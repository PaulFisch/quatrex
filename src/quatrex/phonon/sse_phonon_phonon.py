# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Anharmonic 3-phonon scattering self-energy

Implements
    Sigma^{<,>}_{ab}(omega) = (i hbar / 2) * sum_{c,d,e,f}
        Phi_{a c d}
        * [ G^{<,>}_{cf} * G^{<,>}_{de} ](omega)
        * Phi_{b e f}

The cubic vertex Phi is supplied as a block-sparse dict
(see quatrex.phonon.fc3_loader). Internal units are THz / THz^2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, xp
from qttools.comm import comm as ranks
from qttools.datastructures import DSDBSparse
from qttools.profiling import Profiler

from scipy import sparse

from quatrex.core.config import QuatrexConfig
from quatrex.core.fft_utils import hilbert_transform
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.phonon.bubble import bubble_dense, ring_contract
from quatrex.phonon.fc3_loader import (
    PhiBlocks,
    load_device_fc3,
)
from quatrex.phonon.units import bubble_prefactor_thz

profiler = Profiler()


class SigmaPhononPhonon(ScatteringSelfEnergy):
    """3-phonon SCBA scattering self-energy.

    Parameters
    ----------
    config : QuatrexConfig
        Quatrex configuration object. Reads ``config.phonon.fc3_path``
        and ``config.phonon.retarded_method``.
    phonon_frequencies : NDArray
        The phonon frequency grid in **THz** (uniform). Local slice for
        the calling rank — the full axis is recovered via
        ``comm.stack.all_gather_v`` inside :meth:`compute`.
    block_sizes : NDArray
        Transport-cell DOF sizes ``(N_blocks,)``.
    phi_blocks : PhiBlocks, optional
        Pre-built block-sparse Phi dict in THz^2. If not provided, the
        loader is invoked on ``config.phonon.fc3_path``.
    """

    def __init__(
        self,
        config: QuatrexConfig,
        phonon_frequencies: NDArray,
        block_sizes: NDArray,
        phi_blocks: PhiBlocks | None = None,
    ) -> None:
        # Local energy slice; the full axis is gathered in compute().
        self.local_frequencies = np.asarray(phonon_frequencies)

        self.block_sizes = np.asarray(block_sizes, dtype=int)
        self.n_blocks = int(self.block_sizes.shape[0])
        self.block_offsets = np.concatenate(([0], np.cumsum(self.block_sizes)))

        retarded_method = getattr(config.phonon, "retarded_method", "fft")
        if retarded_method not in ("half", "fft"):
            raise ValueError(
                f"Unknown retarded_method={retarded_method!r}; "
                "use 'half' or 'fft'."
            )
        self.retarded_method = retarded_method

        if phi_blocks is None:
            fc3_path = getattr(config.phonon, "fc3_path", None)
            if fc3_path is None:
                raise ValueError(
                    "config.phonon.fc3_path must be set when phi_blocks "
                    "is not provided to SigmaPhononPhonon."
                )
            phi_blocks = load_device_fc3(
                Path(fc3_path),
                block_sizes=self.block_sizes,
            )
        self.phi_blocks = phi_blocks

        # Inner Green's-function band kept in the contraction. The
        # upstream RGF selected-inversion produces the block-tridiagonal
        # G (diagonal + first off-diagonal, |K-K'| <= 1), so the full
        # off-diagonal ring keeps exactly that band — NOT a diagonal-G
        # approximation. Quads that would require |K-K'| > g_band are
        # never generated, which automatically satisfies the dense
        # reference's g_keys membership gate (se_finite.py:81-84).
        self.g_band = 1

        # Precompute the full off-diagonal pair index: for each output
        # block pair (I, J) with |I-J| <= 1, collect the ring quads
        #   (K1, K2, K1', K2', phi_left, phi_right)
        # with phi_left = Phi[(I, K1, K2)], phi_right = Phi[(J, K2', K1')]
        # and the inner G links (K1, K1'), (K2, K2') inside the band.
        # This mirrors the dense reference _build_pair_index exactly;
        # the contraction uses G_a = G_{K1 K1'}, G_b = G_{K2 K2'}.
        self._phi_pair_index: dict[
            tuple[int, int],
            list[tuple[int, int, int, int, NDArray, NDArray]],
        ] = {}
        for (I, K1, K2), phi_left in self.phi_blocks.items():
            for J in range(max(0, I - 1), min(self.n_blocks, I + 2)):
                for K1p in range(
                    max(0, K1 - self.g_band),
                    min(self.n_blocks, K1 + self.g_band + 1),
                ):
                    for K2p in range(
                        max(0, K2 - self.g_band),
                        min(self.n_blocks, K2 + self.g_band + 1),
                    ):
                        phi_right = self.phi_blocks.get((J, K2p, K1p))
                        if phi_right is None:
                            continue
                        self._phi_pair_index.setdefault((I, J), []).append(
                            (K1, K2, K1p, K2p, phi_left, phi_right)
                        )

        # Distinct inner G band blocks (K, K') referenced by any quad —
        # the exact set that must be gathered to full omega.
        self._g_band_keys: set[tuple[int, int]] = set()
        for quads in self._phi_pair_index.values():
            for K1, K2, K1p, K2p, _pl, _pr in quads:
                self._g_band_keys.add((K1, K1p))
                self._g_band_keys.add((K2, K2p))

        # Lazily-built, cached intermediate τ-domain buffers (length
        # n_fft) for the FFT-first pipeline; keyed by (buffer class,
        # n_fft). Allocated on first compute() from G's sparsity.
        self._tau_cache: tuple | None = None
        self._full_freqs: NDArray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @profiler.profile(label="SigmaPhononPhonon", level="default", comm=comm)
    def compute(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
    ) -> None:
        """Compute the 3-phonon self-energy contribution (FFT-first).

        Memory-optimal distributed pipeline (theory.tex
        §ssec:phph_distribution), reusing the GW polarisation pattern:

        1. ``G(ω)→g(τ)`` by FFT in **nnz** distribution (full ω local
           per nnz slice, so the FFT work and storage divide over the
           ranks — the full-ω band is never replicated on every rank);
        2. ``g(τ)`` **nnz→stack** so each ``comm.stack`` rank owns a
           τ-slice of the full band, block-accessible;
        3. the full off-diagonal ring contraction per τ-slice in stack
           (``G_a=g_{K1K1'}``, ``G_b=g_{K2K2'}``, ``|K-K'|≤1`` — NOT a
           diagonal-G approximation);
        4. ``σ(τ)`` **stack→nnz** (a partition, so each ``(I,J)`` full-τ
           lands on exactly one rank — no ``P_E``-fold double counting);
        5. ``σ(τ)→Σ(ω)`` by IFFT in nnz, added into the outputs; ``Σ^R``
           via the bosonic Hilbert transform.

        The incoming distribution state of every buffer is restored on
        exit. Additive: contributions are added into ``out``.

        Parameters
        ----------
        g_lesser, g_greater
            Phonon Green's functions.
        out
            ``(sigma_lesser, sigma_greater, sigma_retarded)``.
        """
        sigma_lesser, sigma_greater, sigma_retarded = out

        all_bufs = (
            g_lesser, g_greater, sigma_lesser, sigma_greater, sigma_retarded,
        )
        incoming_states = {id(m): m.distribution_state for m in all_bufs}

        # The whole pipeline operates with the data buffers in nnz
        # (full ω local per nnz slice — what the FFT/IFFT and the
        # additive write-back need).
        for m in all_bufs:
            if m.distribution_state != "nnz":
                m.dtranspose()

        try:
            self._compute_fft_first(
                g_lesser, g_greater,
                sigma_lesser, sigma_greater, sigma_retarded,
            )
        finally:
            for m in all_bufs:
                if m.distribution_state != incoming_states[id(m)]:
                    m.dtranspose()

    # ------------------------------------------------------------------
    # Internals — FFT-first pipeline
    # ------------------------------------------------------------------

    def _compute_fft_first(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        sigma_lesser: DSDBSparse,
        sigma_greater: DSDBSparse,
        sigma_retarded: DSDBSparse,
    ) -> None:
        """The FFT-first contraction; all output buffers in nnz on entry."""
        ne_full = int(g_lesser.global_stack_shape[0])
        nk = tuple(int(k) for k in g_lesser.global_stack_shape[1:])
        if any(k > 1 for k in nk):
            raise NotImplementedError(
                "Transverse-q (k>1) phonon-phonon SSE (the Φ̃(q,q') "
                "momentum convolution) is not implemented in this finite "
                "block-tridiagonal path."
            )
        n_fft = 2 * ne_full - 1
        full_freqs = self._full_frequencies(ne_full)
        prefactor = bubble_prefactor_thz(float(full_freqs[1] - full_freqs[0]))

        gtl, gtg, stl, stg = self._ensure_tau_buffers(g_lesser, n_fft)

        # (1) FFT G(ω)→g(τ) in nnz. The τ-buffers share G's exact nnz
        # ordering (asserted at allocation), so the raw-data assignment
        # is index-consistent.
        for m in (gtl, gtg, stl, stg):
            if m.distribution_state != "nnz":
                m.dtranspose(discard=True)
        gtl.data[:] = self._fft_pad(g_lesser.data, n_fft)
        gtg.data[:] = self._fft_pad(g_greater.data, n_fft)

        # (2) g(τ) nnz→stack: τ-slice of the full band per stack rank.
        gtl.dtranspose()
        gtg.dtranspose()
        if stl.distribution_state != "stack":
            stl.dtranspose(discard=True)
        if stg.distribution_state != "stack":
            stg.dtranspose(discard=True)
        stl.data[:] = 0.0
        stg.data[:] = 0.0

        gtlv, gtgv = gtl.stack[...], gtg.stack[...]
        stlv, stgv = stl.stack[...], stg.stack[...]

        # (3) Ring contraction per τ-slice in stack. Under a comm.block
        # split each rank owns outputs (I,J) with min(I,J) in its block
        # window and fetches the off-window band links from its
        # neighbours via a width-N_h halo (arrow partitioning makes every
        # band block addressable by the rank owning min(K,K')).
        start = int(stl.block_section_offsets[ranks.block.rank])
        end = int(stl.block_section_offsets[ranks.block.rank + 1])
        owned = self._owned_outputs(start, end)

        if ranks.block.size > 1:
            halo_l, halo_g = self._exchange_band_halo(gtlv, gtgv, gtl, start, end)
        else:
            halo_l = halo_g = {}

        # Materialise each distinct band link once (a link recurs across
        # many quads; the block indexer otherwise re-gathers it each time).
        gl_blk: dict[tuple[int, int], NDArray] = {}
        gg_blk: dict[tuple[int, int], NDArray] = {}
        for (K, Kp) in self._links_for_range(start, end):
            if start <= min(K, Kp) < end:
                gl_blk[(K, Kp)] = gtlv.blocks[K - start, Kp - start]
                gg_blk[(K, Kp)] = gtgv.blocks[K - start, Kp - start]
            else:
                gl_blk[(K, Kp)] = halo_l[(K, Kp)]
                gg_blk[(K, Kp)] = halo_g[(K, Kp)]

        for (I, J) in owned:
            acc_l = None
            acc_g = None
            for K1, K2, K1p, K2p, phi_left, phi_right in self._phi_pair_index[(I, J)]:
                sl = ring_contract(
                    phi_left, phi_right,
                    gl_blk[(K1, K1p)], gl_blk[(K2, K2p)], xp=xp,
                )
                sg = ring_contract(
                    phi_left, phi_right,
                    gg_blk[(K1, K1p)], gg_blk[(K2, K2p)], xp=xp,
                )
                acc_l = sl if acc_l is None else acc_l + sl
                acc_g = sg if acc_g is None else acc_g + sg
            if acc_l is not None:
                stlv.blocks[I - start, J - start] = acc_l
                stgv.blocks[I - start, J - start] = acc_g

        # (4) σ(τ) stack→nnz: each (I,J) full-τ on exactly one rank.
        stl.dtranspose()
        stg.dtranspose()

        # (5) IFFT σ(τ)→Σ(ω) in nnz; add into outputs; build Σ^R.
        sl_data = prefactor * xp.fft.ifft(stl.data, axis=0)[:ne_full]
        sg_data = prefactor * xp.fft.ifft(stg.data, axis=0)[:ne_full]
        sigma_lesser.data[:] = sigma_lesser.data + sl_data
        sigma_greater.data[:] = sigma_greater.data + sg_data
        delta = sg_data - sl_data
        sr_data = 0.5 * delta
        if self.retarded_method == "fft":
            sr_data = sr_data + 0.5j * hilbert_transform(delta, full_freqs)
        sigma_retarded.data[:] = sigma_retarded.data + sr_data

    @staticmethod
    def _fft_pad(data: NDArray, n_fft: int) -> NDArray:
        """Zero-pad along the energy axis (0) to ``n_fft`` and FFT."""
        pad_shape = (n_fft - data.shape[0],) + tuple(data.shape[1:])
        padded = xp.concatenate(
            [data, xp.zeros(pad_shape, dtype=data.dtype)], axis=0
        )
        return xp.fft.fft(padded, axis=0)

    def _links_for_range(self, a: int, b: int) -> set[tuple[int, int]]:
        """Distinct inner G band links needed by outputs owned (by the
        ``min(I,J)`` rule) by the block range ``[a, b)`` (cached: the
        global pair index is fixed, so each range is scanned once)."""
        cache = getattr(self, "_links_cache", None)
        if cache is None:
            cache = self._links_cache = {}
        if (a, b) in cache:
            return cache[(a, b)]
        links: set[tuple[int, int]] = set()
        for (I, J), quads in self._phi_pair_index.items():
            if not (a <= min(I, J) < b):
                continue
            for K1, K2, K1p, K2p, _pl, _pr in quads:
                links.add((K1, K1p))
                links.add((K2, K2p))
        cache[(a, b)] = links
        return links

    def _owned_outputs(self, start: int, end: int) -> list[tuple[int, int]]:
        """Output pairs (I,J) this comm.block rank owns (by min(I,J)),
        cached per block range."""
        cache = getattr(self, "_owned_cache", None)
        if cache is None:
            cache = self._owned_cache = {}
        if (start, end) not in cache:
            cache[(start, end)] = [
                ij for ij in self._phi_pair_index
                if start <= min(ij) < end
            ]
        return cache[(start, end)]

    def _exchange_band_halo(
        self, gtlv, gtgv, ref: DSDBSparse, start: int, end: int,
    ) -> tuple[dict[tuple[int, int], NDArray], dict[tuple[int, int], NDArray]]:
        """Fetch the off-window band links (for both lesser and greater)
        from the immediate ``comm.block`` neighbours via a width-``N_h``
        halo.

        Arrow partitioning makes each band block addressable by the rank
        owning ``min(K,K')``; this rank therefore sends to a neighbour the
        links that neighbour's outputs need and which fall in this rank's
        window, and receives the symmetric set. The send/receive lists are
        derived from the global pair index and the global block ranges, so
        both ends enumerate them in the same (sorted) order — matching the
        non-overtaking ``Isend``/``Irecv`` per ``(source, tag)``.
        """
        bcomm = ranks.block._mpi_comm
        rank, size = ranks.block.rank, ranks.block.size
        offs = ref.block_section_offsets
        nloc_tau = int(ref.data.shape[0])  # local τ count (stack state)

        def is_local(link):
            return start <= min(link) < end

        def shape(link):
            K, Kp = link
            return (nloc_tau, int(self.block_sizes[K]), int(self.block_sizes[Kp]))

        halo_l: dict[tuple[int, int], NDArray] = {}
        halo_g: dict[tuple[int, int], NDArray] = {}
        reqs = []
        sendbufs = []  # keep buffers alive until Waitall

        def post(neigh, lo, hi, send_tag_l, send_tag_g, recv_tag_l, recv_tag_g):
            # send: links the neighbour's outputs need that I own
            for link in sorted(l for l in self._links_for_range(lo, hi) if is_local(l)):
                K, Kp = link
                bl = xp.ascontiguousarray(gtlv.blocks[K - start, Kp - start])
                bg = xp.ascontiguousarray(gtgv.blocks[K - start, Kp - start])
                sendbufs.extend((bl, bg))
                reqs.append(bcomm.Isend(bl, dest=neigh, tag=send_tag_l))
                reqs.append(bcomm.Isend(bg, dest=neigh, tag=send_tag_g))
            # recv: links my outputs need that the neighbour owns
            for link in sorted(l for l in self._links_for_range(start, end)
                               if lo <= min(l) < hi):
                bl = xp.empty(shape(link), dtype=complex)
                bg = xp.empty(shape(link), dtype=complex)
                halo_l[link], halo_g[link] = bl, bg
                reqs.append(bcomm.Irecv(bl, source=neigh, tag=recv_tag_l))
                reqs.append(bcomm.Irecv(bg, source=neigh, tag=recv_tag_g))

        if rank + 1 < size:  # upper neighbour; my send uses tags 0/1
            post(rank + 1, int(offs[rank + 1]), int(offs[rank + 2]), 0, 1, 2, 3)
        if rank - 1 >= 0:    # lower neighbour; my send uses tags 2/3
            post(rank - 1, int(offs[rank - 1]), int(offs[rank]), 2, 3, 0, 1)

        # Guard: every non-local needed link must be in an immediate
        # neighbour (i.e. each rank owns >= the halo reach). Otherwise the
        # halo is multi-hop and this targeted exchange is insufficient.
        needed_nonlocal = {l for l in self._links_for_range(start, end)
                           if not is_local(l)}
        missing = needed_nonlocal - set(halo_l)
        if missing:
            raise RuntimeError(
                "Phonon-phonon halo reaches beyond the immediate comm.block "
                f"neighbour (links {sorted(missing)}). Give each comm.block "
                "rank at least N_h blocks (reduce comm.block.size)."
            )
        MPI.Request.Waitall(reqs)
        return halo_l, halo_g

    def _ensure_tau_buffers(
        self, g_lesser: DSDBSparse, n_fft: int
    ) -> tuple[DSDBSparse, DSDBSparse, DSDBSparse, DSDBSparse]:
        """Lazily allocate the four n_fft τ-buffers sharing G's pattern.

        Built once via ``from_sparray`` from G's own sparsity (recovered
        with :meth:`spy`), so they carry the identical nnz ordering —
        a hard requirement for the raw-data FFT assignment in
        :meth:`_compute_fft_first` to be index-consistent (asserted).
        """
        cls = type(g_lesser)
        key = (cls, n_fft)
        if self._tau_cache is not None and self._tau_cache[0] == key:
            return self._tau_cache[1]

        rows, cols = g_lesser.spy()
        N = int(np.sum(self.block_sizes))
        pattern = sparse.coo_matrix(
            (
                np.ones(len(rows), dtype=np.complex128),
                (np.asarray(rows), np.asarray(cols)),
            ),
            shape=(N, N),
        ).tocsr()
        nk = tuple(int(k) for k in g_lesser.global_stack_shape[1:])
        bufs = tuple(
            cls.from_sparray(
                pattern, self.block_sizes, global_stack_shape=(n_fft,) + nk
            )
            for _ in range(4)
        )
        # The raw-data FFT path requires the τ-buffers to carry G's exact
        # internal nnz ordering. from_sparray is deterministic for a given
        # pattern, so rebuilding from G's own spy() reproduces G's order;
        # assert the (unsorted) per-index match to be certain.
        gt_rows, gt_cols = bufs[0].spy()
        if not (
            np.array_equal(np.asarray(rows), np.asarray(gt_rows))
            and np.array_equal(np.asarray(cols), np.asarray(gt_cols))
        ):
            raise RuntimeError(
                "τ-buffer sparsity does not match G; cannot FFT raw data."
            )
        self._tau_cache = (key, bufs)
        return bufs

    # ------------------------------------------------------------------
    # Internals — block-bubble FFT contraction
    # ------------------------------------------------------------------

    def _bubble_block(
        self,
        *,
        phi_left: NDArray,
        phi_right: NDArray,
        G_inner_a: NDArray,
        G_inner_b: NDArray,
        n_fft: int,
        prefactor: complex,
    ) -> NDArray:
        """FFT 3-phonon bubble for one block triplet pair (THz²)."""
        ne = G_inner_a.shape[0]
        return bubble_dense(
            phi_left=phi_left,
            phi_right=phi_right,
            G_a=G_inner_a,
            G_b=G_inner_b,
            n_fft=n_fft,
            prefactor=prefactor,
            out_slice=slice(0, ne),
            zero_freq_idx=None,
            xp=xp,
        )

    # ------------------------------------------------------------------
    # Internals — distribution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _block_range(m: DSDBSparse) -> tuple[int, int]:
        """Return the [start, end) global-block range owned by this
        ``comm.block`` rank.
        """
        start = int(m.block_section_offsets[ranks.block.rank])
        end = int(m.block_section_offsets[ranks.block.rank + 1])
        return start, end

    def _full_frequencies(self, ne_full: int) -> NDArray:
        """Full (cached) frequency grid; all-gathers the local slice."""
        if self._full_freqs is not None and self._full_freqs.shape[0] == ne_full:
            return self._full_freqs
        if ranks.stack.size == 1:
            freqs = xp.asarray(self.local_frequencies, dtype=float)
        else:
            freqs = ranks.stack.all_gather_v(
                xp.asarray(self.local_frequencies, dtype=float), axis=0
            )
        self._full_freqs = freqs
        return freqs
