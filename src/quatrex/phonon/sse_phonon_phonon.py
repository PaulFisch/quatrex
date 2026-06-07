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

import warnings
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


def _build_cell_zero_mode_projector(h00, h01, *, floor_thz=0.1):
    """Cell-level rigid-mode projector ``Q = I - V Vᵀ`` (n_dof×n_dof).

    Removes the cell's q=0 rigid-body modes -- the 3 Cartesian
    translations plus any near-zero rotational quasi-Goldstone (e.g. a
    1-D wire's axial twist). A mode is rigid if its frequency is below
    ``floor_thz`` (eigenvalue of the cell Gamma-matrix
    ``h00 + h01 + h01^dagger`` below ``floor_thz**2``; blocks in THz²).
    The floor is ABSOLUTE so a stiff transport-irrelevant mode (e.g. a
    Si-H stretch) cannot inflate the cutoff and over-project real low-ω
    heat carriers. Applied two-sided per device band block it is
    band-local (no cross-slab fill), unlike the dense global projector
    in ``phonon/solver/zero_modes.py``.

    Returns ``(Q, projected_freqs_thz)``.
    """
    h00 = np.asarray(h00)
    h01 = np.asarray(h01)
    n_dof = h00.shape[0]
    dyn = h00 + h01 + h01.conj().T
    dyn = 0.5 * (dyn + dyn.conj().T)
    eigvals, eigvecs = np.linalg.eigh(dyn)
    idx = np.where(eigvals.real < float(floor_thz) ** 2)[0]
    if idx.size == 0:
        return np.eye(n_dof), np.array([])
    V = eigvecs[:, idx]
    Q = np.eye(n_dof, dtype=V.dtype) - V @ V.conj().T
    if np.allclose(Q.imag, 0.0):
        Q = Q.real
    projected_freqs = np.sqrt(np.clip(eigvals[idx].real, 0.0, None))
    return Q, projected_freqs


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
        dynamical_matrix: "DSDBSparse | None" = None,
        qfold: "tuple | None" = None,
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

        # Transversely-periodic (k>1) coupled-q vertices. When present, the
        # 3-phonon bubble couples the transverse momenta (crystal-momentum
        # conservation); the Gamma vertex ``vertices[(0,0)]`` defines the
        # block-pair index (consistent with the dense reference, which also
        # takes the pair index from the (0,0) entry) AND serves as the device
        # FC3 (so ``fc3_path`` is not separately required). Built offline (no
        # G dependence) and loaded as arrays only. See quatrex.phonon.qfold.
        self._qvertices: dict | None = None
        self._q_diff_map: NDArray | None = None
        self._n_kpts: int = 1
        if qfold is None:
            qfold_path = getattr(config.phonon, "qfold_path", None)
            if qfold_path is not None:
                from quatrex.phonon.qfold import load_qfold

                vertices, q_diff_map, nk_shape = load_qfold(Path(qfold_path))
                qfold = (vertices, q_diff_map, int(np.prod(nk_shape)))
        if qfold is not None:
            vertices, q_diff_map, n_kpts = qfold
            self._qvertices = vertices
            self._q_diff_map = np.asarray(q_diff_map, dtype=int)
            self._n_kpts = int(n_kpts)
            if phi_blocks is None:
                phi_blocks = vertices[(0, 0)]

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

        # Optional rigid-body (q=0) zero-mode projector Q_cell (cell-level,
        # n_dof×n_dof), applied two-sided to every band block of Σ^{<,>}.
        # Band-local ⇒ preserves the production block sparsity. Built from
        # the cell dynamical-matrix blocks; requires uniform block sizes
        # (one cell per block). See config.phonon.zero_mode_projection.
        self._zero_mode_Q: NDArray | None = None
        if getattr(config.phonon, "zero_mode_projection", False):
            self._build_zero_mode_projector(config, dynamical_matrix)

        # Optional self-consistent SCP cubic-tadpole static self-energy.
        self._scp_tadpole = bool(getattr(config.phonon, "scp_tadpole", False))
        self._sigma_static: NDArray | None = None
        if self._scp_tadpole:
            self._setup_scp_tadpole(config, dynamical_matrix)

    def _build_zero_mode_projector(self, config, dynamical_matrix) -> None:
        """Build the cell rigid-mode projector ``Q_cell`` from the cell
        dynamical-matrix blocks (H00, H01) and store it on
        ``self._zero_mode_Q``. No-op (with a warning) if the inputs are
        unavailable or the block sizes are non-uniform."""
        if dynamical_matrix is None:
            warnings.warn(
                "zero_mode_projection requested but no dynamical_matrix was "
                "passed to SigmaPhononPhonon; projection DISABLED.",
                stacklevel=2,
            )
            return
        if not np.all(self.block_sizes == self.block_sizes[0]):
            warnings.warn(
                "zero_mode_projection requires uniform block sizes "
                f"(got {self.block_sizes.tolist()}); projection DISABLED.",
                stacklevel=2,
            )
            return
        # Cell on-site / forward-coupling blocks (THz²), ω-independent ⇒
        # take the first stack slice. blocks[0,1] is H01 (cell 0 → cell 1).
        # The stack is (energy, *k_transverse); reducing every leading axis
        # to 0 selects the Gamma-transverse (q⊥=0) cell matrix — the right
        # reference for the rigid/soft modes (handles both the Γ-only k==1
        # device and the transversely-periodic k>1 film).
        h00 = np.asarray(dynamical_matrix.blocks[0, 0])
        h01 = np.asarray(dynamical_matrix.blocks[0, 1])
        while h00.ndim > 2:
            h00 = h00[0]
        while h01.ndim > 2:
            h01 = h01[0]
        floor = float(getattr(config.phonon, "zero_mode_floor_thz", 0.1))
        Q, projected = _build_cell_zero_mode_projector(h00, h01, floor_thz=floor)
        if projected.size == 0:
            warnings.warn(
                f"zero_mode_projection ON but no cell mode below "
                f"{floor} THz; projector is identity (no-op).",
                stacklevel=2,
            )
            self._zero_mode_Q = None
            return
        self._zero_mode_Q = np.ascontiguousarray(Q)
        if comm.rank == 0:
            freqs = ", ".join(f"{f:.4f}" for f in np.sort(projected))
            print(
                f"[SigmaPhononPhonon] zero-mode projection ON: projecting "
                f"{projected.size} cell mode(s) below {floor} THz "
                f"[{freqs}] THz out of Σ^<,>.",
                flush=True,
            )

    def _setup_scp_tadpole(self, config, dynamical_matrix) -> None:
        """Prepare the self-consistent SCP cubic-tadpole static self-energy:
        the dense device FC3 + dynamical matrix, the mixing/floor, and the
        zeroed running ``Sigma_static``. No-op (warning) if prerequisites
        are missing or the layout is unsupported (non-uniform blocks, or a
        block-distributed run — the soft-mode regime is single/few-cell)."""
        from quatrex.phonon.static_self_energy import device_fc3_mass_weighted

        if dynamical_matrix is None:
            warnings.warn(
                "scp_tadpole requested but no dynamical_matrix passed to "
                "SigmaPhononPhonon; SCP tadpole DISABLED.", stacklevel=2)
            self._scp_tadpole = False
            return
        if not np.all(self.block_sizes == self.block_sizes[0]):
            warnings.warn(
                "scp_tadpole requires uniform block sizes; DISABLED.",
                stacklevel=2)
            self._scp_tadpole = False
            return
        if ranks.block.size > 1:
            warnings.warn(
                "scp_tadpole is not yet implemented for block-distributed "
                "runs (comm.block.size>1); SCP tadpole DISABLED. Use a single "
                "block rank (the soft-mode regime is single/few-cell).",
                stacklevel=2)
            self._scp_tadpole = False
            return

        n_dof = int(self.block_sizes[0])
        n_blocks = self.n_blocks
        N_D = n_blocks * n_dof
        # Dense device FC3 in the tadpole mass-weighting (bubble blocks / C_FC3).
        self._fc3_dev_mw = device_fc3_mass_weighted(
            self.phi_blocks, n_blocks, n_dof)
        # Dense device dynamical matrix D (THz²), ω-independent.
        D = np.zeros((N_D, N_D), dtype=float)
        for I in range(n_blocks):
            for J in range(max(0, I - 1), min(n_blocks, I + 2)):
                blk = np.asarray(dynamical_matrix.blocks[I, J])
                if blk.ndim == 3:
                    blk = blk[0]
                D[I * n_dof:(I + 1) * n_dof, J * n_dof:(J + 1) * n_dof] = blk.real
        self._scp_D = 0.5 * (D + D.T)
        self._sigma_static = np.zeros((N_D, N_D), dtype=float)
        self._scp_mix = float(getattr(config.phonon, "scp_static_mixing", 0.1))
        self._scp_floor2 = float(getattr(config.phonon, "scp_floor_thz", 0.5)) ** 2
        if comm.rank == 0:
            print(
                f"[SigmaPhononPhonon] SCP tadpole ON: N_D={N_D}, "
                f"mixing={self._scp_mix}, Phi_eff floor="
                f"{getattr(config.phonon, 'scp_floor_thz', 0.5)} THz.",
                flush=True,
            )

    def _apply_scp_tadpole(self, g_lesser, sigma_retarded) -> None:
        """Self-consistent cubic tadpole, in the nnz state.

        Forms ``<uu>`` from the ω-integral of the device ``G^<`` (full ω
        local per nnz slice), solves the regularised ``mean_displacement``
        against ``Phi_eff = D + Sigma_static``, mixes the resulting static
        ``Sigma_T``, and broadcasts it into ``Sigma^R`` at every frequency
        (≡ stiffening the dynamical matrix). Reuses the bubble's ``G^<`` --
        no recomputation.
        """
        from quatrex.phonon.static_self_energy import (
            equal_time_uu_from_sum, mean_displacement, sigma_tadpole)

        N_D = self._sigma_static.shape[0]
        # ω-integral of G^< (nnz: all ω local). Sum over every stack axis.
        ax = tuple(range(g_lesser.data.ndim - 1))
        g_sum_local = g_lesser.data.sum(axis=ax)             # (local_nnz,)
        rows = np.asarray(g_lesser.rows) + int(g_lesser.global_block_offset)
        cols = np.asarray(g_lesser.cols) + int(g_lesser.global_block_offset)
        g_sum = np.zeros((N_D, N_D), dtype=g_sum_local.dtype)
        g_sum[rows, cols] = np.asarray(g_sum_local)
        # nnz is split over comm.stack (full ω local, spatial elements split).
        if ranks.stack.size > 1:
            recv = xp.zeros_like(g_sum)
            ranks.stack.all_reduce(g_sum, recv, op="sum")
            g_sum = recv
        dw = float(self.local_frequencies[1] - self.local_frequencies[0])
        uu = equal_time_uu_from_sum(g_sum, dw)
        phi_eff = self._scp_D + self._sigma_static
        w_mean = mean_displacement(
            self._fc3_dev_mw, uu, phi_eff,
            omega2_floor_abs=self._scp_floor2)
        sig_new = sigma_tadpole(self._fc3_dev_mw, w_mean)
        self._sigma_static = (
            (1.0 - self._scp_mix) * self._sigma_static + self._scp_mix * sig_new)
        if getattr(self, "_scp_verbose", False) and comm.rank == 0:
            print(f"[SigmaPhononPhonon] ||Sigma_static||={np.linalg.norm(self._sigma_static):.3f} "
                  f"||Sigma_T(new)||={np.linalg.norm(sig_new):.3f} THz^2", flush=True)
        # Broadcast the static self-energy into Σ^R at every frequency.
        sr_static = self._sigma_static[rows, cols].astype(sigma_retarded.data.dtype)
        sigma_retarded.data[:] = sigma_retarded.data + sr_static

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
        nq = int(np.prod(nk)) if len(nk) else 1
        if nq > 1 and self._qvertices is None:
            raise ValueError(
                f"Transverse-q device (mesh {nk}, n_kpts={nq}) requires the "
                "q-folded vertices; set config.phonon.qfold_path (see "
                "quatrex.phonon.qfold)."
            )
        if nq > 1 and self._n_kpts != nq:
            raise ValueError(
                f"q-folded vertices are for n_kpts={self._n_kpts} but the "
                f"Green's function has {nq} transverse momenta {nk}."
            )
        if nq > 1 and ranks.block.size > 1:
            # The transverse-q axis is local in the stack, so q distributes
            # over comm.stack (energy) for free; only the comm.block band
            # halo would need to carry the q-axis. Distribute the film over
            # energy (and/or comm.q), keeping block_comm_size == 1.
            raise NotImplementedError(
                "Transverse-q (k>1) with comm.block.size > 1 is not "
                "supported (the band halo is not q-aware); run the periodic "
                "device with block_comm_size == 1 and distribute over the "
                "energy/stack axis."
            )
        n_fft = 2 * ne_full - 1
        full_freqs = self._full_frequencies(ne_full)
        prefactor = bubble_prefactor_thz(float(full_freqs[1] - full_freqs[0]))
        # Coupled-q convolution carries the 1/N_q mesh-average (matches the
        # dense reference's ``bubble_prefactor(..., n_kpts=N_q)``).
        if nq > 1:
            prefactor = prefactor / nq

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

        Qproj = (
            xp.asarray(self._zero_mode_Q)
            if self._zero_mode_Q is not None
            else None
        )
        if nq == 1:
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
                    if Qproj is not None:
                        # Two-sided rigid-mode projection on every band block
                        # (τ-independent ⇒ commutes with the IFFT; preserves
                        # Σ^<,> Hermiticity and the band sparsity). Q is real
                        # n_dof×n_dof; broadcasts over the leading τ axis.
                        acc_l = Qproj @ acc_l @ Qproj
                        acc_g = Qproj @ acc_g @ Qproj
                    stlv.blocks[I - start, J - start] = acc_l
                    stgv.blocks[I - start, J - start] = acc_g
        else:
            # Coupled-q convolution. The transverse-momentum axis rides as a
            # LOCAL batch dimension of the stack (only ω is split across
            # comm.stack), so the whole q-sum is local — no q communication.
            # For each external q_ext: Σ(q_ext) = Σ_{q'} ring[ Φ̃(q',q2),
            # Φ̃(q2,q'), G(q')_{K1K1'}, G(q2)_{K2K2'} ] with q2 = q_ext − q'.
            qdm = self._q_diff_map
            qv = self._qvertices

            def _qflat(d):
                # (τ, *nk, b, b) → (τ, N_q, b, b); the q-axis is contiguous
                # in the same C-order as global_stack_shape[1:].
                return {
                    kk: v.reshape(v.shape[0], nq, v.shape[-2], v.shape[-1])
                    for kk, v in d.items()
                }

            gl_q = _qflat(gl_blk)
            gg_q = _qflat(gg_blk)
            n_tau = next(iter(gl_q.values())).shape[0]
            dtype = next(iter(gl_q.values())).dtype
            for (I, J) in owned:
                bs_I = int(self.block_sizes[I])
                bs_J = int(self.block_sizes[J])
                out_l = xp.zeros((n_tau, nq, bs_I, bs_J), dtype=dtype)
                out_g = xp.zeros((n_tau, nq, bs_I, bs_J), dtype=dtype)
                wrote = False
                for iq_ext in range(nq):
                    acc_l = None
                    acc_g = None
                    for iqp in range(nq):
                        iq2 = int(qdm[iq_ext, iqp])
                        phiL = qv.get((iqp, iq2))   # legs (q', q_ext−q')
                        phiR = qv.get((iq2, iqp))   # legs (q_ext−q', q')
                        if phiL is None or phiR is None:
                            continue
                        for K1, K2, K1p, K2p, _pl, _pr in self._phi_pair_index[(I, J)]:
                            pl = phiL.get((I, K1, K2))
                            pr = phiR.get((J, K2p, K1p))
                            if pl is None or pr is None:
                                continue
                            sl = ring_contract(
                                pl, pr,
                                gl_q[(K1, K1p)][:, iqp], gl_q[(K2, K2p)][:, iq2],
                                xp=xp,
                            )
                            sg = ring_contract(
                                pl, pr,
                                gg_q[(K1, K1p)][:, iqp], gg_q[(K2, K2p)][:, iq2],
                                xp=xp,
                            )
                            acc_l = sl if acc_l is None else acc_l + sl
                            acc_g = sg if acc_g is None else acc_g + sg
                    if acc_l is not None:
                        if Qproj is not None:
                            acc_l = Qproj @ acc_l @ Qproj
                            acc_g = Qproj @ acc_g @ Qproj
                        out_l[:, iq_ext] = acc_l
                        out_g[:, iq_ext] = acc_g
                        wrote = True
                if wrote:
                    blk_shape = (n_tau,) + tuple(nk) + (bs_I, bs_J)
                    stlv.blocks[I - start, J - start] = out_l.reshape(blk_shape)
                    stgv.blocks[I - start, J - start] = out_g.reshape(blk_shape)

        # (4) σ(τ) stack→nnz: each (I,J) full-τ on exactly one rank.
        stl.dtranspose()
        stg.dtranspose()

        # (5) IFFT σ(τ)→Σ(ω) in nnz; add into outputs; build Σ^R.
        sl_data = prefactor * xp.fft.ifft(stl.data, axis=0)[:ne_full]
        sg_data = prefactor * xp.fft.ifft(stg.data, axis=0)[:ne_full]
        sigma_lesser.data[:] = sigma_lesser.data + sl_data
        sigma_greater.data[:] = sigma_greater.data + sg_data
        # Σ^R contribution: only the DISPERSIVE (Hilbert) part. The
        # anti-Hermitian part ½(Σ^>−Σ^<) is added once by the SCBA loop
        # after all interactions (the GW convention,
        # cf. electron/sse_coulomb_screening); adding it here too would
        # double-count it and break heat-flow conservation.
        if self.retarded_method == "fft":
            delta = sg_data - sl_data
            sigma_retarded.data[:] = (
                sigma_retarded.data + 0.5j * hilbert_transform(delta, full_freqs)
            )

        # Self-consistent SCP cubic-tadpole static self-energy (stiffens the
        # soft mode; added into Σ^R at every frequency). Uses the same G^<.
        if self._scp_tadpole and self._sigma_static is not None:
            self._apply_scp_tadpole(g_lesser, sigma_retarded)

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
