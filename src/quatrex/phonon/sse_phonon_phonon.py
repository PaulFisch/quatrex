# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Anharmonic 3-phonon scattering self-energy

Implements
    Sigma^{<,>}_{ab}(omega) = (i hbar / 2) * sum_{c,d,e,f}
        Phi_{a c d}
        * [ G^{<,>}_{cf} * G^{<,>}_{de} ](omega)
        * Phi_{b e f}

The cubic vertex Phi is supplied as a block-sparse dict.
Internal units are THz / THz^2
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, xp
from qttools.comm import comm as ranks
from qttools.datastructures import DSDBSparse
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_host

from scipy import sparse

from quatrex.core.config import QuatrexConfig
from quatrex.core.fft_utils import hilbert_transform
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.phonon.bubble import (
    _ring_contract_serial,
    configure_ring_pool,
    ring_pool,
    bubble_dense,
    phi_perms,
    ring_contract,
    ring_contract_pre,
)
from quatrex.phonon.fc3_loader import (
    PhiBlocks,
    load_device_fc3,
)
from quatrex.phonon.units import bubble_prefactor_thz

profiler = Profiler()

# Bose argument hbar*omega/(kB*T) per THz per K: x = _THZ_OVER_K * f[THz] / T[K]
# (~47.99 f/T) == units.HBAR_EV * units.THZ_TO_RAD / kB[eV/K]. Kept as the exact
# literal product so the occupation window stays bit-identical.
_THZ_OVER_K = 6.582119569e-16 * 2.0 * np.pi * 1e12 / 8.617333262e-5


class SigmaPhononPhonon(ScatteringSelfEnergy):
    """3-phonon SCBA scattering self-energy.

    Parameters
    ----------
    config : QuatrexConfig
        Quatrex configuration object. Reads config.phonon.fc3_path
        and config.phonon.retarded_method.
    phonon_frequencies : NDArray
        The phonon frequency grid in THz. Local slice for
        the calling rank
    block_sizes : NDArray
        Transport-cell DOF sizes (N_blocks,).
    phi_blocks : PhiBlocks, optional
        Pre-built block-sparse Phi dict in THz^2. If not provided, the
        loader is invoked on config.phonon.fc3_path.
    """

    def __init__(
        self,
        config: QuatrexConfig,
        phonon_frequencies: NDArray,
        block_sizes: NDArray,
        phi_blocks: PhiBlocks | None = None,
        dynamical_matrix: DSDBSparse | None = None,
        qfold: tuple | None = None,
        vfactors=None,
    ) -> None:
        self.local_frequencies = np.asarray(phonon_frequencies)

        self.block_sizes = np.asarray(block_sizes, dtype=int)
        self.n_blocks = int(self.block_sizes.shape[0])
        self.block_offsets = np.concatenate(([0], np.cumsum(self.block_sizes)))

        self._ramp_n = int(getattr(config.phonon, "sse_ramp_iterations", 0))
        self._ramp_it = 0
        self._vertex_scale = float(getattr(config.phonon, "sse_vertex_scale", 1.0))
        # Ring-contraction thread pool: config option, env vars as fallback.
        configure_ring_pool(
            threads=int(getattr(config.phonon, "sse_ring_threads", 0)),
            min_w=getattr(config.phonon, "sse_ring_min_w", None),
        )
        retarded_method = getattr(config.phonon, "retarded_method", "fft")
        if retarded_method not in ("half", "fft"):
            raise ValueError(
                f"Unknown retarded_method={retarded_method!r}; "
                "use 'half' or 'fft'."
            )
        self.retarded_method = retarded_method

        # Transversely-periodic (k>1) coupled-q vertices
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

        # TENSOR-DECOMPOSED coupled-q vertex (exact per-leg factorisation of
        # the q-folded blocks, see quatrex.phonon.vertex_factors). Mutually
        # exclusive with the dense qfold dict; the factored kernel replaces
        # the per-(q', q2) triple-GEMM loop by skinny Grams + a sandwich.
        self._vfactors = None
        if vfactors is None:
            dv_path = getattr(config.phonon, "decomposed_vertices_path", None)
            if dv_path is not None:
                from quatrex.phonon.vertex_factors import load_decomposed

                vfactors = load_decomposed(
                    Path(dv_path),
                    rank=int(getattr(config.phonon, "sse_vertex_rank", 0)),
                )
        if vfactors is not None:
            if self._qvertices is not None:
                raise ValueError(
                    "decomposed_vertices_path and qfold_path/qfold are "
                    "mutually exclusive -- pick the dense or the factored "
                    "coupled-q vertex."
                )
            if np.unique(self.block_sizes).size != 1:
                raise ValueError(
                    "The factored coupled-q SSE requires uniform block "
                    f"sizes; got {np.unique(self.block_sizes)}."
                )
            if int(self.block_sizes[0]) != int(vfactors.D.shape[0]):
                raise ValueError(
                    f"Factored vertex n_dof={int(vfactors.D.shape[0])} does "
                    f"not match the device block size "
                    f"{int(self.block_sizes[0])}."
                )
            self._vfactors = vfactors
            self._q_diff_map = np.asarray(vfactors.q_diff_map, dtype=int)
            self._n_kpts = int(vfactors.n_kpts)
            # Kernel choice for consuming the factors:
            #   "reconstruct" (default): materialise the RANK-LOCAL slice of
            #     the dense q-folded dict from the factors once at first
            #     compute (the vertex is fixed) and run the dense path; the
            #     factored win is MEMORY + build time, not flops.
            #   "gram": the skinny-Gram contraction (bubble_factored) --
            #     fewer flops than dense only at small rank or large block
            #     sizes (memory-bound in R^2 on small-block systems).
            self._vf_kernel = str(getattr(
                config.phonon, "decomposed_kernel", "reconstruct"))
            if self._vf_kernel not in ("gram", "reconstruct"):
                raise ValueError(
                    f"Unknown decomposed_kernel={self._vf_kernel!r}; "
                    "use 'gram' or 'reconstruct'.")
            self._vf_dense_cache: tuple | None = None
            if phi_blocks is None:
                # Gamma-point dense blocks reconstructed from the factors --
                # feeds the (nq-independent) pair index and any Gamma-only
                # consumer; the coupled-q contraction itself stays factored.
                phi_blocks = self._phi_blocks_from_factors(vfactors)

        device_cfg = getattr(config, "device", None)
        if self._n_kpts > 1 and device_cfg is not None:
            # The q-difference index arithmetic ((i-j) mod n), the q -> -q
            # fold and the offline vertices all assume the GAMMA-CENTERED
            # mesh q = k/n. Validate the configured Monkhorst-Pack mesh
            # against it instead of failing silently.
            from quatrex.grid.kpoints import monkhorst_pack

            grid = np.asarray(device_cfg.kpoint_grid, dtype=int)
            shift = np.asarray(device_cfg.kpoint_shift, dtype=float)
            tr = grid > 1
            mesh = np.asarray(monkhorst_pack(grid[tr], shift[tr])) % 1.0
            want = np.stack(
                np.meshgrid(*[np.arange(n) / n for n in grid[tr]],
                            indexing="ij"), axis=-1).reshape(-1, int(tr.sum()))
            # ORDERED comparison: the q-difference/negation index arithmetic
            # requires q_k = k/n at index k, not merely the same point set.
            if mesh.shape != want.shape or not np.allclose(
                mesh % 1.0, want % 1.0, atol=1e-8
            ):
                raise ValueError(
                    "Coupled-q vertices assume the Gamma-centered mesh "
                    "q = k/n per transverse axis; the configured "
                    f"kpoint_grid={tuple(grid)} with kpoint_shift="
                    f"{tuple(shift)} does not produce it. Set "
                    "kpoint_shift[i] = 1/2 - 1/(2*n_i) on the transverse "
                    "axes."
                )

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
        # RGF selected-inversion produces the block-tridiagonal
        # G (diagonal + first off-diagonal, |K-K'| <= 1), so the full
        # off-diagonal ring keeps exactly that band
        self.g_band = 1

        # Precompute the full off-diagonal pair index: for each output
        # block pair (I, J) with |I-J| <= 1, collect the ring quads
        #   (K1, K2, K1', K2', phi_left, phi_right)
        # with phi_left = Phi[(I, K1, K2)], phi_right = Phi[(J, K2', K1')]
        # and the inner G links (K1, K1'), (K2, K2') inside the band.
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

        # Distinct inner G band blocks (K, K') referenced by any quad,
        # that must be gathered to full omega.
        self._g_band_keys: set[tuple[int, int]] = set()
        for quads in self._phi_pair_index.values():
            for K1, K2, K1p, K2p, _pl, _pr in quads:
                self._g_band_keys.add((K1, K1p))
                self._g_band_keys.add((K2, K2p))

        # Cached intermediate tau-domain buffers (length
        # n_fft) for FFTd G
        self._tau_cache: tuple | None = None
        # Cached bosonic-fold plan (local gather perm + neighbour exchange
        # schedule); False = not yet built, None = no pattern, else the tuple.
        self._rev_perm: tuple | None | bool = False
        # Pre-permuted phi factors, built lazily on first compute (after any
        # ballistic zeroing); the FC3 vertex is fixed so this is computed once.
        self._phi_pre: dict | None = None
        self._full_freqs: NDArray | None = None

        # Optional self-consistent SCP cubic-tadpole static self-energy.
        self._scp_tadpole = bool(getattr(config.phonon, "scp_tadpole", False))
        self._sigma_static: NDArray | None = None
        if self._scp_tadpole:
            self._setup_scp_tadpole(config, dynamical_matrix)

    def _phi_blocks_from_factors(self, vf) -> dict:
        """Dense Gamma-point (iq1 = iq2 = 0) vertex blocks reconstructed from
        the factors, over the factor offset window -- feeds the pair index."""
        offs = [int(d) for d in vf.offsets]
        blocks: dict[tuple[int, int, int], np.ndarray] = {}
        for I in range(self.n_blocks):
            for d1 in offs:
                K1 = I + d1
                if not 0 <= K1 < self.n_blocks:
                    continue
                for d2 in offs:
                    K2 = I + d2
                    if not 0 <= K2 < self.n_blocks:
                        continue
                    blocks[(I, K1, K2)] = vf.reconstruct_block(0, 0, d1, d2)
        return blocks

    def _reconstructed_qvertices(self, q_lo: int, q_hi: int, nq: int) -> dict:
        """RANK-LOCAL slice of the dense q-folded vertex dict, reconstructed
        from the factors (decomposed_kernel="reconstruct").

        Only the (q', q2) pairs this rank's external-q slice [q_lo, q_hi)
        touches are materialised, and the per-(d1, d2) offset table is shared
        across the (bulk-homogeneous) block index I -- MBs instead of the
        GB-scale full dict. Built lazily on first compute (after any
        ballistic lambda-zeroing) and cached; the vertex is fixed.
        """
        key = (q_lo, q_hi, nq)
        if self._vf_dense_cache is not None and self._vf_dense_cache[0] == key:
            return self._vf_dense_cache[1]
        vf = self._vfactors
        qdm = self._q_diff_map
        pairs = {(0, 0)}
        for iq_ext in range(q_lo, q_hi):
            for iqp in range(nq):
                iq2 = int(qdm[iq_ext, iqp])
                pairs.add((iqp, iq2))
                pairs.add((iq2, iqp))
        offs = [int(d) for d in vf.offsets]
        pos = vf.offset_index()
        lam_D = vf.D * vf.lambdas[None, :]          # (b, R)
        qv: dict = {}
        for (iq1, iq2) in pairs:
            # One offset-table einsum per pair; blocks are I-independent
            # (bulk vertex, minimum-image offsets), shared across I.
            table = np.einsum(
                "ar,dbr,ecr->deabc", lam_D,
                vf.UB[:, iq1], vf.UC[:, iq2], optimize=True)
            blocks = {}
            for I in range(self.n_blocks):
                for d1 in offs:
                    K1 = I + d1
                    if not 0 <= K1 < self.n_blocks:
                        continue
                    for d2 in offs:
                        K2 = I + d2
                        if not 0 <= K2 < self.n_blocks:
                            continue
                        blocks[(I, K1, K2)] = table[pos[d1], pos[d2]]
            qv[(iq1, iq2)] = blocks
        self._vf_dense_cache = (key, qv)
        return qv

    def _setup_scp_tadpole(self, config, dynamical_matrix) -> None:
        """Prepare the self-consistent SCP cubic-tadpole static self-energy:
        the dense device FC3 + dynamical matrix, the mixing/floor, and the
        zeroed running Sigma_static."""
        from quatrex.phonon.static_self_energy import device_fc3_mass_weighted

        if dynamical_matrix is None:
            warnings.warn(
                "scp_tadpole requested but no dynamical_matrix passed to "
                "SigmaPhononPhonon; SCP tadpole disabled.", stacklevel=2)
            self._scp_tadpole = False
            return
        if not np.all(self.block_sizes == self.block_sizes[0]):
            warnings.warn(
                "scp_tadpole requires uniform block sizes; disabled.",
                stacklevel=2)
            self._scp_tadpole = False
            return
        if ranks.block.size > 1:
            warnings.warn(
                "scp_tadpole is not yet implemented for block-distributed "
                "runs (comm.block.size>1); SCP tadpole disabled. ",
                stacklevel=2)
            self._scp_tadpole = False
            return

        n_dof = int(self.block_sizes[0])
        n_blocks = self.n_blocks
        N_D = n_blocks * n_dof
        # Dense device FC3 in the tadpole mass-weighting
        self._fc3_dev_mw = device_fc3_mass_weighted(
            self.phi_blocks, n_blocks, n_dof)
        # Dense device dynamical matrix D (THz^2), omega-independent.
        D = np.zeros((N_D, N_D), dtype=float)
        for I in range(n_blocks):
            for J in range(max(0, I - 1), min(n_blocks, I + 2)):
                blk = np.asarray(get_host(dynamical_matrix.blocks[I, J]))
                while blk.ndim > 2:
                    # Leading stack / transverse-q axes: take the first
                    # stack entry and the Gamma point (index 0).
                    blk = blk[0]
                D[I * n_dof:(I + 1) * n_dof, J * n_dof:(J + 1) * n_dof] = blk.real
        self._scp_D = 0.5 * (D + D.T)
        self._sigma_static = np.zeros((N_D, N_D), dtype=float)
        self._scp_mix = float(getattr(config.phonon, "scp_static_mixing", 0.1))
        self._scp_floor2 = float(getattr(config.phonon, "scp_floor_thz", 0.5)) ** 2
        if comm.rank == 0:
            print(
                f"[SigmaPhononPhonon] SCP tadpole on: N_D={N_D}, "
                f"mixing={self._scp_mix}, Phi_eff floor="
                f"{getattr(config.phonon, 'scp_floor_thz', 0.5)} THz.",
                flush=True,
            )

    def _apply_scp_tadpole(self, g_lesser, g_greater, sigma_retarded) -> None:
        """Self-consistent cubic tadpole, in the nnz state.

        Forms <uu> from the omega-integral of the device G^{<,>} on the
        one-sided grid: the negative-frequency half enters via the bosonic
        fold G^<_ij(-w) = G^>_ji(w) (w=0 counted once), and the solver's
        occupation-positive G (= -G_textbook) is negated. Solves the
        regularised mean_displacement against Phi_eff = D + Sigma_static,
        mixes the resulting static Sigma_T, and broadcasts it into Sigma^R
        at every frequency.
        """
        from quatrex.phonon.static_self_energy import (
            equal_time_uu_from_sum, mean_displacement, sigma_tadpole)

        N_D = self._sigma_static.shape[0]
        # omega-integral over every stack axis (full omega local per nnz
        # slice); the w=0 bin is excluded from the folded G^> half.
        ax = tuple(range(g_lesser.data.ndim - 1))
        gl_sum_local = g_lesser.data.sum(axis=ax)            # (local_nnz,)
        full_freqs = self._full_frequencies(int(g_lesser.data.shape[0]))
        pos = xp.abs(xp.asarray(full_freqs)) > 1e-12
        gg_sum_local = g_greater.data[pos].sum(axis=ax)
        # Transverse-momentum mesh average (1/N_q).
        nq = int(np.prod(g_lesser.data.shape[1:-1])) or 1
        rows = xp.asarray(g_lesser.rows_nnz)
        cols = xp.asarray(g_lesser.cols_nnz)
        g_sum = xp.zeros((N_D, N_D), dtype=gl_sum_local.dtype)
        g_sum[rows, cols] = gl_sum_local / nq
        gg_sum = xp.zeros((N_D, N_D), dtype=gg_sum_local.dtype)
        gg_sum[rows, cols] = gg_sum_local / nq
        # nnz is split over comm.stack (full omega local, elements split).
        if ranks.stack.size > 1:
            for buf in (g_sum, gg_sum):
                recv = xp.empty_like(buf)
                ranks.stack.all_reduce(xp.ascontiguousarray(buf), recv, op="sum")
                buf[:] = recv
        dw = float(self.local_frequencies[1] - self.local_frequencies[0])
        # Textbook convention (G^< = -i n A) is the NEGATIVE of the solver's
        # occupation-positive storage; the transpose is the ji fold.
        g_total = np.asarray(get_host(-(g_sum + gg_sum.T)))
        uu = equal_time_uu_from_sum(g_total, dw)
        if float(np.trace(uu)) <= 0.0 and comm.rank == 0:
            warnings.warn(
                "SCP tadpole: Tr<uu> <= 0 -- sign-convention violation?",
                stacklevel=2)
        phi_eff = self._scp_D + self._sigma_static
        w_mean = mean_displacement(
            self._fc3_dev_mw, uu, phi_eff,
            omega2_floor_abs=self._scp_floor2)
        sig_new = sigma_tadpole(self._fc3_dev_mw, w_mean)
        self._sigma_static = (
            (1.0 - self._scp_mix) * self._sigma_static + self._scp_mix * sig_new)
        # Broadcast the static self-energy into Sigma^R at every frequency.
        sr_static = xp.asarray(
            self._sigma_static[get_host(rows), get_host(cols)]
        ).astype(sigma_retarded.data.dtype)
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
        """Compute the 3-phonon self-energy contribution

        1. G(omega)->g(tau) by FFT in nnz distribution
        2. g(tau) nnz->stack so each comm.stack rank owns a
           tau-slice of the full band, block-accessible
        3. the off-diagonal ring contraction per tau-slice in stack
        4. sigma(tau) stack->nnz
        5. sigma(tau)->Sigma(omega) by IFFT in nnz

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

    def _compute_fft_first(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        sigma_lesser: DSDBSparse,
        sigma_greater: DSDBSparse,
        sigma_retarded: DSDBSparse,
    ) -> None:
        ne_full = int(g_lesser.global_stack_shape[0])
        nk = tuple(int(k) for k in g_lesser.global_stack_shape[1:])
        nq = int(np.prod(nk)) if len(nk) else 1
        if nq > 1 and self._qvertices is None and self._vfactors is None:
            raise ValueError(
                f"Transverse-q device (mesh {nk}, n_kpts={nq}) requires the "
                "q-folded vertices (config.phonon.qfold_path, see "
                "quatrex.phonon.qfold) or the tensor-decomposed factors "
                "(config.phonon.decomposed_vertices_path)."
            )
        if nq > 1 and self._n_kpts != nq:
            raise ValueError(
                f"q-folded vertices are for n_kpts={self._n_kpts} but the "
                f"Green's function has {nq} transverse momenta {nk}."
            )
        if nq > 1 and ranks.block.size > 1:
            raise NotImplementedError(
                "Transverse-q (k>1) with comm.block.size > 1 is not "
                "supported."
            )
        n_fft = 2 * ne_full - 1
        full_freqs = self._full_frequencies(ne_full)
        prefactor = bubble_prefactor_thz(float(full_freqs[1] - full_freqs[0]))
        if self._vertex_scale != 1.0:
            # Sigma ~ Phi^2 -> lambda^2 on the bubble
            prefactor = prefactor * self._vertex_scale**2
        if self._ramp_n > 0:
            # adiabatic switch-on (config.phonon.sse_ramp_iterations)
            self._ramp_it += 1
            ramp = min(1.0, self._ramp_it / float(self._ramp_n))
            prefactor = prefactor * ramp
            if ranks.rank == 0 and ramp < 1.0:
                print(f"SSE ramp: {ramp:.3f}", flush=True)
        # Coupled-q convolution carries the 1/N_q mesh-average
        if nq > 1:
            prefactor = prefactor / nq

        gtl, gtg, stl, stg, gtlr, gtgr = self._ensure_tau_buffers(g_lesser, n_fft)

        # (1) FFT G(omega)->g(tau) in nnz.
        for m in (gtl, gtg, stl, stg, gtlr, gtgr):
            if m.distribution_state != "nnz":
                m.dtranspose(discard=True)
        gl_in, gg_in = g_lesser.data, g_greater.data
        # The omega=0 bin never enters the bubble (Bose divergence at DC;
        # the zero-measure convention of the theory). The Green's functions
        # stay intact for Dyson/observables; only the copies fed into the
        # 3-phonon convolution are masked, with the matching output mask in
        # step (5).
        sse_mask = xp.abs(xp.asarray(full_freqs)) < 1e-6
        if bool(sse_mask.any()):
            gl_in = gl_in.copy(); gl_in[sse_mask] = 0.0
            gg_in = gg_in.copy(); gg_in[sse_mask] = 0.0
        # NOTE: no IR treatment is applied to the bubble legs -- the device
        # G^< has no 1/omega pole (the bosonic fold and the bounded spectral
        # function force it to cancel). The Bose pole is real only in the
        # lead occupation, where the odd lead broadening Gamma(omega) keeps
        # the injection finite. The bubble is the bare conserving convolution.
        if os.environ.get("QX_DIAG_SPECTRAL") == "1":
            # eta=0 convergence diagnostic: per-omega magnitude of the bubble
            # INPUT G^<, RAW (g_lesser.data) vs WINDOWED/masked (gl_in, what is
            # actually convolved). Full-omega axis here (nnz distribution);
            # rank-local max over nnz is reduced to the global per-omega max
            # over WORLD. Read on rank 0 in engine/run.py. No effect on the math.
            nw = int(gl_in.shape[0])
            graw = np.abs(np.asarray(g_lesser.data)).reshape(nw, -1).max(axis=1)
            gwin = np.abs(np.asarray(gl_in)).reshape(nw, -1).max(axis=1)
            graw = np.ascontiguousarray(graw, dtype=np.float64)
            gwin = np.ascontiguousarray(gwin, dtype=np.float64)
            comm.Allreduce(MPI.IN_PLACE, graw, op=MPI.MAX)
            comm.Allreduce(MPI.IN_PLACE, gwin, op=MPI.MAX)
            self._diag_graw_w = graw
            self._diag_gwin_w = gwin
            self._diag_full_freqs = np.abs(np.asarray(full_freqs).real)
        gtl.data[:] = self._fft_pad(gl_in, n_fft)
        gtg.data[:] = self._fft_pad(gg_in, n_fft)
        # Build the reversed (absorption) legs: DFT index-reversal
        # rev(X)[l] = X[(-l) mod n_fft] of the FFT'd G. The exact bosonic
        # continuation carries the ji-TRANSPOSE (and -q for coupled-q):
        #     G^<_ij(q, -w) = G^>_ji(-q, w).
        # NOTE: the no-transpose shortcut is exact only for the EQUILIBRIUM
        # (complex-symmetric) part of G; skipping the transpose breaks the
        # Phi-derivable energy balance of the bubble off equilibrium
        # (see SCBA._phonon_bubble_energy_balance).
        # The q-negation (middle axes) and the FFT + tau reversal (axis 0)
        # are state-independent and done here in the nnz state; the
        # ji-transpose (nnz/last axis) is applied AFTER the nnz->stack
        # dtranspose below, where the FULL nnz pattern is local on every
        # rank. The transpose commutes with the FFT/reversal (axis 0) and
        # the q-negation (middle axes), so this is exactly the serial fold
        # at any rank count. The q-negation also commutes with the axis-0
        # FFT, so the already-FFT'd decay legs are reused without a second
        # FFT of the reversed source.
        Xl, Xg = gtl.data, gtg.data
        if nq > 1:
            # negate the transverse momentum axes (Gamma-centered IDFT
            # meshes are closed under q -> -q)
            for ax, k in enumerate(nk, start=1):
                neg = (-xp.arange(k)) % k
                Xl = xp.take(Xl, neg, axis=ax)
                Xg = xp.take(Xg, neg, axis=ax)
        gtlr.data[0] = Xl[0]
        gtlr.data[1:] = Xl[:0:-1]
        gtgr.data[0] = Xg[0]
        gtgr.data[1:] = Xg[:0:-1]

        # (2) g(tau) nnz->stack: tau-slice of the full band per stack rank
        gtl.dtranspose()
        gtg.dtranspose()
        gtlr.dtranspose()
        gtgr.dtranspose()
        # ji-transpose of the reversed legs, applied in the "stack" state where
        # the FULL nnz axis is local on every rank (the decay legs gtl/gtg are
        # NOT transposed). The transpose (i,j)->(j,i) is a pure function of the
        # sparsity pattern; with block_comm_size>1 the partner (j,i) of a
        # block-boundary entry lives on a neighbour block rank and is fetched by
        # an immediate-neighbour exchange (BT band => |I-J|<=1). Exact + conserving
        # at any nranks AND any block_comm_size.
        self._fold_reversed_legs(gtlr, gtgr, g_lesser)
        if stl.distribution_state != "stack":
            stl.dtranspose(discard=True)
        if stg.distribution_state != "stack":
            stg.dtranspose(discard=True)
        stl.data[:] = 0.0
        stg.data[:] = 0.0

        gtlv, gtgv = gtl.stack[...], gtg.stack[...]
        gtlrv, gtgrv = gtlr.stack[...], gtgr.stack[...]
        stlv, stgv = stl.stack[...], stg.stack[...]

        # (3) Ring contraction per tau-slice in stack. Under a comm.block
        # split each rank owns outputs (I,J) with min(I,J) in its block
        # window and fetches the off-window band links from its
        # neighbours via a width-N_h halo
        start = int(stl.block_section_offsets[ranks.block.rank])
        end = int(stl.block_section_offsets[ranks.block.rank + 1])
        owned = self._owned_outputs(start, end)

        if ranks.block.size > 1:
            halo_l, halo_g = self._exchange_band_halo(gtlv, gtgv, gtl, start, end)
            halo_lr, halo_gr = self._exchange_band_halo(
                gtlrv, gtgrv, gtlr, start, end)
        else:
            halo_l = halo_g = halo_lr = halo_gr = {}

        # Materialise each distinct band link
        gl_blk: dict[tuple[int, int], NDArray] = {}
        gg_blk: dict[tuple[int, int], NDArray] = {}
        glr_blk: dict[tuple[int, int], NDArray] = {}
        ggr_blk: dict[tuple[int, int], NDArray] = {}
        for (K, Kp) in self._links_for_range(start, end):
            if start <= min(K, Kp) < end:
                gl_blk[(K, Kp)] = gtlv.blocks[K - start, Kp - start]
                gg_blk[(K, Kp)] = gtgv.blocks[K - start, Kp - start]
                glr_blk[(K, Kp)] = gtlrv.blocks[K - start, Kp - start]
                ggr_blk[(K, Kp)] = gtgrv.blocks[K - start, Kp - start]
            else:
                gl_blk[(K, Kp)] = halo_l[(K, Kp)]
                gg_blk[(K, Kp)] = halo_g[(K, Kp)]
                glr_blk[(K, Kp)] = halo_lr[(K, Kp)]
                ggr_blk[(K, Kp)] = halo_gr[(K, Kp)]

        # The full bubble folds the negative-omega contribution
        # into the one-sided grid via G^<(-omega)=G^>(omega): per ring quad
        #   Sigma^<  = ring(g^<_a, g^<_b) + ring(g^<_a, rev g^>_b) + ring(rev g^>_a, g^<_b)
        #   Sigma^>  = ring(g^>_a, g^>_b) + ring(g^>_a, rev g^<_b) + ring(rev g^<_a, g^>_b)
        # The omega/tau batch is parallelised ONCE over the whole
        # (I,J)/phi-pair loop below (single-thread BLAS per chunk), and the
        # fixed phi factors are pre-permuted once -- per (I,J) a list of
        # (K1,K2,K1p,K2p, PL,PR,nI,bK2,nJ) -- to avoid per-call transpose
        # copies.
        if self._phi_pre is None:
            self._phi_pre = {}
            for (I, J), quads in self._phi_pair_index.items():
                self._phi_pre[(I, J)] = [
                    (K1, K2, K1p, K2p) + phi_perms(pl, pr, xp)
                    for (K1, K2, K1p, K2p, pl, pr) in quads
                ]

        def _fold_l(pre, gla, glb, ggra, ggrb):
            PL, PR, nI, bK2, nJ = pre
            return (ring_contract_pre(PL, PR, nI, bK2, nJ, gla, glb, xp)
                    + ring_contract_pre(PL, PR, nI, bK2, nJ, gla, ggrb, xp)
                    + ring_contract_pre(PL, PR, nI, bK2, nJ, ggra, glb, xp))

        def _fold_g(pre, gga, ggb, glra, glrb):
            PL, PR, nI, bK2, nJ = pre
            return (ring_contract_pre(PL, PR, nI, bK2, nJ, gga, ggb, xp)
                    + ring_contract_pre(PL, PR, nI, bK2, nJ, gga, glrb, xp)
                    + ring_contract_pre(PL, PR, nI, bK2, nJ, glra, ggb, xp))

        if nq == 1:
            # Compute Sigma^{<,>}(I,J) for the tau slice [lo:hi]; returns
            # the band blocks for that slice.
            def _contract_tau(lo, hi):
                res = {}
                for (I, J) in owned:
                    acc_l = None
                    acc_g = None
                    for (K1, K2, K1p, K2p, *pre
                         ) in self._phi_pre[(I, J)]:
                        sl = _fold_l(
                            pre,
                            gl_blk[(K1, K1p)][lo:hi], gl_blk[(K2, K2p)][lo:hi],
                            ggr_blk[(K1, K1p)][lo:hi], ggr_blk[(K2, K2p)][lo:hi],
                        )
                        sg = _fold_g(
                            pre,
                            gg_blk[(K1, K1p)][lo:hi], gg_blk[(K2, K2p)][lo:hi],
                            glr_blk[(K1, K1p)][lo:hi], glr_blk[(K2, K2p)][lo:hi],
                        )
                        acc_l = sl if acc_l is None else acc_l + sl
                        acc_g = sg if acc_g is None else acc_g + sg
                    if acc_l is not None:
                        res[(I, J)] = (acc_l, acc_g)
                return res

            n_tau = next(iter(gl_blk.values())).shape[0] if gl_blk else 0
            # Cap the tau-split so each worker keeps >= ~4 tau points: the
            # per-ring contraction scales with threads only while each
            # chunk's BS^3 intermediate stays in cache; shorter chunks
            # regress.
            pool, n_threads = ring_pool()
            nt = min(n_threads, max(1, n_tau // 4))
            if pool is not None and nt > 1:
                bnds = [(i * n_tau // nt, (i + 1) * n_tau // nt) for i in range(nt)]
                chunks = list(pool.map(lambda b: _contract_tau(*b), bnds))
                for (I, J) in owned:
                    parts = [c for c in chunks if (I, J) in c]
                    if parts:
                        stlv.blocks[I - start, J - start] = xp.concatenate(
                            [c[(I, J)][0] for c in parts], axis=0)
                        stgv.blocks[I - start, J - start] = xp.concatenate(
                            [c[(I, J)][1] for c in parts], axis=0)
            else:
                for (I, J), (acc_l, acc_g) in _contract_tau(0, n_tau).items():
                    stlv.blocks[I - start, J - start] = acc_l
                    stgv.blocks[I - start, J - start] = acc_g
        else:
            # Coupled-q convolution.
            qdm = self._q_diff_map
            qv = self._qvertices

            def _qflat(d):
                # (tau, *nk, b, b) -> (tau, N_q, b, b); the q-axis is contiguous
                # in the same C-order as global_stack_shape[1:].
                return {
                    kk: v.reshape(v.shape[0], nq, v.shape[-2], v.shape[-1])
                    for kk, v in d.items()
                }

            gl_q = _qflat(gl_blk)
            gg_q = _qflat(gg_blk)
            glr_q = _qflat(glr_blk)
            ggr_q = _qflat(ggr_blk)
            n_tau = next(iter(gl_q.values())).shape[0]
            dtype = next(iter(gl_q.values())).dtype
            # Distribute the EXTERNAL-q loop over comm.q. The q-folded
            # internal q' Green's functions are kept whole/local on every
            # rank (only the energy axis is split across comm.stack), so each
            # q-rank computes a disjoint subset of iq_ext from the full local
            # q' data -- no internal-q gather -- and the per-rank partial
            # Sigma(q_ext) are summed over comm.q after the loop. This is the
            # dedicated q axis: N_q-way parallelism on top of the energy axis.
            q_lo = ranks.q.rank * nq // ranks.q.size
            q_hi = (ranks.q.rank + 1) * nq // ranks.q.size

            if self._vfactors is not None and self._vf_kernel == "gram":
                self._contract_factored_q(
                    owned, qdm, q_lo, q_hi, nq, nk, n_tau, dtype,
                    gl_q, gg_q, glr_q, ggr_q, stlv, stgv, start, xp,
                )
            else:
                if self._vfactors is not None:
                    # decomposed_kernel="reconstruct": dense path fed the
                    # factor-reconstructed rank-local vertex slice.
                    qv = self._reconstructed_qvertices(q_lo, q_hi, nq)
                self._contract_dense_q(
                    owned, qdm, qv, q_lo, q_hi, nq, nk, n_tau, dtype,
                    gl_q, gg_q, glr_q, ggr_q, _fold_l, _fold_g,
                    stlv, stgv, start, xp,
                )

        # Assemble the external-q distribution: each comm.q rank computed a
        # disjoint subset of iq_ext (others left zero), so sum over comm.q.
        if nq > 1 and ranks.q.size > 1:
            for m in (stl, stg):
                recv = xp.empty_like(m.data)
                ranks.q.all_reduce(xp.ascontiguousarray(m.data), recv, op="sum")
                m.data[:] = recv

        # (4) sigma(tau) stack->nnz
        stl.dtranspose()
        stg.dtranspose()

        # (5) IFFT sigma(tau)->Sigma(omega) in nnz; add into outputs; build Sigma^R.
        sl_data = prefactor * xp.fft.ifft(stl.data, axis=0)[:ne_full]
        sg_data = prefactor * xp.fft.ifft(stg.data, axis=0)[:ne_full]
        # OUTPUT mask, completing the input masking above: the scattering
        # Sigma is NOT applied below the SSE cutoff (transport there stays
        # ballistic), and never at the omega=0 bin (a nonzero Sigma^{<,>}(0)
        # hits the near-singular acoustic G^R(0); the bin carries zero heat
        # anyway).
        if bool(sse_mask.any()):
            sl_data[sse_mask] = 0.0
            sg_data[sse_mask] = 0.0
        # SIGN CONVENTION: fed with this solver's occupation-positive
        # Green's functions (-i G^{<,>} >= 0, the same convention as the
        # lead injection sigma^{<,>} = +i n(+1) gamma), the bubble returns
        # TEXTBOOK-signed sigma^{<,>} (-i sigma^{<,>} <= 0) -- the exact
        # negative of what the Keldysh feedback G^{<,>} = G^R sigma^{<,>} G^A
        # expects (unflipped it injects anti-dissipation), so negate here.
        sigma_lesser.data[:] = sigma_lesser.data - sl_data
        sigma_greater.data[:] = sigma_greater.data - sg_data
        # Sigma^R contribution (from the RAW, textbook-signed values:
        # Gamma = i(sigma^> - sigma^<)_raw >= 0, matching the lead OBC
        # damping sign -- unchanged by the convention flip above).
        if self.retarded_method == "fft":
            delta = sg_data - sl_data
            # transverse_shape: the exact bosonic mirror carries q -> -q on
            # the transverse axes (exact off equilibrium, unlike the plain
            # conjugate shortcut).
            hil = 0.5j * hilbert_transform(delta, full_freqs,
                                           transverse_shape=nk)
            if bool(sse_mask.any()):
                # post-mask the retarded consistently with Sigma^<>
                hil[sse_mask] = 0.0
            sigma_retarded.data[:] = sigma_retarded.data + hil

        # Self-consistent SCP cubic-tadpole static self-energy
        if self._scp_tadpole and self._sigma_static is not None:
            self._apply_scp_tadpole(g_lesser, g_greater, sigma_retarded)

    def _contract_dense_q(
        self, owned, qdm, qv, q_lo, q_hi, nq, nk, n_tau, dtype,
        gl_q, gg_q, glr_q, ggr_q, _fold_l, _fold_g,
        stlv, stgv, start, xp,
    ):
        """DENSE coupled-q vertex-pair contraction (the reference path).

        Consumes the dense q-folded vertex dict ``qv`` and the q-flattened
        tau-domain Green's function band dicts, writes the per-(I, J)
        Sigma^{<,>} tau blocks into the stack views. See
        ``_contract_factored_q`` for the tensor-decomposed equivalent.
        """
        # Resolve the per-(I, J) vertex-pair task list once. The LEFT
        # vertex is CONJUGATED: the bubble at external q pairs
        # Phi(q', q_ext-q')^* with Phi(q_ext-q', q'); the unconjugated
        # pairing breaks momentum bookkeeping (Sigma(-q) != Sigma(q)^T
        # under time reversal). At Gamma the vertices are real, so the
        # Gamma-only (nq==1) path is unaffected.
        # The task list depends only on the (fixed) vertex and mesh, so it
        # is built once and cached; identical (pl, pr) vertex pairs share
        # one pre-permuted copy (the bulk-homogeneous blocks repeat across
        # I and across (iq_ext, iqp) with the same q-difference).
        cache_key = (q_lo, q_hi, nq)
        if getattr(self, "_qtasks_cache_key", None) == cache_key:
            qtasks = self._qtasks_cache
        else:
            perm_cache: dict[tuple[int, int], tuple] = {}
            qtasks = {}
            for iq_ext in range(q_lo, q_hi):
                for iqp in range(nq):
                    iq2 = int(qdm[iq_ext, iqp])
                    phiL = qv.get((iqp, iq2))   # legs (q', q_ext-q')
                    phiR = qv.get((iq2, iqp))   # legs (q_ext-q', q')
                    if phiL is None or phiR is None:
                        continue
                    for (I, J) in owned:
                        for K1, K2, K1p, K2p, _pl, _pr in self._phi_pair_index[(I, J)]:
                            pl = phiL.get((I, K1, K2))
                            pr = phiR.get((J, K2p, K1p))
                            if pl is None or pr is None:
                                continue
                            pkey = (id(pl), id(pr))
                            pre = perm_cache.get(pkey)
                            if pre is None:
                                pre = phi_perms(xp.conj(pl), pr, xp)
                                perm_cache[pkey] = pre
                            qtasks.setdefault((I, J), []).append(
                                (iq_ext, iqp, iq2, K1, K1p, K2, K2p) + pre)
            self._qtasks_cache_key = cache_key
            self._qtasks_cache = qtasks

        # Sigma^{<,>}(I, J, q_ext) for the tau slice [lo:hi]; mirrors the
        # nq==1 _contract_tau so the omega/tau batch parallelises across
        # the ring pool.
        def _contract_tau_q(lo, hi):
            res = {}
            for (I, J), tasks in qtasks.items():
                bs_I = int(self.block_sizes[I])
                bs_J = int(self.block_sizes[J])
                out_l = xp.zeros((hi - lo, nq, bs_I, bs_J), dtype=dtype)
                out_g = xp.zeros((hi - lo, nq, bs_I, bs_J), dtype=dtype)
                for (iq_ext, iqp, iq2, K1, K1p, K2, K2p, *pre) in tasks:
                    out_l[:, iq_ext] += _fold_l(
                        pre,
                        gl_q[(K1, K1p)][lo:hi, iqp],
                        gl_q[(K2, K2p)][lo:hi, iq2],
                        ggr_q[(K1, K1p)][lo:hi, iqp],
                        ggr_q[(K2, K2p)][lo:hi, iq2],
                    )
                    out_g[:, iq_ext] += _fold_g(
                        pre,
                        gg_q[(K1, K1p)][lo:hi, iqp],
                        gg_q[(K2, K2p)][lo:hi, iq2],
                        glr_q[(K1, K1p)][lo:hi, iqp],
                        glr_q[(K2, K2p)][lo:hi, iq2],
                    )
                res[(I, J)] = (out_l, out_g)
            return res

        def _write_q(I, J, out_l, out_g):
            bs_I = int(self.block_sizes[I])
            bs_J = int(self.block_sizes[J])
            blk_shape = (n_tau,) + tuple(nk) + (bs_I, bs_J)
            stlv.blocks[I - start, J - start] = out_l.reshape(blk_shape)
            stgv.blocks[I - start, J - start] = out_g.reshape(blk_shape)

        pool, n_threads = ring_pool()
        nt = min(n_threads, max(1, n_tau // 4))  # >=~4 tau/thread (see nq==1)
        if pool is not None and xp is np and nt > 1:
            bnds = [(i * n_tau // nt, (i + 1) * n_tau // nt)
                    for i in range(nt)]
            chunks = list(pool.map(lambda b: _contract_tau_q(*b), bnds))
            for (I, J) in qtasks:
                _write_q(
                    I, J,
                    xp.concatenate([c[(I, J)][0] for c in chunks], axis=0),
                    xp.concatenate([c[(I, J)][1] for c in chunks], axis=0),
                )
        else:
            for (I, J), (out_l, out_g) in _contract_tau_q(0, n_tau).items():
                _write_q(I, J, out_l, out_g)

    def _contract_factored_q(
        self, owned, qdm, q_lo, q_hi, nq, nk, n_tau, dtype,
        gl_q, gg_q, glr_q, ggr_q, stlv, stgv, start, xp,
    ):
        """Coupled-q contraction with the TENSOR-DECOMPOSED vertex.

        Exact factored equivalent of ``_contract_dense_q`` (same fold terms,
        same left-vertex conjugation -- applied to the row factors inside the
        Grams, see ``quatrex.phonon.bubble_factored``): per (g-variant, band
        link, offset pair, q) one skinny Gram replaces the per-(q', q2)
        vertex-pair triple-GEMMs; the q'-convolution runs entrywise on the
        rank-R Gram tables and a small sandwich restores the band block.
        """
        from quatrex.phonon.bubble_factored import contract_tau_q_factored

        vf = self._vfactors
        quads_by_pair = {
            (I, J): [
                (K1, K2, K1p, K2p)
                for (K1, K2, K1p, K2p, _pl, _pr) in self._phi_pair_index[(I, J)]
            ]
            for (I, J) in owned
            if (I, J) in self._phi_pair_index
        }
        off_pos = vf.offset_index()
        # lambda folded into the external leg ONCE per vertex: the sandwich
        # is Dt @ H @ Dt^T with Dt = D diag(lambda) (D, lambda real).
        Dt = xp.asarray(vf.D * vf.lambdas[None, :])
        UB = xp.asarray(vf.UB)
        UC = UB if vf.UB is vf.UC else xp.asarray(vf.UC)
        g_dicts = {"l": gl_q, "g": gg_q, "lr": glr_q, "gr": ggr_q}
        shared = str(vf.ansatz).upper() == "INDSCAL"

        def _run(lo, hi):
            return contract_tau_q_factored(
                quads_by_pair, self.block_sizes, qdm, q_lo, q_hi, nq,
                g_dicts, Dt, UB, UC, off_pos, lo, hi, xp, shared, dtype,
            )

        def _write(I, J, out_l, out_g):
            bs_I = int(self.block_sizes[I])
            bs_J = int(self.block_sizes[J])
            blk_shape = (n_tau,) + tuple(nk) + (bs_I, bs_J)
            stlv.blocks[I - start, J - start] = out_l.reshape(blk_shape)
            stgv.blocks[I - start, J - start] = out_g.reshape(blk_shape)

        pool, n_threads = ring_pool()
        nt = min(n_threads, max(1, n_tau // 4))  # >=~4 tau/thread (see nq==1)
        if pool is not None and xp is np and nt > 1:
            bnds = [(i * n_tau // nt, (i + 1) * n_tau // nt) for i in range(nt)]
            chunks = list(pool.map(lambda b: _run(*b), bnds))
            for (I, J) in quads_by_pair:
                _write(
                    I, J,
                    xp.concatenate([c[(I, J)][0] for c in chunks], axis=0),
                    xp.concatenate([c[(I, J)][1] for c in chunks], axis=0),
                )
        else:
            for (I, J), (out_l, out_g) in _run(0, n_tau).items():
                _write(I, J, out_l, out_g)

    def _fold_reversed_legs(self, gtlr, gtgr, g: DSDBSparse) -> None:
        """Apply the exact bosonic ji-transpose ``X_ij -> X_ji`` to the reversed
        (absorption) legs in place, at ANY ``comm.block`` size.

        On the local nnz axis the partner ``(j,i)`` of an entry ``(i,j)`` is
        stored locally unless ``(i,j)`` straddles a block-rank boundary, in which
        case ``(j,i)`` lives on the immediate neighbour block rank (BT band =>
        ``|I-J| <= 1``). The pattern-only plan (a local gather permutation plus a
        neighbour request/reply schedule) is built once; per call only the
        boundary entries are exchanged.
        """
        plan = self._build_fold_plan(g)
        if plan is None:  # serial-equivalent (single nnz block / no pattern)
            return
        local_perm, recv_dest, send_src = plan
        for leg in (gtlr, gtgr):
            data = leg.data
            folded = data[..., local_perm]
            for nbr in sorted(set(recv_dest) | set(send_src)):
                sbuf = xp.ascontiguousarray(data[..., send_src[nbr]]) \
                    if nbr in send_src else xp.empty((0,), dtype=data.dtype)
                rshape = data.shape[:-1] + (recv_dest[nbr].size,)
                rbuf = xp.empty(rshape, dtype=data.dtype) if nbr in recv_dest \
                    else xp.empty((0,), dtype=data.dtype)
                self._fold_bcomm.Sendrecv(sbuf, dest=nbr, sendtag=7,
                                          recvbuf=rbuf, source=nbr, recvtag=7)
                if nbr in recv_dest:
                    folded[..., recv_dest[nbr]] = rbuf
            data[:] = folded

    def _build_fold_plan(self, g: DSDBSparse):
        """Pattern-only schedule for :meth:`_fold_reversed_legs` (cached).

        Returns ``(local_perm, recv_dest, send_src)`` where ``local_perm`` gathers
        the locally-resolvable ``(j,i)`` partners (boundary slots point at 0 and
        are overwritten by the exchange), ``recv_dest[nbr]`` are the local nnz
        positions filled from neighbour ``nbr``, and ``send_src[nbr]`` are the
        local nnz positions sent to ``nbr`` (in the order it requests them).
        """
        if self._rev_perm is not False:
            return self._rev_perm
        rows = getattr(g, "rows", None)
        cols = getattr(g, "cols", None)
        if rows is None or cols is None:
            self._rev_perm = None
            return None
        off = int(getattr(g, "global_block_offset", 0))
        r = (np.asarray(rows.get() if hasattr(rows, "get") else rows) + off)
        c = (np.asarray(cols.get() if hasattr(cols, "get") else cols) + off)
        lut = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(r, c))}
        local_perm = np.zeros(r.size, dtype=np.int64)
        want = {}  # nbr -> list of (dest_local_idx, requested (j,i) key)
        for k, (i, j) in enumerate(zip(r, c)):
            src = lut.get((int(j), int(i)))
            if src is not None:
                local_perm[k] = src
            else:
                nbr = self._block_owner(int(j), g)
                want.setdefault(nbr, []).append((k, (int(j), int(i))))
        recv_dest, send_src = {}, {}
        if want or ranks.block.size > 1:
            bcomm = ranks.block._mpi_comm
            self._fold_bcomm = bcomm
            rank, size = ranks.block.rank, ranks.block.size
            for nbr in (rank - 1, rank + 1):
                if not (0 <= nbr < size):
                    continue
                my_keys = [key for _, key in want.get(nbr, [])]
                their_keys = bcomm.sendrecv(my_keys, dest=nbr, sendtag=5,
                                            source=nbr, recvtag=5)
                if nbr in want:
                    recv_dest[nbr] = np.array([d for d, _ in want[nbr]],
                                              dtype=np.int64)
                # reply with MY entry (a,b) the neighbour requested as its (j,i)
                if their_keys:
                    src_idx = []
                    for (a, b) in their_keys:
                        s = lut.get((int(a), int(b)))
                        if s is None:
                            raise RuntimeError(
                                "SSE bosonic fold: neighbour requested a "
                                f"transpose partner ({a},{b}) not stored locally "
                                "-- the ji-band reaches beyond the immediate "
                                "block neighbour; reduce comm.block.size.")
                        src_idx.append(s)
                    send_src[nbr] = np.array(src_idx, dtype=np.int64)
        # every boundary slot must be covered by exactly one neighbour recv
        covered = sum(d.size for d in recv_dest.values())
        missing = [k for k, (i, j) in enumerate(zip(r, c))
                   if lut.get((int(j), int(i))) is None]
        if len(missing) != covered:
            raise RuntimeError(
                "SSE bosonic fold: %d boundary partners unresolved by the "
                "neighbour exchange (covered %d)." % (len(missing), covered))
        local_perm = xp.asarray(local_perm)
        recv_dest = {nbr: xp.asarray(v) for nbr, v in recv_dest.items()}
        send_src = {nbr: xp.asarray(v) for nbr, v in send_src.items()}
        self._rev_perm = (local_perm, recv_dest, send_src)
        return self._rev_perm

    def _block_owner(self, orbital_row: int, g: DSDBSparse) -> int:
        """``comm.block`` rank owning the block that contains global ORBITAL
        row ``orbital_row`` (``block_section_offsets`` is in block units, so map
        it through the orbital block-offsets first)."""
        bso = np.asarray(g.block_section_offsets)
        orbital_bounds = self.block_offsets[bso]
        return int(np.searchsorted(orbital_bounds, orbital_row, side="right") - 1)

    @staticmethod
    def _fft_pad(data: NDArray, n_fft: int) -> NDArray:
        """Zero-pad along the energy axis (0) to n_fft and FFT."""
        pad_shape = (n_fft - data.shape[0],) + tuple(data.shape[1:])
        padded = xp.concatenate(
            [data, xp.zeros(pad_shape, dtype=data.dtype)], axis=0
        )
        return xp.fft.fft(padded, axis=0)

    def _links_for_range(self, a: int, b: int) -> set[tuple[int, int]]:
        """Distinct inner G band links needed by outputs owned
         by the block range [a, b)"""
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
        from the immediate comm.block neighbours via a width-N_h halo.
        """
        bcomm = ranks.block._mpi_comm
        rank, size = ranks.block.rank, ranks.block.size
        offs = ref.block_section_offsets
        nloc_tau = int(ref.data.shape[0])  # local tau count (stack state)

        def is_local(link):
            return start <= min(link) < end

        def shape(link):
            K, Kp = link
            return (nloc_tau, int(self.block_sizes[K]), int(self.block_sizes[Kp]))

        halo_l: dict[tuple[int, int], NDArray] = {}
        halo_g: dict[tuple[int, int], NDArray] = {}
        reqs = []
        sendbufs = []

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
        # neighbour
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
        """Lazily allocate the four n_fft tau-buffers sharing G's pattern.
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
            for _ in range(6)
        )
        gt_rows, gt_cols = bufs[0].spy()
        if not (
            np.array_equal(np.asarray(rows), np.asarray(gt_rows))
            and np.array_equal(np.asarray(cols), np.asarray(gt_cols))
        ):
            raise RuntimeError(
                "tau-buffer sparsity does not match G; cannot FFT raw data."
            )
        self._tau_cache = (key, bufs)
        return bufs

    # ------------------------------------------------------------------
    # block-bubble FFT contraction
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
        """FFT 3-phonon bubble for one block triplet pair (THz^2)."""
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

    def _full_frequencies(self, ne_full: int) -> NDArray:
        """Full (cached) frequency grid, all-gathers the local slice."""
        if self._full_freqs is not None and self._full_freqs.shape[0] == ne_full:
            return self._full_freqs
        if ranks.stack.size == 1:
            freqs = xp.asarray(self.local_frequencies, dtype=float)
        else:
            freqs = ranks.stack.all_gather_v(
                xp.asarray(self.local_frequencies, dtype=float), axis=0
            )
        if int(freqs.shape[0]) != int(ne_full):
            raise ValueError(
                f"Phonon frequency grid ({int(freqs.shape[0])} pts) does not "
                f"match the Green's-function energy grid ({int(ne_full)} "
                "pts). The bubble prefactor carries this grid's spacing, so "
                "a mismatch silently misscales Sigma -- refusing to run. "
                "Regenerate phonon_energies.npy to match the configured "
                "energy window (write_config.py does this) or pass the "
                "solver grid."
            )
        # The FFT bin arithmetic (forward convolution bin n = k + m and the
        # index-reversal fold bin l -> -l) represents omega sums only on a
        # UNIFORM grid ANCHORED AT ZERO; anything else runs silently shifted.
        if int(freqs.shape[0]) > 1:
            df = float(freqs[1] - freqs[0])
            if abs(float(freqs[0])) > 1e-9 * abs(df) or bool(
                xp.max(xp.abs(xp.diff(freqs) - df)) > 1e-9 * abs(df)
            ):
                raise ValueError(
                    "The bubble FFT requires a uniform frequency grid "
                    f"starting at 0 (got start {float(freqs[0]):g}, spacing "
                    f"{df:g}). Set energy_window_min = 0."
                )
        self._full_freqs = freqs
        return freqs
