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
from time import perf_counter

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, xp
from qttools.comm import comm as ranks
from qttools.datastructures import DSDBSparse
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import free_mempool, get_host
from qttools.utils.mpi_utils import get_section_sizes

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
from quatrex.phonon.experimental.frequency_grid import (
    AuxiliaryGrid,
    make_auxiliary_grid,
)
from quatrex.phonon.microblocks import (
    MicroblockLayout,
    build_micro_pair_index,
    micro_link_views,
)
from quatrex.phonon.contraction_support import ContractionSupportMixin
from quatrex.phonon.q_contraction import QContractionMixin
from quatrex.phonon.units import bubble_prefactor_thz

profiler = Profiler()


class SigmaPhononPhonon(
    QContractionMixin, ContractionSupportMixin, ScatteringSelfEnergy
):
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
        qfold: tuple | None = None,
        vfactors=None,
    ) -> None:
        self.local_frequencies = np.asarray(phonon_frequencies)

        self.block_sizes = np.asarray(block_sizes, dtype=int)
        self.n_blocks = int(self.block_sizes.shape[0])
        self.block_offsets = np.concatenate(([0], np.cumsum(self.block_sizes)))
        micro_dof = int(
            getattr(config.phonon, "sse_microblock_dof", 0) or 0)
        micro_band = int(
            getattr(config.phonon, "sse_microblock_g_band", 0) or 0)
        if bool(micro_dof) != bool(micro_band):
            raise ValueError(
                "sse_microblock_dof and sse_microblock_g_band must either "
                "both be zero or both be positive."
            )
        self._micro_layout = (
            MicroblockLayout.from_block_sizes(self.block_sizes, micro_dof)
            if micro_dof else None
        )
        self._vertex_n_blocks = (
            self._micro_layout.n_microblocks
            if self._micro_layout is not None else self.n_blocks
        )

        # Ring-contraction thread pool: config option, env vars as fallback.
        configure_ring_pool(
            threads=int(getattr(config.phonon, "sse_ring_threads", 0)),
            min_w=getattr(config.phonon, "sse_ring_min_w", None),
            workspaces=bool(getattr(config.phonon, "sse_ring_workspaces",
                                    False)),
        )
        self._tau_min_chunk = int(
            getattr(config.phonon, "sse_tau_min_chunk", 4))
        self._tau_chunk_bytes = int(config.phonon.sse_tau_chunk_bytes)
        self._release_legs = bool(
            getattr(config.phonon, "sse_release_leg_blocks", False))
        self._perm_share = str(
            getattr(config.phonon, "sse_perm_cache_share", "off") or "off")
        self._pool_scope = str(
            getattr(config.phonon, "sse_pool_scope", "tau"))
        self._ring_c64 = (
            getattr(config.phonon, "sse_ring_dtype", "complex128")
            == "complex64")
        self._g_from_l = bool(
            getattr(config.phonon, "sse_greater_from_lesser", False))
        self._fold_verify = int(
            getattr(config.phonon, "sse_fold_verify_iterations", 0))
        self._fold_verify_done = 0
        self._herm_pairs = bool(
            getattr(config.phonon, "sse_hermitian_pairs", False))
        retarded_method = getattr(config.phonon, "retarded_method", "fft")
        if retarded_method not in ("half", "fft"):
            raise ValueError(
                f"Unknown retarded_method={retarded_method!r}; "
                "use 'half' or 'fft'."
            )
        self.retarded_method = retarded_method
        _ps = getattr(config.phonon, "pole_sector", None)
        self._pole_injection = None
        if _ps is not None and _ps.enabled:
            from quatrex.phonon.experimental.pole.bubble_state import (
                PoleBubbleState,
            )
            self._pole_injection = PoleBubbleState()
        self._aux_spacing = float(
            getattr(config.phonon, "sse_aux_grid_dw_thz", 0.0) or 0.0)
        self._aux_max_frequency = float(
            getattr(config.phonon, "sse_aux_grid_fmax_thz", 0.0) or 0.0)
        self._aux_restriction = str(
            getattr(config.phonon, "sse_aux_restrict", "adjoint"))
        if self._aux_restriction not in ("adjoint", "sample"):
            raise ValueError(
                f"Unknown sse_aux_restrict={self._aux_restriction!r}; "
                "use 'adjoint' or 'sample'.")
        self._aux_grid: AuxiliaryGrid | None = None

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
            if (self._micro_layout is None
                    and np.unique(self.block_sizes).size != 1):
                raise ValueError(
                    "The factored coupled-q SSE requires uniform block "
                    f"sizes; got {np.unique(self.block_sizes)}."
                )
            vertex_dof = (
                self._micro_layout.micro_dof
                if self._micro_layout is not None
                else int(self.block_sizes[0])
            )
            if vertex_dof != int(vfactors.D.shape[0]):
                raise ValueError(
                    f"Factored vertex n_dof={int(vfactors.D.shape[0])} does "
                    f"not match the SSE vertex block size {vertex_dof}."
                )
            if np.iscomplexobj(vfactors.D) or np.iscomplexobj(vfactors.lambdas):
                raise ValueError(
                    "The external-leg factors D and lambdas must be real."
                )
            self._vfactors = vfactors
            self._q_diff_map = np.asarray(vfactors.q_diff_map, dtype=int)
            self._n_kpts = int(vfactors.n_kpts)
            self._vf_kernel = str(getattr(
                config.phonon, "decomposed_kernel", "reconstruct"))
            if self._vf_kernel not in ("gram", "reconstruct"):
                raise ValueError(
                    f"Unknown decomposed_kernel={self._vf_kernel!r}; "
                    "use 'gram' or 'reconstruct'.")
            self._vf_dense_cache: tuple | None = None
            if phi_blocks is None:
                phi_blocks = self._phi_blocks_from_factors(vfactors)

        device_cfg = getattr(config, "device", None)
        if self._n_kpts > 1 and device_cfg is not None:
            from quatrex.grid.kpoints import monkhorst_pack

            grid = np.asarray(device_cfg.kpoint_grid, dtype=int)
            shift = np.asarray(device_cfg.kpoint_shift, dtype=float)
            tr = grid > 1
            mesh = np.asarray(monkhorst_pack(grid[tr], shift[tr]))
            want = np.stack(
                np.meshgrid(*[np.arange(n) / n for n in grid[tr]],
                            indexing="ij"), axis=-1).reshape(-1, int(tr.sum()))
            residual = (mesh - want + 0.5) % 1.0 - 0.5
            if mesh.shape != want.shape or not np.allclose(
                residual, 0.0, atol=1e-6
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
                block_sizes=(
                    np.full(self._vertex_n_blocks,
                            self._micro_layout.micro_dof, dtype=int)
                    if self._micro_layout is not None else self.block_sizes
                ),
                truncation_warn=config.phonon.phonon_phonon_truncation_warn,
            )

        self.phi_blocks = phi_blocks

        # Retain the Green-function band requested by the vertex layout.
        self.g_band = min(
            (micro_band if self._micro_layout is not None else
             int(getattr(config.phonon, "sse_g_band", 1) or 1)),
            self._vertex_n_blocks - 1,
        )
        # Grouped layouts retain wider primitive support in adjacent blocks.
        self._solver_g_band = (
            min(1, self.n_blocks - 1)
            if self._micro_layout is not None else self.g_band
        )
        self._dense_q_batched = bool(
            getattr(config.phonon, "sse_dense_q_batched", True))
        if self._solver_g_band > 1 and ranks.block.size > 1:
            min_local = int(
                min(get_section_sizes(self.n_blocks, ranks.block.size)[0])
            )
            if min_local < self.g_band + 1:
                raise ValueError(
                    f"sse_g_band={self._solver_g_band} with block_comm_size="
                    f"{ranks.block.size} leaves a comm.block rank with only "
                    f"{min_local} blocks (need >= {self.g_band + 1} so the "
                    "band halo spans only immediate neighbours); reduce "
                    "block_comm_size."
                )

        if self._micro_layout is not None:
            unsupported = []
            if self._g_from_l:
                unsupported.append("sse_greater_from_lesser")
            if self._herm_pairs:
                unsupported.append("sse_hermitian_pairs")
            if self._ring_c64:
                unsupported.append("sse_ring_dtype=complex64")
            if unsupported:
                raise NotImplementedError(
                    "The primitive-microblock SSE currently requires the "
                    "plain complex128 six-ring path; disable "
                    + ", ".join(unsupported)
                )

        self._phi_pair_index: dict[
            tuple[int, int],
            list[tuple[int, int, int, int, NDArray, NDArray]],
        ] = {}
        self._micro_pair_index = None
        if self._micro_layout is not None:
            (self._micro_pair_index, self._vertex_span,
             self._sigma_micro_span) = build_micro_pair_index(
                self.phi_blocks, self._micro_layout, self.g_band)
            if ranks.rank == 0:
                print(
                    "[SigmaPhononPhonon] primitive microblocks: "
                    f"groups={self._micro_layout.cells_per_block}, "
                    f"d={self._micro_layout.micro_dof}, G band={self.g_band}, "
                    f"FC3 span={self._vertex_span}, generated Sigma "
                    f"span={self._sigma_micro_span}, factor rank="
                    f"{getattr(self._vfactors, 'rank', 0) or 'dense'}",
                    flush=True,
                )
        else:
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

        self._g_band_keys: set[tuple[int, int]] = set()
        if self._micro_pair_index is not None:
            for pairs in self._micro_pair_index.values():
                for quads in pairs.values():
                    for quad in quads:
                        self._g_band_keys.add(
                            self._micro_layout.grouped_pair(
                                quad.k1, quad.k1p))
                        self._g_band_keys.add(
                            self._micro_layout.grouped_pair(
                                quad.k2, quad.k2p))
        else:
            for quads in self._phi_pair_index.values():
                for K1, K2, K1p, K2p, _pl, _pr in quads:
                    self._g_band_keys.add((K1, K1p))
                    self._g_band_keys.add((K2, K2p))

        self._tau_cache: tuple | None = None
        self._rev_perm: tuple | bool = False
        self._phi_pre: dict | None = None
        self._full_freqs: NDArray | None = None

    def _phi_blocks_from_factors(self, vf) -> dict:
        """Dense Gamma-point (iq1 = iq2 = 0) vertex blocks reconstructed from
        the factors, over the factor offset window -- feeds the pair index.

        With ``support_pairs`` in the factor metadata (builder option
        ``--decompose-support dense``) only the (d1, d2) offset pairs the
        DENSE FC3 actually populates are emitted: the full offs x offs
        window otherwise manufactures vertex blocks the force constants do
        not contain, weighted only by the global fit residual (the
        "vertex-support asymmetry" -- severe for offset-diagonal vertices).
        """
        offs = [int(d) for d in vf.offsets]
        support = vf.meta.get("support_pairs")
        if support is not None:
            support = {(int(a), int(b)) for a, b in support}
        blocks: dict[tuple[int, int, int], np.ndarray] = {}
        for I in range(self._vertex_n_blocks):
            for d1 in offs:
                K1 = I + d1
                if not 0 <= K1 < self._vertex_n_blocks:
                    continue
                for d2 in offs:
                    K2 = I + d2
                    if not 0 <= K2 < self._vertex_n_blocks:
                        continue
                    if support is not None and (d1, d2) not in support:
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
        support = vf.meta.get("support_pairs")
        if support is not None:
            support = {(int(a), int(b)) for a, b in support}
        lam_D = vf.D * vf.lambdas[None, :]          # (b, R)
        qv: dict = {}
        for (iq1, iq2) in pairs:
            table = np.einsum(
                "ar,dbr,ecr->deabc", lam_D,
                vf.UB[:, iq1], vf.UC[:, iq2], optimize=True)
            blocks = {}
            for I in range(self._vertex_n_blocks):
                for d1 in offs:
                    K1 = I + d1
                    if not 0 <= K1 < self._vertex_n_blocks:
                        continue
                    for d2 in offs:
                        K2 = I + d2
                        if not 0 <= K2 < self._vertex_n_blocks:
                            continue
                        if support is not None and (d1, d2) not in support:
                            continue
                        blocks[(I, K1, K2)] = table[pos[d1], pos[d2]]
            qv[(iq1, iq2)] = blocks
        self._vf_dense_cache = (key, qv)
        return qv


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

    @profiler.profile(
        label="SigmaPhononPhonon.linearized", level="default", comm=comm
    )
    def compute_linearized(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        dg_lesser: DSDBSparse,
        dg_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
    ) -> None:
        """Exact directional derivative of the bubble: the mixed-leg form.

        The bubble is a homogeneous quadratic map S(G) = B(G, G) with B
        bilinear in the two internal legs, so its Frechet derivative in
        the direction dG is the cut-line (2PI-kernel) contraction

            dSigma = B(dG, G) + B(G, dG),

        evaluated here through the SAME FFT/fold/ring pipeline as
        :meth:`compute`, with each ring replaced by its two mixed-leg
        variants. Algebraically identical to the polarisation identity
        S(G+dG) - S(G) - S(dG), but with no subtraction of large terms:
        uniformly exact to rounding even for directions that are small
        or nearly annihilated by the kernel.

        Gamma-only (nq == 1), single block rank, plain 6-ring path (the
        symmetry fast paths fuse legs across the (I,J) mirror and do not
        commute with the asymmetric cross). Does not advance the
        adiabatic-ramp counter; the current ramp prefactor is reused.
        Future work: coupled-q cross, fast-path product rules, the
        adjoint (VJP) pipeline.

        Parameters
        ----------
        g_lesser, g_greater
            The frozen Green's function (the linearisation point).
        dg_lesser, dg_greater
            The direction. Must share G's sparsity pattern.
        out
            ``(dsigma_lesser, dsigma_greater, dsigma_retarded)``.
        """
        dsigma_lesser, dsigma_greater, dsigma_retarded = out

        all_bufs = (
            g_lesser, g_greater, dg_lesser, dg_greater,
            dsigma_lesser, dsigma_greater, dsigma_retarded,
        )
        incoming_states = {id(m): m.distribution_state for m in all_bufs}

        for m in all_bufs:
            if m.distribution_state != "nnz":
                m.dtranspose()

        try:
            self._compute_fft_first(
                g_lesser, g_greater,
                dsigma_lesser, dsigma_greater, dsigma_retarded,
                dg_lesser=dg_lesser, dg_greater=dg_greater,
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
        dg_lesser: DSDBSparse | None = None,
        dg_greater: DSDBSparse | None = None,
    ) -> None:
        ne_full = int(g_lesser.global_stack_shape[0])
        nk = tuple(int(k) for k in g_lesser.global_stack_shape[1:])
        nq = int(np.prod(nk)) if len(nk) else 1
        q_split = (nq > 1
                   and getattr(g_lesser, "q_section_offsets", None) is not None)
        if nq > 1 and self._qvertices is None and self._vfactors is None:
            raise ValueError(
                f"Transverse-q device (mesh {nk}, n_kpts={nq}) requires the "
                "q-folded vertices (config.phonon.qfold_path, see "
                "quatrex.phonon.qfold) or the tensor-decomposed factors "
                "(config.phonon.decomposed_vertices_path)."
            )
        if (nq > 1 or self._vfactors is not None) and self._n_kpts != nq:
            raise ValueError(
                f"The vertices are for n_kpts={self._n_kpts} but the Green's "
                f"function has {nq} transverse momenta {nk}."
            )
        if nq > 1 and self._vfactors is not None:
            if tuple(self._vfactors.nk_shape) != nk:
                raise ValueError(
                    "The decomposed vertices are for a transverse mesh "
                    f"{tuple(self._vfactors.nk_shape)} but the Green's function "
                    f"has {nk}."
                )

        # Linearized (mixed-leg cross) mode: dG legs provided.
        lin = dg_lesser is not None
        if lin:
            if nq > 1 or self._vfactors is not None:
                raise NotImplementedError(
                    "compute_linearized supports the Gamma-only dense-"
                    "vertex path (nq == 1, no factored vertices)."
                )
            if ranks.block.size > 1:
                raise NotImplementedError(
                    "compute_linearized requires block_comm_size == 1."
                )

        full_freqs = self._full_frequencies(ne_full)
        aux_on = self._aux_spacing > 0.0
        if aux_on:
            aux_grid = self._get_auxiliary_grid(full_freqs)
            conv_freqs = aux_grid.frequencies
            ne_conv = int(conv_freqs.shape[0])
            prefactor = bubble_prefactor_thz(self._aux_spacing)
        else:
            aux_grid = None
            conv_freqs = full_freqs
            ne_conv = ne_full
            prefactor = bubble_prefactor_thz(
                float(full_freqs[1] - full_freqs[0]))
        n_fft = 2 * ne_conv - 1
        # Coupled-q convolution carries the 1/N_q mesh-average
        if nq > 1:
            prefactor = prefactor / nq

        _gram_now = (self._vfactors is not None
                     and self._vf_kernel == "gram")
        if self._g_from_l and _gram_now:
            raise ValueError(
                "sse_greater_from_lesser supports the DENSE kernels only "
                "(no decomposed gram kernel)."
            )
        if self._g_from_l and nq > 1 and self._vfactors is not None:
            raise ValueError(
                "sse_greater_from_lesser at nq > 1 requires the explicit "
                "dense q-folded vertex (qfold); the factor-reconstructed "
                "vertex is not audited for the reality condition "
                "Phi(-q1,-q2) = conj(Phi(q1,q2)) the fold relies on."
            )
        verify_now = self._g_from_l and self._fold_verify_done < self._fold_verify
        if verify_now and ranks.size > 1:
            if ranks.rank == 0 and self._fold_verify_done == 0:
                warnings.warn(
                    "sse_fold_verify_iterations requires a single-rank run; "
                    "skipping the in-process gate (verify with a run-level "
                    "A/B against sse_greater_from_lesser=false instead)."
                )
            verify_now = False
            self._fold_verify_done = self._fold_verify
        fast_now = self._g_from_l and not verify_now
        if self._herm_pairs and (nq > 1 or _gram_now):
            raise ValueError(
                "sse_hermitian_pairs supports the Gamma-only DENSE kernel "
                "only (nq == 1, no decomposed gram kernel)."
            )
        if self._herm_pairs and ranks.size > 1:
            raise NotImplementedError(
                "sse_hermitian_pairs needs the full local nnz pattern for "
                "the mirror masks; single-rank runs only."
            )
        halve_now = self._herm_pairs and not verify_now
        if lin:
            fast_now = halve_now = verify_now = False
        if q_split:
            fast_now = False

        bufs = self._ensure_tau_buffers(g_lesser, n_fft, with_dg=lin)
        gtl, gtg, stl, stg, gtlr, gtgr = bufs[:6]
        dgtl = dgtg = dgtlr = dgtgr = None
        if lin:
            dgtl, dgtg, dgtlr, dgtgr = bufs[6:10]
        stx = gtlr  # fast mode: cross-term accumulator in the gtlr slot

        # (1) FFT G(omega)->g(tau) in nnz.
        with profiler.profile_range(
            "PhPh SSE: 1 FFT G->tau", level="default", comm=comm,
        ):
            for m in (gtl, gtg, stl, stg, gtlr, gtgr):
                if m.distribution_state != "nnz":
                    m.dtranspose(discard=True)
            gl_in, gg_in = g_lesser.data, g_greater.data
            pole = self._pole_injection
            if pole is not None and pole.channel is not None:
                p_l, p_g = pole.channel
                gl_in = gl_in - p_l
                gg_in = gg_in - p_g
            conv_mask = xp.abs(xp.asarray(conv_freqs)) < 1e-6
            out_mask = (conv_mask if not aux_on
                        else xp.abs(xp.asarray(full_freqs)) < 1e-6)
            if aux_on:
                if bool(out_mask.any()):
                    gl_in = gl_in.copy(); gl_in[out_mask] = 0.0
                    gg_in = gg_in.copy(); gg_in[out_mask] = 0.0
                gl_in = aux_grid.interpolate(gl_in)
                gg_in = aux_grid.interpolate(gg_in)
                if bool(conv_mask.any()):
                    gl_in[conv_mask] = 0.0
                    gg_in[conv_mask] = 0.0
            elif bool(conv_mask.any()):
                gl_in = gl_in.copy(); gl_in[conv_mask] = 0.0
                gg_in = gg_in.copy(); gg_in[conv_mask] = 0.0
            if os.environ.get("QX_DIAG_SPECTRAL") == "1":
                nw_raw = int(g_lesser.data.shape[0])
                nw_win = int(gl_in.shape[0])
                graw = np.abs(np.asarray(get_host(g_lesser.data))).reshape(
                    nw_raw, -1).max(axis=1)
                gwin = np.abs(np.asarray(get_host(gl_in))).reshape(
                    nw_win, -1).max(axis=1)
                graw = np.ascontiguousarray(graw, dtype=np.float64)
                gwin = np.ascontiguousarray(gwin, dtype=np.float64)
                comm.Allreduce(MPI.IN_PLACE, graw, op=MPI.MAX)
                comm.Allreduce(MPI.IN_PLACE, gwin, op=MPI.MAX)
                self._diag_graw_w = graw
                self._diag_gwin_w = gwin
                self._diag_full_freqs = np.abs(
                    np.asarray(get_host(full_freqs)).real)
                self._diag_win_freqs = np.abs(
                    np.asarray(get_host(conv_freqs)).real)
            gtl.data[:] = self._fft_pad(gl_in, n_fft)
            gtg.data[:] = self._fft_pad(gg_in, n_fft)
            Xl, Xg = gtl.data, gtg.data
            if nq > 1 and q_split:
                _lo = int(g_lesser.local_q_offset)
                _hi = _lo + int(g_lesser.local_q_shape[0])
                Xl = self._negate_q_across_comm(Xl, nq, _lo, _hi, xp)
                Xg = self._negate_q_across_comm(Xg, nq, _lo, _hi, xp)
            elif nq > 1:
                for ax, k in enumerate(nk, start=1):
                    neg = (-xp.arange(k)) % k
                    Xl = xp.take(Xl, neg, axis=ax)
                    Xg = xp.take(Xg, neg, axis=ax)
            if not fast_now:
                gtlr.data[0] = Xl[0]
                gtlr.data[1:] = Xl[:0:-1]
            gtgr.data[0] = Xg[0]
            gtgr.data[1:] = Xg[:0:-1]

            if lin:
                # Direction legs: identical FFT + DC mask + reversal.
                for m in (dgtl, dgtg, dgtlr, dgtgr):
                    if m.distribution_state != "nnz":
                        m.dtranspose(discard=True)
                dgl_in, dgg_in = dg_lesser.data, dg_greater.data
                if aux_on:
                    if bool(out_mask.any()):
                        dgl_in = dgl_in.copy(); dgl_in[out_mask] = 0.0
                        dgg_in = dgg_in.copy(); dgg_in[out_mask] = 0.0
                    dgl_in = aux_grid.interpolate(dgl_in)
                    dgg_in = aux_grid.interpolate(dgg_in)
                    if bool(conv_mask.any()):
                        dgl_in[conv_mask] = 0.0
                        dgg_in[conv_mask] = 0.0
                elif bool(conv_mask.any()):
                    dgl_in = dgl_in.copy(); dgl_in[conv_mask] = 0.0
                    dgg_in = dgg_in.copy(); dgg_in[conv_mask] = 0.0
                dgtl.data[:] = self._fft_pad(dgl_in, n_fft)
                dgtg.data[:] = self._fft_pad(dgg_in, n_fft)
                dXl, dXg = dgtl.data, dgtg.data
                dgtlr.data[0] = dXl[0]
                dgtlr.data[1:] = dXl[:0:-1]
                dgtgr.data[0] = dXg[0]
                dgtgr.data[1:] = dXg[:0:-1]

        # (2) g(tau) nnz->stack: tau-slice of the full band per stack rank
        with profiler.profile_range(
            "PhPh SSE: 2 tau nnz->stack + fold", level="default", comm=comm,
        ):
            gtl.dtranspose()
            gtg.dtranspose()
            if fast_now:
                stx.dtranspose(discard=True)
            else:
                gtlr.dtranspose()
            gtgr.dtranspose()
            if lin:
                for m in (dgtl, dgtg, dgtlr, dgtgr):
                    m.dtranspose()
            self._fold_reversed_legs(
                (gtlr, gtgr, dgtlr, dgtgr) if lin
                else ((gtgr,) if fast_now else (gtlr, gtgr)), g_lesser)
            if fast_now:
                stx.data[:] = 0.0
            if stl.distribution_state != "stack":
                stl.dtranspose(discard=True)
            if stg.distribution_state != "stack":
                stg.dtranspose(discard=True)
            stl.data[:] = 0.0
            stg.data[:] = 0.0

        with profiler.profile_range(
            "PhPh SSE: 3 ring contraction", level="default", comm=comm,
        ):
            gtlv, gtgv = gtl.stack[...], gtg.stack[...]
            gtlrv, gtgrv = gtlr.stack[...], gtgr.stack[...]
            stlv, stgv = stl.stack[...], stg.stack[...]
            if lin:
                dgtlv, dgtgv = dgtl.stack[...], dgtg.stack[...]
                dgtlrv, dgtgrv = dgtlr.stack[...], dgtgr.stack[...]

            start = int(stl.block_section_offsets[ranks.block.rank])
            end = int(stl.block_section_offsets[ranks.block.rank + 1])
            owned = self._owned_outputs(start, end)

            if ranks.block.size > 1:
                halo_l, halo_g = self._exchange_band_halo(gtlv, gtgv, gtl, start, end)
                halo_lr, halo_gr = self._exchange_band_halo(
                    gtlrv, gtgrv, gtlr, start, end)
            else:
                halo_l = halo_g = halo_lr = halo_gr = {}

            _c = ((lambda a: xp.ascontiguousarray(a, dtype=xp.complex64))
                  if (self._ring_c64 and nq == 1) else (lambda a: a))
            gl_blk: dict[tuple[int, int], NDArray] = {}
            gg_blk: dict[tuple[int, int], NDArray] = {}
            glr_blk: dict[tuple[int, int], NDArray] = {}
            ggr_blk: dict[tuple[int, int], NDArray] = {}
            dgl_blk: dict[tuple[int, int], NDArray] = {}
            dgg_blk: dict[tuple[int, int], NDArray] = {}
            dglr_blk: dict[tuple[int, int], NDArray] = {}
            dggr_blk: dict[tuple[int, int], NDArray] = {}
            for (K, Kp) in self._links_for_range(start, end):
                if start <= min(K, Kp) < end:
                    gl_blk[(K, Kp)] = _c(gtlv.blocks[K - start, Kp - start])
                    gg_blk[(K, Kp)] = _c(gtgv.blocks[K - start, Kp - start])
                    glr_blk[(K, Kp)] = _c(gtlrv.blocks[K - start, Kp - start])
                    ggr_blk[(K, Kp)] = _c(gtgrv.blocks[K - start, Kp - start])
                    if lin:
                        dgl_blk[(K, Kp)] = _c(
                            dgtlv.blocks[K - start, Kp - start])
                        dgg_blk[(K, Kp)] = _c(
                            dgtgv.blocks[K - start, Kp - start])
                        dglr_blk[(K, Kp)] = _c(dgtlrv.blocks[
                            K - start, Kp - start])
                        dggr_blk[(K, Kp)] = _c(dgtgrv.blocks[
                            K - start, Kp - start])
                else:
                    gl_blk[(K, Kp)] = _c(halo_l[(K, Kp)])
                    gg_blk[(K, Kp)] = _c(halo_g[(K, Kp)])
                    glr_blk[(K, Kp)] = _c(halo_lr[(K, Kp)])
                    ggr_blk[(K, Kp)] = _c(halo_gr[(K, Kp)])

            use_factored = (
                self._vfactors is not None and self._vf_kernel == "gram"
            )
            if (self._phi_pre is None and not use_factored
                    and self._micro_pair_index is None):
                self._phi_pre = {}
                _dt = xp.complex64 if self._ring_c64 else None
                _sl = _sr = 1.0
                if self._ring_c64:
                    m = max((max(float(np.abs(pl).max()),
                                 float(np.abs(pr).max()))
                             for quads in self._phi_pair_index.values()
                             for (*_, pl, pr) in quads), default=1.0)
                    e = int(np.ceil(np.log2(max(m, 1e-300))))
                    _sl = _sr = 2.0 ** -e
                    self._ring_unscale = 2.0 ** (2 * e)
                for (I, J), quads in self._phi_pair_index.items():
                    pre_list = []
                    for (K1, K2, K1p, K2p, pl, pr) in quads:
                        pl_eff = pl
                        pr_eff = pr
                        if self._ring_c64:
                            pl_eff = pl_eff * _sl
                            pr_eff = pr * _sr
                        pre_list.append(
                            (K1, K2, K1p, K2p) + phi_perms(
                                xp.asarray(pl_eff, dtype=_dt),
                                xp.asarray(pr_eff, dtype=_dt), xp))
                    self._phi_pre[(I, J)] = pre_list

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

            if self._micro_pair_index is not None:
                micro_links = self._micro_links_for_outputs(owned)
                mgl_blk = micro_link_views(
                    gl_blk, self._micro_layout, micro_links)
                mgg_blk = micro_link_views(
                    gg_blk, self._micro_layout, micro_links)
                mglr_blk = micro_link_views(
                    glr_blk, self._micro_layout, micro_links)
                mggr_blk = micro_link_views(
                    ggr_blk, self._micro_layout, micro_links)
            else:
                mgl_blk, mgg_blk = gl_blk, gg_blk
                mglr_blk, mggr_blk = glr_blk, ggr_blk

            if use_factored:
                def _qflat_f(d):
                    return {
                        kk: v.reshape(v.shape[0], nq, v.shape[-2], v.shape[-1])
                        for kk, v in d.items()
                    }

                gl_q = _qflat_f(mgl_blk)
                n_tau = next(iter(gl_q.values())).shape[0]
                dtype = next(iter(gl_q.values())).dtype
                q_lo = ranks.q.rank * nq // ranks.q.size
                q_hi = (ranks.q.rank + 1) * nq // ranks.q.size
                self._contract_factored_q(
                    owned, q_lo, q_hi, nq, nk, n_tau, dtype,
                    gl_q, _qflat_f(mgg_blk),
                    _qflat_f(mglr_blk), _qflat_f(mggr_blk),
                    stlv, stgv, start, xp,
                )
            elif self._micro_pair_index is not None and nq == 1:
                self._contract_micro_dense_gamma(
                    owned, n_tau=next(iter(mgl_blk.values())).shape[0],
                    dtype=stl.data.dtype, gl=mgl_blk, gg=mgg_blk,
                    glr=mglr_blk, ggr=mggr_blk, stlv=stlv, stgv=stgv,
                    start=start, xp=xp,
                )
            elif nq == 1:
                _pair_debug = os.environ.get("QTX_PROFILE_LEVEL") == "debug"

                pairs_c = ([ij for ij in owned if ij[0] <= ij[1]]
                           if halve_now else owned)

                n_tau = next(iter(gl_blk.values())).shape[0] if gl_blk else 0
                _dt = stl.data.dtype
                out_l = {ij: xp.empty(
                    (n_tau, int(self.block_sizes[ij[0]]),
                     int(self.block_sizes[ij[1]])), dtype=_dt)
                    for ij in pairs_c}
                out_g = {ij: xp.empty_like(out_l[ij]) for ij in pairs_c}
                out_x = ({ij: xp.empty_like(out_l[ij]) for ij in pairs_c}
                         if (fast_now or verify_now) else None)
                out_t56 = ({ij: xp.empty_like(out_l[ij]) for ij in pairs_c}
                           if verify_now else None)

                def _rings3(pre, da, db, ra, rb):
                    PL, PR, nI, bK2, nJ = pre
                    return (
                        ring_contract_pre(PL, PR, nI, bK2, nJ, da, db, xp),
                        ring_contract_pre(PL, PR, nI, bK2, nJ, da, rb, xp),
                        ring_contract_pre(PL, PR, nI, bK2, nJ, ra, db, xp),
                    )

                def _contract_tau(lo, hi, pairs):
                    times = {} if _pair_debug else None
                    for (I, J) in pairs:
                        t0 = perf_counter() if _pair_debug else 0.0
                        acc_l = acc_g = acc_x = acc_t56 = None
                        for (K1, K2, K1p, K2p, *pre
                             ) in self._phi_pre[(I, J)]:
                            gl_a = gl_blk[(K1, K1p)][lo:hi]
                            gl_b = gl_blk[(K2, K2p)][lo:hi]
                            gg_a = gg_blk[(K1, K1p)][lo:hi]
                            gg_b = gg_blk[(K2, K2p)][lo:hi]
                            ggr_a = ggr_blk[(K1, K1p)][lo:hi]
                            ggr_b = ggr_blk[(K2, K2p)][lo:hi]
                            if lin:
                                dgl_a = dgl_blk[(K1, K1p)][lo:hi]
                                dgl_b = dgl_blk[(K2, K2p)][lo:hi]
                                dgg_a = dgg_blk[(K1, K1p)][lo:hi]
                                dgg_b = dgg_blk[(K2, K2p)][lo:hi]
                                dggr_a = dggr_blk[(K1, K1p)][lo:hi]
                                dggr_b = dggr_blk[(K2, K2p)][lo:hi]
                                glr_a = glr_blk[(K1, K1p)][lo:hi]
                                glr_b = glr_blk[(K2, K2p)][lo:hi]
                                dglr_a = dglr_blk[(K1, K1p)][lo:hi]
                                dglr_b = dglr_blk[(K2, K2p)][lo:hi]
                                a1, a2, a3 = _rings3(
                                    pre, dgl_a, gl_b, dggr_a, ggr_b)
                                b1, b2, b3 = _rings3(
                                    pre, gl_a, dgl_b, ggr_a, dggr_b)
                                sl = ((a1 + a2) + a3) + ((b1 + b2) + b3)
                                a4, a5, a6 = _rings3(
                                    pre, dgg_a, gg_b, dglr_a, glr_b)
                                b4, b5, b6 = _rings3(
                                    pre, gg_a, dgg_b, glr_a, dglr_b)
                                sg = ((a4 + a5) + a6) + ((b4 + b5) + b6)
                                tx = None
                                acc_l = sl if acc_l is None else acc_l + sl
                                acc_g = sg if acc_g is None else acc_g + sg
                                continue
                            r1, r2, r3 = _rings3(pre, gl_a, gl_b, ggr_a, ggr_b)
                            sl = (r1 + r2) + r3  # == legacy _fold_l order
                            if fast_now:
                                PL, PR, nI, bK2, nJ = pre
                                sg = ring_contract_pre(
                                    PL, PR, nI, bK2, nJ, gg_a, gg_b, xp)
                                tx = r2 + r3
                            else:
                                glr_a = glr_blk[(K1, K1p)][lo:hi]
                                glr_b = glr_blk[(K2, K2p)][lo:hi]
                                r4, r5, r6 = _rings3(
                                    pre, gg_a, gg_b, glr_a, glr_b)
                                sg = (r4 + r5) + r6  # == legacy _fold_g order
                                tx = (r2 + r3) if verify_now else None
                                if verify_now:
                                    t56 = r5 + r6
                            acc_l = (sl.astype(xp.complex128, copy=False)
                                     if acc_l is None else acc_l + sl)
                            acc_g = (sg.astype(xp.complex128, copy=False)
                                     if acc_g is None else acc_g + sg)
                            if tx is not None:
                                acc_x = (tx.astype(xp.complex128, copy=False)
                                         if acc_x is None else acc_x + tx)
                            if verify_now:
                                acc_t56 = (t56.astype(xp.complex128,
                                                      copy=False)
                                           if acc_t56 is None
                                           else acc_t56 + t56)
                        if acc_l is not None:
                            if self._ring_c64:
                                u = self._ring_unscale
                                acc_l = acc_l * u
                                acc_g = acc_g * u
                                if acc_x is not None:
                                    acc_x = acc_x * u
                                if acc_t56 is not None:
                                    acc_t56 = acc_t56 * u
                            out_l[(I, J)][lo:hi] = acc_l
                            out_g[(I, J)][lo:hi] = acc_g
                            if acc_x is not None:
                                out_x[(I, J)][lo:hi] = acc_x
                            if acc_t56 is not None:
                                out_t56[(I, J)][lo:hi] = acc_t56
                        if _pair_debug:
                            times[(I, J)] = perf_counter() - t0
                    return times

                pool, n_threads = ring_pool()
                nt = min(n_threads, max(1, n_tau // self._tau_min_chunk))
                self._print_ring_stats(
                    pairs_c, n_tau, n_threads, nt,
                    rings_per_quad=12 if lin else (4 if fast_now else 6))
                pair_times: dict | None = {} if _pair_debug else None
                if pool is not None and xp is np and nt > 1:
                    bnds = [(i * n_tau // nt, (i + 1) * n_tau // nt) for i in range(nt)]
                    if self._pool_scope == "pair_tau":
                        tasks = [(ij, lo, hi)
                                 for ij in pairs_c for (lo, hi) in bnds]
                        tlists = list(pool.map(
                            lambda t: _contract_tau(t[1], t[2], (t[0],)),
                            tasks))
                    else:
                        tlists = list(pool.map(
                            lambda b: _contract_tau(b[0], b[1], pairs_c),
                            bnds))
                    if _pair_debug:
                        for t in tlists:
                            for k, v in t.items():
                                pair_times[k] = pair_times.get(k, 0.0) + v
                else:
                    pair_times = _contract_tau(0, n_tau, pairs_c)
                for ij in pairs_c:
                    stlv.blocks[ij[0] - start, ij[1] - start] = out_l[ij]
                    stgv.blocks[ij[0] - start, ij[1] - start] = out_g[ij]
                if halve_now:
                    if abs(prefactor.real) > 1e-30 * abs(prefactor):
                        raise RuntimeError(
                            "sse_hermitian_pairs: the bubble prefactor is "
                            "not purely imaginary; mirror sign undefined."
                        )
                    for (I, J) in pairs_c:
                        if I == J:
                            continue
                        stlv.blocks[J - start, I - start] = xp.conj(
                            out_l[(I, J)].swapaxes(-1, -2))
                        stgv.blocks[J - start, I - start] = xp.conj(
                            out_g[(I, J)].swapaxes(-1, -2))
                if fast_now:
                    stxv = stx.stack[...]
                    for ij in pairs_c:
                        stxv.blocks[ij[0] - start, ij[1] - start] = out_x[ij]
                    if halve_now:
                        for (I, J) in pairs_c:
                            if I == J:
                                continue
                            stxv.blocks[J - start, I - start] = xp.conj(
                                out_x[(I, J)].swapaxes(-1, -2))
                    self._fold_reversed_legs((stx,), g_lesser)
                if verify_now:
                    rev = (-xp.arange(n_tau)) % n_tau
                    worst = worst_rel = 0.0
                    for (I, J) in owned:
                        rec = out_x[(J, I)][rev].swapaxes(-1, -2)
                        d = float(xp.max(xp.abs(out_t56[(I, J)] - rec)))
                        scale = float(xp.max(xp.abs(out_t56[(I, J)]))) or 1.0
                        worst = max(worst, d)
                        worst_rel = max(worst_rel, d / scale)
                    self._fold_verify_done += 1
                    if ranks.rank == 0:
                        print(
                            "PhPh SSE fold-verify "
                            f"[{self._fold_verify_done}/{self._fold_verify}]"
                            f": max|d|={worst:.3e} rel={worst_rel:.3e} "
                            f"({'OK' if worst_rel < 1e-10 else 'MISMATCH'})",
                            flush=True,
                        )
                if _pair_debug and ranks.rank == 0 and pair_times:
                    tot = sum(pair_times.values())
                    per = "  ".join(
                        f"({I},{J}):{t:.2f}s/{len(self._phi_pre[(I, J)])}q"
                        for (I, J), t in sorted(pair_times.items()))
                    print(f"PhPh SSE pairs [cpu-s/quads]: {per}  "
                          f"total {tot:.2f}s", flush=True)
            else:
                # Coupled-q convolution.
                qdm = self._q_diff_map
                qv = self._qvertices

                def _qflat(d):
                    return {
                        kk: v.reshape(v.shape[0], -1, v.shape[-2], v.shape[-1])
                        for kk, v in d.items()
                    }

                gl_q = _qflat(mgl_blk)
                gg_q = _qflat(mgg_blk)
                glr_q = _qflat(mglr_blk)
                ggr_q = _qflat(mggr_blk)

                def _release_leg_blocks():
                    """Drop every reference to the densified leg blocks.

                    `_qflat` returns reshape VIEWS, and the halo dicts alias
                    the same arrays, so all three levels have to let go
                    before the pool can reclaim anything. Called by the
                    batched coupled-q kernel once its stacks hold the same
                    values; see `phonon.sse_release_leg_blocks`.
                    """
                    for _d in (gl_blk, gg_blk, glr_blk, ggr_blk,
                               gl_q, gg_q, glr_q, ggr_q,
                               halo_l, halo_g, halo_lr, halo_gr):
                        _d.clear()
                    free_mempool()
                n_tau = next(iter(gl_q.values())).shape[0]
                dtype = next(iter(gl_q.values())).dtype
                q_lo = ranks.q.rank * nq // ranks.q.size
                q_hi = (ranks.q.rank + 1) * nq // ranks.q.size

                if self._vfactors is not None:
                    qv = self._reconstructed_qvertices(q_lo, q_hi, nq)
                self._rotate_internal_q(
                    owned, qdm, qv, q_lo, q_hi, nq, nk, n_tau, dtype,
                    (gl_q, gg_q, glr_q, ggr_q), _fold_l, _fold_g,
                    stlv, stgv, start, xp,
                    q_split=q_split,
                    fast_now=fast_now, verify_now=verify_now,
                    stxv=stx.stack[...] if fast_now else None,
                    release=(_release_leg_blocks
                             if self._release_legs else None),
                )
                if fast_now:
                    self._fold_reversed_legs((stx,), g_lesser)

            if nq > 1 and ranks.q.size > 1:
                for m in ((stl, stg, stx) if fast_now else (stl, stg)):
                    recv = xp.empty_like(m.data)
                    ranks.q.all_reduce(xp.ascontiguousarray(m.data), recv, op="sum")
                    m.data[:] = recv

        # (4) sigma(tau) stack->nnz
        with profiler.profile_range(
            "PhPh SSE: 4 tau stack->nnz", level="default", comm=comm,
        ):
            stl.dtranspose()
            stg.dtranspose()
            if fast_now:
                stx.dtranspose()

        # (5) IFFT sigma(tau)->Sigma(omega) in nnz; add into outputs; build Sigma^R.
        with profiler.profile_range(
            "PhPh SSE: 5 IFFT + Hilbert", level="default", comm=comm,
        ):
            if halve_now:
                ml, mu = self._herm_masks(g_lesser)
                stl.data[1:, ..., ml] = stl.data[:0:-1, ..., ml]
                stg.data[1:, ..., ml] = stg.data[:0:-1, ..., ml]
                if fast_now:
                    stx.data[1:, ..., mu] = stx.data[:0:-1, ..., mu]
            if fast_now:
                xs = stx.data
                if nq > 1:
                    for ax, k in enumerate(nk, start=1):
                        xs = xp.take(xs, (-xp.arange(k)) % k, axis=ax)
                stg.data[0] += xs[0]
                stg.data[1:] += xs[:0:-1]
            sl_conv = prefactor * xp.fft.ifft(stl.data, axis=0)[:ne_conv]
            sg_conv = prefactor * xp.fft.ifft(stg.data, axis=0)[:ne_conv]
            if q_split:
                q_out = sigma_lesser.local_q_slice
                sl_conv = sl_conv[:, q_out]
                sg_conv = sg_conv[:, q_out]
            if pole is not None and pole.covariance is not None:
                _cl, _cg = pole.covariance
                sl_conv = sl_conv + _cl
                sg_conv = sg_conv + _cg
            if pole is not None and pole.mixed is not None:
                _mx_l, _mx_g = pole.mixed
                sl_conv = sl_conv + _mx_l
                sg_conv = sg_conv + _mx_g
            if bool(conv_mask.any()):
                sl_conv[conv_mask] = 0.0
                sg_conv[conv_mask] = 0.0
            if aux_on:
                sl_data = aux_grid.restrict(sl_conv)
                sg_data = aux_grid.restrict(sg_conv)
                if bool(out_mask.any()):
                    sl_data[out_mask] = 0.0
                    sg_data[out_mask] = 0.0
            else:
                sl_data, sg_data = sl_conv, sg_conv
            if pole is not None and pole.self_energy is not None:
                _ss_l, _ss_g, _ss_r = pole.self_energy
                if bool(out_mask.any()):
                    _ss_l = _ss_l.copy(); _ss_l[out_mask] = 0.0
                    _ss_g = _ss_g.copy(); _ss_g[out_mask] = 0.0
                sl_data = sl_data + _ss_l
                sg_data = sg_data + _ss_g
            sigma_lesser.data[:] = sigma_lesser.data - sl_data
            sigma_greater.data[:] = sigma_greater.data - sg_data
            if self.retarded_method == "fft":
                delta = sg_conv - sl_conv
                self._check_kk_grid_support(delta)
                hil = 0.5j * hilbert_transform(delta, conv_freqs,
                                               transverse_shape=nk)
                if aux_on:
                    hil = aux_grid.restrict(hil)
                if bool(out_mask.any()):
                    # post-mask the retarded consistently with Sigma^<>
                    hil[out_mask] = 0.0
                sigma_retarded.data[:] = sigma_retarded.data + hil
            if pole is not None and pole.self_energy is not None:
                _ss_r = pole.self_energy[2]
                if bool(out_mask.any()):
                    _ss_r = _ss_r.copy()
                    _ss_r[out_mask] = 0.0
                sigma_retarded.data[:] = sigma_retarded.data + _ss_r
            if pole is not None:
                pole.clear()

    def _check_kk_grid_support(self, delta: NDArray) -> None:
        """Warns (once) if the bubble is still alive at the top of the grid.

        The Hilbert transform reconstructs Re Sigma^R from Sigma^> - Sigma^<
        sampled on [0, omega_max]. The 3-phonon bubble has support up to twice
        the phonon band top, so a grid that stops at the band top truncates the
        Kramers-Kronig integral. A spectrum that has not decayed by the last bin
        is the symptom.
        """
        if getattr(self, "_kk_grid_checked", False):
            return
        self._kk_grid_checked = True

        if delta.size == 0:
            return

        peak = float(get_host(xp.max(xp.abs(delta))))
        if peak == 0.0:
            return

        edge = float(get_host(xp.max(xp.abs(delta[-1]))))
        if edge > 1e-2 * peak:
            advice = (
                "Raise phonon.sse_aux_grid_fmax_thz to about twice the "
                "phonon band top (the auxiliary bubble grid sets the KK "
                "support, not the energy window)."
                if self._aux_spacing > 0.0 else
                "Extend energy_window_max (or the grid in "
                "phonon_energies.npy) to about twice the phonon band top."
            )
            warnings.warn(
                f"The 3-phonon bubble still carries {edge / peak:.1%} of its "
                "peak weight at the top of the frequency grid, so the "
                f"Kramers-Kronig integral for Re Sigma^R is truncated. "
                f"{advice}",
                stacklevel=2,
            )

    def _print_ring_stats(self, owned, n_tau, n_threads, n_chunks,
                          rings_per_quad=6):
        """One-time (rank 0) stage-3 shape/cost summary: pair/quad counts,
        tau geometry, pool layout, and the GEMM flop model of one full ring
        pass -- divide by the measured 'PhPh SSE: 3' time for achieved GF/s."""
        if getattr(self, "_ring_stats_printed", False) or ranks.rank != 0:
            return
        self._ring_stats_printed = True
        n_quads = sum(len(self._phi_pre[p]) for p in owned)
        flops = 0.0
        for p in owned:
            for (_, _, _, _, PL, PR, nI, bK2, nJ) in self._phi_pre[p]:
                bK1 = PL.shape[1]
                bK2p = PR.shape[0]
                bK1p = PR.shape[1] // nJ
                ring = 8 * n_tau * (
                    nI * bK2 * bK1 * bK1p
                    + bK2 * bK2p * bK1p * nJ
                    + nI * bK2 * bK1p * nJ
                )
                flops += rings_per_quad * ring
        self._ring_model_gflop = flops / 1e9
        print(
            f"PhPh SSE ring: pairs={len(owned)} quads={n_quads} "
            f"n_tau={n_tau} pool={n_threads} chunks={n_chunks} "
            f"rings/quad={rings_per_quad} "
            f"model={flops / 1e9:.1f} GFLOP/pass",
            flush=True,
        )

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

    def set_pole_channel(self, g_pp_lesser, g_pp_greater) -> None:
        """Inject pole-sector Green-function legs for one iteration."""
        self._pole_injection.channel = (g_pp_lesser, g_pp_greater)

    def set_pole_mixed(self, sigma_l, sigma_g) -> None:
        """Inject mixed pole-background self-energies for one iteration."""
        self._pole_injection.mixed = (sigma_l, sigma_g)

    def set_bubble_correction(self, corr_l, corr_g) -> None:
        """Inject a subcell covariance correction for one iteration."""
        self._pole_injection.covariance = (corr_l, corr_g)

    def set_pole_self_energy(self, sigma_l, sigma_g, sigma_r) -> None:
        """Inject an analytic pole self-energy for one iteration."""
        self._pole_injection.self_energy = (sigma_l, sigma_g, sigma_r)

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
        if int(freqs.shape[0]) > 1:
            if float(freqs[0]) < 0.0 or bool(xp.min(xp.diff(freqs)) <= 0.0):
                raise ValueError(
                    "The phonon frequency grid must be strictly ascending "
                    "and non-negative."
                )
            if self._aux_spacing <= 0.0:
                df = float(freqs[1] - freqs[0])
                if abs(float(freqs[0])) > 1e-9 * abs(df) or bool(
                    xp.max(xp.abs(xp.diff(freqs) - df)) > 1e-9 * abs(df)
                ):
                    raise ValueError(
                        "The bubble FFT requires a uniform frequency grid "
                        f"starting at 0 (got start {float(freqs[0]):g}, "
                        f"spacing {df:g}). Set energy_window_min = 0, or set "
                        "phonon.sse_aux_grid_dw_thz > 0 to run the bubble on "
                        "an auxiliary uniform grid (non-uniform primary "
                        "grids)."
                    )
        self._full_freqs = freqs
        return freqs

    def _get_auxiliary_grid(self, frequencies: NDArray) -> AuxiliaryGrid:
        if self._aux_grid is None:
            self._aux_grid = make_auxiliary_grid(
                frequencies,
                self._aux_spacing,
                self._aux_max_frequency,
                self._aux_restriction,
            )
        return self._aux_grid
