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

from qttools import NDArray, sparse, xp
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
from quatrex.phonon.microblocks import (
    MicroblockLayout,
    build_micro_pair_index,
    micro_link_views,
)
from quatrex.phonon.units import bubble_prefactor_thz

profiler = Profiler()

#: Slots in the ``_ensure_tau_buffers`` tuple that hold the Sigma(tau)
#: ACCUMULATORS rather than a leg. The order is
#: ``(gtl, gtg, stl, stg, gtlr, gtgr[, dgtl, dgtg, dgtlr, dgtgr])``.
#: They are the only buffers that must stay whole when ``comm.q`` sections the
#: transverse axis, because the bubble is a convolution over q and one slice
#: pair contributes to every external momentum.
_SIGMA_TAU_SLOTS = frozenset({2, 3})

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
        # Sigma^> reconstruction from the Sigma^< cross terms (WP3): exact
        # bosonic tau-domain identity, 4 instead of 6 ring calls per quad.
        self._g_from_l = bool(
            getattr(config.phonon, "sse_greater_from_lesser", False))
        self._fold_verify = int(
            getattr(config.phonon, "sse_fold_verify_iterations", 0))
        self._fold_verify_done = 0
        # Hermitian pair-halving (WP4): contract I <= J only, mirror the
        # lower blocks via Sigma_JI = -Sigma_IJ^dagger.
        self._herm_pairs = bool(
            getattr(config.phonon, "sse_hermitian_pairs", False))
        retarded_method = getattr(config.phonon, "retarded_method", "fft")
        if retarded_method not in ("half", "fft"):
            raise ValueError(
                f"Unknown retarded_method={retarded_method!r}; "
                "use 'half' or 'fft'."
            )
        self.retarded_method = retarded_method
        # Pole-subtracted SCBA sector. Narrow resonances of G are removed from
        # the bubble LEGS and re-added analytically, so the frequency grid no
        # longer has to resolve the sharpest linewidth in the problem. The
        # split G = G_S + G_R is exact: this changes the representation, not
        # the diagram. See phonon/docs/pole_scba_implemented.md.
        _ps = getattr(config.phonon, "pole_sector", None)
        self._pole_cfg = _ps
        self._pole_enabled = bool(_ps is not None and _ps.enabled)
        self._pole_channel = None   # (g_pp_lesser, g_pp_greater) on the primary grid
        self._pole_sigma_ss = None  # (sigma_ss_l, sigma_ss_g, sigma_ss_r)
        self._pole_sigma_mixed = None  # (sigma_sr_l, sigma_sr_g)
        self._bubble_correction = None  # subcell covariance, pre-KK
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
                # The kernel conjugates only the contracted-leg row factor; the
                # external leg carries no conjugate because D and lambda are real.
                raise ValueError(
                    "The external-leg factors D and lambdas must be real."
                )
            self._vfactors = vfactors
            self._q_diff_map = np.asarray(vfactors.q_diff_map, dtype=int)
            self._n_kpts = int(vfactors.n_kpts)
            # Kernel choice for consuming the factors:
            #   "reconstruct": materialise the RANK-LOCAL slice of
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
            mesh = np.asarray(monkhorst_pack(grid[tr], shift[tr]))
            want = np.stack(
                np.meshgrid(*[np.arange(n) / n for n in grid[tr]],
                            indexing="ij"), axis=-1).reshape(-1, int(tr.sum()))
            # ORDERED comparison: the q-difference/negation index arithmetic
            # requires q_k = k/n at index k, not merely the same point set. The
            # mesh lives on a torus, so the residual is the SIGNED circular
            # distance -- q = -1e-11 and q = 0 are the same point, whereas a
            # plain modulo maps them to opposite ends of the cell.
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

        # Inner Green's-function band kept in the contraction. The default
        # RGF selected-inversion produces the block-tridiagonal G
        # (diagonal + first off-diagonal, |K-K'| <= 1), and the ring masks
        # the bubble kernel G(x)G to that band. That mask is NOT a
        # congruence: it breaks the positive-semidefiniteness of Sigma^{<,>}
        # on interior slabs (Schur product with the indefinite
        # tridiagonal-ones mask), injecting non-causal gain. With
        # sse_g_band = 2 the solver additionally produces the second
        # off-diagonal G^{<,>} blocks and the kernel is complete for the
        # nearest-neighbour vertex span (diagonal Sigma blocks exact).
        self.g_band = min(
            (micro_band if self._micro_layout is not None else
             int(getattr(config.phonon, "sse_g_band", 1) or 1)),
            self._vertex_n_blocks - 1,
        )
        # The selected solve is expressed in grouped Dyson blocks.  A valid
        # microblock grouping is required below to place every retained
        # primitive G link in the same or an adjacent grouped block, so the
        # grouped solve remains block tridiagonal even when the primitive band
        # is much wider.
        self._solver_g_band = (
            min(1, self.n_blocks - 1)
            if self._micro_layout is not None else self.g_band
        )
        # Coupled-q dense ring: strided-batched GEMMs over the (q', quad)
        # task axis instead of one Python task at a time (the per-task
        # dispatch cost ~200 us dominated the film ring at 4-10% of peak).
        self._dense_q_batched = bool(
            getattr(config.phonon, "sse_dense_q_batched", True))
        if self._solver_g_band > 1 and ranks.block.size > 1:
            # The band halo exchange and the bosonic-fold plan only span
            # the IMMEDIATE comm.block neighbours; with every rank owning
            # at least g_band + 1 blocks all band links land there (the
            # halo width itself is data-driven via _links_for_range). The
            # distributed RGF enforces the same bound for its
            # off-diagonal post-pass.
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

        # Precompute the full off-diagonal pair index: for each output
        # block pair (I, J) with |I-J| <= 1, collect the ring quads
        #   (K1, K2, K1', K2', phi_left, phi_right)
        # with phi_left = Phi[(I, K1, K2)], phi_right = Phi[(J, K2', K1')]
        # and the inner G links (K1, K1'), (K2, K2') inside the band.
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

        # Distinct inner G band blocks (K, K') referenced by any quad,
        # that must be gathered to full omega.
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

        # Cached intermediate tau-domain buffers (length
        # n_fft) for FFTd G
        self._tau_cache: tuple | None = None
        # Cached bosonic-fold plan (local gather perm + neighbour exchange
        # schedule); False = not yet built, None = no pattern, else the tuple.
        self._rev_perm: tuple | bool = False
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
            # One offset-table einsum per pair; blocks are I-independent
            # (bulk vertex, minimum-image offsets), shared across I.
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
        self._scp_cfg = config.phonon
        # Quartic (SCP) loop: Sigma_L = 1/2 Phi4 : <uu> -- the
        # stiffening counterterm that GROWS with <uu>, restoring the
        # negative feedback the cubic-only bubble lacks on soft-mode
        # structures (the MoS2 film spiral instability).
        self._fc4_blocks = None
        if bool(getattr(config.phonon, "scp_loop", False)):
            import h5py as _h5py
            from pathlib import Path as _Path
            _fc4p = getattr(config.phonon, "scp_fc4_path", None)
            if _fc4p is None and getattr(config.phonon, "fc3_path", None):
                _fc4p = str(_Path(config.phonon.fc3_path).parent
                            / "fc4_blocks.hdf5")
            if _fc4p is None or not _Path(_fc4p).exists():
                raise FileNotFoundError(
                    f"scp_loop requested but fc4_blocks not found "
                    f"({_fc4p!r}); run the fc4 reap first.")
            self._fc4_blocks = {}
            with _h5py.File(_fc4p, "r") as _f:
                for _k, _d in _f["fc4_blocks"].items():
                    self._fc4_blocks[tuple(int(x) for x in
                                           _k.split("_"))] = _d[()]
            if ranks.rank == 0:
                print(f"SCP loop: {len(self._fc4_blocks)} fc4 device "
                      f"blocks from {_fc4p}", flush=True)
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

        from quatrex.grid.energies import (
            frequency_cell_widths, is_uniform_grid)

        N_D = self._sigma_static.shape[0]
        # omega-integral over every stack axis (full omega local per nnz
        # slice); the w=0 bin is excluded from the folded G^> half.
        ax = tuple(range(g_lesser.data.ndim - 1))
        full_freqs = self._full_frequencies(int(g_lesser.data.shape[0]))
        # <uu> quadrature floor: bins below the lowest device mode
        # carry only eta=0 resolvent tails/leakage (no spectral
        # weight), yet on IR-resolved grids they can dominate and
        # even flip Tr<uu> negative. scp_uu_min_thz (default 0 =
        # legacy) excludes them from BOTH G^< and the folded G^>.
        _uu_min = float(getattr(self._scp_cfg, "scp_uu_min_thz", 0.0)
                        if hasattr(self, "_scp_cfg") else 0.0)
        _keep = xp.abs(xp.asarray(full_freqs)) >= max(_uu_min, 0.0)
        pos = (xp.abs(xp.asarray(full_freqs)) > 1e-12) & _keep
        if is_uniform_grid(full_freqs):
            # Legacy path: plain sum, scalar dw applied downstream. The dw
            # source is the rank-LOCAL first gap, as before this change:
            # on linspace grids the local and global first gaps can differ
            # in the last ulp, and the legacy bit-identity contract wins.
            gl_sum_local = g_lesser.data[_keep].sum(axis=ax)
            gg_sum_local = g_greater.data[pos].sum(axis=ax)
            lf = np.asarray(self.local_frequencies)
            dw = (float(lf[1] - lf[0]) if int(lf.shape[0]) >= 2
                  else float(full_freqs[1] - full_freqs[0]))
        else:
            # Non-uniform grid: per-bin quadrature weights, dw folded in.
            cw = frequency_cell_widths(full_freqs).reshape(
                (-1,) + (1,) * (g_lesser.data.ndim - 1))
            gl_sum_local = (cw * g_lesser.data)[_keep].sum(axis=ax)
            gg_sum_local = (cw * g_greater.data)[pos].sum(axis=ax)
            dw = 1.0
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
        # Textbook convention (G^< = -i n A) is the NEGATIVE of the solver's
        # occupation-positive storage; the transpose is the ji fold.
        g_total = np.asarray(get_host(-(g_sum + gg_sum.T)))
        uu = equal_time_uu_from_sum(g_total, dw)
        if float(np.trace(uu)) <= 0.0 and comm.rank == 0:
            warnings.warn(
                "SCP tadpole: Tr<uu> <= 0 -- sign-convention violation?",
                stacklevel=2)
        phi_eff = self._scp_D + self._sigma_static
        if str(getattr(self._scp_cfg, "scp_uu_source", "g")) == "dressed":
            # SCP closed form on the dressed model -- the eta=0 G^<
            # quadrature is ill-conditioned on IR-resolved grids and
            # produced O(10x) <uu> errors driving the static terms
            # unphysical (2026-08-02).
            from quatrex.phonon.static_self_energy import (
                equal_time_uu_dressed)
            _tbar = 0.5 * (float(self._scp_cfg.left_temperature)
                           + float(self._scp_cfg.right_temperature))
            uu = equal_time_uu_dressed(phi_eff, _tbar)
        if bool(getattr(self._scp_cfg, "scp_tadpole_term", True)):
            w_mean = mean_displacement(
                self._fc3_dev_mw, uu, phi_eff,
                omega2_floor_abs=self._scp_floor2)
            sig_new = sigma_tadpole(self._fc3_dev_mw, w_mean)
        else:
            # Centrosymmetric crystals (2H-MoS2, P6_3/mmc): <u> = 0 by
            # inversion symmetry -- the tadpole is pure numerical noise
            # amplified through Phi_eff^+ (the loop3 runaway). Keep the
            # inversion-even quartic loop only.
            sig_new = np.zeros_like(self._sigma_static)
        if self._fc4_blocks is not None:
            from quatrex.phonon.static_self_energy import sigma_loop_blocks
            sig_new = sig_new + sigma_loop_blocks(
                self._fc4_blocks, uu,
                self.n_blocks, int(self.block_sizes[0]))
        self._sigma_static = (
            (1.0 - self._scp_mix) * self._sigma_static + self._scp_mix * sig_new)
        if ranks.rank == 0:
            _ev = np.linalg.eigvalsh(
                0.5 * (phi_eff + phi_eff.T)
                + (self._sigma_static - (phi_eff - self._scp_D)))
            _w = np.sign(_ev[:6]) * np.sqrt(np.abs(_ev[:6]))
            print(f"SCP static: dressed low-6 (THz) "
                  f"{np.round(_w, 4)}", flush=True)
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
        # A q-SECTIONED G: the legs carry only this rank's slice of the
        # transverse mesh. The bubble is a convolution over q, so the rotation
        # in :meth:`_rotate_internal_q` accumulates at EVERY external momentum
        # and only the reduction over ``comm.q`` completes it -- the asymmetry
        # recorded in phonon/docs/bubble_positivity.md Sec. 7 (the legs
        # section, the Sigma accumulator does not until that reduction).
        #
        # The two halves of the reduce-scatter therefore live apart: the
        # accumulators are allocated WHOLE (``_SIGMA_TAU_SLOTS``), summed over
        # comm.q after the rotation, and only then cut down to this rank's
        # section, in stage (5) where the outputs are formed. Covered by
        # test_internal_q_rotation_reproduces_the_replicated_result.
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
            # The momentum index is flattened in C order, so the per-axis mesh
            # must match, not just its product: (3, 9) and (9, 3) both give
            # n_kpts = 27 but index every transverse momentum differently.
            if tuple(self._vfactors.nk_shape) != nk:
                raise ValueError(
                    "The decomposed vertices are for a transverse mesh "
                    f"{tuple(self._vfactors.nk_shape)} but the Green's function "
                    f"has {nk}."
                )
        # Transverse q with block-parallel transport. The band halo derives
        # its buffer shape from the leg buffers (_exchange_band_halo), so the
        # nk axes ride through it, and the bosonic fold already works on the
        # nnz axis alone with data.shape[:-1] preserved -- those two were the
        # blockers. Covered by tests/quatrex/phonon/test_sse_coupled_q_dist.py.

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
        conv_freqs = full_freqs
        ne_conv = ne_full
        prefactor = bubble_prefactor_thz(
            float(full_freqs[1] - full_freqs[0]))
        n_fft = 2 * ne_conv - 1
        # Coupled-q convolution carries the 1/N_q mesh-average
        if nq > 1:
            prefactor = prefactor / nq

        # Sigma^> reconstruction mode (sse_greater_from_lesser): during the
        # first sse_fold_verify_iterations calls the LEGACY 6-ring path runs
        # and the reconstruction identity is checked in place (single-rank
        # runs only); afterwards the 4-ring fast path takes over and the
        # reversed-lesser leg buffer (gtlr) is repurposed as the cross-term
        # accumulator stx.
        _gram_now = (self._vfactors is not None
                     and self._vf_kernel == "gram")
        if self._g_from_l and _gram_now:
            # gram takes precedence over the dense path since the
            # two-collapse commit; with greater_from_lesser the
            # repurposed gtlr/stx buffers are never written by the
            # factored kernel -> silently wrong Sigma^>. Refuse.
            raise ValueError(
                "sse_greater_from_lesser supports the DENSE kernels only "
                "(no decomposed gram kernel)."
            )
        if self._g_from_l and nq > 1 and self._vfactors is not None:
            # The coupled-q fold identity
            #   Sigma^>_IJ(q, tau) = Sigma^<_JI(-q, -tau)^T
            # relies on the vertex reality Phi(-q1,-q2) = conj(Phi(q1,q2))
            # (real real-space FC3). Production qfold vertices satisfy it;
            # the factor-reconstructed slice is not audited for it. Refuse.
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
            # The factored kernel contracts ALL owned pairs; the stage-5
            # mirror completion would then overwrite directly-computed
            # blocks with the tau reversal -> garbage. Refuse.
            raise ValueError(
                "sse_hermitian_pairs supports the Gamma-only DENSE kernel "
                "only (nq == 1, no decomposed gram kernel)."
            )
        if self._herm_pairs and ranks.size > 1:
            raise NotImplementedError(
                "sse_hermitian_pairs needs the full local nnz pattern for "
                "the mirror masks; single-rank runs only."
            )
        # Halving is suspended during the fold-verify iterations (the gate
        # needs both triangles contracted directly).
        halve_now = self._herm_pairs and not verify_now
        if lin:
            # The symmetry fast paths fuse legs across the (I,J)<->(J,I)
            # mirror -- exact for the symmetric quadratic form B(G, G),
            # not for the asymmetric cross B(dG, G) + B(G, dG). The
            # linearized bubble always runs the plain 6-ring path.
            fast_now = halve_now = verify_now = False
        if q_split:
            # The fast path parks its cross-term accumulator in the gtlr LEG
            # slot (``stx = gtlr`` below). Under q sectioning a leg is a slice
            # and an accumulator is whole, so one buffer cannot be both. The
            # fast path is an optimisation, not a correctness requirement, so
            # it stands down rather than the sectioning.
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
            if self._pole_enabled and self._pole_channel is not None:
                # Remove the pole sector from the legs before the FFT.
                p_l, p_g = self._pole_channel
                gl_in = gl_in - p_l
                gg_in = gg_in - p_g
            # The omega=0 bin never enters the bubble (Bose divergence at DC;
            # the zero-measure convention of the theory). The Green's functions
            # stay intact for Dyson/observables; only the copies fed into the
            # 3-phonon convolution are masked, with the matching output mask in
            # step (5).
            out_mask = xp.abs(xp.asarray(full_freqs)) < 1e-6
            if bool(out_mask.any()):
                gl_in = gl_in.copy(); gl_in[out_mask] = 0.0
                gg_in = gg_in.copy(); gg_in[out_mask] = 0.0
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
                    np.asarray(get_host(full_freqs)).real)
            gtl.data[:] = self._fft_pad(gl_in, n_fft)
            gtg.data[:] = self._fft_pad(gg_in, n_fft)
            # Reversed (absorption) legs: rev(X)[l] = X[(-l) mod n_fft] of the
            # FFT'd G. The bosonic continuation carries the ji-TRANSPOSE (and
            # -q for coupled-q): G^<_ij(q, -w) = G^>_ji(-q, w). The
            # no-transpose shortcut is exact only for the equilibrium
            # (complex-symmetric) part of G and breaks the Phi-derivable energy
            # balance off equilibrium.
            #
            # Ordering: q-negation and FFT + tau reversal are state-independent
            # and done here in the nnz state; the ji-transpose is applied after
            # the nnz->stack dtranspose, where the full nnz pattern is local on
            # every rank. All three commute, so this is exactly the serial fold
            # at any rank count, and the already-FFT'd decay legs are reused.
            Xl, Xg = gtl.data, gtg.data
            if nq > 1 and q_split:
                # Sectioned: q -> -q lands on another rank's slice, so the
                # axis has to be reassembled for the negation. One transverse
                # axis by construction under q_distributed.
                _lo = int(g_lesser.local_q_offset)
                _hi = _lo + int(g_lesser.local_q_shape[0])
                Xl = self._negate_q_across_comm(Xl, nq, _lo, _hi, xp)
                Xg = self._negate_q_across_comm(Xg, nq, _lo, _hi, xp)
            elif nq > 1:
                # negate the transverse momentum axes (Gamma-centered IDFT
                # meshes are closed under q -> -q)
                for ax, k in enumerate(nk, start=1):
                    neg = (-xp.arange(k)) % k
                    Xl = xp.take(Xl, neg, axis=ax)
                    Xg = xp.take(Xg, neg, axis=ax)
            if not fast_now:
                # (skipped in fast mode: only Sigma^> consumed the
                # reversed-lesser leg, and Sigma^> is reconstructed)
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
                if bool(out_mask.any()):
                    dgl_in = dgl_in.copy(); dgl_in[out_mask] = 0.0
                    dgg_in = dgg_in.copy(); dgg_in[out_mask] = 0.0
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
            # ji-transpose of the reversed legs, applied in the "stack" state where
            # the FULL nnz axis is local on every rank (the decay legs gtl/gtg are
            # NOT transposed). The transpose (i,j)->(j,i) is a pure function of the
            # sparsity pattern; with block_comm_size>1 the partner (j,i) of a
            # block-boundary entry lives on a neighbour block rank and is fetched by
            # an immediate-neighbour exchange (BT band => |I-J|<=1). Exact + conserving
            # at any nranks AND any block_comm_size.
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

            # Materialise each distinct band link. With the
            # single-precision ring the links are downcast copies (the
            # tau buffers themselves stay complex128). Gamma-only: the
            # coupled-q kernels ignore sse_ring_dtype.
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

            # The full bubble folds the negative-omega contribution
            # into the one-sided grid via G^<(-omega)=G^>(omega): per ring quad
            #   Sigma^<  = ring(g^<_a, g^<_b) + ring(g^<_a, rev g^>_b) + ring(rev g^>_a, g^<_b)
            #   Sigma^>  = ring(g^>_a, g^>_b) + ring(g^>_a, rev g^<_b) + ring(rev g^<_a, g^>_b)
            # The omega/tau batch is parallelised ONCE over the whole
            # (I,J)/phi-pair loop below (single-thread BLAS per chunk), and the
            # fixed phi factors are pre-permuted once -- per (I,J) a list of
            # (K1,K2,K1p,K2p, PL,PR,nI,bK2,nJ) -- to avoid per-call transpose
            # copies.
            # Only the dense ring consumes the pre-permuted phi factors.
            use_factored = (
                self._vfactors is not None and self._vf_kernel == "gram"
            )
            if (self._phi_pre is None and not use_factored
                    and self._micro_pair_index is None):
                self._phi_pre = {}
                # xp.asarray stages the host vertex factors to the
                # device once here (free view under numpy). With the
                # single-precision ring the copies are downcast so the
                # per-quad GEMMs run as batched CGEMM -- prescaled by an
                # exact power of two: the raw FC3 elements are ~1e20, so
                # the Phi G G Phi product would overflow float32 (~1e38).
                # The 2^-e / 2^e pair is exact in floating point, adding
                # no rounding beyond the downcast itself.
                _dt = xp.complex64 if self._ring_c64 else None
                _sl = _sr = 1.0
                if self._ring_c64:
                    m = max((max(float(np.abs(pl).max()),
                                 float(np.abs(pr).max()))
                             for quads in self._phi_pair_index.values()
                             for (*_, pl, pr) in quads), default=1.0)
                    # Full scale on EACH side: the T = Phi@G and
                    # U = G@Phi intermediates must stay ~O(|G| b), or
                    # the S = T@U product overflows float32 in the
                    # Bose-enhanced low-omega bins.
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
                # The factored kernel serves the Gamma-only device too: at
                # nq == 1 the momentum convolution is the identity and what
                # remains is the Gram collapse, b^4 -> R b^2 + R^2 b.
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

                # Hermitian pair-halving: contract only I <= J; the (J, I)
                # blocks are mirrored below via Sigma_JI = -Sigma_IJ^dagger.
                pairs_c = ([ij for ij in owned if ij[0] <= ij[1]]
                           if halve_now else owned)

                n_tau = next(iter(gl_blk.values())).shape[0] if gl_blk else 0
                # Sigma dtype, NOT the leg dtype: with the c64 ring the
                # accumulators are complex128 (unscaled magnitudes far
                # beyond float32 range) -- a c64 output buffer would clip
                # the largest Sigma^>(tau) elements to inf on assignment.
                _dt = stl.data.dtype
                # Preallocated per-pair outputs: pool tasks write disjoint
                # tau slices directly (race-free), removing the per-pair
                # concatenate alloc+copy of the chunked path.
                out_l = {ij: xp.empty(
                    (n_tau, int(self.block_sizes[ij[0]]),
                     int(self.block_sizes[ij[1]])), dtype=_dt)
                    for ij in pairs_c}
                out_g = {ij: xp.empty_like(out_l[ij]) for ij in pairs_c}
                # Cross-term (t2+t3) accumulator for the Sigma^>
                # reconstruction; in verify mode also the direct t5+t6 for
                # the identity gate.
                out_x = ({ij: xp.empty_like(out_l[ij]) for ij in pairs_c}
                         if (fast_now or verify_now) else None)
                out_t56 = ({ij: xp.empty_like(out_l[ij]) for ij in pairs_c}
                           if verify_now else None)

                def _rings3(pre, da, db, ra, rb):
                    # The 3-term bosonic fold, pieces kept separate:
                    # ring(decay, decay), ring(decay, rev), ring(rev, decay).
                    PL, PR, nI, bK2, nJ = pre
                    return (
                        ring_contract_pre(PL, PR, nI, bK2, nJ, da, db, xp),
                        ring_contract_pre(PL, PR, nI, bK2, nJ, da, rb, xp),
                        ring_contract_pre(PL, PR, nI, bK2, nJ, ra, db, xp),
                    )

                # Compute Sigma^{<,>} for the tau slice [lo:hi] of the given
                # pairs, writing into the preallocated outputs; returns the
                # per-pair wall time under profile-level debug.
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
                                # Mixed-leg cross (product rule): each of
                                # the 3 bosonic-fold rings with leg a, then
                                # leg b, carrying the direction. Exactly
                                # B(dG, G) + B(G, dG); the reversed
                                # direction legs are rev(dG) (R-linear).
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
                                # Exact reconstruction: only the diagonal
                                # term ring(g^>, g^>) is contracted; the
                                # cross terms come from (J, I)'s t2+t3.
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
                                # exact power-of-two unscale (see the
                                # _phi_pre build)
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

                # Cap the tau-split so each worker keeps >= sse_tau_min_chunk
                # tau points: the per-ring contraction scales with threads
                # only while each chunk's BS^3 intermediate stays in cache;
                # shorter chunks regress into dispatch/allocator churn.
                pool, n_threads = ring_pool()
                nt = min(n_threads, max(1, n_tau // self._tau_min_chunk))
                self._print_ring_stats(
                    pairs_c, n_tau, n_threads, nt,
                    rings_per_quad=12 if lin else (4 if fast_now else 6))
                pair_times: dict | None = {} if _pair_debug else None
                # The pool only pays off for the CPU backend (single-thread BLAS
                # per chunk); on GPU the GEMMs are already batched on-device.
                if pool is not None and xp is np and nt > 1:
                    bnds = [(i * n_tau // nt, (i + 1) * n_tau // nt) for i in range(nt)]
                    if self._pool_scope == "pair_tau":
                        # (pair x tau-chunk) tiles: fat chunks still fill
                        # the pool.
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
                    # Mirror the lower blocks: Sigma_JI = -Sigma_IJ^dagger
                    # with sign -conj(pref)/pref = +1 (purely imaginary
                    # bubble prefactor); the tau reversal that completes the
                    # identity is applied in the nnz state before the IFFT.
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
                        # Mirror the cross terms too; after the fold the
                        # upper stx entries then hold conj((t2+t3)_IJ), and
                        # the reversal in stage 5 (upper mask) makes the
                        # merge deliver Sigma^>_IJ[l] += conj((t2+t3)_IJ[l]).
                        for (I, J) in pairs_c:
                            if I == J:
                                continue
                            stxv.blocks[J - start, I - start] = xp.conj(
                                out_x[(I, J)].swapaxes(-1, -2))
                    # ji-transpose the cross terms in the stack state; the
                    # tau reversal happens after the stage-4 dtranspose (nnz
                    # state, full tau axis), completing
                    #   Sigma^>_IJ[l] = t4_IJ[l] + (t2+t3)_JI[(-l) mod n]^T.
                    self._fold_reversed_legs((stx,), g_lesser)
                if verify_now:
                    # Identity gate (single rank: full tau + all pairs
                    # local): (t5+t6)_IJ[l, a, b] == (t2+t3)_JI[-l, b, a].
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
                    # Summed over pool chunks: CPU time per output pair (the
                    # wall time of one chunk's pair-loop pass, accumulated).
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
                    # (tau, *nk, b, b) -> (tau, N_q_local, b, b); the q-axis is
                    # contiguous in the same C-order as global_stack_shape[1:].
                    # Sized from the ARRAY rather than from nq: when comm.q
                    # sections the axis the local width is smaller, and the
                    # rotation carries the global count separately (it needs
                    # global bounds, the legs are local).
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
                # Distribute the EXTERNAL-q loop over comm.q. The q-folded
                # internal q' Green's functions are kept whole/local on every
                # rank (only the energy axis is split across comm.stack), so each
                # q-rank computes a disjoint subset of iq_ext from the full local
                # q' data -- no internal-q gather -- and the per-rank partial
                # Sigma(q_ext) are summed over comm.q after the loop. This is the
                # dedicated q axis: N_q-way parallelism on top of the energy axis.
                q_lo = ranks.q.rank * nq // ranks.q.size
                q_hi = (ranks.q.rank + 1) * nq // ranks.q.size

                if self._vfactors is not None:
                    # decomposed_kernel="reconstruct": dense path fed the
                    # factor-reconstructed rank-local vertex slice.
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
                    # ji-transpose the cross terms in the stack state (same
                    # exchange pattern as the Gamma fast path); the tau
                    # reversal AND the q_ext -> -q_ext gather complete the
                    # fold in stage 5 (nnz state, full tau/q axes local).
                    self._fold_reversed_legs((stx,), g_lesser)

            # Assemble the external-q distribution: each comm.q rank computed a
            # disjoint subset of iq_ext (others left zero), so sum over comm.q
            # (in fast mode also the cross-term accumulator stx -- the fold
            # permutation is rank-independent, so it commutes with the sum).
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
                # Complete the mirrored lower blocks (and, in fast mode, the
                # folded upper cross terms) with their tau reversal: the
                # stack-state fill could only conjugate-transpose (tau is
                # split there); the reversal is local here (full tau axis).
                # RHS advanced indexing copies first -- alias-safe.
                ml, mu = self._herm_masks(g_lesser)
                stl.data[1:, ..., ml] = stl.data[:0:-1, ..., ml]
                stg.data[1:, ..., ml] = stg.data[:0:-1, ..., ml]
                if fast_now:
                    stx.data[1:, ..., mu] = stx.data[:0:-1, ..., mu]
            if fast_now:
                # Complete the Sigma^> reconstruction: add the ji-transposed
                # cross terms with the tau axis reversed (circular; l = 0
                # self-maps, n_fft odd so there is no Nyquist bin). At
                # coupled-q the identity carries exactly one extra
                # operation, the external q -> -q gather on the transverse
                # axes (Gamma-centered mesh, closed under q -> -q; the same
                # negation as the stage-1 reversed legs).
                xs = stx.data
                if nq > 1:
                    for ax, k in enumerate(nk, start=1):
                        xs = xp.take(xs, (-xp.arange(k)) % k, axis=ax)
                stg.data[0] += xs[0]
                stg.data[1:] += xs[:0:-1]
            sl_conv = prefactor * xp.fft.ifft(stl.data, axis=0)[:ne_conv]
            sg_conv = prefactor * xp.fft.ifft(stg.data, axis=0)[:ne_conv]
            if q_split:
                # The SCATTER half of the reduce-scatter. The rotation
                # accumulated at every external momentum and the comm.q sum
                # above completed them, so every rank now holds the WHOLE
                # Sigma; the outputs are sectioned, so each keeps only the
                # section it owns. Cut here, before the corrections and masks,
                # so everything downstream sees one consistent q extent.
                #
                # ``q_distributed`` admits exactly one transverse axis
                # (dsdbsparse.py: a mesh with several must be flattened first),
                # so this is axis 1 and the slice is the buffer's own.
                q_out = sigma_lesser.local_q_slice
                sl_conv = sl_conv[:, q_out]
                sg_conv = sg_conv[:, q_out]
            if self._bubble_correction is not None:
                # Before the masks and before delta: see set_bubble_correction.
                _cl, _cg = self._bubble_correction
                sl_conv = sl_conv + _cl
                sg_conv = sg_conv + _cg
            if self._pole_enabled and self._pole_sigma_mixed is not None:
                # The mixed sectors join the RAW bubble output here, before the
                # masks and before delta is formed, so the existing
                # Kramers-Kronig transform covers them along with Sigma_RR.
                _mx_l, _mx_g = self._pole_sigma_mixed
                sl_conv = sl_conv + _mx_l
                sg_conv = sg_conv + _mx_g
            # OUTPUT mask, completing the input masking above: the scattering
            # Sigma is NOT applied below the SSE cutoff (transport there stays
            # ballistic), and never at the omega=0 bin (a nonzero Sigma^{<,>}(0)
            # hits the near-singular acoustic G^R(0); the bin carries zero heat
            # anyway).
            if bool(out_mask.any()):
                sl_conv[out_mask] = 0.0
                sg_conv[out_mask] = 0.0
            sl_data, sg_data = sl_conv, sg_conv
            # SIGN CONVENTION: fed with this solver's occupation-positive
            # Green's functions (-i G^{<,>} >= 0, the same convention as the
            # lead injection sigma^{<,>} = +i n(+1) gamma), the bubble returns
            # TEXTBOOK-signed sigma^{<,>} (-i sigma^{<,>} <= 0) -- the exact
            # negative of what the Keldysh feedback G^{<,>} = G^R sigma^{<,>} G^A
            # expects (unflipped it injects anti-dissipation), so negate here.
            if self._pole_enabled and self._pole_sigma_ss is not None:
                # Add the analytic pole-pole contribution on the PRIMARY grid.
                _ss_l, _ss_g, _ss_r = self._pole_sigma_ss
                if bool(out_mask.any()):
                    # The omega = 0 bin carries no scattering by convention (a
                    # nonzero Sigma^{<,>}(0) hits the near-singular acoustic
                    # G^R(0)). The analytic term obeys the same mask as the
                    # numerical one, or the two halves of Sigma disagree about
                    # which bins exist.
                    _ss_l = _ss_l.copy(); _ss_l[out_mask] = 0.0
                    _ss_g = _ss_g.copy(); _ss_g[out_mask] = 0.0
                sl_data = sl_data + _ss_l
                sg_data = sg_data + _ss_g
            sigma_lesser.data[:] = sigma_lesser.data - sl_data
            sigma_greater.data[:] = sigma_greater.data - sg_data
            # Sigma^R contribution (from the RAW, textbook-signed values:
            # Gamma = i(sigma^> - sigma^<)_raw >= 0, matching the lead OBC
            # damping sign -- unchanged by the convention flip above).
            if self.retarded_method == "fft":
                delta = sg_conv - sl_conv
                self._check_kk_grid_support(delta)
                # transverse_shape: the exact bosonic mirror carries q -> -q on
                # the transverse axes (exact off equilibrium, unlike the plain
                # conjugate shortcut).
                hil = 0.5j * hilbert_transform(delta, conv_freqs,
                                               transverse_shape=nk)
                if bool(out_mask.any()):
                    # post-mask the retarded consistently with Sigma^<>
                    hil[out_mask] = 0.0
                sigma_retarded.data[:] = sigma_retarded.data + hil
            if self._pole_enabled and self._pole_sigma_ss is not None:
                # Closed-form Kramers-Kronig partner of the analytic sector.
                # Routing it through the discrete Hilbert transform would both
                # cost resolution and put back the grid dependence the sector
                # removes. It uses the same endpoint mask as the numerical half.
                _ss_r = self._pole_sigma_ss[2]
                if bool(out_mask.any()):
                    _ss_r = _ss_r.copy()
                    _ss_r[out_mask] = 0.0
                sigma_retarded.data[:] = sigma_retarded.data + _ss_r
            # Never let an injected channel survive into the next iteration: it
            # was built from THIS iterate's self-energy.
            self._pole_channel = None
            self._pole_sigma_ss = None
            self._pole_sigma_mixed = None
            self._bubble_correction = None

        # Self-consistent SCP cubic-tadpole static self-energy
        if self._scp_tadpole and self._sigma_static is not None:
            self._apply_scp_tadpole(g_lesser, g_greater, sigma_retarded)

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

        # A rank can own NO frequencies: the stack split pads up to
        # ceil(ne / n_ranks) per rank, so e.g. 121 frequencies over 32 ranks
        # leaves the last rank empty. There is nothing to check there, and
        # reducing over the empty slice raises.
        if delta.size == 0:
            return

        peak = float(get_host(xp.max(xp.abs(delta))))
        if peak == 0.0:
            return

        edge = float(get_host(xp.max(xp.abs(delta[-1]))))
        if edge > 1e-2 * peak:
            warnings.warn(
                f"The 3-phonon bubble still carries {edge / peak:.1%} of its "
                "peak weight at the top of the frequency grid, so the "
                f"Kramers-Kronig integral for Re Sigma^R is truncated. "
                "Extend energy_window_max (or the grid in "
                "phonon_energies.npy) to about twice the phonon band top.",
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
                # 3 GEMMs per ring call; 6 ring calls per quad in the legacy
                # 3-term fold for each of Sigma^{<,>}, 4 with the Sigma^>
                # reconstruction; 8 real flops per complex MAC.
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

    @staticmethod
    def _qfold_is_translation_invariant(qv, xp) -> bool:
        """Exact test: does every q-folded vertex block satisfy
        ``Phi[(I+1, K1+1, K2+1)] == Phi[(I, K1, K2)]``?

        Only shifts whose image is also present are compared, and at least
        one such pair must exist -- otherwise there is nothing to share and
        we must not claim invariance. Any mismatch (including a missing
        partner where a present one is expected) returns False, so the
        caller falls back to the absolute key.
        """
        compared = 0
        for phi in qv.values():
            if not isinstance(phi, dict):
                return False
            for (I, K1, K2), blk in phi.items():
                shifted = phi.get((I + 1, K1 + 1, K2 + 1))
                if shifted is None:
                    continue
                a = xp.asarray(blk)
                b = xp.asarray(shifted)
                if a.shape != b.shape or not bool(xp.all(a == b)):
                    return False
                compared += 1
        return compared > 0

    def _rotate_internal_q(
        self, owned, qdm, qv, q_lo, q_hi, nq, nk, n_tau, dtype,
        legs, _fold_l, _fold_g, stlv, stgv, start, xp,
        q_split=False, **kw,
    ):
        """Drive :meth:`_contract_dense_q` over the internal-momentum pairs.

        With the transverse axis replicated this is one call over the whole
        axis -- the pre-rotation behaviour, bit for bit.

        With it sectioned (``q_split``) the bubble cannot be evaluated from a
        rank's own slice alone: it is a CONVOLUTION over q, so every term at
        an external momentum needs a SECOND internal momentum, generally on
        another rank. The fix is systolic rather than a gather, because a
        gather would restore the replication the sectioning exists to remove:

        * leg A stays this rank's own slice throughout;
        * leg B starts as a copy of it and is passed once around ``comm.q``,
          so after step ``s`` rank ``r`` holds the slice owned by
          ``(r + s) mod P_q``;
        * each step contracts the pairs it now holds and ACCUMULATES.

        Across all ranks and steps every ordered slice pair occurs exactly
        once, so the q sum is complete
        (``test_internal_q_slice_pairs_tile_the_convolution``). The existing
        ``all_reduce`` over ``comm.q`` still finishes the job: each rank now
        contributes partial sums to every external momentum instead of
        complete sums to a disjoint few, which that collective handles
        identically.

        See ``phonon/docs/bubble_positivity.md`` Sec. 7.
        """
        bounds = [r * nq // ranks.q.size for r in range(ranks.q.size + 1)]
        contract = (
            self._contract_micro_dense_q
            if self._micro_pair_index is not None else self._contract_dense_q
        )
        if not q_split:
            # Whole axis local: one pass, offsets zero, plain assignment.
            contract(
                owned, qdm, qv, q_lo, q_hi, nq, nk, n_tau, dtype,
                *legs, _fold_l, _fold_g, stlv, stgv, start, xp, **kw)
            return

        rank, size = ranks.q.rank, ranks.q.size
        a_lo, a_hi = bounds[rank], bounds[rank + 1]
        # Leg B rotates; start from a copy so leg A is never aliased by the
        # receive buffer (the first step contracts A against itself).
        buf = tuple({k: xp.array(v, copy=True) for k, v in d.items()}
                    for d in legs)
        for step in range(size):
            src = (rank + step) % size
            b_lo, b_hi = bounds[src], bounds[src + 1]
            # NOT (q_lo, q_hi): the external-q split is how the REPLICATED
            # path divides work, and it is mutually exclusive with this one.
            # Here every rank produces partial sums at every external
            # momentum -- its own slice paired with whatever it currently
            # holds -- and the all_reduce over comm.q completes them.
            contract(
                owned, qdm, qv, 0, nq, nq, nk, n_tau, dtype,
                *legs, _fold_l, _fold_g, stlv, stgv, start, xp,
                a_slice=(a_lo, a_hi), b_slice=(b_lo, b_hi),
                a_off=a_lo, b_off=b_lo, legs_b=buf,
                accumulate=(step > 0), **kw)
            if step == size - 1:
                break
            buf = self._rotate_q_buffers(buf, bounds, rank, size, step, xp)

    def _micro_links_for_outputs(
        self, outputs: list[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        """Primitive Green links consumed by grouped output pairs."""
        links: set[tuple[int, int]] = set()
        for pair in outputs:
            for quads in self._micro_pair_index.get(pair, {}).values():
                for quad in quads:
                    links.add((quad.k1, quad.k1p))
                    links.add((quad.k2, quad.k2p))
        return links

    def _contract_micro_dense_gamma(
        self, owned, *, n_tau, dtype, gl, gg, glr, ggr,
        stlv, stgv, start, xp,
    ) -> None:
        """Dense Gamma bubble on primitive FC3 and grouped Dyson blocks."""
        if getattr(self, "_micro_gamma_pre", None) is None:
            pre = {}
            for group_pair, primitive_pairs in self._micro_pair_index.items():
                tasks = []
                for (i, j), quads in primitive_pairs.items():
                    _bi, si = self._micro_layout.locate(i)
                    _bj, sj = self._micro_layout.locate(j)
                    for quad in quads:
                        tasks.append((
                            si.start, sj.start, quad.k1, quad.k1p,
                            quad.k2, quad.k2p,
                        ) + phi_perms(
                            xp.asarray(quad.phi_left),
                            xp.asarray(quad.phi_right), xp,
                        ))
                pre[group_pair] = tasks
            self._micro_gamma_pre = pre

        def _run(lo, hi):
            result = {}
            for I, J in owned:
                out_l = xp.zeros(
                    (hi - lo, int(self.block_sizes[I]),
                     int(self.block_sizes[J])), dtype=dtype)
                out_g = xp.zeros_like(out_l)
                for (oi, oj, k1, k1p, k2, k2p, *p) in (
                    self._micro_gamma_pre.get((I, J), ())
                ):
                    PL, PR, nI, bK2, nJ = p
                    gla, glb = gl[(k1, k1p)][lo:hi], gl[(k2, k2p)][lo:hi]
                    gga, ggb = gg[(k1, k1p)][lo:hi], gg[(k2, k2p)][lo:hi]
                    ggra, ggrb = ggr[(k1, k1p)][lo:hi], ggr[(k2, k2p)][lo:hi]
                    glra, glrb = glr[(k1, k1p)][lo:hi], glr[(k2, k2p)][lo:hi]

                    def ring(a, b):
                        return ring_contract_pre(
                            PL, PR, nI, bK2, nJ, a, b, xp)

                    sl = ring(gla, glb) + ring(gla, ggrb) + ring(ggra, glb)
                    sg = ring(gga, ggb) + ring(gga, glrb) + ring(glra, ggb)
                    d = self._micro_layout.micro_dof
                    out_l[:, oi:oi + d, oj:oj + d] += sl
                    out_g[:, oi:oi + d, oj:oj + d] += sg
                result[(I, J)] = (out_l, out_g)
            return result

        pool, n_threads = ring_pool()
        nt = min(n_threads, max(1, n_tau // self._tau_min_chunk))
        if pool is not None and xp is np and nt > 1:
            bounds = [(i * n_tau // nt, (i + 1) * n_tau // nt)
                      for i in range(nt)]
            chunks = list(pool.map(lambda bound: _run(*bound), bounds))
            result = {
                pair: tuple(xp.concatenate([c[pair][side] for c in chunks],
                                           axis=0)
                            for side in (0, 1))
                for pair in owned
            }
        else:
            result = _run(0, n_tau)
        for (I, J), (out_l, out_g) in result.items():
            stlv.blocks[I - start, J - start] = out_l
            stgv.blocks[I - start, J - start] = out_g

    def _contract_micro_dense_q(
        self, owned, qdm, qv, q_lo, q_hi, nq, nk, n_tau, dtype,
        gl_q, gg_q, glr_q, ggr_q, _fold_l, _fold_g,
        stlv, stgv, start, xp,
        fast_now=False, verify_now=False, stxv=None, release=None,
        a_slice=None, b_slice=None, a_off=0, b_off=0, legs_b=None,
        accumulate=False,
    ) -> None:
        """Dense coupled-q reference contraction on primitive microblocks."""
        if fast_now or verify_now:
            raise NotImplementedError(
                "primitive microblocks use the plain six-ring coupled-q path")
        del stxv, release
        a_lo, a_hi = (0, nq) if a_slice is None else a_slice
        b_lo, b_hi = (0, nq) if b_slice is None else b_slice
        gl_a, gg_a, glr_a, ggr_a = gl_q, gg_q, glr_q, ggr_q
        gl_b, gg_b, glr_b, ggr_b = (
            (gl_q, gg_q, glr_q, ggr_q) if legs_b is None else legs_b)
        cache_key = (q_lo, q_hi, nq, a_lo, a_hi, b_lo, b_hi, tuple(owned))
        # A q-sectioned calculation visits one (q_a, q_b) tile at a time as
        # leg B rotates around comm.q.  Do not accumulate every tile's task
        # metadata on the instance: taken together they are the complete
        # O(nq**2) plan, whereas a rotation consumes them one at a time.  The
        # contraction below also caches left and right FC3 permutations
        # independently, rather than caching their combinatorial pairing.
        # These are exact scheduling changes; the same q pairs and ring
        # contractions are accumulated below.
        transient_tile = a_slice is not None or b_slice is not None
        cache = getattr(self, "_micro_qtasks_cache", {})
        qtasks = None if transient_tile else cache.get(cache_key)
        if qtasks is None:
            # qdm[Q, q1] = q2 is the authoritative mesh arithmetic.  Its
            # inverse gives Q for each (q1, q2).  Flat-index addition is only
            # valid for a one-dimensional mesh; on a film mesh such as 9x9 it
            # carries from the second transverse coordinate into the first
            # and changes roughly half the momentum-conserving triples.
            qsum = np.empty((nq, nq), dtype=np.int64)
            qprime = np.arange(nq, dtype=np.int64)
            for iq_ext in range(nq):
                qsum[qprime, np.asarray(qdm[iq_ext], dtype=np.int64)] = iq_ext
            # Cache the two fixed vertex permutations INDEPENDENTLY.  A paired
            # cache duplicates one PL for every PR it meets (and vice versa),
            # so its memory follows the number of ring tasks; exact q5 Si
            # filled a 96-GiB GH200 before map zero.  Separate caches contain
            # at most one PL and one PR per q-folded FC3 block, hence follow
            # the input vertex size instead of the combinatorial pairing.
            perm_left: dict[tuple, NDArray] = {}
            perm_right: dict[tuple, NDArray] = {}
            qtasks = {}
            bulk_vertex = self._vfactors is not None
            for iqp in range(a_lo, a_hi):
                for iq2 in range(b_lo, b_hi):
                    iq_ext = int(qsum[iqp, iq2])
                    if not q_lo <= iq_ext < q_hi:
                        continue
                    phi_l = qv.get((iqp, iq2))
                    phi_r = qv.get((iq2, iqp))
                    if phi_l is None or phi_r is None:
                        continue
                    for group_pair in owned:
                        for (i, j), quads in self._micro_pair_index.get(
                            group_pair, {}
                        ).items():
                            _bi, si = self._micro_layout.locate(i)
                            _bj, sj = self._micro_layout.locate(j)
                            for quad in quads:
                                pl = phi_l.get((i, quad.k1, quad.k2))
                                pr = phi_r.get((j, quad.k2p, quad.k1p))
                                if pl is None or pr is None:
                                    continue
                                left_key = (
                                    (iqp, iq2, quad.k1 - i, quad.k2 - i,
                                     "left")
                                    if bulk_vertex else
                                    (iqp, iq2, i, quad.k1, quad.k2)
                                )
                                right_key = (
                                    (iq2, iqp, quad.k2p - j, quad.k1p - j,
                                     "right")
                                    if bulk_vertex else
                                    (iq2, iqp, j, quad.k2p, quad.k1p)
                                )
                                PL = perm_left.get(left_key)
                                if PL is None:
                                    nI, bK1, bK2 = pl.shape
                                    PL = xp.ascontiguousarray(
                                        xp.conj(xp.asarray(pl)).transpose(
                                            0, 2, 1)
                                    ).reshape(nI * bK2, bK1)
                                    perm_left[left_key] = PL
                                else:
                                    nI, bK1, bK2 = pl.shape
                                PR = perm_right.get(right_key)
                                if PR is None:
                                    nJ, bK2p, bK1p = pr.shape
                                    PR = xp.ascontiguousarray(
                                        xp.asarray(pr).transpose(1, 2, 0)
                                    ).reshape(bK2p, bK1p * nJ)
                                    perm_right[right_key] = PR
                                else:
                                    nJ = pr.shape[0]
                                p = (PL, PR, nI, bK2, nJ)
                                qtasks.setdefault(group_pair, []).append((
                                    iq_ext, iqp, iq2, si.start, sj.start,
                                    quad.k1, quad.k1p, quad.k2, quad.k2p,
                                ) + p)
            if not transient_tile:
                cache[cache_key] = qtasks
                self._micro_qtasks_cache = cache

        if not getattr(self, "_ring_stats_printed", False) and ranks.rank == 0:
            self._ring_stats_printed = True
            ntasks = sum(len(v) for v in qtasks.values())
            print(
                "PhPh SSE ring (microblock dense coupled-q): "
                f"group_pairs={len(qtasks)} qtasks={ntasks} nq={nq} "
                f"q_local={q_hi - q_lo} n_tau={n_tau}", flush=True)

        def _run(lo, hi):
            result = {}
            d = self._micro_layout.micro_dof
            for I, J in owned:
                out_l = xp.zeros(
                    (hi - lo, nq, int(self.block_sizes[I]),
                     int(self.block_sizes[J])), dtype=dtype)
                out_g = xp.zeros_like(out_l)
                for (iqe, iqp, iq2, oi, oj, k1, k1p, k2, k2p,
                     *p) in qtasks.get((I, J), ()):
                    gla = gl_a[(k1, k1p)][lo:hi, iqp - a_off]
                    glb = gl_b[(k2, k2p)][lo:hi, iq2 - b_off]
                    gga = gg_a[(k1, k1p)][lo:hi, iqp - a_off]
                    ggb = gg_b[(k2, k2p)][lo:hi, iq2 - b_off]
                    ggra = ggr_a[(k1, k1p)][lo:hi, iqp - a_off]
                    ggrb = ggr_b[(k2, k2p)][lo:hi, iq2 - b_off]
                    glra = glr_a[(k1, k1p)][lo:hi, iqp - a_off]
                    glrb = glr_b[(k2, k2p)][lo:hi, iq2 - b_off]
                    sl = _fold_l(p, gla, glb, ggra, ggrb)
                    sg = _fold_g(p, gga, ggb, glra, glrb)
                    out_l[:, iqe, oi:oi + d, oj:oj + d] += sl
                    out_g[:, iqe, oi:oi + d, oj:oj + d] += sg
                result[(I, J)] = (out_l, out_g)
            return result

        def _run_batched(lo, hi):
            """GPU/NumPy task-batched form of the same primitive ring.

            The original microblock reference loop launched six sets of three
            tiny matrix multiplications for every q/quad task.  At q9 L5 that
            is more than five million Python iterations.  Flattening the task
            axis into strided batched GEMMs changes only the final reduction
            order and is the same optimisation used by the grouped dense
            kernel below.
            """
            w = hi - lo

            def _stack(family, keys):
                if set(family) != set(keys):
                    return None
                return xp.stack([
                    xp.ascontiguousarray(family[key]) for key in keys
                ])

            a_keys = sorted(gl_a)
            b_keys = sorted(gl_b)
            a_index = {key: i for i, key in enumerate(a_keys)}
            b_index = {key: i for i, key in enumerate(b_keys)}
            GL = _stack(gl_a, a_keys)
            GG = _stack(gg_a, a_keys)
            GGR = _stack(ggr_a, a_keys)
            GLR = _stack(glr_a, a_keys)
            GLb = _stack(gl_b, b_keys)
            GGb = _stack(gg_b, b_keys)
            GGRb = _stack(ggr_b, b_keys)
            GLRb = _stack(glr_b, b_keys)
            if any(value is None for value in
                   (GL, GG, GGR, GLR, GLb, GGb, GGRb, GLRb)):
                return _run(lo, hi)

            def _scatter_micro(out, iqe, oi, oj, values, bs_i, bs_j):
                # values: (C, wt, d, d).  Flatten (q,row,col) into one
                # target index and retain tau as the contiguous value axis.
                d = values.shape[-1]
                rr = xp.arange(d, dtype=xp.int64)[None, :, None]
                cc = xp.arange(d, dtype=xp.int64)[None, None, :]
                target = (
                    iqe[:, None, None] * (bs_i * bs_j)
                    + (oi[:, None, None] + rr) * bs_j
                    + (oj[:, None, None] + cc)
                ).reshape(-1)
                vals = values.transpose(0, 2, 3, 1).reshape(-1, values.shape[1])
                out_flat = out.reshape(out.shape[0], -1).T
                if xp is np:
                    np.add.at(out_flat, target, vals)
                else:
                    import cupyx
                    cupyx.scatter_add(out_flat.real, target, vals.real)
                    cupyx.scatter_add(out_flat.imag, target, vals.imag)

            result = {}
            for I, J in owned:
                tasks = qtasks.get((I, J), ())
                bs_i = int(self.block_sizes[I])
                bs_j = int(self.block_sizes[J])
                out_l = xp.zeros((w, nq, bs_i, bs_j), dtype=dtype)
                out_g = xp.zeros_like(out_l)
                if not tasks:
                    result[(I, J)] = (out_l, out_g)
                    continue

                groups: dict[tuple, list] = {}
                for task in tasks:
                    PL, PR, nI, bK2, nJ = task[9:14]
                    groups.setdefault(
                        (PL.shape, PR.shape, nI, bK2, nJ), []
                    ).append(task)
                for (shape_l, shape_r, nI, bK2, nJ), ts in groups.items():
                    n_tasks = len(ts)
                    bK1 = shape_l[1]
                    bK1p = shape_r[1] // nJ
                    per_task_tau = 16 * (
                        2 * bK1 * bK1p + shape_l[0] * bK1p
                        + bK2 * shape_r[1] + 2 * nI * nJ
                    )
                    preferred_tasks = min(n_tasks, 256)
                    tau_width = int(max(1, min(
                        w, self._tau_chunk_bytes
                        // max(per_task_tau * preferred_tasks, 1)
                    )))
                    task_width = max(1, min(
                        n_tasks, self._tau_chunk_bytes
                        // max(per_task_tau * tau_width, 1)
                    ))
                    iqe_all = xp.asarray([t[0] for t in ts], dtype=xp.int64)
                    qa_all = xp.asarray([t[1] for t in ts], dtype=xp.int64) - a_off
                    qb_all = xp.asarray([t[2] for t in ts], dtype=xp.int64) - b_off
                    oi_all = xp.asarray([t[3] for t in ts], dtype=xp.int64)
                    oj_all = xp.asarray([t[4] for t in ts], dtype=xp.int64)
                    ai_all = xp.asarray([
                        a_index[(t[5], t[6])] for t in ts
                    ], dtype=xp.int64)
                    bi_all = xp.asarray([
                        b_index[(t[7], t[8])] for t in ts
                    ], dtype=xp.int64)

                    for c0 in range(0, n_tasks, task_width):
                        c1 = min(c0 + task_width, n_tasks)
                        PLc = xp.stack([t[9] for t in ts[c0:c1]])[:, None]
                        PRc = xp.stack([t[10] for t in ts[c0:c1]])[:, None]
                        ai, bi = ai_all[c0:c1], bi_all[c0:c1]
                        qa, qb = qa_all[c0:c1], qb_all[c0:c1]

                        for w0 in range(0, w, tau_width):
                            w1 = min(w0 + tau_width, w)
                            tau = xp.arange(lo + w0, lo + w1)[None, :]

                            def _legs(A, B):
                                return (
                                    A[ai[:, None], tau, qa[:, None]],
                                    B[bi[:, None], tau, qb[:, None]],
                                )

                            def _ring(Ga, Gb):
                                left = PLc @ Ga
                                right = Gb @ PRc
                                return (
                                    left.reshape(c1 - c0, w1 - w0,
                                                 nI, bK2 * bK1p)
                                    @ right.reshape(c1 - c0, w1 - w0,
                                                    bK2 * bK1p, nJ)
                                )

                            sl = None
                            for A, B in ((GL, GLb), (GL, GGRb), (GGR, GLb)):
                                Ga, Gb = _legs(A, B)
                                term = _ring(Ga, Gb)
                                sl = term if sl is None else sl + term
                            sg = None
                            for A, B in ((GG, GGb), (GG, GLRb), (GLR, GGb)):
                                Ga, Gb = _legs(A, B)
                                term = _ring(Ga, Gb)
                                sg = term if sg is None else sg + term
                            _scatter_micro(
                                out_l[w0:w1], iqe_all[c0:c1],
                                oi_all[c0:c1], oj_all[c0:c1],
                                sl.astype(dtype, copy=False), bs_i, bs_j,
                            )
                            _scatter_micro(
                                out_g[w0:w1], iqe_all[c0:c1],
                                oi_all[c0:c1], oj_all[c0:c1],
                                sg.astype(dtype, copy=False), bs_i, bs_j,
                            )
                result[(I, J)] = (out_l, out_g)
            return result

        result = (
            _run_batched(0, n_tau) if self._dense_q_batched
            else _run(0, n_tau)
        )
        for (I, J), (out_l, out_g) in result.items():
            shape = ((n_tau,) + tuple(nk) +
                     (int(self.block_sizes[I]), int(self.block_sizes[J])))
            if accumulate:
                stlv.blocks[I - start, J - start] = (
                    stlv.blocks[I - start, J - start] + out_l.reshape(shape))
                stgv.blocks[I - start, J - start] = (
                    stgv.blocks[I - start, J - start] + out_g.reshape(shape))
            else:
                stlv.blocks[I - start, J - start] = out_l.reshape(shape)
                stgv.blocks[I - start, J - start] = out_g.reshape(shape)

    @staticmethod
    def _negate_q_across_comm(x, nq, lo, hi, xp):
        """``q -> -q`` on an axis that ``comm.q`` has sectioned.

        The negation is not local: it maps a rank's own section onto the
        section of whichever rank owns the negated momenta, so it needs the
        whole axis. Rebuild it by summing zero-padded contributions -- the
        sections are disjoint, so the sum IS the gather, and it needs no
        padding arithmetic for an uneven split (``all_gather_v`` would pad to
        the widest section and the trim would have to be undone by hand).

        The full-axis temporary is transient and one leg wide; the buffers
        themselves stay sectioned, which is where the memory is.
        """
        full = xp.zeros(x.shape[:1] + (nq,) + x.shape[2:], dtype=x.dtype)
        full[:, lo:hi] = x
        recv = xp.empty_like(full)
        ranks.q.all_reduce(xp.ascontiguousarray(full), recv, op="sum")
        neg = (-xp.arange(nq)) % nq
        return xp.take(recv, neg, axis=1)[:, lo:hi]

    @staticmethod
    def _rotate_q_buffers(buf, bounds, rank, size, step, xp):
        """One hop of the leg-B ring: send to ``rank-1``, receive ``rank+1``.

        The next slice a rank needs is the one its successor currently holds,
        so the ring turns one way only. Buffers are sized from the SOURCE
        slice, which generally differs from the local one when ``nq`` does not
        divide ``P_q`` -- assuming they match is the obvious way to get a
        truncated message here.
        """
        dst = (rank - 1) % size
        src = (rank + 1) % size
        nxt = (rank + step + 1) % size
        width = bounds[nxt + 1] - bounds[nxt]
        out = []
        for d in buf:
            new = {}
            # SORTED, not dict order: the peers must post their messages in
            # one agreed sequence (there are no tags), and these dicts are
            # populated by iterating _links_for_range, which returns a SET.
            for key in sorted(d):
                val = d[key]
                shape = val.shape[:1] + (width,) + val.shape[2:]
                recv = xp.empty(shape, dtype=val.dtype)
                ranks.q.send_recv(xp.ascontiguousarray(val), dst, recv, src)
                new[key] = recv
            out.append(new)
        return tuple(out)

    def _contract_dense_q(
        self, owned, qdm, qv, q_lo, q_hi, nq, nk, n_tau, dtype,
        gl_q, gg_q, glr_q, ggr_q, _fold_l, _fold_g,
        stlv, stgv, start, xp,
        fast_now=False, verify_now=False, stxv=None, release=None,
        a_slice=None, b_slice=None, a_off=0, b_off=0, legs_b=None,
        accumulate=False,
    ):
        """DENSE coupled-q vertex-pair contraction (the reference path).

        Consumes the dense q-folded vertex dict ``qv`` and the q-flattened
        tau-domain Green's function band dicts, writes the per-(I, J)
        Sigma^{<,>} tau blocks into the stack views. See
        ``_contract_factored_q`` for the tensor-decomposed equivalent.

        ``fast_now`` (sse_greater_from_lesser): 4-ring bosonic tau fold --
        Sigma^> is contracted from the (gg, gg) ring only and its two
        absorption cross terms are the mixed Sigma^< rings of pair (J, I)
        at negated external q, folded downstream (JI/orbital transpose in
        the stack state, tau reversal + q negation in stage 5); the mixed
        rings are accumulated per (I, J, q_ext) and written into ``stxv``.
        ``verify_now``: legacy 6-ring result shipped, the reconstruction
        identity checked in place (single-rank runs only).
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
        # Which internal-momentum slices this rank currently holds. Whole
        # axis by default, which is what the replicated (non-q-distributed)
        # path always has.
        a_lo, a_hi = (0, nq) if a_slice is None else a_slice
        b_lo, b_hi = (0, nq) if b_slice is None else b_slice
        # Leg A is read at the first internal momentum, leg B at the second.
        # They come from one family in the replicated case and from two -- own
        # slice and rotating slice -- under the q rotation. ``a_off``/``b_off``
        # are the GLOBAL index of each family's first stored momentum, so they
        # are 0 whenever the arrays span the whole axis.
        gl_a, gg_a, glr_a, ggr_a = gl_q, gg_q, glr_q, ggr_q
        gl_b, gg_b, glr_b, ggr_b = (
            (gl_q, gg_q, glr_q, ggr_q) if legs_b is None else legs_b)
        cache_key = (q_lo, q_hi, nq, a_lo, a_hi, b_lo, b_hi)
        if getattr(self, "_qtasks_cache_key", None) == cache_key:
            qtasks = self._qtasks_cache
        else:
            bulk_vertex = self._vfactors is not None
            if not bulk_vertex and self._perm_share == "auto":
                # The dense q-folded vertex is I-dependent in general, but
                # for a bulk-homogeneous device its blocks repeat along the
                # block row. Verify that EXACTLY (once) rather than assume
                # it; on success the transport-offset key is equivalent and
                # collapses the cache to a single distinct key on every
                # device shipped so far.
                bulk_vertex = self._qfold_is_translation_invariant(qv, xp)
                if ranks.rank == 0:
                    print(f"PhPh SSE perm cache: q-folded vertex "
                          f"{'IS' if bulk_vertex else 'is NOT'} translation "
                          f"invariant -> "
                          f"{'offset' if bulk_vertex else 'absolute'} key",
                          flush=True)
            perm_cache: dict[tuple, tuple] = {}
            qtasks = {}
            # Indexed by the two INTERNAL momenta rather than by the
            # external one. They determine it -- ``qdm[iq_ext, iqp]`` is
            # ``(iq_ext - iqp) mod nq``, so ``iq_ext = (iqp + iq2) mod nq`` --
            # and this is the form the q-distributed rotation needs: leg A is
            # read at ``iqp`` and leg B at ``iq2``, so restricting the two
            # loops to the two slices a rank currently holds selects exactly
            # the pairs it can compute. See phonon/docs/bubble_positivity.md
            # Sec. 7.
            #
            # Bit-identical to iterating ``iq_ext`` outer: the map is a
            # bijection on the pairs, distinct ``iq_ext`` accumulate into
            # distinct memory, and for a fixed ``iq_ext`` the ``iqp`` still
            # arrive ascending, so no sum is reassociated.
            for iqp in range(a_lo, a_hi):
                for iq2 in range(b_lo, b_hi):
                    iq_ext = (iqp + iq2) % nq
                    if not q_lo <= iq_ext < q_hi:
                        continue
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
                            # Only the factor-reconstructed vertex is
                            # translationally invariant, so only there may the
                            # permuted pair be shared across the block row via
                            # the transport offsets. A dense q-folded vertex has
                            # I-dependent blocks and must be keyed in full.
                            # (Keying on id() shares nothing at all: the
                            # reconstructed blocks are fresh views per I.)
                            pkey = (
                                (iqp, iq2, K1 - I, K2 - I, K2p - J, K1p - J)
                                if bulk_vertex
                                else (iqp, iq2, I, K1, K2, J, K2p, K1p)
                            )
                            pre = perm_cache.get(pkey)
                            if pre is None:
                                pre = phi_perms(
                                    xp.conj(xp.asarray(pl)), xp.asarray(pr), xp
                                )
                                perm_cache[pkey] = pre
                            qtasks.setdefault((I, J), []).append(
                                (iq_ext, iqp, iq2, K1, K1p, K2, K2p) + pre)
            self._qtasks_cache_key = cache_key
            self._qtasks_cache = qtasks

        # One-time stage-3 cost model for the COUPLED-Q ring (mirrors
        # _print_ring_stats): 6 ring calls of 3 GEMMs per task, 8 real
        # flops per complex MAC. Without this every film run reported
        # _ring_model_gflop = 0 and no in-engine GF/s could be derived.
        if not getattr(self, "_ring_stats_printed", False) and ranks.rank == 0:
            self._ring_stats_printed = True
            n_tasks = sum(len(t) for t in qtasks.values())
            n_rings = 4 if fast_now else 6
            flops = 0.0
            for tasks in qtasks.values():
                for t in tasks:
                    PL, PR, nI, bK2, nJ = t[7], t[8], t[9], t[10], t[11]
                    bK1 = PL.shape[1]
                    bK2p = PR.shape[0]
                    bK1p = PR.shape[1] // nJ
                    flops += n_rings * 8 * n_tau * (
                        nI * bK2 * bK1 * bK1p
                        + bK2 * bK2p * bK1p * nJ
                        + nI * bK2 * bK1p * nJ
                    )
            self._ring_model_gflop = flops / 1e9
            print(
                f"PhPh SSE ring (coupled-q): pairs={len(qtasks)} "
                f"qtasks={n_tasks} nq={nq} q_local={q_hi - q_lo} "
                f"n_tau={n_tau} model={flops / 1e9:.1f} GFLOP/pass",
                flush=True,
            )

        # Sigma^{<,>}(I, J, q_ext) for the tau slice [lo:hi]; mirrors the
        # nq==1 _contract_tau so the omega/tau batch parallelises across
        # the ring pool. In fast/verify mode the two mixed (absorption)
        # Sigma^< rings are kept separate and accumulated into out_x --
        # they double as the Sigma^> cross-term source of the bosonic tau
        # fold; in verify mode the direct mixed Sigma^> rings go to
        # out_t56 for the in-place identity gate.
        def _contract_tau_q(lo, hi):
            res = {}
            need_x = fast_now or verify_now
            for (I, J), tasks in qtasks.items():
                bs_I = int(self.block_sizes[I])
                bs_J = int(self.block_sizes[J])
                out_l = xp.zeros((hi - lo, nq, bs_I, bs_J), dtype=dtype)
                out_g = xp.zeros((hi - lo, nq, bs_I, bs_J), dtype=dtype)
                out_x = (xp.zeros((hi - lo, nq, bs_I, bs_J), dtype=dtype)
                         if need_x else None)
                out_t56 = (xp.zeros((hi - lo, nq, bs_I, bs_J), dtype=dtype)
                           if verify_now else None)
                for (iq_ext, iqp, iq2, K1, K1p, K2, K2p, *pre) in tasks:
                    gla = gl_a[(K1, K1p)][lo:hi, iqp - a_off]
                    glb = gl_b[(K2, K2p)][lo:hi, iq2 - b_off]
                    ggra = ggr_a[(K1, K1p)][lo:hi, iqp - a_off]
                    ggrb = ggr_b[(K2, K2p)][lo:hi, iq2 - b_off]
                    tx = t56 = None
                    if need_x:
                        PL, PR, nI, bK2, nJ = pre
                        tx = (ring_contract_pre(
                                  PL, PR, nI, bK2, nJ, gla, ggrb, xp)
                              + ring_contract_pre(
                                  PL, PR, nI, bK2, nJ, ggra, glb, xp))
                        sl = ring_contract_pre(
                            PL, PR, nI, bK2, nJ, gla, glb, xp) + tx
                    else:
                        sl = _fold_l(pre, gla, glb, ggra, ggrb)
                    gga = gg_a[(K1, K1p)][lo:hi, iqp - a_off]
                    ggb = gg_b[(K2, K2p)][lo:hi, iq2 - b_off]
                    if fast_now:
                        # Exact reconstruction: only the diagonal ring
                        # (g^>, g^>); the cross terms come from (J, I)'s
                        # folded out_x at negated external q.
                        PL, PR, nI, bK2, nJ = pre
                        sg = ring_contract_pre(
                            PL, PR, nI, bK2, nJ, gga, ggb, xp)
                    elif verify_now:
                        PL, PR, nI, bK2, nJ = pre
                        t56 = (ring_contract_pre(
                                   PL, PR, nI, bK2, nJ, gga,
                                   glr_b[(K2, K2p)][lo:hi, iq2 - b_off], xp)
                               + ring_contract_pre(
                                   PL, PR, nI, bK2, nJ,
                                   glr_a[(K1, K1p)][lo:hi, iqp - a_off], ggb, xp))
                        sg = ring_contract_pre(
                            PL, PR, nI, bK2, nJ, gga, ggb, xp) + t56
                    else:
                        sg = _fold_g(
                            pre,
                            gga,
                            ggb,
                            glr_a[(K1, K1p)][lo:hi, iqp - a_off],
                            glr_b[(K2, K2p)][lo:hi, iq2 - b_off],
                        )
                    out_l[:, iq_ext] += sl
                    out_g[:, iq_ext] += sg
                    if tx is not None:
                        out_x[:, iq_ext] += tx
                    if t56 is not None:
                        out_t56[:, iq_ext] += t56
                res[(I, J)] = (out_l, out_g, out_x, out_t56)
            return res

        def _write_q(I, J, out_l, out_g, out_x=None):
            """Write one slice pair's contribution into the Sigma stack.

            ACCUMULATES when ``accumulate`` is set, which is what the
            q-distributed rotation needs: a rank visits ``P_q`` slice pairs
            and each contributes to the same external momenta, so a plain
            assignment would keep only the last step. Assignment stays the
            default so the replicated path -- one pass over the whole axis --
            is untouched.
            """
            bs_I = int(self.block_sizes[I])
            bs_J = int(self.block_sizes[J])
            blk_shape = (n_tau,) + tuple(nk) + (bs_I, bs_J)
            i, j = I - start, J - start
            if accumulate:
                stlv.blocks[i, j] = stlv.blocks[i, j] + out_l.reshape(blk_shape)
                stgv.blocks[i, j] = stgv.blocks[i, j] + out_g.reshape(blk_shape)
                if out_x is not None:
                    stxv.blocks[i, j] = (stxv.blocks[i, j]
                                         + out_x.reshape(blk_shape))
                return
            stlv.blocks[i, j] = out_l.reshape(blk_shape)
            stgv.blocks[i, j] = out_g.reshape(blk_shape)
            if out_x is not None:
                stxv.blocks[i, j] = out_x.reshape(blk_shape)

        def _scatter_add_q(out, idx, vals):
            # out (w, nq, bI, bJ); vals (C, w, bI, bJ); idx (C,) with
            # duplicates -> accumulate on axis 1 of out via a transposed
            # view (both add.at and scatter_add handle repeated indices).
            outT = out.transpose(1, 0, 2, 3)          # view
            if xp is np:
                np.add.at(outT, idx, vals)
            else:
                import cupyx
                # cupy.add.at has no complex support -> scatter the real
                # and imaginary float64 views separately.
                #
                # This is why the GPU bubble is not reproducible run to
                # run: cupyx.scatter_add resolves duplicate indices with
                # atomicAdd, whose ORDER is not fixed. Measured on an RTX
                # A1000 (2026-08-16): repeated scatter_add of 4e5 values
                # into 4096 slots varies at 5.4e-13 relative, and the
                # whole compute() varies at ~1e-13 in ~88% of Sigma's
                # entries with the bed, the settings and the process all
                # held fixed. np.add.at above is exact, so the CPU path
                # IS bit-reproducible -- run bit-identity gates under
                # QTX_ARRAY_MODULE=numpy, never on the GPU. The jitter is
                # ~10 orders below the SCBA residual floor and explains
                # no convergence behaviour.
                cupyx.scatter_add(outT.real, idx, vals.real)
                cupyx.scatter_add(outT.imag, idx, vals.imag)

        def _contract_tau_q_batched(lo, hi):
            """Same math as _contract_tau_q with the (q', quad) task axis
            flattened into strided-batched GEMMs (per (I, J), grouped by
            ring shape). Reduction order differs from the task loop only
            within the scatter-add -> gate at ~1e-12, not bit."""
            w = hi - lo
            # Stack each G dict once: {(K, K'): (n_tau, nq, b, b)} ->
            # (n_pairs, n_tau, nq, b, b) + index map (per shape group the
            # block sizes are uniform, so one stack per dict suffices for
            # uniform-block devices; mixed-block devices fall back).
            def _stack(d):
                keys = sorted(d.keys())
                shapes = {d[k].shape[-2:] for k in keys}
                if len(shapes) > 1:
                    return None, None
                return xp.stack([xp.ascontiguousarray(d[k]) for k in keys]), \
                    {k: i for i, k in enumerate(keys)}
            # Leg A is always this rank's own family; leg B is the SAME
            # family on the replicated path and the ROTATING one under the q
            # rotation, where it is a different rank's slice and generally a
            # different width. Stacking only the A family (as this path did
            # until 2026-08-16) silently contracts a rank's own slice against
            # itself and, once the widths differ, indexes out of bounds. The
            # serial path above always read gl_b/gg_b/glr_b/ggr_b.
            GL, gl_i = _stack(gl_a)
            GG, gg_i = _stack(gg_a)
            GGR, ggr_i = _stack(ggr_a)
            # Fast mode never contracts the reversed-lesser legs (its
            # buffer is repurposed as stx and holds zeros here).
            GLR, _glr_i = (GG, gg_i) if fast_now else _stack(glr_a)
            if legs_b is None:
                GLb, GGb, GGRb, GLRb, glb_i = GL, GG, GGR, GLR, gl_i
            else:
                GLb, glb_i = _stack(gl_b)
                GGb, _ = _stack(gg_b)
                GGRb, _ = _stack(ggr_b)
                GLRb = GGb if fast_now else _stack(glr_b)[0]
            if any(x is None for x in
                   (GL, GG, GGR, GLR, GLb, GGb, GGRb, GLRb)):
                return _contract_tau_q(lo, hi)
            if release is not None:
                # The stacks now hold every value the leg dicts did, and the
                # dicts are dead from here on (the fallback above already
                # returned). Releasing them here rather than at the end of
                # the SSE removes one full copy of 4L (n_tau, N_q, b, b)
                # from the peak -- 13.9 GB of a 100 GB phase on 6-cell MoS2.
                release()

            need_x = fast_now or verify_now
            res = {}
            for (I, J), tasks in qtasks.items():
                bs_I = int(self.block_sizes[I])
                bs_J = int(self.block_sizes[J])
                out_l = xp.zeros((w, nq, bs_I, bs_J), dtype=dtype)
                out_g = xp.zeros((w, nq, bs_I, bs_J), dtype=dtype)
                out_x = (xp.zeros((w, nq, bs_I, bs_J), dtype=dtype)
                         if need_x else None)
                out_t56 = (xp.zeros((w, nq, bs_I, bs_J), dtype=dtype)
                           if verify_now else None)
                # group tasks by ring shape (uniform blocks -> one group)
                groups: dict[tuple, list] = {}
                for t in tasks:
                    PL, PR, nI, bK2, nJ = t[7], t[8], t[9], t[10], t[11]
                    groups.setdefault(
                        (PL.shape, PR.shape, nI, bK2, nJ), []).append(t)
                for (shpL, shpR, nI, bK2, nJ), ts in groups.items():
                    Tn = len(ts)
                    bK1 = shpL[1]
                    bK1p = shpR[1] // nJ
                    # Memory budget bounds the GEMM batch VOLUME C * wt
                    # (tasks x tau window). The old code fixed wt = w and
                    # let C absorb the budget with a floor of 16, so at
                    # large n_tau (fine aux grids) the intermediates blew
                    # past the budget ~10x. Now the tau axis is windowed
                    # too: pick wt so a preferred task depth fits, then C
                    # from the remaining budget. Throughput is flat once
                    # C * wt saturates the b-ceiling (microbench: by a few
                    # hundred), so this is a memory fix, not a speed knob.
                    per_tt = 16 * (
                        2 * bK1 * bK1p + shpL[0] * bK1p
                        + bK2 * shpR[1] + 2 * nI * nJ)
                    c_pref = min(Tn, 256)
                    wt = int(max(16, min(
                        w, self._tau_chunk_bytes // max(per_tt * c_pref, 1))))
                    C = max(16, min(Tn, self._tau_chunk_bytes
                                    // max(per_tt * wt, 1)))
                    iqe = xp.asarray([t[0] for t in ts], dtype=xp.int64)
                    a_id = xp.asarray([gl_i[(t[3], t[4])] for t in ts],
                                      dtype=xp.int64)
                    # B-family index map: the rotating stack is keyed by its
                    # own sorted links, which need not match leg A's.
                    b_id = xp.asarray([glb_i[(t[5], t[6])] for t in ts],
                                      dtype=xp.int64)
                    # GLOBAL momenta in the task tuples, LOCAL rows in the leg
                    # arrays: subtract the slice offsets, exactly as the serial
                    # path does at ``iqp - a_off`` / ``iq2 - b_off``. Both are
                    # zero unless the q rotation is running, so this is a no-op
                    # on the replicated path.
                    qa = xp.asarray([t[1] for t in ts], dtype=xp.int64) - a_off
                    qb = xp.asarray([t[2] for t in ts], dtype=xp.int64) - b_off
                    for c0 in range(0, Tn, C):
                        c1 = min(c0 + C, Tn)
                        PLc = xp.stack([t[7] for t in ts[c0:c1]])
                        PRc = xp.stack([t[8] for t in ts[c0:c1]])
                        PLc = PLc[:, None]            # (C,1,nIbK2,bK1)
                        PRc = PRc[:, None]            # (C,1,bK2p,bK1pnJ)
                        ai, bi = a_id[c0:c1], b_id[c0:c1]
                        qi, q2 = qa[c0:c1], qb[c0:c1]

                        for w0 in range(0, w, wt):
                            w1 = min(w0 + wt, w)
                            _tau_ix = xp.arange(lo + w0, lo + w1)[None, :]

                            def _legs(A, B):
                                # single gather -> (C, wt, b, b); no
                                # (C, wt, nq, b, b) intermediate (25x the
                                # needed memory).
                                Ga = A[ai[:, None], _tau_ix, qi[:, None]]
                                Gb = B[bi[:, None], _tau_ix, q2[:, None]]
                                return Ga, Gb

                            def _ring(Ga, Gb):
                                Tm = PLc @ Ga         # (C,wt,nIbK2,bK1p)
                                Um = Gb @ PRc         # (C,wt,bK2,bK1pnJ)
                                return (
                                    Tm.reshape(c1 - c0, w1 - w0,
                                               nI, bK2 * bK1p)
                                    @ Um.reshape(c1 - c0, w1 - w0,
                                                 bK2 * bK1p, nJ))

                            tx = t56 = None
                            if need_x:
                                # mixed lesser rings kept separate: they
                                # double as the Sigma^> cross-term source
                                # of the bosonic tau fold
                                Ga, Gb = _legs(GL, GGRb)
                                tx = _ring(Ga, Gb)
                                Ga, Gb = _legs(GGR, GLb)
                                tx = tx + _ring(Ga, Gb)
                                Ga, Gb = _legs(GL, GLb)
                                sl = _ring(Ga, Gb) + tx
                            else:
                                sl = None
                                for A, B in ((GL, GLb), (GL, GGRb), (GGR, GLb)):
                                    Ga, Gb = _legs(A, B)
                                    s = _ring(Ga, Gb)
                                    sl = s if sl is None else sl + s
                            if fast_now:
                                # Exact reconstruction: only the diagonal
                                # ring (g^>, g^>); the cross terms come
                                # from (J, I)'s folded out_x at negated
                                # external q.
                                Ga, Gb = _legs(GG, GGb)
                                sg = _ring(Ga, Gb)
                            elif verify_now:
                                Ga, Gb = _legs(GG, GLRb)
                                t56 = _ring(Ga, Gb)
                                Ga, Gb = _legs(GLR, GGb)
                                t56 = t56 + _ring(Ga, Gb)
                                Ga, Gb = _legs(GG, GGb)
                                sg = _ring(Ga, Gb) + t56
                            else:
                                sg = None
                                for A, B in ((GG, GGb), (GG, GLRb), (GLR, GGb)):
                                    Ga, Gb = _legs(A, B)
                                    s = _ring(Ga, Gb)
                                    sg = s if sg is None else sg + s
                            _scatter_add_q(
                                out_l[w0:w1], iqe[c0:c1],
                                sl.astype(dtype, copy=False))
                            _scatter_add_q(
                                out_g[w0:w1], iqe[c0:c1],
                                sg.astype(dtype, copy=False))
                            if tx is not None:
                                _scatter_add_q(
                                    out_x[w0:w1], iqe[c0:c1],
                                    tx.astype(dtype, copy=False))
                            if t56 is not None:
                                _scatter_add_q(
                                    out_t56[w0:w1], iqe[c0:c1],
                                    t56.astype(dtype, copy=False))
                res[(I, J)] = (out_l, out_g, out_x, out_t56)
            return res

        pool, n_threads = ring_pool()
        nt = min(n_threads, max(1, n_tau // self._tau_min_chunk))  # see nq==1
        if self._dense_q_batched:
            res = _contract_tau_q_batched(0, n_tau)
        elif pool is not None and xp is np and nt > 1:
            bnds = [(i * n_tau // nt, (i + 1) * n_tau // nt)
                    for i in range(nt)]
            chunks = list(pool.map(lambda b: _contract_tau_q(*b), bnds))
            res = {
                ij: tuple(
                    xp.concatenate([c[ij][t] for c in chunks], axis=0)
                    if chunks[0][ij][t] is not None else None
                    for t in range(4)
                )
                for ij in qtasks
            }
        else:
            res = _contract_tau_q(0, n_tau)
        for (I, J), (out_l, out_g, out_x, _t56) in res.items():
            # out_x reaches the stx buffer only in fast mode; in verify
            # mode that buffer still holds the reversed-lesser legs.
            _write_q(I, J, out_l, out_g, out_x if fast_now else None)

        if verify_now:
            # Identity gate (single rank: full tau + all q_ext + all pairs
            # local), the coupled-q mirror of the Gamma gate:
            #   (t5+t6)_IJ[l, qe, a, b] == (t2+t3)_JI[-l, -qe, b, a].
            rev = (-xp.arange(n_tau)) % n_tau
            # Flat-index permutation of the per-axis q -> -q negation
            # (C-order flattening of the transverse mesh).
            qneg = xp.arange(nq).reshape(nk)
            for ax, k in enumerate(nk):
                qneg = xp.take(qneg, (-xp.arange(k)) % k, axis=ax)
            qneg = qneg.reshape(-1)
            worst = worst_rel = 0.0
            for (I, J) in res:
                rec = res[(J, I)][2][rev][:, qneg].swapaxes(-1, -2)
                d = float(xp.max(xp.abs(res[(I, J)][3] - rec)))
                scale = float(xp.max(xp.abs(res[(I, J)][3]))) or 1.0
                worst = max(worst, d)
                worst_rel = max(worst_rel, d / scale)
            self._fold_verify_done += 1
            if ranks.rank == 0:
                print(
                    "PhPh SSE fold-verify (coupled-q) "
                    f"[{self._fold_verify_done}/{self._fold_verify}]"
                    f": max|d|={worst:.3e} rel={worst_rel:.3e} "
                    f"({'OK' if worst_rel < 1e-10 else 'MISMATCH'})",
                    flush=True,
                )

    def _contract_factored_q(
        self, owned, q_lo, q_hi, nq, nk, n_tau, dtype,
        gl_q, gg_q, glr_q, ggr_q, stlv, stgv, start, xp,
    ):
        """Coupled-q contraction with the TENSOR-DECOMPOSED vertex.

        Exact factored equivalent of ``_contract_dense_q`` (same fold terms,
        same left-vertex conjugation -- applied to the row factors inside the
        Grams). The quad sum collapses onto two summed Grams and the
        q'-convolution runs as an FFT; see ``quatrex.phonon.bubble_factored``.
        """
        from quatrex.phonon.bubble_factored import contract_tau_q_factored

        vf = self._vfactors
        if self._micro_pair_index is not None:
            quads_by_pair = {
                pair: [
                    (q.k1, q.k2, q.k1p, q.k2p) for q in quads
                ]
                for group_pair in owned
                for pair, quads in self._micro_pair_index.get(
                    group_pair, {}).items()
            }
            factor_block_sizes = np.full(
                self._micro_layout.n_microblocks,
                self._micro_layout.micro_dof, dtype=int)
        else:
            quads_by_pair = {
                (I, J): [
                    (K1, K2, K1p, K2p)
                    for (K1, K2, K1p, K2p, _pl, _pr)
                    in self._phi_pair_index[(I, J)]
                ]
                for (I, J) in owned
                if (I, J) in self._phi_pair_index
            }
            factor_block_sizes = self.block_sizes
        off_pos = vf.offset_index()
        # lambda folded into the external leg ONCE per vertex: the sandwich
        # is Dt @ H @ Dt^T with Dt = D diag(lambda) (D, lambda real).
        Dt = xp.asarray(vf.D * vf.lambdas[None, :])
        UB = xp.asarray(vf.UB)
        UC = UB if vf.UB is vf.UC else xp.asarray(vf.UC)
        g_dicts = {"l": gl_q, "g": gg_q, "lr": glr_q, "gr": ggr_q}
        # Whether the two contracted legs coincide is a property of the ARRAYS,
        # not of the ansatz label: a mislabelled file would otherwise silently
        # serve an a-role Gram for a b-role request.
        shared = UB is UC or bool(xp.array_equal(UB, UC))

        def _run(lo, hi):
            return contract_tau_q_factored(
                quads_by_pair, factor_block_sizes, tuple(nk), q_lo, q_hi, nq,
                g_dicts, Dt, UB, UC, off_pos, lo, hi, xp, shared, dtype,
            )

        def _write_grouped(result):
            if self._micro_pair_index is None:
                for (I, J), (out_l, out_g) in result.items():
                    bs_I = int(self.block_sizes[I])
                    bs_J = int(self.block_sizes[J])
                    shape = (n_tau,) + tuple(nk) + (bs_I, bs_J)
                    stlv.blocks[I - start, J - start] = out_l.reshape(shape)
                    stgv.blocks[I - start, J - start] = out_g.reshape(shape)
                return

            grouped = {}
            d = self._micro_layout.micro_dof
            for I, J in owned:
                grouped[(I, J)] = (
                    xp.zeros((n_tau, nq, int(self.block_sizes[I]),
                              int(self.block_sizes[J])), dtype=dtype),
                    xp.zeros((n_tau, nq, int(self.block_sizes[I]),
                              int(self.block_sizes[J])), dtype=dtype),
                )
            for (i, j), (out_l, out_g) in result.items():
                I, si = self._micro_layout.locate(i)
                J, sj = self._micro_layout.locate(j)
                grouped[(I, J)][0][..., si, sj] += out_l
                grouped[(I, J)][1][..., si, sj] += out_g
            for (I, J), (out_l, out_g) in grouped.items():
                shape = ((n_tau,) + tuple(nk) +
                         (int(self.block_sizes[I]), int(self.block_sizes[J])))
                stlv.blocks[I - start, J - start] = out_l.reshape(shape)
                stgv.blocks[I - start, J - start] = out_g.reshape(shape)

        # The Gram tables are (nq, n_tau, R, R), so the working set grows with the
        # tau slice AND with R^2 -- at R=128 the full local tau axis is tens of
        # GB. Bound it by splitting tau, whether or not there is a pool to
        # parallelise over (on the GPU there is none).
        pool, n_threads = ring_pool()
        bytes_per_table_tau = (
            nq * Dt.shape[1] ** 2 * xp.dtype(dtype).itemsize
        )
        # GramTables caches four variants for every distinct contracted link
        # while one output pair is evaluated.  The old budget counted only
        # ONE such table, so R=64 q9 already exceeded a 6 GB GPU although the
        # configured temporary budget was 256 MB.  Add the exact cache count
        # (roles share it for INDSCAL) and space for the live sums/FFT plans.
        table_multiplier = 1
        for quads in quads_by_pair.values():
            a_links = {(q[0], q[2]) for q in quads}
            b_links = {(q[1], q[3]) for q in quads}
            n_links = (len(a_links | b_links) if shared
                       else len(a_links) + len(b_links))
            table_multiplier = max(table_multiplier, 4 * n_links + 16)
        bytes_per_tau = bytes_per_table_tau * table_multiplier
        cap = max(1, self._tau_chunk_bytes // max(bytes_per_tau, 1))
        n_chunks = max(1, -(n_tau // -int(cap)))                 # ceil
        if pool is not None and xp is np:
            n_chunks = max(
                n_chunks, min(n_threads, max(1, n_tau // self._tau_min_chunk))
            )

        bnds = [(i * n_tau // n_chunks, (i + 1) * n_tau // n_chunks)
                for i in range(n_chunks)]
        bnds = [(lo, hi) for lo, hi in bnds if hi > lo]

        if len(bnds) == 1:
            _write_grouped(_run(*bnds[0]))
            return

        if pool is not None and xp is np and n_threads > 1:
            chunks = list(pool.map(lambda b: _run(*b), bnds))
        else:
            chunks = [_run(lo, hi) for lo, hi in bnds]
        joined = {
            pair: (
                xp.concatenate([c[pair][0] for c in chunks], axis=0),
                xp.concatenate([c[pair][1] for c in chunks], axis=0),
            )
            for pair in quads_by_pair
        }
        _write_grouped(joined)

    def _fold_reversed_legs(self, legs: tuple, g: DSDBSparse) -> None:
        """Apply the exact bosonic ji-transpose ``X_ij -> X_ji`` to the given
        stack-state buffers in place, at ANY ``comm.block`` size. Used on the
        reversed (absorption) G legs, and in the Sigma^>-reconstruction mode
        also on the cross-term accumulator (same pattern by construction).

        On the local nnz axis the partner ``(j,i)`` of an entry ``(i,j)`` is
        stored locally unless ``(i,j)`` straddles a block-rank boundary, in which
        case ``(j,i)`` lives on the immediate neighbour block rank (BT band =>
        ``|I-J| <= 1``). The pattern-only plan (a local gather permutation plus a
        neighbour request/reply schedule) is built once; per call only the
        boundary entries are exchanged.
        """
        local_perm, recv_dest, send_src = self._build_fold_plan(g)
        for leg in legs:
            data = leg.data
            folded = data[..., local_perm]
            for nbr in sorted(set(recv_dest) | set(send_src)):
                sbuf = xp.ascontiguousarray(data[..., send_src[nbr]]) \
                    if nbr in send_src else xp.empty((0,), dtype=data.dtype)
                rshape = data.shape[:-1] + (recv_dest[nbr].size,)
                rbuf = xp.empty(rshape, dtype=data.dtype) if nbr in recv_dest \
                    else xp.empty((0,), dtype=data.dtype)
                ranks.block.send_recv(sbuf, nbr, rbuf, nbr)
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
            # Skipping the fold would silently drop the ji-transpose of the
            # absorption legs and break the bubble's energy balance, so refuse.
            raise TypeError(
                f"The 3-phonon bosonic fold needs the (row, col) nnz pattern, "
                f"which {type(g).__name__} does not expose. Use DSDBCOO "
                "(config.compute.dsdbsparse_type)."
            )
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
            # Pattern-only key exchange: pickled python objects, one-time,
            # host-side -- stays on the raw mpi4py communicator.
            bcomm = ranks.block._mpi_comm
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

    def _herm_masks(self, g: DSDBSparse):
        """Boolean masks over the (full, single-rank) nnz axis selecting the
        strictly-lower and strictly-upper block-triangle entries -- the
        mirrored placeholders of the Hermitian pair-halving. Cached."""
        cached = getattr(self, "_herm_mask_cache", None)
        if cached is None:
            rows, cols = g.spy()
            r = np.asarray(rows.get() if hasattr(rows, "get") else rows)
            c = np.asarray(cols.get() if hasattr(cols, "get") else cols)
            rb = np.searchsorted(self.block_offsets, r, side="right") - 1
            cb = np.searchsorted(self.block_offsets, c, side="right") - 1
            cached = self._herm_mask_cache = (
                xp.asarray(rb > cb), xp.asarray(rb < cb))
        return cached

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
        if self._micro_pair_index is not None:
            for (I, J), pairs in self._micro_pair_index.items():
                if not (a <= min(I, J) < b):
                    continue
                for quads in pairs.values():
                    for quad in quads:
                        links.add(self._micro_layout.grouped_pair(
                            quad.k1, quad.k1p))
                        links.add(self._micro_layout.grouped_pair(
                            quad.k2, quad.k2p))
        else:
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
                ij for ij in (self._micro_pair_index
                              if self._micro_pair_index is not None
                              else self._phi_pair_index)
                if start <= min(ij) < end
            ]
        return cache[(start, end)]

    def _exchange_band_halo(
        self, gtlv, gtgv, ref: DSDBSparse, start: int, end: int,
    ) -> tuple[dict[tuple[int, int], NDArray], dict[tuple[int, int], NDArray]]:
        """Fetch the off-window band links (for both lesser and greater)
        from the immediate comm.block neighbours via a width-N_h halo.

        No MPI tags: both peers enumerate the identical sorted link set
        per direction, so messages between a fixed rank pair match in
        posting order, and the exchange runs through the backend-agnostic
        ``comm.block`` group isend/irecv.
        """
        rank, size = ranks.block.rank, ranks.block.size
        offs = ref.block_section_offsets
        # Leading axes of a block: (local tau,) in the Gamma-only case, and
        # (local tau, *nk) once the device is transversely periodic. Read off
        # the buffer rather than assumed, because hardcoding the Gamma shape
        # here is what made transverse q and comm.block.size > 1 mutually
        # exclusive: the halo would post buffers nq times too small.
        lead = tuple(int(n) for n in ref.data.shape[:-1])

        def is_local(link):
            return start <= min(link) < end

        def shape(link):
            K, Kp = link
            return lead + (int(self.block_sizes[K]),
                           int(self.block_sizes[Kp]))

        neighbours = [
            (nbr, int(offs[nbr]), int(offs[nbr + 1]))
            for nbr in (rank + 1, rank - 1)
            if 0 <= nbr < size
        ]

        # Guard: every non-local needed link must be in an immediate
        # neighbour. Checked BEFORE posting so the (rank-symmetric)
        # failure leaves no dangling group state.
        needed_nonlocal = {l for l in self._links_for_range(start, end)
                           if not is_local(l)}
        received = set()
        for _neigh, lo, hi in neighbours:
            received |= {l for l in self._links_for_range(start, end)
                         if lo <= min(l) < hi}
        missing = needed_nonlocal - received
        if missing:
            raise RuntimeError(
                "Phonon-phonon halo reaches beyond the immediate comm.block "
                f"neighbour (links {sorted(missing)}). Give each comm.block "
                "rank at least N_h blocks (reduce comm.block.size)."
            )

        halo_l: dict[tuple[int, int], NDArray] = {}
        halo_g: dict[tuple[int, int], NDArray] = {}
        reqs = []
        sendbufs = []

        backend = ranks.block._config["send_recv"]
        ranks.block.group_start(backend)
        for neigh, lo, hi in neighbours:
            # send: links the neighbour's outputs need that I own
            for link in sorted(l for l in self._links_for_range(lo, hi)
                               if is_local(l)):
                K, Kp = link
                bl = xp.ascontiguousarray(gtlv.blocks[K - start, Kp - start])
                bg = xp.ascontiguousarray(gtgv.blocks[K - start, Kp - start])
                sendbufs.extend((bl, bg))
                reqs.append(ranks.block.isend(bl, dest=neigh))
                reqs.append(ranks.block.isend(bg, dest=neigh))
            # recv: links my outputs need that the neighbour owns
            for link in sorted(l for l in self._links_for_range(start, end)
                               if lo <= min(l) < hi):
                bl = xp.empty(shape(link), dtype=complex)
                bg = xp.empty(shape(link), dtype=complex)
                halo_l[link], halo_g[link] = bl, bg
                reqs.append(ranks.block.irecv(bl, source=neigh))
                reqs.append(ranks.block.irecv(bg, source=neigh))
        ranks.block.group_end(backend, reqs)
        return halo_l, halo_g

    def _ensure_tau_buffers(
        self, g_lesser: DSDBSparse, n_fft: int, with_dg: bool = False
    ) -> tuple[DSDBSparse, ...]:
        """Lazily allocate the n_fft tau-buffers sharing G's pattern.

        Six buffers for the quadratic bubble (decay + sigma accumulators
        + reversed legs); ``with_dg`` extends the SAME cached set by four
        more (the direction's decay + reversed legs) for the linearized
        bubble -- allocated only when first requested, so runs that never
        call :meth:`compute_linearized` pay nothing.
        """
        cls = type(g_lesser)
        key = (cls, n_fft)
        n_needed = 10 if with_dg else 6
        if self._tau_cache is not None and self._tau_cache[0] == key:
            bufs = self._tau_cache[1]
            if len(bufs) >= n_needed:
                return bufs[:n_needed]

        rows, cols = g_lesser.spy()
        N = int(np.sum(self.block_sizes))
        pattern = sparse.coo_matrix(
            (
                xp.ones(len(rows), dtype=xp.complex128),
                (xp.asarray(rows), xp.asarray(cols)),
            ),
            shape=(N, N),
        ).tocsr()
        nk = tuple(int(k) for k in g_lesser.global_stack_shape[1:])
        existing = (
            self._tau_cache[1]
            if self._tau_cache is not None and self._tau_cache[0] == key
            else ()
        )
        q_split = getattr(g_lesser, "q_section_offsets", None) is not None
        bufs = existing + tuple(
            cls.from_sparray(
                pattern, self.block_sizes, global_stack_shape=(n_fft,) + nk,
                # Inherit the source's transverse sectioning -- but only on the
                # LEGS. Without it the tau buffers span the WHOLE mesh while the
                # G legs written into them carry only this rank's slice, and the
                # assignment fails on the shape -- asymmetrically, so one rank
                # raises while its peers block in the next collective and the
                # run looks like a deadlock rather than an error.
                #
                # Slots 2 and 3 are the Sigma ACCUMULATORS (stl, stg) and must
                # stay whole. The rotation is a convolution over q: one slice
                # pair contributes to EVERY external momentum, so a rank
                # legitimately accumulates across the whole mesh and only owns a
                # section of the RESULT. Sectioning them here is what produced
                #   IndexError: index 2 is out of bounds for axis 1 with size 1
                # The section is taken after the comm.q reduction instead --
                # ``_scatter_sigma_q`` below.
                q_distributed=q_split and slot not in _SIGMA_TAU_SLOTS,
            )
            for slot in range(len(existing), n_needed)
        )
        gt_rows, gt_cols = bufs[0].spy()
        if not (
            bool(xp.array_equal(xp.asarray(rows), xp.asarray(gt_rows)))
            and bool(xp.array_equal(xp.asarray(cols), xp.asarray(gt_cols)))
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

    def set_pole_channel(self, g_pp_lesser, g_pp_greater) -> None:
        """Inject the pole-sector Keldysh legs for this SCBA iteration.

        ``g_pp_{lesser,greater}`` are ``G_PP^{<,>}`` already projected onto the
        stored sparsity pattern, on the PRIMARY grid and in the solver's
        occupation-positive convention. They are subtracted from the bubble legs
        before any interpolation or masking; the matching analytic contribution
        is supplied separately through :meth:`set_pole_self_energy`.

        Cleared after every :meth:`compute` so a stale channel can never be
        consumed against a self-energy it was not built from.
        """
        self._pole_channel = (g_pp_lesser, g_pp_greater)

    def set_pole_mixed(self, sigma_l, sigma_g) -> None:
        """Inject the mixed pole-background sectors ``Sigma_SR + Sigma_RS``.

        These go in BEFORE the Kramers-Kronig transform, unlike the pole-pole
        channel. They carry only ONE narrow factor against a smooth background,
        so they are ordinary grid objects and the existing numerical Hilbert
        transform is the right tool for their retarded partner -- no separate
        analytic reconstruction is needed, and using one would risk
        double-counting against the transform that already sees them.
        """
        self._pole_sigma_mixed = (sigma_l, sigma_g)

    def set_bubble_correction(self, corr_l, corr_g) -> None:
        """Inject the subcell covariance correction for this iteration.

        This is what the cell-averaged ring LEAVES BEHIND, not a replacement
        for anything it computed: the correction is added to the raw bubble
        output, nothing is subtracted, and with no active cell it is exactly
        zero. That is why it needs no matching channel on the legs -- unlike
        the pole-pole route, where the leg removed and the sector restored have
        to be the same function.

        Placed with the mixed sectors, BEFORE ``delta`` is formed, so the
        existing Kramers-Kronig transform supplies its retarded partner. It
        carries narrow structure at combination frequencies and has no
        closed-form causal continuation of its own, so routing it after the
        transform would leave ``Sigma^R`` missing exactly the dispersive part
        ``Sigma^{<,>}`` had gained.
        """
        self._bubble_correction = (corr_l, corr_g)

    def set_pole_self_energy(self, sigma_l, sigma_g, sigma_r) -> None:
        """Inject the analytic pole-sector self-energy for this iteration."""
        self._pole_sigma_ss = (sigma_l, sigma_g, sigma_r)

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
            df = float(freqs[1] - freqs[0])
            if abs(float(freqs[0])) > 1e-9 * abs(df) or bool(
                xp.max(xp.abs(xp.diff(freqs) - df)) > 1e-9 * abs(df)
            ):
                raise ValueError(
                    "The bubble FFT requires a uniform frequency grid "
                    f"starting at 0 (got start {float(freqs[0]):g}, "
                    f"spacing {df:g}). Set energy_window_min = 0."
                )
        self._full_freqs = freqs
        return freqs
