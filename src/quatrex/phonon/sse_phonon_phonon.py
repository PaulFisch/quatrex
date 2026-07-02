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

from scipy import sparse

from quatrex.core.config import QuatrexConfig
from quatrex.core.fft_utils import hilbert_transform
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.phonon.bubble import (
    _RING_POOL,
    _RING_THREADS,
    _ring_contract_serial,
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
        self._sse_cutoff = float(
            getattr(config.phonon, "sse_low_freq_cutoff_thz", 0.0))
        # Band-limited SSE: scatter only where there are phonon states (mask the
        # bubble where the spectral function A(w)=i(G^>-G^<) is negligible). The
        # generic, automatic cutoff that fixes the eta->0 divergence (spurious
        # self-energy in the empty above-band grid region amplifying with no
        # damping) without a hand-set frequency.
        self._band_limit = bool(getattr(config.phonon, "band_limit_sse", False))
        self._band_tol = float(getattr(config.phonon, "spectral_support_tol", 1e-4))
        self._sharp_cap = float(getattr(config.phonon, "spectral_sharp_cap", 0.0))
        self._sharp_state = None  # persistent hysteresis mask of near-singular bins
        self._band_support_mask = None  # FROZEN empty-region mask (A-threshold fallback)
        # Fixed harmonic band top (THz) for the above-band mask. The eta->0
        # divergence lives in the empty grid region ABOVE the phonon band; the
        # band top is a G-independent property of the dynamical matrix, so the
        # masked set never moves (flicker-free). This supersedes the A(w)-
        # threshold support, which is unreliable for multi-cell devices: the
        # full n_cells x n_cells block nnz pattern pads the physically-zero
        # corner blocks with 1.0, flooring max_nnz|G^>-G^<| and hiding the
        # above-band emptiness (cnt33 L3 masked 0 bins -> eta=0 diverged).
        self._band_top = None
        self._band_top_announced = False
        if self._band_limit and dynamical_matrix is not None:
            bt_local = 0.0
            try:
                bt_local = self._compute_band_top(dynamical_matrix)
            except Exception as exc:  # noqa: BLE001 - fall back, never fatal
                if ranks.rank == 0:
                    print(f"SSE band-limit: band-top computation failed "
                          f"({exc!r}); falling back to A-threshold support",
                          flush=True)
            # Agree across ranks: the dynamical matrix may be block-distributed,
            # so only the rank(s) owning blocks (0,0)/(0,1) get a real value.
            bt = np.array(float(bt_local), dtype=float)
            comm.Allreduce(MPI.IN_PLACE, bt, op=MPI.MAX)
            if float(bt) > 0.0:
                self._band_top = float(bt)
                if comm.rank == 0:
                    print(f"SSE band-limit: harmonic band top "
                          f"{self._band_top:.3f} THz (fixed above-band mask)",
                          flush=True)
        # HARMONIC spectral-support masking (band_support_margin_thz > 0): mask
        # SSE bins far from EVERY harmonic band frequency -- above-band AND
        # interior gaps AND between sparse high-freq modes (the single band-top
        # mask misses the latter two). G-independent + frozen. self._support_freqs
        # is local (only the rank(s) owning blocks (0,0)/(0,1) get a real value);
        # the per-bin empty mask is OR-reduced across ranks at first use.
        self._support_margin = float(
            getattr(config.phonon, "band_support_margin_thz", 0.0))
        self._support_freqs = None
        self._support_empty = None
        if (self._band_limit and self._support_margin > 0.0
                and dynamical_matrix is not None):
            try:
                self._support_freqs = self._compute_band_support_freqs(
                    dynamical_matrix)
            except Exception as exc:  # noqa: BLE001 - never fatal
                if ranks.rank == 0:
                    print(f"SSE band-limit: band-support computation failed "
                          f"({exc!r}); harmonic-support mask off", flush=True)
        # Occupation-freeze mask (sse_freeze_occupation > 0): mask SSE bins whose
        # Bose occupation n(omega, T_hot) is below the threshold -- thermally
        # frozen spectator modes that carry ~zero heat (e.g. the isolated Si-H
        # stretch island). T_hot = warmer lead, so a mode is frozen only if
        # frozen at the hot contact too. x = hbar*omega/(kB*T_hot).
        self._freeze_n = float(getattr(config.phonon, "sse_freeze_occupation", 0.0))
        self._T_hot = max(float(getattr(config.phonon, "left_temperature", 300.0)),
                          float(getattr(config.phonon, "right_temperature", 300.0)))
        self._freeze_mask = None
        # SMOOTH band-limit window (sse_smooth_window): replace the hard masks
        # with a smooth multiplicative window -> no Sigma discontinuity -> no
        # Hilbert/Gibbs edge ring -> no manufactured eta=0 marginal mode.
        self._smooth_window = bool(getattr(config.phonon, "sse_smooth_window",
                                           False))
        self._support_taper_cells = float(
            getattr(config.phonon, "support_taper_cells", 4.0))
        # NB: the IR Bose-singularity subtraction (sse_ir_subtraction) is handled
        # in phonon/solver.py at the LEAD OCCUPATION level, not in the bubble --
        # the device G^< has no 1/omega pole to subtract (the bosonic fold +
        # bounded spectral A force it to cancel), so a bubble-leg subtraction
        # both does nothing physical AND breaks the Phi-derivable conservation.
        self._sse_window = None  # cached (ne_full, 1, ...) float window
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
            #     compute (the vertex is fixed) and run the dense path at
            #     full speed. The factored win here is MEMORY + build time
            #     (~MBs vs the GB-scale replicated qfold dict), not flops.
            #   "gram": the skinny-Gram contraction (bubble_factored) --
            #     fewer flops than dense only while R^2 < ~3*b^4, i.e. small
            #     R or LARGE blocks; on small-block films it is memory-bound
            #     in R^2 and loses to dense beyond R ~ 16 (2026-07-02
            #     micro-benchmark, phonon/studies/_bench_factored_sse.py).
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

    @staticmethod
    def _compute_band_top(dynamical_matrix, n_k: int = 512) -> float:
        """Harmonic phonon band top (THz) from the dynamical matrix.

        The device matrix is block-tridiagonal in the (nearest-neighbour) unit
        cells, so the bulk dispersion is ``D(k) = D0 + D1 e^{ik} + D1^H
        e^{-ik}`` with the on-site block ``D0 = blocks[0,0]`` and the coupling
        ``D1 = blocks[0,1]``; its eigenvalues are ``omega^2(k)``. The band top
        ``max_k sqrt(eig D(k))`` is the highest frequency with phonon states --
        above it the grid is empty (the eta->0 divergence region). This is a
        fixed property of D (no Green's function), so it is robust to the
        constant-1.0 corner-block padding that corrupts the A(w) support.
        """
        D0 = np.asarray(dynamical_matrix.blocks[0, 0])
        if D0.ndim == 3:
            D0 = D0[0]
        D1 = np.asarray(dynamical_matrix.blocks[0, 1])
        if D1.ndim == 3:
            D1 = D1[0]
        D0 = 0.5 * (D0 + D0.conj().T)
        ks = np.linspace(0.0, np.pi, int(n_k))
        w_top = 0.0
        for k in ks:
            phase = np.exp(1j * k)
            Dk = D0 + D1 * phase + D1.conj().T * np.conj(phase)
            Dk = 0.5 * (Dk + Dk.conj().T)
            ev_max = float(np.linalg.eigvalsh(Dk)[-1])
            if ev_max > w_top:
                w_top = ev_max
        return float(np.sqrt(max(w_top, 0.0)))

    @staticmethod
    def _compute_band_support_freqs(dynamical_matrix, n_k: int = 512):
        """ALL harmonic mode frequencies (THz) of the bulk dispersion
        ``D(k)=D0+D1 e^{ik}+D1^H e^{-ik}``, k in [0, pi] -- the union over k and
        bands is the phonon spectral SUPPORT. Grid bins far from every band
        frequency are EMPTY (no states); masking them out of the SSE removes the
        eta->0 FFT-leakage divergence in the above-band AND the interior gaps,
        including for SPARSE high-frequency modes (e.g. discrete Si-H surface
        modes) that the single band-top cutoff misses. Like
        ``_compute_band_top`` but keeps the full spectrum; G-independent (robust
        to corner-block padding). Returns sorted unique frequencies (THz)."""
        D0 = np.asarray(dynamical_matrix.blocks[0, 0])
        if D0.ndim == 3:
            D0 = D0[0]
        D1 = np.asarray(dynamical_matrix.blocks[0, 1])
        if D1.ndim == 3:
            D1 = D1[0]
        D0 = 0.5 * (D0 + D0.conj().T)
        out = []
        for k in np.linspace(0.0, np.pi, int(n_k)):
            phase = np.exp(1j * k)
            Dk = D0 + D1 * phase + D1.conj().T * np.conj(phase)
            Dk = 0.5 * (Dk + Dk.conj().T)
            ev = np.linalg.eigvalsh(Dk)
            out.append(np.sqrt(np.clip(ev.real, 0.0, None)))
        return np.unique(np.concatenate(out))

    def _build_sse_window(self, full_freqs):
        """SMOOTH band-limit window w(omega) in [0,1] (1D, cached). Replaces the
        hard support+freeze masks: w_supp (raised-cosine ramp of width
        support_taper_cells*dw on the GLOBAL distance to the harmonic support --
        subsumes band-top + gaps) times w_occ (smooth logistic in the Bose
        occupation at T_hot -- subsumes the freeze). G-independent + frozen, so
        no live-A limit cycle; C^1 smooth, so its Hilbert partner has no edge
        ring; ramps ~dw, so -> the exact band indicator as dw->0."""
        af = np.abs(np.asarray(full_freqs).real).astype(float)
        dw = float(abs(full_freqs[1] - full_freqs[0])) if af.size > 1 else 1.0
        w = np.ones(af.shape, dtype=float)
        if self._support_margin > 0.0:
            sf = self._support_freqs
            if sf is not None and sf.size:
                dist = np.min(np.abs(af[:, None] - sf[None, :]), axis=1)
            else:
                dist = np.full(af.shape, np.inf)
            # global nearest-harmonic-mode distance (support is the union of ranks)
            comm.Allreduce(MPI.IN_PLACE, dist, op=MPI.MIN)
            ramp = max(self._support_taper_cells * dw, 1e-30)
            s = np.clip((dist - self._support_margin) / ramp, 0.0, 1.0)
            w *= 0.5 * (1.0 + np.cos(np.pi * s))  # 1 at dist<=m, 0 at dist>=m+ramp
        if self._freeze_n > 0.0:
            x = _THZ_OVER_K * af / max(self._T_hot, 1e-30)
            with np.errstate(over="ignore", divide="ignore"):
                n_bose = 1.0 / np.expm1(np.clip(x, 1e-30, None))
            w *= 1.0 / (1.0 + (self._freeze_n
                               / np.maximum(n_bose, 1e-300)) ** 3)
        # The omega=0 bin is always zeroed (a nonzero Sigma^≷(0) hits the singular
        # acoustic G^R(0) -> DC spike). The IR occupancy taper (ir_taper_cells)
        # makes the approach ~omega^2 smooth, so this single-bin zero does not ring.
        w[af < 1e-6] = 0.0
        if comm.rank == 0:
            print(f"SSE band-limit: SMOOTH window (support ramp "
                  f"{self._support_taper_cells:g}*dw, freeze n<{self._freeze_n:g})"
                  f" -> mean w={float(w.mean()):.3f}, {int((w < 0.5).sum())}/"
                  f"{w.size} bins below 0.5", flush=True)
        return xp.asarray(w)

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
        # Dense device dynamical matrix D (THz²), omega-independent.
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
                f"[SigmaPhononPhonon] SCP tadpole on: N_D={N_D}, "
                f"mixing={self._scp_mix}, Phi_eff floor="
                f"{getattr(config.phonon, 'scp_floor_thz', 0.5)} THz.",
                flush=True,
            )

    def _apply_scp_tadpole(self, g_lesser, sigma_retarded) -> None:
        """Self-consistent cubic tadpole, in the nnz state.

        Forms <uu> from the omega-integral of the device G^< (full omega-
        local per nnz slice), solves the regularised mean_displacement
        against Phi_eff = D + Sigma_static, mixes the resulting static
        Sigma_T, and broadcasts it into Sigma^R at every frequency.
        """
        from quatrex.phonon.static_self_energy import (
            equal_time_uu_from_sum, mean_displacement, sigma_tadpole)

        N_D = self._sigma_static.shape[0]
        # omega-integral of G^< summed over every stack axis.
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
        # Broadcast the static self-energy into Sigma^R at every frequency.
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
        """Compute the 3-phonon self-energy contribution

        1. G(omega)->g(tau) by FFT in nnz distribution
        2. g(tau) nnz->stack so each comm.stack rank owns a
           tau-slice of the full band, block-accessible
        3. the off-diagonal ring contraction per tau-slice in stack
        4. sgima(tau) stack->nnz
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
        # Low-frequency masking of the bubble INPUT: the Green's functions
        # stay intact for Dyson/observables; only the copies fed into the
        # 3-phonon convolution are masked. The omega=0 bin is always
        # excluded (Bose divergence at DC); below
        # config.phonon.sse_low_freq_cutoff_thz (0 = off) the modes do not
        # participate in the SSE at all -- with the matching OUTPUT mask in
        # step (5), transport below the cutoff is purely ballistic.
        sse_mask = xp.abs(xp.asarray(full_freqs)) < max(self._sse_cutoff, 1e-6)
        if self._band_limit:
            af = xp.abs(xp.asarray(full_freqs))
            low = xp.zeros(af.shape, dtype=bool)
            if self._band_top is not None:
                # PRIMARY: fixed harmonic band-top mask. The eta->0 divergence
                # lives in the empty grid region ABOVE the phonon band (no states
                # there -> the resolvent is real -> the SCBA amplifies
                # convolution leakage with no damping). omega_top is a fixed,
                # G-independent property of the dynamical matrix, so the masked
                # set never moves: a LIVE A(w) threshold instead makes the SCBA
                # map discontinuous and it limit-cycles (Anderson on cnt33 L2
                # converges to 9e-5 with no mask but cycles 0.09<->0.38 with a
                # live mask). It also dodges the constant-1.0 corner-block
                # padding of the full n_cells x n_cells nnz pattern, which floors
                # max_nnz|G^>-G^<| at 1.0 on multi-cell devices and hides the
                # above-band emptiness (cnt33 L3 masked 0 bins -> eta=0 diverged).
                margin = (float(abs(full_freqs[1] - full_freqs[0]))
                          if full_freqs.size > 1 else 0.0)
                low = af > (self._band_top + margin)
                if comm.rank == 0 and not self._band_top_announced:
                    self._band_top_announced = True
                    print(f"SSE band-limit: masking {int(low.sum())} bin(s) "
                          f"above {self._band_top:.3f} THz", flush=True)
            else:
                # FALLBACK (no dynamical matrix passed): freeze the empty-region
                # A(w)-threshold mask at the first (ballistic) call so the masked
                # set does not flicker as the modes shift under Sigma. Unreliable
                # on multi-cell devices (corner-block padding) -- prefer the band
                # top above whenever the dynamical matrix is available.
                a_w = xp.abs(gg_in - gl_in)
                spec = a_w.reshape(a_w.shape[0], -1).max(axis=1)
                spec_peak = float(spec.max())
                if spec_peak > 0.0:
                    if (self._band_support_mask is None
                            or self._band_support_mask.shape != spec.shape):
                        self._band_support_mask = spec < self._band_tol * spec_peak
                        if ranks.rank == 0:
                            print(f"SSE: froze band-support mask "
                                  f"({int(self._band_support_mask.sum())}/"
                                  f"{spec.size} empty bins, "
                                  f"tol={self._band_tol:g})", flush=True)
                    low = self._band_support_mask
            if self._support_margin > 0.0:
                # HARMONIC spectral-support empty-bin mask (above-band + interior
                # gaps + between sparse high-freq modes). Computed once, frozen.
                if self._support_empty is None:
                    af_h = np.abs(np.asarray(full_freqs).real).astype(float)
                    sf = self._support_freqs
                    if sf is not None and sf.size:
                        dist = np.min(np.abs(af_h[:, None] - sf[None, :]), axis=1)
                        loc_sup = (dist <= self._support_margin).astype(np.int32)
                    else:
                        loc_sup = np.zeros(af_h.shape, dtype=np.int32)
                    # OR across ranks: the block-owning rank holds the support.
                    comm.Allreduce(MPI.IN_PLACE, loc_sup, op=MPI.MAX)
                    self._support_empty = xp.asarray(loc_sup == 0)
                    if comm.rank == 0:
                        print(f"SSE band-limit: harmonic-support mask -> "
                              f"{int((loc_sup == 0).sum())}/{loc_sup.size} empty "
                              f"bins masked (margin {self._support_margin:g} THz)",
                              flush=True)
                low = low | self._support_empty
            if self._freeze_n > 0.0:
                # Occupation-freeze: mask thermally-frozen spectator bins
                # (n(omega, T_hot) < threshold). Frozen once (T-fixed).
                if self._freeze_mask is None:
                    af_h = np.abs(np.asarray(full_freqs).real).astype(float)
                    # hbar*omega/(kB*T): omega in THz -> 47.99 * f / T
                    x = _THZ_OVER_K * af_h / max(self._T_hot, 1e-30)
                    with np.errstate(over="ignore", divide="ignore"):
                        n_bose = 1.0 / np.expm1(np.clip(x, 1e-30, None))
                    frozen = (n_bose < self._freeze_n) & (af_h > 0.0)
                    self._freeze_mask = xp.asarray(frozen)
                    if comm.rank == 0:
                        print(f"SSE band-limit: occupation-freeze mask -> "
                              f"{int(frozen.sum())}/{frozen.size} frozen bins "
                              f"(n<{self._freeze_n:g}, T_hot={self._T_hot:g}K)",
                              flush=True)
                low = low | self._freeze_mask
            sse_mask = sse_mask | low
            if self._sharp_cap > 0.0:
                # Near-singular (sharp) modes: A spikes far above the in-band
                # median (sub-grid linewidth). Zero them for THIS iteration with
                # HYSTERESIS (flag at cap x median, un-flag below cap/4) so the
                # mask itself does not flicker; they re-enter once Sigma broadens
                # them. (The in-band median is corrupted by corner-block padding
                # on multi-cell devices -- this knob is for single-cell soft
                # modes; default spectral_sharp_cap=0 keeps it off.)
                a_w = xp.abs(gg_in - gl_in)
                spec = a_w.reshape(a_w.shape[0], -1).max(axis=1)
                inband = spec[(~low) & (spec > 0.0)]
                if inband.size:
                    med = float(xp.median(inband))
                    if med > 0.0:
                        newly = spec > self._sharp_cap * med
                        healed = spec < 0.25 * self._sharp_cap * med
                        if self._sharp_state is None or \
                                self._sharp_state.shape != spec.shape:
                            self._sharp_state = xp.zeros_like(spec, dtype=bool)
                        self._sharp_state = (self._sharp_state & ~healed) | newly
                        if ranks.rank == 0 and bool(self._sharp_state.any()):
                            print(f"SSE: masked {int(self._sharp_state.sum())} "
                                  f"near-singular bin(s) (A > {self._sharp_cap:g}"
                                  f"x median, hysteresis)", flush=True)
                        sse_mask = sse_mask | self._sharp_state
        if self._smooth_window:
            # SMOOTH window (replaces the hard masks). The reversed/absorption
            # legs are built from gl_in/gg_in below, so they inherit it -> the
            # bosonic fold stays exact (w is a function of |omega|).
            if self._sse_window is None:
                self._sse_window = self._build_sse_window(full_freqs)
            _w = self._sse_window.reshape((-1,) + (1,) * (gl_in.ndim - 1))
            gl_in = gl_in * _w
            gg_in = gg_in * _w
        elif bool(sse_mask.any()):
            gl_in = gl_in.copy(); gl_in[sse_mask] = 0.0
            gg_in = gg_in.copy(); gg_in[sse_mask] = 0.0
        # NOTE: the IR Bose subtraction is NOT applied to the bubble legs -- the
        # device G^< has no 1/omega pole (the bosonic fold + bounded spectral A
        # force it to cancel; data confirm |G^<|~omega^-0.5 bounded). The pole is
        # real only in the lead OCCUPATION; the sse_ir_subtraction treatment
        # lives there (phonon/solver.py: use the full physical occupation instead
        # of the omega^2 taper). The bubble stays the bare (conserving) convolution.
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
        # DFT index-reversal rev(X)[l]=X[(-l) mod n_fft] of the FFT'd G, for
        # the absorption (negative-omega') terms. The exact bosonic
        # continuation carries the ji-TRANSPOSE (and -q for coupled-q):
        #     G^<_ij(q, -w) = G^>_ji(-q, w).
        # The no-transpose shortcut is exact only for the EQUILIBRIUM
        # (complex-symmetric) part of G; the current-carrying asymmetry of
        # the nonequilibrium G^< (~2% on cnt33 at dT=10 K) was previously
        # folded without it, breaking the Phi-derivable energy balance of
        # the bubble at ~1e-5 (see SCBA._phonon_bubble_energy_balance).
        # Build the reversed (absorption) legs. The q-negation (middle axes) and
        # the FFT + tau reversal (axis 0) are state-independent and done here in
        # the nnz state; the ji-TRANSPOSE (nnz/last axis) is applied AFTER the
        # nnz->stack dtranspose below, where the FULL nnz pattern is local on
        # every rank. The transpose (axis -1) commutes with the FFT/reversal
        # (axis 0) and q-negation (middle axes), so this is EXACTLY the serial
        # fold G^<_ij(q,-w)=G^>_ji(-q,w) -- now at ANY nranks (block_comm=1),
        # not just nranks=1. The old code applied the transpose in the nnz state
        # (nnz split over comm.stack -> partner non-local) and silently fell back
        # to the no-transpose equilibrium continuation, breaking the
        # Phi-derivable energy balance (~1e-5) for every multi-rank run.
        gl_rev_src, gg_rev_src = gl_in, gg_in
        if nq > 1:
            # negate the transverse momentum axes (Gamma-centered IDFT
            # meshes are closed under q -> -q)
            for ax, k in enumerate(nk, start=1):
                neg = (-xp.arange(k)) % k
                gl_rev_src = xp.take(gl_rev_src, neg, axis=ax)
                gg_rev_src = xp.take(gg_rev_src, neg, axis=ax)
        Xl = self._fft_pad(gl_rev_src, n_fft)
        Xg = self._fft_pad(gg_rev_src, n_fft)
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
        # into the one-sided grid via G^<(-ω)=G^>(ω): for each ring quad
        #   Sigma^<  = ring(g^<_a, g^<_b) + ring(g^<_a, rev g^>_b) + ring(rev g^>_a, g^<_b)
        #   Sigma^>  = ring(g^>_a, g^>_b) + ring(g^>_a, rev g^<_b) + ring(rev g^<_a, g^>_b)
        # Serial ring contraction (single-thread BLAS); the omega/tau batch is
        # parallelised ONCE over the whole (I,J)/phi-pair loop below, not per
        # call, so the worker threads stay busy across the whole contraction
        # instead of idling on the GIL-held Python between 100s of short calls.
        # Pre-permute the (fixed) phi factors once: per (I,J) a list of
        # (K1,K2,K1p,K2p, PL,PR,nI,bK2,nJ). Removes the per-call transpose copy
        # that dominates the bubble on small-block systems.
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
            # Cap the tau-split so each worker keeps >=~4 tau points. The per-ring
            # contraction is cache-bound per chunk (the BS^3 ring intermediate
            # stays in cache only while the chunk is small), so it scales near-
            # linearly with threads UNTIL the chunk gets too short -- measured on
            # cnt33 L2 (w=241): 64 threads = 56x (1232 GF/s), 128 threads regress
            # to 791 GF/s purely from <2 tau/thread. n_tau//4 keeps us at the
            # sweet spot and lets large-w cells use proportionally more threads.
            nt = min(_RING_THREADS, max(1, n_tau // 4))
            if _RING_POOL is not None and nt > 1:
                bnds = [(i * n_tau // nt, (i + 1) * n_tau // nt) for i in range(nt)]
                chunks = list(_RING_POOL.map(lambda b: _contract_tau(*b), bnds))
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
                # (tau, *nk, b, b) → (tau, N_q, b, b); the q-axis is contiguous
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
                ranks.q.all_reduce(np.ascontiguousarray(m.data), recv, op="sum")
                m.data[:] = recv

        # (4) sigma(tau) stack->nnz
        stl.dtranspose()
        stg.dtranspose()

        # (5) IFFT sigma(τ)→Sigma(omega) in nnz; add into outputs; build Sigma^R.
        sl_data = prefactor * xp.fft.ifft(stl.data, axis=0)[:ne_full]
        sg_data = prefactor * xp.fft.ifft(stg.data, axis=0)[:ne_full]
        # OUTPUT mask, completing the input masking above: the scattering
        # Sigma is NOT applied below the SSE cutoff (transport there stays
        # ballistic), and never at the omega=0 bin (a nonzero Sigma^≷(0)
        # hits the near-singular acoustic G^R(0) and produces a x1e5 DC
        # spike in I(0) on soft wires; the bin carries zero heat anyway).
        if self._smooth_window:
            _w = self._sse_window.reshape((-1,) + (1,) * (sl_data.ndim - 1))
            sl_data = sl_data * _w
            sg_data = sg_data * _w
        elif bool(sse_mask.any()):
            sl_data[sse_mask] = 0.0
            sg_data[sse_mask] = 0.0
        # SIGN CONVENTION (2026-06-12 fix): the bubble formula (textbook
        # G^< = -i n A) is quadratic in G, so fed with this solver's
        # occupation-positive Green's functions (-iG^≷ >= 0, the same
        # convention as the lead injection sigma^≷ = +i n(+1) gamma) it
        # returns TEXTBOOK-signed sigma^≷ (-i sigma^≷ <= 0) -- the exact
        # NEGATIVE of what the Keldysh feedback G^≷ = G^R sigma^≷ G^A
        # expects. Feeding it unflipped injects negative occupation
        # (anti-dissipation): the SCBA then diverges at any coupling
        # (2026-06-12: even lambda=0.25 diverged while the equilibrium
        # textbook-convention loop converged). Negate sigma^≷ here; the
        # retarded part below keeps its (damping-correct) sign by using
        # the raw values.
        sigma_lesser.data[:] = sigma_lesser.data - sl_data
        sigma_greater.data[:] = sigma_greater.data - sg_data
        # Sigma^R contribution (from the RAW, textbook-signed values:
        # Gamma = i(sigma^> - sigma^<)_raw >= 0, matching the lead OBC
        # damping sign -- unchanged by the convention flip above).
        if self.retarded_method == "fft":
            delta = sg_data - sl_data
            hil = 0.5j * hilbert_transform(delta, full_freqs)
            if not self._smooth_window and bool(sse_mask.any()):
                # HARD path: post-mask the retarded too. (SMOOTH path: sl/sg are
                # already windowed -> delta is C^1 smooth -> the Hilbert has NO
                # edge ring; post-masking here would RE-INTRODUCE the step after
                # the global KK transform -- the Gibbs source -- so we skip it.)
                hil[sse_mask] = 0.0
            sigma_retarded.data[:] = sigma_retarded.data + hil

        # Self-consistent SCP cubic-tadpole static self-energy
        if self._scp_tadpole and self._sigma_static is not None:
            self._apply_scp_tadpole(g_lesser, sigma_retarded)

    def _contract_dense_q(
        self, owned, qdm, qv, q_lo, q_hi, nq, nk, n_tau, dtype,
        gl_q, gg_q, glr_q, ggr_q, _fold_l, _fold_g,
        stlv, stgv, start, xp,
    ):
        """DENSE coupled-q vertex-pair contraction (the reference path).

        Extracted verbatim from ``_compute_fft_first``; consumes the dense
        q-folded vertex dict ``qv`` and the q-flattened tau-domain Green's
        function band dicts, writes the per-(I, J) Sigma^{<,>} tau blocks
        into the stack views. See ``_contract_factored_q`` for the
        tensor-decomposed equivalent.
        """
        # Resolve the per-(I, J) vertex-pair task list once. The LEFT
        # vertex is CONJUGATED: the bubble at external q pairs
        # Phi(q', q_ext-q')^* with Phi(q_ext-q', q'); the unconjugated
        # pairing breaks momentum bookkeeping (Sigma(-q) != Sigma(q)^T
        # under time reversal) and disagrees with a real-space supercell
        # ground truth, see phonon/scripts/verify/audit_qfold_trs.py.
        # At Gamma the vertices are real, so the Gamma-only (nq==1)
        # path is unaffected.
        qtasks: dict[tuple[int, int], list] = {}
        for iq_ext in range(q_lo, q_hi):
            for iqp in range(nq):
                iq2 = int(qdm[iq_ext, iqp])
                phiL = qv.get((iqp, iq2))   # legs (q', q_ext−q')
                phiR = qv.get((iq2, iqp))   # legs (q_ext−q', q')
                if phiL is None or phiR is None:
                    continue
                for (I, J) in owned:
                    for K1, K2, K1p, K2p, _pl, _pr in self._phi_pair_index[(I, J)]:
                        pl = phiL.get((I, K1, K2))
                        pr = phiR.get((J, K2p, K1p))
                        if pl is None or pr is None:
                            continue
                        qtasks.setdefault((I, J), []).append(
                            (iq_ext, iqp, iq2, K1, K1p, K2, K2p)
                            + phi_perms(xp.conj(pl), pr, xp))

        # Sigma^{<,>}(I, J, q_ext) for the tau slice [lo:hi]; mirrors the
        # nq==1 _contract_tau so the omega/tau batch parallelises across
        # the ring pool (the coupled-q path previously ran serial).
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

        nt = min(_RING_THREADS, max(1, n_tau // 4))  # >=~4 tau/thread (see nq==1)
        if _RING_POOL is not None and xp is np and nt > 1:
            bnds = [(i * n_tau // nt, (i + 1) * n_tau // nt)
                    for i in range(nt)]
            chunks = list(_RING_POOL.map(lambda b: _contract_tau_q(*b), bnds))
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

        nt = min(_RING_THREADS, max(1, n_tau // 4))  # >=~4 tau/thread (see nq==1)
        if _RING_POOL is not None and xp is np and nt > 1:
            bnds = [(i * n_tau // nt, (i + 1) * n_tau // nt) for i in range(nt)]
            chunks = list(_RING_POOL.map(lambda b: _run(*b), bnds))
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
                "τ-buffer sparsity does not match G; cannot FFT raw data."
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
        self._full_freqs = freqs
        return freqs
