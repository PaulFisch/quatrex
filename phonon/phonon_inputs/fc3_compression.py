"""FC3 tensor compression via tensor decompositions.

Implements the five canonical ansatze from the FC3 compression section of the
thesis, plus PCP (Luo et al. 2025):

    mSVD     — truncated matricization SVD (Tucker1, mode-3)
    HOSVD    — truncated higher-order SVD (Tucker) with S2 symmetry on legs 2,3
               and a few steps of symmetric HOOI
    CP       — unconstrained canonical polyadic decomposition
    INDSCAL  — CP with S2 symmetry on legs 2,3 (b_r = c_r)
    Waring   — symmetric CP on the S3-symmetrised supercell-lifted tensor
               (A=B=C, equivalent to sum of cubes of linear forms)
    PCP      — permanent CP of Luo et al. 2025 (wraps phonon_inputs.pcp)

All decompositions operate on the same mass-weighted real-space target
    T[mu, j, k]   shape (n_dof, dim_sc, dim_sc)
with
    mu = 3*i_prim + alpha    (primitive DOF; alpha = 0,1,2)
    j  = 3*s + beta          (supercell DOF; beta = 0,1,2)
    k  = 3*s + gamma         (supercell DOF; gamma = 0,1,2)
in THz^{5/2} units.  For the S3-symmetric decompositions the tensor is lifted
to shape (dim_sc, dim_sc, dim_sc) via the primitive -> supercell map p2s_map
and symmetrised over S3 before fitting; reconstructions are sliced back to
(n_dof, dim_sc, dim_sc) for error comparison.

Every fitter returns a ``CompressionResult`` dataclass with a uniform API:
rank, parameter count, relative Frobenius error, and a ``factors`` dict whose
contents depend on the ansatz.  ``reconstruct(result)`` always yields a
dense (n_dof, dim_sc, dim_sc) array in THz^{5/2} units.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # optional; only needed for iterative refinement

try:
    import tensorly as tl
    from tensorly.decomposition import parafac as _tl_parafac
    from tensorly.decomposition import tucker as _tl_tucker
    from tensorly.decomposition import symmetric_parafac_power_iteration as _tl_sym_cp
    _HAVE_TENSORLY = True
except ImportError:
    _HAVE_TENSORLY = False

from .constants import CONVERSION_FC3_THZ
from .pcp import (
    _build_target as _pcp_build_target,
    fit_pcp,
    fit_pcp_greedy,
    build_cell_mapping,
    build_cell_diff_table,
    _pcp_forward_torch,
    _build_index_tables,
)


def _build_s3_symmetric_lift(
    T_compact: np.ndarray, phonon
) -> tuple[np.ndarray, np.ndarray]:
    """Lift the compact FC3 to a fully S3-symmetric (dim_sc, dim_sc, dim_sc) tensor.

    Phono3py stores Phi(l1=0, kappa1; l2 kappa2; l3 kappa3) only.  The full
    lattice FC3 is S3-symmetric in the three (atom, Cartesian) labels and is
    translation-invariant:

        Phi(l1 k1, l2 k2, l3 k3) = Phi(0 k1, (l2-l1) k2, (l3-l1) k3)

    This routine applies the translation identity to populate all rows on leg 1
    (including l1 != 0), producing a tensor that is exact S3-symmetric (no
    averaging needed).

    Parameters
    ----------
    T_compact : ndarray, shape (nat_prim, n_super, n_super, 3, 3, 3)
    phonon : Phonopy

    Returns
    -------
    T_lifted : ndarray, shape (dim_sc, dim_sc, dim_sc)
    p2s_map : ndarray, shape (nat_prim,)
    """
    nat_prim, n_super = T_compact.shape[0], T_compact.shape[1]
    dim_sc = 3 * n_super

    sc_to_cell, sc_to_prim, _, n_cells, sc_mat = build_cell_mapping(phonon)
    cell_diff, _ = build_cell_diff_table(n_cells, sc_mat)

    # (cell_idx, prim_idx) -> supercell atom index (inverse of sc_to_{cell,prim})
    cell_atom_to_sc = np.full((n_cells, nat_prim), -1, dtype=int)
    for s in range(n_super):
        cell_atom_to_sc[sc_to_cell[s], sc_to_prim[s]] = s
    assert np.all(cell_atom_to_sc >= 0), "cell_atom_to_sc has missing entries"

    # For each s1, precompute the j-shift map: shifted_s[s1, s2] = equivalent s2
    # in the "s1 at cell 0" frame: cell(s2) -> cell(s2) - cell(s1).
    shifted = np.zeros((n_super, n_super), dtype=np.int64)
    for s1 in range(n_super):
        l1 = sc_to_cell[s1]
        for s2 in range(n_super):
            l2 = sc_to_cell[s2]
            shifted[s1, s2] = cell_atom_to_sc[cell_diff[l2, l1], sc_to_prim[s2]]

    # T_lifted[s1*3+α1, s2*3+α2, s3*3+α3] = T_compact[κ1, shifted[s1,s2], shifted[s1,s3], α1, α2, α3]
    # Vectorise over s2, s3, and Cartesian components per s1.
    T_lifted = np.zeros((dim_sc, dim_sc, dim_sc), dtype=T_compact.dtype)
    for s1 in range(n_super):
        k1 = sc_to_prim[s1]
        s2_idx = shifted[s1]             # (n_super,)
        s3_idx = shifted[s1]
        # Block: (n_super, n_super, 3, 3, 3) = T_compact[k1][s2_idx][:, s3_idx]
        block = T_compact[k1][s2_idx][:, s3_idx]          # (n_super, n_super, 3, 3, 3)
        # Collapse (s2, α2) and (s3, α3); leg α1 is the first Cartesian axis of block[..., α1, α2, α3]
        # block axes: (s2, s3, α1, α2, α3).  Move α1 to front -> (α1, s2, α2, s3, α3)
        block_r = np.transpose(block, (2, 0, 3, 1, 4)).reshape(3, dim_sc, dim_sc)
        T_lifted[3 * s1:3 * s1 + 3] = block_r

    return T_lifted, phonon.primitive.p2s_map.astype(np.int64)


S3_PERMS = tuple(itertools.permutations(range(3)))


# =====================================================================
# ASR (acoustic-sum-rule) projection helpers
# =====================================================================
#
# For a tensor index of the form j = 3*s + beta (supercell atom index s in
# [0, n_super), Cartesian beta in {0,1,2}), the ASR null-space on that leg is
#   { v : sum_s v[3*s + beta] = 0  for each beta in {0,1,2} }
# i.e. three linear constraints per leg.  The orthogonal projector subtracts
# the Cartesian-by-Cartesian mean over supercell atoms.
#
# Null-space reparametrisation is used here: for multilinear ansatze (CP,
# Tucker, symmetric CP) projecting each factor-column onto this null-space is
# both necessary and sufficient for the reconstructed tensor to satisfy ASR
# on the corresponding leg (see e.g. Comon-Golub-Lim-Mourrain 2008; Kolda-Bader
# 2009; and phonon-community practice in hiPhive / phono3py / ALAMODE).


def _torch_device():
    """Fit device: QX_FIT_DEVICE=cuda|cpu, else cuda when available.
    The L-BFGS refinement is dense f64 linear algebra -- one GH200 runs
    the 16-restart production fits ~an order of magnitude faster than a
    CPU node (Paul 2026-08-02: use GPU where significantly faster)."""
    import os
    dev = os.environ.get("QX_FIT_DEVICE")
    if dev:
        return dev
    try:
        import torch as _t
        return "cuda" if _t.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def asr_project_factor(V: np.ndarray, n_super: int, axis: int = 0,
                       weights: np.ndarray | None = None) -> np.ndarray:
    """Project V along ``axis`` (size dim_sc = 3*n_super) onto the ASR null-space.

    ``weights`` is the per-supercell-atom constraint weight w_s. The fitted
    target is the MASS-WEIGHTED tensor M = Phi / (sqrt(m_i) sqrt(m_s)
    sqrt(m_s')), so the physical acoustic sum rule sum_s Phi[..,s,..] = 0
    reads sum_s sqrt(m_s) M[..,3s+beta,..] = 0 -- the constraint weight is
    w_s = sqrt(m_s), NOT the plain mean. ``weights=None`` keeps the legacy
    unweighted projector, which is identical iff all masses are equal
    (mono-species systems: Si). Idempotent either way.
    """
    V = np.moveaxis(V, axis, 0)
    dim_sc = V.shape[0]
    assert dim_sc == 3 * n_super, f"axis size {dim_sc} != 3*n_super = {3*n_super}"
    rest = V.shape[1:]
    V_r = V.reshape(n_super, 3, *rest)
    if weights is None:
        V_r = V_r - V_r.mean(axis=0, keepdims=True)
    else:
        w = np.asarray(weights, dtype=V_r.real.dtype).reshape(
            (n_super,) + (1,) * (V_r.ndim - 1))
        V_r = V_r - w * ((w * V_r).sum(axis=0, keepdims=True)
                         / (w * w).sum())
    V = V_r.reshape(dim_sc, *rest)
    return np.moveaxis(V, 0, axis)


def _asr_project_torch(V, n_super: int, weights=None):
    """Torch version of asr_project_factor on the first axis.  Differentiable
    in V; ``weights`` (torch tensor, (n_super,)) is a constant."""
    dim_sc = V.shape[0]
    rest = V.shape[1:]
    V_r = V.reshape(n_super, 3, *rest)
    if weights is None:
        V_r = V_r - V_r.mean(dim=0, keepdim=True)
    else:
        w = weights.reshape((n_super,) + (1,) * (V_r.dim() - 1)).to(V_r.dtype)
        V_r = V_r - w * ((w * V_r).sum(dim=0, keepdim=True) / (w * w).sum())
    return V_r.reshape(dim_sc, *rest)


def asr_residual(T_hat: np.ndarray, n_super: int,
                 weights: np.ndarray | None = None) -> dict[str, float]:
    """Return ||ASR_leg(T_hat)|| on legs 1, 2, 3 (axes 0, 1, 2) in Frobenius norm.

    For the (n_dof, dim_sc, dim_sc) target, only legs 2 and 3 are directly
    enforceable (leg 1 is primitive-DOF indexed, not a supercell atom sum).
    With ``weights`` the PHYSICAL (sqrt-mass-weighted) sum is measured --
    see asr_project_factor.
    """
    out = {}
    out["norm"] = float(np.linalg.norm(T_hat))
    for axis_name, axis in [("leg_j", 1), ("leg_k", 2)]:
        R = T_hat.reshape(
            T_hat.shape[0], n_super, 3, n_super, 3
        )
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            R = R * (w[None, :, None, None, None] if axis == 1
                     else w[None, None, None, :, None])
        if axis == 1:
            s = R.sum(axis=1)   # sum over leg-2 supercell atoms
        else:
            s = R.sum(axis=3)
        out[axis_name] = float(np.linalg.norm(s))
    return out


# =====================================================================
# Target construction
# =====================================================================


@dataclass
class FC3Target:
    """Dense mass-weighted FC3 target tensor with metadata."""

    T: np.ndarray                 # (n_dof, dim_sc, dim_sc), real, THz^{5/2}
    T_lifted: np.ndarray          # (dim_sc, dim_sc, dim_sc), real, S3-symmetric
    T_lifted_sym: np.ndarray      # S3-symmetrised T_lifted
    p2s_map: np.ndarray           # (nat_prim,) -> supercell index
    nat_prim: int
    n_super: int
    n_dof: int
    dim_sc: int
    target_norm: float
    s2_residual: float = 0.0      # ||T - T^(j<->k)|| / ||T|| of the INPUT
    asr_weights: np.ndarray | None = None  # (n_super,) sqrt-masses; None = uniform


def build_fc3_target(fc3_raw: np.ndarray, phonon,
                     build_lift: bool = True) -> FC3Target:
    """Build the mass-weighted FC3 target tensor and its S3-lifted version.

    Parameters
    ----------
    fc3_raw : ndarray
        FC3 tensor in compact (nat_prim, n_super, n_super, 3, 3, 3) or full
        (n_super, n_super, n_super, 3, 3, 3) format.
    phonon : Phonopy
    build_lift : bool
        The S3 lift is a (dim_sc)^3 array -- 91 GB for a 5^3 bulk supercell.
        Pass False to skip it (only Waring needs the lift); T_lifted is then
        a zero-size placeholder and fit_waring will raise.
    """
    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)
    n_dof = 3 * nat_prim
    dim_sc = 3 * n_super
    p2s_map = phonon.primitive.p2s_map.astype(np.int64)

    # Compact mass-weighted tensor (nat_prim, n_super, n_super, 3, 3, 3).
    T_compact = _pcp_build_target(fc3_raw, phonon).astype(np.float64)

    # Flatten compact T to (n_dof, dim_sc, dim_sc).
    T = T_compact.transpose(0, 3, 1, 4, 2, 5).reshape(n_dof, dim_sc, dim_sc)

    # Translation-invariant S3-symmetric lift to (dim_sc, dim_sc, dim_sc).
    # By construction this tensor is exactly S3-symmetric (no averaging needed).
    if build_lift:
        T_lifted, _ = _build_s3_symmetric_lift(T_compact, phonon)
    else:
        T_lifted = np.zeros((0, 0, 0))
    T_lifted_sym = T_lifted  # already S3-symmetric by physical identity

    target_norm = float(np.linalg.norm(T))

    # A physical FC3 is exactly S2-symmetric in (j,k) even in the compact
    # primitive-row convention; a violation means a broken input tensor.
    s2_residual = float(
        np.linalg.norm(T - T.transpose(0, 2, 1)) / (target_norm or 1.0))
    if s2_residual > 1e-10:
        import warnings
        warnings.warn(
            f"FC3 target violates S2 leg symmetry: rel residual "
            f"{s2_residual:.3e} > 1e-10 -- the input tensor is suspect.",
            stacklevel=2)

    return FC3Target(
        T=T,
        T_lifted=T_lifted,
        T_lifted_sym=T_lifted_sym,
        p2s_map=p2s_map,
        nat_prim=nat_prim,
        n_super=n_super,
        n_dof=n_dof,
        dim_sc=dim_sc,
        target_norm=target_norm,
        s2_residual=s2_residual,
        asr_weights=np.sqrt(np.asarray(phonon.supercell.masses, dtype=float)),
    )


def target_from_dense(T: np.ndarray, n_super: int,
                      asr_weights: np.ndarray | None = None) -> FC3Target:
    """FC3Target from a dense (n_dof, dim_sc, dim_sc) tensor, no phonopy.

    Used by the production factor pipeline, which fits the SAME mass-weighted
    ``M_stacked`` (solver units, ``build_supercell_mapping`` gauge) that the
    q-folded vertices are built from -- so the factorisation matches the dense
    vertex chain exactly, modulo the fit error. No S3 lift (Waring excluded).
    """
    n_dof, dim_sc, dim_sc2 = T.shape
    assert dim_sc == dim_sc2 == 3 * n_super
    target_norm = float(np.linalg.norm(T))
    s2_residual = float(
        np.linalg.norm(T - T.transpose(0, 2, 1)) / (target_norm or 1.0))
    return FC3Target(
        T=T, T_lifted=np.zeros((0, 0, 0)), T_lifted_sym=np.zeros((0, 0, 0)),
        p2s_map=np.arange(n_dof // 3), nat_prim=n_dof // 3, n_super=n_super,
        n_dof=n_dof, dim_sc=dim_sc, target_norm=target_norm,
        s2_residual=s2_residual, asr_weights=asr_weights)


def _symmetrise_s3(A: np.ndarray) -> np.ndarray:
    """S3 symmetrisation of a cubic tensor on (d, d, d)."""
    return (
        A
        + A.transpose(0, 2, 1)
        + A.transpose(1, 0, 2)
        + A.transpose(1, 2, 0)
        + A.transpose(2, 0, 1)
        + A.transpose(2, 1, 0)
    ) / 6.0


def _slice_to_ndof(A_lifted: np.ndarray, p2s_map: np.ndarray) -> np.ndarray:
    """Extract the primitive-DOF rows from a (dim_sc, dim_sc, dim_sc) lift."""
    dim_sc = A_lifted.shape[1]
    nat_prim = len(p2s_map)
    n_dof = 3 * nat_prim
    out = np.zeros((n_dof, dim_sc, dim_sc), dtype=A_lifted.dtype)
    for i in range(nat_prim):
        s_i = int(p2s_map[i])
        for alpha in range(3):
            out[3 * i + alpha] = A_lifted[3 * s_i + alpha]
    return out


# =====================================================================
# Unified result dataclass
# =====================================================================


@dataclass
class CompressionResult:
    name: str                       # 'mSVD', 'HOSVD', 'CP', 'INDSCAL', 'Waring', 'PCP'
    rank: int | tuple               # scalar R or (R1, R2)
    n_params: int
    rel_err: float                  # ||T - T_approx||_F / ||T||_F (on (n_dof, dim_sc, dim_sc))
    fit_time_s: float
    factors: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


# =====================================================================
# 1. Truncated matricization SVD (mSVD)
# =====================================================================


def fit_msvd(
    target: FC3Target, rank: int, enforce_asr: bool = True,
) -> CompressionResult:
    """Truncated (mu,j)|k matricization SVD.

    Stacks the n_dof matrices M_mu = T[mu, :, :] row-wise into a
    (n_dof * dim_sc, dim_sc) matrix and truncates at ``rank``.
    Eckart-Young optimal at this bipartition.

    If ``enforce_asr`` is True the target is first projected onto the ASR
    null-space on axes j (=1) and k (=2); the resulting U_R, V_R then encode
    an approximation whose legs 2 and 3 satisfy ASR exactly.  For a physical
    FC3 which already satisfies ASR the pre-projection is a no-op (up to
    rounding); for a generic tensor this removes the ASR-violating mass.
    """
    t0 = time.time()
    T = target.T
    n_dof, dim_sc, _ = T.shape
    T_fit = T
    if enforce_asr:
        T_fit = asr_project_factor(T_fit, target.n_super, axis=1, weights=target.asr_weights)
        T_fit = asr_project_factor(T_fit, target.n_super, axis=2, weights=target.asr_weights)
    M = T_fit.reshape(n_dof * dim_sc, dim_sc)

    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    R = min(rank, len(S))
    U_R = U[:, :R] * S[:R]        # absorb sigma into the large factor
    V_R = Vt[:R, :]               # (R, dim_sc)

    if enforce_asr:
        # Safety: re-project V_R columns (axis 1 of V_R = dim_sc), and reshape
        # U_R's second factor (j) in the (n_dof, dim_sc, R) lift to re-project.
        V_R = asr_project_factor(V_R, target.n_super, axis=1, weights=target.asr_weights)
        U_R_cube = U_R.reshape(n_dof, dim_sc, R)
        U_R_cube = asr_project_factor(U_R_cube, target.n_super, axis=1, weights=target.asr_weights)
        U_R = U_R_cube.reshape(n_dof * dim_sc, R)

    M_approx = U_R @ V_R
    M_full = T.reshape(n_dof * dim_sc, dim_sc)
    rel_err = float(np.linalg.norm(M_full - M_approx) / (target.target_norm or 1.0))

    return CompressionResult(
        name="mSVD",
        rank=R,
        n_params=n_params_msvd(R, n_dof, dim_sc),
        rel_err=rel_err,
        fit_time_s=time.time() - t0,
        factors={"U_R": U_R.copy(), "V_R": V_R.copy()},
        info={"singular_values": S.copy(), "enforce_asr": enforce_asr},
    )


def reconstruct_msvd(result: CompressionResult, target: FC3Target) -> np.ndarray:
    U_R = result.factors["U_R"]
    V_R = result.factors["V_R"]
    return (U_R @ V_R).reshape(target.n_dof, target.dim_sc, target.dim_sc)


def n_params_msvd(R: int, n_dof: int, dim_sc: int) -> int:
    """R * ((n_dof+1)*dim_sc + 1)  — equivalent to R*((3*Nprim+1)*ndof+1)."""
    return R * ((n_dof + 1) * dim_sc + 1)


# =====================================================================
# 2. Truncated HOSVD (Tucker) with S2 symmetry
# =====================================================================


def fit_hosvd(
    target: FC3Target,
    R1: int,
    R2: int,
    refine: bool = True,
    hooi_iters: int = 6,
    enforce_asr: bool = True,
) -> CompressionResult:
    """S2-symmetric Tucker decomposition via HOSVD + optional HOOI refinement.

    Factors ``A`` on mode 1 (size n_dof x R1) and a shared ``B`` on modes 2, 3
    (size dim_sc x R2).  Core G is (R1, R2, R2) and S2-symmetric.

    If ``enforce_asr`` is True each HOSVD/HOOI update restricts B to the ASR
    null-space on its dim_sc axis.  Because ``B`` is shared between legs 2
    and 3, the reconstruction then satisfies ASR on both legs (the core and
    A remain unconstrained).
    """
    t0 = time.time()
    T = target.T
    n_dof, dim_sc, _ = T.shape
    n_super = target.n_super

    def _proj_B(B_: np.ndarray) -> np.ndarray:
        if not enforce_asr:
            return B_
        Bp = asr_project_factor(B_, n_super, axis=0, weights=target.asr_weights)
        Q, _ = np.linalg.qr(Bp)
        return Q

    # --- Closed-form HOSVD ---
    # Mode-1 unfolding: (n_dof, dim_sc*dim_sc)
    U1, _, _ = np.linalg.svd(T.reshape(n_dof, -1), full_matrices=False)
    A = U1[:, :R1]

    # Symmetric mode-(2,3) unfolding: combine the mode-2 and mode-3 unfoldings
    # so that the leading singular vectors are S2-compatible.
    M23 = np.concatenate(
        [
            T.transpose(1, 0, 2).reshape(dim_sc, -1),
            T.transpose(2, 0, 1).reshape(dim_sc, -1),
        ],
        axis=1,
    )
    U2, _, _ = np.linalg.svd(M23, full_matrices=False)
    B = _proj_B(U2[:, :R2])

    def _core(A_, B_):
        G_ = np.einsum("mjk,mp,jq,kr->pqr", T, A_, B_, B_, optimize=True)
        return 0.5 * (G_ + G_.transpose(0, 2, 1))   # exact S2

    G = _core(A, B)

    # --- Optional symmetric HOOI refinement ---
    hooi_errs = []
    if refine:
        for _ in range(hooi_iters):
            # Update A holding B fixed: SVD of mode-1 unfolding of T x_2 B^T x_3 B^T.
            TB = np.einsum("mjk,jq,kr->mqr", T, B, B, optimize=True)
            U1, _, _ = np.linalg.svd(TB.reshape(n_dof, -1), full_matrices=False)
            A = U1[:, :R1]

            # Update B holding A fixed, using a joint mode-(2,3) unfolding
            # so the updated B still respects the S2 structure.
            TA = np.einsum("mjk,mp->pjk", T, A, optimize=True)
            M23 = np.concatenate(
                [TA.transpose(1, 0, 2).reshape(dim_sc, -1),
                 TA.transpose(2, 0, 1).reshape(dim_sc, -1)],
                axis=1,
            )
            U2, _, _ = np.linalg.svd(M23, full_matrices=False)
            B = _proj_B(U2[:, :R2])

            G = _core(A, B)

            # Monitor reconstruction error
            T_approx = np.einsum("pqr,mp,jq,kr->mjk", G, A, B, B, optimize=True)
            hooi_errs.append(
                float(np.linalg.norm(T - T_approx) / (target.target_norm or 1.0))
            )

    T_approx = np.einsum("pqr,mp,jq,kr->mjk", G, A, B, B, optimize=True)
    rel_err = float(np.linalg.norm(T - T_approx) / (target.target_norm or 1.0))

    return CompressionResult(
        name="HOSVD",
        rank=(R1, R2),
        n_params=n_params_hosvd(R1, R2, n_dof, dim_sc),
        rel_err=rel_err,
        fit_time_s=time.time() - t0,
        factors={"A": A.copy(), "B": B.copy(), "G": G.copy()},
        info={"hooi_errs": hooi_errs, "enforce_asr": enforce_asr},
    )


def reconstruct_hosvd(result: CompressionResult, target: FC3Target) -> np.ndarray:
    A = result.factors["A"]
    B = result.factors["B"]
    G = result.factors["G"]
    return np.einsum("pqr,mp,jq,kr->mjk", G, A, B, B, optimize=True)


def n_params_hosvd(R1: int, R2: int, n_dof: int, dim_sc: int) -> int:
    # S2-symmetric core: R1 * R2 * (R2+1) // 2
    return n_dof * R1 + dim_sc * R2 + R1 * R2 * (R2 + 1) // 2


# =====================================================================
# 3. Canonical Polyadic decomposition (CP)
# =====================================================================


def fit_cp(
    target: FC3Target,
    rank: int,
    n_restarts: int = 5,
    max_iter: int = 500,
    tol: float = 1e-10,
    seed: int = 0,
    use_linesearch: bool = True,
    lbfgs_refine: bool = True,
    lbfgs_iters: int = 200,
    verbose: bool = False,
    enforce_asr: bool = True,
) -> CompressionResult:
    """Unconstrained CP via tensorly ALS with ELS + optional L-BFGS refinement.

    Runs ``n_restarts`` ALS fits from random inits, keeps the best by
    residual, and (if requested) refines the top candidate with L-BFGS on
    (A, B, C, lambda).  Factors on the two internal modes are left
    independent — symmetry is NOT imposed.
    """
    if not _HAVE_TENSORLY:
        raise RuntimeError("tensorly is required for fit_cp; please install it.")
    t0 = time.time()
    T = target.T
    T_tl = tl.tensor(T, dtype=tl.float64)

    best_err = np.inf
    best_factors: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    rng = np.random.default_rng(seed)
    restart_errs: list[float] = []

    for trial in range(n_restarts):
        init = "random" if trial > 0 else "svd"  # one SVD init, rest random
        try:
            weights, factors = _tl_parafac(
                T_tl,
                rank=rank,
                n_iter_max=max_iter,
                init=init,
                normalize_factors=True,
                tol=tol,
                random_state=int(rng.integers(0, 2**31 - 1)),
                linesearch=use_linesearch,
            )
        except Exception as exc:  # rank-degeneracy or failure
            if verbose:
                print(f"    CP restart {trial}: ALS failed ({exc})")
            continue
        A, B, C = (np.asarray(f) for f in factors)
        lam = np.asarray(weights)
        err = _cp_error(T, A, B, C, lam) / (target.target_norm or 1.0)
        restart_errs.append(err)
        if err < best_err:
            best_err = err
            best_factors = (A.copy(), B.copy(), C.copy(), lam.copy())

    if best_factors is None:
        raise RuntimeError("CP failed on all restarts")

    A, B, C, lam = best_factors

    # --- Initial ASR projection of B and C (necessary & sufficient for ASR on
    # legs 2 and 3 of the rank-1 outer products).  We then re-solve for A and
    # lambdas in closed form given the projected (B, C) to stabilise the init.
    if enforce_asr:
        B = asr_project_factor(B, target.n_super, weights=target.asr_weights)
        C = asr_project_factor(C, target.n_super, weights=target.asr_weights)

    # --- L-BFGS refinement in PyTorch (only keep if it improves) ---
    if lbfgs_refine and torch is not None:
        A2, B2, C2, lam2, lbfgs_err = _cp_lbfgs_refine(
            T, A, B, C, lam, n_iter=lbfgs_iters, target_norm=target.target_norm,
            enforce_asr=enforce_asr, n_super=target.n_super,
            weights=target.asr_weights,
        )
        if lbfgs_err < best_err:
            best_err = lbfgs_err
            A, B, C, lam = A2, B2, C2, lam2

    if enforce_asr:
        B = asr_project_factor(B, target.n_super, weights=target.asr_weights)
        C = asr_project_factor(C, target.n_super, weights=target.asr_weights)
        best_err = _cp_error(T, A, B, C, lam) / (target.target_norm or 1.0)

    # Sort by |lambda| descending
    order = np.argsort(-np.abs(lam))
    A, B, C, lam = A[:, order], B[:, order], C[:, order], lam[order]

    n_dof, dim_sc = target.n_dof, target.dim_sc
    return CompressionResult(
        name="CP",
        rank=rank,
        n_params=n_params_cp(rank, n_dof, dim_sc),
        rel_err=best_err,
        fit_time_s=time.time() - t0,
        factors={"A": A, "B": B, "C": C, "lambdas": lam},
        info={"restart_errs": restart_errs, "enforce_asr": enforce_asr},
    )


def _cp_error(
    T: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, lam: np.ndarray
) -> float:
    T_approx = np.einsum("r,mr,jr,kr->mjk", lam, A, B, C, optimize=True)
    return float(np.linalg.norm(T - T_approx))


def _cp_lbfgs_refine(
    T, A, B, C, lam, n_iter: int, target_norm: float,
    enforce_asr: bool = False, n_super: int | None = None,
    weights=None,
):
    Tn = target_norm or 1.0
    _dev = _torch_device()
    Tt = torch.tensor(T / Tn, dtype=torch.float64, device=_dev)
    At = torch.tensor(A, dtype=torch.float64, device=_dev, requires_grad=True)
    Bt = torch.tensor(B, dtype=torch.float64, device=_dev, requires_grad=True)
    Ct = torch.tensor(C, dtype=torch.float64, device=_dev, requires_grad=True)
    lamt = torch.tensor(lam / Tn, dtype=torch.float64, device=_dev,
                        requires_grad=True)
    _wt = (None if weights is None else
           torch.tensor(np.asarray(weights, dtype=float), device=_dev))

    opt = torch.optim.LBFGS(
        [At, Bt, Ct, lamt],
        lr=1.0,
        max_iter=max(n_iter, 1),
        history_size=30,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
    )

    def closure():
        opt.zero_grad()
        B_eff = _asr_project_torch(Bt, n_super, _wt) if enforce_asr else Bt
        C_eff = _asr_project_torch(Ct, n_super, _wt) if enforce_asr else Ct
        Tap = torch.einsum("r,mr,jr,kr->mjk", lamt, At, B_eff, C_eff)
        loss = torch.sum((Tap - Tt) ** 2)
        loss.backward()
        return loss

    opt.step(closure)

    A = At.detach().cpu().numpy()
    B = Bt.detach().cpu().numpy()
    C = Ct.detach().cpu().numpy()
    lam = lamt.detach().cpu().numpy() * Tn
    if enforce_asr:
        B = asr_project_factor(B, n_super, weights=weights)
        C = asr_project_factor(C, n_super, weights=weights)
    err = _cp_error(T, A, B, C, lam) / Tn
    return A, B, C, lam, err


def reconstruct_cp(result: CompressionResult, target: FC3Target) -> np.ndarray:
    f = result.factors
    return np.einsum(
        "r,mr,jr,kr->mjk", f["lambdas"], f["A"], f["B"], f["C"], optimize=True
    )


def n_params_cp(R: int, n_dof: int, dim_sc: int) -> int:
    return R * (n_dof + 2 * dim_sc + 1)


# =====================================================================
# 3b. Contracted-leg-symmetric paired CP
# =====================================================================


def symmetrise_cp_result(
    result: CompressionResult, target: FC3Target
) -> CompressionResult:
    """Pair a CP fit with its contracted-leg transpose.

    The orthogonal projection ``(T_hat + T_hat.swapaxes(1, 2)) / 2`` is
    represented without reconstructing the dense tensor by concatenating
    every ``A*B*C`` component with ``A*C*B``.  The exported rank doubles,
    while exact S2 symmetry and the ASR of both contracted factors are
    retained.  Projection cannot increase the error for an S2-symmetric
    target.
    """
    if result.name != "CP":
        raise ValueError("only a CP result can be paired")
    A, B, C = (np.asarray(result.factors[key]) for key in ("A", "B", "C"))
    lam = np.asarray(result.factors.get("lambdas", np.ones(A.shape[1])))
    paired = CompressionResult(
        name="S2CP",
        rank=2 * int(result.rank),
        n_params=n_params_cp(2 * int(result.rank), target.n_dof, target.dim_sc),
        rel_err=np.nan,
        fit_time_s=result.fit_time_s,
        factors={
            "A": np.concatenate((A, A), axis=1),
            "B": np.concatenate((B, C), axis=1),
            "C": np.concatenate((C, B), axis=1),
            "lambdas": 0.5 * np.concatenate((lam, lam)),
        },
        info={**result.info, "paired_base_rank": int(result.rank)},
    )
    approximation = reconstruct_cp(paired, target)
    paired.rel_err = float(
        np.linalg.norm(target.T - approximation)
        / (target.target_norm or 1.0)
    )
    return paired


def fit_s2cp(
    target: FC3Target,
    rank: int,
    **kwargs,
) -> CompressionResult:
    """Fit CP at half rank and pair it to enforce contracted-leg symmetry."""
    if rank < 2 or rank % 2:
        raise ValueError("S2CP final rank must be a positive even integer")
    return symmetrise_cp_result(
        fit_cp(target, rank=rank // 2, **kwargs), target
    )


# =====================================================================
# 4. INDSCAL (CP with internal-leg symmetry b_r = c_r)
# =====================================================================


def fit_indscal(
    target: FC3Target,
    rank: int,
    n_restarts: int = 5,
    max_iter: int = 500,
    lbfgs_iters: int = 300,
    tol: float = 1e-10,
    seed: int = 0,
    use_algebraic_init: bool = True,
    verbose: bool = False,
    enforce_asr: bool = True,
) -> CompressionResult:
    """INDSCAL: T[mu, j, k] ~ sum_r D[mu, r] * V[j, r] * V[k, r].

    Strategy: algebraic init via SVD + eigendecomposition of symmetrised
    slices (cheap, non-optimal), then joint L-BFGS refinement under the
    symmetry constraint.  Multiple random restarts are used in addition to
    the algebraic init.
    """
    if torch is None:
        raise RuntimeError("INDSCAL refinement requires torch.")
    t0 = time.time()
    T = target.T
    n_dof, dim_sc, _ = T.shape
    rng = np.random.default_rng(seed)
    restart_errs = []

    # Symmetrise target on (j,k) to respect the INDSCAL constraint exactly.
    T_sym = 0.5 * (T + T.transpose(0, 2, 1))

    best_err = np.inf
    best_D = best_V = None

    for trial in range(n_restarts + (1 if use_algebraic_init else 0)):
        if use_algebraic_init and trial == 0:
            D0, V0 = _indscal_algebraic_init(T_sym, rank, rng=rng)
        else:
            D0 = rng.normal(0, 1.0 / np.sqrt(rank), (n_dof, rank))
            V0 = rng.normal(0, 1.0 / np.sqrt(rank), (dim_sc, rank))

        if enforce_asr:
            V0 = asr_project_factor(V0, target.n_super, weights=target.asr_weights)

        D, V, err = _indscal_als(
            T_sym, D0, V0, max_iter=max_iter, tol=tol, target_norm=target.target_norm,
            enforce_asr=enforce_asr, n_super=target.n_super,
            weights=target.asr_weights,
        )
        restart_errs.append(err)
        if err < best_err:
            best_err = err
            best_D, best_V = D.copy(), V.copy()

    # L-BFGS joint refinement on (D, V)
    D, V, lbfgs_err = _indscal_lbfgs(
        T_sym, best_D, best_V, n_iter=lbfgs_iters, target_norm=target.target_norm,
        enforce_asr=enforce_asr, n_super=target.n_super,
        weights=target.asr_weights,
    )
    if lbfgs_err < best_err:
        best_err = lbfgs_err
        best_D, best_V = D.copy(), V.copy()

    if enforce_asr:
        best_V = asr_project_factor(best_V, target.n_super, weights=target.asr_weights)

    # Final error on the original (non-symmetrised) target
    T_approx = np.einsum("mr,jr,kr->mjk", best_D, best_V, best_V, optimize=True)
    rel_err = float(np.linalg.norm(T - T_approx) / (target.target_norm or 1.0))

    # The irreducible floor from fitting the (j,k)-symmetrised target: any
    # S2-asymmetric content of T cannot be represented by the ansatz.
    s2_dropped = float(
        np.linalg.norm(T - T_sym) / (target.target_norm or 1.0))

    return CompressionResult(
        name="INDSCAL",
        rank=rank,
        n_params=n_params_indscal(rank, n_dof, dim_sc),
        rel_err=rel_err,
        fit_time_s=time.time() - t0,
        factors={"D": best_D, "V": best_V},
        info={"restart_errs": restart_errs, "sym_err": best_err,
              "enforce_asr": enforce_asr, "seed": seed,
              "best_trial": int(np.argmin(restart_errs)),
              "s2_asymmetry_dropped": s2_dropped},
    )


def _indscal_algebraic_init(T_sym: np.ndarray, rank: int,
                            rng: np.random.Generator | None = None):
    """Algebraic init: SVD of T_sym.reshape(n_dof, dim_sc**2), then
    eigendecompose each right-singular slice (Carroll-Chang / ten Berge).

    ``rng`` seeds the (rare) padding when the algebraic spectrum yields fewer
    than ``rank`` components; threaded from the caller for determinism.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_dof, dim_sc, _ = T_sym.shape
    Phi_flat = T_sym.reshape(n_dof, dim_sc * dim_sc)
    U, S, Vt = np.linalg.svd(Phi_flat, full_matrices=False)

    d_list, v_list, amp_list = [], [], []
    for r in range(len(S)):
        if S[r] < 1e-12 * S[0]:
            break
        W = (S[r] * Vt[r]).reshape(dim_sc, dim_sc)
        W = 0.5 * (W + W.T)
        eigvals, eigvecs = np.linalg.eigh(W)
        order = np.argsort(-np.abs(eigvals))
        for p in order:
            if np.abs(eigvals[p]) < 1e-12 * np.max(np.abs(eigvals)):
                continue
            d_list.append(U[:, r] * eigvals[p])
            v_list.append(eigvecs[:, p])
            amp_list.append(np.abs(eigvals[p]) * np.linalg.norm(U[:, r]))

    order = np.argsort(-np.asarray(amp_list))[:rank]
    d_arr = np.stack([d_list[i] for i in order], axis=1)  # (n_dof, R)
    v_arr = np.stack([v_list[i] for i in order], axis=1)  # (dim_sc, R)
    if d_arr.shape[1] < rank:
        pad = rank - d_arr.shape[1]
        d_arr = np.concatenate(
            [d_arr, 1e-3 * rng.normal(size=(n_dof, pad))], axis=1
        )
        v_arr = np.concatenate(
            [v_arr, 1e-3 * rng.normal(size=(dim_sc, pad))], axis=1
        )
    return d_arr, v_arr


def _indscal_als(
    T_sym, D, V, max_iter: int, tol: float, target_norm: float,
    enforce_asr: bool = False, n_super: int | None = None,
    weights=None,
):
    """Symmetric ALS: update D in closed form, update V via symmetric solve."""
    n_dof, dim_sc, _ = T_sym.shape
    prev_err = np.inf
    Tn = target_norm or 1.0
    for _ in range(max_iter):
        # Update D: T_{m,j,k} = sum_r D[m,r] V[j,r] V[k,r]
        # => T.reshape(n_dof, dim_sc*dim_sc) = D @ KR(V,V).T   (KR = Khatri-Rao)
        KR = (V[:, None, :] * V[None, :, :]).reshape(dim_sc * dim_sc, -1)
        Gram = KR.T @ KR
        RHS = T_sym.reshape(n_dof, -1) @ KR
        D = np.linalg.solve(Gram + 1e-12 * np.eye(Gram.shape[0]), RHS.T).T

        # Update V: V_new = (unfold_mode(T) @ KR) (KR^T KR)^{-1} averaged
        # across modes 2 and 3 to respect the internal-leg symmetry.
        # KR_vd[(k,m), r] = V[k, r] * D[m, r]; Gram = (V^T V) * (D^T D).
        KR_vd = (V[:, None, :] * D[None, :, :]).reshape(-1, V.shape[1])
        G_sys = (V.T @ V) * (D.T @ D)
        RHS2 = T_sym.transpose(1, 2, 0).reshape(dim_sc, -1) @ KR_vd
        RHS3 = T_sym.transpose(2, 1, 0).reshape(dim_sc, -1) @ KR_vd
        V1 = np.linalg.solve(G_sys + 1e-12 * np.eye(G_sys.shape[0]), RHS2.T).T
        V2 = np.linalg.solve(G_sys + 1e-12 * np.eye(G_sys.shape[0]), RHS3.T).T
        V = 0.5 * (V1 + V2)
        if enforce_asr:
            V = asr_project_factor(V, n_super, weights=weights)

        T_approx = np.einsum("mr,jr,kr->mjk", D, V, V, optimize=True)
        err = float(np.linalg.norm(T_sym - T_approx) / Tn)
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    return D, V, err


def _indscal_lbfgs(
    T_sym, D, V, n_iter: int, target_norm: float,
    enforce_asr: bool = False, n_super: int | None = None,
    weights=None,
):
    Tn = target_norm or 1.0
    _dev = _torch_device()
    Tt = torch.tensor(T_sym / Tn, dtype=torch.float64, device=_dev)
    Dt = torch.tensor(D, dtype=torch.float64, device=_dev, requires_grad=True)
    Vt = torch.tensor(V, dtype=torch.float64, device=_dev, requires_grad=True)
    _wt = (None if weights is None else
           torch.tensor(np.asarray(weights, dtype=float), device=_dev))

    opt = torch.optim.LBFGS(
        [Dt, Vt],
        lr=1.0,
        max_iter=max(n_iter, 1),
        history_size=30,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
    )

    def closure():
        opt.zero_grad()
        V_eff = _asr_project_torch(Vt, n_super, _wt) if enforce_asr else Vt
        Tap = torch.einsum("mr,jr,kr->mjk", Dt, V_eff, V_eff)
        loss = torch.sum((Tap - Tt) ** 2)
        loss.backward()
        return loss

    opt.step(closure)

    D = Dt.detach().cpu().numpy()
    V = Vt.detach().cpu().numpy()
    if enforce_asr:
        V = asr_project_factor(V, n_super, weights=weights)
    T_approx = np.einsum("mr,jr,kr->mjk", D, V, V, optimize=True)
    err = float(np.linalg.norm(T_sym - T_approx) / Tn)
    return D, V, err


def reconstruct_indscal(result: CompressionResult, target: FC3Target) -> np.ndarray:
    D = result.factors["D"]
    V = result.factors["V"]
    return np.einsum("mr,jr,kr->mjk", D, V, V, optimize=True)


def n_params_indscal(R: int, n_dof: int, dim_sc: int) -> int:
    return R * (n_dof + dim_sc)


# =====================================================================
# 5. Symmetric CP (Waring)
# =====================================================================


def fit_waring(
    target: FC3Target,
    rank: int,
    n_restarts: int = 3,
    n_power_repeats: int = 5,
    n_power_iters: int = 100,
    lbfgs_iters: int = 200,
    seed: int = 0,
    verbose: bool = False,
    cp_init: bool = True,
    power_init: bool = True,
    loss: str = "primitive",
    enforce_asr: bool = True,
    early_stop_rel_err: float = 1e-8,
    max_time_s: float | None = None,
) -> CompressionResult:
    r"""Symmetric CP (Waring) with factors shared across the three legs.

    The ansatz stores factors ``V`` of shape (dim_sc, R) and ``lambdas`` of
    shape (R,) and reconstructs the primitive-row FC3 as

        T_approx[3i+alpha, j, k] = sum_r lam_r V[3*p2s_map[i]+alpha, r]
                                          * V[j, r] * V[k, r]

    If ``enforce_asr`` is True the factor V is constrained to the ASR
    null-space on its supercell-atom axis (per-column Cartesian-mean removed).
    This is necessary and sufficient for the reconstruction to satisfy the
    acoustic sum rule on legs 2 and 3 (see asr_project_factor docstring).

    Parameters
    ----------
    rank : int
    n_restarts : int, default 3
        Random restarts in addition to the power-iteration and optional CP init.
    n_power_repeats, n_power_iters : int, int, default 5, 100
        Arguments to tensorly's shifted symmetric power iteration.  The
        original defaults (30, 300) were dominated the total runtime — 9000
        inner iterations per power-init call — without measurable gain on the
        L-BFGS refined output.
    lbfgs_iters : int, default 200
        Max L-BFGS iterations per candidate.  Strong-Wolfe line search does
        multiple closures per iter; the optimiser typically terminates far
        earlier via ``tolerance_change``.
    cp_init : bool, default True
        Run an unconstrained CP fit and symmetrise as an additional init.
    power_init : bool, default True
        Use tensorly's shifted symmetric power iteration as an init.
    loss : {"primitive", "lift"}
        "primitive" (default) fits the (n_dof, dim_sc, dim_sc) target
        directly; "lift" fits the full S3 lift.
    enforce_asr : bool, default False
        Require V columns to live in the ASR null-space on the supercell axis.
    early_stop_rel_err : float, default 1e-8
        Stop the restart loop once any candidate achieves this relative error.
    max_time_s : float or None
        Abort the restart loop (not the current candidate) after this many
        wall-clock seconds and return the best so far.
    """
    if torch is None:
        raise RuntimeError("Waring refinement requires torch.")
    if loss not in ("primitive", "lift"):
        raise ValueError(f"loss must be 'primitive' or 'lift', got {loss!r}")
    if target.T_lifted_sym.size == 0:
        raise RuntimeError(
            "Waring needs the S3 lift; rebuild the target with "
            "build_fc3_target(..., build_lift=True)")
    t0 = time.time()
    T_lifted = target.T_lifted_sym
    T_lifted_norm = float(np.linalg.norm(T_lifted))
    dim_sc = target.dim_sc
    n_super = target.n_super
    Tn = target.target_norm or 1.0
    rng = np.random.default_rng(seed)

    # Primitive-row index: prim_idx[3i+alpha] = 3*p2s_map[i] + alpha
    prim_idx = np.array(
        [3 * int(target.p2s_map[i]) + a for i in range(target.nat_prim) for a in range(3)],
        dtype=np.int64,
    )

    if loss == "primitive":
        def refine(V0, lam0):
            return _waring_lbfgs_primitive(
                target.T, V0, lam0, prim_idx, lbfgs_iters, Tn,
                enforce_asr=enforce_asr, n_super=n_super,
            )
    else:
        def refine(V0, lam0):
            return _waring_lbfgs_lift(
                T_lifted, V0, lam0, lbfgs_iters, T_lifted_norm,
                enforce_asr=enforce_asr, n_super=n_super,
            )

    candidates: list[tuple[float, np.ndarray, np.ndarray, str]] = []

    # Init magnitudes: |v_r| ~ 1, |lam_r| ~ |T_target|/sqrt(rank) so the
    # initial approximation has magnitude ~ |T_target| when components are
    # roughly orthogonal. T_target is whichever tensor the loss actually
    # compares against — the lift for ``loss="lift"`` and the primitive
    # slice (norm Tn) for ``loss="primitive"``. Previously this was
    # hard-coded to ``T_lifted_norm``, so the primitive path started with
    # an initial Tap that is ~6× the primitive target (T_lifted_norm is
    # typically ~sqrt(6) × Tn for an S3-symmetric lift), causing the
    # L-BFGS refinement to overshoot and collapse to λ = 0 — the
    # rank-2/rank-4 "frob_rel_err = 1.0" pathology observed on SiNW.
    target_norm_for_init = Tn if loss == "primitive" else (T_lifted_norm or 1.0)
    lam_scale = target_norm_for_init / np.sqrt(rank)

    def _seed_random() -> tuple[np.ndarray, np.ndarray]:
        V = rng.normal(0, 1.0 / np.sqrt(dim_sc), (dim_sc, rank))
        if enforce_asr:
            V = asr_project_factor(V, n_super)
        V = V / np.maximum(np.linalg.norm(V, axis=0, keepdims=True), 1e-30)
        lam = lam_scale * rng.choice([-1.0, 1.0], size=rank)
        return V, lam

    def _should_stop() -> bool:
        if candidates and candidates[0][0] <= early_stop_rel_err:
            return True
        if max_time_s is not None and (time.time() - t0) >= max_time_s:
            return True
        return False

    def _register(V_ref, lam_ref, err_ref, tag):
        candidates.append((err_ref, V_ref, lam_ref, tag))
        candidates.sort(key=lambda c: c[0])
        if verbose:
            print(f"    {tag}: err={err_ref:.4e}  (best={candidates[0][0]:.4e})",
                  flush=True)

    def _rescale_lam_for_target(lam0: np.ndarray, V0: np.ndarray) -> np.ndarray:
        """Power/CP init returns lambdas scaled to the LIFTED tensor. When
        the L-BFGS loss is on the primitive slice, that initial Tap is
        too big (the primitive's Frobenius is smaller than the lift's by
        roughly ``sqrt(6) × sqrt(n_dof/dim_sc)``). Rescale once so the
        initial loss is close to ‖T‖² rather than ‖T_lifted‖²; that
        avoids the L-BFGS step-1 overshoot that drives λ → 0.
        """
        if loss != "primitive":
            return lam0
        # Compute the actual primitive reconstruction norm at this init.
        Vs = V0[prim_idx]
        T_init = np.einsum("r,mr,jr,kr->mjk", lam0, Vs, V0, V0, optimize=True)
        init_norm = float(np.linalg.norm(T_init))
        if init_norm < 1e-30:
            return lam0
        return lam0 * (Tn / init_norm)

    # --- Init: tensorly power iteration (deflation-based on the lift) ---
    if power_init and _HAVE_TENSORLY:
        try:
            lam0, V0 = _waring_power_init(
                T_lifted, rank, n_power_repeats, n_power_iters, seed
            )
            if enforce_asr:
                V0 = asr_project_factor(V0, n_super)
            lam0 = _rescale_lam_for_target(lam0, V0)
            V_ref, lam_ref, err_ref = refine(V0, lam0)
            _register(V_ref, lam_ref, err_ref, "power")
        except Exception as exc:
            if verbose:
                print(f"    power init failed: {exc}")

    # --- Init: unconstrained CP on the lift, then symmetrise ---
    if cp_init and _HAVE_TENSORLY and not _should_stop():
        try:
            V0, lam0 = _waring_cp_init(T_lifted, rank, seed)
            if enforce_asr:
                V0 = asr_project_factor(V0, n_super)
            lam0 = _rescale_lam_for_target(lam0, V0)
            V_ref, lam_ref, err_ref = refine(V0, lam0)
            _register(V_ref, lam_ref, err_ref, "cp")
        except Exception as exc:
            if verbose:
                print(f"    cp init failed: {exc}")

    # --- Random inits ---
    for trial in range(n_restarts):
        if _should_stop():
            break
        V0, lam0 = _seed_random()
        V_ref, lam_ref, err_ref = refine(V0, lam0)
        _register(V_ref, lam_ref, err_ref, f"random_{trial}")

    if not candidates:
        raise RuntimeError("Waring: all inits failed")

    best_err, V, lam, best_kind = candidates[0]
    # Final projection to enforce ASR exactly on the stored factor.
    if enforce_asr:
        V = asr_project_factor(V, n_super)

    order = np.argsort(-np.abs(lam))
    lam = lam[order]
    V = V[:, order]

    T_lifted_approx = np.einsum("r,ir,jr,kr->ijk", lam, V, V, V, optimize=True)
    T_approx_ndof = _slice_to_ndof(T_lifted_approx, target.p2s_map)
    rel_err = float(np.linalg.norm(target.T - T_approx_ndof) / Tn)
    full_lift_err = float(
        np.linalg.norm(T_lifted - T_lifted_approx) / (T_lifted_norm or 1.0)
    )

    return CompressionResult(
        name="Waring",
        rank=rank,
        n_params=n_params_waring(rank, dim_sc),
        rel_err=rel_err,
        fit_time_s=time.time() - t0,
        factors={"V": V, "lambdas": lam},
        info={
            "best_init": best_kind,
            "restart_errs": [c[0] for c in candidates],
            "full_lift_rel_err": full_lift_err,
            "loss": loss,
        },
    )


def _waring_power_init(T_lifted, rank, n_repeats, n_iters, seed):
    """Tensorly's shifted symmetric power iteration.

    tensorly's symmetric power iteration has no random_state argument, so the
    global numpy RNG state is saved/seeded/restored around the call --
    deterministic without leaking the seed into the caller's RNG stream.
    """
    T_tl = tl.tensor(T_lifted, dtype=tl.float64)
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        lam, V = _tl_sym_cp(T_tl, rank=rank, n_repeat=n_repeats,
                            n_iteration=n_iters)
    finally:
        np.random.set_state(state)
    return np.asarray(lam), np.asarray(V)


def _waring_cp_init(T_lifted, rank, seed):
    """Run unconstrained CP on the S3-symmetric lift and symmetrise.

    For an exactly symmetric target, the CP optimum has near-symmetric
    factor matrices (up to column permutation and sign).  We align the
    three factor matrices to A (via signed-permutation matching) and then
    average to obtain a symmetric initialisation.
    """
    T_tl = tl.tensor(T_lifted, dtype=tl.float64)
    # random_state=seed fully determines the init; no global np.random.seed.
    weights, factors = _tl_parafac(
        T_tl,
        rank=rank,
        n_iter_max=300,
        init="random",
        normalize_factors=True,
        tol=1e-9,
        random_state=seed,
        linesearch=True,
    )
    A, B, C = (np.asarray(f) for f in factors)
    lam = np.asarray(weights)

    # Align B and C columns to A by sign (columns are already unit-norm).
    for r in range(rank):
        if np.dot(B[:, r], A[:, r]) < 0:
            B[:, r] = -B[:, r]
            lam[r] = -lam[r]
        if np.dot(C[:, r], A[:, r]) < 0:
            C[:, r] = -C[:, r]
            lam[r] = -lam[r]
    V = (A + B + C) / 3.0
    # Re-normalise
    V_norms = np.linalg.norm(V, axis=0, keepdims=True)
    V_norms[V_norms == 0] = 1.0
    V = V / V_norms
    lam = lam * (V_norms.ravel() ** 3)
    return V, lam


def _waring_error(T_lifted_sym: np.ndarray, V: np.ndarray, lam: np.ndarray) -> float:
    T_approx = np.einsum("r,ir,jr,kr->ijk", lam, V, V, V, optimize=True)
    return float(np.linalg.norm(T_lifted_sym - T_approx))


def _waring_lbfgs_lift(
    T_lifted_sym, V, lam, n_iter: int, norm: float,
    enforce_asr: bool = False, n_super: int | None = None,
):
    """L-BFGS on ||T_lifted - sum_r lam_r v_r^{otimes 3}||^2."""
    Tn = norm or 1.0
    Tt = torch.tensor(T_lifted_sym / Tn, dtype=torch.float64)
    Vt = torch.tensor(V, dtype=torch.float64, requires_grad=True)
    lamt = torch.tensor(lam / Tn, dtype=torch.float64, requires_grad=True)

    opt = torch.optim.LBFGS(
        [Vt, lamt],
        lr=1.0,
        max_iter=max(n_iter, 1),
        history_size=30,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
    )

    def closure():
        opt.zero_grad()
        V_eff = _asr_project_torch(Vt, n_super) if enforce_asr else Vt
        Tap = torch.einsum("r,ir,jr,kr->ijk", lamt, V_eff, V_eff, V_eff)
        loss = torch.sum((Tap - Tt) ** 2)
        loss.backward()
        return loss

    opt.step(closure)

    V = Vt.detach().cpu().numpy()
    lam = lamt.detach().cpu().numpy() * Tn
    V_out = asr_project_factor(V, n_super) if enforce_asr else V
    err = float(np.linalg.norm(
        T_lifted_sym - np.einsum("r,ir,jr,kr->ijk", lam, V_out, V_out, V_out, optimize=True)
    ))
    return V_out, lam, err


def _waring_lbfgs_primitive(
    T, V, lam, prim_idx, n_iter: int, norm: float,
    enforce_asr: bool = False, n_super: int | None = None,
):
    """L-BFGS on ||T - slice_prim(sum_r lam_r v_r^{otimes 3})||^2.

    Keeps the Waring ansatz (V, lam) but optimises directly against the
    primitive-row target ``T`` of shape (n_dof, dim_sc, dim_sc).  This is
    unbiased for the reported ``rel_err`` and lets low-rank fits
    concentrate accuracy on the rows we actually evaluate.
    """
    Tn = norm or 1.0
    Tt = torch.tensor(T / Tn, dtype=torch.float64)
    Vt = torch.tensor(V, dtype=torch.float64, requires_grad=True)
    lamt = torch.tensor(lam / Tn, dtype=torch.float64, requires_grad=True)
    idx = torch.tensor(prim_idx, dtype=torch.long)

    opt = torch.optim.LBFGS(
        [Vt, lamt],
        lr=1.0,
        max_iter=max(n_iter, 1),
        history_size=30,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
    )

    def closure():
        opt.zero_grad()
        V_eff = _asr_project_torch(Vt, n_super) if enforce_asr else Vt
        Vs = V_eff.index_select(0, idx)
        Tap = torch.einsum("r,mr,jr,kr->mjk", lamt, Vs, V_eff, V_eff)
        loss = torch.sum((Tap - Tt) ** 2)
        loss.backward()
        return loss

    opt.step(closure)

    V = Vt.detach().cpu().numpy()
    lam = lamt.detach().cpu().numpy() * Tn
    V_out = asr_project_factor(V, n_super) if enforce_asr else V
    Vs = V_out[prim_idx]
    err = float(np.linalg.norm(
        T - np.einsum("r,mr,jr,kr->mjk", lam, Vs, V_out, V_out, optimize=True)
    ))
    return V_out, lam, err


def reconstruct_waring(result: CompressionResult, target: FC3Target) -> np.ndarray:
    V = result.factors["V"]
    lam = result.factors["lambdas"]
    T_lifted_approx = np.einsum("r,ir,jr,kr->ijk", lam, V, V, V, optimize=True)
    return _slice_to_ndof(T_lifted_approx, target.p2s_map)


def n_params_waring(R: int, dim_sc: int) -> int:
    return R * (dim_sc + 1)


# =====================================================================
# 6. PCP (Luo et al. 2025)
# =====================================================================


def fit_pcp_wrapped(
    target: FC3Target,
    rank: int,
    phonon,
    fc3_raw,
    max_iter: int = 2000,
    greedy: bool = False,
    verbose: bool = False,
) -> CompressionResult:
    """PCP via phonon_inputs.pcp.  Reconstructs to (n_dof, dim_sc, dim_sc)."""
    t0 = time.time()

    if greedy:
        A_modes, lambdas, info = fit_pcp_greedy(
            fc3_raw, phonon, N_c=rank, iters_per_rank=max(100, max_iter // rank),
            verbose=verbose,
        )
    else:
        A_modes, lambdas, info = fit_pcp(
            fc3_raw, phonon, N_c=rank, max_iter=max_iter, verbose=verbose,
        )

    # Evaluate PCP in the primitive-DOF convention.
    nat_prim = target.nat_prim
    n_super = target.n_super
    sc_to_cell, sc_to_prim, _, n_cells, sc_mat = build_cell_mapping(phonon)
    cell_diff, _ = build_cell_diff_table(n_cells, sc_mat)
    idx1_cell_np, idx_jk_cell_np, sc_to_prim_i = _build_index_tables(
        sc_to_cell, sc_to_prim, cell_diff, nat_prim, n_super, n_cells
    )
    idx1_cell = torch.tensor(idx1_cell_np, dtype=torch.long)
    idx_jk_cell = torch.tensor(idx_jk_cell_np, dtype=torch.long)
    sc_to_prim_t = torch.tensor(sc_to_prim_i, dtype=torch.long)
    with torch.no_grad():
        A_t = torch.tensor(A_modes, dtype=torch.float64)
        lam_t = torch.tensor(lambdas / info["target_norm"], dtype=torch.float64)
        fc3_compact = _pcp_forward_torch(
            A_t, lam_t, idx1_cell, idx_jk_cell, sc_to_prim_t,
            nat_prim, n_super, n_cells, rank,
        ).numpy() * info["target_norm"]

    # (nat_prim, n_super, n_super, 3, 3, 3) -> (n_dof, dim_sc, dim_sc)
    T_approx = fc3_compact.transpose(0, 3, 1, 4, 2, 5).reshape(
        target.n_dof, target.dim_sc, target.dim_sc
    )
    rel_err = float(np.linalg.norm(target.T - T_approx) / (target.target_norm or 1.0))

    return CompressionResult(
        name="PCP",
        rank=rank,
        n_params=n_params_pcp(rank, target.dim_sc),
        rel_err=rel_err,
        fit_time_s=time.time() - t0,
        factors={"A_modes": A_modes, "lambdas": lambdas},
        info=info,
    )


def reconstruct_pcp(
    result: CompressionResult, target: FC3Target, phonon
) -> np.ndarray:
    nat_prim = target.nat_prim
    n_super = target.n_super
    A_modes = result.factors["A_modes"]
    lambdas = result.factors["lambdas"]
    rank = len(lambdas)

    sc_to_cell, sc_to_prim, _, n_cells, sc_mat = build_cell_mapping(phonon)
    cell_diff, _ = build_cell_diff_table(n_cells, sc_mat)
    idx1_cell_np, idx_jk_cell_np, sc_to_prim_i = _build_index_tables(
        sc_to_cell, sc_to_prim, cell_diff, nat_prim, n_super, n_cells
    )
    idx1_cell = torch.tensor(idx1_cell_np, dtype=torch.long)
    idx_jk_cell = torch.tensor(idx_jk_cell_np, dtype=torch.long)
    sc_to_prim_t = torch.tensor(sc_to_prim_i, dtype=torch.long)
    with torch.no_grad():
        A_t = torch.tensor(A_modes, dtype=torch.float64)
        lam_t = torch.tensor(lambdas, dtype=torch.float64)
        fc3_compact = _pcp_forward_torch(
            A_t, lam_t, idx1_cell, idx_jk_cell, sc_to_prim_t,
            nat_prim, n_super, n_cells, rank,
        ).numpy()
    return fc3_compact.transpose(0, 3, 1, 4, 2, 5).reshape(
        target.n_dof, target.dim_sc, target.dim_sc
    )


def n_params_pcp(R: int, dim_sc: int) -> int:
    return R * (3 * dim_sc + 1)


# =====================================================================
# Driver
# =====================================================================


FITTERS: dict[str, Callable[..., CompressionResult]] = {
    "mSVD": fit_msvd,
    "HOSVD": fit_hosvd,
    "CP": fit_cp,
    "S2CP": fit_s2cp,
    "INDSCAL": fit_indscal,
    "Waring": fit_waring,
    "PCP": fit_pcp_wrapped,
}


def annotate_result(
    result: CompressionResult, target: FC3Target, phonon=None
) -> CompressionResult:
    """Attach the ASR and S2 residuals of the RECONSTRUCTION to result.info.

    Every production decision reads these — a fit is only conserving if the
    reconstructed tensor's ASR legs vanish, regardless of what the fitter
    claims it enforced.
    """
    T_hat = reconstruct(result, target, phonon=phonon)
    asr = asr_residual(T_hat, target.n_super,
                       weights=target.asr_weights)
    result.info["asr"] = asr
    result.info["s2_recon"] = float(
        np.linalg.norm(T_hat - T_hat.transpose(0, 2, 1))
        / (target.target_norm or 1.0)
    )
    return result


def compute_all(
    target: FC3Target,
    ranks_per_method: dict[str, list[int] | list[tuple[int, int]]],
    extra_kwargs: dict[str, dict] | None = None,
    verbose: bool = True,
    enforce_asr: bool = True,
    phonon=None,
) -> list[CompressionResult]:
    """Fit every (method, rank) combination requested.

    Parameters
    ----------
    target : FC3Target
    ranks_per_method : dict
        ``{method_name: [rank, rank, ...]}`` — for HOSVD, each entry is a
        (R1, R2) tuple.
    extra_kwargs : dict, optional
        Per-method keyword arguments (e.g. ``{"CP": {"n_restarts": 3}}``).
    enforce_asr : bool
        Threaded to every fitter (top-level, so a study cannot silently run
        ASR-off as the d11a transport-quality sweep did).  PCP has no
        factor-ASR hook and is skipped by this switch (it receives only its
        own ``extra_kwargs``); it is not a production candidate.
    phonon : Phonopy, optional
        Needed only to annotate PCP reconstructions.
    """
    extra_kwargs = extra_kwargs or {}
    results: list[CompressionResult] = []

    for method, ranks in ranks_per_method.items():
        fitter = FITTERS[method]
        kwargs = dict(extra_kwargs.get(method, {}))
        if method != "PCP":
            kwargs.setdefault("enforce_asr", enforce_asr)
        for rank in ranks:
            if verbose:
                print(f"[{method}] rank={rank} ...", flush=True)
            try:
                if method == "HOSVD":
                    R1, R2 = rank
                    res = fitter(target, R1=R1, R2=R2, **kwargs)
                else:
                    res = fitter(target, rank=rank, **kwargs)
                annotate_result(res, target, phonon=phonon)
            except Exception as exc:
                print(f"  {method} rank={rank} failed: {exc}")
                continue
            results.append(res)
            if verbose:
                asr = res.info.get("asr", {})
                print(
                    f"    n_params={res.n_params}, rel_err={res.rel_err:.4e}, "
                    f"asr_j={asr.get('leg_j', float('nan')):.2e}, "
                    f"t={res.fit_time_s:.1f}s"
                )
    return results


def fit_production(
    target: FC3Target,
    rank: int,
    ansatz: str = "INDSCAL",
    n_restarts: int = 16,
    seed: int = 0,
    leg_weights: np.ndarray | None = None,
    verbose: bool = False,
    **kwargs,
) -> CompressionResult:
    """Production fit: INDSCAL (CP fallback), ASR-on, restart-rich, gated.

    Hard-fails when the reconstruction violates the acoustic sum rule
    (legs j,k) beyond 1e-10 of the target norm — a non-conserving vertex must
    never reach the solver.

    ``leg_weights`` (optional, off by default): diagonal per-DOF weights
    applied to legs 2,3 of the target before fitting (fit T·w_j·w_k, factors
    divided by w afterwards); the ASR projection is applied in the UNSCALED
    metric after rescaling, so conservation is unaffected.
    """
    if ansatz not in ("INDSCAL", "CP", "S2CP"):
        raise ValueError("production ansatz must be INDSCAL, CP or S2CP")
    fitter = FITTERS[ansatz]
    enforce_asr = kwargs.pop("enforce_asr", True)

    if leg_weights is not None:
        w = np.asarray(leg_weights, dtype=np.float64)
        assert w.shape == (target.dim_sc,) and np.all(w > 0)
        T_w = target.T * w[None, :, None] * w[None, None, :]
        target_w = FC3Target(
            T=T_w, T_lifted=target.T_lifted, T_lifted_sym=target.T_lifted_sym,
            p2s_map=target.p2s_map, nat_prim=target.nat_prim,
            n_super=target.n_super, n_dof=target.n_dof, dim_sc=target.dim_sc,
            target_norm=float(np.linalg.norm(T_w)),
        )
        res = fitter(target_w, rank=rank, n_restarts=n_restarts, seed=seed,
                     enforce_asr=False, verbose=verbose, **kwargs)
        # Unscale legs 2,3, project ASR in the physical metric, refresh error.
        for key in ("V", "B", "C"):
            if key in res.factors:
                res.factors[key] = asr_project_factor(
                    res.factors[key] / w[:, None], target.n_super)
        annotate_result(res, target)
        T_hat = reconstruct(res, target)
        res.rel_err = float(
            np.linalg.norm(target.T - T_hat) / (target.target_norm or 1.0))
    else:
        res = fitter(target, rank=rank, n_restarts=n_restarts, seed=seed,
                     enforce_asr=enforce_asr, verbose=verbose, **kwargs)
        annotate_result(res, target)

    asr = res.info["asr"]
    tol = 1e-10 * (asr["norm"] or 1.0)
    if asr["leg_j"] > tol or asr["leg_k"] > tol:
        raise RuntimeError(
            f"production fit is not conserving: ASR legs "
            f"j={asr['leg_j']:.3e}, k={asr['leg_k']:.3e} exceed "
            f"1e-10*norm={tol:.3e}")
    res.info["production"] = {"ansatz": ansatz, "n_restarts": n_restarts,
                              "seed": seed,
                              "leg_weighted": leg_weights is not None}
    return res


def export_production_factors(
    result: CompressionResult, target: FC3Target
) -> dict:
    """The content contract consumed by the solver-side factor pipeline.

    Returns {D (n_dof,R), V (dim_sc,R), lambdas (R,), meta}.  INDSCAL stores
    the CP weight inside D (columns unnormalised), so lambdas is ones; CP
    exports (A,B,C) with its weights.  Columns are sorted by descending
    contribution so rank truncation at load time keeps the leading terms.
    """
    meta = {
        "method": result.name, "rank": int(result.rank),
        "rel_err": float(result.rel_err),
        "asr": dict(result.info.get("asr", {})),
        "s2_recon": float(result.info.get("s2_recon", np.nan)),
        "seed": result.info.get("production", {}).get("seed", None),
        "n_dof": target.n_dof, "dim_sc": target.dim_sc,
        "n_super": target.n_super,
    }
    if "paired_base_rank" in result.info:
        meta["paired_base_rank"] = int(result.info["paired_base_rank"])
    if result.name == "INDSCAL":
        D, V = result.factors["D"], result.factors["V"]
        contrib = np.linalg.norm(D, axis=0) * np.linalg.norm(V, axis=0) ** 2
        order = np.argsort(-contrib)
        return {"D": D[:, order].copy(), "V": V[:, order].copy(),
                "lambdas": np.ones(D.shape[1]), "meta": meta}
    if result.name in ("CP", "S2CP"):
        A, B, C = (result.factors[k] for k in ("A", "B", "C"))
        lam = result.factors.get("lambdas", np.ones(A.shape[1]))
        contrib = (np.abs(lam) * np.linalg.norm(A, axis=0)
                   * np.linalg.norm(B, axis=0) * np.linalg.norm(C, axis=0))
        order = np.argsort(-contrib)
        return {"A": A[:, order].copy(), "B": B[:, order].copy(),
                "C": C[:, order].copy(), "lambdas": lam[order].copy(),
                "meta": meta}
    raise ValueError(f"no production export for ansatz {result.name}")


# =====================================================================
# Convenience: reconstruct dispatcher
# =====================================================================


def reconstruct(
    result: CompressionResult, target: FC3Target, phonon=None
) -> np.ndarray:
    """Reconstruct the full (n_dof, dim_sc, dim_sc) approximation."""
    if result.name == "mSVD":
        return reconstruct_msvd(result, target)
    if result.name == "HOSVD":
        return reconstruct_hosvd(result, target)
    if result.name in ("CP", "S2CP"):
        return reconstruct_cp(result, target)
    if result.name == "INDSCAL":
        return reconstruct_indscal(result, target)
    if result.name == "Waring":
        return reconstruct_waring(result, target)
    if result.name == "PCP":
        if phonon is None:
            raise ValueError("PCP reconstruction requires the phonon object")
        return reconstruct_pcp(result, target, phonon)
    raise ValueError(f"Unknown method {result.name}")
