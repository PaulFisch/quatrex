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


def build_fc3_target(fc3_raw: np.ndarray, phonon) -> FC3Target:
    """Build the mass-weighted FC3 target tensor and its S3-lifted version.

    Parameters
    ----------
    fc3_raw : ndarray
        FC3 tensor in compact (nat_prim, n_super, n_super, 3, 3, 3) or full
        (n_super, n_super, n_super, 3, 3, 3) format.
    phonon : Phonopy
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
    T_lifted, _ = _build_s3_symmetric_lift(T_compact, phonon)
    T_lifted_sym = T_lifted  # already S3-symmetric by physical identity

    target_norm = float(np.linalg.norm(T))

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
    )


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


def fit_msvd(target: FC3Target, rank: int) -> CompressionResult:
    """Truncated (mu,j)|k matricization SVD.

    Stacks the n_dof matrices M_mu = T[mu, :, :] row-wise into a
    (n_dof * dim_sc, dim_sc) matrix and truncates at ``rank``.
    Eckart-Young optimal at this bipartition.
    """
    t0 = time.time()
    T = target.T
    n_dof, dim_sc, _ = T.shape
    M = T.reshape(n_dof * dim_sc, dim_sc)

    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    R = min(rank, len(S))
    U_R = U[:, :R] * S[:R]        # absorb sigma into the large factor
    V_R = Vt[:R, :]               # (R, dim_sc)

    M_approx = U_R @ V_R
    rel_err = float(np.linalg.norm(M - M_approx) / (target.target_norm or 1.0))

    return CompressionResult(
        name="mSVD",
        rank=R,
        n_params=n_params_msvd(R, n_dof, dim_sc),
        rel_err=rel_err,
        fit_time_s=time.time() - t0,
        factors={"U_R": U_R.copy(), "V_R": V_R.copy()},
        info={"singular_values": S.copy()},
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
) -> CompressionResult:
    """S2-symmetric Tucker decomposition via HOSVD + optional HOOI refinement.

    Factors ``A`` on mode 1 (size n_dof x R1) and a shared ``B`` on modes 2, 3
    (size dim_sc x R2).  Core G is (R1, R2, R2) and S2-symmetric.
    """
    t0 = time.time()
    T = target.T
    n_dof, dim_sc, _ = T.shape

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
    B = U2[:, :R2]

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
            B = U2[:, :R2]

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
        info={"hooi_errs": hooi_errs},
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

    # --- L-BFGS refinement in PyTorch (only keep if it improves) ---
    if lbfgs_refine and torch is not None:
        A2, B2, C2, lam2, lbfgs_err = _cp_lbfgs_refine(
            T, A, B, C, lam, n_iter=lbfgs_iters, target_norm=target.target_norm
        )
        if lbfgs_err < best_err:
            best_err = lbfgs_err
            A, B, C, lam = A2, B2, C2, lam2

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
        info={"restart_errs": restart_errs},
    )


def _cp_error(
    T: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, lam: np.ndarray
) -> float:
    T_approx = np.einsum("r,mr,jr,kr->mjk", lam, A, B, C, optimize=True)
    return float(np.linalg.norm(T - T_approx))


def _cp_lbfgs_refine(T, A, B, C, lam, n_iter: int, target_norm: float):
    Tn = target_norm or 1.0
    Tt = torch.tensor(T / Tn, dtype=torch.float64)
    At = torch.tensor(A, dtype=torch.float64, requires_grad=True)
    Bt = torch.tensor(B, dtype=torch.float64, requires_grad=True)
    Ct = torch.tensor(C, dtype=torch.float64, requires_grad=True)
    lamt = torch.tensor(lam / Tn, dtype=torch.float64, requires_grad=True)

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
        Tap = torch.einsum("r,mr,jr,kr->mjk", lamt, At, Bt, Ct)
        loss = torch.sum((Tap - Tt) ** 2)
        loss.backward()
        return loss

    opt.step(closure)

    A = At.detach().numpy()
    B = Bt.detach().numpy()
    C = Ct.detach().numpy()
    lam = lamt.detach().numpy() * Tn
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
            D0, V0 = _indscal_algebraic_init(T_sym, rank)
        else:
            D0 = rng.normal(0, 1.0 / np.sqrt(rank), (n_dof, rank))
            V0 = rng.normal(0, 1.0 / np.sqrt(rank), (dim_sc, rank))

        D, V, err = _indscal_als(
            T_sym, D0, V0, max_iter=max_iter, tol=tol, target_norm=target.target_norm
        )
        restart_errs.append(err)
        if err < best_err:
            best_err = err
            best_D, best_V = D.copy(), V.copy()

    # L-BFGS joint refinement on (D, V)
    D, V, lbfgs_err = _indscal_lbfgs(
        T_sym, best_D, best_V, n_iter=lbfgs_iters, target_norm=target.target_norm
    )
    if lbfgs_err < best_err:
        best_err = lbfgs_err
        best_D, best_V = D.copy(), V.copy()

    # Final error on the original (non-symmetrised) target
    T_approx = np.einsum("mr,jr,kr->mjk", best_D, best_V, best_V, optimize=True)
    rel_err = float(np.linalg.norm(T - T_approx) / (target.target_norm or 1.0))

    return CompressionResult(
        name="INDSCAL",
        rank=rank,
        n_params=n_params_indscal(rank, n_dof, dim_sc),
        rel_err=rel_err,
        fit_time_s=time.time() - t0,
        factors={"D": best_D, "V": best_V},
        info={"restart_errs": restart_errs, "sym_err": best_err},
    )


def _indscal_algebraic_init(T_sym: np.ndarray, rank: int):
    """Algebraic init: SVD of T_sym.reshape(n_dof, dim_sc**2), then
    eigendecompose each right-singular slice (Carroll-Chang / ten Berge)."""
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
            [d_arr, 1e-3 * np.random.default_rng(0).normal(size=(n_dof, pad))], axis=1
        )
        v_arr = np.concatenate(
            [v_arr, 1e-3 * np.random.default_rng(1).normal(size=(dim_sc, pad))], axis=1
        )
    return d_arr, v_arr


def _indscal_als(T_sym, D, V, max_iter: int, tol: float, target_norm: float):
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

        T_approx = np.einsum("mr,jr,kr->mjk", D, V, V, optimize=True)
        err = float(np.linalg.norm(T_sym - T_approx) / Tn)
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    return D, V, err


def _indscal_lbfgs(T_sym, D, V, n_iter: int, target_norm: float):
    Tn = target_norm or 1.0
    Tt = torch.tensor(T_sym / Tn, dtype=torch.float64)
    Dt = torch.tensor(D, dtype=torch.float64, requires_grad=True)
    Vt = torch.tensor(V, dtype=torch.float64, requires_grad=True)

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
        Tap = torch.einsum("mr,jr,kr->mjk", Dt, Vt, Vt)
        loss = torch.sum((Tap - Tt) ** 2)
        loss.backward()
        return loss

    opt.step(closure)

    D = Dt.detach().numpy()
    V = Vt.detach().numpy()
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
    n_restarts: int = 5,
    n_power_repeats: int = 10,
    n_power_iters: int = 200,
    lbfgs_iters: int = 400,
    seed: int = 0,
    verbose: bool = False,
    cp_init: bool = True,
    loss: str = "primitive",
) -> CompressionResult:
    r"""Symmetric CP (Waring) with factors shared across the three legs.

    The ansatz stores factors ``V \in R^{dim_sc x R}`` and ``lambdas \in R^R``
    and reconstructs the primitive-row FC3 as

        T_approx[3i+alpha, j, k] = sum_r lam_r V[3*p2s_map[i]+alpha, r]
                                          * V[j, r] * V[k, r]

    i.e. a rank-R sum of cubes of linear forms ``v_r`` restricted to the
    primitive-atom rows on leg 1.  The underlying full-lifted tensor is
    exactly S3-symmetric by construction.

    Parameters
    ----------
    loss : {"primitive", "lift"}
        ``"primitive"`` (default) minimises ``||T - slice(approx)||`` on the
        (n_dof, dim_sc, dim_sc) target we actually want to compress.  This
        is unbiased for the reported ``rel_err``.
        ``"lift"`` minimises ``||T_lifted - approx||`` on the full
        (dim_sc, dim_sc, dim_sc) lift — biased towards non-primitive rows
        at low rank but preserves S3-symmetric optimality at high rank.

    Inits:
      * shifted symmetric power iteration (tensorly),
      * optionally an unconstrained CP fit symmetrised by column alignment
        (the CP optimum of a symmetric tensor is near-symmetric),
      * ``n_restarts`` random inits with properly scaled magnitudes.

    Every init is refined jointly with L-BFGS on (V, lambda) in a single
    outer call so the Hessian history is preserved.
    """
    if torch is None:
        raise RuntimeError("Waring refinement requires torch.")
    if loss not in ("primitive", "lift"):
        raise ValueError(f"loss must be 'primitive' or 'lift', got {loss!r}")
    t0 = time.time()
    T_lifted = target.T_lifted_sym
    T_lifted_norm = float(np.linalg.norm(T_lifted))
    dim_sc = target.dim_sc
    Tn = target.target_norm or 1.0
    rng = np.random.default_rng(seed)

    # Primitive-row index: prim_idx[3i+alpha] = 3*p2s_map[i] + alpha
    prim_idx = np.array(
        [3 * int(target.p2s_map[i]) + a for i in range(target.nat_prim) for a in range(3)],
        dtype=np.int64,
    )

    if loss == "primitive":
        refine = lambda V0, lam0: _waring_lbfgs_primitive(
            target.T, V0, lam0, prim_idx, lbfgs_iters, Tn
        )
    else:
        refine = lambda V0, lam0: _waring_lbfgs_lift(
            T_lifted, V0, lam0, lbfgs_iters, T_lifted_norm
        )

    candidates: list[tuple[float, np.ndarray, np.ndarray, str]] = []

    # Init magnitudes: |v_r| ~ 1, |lam_r| ~ |T_lifted|/sqrt(rank) so that the
    # initial approx has magnitude ~ |T_lifted| when components are roughly
    # orthogonal.
    lam_scale = T_lifted_norm / np.sqrt(rank)

    def _seed_random() -> tuple[np.ndarray, np.ndarray]:
        V = rng.normal(0, 1.0 / np.sqrt(dim_sc), (dim_sc, rank))
        V = V / np.linalg.norm(V, axis=0, keepdims=True)
        lam = lam_scale * rng.choice([-1.0, 1.0], size=rank)
        return V, lam

    # --- Init: tensorly power iteration (deflation-based on the lift) ---
    if _HAVE_TENSORLY:
        try:
            lam0, V0 = _waring_power_init(
                T_lifted, rank, n_power_repeats, n_power_iters, seed
            )
            V_ref, lam_ref, err_ref = refine(V0, lam0)
            candidates.append((err_ref, V_ref, lam_ref, "power"))
            if verbose:
                print(f"    power init: err={err_ref:.4e}")
        except Exception as exc:
            if verbose:
                print(f"    power init failed: {exc}")

    # --- Init: unconstrained CP on the lift, then symmetrise ---
    if cp_init and _HAVE_TENSORLY:
        try:
            V0, lam0 = _waring_cp_init(T_lifted, rank, seed)
            V_ref, lam_ref, err_ref = refine(V0, lam0)
            candidates.append((err_ref, V_ref, lam_ref, "cp"))
            if verbose:
                print(f"    cp init: err={err_ref:.4e}")
        except Exception as exc:
            if verbose:
                print(f"    cp init failed: {exc}")

    # --- Random inits ---
    for trial in range(n_restarts):
        V0, lam0 = _seed_random()
        V_ref, lam_ref, err_ref = refine(V0, lam0)
        candidates.append((err_ref, V_ref, lam_ref, f"random_{trial}"))
        if verbose:
            print(f"    random {trial}: err={err_ref:.4e}")

    candidates.sort(key=lambda c: c[0])
    best_err, V, lam, best_kind = candidates[0]

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
    """Tensorly's shifted symmetric power iteration."""
    T_tl = tl.tensor(T_lifted, dtype=tl.float64)
    np.random.seed(seed)
    lam, V = _tl_sym_cp(T_tl, rank=rank, n_repeat=n_repeats, n_iteration=n_iters)
    return np.asarray(lam), np.asarray(V)


def _waring_cp_init(T_lifted, rank, seed):
    """Run unconstrained CP on the S3-symmetric lift and symmetrise.

    For an exactly symmetric target, the CP optimum has near-symmetric
    factor matrices (up to column permutation and sign).  We align the
    three factor matrices to A (via signed-permutation matching) and then
    average to obtain a symmetric initialisation.
    """
    T_tl = tl.tensor(T_lifted, dtype=tl.float64)
    np.random.seed(seed)
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


def _waring_lbfgs_lift(T_lifted_sym, V, lam, n_iter: int, norm: float):
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
        Tap = torch.einsum("r,ir,jr,kr->ijk", lamt, Vt, Vt, Vt)
        loss = torch.sum((Tap - Tt) ** 2)
        loss.backward()
        return loss

    opt.step(closure)

    V = Vt.detach().numpy()
    lam = lamt.detach().numpy() * Tn
    err = float(np.linalg.norm(T_lifted_sym - np.einsum("r,ir,jr,kr->ijk", lam, V, V, V, optimize=True)))
    return V, lam, err


def _waring_lbfgs_primitive(T, V, lam, prim_idx, n_iter: int, norm: float):
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
        Vs = Vt.index_select(0, idx)
        Tap = torch.einsum("r,mr,jr,kr->mjk", lamt, Vs, Vt, Vt)
        loss = torch.sum((Tap - Tt) ** 2)
        loss.backward()
        return loss

    opt.step(closure)

    V = Vt.detach().numpy()
    lam = lamt.detach().numpy() * Tn
    Vs = V[prim_idx]
    err = float(np.linalg.norm(T - np.einsum("r,mr,jr,kr->mjk", lam, Vs, V, V, optimize=True)))
    return V, lam, err


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
    "INDSCAL": fit_indscal,
    "Waring": fit_waring,
    "PCP": fit_pcp_wrapped,
}


def compute_all(
    target: FC3Target,
    ranks_per_method: dict[str, list[int] | list[tuple[int, int]]],
    extra_kwargs: dict[str, dict] | None = None,
    verbose: bool = True,
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
    """
    extra_kwargs = extra_kwargs or {}
    results: list[CompressionResult] = []

    for method, ranks in ranks_per_method.items():
        fitter = FITTERS[method]
        kwargs = extra_kwargs.get(method, {})
        for rank in ranks:
            if verbose:
                print(f"[{method}] rank={rank} ...", flush=True)
            try:
                if method == "HOSVD":
                    R1, R2 = rank
                    res = fitter(target, R1=R1, R2=R2, **kwargs)
                else:
                    res = fitter(target, rank=rank, **kwargs)
            except Exception as exc:
                print(f"  {method} rank={rank} failed: {exc}")
                continue
            results.append(res)
            if verbose:
                print(
                    f"    n_params={res.n_params}, rel_err={res.rel_err:.4e}, "
                    f"t={res.fit_time_s:.1f}s"
                )
    return results


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
    if result.name == "CP":
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
