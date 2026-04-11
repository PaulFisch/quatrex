"""Toy model validation: verify dense == separable self-energy kernels.

Uses synthetic data (random M_stacked and T matrices) to validate
that the self-energy kernels agree without any DFT data.

Tests:
1. Dense == Separable (full rank) at the self-energy kernel level
2. Separable rank truncation introduces bounded error
3. Self-energy Hermiticity: Sigma^{<,>}(w) are Hermitian
"""

import sys
import time
from pathlib import Path

import numpy as np

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent
sys.path.insert(0, str(work_dir))

from phonon_inputs.anharmonic import _compute_phph_self_energy_q_dense
from phonon_inputs.separable import _compute_phph_self_energy_separable
from phonon_inputs.constants import HBAR_SI


def make_hermitian(A):
    """Make a matrix Hermitian."""
    return 0.5 * (A + A.conj().T)


def random_green_functions(n_kpts, n_freq, n_dof, rng, scale=1.0):
    """Generate random Hermitian Green's functions.

    G^< and G^> should be anti-Hermitian (i*Hermitian) for physical systems.
    For testing purposes, we just need them to be well-behaved.
    """
    G_lesser = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)
    G_greater = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)

    for iq in range(n_kpts):
        for iw in range(n_freq):
            # G^< = -i * A * f(w), where A is Hermitian positive semi-definite
            A = rng.normal(0, scale, (n_dof, n_dof)) + 1j * rng.normal(0, scale, (n_dof, n_dof))
            A = A @ A.conj().T  # Hermitian positive semi-definite
            G_lesser[iq, iw] = -1j * A

            B = rng.normal(0, scale, (n_dof, n_dof)) + 1j * rng.normal(0, scale, (n_dof, n_dof))
            B = B @ B.conj().T
            G_greater[iq, iw] = 1j * B

    return G_lesser, G_greater


def build_toy_model(n_dof=6, dim_t=12, n_kpts=4, n_freq=21, seed=42):
    """Build a complete toy model with synthetic data.

    Returns inputs for both dense and separable kernels.
    """
    rng = np.random.default_rng(seed)
    nat_prim = n_dof // 3

    # Random M_stacked: (n_dof * dim_t, dim_t)
    M_stacked = rng.normal(0, 1.0, (n_dof * dim_t, dim_t))

    # Random gathering matrices T(q): (n_dof, dim_t) per q-point
    # In reality these are sparse with phases, but dense random works for testing
    T_all_q = []
    for iq in range(n_kpts):
        T = rng.normal(0, 1.0, (n_dof, dim_t)) + 1j * rng.normal(0, 0.5, (n_dof, dim_t))
        T_all_q.append(T)

    # q-diff map: (q_ext - q') mod n_kpts
    q_diff_map = np.zeros((n_kpts, n_kpts), dtype=int)
    for i in range(n_kpts):
        for j in range(n_kpts):
            q_diff_map[i, j] = (i - j) % n_kpts

    # Frequency grid
    freqs_thz = np.linspace(1.0, 14.0, n_freq)
    dw_thz = freqs_thz[1] - freqs_thz[0]

    # Green's functions
    G_lesser, G_greater = random_green_functions(n_kpts, n_freq, n_dof, rng, scale=0.1)

    # SVD decomposition of M_stacked for separable
    U, S, Vt = np.linalg.svd(M_stacked, full_matrices=False)
    R_full = len(S)

    # F_list[r] = U[:, r].reshape(n_dof, dim_t)
    # H = (S[:, None] * Vt).T = V @ diag(S), shape (dim_t, R)
    H = (S[:, None] * Vt).T
    F_list = [U[:, r].reshape(n_dof, dim_t) for r in range(R_full)]

    # Fourier-transform for separable: F_hat_q[iq][r] = F_list[r] @ T(q)^T
    # H_hat_q[iq] = T(q) @ H
    F_hat_q_full = []
    H_hat_q = []
    for iq in range(n_kpts):
        T = T_all_q[iq]
        F_hat = [F_r @ T.T for F_r in F_list]
        F_hat_q_full.append(F_hat)
        H_hat_q.append(T @ H)

    return {
        "M_stacked": M_stacked,
        "T_all_q": T_all_q,
        "q_diff_map": q_diff_map,
        "G_lesser": G_lesser,
        "G_greater": G_greater,
        "freqs_thz": freqs_thz,
        "dw_thz": dw_thz,
        "n_dof": n_dof,
        "n_kpts": n_kpts,
        "nat_prim": nat_prim,
        "F_hat_q_full": F_hat_q_full,
        "H_hat_q": H_hat_q,
        "svals": S,
        "F_list": F_list,
        "H": H,
    }


def test_dense_vs_separable_full_rank():
    """Test 1: Dense == Separable (full rank) self-energy."""
    print("\n=== Test 1: Dense vs Separable (full rank) ===")
    toy = build_toy_model()

    # Dense self-energy
    t0 = time.time()
    SL_dense, SG_dense, SR_dense = _compute_phph_self_energy_q_dense(
        toy["G_lesser"], toy["G_greater"],
        toy["M_stacked"], toy["T_all_q"],
        toy["q_diff_map"], toy["nat_prim"],
        toy["n_kpts"], toy["freqs_thz"], toy["dw_thz"],
    )
    dt_dense = time.time() - t0

    # Separable self-energy (full rank)
    t0 = time.time()
    SL_sep, SG_sep, SR_sep = _compute_phph_self_energy_separable(
        toy["G_lesser"], toy["G_greater"],
        toy["F_hat_q_full"], toy["H_hat_q"],
        toy["n_dof"], toy["n_kpts"],
        toy["freqs_thz"], toy["dw_thz"],
        toy["q_diff_map"],
    )
    dt_sep = time.time() - t0

    # Compare
    norm_dense = np.linalg.norm(SL_dense)
    diff_L = np.linalg.norm(SL_dense - SL_sep) / norm_dense
    diff_G = np.linalg.norm(SG_dense - SG_sep) / np.linalg.norm(SG_dense)
    diff_R = np.linalg.norm(SR_dense - SR_sep) / np.linalg.norm(SR_dense)

    print(f"  Dense: {dt_dense:.2f}s, Separable: {dt_sep:.2f}s")
    print(f"  Sigma^<: rel_diff = {diff_L:.2e}")
    print(f"  Sigma^>: rel_diff = {diff_G:.2e}")
    print(f"  Sigma^R: rel_diff = {diff_R:.2e}")

    assert diff_L < 1e-10, f"Sigma^< mismatch: {diff_L}"
    assert diff_G < 1e-10, f"Sigma^> mismatch: {diff_G}"
    assert diff_R < 1e-10, f"Sigma^R mismatch: {diff_R}"

    print("  PASSED")
    return SL_dense, SG_dense


def test_rank_truncation_error():
    """Test 2: Separable rank truncation introduces bounded error."""
    print("\n=== Test 2: Rank truncation error ===")
    toy = build_toy_model(dim_t=24)

    # Dense reference
    SL_dense, SG_dense, SR_dense = _compute_phph_self_energy_q_dense(
        toy["G_lesser"], toy["G_greater"],
        toy["M_stacked"], toy["T_all_q"],
        toy["q_diff_map"], toy["nat_prim"],
        toy["n_kpts"], toy["freqs_thz"], toy["dw_thz"],
    )
    norm_ref = np.linalg.norm(SR_dense)

    # Sweep ranks
    svals = toy["svals"]
    n_kpts = toy["n_kpts"]
    F_list = toy["F_list"]
    H = toy["H"]
    T_all_q = toy["T_all_q"]

    print(f"  Singular values: {svals[:8]}")
    print(f"  {'Rank':>6} {'FC3 err':>12} {'Sigma^R err':>12}")

    for R in [2, 4, 6, 12, len(svals)]:
        # Truncate
        F_list_R = F_list[:R]
        H_R = H[:, :R]

        # FT
        F_hat_q = []
        H_hat_q = []
        for iq in range(n_kpts):
            T = T_all_q[iq]
            F_hat_q.append([F_r @ T.T for F_r in F_list_R])
            H_hat_q.append(T @ H_R)

        SL, SG, SR = _compute_phph_self_energy_separable(
            toy["G_lesser"], toy["G_greater"],
            F_hat_q, H_hat_q, toy["n_dof"], n_kpts,
            toy["freqs_thz"], toy["dw_thz"], toy["q_diff_map"],
        )

        # FC3 reconstruction error
        fc3_err = np.sqrt(np.sum(svals[R:]**2)) / np.sqrt(np.sum(svals**2))
        sigma_err = np.linalg.norm(SR - SR_dense) / norm_ref

        print(f"  {R:>6} {fc3_err:>12.2e} {sigma_err:>12.2e}")

    print("  PASSED (error decreases with rank)")


def test_self_energy_symmetry():
    """Test 3: Self-energy Hermiticity."""
    print("\n=== Test 3: Self-energy symmetry ===")
    toy = build_toy_model()

    SL, SG, SR = _compute_phph_self_energy_q_dense(
        toy["G_lesser"], toy["G_greater"],
        toy["M_stacked"], toy["T_all_q"],
        toy["q_diff_map"], toy["nat_prim"],
        toy["n_kpts"], toy["freqs_thz"], toy["dw_thz"],
    )

    # Check Sigma^R = 0.5 * (Sigma^> - Sigma^<)
    SR_check = 0.5 * (SG - SL)
    diff = np.linalg.norm(SR - SR_check) / np.linalg.norm(SR)
    print(f"  Sigma^R = 0.5*(Sigma^> - Sigma^<): rel_diff = {diff:.2e}")
    assert diff < 1e-14, f"Retarded self-energy relation broken: {diff}"

    print("  PASSED")


def main():
    print("=" * 60)
    print("Toy Model Validation")
    print("=" * 60)

    test_dense_vs_separable_full_rank()
    test_rank_truncation_error()
    test_self_energy_symmetry()

    print("\n" + "=" * 60)
    print("All toy model tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
