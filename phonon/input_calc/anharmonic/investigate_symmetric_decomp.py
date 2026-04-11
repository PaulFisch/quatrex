"""In-depth investigation of symmetric decompositions for FC3.

Mathematical framework:
======================

The mass-weighted FC3 has shape Phi[a, j, k] where:
  a ∈ {1..n_dof}     (primitive cell DOFs, external leg)
  j, k ∈ {1..N_sc}   (supercell DOFs, internal legs)

Key symmetry: Phi[a, j, k] = Phi[a, k, j] (S2 of internal legs).

This means for each external DOF a, the "slice" M_a[j,k] = Phi[a,j,k]
is a SYMMETRIC N_sc × N_sc matrix.

We investigate three decompositions:

1. **SVD separable** (existing):
   Phi[a,j,k] = sum_r F_r[a,j] h_r[k]
   Does NOT respect S2 symmetry. F and h are different objects.
   Kernel cost: O(n_dof^4) per (q,q',w) — the M1/M2/conv_hat chain.

2. **Algebraic partially symmetric CP** (new, this script):
   Step 1: Reshape Phi to (n_dof, N_sc^2), SVD → U_r[a], W_r[j,k]
           W_r is symmetric (proven: data rows lie in symmetric subspace)
   Step 2: Eigendecompose W_r = Q_r Lam_r Q_r^T
   Result: Phi[a,j,k] = sum_{r,p} d_{rp}[a] v_{rp}[j] v_{rp}[k]
   This is EXACT with R_svd × N_sc terms; truncate by eigenvalue.
   Kernel cost: O(R^2) per (q,q',w) — scalar projections, no S3 loop.

3. **PCP → partially symmetric via polarization identity** (new):
   Each PCP component has S3 symmetrization of 3 modes A_1, A_2, A_3.
   Group by external leg, apply partial polarization:
     A_i ⊗ (A_j A_k + A_k A_j) = A_i ⊗ [(A_j+A_k)^2 - A_j^2 - A_k^2]
   Converts each PCP component to 9 partially symmetric CP terms.
   Preserves PCP fitting quality, eliminates S3 loop in kernel.

Key questions this script answers:
- How fast do the eigenvalues of W_r decay? (determines R_total)
- At what R_total does the algebraic decomposition match PCP accuracy?
- How does the partially symmetric CP kernel compare in speed?
- Does PCP's cell-shift structure help or hurt?
"""

import sys
from pathlib import Path
import numpy as np
import time

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs.pcp import (
    fit_pcp, fit_supercell_cp,
    fourier_transform_supercell_cp,
    _compute_phph_self_energy_pcp,
    _build_target,
)
from phonon_inputs.separable import (
    build_supercell_mapping, build_realspace_fc3_matrices,
    build_gathering_matrix, build_q_diff_map,
    _compute_phph_self_energy_separable,
    fourier_transform_factors,
)
from phonon_inputs.anharmonic import _compute_phph_self_energy_q_dense
from phonon_inputs.constants import CONVERSION_FC3_THZ, HBAR_SI
import h5py


# ========================================================================
# Part 1: Algebraic partially symmetric CP decomposition
# ========================================================================

def algebraic_pscp(M_stacked, nat_prim, n_super, threshold=1e-10):
    """Algebraic partially symmetric CP decomposition.

    Given M_stacked (n_dof * N_sc, N_sc), exploits the S2 symmetry
    of the internal legs via SVD + eigendecomposition.

    Returns
    -------
    d_ext : ndarray, shape (R_total, n_dof) — external weights
    v_int : ndarray, shape (R_total, N_sc)  — internal modes (shared for both legs)
    info : dict with diagnostic information
    """
    n_dof = nat_prim * 3
    N_sc = n_super * 3

    # Reshape M_stacked to (n_dof, N_sc, N_sc) tensor
    Phi = M_stacked.reshape(n_dof, N_sc, N_sc)

    # Verify S2 symmetry
    sym_err = np.linalg.norm(Phi - Phi.transpose(0, 2, 1)) / np.linalg.norm(Phi)
    print(f"  S2 symmetry violation: {sym_err:.2e}")

    # Step 1: SVD of (n_dof, N_sc^2) matrix
    Phi_flat = Phi.reshape(n_dof, N_sc * N_sc)
    U_ext, sigma, Vt = np.linalg.svd(Phi_flat, full_matrices=False)
    R_svd = len(sigma)  # = min(n_dof, N_sc^2) = n_dof typically

    print(f"  SVD: R_svd = {R_svd}")
    print(f"  Singular values: {sigma}")
    print(f"  Relative singular values: {sigma / sigma[0]}")

    # Step 2: Eigendecompose each right singular vector (symmetric matrix)
    all_d = []
    all_v = []
    all_eig_info = []

    for r in range(R_svd):
        W_r = (sigma[r] * Vt[r]).reshape(N_sc, N_sc)
        # Force exact symmetry
        W_r = 0.5 * (W_r + W_r.T)

        eigenvalues, eigenvectors = np.linalg.eigh(W_r)
        # Sort by magnitude
        order = np.argsort(-np.abs(eigenvalues))
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        all_eig_info.append({
            'eigenvalues': eigenvalues,
            'n_nonzero': np.sum(np.abs(eigenvalues) > threshold * np.abs(eigenvalues[0]))
        })

        for p in range(len(eigenvalues)):
            if np.abs(eigenvalues[p]) > threshold * np.abs(eigenvalues[0]):
                d_rp = U_ext[:, r] * eigenvalues[p]  # (n_dof,)
                v_rp = eigenvectors[:, p]  # (N_sc,)
                all_d.append(d_rp)
                all_v.append(v_rp)

    d_ext = np.array(all_d)  # (R_total, n_dof)
    v_int = np.array(all_v)  # (R_total, N_sc)
    R_total = len(all_d)

    print(f"  Total partially symmetric CP terms: {R_total}")
    for r in range(R_svd):
        info = all_eig_info[r]
        n_nz = info['n_nonzero']
        eigs = info['eigenvalues']
        print(f"    SVD component {r}: {n_nz} eigenvectors, "
              f"eig range [{eigs[-1]:.2e}, {eigs[0]:.2e}]")

    info = {
        'R_svd': R_svd,
        'R_total': R_total,
        'sigma': sigma,
        'eig_info': all_eig_info,
    }
    return d_ext, v_int, info


def truncate_pscp(d_ext, v_int, R_max):
    """Truncate partially symmetric CP to R_max terms by |d_rp| norm."""
    norms = np.linalg.norm(d_ext, axis=1)  # per-term weight magnitude
    order = np.argsort(-norms)[:R_max]
    return d_ext[order], v_int[order]


def reconstruction_error_pscp(d_ext, v_int, M_stacked, nat_prim, n_super):
    """Compute relative reconstruction error of PSCP decomposition."""
    n_dof = nat_prim * 3
    N_sc = n_super * 3
    R = len(d_ext)

    # Reconstruct: Phi[a,j,k] = sum_r d_r[a] v_r[j] v_r[k]
    # M_approx[a*N_sc + j, k] = sum_r d_r[a] v_r[j] v_r[k]
    M_approx = np.zeros_like(M_stacked)
    for a in range(n_dof):
        # sum_r d_r[a] v_r @ v_r^T
        weighted_v = d_ext[:, a:a+1] * v_int  # (R, N_sc) weighted by d[a]
        M_a = weighted_v.T @ v_int  # (N_sc, N_sc)
        M_approx[a * N_sc:(a + 1) * N_sc, :] = M_a

    return np.linalg.norm(M_approx - M_stacked) / np.linalg.norm(M_stacked)


# ========================================================================
# Part 2: Partially symmetric CP self-energy kernel
# ========================================================================

def fourier_transform_pscp(d_ext, v_int, prim_indices, cell_frac,
                            q_points, nat_prim, transport_direction='x'):
    """FT the partially symmetric CP modes.

    Returns
    -------
    f_int_q : ndarray, shape (n_kpts, R, n_dof), complex
        FT'd internal modes ṽ_r(q) = T(q) @ v_r
    d_ext : ndarray, shape (R, n_dof), real
        External weights (unchanged, no FT needed)
    """
    R = len(d_ext)
    n_dof = nat_prim * 3
    n_kpts = len(q_points)

    f_int_q = np.zeros((n_kpts, R, n_dof), dtype=complex)
    for iq, (qx, qy) in enumerate(q_points):
        T_q = build_gathering_matrix(prim_indices, cell_frac,
                                     (qx, qy), nat_prim, transport_direction)
        f_int_q[iq] = v_int @ T_q.T  # (R, N_sc) @ (N_sc, n_dof)^T → (R, n_dof)

    return f_int_q, d_ext


def compute_self_energy_pscp(
    G_lesser_q, G_greater_q,
    f_int_q, d_ext,
    n_dof, n_kpts, omega_grid_thz, dw_thz,
    q_diff_map=None,
):
    """Self-energy kernel for partially symmetric CP.

    Sigma[a,a'](q,w) = sum_{r,s} d_r[a] d_s[a']
                        × (1/N_q) sum_{q'} conv_w(g_rs(q'), g_rs(q-q'))

    g_rs(q,w) = ṽ_r(q)^T G(q,w) ṽ_s*(q)  — scalar projection.

    No S3 permutation loop. No cell-shift sum.
    """
    n_freq = len(omega_grid_thz)
    R = len(d_ext)

    n_low = max(0, int(np.round(omega_grid_thz[0] / dw_thz)))
    n_ext = n_low + n_freq
    n_fft = 2 * n_ext
    freq_sl = slice(n_low, n_low + n_freq)

    prefactor = 0.5j * HBAR_SI * dw_thz / (2 * np.pi) / n_kpts

    def _pad(G_q):
        out = np.zeros((n_kpts, n_fft, n_dof, n_dof), dtype=complex)
        out[:, n_low:n_low + n_freq] = G_q
        return out

    GL = _pad(G_lesser_q)
    GG = _pad(G_greater_q)

    if q_diff_map is None:
        q_diff_map = np.array([[(i - j) % n_kpts for j in range(n_kpts)]
                                for i in range(n_kpts)])

    # Phase 1: Scalar projections g[r,s,q,w] = f_r^T G f_s*
    gL = np.zeros((R, R, n_kpts, n_fft), dtype=complex)
    gG = np.zeros_like(gL)

    for iq in range(n_kpts):
        f = f_int_q[iq]           # (R, n_dof)
        f_conj = np.conj(f)       # (R, n_dof)

        # g_rs = f_r @ G @ f_s*^T
        Gfc_L = GL[iq] @ f_conj.T   # (n_fft, n_dof, R)
        Gfc_G = GG[iq] @ f_conj.T

        gL[:, :, iq, :] = np.einsum('rm, wmq -> rqw', f, Gfc_L)
        gG[:, :, iq, :] = np.einsum('rm, wmq -> rqw', f, Gfc_G)

    # Phase 2: FFT over omega
    gL_hat = np.fft.fft(gL, axis=-1)
    gG_hat = np.fft.fft(gG, axis=-1)

    # Phase 3: For each q_ext, product + IFFT + sum over q'
    Sigma_lesser = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)
    Sigma_greater = np.zeros_like(Sigma_lesser)

    for iq_ext in range(n_kpts):
        iq_diffs = q_diff_map[iq_ext]  # (n_kpts,)

        gLd = gL_hat[:, :, iq_diffs, :]  # (R, R, n_kpts, n_fft)
        gGd = gG_hat[:, :, iq_diffs, :]

        # Product in Fourier domain — single element-wise multiply
        prod_L = gL_hat * gLd  # (R, R, n_kpts, n_fft)
        prod_G = gG_hat * gGd

        # IFFT, extract freq range, sum over q'
        conv_L = np.sum(np.fft.ifft(prod_L, axis=-1)[:, :, :, freq_sl], axis=2)
        conv_G = np.sum(np.fft.ifft(prod_G, axis=-1)[:, :, :, freq_sl], axis=2)
        # Shape: (R, R, n_freq)

        # Phase 4: Accumulate Sigma = d^T C d
        CL = conv_L.transpose(2, 0, 1)  # (n_freq, R, R)
        CG = conv_G.transpose(2, 0, 1)

        Sigma_lesser[iq_ext] = prefactor * np.einsum('ra,wrs,sb->wab', d_ext, CL, d_ext)
        Sigma_greater[iq_ext] = prefactor * np.einsum('ra,wrs,sb->wab', d_ext, CG, d_ext)

    Sigma_retarded = 0.5 * (Sigma_greater - Sigma_lesser)
    return Sigma_lesser, Sigma_greater, Sigma_retarded


# ========================================================================
# Part 3: PCP → partially symmetric CP via polarization identity
# ========================================================================

def pcp_to_pscp(A_modes, lambdas, phonon):
    """Convert PCP decomposition to partially symmetric CP via polarization.

    PCP: Phi = sum_xi (lam/6) sum_{sigma in S3} A_{s1}[ext] A_{s2}[j] A_{s3}[k]

    Group by external leg, then use partial polarization on internal pair:
      A_i ⊗ (A_j⊗A_k + A_k⊗A_j) = A_i ⊗ [(A_j+A_k)^2 - A_j^2 - A_k^2]

    Each PCP component → 9 PSCP terms (3 external choices × 3 polarization terms).

    Parameters
    ----------
    A_modes : ndarray, shape (3, N_c, n_cells, nat_prim, 3)
    lambdas : ndarray, shape (N_c,)
    phonon : Phonopy

    Returns
    -------
    d_ext : ndarray, shape (R_total, n_dof)
    v_int : ndarray, shape (R_total, N_sc)
    """
    from phonon_inputs.pcp import build_cell_mapping, build_cell_diff_table

    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)
    n_dof = nat_prim * 3
    N_sc = n_super * 3
    N_c = len(lambdas)

    sc_to_cell, sc_to_prim, _, n_cells, sc_mat = build_cell_mapping(phonon)
    cell_diff, _ = build_cell_diff_table(n_cells, sc_mat)
    p2s = phonon.primitive.p2s_map

    # For each PCP component xi, we have 3 modes A_1, A_2, A_3
    # Each A_s has shape (n_cells, nat_prim, 3)
    # Convert to supercell indexing: mode_s[j] = A_s[cell_j, prim_j, cart_j]

    def mode_to_supercell(A_s_xi):
        """Convert PCP mode to supercell-indexed vector."""
        v = np.zeros(N_sc)
        for j in range(n_super):
            cell_j = sc_to_cell[j]
            prim_j = sc_to_prim[j]
            for beta in range(3):
                v[j * 3 + beta] = A_s_xi[cell_j, prim_j, beta]
        return v

    def mode_to_ext(A_s_xi):
        """Convert PCP mode to external (cell-0) weight vector.

        For PCP, the external weight when mode s is on the external leg
        uses cell-0 restriction. But PCP modes depend on cell shifts via
        idx1_cell mapping. For the external leg (i_prim), the cell index
        depends on the reference cell l in the PCP sum.

        For the standard PCP self-energy kernel (not shifted), the
        external mode at cell 0 is A_s(0, kappa, alpha).
        """
        u = np.zeros(n_dof)
        for kappa in range(nat_prim):
            for alpha in range(3):
                u[kappa * 3 + alpha] = A_s_xi[0, kappa, alpha]
        return u

    all_d = []
    all_v = []

    for xi in range(N_c):
        lam = lambdas[xi] / 6.0  # the 1/6 from S3 normalization

        # Get the 3 supercell-indexed modes
        modes_sc = [mode_to_supercell(A_modes[s, xi]) for s in range(3)]
        modes_ext = [mode_to_ext(A_modes[s, xi]) for s in range(3)]

        # S3 permutations, grouped by which mode is external:
        # Pair (ext_mode, {int_mode_1, int_mode_2}) with symmetrized internal pair
        #
        # S3 = {(1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1)}
        # Grouped by first element (external):
        #   ext=1: (2,3) and (3,2) → symmetric pair A_2, A_3
        #   ext=2: (1,3) and (3,1) → symmetric pair A_1, A_3
        #   ext=3: (1,2) and (2,1) → symmetric pair A_1, A_2

        groups = [
            (0, 1, 2),  # ext=A_0, internal pair = (A_1, A_2)
            (1, 0, 2),  # ext=A_1, internal pair = (A_0, A_2)
            (2, 0, 1),  # ext=A_2, internal pair = (A_0, A_1)
        ]

        for i_ext, i_int1, i_int2 in groups:
            u_ext = modes_ext[i_ext]  # (n_dof,)
            v1 = modes_sc[i_int1]     # (N_sc,)
            v2 = modes_sc[i_int2]     # (N_sc,)

            # Partial polarization identity:
            # v1⊗v2 + v2⊗v1 = (v1+v2)^{⊗2} - v1^{⊗2} - v2^{⊗2}

            # Term 1: +(v1+v2)^{⊗2} with weight lam
            all_d.append(lam * u_ext)
            all_v.append(v1 + v2)

            # Term 2: -v1^{⊗2} with weight lam
            all_d.append(-lam * u_ext)
            all_v.append(v1)

            # Term 3: -v2^{⊗2} with weight lam
            all_d.append(-lam * u_ext)
            all_v.append(v2)

    d_ext = np.array(all_d)  # (9*N_c, n_dof)
    v_int = np.array(all_v)  # (9*N_c, N_sc)

    return d_ext, v_int


# ========================================================================
# Part 4: Cost analysis
# ========================================================================

def print_cost_analysis(n_dof, N_sc, n_fft, R_pscp, R_svd, N_c_pcp):
    """Print FLOP comparison for different kernels."""
    # Per (q, q') pair costs:
    # Dense: dominated by V G G V* contraction
    dense = n_fft * n_dof**5 + 2 * n_fft * n_dof**4

    # SVD separable: M1/M2 construction + contraction
    svd = n_fft * (2 * n_dof**2 * R_svd + n_dof**4)

    # SC-CP: 36 permutation pairs × N_c^2 products
    sccp = n_fft * (36 * N_c_pcp**2 + 9 * N_c_pcp * n_dof)

    # PSCP: R^2 products (no S3 loop) + accumulation
    pscp = n_fft * R_pscp**2 + n_fft * R_pscp * n_dof

    print(f"\n  n_dof={n_dof}, N_sc={N_sc}, n_fft={n_fft}")
    print(f"  R_pscp={R_pscp}, R_svd={R_svd}, N_c_pcp={N_c_pcp}")
    print(f"    Dense:        {dense:>12.2e}")
    print(f"    SVD sep:      {svd:>12.2e}  ({dense/svd:.1f}x)")
    print(f"    SC-CP:        {sccp:>12.2e}  ({dense/sccp:.1f}x)")
    print(f"    PSCP:         {pscp:>12.2e}  ({dense/pscp:.1f}x)")


# ========================================================================
# Main
# ========================================================================

def main():
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    nat_prim = len(phonon.primitive.masses)
    n_dof = nat_prim * 3
    n_super = len(phonon.supercell.masses)
    N_sc = n_super * 3

    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(phonon)
    M_stacked = build_realspace_fc3_matrices(fc3_raw, nat_prim,
                                              phonon.supercell.masses, ref_sc_atoms)

    # ==================================================================
    print("=" * 70)
    print("PART 1: Algebraic partially symmetric CP decomposition")
    print("=" * 70)

    d_ext, v_int, info = algebraic_pscp(M_stacked, nat_prim, n_super,
                                         threshold=1e-12)

    # Reconstruction error at various truncation levels
    print(f"\n  Reconstruction error vs truncation:")
    print(f"  {'R_total':>8s} {'rel_err':>12s}")
    print(f"  {'-'*22}")
    for R in [6, 12, 24, 48, 96, 144, 192, info['R_total']]:
        if R > info['R_total']:
            continue
        d_t, v_t = truncate_pscp(d_ext, v_int, R)
        err = reconstruction_error_pscp(d_t, v_t, M_stacked, nat_prim, n_super)
        print(f"  {R:8d} {err:12.4e}")

    # ==================================================================
    print(f"\n{'='*70}")
    print("PART 2: Eigenvalue spectrum analysis")
    print("=" * 70)

    for r in range(info['R_svd']):
        eigs = info['eig_info'][r]['eigenvalues']
        print(f"\n  SVD component {r} (sigma={info['sigma'][r]:.4e}):")
        print(f"    Top 10 |eigenvalues|: {np.abs(eigs[:10])}")
        cum_energy = np.cumsum(eigs**2) / np.sum(eigs**2)
        for frac in [0.9, 0.99, 0.999, 0.9999]:
            idx = np.searchsorted(cum_energy, frac)
            print(f"    {frac*100:6.2f}% energy captured by {idx+1} eigenvectors")

    # ==================================================================
    print(f"\n{'='*70}")
    print("PART 3: Self-energy kernel validation")
    print("=" * 70)

    nk = 4
    q_points = [(i/nk, j/nk) for i in range(nk) for j in range(nk)]
    n_kpts = len(q_points)
    q_diff_map = build_q_diff_map(nk, nk)

    T_all = [build_gathering_matrix(prim_indices, cell_frac, q, nat_prim, 'x')
             for q in q_points]

    n_freq = 21
    freqs = np.linspace(1.0, 14.0, n_freq)
    dw = freqs[1] - freqs[0]

    rng = np.random.default_rng(42)
    G_lesser = rng.standard_normal((n_kpts, n_freq, n_dof, n_dof)) + \
               1j * rng.standard_normal((n_kpts, n_freq, n_dof, n_dof))
    G_greater = rng.standard_normal((n_kpts, n_freq, n_dof, n_dof)) + \
                1j * rng.standard_normal((n_kpts, n_freq, n_dof, n_dof))
    G_lesser = 0.5 * (G_lesser - np.conj(G_lesser.transpose(0, 1, 3, 2)))
    G_greater = 0.5 * (G_greater - np.conj(G_greater.transpose(0, 1, 3, 2)))

    # Dense reference
    print("\n  Dense reference...")
    t0 = time.time()
    SL_d, SG_d, SR_d = _compute_phph_self_energy_q_dense(
        G_lesser, G_greater, M_stacked, T_all, q_diff_map,
        nat_prim, n_kpts, freqs, dw)
    t_dense = time.time() - t0
    norm_R = np.linalg.norm(SR_d)
    print(f"  Time: {t_dense:.3f}s")

    # SVD separable reference
    U_svd, S_svd, Vt_svd = np.linalg.svd(M_stacked, full_matrices=False)
    for R_svd in [6]:
        F_list = [U_svd[:, r:r+1].reshape(n_dof, N_sc) * S_svd[r]
                  for r in range(R_svd)]
        H = Vt_svd[:R_svd, :].T
        F_hat_q = []
        H_hat_q = []
        for T_q in T_all:
            F_hat_q.append([F_r @ T_q.T for F_r in F_list])
            H_hat_q.append(T_q @ H)

        t0 = time.time()
        SL_svd, SG_svd, SR_svd = _compute_phph_self_energy_separable(
            G_lesser, G_greater, F_hat_q, H_hat_q,
            n_dof, n_kpts, freqs, dw, q_diff_map=q_diff_map)
        t_svd = time.time() - t0
        err_svd = np.linalg.norm(SR_svd - SR_d) / norm_R
        print(f"\n  SVD sep R={R_svd}: {t_svd:.3f}s  err={err_svd:.4e}")

    # PSCP at various truncation levels
    print(f"\n  PSCP self-energy (algebraic decomposition):")
    print(f"  {'R':>6s} {'time':>8s} {'SE err':>12s} {'fit err':>12s} {'speedup':>8s}")
    print(f"  {'-'*50}")

    for R in [12, 24, 48, 96, info['R_total']]:
        if R > info['R_total']:
            continue
        d_t, v_t = truncate_pscp(d_ext, v_int, R)
        fit_err = reconstruction_error_pscp(d_t, v_t, M_stacked, nat_prim, n_super)

        f_int_q, d_t_out = fourier_transform_pscp(
            d_t, v_t, prim_indices, cell_frac, q_points, nat_prim, 'x')

        t0 = time.time()
        SL_p, SG_p, SR_p = compute_self_energy_pscp(
            G_lesser, G_greater, f_int_q, d_t_out,
            n_dof, n_kpts, freqs, dw, q_diff_map=q_diff_map)
        t_pscp = time.time() - t0

        err_p = np.linalg.norm(SR_p - SR_d) / norm_R
        print(f"  {R:6d} {t_pscp:8.3f}s {err_p:12.4e} {fit_err:12.4e} {t_dense/t_pscp:8.2f}x")

    # ==================================================================
    print(f"\n{'='*70}")
    print("PART 4: PCP → PSCP via polarization identity")
    print("=" * 70)

    for N_c in [24, 48]:
        print(f"\n  --- PCP N_c={N_c} ---")
        A_modes, lambdas_pcp, info_pcp = fit_pcp(
            fc3_raw, phonon, N_c=N_c, max_iter=2000, verbose=False)
        print(f"  PCP fit err: {info_pcp['rel_err']:.6e}")

        d_pcp, v_pcp = pcp_to_pscp(A_modes, lambdas_pcp, phonon)
        R_polar = len(d_pcp)
        print(f"  Polarization gives {R_polar} PSCP terms (= 9 × {N_c})")

        # Check reconstruction quality
        fit_err_pscp = reconstruction_error_pscp(d_pcp, v_pcp, M_stacked,
                                                  nat_prim, n_super)
        print(f"  PSCP reconstruction err: {fit_err_pscp:.6e}")

        # Self-energy
        f_int_q, d_out = fourier_transform_pscp(
            d_pcp, v_pcp, prim_indices, cell_frac, q_points, nat_prim, 'x')

        t0 = time.time()
        SL_pol, SG_pol, SR_pol = compute_self_energy_pscp(
            G_lesser, G_greater, f_int_q, d_out,
            n_dof, n_kpts, freqs, dw, q_diff_map=q_diff_map)
        t_pol = time.time() - t0

        err_pol = np.linalg.norm(SR_pol - SR_d) / norm_R
        print(f"  SE err: {err_pol:.6e}, time: {t_pol:.3f}s ({t_dense/t_pol:.2f}x)")

        # Try truncating the polarization result
        print(f"  Truncated polarization PSCP:")
        for R_t in [24, 48, 96]:
            if R_t > R_polar:
                continue
            d_t, v_t = truncate_pscp(d_pcp, v_pcp, R_t)
            f_int_q, d_out = fourier_transform_pscp(
                d_t, v_t, prim_indices, cell_frac, q_points, nat_prim, 'x')
            t0 = time.time()
            _, _, SR_t = compute_self_energy_pscp(
                G_lesser, G_greater, f_int_q, d_out,
                n_dof, n_kpts, freqs, dw, q_diff_map=q_diff_map)
            t_t = time.time() - t0
            err_t = np.linalg.norm(SR_t - SR_d) / norm_R
            fit_t = reconstruction_error_pscp(d_t, v_t, M_stacked, nat_prim, n_super)
            print(f"    R={R_t:3d}: SE err={err_t:.4e}, fit={fit_t:.4e}, "
                  f"time={t_t:.3f}s ({t_dense/t_t:.2f}x)")

    # ==================================================================
    print(f"\n{'='*70}")
    print("PART 5: Theoretical cost scaling comparison")
    print("=" * 70)

    n_fft = 200
    for n_d, n_cells_est, label in [
        (6, 8, "Si (2 atoms)"),
        (12, 8, "4 atoms"),
        (30, 27, "10 atoms"),
        (60, 64, "20 atoms"),
        (120, 64, "40 atoms"),
    ]:
        N_sc_est = n_d * n_cells_est // 3 * 3  # approximate
        # R_pscp estimate: n_dof × effective_rank_per_component
        # Conservative: ~2*n_dof for well-converged PSCP
        R_pscp_est = min(2 * n_d, N_sc_est)
        R_svd_est = n_d
        N_c_est = 24

        print(f"\n  {label} (n_dof={n_d}, N_sc≈{N_sc_est}):")
        print_cost_analysis(n_d, N_sc_est, n_fft, R_pscp_est, R_svd_est, N_c_est)

    # ==================================================================
    print(f"\n{'='*70}")
    print("PART 6: Memory comparison")
    print("=" * 70)
    for n_d, N_w, R_est, label in [
        (6, 200, 12, "Si"),
        (30, 200, 60, "10 atoms"),
        (60, 200, 120, "20 atoms"),
        (120, 200, 240, "40 atoms"),
    ]:
        # Dense K tensor: N_w × n_dof^4 (for SVD separable intermediate)
        K_bytes = N_w * n_d**4 * 16
        # PSCP g tensor: R^2 × N_q × N_fft (scalar projections)
        N_q = 16
        g_bytes = R_est**2 * N_q * N_w * 2 * 16  # n_fft ≈ 2*N_w
        # SVD V/U arrays: N_q × n_fft × n_dof × R
        vu_bytes = 4 * N_q * N_w * 2 * n_d * min(n_d, 48) * 16
        print(f"  {label:8s}: SVD K={K_bytes/1e9:.2f} GB, "
              f"PSCP g={g_bytes/1e6:.0f} MB, SVD V/U={vu_bytes/1e6:.0f} MB")


if __name__ == "__main__":
    main()
