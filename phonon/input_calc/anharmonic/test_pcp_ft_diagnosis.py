"""Deep diagnosis of PCP Fourier transform at non-commensurate q.

The PCP decomposition is:
  FC3[i, j, k] = sum_xi (lam/6) sum_sigma sum_l
      A_{s1}(cell_diff[0,l], prim_i, alpha)
    * A_{s2}(cell_diff[cell_j, l], prim_j, beta)
    * A_{s3}(cell_diff[cell_k, l], prim_k, gamma)

The dense FT computes the vertex correctly at any q:
  Phi[a, c, d] = T(q') @ M_a @ T(q-q')^T

where M_a is the FC3 matrix in supercell-atom indexing.

The "factorized" PCP FT claims:
  Phi[a, c, d] = sum_xi (lam/6) sum_sigma f_{s1}(q_ext)[a] f_{s2}(q')[c] f_{s3}(q-q')[d]

where f_s(q) = sum_l exp(-2pi*i*q*R_l) A_s(l, kappa, alpha).

This test proves:
  1. The "factorized" claim fails at non-commensurate q (confirms known result)
  2. A "shifted-FT" approach (sum over l of T(q) @ circshift(A, l)) gives EXACT
     results at ANY q, proving the error is purely in the phase factorization
  3. The shifted-FT approach can be used to build a correct PCP kernel
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
    fit_pcp, fourier_transform_pcp, _compute_phph_self_energy_pcp,
    build_cell_mapping, build_cell_diff_table,
    CONVERSION_FC3_THZ, S3_PERMS,
    reconstruct_fc3_from_pcp,
)
from phonon_inputs.separable import (
    build_supercell_mapping, build_realspace_fc3_matrices,
    build_gathering_matrix, build_q_diff_map,
)
from phonon_inputs.anharmonic import _compute_phph_self_energy_q_dense
from phonon_inputs.constants import HBAR_SI
import h5py


def build_dense_vertex(M_stacked, T_q1, T_q2, n_dof):
    """Dense vertex: Phi[a,c,d] = T(q1) @ M_a @ T(q2)^T."""
    dim_t = M_stacked.shape[1]
    Phi = np.zeros((n_dof, n_dof, n_dof), dtype=complex)
    for a in range(n_dof):
        Phi[a] = T_q1 @ M_stacked[a*dim_t:(a+1)*dim_t, :] @ T_q2.T
    return Phi


def build_pcp_vertex_factorized(f_modes_all_q, lambdas, iq_ext, iq1, iq2):
    """CURRENT code's factorized vertex: product of FT'd modes."""
    N_c = len(lambdas)
    n_dof = f_modes_all_q.shape[-1]
    V = np.zeros((n_dof, n_dof, n_dof), dtype=complex)
    for xi in range(N_c):
        for s1, s2, s3 in S3_PERMS:
            fa = f_modes_all_q[iq_ext, s1, xi, :]
            fc = f_modes_all_q[iq1, s2, xi, :]
            fd = f_modes_all_q[iq2, s3, xi, :]
            V += (lambdas[xi] / 6.0) * fa[:, None, None] * fc[None, :, None] * fd[None, None, :]
    return V


def build_pcp_vertex_shifted_ft(A_modes, lambdas, phonon, T_q1, T_q2, T_qext,
                                 pcp_info):
    """Correct PCP vertex using shifted-FT approach (works at any q).

    For each (xi, sigma, l), builds supercell-indexed vectors:
      v_l[j*3+beta] = A_{s2}(cell_diff[cell_j, l], prim_j, beta)
      w_l[k*3+gamma] = A_{s3}(cell_diff[cell_k, l], prim_k, gamma)

    Then applies T(q') @ v_l and T(q-q') @ w_l, which is the EXACT FT
    without requiring the phase factorization identity.
    """
    nat_prim = len(phonon.primitive.masses)
    n_dof = nat_prim * 3
    n_super = len(phonon.supercell.masses)
    dim_sc = n_super * 3
    N_c = len(lambdas)

    sc_to_cell, sc_to_prim, _, n_cells, sc_mat = build_cell_mapping(phonon)
    cell_diff, idx_to_t = build_cell_diff_table(n_cells, sc_mat)

    V = np.zeros((n_dof, n_dof, n_dof), dtype=complex)

    for xi in range(N_c):
        for s1, s2, s3 in S3_PERMS:
            for l in range(n_cells):
                # External mode: scalar for each DOF a=(i_prim, alpha)
                cell_ext = cell_diff[0, l]  # (-l) mod N
                ext_vals = A_modes[s1, xi, cell_ext, :, :]  # (nat_prim, 3)
                ext_flat = ext_vals.flatten()  # (n_dof,)

                # Build supercell vector for leg 2: v[j*3+beta]
                v = np.zeros(dim_sc)
                for j in range(n_super):
                    cj = cell_diff[sc_to_cell[j], l]
                    pj = sc_to_prim[j]
                    v[j*3:j*3+3] = A_modes[s2, xi, cj, pj, :]

                # Build supercell vector for leg 3: w[k*3+gamma]
                w = np.zeros(dim_sc)
                for k in range(n_super):
                    ck = cell_diff[sc_to_cell[k], l]
                    pk = sc_to_prim[k]
                    w[k*3:k*3+3] = A_modes[s3, xi, ck, pk, :]

                # Apply gathering matrices (exact FT, no phase factorization!)
                fv = T_q1 @ v   # (n_dof,)
                fw = T_q2 @ w   # (n_dof,)

                # Accumulate vertex
                V += (lambdas[xi] / 6.0) * ext_flat[:, None, None] * fv[None, :, None] * fw[None, None, :]

    return V


def test_vertex_comparison():
    """Compare three vertex computation methods at various q-points."""
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    nat_prim = len(phonon.primitive.masses)
    n_dof = nat_prim * 3
    masses_super = phonon.supercell.masses

    print("Fitting PCP N_c=24...")
    A_modes, lambdas, pcp_info = fit_pcp(fc3_raw, phonon, N_c=24,
                                          max_iter=2000, verbose=False)
    print(f"  rel_err = {pcp_info['rel_err']:.6e}")

    # Dense infrastructure from PCP-reconstructed FC3
    fc3_recon = reconstruct_fc3_from_pcp(A_modes, lambdas, phonon, pcp_info)
    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(phonon)
    M_stacked = build_realspace_fc3_matrices(fc3_recon, nat_prim, masses_super, ref_sc_atoms)

    # Also from raw FC3 for reference
    M_stacked_raw = build_realspace_fc3_matrices(fc3_raw, nat_prim, masses_super, ref_sc_atoms)
    recon_err = np.linalg.norm(M_stacked - M_stacked_raw) / np.linalg.norm(M_stacked_raw)
    print(f"  M_stacked recon error: {recon_err:.6e}")

    # ====================================================================
    # TEST: Compare vertices at 4x4 mesh (non-commensurate q-points)
    # ====================================================================
    print("\n" + "="*70)
    print("VERTEX COMPARISON: Dense vs Factorized-FT vs Shifted-FT")
    print("="*70)
    print("\nDense = T(q') @ M_a @ T(q-q')^T (ground truth, from PCP-reconstructed FC3)")
    print("Fact  = product of FT'd PCP modes (current code's approach)")
    print("Shift = sum_l of T(q) applied to shifted supercell vectors (new approach)")

    nk = 4
    q_points = [(i/nk, j/nk) for i in range(nk) for j in range(nk)]
    q_diff_map = build_q_diff_map(nk, nk)

    # PCP FT modes for factorized approach
    f_modes = np.zeros((len(q_points), 3, 24, n_dof), dtype=complex)
    for iq, (qx, qy) in enumerate(q_points):
        f_modes[iq] = fourier_transform_pcp(
            A_modes, lambdas, phonon, (qx, qy), 'x', info=pcp_info)

    # Gathering matrices for all q-points
    T_all = []
    for qx, qy in q_points:
        T = build_gathering_matrix(prim_indices, cell_frac, (qx, qy), nat_prim, 'x')
        T_all.append(T)

    sc_mat = np.diag(phonon.supercell_matrix).astype(int)

    # Test at selected q-point triples
    test_cases = [
        (0, 0, "Gamma,Gamma"),
        (0, 1, "Gamma,(0,0.25)"),
        (0, 5, "Gamma,(0.25,0.25)"),
        (1, 5, "(0,0.25),(0.25,0.25)"),
        (3, 7, "(0,0.75),(0.25,0.75)"),
        (5, 10, "(0.25,0.25),(0.5,0.5)"),
        (6, 11, "(0.25,0.5),(0.5,0.75)"),
    ]

    print(f"\n  {'q_ext,q\'':<30s}  {'Dense norm':>10s}  {'Fact err':>10s}  {'Shift err':>10s}  {'Comm?':>6s}")
    print("  " + "-"*76)

    for iq_ext, iq_prime, label in test_cases:
        iq_diff = q_diff_map[iq_ext, iq_prime]
        qp = q_points[iq_prime]
        qd = q_points[iq_diff]

        # Check commensurability
        comm_p = all(abs(q * sc_mat[i+1] - round(q * sc_mat[i+1])) < 1e-10
                     for i, q in enumerate(qp))
        comm_d = all(abs(q * sc_mat[i+1] - round(q * sc_mat[i+1])) < 1e-10
                     for i, q in enumerate(qd))
        comm = "YES" if (comm_p and comm_d) else "NO"

        # Dense vertex (ground truth)
        V_dense = build_dense_vertex(M_stacked, T_all[iq_prime], T_all[iq_diff], n_dof)
        norm = np.linalg.norm(V_dense)
        if norm < 1e-15:
            continue

        # Factorized vertex (current code)
        V_fact = build_pcp_vertex_factorized(f_modes, lambdas, iq_ext, iq_prime, iq_diff)
        err_fact = np.linalg.norm(V_fact - V_dense) / norm

        # Shifted-FT vertex (new approach)
        V_shift = build_pcp_vertex_shifted_ft(
            A_modes, lambdas, phonon,
            T_all[iq_prime], T_all[iq_diff], T_all[iq_ext],
            pcp_info,
        )
        err_shift = np.linalg.norm(V_shift - V_dense) / norm

        print(f"  {label:<30s}  {norm:>10.4e}  {err_fact:>10.2e}  {err_shift:>10.2e}  {comm:>6s}")

    # ====================================================================
    # Diagnose: WHY does the shifted-FT work?
    # ====================================================================
    print("\n" + "="*70)
    print("DIAGNOSIS: Phase factorization breakdown at non-commensurate q")
    print("="*70)

    sc_to_cell, sc_to_prim, cell_frac_pcp, n_cells, sc_mat_arr = build_cell_mapping(phonon)
    cell_diff, idx_to_t = build_cell_diff_table(n_cells, sc_mat_arr)

    # Pick a non-commensurate q-point: (0.25, 0)
    q_test = (0.25, 0.0)
    tidx = 0  # x transport
    perp_idx = [1, 2]  # y, z
    q_full = np.zeros(3)
    q_full[perp_idx[0]] = q_test[0]
    q_full[perp_idx[1]] = q_test[1]

    print(f"\n  q = {q_test}, sc_mat = {sc_mat_arr}")
    print(f"  q * N_sc = {q_full * sc_mat_arr}")
    print(f"  Commensurate? {all(abs(q_full[i]*sc_mat_arr[i] - round(q_full[i]*sc_mat_arr[i])) < 1e-10 for i in range(3))}")

    # Show phase factorization for each cell shift l
    print(f"\n  Phase factorization test: exp(-iq*R_{{(t+l)%N}}) vs exp(-iq*R_t)*exp(-iq*R_l)")
    print(f"  {'l':>3s} {'R_l':>12s} {'phase(R_l)':>16s} {'max |ratio - 1|':>18s}")
    print("  " + "-"*55)

    for l in range(n_cells):
        R_l = idx_to_t[l]
        phase_l = np.exp(-2j * np.pi * R_l @ q_full)

        max_err = 0.0
        for t in range(n_cells):
            R_t = idx_to_t[t]
            tl_mod = ((R_t + R_l) % sc_mat_arr)
            # Find cell index of (t+l) mod N
            idx_tl = ((tl_mod[0]) * sc_mat_arr[1] + tl_mod[1]) * sc_mat_arr[2] + tl_mod[2]
            R_tl = idx_to_t[idx_tl]

            phase_exact = np.exp(-2j * np.pi * R_tl @ q_full)
            phase_product = np.exp(-2j * np.pi * R_t @ q_full) * phase_l

            if abs(phase_exact) > 1e-15:
                ratio = phase_product / phase_exact
                err = abs(ratio - 1.0)
                max_err = max(max_err, err)

        print(f"  {l:>3d} {str(R_l):>12s} {phase_l.real:>+8.4f}{phase_l.imag:>+8.4f}i {max_err:>18.2e}")

    # ====================================================================
    # Self-energy comparison with shifted-FT kernel
    # ====================================================================
    print("\n" + "="*70)
    print("SELF-ENERGY COMPARISON: 4x4 mesh")
    print("="*70)

    nk = 4
    q_points = [(i/nk, j/nk) for i in range(nk) for j in range(nk)]
    n_kpts = len(q_points)
    q_diff_map = build_q_diff_map(nk, nk)

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

    T_all = []
    for qx, qy in q_points:
        T = build_gathering_matrix(prim_indices, cell_frac, (qx, qy), nat_prim, 'x')
        T_all.append(T)

    # Dense self-energy (ground truth)
    t0 = time.time()
    SL_dense, SG_dense, SR_dense = _compute_phph_self_energy_q_dense(
        G_lesser, G_greater, M_stacked, T_all, q_diff_map,
        nat_prim, n_kpts, freqs, dw,
    )
    t_dense = time.time() - t0

    # PCP factorized self-energy (current code)
    f_modes = np.zeros((n_kpts, 3, 24, n_dof), dtype=complex)
    for iq, (qx, qy) in enumerate(q_points):
        f_modes[iq] = fourier_transform_pcp(
            A_modes, lambdas, phonon, (qx, qy), 'x', info=pcp_info)

    t0 = time.time()
    SL_fact, SG_fact, SR_fact = _compute_phph_self_energy_pcp(
        G_lesser, G_greater, f_modes, lambdas,
        n_dof, n_kpts, freqs, dw, q_diff_map,
    )
    t_fact = time.time() - t0

    norm_R = np.linalg.norm(SR_dense)
    err_fact = np.linalg.norm(SR_fact - SR_dense) / norm_R
    print(f"\n  Dense SE time:      {t_dense:.2f}s")
    print(f"  Factorized SE time: {t_fact:.2f}s")
    print(f"  Factorized error:   {err_fact:.6e}")

    # Now test: build "shifted-FT" mode arrays and use the PCP kernel
    # The idea: instead of f_modes[iq, s, xi, :] = FT of A_s^xi,
    # we need f_modes_shifted[iq, s, xi, l, :] for each cell shift l.
    # Then the kernel sums over l with the appropriate external weights.

    # But the existing _compute_phph_self_energy_pcp doesn't support l-dependent
    # modes. Let's instead test at the vertex level and confirm the shifted-FT
    # is exact, which we already did above.

    # Alternative: for each l, build the l-specific modes and run the kernel
    # This is O(N_cells) times more expensive but should be exact.

    print("\n  Testing shifted-FT self-energy (sum over cell shifts)...")

    # Precompute shifted mode arrays for each l
    # For cell shift l: f_shifted[iq, s, xi, :] = T(q) @ supercell_vector(A_s^xi shifted by l)
    n_cells = pcp_info['n_cells']
    N_c = len(lambdas)
    n_super = len(phonon.supercell.masses)

    # Precompute: for each (l, s, xi, q), the shifted-FT vector
    # f_shifted_all[l, iq, s, xi, :] = T(q) @ v_{s,xi,l}
    print(f"  Precomputing shifted FT modes: {n_cells} cells x {n_kpts} q-points x 3 legs x {N_c} ranks...")
    t0 = time.time()

    f_shifted_all = np.zeros((n_cells, n_kpts, 3, N_c, n_dof), dtype=complex)
    for l in range(n_cells):
        for iq, (qx, qy) in enumerate(q_points):
            T_q = T_all[iq]  # (n_dof, dim_sc)
            for s in range(3):
                for xi in range(N_c):
                    # Build supercell vector
                    v = np.zeros(n_super * 3)
                    for j in range(n_super):
                        cj = cell_diff[sc_to_cell[j], l]
                        pj = sc_to_prim[j]
                        v[j*3:j*3+3] = A_modes[s, xi, cj, pj, :]
                    f_shifted_all[l, iq, s, xi, :] = T_q @ v

    t_precomp = time.time() - t0
    print(f"  Shifted-FT precomputation: {t_precomp:.2f}s")

    # Verify: at l=0, shifted-FT should match standard FT
    for iq in range(n_kpts):
        err_l0 = np.linalg.norm(f_shifted_all[0, iq] - f_modes[iq]) / np.linalg.norm(f_modes[iq])
        if err_l0 > 1e-10:
            print(f"  WARNING: l=0 shifted FT != standard FT at iq={iq}: err={err_l0:.2e}")

    # Verify: at commensurate q, shifted FT = phase * standard FT
    # q=(0,0) and q=(0.5,0.5) are commensurate with 2x2x2
    for iq in [0, 10]:  # (0,0) and (0.5,0.5) on 4x4 mesh
        qx, qy = q_points[iq]
        q_full = np.zeros(3)
        q_full[1] = qx  # perp_idx for x-transport: y,z
        q_full[2] = qy
        for l in range(n_cells):
            R_l = idx_to_t[l]
            phase_l = np.exp(-2j * np.pi * R_l @ q_full)
            f_predicted = phase_l * f_modes[iq]
            f_actual = f_shifted_all[l, iq]
            err = np.linalg.norm(f_actual - f_predicted) / np.linalg.norm(f_actual)
            if err > 1e-10 and iq == 0:
                print(f"  Commensurate q={q_points[iq]}, l={l}: phase shift err = {err:.2e}")

    # Check non-commensurate: q=(0.25,0)
    iq_test = 1  # (0, 0.25) on 4x4 mesh
    qx, qy = q_points[iq_test]
    q_full = np.zeros(3)
    q_full[1] = qx
    q_full[2] = qy
    print(f"\n  Non-commensurate q={q_points[iq_test]}:")
    for l in range(n_cells):
        R_l = idx_to_t[l]
        phase_l = np.exp(-2j * np.pi * R_l @ q_full)
        f_predicted = phase_l * f_modes[iq_test]
        f_actual = f_shifted_all[l, iq_test]
        err = np.linalg.norm(f_actual - f_predicted) / np.linalg.norm(f_actual)
        print(f"    l={l} R_l={R_l}: |f_shifted - phase*f_standard| / |f_shifted| = {err:.6e}")

    # ====================================================================
    # Compute shifted-FT self-energy
    # ====================================================================
    print("\n  Computing shifted-FT self-energy...")
    t0 = time.time()

    # The self-energy with shifted FT:
    # Sigma^<[a,b](q_ext, w) = pref * sum_{q'} sum_{xi,xi'} (lam*lam'/36)
    #   sum_{sigma,sigma'} sum_{l,l'}
    #   A_{s1}^xi((-l)%N, i_a, alpha_a) * conj(A_{s1'}^{xi'}((-l')%N, i_b, alpha_b))
    #   * IFFT{ g^{l,l'}_{s2,s2'}(q',w) * g^{l,l'}_{s3,s3'}(q-q',w) }
    #
    # where g^{l,l'}_{s,s'}(q,w) = f_{s,xi,l}(q)^H @ G(q,w) @ f_{s',xi',l'}(q)
    #
    # This is O(N_c^2 * N_cells^2 * ...) which is N_cells^2 times more expensive.
    # For N_cells=8, that's 64x overhead.

    # For the test, let's use a simplified version: direct vertex substitution
    # Sigma = sum_{q'} Phi_L @ K @ Phi_R where Phi is built from shifted FT

    n_low = max(0, int(np.round(freqs[0] / dw)))
    n_ext = n_low + n_freq
    n_fft = 2 * n_ext
    freq_sl = slice(n_low, n_low + n_freq)
    prefactor = 0.5j * HBAR_SI * dw / (2 * np.pi) / n_kpts

    # Pad and FFT Green's functions
    def _pad_fft(G_q):
        out = np.zeros((n_kpts, n_fft, n_dof, n_dof), dtype=complex)
        out[:, n_low:n_low + n_freq] = G_q
        return np.fft.fft(out, axis=1)

    GL_fft = _pad_fft(G_lesser)
    GG_fft = _pad_fft(G_greater)

    SR_shifted = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)
    SL_shifted = np.zeros_like(SR_shifted)
    SG_shifted = np.zeros_like(SR_shifted)

    # For each q_ext, q': build Phi_L and Phi_R using shifted FT, then contract
    for iq_ext in range(n_kpts):
        for iq_prime in range(n_kpts):
            iq_diff = q_diff_map[iq_ext, iq_prime]

            # Build Phi_L[a, c, d] using shifted-FT
            Phi_L = np.zeros((n_dof, n_dof, n_dof), dtype=complex)
            Phi_R_raw = np.zeros((n_dof, n_dof, n_dof), dtype=complex)

            for xi in range(N_c):
                for s1, s2, s3 in S3_PERMS:
                    for l in range(n_cells):
                        cell_ext = cell_diff[0, l]
                        ext_a = A_modes[s1, xi, cell_ext, :, :].flatten()  # (n_dof,)
                        fv = f_shifted_all[l, iq_prime, s2, xi, :]  # (n_dof,)
                        fw = f_shifted_all[l, iq_diff, s3, xi, :]   # (n_dof,)
                        Phi_L += (lambdas[xi] / 6.0) * ext_a[:, None, None] * fv[None, :, None] * fw[None, None, :]

                        # Phi_R: swap q' and q-q' roles, take conjugate
                        fv_r = f_shifted_all[l, iq_diff, s2, xi, :]
                        fw_r = f_shifted_all[l, iq_prime, s3, xi, :]
                        Phi_R_raw += (lambdas[xi] / 6.0) * ext_a[:, None, None] * fv_r[None, :, None] * fw_r[None, None, :]

            Phi_R = np.conj(Phi_R_raw)

            # Contract with K = IFFT(G(q') * G(q-q'))
            for G_fft, S_out in [(GL_fft, SL_shifted), (GG_fft, SG_shifted)]:
                product = (G_fft[iq_prime][:, :, None, :, None]
                           * G_fft[iq_diff][:, None, :, None, :])
                K = np.fft.ifft(product, axis=0)[freq_sl]
                temp = np.einsum('acd,wcdfe->wafe', Phi_L, K)
                S_out[iq_ext] += prefactor * np.einsum('wafe,bef->wab', temp, Phi_R)

    SR_shifted = 0.5 * (SG_shifted - SL_shifted)
    t_shifted = time.time() - t0

    err_shifted = np.linalg.norm(SR_shifted - SR_dense) / norm_R
    print(f"  Shifted-FT SE time:   {t_shifted:.2f}s")
    print(f"  Shifted-FT SE error:  {err_shifted:.6e}")

    # Per q-point breakdown
    print(f"\n  {'q-point':<16s} {'Factorized err':>15s} {'Shifted-FT err':>15s} {'Comm?':>6s}")
    print("  " + "-"*56)
    for iq in range(n_kpts):
        qx, qy = q_points[iq]
        comm = all(abs(q * 2 - round(q * 2)) < 1e-10 for q in (qx, qy))
        label = "YES" if comm else "NO"
        norm_q = np.linalg.norm(SR_dense[iq])
        if norm_q > 0:
            err_f = np.linalg.norm(SR_fact[iq] - SR_dense[iq]) / norm_q
            err_s = np.linalg.norm(SR_shifted[iq] - SR_dense[iq]) / norm_q
            print(f"  {str(q_points[iq]):<16s} {err_f:>15.2e} {err_s:>15.2e} {label:>6s}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    if err_shifted < 1e-10:
        print("  The shifted-FT approach gives EXACT results at ALL q-points.")
        print("  The error in the current factorized code is purely due to the")
        print("  phase-shift identity exp(-iq*R_{(t+l)%N}) = exp(-iq*R_t)*exp(-iq*R_l)")
        print("  which breaks at non-commensurate q.")
        print("")
        print("  SOLUTION: Use shifted-FT mode vectors instead of standard FT modes.")
        print("  Cost: O(N_cells) more FT precomputation, but the scalar projection")
        print("  structure is preserved, so the SCBA loop remains efficient.")
    else:
        print(f"  Shifted-FT error = {err_shifted:.2e} — investigate further.")


if __name__ == "__main__":
    test_vertex_comparison()
