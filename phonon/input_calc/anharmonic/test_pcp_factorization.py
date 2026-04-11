"""Test whether PCP factorized self-energy is mathematically exact.

The self-energy contraction is:
  Sigma[a,b] = sum_{c,d,e,f} Phi_L[a,c,d] K[c,d,f,e] Phi_R[b,e,f]

where K[c,d,f,e] = IFFT{ G(q')[c,f] * G(q-q')[d,e] }

Key index pairing:
  G(q')   carries indices (c, f)
  G(q-q') carries indices (d, e)

For PCP vertex:
  Phi_L[a,c,d] = sum_xi lam_xi/6 sum_sigma f_{s1}(q)_a f_{s2}(q')_c f_{s3}(q-q')_d
  Phi_R[b,e,f] = conj of Phi(q-q',q')
               = sum_xi' lam_{xi'}/6 sum_sigma' f*_{s1'}(q)_b f*_{s2'}(q-q')_e f*_{s3'}(q')_f

Substituting and grouping by G index pairs:
  (c, f) both at q': f_{s2}(q')_c and f*_{s3'}(q')_f -> g_{s2,s3'}(q')
  (d, e) both at q-q': f_{s3}(q-q')_d and f*_{s2'}(q-q')_e -> g_{s3,s2'}(q-q')

So the factorized PCP self-energy uses scalar projections g = f^H G f,
and the question is: does this factorization give the SAME result as dense?

Test 1: Vertex comparison at commensurate vs non-commensurate q-points
Test 2: Self-energy comparison (factorized PCP vs dense) at commensurate q
Test 3: Self-energy comparison at non-commensurate q (4x4 mesh on 2x2x2 SC)
"""

import sys
from pathlib import Path
import numpy as np

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs.pcp import (
    fit_pcp, fourier_transform_pcp, _compute_phph_self_energy_pcp,
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
    """Dense vertex Phi(q1, q2) via T(q1) @ M_a @ T(q2)^T."""
    dim_t = M_stacked.shape[1]
    Phi = np.zeros((n_dof, n_dof, n_dof), dtype=complex)
    for a in range(n_dof):
        Phi[a] = T_q1 @ M_stacked[a*dim_t:(a+1)*dim_t, :] @ T_q2.T
    return Phi


def build_pcp_vertex(f_modes_all_q, lambdas, iq_ext, iq1, iq2):
    """PCP vertex from FT modes: Phi(q1,q2)[a,c,d]."""
    N_c = len(lambdas)
    n_dof = f_modes_all_q.shape[-1]
    V = np.zeros((n_dof, n_dof, n_dof), dtype=complex)
    for xi in range(N_c):
        for s1, s2, s3 in S3_PERMS:
            fa = f_modes_all_q[iq_ext, s1, xi, :]  # (n_dof,)
            fc = f_modes_all_q[iq1, s2, xi, :]
            fd = f_modes_all_q[iq2, s3, xi, :]
            V += (lambdas[xi] / 6.0) * fa[:, None, None] * fc[None, :, None] * fd[None, None, :]
    return V


def main():
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    nat_prim = len(phonon.primitive.masses)
    n_dof = nat_prim * 3
    masses_super = phonon.supercell.masses

    # Fit PCP
    print("Fitting PCP N_c=24...")
    A_modes, lambdas, pcp_info = fit_pcp(fc3_raw, phonon, N_c=24, max_iter=2000, verbose=False)
    print(f"  rel_err = {pcp_info['rel_err']:.6e}")

    # Build dense infrastructure from RAW FC3
    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(phonon)
    M_stacked_raw = build_realspace_fc3_matrices(fc3_raw, nat_prim, masses_super, ref_sc_atoms)

    # Build dense infrastructure from RECONSTRUCTED FC3
    fc3_recon = reconstruct_fc3_from_pcp(A_modes, lambdas, phonon, pcp_info)
    M_stacked_pcp = build_realspace_fc3_matrices(fc3_recon, nat_prim, masses_super, ref_sc_atoms)

    recon_err = np.linalg.norm(M_stacked_pcp - M_stacked_raw) / np.linalg.norm(M_stacked_raw)
    print(f"  M_stacked reconstruction error: {recon_err:.6e}")

    # ====================================================================
    # TEST 1: Vertex comparison at commensurate vs non-commensurate q
    # ====================================================================
    print("\n" + "="*70)
    print("TEST 1: Vertex comparison (PCP FT modes vs dense FT)")
    print("="*70)

    # 2x2 mesh: all commensurate with 2x2x2 supercell
    # 4x4 mesh: half non-commensurate
    for nk, label in [(2, "2x2 (all commensurate)"), (4, "4x4 (mixed)")]:
        print(f"\n  q-mesh: {label}")
        q_points = [(i/nk, j/nk) for i in range(nk) for j in range(nk)]
        q_diff_map = build_q_diff_map(nk, nk)

        # PCP FT modes
        f_modes = np.zeros((len(q_points), 3, 24, n_dof), dtype=complex)
        for iq, (qx, qy) in enumerate(q_points):
            f_modes[iq] = fourier_transform_pcp(
                A_modes, lambdas, phonon, (qx, qy), 'x', info=pcp_info)

        # Dense gathering matrices
        T_all = []
        for qx, qy in q_points:
            T = build_gathering_matrix(prim_indices, cell_frac, (qx, qy), nat_prim, 'x')
            T_all.append(T)

        # Compare vertices at a few q-point pairs
        test_pairs = [(0, 1), (1, 2)]
        if nk == 4:
            test_pairs += [(3, 5), (7, 3)]  # likely non-commensurate

        for iq_ext, iq_prime in test_pairs:
            iq_diff = q_diff_map[iq_ext, iq_prime]
            qext = q_points[iq_ext]
            qp = q_points[iq_prime]
            qd = q_points[iq_diff]

            # Is q' commensurate with 2x2x2 supercell?
            comm_p = all(abs(q*2 - round(q*2)) < 1e-10 for q in qp)
            comm_d = all(abs(q*2 - round(q*2)) < 1e-10 for q in qd)
            comm_str = "COMM" if (comm_p and comm_d) else "NON-COMM"

            # Dense vertex from raw FC3
            V_dense = build_dense_vertex(M_stacked_raw, T_all[iq_prime], T_all[iq_diff], n_dof)
            # Dense vertex from PCP-reconstructed FC3
            V_recon = build_dense_vertex(M_stacked_pcp, T_all[iq_prime], T_all[iq_diff], n_dof)
            # PCP factorized vertex
            V_pcp = build_pcp_vertex(f_modes, lambdas, iq_ext, iq_prime, iq_diff)

            norm_raw = np.linalg.norm(V_dense)
            err_recon = np.linalg.norm(V_recon - V_dense) / norm_raw
            err_pcp = np.linalg.norm(V_pcp - V_dense) / norm_raw
            err_pcp_vs_recon = np.linalg.norm(V_pcp - V_recon) / np.linalg.norm(V_recon)

            print(f"    q_ext={qext}, q'={qp}, q-q'={qd}  [{comm_str}]")
            print(f"      V_recon vs V_raw:     {err_recon:.6e}")
            print(f"      V_pcp   vs V_raw:     {err_pcp:.6e}")
            print(f"      V_pcp   vs V_recon:   {err_pcp_vs_recon:.6e}")

    # ====================================================================
    # TEST 2: Self-energy at commensurate q (2x2 mesh)
    # ====================================================================
    print("\n" + "="*70)
    print("TEST 2: Self-energy comparison (2x2 mesh, all commensurate)")
    print("="*70)

    nk = 2
    q_points = [(i/nk, j/nk) for i in range(nk) for j in range(nk)]
    n_kpts = len(q_points)
    q_diff_map = build_q_diff_map(nk, nk)

    # Frequency grid
    n_freq = 31
    freqs = np.linspace(1.0, 14.0, n_freq)
    dw = freqs[1] - freqs[0]

    # Build fake Green's functions (random, for structural test)
    rng = np.random.default_rng(42)
    G_lesser = rng.standard_normal((n_kpts, n_freq, n_dof, n_dof)) + \
               1j * rng.standard_normal((n_kpts, n_freq, n_dof, n_dof))
    G_greater = rng.standard_normal((n_kpts, n_freq, n_dof, n_dof)) + \
                1j * rng.standard_normal((n_kpts, n_freq, n_dof, n_dof))
    # Make Hermitian-like for physical consistency
    G_lesser = 0.5 * (G_lesser - np.conj(G_lesser.transpose(0, 1, 3, 2)))
    G_greater = 0.5 * (G_greater - np.conj(G_greater.transpose(0, 1, 3, 2)))

    # Dense self-energy (from PCP-reconstructed FC3)
    T_all = []
    for qx, qy in q_points:
        T = build_gathering_matrix(prim_indices, cell_frac, (qx, qy), nat_prim, 'x')
        T_all.append(T)

    SL_dense, SG_dense, SR_dense = _compute_phph_self_energy_q_dense(
        G_lesser, G_greater, M_stacked_pcp, T_all, q_diff_map,
        nat_prim, n_kpts, freqs, dw,
    )

    # PCP factorized self-energy
    f_modes = np.zeros((n_kpts, 3, 24, n_dof), dtype=complex)
    for iq, (qx, qy) in enumerate(q_points):
        f_modes[iq] = fourier_transform_pcp(
            A_modes, lambdas, phonon, (qx, qy), 'x', info=pcp_info)

    SL_pcp, SG_pcp, SR_pcp = _compute_phph_self_energy_pcp(
        G_lesser, G_greater, f_modes, lambdas,
        n_dof, n_kpts, freqs, dw, q_diff_map,
    )

    norm_L = np.linalg.norm(SL_dense)
    norm_G = np.linalg.norm(SG_dense)
    norm_R = np.linalg.norm(SR_dense)

    err_L = np.linalg.norm(SL_pcp - SL_dense) / norm_L
    err_G = np.linalg.norm(SG_pcp - SG_dense) / norm_G
    err_R = np.linalg.norm(SR_pcp - SR_dense) / norm_R

    print(f"  Sigma_lesser  error: {err_L:.6e}")
    print(f"  Sigma_greater error: {err_G:.6e}")
    print(f"  Sigma_R       error: {err_R:.6e}")

    # Per q-point breakdown
    for iq in range(n_kpts):
        norm_q = np.linalg.norm(SR_dense[iq])
        if norm_q > 0:
            err_q = np.linalg.norm(SR_pcp[iq] - SR_dense[iq]) / norm_q
            print(f"    q={q_points[iq]}: |SR_pcp - SR_dense|/|SR_dense| = {err_q:.6e}")

    # ====================================================================
    # TEST 3: Self-energy at 4x4 mesh (mixed commensurate/non-commensurate)
    # ====================================================================
    print("\n" + "="*70)
    print("TEST 3: Self-energy comparison (4x4 mesh, mixed commensurability)")
    print("="*70)

    nk = 4
    q_points_4 = [(i/nk, j/nk) for i in range(nk) for j in range(nk)]
    n_kpts_4 = len(q_points_4)
    q_diff_map_4 = build_q_diff_map(nk, nk)

    G_lesser_4 = rng.standard_normal((n_kpts_4, n_freq, n_dof, n_dof)) + \
                 1j * rng.standard_normal((n_kpts_4, n_freq, n_dof, n_dof))
    G_greater_4 = rng.standard_normal((n_kpts_4, n_freq, n_dof, n_dof)) + \
                  1j * rng.standard_normal((n_kpts_4, n_freq, n_dof, n_dof))
    G_lesser_4 = 0.5 * (G_lesser_4 - np.conj(G_lesser_4.transpose(0, 1, 3, 2)))
    G_greater_4 = 0.5 * (G_greater_4 - np.conj(G_greater_4.transpose(0, 1, 3, 2)))

    T_all_4 = []
    for qx, qy in q_points_4:
        T = build_gathering_matrix(prim_indices, cell_frac, (qx, qy), nat_prim, 'x')
        T_all_4.append(T)

    SL_dense_4, SG_dense_4, SR_dense_4 = _compute_phph_self_energy_q_dense(
        G_lesser_4, G_greater_4, M_stacked_pcp, T_all_4, q_diff_map_4,
        nat_prim, n_kpts_4, freqs, dw,
    )

    f_modes_4 = np.zeros((n_kpts_4, 3, 24, n_dof), dtype=complex)
    for iq, (qx, qy) in enumerate(q_points_4):
        f_modes_4[iq] = fourier_transform_pcp(
            A_modes, lambdas, phonon, (qx, qy), 'x', info=pcp_info)

    SL_pcp_4, SG_pcp_4, SR_pcp_4 = _compute_phph_self_energy_pcp(
        G_lesser_4, G_greater_4, f_modes_4, lambdas,
        n_dof, n_kpts_4, freqs, dw, q_diff_map_4,
    )

    norm_R_4 = np.linalg.norm(SR_dense_4)
    err_R_4 = np.linalg.norm(SR_pcp_4 - SR_dense_4) / norm_R_4
    print(f"  Overall Sigma_R error: {err_R_4:.6e}")

    # Per q-point breakdown, separating commensurate and non-commensurate
    for iq in range(n_kpts_4):
        qx, qy = q_points_4[iq]
        comm = all(abs(q*2 - round(q*2)) < 1e-10 for q in (qx, qy))
        label = "COMM" if comm else "NON-COMM"
        norm_q = np.linalg.norm(SR_dense_4[iq])
        if norm_q > 0:
            err_q = np.linalg.norm(SR_pcp_4[iq] - SR_dense_4[iq]) / norm_q
            print(f"    q={str(q_points_4[iq]):12s} [{label:8s}]: {err_q:.6e}")

    # ====================================================================
    # TEST 4: Permutation index pairing verification
    # ====================================================================
    print("\n" + "="*70)
    print("TEST 4: Verify permutation index pairing")
    print("="*70)
    print("  Testing whether g[s2,s2']*g[s3,s3'] == g[s2,s3']*g[s3,s2']")
    print("  after summing over all permutation pairs...")

    # Use 2x2 mesh data
    n_fft = 62  # from padding
    g_test = rng.standard_normal((24, 24, 3, 3, n_kpts, n_fft)) + \
             1j * rng.standard_normal((24, 24, 3, 3, n_kpts, n_fft))

    iq_p, iq_d = 1, 2

    # Method A: code's pairing g[s2,s2'] * g[s3,s3']
    sum_A = np.zeros((24, 24, 3, 3, n_fft), dtype=complex)
    for s1, s2, s3 in S3_PERMS:
        for s1p, s2p, s3p in S3_PERMS:
            sum_A[:, :, s1, s1p, :] += (
                g_test[:, :, s2, s2p, iq_p, :] *
                g_test[:, :, s3, s3p, iq_d, :]
            )

    # Method B: derived pairing g[s2,s3'] * g[s3,s2']
    sum_B = np.zeros((24, 24, 3, 3, n_fft), dtype=complex)
    for s1, s2, s3 in S3_PERMS:
        for s1p, s2p, s3p in S3_PERMS:
            sum_B[:, :, s1, s1p, :] += (
                g_test[:, :, s2, s3p, iq_p, :] *
                g_test[:, :, s3, s2p, iq_d, :]
            )

    diff = np.linalg.norm(sum_A - sum_B) / np.linalg.norm(sum_A)
    print(f"  |sum_A - sum_B| / |sum_A| = {diff:.6e}")
    if diff < 1e-14:
        print("  -> Pairings are EQUIVALENT (sums commute over S3 x S3)")
    else:
        print("  -> Pairings DIFFER")


if __name__ == "__main__":
    main()
