"""Diagnose PCP self-energy: vertex and FT correctness.

Tests:
1. PCP vertex from FT modes vs full vertex from raw FC3
2. PCP vertex from FT modes vs full vertex from reconstructed FC3
   (isolates whether error is in FT or reconstruction)
3. Verify that vertex derivation holds: V = sum f(q_ext) f(q') f(q-q')
"""

import sys
from pathlib import Path

import h5py
import numpy as np

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs.pcp import (
    build_cell_mapping, build_cell_diff_table, _build_index_tables,
    _pcp_forward_torch, _build_target, fit_pcp,
    fourier_transform_pcp,
    CONVERSION_FC3_THZ, S3_PERMS,
)
from phonon_inputs.separable import build_q_diff_map
import torch


def build_full_vertex(fc3_raw, phonon, q_prime, q_double_prime,
                      transport_direction='x'):
    """Full FT vertex from FC3 (compact or non-compact)."""
    prim = phonon.primitive
    sc = phonon.supercell
    nat_prim = len(prim.masses)
    n_super = len(sc.masses)
    n_dof = nat_prim * 3
    p2s = prim.p2s_map
    is_compact = fc3_raw.shape[0] == nat_prim

    inv_cell = np.linalg.inv(prim.cell.T)
    sc_to_prim = np.zeros(n_super, dtype=int)
    cell_frac = np.zeros((n_super, 3))
    for s in range(n_super):
        for a in range(nat_prim):
            diff = sc.positions[s] - prim.positions[a]
            frac = inv_cell @ diff
            R_int = np.round(frac).astype(int)
            if np.linalg.norm(diff - prim.cell.T @ R_int) < 0.1:
                sc_to_prim[s] = a
                cell_frac[s] = R_int
                break

    tidx = "xyz".index(transport_direction)
    perp_idx = [i for i in range(3) if i != tidx]

    q_full_p = np.zeros(3)
    q_full_p[perp_idx[0]] = q_prime[0]
    q_full_p[perp_idx[1]] = q_prime[1]

    q_full_d = np.zeros(3)
    q_full_d[perp_idx[0]] = q_double_prime[0]
    q_full_d[perp_idx[1]] = q_double_prime[1]

    phases_p = np.exp(-2j * np.pi * cell_frac @ q_full_p)
    phases_d = np.exp(-2j * np.pi * cell_frac @ q_full_d)

    V = np.zeros((n_dof, n_dof, n_dof), dtype=complex)
    for i_prim in range(nat_prim):
        fc3_idx = i_prim if is_compact else int(p2s[i_prim])
        m_i = sc.masses[int(p2s[i_prim])]
        for j in range(n_super):
            kappa_j = sc_to_prim[j]
            m_j = sc.masses[j]
            for k in range(n_super):
                kappa_k = sc_to_prim[k]
                m_k = sc.masses[k]
                mf = np.sqrt(m_i * m_j * m_k)
                ph = phases_p[j] * phases_d[k]
                for a1 in range(3):
                    a = i_prim * 3 + a1
                    for a2 in range(3):
                        c = kappa_j * 3 + a2
                        for a3 in range(3):
                            d = kappa_k * 3 + a3
                            V[a, c, d] += ph * fc3_raw[fc3_idx, j, k, a1, a2, a3] / mf * CONVERSION_FC3_THZ

    return V


def build_pcp_vertex(f_modes_all_q, lambdas, iq_ext, iq_prime, iq_diff, conj_leg1=True):
    """Build PCP vertex from FT modes."""
    N_c = len(lambdas)
    n_dof = f_modes_all_q.shape[-1]
    V = np.zeros((n_dof, n_dof, n_dof), dtype=complex)
    for xi in range(N_c):
        for s1, s2, s3 in S3_PERMS:
            f1 = np.conj(f_modes_all_q[iq_ext, s1, xi, :]) if conj_leg1 else f_modes_all_q[iq_ext, s1, xi, :]
            f2 = f_modes_all_q[iq_prime, s2, xi, :]
            f3 = f_modes_all_q[iq_diff, s3, xi, :]
            V += (lambdas[xi] / 6.0) * f1[:, None, None] * f2[None, :, None] * f3[None, None, :]
    return V


def main():
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)
    n_dof = nat_prim * 3

    # Fit PCP
    print("Fitting PCP N_c=24...")
    A_modes, lambdas, pcp_info = fit_pcp(fc3_raw, phonon, N_c=24, max_iter=2000, verbose=False)
    print(f"  rel_err = {pcp_info['rel_err']:.6e}")

    # Reconstruct FC3 from PCP in real space
    print("Reconstructing FC3 from PCP...")
    sc_to_cell, sc_to_prim_np, _, n_cells, sc_mat = build_cell_mapping(phonon)
    cell_diff, idx_to_t = build_cell_diff_table(n_cells, sc_mat)
    idx1_cell, idx_jk_cell, sc_to_prim_i = _build_index_tables(
        sc_to_cell, sc_to_prim_np, cell_diff, nat_prim, n_super, n_cells)

    target_norm = pcp_info['target_norm']
    with torch.no_grad():
        A_t = torch.tensor(A_modes, dtype=torch.float64)
        lam_t = torch.tensor(lambdas / target_norm, dtype=torch.float64)
        fc3_pcp = _pcp_forward_torch(
            A_t, lam_t,
            torch.tensor(idx1_cell, dtype=torch.long),
            torch.tensor(idx_jk_cell, dtype=torch.long),
            torch.tensor(sc_to_prim_i, dtype=torch.long),
            nat_prim, n_super, n_cells, 24,
        ).numpy() * target_norm

    # Verify reconstruction matches target
    target = _build_target(fc3_raw, phonon)
    recon_err = np.linalg.norm(fc3_pcp - target) / np.linalg.norm(target)
    print(f"  Reconstruction error: {recon_err:.6e}")

    # Setup q-points
    nkx, nky = 4, 4
    q_points = [(i/nkx, j/nky) for i in range(nkx) for j in range(nky)]
    q_diff_map = build_q_diff_map(nkx, nky)

    # PCP FT modes
    N_c = len(lambdas)
    f_modes_all_q = np.zeros((len(q_points), 3, N_c, n_dof), dtype=complex)
    for iq, (qx, qy) in enumerate(q_points):
        f_modes_all_q[iq] = fourier_transform_pcp(
            A_modes, lambdas, phonon, (qx, qy), 'x', info=pcp_info)

    # ---- TEST 1: Vertex comparison at several q-points ----
    print("\n=== TEST 1: Vertex comparison ===")

    test_cases = [(3, 5), (0, 0), (0, 7), (7, 3), (1, 1)]
    for iq_ext, iq_prime in test_cases:
        iq_diff = q_diff_map[iq_ext, iq_prime]
        q_ext = q_points[iq_ext]
        q_p = q_points[iq_prime]
        q_d = q_points[iq_diff]

        # Full vertex from RAW FC3
        V_raw = build_full_vertex(fc3_raw, phonon, q_p, q_d)

        # Full vertex from RECONSTRUCTED FC3
        V_recon = build_full_vertex_from_target(fc3_pcp, phonon, q_p, q_d)

        # PCP vertex from FT modes
        V_pcp_conj = build_pcp_vertex(f_modes_all_q, lambdas, iq_ext, iq_prime, iq_diff, conj_leg1=True)
        V_pcp_plain = build_pcp_vertex(f_modes_all_q, lambdas, iq_ext, iq_prime, iq_diff, conj_leg1=False)

        norm_raw = np.linalg.norm(V_raw)
        err_recon = np.linalg.norm(V_recon - V_raw) / norm_raw
        err_pcp_conj = np.linalg.norm(V_pcp_conj - V_raw) / norm_raw
        err_pcp_plain = np.linalg.norm(V_pcp_plain - V_raw) / norm_raw
        err_pcp_conj_recon = np.linalg.norm(V_pcp_conj - V_recon) / np.linalg.norm(V_recon) if np.linalg.norm(V_recon) > 0 else float('inf')
        err_pcp_plain_recon = np.linalg.norm(V_pcp_plain - V_recon) / np.linalg.norm(V_recon) if np.linalg.norm(V_recon) > 0 else float('inf')

        print(f"  q_ext={q_ext}, q'={q_p}, q-q'={q_d}")
        print(f"    V_recon vs V_raw:      {err_recon:.6e}")
        print(f"    V_pcp(conj)  vs V_raw: {err_pcp_conj:.6e}")
        print(f"    V_pcp(plain) vs V_raw: {err_pcp_plain:.6e}")
        print(f"    V_pcp(conj)  vs V_recon: {err_pcp_conj_recon:.6e}")
        print(f"    V_pcp(plain) vs V_recon: {err_pcp_plain_recon:.6e}")

    # ---- TEST 2: Detailed FT check ----
    print("\n=== TEST 2: FT consistency check ===")
    # Manually FT the reconstructed FC3 and compare to f-mode vertex
    iq_ext, iq_prime = 3, 5
    iq_diff = q_diff_map[iq_ext, iq_prime]
    q_ext = q_points[iq_ext]
    q_p = q_points[iq_prime]
    q_d = q_points[iq_diff]

    # The PCP FC3 is: fc3_pcp[b1, j, k, a1, a2, a3] (mass-weighted, in THz units)
    # But build_full_vertex applies mass-weighting again from fc3_raw!
    # This is wrong for the reconstructed case -- it's already mass-weighted.

    # Let me check: _build_target divides by masses and multiplies by CONVERSION_FC3_THZ.
    # fc3_pcp = forward(A, lam) * target_norm, and target = fc3_raw / masses * conv.
    # So fc3_pcp is ALREADY mass-weighted. The vertex from fc3_pcp should NOT apply masses again.
    print(f"  fc3_pcp is already mass-weighted? Let me verify...")
    print(f"  target[0,0,0,:,:,:] sample: {target[0,0,0,0,0,0]:.4e}")
    print(f"  fc3_pcp[0,0,0,:,:,:] sample: {fc3_pcp[0,0,0,0,0,0]:.4e}")
    print(f"  fc3_raw[0,0,0,:,:,:] sample: {fc3_raw[0,0,0,0,0,0]:.4e}")
    m0 = phonon.supercell.masses[0]
    manual_mw = fc3_raw[0,0,0,0,0,0] / (m0 * np.sqrt(m0)) * CONVERSION_FC3_THZ
    print(f"  Manual mass-weight of fc3_raw[0,0,0,0,0,0]: {manual_mw:.4e}")
    print(f"  (fc3_pcp is the mass-weighted target from _build_target)")


def build_full_vertex_from_target(fc3_mw, phonon, q_prime, q_double_prime,
                                   transport_direction='x'):
    """Full FT vertex from already-mass-weighted FC3 in compact format."""
    prim = phonon.primitive
    sc = phonon.supercell
    nat_prim = len(prim.masses)
    n_super = len(sc.masses)
    n_dof = nat_prim * 3

    inv_cell = np.linalg.inv(prim.cell.T)
    sc_to_prim = np.zeros(n_super, dtype=int)
    cell_frac = np.zeros((n_super, 3))
    for s in range(n_super):
        for a in range(nat_prim):
            diff = sc.positions[s] - prim.positions[a]
            frac = inv_cell @ diff
            R_int = np.round(frac).astype(int)
            if np.linalg.norm(diff - prim.cell.T @ R_int) < 0.1:
                sc_to_prim[s] = a
                cell_frac[s] = R_int
                break

    tidx = "xyz".index(transport_direction)
    perp_idx = [i for i in range(3) if i != tidx]

    q_full_p = np.zeros(3)
    q_full_p[perp_idx[0]] = q_prime[0]
    q_full_p[perp_idx[1]] = q_prime[1]

    q_full_d = np.zeros(3)
    q_full_d[perp_idx[0]] = q_double_prime[0]
    q_full_d[perp_idx[1]] = q_double_prime[1]

    phases_p = np.exp(-2j * np.pi * cell_frac @ q_full_p)
    phases_d = np.exp(-2j * np.pi * cell_frac @ q_full_d)

    V = np.zeros((n_dof, n_dof, n_dof), dtype=complex)
    for i_prim in range(nat_prim):
        for j in range(n_super):
            kappa_j = sc_to_prim[j]
            for k in range(n_super):
                kappa_k = sc_to_prim[k]
                ph = phases_p[j] * phases_d[k]
                for a1 in range(3):
                    a = i_prim * 3 + a1
                    for a2 in range(3):
                        c = kappa_j * 3 + a2
                        for a3 in range(3):
                            d = kappa_k * 3 + a3
                            # Already mass-weighted, no mass factor needed
                            V[a, c, d] += ph * fc3_mw[i_prim, j, k, a1, a2, a3]

    return V


if __name__ == "__main__":
    main()
