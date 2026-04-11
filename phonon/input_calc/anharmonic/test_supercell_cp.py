"""Validate supercell CP decomposition and self-energy kernel.

Tests:
1. Fitting quality: supercell CP vs standard PCP at same rank
2. Self-energy accuracy: supercell CP kernel vs dense reference
   using FC3 reconstructed FROM the supercell CP modes (apples-to-apples)
3. Performance comparison
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
    _supercell_cp_forward_torch,
)
from phonon_inputs.separable import (
    build_supercell_mapping, build_realspace_fc3_matrices,
    build_gathering_matrix, build_q_diff_map,
)
from phonon_inputs.anharmonic import _compute_phph_self_energy_q_dense
from phonon_inputs.constants import CONVERSION_FC3_THZ, HBAR_SI
import h5py
import torch


def reconstruct_fc3_from_supercell_cp(u_modes, lambdas, phonon, info):
    """Reconstruct raw FC3 from supercell CP modes."""
    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)
    masses = phonon.supercell.masses
    p2s = phonon.primitive.p2s_map.astype(np.int64)
    N_c = len(lambdas)
    target_norm = info['target_norm']

    with torch.no_grad():
        u_t = torch.tensor(u_modes, dtype=torch.float64)
        lam_t = torch.tensor(lambdas / target_norm, dtype=torch.float64)
        p2s_t = torch.tensor(p2s, dtype=torch.long)
        fc3_mw = _supercell_cp_forward_torch(
            u_t, lam_t, p2s_t, nat_prim, n_super, N_c,
        ).numpy() * target_norm

    # Un-mass-weight
    fc3_recon = np.zeros_like(fc3_mw)
    for i_prim in range(nat_prim):
        m_i = masses[int(p2s[i_prim])]
        mass_jk = np.sqrt(m_i * masses[:, None] * masses[None, :])
        fc3_recon[i_prim] = fc3_mw[i_prim] * mass_jk[:, :, None, None, None] / CONVERSION_FC3_THZ

    return fc3_recon


def main():
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    nat_prim = len(phonon.primitive.masses)
    n_dof = nat_prim * 3
    n_super = len(phonon.supercell.masses)

    # ---- Step 1: Fitting comparison ----
    print("=" * 60)
    print("Step 1: Fitting quality comparison")
    print("=" * 60)

    for N_c in [4, 8, 24]:
        print(f"\n--- N_c = {N_c} ---")
        _, _, info_pcp = fit_pcp(fc3_raw, phonon, N_c=N_c, max_iter=2000, verbose=False)
        _, _, info_sc = fit_supercell_cp(fc3_raw, phonon, N_c=N_c, max_iter=2000, verbose=False)
        print(f"  PCP rel_err:          {info_pcp['rel_err']:.6e}")
        print(f"  Supercell CP rel_err: {info_sc['rel_err']:.6e}")

    # ---- Step 2: Self-energy validation (apples-to-apples) ----
    print("\n" + "=" * 60)
    print("Step 2: Self-energy kernel validation")
    print("=" * 60)

    # Use N_c=24 supercell CP modes
    print("\nFitting supercell CP (N_c=24)...")
    u_modes, lambdas_sc, info_sc = fit_supercell_cp(
        fc3_raw, phonon, N_c=24, max_iter=2000, verbose=False)
    print(f"  rel_err = {info_sc['rel_err']:.6e}")

    # Reconstruct FC3 from SAME supercell CP modes (apples-to-apples)
    print("Reconstructing FC3 from supercell CP modes...")
    fc3_recon = reconstruct_fc3_from_supercell_cp(u_modes, lambdas_sc, phonon, info_sc)

    # Build dense infrastructure from the SAME FC3
    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(phonon)
    masses_super = phonon.supercell.masses
    M_stacked = build_realspace_fc3_matrices(fc3_recon, nat_prim, masses_super, ref_sc_atoms)

    # 4x4 q-mesh (non-commensurate with 2x2x2 supercell)
    nk = 4
    q_points = [(i/nk, j/nk) for i in range(nk) for j in range(nk)]
    n_kpts = len(q_points)
    q_diff_map = build_q_diff_map(nk, nk)

    T_all = []
    for qx, qy in q_points:
        T = build_gathering_matrix(prim_indices, cell_frac, (qx, qy), nat_prim, 'x')
        T_all.append(T)

    # FT supercell CP modes
    print("FT supercell CP modes...")
    f_modes_sc, f_ext = fourier_transform_supercell_cp(
        u_modes, lambdas_sc, phonon, q_points, transport_direction='x')
    print(f"  f_ext shape: {f_ext.shape}")

    # Generate test Green's functions
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

    # Dense self-energy (using FC3 from supercell CP reconstruction)
    print("\nComputing dense self-energy (reference from same FC3)...")
    t0 = time.time()
    SL_dense, SG_dense, SR_dense = _compute_phph_self_energy_q_dense(
        G_lesser, G_greater, M_stacked, T_all, q_diff_map,
        nat_prim, n_kpts, freqs, dw,
    )
    t_dense = time.time() - t0
    print(f"  Time: {t_dense:.2f}s")

    # Supercell CP self-energy (using same modes)
    print("Computing supercell CP self-energy...")
    t0 = time.time()
    SL_sc, SG_sc, SR_sc = _compute_phph_self_energy_pcp(
        G_lesser, G_greater,
        f_modes_sc, lambdas_sc,
        n_dof, n_kpts, freqs, dw,
        q_diff_map=q_diff_map,
        f_ext=f_ext,
    )
    t_sc = time.time() - t0
    print(f"  Time: {t_sc:.2f}s")

    # Compare
    norm_R = np.linalg.norm(SR_dense)
    norm_L = np.linalg.norm(SL_dense)
    norm_G = np.linalg.norm(SG_dense)

    err_R = np.linalg.norm(SR_sc - SR_dense) / norm_R
    err_L = np.linalg.norm(SL_sc - SL_dense) / norm_L
    err_G = np.linalg.norm(SG_sc - SG_dense) / norm_G

    print(f"\n{'='*60}")
    print(f"RESULTS (same FC3, N_c=24)")
    print(f"{'='*60}")
    print(f"  Dense time:        {t_dense:.2f}s")
    print(f"  Supercell CP time: {t_sc:.2f}s")
    print(f"  Speedup:           {t_dense/t_sc:.2f}x")
    print(f"")
    print(f"  Sigma^R error:     {err_R:.6e}")
    print(f"  Sigma^< error:     {err_L:.6e}")
    print(f"  Sigma^> error:     {err_G:.6e}")

    # Per q-point breakdown
    print(f"\n  {'q-point':<16s} {'|Sigma^R|':>10s} {'rel error':>12s}")
    print("  " + "-" * 42)
    for iq in range(n_kpts):
        norm_q = np.linalg.norm(SR_dense[iq])
        if norm_q > 0:
            err_q = np.linalg.norm(SR_sc[iq] - SR_dense[iq]) / norm_q
            print(f"  {str(q_points[iq]):<16s} {norm_q:>10.4e} {err_q:>12.2e}")

    print(f"\n{'='*60}")
    if err_R < 1e-10:
        print(f"PASS: Kernel matches dense reference (machine precision)")
    elif err_R < 1e-6:
        print(f"PASS: Kernel matches dense reference (err={err_R:.2e})")
    else:
        print(f"FAIL: Error {err_R:.2e} exceeds threshold")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
