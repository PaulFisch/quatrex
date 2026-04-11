"""Validate the shifted-FT PCP self-energy kernel against the dense reference.

Tests _compute_phph_self_energy_pcp_shifted by comparing its output to
_compute_phph_self_energy_q_dense (using PCP-reconstructed FC3) on a 4x4
q-mesh (non-commensurate with 2x2x2 supercell).
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
    fit_pcp, fourier_transform_pcp_shifted,
    _compute_phph_self_energy_pcp_shifted,
    reconstruct_fc3_from_pcp,
)
from phonon_inputs.separable import (
    build_supercell_mapping, build_realspace_fc3_matrices,
    build_gathering_matrix, build_q_diff_map,
)
from phonon_inputs.anharmonic import _compute_phph_self_energy_q_dense
from phonon_inputs.constants import HBAR_SI
import h5py


def main():
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    nat_prim = len(phonon.primitive.masses)
    n_dof = nat_prim * 3

    # Fit PCP at N_c=4 (the practical compression level)
    print("Fitting PCP N_c=4...")
    A_modes, lambdas, pcp_info = fit_pcp(fc3_raw, phonon, N_c=4,
                                          max_iter=2000, verbose=False)
    print(f"  rel_err = {pcp_info['rel_err']:.6e}")

    # Dense infrastructure from PCP-reconstructed FC3
    fc3_recon = reconstruct_fc3_from_pcp(A_modes, lambdas, phonon, pcp_info)
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

    # Precompute shifted-FT modes
    print("Precomputing shifted-FT modes...")
    t0 = time.time()
    f_shifted, ext_weights = fourier_transform_pcp_shifted(
        A_modes, lambdas, phonon, q_points,
        transport_direction='x', info=pcp_info,
    )
    t_precomp = time.time() - t0
    print(f"  Time: {t_precomp:.2f}s")
    print(f"  f_shifted shape: {f_shifted.shape}")
    print(f"  ext_weights shape: {ext_weights.shape}")

    # Generate test Green's functions
    n_freq = 21
    freqs = np.linspace(1.0, 14.0, n_freq)
    dw = freqs[1] - freqs[0]

    rng = np.random.default_rng(42)
    G_lesser = rng.standard_normal((n_kpts, n_freq, n_dof, n_dof)) + \
               1j * rng.standard_normal((n_kpts, n_freq, n_dof, n_dof))
    G_greater = rng.standard_normal((n_kpts, n_freq, n_dof, n_dof)) + \
                1j * rng.standard_normal((n_kpts, n_freq, n_dof, n_dof))
    # Make anti-Hermitian (physical property of G^<, G^>)
    G_lesser = 0.5 * (G_lesser - np.conj(G_lesser.transpose(0, 1, 3, 2)))
    G_greater = 0.5 * (G_greater - np.conj(G_greater.transpose(0, 1, 3, 2)))

    # Dense self-energy (ground truth)
    print("\nComputing dense self-energy (reference)...")
    t0 = time.time()
    SL_dense, SG_dense, SR_dense = _compute_phph_self_energy_q_dense(
        G_lesser, G_greater, M_stacked, T_all, q_diff_map,
        nat_prim, n_kpts, freqs, dw,
    )
    t_dense = time.time() - t0
    print(f"  Time: {t_dense:.2f}s")

    # Shifted-FT PCP self-energy
    print("Computing shifted-FT PCP self-energy...")
    t0 = time.time()
    SL_pcp, SG_pcp, SR_pcp = _compute_phph_self_energy_pcp_shifted(
        G_lesser, G_greater,
        f_shifted, ext_weights, lambdas,
        n_dof, n_kpts, freqs, dw,
        q_diff_map=q_diff_map,
    )
    t_pcp = time.time() - t0
    print(f"  Time: {t_pcp:.2f}s")

    # Compare
    norm_R = np.linalg.norm(SR_dense)
    norm_L = np.linalg.norm(SL_dense)
    norm_G = np.linalg.norm(SG_dense)

    err_R = np.linalg.norm(SR_pcp - SR_dense) / norm_R
    err_L = np.linalg.norm(SL_pcp - SL_dense) / norm_L
    err_G = np.linalg.norm(SG_pcp - SG_dense) / norm_G

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Dense time:       {t_dense:.2f}s")
    print(f"  Shifted PCP time: {t_pcp:.2f}s")
    print(f"  Speedup:          {t_dense/t_pcp:.2f}x")
    print(f"")
    print(f"  Sigma^R error:    {err_R:.6e}")
    print(f"  Sigma^< error:    {err_L:.6e}")
    print(f"  Sigma^> error:    {err_G:.6e}")

    # Per q-point breakdown
    print(f"\n  {'q-point':<16s} {'|Sigma^R|':>10s} {'rel error':>12s}")
    print("  " + "-"*42)
    for iq in range(n_kpts):
        norm_q = np.linalg.norm(SR_dense[iq])
        if norm_q > 0:
            err_q = np.linalg.norm(SR_pcp[iq] - SR_dense[iq]) / norm_q
            print(f"  {str(q_points[iq]):<16s} {norm_q:>10.4e} {err_q:>12.2e}")

    print(f"\n{'='*60}")
    if err_R < 1e-10:
        print("PASS: Shifted-FT PCP kernel matches dense reference (machine precision)")
    else:
        print(f"FAIL: Error {err_R:.2e} exceeds machine precision")
    print(f"{'='*60}")

    # Also test at N_c=24 (full rank)
    print("\n\nAlso testing at full rank N_c=24...")
    A_modes_24, lambdas_24, info_24 = fit_pcp(fc3_raw, phonon, N_c=24,
                                                max_iter=2000, verbose=False)
    print(f"  rel_err = {info_24['rel_err']:.6e}")

    fc3_recon_24 = reconstruct_fc3_from_pcp(A_modes_24, lambdas_24, phonon, info_24)
    M_stacked_24 = build_realspace_fc3_matrices(fc3_recon_24, nat_prim, masses_super, ref_sc_atoms)

    f_shifted_24, ext_weights_24 = fourier_transform_pcp_shifted(
        A_modes_24, lambdas_24, phonon, q_points,
        transport_direction='x', info=info_24,
    )

    t0 = time.time()
    SL_dense_24, SG_dense_24, SR_dense_24 = _compute_phph_self_energy_q_dense(
        G_lesser, G_greater, M_stacked_24, T_all, q_diff_map,
        nat_prim, n_kpts, freqs, dw,
    )
    t_dense_24 = time.time() - t0

    t0 = time.time()
    SL_pcp_24, SG_pcp_24, SR_pcp_24 = _compute_phph_self_energy_pcp_shifted(
        G_lesser, G_greater,
        f_shifted_24, ext_weights_24, lambdas_24,
        n_dof, n_kpts, freqs, dw,
        q_diff_map=q_diff_map,
    )
    t_pcp_24 = time.time() - t0

    err_24 = np.linalg.norm(SR_pcp_24 - SR_dense_24) / np.linalg.norm(SR_dense_24)
    print(f"  N_c=24: error = {err_24:.6e}")
    print(f"  Dense time: {t_dense_24:.2f}s, PCP shifted time: {t_pcp_24:.2f}s")
    print(f"  Speedup: {t_dense_24/t_pcp_24:.2f}x")


if __name__ == "__main__":
    main()
