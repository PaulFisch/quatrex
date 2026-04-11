"""Benchmark all self-energy kernel implementations."""

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
    reconstruct_fc3_from_pcp,
    _supercell_cp_forward_torch,
)
from phonon_inputs.separable import (
    build_supercell_mapping, build_realspace_fc3_matrices,
    build_gathering_matrix, build_q_diff_map,
    _compute_phph_self_energy_separable,
)
from phonon_inputs.anharmonic import _compute_phph_self_energy_q_dense
from phonon_inputs.constants import CONVERSION_FC3_THZ
import h5py
import torch


def main():
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    nat_prim = len(phonon.primitive.masses)
    n_dof = nat_prim * 3
    n_super = len(phonon.supercell.masses)
    p2s = phonon.primitive.p2s_map
    masses = phonon.supercell.masses

    # Dense infrastructure from raw FC3
    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(phonon)
    M_stacked = build_realspace_fc3_matrices(fc3_raw, nat_prim, masses, ref_sc_atoms)
    U, S, Vt = np.linalg.svd(M_stacked, full_matrices=False)

    print(f"{'='*70}")
    print(f"BENCHMARK: Si (nat_prim={nat_prim}, n_dof={n_dof}, n_super={n_super})")
    print(f"{'='*70}")

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

    T_all = [build_gathering_matrix(prim_indices, cell_frac, q, nat_prim, 'x')
             for q in q_points]

    print(f"\n  q-mesh: {nk}x{nk} = {n_kpts}, n_freq={n_freq}")

    # Dense reference
    t0 = time.time()
    SL_d, SG_d, SR_d = _compute_phph_self_energy_q_dense(
        G_lesser, G_greater, M_stacked, T_all, q_diff_map,
        nat_prim, n_kpts, freqs, dw)
    t_dense = time.time() - t0
    norm_R = np.linalg.norm(SR_d)
    print(f"\n  Dense:     {t_dense:.3f}s (reference)")

    # SVD at various ranks
    for R in [6, 12, 24, 48]:
        F_list = [U[:, r:r+1].reshape(n_dof, n_super*3) * S[r] for r in range(R)]
        H = Vt[:R, :].T

        F_hat_q = []
        H_hat_q = []
        for T_q in T_all:
            F_hat_q.append([F_r @ T_q.T for F_r in F_list])
            H_hat_q.append(T_q @ H)

        t0 = time.time()
        SL_s, SG_s, SR_s = _compute_phph_self_energy_separable(
            G_lesser, G_greater, F_hat_q, H_hat_q,
            n_dof, n_kpts, freqs, dw, q_diff_map=q_diff_map)
        t_svd = time.time() - t0
        err = np.linalg.norm(SR_s - SR_d) / norm_R
        print(f"  SVD  R={R:2d}:  {t_svd:.3f}s  err={err:.2e}  ({t_dense/t_svd:.2f}x)")

    # Supercell CP at various ranks
    for N_c in [4, 8, 12, 24]:
        u_modes, lam_sc, info_sc = fit_supercell_cp(
            fc3_raw, phonon, N_c=N_c, max_iter=2000, verbose=False)
        f_modes_q, f_ext = fourier_transform_supercell_cp(
            u_modes, lam_sc, phonon, q_points, transport_direction='x')

        t0 = time.time()
        SL_c, SG_c, SR_c = _compute_phph_self_energy_pcp(
            G_lesser, G_greater, f_modes_q, lam_sc,
            n_dof, n_kpts, freqs, dw,
            q_diff_map=q_diff_map, f_ext=f_ext)
        t_cp = time.time() - t0
        err = np.linalg.norm(SR_c - SR_d) / norm_R
        print(f"  SC-CP N_c={N_c:2d}: {t_cp:.3f}s  err={err:.2e}  ({t_dense/t_cp:.2f}x)  fit={info_sc['rel_err']:.2e}")

    # FLOP analysis
    print(f"\n{'='*70}")
    print(f"THEORETICAL COST SCALING (FLOPs per (q,q') pair)")
    print(f"{'='*70}")
    for nd, label in [(6, "Si (2 atoms)"), (12, "4 atoms"), (30, "10 atoms"),
                       (60, "20 atoms"), (120, "40 atoms")]:
        Nw = 42
        dense_cost = Nw * (nd**5 + 2*nd**4)
        print(f"\n  {label} (n_dof={nd}):")
        print(f"    Dense:           {dense_cost:>12.2e}")
        for R in [6, min(nd, 30)]:
            svd_cost = Nw * (2*nd**3*R + nd**4)
            print(f"    SVD R={R:<3d}:        {svd_cost:>12.2e}  ({dense_cost/svd_cost:.0f}x)")
        for Nc in [4, 12, 24]:
            # S3 permutation: 36*Nc^2*Nw, IFFT: 9*Nc^2*Nw, accum: 9*Nc*nd*Nw
            cp_cost = Nw * (36*Nc**2 + 9*Nc**2 + 9*Nc*nd)
            print(f"    SC-CP N_c={Nc:<3d}:   {cp_cost:>12.2e}  ({dense_cost/cp_cost:.0f}x)")

    print(f"\n{'='*70}")
    print(f"MEMORY: dominant intermediate per (q,q') pair")
    print(f"{'='*70}")
    for nd, Nw, label in [(6, 42, "Si"), (30, 200, "10at"), (60, 200, "20at"), (120, 200, "40at")]:
        K_bytes = Nw * nd**4 * 16
        g_bytes_24 = 24*24 * 9 * Nw * 16  # g has no n_dof dependence!
        print(f"  {label:5s}: Dense K = {K_bytes/1e9:.2f} GB, SC-CP g(N_c=24) = {g_bytes_24/1e6:.0f} MB")


if __name__ == "__main__":
    main()
