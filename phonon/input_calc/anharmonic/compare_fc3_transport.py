"""Compare FC3 approximation methods via quantum transport observable G_anh.

For each method at various ranks:
  1. Reconstruct M_stacked from the approximation
  2. Run the full dense SCBA transport simulation
  3. Compare G_anh to the reference (full FC3)

Plots: relative error in G_anh vs number of parameters for each method.

Requires:
  - fc3_prim/fc3.hdf5
  - fc3_prim/phono3py_disp.yaml
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs.constants import (
    CONVERSION_FC3_THZ, CONVERSION_THZ2, HBAR_SI, KB_SI, THZ_TO_RAD,
)
from phonon_inputs.separable import (
    build_supercell_mapping,
    build_realspace_fc3_matrices,
    build_gathering_matrix,
    build_q_diff_map,
)
from phonon_inputs.anharmonic import (
    _build_device_hamiltonian,
    _compute_obc_self_energies,
    _solve_green_functions,
    _compute_phph_self_energy_q_dense,
)
from phonon_inputs.convention import get_btd_blocks
from phonon_inputs.validation import _ballistic_transmission
from phonon_inputs.pcp import _build_target, fit_supercell_cp

from compare_fc3_approximations import (
    svd_approximation, svd_n_params,
    pscp_decomposition, pscp_reconstruct, pscp_n_params,
    scp3_reconstruct_M_stacked, scp3_n_params,
    fit_fscp, fscp_reconstruct_M_stacked, fscp_n_params,
)


# =====================================================================
# Transport computation with a given M_stacked
# =====================================================================

def compute_transport(
    phonon, M_stacked,
    q_mesh_transverse=(4, 4),
    freq_range_thz=(0.0, 15.0, 51),
    transport_direction="x",
    eta_factor=0.5,
    temperature=300.0,
    delta_T=10.0,
    max_scba_iter=20,
    scba_tol=0.005,
    mixing=0.3,
    n_slabs=1,
    verbose=False,
):
    """Run dense SCBA transport with a given M_stacked.

    Same as anharmonic_transmission_q but takes M_stacked directly.
    Returns dict with thermal_conductance_anharmonic, etc.
    """
    _fmin, fmax, nfreq_pos = freq_range_thz
    nfreq_pos = int(nfreq_pos)
    freqs_pos = np.linspace(0.0, fmax, nfreq_pos)
    freqs_thz = np.concatenate((-freqs_pos[:0:-1], freqs_pos))
    nfreq = len(freqs_thz)
    dw_thz = freqs_pos[1] - freqs_pos[0]
    omega_sq_thz2 = freqs_thz ** 2
    eta = dw_thz ** 2 * eta_factor
    pos_mask = freqs_thz >= 0.0

    n_atoms = len(phonon.primitive.masses)
    n_dof = 3 * n_atoms
    N_D = n_slabs * n_dof

    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(
        phonon, transport_direction)

    # q-mesh
    nkx, nky = q_mesh_transverse
    q_points = [(i / nkx, j / nky) for i in range(nkx) for j in range(nky)]
    n_kpts = len(q_points)
    q_diff_map = build_q_diff_map(nkx, nky)

    T_all_q = [build_gathering_matrix(prim_indices, cell_frac, q, n_atoms,
                                       transport_direction) for q in q_points]

    # Bose-Einstein (SI units, expm1 for numerical stability)
    def bose_einstein(freq_thz_arr, T):
        omega_rad_s = np.abs(freq_thz_arr) * THZ_TO_RAD
        x = HBAR_SI * omega_rad_s / (KB_SI * T)
        n = np.zeros_like(x)
        valid = x > 1e-12
        n[valid] = 1.0 / np.expm1(x[valid])
        return n

    T_L = temperature + delta_T / 2.0
    T_R = temperature - delta_T / 2.0
    n_bose_L = bose_einstein(freqs_thz, T_L)
    n_bose_R = bose_einstein(freqs_thz, T_R)

    # BTD blocks
    btd_blocks = []
    for q in q_points:
        H_00, H_01 = get_btd_blocks(phonon, q, transport_direction=transport_direction,
                                      conversion_factor=CONVERSION_THZ2)
        btd_blocks.append((H_00, H_01))

    # Ballistic
    trans_ballistic = np.zeros(nfreq)
    for iq, (H_00, H_01) in enumerate(btd_blocks):
        H_D = _build_device_hamiltonian(H_00, H_01, n_slabs)
        H_LD = np.zeros((n_dof, N_D), dtype=complex)
        H_LD[:, :n_dof] = H_01
        H_DR = np.zeros((N_D, n_dof), dtype=complex)
        H_DR[-n_dof:, :] = H_01
        for iw, w2 in enumerate(omega_sq_thz2):
            trans_ballistic[iw] += _ballistic_transmission(
                w2, H_D, H_00, H_01, H_00, H_01, H_LD, H_DR, eta=eta)
    trans_ballistic /= n_kpts

    # Cross-sectional area
    lattice = phonon.primitive.cell
    tidx = "xyz".index(transport_direction)
    perp_idx = [i for i in range(3) if i != tidx]
    a1 = lattice[perp_idx[0]]
    a2 = lattice[perp_idx[1]]
    A_c = np.linalg.norm(np.cross(a1, a2)) * 1e-20

    omega_rad = freqs_thz * THZ_TO_RAD
    spectral_J_ball = HBAR_SI * omega_rad * (n_bose_L - n_bose_R) * trans_ballistic
    J_ball_total = np.sum(spectral_J_ball[pos_mask]) * dw_thz * 1e12
    G_ball = J_ball_total / (A_c * delta_T)

    # SCBA
    Sigma_R_q = np.zeros((n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex)
    Sigma_l_q = np.zeros_like(Sigma_R_q)
    Sigma_g_q = np.zeros_like(Sigma_R_q)

    spectral_J_L = np.zeros(nfreq)
    spectral_J_R = np.zeros(nfreq)
    convergence_history = []
    J_total_prev = 0.0

    for scba_iter in range(max_scba_iter):
        G_lesser_slab_q = np.zeros((n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex)
        G_greater_slab_q = np.zeros_like(G_lesser_slab_q)
        spectral_J_L[:] = 0.0
        spectral_J_R[:] = 0.0

        for iq, (H_00, H_01) in enumerate(btd_blocks):
            H_D = _build_device_hamiltonian(H_00, H_01, n_slabs)
            for iw, w2 in enumerate(omega_sq_thz2):
                Sig_R_dev = np.zeros((N_D, N_D), dtype=complex)
                Sig_l_dev = np.zeros((N_D, N_D), dtype=complex)
                Sig_g_dev = np.zeros((N_D, N_D), dtype=complex)
                for l in range(n_slabs):
                    sl = slice(l * n_dof, (l + 1) * n_dof)
                    Sig_R_dev[sl, sl] = Sigma_R_q[l, iq, iw]
                    Sig_l_dev[sl, sl] = Sigma_l_q[l, iq, iw]
                    Sig_g_dev[sl, sl] = Sigma_g_q[l, iq, iw]

                obc = _compute_obc_self_energies(
                    w2, H_00, H_01, eta, n_bose_L[iw], n_bose_R[iw], n_slabs=n_slabs)
                _, G_less, G_great = _solve_green_functions(
                    w2, H_D, obc, Sig_R_dev, Sig_l_dev, Sig_g_dev, eta)

                for l in range(n_slabs):
                    sl = slice(l * n_dof, (l + 1) * n_dof)
                    G_lesser_slab_q[l, iq, iw] = G_less[sl, sl]
                    G_greater_slab_q[l, iq, iw] = G_great[sl, sl]

                sl0 = slice(0, n_dof)
                spectral_J_L[iw] += HBAR_SI * omega_rad[iw] * np.real(np.trace(
                    obc["Sigma_L_greater"][sl0, sl0] @ G_less[sl0, sl0]
                    - obc["Sigma_L_lesser"][sl0, sl0] @ G_great[sl0, sl0]))
                sl_last = slice((n_slabs - 1) * n_dof, n_slabs * n_dof)
                spectral_J_R[iw] += HBAR_SI * omega_rad[iw] * np.real(np.trace(
                    obc["Sigma_R_lesser"][sl_last, sl_last] @ G_great[sl_last, sl_last]
                    - obc["Sigma_R_greater"][sl_last, sl_last] @ G_less[sl_last, sl_last]))

        spectral_J_L /= n_kpts
        spectral_J_R /= n_kpts

        J_L_total = np.sum(spectral_J_L[pos_mask]) * dw_thz * 1e12
        J_R_total = np.sum(spectral_J_R[pos_mask]) * dw_thz * 1e12
        J_total = 0.5 * (J_L_total + J_R_total)
        J_denom = abs(J_L_total) + abs(J_R_total)
        conservation_err = abs(J_L_total - J_R_total) / J_denom if J_denom > 0 else 0.0

        # Self-energy update
        Sigma_l_new = np.zeros_like(Sigma_l_q)
        Sigma_g_new = np.zeros_like(Sigma_g_q)
        Sigma_r_new = np.zeros_like(Sigma_R_q)

        for l in range(n_slabs):
            sl_n, sg_n, sr_n = _compute_phph_self_energy_q_dense(
                G_lesser_slab_q[l], G_greater_slab_q[l],
                M_stacked, T_all_q, q_diff_map,
                n_atoms, n_kpts, freqs_thz, dw_thz)
            Sigma_l_new[l] = sl_n
            Sigma_g_new[l] = sg_n
            Sigma_r_new[l] = sr_n

        if scba_iter > 0:
            Sigma_l_q = (1 - mixing) * Sigma_l_q + mixing * Sigma_l_new
            Sigma_g_q = (1 - mixing) * Sigma_g_q + mixing * Sigma_g_new
            Sigma_R_q = (1 - mixing) * Sigma_R_q + mixing * Sigma_r_new
        else:
            Sigma_l_q = Sigma_l_new.copy()
            Sigma_g_q = Sigma_g_new.copy()
            Sigma_R_q = Sigma_r_new.copy()

        if scba_iter > 0:
            rel_change = abs(J_total - J_total_prev) / (abs(J_total_prev) + 1e-30)
            convergence_history.append(rel_change)
            if verbose:
                print(f"      SCBA {scba_iter+1}: J={J_total:.4e}, "
                      f"conserv={conservation_err:.4e}, change={rel_change:.4e}")
            if conservation_err < scba_tol:
                break
        else:
            if verbose:
                print(f"      SCBA 1: J_L={J_L_total:.4e}, J_R={J_R_total:.4e}")

        J_total_prev = J_total

    spectral_J_anh = 0.5 * (spectral_J_L + spectral_J_R)
    J_anh_total = np.sum(spectral_J_anh[pos_mask]) * dw_thz * 1e12
    G_anh = J_anh_total / (A_c * delta_T)

    return {
        "thermal_conductance_ballistic": G_ball,
        "thermal_conductance_anharmonic": G_anh,
        "heat_flow_conservation": conservation_err,
        "spectral_heat_current": spectral_J_anh[pos_mask],
        "spectral_heat_current_L": spectral_J_L[pos_mask].copy(),
        "spectral_heat_current_R": spectral_J_R[pos_mask].copy(),
        "spectral_heat_current_ballistic": spectral_J_ball[pos_mask],
        "freqs_thz": freqs_thz[pos_mask],
        "n_scba_iterations": scba_iter + 1,
        "convergence_history": convergence_history,
    }


# =====================================================================
# Main sweep
# =====================================================================

def main():
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"
    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)
    n_dof = 3 * nat_prim
    dim_sc = 3 * n_super
    masses_super = phonon.supercell.masses

    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(phonon)

    # Build reference M_stacked
    M_ref = build_realspace_fc3_matrices(fc3_raw, nat_prim, masses_super, ref_sc_atoms)
    M_norm = np.linalg.norm(M_ref, 'fro')

    # Full SVD for rank info
    _, S_full, _ = np.linalg.svd(M_ref, full_matrices=False)
    R_full = len(S_full)

    # Full PSCP
    d_pscp_full, v_pscp_full, norms_pscp = pscp_decomposition(M_ref, n_dof, dim_sc)

    print(f"System: nat_prim={nat_prim}, n_super={n_super}")
    print(f"  n_dof={n_dof}, dim_sc={dim_sc}, SVD rank={R_full}")
    print(f"  PSCP full rank={len(d_pscp_full)}")

    # Transport parameters (small mesh for reasonable runtime)
    transport_kw = dict(
        q_mesh_transverse=(4, 4),
        freq_range_thz=(0.0, 15.0, 51),
        transport_direction="x",
        eta_factor=0.5,
        temperature=300.0,
        delta_T=10.0,
        max_scba_iter=20,
        scba_tol=0.005,
        mixing=0.3,
        n_slabs=1,
        verbose=False,
    )

    # ---- Reference: full M_stacked ----
    print("\n" + "=" * 60)
    print("Reference: full FC3 (dense)")
    print("=" * 60)
    t0 = time.time()
    ref_result = compute_transport(phonon, M_ref, **transport_kw)
    dt_ref = time.time() - t0
    G_ref = ref_result["thermal_conductance_anharmonic"]
    G_ball = ref_result["thermal_conductance_ballistic"]
    print(f"  G_ball = {G_ball/1e6:.2f} MW/(m^2 K)")
    print(f"  G_anh  = {G_ref/1e6:.2f} MW/(m^2 K)  ({dt_ref:.1f}s)")

    results = {
        'SVD': {'ranks': [], 'n_params': [], 'G_anh': [], 'frob_err': []},
        'PSCP': {'ranks': [], 'n_params': [], 'G_anh': [], 'frob_err': []},
        'SCP3': {'ranks': [], 'n_params': [], 'G_anh': [], 'frob_err': []},
        'FSCP': {'ranks': [], 'n_params': [], 'G_anh': [], 'frob_err': []},
    }

    def run_one(method, rank, M_approx, n_params):
        frob_err = np.linalg.norm(M_ref - M_approx, 'fro') / M_norm
        t0 = time.time()
        res = compute_transport(phonon, M_approx, **transport_kw)
        dt = time.time() - t0
        G = res["thermal_conductance_anharmonic"]
        results[method]['ranks'].append(rank)
        results[method]['n_params'].append(n_params)
        results[method]['G_anh'].append(G)
        results[method]['frob_err'].append(frob_err)
        G_err = abs(G - G_ref) / abs(G_ref)
        print(f"  R={rank:3d}: params={n_params:6d}, "
              f"frob_err={frob_err:.4e}, "
              f"G_anh={G/1e6:.2f} MW/(m^2K), "
              f"G_err={G_err:.4e} ({dt:.1f}s)")

    # ---- SVD ----
    print("\n" + "=" * 60)
    print("Method 1: Truncated SVD")
    print("=" * 60)
    for R in [4, 8, 12, 16, 24]:
        if R > R_full:
            continue
        M_approx = svd_approximation(M_ref, R)
        run_one('SVD', R, M_approx, svd_n_params(R, n_dof, dim_sc))

    # ---- PSCP ----
    print("\n" + "=" * 60)
    print("Method 2: PSCP")
    print("=" * 60)
    for R in [6, 12, 24, 36, 48]:
        if R > len(d_pscp_full):
            continue
        M_approx = pscp_reconstruct(d_pscp_full[:R], v_pscp_full[:R], n_dof, dim_sc)
        run_one('PSCP', R, M_approx, pscp_n_params(R, n_dof, dim_sc))

    # ---- SCP3 ----
    print("\n" + "=" * 60)
    print("Method 3: Symmetric CP (3 modes)")
    print("=" * 60)
    for N_c in [4, 8, 16, 24]:
        print(f"  Fitting N_c={N_c}...")
        u_modes, lambdas, info = fit_supercell_cp(
            fc3_raw, phonon, N_c=N_c, max_iter=2000, verbose=False)
        M_approx = scp3_reconstruct_M_stacked(
            u_modes, lambdas, phonon, info['target_norm'], n_dof, dim_sc)
        run_one('SCP3', N_c, M_approx, scp3_n_params(N_c, dim_sc))

    # ---- FSCP ----
    print("\n" + "=" * 60)
    print("Method 4: Fully Symmetric CP (1 mode)")
    print("=" * 60)
    for R in [8, 16, 24, 48]:
        print(f"  Fitting R={R}...")
        v_modes, lambdas, info = fit_fscp(
            fc3_raw, phonon, R=R, max_iter=2000, verbose=False)
        M_approx = fscp_reconstruct_M_stacked(
            v_modes, lambdas, phonon, info['target_norm'], n_dof, dim_sc)
        run_one('FSCP', R, M_approx, fscp_n_params(R, dim_sc))

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Reference G_anh = {G_ref/1e6:.2f} MW/(m^2 K)")
    print(f"\n{'Method':<8} {'Rank':>6} {'Params':>8} {'Frob Err':>10} "
          f"{'G_anh [MW/m^2K]':>16} {'G_err':>10}")
    print("-" * 64)
    for method in ['SVD', 'PSCP', 'SCP3', 'FSCP']:
        r = results[method]
        for i in range(len(r['ranks'])):
            G_err = abs(r['G_anh'][i] - G_ref) / abs(G_ref)
            print(f"{method:<8} {r['ranks'][i]:>6} {r['n_params'][i]:>8} "
                  f"{r['frob_err'][i]:>10.4e} "
                  f"{r['G_anh'][i]/1e6:>16.2f} {G_err:>10.4e}")
        print("-" * 64)

    # ---- Plots ----
    fig_dir = script_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    colors = {'SVD': 'C0', 'PSCP': 'C1', 'SCP3': 'C2', 'FSCP': 'C3'}
    markers = {'SVD': 'o', 'PSCP': 's', 'SCP3': '^', 'FSCP': 'D'}

    # Plot 1: G_anh error vs number of parameters
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ['SVD', 'PSCP', 'SCP3', 'FSCP']:
        r = results[method]
        if not r['n_params']:
            continue
        G_errs = [abs(G - G_ref) / abs(G_ref) for G in r['G_anh']]
        ax.semilogy(r['n_params'], G_errs,
                    f'-{markers[method]}', color=colors[method],
                    label=method, markersize=6)
    ax.set_xlabel('Number of parameters')
    ax.set_ylabel(r'Relative error in $G_{\mathrm{anh}}$')
    ax.set_title(r'Transport error vs FC3 approximation parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "transport_error_vs_params.pdf")
    fig.savefig(fig_dir / "transport_error_vs_params.png", dpi=150)
    print(f"\nSaved: {fig_dir / 'transport_error_vs_params.pdf'}")

    # Plot 2: Frobenius error vs G_anh error (correlation)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ['SVD', 'PSCP', 'SCP3', 'FSCP']:
        r = results[method]
        if not r['frob_err']:
            continue
        G_errs = [abs(G - G_ref) / abs(G_ref) for G in r['G_anh']]
        ax.loglog(r['frob_err'], G_errs,
                  f'{markers[method]}', color=colors[method],
                  label=method, markersize=8)
    ax.set_xlabel(r'Frobenius norm error $\|\Phi - \tilde\Phi\|_F / \|\Phi\|_F$')
    ax.set_ylabel(r'Relative error in $G_{\mathrm{anh}}$')
    ax.set_title('Frobenius error vs transport error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Guide line
    frob_range = np.logspace(-4, 0, 50)
    ax.plot(frob_range, frob_range, 'k--', alpha=0.3, label='1:1')
    fig.tight_layout()
    fig.savefig(fig_dir / "frob_vs_transport_error.pdf")
    fig.savefig(fig_dir / "frob_vs_transport_error.png", dpi=150)
    print(f"Saved: {fig_dir / 'frob_vs_transport_error.pdf'}")

    # Plot 3: G_anh error vs rank
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ['SVD', 'PSCP', 'SCP3', 'FSCP']:
        r = results[method]
        if not r['ranks']:
            continue
        G_errs = [abs(G - G_ref) / abs(G_ref) for G in r['G_anh']]
        ax.semilogy(r['ranks'], G_errs,
                    f'-{markers[method]}', color=colors[method],
                    label=method, markersize=6)
    ax.set_xlabel('Rank R')
    ax.set_ylabel(r'Relative error in $G_{\mathrm{anh}}$')
    ax.set_title('Transport error vs rank')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "transport_error_vs_rank.pdf")
    fig.savefig(fig_dir / "transport_error_vs_rank.png", dpi=150)
    print(f"Saved: {fig_dir / 'transport_error_vs_rank.pdf'}")

    plt.close('all')


if __name__ == "__main__":
    main()
