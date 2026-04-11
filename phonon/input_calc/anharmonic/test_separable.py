"""Test the separable FC3 self-energy implementation.

Three levels of testing:
1. Decomposition: SVD rank and singular values for FD and DFPT FC3
2. Kernel: full-rank separable reproduces the dense self-energy
3. SCBA: full transmission comparison (FD vs DFPT, dense vs separable)

Requires:
  - fc3_prim/fc3.hdf5     (FD, run fc3-reap first)
  - dfpt/fc3.hdf5          (DFPT, run dfpt pipeline first)
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
work_dir = script_dir.parent  # input_calc/
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell, load_primitive_cell_dfpt
from phonon_inputs.constants import CONVERSION_FC3_THZ
from phonon_inputs.separable import (
    build_supercell_mapping,
    decompose_fc3_supercell,
    reconstruction_error,
    separable_anharmonic_transmission,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _get_prim_sc_indices(phonon):
    """Return p2s_map: supercell indices for each primitive atom at cell (0,0,0)."""
    return phonon.primitive.p2s_map


def _load_fc3_raw(hdf5_path):
    """Load raw FC3 array from HDF5."""
    with h5py.File(hdf5_path, "r") as f:
        return np.array(f["fc3"])


def _svd_per_dof(fc3_raw, p2s_map, nat_prim):
    """Per-(prim_atom, mu) SVD analysis, correctly handling full-format FC3.

    For full FC3 (n_super, n_super, n_super, 3, 3, 3), index with p2s_map
    to get the correct supercell atom for each primitive atom.
    For compact FC3 (n_prim, n_super, n_super, 3, 3, 3), index directly.
    """
    is_compact = fc3_raw.shape[0] == nat_prim
    n_super_cols = fc3_raw.shape[1]
    dim = n_super_cols * 3

    results = []
    for i_prim in range(nat_prim):
        s_i = i_prim if is_compact else int(p2s_map[i_prim])
        for mu in range(3):
            block = fc3_raw[s_i, :, :, mu, :, :]
            mat = block.transpose(0, 2, 1, 3).reshape(dim, dim)
            sv = np.linalg.svd(mat, compute_uv=False)
            total = np.sum(sv**2)
            if total < 1e-30:
                results.append(dict(i_prim=i_prim, mu=mu, sv=sv,
                                    r99=0, n_nonzero=0))
                continue
            cum = np.cumsum(sv**2) / total
            r99 = int(np.searchsorted(cum, 0.99)) + 1
            n_nz = int(np.sum(sv > 1e-10 * sv[0]))
            results.append(dict(i_prim=i_prim, mu=mu, sv=sv,
                                r99=r99, n_nonzero=n_nz))
    return results


# -----------------------------------------------------------------------
# Test 1: Decomposition
# -----------------------------------------------------------------------

def test_decomposition(phonon, fc3_hdf5, label=""):
    """Test SVD decomposition quality for an FC3 source."""
    print(f"\n{'=' * 60}")
    print(f"Test 1: Supercell FC3 decomposition [{label}]")
    print("=" * 60)

    fc3_raw = _load_fc3_raw(fc3_hdf5)
    n_atoms = len(phonon.primitive.masses)
    masses_super = phonon.supercell.masses
    p2s_map = _get_prim_sc_indices(phonon)

    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(
        phonon, "x"
    )
    trans_atoms = np.where(slab_indices == 0)[0]
    dim_t = len(trans_atoms) * 3

    print(f"  FC3 shape: {fc3_raw.shape}")
    print(f"  p2s_map: {p2s_map}")
    print(f"  Same-slab atoms: {len(trans_atoms)}, dim_trans = {dim_t}")

    # Rank sweep
    for rank in [1, 3, 6, 10, None]:
        F_list, H, svals, ta = decompose_fc3_supercell(
            fc3_raw, n_atoms, masses_super, prim_indices, slab_indices,
            ref_sc_atoms, rank=rank,
        )
        R = len(F_list)
        err = reconstruction_error(
            fc3_raw, n_atoms, masses_super, prim_indices, slab_indices,
            ref_sc_atoms, F_list, H, ta,
        )
        print(f"  rank={R:2d}/{dim_t}: recon error = {err:.6e}")

    # Per-(i_prim, mu) SVD
    print(f"\n  Per-(i_prim, mu) SVD:")
    svd_results = _svd_per_dof(fc3_raw, p2s_map, n_atoms)
    for r in svd_results:
        sv = r["sv"]
        print(f"    atom {r['i_prim']}, {'xyz'[r['mu']]}: "
              f"R(99%) = {r['r99']}, sv[0] = {sv[0]:.4e}, "
              f"nonzero = {r['n_nonzero']}")

    return svd_results


# -----------------------------------------------------------------------
# Test 2: Fourier transform consistency (q=0 check)
# -----------------------------------------------------------------------

def test_fourier_transform(phonon, fc3_hdf5, label=""):
    """Test that FT factors at q=0 reproduce the direct sum of raw FC3."""
    from phonon_inputs.separable import fourier_transform_factors

    print(f"\n{'=' * 60}")
    print(f"Test 2: Fourier transform consistency [{label}]")
    print("=" * 60)

    fc3_raw = _load_fc3_raw(fc3_hdf5)
    n_atoms = len(phonon.primitive.masses)
    masses_super = phonon.supercell.masses
    p2s_map = _get_prim_sc_indices(phonon)

    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(
        phonon, "x"
    )

    F_list, H, svals, trans_atoms = decompose_fc3_supercell(
        fc3_raw, n_atoms, masses_super, prim_indices, slab_indices, ref_sc_atoms,
        rank=None, tol=1e-12,
    )
    R = len(F_list)

    # FT at q=(0,0)
    F_hat_0, H_hat_0 = fourier_transform_factors(
        F_list, H, trans_atoms, prim_indices, cell_frac,
        (0.0, 0.0), n_atoms, "x",
    )

    # Reconstruct FC3 tensor from FT'd factors at q=0
    n_dof = n_atoms * 3
    Phi3_reconstructed = np.zeros((n_dof, n_dof, n_dof), dtype=complex)
    for r in range(R):
        Phi3_reconstructed += np.einsum(
            'ac,d->acd', F_hat_0[r], H_hat_0[:, r]
        )

    # Reference: direct sum of raw FC3 over same-slab atoms
    is_compact = fc3_raw.shape[0] == n_atoms
    Phi3_ref = np.zeros((n_dof, n_dof, n_dof))
    for i_prim in range(n_atoms):
        s_i = i_prim if is_compact else int(p2s_map[i_prim])
        m_i = masses_super[ref_sc_atoms[i_prim]]
        for alpha in range(3):
            a = 3 * i_prim + alpha
            for s_j in trans_atoms:
                kappa_j = prim_indices[s_j]
                m_j = masses_super[s_j]
                for s_k in trans_atoms:
                    kappa_k = prim_indices[s_k]
                    m_k = masses_super[s_k]
                    mass_factor = np.sqrt(m_i * m_j * m_k)
                    for beta in range(3):
                        for gamma in range(3):
                            Phi3_ref[a, 3*kappa_j+beta, 3*kappa_k+gamma] += (
                                fc3_raw[s_i, s_j, s_k, alpha, beta, gamma]
                                / mass_factor * CONVERSION_FC3_THZ
                            )

    max_ref = np.max(np.abs(Phi3_ref))
    diff = np.max(np.abs(Phi3_ref - Phi3_reconstructed.real))
    rel_err = diff / max_ref if max_ref > 0 else 0

    print(f"  Max |Phi3_ref (direct)|:  {max_ref:.4e}")
    print(f"  Max |diff|:              {diff:.4e}")
    print(f"  Relative error:          {rel_err:.2e}")
    print(f"  Max |imag|:              {np.max(np.abs(Phi3_reconstructed.imag)):.2e}")

    status = "PASS" if rel_err < 1e-10 else f"MISMATCH (rel_err = {rel_err:.2e})"
    print(f"  FT at q=0: {status}")
    return rel_err


# -----------------------------------------------------------------------
# Test 3: SCBA comparison (dense Approx III vs separable, FD vs DFPT)
# -----------------------------------------------------------------------

def test_scba(phonon_fd, fc3_hdf5_fd, phonon_dfpt, fc3_hdf5_dfpt):
    """Compare separable full rank vs truncated, FD vs DFPT."""
    print(f"\n{'=' * 60}")
    print("Test 3: Separable SCBA — FD vs DFPT, full vs truncated")
    print("=" * 60)

    common = dict(
        freq_range_thz=(1.0, 14.0, 31),
        transport_direction="x",
        eta_factor=0.5,
        temperature=300.0,
        delta_T=10.0,
        max_scba_iter=10,
        scba_tol=0.005,
        mixing=0.3,
        n_slabs=1,
    )

    results = {}
    for label, phonon, fc3_h5 in [
        ("FD",   phonon_fd,   fc3_hdf5_fd),
        ("DFPT", phonon_dfpt, fc3_hdf5_dfpt),
    ]:
        print(f"\n  --- {label}: Separable full rank (4x4) ---")
        t0 = time.time()
        res_full = separable_anharmonic_transmission(
            phonon, str(fc3_h5),
            q_mesh_transverse=(4, 4),
            rank=None, svd_tol=1e-12,
            verbose=True, **common,
        )
        t_full = time.time() - t0

        print(f"\n  --- {label}: Separable R=6 (4x4) ---")
        t0 = time.time()
        res_r6 = separable_anharmonic_transmission(
            phonon, str(fc3_h5),
            q_mesh_transverse=(4, 4),
            rank=6,
            verbose=True, **common,
        )
        t_r6 = time.time() - t0

        results[label] = {
            "full": res_full,
            "r6": res_r6,
            "t_full": t_full,
            "t_r6": t_r6,
        }

    # Print summary table
    print(f"\n{'-' * 66}")
    print(f"{'Case':8s} {'G_ball':>10s} {'G_full':>10s} {'G_R6':>10s} "
          f"{'conserv_full':>13s} {'conserv_R6':>11s}")
    print(f"{' ':8s} {'MW/m²K':>10s} {'MW/m²K':>10s} {'MW/m²K':>10s}")
    print("-" * 66)
    for label in ["FD", "DFPT"]:
        r = results[label]
        Gb = r["full"]["thermal_conductance_ballistic"] / 1e6
        Gf = r["full"]["thermal_conductance_anharmonic"] / 1e6
        Gr = r["r6"]["thermal_conductance_anharmonic"] / 1e6
        cf = r["full"]["heat_flow_conservation"]
        cr = r["r6"]["heat_flow_conservation"]
        print(f"{label:8s} {Gb:10.2f} {Gf:10.2f} {Gr:10.2f} {cf:13.4e} {cr:11.4e}")

    return results


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_comparison(results, svd_fd, svd_dfpt, fc3_hdf5_fd, fc3_hdf5_dfpt,
                    phonon_fd, phonon_dfpt):
    """Plot FD vs DFPT comparison: spectral current, SVD decay, conductance."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ---- Row 1: Spectral heat current per source ----
    for col, label in enumerate(["FD", "DFPT"]):
        ax = axes[0, col]
        r = results[label]
        freqs = r["full"]["freqs_thz"]
        ax.plot(freqs, r["full"]["spectral_heat_current_ballistic"],
                "k-", lw=0.8, alpha=0.4, label="Ballistic")
        Gf = r["full"]["thermal_conductance_anharmonic"] / 1e6
        Gr = r["r6"]["thermal_conductance_anharmonic"] / 1e6
        ax.plot(freqs, r["full"]["spectral_heat_current"], "b-", lw=1.8,
                label=f"Full rank ({Gf:.0f})")
        ax.plot(r["r6"]["freqs_thz"],
                r["r6"]["spectral_heat_current"], "r--", lw=1.5,
                label=f"R=6 ({Gr:.0f})")
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("Spectral heat current (W)")
        ax.set_title(f"{label}: spectral heat current")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ---- Row 1 col 3: FD vs DFPT overlay (full rank) ----
    ax = axes[0, 2]
    for label, color, ls in [("FD", "tab:blue", "-"), ("DFPT", "tab:red", "--")]:
        r = results[label]
        freqs = r["full"]["freqs_thz"]
        Gf = r["full"]["thermal_conductance_anharmonic"] / 1e6
        ax.plot(freqs, r["full"]["spectral_heat_current_ballistic"],
                color=color, ls=ls, lw=0.8, alpha=0.4)
        ax.plot(freqs, r["full"]["spectral_heat_current"],
                color=color, ls=ls, lw=1.8,
                label=f"{label} ({Gf:.0f} MW/m²K)")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Spectral heat current (W)")
    ax.set_title("FD vs DFPT (full rank)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Row 2 col 1: SVD singular value decay (per DOF) ----
    ax = axes[1, 0]
    for svd_results, label, color in [(svd_fd, "FD", "tab:blue"),
                                       (svd_dfpt, "DFPT", "tab:red")]:
        for r in svd_results:
            sv = r["sv"]
            if sv[0] < 1e-30:
                continue
            linestyle = ["-", "--", ":"][r["mu"]]
            ax.semilogy(sv / sv[0], color=color, ls=linestyle, lw=1.0,
                        alpha=0.7, markersize=2)
        ax.plot([], [], color=color, lw=1.5, label=label)
    ax.set_xlabel("Singular value index r")
    ax.set_ylabel(r"$\sigma_r / \sigma_0$")
    ax.set_title("FC3 singular value decay (per DOF)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Row 2 col 2: Stacked SVD decay ----
    ax = axes[1, 1]
    for fc3_h5, phonon, label, color in [
        (fc3_hdf5_fd, phonon_fd, "FD", "tab:blue"),
        (fc3_hdf5_dfpt, phonon_dfpt, "DFPT", "tab:red"),
    ]:
        fc3_raw = _load_fc3_raw(fc3_h5)
        n_atoms = len(phonon.primitive.masses)
        prim_indices, cell_frac, slab_indices, ref_sc_atoms = (
            build_supercell_mapping(phonon, "x"))
        _, _, svals, _ = decompose_fc3_supercell(
            fc3_raw, n_atoms, phonon.supercell.masses,
            prim_indices, slab_indices, ref_sc_atoms, rank=None, tol=0,
        )
        ax.semilogy(range(1, len(svals) + 1), svals / svals[0],
                     "o-", color=color, ms=4, lw=1.2, label=label)
    ax.set_xlabel("Rank r")
    ax.set_ylabel(r"$\sigma_r / \sigma_1$")
    ax.set_title("Stacked SVD: singular value decay")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Row 2 col 3: Bar chart of conductances ----
    ax = axes[1, 2]
    labels_bar = ["FD", "DFPT"]
    x = np.arange(len(labels_bar))
    width = 0.25
    for i, (key, color, name) in enumerate([
        ("full", "tab:blue", "Full rank"),
        ("r6", "tab:green", "R=6"),
    ]):
        vals = [results[l][key]["thermal_conductance_anharmonic"] / 1e6
                for l in labels_bar]
        ax.bar(x + (i - 0.5) * width, vals, width, color=color, label=name)
    for j, l in enumerate(labels_bar):
        Gb = results[l]["full"]["thermal_conductance_ballistic"] / 1e6
        ax.plot([j - width, j + width], [Gb, Gb],
                "k--", lw=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar)
    ax.set_ylabel("G (MW / m² K)")
    ax.set_title("Thermal conductance comparison")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(script_dir / "separable_test.png", dpi=150)
    plt.close("all")
    print(f"\n  Saved separable_test.png")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    # --- Load FD ---
    print("Loading FD (phono3py) data...")
    phonon_fd, _ = load_primitive_cell(work_dir)
    fc3_hdf5_fd = work_dir / "fc3_prim" / "fc3.hdf5"

    # --- Load DFPT ---
    print("Loading DFPT data...")
    phonon_dfpt, _ = load_primitive_cell_dfpt(work_dir)
    fc3_hdf5_dfpt = work_dir / "dfpt" / "fc3.hdf5"

    # --- Test 1: Decomposition ---
    svd_fd = test_decomposition(phonon_fd, fc3_hdf5_fd, label="FD")
    svd_dfpt = test_decomposition(phonon_dfpt, fc3_hdf5_dfpt, label="DFPT")

    # --- Test 2: FT consistency ---
    test_fourier_transform(phonon_fd, fc3_hdf5_fd, label="FD")
    test_fourier_transform(phonon_dfpt, fc3_hdf5_dfpt, label="DFPT")

    # --- Test 3: SCBA comparison ---
    results = test_scba(phonon_fd, fc3_hdf5_fd, phonon_dfpt, fc3_hdf5_dfpt)

    # --- Plot ---
    plot_comparison(results, svd_fd, svd_dfpt,
                    fc3_hdf5_fd, fc3_hdf5_dfpt,
                    phonon_fd, phonon_dfpt)

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)
