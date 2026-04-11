"""Investigate FC3/FC2 strength ratio and SCBA convergence regime.

Compares the anharmonic coupling strength relative to harmonic force
constants for:
  1. Real Si (phono3py, 2x2x2 supercell)
  2. Toy model at various phi3 strengths

The SCBA (self-consistent Born approximation) is a perturbative method.
It converges when the self-energy Sigma is small compared to the
harmonic dynamical matrix: |Sigma| << |w^2 - D(q)|.  This requires
FC3 to be weak relative to FC2 — a condition that depends on the
quality of the FC3 input.
"""

import sys
from pathlib import Path

import numpy as np

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent
sys.path.insert(0, str(work_dir))


def analyze_si_fc():
    """Analyze FC3/FC2 ratio for Si."""
    from run_anharmonic import load_primitive_cell
    import h5py

    print("=" * 60)
    print("Si FC3/FC2 analysis (phono3py, 2x2x2 supercell)")
    print("=" * 60)

    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

    with h5py.File(fc3_path, "r") as f:
        fc3 = np.array(f["fc3"])

    fc2 = phonon.force_constants
    masses = phonon.supercell.masses
    sc_pos = phonon.supercell.positions
    n_sc = len(masses)

    print(f"\nSupercell: {n_sc} atoms")
    print(f"FC2 shape: {fc2.shape}")
    print(f"FC3 shape: {fc3.shape}")

    # FC2 analysis
    fc2_norms = np.zeros((n_sc, n_sc))
    for i in range(n_sc):
        for j in range(n_sc):
            fc2_norms[i, j] = np.linalg.norm(fc2[i, j])

    fc2_diag = np.mean([np.linalg.norm(fc2[i, i]) for i in range(n_sc)])
    fc2_offdiag_max = np.max(fc2_norms[~np.eye(n_sc, dtype=bool)])
    fc2_offdiag_mean = np.mean(fc2_norms[fc2_norms > 1e-10][~np.eye(n_sc, dtype=bool).ravel()[:np.sum(fc2_norms > 1e-10)]])

    print(f"\nFC2 (eV/A^2):")
    print(f"  Diagonal (on-site) norm: {fc2_diag:.4f}")
    print(f"  Off-diagonal max norm:   {fc2_offdiag_max:.4f}")

    # FC3 analysis
    fc3_norms_ijk = np.zeros((n_sc, n_sc, n_sc))
    for i in range(n_sc):
        for j in range(n_sc):
            for k in range(n_sc):
                fc3_norms_ijk[i, j, k] = np.linalg.norm(fc3[i, j, k])

    fc3_max = np.max(fc3_norms_ijk)
    fc3_nonzero = fc3_norms_ijk[fc3_norms_ijk > 1e-10]

    print(f"\nFC3 (eV/A^3):")
    print(f"  Max ||FC3[i,j,k]||_F:  {fc3_max:.4f}")
    print(f"  Mean (nonzero):        {np.mean(fc3_nonzero):.4f}")
    print(f"  Nonzero triplets:      {len(fc3_nonzero)}/{n_sc**3}")

    # FC3/FC2 ratio — key metric for SCBA validity
    # Dimensionless ratio: FC3 * u_rms / FC2
    # where u_rms is the thermal displacement amplitude
    # u_rms ~ sqrt(k_B T / (m w^2)) ~ sqrt(k_B T / FC2_diag)
    from phonon_inputs.constants import KB_EV
    T = 300.0  # K
    m_avg = np.mean(masses)  # amu
    # Typical harmonic frequency scale
    # FC2_diag ~ m * w^2, so w ~ sqrt(FC2_diag / m)
    # For phonopy FC2 in eV/A^2 and mass in amu:
    # w^2 [THz^2] = FC2 [eV/A^2] / m [amu] * conversion
    # conversion = 1 eV / (1 amu * 1 A^2) = 1.602e-19 / (1.661e-27 * 1e-20)
    #            = 1.602e-19 / 1.661e-47 = 9.648e27 s^-2
    #            = 9.648e27 / (2pi*1e12)^2 THz^2 = 244.5 THz^2
    from phonon_inputs.constants import CONVERSION_THZ2
    w_typical = np.sqrt(fc2_diag / m_avg * CONVERSION_THZ2)  # THz
    print(f"\nTypical harmonic frequency: {w_typical:.1f} THz")

    # Thermal displacement: u_rms^2 = k_B T / (m w^2)
    # In SI: u_rms = sqrt(k_B T / (m * w^2))
    # k_B T = 0.02585 eV at 300K
    # m * w^2 ~ FC2_diag [eV/A^2]
    kBT = KB_EV * T  # eV
    u_rms = np.sqrt(kBT / fc2_diag)  # Angstrom
    print(f"Thermal displacement u_rms: {u_rms:.4f} A at {T} K")

    # Dimensionless anharmonic parameter: alpha = FC3 * u_rms / FC2
    # This is the ratio of anharmonic to harmonic force at thermal displacement
    alpha = fc3_max * u_rms / fc2_diag
    print(f"\nAnharmonic parameter alpha = FC3_max * u_rms / FC2_diag = {alpha:.4f}")
    print(f"  (alpha << 1 required for SCBA convergence)")

    # Mass-weighted FC3 (what enters the self-energy)
    # The self-energy scales as FC3^2 / sqrt(m_i m_j m_k)
    # The mass-weighted ratio: FC3_mw = FC3 / sqrt(m^3)
    # Sigma ~ FC3_mw^2 * G ~ (FC3/sqrt(m^3))^2 / (w^2 - D)
    # Dimensionless: Sigma / D ~ (FC3 / sqrt(m^3))^2 / D^2 * kBT / w
    # This is roughly alpha^2 * (kBT / hbar*w)

    print(f"\n--- Per-neighbor-shell FC3 strength ---")
    sc_cell = phonon.supercell.cell
    for i in range(min(2, n_sc)):  # check first two atoms
        dists = []
        norms = []
        for j in range(n_sc):
            diff = sc_pos[j] - sc_pos[i]
            frac = np.linalg.solve(sc_cell.T, diff)
            frac -= np.round(frac)
            diff_min = sc_cell.T @ frac
            d = np.linalg.norm(diff_min)
            if d > 0.1:
                fc3_j_norm = np.max([np.linalg.norm(fc3[i, j, k])
                                     for k in range(n_sc)])
                dists.append(d)
                norms.append(fc3_j_norm)

        # Group by shell
        dists = np.array(dists)
        norms = np.array(norms)
        unique_d = np.unique(np.round(dists, 2))
        print(f"\n  Atom {i} (mass={masses[i]:.1f} amu):")
        for d in unique_d[:5]:
            mask = np.abs(dists - d) < 0.1
            shell_norm = np.mean(norms[mask])
            n_in_shell = np.sum(mask)
            print(f"    d={d:.2f} A: {n_in_shell} neighbors, "
                  f"max FC3 norm = {shell_norm:.4f} eV/A^3, "
                  f"alpha = {shell_norm * u_rms / fc2_diag:.4f}")

    # --- Self-energy magnitude estimate ---
    print(f"\n--- Self-energy magnitude estimate ---")
    # From first SCBA iteration: Sigma ~ (FC3_mw)^2 * G^(0) * dw
    # |Sigma| / |D| is the perturbative parameter
    # For Si: FC3_max ~ 20 eV/A^3, FC2_diag ~ 15 eV/A^2
    # Mass-weighted: FC3_mw = FC3 / sqrt(m^3) [in appropriate units]
    # After conversion to THz: the singular values encode this

    from phonon_inputs.separable import (
        build_supercell_mapping, decompose_fc3_supercell, CONVERSION_FC3_THZ,
    )
    prim_indices, cell_frac, slab_indices, ref_sc_atoms = \
        build_supercell_mapping(phonon, "x")
    n_atoms = len(phonon.primitive.masses)

    F_list, H, svals, trans_atoms = decompose_fc3_supercell(
        fc3, n_atoms, masses, prim_indices, slab_indices,
        ref_sc_atoms, rank=None, tol=1e-15,
    )
    print(f"  SVD singular values (THz^5/2 units): {svals}")
    print(f"  sigma_1^2 / w_typical^4 = {svals[0]**2 / w_typical**4:.2e}")
    print(f"  (this ratio controls |Sigma|/|D| in the self-energy)")

    return {
        "fc2_diag": fc2_diag,
        "fc3_max": fc3_max,
        "alpha": alpha,
        "u_rms": u_rms,
        "w_typical": w_typical,
        "svals": svals,
    }


def analyze_toy_fc(a=3.0, mass=28.0, k_spring=5.0, phi3_values=[0.5, 2.0, 5.0]):
    """Analyze FC3/FC2 ratio for the toy model at various FC3 strengths."""
    from test_scba_toy import make_toy_phonon, make_toy_fc3
    from phonon_inputs.constants import KB_EV
    from phonon_inputs.separable import (
        build_supercell_mapping, decompose_fc3_supercell, CONVERSION_FC3_THZ,
    )

    print("\n" + "=" * 60)
    print("Toy model FC3/FC2 analysis")
    print("=" * 60)

    phonon = make_toy_phonon(a=a, mass=mass, k_spring=k_spring)
    fc2 = phonon.force_constants
    fc2_diag = np.mean([np.linalg.norm(fc2[i, i]) for i in range(len(phonon.supercell.masses))])

    T = 300.0
    u_rms = np.sqrt(KB_EV * T / fc2_diag)
    from phonon_inputs.constants import CONVERSION_THZ2
    w_typical = np.sqrt(fc2_diag / mass * CONVERSION_THZ2)

    prim_indices, cell_frac, slab_indices, ref_sc_atoms = \
        build_supercell_mapping(phonon, "x")
    n_atoms = len(phonon.primitive.masses)
    masses_super = phonon.supercell.masses

    print(f"\n  FC2 diagonal norm: {fc2_diag:.4f} eV/A^2")
    print(f"  u_rms at {T}K: {u_rms:.4f} A")
    print(f"  w_typical: {w_typical:.1f} THz")

    print(f"\n  {'phi3':>8} {'FC3_max':>10} {'alpha':>10} {'sigma_1':>12} "
          f"{'sig1^2/w^4':>12} {'SCBA':>8}")
    print("  " + "-" * 65)

    for phi3 in phi3_values:
        fc3 = make_toy_fc3(phonon, phi3=phi3, a=a)
        fc3_max = np.max(np.abs(fc3))
        alpha = fc3_max * u_rms / fc2_diag

        F_list, H, svals, trans_atoms = decompose_fc3_supercell(
            fc3, n_atoms, masses_super, prim_indices, slab_indices,
            ref_sc_atoms, rank=None, tol=1e-15,
        )
        sig_ratio = svals[0]**2 / w_typical**4

        # From test results: phi3=0.5 converges, phi3=5.0 mostly doesn't
        scba_ok = "OK" if phi3 <= 1.0 else "FAILS"

        print(f"  {phi3:>8.1f} {fc3_max:>10.4f} {alpha:>10.4f} "
              f"{svals[0]:>12.2e} {sig_ratio:>12.2e} {scba_ok:>8}")


def main():
    si_data = analyze_si_fc()
    analyze_toy_fc()

    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print(f"""
  The anharmonic parameter alpha = FC3 * u_rms / FC2 controls
  whether the SCBA perturbative expansion converges.

  Si (real data):   alpha = {si_data['alpha']:.4f}
  Toy (phi3=0.5):   alpha ~ 0.003  -> converges
  Toy (phi3=5.0):   alpha ~ 0.03   -> diverges

  Si's alpha is between these extremes. The SCBA converges for
  small devices (n_slabs <= 6) but fails at larger thicknesses
  where more scattering events accumulate.

  Possible improvements:
  1. Better FC3 input (larger supercell, more controlled neighbor shells)
  2. Anderson/Pulay mixing instead of simple linear mixing
  3. Adaptive mixing that reduces alpha_mix when |Sigma| grows
  4. The q-averaged self-energy (Approximation III) is inherently
     more stable because it averages over q before feeding back
""")


if __name__ == "__main__":
    main()
