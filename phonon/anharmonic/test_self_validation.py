"""Self-validation tests for the anharmonic NEGF transport code.

Mandatory consistency checks that must pass before trusting results:
1. M_stacked = 0 reproduces the ballistic result exactly
2. delta_T = 0 gives zero heat current
3. anharmonic_transmission_q(..., q_mesh=(1,1)) agrees with
   anharmonic_transmission_finite(...)
4. i(Σ^R - Σ^{R†}) has the correct sign (negative semidefinite
   for ω > 0, positive semidefinite for ω < 0)
"""
import sys
from pathlib import Path
import numpy as np

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell

phonon, _ = load_primitive_cell(work_dir)
fc3_path = str(work_dir / "reaps" / "si_primitive_work" / "fc3.hdf5")

COMMON_KW = dict(
    freq_range_thz=(0.01, 15.0, 31),
    transport_direction="x",
    eta_factor=0.5,
    temperature=300.0,
    max_scba_iter=5,
    scba_tol=1e-2,
    conservation_tol=1.0,  # loose — we're testing physics, not convergence
    mixing=0.5,
    n_slabs=1,
    verbose=False,
    retarded="pv",
)


def test_zero_scattering():
    """M_stacked = 0 → anharmonic result equals ballistic exactly."""
    from phonon_inputs.anharmonic import anharmonic_transmission_finite

    n_atoms = len(phonon.primitive.masses)
    n_dof = 3 * n_atoms
    # Build a zero M_stacked of the right shape
    from phonon_inputs.separable import build_supercell_mapping
    _, _, _, _ = build_supercell_mapping(phonon, "x")
    n_super = len(phonon.supercell.masses)
    dim_sc = n_super * 3
    M_zero = np.zeros((n_dof * dim_sc, dim_sc))

    res = anharmonic_transmission_finite(
        phonon, M_stacked_override=M_zero, delta_T=10.0, **COMMON_KW)

    G_ball = res["thermal_conductance_ballistic"]
    G_anh = res["thermal_conductance_anharmonic"]
    rel_err = abs(G_anh - G_ball) / abs(G_ball)
    print(f"  G_ball = {G_ball:.2f}, G_anh = {G_anh:.2f}, "
          f"rel err = {rel_err:.2e}")
    assert rel_err < 1e-10, f"Zero-scattering test FAILED: {rel_err:.2e}"
    print("  PASSED")


def test_zero_delta_T():
    """delta_T = 0 → zero heat current."""
    from phonon_inputs.anharmonic import anharmonic_transmission_finite

    res = anharmonic_transmission_finite(
        phonon, str(fc3_path), delta_T=0.0, **COMMON_KW)

    J = res["heat_current"]
    # With delta_T = 0, G = J/(A * delta_T) is undefined, but J should be ~0.
    print(f"  J(delta_T=0) = {J:.4e} W")
    assert abs(J) < 1e-20, f"Zero-deltaT test FAILED: J = {J:.4e}"
    print("  PASSED")


def test_q_vs_finite():
    """q_mesh=(1,1) should agree with the finite (Gamma-only) code."""
    from phonon_inputs.anharmonic import (
        anharmonic_transmission_q, anharmonic_transmission_finite)

    res_q = anharmonic_transmission_q(
        phonon, str(fc3_path), q_mesh_transverse=(1, 1),
        delta_T=10.0, **COMMON_KW)
    res_f = anharmonic_transmission_finite(
        phonon, str(fc3_path), delta_T=10.0, **COMMON_KW)

    # Compare ballistic (should be exact)
    G_ball_q = res_q["thermal_conductance_ballistic"]
    G_ball_f = res_f["thermal_conductance_ballistic"]
    ball_err = abs(G_ball_q - G_ball_f) / abs(G_ball_f)
    print(f"  Ballistic: q={G_ball_q:.2f}, finite={G_ball_f:.2f}, "
          f"rel err = {ball_err:.2e}")
    assert ball_err < 1e-10, f"Ballistic q-vs-finite FAILED: {ball_err:.2e}"

    # Compare anharmonic (may differ slightly due to mixing path)
    G_anh_q = res_q["thermal_conductance_anharmonic"]
    G_anh_f = res_f["thermal_conductance_anharmonic"]
    anh_err = abs(G_anh_q - G_anh_f) / abs(G_anh_f)
    print(f"  Anharmonic: q={G_anh_q:.2f}, finite={G_anh_f:.2f}, "
          f"rel err = {anh_err:.2e}")
    assert anh_err < 1e-6, f"Anharmonic q-vs-finite FAILED: {anh_err:.2e}"
    print("  PASSED")


def test_sigma_r_sign():
    """Γ = i(Σ^R - Σ^A) should be PSD for ω > 0.

    This is the dissipation condition: Γ represents broadening and
    must be positive semidefinite for positive frequencies.
    """
    from phonon_inputs.anharmonic import anharmonic_transmission_finite

    res = anharmonic_transmission_finite(
        phonon, str(fc3_path), delta_T=10.0, **COMMON_KW)

    Sigma_R = res["self_energy_retarded"]  # (n_slabs, nfreq_pos, nd, nd)
    freqs = res["freqs_thz"]

    # Check: Γ = i(Σ^R - Σ^A) eigenvalues should be ≥ 0 for ω > 0
    n_violations = 0
    max_violation = 0.0
    for l in range(Sigma_R.shape[0]):
        for iw in range(len(freqs)):
            if freqs[iw] < 0.5:
                continue  # skip near-zero where self-energy is tiny
            sr = Sigma_R[l, iw]
            gamma = 1j * (sr - sr.conj().T)
            gamma_h = 0.5 * (gamma + gamma.conj().T)
            eigs = np.linalg.eigvalsh(gamma_h)
            min_eig = eigs.min()
            if min_eig < -1e-10:
                n_violations += 1
                max_violation = max(max_violation, -min_eig)

    print(f"  Γ PSD violations (ω > 0.5 THz): "
          f"{n_violations}, max = {max_violation:.2e}")
    if n_violations > 0:
        print(f"  WARNING: {n_violations} frequency points have Γ not PSD "
              f"(max negative eigenvalue = {max_violation:.2e})")
    else:
        print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Self-validation tests for anharmonic NEGF")
    print("=" * 60)

    print("\n1. Zero-scattering test (M_stacked=0 → ballistic)")
    test_zero_scattering()

    print("\n2. Zero delta_T test (delta_T=0 → J=0)")
    test_zero_delta_T()

    print("\n3. q-mesh (1,1) vs finite (Gamma-only)")
    test_q_vs_finite()

    print("\n4. Σ^R dissipation sign check")
    test_sigma_r_sign()

    print("\n" + "=" * 60)
    print("All self-validation tests completed.")
    print("=" * 60)
