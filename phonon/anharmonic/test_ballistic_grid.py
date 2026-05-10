"""Compare ballistic G on positive-only vs symmetric grid.

Verifies that the symmetric grid gives the same ballistic result
as the old positive-only grid (which was validated against Guo Table III).
"""
import sys
from pathlib import Path
import numpy as np

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs.constants import CONVERSION_THZ2, HBAR_SI, KB_SI, THZ_TO_RAD
from phonon_inputs.convention import get_btd_blocks
from phonon_inputs.anharmonic import _build_device_hamiltonian
from phonon_inputs.validation import _ballistic_transmission

phonon, _ = load_primitive_cell(work_dir)
n_atoms = len(phonon.primitive.masses)
n_dof = 3 * n_atoms
nkx, nky = 4, 4

# Cross-sectional area
lattice = phonon.primitive.cell
perp_idx = [1, 2]  # transport along x
a1 = lattice[perp_idx[0]]
a2 = lattice[perp_idx[1]]
A_c = np.linalg.norm(np.cross(a1, a2)) * 1e-20

T = 300.0
delta_T = 10.0
T_L, T_R = T + delta_T/2, T - delta_T/2

def bose_einstein(freq_thz_arr, T):
    omega_rad_s = freq_thz_arr * THZ_TO_RAD  # signed
    x = HBAR_SI * omega_rad_s / (KB_SI * T)
    n = np.zeros_like(x)
    valid = np.abs(x) > 1e-12
    n[valid] = 1.0 / np.expm1(x[valid])
    return n


def compute_G_ball(freqs, label):
    """Compute ballistic thermal conductance on the given frequency grid."""
    nfreq = len(freqs)
    dw = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    eta_w = dw * 0.5
    z2_arr = freqs**2 + 2j * freqs * eta_w

    n_L = bose_einstein(freqs, T_L)
    n_R = bose_einstein(freqs, T_R)

    # q-mesh
    q_points = [(i/nkx, j/nky) for i in range(nkx) for j in range(nky)]
    n_kpts = len(q_points)

    trans = np.zeros(nfreq)
    for qx, qy in q_points:
        H_00, H_01 = get_btd_blocks(
            phonon, (qx, qy), transport_direction="x",
            conversion_factor=CONVERSION_THZ2)
        H_D = _build_device_hamiltonian(H_00, H_01, 1)
        N_D = H_D.shape[0]
        H_LD = np.zeros((n_dof, N_D), dtype=complex)
        H_LD[:, :n_dof] = H_01
        H_DR = np.zeros((N_D, n_dof), dtype=complex)
        H_DR[-n_dof:, :] = H_01
        for iw, z2 in enumerate(z2_arr):
            trans[iw] += _ballistic_transmission(z2, H_D, H_00, H_01, H_00, H_01, H_LD, H_DR)
    trans /= n_kpts

    omega_rad = freqs * THZ_TO_RAD
    spectral_J = HBAR_SI * omega_rad * (n_L - n_R) * trans

    # Which frequencies to integrate?
    pos_mask = freqs >= 0.0
    J_total = np.sum(spectral_J[pos_mask]) * dw * 1e12
    G = J_total / (A_c * delta_T)

    print(f"\n--- {label} ---")
    print(f"  Grid: {nfreq} pts, [{freqs[0]:.2f}, {freqs[-1]:.2f}], dw={dw:.4f}")
    print(f"  eta_w = {eta_w:.4e} THz")
    print(f"  max T(w) = {trans.max():.4f}")
    print(f"  Sum spectral_J (pos) = {np.sum(spectral_J[pos_mask]):.6e}")
    print(f"  G_ball = {G/1e6:.2f} MW/(m^2 K)")

    # Also check: what if we integrate ALL frequencies (not just positive)?
    J_all = np.sum(spectral_J) * dw * 1e12
    G_all = J_all / (A_c * delta_T)
    print(f"  G_ball (all freqs) = {G_all/1e6:.2f} MW/(m^2 K)")

    # Check symmetry properties
    if freqs[0] < 0:
        mid = (nfreq - 1) // 2
        print(f"  T(w=0) = {trans[mid]:.4f}")
        print(f"  T symmetric? max|T(w) - T(-w)| = {np.max(np.abs(trans - trans[::-1])):.2e}")
        print(f"  spectral_J antisymmetric? max|J(w)+J(-w)| = {np.max(np.abs(spectral_J + spectral_J[::-1])):.2e}")

    return G, trans, spectral_J


print("=" * 60)
print("Test: Ballistic G on different grids")
print(f"Expected: ~1050 MW/(m^2 K) (Guo Table III, Si [100] 4x4)")
print("=" * 60)

# Grid 1: Old positive-only grid (known-good)
freqs_old = np.linspace(0.5, 15.0, 51)
G_old, _, _ = compute_G_ball(freqs_old, "OLD: positive-only [0.5, 15.0]")

# Grid 2: Positive-only starting from ~0
freqs_pos0 = np.linspace(0.0, 15.0, 51)
G_pos0, _, _ = compute_G_ball(freqs_pos0, "Positive [0.0, 15.0], 51 pts")

# Grid 3: Symmetric grid (current code)
nfreq_pos = 51
freqs_pos = np.linspace(0.0, 15.0, nfreq_pos)
freqs_sym = np.concatenate((-freqs_pos[:0:-1], freqs_pos))
G_sym, _, _ = compute_G_ball(freqs_sym, "SYMMETRIC [-15.0, 15.0]")

# Grid 4: Positive-only with more points for convergence check
freqs_fine = np.linspace(0.3, 15.0, 101)
G_fine, _, _ = compute_G_ball(freqs_fine, "Positive [0.3, 15.0], 101 pts")

# Grid 5: Positive with small fmin
freqs_small = np.linspace(0.1, 15.0, 51)
G_small, _, _ = compute_G_ball(freqs_small, "Positive [0.1, 15.0], 51 pts")

print("\n" + "=" * 60)
print("Summary:")
print(f"  Old pos-only [0.5, 15]:  {G_old/1e6:.2f} MW/(m^2 K)")
print(f"  Pos [0.0, 15]:           {G_pos0/1e6:.2f} MW/(m^2 K)")
print(f"  Symmetric [-15, 15]:     {G_sym/1e6:.2f} MW/(m^2 K)")
print(f"  Pos fine [0.3, 15]:      {G_fine/1e6:.2f} MW/(m^2 K)")
print(f"  Pos [0.1, 15]:           {G_small/1e6:.2f} MW/(m^2 K)")
print(f"  Expected (Guo):          ~1050 MW/(m^2 K)")
