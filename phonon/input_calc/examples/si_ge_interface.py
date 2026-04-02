"""Ballistic phonon transmission through a Si/Ge interface.

Mass-mismatch model: Si force constants everywhere, Ge masses in the
right contact. Following Latour et al., PRB 96, 104310 (2017) and
Tian et al., PRB 86, 235304 (2012).

Requires an existing phonopy QE calculation for Si in scf_disp/.
"""

import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

work_dir = Path(__file__).resolve().parent
parent_dir = work_dir.parent
sys.path.insert(0, str(parent_dir))

from phonon_inputs.structure import load_phonopy_calculation, clone_with_masses
from phonon_inputs.validation import (
    interface_transmission,
    reference_transmission,
    thermal_conductance,
)

# ---------------------------------------------------------------------------
# 1. Load Si force constants from existing phonopy QE calculation
# ---------------------------------------------------------------------------
print("Loading Si force constants (phonopy QE native)...")
phonon_si = load_phonopy_calculation(
    phonopy_yaml=parent_dir / "scf_disp" / "phonopy_disp.yaml",
    force_sets_filename=parent_dir / "scf_disp" / "FORCE_SETS",
    calculator="qe",
)

a_si = phonon_si.unitcell.cell[0, 0]
print(f"  Si lattice constant: a = {a_si:.4f} A")
phonon_si.run_qpoints([[0, 0, 0]])
freqs_si = phonon_si.get_qpoints_dict()["frequencies"][0]
print(f"  Si Gamma max: {freqs_si.max():.2f} THz")

# ---------------------------------------------------------------------------
# 2. Create Ge and interface phonopy objects (mass-mismatch model)
# ---------------------------------------------------------------------------
# Right contact: Si FC with Ge masses
phonon_ge = clone_with_masses(phonon_si, symbols=["Ge"] * 8)
phonon_ge.run_qpoints([[0, 0, 0]])
freqs_ge = phonon_ge.get_qpoints_dict()["frequencies"][0]
print(f"  Ge Gamma max (mass-mismatch): {freqs_ge.max():.2f} THz")

# Device: Si FC with mixed masses (Si on z<0.5, Ge on z>=0.5)
frac_positions = phonon_si.unitcell.scaled_positions
frac_z = frac_positions[:, 2]
ifc_symbols = ["Ge" if z >= 0.5 else "Si" for z in frac_z]
phonon_ifc = clone_with_masses(phonon_si, symbols=ifc_symbols)
phonon_ifc.run_qpoints([[0, 0, 0]])
freqs_ifc = phonon_ifc.get_qpoints_dict()["frequencies"][0]
print(f"  Interface Gamma: min={freqs_ifc.min():.4f}, max={freqs_ifc.max():.2f} THz")
print(f"  Interface symbols: {ifc_symbols}")

# ---------------------------------------------------------------------------
# 3. Compute Si/Ge interface transmission
# ---------------------------------------------------------------------------
nk = 20
print(f"\nComputing Si/Ge interface transmission ({nk}x{nk} k-mesh)...")
freqs_thz, trans = interface_transmission(
    phonon_left=phonon_si,
    phonon_right=phonon_ge,
    phonon_device=phonon_ifc,
    q_mesh_transverse=(nk, nk),
    freq_range_thz=(0.01, 16.0, 201),
    transport_direction="z",
)
print(f"Max transmission: {trans.max():.4f}")

# ---------------------------------------------------------------------------
# 4. Pure Si reference transmission
# ---------------------------------------------------------------------------
print("\nComputing pure Si reference transmission...")
freqs_thz_si, trans_si = reference_transmission(
    phonon_si,
    q_mesh_transverse=(nk, nk),
    freq_range_thz=(0.01, 16.0, 201),
    transport_direction="z",
)
print(f"Max Si transmission: {trans_si.max():.4f}")

# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(freqs_thz, trans, "b-", lw=1.5, label="Si/Ge interface")
ax.plot(freqs_thz_si, trans_si, "r--", lw=1, alpha=0.5, label="Pure Si")
ax.set_xlabel("Frequency (THz)")
ax.set_ylabel("Transmission")
ax.set_title("Ballistic transmission: Si/Ge interface (mass-mismatch)")
ax.set_xlim(0, 16)
ax.set_ylim(0, None)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
transmissivity = np.zeros_like(trans)
mask = trans_si > 0.01
transmissivity[mask] = trans[mask] / trans_si[mask]
ax.plot(freqs_thz, transmissivity, "b-", lw=1.5)
ax.set_xlabel("Frequency (THz)")
ax.set_ylabel(r"$\Gamma_\mathrm{Si \to Ge}$")
ax.set_title("Interfacial transmissivity")
ax.set_xlim(0, 16)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(work_dir / "si_ge_interface_transmission.png", dpi=150)
plt.close("all")
print("\nSaved si_ge_interface_transmission.png")

# ---------------------------------------------------------------------------
# 6. Thermal conductance
# ---------------------------------------------------------------------------
from phonon_inputs.constants import THZ_TO_RAD, HBAR_EV, KB_EV, EV_TO_J

T_kelvin = 300.0
G = thermal_conductance(
    freqs_thz, trans, T_kelvin, phonon_si.primitive.cell, "z",
)
print(f"\nSi/Ge interface thermal conductance @ {T_kelvin} K: {G / 1e6:.1f} MW/(m^2 K)")

# Reference values from literature:
# Latour et al. (PRB 96): G = 2.10e8 W/(m^2 K)
# Tian et al. (PRB 86): G = 2.09e8 W/(m^2 K)  (DFT)
# Tian et al. (PRB 86): G = 2.8e8 W/(m^2 K)   (SW potential)
