"""Si/Ge interface: quatrex NEGF vs reference Sancho-Rubio comparison.

Runs both bulk Si and Si/Ge interface (mass-mismatch model) through
quatrex's PhononSolver, compares with the reference Caroli formula,
and saves a comparison plot.

Requires:
- Existing phonopy QE calculation for Si in scf_disp/
- quatrex installed (src/ on PYTHONPATH)
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
work_dir = Path(__file__).resolve().parent
parent_dir = work_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir.parents[1] / "src"))

from phonon_inputs.structure import load_phonopy_calculation, clone_with_masses
from phonon_inputs.convention import extract_blocks
from phonon_inputs.quatrex_writer import write_all
from phonon_inputs.config import QuatrexOutputConfig
from phonon_inputs.validation import (
    reference_transmission,
    interface_transmission,
    thermal_conductance,
)
from phonon_inputs.constants import THZ_TO_RAD, HBAR_EV, KB_EV, EV_TO_J

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
nk = 20  # transverse k-mesh (nk x nk)
n_freq = 101  # reduced from 201 to fit 20x20 k-mesh in memory
freq_range = (0.01, 16.0, n_freq)
num_transport_cells = 5  # total cells in BTD matrix
num_left_cells = 2
num_right_cells = 2
T_L, T_R = 301.0, 299.0
T_ref = 300.0

# ---------------------------------------------------------------------------
# 1. Load Si force constants from phonopy QE calculation
# ---------------------------------------------------------------------------
print("Loading Si force constants...")
phonon_si = load_phonopy_calculation(
    phonopy_yaml=parent_dir / "scf_disp" / "phonopy_disp.yaml",
    force_sets_filename=parent_dir / "scf_disp" / "FORCE_SETS",
    calculator="qe",
)
a_si = phonon_si.unitcell.cell[0, 0]
print(f"  Si lattice constant: a = {a_si:.4f} A")

# ---------------------------------------------------------------------------
# 2. Create Ge and interface phonopy objects (mass-mismatch)
# ---------------------------------------------------------------------------
phonon_ge = clone_with_masses(phonon_si, symbols=["Ge"] * 8)

frac_z = phonon_si.unitcell.scaled_positions[:, 2]
ifc_symbols = ["Ge" if z >= 0.5 else "Si" for z in frac_z]
phonon_ifc = clone_with_masses(phonon_si, symbols=ifc_symbols)

print(f"  Interface symbols: {ifc_symbols}")

# ---------------------------------------------------------------------------
# 3. Extract Convention B real-space blocks for each material
# ---------------------------------------------------------------------------
print("Extracting dynamical matrix blocks...")
blocks_si = extract_blocks(phonon_si, q_mesh=(3, 3, 3))
blocks_ge = extract_blocks(phonon_ge, q_mesh=(3, 3, 3))
blocks_ifc = extract_blocks(phonon_ifc, q_mesh=(3, 3, 3))

print(f"  Si: {len(blocks_si)} blocks, Ge: {len(blocks_ge)} blocks, "
      f"Interface: {len(blocks_ifc)} blocks")

# ---------------------------------------------------------------------------
# 4. Write quatrex inputs: bulk Si (homogeneous)
# ---------------------------------------------------------------------------
print("\nWriting quatrex inputs for bulk Si...")
si_config = QuatrexOutputConfig(
    output_dir=str(work_dir / "outputs" / "quatrex_si"),
    num_transport_cells=num_transport_cells,
    kpoint_grid=[nk, nk, 1],
    neighbor_cell_cutoff=[1, 1, 1],
    eta=1e-8,
    left_temperature=T_L,
    right_temperature=T_R,
)
si_dir = write_all(
    cell=phonon_si.unitcell,
    blocks=blocks_si,
    config=si_config,
    transport_direction="z",
)
print(f"  -> {si_dir}")

# ---------------------------------------------------------------------------
# 5. Write quatrex inputs: Si/Ge interface (heterogeneous)
# ---------------------------------------------------------------------------
print("Writing quatrex inputs for Si/Ge interface...")
ifc_config = QuatrexOutputConfig(
    output_dir=str(work_dir / "outputs" / "quatrex_interface"),
    num_transport_cells=num_transport_cells,
    kpoint_grid=[nk, nk, 1],
    neighbor_cell_cutoff=[1, 1, 1],
    eta=1e-8,
    left_temperature=T_L,
    right_temperature=T_R,
    num_left_cells=num_left_cells,
    num_right_cells=num_right_cells,
    left_matrix="dynamical_matrix_si",
    right_matrix="dynamical_matrix_ge",
)
ifc_dir = write_all(
    cell=phonon_si.unitcell,
    blocks=blocks_ifc,
    config=ifc_config,
    transport_direction="z",
    left_blocks=blocks_si,
    right_blocks=blocks_ge,
)
print(f"  -> {ifc_dir}")

# ---------------------------------------------------------------------------
# 6. Run quatrex for both cases
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")

from qttools import xp, sparse
from qttools.comm import comm
from qttools.utils.mpi_utils import get_local_slice
from quatrex.core.config import parse_config
from quatrex.core.statistics import bose_einstein
from quatrex.phonon.solver import PhononSolver
from quatrex.device.inputs import load_matrix

# Frequency grid
freqs_thz = np.linspace(*freq_range)
omega_rad = freqs_thz * THZ_TO_RAD
omega_sq = omega_rad ** 2

# Eta: match reference solver broadening
dw = (freqs_thz[1] - freqs_thz[0]) * THZ_TO_RAD
eta_match = dw ** 2 * 0.5


def run_quatrex_phonon(config_dir, label):
    """Run quatrex PhononSolver and return per-frequency transmission."""
    print(f"\n--- quatrex: {label} ---")
    saved_cwd = os.getcwd()
    os.chdir(config_dir)

    try:
        config = parse_config("quatrex_config.toml")
        config.phonon.eta = eta_match

        # Load DM for sparsity pattern
        dm_temp, sp = load_matrix(
            config=config,
            matrix_name="dynamical_matrix",
            sparsity_pattern=None,
            shift_kpoints=False,
        )
        sparsity_pattern = sp.copy()
        del dm_temp

        # Create solver
        solver = PhononSolver(config, omega_sq, sparsity_pattern)

        local_freqs = get_local_slice(omega_sq, comm.stack)
        local_omega_rad = np.sqrt(np.abs(local_freqs))
        hbar_omega_eV = HBAR_EV * local_omega_rad
        solver.left_occupancies = bose_einstein(hbar_omega_eV, T_L)
        solver.right_occupancies = bose_einstein(hbar_omega_eV, T_R)
        solver.left_temperature = T_L
        solver.right_temperature = T_R

        print(f"  block sizes: {solver.block_sizes}")
        print(f"  eta = {config.phonon.eta:.2e} (rad/s)^2")

        # Allocate zero self-energies (ballistic)
        DSDB = type(solver.dynamical_matrix)
        stack_shape = omega_sq.shape + tuple(
            int(k) for k in config.device.kpoint_grid if k > 1
        )

        g_lesser = DSDB.from_sparray(
            sp.astype(xp.complex128),
            block_sizes=solver.block_sizes,
            global_stack_shape=stack_shape,
        )
        g_greater = DSDB.zeros_like(g_lesser)
        g_retarded = DSDB.zeros_like(g_lesser)
        sse_lesser = DSDB.zeros_like(g_lesser)
        sse_greater = DSDB.zeros_like(g_lesser)
        sse_retarded = DSDB.zeros_like(g_lesser)

        out = (g_lesser, g_greater, g_retarded)
        print("  Solving...")
        solver.solve(sse_lesser, sse_greater, sse_retarded, out)
        print("  Done.")

        # Extract transmission
        # Use the last interface (closest to right contact) for current.
        # In the RGF backward sweep, the last interface gives the most
        # numerically accurate Meir-Wingreen current.
        current = solver.meir_wingreen_current
        dn = solver.left_occupancies - solver.right_occupancies
        n_kpoints = nk * nk
        n_interfaces = current.shape[-1]

        spectral_current = current[:, :, :, n_interfaces - 1].real
        total_current = np.sum(spectral_current, axis=(1, 2))

        mask = np.abs(dn) > 1e-20
        trans = np.zeros(n_freq)
        trans[mask] = total_current[mask] / dn[mask] / n_kpoints

        # Current conservation check
        n_interfaces = current.shape[-1]
        for i in range(n_interfaces):
            ci = np.sum(current[:, :, :, i].real, axis=(1, 2))
            max_ci = np.max(np.abs(ci))
            print(f"  Interface {i}: max |I| = {max_ci:.4e}")

        print(f"  Max transmission: {trans.max():.4f}")
        return trans

    finally:
        os.chdir(saved_cwd)


# Run bulk Si
trans_si_quatrex = run_quatrex_phonon(si_dir, "Bulk Si")

# Run Si/Ge interface
trans_ifc_quatrex = run_quatrex_phonon(ifc_dir, "Si/Ge interface")

# ---------------------------------------------------------------------------
# 7. Reference Sancho-Rubio transmission
# ---------------------------------------------------------------------------
print("\n--- Reference: Bulk Si ---")
freqs_ref, trans_si_ref = reference_transmission(
    phonon_si,
    q_mesh_transverse=(nk, nk),
    freq_range_thz=freq_range,
    transport_direction="z",
)
print(f"  Max transmission: {trans_si_ref.max():.4f}")

print("\n--- Reference: Si/Ge interface ---")
freqs_ref_ifc, trans_ifc_ref = interface_transmission(
    phonon_left=phonon_si,
    phonon_right=phonon_ge,
    phonon_device=phonon_ifc,
    q_mesh_transverse=(nk, nk),
    freq_range_thz=freq_range,
    transport_direction="z",
)
print(f"  Max transmission: {trans_ifc_ref.max():.4f}")

# ---------------------------------------------------------------------------
# 8. Thermal conductance
# ---------------------------------------------------------------------------
lattice = phonon_si.primitive.cell

G_si_ref = thermal_conductance(freqs_ref, trans_si_ref, T_ref, lattice, "z")
G_si_qtx = thermal_conductance(freqs_thz, trans_si_quatrex, T_ref, lattice, "z")
G_ifc_ref = thermal_conductance(freqs_ref_ifc, trans_ifc_ref, T_ref, lattice, "z")
G_ifc_qtx = thermal_conductance(freqs_thz, trans_ifc_quatrex, T_ref, lattice, "z")

print(f"\nThermal conductance @ {T_ref} K:")
print(f"  Bulk Si:     ref={G_si_ref/1e6:.1f}, quatrex={G_si_qtx/1e6:.1f} MW/(m^2 K)")
print(f"  Si/Ge ifc:   ref={G_ifc_ref/1e6:.1f}, quatrex={G_ifc_qtx/1e6:.1f} MW/(m^2 K)")

# ---------------------------------------------------------------------------
# 9. Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bulk Si
ax = axes[0]
ax.plot(freqs_ref, trans_si_ref, "b-", lw=1.5, label="Reference (Caroli)")
ax.plot(freqs_thz, trans_si_quatrex, "r--", lw=1.5, label="quatrex NEGF")
ax.set_xlabel("Frequency (THz)")
ax.set_ylabel("Transmission")
ax.set_title("Bulk Si (homogeneous)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 16)
ax.set_ylim(0, None)

# Si/Ge interface
ax = axes[1]
ax.plot(freqs_ref_ifc, trans_ifc_ref, "b-", lw=1.5, label="Reference (Caroli)")
ax.plot(freqs_thz, trans_ifc_quatrex, "r--", lw=1.5, label="quatrex NEGF")

# Overlay digitized Guo et al. (2020) data if available
lit_file = work_dir / "literature_fig5b.npz"
if lit_file.exists():
    lit = np.load(lit_file)
    if len(lit["guo_freq"]) > 0:
        ax.plot(lit["guo_freq"], lit["guo_trans"], "g-", lw=1, alpha=0.7,
                label="Guo et al. (2020)")

ax.set_xlabel("Frequency (THz)")
ax.set_ylabel("Transmission")
ax.set_title("Si/Ge interface (heterogeneous)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 16)
ax.set_ylim(0, None)

plt.tight_layout()
fig.savefig(work_dir / "si_ge_quatrex_comparison.png", dpi=150)
plt.close("all")
print(f"\nSaved si_ge_quatrex_comparison.png")
