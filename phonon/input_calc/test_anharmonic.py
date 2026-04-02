"""Test the anharmonic phonon-phonon NEGF reference implementation.

Loads Si force constants (harmonic + FC3) and runs the SCBA to compute
the effect of phonon-phonon scattering on the transmission function.

Key issue: the FC3 was computed for the 2-atom FCC primitive cell while
the phonopy harmonic FC are for the 8-atom conventional cubic cell.
This script remaps the FC3 to the conventional cell basis.
"""

import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setup paths
work_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(work_dir.parents[1] / "src"))

from phonon_inputs.structure import load_phonopy_calculation
from phonon_inputs.force_constants import load_fc3_thirdorder
from phonon_inputs.anharmonic import (
    anharmonic_transmission,
    _assemble_fc3_full,
    CONVERSION_FC3,
    HBAR_SI,
)
from phonon_inputs.constants import CONVERSION


# ---------------------------------------------------------------------------
# FC3 remapping: FCC 2-atom -> conventional 8-atom
# ---------------------------------------------------------------------------

def remap_fc3_to_conventional(fc3_data, fcc_cell, fcc_frac, conv_cell, conv_frac):
    """Remap FC3 from FCC 2-atom primitive to conventional 8-atom cell.

    The phonopy conventional cell has an origin shift of (-a/8, -a/8, -a/8)
    relative to the standard diamond convention where FCC sublattice atoms
    sit at (0,0,0). This shift is detected and accounted for.

    Parameters
    ----------
    fc3_data : dict
        FC3 from load_fc3_thirdorder() (2-atom FCC basis).
    fcc_cell : (3, 3) array
        FCC primitive lattice vectors (rows), scaled to phonopy lattice constant.
    fcc_frac : (2, 3) array
        Fractional positions of atoms in FCC cell.
    conv_cell : (3, 3) array
        Conventional cell lattice vectors (rows).
    conv_frac : (n, 3) array
        Fractional positions of atoms in conventional cell.
    """
    fcc_cart = fcc_frac @ fcc_cell
    conv_cart = conv_frac @ conv_cell
    conv_inv = np.linalg.inv(conv_cell.T)
    fcc_inv = np.linalg.inv(fcc_cell.T)
    n_conv = len(conv_frac)

    # Detect origin shift: standard diamond has FCC atom at (0,0,0) conv frac.
    # Phonopy has it at conv_frac[0] = (7/8, 7/8, 7/8) typically.
    # The shift to add to phonopy Cartesian positions to get standard positions:
    # We find it by matching the first conv atom to an FCC lattice point.
    # For diamond Si, shift = (a/8, a/8, a/8) where a = conventional lattice constant.
    a_conv = conv_cell[0, 0]
    shift = np.array([a_conv / 8, a_conv / 8, a_conv / 8])

    # Verify the shift works for all atoms
    mapping_ok = True
    for i in range(n_conv):
        shifted = conv_cart[i] + shift
        found = False
        for j in range(2):
            diff = shifted - fcc_cart[j]
            fcc_frac_diff = fcc_inv @ diff
            if np.allclose(fcc_frac_diff, np.round(fcc_frac_diff), atol=0.05):
                found = True
                break
        if not found:
            mapping_ok = False
            break

    if not mapping_ok:
        # Try the negative shift
        shift = -shift
        mapping_ok = True
        for i in range(n_conv):
            shifted = conv_cart[i] + shift
            found = False
            for j in range(2):
                diff = shifted - fcc_cart[j]
                fcc_frac_diff = fcc_inv @ diff
                if np.allclose(fcc_frac_diff, np.round(fcc_frac_diff), atol=0.05):
                    found = True
                    break
            if not found:
                mapping_ok = False
                break

    if not mapping_ok:
        raise ValueError("Could not determine origin shift. Non-standard diamond cell?")

    print(f"  Origin shift: ({shift[0]:.4f}, {shift[1]:.4f}, {shift[2]:.4f}) A")

    # Print the mapping
    for i in range(n_conv):
        shifted = conv_cart[i] + shift
        for j in range(2):
            diff = shifted - fcc_cart[j]
            fcc_frac_diff = fcc_inv @ diff
            if np.allclose(fcc_frac_diff, np.round(fcc_frac_diff), atol=0.05):
                R = np.round(fcc_frac_diff).astype(int)
                print(f"    conv {i} -> FCC atom {j}, cell ({R[0]},{R[1]},{R[2]})")
                break

    def _fcc_to_conv(fcc_atom, fcc_R_cart):
        """Map FCC (atom, R_cart) -> conv (atom_index, cell_int_vector)."""
        pos = fcc_cart[fcc_atom] + fcc_R_cart
        conv_pos = pos - shift
        for i in range(n_conv):
            diff = conv_pos - conv_cart[i]
            diff_frac = conv_inv @ diff
            cell_idx = np.round(diff_frac).astype(int)
            err = np.linalg.norm(diff - cell_idx.astype(float) @ conv_cell)
            if err < 0.5:
                return i, cell_idx
        raise ValueError(f"Failed: FCC atom {fcc_atom}, R={fcc_R_cart}")

    # Remap FC3 blocks
    # Scale factor for FC3 cell vectors (thirdorder lattice -> phonopy lattice)
    a_fcc_orig = 2 * np.linalg.norm(fcc_cell[0]) / np.sqrt(2)  # recover original a
    scale = a_conv / a_fcc_orig if abs(a_conv / a_fcc_orig - 1.0) > 1e-6 else 1.0

    new_blocks = []
    for block in fc3_data["blocks"]:
        R_j = block["cell_j"] * scale
        R_k = block["cell_k"] * scale

        ai_conv, Ri = _fcc_to_conv(block["atom_i"], np.zeros(3))
        aj_conv, Rj = _fcc_to_conv(block["atom_j"], R_j)
        ak_conv, Rk = _fcc_to_conv(block["atom_k"], R_k)

        new_R_j = (Rj - Ri).astype(float) @ conv_cell
        new_R_k = (Rk - Ri).astype(float) @ conv_cell

        new_blocks.append({
            "cell_j": new_R_j,
            "cell_k": new_R_k,
            "atom_i": ai_conv,
            "atom_j": aj_conv,
            "atom_k": ak_conv,
            "tensor": block["tensor"],
        })

    return {"n_blocks": len(new_blocks), "blocks": new_blocks}


if __name__ == "__main__":

    # ---------------------------------------------------------------------------
    # 1. Load Si harmonic force constants from phonopy
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("Loading Si phonopy calculation...")
    phonon_si = load_phonopy_calculation(
        phonopy_yaml=work_dir / "scf_disp" / "phonopy_disp.yaml",
        force_sets_filename=work_dir / "scf_disp" / "FORCE_SETS",
        calculator="qe",
    )
    n_atoms_conv = len(phonon_si.primitive.masses)
    masses = phonon_si.primitive.masses
    a_conv = phonon_si.unitcell.cell[0, 0]
    print(f"  Conventional cell: {n_atoms_conv} atoms, a = {a_conv:.4f} A")

    # -----------------------------------------------------------------------
    # 2. Load FC3 and remap to conventional cell
    # -----------------------------------------------------------------------
    print("\nLoading FC3...")
    fc3_data = load_fc3_thirdorder(work_dir / "fc3_si" / "FORCE_CONSTANTS_3RD")
    print(f"  FC3 blocks (FCC 2-atom): {fc3_data['n_blocks']}")

    print("\nRemapping FC3 to conventional cell...")
    fcc_cell = np.array([
        [0.0, a_conv / 2, a_conv / 2],
        [a_conv / 2, 0.0, a_conv / 2],
        [a_conv / 2, a_conv / 2, 0.0],
    ])
    fcc_frac = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])

    fc3_conv = remap_fc3_to_conventional(
        fc3_data, fcc_cell, fcc_frac,
        phonon_si.unitcell.cell, phonon_si.unitcell.scaled_positions,
    )
    print(f"  Remapped: {fc3_conv['n_blocks']} blocks")

    # -----------------------------------------------------------------------
    # 3. Assemble and check FC3 in conventional basis
    # -----------------------------------------------------------------------
    print("\nAssembling FC3 tensor (conventional cell)...")
    conv_cell = phonon_si.unitcell.cell
    Phi3_full, n_inc, n_tot = _assemble_fc3_full(
        fc3_conv, n_atoms_conv, masses, conv_cell, "z"
    )
    n_nz = np.count_nonzero(Phi3_full)
    print(f"  delta_l=0: {n_inc}/{n_tot} blocks, {n_nz} non-zero / {Phi3_full.size}")
    print(f"  Max |Phi3| (mass-weighted): {np.max(np.abs(Phi3_full)):.4e}")

    # -----------------------------------------------------------------------
    # 4. Run the anharmonic SCBA (small test)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running anharmonic SCBA...")
    print("  4x4 q-mesh, 51 freq points, 20 SCBA iterations")
    print("=" * 60)

    t0 = time.time()
    result = anharmonic_transmission(
        phonon_si,
        fc3_conv,
        q_mesh_transverse=(4, 4),
        freq_range_thz=(0.5, 15.0, 51),
        transport_direction="z",
        eta_factor=0.5,
        temperature=300.0,
        max_scba_iter=20,
        scba_tol=0.005,
        mixing=0.3,
        fc3_mode="full",
        verbose=True,
    )
    t1 = time.time()

    print(f"\nCompleted in {t1 - t0:.1f} s")
    print(f"  Ballistic max T:    {result['transmission_ballistic'].max():.4f}")
    G_ball = result['thermal_conductance_ballistic']
    G_anh = result['thermal_conductance_anharmonic']
    print(f"  G_ballistic:        {G_ball/1e6:.2f} MW/(m^2 K)")
    print(f"  G_anharmonic:       {G_anh/1e6:.2f} MW/(m^2 K)")
    print(f"  Heat flow conserv.: {result['heat_flow_conservation']:.2e}")
    if G_ball > 0:
        print(f"  Reduction:          {(1 - G_anh/G_ball)*100:.1f}%")

    # -----------------------------------------------------------------------
    # 5. Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    freqs = result["freqs_thz"]

    ax = axes[0]
    ax.plot(freqs, result["spectral_heat_current_ballistic"], "b-", lw=1.5,
            label="Ballistic")
    ax.plot(freqs, result["spectral_heat_current"], "r--", lw=1.5,
            label="Anharmonic (SCBA)")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Spectral heat current (W)")
    ax.set_title("Si bulk: ballistic vs anharmonic (300 K)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, None)

    ax = axes[1]
    J_ball = result["spectral_heat_current_ballistic"]
    J_anh = result["spectral_heat_current"]
    mask = np.abs(J_ball) > np.max(np.abs(J_ball)) * 0.01
    ratio = np.ones_like(freqs)
    ratio[mask] = J_anh[mask] / J_ball[mask]
    ax.plot(freqs[mask], ratio[mask], "k-", lw=1.5)
    ax.axhline(y=1.0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("J_anharmonic / J_ballistic")
    ax.set_title("Heat current ratio")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 1.5)

    plt.tight_layout()
    fig.savefig(work_dir / "anharmonic_test.png", dpi=150)
    plt.close("all")
    print(f"\nSaved anharmonic_test.png")
