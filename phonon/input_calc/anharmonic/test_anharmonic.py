"""Test the anharmonic phonon-phonon NEGF reference implementation.

Loads Si force constants (FC2 + FC3) from phono3py for the 2-atom FCC
primitive cell and runs the SCBA to compute the effect of phonon-phonon
scattering on the transmission function.

Requires fc3_prim/fc3.hdf5 (run fc3-reap first).
"""

import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent  # input_calc/
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs.anharmonic import (
    anharmonic_transmission,
    _assemble_fc3_full,
)


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 1. Load Si primitive cell + FC3
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading Si primitive cell (phono3py)...")
    phonon, fc3_data = load_primitive_cell(work_dir)

    n_atoms = len(phonon.primitive.masses)
    masses = phonon.primitive.masses
    prim_cell = phonon.primitive.cell
    print(f"  Primitive cell: {n_atoms} atoms")
    print(f"  FC3 blocks: {fc3_data['n_blocks']}")

    # ------------------------------------------------------------------
    # 2. Assemble and check FC3
    # ------------------------------------------------------------------
    print("\nAssembling FC3 tensor (primitive cell)...")
    Phi3_full, n_inc, n_tot = _assemble_fc3_full(
        fc3_data, n_atoms, masses, prim_cell, "x"
    )
    n_nz = np.count_nonzero(Phi3_full)
    print(f"  delta_l=0: {n_inc}/{n_tot} blocks, {n_nz} non-zero / {Phi3_full.size}")
    print(f"  Max |Phi3| (mass-weighted): {np.max(np.abs(Phi3_full)):.4e}")

    # ------------------------------------------------------------------
    # 3. Run the anharmonic SCBA (small test)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running anharmonic SCBA...")
    print("  4x4 q-mesh, 51 freq points, 20 SCBA iterations")
    print("=" * 60)

    t0 = time.time()
    result = anharmonic_transmission(
        phonon,
        fc3_data,
        q_mesh_transverse=(4, 4),
        freq_range_thz=(0.0, 15.0, 51),
        transport_direction="x",
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

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
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
    fig.savefig(script_dir / "anharmonic_test.png", dpi=150)
    plt.close("all")
    print(f"\nSaved anharmonic_test.png")
