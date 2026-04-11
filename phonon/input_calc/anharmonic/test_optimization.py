"""Quick validation that optimized kernels produce correct results.

Runs dense, separable (full rank), and PCP on a small freq grid
and compares thermal conductances.
"""

import sys
import time
from pathlib import Path

import numpy as np

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent  # input_calc/
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs.pcp import pcp_anharmonic_transmission
from phonon_inputs.separable import separable_anharmonic_transmission
from phonon_inputs.anharmonic import anharmonic_transmission_q


def main():
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

    common = dict(
        q_mesh_transverse=(4, 4),
        freq_range_thz=(1.0, 14.0, 51),  # coarser grid for speed
        max_scba_iter=5,
        scba_tol=0.01,
        mixing=0.3,
        n_slabs=1,
        verbose=True,
    )

    results = {}

    # Dense (reference)
    print("\n=== Dense ===")
    t0 = time.time()
    r = anharmonic_transmission_q(phonon, str(fc3_path), **common)
    dt = time.time() - t0
    results["dense"] = r
    print(f"  Time: {dt:.1f}s, G_anh: {r['thermal_conductance_anharmonic']:.2e}")

    # Separable full rank
    print("\n=== Separable (full rank) ===")
    t0 = time.time()
    r = separable_anharmonic_transmission(phonon, str(fc3_path), rank=None, **common)
    dt = time.time() - t0
    results["sep_full"] = r
    print(f"  Time: {dt:.1f}s, G_anh: {r['thermal_conductance_anharmonic']:.2e}")

    # Separable R=6
    print("\n=== Separable R=6 ===")
    t0 = time.time()
    r = separable_anharmonic_transmission(phonon, str(fc3_path), rank=6, **common)
    dt = time.time() - t0
    results["sep_r6"] = r
    print(f"  Time: {dt:.1f}s, G_anh: {r['thermal_conductance_anharmonic']:.2e}")

    # PCP N_c=8
    print("\n=== PCP N_c=8 ===")
    t0 = time.time()
    r = pcp_anharmonic_transmission(phonon, str(fc3_path), pcp_rank=8, **common)
    dt = time.time() - t0
    results["pcp_8"] = r
    print(f"  Time: {dt:.1f}s, G_anh: {r['thermal_conductance_anharmonic']:.2e}")

    # Summary
    print(f"\n{'Method':<18} {'G_ball':>12} {'G_anh':>12} {'G_anh/G_ball':>14} {'Time':>8}")
    for key, r in results.items():
        g_ball = r["thermal_conductance_ballistic"]
        g_anh = r["thermal_conductance_anharmonic"]
        print(f"{key:<18} {g_ball:>12.2e} {g_anh:>12.2e} {g_anh/g_ball:>14.4f}")

    # Validation: dense ≈ separable full rank
    g_dense = results["dense"]["thermal_conductance_anharmonic"]
    g_sep = results["sep_full"]["thermal_conductance_anharmonic"]
    rel_diff = abs(g_dense - g_sep) / abs(g_dense)
    print(f"\nDense vs Sep(full): rel_diff = {rel_diff:.2e}")
    if rel_diff < 0.05:
        print("PASSED: Dense ≈ Separable (full rank)")
    else:
        print("WARNING: Dense and Separable (full rank) differ significantly!")


if __name__ == "__main__":
    main()
