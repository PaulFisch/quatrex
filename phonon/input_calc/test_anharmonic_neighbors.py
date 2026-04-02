"""Investigate the effect of FC3 neighbor cutoffs on anharmonic transmission.

Runs the SCBA with different FC3 interaction ranges:
  1) All interactions (delta_l=0 along transport)
  2) Only nearest-neighbor triplets (max distance < 1NN cutoff)
  3) Up to second nearest-neighbor triplets
  4) On-site only (R'=R''=0)

This helps determine how many neighbor shells are needed for
converged phonon-phonon scattering.
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
from test_anharmonic import remap_fc3_to_conventional


# ---------------------------------------------------------------------------
# 1. Load data
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
conv_cell = phonon_si.unitcell.cell
print(f"  Conventional cell: {n_atoms_conv} atoms, a = {a_conv:.4f} A")

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


# ---------------------------------------------------------------------------
# 2. Analyze FC3 distance structure
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Analyzing FC3 neighbor structure...")
print("=" * 60)

# For each FC3 block, compute the distances |R'| and |R''| from atom i
# These are the cell vectors in Cartesian coordinates
inv_cell = np.linalg.inv(conv_cell.T)
tidx = 2  # z-direction

distances_j = []
distances_k = []
distances_max = []
frac_j_list = []
frac_k_list = []

for block in fc3_conv["blocks"]:
    cell_j = block["cell_j"]  # Cartesian Angstrom
    cell_k = block["cell_k"]

    # Atom positions within the cell
    ai, aj, ak = block["atom_i"], block["atom_j"], block["atom_k"]
    pos_i = phonon_si.unitcell.scaled_positions[ai] @ conv_cell
    pos_j = phonon_si.unitcell.scaled_positions[aj] @ conv_cell + cell_j
    pos_k = phonon_si.unitcell.scaled_positions[ak] @ conv_cell + cell_k

    dj = np.linalg.norm(pos_j - pos_i)
    dk = np.linalg.norm(pos_k - pos_i)

    distances_j.append(dj)
    distances_k.append(dk)
    distances_max.append(max(dj, dk))

    frac_j = inv_cell @ cell_j
    frac_k = inv_cell @ cell_k
    frac_j_list.append(frac_j)
    frac_k_list.append(frac_k)

distances_j = np.array(distances_j)
distances_k = np.array(distances_k)
distances_max = np.array(distances_max)

# Identify neighbor shells from unique distances
unique_dists = np.unique(np.round(distances_max, 3))
print(f"\nUnique max(d_ij, d_ik) distances (Angstrom):")
for d in unique_dists[:15]:
    n = np.sum(np.abs(distances_max - d) < 0.01)
    print(f"  {d:.3f} A  ({n} blocks)")

# Reference distances for FCC/diamond
a = a_conv
nn1 = a * np.sqrt(3) / 4   # 1NN distance in diamond (bond length)
nn2 = a * np.sqrt(2) / 2   # 2NN distance
nn3 = a * np.sqrt(11) / 4  # 3NN distance
nn4 = a                     # 4NN distance
print(f"\nDiamond Si reference distances:")
print(f"  1NN bond length:  {nn1:.3f} A")
print(f"  2NN distance:     {nn2:.3f} A")
print(f"  3NN distance:     {nn3:.3f} A")
print(f"  4NN distance:     {nn4:.3f} A")

# Count blocks within each cutoff (for delta_l=0 constraint)
for block in fc3_conv["blocks"]:
    frac_j = inv_cell @ block["cell_j"]
    frac_k = inv_cell @ block["cell_k"]

# Determine which blocks are in each neighbor range
# Also check which ones have delta_l=0 (same slab along transport)
onsite_mask = np.array([
    np.allclose(b["cell_j"], 0.0) and np.allclose(b["cell_k"], 0.0)
    for b in fc3_conv["blocks"]
])

dl0_mask = np.array([
    abs(int(np.round((inv_cell @ b["cell_j"])[tidx]))) == 0 and
    abs(int(np.round((inv_cell @ b["cell_k"])[tidx]))) == 0
    for b in fc3_conv["blocks"]
])

print(f"\nBlock counts:")
print(f"  Total blocks:        {len(fc3_conv['blocks'])}")
print(f"  On-site (R'=R''=0):  {np.sum(onsite_mask)}")
print(f"  delta_l=0 (all):     {np.sum(dl0_mask)}")

# Define cutoff ranges based on max(d_ij, d_ik)
cutoffs = {
    "1NN": nn1 + 0.1,   # Up to and including 1st nearest neighbors
    "2NN": nn2 + 0.1,   # Up to 2nd nearest neighbors
    "3NN": nn3 + 0.1,   # Up to 3rd nearest neighbors
    "all": 1e10,         # All delta_l=0 blocks
}

for label, cutoff in cutoffs.items():
    mask = dl0_mask & (distances_max < cutoff)
    n_blocks = np.sum(mask)
    max_phi = 0.0
    for i, b in enumerate(fc3_conv["blocks"]):
        if mask[i]:
            max_phi = max(max_phi, np.max(np.abs(b["tensor"])))
    print(f"  {label} (cutoff={cutoff:.2f} A): {n_blocks} blocks, "
          f"max|Phi3|={max_phi:.4e} eV/A^3")


# ---------------------------------------------------------------------------
# 3. Create filtered FC3 datasets
# ---------------------------------------------------------------------------

def filter_fc3_by_distance(fc3_data, conv_cell, scaled_positions, cutoff,
                           transport_direction="z"):
    """Filter FC3 blocks by maximum pairwise distance and delta_l=0."""
    inv_cell = np.linalg.inv(conv_cell.T)
    tidx = "xyz".index(transport_direction)

    filtered_blocks = []
    for block in fc3_data["blocks"]:
        cell_j = block["cell_j"]
        cell_k = block["cell_k"]

        # Check delta_l=0
        frac_j = inv_cell @ cell_j
        frac_k = inv_cell @ cell_k
        dl_j = int(np.round(frac_j[tidx]))
        dl_k = int(np.round(frac_k[tidx]))
        if dl_j != 0 or dl_k != 0:
            continue

        # Compute pairwise distances
        ai, aj, ak = block["atom_i"], block["atom_j"], block["atom_k"]
        pos_i = scaled_positions[ai] @ conv_cell
        pos_j = scaled_positions[aj] @ conv_cell + cell_j
        pos_k = scaled_positions[ak] @ conv_cell + cell_k

        dij = np.linalg.norm(pos_j - pos_i)
        dik = np.linalg.norm(pos_k - pos_i)
        d_max = max(dij, dik)

        if d_max < cutoff:
            filtered_blocks.append(block)

    return {"n_blocks": len(filtered_blocks), "blocks": filtered_blocks}


# ---------------------------------------------------------------------------
# 4. Run SCBA for each cutoff
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Running SCBA for different FC3 neighbor cutoffs...")
print("=" * 60)

# Common parameters — small mesh for speed
q_mesh = (4, 4)
freq_range = (0.5, 15.0, 51)
scba_params = dict(
    q_mesh_transverse=q_mesh,
    freq_range_thz=freq_range,
    transport_direction="z",
    eta_factor=0.5,
    temperature=300.0,
    max_scba_iter=20,
    scba_tol=0.005,
    mixing=0.3,
    fc3_mode="full",
    verbose=False,
)

# Cases to run
cases = [
    ("on-site", None, "onsite"),      # Only R'=R''=0
    ("1NN", nn1 + 0.1, "filtered"),   # Up to 1st nearest neighbor
    ("2NN", nn2 + 0.1, "filtered"),   # Up to 2nd nearest neighbor
    ("3NN", nn3 + 0.1, "filtered"),   # Up to 3rd nearest neighbor
    ("all", None, "full"),             # All delta_l=0
]

results = {}

for label, cutoff, mode in cases:
    print(f"\n--- Case: {label} ---")

    if mode == "onsite":
        # Use on-site mode
        params = {**scba_params, "fc3_mode": "onsite", "verbose": True}
        fc3_input = fc3_conv
    elif mode == "filtered":
        fc3_filtered = filter_fc3_by_distance(
            fc3_conv, conv_cell, phonon_si.unitcell.scaled_positions,
            cutoff, transport_direction="z"
        )
        print(f"  Filtered to {fc3_filtered['n_blocks']} blocks "
              f"(cutoff={cutoff:.2f} A)")
        params = {**scba_params, "fc3_mode": "full", "verbose": True}
        fc3_input = fc3_filtered
    else:
        params = {**scba_params, "fc3_mode": "full", "verbose": True}
        fc3_input = fc3_conv

    t0 = time.time()
    result = anharmonic_transmission(phonon_si, fc3_input, **params)
    t1 = time.time()

    results[label] = result
    G_ball = result["thermal_conductance_ballistic"]
    G_anh = result["thermal_conductance_anharmonic"]
    reduction = (1 - G_anh / G_ball) * 100 if G_ball > 0 else 0
    print(f"  Time: {t1-t0:.1f} s")
    print(f"  Ballistic G:   {G_ball/1e6:.2f} MW/(m^2 K)")
    print(f"  Anharmonic G:  {G_anh/1e6:.2f} MW/(m^2 K)")
    print(f"  Reduction:     {reduction:.1f}%")
    print(f"  Max J (ball):  {result['spectral_heat_current_ballistic'].max():.4e}")
    print(f"  Max J (anh):   {result['spectral_heat_current'].max():.4e}")
    print(f"  SCBA iters:    {result['n_scba_iterations']}")


# ---------------------------------------------------------------------------
# 5. Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY: Effect of FC3 neighbor cutoffs")
print("=" * 60)
print(f"{'Case':<10} {'G_ball [MW/m²K]':>16} {'G_anh [MW/m²K]':>16} "
      f"{'Reduction [%]':>14} {'SCBA iters':>11}")
print("-" * 70)

for label, _, _ in cases:
    r = results[label]
    G_b = r["thermal_conductance_ballistic"]
    G_a = r["thermal_conductance_anharmonic"]
    red = (1 - G_a / G_b) * 100 if G_b > 0 else 0
    print(f"{label:<10} {G_b/1e6:>16.2f} {G_a/1e6:>16.2f} "
          f"{red:>14.1f} {r['n_scba_iterations']:>11d}")


# ---------------------------------------------------------------------------
# 6. Plot comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
freqs = results["all"]["freqs_thz"]

# (a) Spectral heat current
ax = axes[0]
ax.plot(freqs, results["all"]["spectral_heat_current_ballistic"],
        "k-", lw=2, label="Ballistic")
colors = {"on-site": "C0", "1NN": "C1", "2NN": "C2", "3NN": "C3", "all": "C4"}
styles = {"on-site": ":", "1NN": "-.", "2NN": "--", "3NN": "-", "all": "-"}

for label, _, _ in cases:
    ax.plot(freqs, results[label]["spectral_heat_current"],
            color=colors[label], ls=styles[label], lw=1.5,
            label=f"{label}")
ax.set_xlabel("Frequency (THz)")
ax.set_ylabel("Spectral heat current (W)")
ax.set_title("(a) Heat current vs FC3 cutoff")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 16)
ax.set_ylim(0, None)

# (b) Heat current ratio
ax = axes[1]
J_ball = results["all"]["spectral_heat_current_ballistic"]
mask = J_ball > np.max(J_ball) * 0.01
for label, _, _ in cases:
    J_anh = results[label]["spectral_heat_current"]
    ratio = np.ones_like(freqs)
    ratio[mask] = J_anh[mask] / J_ball[mask]
    ax.plot(freqs[mask], ratio[mask],
            color=colors[label], ls=styles[label], lw=1.5,
            label=f"{label}")
ax.axhline(y=1.0, color="gray", ls="--", alpha=0.5)
ax.set_xlabel("Frequency (THz)")
ax.set_ylabel("J_anh / J_ball")
ax.set_title("(b) Heat current ratio")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 16)
ax.set_ylim(0, 1.5)

# (c) Bar chart of thermal conductance reduction
ax = axes[2]
labels_short = [c[0] for c in cases]
reductions = []
for label, _, _ in cases:
    r = results[label]
    G_b = r["thermal_conductance_ballistic"]
    G_a = r["thermal_conductance_anharmonic"]
    reductions.append((1 - G_a / G_b) * 100 if G_b > 0 else 0)

bars = ax.bar(labels_short, reductions, color=[colors[c[0]] for c in cases],
              edgecolor="black", linewidth=0.5)
ax.set_ylabel("Thermal conductance reduction (%)")
ax.set_title("(c) Effect of FC3 cutoff")
ax.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, reductions):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
fig.savefig(work_dir / "anharmonic_neighbor_study.png", dpi=150)
plt.close("all")
print(f"\nSaved anharmonic_neighbor_study.png")
