"""Distributed scaling of the production quatrex phph self-energy: speed-up vs
ranks along the energy(stack) axis vs the block axis, with the q-point axis
(embarrassingly parallel, projected) for contrast.

Data measured by phph_dist_scaling.py (NBLK=6, BS=16, NE=96):
  energy(stack)-parallel (block_comm_size=1): np 1/2/4 -> 4.13/4.27/4.64 s
  block-parallel        (block_comm_size=np): np 1/2/3/6 -> 4.13/2.72/1.74/0.77 s
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, "/usr/scratch/mont-fort11/pfischill/quatrex/phonon")
from finite_analysis.plot_style import set_publication_style  # noqa: E402
set_publication_style()

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
OUTS = [_REPO / "phonon/scripts/out/phph_physics",
        _REPO / "document/fig/transport_sweeps"]
for o in OUTS:
    o.mkdir(parents=True, exist_ok=True)

t1 = 4.13
energy_np = [1, 2, 4];      energy_t = [4.13, 4.27, 4.64]
block_np = [1, 2, 3, 6];    block_t = [4.13, 2.72, 1.74, 0.77]

fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.plot(block_np, [t1 / t for t in block_t], "s-", color="#2ca02c",
        label="block-parallel (the $(I,J)$ loop)")
ax.plot(energy_np, [t1 / t for t in energy_t], "o-", color="#d62728",
        label="energy/stack-parallel (bubble replicated)")
rr = np.array([1, 2, 3, 4, 6])
ax.plot(rr, rr, "k:", lw=1.0, label="ideal (= q-point axis, projected)")
ax.set_xlabel("MPI ranks")
ax.set_ylabel("phph self-energy speed-up")
ax.set_xticks([1, 2, 3, 4, 6])
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
for o in OUTS:
    fig.savefig(o / "phph_scaling.pdf")
plt.close(fig)
print("block speed-up:", [round(t1/t, 2) for t in block_t])
print("energy speed-up:", [round(t1/t, 2) for t in energy_t])
print("[done] wrote phph_scaling.pdf")
