"""Solver-scaling plots (Part 1b/1c): RGF (linear) vs dense (cubic) selected
inversion vs number of blocks, and distributed energy-parallel speedup.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
OUTS = [_REPO / "phonon/scripts/out/ballistic_curves",
        _REPO / "document/fig/transport_sweeps"]
for o in OUTS:
    o.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(open(_REPO / "phonon/scripts/out/rgf_vs_dense_scaling.csv")))
nb = [int(r["num_blocks"]) for r in rows]
t_rgf = [float(r["t_rgf"]) for r in rows]
t_dense = [float(r["t_dense"]) for r in rows]

# distributed energy-parallel (dist_scaling.py, N=1536, 128 E-pts)
ranks = [1, 2, 4, 8]
wall = [2.053, 1.108, 0.722, 0.384]
speedup = [wall[0] / w for w in wall]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
ax1.loglog(nb, t_rgf, "o-", color="#2ca02c", label=r"RGF  $\mathcal{O}(N_\mathrm{blk})$")
ax1.loglog(nb, t_dense, "s-", color="#d62728", label=r"dense  $\mathcal{O}(N_\mathrm{blk}^3)$")
# guide lines
ax1.loglog(nb, [t_rgf[0]*(n/nb[0]) for n in nb], "k:", lw=0.8, alpha=0.6)
ax1.loglog(nb, [t_dense[0]*(n/nb[0])**3 for n in nb], "k--", lw=0.8, alpha=0.6)
ax1.set_xlabel("number of blocks (transport cells)")
ax1.set_ylabel("selected-inversion wall time (s)")
ax1.set_title("RGF vs dense scaling (block size 192)")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3, which="both")

ax2.plot(ranks, speedup, "o-", color="#1f77b4", label="measured")
ax2.plot(ranks, ranks, "k:", lw=0.8, label="ideal")
ax2.set_xlabel("MPI ranks (energy-parallel)")
ax2.set_ylabel("speed-up")
ax2.set_title("Distributed selected inversion (N=1536, 128 E-pts)")
ax2.set_xticks(ranks)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
fig.tight_layout()
for o in OUTS:
    fig.savefig(o / "solver_scaling.pdf")
plt.close(fig)
print(f"dense/RGF at {nb[-1]} blocks: {t_dense[-1]/t_rgf[-1]:.0f}x; "
      f"distributed {speedup[-1]:.1f}x at {ranks[-1]} ranks")
print("[done] wrote solver_scaling.pdf")
