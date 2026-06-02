"""Length-scaling plots (Part 1a): anharmonic resistance accumulation on the CNT
and ballistic conductance vs length for all three wires.
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

# --- CNT anharmonic G(L) from checkpoints ---
ck = _REPO / "phonon/scripts/out/cnt33_transport/checkpoints/length"
pts = []
for f in sorted(ck.glob("*.npz")):
    d = np.load(f, allow_pickle=False)
    pts.append((int(d["n_slabs"]), float(d["thermal_conductance_ballistic"]),
                float(d["thermal_conductance_anharmonic"])))
pts.sort()
L = [p[0] for p in pts]
gb = [p[1] / 1e6 for p in pts]
ga = [p[2] / 1e6 for p in pts]
ratio = [p[2] / p[1] for p in pts]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
ax1.plot(L, gb, "s--", color="#2ca02c", alpha=0.5, label="ballistic")
ax1.plot(L, ga, "^-", color="#2ca02c", label="anharmonic (SCBA)")
ax1.set_xlabel("device length (transport cells)")
ax1.set_ylabel(r"$G$  (MW m$^{-2}$ K$^{-1}$)")
ax1.set_title("(3,3) CNT conductance vs length, T=300 K")
ax1.set_xticks(L)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1b = ax1.twinx()
ax1b.plot(L, ratio, "o:", color="#d62728")
ax1b.set_ylabel(r"$G_\mathrm{anh}/G_\mathrm{ball}$", color="#d62728")
ax1b.tick_params(axis="y", labelcolor="#d62728")

# --- 3-wire ballistic G(L) from ballistic.csv ---
rows = list(csv.DictReader(open(_REPO / "phonon/scripts/out/ballistic_curves/ballistic.csv")))
STYLE = {"d5a": ("#1f77b4", "o", "d5a SiNW"), "d11a": ("#d62728", "s", "d11a SiNW"),
         "cnt33": ("#2ca02c", "^", "(3,3) CNT")}
for wire, (c, m, lab) in STYLE.items():
    ws = sorted([r for r in rows if r["wire"] == wire and int(float(r["T"])) == 300],
                key=lambda r: int(float(r["n_slabs"])))
    ax2.semilogy([int(float(r["n_slabs"])) for r in ws],
                 [float(r["G_ball"]) / 1e6 for r in ws], marker=m, color=c, label=lab)
ax2.set_xlabel("device length (transport cells)")
ax2.set_ylabel(r"$G_\mathrm{ball}$  (MW m$^{-2}$ K$^{-1}$)")
ax2.set_title("Ballistic conductance vs length (all wires)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, which="both")
fig.tight_layout()
for o in OUTS:
    fig.savefig(o / "length_scaling.pdf")
plt.close(fig)
print("CNT G_anh/G_ball vs L:", list(zip(L, [round(r, 3) for r in ratio])))
print(f"[done] wrote length_scaling.pdf")
