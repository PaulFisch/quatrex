"""Plot the direct ballistic conductance curves for d5a and d11a.

Reads ``ballistic_curves/ballistic.csv`` and writes both into the script
output dir and the LaTeX figure tree (document/fig/transport_sweeps/).
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
CSV = _REPO / "phonon/scripts/out/ballistic_curves/ballistic.csv"
OUTS = [_REPO / "phonon/scripts/out/ballistic_curves",
        _REPO / "document/fig/transport_sweeps"]
for o in OUTS:
    o.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(open(CSV)))
for r in rows:
    for k in ("n_slabs", "T"):
        r[k] = int(float(r[k]))
    r["G_ball"] = float(r["G_ball"])

STYLE = {"d5a": dict(color="#1f77b4", marker="o", label="d5a SiNW (63 DOF/cell)"),
         "d11a": dict(color="#d62728", marker="s", label="d11a SiNW (135 DOF/cell)"),
         "cnt33": dict(color="#2ca02c", marker="^", label="(3,3) CNT (36 DOF/cell)")}


def save(fig, name):
    for o in OUTS:
        fig.savefig(o / name)
    plt.close(fig)


# G_ball vs T at L=1.
fig, ax = plt.subplots(figsize=(5.2, 4.0))
for wire, st in STYLE.items():
    pts = sorted([r for r in rows if r["wire"] == wire and r["n_slabs"] == 1],
                 key=lambda r: r["T"])
    ax.plot([r["T"] for r in pts], [r["G_ball"] / 1e6 for r in pts], **st)
ax.set_xlabel("temperature (K)")
ax.set_ylabel(r"$G_\mathrm{ball}$  (MW m$^{-2}$ K$^{-1}$)")
ax.set_title("Ballistic conductance vs temperature (one transport cell)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "ballistic_vs_T_d5_d11.pdf")

# G_ball vs length at T=300 (finite-eta coherence attenuation).
fig, ax = plt.subplots(figsize=(5.2, 4.0))
for wire, st in STYLE.items():
    pts = sorted([r for r in rows if r["wire"] == wire and r["T"] == 300],
                 key=lambda r: r["n_slabs"])
    ax.plot([r["n_slabs"] for r in pts], [r["G_ball"] / 1e6 for r in pts], **st)
ax.set_xlabel("device length (transport cells)")
ax.set_ylabel(r"$G_\mathrm{ball}$  (MW m$^{-2}$ K$^{-1}$)")
ax.set_title(r"Coherent conductance vs length at $T=300$ K"
             "\n(finite-$\\eta$ attenuation)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "ballistic_vs_length_d5_d11.pdf")

print(f"[plot] wrote ballistic figures to {OUTS[0]} and {OUTS[1]}", flush=True)
