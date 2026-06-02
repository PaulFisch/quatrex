"""Plot the d5a vertex-cutoff convergence of the anharmonic conductance.

Reads ``d5a_cutoff_proj/summary.csv`` (written by d5_cutoff_sweep.py) and
plots G_anh and heat-flow conservation vs the vertex cutoff radius, into the
LaTeX figure tree.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
CSV = _REPO / "phonon/scripts/out/d5a_cutoff_proj/summary.csv"
OUTS = [_REPO / "phonon/scripts/out/d5a_cutoff_proj",
        _REPO / "document/fig/transport_sweeps"]
for o in OUTS:
    o.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(open(CSV)))


def vkey(r):
    v = str(r["vertex_cutoff"])
    return 1e9 if v in ("Inf", "inf") else float(v)


rows = sorted(rows, key=vkey)
labels = [("full" if str(r["vertex_cutoff"]) in ("Inf", "inf")
           else str(r["vertex_cutoff"])) for r in rows]
xs = list(range(len(rows)))
G = [float(r["G_anh"]) / 1e6 for r in rows]
cons = [float(r["conservation"]) for r in rows]

fig, ax1 = plt.subplots(figsize=(5.6, 4.0))
ax1.plot(xs, G, "o-", color="#1f77b4", label=r"$G_\mathrm{anh}$")
ax1.set_xlabel("vertex cutoff (transport cells)")
ax1.set_ylabel(r"$G_\mathrm{anh}$  (MW m$^{-2}$ K$^{-1}$)", color="#1f77b4")
ax1.set_xticks(xs)
ax1.set_xticklabels(labels)
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax2 = ax1.twinx()
ax2.plot(xs, cons, "s--", color="#d62728", label="conservation")
ax2.set_ylabel(r"heat-flow non-conservation $|J_L-J_R|/J$", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax1.set_title("d5a vertex-cutoff convergence (n_slabs=2, projected FC3)")
fig.tight_layout()
for o in OUTS:
    fig.savefig(o / "cutoff_vertex_d5a.pdf")
plt.close(fig)
print("[plot] vertex-cutoff figure written; data:")
for lab, g, c in zip(labels, G, cons):
    print(f"  vertex={lab:>4}: G_anh={g:.3f} MW/m2K, conservation={c:.3e}")
