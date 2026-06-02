"""Overlay the finite-analysis FC3/self-energy cutoff convergence for both wires.

Reads the ``cutoffs_sweep.csv`` produced by the finite-analysis cutoff study
(self-energy Sigma^< relative error vs FC3 magnitude-threshold and vs the
diagonal-G approximation) for d5a and d11a and overlays them.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
SRC = {
    "d5a": _REPO / "phonon/reaps_old/hiphive_sinw100_d5a_vasp/cutoffs/cutoffs_sweep.csv",
    "d11a": _REPO / "phonon/configs/sinw/reaps/sinw100_d11a_vasp_sc4/cutoffs/cutoffs_sweep.csv",
}
OUTS = [_REPO / "phonon/scripts/out/ballistic_curves",
        _REPO / "document/fig/transport_sweeps"]
for o in OUTS:
    o.mkdir(parents=True, exist_ok=True)

STYLE = {"d5a": dict(color="#1f77b4", marker="o", label="d5a (63 DOF/cell)"),
         "d11a": dict(color="#d62728", marker="s", label="d11a (135 DOF/cell)")}
THRESH = {"mag_thresh_1e-2": 1e-2, "mag_thresh_1e-3": 1e-3, "mag_thresh_1e-4": 1e-4}

fig, ax = plt.subplots(figsize=(5.6, 4.0))
diag = {}
for wire, path in SRC.items():
    rows = {r["label"]: float(r["mean_rel_diff_lesser"]) for r in csv.DictReader(open(path))}
    xs = sorted(THRESH.values(), reverse=True)
    ys = [rows[lab] for lab in
          sorted(THRESH, key=lambda k: THRESH[k], reverse=True)]
    ax.loglog(xs, ys, **STYLE[wire])
    diag[wire] = rows.get("diag_G_in_se", float("nan"))
ax.set_xlabel("FC3 magnitude-threshold (fraction of max)")
ax.set_ylabel(r"$\langle|\Delta\Sigma^<|\rangle$ rel.\ to full vertex")
ax.set_title("FC3 vertex magnitude-cutoff convergence")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")
# annotate the catastrophic diagonal-G approximation
txt = "diagonal-$G$ approx.: " + ", ".join(
    f"{w} {diag[w]:.0f}$\\times$" for w in ("d5a", "d11a"))
ax.text(0.5, 0.04, txt, transform=ax.transAxes, fontsize=8,
        ha="center", va="bottom",
        bbox=dict(boxstyle="round", fc="#fff3cd", ec="#999"))
fig.tight_layout()
for o in OUTS:
    fig.savefig(o / "cutoff_sse_d5_d11.pdf")
plt.close(fig)
print("[plot] cutoff (SSE) overlay written; diag_G rel-diff:", diag, flush=True)
