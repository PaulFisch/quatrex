"""Solver-scaling figure (fig:res_solver_scaling): RGF vs dense selected
inversion, and distributed energy-parallel speedup.

Left panel : selected-inversion wall time vs number of transport-cell blocks
             (fixed 192-dof block) -- RGF linear, dense cubic, with slope
             guides.  Data: phonon/scripts/out/rgf_vs_dense_scaling.csv.
Right panel: distributed energy-parallel speedup vs MPI ranks with the ideal
             slope-1 guide.
             PROVENANCE: the wall times are literals from the retired
             dist_scaling.py run (N=1536 phonon-Dyson problem, 128 energy
             points; ranks 1/2/4/8 -> 2.053/1.108/0.722/0.384 s), carried
             here verbatim because that benchmark script no longer exists.
             NB the report body quotes 1.55->0.28 s (a different run of the
             same benchmark); both give the same ~5.3-5.5x at 8 ranks.

Style: phonon/studies/style.py (unified). Successor of the retired
phonon/scripts/verify/plot_scaling.py (git 843c3069^).

Run:  python phonon/scripts/figures/solver_scaling.py
Figure -> document/fig/transport_sweeps/solver_scaling.{png,pdf}
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style

CSV = ROOT / "phonon/scripts/out/rgf_vs_dense_scaling.csv"
FIGDIR = ROOT / "document/fig/transport_sweeps"

# retired dist_scaling.py measurement (see docstring)
RANKS = np.array([1, 2, 4, 8])
WALL = np.array([2.053, 1.108, 0.722, 0.384])


def main() -> None:
    with open(CSV) as fh:
        rows = list(csv.DictReader(fh))
    nb = np.array([int(r["num_blocks"]) for r in rows])
    t_rgf = np.array([float(r["t_rgf"]) for r in rows])
    t_dense = np.array([float(r["t_dense"]) for r in rows])
    maxerr = max(float(r["rgf_vs_dense_maxerr"]) for r in rows)
    speedup = WALL[0] / WALL

    fig, (ax1, ax2) = style.figure(ncols=2, width=4.4, height=3.3)
    ax1.loglog(nb, t_rgf, "o-", color="C2",
               label=r"RGF  $\mathcal{O}(N_\mathrm{blk})$")
    ax1.loglog(nb, t_dense, "s-", color="C3",
               label=r"dense  $\mathcal{O}(N_\mathrm{blk}^3)$")
    # slope guides anchored at the largest problem (the asymptotic side)
    ax1.loglog(nb, t_rgf[-1] * (nb / nb[-1]), "k:", lw=0.9, alpha=0.6)
    ax1.loglog(nb, t_dense[-1] * (nb / nb[-1]) ** 3, "k--", lw=0.9, alpha=0.6)
    ax1.annotate(r"$\propto N_\mathrm{blk}$",
                 (nb[-2], t_rgf[-1] * nb[-2] / nb[-1]),
                 textcoords="offset points", xytext=(4, -12), fontsize=7.5,
                 color="0.25")
    ax1.annotate(r"$\propto N_\mathrm{blk}^3$",
                 (nb[-3], t_dense[-1] * (nb[-3] / nb[-1]) ** 3),
                 textcoords="offset points", xytext=(-32, -16), fontsize=7.5,
                 color="0.25")
    ax1.plot([nb[-1], nb[-1]], [t_rgf[-1], t_dense[-1]], "-", color="0.6",
             lw=0.9, zorder=1)
    ax1.annotate(f"{t_dense[-1] / t_rgf[-1]:.0f}$\\times$",
                 (nb[-1], np.sqrt(t_rgf[-1] * t_dense[-1])),
                 textcoords="offset points", xytext=(-4, 0),
                 fontsize=8, ha="right", va="center", color="0.15")
    ax1.set_xticks(nb)
    ax1.set_xticklabels([str(n) for n in nb])
    ax1.minorticks_off()
    ax1.set_xlabel(r"number of blocks $N_\mathrm{blk}$ (transport cells)")
    ax1.set_ylabel("selected-inversion wall time (s)")
    ax1.legend(fontsize=7.5, loc="upper left")

    ax2.loglog(RANKS, speedup, "o-", color="C0", label="measured")
    ax2.loglog(RANKS, RANKS, "k:", lw=0.9, label="ideal")
    ax2.set_xticks(RANKS)
    ax2.set_xticklabels([str(r) for r in RANKS])
    ax2.set_yticks([1, 2, 4, 8])
    ax2.set_yticklabels(["1", "2", "4", "8"])
    ax2.minorticks_off()
    ax2.annotate(f"{speedup[-1]:.1f}$\\times$ at {RANKS[-1]} ranks",
                 (7.9, 1.35), fontsize=7.5, ha="right", color="0.15")
    ax2.set_xlabel("MPI ranks (energy-parallel)")
    ax2.set_ylabel("speed-up")
    ax2.legend(fontsize=7.5, loc="upper left")

    style.save(fig, "solver_scaling", directory=FIGDIR)
    print(f"dense/RGF at {nb[-1]} blocks: {t_dense[-1] / t_rgf[-1]:.1f}x "
          f"(report: 58x); max RGF-dense deviation {maxerr:.1e} "
          f"(report: ~1e-13)")
    print(f"energy-parallel: {speedup[-1]:.2f}x at {RANKS[-1]} ranks "
          f"({100 * speedup[-1] / RANKS[-1]:.0f}% efficiency; report: "
          f"5.3-5.5x / 68%)")


if __name__ == "__main__":
    main()
