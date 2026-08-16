"""FC3/self-energy cutoff convergence overlay for the d5a + d11a nanowires
(fig:res_cutoffs).

Data:
  Data (cached CSVs only, no reruns):
    d11a : phonon/configs/sinw/reaps/sinw100_d11a_vasp_sc4/cutoffs/cutoffs_sweep.csv
    d5a  : phonon/reaps/hiphive_sinw100_d5a_vasp/cutoffs/cutoffs_sweep.csv
           PROVENANCE: the on-disk copy was purged with reaps_old/; this file was
           restored verbatim from git (commit 9ee70acf "cutoff data", path
           phonon/reaps/hiphive_sinw100_d5a_vasp/...). It is the pre-sc4 d5a reap
           the retired plot_cutoff_sse.py used, and it reproduces the report's
           numbers exactly (diag-G 5.0x, 1e-3 threshold -> 0.68%). Note git also
           carries an sc4-era d5a sweep (sinw100_d5a_vasp_sc4, diag-G 77x) --
           rerun the cutoff study on the current reap if the sc4 fit is wanted.

Run:  python phonon/scripts/figures/cutoff_sse.py
Figure -> document/fig/transport_sweeps/cutoff_sse_d5_d11.{png,pdf}
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style

SRC = {
    "d5a": ROOT / "phonon/reaps/hiphive_sinw100_d5a_vasp/cutoffs/cutoffs_sweep.csv",
    "d11a": ROOT / "phonon/configs/sinw/reaps/sinw100_d11a_vasp_sc4/cutoffs/cutoffs_sweep.csv",
}
FIGDIR = ROOT / "document/fig/transport_sweeps"
STYLE = {"d5a": dict(color="C0", marker="o", label="d5a (63 DOF/cell)"),
         "d11a": dict(color="C3", marker="s", label="d11a (135 DOF/cell)")}
THRESH = {"mag_thresh_1e-2": 1e-2, "mag_thresh_1e-3": 1e-3,
          "mag_thresh_1e-4": 1e-4}
FLOOR = 1e-4  # d5a's 1e-4 threshold changes nothing (exact 0) -> plot at floor


def main() -> None:
    fig, ax = style.figure(width=4.8, height=3.5)
    diag = {}
    for wire, path in SRC.items():
        if not path.exists():
            print(f"WARNING: {wire} cutoffs_sweep.csv missing at {path} -- "
                  f"series skipped; rerun the finite-analysis cutoff study "
                  f"(or restore it from git commit 9ee70acf).")
            continue
        with open(path) as fh:
            rows = {r["label"]: float(r["mean_rel_diff_lesser"])
                    for r in csv.DictReader(fh)}
        c = STYLE[wire]["color"]
        xs = sorted(THRESH.values(), reverse=True)
        labs = sorted(THRESH, key=lambda k: THRESH[k], reverse=True)
        ys = [max(rows[la], FLOOR / 10) for la in labs]
        zero = [rows[la] == 0.0 for la in labs]
        ax.loglog(xs, ys, ls="-", **STYLE[wire])
        for xi, yi, z in zip(xs, ys, zero):
            if z:  # exact zero pinned to the floor decade
                ax.annotate("exact (0)", (xi, yi), textcoords="offset points",
                            xytext=(-7, 6), fontsize=7, color=c, ha="right")
        # review fix: diagonal-G as a per-wire dashed line at its true value
        d = rows.get("diag_G_in_se", float("nan"))
        diag[wire] = d
        ax.axhline(d, color=c, ls="--", lw=1.1, alpha=0.85)
        ax.annotate(f"diagonal $G$ ({wire}): {d:.0f}$\\times$",
                    (1.15e-4, d * 1.25), fontsize=7.5, color=c)

    ax.set_xlabel("FC3 magnitude threshold (fraction of max)")
    ax.set_ylabel(r"$\langle|\Delta\Sigma^<|\rangle$ rel. to full vertex")
    ax.set_xlim(1.5e-2, 0.7e-4)  # tighter threshold to the right
    ax.set_ylim(0.5 * FLOOR / 10, 2e2)
    ax.legend(fontsize=8, loc="lower left")
    style.save(fig, "cutoff_sse_d5_d11", directory=FIGDIR)

    print("diag-G mean rel. error:",
          {w: f"{v:.2f}x" for w, v in diag.items()})
    for wire, path in SRC.items():
        if path.exists():
            with open(path) as fh:
                rows = {r["label"]: float(r["mean_rel_diff_lesser"])
                        for r in csv.DictReader(fh)}
            print(f"{wire}: 1e-3 threshold -> "
                  f"{rows['mag_thresh_1e-3']*100:.2f}% mean dSigma^<")


if __name__ == "__main__":
    main()
