"""cnt33 finite-eta bias demonstration (results/80_approx, fig:res_cnt_T).

Data:
  sweeps (phonon/docs/lab_notebook_archive.md, F30 -- the raw npz were purged in

Run:  python phonon/scripts/figures/cnt33_finite_eta_bias.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style

PROD = ROOT / "phonon/scripts/out/prod/cnt33_eta0"
FIGDIR = ROOT / "document/fig/transport_sweeps"

# lab_notebook_archive.md F30: dense L=1 fine-grid T-sweep (eta_w = 0.050 THz)
DENSE_T = [  # (T K, G_ball, G_anh, ratio)
    (30, 5.49e7, 5.04e7, 0.919),
    (50, 9.38e7, 8.25e7, 0.880),
    (100, 2.14e8, 1.72e8, 0.805),
    (150, 3.60e8, 2.74e8, 0.762),
    (200, 5.06e8, 3.73e8, 0.737),
    (300, 7.73e8, 5.48e8, 0.709),
]
# F30 ladder at eta_w = 0.206 THz: the SAME L=1/300 K point moves to 0.810;
# the two-cell point (matching the eta=0 sweep's length) is 0.702.
LADDER_L1_ETA206 = 0.810
LADDER_L2_ETA206 = 0.702


def main():
    rows = json.load(open(PROD / "summary.json"))
    e0 = sorted((r["t_mean"], r["ratio"], r["G_ball_W_per_m2_K"],
                 r["G_anh_W_per_m2_K"]) for r in rows
                if r.get("sweep") == "temperature" and r.get("anh_converged"))

    T = [t for t, *_ in DENSE_T]
    fig, ax = style.figure(width=4.6, height=3.4)
    ax.plot(T, [b for _, b, _, _ in DENSE_T], "o--", color="C0",
            label=r"$G_\mathrm{ball}$ (dense, $\eta_w{=}0.05$)")
    ax.plot(T, [a for _, _, a, _ in DENSE_T], "o-", color="C0",
            label=r"$G_\mathrm{anh}$ (dense, $\eta_w{=}0.05$)")
    ax.plot([t for t, *_ in e0], [g for _, _, g, _ in e0], "s--", color="C3",
            label=r"$G_\mathrm{ball}$ ($\eta=0$, two cells)")
    ax.plot([t for t, *_ in e0], [g for _, _, _, g in e0], "s-", color="C3",
            label=r"$G_\mathrm{anh}$ ($\eta=0$, two cells)")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"$G$ (W m$^{-2}$ K$^{-1}$)")
    ax.legend(fontsize=7, loc="upper left")
    style.save(fig, "cnt33_temperature_g", directory=FIGDIR)

    fig, ax = style.figure(width=4.6, height=3.4)
    ax.plot(T, [r for *_, r in DENSE_T], "o-", color="C0",
            label=r"finite $\eta_w{=}0.05$ THz (one cell)")
    ax.plot(300, LADDER_L1_ETA206, "D", ms=7, mfc="none", color="C0",
            label=r"finite $\eta_w{=}0.206$ THz (one cell)")
    ax.plot(300, LADDER_L2_ETA206, "^", ms=7, mfc="none", color="C2",
            label=r"finite $\eta_w{=}0.206$ THz (two cells)")
    ax.plot([t for t, *_ in e0], [r for _, r, *_ in e0], "s-", color="C3",
            label=r"conserving $\eta=0$ (two cells)")
    # the honest same-length comparison at 300 K: 0.702 -> 0.574
    ax.annotate("", xy=(300, 0.586), xytext=(300, 0.692),
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0))
    ax.annotate("same length:\n$0.702\\to0.574$", (288, 0.635), fontsize=7,
                ha="right", color="0.35")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"$G_\mathrm{anh}/G_\mathrm{ball}$")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=7, loc="lower left")
    style.save(fig, "cnt33_temperature_ratio", directory=FIGDIR)

    print("dense 300K: 0.709 (eta_w=0.05) / 0.810 (eta_w=0.206);  eta=0:",
          [(t, round(r, 3)) for t, r, *_ in e0])


def _interp_e0(e0, t):
    import numpy as np
    ts = [x[0] for x in e0]
    rs = [x[1] for x in e0]
    return float(np.interp(t, ts, rs))


if __name__ == "__main__":
    main()
