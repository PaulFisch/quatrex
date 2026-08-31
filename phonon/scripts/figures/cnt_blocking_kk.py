"""CNT blocking and the Kramers-Kronig form (fig:res_cnt_blocking).

Two interventions on ONE sixteen-cell CNT (3,3) device at eta = 0, band 3,
ne = 161, each changing a single variable against the 16 x 1 baseline that
every CNT run in the corpus used:

  blocking  8 blocks of 2 cells instead of 16 of 1 (c16x2h)
  Sigma^R   the full Kramers-Kronig form instead of the antihermitian
            half (c16-kk)

Data: cluster/{c16-ball,c16-half,c16x2h,c16-kk}/run.npz, campaign of
2026-08-28 on 4 GH200 (see phonon/docs/cnt_campaign_2026-08.md).

Run:  python phonon/scripts/figures/cnt_blocking_kk.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.studies import style  # noqa: E402

CL = ROOT / "cluster"
FIGDIR = ROOT / "document/fig/transport_sweeps"

# (run dir, label, converged-by-both-gates)
ARMS = [
    ("c16-half", r"$16\times1$, half", True),
    ("c16x2h",   r"$8\times2$, half",  True),
    ("c16-kk",   r"$16\times1$, KK",   False),
]


def load(name):
    z = np.load(CL / name / "run.npz", allow_pickle=True)
    dw = float(np.asarray(z["frequency_cell_widths"], float)[1])
    return (float(z["lead_current"]) * dw,      # integral units
            float(z["internal_spread"]),
            int(z["n_iter"]))


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    i_ball, _, _ = load("c16-ball")

    rows = [(lab, *load(d), ok) for d, lab, ok in ARMS]
    y = np.arange(len(rows))[::-1]          # first row at the top

    fig, (ax_r, ax_s) = style.doc_figure(ncols=2, aspect=0.42)

    for yi, (lab, I, spread, nit, ok) in zip(y, rows):
        # Converged arms are solid; the arm that met neither gate is drawn
        # hollow with a hatch, so "not converged" is not carried by colour.
        c = style.C_ANHARMONIC if ok else style.C_THIRD
        kw = (dict(color=c) if ok else
              dict(facecolor="none", edgecolor=c, hatch="///", linewidth=1.2))
        ax_r.barh(yi, I / i_ball, height=0.55, **kw)
        ax_s.barh(yi, spread, height=0.55, **kw)
        ax_r.text(I / i_ball + 0.015, yi, f"{I / i_ball:.3f}",
                  va="center", ha="left", fontsize=7.5, color="0.15")
        ax_s.text(spread * 1.25, yi, f"{spread:.1e}".replace("e-0", r"$\times10^{-}$")
                  if False else f"{spread:.1e}",
                  va="center", ha="left", fontsize=7.5, color="0.15")
        if not ok:
            ax_r.text(0.02, yi - 0.42, f"did not converge in {nit} iterations",
                      va="top", ha="left", fontsize=6.8, color=c, style="italic")

    ax_r.axvline(1.0, color=style.C_BALLISTIC, lw=1.4, zorder=0)
    ax_r.text(1.0, y[0] + 0.55, "ballistic", color=style.C_BALLISTIC,
              fontsize=7.5, ha="right", va="bottom")

    for ax in (ax_r, ax_s):
        ax.set_yticks(y)
        ax.set_ylim(y.min() - 0.85, y.max() + 0.95)
    ax_r.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax_s.set_yticklabels([])       # categories are shared; label them once
    ax_r.set_xlabel(r"$r = G/G_{\mathrm{ball}}$")
    ax_r.set_xlim(0, 1.12)
    ax_s.set_xscale("log")
    ax_s.set_xlabel("interior heat-profile spread")
    ax_s.set_xlim(2e-4, 4.0)
    ax_s.axvline(1e-2, color=style.C_REFERENCE, lw=1.0, ls=":", zorder=0)
    ax_s.text(1.1e-2, y[0] + 0.55, "conservation gate", color=style.C_REFERENCE,
              fontsize=7, ha="left", va="bottom")

    style.save(fig, "cnt_blocking_kk", directory=FIGDIR)


if __name__ == "__main__":
    main()
