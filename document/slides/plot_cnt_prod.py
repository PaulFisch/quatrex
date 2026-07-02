"""CNT(3,3) and CNT(8,0) transport from the PRODUCTION runs (cluster/prod).

  (a) CNT(3,3) G_anh/G_ball vs T           <- cnt33/summary.csv
  (b) G_anh/G_ball vs device length         <- cnt33 + cnt80 summary.csv
  (c) CNT(3,3) spectral heat current @300 K <- prod current_spectrum (anh) vs
  (d) CNT(8,0) spectral heat current @300 K    ballistic shape (intrinsic), both
                                               normalised to the prod conductances.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PROD = HERE.parents[1] / "cluster" / "prod"
EXPL = HERE.parents[1] / "cluster"

C = {"cnt33": "#228833", "cnt80": "#CCBB44", "ball": "#BBBBBB",
     "anh": "#AA3377", "blue": "#4477AA"}

plt.rcParams.update({
    "font.size": 10, "legend.fontsize": 8, "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True, "lines.linewidth": 1.8,
})


def rows(system):
    with open(PROD / system / "summary.csv") as fh:
        return list(csv.DictReader(fh))


def tag(ax, s):
    ax.text(0.025, 0.97, s, transform=ax.transAxes, fontweight="bold", va="top")


def spectral(checkpoint):
    """Return (omega, J_ball, J_anh) as unit-area shapes from a matched run.

    Prod ballistic runs do not save a spectrum, and prod's current_spectrum
    uses a different omega-weighting than the spectral_heat_current arrays, so
    the only self-consistent anharmonic-vs-ballistic spectral PAIR comes from
    one transport checkpoint. Shown as normalised shapes (the quantitative
    G_anh/G_ball is in panels a,b from the prod summary).
    """
    z = np.load(EXPL / checkpoint, allow_pickle=True)
    w = np.asarray(z["freqs_thz"])
    jb = np.abs(np.asarray(z["spectral_heat_current_ballistic"]))
    ja = np.abs(np.asarray(z["spectral_heat_current"]))
    jb = jb / np.trapezoid(jb, w)
    ja = ja / np.trapezoid(ja, w)
    return w, jb, ja


def main():
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 7.0))

    # (a) CNT(3,3) ratio vs T
    a = ax[0, 0]
    tr = [r for r in rows("cnt33") if r["sweep"] == "temperature"]
    T = [float(r["t_mean"]) for r in tr]; rat = [float(r["ratio"]) for r in tr]
    a.plot(T, rat, "o-", color=C["cnt33"], label="CNT(3,3)")
    a.axhline(1.0, ls=":", color=C["ball"])
    a.set_xlabel("temperature [K]"); a.set_ylabel(r"$G_{\rm anh}/G_{\rm ball}$")
    a.set_ylim(0.84, 1.01); a.legend(loc="lower left"); tag(a, "(a)")

    # (b) ratio vs length, both tubes
    b = ax[0, 1]
    for sysn, col, lbl in [("cnt33", C["cnt33"], "CNT(3,3)"), ("cnt80", C["cnt80"], "CNT(8,0)")]:
        lr = [r for r in rows(sysn) if r["sweep"] == "length"]
        L = [int(r["n_slabs"]) for r in lr]; rat = [float(r["ratio"]) for r in lr]
        b.plot(L, rat, "s-" if sysn == "cnt33" else "D-", color=col, label=lbl)
    b.axhline(1.0, ls=":", color=C["ball"])
    b.set_xticks([2, 3, 4]); b.set_xlabel("device length [unit cells]")
    b.set_ylabel(r"$G_{\rm anh}/G_{\rm ball}$ @300 K"); b.legend(loc="lower left"); tag(b, "(b)")

    # (c,d) spectral heat current shape: anharmonic vs ballistic
    panels = [
        (ax[1, 0], "cnt33", "cnt33_tempsweep/checkpoints/temperature/T300_dT10_L1_scba.npz", "(c)", "CNT(3,3)"),
        (ax[1, 1], "cnt80", "cnt80_transport/checkpoints/length/L1_T300_dT10_scba.npz", "(d)", "CNT(8,0)"),
    ]
    for axp, sysn, ckpt, lt, lbl in panels:
        w, jb, ja = spectral(ckpt)
        axp.fill_between(w, 0, jb, color=C["ball"], alpha=0.7, label="ballistic")
        axp.plot(w, ja, "-", color=C["anh"], lw=1.6, label="anharmonic")
        axp.set_xlim(0, w[jb > jb.max() * 1e-3].max() * 1.05)
        axp.set_ylim(bottom=0)
        axp.set_xlabel(r"$\omega$ [THz]")
        axp.set_ylabel(r"normalised $J(\omega)$  ($\int\!=\!1$)")
        axp.text(0.5, 0.9, lbl, transform=axp.transAxes, ha="center", fontsize=9,
                 fontweight="bold", color=C[sysn])
        axp.legend(loc="upper right"); tag(axp, lt)

    fig.tight_layout()
    out = HERE / "fig" / "cnt_prod.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
