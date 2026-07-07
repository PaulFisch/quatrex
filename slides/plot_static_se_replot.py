"""d5a SiNW static-self-energy summary (old se_study data, dense solver).

Two panels:
  (a) self-energy magnitude vs T  -- loop / tadpole / bubble (Re shift, Im linewidth)
  (b) spectral heat current @300 K (loop+tadpole) vs ballistic, conservation annotated
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SNAP = HERE.parents[1] / "cluster" / "snapshots"          # 7-T coverage (panel a)
SE = HERE.parents[1] / "cluster" / "tortin3-tmp" / "se_study"  # all 4 modes have J @300K (panel b)

C = {"blue": "#4477AA", "green": "#228833", "yellow": "#CCBB44",
     "purple": "#AA3377", "red": "#EE6677", "grey": "#BBBBBB"}

plt.rcParams.update({"font.size": 10, "legend.fontsize": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True, "lines.linewidth": 1.8})


def load():
    d = {}
    for f in glob.glob(str(SNAP / "study_d5a_T*_*.npz")):
        z = np.load(f, allow_pickle=True)
        d[(float(z["temp"]), str(z["mode"]))] = z
    return d


def tag(ax, s):
    ax.text(0.025, 0.97, s, transform=ax.transAxes, fontweight="bold", va="top")


def main():
    data = load()
    temps = sorted({k[0] for k in data})

    def series(mode, field):
        return [float(data[(T, mode)][field]) if (T, mode) in data else np.nan
                for T in temps]

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.8))

    # (a) self-energy magnitude vs T
    a.semilogy(temps, series("loop", "sigma_static_norm"), "s-", color=C["blue"], label=r"$\|\Sigma_L\|$ loop")
    a.semilogy(temps, series("tadpole", "sigma_static_norm"), "^-", color=C["green"], label=r"$\|\Sigma_T\|$ tadpole")
    a.semilogy(temps, series("bubble", "reB"), "o-", color=C["yellow"], label=r"max$|{\rm Re}\,\Sigma_B|$ (shift)")
    a.semilogy(temps, series("bubble", "imB"), "v-", color=C["purple"], label=r"max$|{\rm Im}\,\Sigma_B|$ (linewidth)")
    a.set_xlabel("temperature [K]"); a.set_ylabel(r"self-energy magnitude [THz$^2$]")
    a.legend(loc="lower right"); tag(a, "(a)")

    # (b) spectral heat current @300 K: all four self-energy variants vs ballistic
    def se300(mode):
        return np.load(SE / f"study_d5a_T300_{mode}.npz", allow_pickle=True)
    z0 = se300("bubble")
    w = np.asarray(z0["freqs"]); jb = np.asarray(z0["J_ball"])
    b.fill_between(w, 0, jb, color=C["grey"], alpha=0.7, label="ballistic")
    for mode, lbl, col in [("bubble", "bubble", C["blue"]),
                           ("loop", "bubble+loop", C["green"]),
                           ("tadpole", "bubble+tadpole", C["yellow"]),
                           ("loop_tadpole", "bubble+loop+tadpole", C["red"])]:
        z = se300(mode)
        b.plot(np.asarray(z["freqs"]), np.asarray(z["J_anh"]), "-", color=col, lw=1.4,
               label=rf"{lbl}: $G_a/G_b{{=}}{float(z['Ga_over_Gb']):.3f}$, "
                     rf"cons {float(z['conservation']):.1e}")
    nz = w[jb > jb.max() * 1e-3]
    if nz.size:
        b.set_xlim(0, 1.08 * float(nz.max()))
    b.set_ylim(bottom=0)
    b.set_xlabel(r"$\omega$ [THz]"); b.set_ylabel(r"spectral heat current $J(\omega)$ [a.u.]")
    b.legend(loc="upper right", fontsize=7.5); tag(b, "(b)")

    fig.tight_layout()
    out = HERE / "fig" / "static_se_replot.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
