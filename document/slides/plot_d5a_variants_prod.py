"""d5a SiNW static-SE variants from the PRODUCTION runs (cluster/prod/d5a_variants).

Unlike the dense se_study, prod completed only bubble / bubble+KK /
bubble+tadpole -- every loop variant (bub_loop, bub_loop_tad, bub_kk_loop_tad)
crashed with a NaN loop stage. All completed runs hit the 60-iteration cap
(converged=False) and oscillate (period-2 mixing instability); the headline
ratios are the best-iteration values.

  (a) G_anh/G_ball vs T          <- lead_current / ballistic
  (b) spectral heat current @300K <- current_spectrum (DC point masked)
  (c) SCBA convergence            <- iter_heat (|dJ|/J per iteration) + conservation
  (d) ratio vs device length      <- L2 vs L3
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "cluster" / "prod" / "d5a_variants"

C = {"blue": "#4477AA", "green": "#228833", "purple": "#AA3377",
     "red": "#EE6677", "grey": "#BBBBBB"}
VAR = [("bub", "bubble", C["blue"], "o-"),
       ("bub_kk", "bubble + full KK", C["green"], "s-"),
       ("bub_tad", "bubble + tadpole", C["purple"], "^-")]
TS = [30, 100, 300]

plt.rcParams.update({
    "font.size": 10, "legend.fontsize": 8, "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True, "lines.linewidth": 1.8,
})


def npz(name):
    return np.load(SRC / name, allow_pickle=True)


def lead(z):
    return float(np.atleast_1d(z["lead_current"]).ravel()[0])


def ratio(tag, v):
    return lead(npz(f"{tag}_{v}.npz")) / lead(npz(f"{tag}-ball_ball.npz"))


def tag(ax, letter):
    ax.text(0.025, 0.97, letter, transform=ax.transAxes, fontweight="bold",
            va="top", ha="left")


def main():
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 7.0))

    # (a) ratio vs T -----------------------------------------------------
    a = ax[0, 0]
    for v, lbl, col, mk in VAR:
        r = [ratio(f"T{T}", v) for T in TS]
        a.plot(TS, r, mk, color=col, label=lbl)
    a.axhline(1.0, ls=":", color=C["grey"])
    a.set_ylim(0.55, 1.05); a.set_xlabel("temperature [K]")
    a.set_ylabel(r"$G_{\rm anh}/G_{\rm ball}$")
    a.annotate("bubble(30 K)=123\n(no-KK blow-up, off-scale)", xy=(30, 0.80),
               xytext=(70, 0.62), fontsize=7, color=C["blue"],
               arrowprops=dict(arrowstyle="->", color=C["blue"], lw=0.8))
    a.legend(loc="upper right"); tag(a, "(a)")

    # (b) spectral heat current @300 K (DC point masked) -----------------
    b = ax[0, 1]
    zb = npz("T300-ball_ball.npz")
    e = np.asarray(zb["energies"])[1:]
    b.fill_between(e, 0, np.asarray(zb["current_spectrum"])[1:, 0], color=C["grey"],
                   alpha=0.6, label="ballistic")
    for v, lbl, col, mk in VAR:
        cs = np.asarray(npz(f"T300_{v}.npz")["current_spectrum"])[1:, 0]
        b.plot(e, cs, "-", color=col, lw=1.5, label=f"{lbl} ($G_a/G_b$={ratio('T300', v):.3f})")
    b.set_xlabel(r"$\omega$ [THz]"); b.set_ylabel(r"spectral heat current $J(\omega)$ [a.u.]")
    b.set_xlim(0, e.max()); b.legend(loc="upper right"); tag(b, "(b)")

    # (c) convergence: |dJ|/J per iteration + conservation ---------------
    c = ax[1, 0]
    for v, lbl, col, mk in VAR:
        z = npz(f"T300_{v}.npz")
        J = np.asarray(z["iter_heat"])[:, 0]
        res = np.abs(np.diff(J)) / np.maximum(np.abs(J[1:]), 1e-30)
        cons = float(np.atleast_1d(z["best_cons"]).ravel()[0])
        conv = bool(np.atleast_1d(z["converged"]).ravel()[0])
        c.semilogy(np.arange(1, len(res) + 1), res, "-", color=col,
                   label=f"{lbl} [{'conv' if conv else 'no conv'}, cons {cons:.1e}]")
    c.axhline(1e-3, ls=":", color=C["grey"]); c.text(1, 1.3e-3, "scba_tol", fontsize=7, color="grey")
    c.set_xlabel("SCBA iteration"); c.set_ylabel(r"$|\Delta J|/J$ per iteration")
    c.legend(loc="lower left", fontsize=7); tag(c, "(c)")

    # (d) ratio vs device length (L2 = T300, L3) -------------------------
    d = ax[1, 1]
    Ls = [2, 3]
    for v, lbl, col, mk in VAR:
        r = [ratio("T300", v), ratio("L3", v)]
        d.plot(Ls, r, mk, color=col, label=lbl)
    d.axhline(1.0, ls=":", color=C["grey"])
    d.set_xticks(Ls); d.set_xlabel("device length [unit cells]")
    d.set_ylabel(r"$G_{\rm anh}/G_{\rm ball}$ @300 K")
    d.legend(loc="upper right"); tag(d, "(d)")

    fig.tight_layout()
    out = HERE / "fig" / "d5a_variants_prod.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
