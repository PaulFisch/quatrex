"""Energy-grid resolution at eta=0 (fig:res_grid_comb, fig:res_grid_lottery).

  grid_comb     (a) the ballistic spectral heat current of the MoS2
                film on the resolution-matched non-uniform grid
                (262 points, min spacing 0.020 THz): a comb of sharp
                lines; the inset zooms the interlayer band with the
                grid points marked. The narrowest resolved features
                are a few grid spacings wide -- at eta=0 the line
                width is set by the physics (escape rate), not by any
                broadening, and a uniform grid at the coarse spacings
                affordable over the full band steps OVER them.
                (b) the CNT L4 converged A/B: the 287-point
                non-uniform grid against the 361-point uniform grid,
                spectral currents overlaid; integrals agree to 0.9%.
  grid_lottery  the uniform-grid ne scan on the CNT L4 fixed point
                under identical linear mixing: diverged at ne=161,
                converged at 201, neither at 271 (350 iterations),
                diverged again at 361 -- convergence is a lottery in
                the grid density because the grid-to-line registration
                changes with ne; the non-uniform grid (nu point)
                converges and needs no lottery.

Data: phonon/scripts/data/grid_aliasing.npz (distilled by
_extract_grid_aliasing.py). All currents integral-convention
(uniform sums multiplied by dw). d5a ladder context printed: both
uniform legs (nf=181, 721) diverge -- density alone does not converge
d5a; its eta=0 record needs the guarded Anderson scheme
(sec:res_campaign).

Run:  python phonon/scripts/figures/energy_grid_aliasing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.studies import style

DATA = ROOT / "phonon/scripts/data/grid_aliasing.npz"
FIGDIR = ROOT / "document/fig/transport_sweeps"


def feature_widths(e: np.ndarray, y: np.ndarray) -> list[float]:
    """FWHM of the peaks in y(e), by local half-max crossing."""
    widths = []
    ymax = y.max()
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] > 0.05 * ymax:
            half = y[i] / 2
            lo = i
            while lo > 0 and y[lo] > half:
                lo -= 1
            hi = i
            while hi < len(y) - 1 and y[hi] > half:
                hi += 1
            widths.append(e[hi] - e[lo])
    return widths


def fig_comb(d) -> None:
    fig, (ax_a, ax_b) = style.figure(ncols=2, width=4.4, height=3.3)
    colors = style.RC["axes.prop_cycle"].by_key()["color"]

    e, y = d["film_e"], d["film_ball_spec"]
    ax_a.plot(e, y, color=colors[0], lw=0.9)
    ax_a.set_xlabel(r"$\omega$ (THz)")
    ax_a.set_ylabel("spectral heat current (arb.)")
    ax_a.set_xlim(0, 16)
    axins = ax_a.inset_axes([0.42, 0.45, 0.55, 0.5])
    m = (e > 0.5) & (e < 2.5)
    axins.plot(e[m], y[m], color=colors[0], lw=0.9)
    axins.plot(e[m], np.zeros(m.sum()), "|", color="0.4", markersize=3)
    axins.set_xlim(0.5, 2.5)
    axins.tick_params(labelsize=6.5)
    ax_a.indicate_inset_zoom(axins, edgecolor="0.6")

    ax_b.plot(d["cnt_uni_e"], d["cnt_uni_spec"]
              / max(d["cnt_uni_spec"].max(), 1e-30),
              color=colors[1], lw=0.9, label="uniform 361")
    ax_b.plot(d["cnt_nu2_e"], d["cnt_nu2_spec"]
              / max(d["cnt_nu2_spec"].max(), 1e-30),
              color=colors[0], lw=0.9, alpha=0.8, label="non-uniform 287")
    ax_b.set_xlabel(r"$\omega$ (THz)")
    ax_b.set_ylabel("spectral heat current (norm.)")
    ax_b.legend()

    style.save(fig, "grid_comb", directory=FIGDIR)

    wid = feature_widths(e, y)
    dw_min = float(np.diff(e).min())
    print(f"film comb: {len(wid)} resolved features; narrowest FWHM "
          f"{min(wid):.3f} THz at grid min spacing {dw_min:.3f} THz "
          f"({min(wid) / dw_min:.1f} spacings); "
          f"uniform equivalent at that spacing over 0-16 THz: "
          f"{int(16 / dw_min)} points vs {len(e)} non-uniform "
          f"({16 / dw_min / len(e):.1f}x)")
    print(f"CNT A/B integrals: nu {float(d['cnt_nu2_I']):.3f} vs uniform "
          f"{float(d['cnt_uni_I']):.3f} "
          f"({float(d['cnt_nu2_I']) / float(d['cnt_uni_I']) - 1:+.1%}); "
          f"resolution-matched uniform equivalent of the nu grid: "
          f"{int(55 / np.diff(d['cnt_nu2_e']).min())} points vs 287 "
          f"({55 / np.diff(d['cnt_nu2_e']).min() / 287:.1f}x)")


def fig_lottery(d) -> None:
    fig, ax = style.figure(width=5.0, height=3.2)
    colors = style.RC["axes.prop_cycle"].by_key()["color"]

    rows = d["nescan"]
    for ne, conv, div, cur, nit in rows:
        c = colors[2] if conv else (colors[3] if div else colors[1])
        mk = "o" if conv else ("x" if div else "^")
        y = cur if (conv or not div) else np.nan
        ax.plot([ne], [y], mk, color=c, markersize=8)
        lab = ("converged" if conv else
               ("diverged" if div else f"limbo ({int(nit)} it)"))
        ax.annotate(lab, (ne, cur if not div else 10.0),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8,
                    color=c)
    nu_i = float(d["cnt_nu2_I"])
    ax.axhline(nu_i, color=colors[0], lw=1.1, ls="--")
    ax.annotate("non-uniform 287 (converged)", (355, nu_i),
                textcoords="offset points", xytext=(0, -12), ha="right",
                fontsize=8, color=colors[0])
    ax.set_xlabel("uniform grid points $n_e$")
    ax.set_ylabel("lead heat current (integral units)")
    ax.set_ylim(9.4, 11.4)

    style.save(fig, "grid_lottery", directory=FIGDIR)

    print("ne scan (ne, conv, div, I, n_it):")
    for row in rows:
        print("  ", [int(row[0]), int(row[1]), int(row[2]),
                     round(float(row[3]), 3), int(row[4])])
    print("d5a uniform ladder (nf, conv, div, n_it):",
          [[int(x) for x in r] for r in d["d5a_ladder"]])


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA)
    fig_comb(d)
    fig_lottery(d)


if __name__ == "__main__":
    main()
