"""Long-chain CNT g_band ladder (fig:res_gband_ladder).

Data:
  phonon/scripts/data/gband_ladder.npz, distilled by
  _extract_gband_ladder.py from the cluster/l{16,24,32}f-* GPU runs
  (2x4 GH200, eta=0, all converged) and the L16 ne=361 pair
  (jobs 4321907/4321908).

Run:  python phonon/scripts/figures/gband_length_grid.py
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

DATA = ROOT / "phonon/scripts/data/gband_ladder.npz"
FIGDIR = ROOT / "document/fig/transport_sweeps"


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    lad = np.load(DATA)["ladder"]  # L, band, ne, conv, I, n_it, spread

    fig, (ax_a, ax_b) = style.doc_figure(ncols=2, aspect=0.38)
    colors = style.RC["axes.prop_cycle"].by_key()["color"]

    sel = {}
    for band, ne in ((3, 161), (1, 161), (3, 361), (1, 361)):
        m = (lad[:, 1] == band) & (lad[:, 2] == ne)
        sel[(band, ne)] = lad[m][np.argsort(lad[m][:, 0])]

    # Band 3 only. The band-1 Bartlett-taper arm was dropped from the
    # report: it converges, but to a catastrophically non-conserving
    # answer (interior spread up to 86 %), so it is not a bracket.
    c = colors[0]
    r = sel[(3, 161)]
    ax_a.plot(r[:, 0], r[:, 4], "o-", color=c, label="$b_G=3$, $n_e=161$")
    r3 = sel[(3, 361)]
    ax_a.plot(r3[:, 0], r3[:, 4], "o", color=c, markerfacecolor="none",
              markersize=8, label="$n_e=361$")
    ax_b.plot(r[:, 0], 100 * r[:, 6], "o-", color=c)
    ax_b.plot(r3[:, 0], 100 * r3[:, 6], "o", color=c,
              markerfacecolor="none", markersize=8)
    ax_a.set_xlabel("device length (transport cells)")
    ax_a.set_ylabel("lead heat current (integral units)")
    ax_a.set_ylim(0, 21)
    ax_a.legend(fontsize=7.5)
    ax_b.set_xlabel("device length (transport cells)")
    ax_b.set_ylabel("interior heat-profile spread (%)")

    style.save(fig, "gband_ladder", directory=FIGDIR)

    g3 = sel[(3, 161)]
    print("g3  I(L):", {int(L): round(i, 2) for L, i in zip(g3[:, 0], g3[:, 4])})
    i361_3 = sel[(3, 361)][0, 4]
    print(f"L16 grid check: g3 {g3[0, 4]:.3f} -> {i361_3:.3f} "
          f"({i361_3 / g3[0, 4] - 1:+.1%})")
    print("interior spread (%):",
          {f"L{int(L)} b{int(b)} ne{int(n)}": round(100 * s, 1)
           for L, b, n, s in zip(lad[:, 0], lad[:, 1], lad[:, 2],
                                 lad[:, 6])})
    print("all runs converged:", bool(lad[:, 3].all()))


if __name__ == "__main__":
    main()
