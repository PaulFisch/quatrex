"""Results 1: the rigid-layer modes that decide which MoS2 fit to believe.

A level diagram. The two lowest optical modes of 2H-MoS2 at Gamma are the
rigid-layer shear (layers sliding) and breathing (layers separating) modes.
They are set by the interlayer force constants the cross-plane problem
depends on, and they are measured -- so they judge a fit with no transport
calculation in the loop.

The splitting is the discriminator, which is why the figure draws it: a
sparsifying prior that prunes the cross-gap third-order blocks also flattens
the harmonic doublet, and the collapse is visible at a glance.

Frequencies come from phonon.studies._interlayer_modes.modes(), the same
function the text report calls, so the figure cannot drift from the number
in the table.

Run:  QTX_ARRAY_MODULE=numpy python phonon/scripts/figures/r1_interlayer_modes.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Offline analysis: no GPU, and identical numbers on any machine. Must be set
# before the first quatrex/qttools import.
os.environ.setdefault("QTX_ARRAY_MODULE", "numpy")

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT.parent, ROOT.parent / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phonon.studies import style                      # noqa: E402
from phonon.studies._interlayer_modes import (        # noqa: E402
    EXPT_BREATHING_THZ, EXPT_SHEAR_THZ, modes,
)

REAPS = [
    (ROOT.parent / "cluster" / "mos2_film_reap", "sparsifying\n(ARD)"),
    (ROOT.parent / "cluster" / "mos2_film_reap_ls", "least\nsquares"),
]


def main() -> int:
    rows = [(modes(path), label) for path, label in REAPS]
    rows.append(
        ({"shear": EXPT_SHEAR_THZ, "breathing": EXPT_BREATHING_THZ,
          "splitting": EXPT_BREATHING_THZ - EXPT_SHEAR_THZ},
         "experiment"),
    )

    fig, ax = style.doc_figure(frac=0.62, aspect=0.86)
    half = 0.30

    # The measured values run the width of the panel, so each fit's deviation
    # is read against them directly rather than from the tick labels.
    for freq in (EXPT_SHEAR_THZ, EXPT_BREATHING_THZ):
        ax.axhline(freq, color=style.C_REFERENCE, lw=0.7, ls=":", zorder=1)

    for i, (m, label) in enumerate(rows):
        is_expt = label == "experiment"
        colour = style.C_REFERENCE if is_expt else (
            style.C_ANHARMONIC if i == 0 else style.C_BALLISTIC)
        for freq in (m["shear"], m["breathing"]):
            ax.hlines(freq, i - half, i + half, color=colour, lw=2.6, zorder=3)
        # The splitting is the discriminator, so it is the only annotation
        # each column carries. Below ~0.1 THz the arrowheads have no room,
        # so the value moves off to the side and the arrow is dropped.
        gap = m["splitting"]
        if gap > 0.12:
            ax.annotate("", xy=(i, m["breathing"]), xytext=(i, m["shear"]),
                        arrowprops=dict(arrowstyle="<->", color=colour,
                                        lw=1.0, shrinkA=0, shrinkB=0))
            ax.text(i + 0.05, 0.5 * (m["shear"] + m["breathing"]),
                    f"{gap:.2f}", color=colour, fontsize=9,
                    ha="left", va="center")
        else:
            ax.text(i, m["breathing"] + 0.045, f"{gap:.2f}", color=colour,
                    fontsize=9, ha="center", va="bottom")

    # Name the two branches once, on the column where they are far apart.
    last = len(rows) - 1
    for freq, name in ((EXPT_SHEAR_THZ, "shear"),
                       (EXPT_BREATHING_THZ, "breathing")):
        ax.text(last + half + 0.08, freq, name, fontsize=8.5,
                color=style.C_REFERENCE, ha="left", va="center")

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([label for _, label in rows])
    ax.set_xlim(-0.55, len(rows) - 0.05)
    ax.set_ylabel(r"$\Gamma$ frequency (THz)")
    ax.set_ylim(0.85, 1.82)
    ax.grid(axis="x", visible=False)

    style.save(fig, "r1_interlayer_modes", directory=style.DOC_FIG)

    # the claim-verification trail: every number the caption or text quotes
    for m, label in rows:
        print(f"  {label.replace(chr(10), ' '):<20} "
              f"shear {m['shear']:.4f}  breathing {m['breathing']:.4f}  "
              f"splitting {m['splitting']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
