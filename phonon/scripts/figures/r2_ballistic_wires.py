"""Results 2: ballistic conductance of the two silicon nanowires.

Run:  python phonon/scripts/figures/r2_ballistic_wires.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QTX_ARRAY_MODULE", "numpy")

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT.parent, ROOT.parent / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np                                  # noqa: E402

from phonon.studies import style                    # noqa: E402

NPZ = ROOT / "studies" / "out" / "ballistic_curves.npz"
# Both series here ARE ballistic, so neither may wear the chapter's
# anharmonic vermillion -- colour follows the entity, and vermillion
# already means "anharmonic" in every other figure. Blue+green also
# clears 3:1 contrast where the purple slot only WARNs.
# Validated pair: dE 18.5 normal / 17.7 protan, both >= 3:1.
WIRES = (("d5a", style.C_BALLISTIC, "o"), ("d11a", style.C_THIRD, "s"))


def main() -> int:
    z = np.load(NPZ, allow_pickle=True)
    wire, ns, T, G = z["wire"], z["n_slabs"], z["T"], z["G_ball"]

    fig, ax = style.doc_figure(frac=0.62, aspect=0.72)
    curves = {}
    for name, colour, marker in WIRES:
        m = (wire == name) & (ns == 1)
        order = np.argsort(T[m])
        t, g = T[m][order], G[m][order] / 1e7
        curves[name] = (t, g)
        ax.plot(t, g, marker=marker, color=colour, label=name)

    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"$G_\mathrm{ball}$ ($10^{7}$ W m$^{-2}$K$^{-1}$)")
    ax.legend(loc="upper left")

    # The ratio is the claim in the text, so it is drawn rather than asserted.
    t5, g5 = curves["d5a"]
    t11, g11 = curves["d11a"]
    assert np.array_equal(t5, t11)
    ratio = g11 / g5
    # Lower right: the only region no series passes through. The inset used
    # to sit at upper left, directly on top of the d11a curve.
    # The ratio is a DERIVED quantity, not a third entity, so it wears the
    # reference grey rather than a series hue -- green now means d11a.
    inset = ax.inset_axes((0.54, 0.10, 0.42, 0.30))
    inset.plot(t5, ratio, color=style.C_REFERENCE, marker="^", ms=3.5, lw=1.1)
    inset.set_ylim(1.4, 1.9)
    inset.set_xlabel("T (K)", fontsize=7, labelpad=1)
    inset.set_ylabel("ratio", fontsize=7, labelpad=1)
    inset.tick_params(labelsize=6.5, pad=1)
    inset.grid(alpha=0.25)

    style.save(fig, "r2_ballistic_wires", directory=style.DOC_FIG)
    for name, (t, g) in curves.items():
        print(f"  {name:6s} G(300 K) = {g[t == 300][0]:.4f}e7 W/m^2/K")
    print(f"  ratio d11a/d5a = {ratio.min():.3f}-{ratio.max():.3f} "
          f"over {t5.min():.0f}-{t5.max():.0f} K, "
          f"{ratio[t5 == 300][0]:.3f} at 300 K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
