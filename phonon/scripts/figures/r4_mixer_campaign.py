"""Results 4.5: the mixer ranking inverts between the two systems.

Residual histories on the exact production bubble, one panel per system.
The claim the figure has to carry is that no scheme dominates: what wins on
the stiff nanotube is not what wins on the soft nanowire, which is why the
solver choices in this work rest on measured Jacobians rather than on the
resonance-gain model.

Four schemes per panel, because four is the ceiling of the validated
palette (see phonon/studies/style.py). They are chosen to span the
families -- damped linear, Anderson, and the two extrapolation/projection
routes -- not to be a complete sweep; the full sweep is the campaign report.

Run:  python phonon/scripts/figures/r4_mixer_campaign.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("QTX_ARRAY_MODULE", "numpy")

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT.parent, ROOT.parent / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np                                  # noqa: E402

from phonon.studies import style                    # noqa: E402

BASE = ROOT / "studies" / "out" / "anderson_test"
RES = re.compile(r"rel Sigma\^R residual ([0-9.eE+-]+)")

# (directory, label, colour). Colour is by SCHEME FAMILY and is the same in
# both panels, so a reader compares like with like across the inversion.
C = style
PANELS = [
    ("CNT(3,3), two cells", BASE / "mixer_campaign_L2", [
        ("lin02",   "linear",   C.C_REFERENCE),
        ("and_d8",  "Anderson", C.C_BALLISTIC),
        ("broyden", "Broyden",  C.C_ANHARMONIC),
        ("jfnk",    "JFNK",     C.C_THIRD),
    ]),
    ("\\texttt{d5a} nanowire", BASE / "mixer_campaign_d5a_v2", [
        ("and_d8",  "Anderson", C.C_BALLISTIC),
        ("rpm",     "RPM",      C.C_ANHARMONIC),
        ("rre_c8",  "RRE(8)",   C.C_THIRD),
        ("rre_c12", "RRE(12)",  "#cc78bc"),
    ]),
]


def residuals(log: Path) -> np.ndarray:
    """Residual history of the LAST run in a possibly-appended log."""
    text = log.read_text(errors="ignore")
    cut = text.rfind("Entering SCBA loop")
    if cut > 0:
        text = text[cut:]
    return np.array([float(m) for m in RES.findall(text)])


def main() -> int:
    fig, axes = style.doc_figure(ncols=2, aspect=0.42)
    for ax, (title, root, schemes) in zip(axes, PANELS):
        for sub, label, colour in schemes:
            log = root / sub / "run.log"
            if not log.exists():
                print(f"  MISSING {log}")
                continue
            r = residuals(log)
            if r.size == 0:
                print(f"  EMPTY   {log}")
                continue
            ax.semilogy(np.arange(1, r.size + 1), r, color=colour, lw=1.3,
                        label=label)
            print(f"  {root.name:24s} {label:9s} {r.size:4d} it  "
                  f"final {r[-1]:.2e}  best {r.min():.2e}")
        ax.set_xlabel("iteration")
        ax.set_ylim(1e-5, 5)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    axes[0].set_ylabel(r"rel $\Sigma^R$ residual")

    style.save(fig, "r4_mixer_campaign", directory=style.DOC_FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
