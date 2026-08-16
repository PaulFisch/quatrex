"""Interlayer shear and breathing frequencies -- the experimental check on
which MoS2 force-constant fit to believe.

Usage::
    python -m phonon.studies._interlayer_modes cluster/mos2_film_reap \
        cluster/mos2_film_reap_ls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "phonon", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EXPT_SHEAR_THZ = 1.0
EXPT_BREATHING_THZ = 1.7


def modes(reap: Path) -> dict:
    """Gamma frequencies of one reap: the rigid-layer shear and breathing.

    Split out of ``main`` so the report figure (``phonon/scripts/figures/
    r1_interlayer_modes.py``) computes the same numbers this prints, rather
    than a second implementation of them.

    The three lowest Gamma modes are acoustic and must be zero; the shear is
    the doubly-degenerate pair above them and the breathing mode the next
    one. ``acoustic`` is returned so a caller can check the sum rule.
    """
    from phonon.studies.engine.build_inputs import _load_bulk_hiphive

    phonon, _ = _load_bulk_hiphive(Path(reap).resolve())
    freqs = np.real(phonon.get_frequencies_with_eigenvectors(np.zeros(3))[0])
    ordered = np.sort(freqs)
    acoustic, optical = ordered[:3], ordered[3:]
    shear, breathing = float(optical[0]), float(optical[2])
    return {
        "name": Path(reap).name,
        "acoustic": acoustic,
        "shear": shear,
        "breathing": breathing,
        "splitting": breathing - shear,
        "shear_pair": (float(optical[0]), float(optical[1])),
        "d_shear": abs(shear - EXPT_SHEAR_THZ) / EXPT_SHEAR_THZ,
        "d_breathing": abs(breathing - EXPT_BREATHING_THZ) / EXPT_BREATHING_THZ,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("reaps", nargs="+", type=Path)
    args = p.parse_args(argv)

    print(f"experiment: shear {EXPT_SHEAR_THZ:.2f} THz, "
          f"breathing {EXPT_BREATHING_THZ:.2f} THz (2H-MoS2 bulk)\n")
    print(f"{'reap':<26} {'shear':>8} {'breathing':>10} {'splitting':>10} "
          f"{'|d| shear':>10} {'|d| breath':>11}")
    for reap in args.reaps:
        m = modes(reap)
        if np.abs(m["acoustic"]).max() > 1e-2:
            print(f"  WARNING {m['name']}: acoustic modes at Gamma are "
                  f"{m['acoustic'].round(4)}, not zero -- the fit violates the "
                  f"acoustic sum rule and the numbers below are unreliable")
        print(f"{m['name']:<26} {m['shear']:>8.4f} {m['breathing']:>10.4f} "
              f"{m['splitting']:>10.4f} "
              f"{m['d_shear']:>9.1%} {m['d_breathing']:>10.1%}")
        if not np.isclose(*m["shear_pair"], rtol=1e-3):
            print(f"  note {m['name']}: the shear pair is not degenerate "
                  f"({m['shear_pair'][0]:.4f}, {m['shear_pair'][1]:.4f}) -- "
                  f"the mode assignment above assumes it is")
    print("\nsplitting is the discriminator: the two rigid-layer modes are well "
          "separated in experiment (~0.7 THz), and a fit that collapses them is "
          "reporting the wrong interlayer physics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
