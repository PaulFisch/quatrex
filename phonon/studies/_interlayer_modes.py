"""Interlayer shear and breathing frequencies -- the experimental check on which
MoS2 force-constant fit to believe.

The two lowest optical modes at Gamma of 2H-MoS2 are rigid-layer modes: a
doubly-degenerate shear (E, layers sliding) and a breathing mode (A, layers
moving apart). They are the degrees of freedom that carry cross-plane heat, they
are set by the same interlayer force constants the cross-plane ladder depends
on, and they are measured: shear ~1.0 THz (32 cm^-1), breathing ~1.7 THz
(57 cm^-1).

So they are a free, independent judgement on a fit -- no transport calculation
in the loop.

Usage::

    python -m phonon.studies._interlayer_modes cluster/mos2_film_reap \\
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("reaps", nargs="+", type=Path)
    args = p.parse_args(argv)

    from phonon.studies.engine.build_inputs import _load_bulk_hiphive

    print(f"experiment: shear {EXPT_SHEAR_THZ:.2f} THz, "
          f"breathing {EXPT_BREATHING_THZ:.2f} THz (2H-MoS2 bulk)\n")
    print(f"{'reap':<26} {'shear':>8} {'breathing':>10} {'splitting':>10} "
          f"{'|d| shear':>10} {'|d| breath':>11}")
    for reap in args.reaps:
        phonon, _ = _load_bulk_hiphive(reap.resolve())
        freqs = np.real(
            phonon.get_frequencies_with_eigenvectors(np.zeros(3))[0]
        )
        acoustic = np.sort(freqs)[:3]
        if np.abs(acoustic).max() > 1e-2:
            print(f"  WARNING {reap.name}: acoustic modes at Gamma are "
                  f"{acoustic.round(4)}, not zero -- the fit violates the "
                  f"acoustic sum rule and the numbers below are unreliable")
        optical = np.sort(freqs)[3:]
        shear, breathing = float(optical[0]), float(optical[2])
        print(f"{reap.name:<26} {shear:>8.4f} {breathing:>10.4f} "
              f"{breathing - shear:>10.4f} "
              f"{abs(shear - EXPT_SHEAR_THZ) / EXPT_SHEAR_THZ:>9.1%} "
              f"{abs(breathing - EXPT_BREATHING_THZ) / EXPT_BREATHING_THZ:>10.1%}")
        if not np.isclose(optical[0], optical[1], rtol=1e-3):
            print(f"  note {reap.name}: the shear pair is not degenerate "
                  f"({optical[0]:.4f}, {optical[1]:.4f}) -- the mode "
                  f"assignment above assumes it is")
    print("\nsplitting is the discriminator: the two rigid-layer modes are well "
          "separated in experiment (~0.7 THz), and a fit that collapses them is "
          "reporting the wrong interlayer physics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
