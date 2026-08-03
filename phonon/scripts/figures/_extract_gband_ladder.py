"""Distil the long-chain CNT g_band ladder into a committed dataset.

Sources (uncommitted GPU-campaign run dirs, pulled from daint):
  cluster/l{16,24,32}f-{g3,g1t}/run.npz      ne=161 pairs (2x4 GH200)
  cluster/l16f-{g3,g1t}-361/run.npz          ne=361 grid check at L16

Writes phonon/scripts/data/gband_ladder.npz with one row per run:
  (L, band [3 or 1=taper], ne, converged, I_integral, n_iter,
   internal_spread) -- lead currents converted to the integral
  convention (sum x dw on uniform grids).

Run:  python phonon/scripts/figures/_extract_gband_ladder.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "phonon/scripts/data/gband_ladder.npz"

RUNS = [
    (16, 3, "cluster/l16f-g3/run.npz"),
    (16, 1, "cluster/l16f-g1t/run.npz"),
    (24, 3, "cluster/l24f-g3/run.npz"),
    (24, 1, "cluster/l24f-g1t/run.npz"),
    (32, 3, "cluster/l32f-g3/run.npz"),
    (32, 1, "cluster/l32f-g1t/run.npz"),
    (16, 3, "cluster/l16f-g3-361/run.npz"),
    (16, 1, "cluster/l16f-g1t-361/run.npz"),
]


def main() -> None:
    rows = []
    for L, band, rel in RUNS:
        d = np.load(ROOT / rel)
        e = np.asarray(d["energies"])
        dw = float(np.diff(e).mean())
        lc = float(d["lead_current"])
        if bool(d.get("uniform_frequency_grid", True)):
            lc *= dw
        rows.append([L, band, len(e), int(d["converged"]), lc,
                     int(d["n_iter"]), float(d["internal_spread"])])
    arr = np.array(rows)
    np.savez_compressed(OUT, ladder=arr)
    print(f"wrote {OUT}")
    for r in rows:
        print(f"  L{int(r[0]):2d} band{int(r[1])} ne={int(r[2]):3d} "
              f"conv={int(r[3])} I={r[4]:8.3f} it={int(r[5]):3d} "
              f"spread={r[6]:.3g}")


if __name__ == "__main__":
    main()
