#!/usr/bin/env python
"""Profile a single SCBA point for d5a (current phonon.solver.dense path).

Runs ``transmission_finite`` once with a small ω grid and a 3-iteration
cap so the run finishes in tens of seconds, wrapped in :mod:`cProfile`.
Prints the top-N cumulative-time entries and writes the raw stats to
``profile_d5_dense.prof`` so the user can re-inspect with
``snakeviz`` / ``gprof2dot``.
"""

from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

# --- sys.path bootstrap (same as d5_transport_sweep.py) ----------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PHONON_DIR = _REPO_ROOT / "phonon"
if str(_PHONON_DIR) not in sys.path:
    sys.path.insert(0, str(_PHONON_DIR))

from phonon.finite_analysis.loader import load_system  # noqa: E402
from phonon.solver.dense import transmission_finite  # noqa: E402


def run_one() -> None:
    cfg = _REPO_ROOT / "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml"
    bundle = load_system(cfg, validate=False, transport_axis=2)
    fc3_hdf5 = bundle.meta["fc3_path"]
    print(f"  Profiling single SCBA point, fc3={fc3_hdf5}", flush=True)

    t0 = time.time()
    _ = transmission_finite(
        bundle.phonon,
        fc3_hdf5=str(fc3_hdf5),
        freq_range_thz=(0.01, 25.0, 100),
        transport_direction="z",
        eta_factor=0.05,
        temperature=300.0,
        delta_T=10.0,
        max_scba_iter=8,
        scba_tol=1e-5,
        conservation_tol=1e-3,
        mixing=0.5,
        anderson_mixing=True,
        n_slabs=2,
        verbose=True,
    )
    print(f"  Total wall: {time.time() - t0:.1f} s", flush=True)


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    run_one()
    profiler.disable()

    out_prof = Path(__file__).resolve().parent / "profile_d5_dense.prof"
    profiler.dump_stats(out_prof)
    print(f"\nRaw stats: {out_prof}", flush=True)

    print("\n=== Top 30 by cumulative time ===")
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(30)

    print("\n=== Top 30 by total (self) time ===")
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(30)


if __name__ == "__main__":
    main()
