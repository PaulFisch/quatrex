#!/usr/bin/env python
"""End-to-end profile of the multi-slab dense SCBA path (no cutoffs).

Runs a single SCBA point at the user-supplied (n_slabs, ne, max_iter)
under :mod:`cProfile`, dumps the raw stats to
``out/profiles/multislab_<n_slabs>x<ne>x<iter>.prof``, and prints
the top entries by cumulative and total (self) time. Also prints a
small breakdown of:

  * ``bubble_dense`` (per-call + total wall in the SSE kernel),
  * ``compute_phph_self_energy_finite_multi_slab`` (per-iter SSE),
  * ``solve_green_batch`` (per-iter Dyson solve),
  * ``build_retarded`` (per-iter Hilbert/PV),
  * ``build_device_fc3_blocks`` (one-shot, outside the SCBA loop).

Threading
---------
The SSE kernel parallelises over individual bubble calls via a
``ThreadPoolExecutor``. To exploit the parallelism on a many-core host:

  * Set ``QUATREX_PHPH_THREADS=<N>`` to pick the worker count
    explicitly (default = ``os.cpu_count()``).
  * Set ``OPENBLAS_NUM_THREADS=1`` and ``OMP_NUM_THREADS=1`` to stop
    BLAS from over-subscribing the same cores the worker pool is
    using. The kernel uses ``threadpoolctl`` to do this
    automatically inside the parallel region, but exporting the env
    vars at job-launch is the cleanest way to make sure no other
    library oversubscribes.

cProfile + threads is well known to serialise: the profiler holds an
internal lock that turns the parallel work into a sequence. Use this
script with ``QUATREX_PHPH_THREADS=1`` for hot-spot analysis, and a
plain (non-cProfile) timer for wall-time measurements of the parallel
scaling.

Cluster invocation example::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
    QUATREX_PHPH_THREADS=32 \\
        /home/paul/miniconda3/envs/quatrex-dev/bin/python \\
        phonon/scripts/profile_multislab.py \\
        --n-slabs 4 --ne 41 --max-iter 2 --top 30
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PHONON_DIR = _REPO_ROOT / "phonon"
if str(_PHONON_DIR) not in sys.path:
    sys.path.insert(0, str(_PHONON_DIR))


def _print_env_threads() -> None:
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "QUATREX_PHPH_THREADS",
    )
    print("  thread env: " + ", ".join(
        f"{k}={os.environ.get(k, '?')}" for k in keys
    ))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path,
        default=_REPO_ROOT / "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=_REPO_ROOT / "phonon/scripts/out/profiles",
    )
    parser.add_argument("--n-slabs", type=int, default=2)
    parser.add_argument("--ne", type=int, default=21,
                        help="number of frequency points "
                        "(must be odd for the symmetric grid)")
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--delta-T", type=float, default=10.0)
    parser.add_argument("--transport-direction", default="z",
                        choices=("x", "y", "z"))
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--sigma-cutoff", type=str, default="None",
        help="None | <int>. None = no truncation (multi-slab).",
    )
    parser.add_argument(
        "--vertex-cutoff", type=str, default="None",
        help="None | <int>.",
    )
    parser.add_argument(
        "--g-cutoff", type=str, default="None",
        help="None | <int>.",
    )
    parser.add_argument(
        "--dc-handling", default="interpolate",
        choices=("zero", "interpolate", "keep"),
    )
    args = parser.parse_args()

    def _cutoff(v):
        return None if str(v).lower() in ("none", "inf", "null") else int(v)

    from phonon.finite_analysis.loader import load_system
    from phonon.solver.dense import transmission_finite

    print("=== profile_multislab ===", flush=True)
    print(f"  n_slabs={args.n_slabs}, ne={args.ne}, max_iter={args.max_iter}",
          flush=True)
    print(f"  cutoffs: sigma={args.sigma_cutoff}, vertex={args.vertex_cutoff}, "
          f"g={args.g_cutoff}, dc={args.dc_handling}", flush=True)
    _print_env_threads()

    bundle = load_system(
        args.config, validate=False,
        transport_axis="xyz".index(args.transport_direction),
    )
    fc3_hdf5 = bundle.meta["fc3_path"]
    print(f"  fc3: {fc3_hdf5}", flush=True)

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    res = transmission_finite(
        bundle.phonon, fc3_hdf5=str(fc3_hdf5),
        freq_range_thz=(0.01, 18.0, args.ne),
        transport_direction=args.transport_direction,
        eta_factor=0.05,
        temperature=args.temperature, delta_T=args.delta_T,
        max_scba_iter=args.max_iter, scba_tol=1e-3,
        conservation_tol=1e-2,
        mixing=0.5, anderson_mixing=False,
        n_slabs=args.n_slabs, verbose=False,
        sigma_cutoff=_cutoff(args.sigma_cutoff),
        vertex_cutoff=_cutoff(args.vertex_cutoff),
        g_cutoff=_cutoff(args.g_cutoff),
        dc_handling=args.dc_handling,
    )
    profiler.disable()
    wall = time.perf_counter() - t0

    print(f"\n  wall total: {wall:.1f} s", flush=True)
    print(f"  G_ball     : {res['thermal_conductance_ballistic']:.3g}")
    print(f"  G_anh      : {res['thermal_conductance_anharmonic']:.3g}")
    print(f"  conservation: {res['heat_flow_conservation']:.3e}")
    print(f"  SCBA iters  : {res['n_scba_iterations']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prof_path = args.out_dir / (
        f"multislab_n{args.n_slabs}_ne{args.ne}_iter{args.max_iter}.prof"
    )
    profiler.dump_stats(prof_path)
    print(f"\n  raw stats: {prof_path}", flush=True)

    stats = pstats.Stats(profiler)

    print(f"\n=== Top {args.top} by cumulative time ===")
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(args.top)

    print(f"\n=== Top {args.top} by total (self) time ===")
    pstats.Stats(profiler).sort_stats("tottime").print_stats(args.top)

    print("\n=== Hot-spot breakdown ===")
    keys_of_interest = (
        "bubble_dense",
        "compute_phph_self_energy_finite_multi_slab",
        "build_device_fc3_blocks",
        "solve_green_batch",
        "build_retarded",
        "hilbert_transform_axis",
        "fft",
        "ifft",
        "tensordot",
    )
    stats_dict = stats.stats
    rows = []
    for func, (cc, nc, tt, ct, _) in stats_dict.items():
        name = func[2]
        if any(k in name for k in keys_of_interest):
            rows.append((name, nc, tt, ct, func[0]))
    rows.sort(key=lambda r: -r[3])
    print(f"  {'function':50s} {'calls':>8s} {'self_s':>10s} {'cum_s':>10s}")
    for name, calls, self_t, cum_t, _file in rows[:20]:
        print(f"  {name:50s} {calls:>8d} {self_t:>10.2f} {cum_t:>10.2f}")


if __name__ == "__main__":
    main()
