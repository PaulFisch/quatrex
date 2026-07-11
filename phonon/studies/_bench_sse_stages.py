"""Micro-benchmark: per-stage SSE cost across threading/allocator configs.

Runs the REAL production SCBA loop (a few iterations of the configured
device) in a child process per (ring-pool width x BLAS threads x malloc
knobs) combination -- both thread counts must be fixed before numpy loads,
hence the parent/child split -- and reads the per-stage times of the WP1
profiler ranges (``PhPh SSE: 1..5``) from the in-process eventlog. Prints a
ranked table and asserts that every combination produced the same Sigma
(the ring result is bit-identical for any pool width; BLAS/allocator knobs
must not change it either).

Usage (laptop smoke, tiny sweep):
    python phonon/studies/_bench_sse_stages.py \
        --config phonon/studies/out/anderson_test/local_L2/quatrex_config.toml \
        --ring 1,4 --blas 1,8 --iters 3

Cluster sweep (one idle tortin node, ~13 x iters minutes):
    python phonon/studies/_bench_sse_stages.py --config <cluster L2 toml> \
        --ring 1,8,32,64 --blas 1,8,0 --malloc default,arena2

June->July attribution (manual; old commits need era-matched configs since
PhononConfig forbids unknown keys):
    git worktree add ../qx-pre-fold 1550675d~1   # before exact 3-term fold
    git worktree add ../qx-window  47794944~1    # spectral window present
    (cd ../qx-pre-fold && PYTHONPATH=src python phonon/studies/_bench_sse_stages.py ...)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

STAGES = [
    "PhPh SSE: 1 FFT G->tau",
    "PhPh SSE: 2 tau nnz->stack + fold",
    "PhPh SSE: 3 ring contraction",
    "PhPh SSE: 4 tau stack->nnz",
    "PhPh SSE: 5 IFFT + Hilbert",
]
MALLOC_KNOBS = {
    "default": {},
    # one arena per NUMA-ish group: kills the per-thread arena explosion
    "arena2": {"MALLOC_ARENA_MAX": "2"},
    # never mmap/trim: temp buffers recycle through the main arena
    "nommap": {"MALLOC_MMAP_MAX_": "0", "MALLOC_TRIM_THRESHOLD_": "-1"},
}


def child(args) -> None:
    os.environ["QX_CONFIG"] = args.config  # engine convention
    from quatrex.core.config import parse_config, setup_context

    cfg = parse_config(args.config)
    cfg.scba.max_iterations = args.iters
    cfg.scba.min_iterations = args.iters  # no early convergence exit
    setup_context(cfg)

    from qttools.profiling import Profiler
    from quatrex.core.scba import SCBA

    scba = SCBA(cfg)
    try:
        scba.run()
    except Exception as exc:  # noqa: BLE001 -- report, still dump timings
        print(f"BENCH_RUN_ERROR: {exc!r}", flush=True)

    import numpy as np

    sl = np.asarray(scba.data.sigma_lesser.data)
    checksum = float(np.abs(sl).sum())
    sigma_max = float(np.abs(sl).max())

    # Aggregate the per-stage ranges from the in-process eventlog:
    # entries are (timestamp, depth, label, call_time, after_barrier_time).
    per_stage: dict[str, list[float]] = {s: [] for s in STAGES}
    iters: list[float] = []
    for ev in Profiler().eventlog:
        label, t = ev[2], float(ev[3])
        if label in per_stage:
            per_stage[label].append(t)
        elif label == "SCBA: Iteration":
            iters.append(t)

    def steady(v):  # drop the first (cache/pool warm-up) sample
        v = v[1:] if len(v) > 1 else v
        return sum(v) / len(v) if v else 0.0

    out = {
        "ring": os.environ.get("QUATREX_PHPH_RING_THREADS", "unset"),
        "blas": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "malloc": args.malloc_tag,
        "s_iter": steady(iters),
        "checksum": checksum,
        "sigma_max": sigma_max,
        **{f"s{i+1}": steady(per_stage[s]) for i, s in enumerate(STAGES)},
    }
    print("BENCH_JSON: " + json.dumps(out), flush=True)


def parent(args) -> int:
    rings = [r.strip() for r in args.ring.split(",")]
    blases = [b.strip() for b in args.blas.split(",")]
    mallocs = [m.strip() for m in args.malloc.split(",")]
    rows = []
    for ring in rings:
        for blas in blases:
            for mal in mallocs:
                env = os.environ.copy()
                env["QUATREX_PHPH_RING_THREADS"] = ring
                env.setdefault("QTX_PROFILE_LEVEL", "default")
                for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS"):
                    env.pop(var, None)
                    if blas != "0":  # 0 = leave unset (library default)
                        env[var] = blas
                env.update(MALLOC_KNOBS[mal])
                cmd = [sys.executable, __file__, "--child",
                       "--config", args.config,
                       "--iters", str(args.iters),
                       "--malloc-tag", mal]
                print(f"--- ring={ring} blas={blas} malloc={mal}", flush=True)
                proc = subprocess.run(
                    cmd, env=env, capture_output=True, text=True)
                line = next((ln for ln in proc.stdout.splitlines()[::-1]
                             if ln.startswith("BENCH_JSON: ")), None)
                if line is None:
                    print(proc.stdout[-2000:])
                    print(proc.stderr[-2000:])
                    print("    FAILED")
                    continue
                row = json.loads(line[len("BENCH_JSON: "):])
                rows.append(row)
                print(f"    iter {row['s_iter']:8.2f}s   "
                      f"ring-stage {row['s3']:8.2f}s", flush=True)

    if not rows:
        return 1
    ref = rows[0]["checksum"]
    print(f"\n{'ring':>5} {'blas':>5} {'malloc':>8} {'s/iter':>9} "
          f"{'s1':>7} {'s2':>7} {'s3 ring':>9} {'s4':>7} {'s5':>7}")
    for r in sorted(rows, key=lambda r: r["s_iter"]):
        print(f"{r['ring']:>5} {r['blas']:>5} {r['malloc']:>8} "
              f"{r['s_iter']:9.2f} {r['s1']:7.2f} {r['s2']:7.2f} "
              f"{r['s3']:9.2f} {r['s4']:7.2f} {r['s5']:7.2f}")
    worst = max(abs(r["checksum"] - ref) / max(ref, 1e-300) for r in rows)
    print(f"\nchecksum spread across configs: {worst:.3e} "
          f"({'OK — identical' if worst < 1e-12 else 'MISMATCH!'})")
    return 0 if worst < 1e-12 else 2


def gemm_roofline() -> int:
    """Single-thread zgemm ceiling for the ACTUAL ring shapes (b=36 CNT
    blocks): the honest per-core roofline that the tau pool multiplies.
    Run with OPENBLAS_NUM_THREADS=1."""
    import time

    import numpy as np

    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        print("WARNING: OPENBLAS_NUM_THREADS != 1 -- per-core numbers off")
    rng = np.random.default_rng(0)
    b = 36
    PL = rng.standard_normal((b * b, b)) + 1j * rng.standard_normal((b * b, b))
    PR = rng.standard_normal((b, b * b)) + 1j * rng.standard_normal((b, b * b))
    print(f"{'w':>5} {'T=PL@Ga':>9} {'U=Gb@PR':>9} {'full ring':>10}"
          f"   GF/s (1 thread)")
    for w in (6, 45, 361):
        Ga = rng.standard_normal((w, b, b)) + 1j * rng.standard_normal((w, b, b))
        Gb = Ga.copy()
        gemm_flops = 8 * w * b**4  # one of the three ring GEMMs
        reps = max(3, int(2e9 / (3 * gemm_flops)))
        res = {}
        for name, fn, f in (
            ("T", lambda: PL @ Ga, gemm_flops),
            ("U", lambda: Gb @ PR, gemm_flops),
            ("ring", lambda: (PL @ Ga).reshape(w, b, b * b)
                @ (Gb @ PR).reshape(w, b * b, b), 3 * gemm_flops),
        ):
            fn()  # warm
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            dt = (time.perf_counter() - t0) / reps
            res[name] = f / dt / 1e9
        print(f"{w:>5} {res['T']:9.1f} {res['U']:9.1f} {res['ring']:10.1f}")
    print("\nEPYC 7742 reference: 36 GF/s/core @2.25 GHz base "
          "(~54 boosted); 2.3 TF/s/socket, 4.6 TF/s/node.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gemm-roofline", action="store_true",
                   help="print the single-thread zgemm ceiling for the ring "
                        "shapes and exit")
    p.add_argument("--config", required=False)
    p.add_argument("--iters", type=int, default=3,
                   help="SCBA iterations per child (first one discarded)")
    p.add_argument("--ring", default="1,8,32,64")
    p.add_argument("--blas", default="1,8,0",
                   help="comma list; 0 = leave BLAS env unset")
    p.add_argument("--malloc", default="default",
                   help=f"comma list of {sorted(MALLOC_KNOBS)}")
    p.add_argument("--child", action="store_true")
    p.add_argument("--malloc-tag", default="default")
    args = p.parse_args()
    if args.gemm_roofline:
        return gemm_roofline()
    if not args.config:
        p.error("--config is required (unless --gemm-roofline)")
    if args.child:
        child(args)
        return 0
    return parent(args)


if __name__ == "__main__":
    sys.exit(main())
