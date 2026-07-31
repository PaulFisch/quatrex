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

    from qttools.utils.gpu_utils import get_host

    sl = np.asarray(get_host(scba.data.sigma_lesser.data))
    checksum = float(np.abs(sl).sum())
    sigma_max = float(np.abs(sl).max())

    # Stage-3 flop model (stashed by _print_ring_stats on rank 0) -> GF/s.
    gflop = 0.0
    for inter in getattr(scba, "interactions", []):
        sse = getattr(inter, "sigma_phonon_phonon", None)
        if sse is not None:
            gflop = float(getattr(sse, "_ring_model_gflop", 0.0))
            break

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

    from qttools import xp

    out = {
        "backend": xp.__name__,
        "ring": os.environ.get("QUATREX_PHPH_RING_THREADS", "unset"),
        "blas": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "malloc": args.malloc_tag,
        "s_iter": steady(iters),
        "checksum": checksum,
        "sigma_max": sigma_max,
        "gflop": gflop,
        **{f"s{i+1}": steady(per_stage[s]) for i, s in enumerate(STAGES)},
    }
    print("BENCH_JSON: " + json.dumps(out), flush=True)


def parent(args) -> int:
    rings = [r.strip() for r in args.ring.split(",")]
    blases = [b.strip() for b in args.blas.split(",")]
    mallocs = [m.strip() for m in args.malloc.split(",")]
    backends = [b.strip() for b in args.backend.split(",")]
    rows = []
    for backend in backends:
        for ring in rings:
            for blas in blases:
                for mal in mallocs:
                    env = os.environ.copy()
                    env["QTX_ARRAY_MODULE"] = backend
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
                    print(f"--- backend={backend} ring={ring} blas={blas} "
                          f"malloc={mal}", flush=True)
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
                    if row["backend"] != backend:
                        # cupy import failed and qttools fell back silently
                        print(f"    SKIPPED (backend fell back to "
                              f"{row['backend']})")
                        continue
                    rows.append(row)
                    print(f"    iter {row['s_iter']:8.2f}s   "
                          f"ring-stage {row['s3']:8.2f}s", flush=True)

    if not rows:
        return 1
    ref = rows[0]["checksum"]

    def gfs(r):  # achieved stage-3 GF/s from the SSE flop model
        return r["gflop"] / r["s3"] if r.get("gflop") and r["s3"] else 0.0

    print(f"\n{'backend':>7} {'ring':>5} {'blas':>5} {'malloc':>8} "
          f"{'s/iter':>9} {'s1':>7} {'s2':>7} {'s3 ring':>9} "
          f"{'s4':>7} {'s5':>7} {'GF/s':>8}")
    for r in sorted(rows, key=lambda r: r["s_iter"]):
        print(f"{r.get('backend', '?'):>7} {r['ring']:>5} {r['blas']:>5} "
              f"{r['malloc']:>8} {r['s_iter']:9.2f} {r['s1']:7.2f} "
              f"{r['s2']:7.2f} {r['s3']:9.2f} {r['s4']:7.2f} "
              f"{r['s5']:7.2f} {gfs(r):8.1f}")
    # Bit-identity holds only within one backend (cuFFT/cuBLAS reorder);
    # across backends gate on a loose relative agreement instead.
    worst = 0.0
    cross = 0.0
    for r in rows:
        if r.get("backend") == rows[0].get("backend"):
            worst = max(worst, abs(r["checksum"] - ref) / max(ref, 1e-300))
        else:
            cross = max(cross, abs(r["checksum"] - ref) / max(ref, 1e-300))
    print(f"\nchecksum spread within backend: {worst:.3e} "
          f"({'OK — identical' if worst < 1e-12 else 'MISMATCH!'})")
    if cross:
        print(f"checksum spread across backends: {cross:.3e} "
              f"({'OK' if cross < 1e-10 else 'MISMATCH!'})")
    return 0 if worst < 1e-12 and cross < 1e-10 else 2


def gemm_roofline() -> int:
    """Batched-zgemm ceiling for the ACTUAL ring shapes -- the honest
    denominator the stage-3 GF/s is judged against. Backend-agnostic:
    runs on the backend selected by QTX_ARRAY_MODULE (numpy: run with
    OPENBLAS_NUM_THREADS=1 for per-core numbers; the tau pool multiplies
    that ceiling). Block sizes cover the production devices (d5a 15,
    cnt33 folded 36 / unit-cell 63, d11a 135); the tau batch w is capped
    per block size so the T workspace stays ~2 GB."""
    import time

    from qttools import xp
    from qttools.utils.gpu_utils import synchronize_device

    if xp.__name__ == "numpy" and os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        print("WARNING: OPENBLAS_NUM_THREADS != 1 -- per-core numbers off")
    print(f"backend: {xp.__name__}")
    rng = xp.random.default_rng(0)
    print(f"{'b':>5} {'w':>5} {'T=PL@Ga':>9} {'U=Gb@PR':>9} {'full ring':>10}"
          f"   GF/s")
    for b in (15, 36, 63, 135):
        PL = rng.standard_normal((b * b, b)) + 1j * rng.standard_normal(
            (b * b, b))
        PR = rng.standard_normal((b, b * b)) + 1j * rng.standard_normal(
            (b, b * b))
        w_cap = max(8, int(2e9 / (16 * b**3)))  # T buffer w*b^2*b ~2 GB
        for w in (60, 241, 481):
            w = min(w, w_cap)
            Ga = rng.standard_normal((w, b, b)) + 1j * rng.standard_normal(
                (w, b, b))
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
                fn()  # warm (also builds cuBLAS handles / plans)
                synchronize_device()
                t0 = time.perf_counter()
                for _ in range(reps):
                    fn()
                synchronize_device()
                dt = (time.perf_counter() - t0) / reps
                res[name] = f / dt / 1e9
            print(f"{b:>5} {w:>5} {res['T']:9.1f} {res['U']:9.1f} "
                  f"{res['ring']:10.1f}", flush=True)
    print("\nEPYC 7742 reference: 36 GF/s/core @2.25 GHz base "
          "(~54 boosted); 2.3 TF/s/socket, 4.6 TF/s/node. "
          "H100/GH200 FP64 peak 67 TF/s (small-n tiles land well below).")
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
    p.add_argument("--backend", default="numpy",
                   help="comma list of QTX_ARRAY_MODULE values for the "
                        "children (e.g. numpy,cupy); on cupy the ring "
                        "pool is bypassed, so pair with --ring 1")
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
