"""Profile SCBA iteration breakdown with 3x3x3 FC3 data.

Instruments anharmonic_transmission_q to report timing for each phase:
OBC setup, ballistic, self-energy kernel, Green's function solve, mixing.

Usage:
    python profile_scba.py                     # auto workers, 2 SCBA iters
    python profile_scba.py --workers 1         # single process
    python profile_scba.py --blas-threads 1    # restrict BLAS to 1 thread
    python profile_scba.py --q-mesh 4 4        # smaller q-mesh for quick test
    python profile_scba.py --max-iter 5        # run more SCBA iterations
"""

import argparse
import os
import sys
import time
import resource
from pathlib import Path

# Parse args before heavy imports so --blas-threads takes effect
parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=None)
parser.add_argument("--blas-threads", type=int, default=None)
parser.add_argument("--q-mesh", type=int, nargs=2, default=[8, 8])
parser.add_argument("--max-iter", type=int, default=2,
                    help="SCBA iterations to run (default 2, enough to profile)")
args = parser.parse_args()

if args.blas_threads is not None:
    for var in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "OMP_NUM_THREADS", "BLAS_NUM_THREADS"]:
        os.environ[var] = str(args.blas_threads)

import numpy as np
import h5py

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs.anharmonic import (
    _se_worker_iq,
    _compute_phph_self_energy_q_dense,
    anharmonic_transmission_q,
    HBAR_SI, THZ_TO_RAD,
)


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    print("=" * 72)
    print("SCBA Performance Profile — 3x3x3 Si")
    print("=" * 72)

    # System info
    print(f"\nSystem:")
    print(f"  CPU count:            {os.cpu_count()}")
    print(f"  Requested workers:    {args.workers or 'auto'}")
    for var in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                "BLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "GOTO_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        print(f"  {var:28s} {os.environ.get(var, '(unset)')}")
    print(f"  q-mesh:               {args.q_mesh[0]}x{args.q_mesh[1]}")
    print(f"  max SCBA iter:        {args.max_iter}")
    print(f"  RSS at start:         {rss_mb():.0f} MB")

    # BLAS diagnostics
    print(f"\nBLAS diagnostics:")
    print(f"  numpy version:        {np.__version__}")
    print(f"  numpy location:       {np.__file__}")

    # np.show_config() — the authoritative source
    try:
        # numpy >= 2.0: show_config(mode='dicts') returns a dict
        cfg = np.show_config(mode='dicts')
        for section in ["Build Dependencies", "Python Dependencies"]:
            if section in cfg:
                for lib, info in cfg[section].items():
                    if any(k in lib.lower() for k in ["blas", "lapack"]):
                        print(f"    {lib}: {info}")
        # Also try to get the raw build info
        if "BLAS" in cfg.get("Build Dependencies", {}):
            print(f"    BLAS detail: {cfg['Build Dependencies']['BLAS']}")
    except (TypeError, AttributeError):
        # older numpy: show_config() prints to stdout
        import contextlib
        from io import StringIO
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            np.show_config()
        config_str = buf.getvalue()
        for line in config_str.splitlines():
            ll = line.lower()
            if any(k in ll for k in ["blas", "lapack", "openblas", "mkl",
                                       "library", "threading"]):
                print(f"    {line.strip()}")

    # Check linked .so files for numpy
    try:
        import subprocess
        np_dir = str(Path(np.__file__).parent)
        ldd_out = subprocess.run(
            ["bash", "-c",
             f"find {np_dir} -name '*.so' | head -5 | xargs ldd 2>/dev/null"
             " | grep -iE 'blas|lapack|mkl|openblas|libgomp|libiomp'"],
            capture_output=True, text=True, timeout=5
        )
        if ldd_out.stdout.strip():
            print(f"\n  Linked BLAS/LAPACK libraries (ldd):")
            for line in ldd_out.stdout.strip().splitlines():
                print(f"    {line.strip()}")
        else:
            print(f"\n  (no BLAS .so found via ldd — may be statically linked)")
    except Exception as e:
        print(f"\n  (ldd check failed: {e})")

    # Runtime thread count test: time a matmul and watch thread count
    print(f"\n  matmul thread test (1000x1000 @ 1000x1000):")
    A = np.random.randn(1000, 1000)
    B = np.random.randn(1000, 1000)
    # Warmup
    _ = A @ B
    # Check /proc/self/status for thread count during matmul
    try:
        import threading
        import subprocess
        thread_counts = []
        stop_flag = threading.Event()
        def poll_threads():
            while not stop_flag.is_set():
                try:
                    with open("/proc/self/status") as f:
                        for line in f:
                            if line.startswith("Threads:"):
                                thread_counts.append(int(line.split()[1]))
                                break
                except Exception:
                    pass
                time.sleep(0.005)
        t = threading.Thread(target=poll_threads, daemon=True)
        t.start()
        t0 = time.perf_counter()
        for _ in range(10):
            _ = A @ B
        dt = time.perf_counter() - t0
        stop_flag.set()
        t.join(timeout=1)
        if thread_counts:
            print(f"    10x matmul:  {dt:.3f}s  ({dt/10*1000:.1f}ms per call)")
            print(f"    Threads observed: min={min(thread_counts)}, "
                  f"max={max(thread_counts)}, "
                  f"samples={len(thread_counts)}")
            if max(thread_counts) <= 2:
                print(f"    WARNING: BLAS appears single-threaded!")
                print(f"    This means numpy is using a non-threaded BLAS.")
                print(f"    Fix: install openblas/mkl or set threads env var.")
        else:
            print(f"    10x matmul:  {dt:.3f}s  (could not poll threads)")
    except Exception as e:
        print(f"    (thread polling failed: {e})")
        t0 = time.perf_counter()
        for _ in range(10):
            _ = A @ B
        dt = time.perf_counter() - t0
        print(f"    10x matmul:  {dt:.3f}s  ({dt/10*1000:.1f}ms per call)")

    # Monkey-patch _compute_phph_self_energy_q_dense to instrument it
    import phonon_inputs.anharmonic as anh_mod
    _orig_se = anh_mod._compute_phph_self_energy_q_dense
    _orig_obc = anh_mod._compute_obc_batch
    _orig_gf = anh_mod._solve_green_batch

    timings = {
        "obc_calls": 0, "obc_time": 0.0,
        "se_calls": 0, "se_time": 0.0,
        "gf_calls": 0, "gf_time": 0.0,
        "iter_starts": [],  # wall-clock timestamps at iteration start
    }

    forced_workers = args.workers

    def patched_se(*a, **kw):
        if forced_workers is not None:
            kw["n_workers"] = forced_workers
        t0 = time.perf_counter()
        result = _orig_se(*a, **kw)
        timings["se_time"] += time.perf_counter() - t0
        timings["se_calls"] += 1
        return result

    def patched_obc(*a, **kw):
        t0 = time.perf_counter()
        result = _orig_obc(*a, **kw)
        timings["obc_time"] += time.perf_counter() - t0
        timings["obc_calls"] += 1
        return result

    def patched_gf(*a, **kw):
        t0 = time.perf_counter()
        result = _orig_gf(*a, **kw)
        timings["gf_time"] += time.perf_counter() - t0
        timings["gf_calls"] += 1
        return result

    anh_mod._compute_phph_self_energy_q_dense = patched_se
    anh_mod._compute_obc_batch = patched_obc
    anh_mod._solve_green_batch = patched_gf

    # Load data
    print(f"\n--- Loading 3x3x3 data ---")
    t0 = time.perf_counter()
    phonon, _ = load_primitive_cell(work_dir, "fc3_prim_vasp")
    fc3_path = str(work_dir / "fc3_prim_vasp" / "fc3.hdf5")
    dt_load = time.perf_counter() - t0
    print(f"  Load time: {dt_load:.2f}s, RSS: {rss_mb():.0f} MB")

    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)
    print(f"  nat_prim={nat_prim}, n_super={n_super}, "
          f"n_dof={3*nat_prim}, dim_sc={3*n_super}")

    # Run transport
    print(f"\n--- Running anharmonic_transmission_q ---")
    t0_total = time.perf_counter()
    result = anharmonic_transmission_q(
        phonon, fc3_path,
        q_mesh_transverse=tuple(args.q_mesh),
        freq_range_thz=(0.1, 18.0, 141),
        transport_direction="x",
        eta_factor=0.5,
        temperature=300.0,
        delta_T=10.0,
        max_scba_iter=args.max_iter,
        scba_tol=0.001,
        mixing=0.1,
        n_slabs=1,
        verbose=True,
    )
    dt_total = time.perf_counter() - t0_total

    # Restore originals
    anh_mod._compute_phph_self_energy_q_dense = _orig_se
    anh_mod._compute_obc_batch = _orig_obc
    anh_mod._solve_green_batch = _orig_gf

    # Report
    n_iters = result["n_scba_iterations"]
    n_kpts = args.q_mesh[0] * args.q_mesh[1]
    dt_other = dt_total - timings["obc_time"] - timings["se_time"] - timings["gf_time"]

    print(f"\n{'='*72}")
    print(f"TIMING BREAKDOWN")
    print(f"{'='*72}")
    print(f"  Total wall time:       {dt_total:8.2f}s")
    print(f"  SCBA iterations:       {n_iters}")
    print(f"  q-points:              {n_kpts}")
    print(f"")
    print(f"  {'Phase':<30s} {'Time':>8s} {'Calls':>6s} {'Per-call':>10s} {'Frac':>7s}")
    print(f"  {'-'*65}")

    phases = [
        ("OBC self-energies", timings["obc_time"], timings["obc_calls"]),
        ("Self-energy kernel", timings["se_time"], timings["se_calls"]),
        ("GF solve", timings["gf_time"], timings["gf_calls"]),
        ("Other (setup, mixing, I/O)", dt_other, 1),
    ]
    for name, dt, calls in phases:
        per_call = dt / calls if calls > 0 else 0
        frac = dt / dt_total * 100
        print(f"  {name:<30s} {dt:8.2f}s {calls:>6d} {per_call:>10.2f}s {frac:>6.1f}%")

    print(f"\n  Per SCBA iteration:")
    if n_iters > 0:
        se_per_iter = timings["se_time"] / n_iters
        gf_per_iter = timings["gf_time"] / n_iters
        print(f"    SE kernel:   {se_per_iter:.2f}s")
        print(f"    GF solve:    {gf_per_iter:.2f}s  "
              f"({timings['gf_calls']/n_iters:.0f} calls/iter)")
        print(f"    Total:       {(se_per_iter + gf_per_iter):.2f}s")
        print(f"")
        print(f"  Estimated time for 20 iterations: "
              f"{(se_per_iter + gf_per_iter)*20:.1f}s "
              f"({(se_per_iter + gf_per_iter)*20/60:.1f} min)")
        # Full plot_quality estimate: dense + 4 methods * ~5 ranks each = ~21 runs
        n_runs = 1 + 5 + 5 + 4 + 4  # dense + SVD + PSCP + SCP3 + FSCP
        est_per_run = (se_per_iter + gf_per_iter) * 20 + timings["obc_time"]
        print(f"  Estimated full plot_quality ({n_runs} runs): "
              f"{est_per_run * n_runs / 60:.0f} min "
              f"({est_per_run * n_runs / 3600:.1f} hr)")

    print(f"\n  Peak RSS: {rss_mb():.0f} MB")
    print(f"  G_anh = {result['thermal_conductance_anharmonic']/1e6:.2f} MW/(m^2 K)")
    print(f"  Conservation = {result['heat_flow_conservation']:.4e}")


if __name__ == "__main__":
    main()
