"""Ring-restructure prototypes vs the cuBLAS composite (kernel round 2).

Two whole-operation rewrites of the fused three-phonon ring
  Tm = PL_c @ Ga_{c,t};  Um = Gb_{c,t} @ PR_c;
  S  = Tm.reshape(B, B*B) @ Um.reshape(B*B, B)
benchmarked with the same protocol as _fused_ring_bench.py (which
this module imports for inputs/reference/timing):

  cutensor   the three stages as einsums with CUPY_ACCELERATORS=
             cutensor (cuTENSOR picks its own kernels; complex128).
  grouped    the production-sharing restructure: tasks share their
             Ga/Gb blocks (in real runs the G dicts hold ~n_pairs*nq
             distinct blocks gathered by ~1e5 tasks), so stages 1/2
             become per-group TALL GEMMs (m = B^2 * C_g, n = k = B --
             ONE padded dimension instead of three) and stage 3 stays
             the well-shaped k = B^2 batched GEMM. The bench models
             the sharing with n_groups distinct G blocks and a
             task -> group map; gather/scatter overhead is included
             in the timing.

Parity on any cupy device (laptop sm_86 ok -- FP64 1:64, timing
meaningless there); timing on a GH200 debug job:
  python _ring_variants_bench.py check
  python _ring_variants_bench.py bench [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os

# cuTENSOR opt-in must be set before cupy import to take effect.
os.environ.setdefault("CUPY_ACCELERATORS", "cutensor,cub")

import cupy as cp  # noqa: E402

from _fused_ring_bench import (  # noqa: E402
    CASES,
    _device_banner,
    _time_gpu,
    cublas_ring,
    rand_inputs,
)


def cutensor_ring(PL, PR, Ga, Gb):
    """Three-stage ring as einsums (cuTENSOR dispatch when active)."""
    C, W, B = Ga.shape[0], Ga.shape[1], Ga.shape[2]
    Tm = cp.einsum("cmk,ctkp->ctmp", PL, Ga)            # (C,W,B*B,B)
    Um = cp.einsum("ctkd,cdp->ctkp", Gb, PR)            # (C,W,B,B*B)
    TmR = Tm.reshape(C, W, B, B * B)
    UmR = Um.reshape(C, W, B * B, B)
    return TmR @ UmR


def make_grouped_inputs(b, C, W, n_groups, seed=1):
    """Production-sharing model: n_groups distinct G blocks, tasks
    mapped round-robin; PL/PR stay per-task."""
    PL, PR, Ga, Gb = rand_inputs(b, C, W, seed=seed)
    n_groups = min(n_groups, C)
    gmap = cp.arange(C) % n_groups
    GaB = Ga[:n_groups].copy()                          # (G,W,B,B)
    GbB = Gb[:n_groups].copy()
    Ga_full = GaB[gmap]                                 # view-expanded
    Gb_full = GbB[gmap]
    return PL, PR, GaB, GbB, gmap, Ga_full, Gb_full


def grouped_ring(PL, PR, GaB, GbB, gmap, b):
    """Stages 1/2 as per-group tall GEMMs, stage 3 as one batched
    k=B^2 GEMM. Gather/scatter included."""
    C = PL.shape[0]
    W = GaB.shape[1]
    B = b
    n_groups = GaB.shape[0]
    order = cp.argsort(gmap)                        # tasks sorted by group
    counts = cp.bincount(gmap, minlength=n_groups)
    Tm = cp.empty((C, W, B * B, B), dtype=PL.dtype)
    Um = cp.empty((C, W, B, B * B), dtype=PL.dtype)
    PLs = PL[order]                                 # (C, B*B, B)
    PRs = PR[order]
    start = 0
    counts_h = counts.get()
    for g in range(n_groups):
        cg = int(counts_h[g])
        if cg == 0:
            continue
        sel = slice(start, start + cg)
        start += cg
        # (W, cg*B*B, B) @ (W, B, B) -> tall GEMM per tau
        plg = PLs[sel].reshape(cg * B * B, B)
        Tm_g = cp.matmul(plg[None], GaB[g])         # (W, cg*B*B, B)
        Tm[order[sel]] = (
            Tm_g.reshape(W, cg, B * B, B).transpose(1, 0, 2, 3))
        # stage 2: (W, B, B) taken as LEFT of (cg) tasks' PR:
        # Um[c,t] = GbB[g,t] @ PR[c] -> batch as (W,B,B) @ (cg*B? ) --
        # regroup: for each task in g: (B,B)@(B,B*B); stack tasks on
        # the N dimension: PR_cat (B, cg*B*B) -> (W, B, cg*B*B)
        prg = PRs[sel].transpose(1, 0, 2).reshape(B, cg * B * B)
        Um_g = cp.matmul(GbB[g], prg[None])         # (W, B, cg*B*B)
        Um[order[sel]] = (
            Um_g.reshape(W, B, cg, B * B).transpose(2, 0, 1, 3))
    TmR = Tm.reshape(C, W, B, B * B)
    UmR = Um.reshape(C, W, B * B, B)
    return TmR @ UmR


def cmd_check(args) -> int:
    ok = True
    for b, C, W in ((18, 5, 20), (18, 12, 30), (36, 4, 10)):
        PL, PR, GaB, GbB, gmap, Ga_f, Gb_f = make_grouped_inputs(
            b, C, W, n_groups=max(2, C // 3))
        ref = cublas_ring(PL, PR, Ga_f, Gb_f)
        for name, out in (
                ("cutensor", cutensor_ring(PL, PR, Ga_f, Gb_f)),
                ("grouped", grouped_ring(PL, PR, GaB, GbB, gmap, b))):
            err = float(cp.abs(out - ref).max() / cp.abs(ref).max())
            stat = "ok " if err <= 1e-13 else "FAIL"
            ok = ok and err <= 1e-13
            print(f"b={b:>2} C={C:>3} W={W:>3} {name:<9} {stat} "
                  f"maxrel={err:.2e}", flush=True)
    print("CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_bench(args) -> int:
    _device_banner()
    print(f"CUPY_ACCELERATORS={os.environ.get('CUPY_ACCELERATORS')}")
    rows = []
    for b, C, W in CASES:
        cp.get_default_memory_pool().free_all_blocks()
        n_groups = max(2, min(45, C))   # ~n_pairs-scale sharing
        PL, PR, GaB, GbB, gmap, Ga_f, Gb_f = make_grouped_inputs(
            b, C, W, n_groups)
        flops = 24.0 * C * W * b**4
        reps = int(max(2, min(20, 2e11 / flops)))
        t_blas = _time_gpu(lambda: cublas_ring(PL, PR, Ga_f, Gb_f),
                           reps=reps)
        ref = cublas_ring(PL, PR, Ga_f, Gb_f)
        for name, fn in (
                ("cutensor", lambda: cutensor_ring(PL, PR, Ga_f, Gb_f)),
                ("grouped",
                 lambda: grouped_ring(PL, PR, GaB, GbB, gmap, b))):
            t = _time_gpu(fn, reps=reps)
            out = fn()
            err = float(cp.abs(out - ref).max() / cp.abs(ref).max())
            row = dict(b=b, C=C, W=W, variant=name,
                       cublas_tfs=flops / t_blas / 1e12,
                       var_tfs=flops / t / 1e12,
                       ratio=t_blas / t, maxrel=err)
            rows.append(row)
            print(f"b={b:>2} C*W={C * W:>7} {name:<9} "
                  f"cuBLAS {row['cublas_tfs']:6.2f}  "
                  f"{name} {row['var_tfs']:6.2f} TF/s  "
                  f"ratio {row['ratio']:5.2f}  maxrel {err:.1e}",
                  flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("check")
    q = sub.add_parser("bench")
    q.add_argument("--json", default=None)
    args = p.parse_args()
    return {"check": cmd_check, "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
