"""C++/CUDA fused ring-contraction kernel vs the cuBLAS-batched path.

The SSE three-phonon ring S = (PL @ Ga) @ (Gb @ PR) runs in production
as three strided-batched ZGEMMs (sse_phonon_phonon._contract_tau_q_batched
-> bubble.ring_contract_pre); at b=18 that path sits at the cuBLAS
tile-quantization ceiling (~5.5-6.8 TF/s FP64 on GH200: 32-wide FP64
tiles waste (32/18)^2 of the padded flops per GEMM dim, and the T/U
intermediates round-trip HBM). This study fuses the whole sandwich into
ONE real CUDA C++ kernel (_fused_ring.cu, nvcc -arch=sm_90): one
threadblock per (task, tau), the b*b-long middle contraction streamed
one k1'-panel at a time through shared memory, S accumulated in
registers -- T/U never touch global memory. Python here is ONLY the
loader/bench driver (cupy RawModule on the nvcc-built .cubin); there is
no python-side kernel.

Effective TF/s uses the complex flop model
    flops = 3 GEMMs * b^4 complex MACs * 8 real flops = 24 * C * W * b^4
(GEMM1 (b^2,b)@(b,b), GEMM2 (b,b)@(b,b^2), GEMM3 (b,b^2)@(b^2,b): b^4
complex MACs each), identical for both paths, so the numbers compare
directly with the cuBLAS figures.

Usage:
  python _fused_ring_bench.py build [--arch sm_90] [--no-fast-math]
  python _fused_ring_bench.py check [--quick]     # parity vs cuBLAS
  python _fused_ring_bench.py bench [--json f]    # TF/s sweep

Laptop (sm_86): functional dev + correctness only (FP64 is 1:64 there).
GH200 numbers via phonon/scripts/daint.py, job _fused_ring_job.sh.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import cupy as cp

HERE = Path(__file__).resolve().parent
CU = HERE / "_fused_ring.cu"
CUBIN = HERE / "_fused_ring.cubin"

# kernel variants: b -> [(name, threads)]
VARIANTS = {
    18: [("fused_ring_b18_o1", 324),
         ("fused_ring_b18_o1x2", 324),
         ("fused_ring_b18_o1nr", 324),
         ("fused_ring_b18_o2", 162),
         ("fused_ring_b18_o3", 108),
         # v2: register-tiled stage 3 (round 2, post-ncu)
         ("fused_ring_b18_rt22", 81),
         ("fused_ring_b18_rt22x4", 81),
         ("fused_ring_b18_rt23", 54),
         ("fused_ring_b18_rt33", 36),
         ("fused_ring_b18_v3", 54),
         ("fused_ring_b18_v3x3", 54)],
    36: [("fused_ring_b36_o2", 648),
         ("fused_ring_b36_o4", 324),
         ("fused_ring_b36_rt22", 324),
         ("fused_ring_b36_rt33", 144)],
}


def smem_bytes(b: int) -> int:
    # Gas + Gbs (b*b each) + Tc + Ur (b*(b+1) each), complex128.
    return (2 * b * b + 2 * b * (b + 1)) * 16


def build(arch: str = "sm_90", fast_math: bool = True) -> None:
    cmd = ["nvcc", f"-arch={arch}", "-O3", "-std=c++17", "-cubin",
           "-Xptxas=-v", "-o", str(CUBIN), str(CU)]
    if fast_math:
        cmd.insert(4, "--use_fast_math")
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


class FusedRing:
    """Loader for the nvcc-built cubin (python = launch glue only)."""

    def __init__(self, b: int):
        self.b = b
        self.mod = cp.RawModule(path=str(CUBIN))
        self.smem = smem_bytes(b)
        self.fns = []
        for name, nt in VARIANTS[b]:
            fn = self.mod.get_function(name)
            if self.smem > 48 * 1024:
                fn.max_dynamic_shared_size_bytes = self.smem
            self.fns.append((name, fn, nt))

    def run(self, fn, nt, PL, PR, Ga, Gb, out=None):
        C, W = Ga.shape[0], Ga.shape[1]
        if out is None:
            out = cp.empty((C, W, self.b, self.b), dtype=cp.complex128)
        fn((C * W,), (nt,),
           (PL, PR, Ga, Gb, out, np.int32(C), np.int32(W)),
           shared_mem=self.smem)
        return out


def cublas_ring(PL, PR, Ga, Gb):
    """The production composite: three strided-batched cuBLAS GEMMs,
    exactly _contract_tau_q_batched._ring (T and U round-trip HBM)."""
    C, W, b = Ga.shape[0], Ga.shape[1], Ga.shape[3]
    Tm = PL[:, None] @ Ga                    # (C, W, b*b, b)
    Um = Gb @ PR[:, None]                    # (C, W, b, b*b)
    return (Tm.reshape(C, W, b, b * b)
            @ Um.reshape(C, W, b * b, b))    # (C, W, b, b)


def rand_inputs(b, C, W, seed):
    rng = cp.random.default_rng(seed)

    def cx(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    return (cx(C, b * b, b), cx(C, b, b * b),
            cx(C, W, b, b), cx(C, W, b, b))


def _time_gpu(fn, reps=10, repeats=5):
    times = []
    for _ in range(2):
        fn()
    cp.cuda.Device().synchronize()
    for _ in range(repeats):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end) / 1e3 / reps)
    return sorted(times)[len(times) // 2]


# ----------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------

def _device_banner():
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"device: {props['name'].decode()}  "
          f"CC {cp.cuda.Device().compute_capability}", flush=True)


def cmd_build(args) -> int:
    build(arch=args.arch, fast_math=not args.no_fast_math)
    return 0


def cmd_check(args) -> int:
    _device_banner()
    ok = True
    bs = (18,) if args.quick else (18, 36)
    for b in bs:
        kern = FusedRing(b)
        for (C, W) in ((3, 17), (7, 40)):
            PL, PR, Ga, Gb = rand_inputs(b, C, W, seed=b + C + W)
            ref = cublas_ring(PL, PR, Ga, Gb)
            for name, fn, nt in kern.fns:
                got = kern.run(fn, nt, PL, PR, Ga, Gb)
                err = float(cp.abs(got - ref).max() / cp.abs(ref).max())
                stat = "PASS" if err <= 1e-13 else "FAIL"
                ok = ok and err <= 1e-13
                print(f"b={b:>2} C={C:>3} W={W:>3} {name:<18} {stat} "
                      f"maxrel={err:.2e}", flush=True)
    print("CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# (b, C, W): C*W in {1e3, 1e4, 1e5} with real-run-like tau depths.
# b=36 at C*W=1e5 skipped: the cuBLAS T/U intermediates alone would
# need 2 * 1e5 * 36^3 * 36 * 16 B ~ 150 GB.
CASES = [(18, 5, 200), (18, 25, 400), (18, 125, 800),
         (36, 5, 200), (36, 25, 400)]


def cmd_bench(args) -> int:
    _device_banner()
    print("flop model: 3 GEMMs x b^4 cMACs x 8 flops = 24*C*W*b^4/call")
    rows = []
    print(f"{'b':>3} {'C':>4} {'W':>4} {'C*W':>7} | {'cuBLAS':>7} | "
          f"{'variant':>18} {'fused':>7} {'ratio':>6} {'maxrel':>9}")
    for b, C, W in CASES:
        cp.get_default_memory_pool().free_all_blocks()
        PL, PR, Ga, Gb = rand_inputs(b, C, W, seed=1)
        flops = 24.0 * C * W * b**4
        reps = int(max(2, min(20, 2e11 / flops)))
        t_blas = _time_gpu(lambda: cublas_ring(PL, PR, Ga, Gb), reps=reps)
        blas_tfs = flops / t_blas / 1e12
        ref = cublas_ring(PL, PR, Ga, Gb)

        kern = FusedRing(b)
        out = cp.empty_like(ref)
        best = None
        for name, fn, nt in kern.fns:
            t = _time_gpu(lambda: kern.run(fn, nt, PL, PR, Ga, Gb, out=out),
                          reps=reps)
            err = float(cp.abs(out - ref).max() / cp.abs(ref).max())
            tfs = flops / t / 1e12
            print(f"{b:>3} {C:>4} {W:>4} {C * W:>7} | {blas_tfs:7.2f} | "
                  f"{name:>18} {tfs:7.2f} {tfs / blas_tfs:6.2f} "
                  f"{err:9.2e}", flush=True)
            if best is None or tfs > best["fused_tfs"]:
                best = dict(b=b, C=C, W=W, variant=name,
                            cublas_tfs=blas_tfs, fused_tfs=tfs,
                            ratio=tfs / blas_tfs, maxrel=err)
        rows.append(best)
    print("\nbest per case:")
    for r in rows:
        print(f"  b={r['b']:>2} C*W={r['C'] * r['W']:>7} "
              f"cuBLAS {r['cublas_tfs']:6.2f} TF/s  fused "
              f"{r['fused_tfs']:6.2f} TF/s  ratio {r['ratio']:5.2f} "
              f"({r['variant']}, maxrel {r['maxrel']:.1e})")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("build")
    q.add_argument("--arch", default="sm_90")
    q.add_argument("--no-fast-math", action="store_true")
    q = sub.add_parser("check")
    q.add_argument("--quick", action="store_true")
    q = sub.add_parser("bench")
    q.add_argument("--json", default=None)
    args = p.parse_args()
    return {"build": cmd_build, "check": cmd_check,
            "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
