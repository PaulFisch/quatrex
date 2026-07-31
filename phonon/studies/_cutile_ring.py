"""cuTile fused ring-contraction experiment vs the cuBLAS-batched path.

The SSE ring S[tau] = (PL @ Ga[tau]) @ (Gb[tau] @ PR) runs at cuBLAS's
batched-ZGEMM ceiling for its block sizes (11.4-13.5 TF/s at b=36 on
GH200: 32/64-wide FP64 tensor-core tiles quantize badly against b=36,
and T/U round-trip HBM). This study reimplements the ring as ONE fused
cuda.tile kernel: tile sizes matched to b, T/U kept on-chip, complex128
as an in-kernel 4M real split (no complex dtypes in Tile IR; on-chip
split has none of the HBM penalty that makes host-side 4M 2x slower).
Effective TF/s uses the complex flop model 24*w*b^4*nb, directly
comparable with the cuBLAS numbers in phonon/docs/gpu_campaign_2026-07.md.

Usage:
  python phonon/studies/_cutile_ring.py smoke            # env gate
  python phonon/studies/_cutile_ring.py check [--fold]   # vs production ring
  python phonon/studies/_cutile_ring.py bench [--json f] # TF/s sweep

Laptop (sm_86): functional dev + correctness (FP64 is 1:64 there).
GH200 numbers via phonon/scripts/daint.py jobs cutile-{gate,check,bench}
(needs nvidia-cuda-tileiras >= 13.3 for sm_90).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import cupy as cp

# tileiras is not self-contained: it resolves its nvvm/ptxas backend
# via PATH / CUDA_HOME / CUDA_PATH / **CUDA_ROOT**. A system CUDA
# toolkit (13.1 here) reachable through any of those shadows the
# wheel's matched 13.3 components and yields the generic "failed to
# compile Tile IR program" for every mma / lineinfo compile (same root
# cause as NVIDIA/cutile-python#72, which was a bad CUDA_HOME). The
# library sanitizes CUDA_HOME/CUDA_PATH in the compiler subprocess but
# NOT CUDA_ROOT -- bisected here 2026-07-31. Scrub and prepend the
# wheel toolkit BEFORE cuda.tile captures the environment.
import os
from pathlib import Path

import nvidia

_cu13 = Path(next(iter(nvidia.__path__))) / "cu13"
for _v in ("CUDA_ROOT", "CUDA_HOME", "CUDA_PATH"):
    os.environ.pop(_v, None)
os.environ["PATH"] = f"{_cu13 / 'bin'}:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = (
    f"{_cu13 / 'nvvm' / 'lib64'}:{os.environ.get('LD_LIBRARY_PATH', '')}")

import cuda.tile as ct  # noqa: E402

F64 = ct.float64
ZERO = ct.PaddingMode.ZERO


# ----------------------------------------------------------------------
# kernel: fused (PL @ Ga) @ (Gb @ PR), 4M complex split
# ----------------------------------------------------------------------
# Layouts (host-prepared, all float64, C-contiguous):
#   PL3[e, a, c] = phi_left[a, c, e]   (B, B, B)
#   PR3[d, p, j] = phi_right[j, d, p]  (B, B, B)   (p = b')
#   G*[nb, w, B, B], S*[nb, w, B, B]
# Math (per ib, tau):  S[a, j] = sum_{e,p} T[e,a,p] * U[e,p,j]
#   T[e,a,p] = sum_c PL3[e,a,c] * Ga[c,p]
#   U[e,p,j] = sum_d Gb[e,d]    * PR3[d,p,j]
# Grid: (cdiv(B,T)^2, w, nb); per CTA: loop e-chunks (E) x p-tiles (T),
# accumulate acc3[E, T, T] over both loops, reduce over E at the end.


@ct.kernel
def ring4m(PLr, PLi, PRr, PRi, Gar, Gai, Gbr, Gbi, Sr, Si,
           B: ct.Constant[int], T: ct.Constant[int], E: ct.Constant[int]):
    ntile = ct.cdiv(B, T)
    ia = ct.bid(0) // ntile
    ij = ct.bid(0) % ntile
    tau = ct.bid(1)
    ib = ct.bid(2)

    acc3r = ct.zeros((E, T, T), F64)
    acc3i = ct.zeros((E, T, T), F64)
    for ie in range(ct.cdiv(B, E)):
        for jp in range(ntile):
            # ---- gemm1: T3[E, T(a), T(p)] over c ----
            t3r = ct.zeros((E, T, T), F64)
            t3i = ct.zeros((E, T, T), F64)
            for kc in range(ntile):
                plr = ct.load(PLr, index=(ie, ia, kc), shape=(E, T, T),
                              padding_mode=ZERO)
                pli = ct.load(PLi, index=(ie, ia, kc), shape=(E, T, T),
                              padding_mode=ZERO)
                gar = ct.reshape(
                    ct.load(Gar, index=(ib, tau, kc, jp), shape=(1, 1, T, T),
                            padding_mode=ZERO), (T, T))
                gai = ct.reshape(
                    ct.load(Gai, index=(ib, tau, kc, jp), shape=(1, 1, T, T),
                            padding_mode=ZERO), (T, T))
                t3r = ct.mma(plr, gar, t3r)
                t3r = ct.mma(ct.negative(pli), gai, t3r)
                t3i = ct.mma(plr, gai, t3i)
                t3i = ct.mma(pli, gar, t3i)
            # ---- gemm2: U3[E, T(p), T(j)] over d ----
            u2r = ct.zeros((E, T * T), F64)
            u2i = ct.zeros((E, T * T), F64)
            for kd in range(ntile):
                gbr = ct.reshape(
                    ct.load(Gbr, index=(ib, tau, ie, kd), shape=(1, 1, E, T),
                            padding_mode=ZERO), (E, T))
                gbi = ct.reshape(
                    ct.load(Gbi, index=(ib, tau, ie, kd), shape=(1, 1, E, T),
                            padding_mode=ZERO), (E, T))
                prr = ct.reshape(
                    ct.load(PRr, index=(kd, jp, ij), shape=(T, T, T),
                            padding_mode=ZERO), (T, T * T))
                pri = ct.reshape(
                    ct.load(PRi, index=(kd, jp, ij), shape=(T, T, T),
                            padding_mode=ZERO), (T, T * T))
                u2r = ct.mma(gbr, prr, u2r)
                u2r = ct.mma(ct.negative(gbi), pri, u2r)
                u2i = ct.mma(gbr, pri, u2i)
                u2i = ct.mma(gbi, prr, u2i)
            u3r = ct.reshape(u2r, (E, T, T))
            u3i = ct.reshape(u2i, (E, T, T))
            # ---- gemm3 (batched over E), accumulated over (ie, jp) ----
            acc3r = ct.mma(t3r, u3r, acc3r)
            acc3r = ct.mma(ct.negative(t3i), u3i, acc3r)
            acc3i = ct.mma(t3r, u3i, acc3i)
            acc3i = ct.mma(t3i, u3r, acc3i)

    sr = ct.sum(acc3r, axis=0)
    si = ct.sum(acc3i, axis=0)
    ct.store(Sr, index=(ib, tau, ia, ij), tile=ct.reshape(sr, (1, 1, T, T)))
    ct.store(Si, index=(ib, tau, ia, ij), tile=ct.reshape(si, (1, 1, T, T)))


# ----------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------

def prep_phis(phi_left, phi_right):
    """3D permutations with the e/d index outermost, split re/im."""
    pl3 = cp.ascontiguousarray(phi_left.transpose(2, 0, 1))
    pr3 = cp.ascontiguousarray(phi_right.transpose(1, 2, 0))
    return (cp.ascontiguousarray(pl3.real), cp.ascontiguousarray(pl3.imag),
            cp.ascontiguousarray(pr3.real), cp.ascontiguousarray(pr3.imag))


def split_g(g):
    return cp.ascontiguousarray(g.real), cp.ascontiguousarray(g.imag)


def run_ring4m(phis, ga, gb, b, tile, echunk, out=None):
    """Launch the fused kernel; ga/gb: (nb, w, b, b) complex128."""
    nb, w = ga.shape[0], ga.shape[1]
    gar, gai = split_g(ga)
    gbr, gbi = split_g(gb)
    if out is None:
        sr = cp.empty((nb, w, b, b), dtype=cp.float64)
        si = cp.empty((nb, w, b, b), dtype=cp.float64)
    else:
        sr, si = out
    ntile = math.ceil(b / tile)
    grid = (ntile * ntile, w, nb)
    ct.launch(cp.cuda.get_current_stream(), grid, ring4m,
              (*phis, gar, gai, gbr, gbi, sr, si, b, tile, echunk))
    return sr, si


def reference_ring(phi_left, phi_right, ga, gb, b):
    """The production path: phi_perms + ring_contract_pre (cuBLAS)."""
    from quatrex.phonon.bubble import phi_perms, ring_contract_pre
    PL, PR, nI, bK2, nJ = phi_perms(phi_left, phi_right, cp)
    nb = ga.shape[0]
    return cp.stack([
        ring_contract_pre(PL, PR, nI, bK2, nJ, ga[i], gb[i], cp)
        for i in range(nb)
    ])


def rand_inputs(b, w, nb, seed):
    rng = cp.random.default_rng(seed)

    def cx(*shape):
        return (rng.standard_normal(shape) +
                1j * rng.standard_normal(shape))

    return cx(b, b, b), cx(b, b, b), cx(nb, w, b, b), cx(nb, w, b, b)


def tile_configs(b):
    """Legal (tile, echunk) sweep for one block size."""
    cfgs = []
    for tile in (8, 16, 32):
        if tile > b:
            continue
        for echunk in (2, 4, 8):
            # on-chip f64 working set (re+im), bytes; keep under ~200 KB
            fp = 2 * 8 * (echunk * tile * tile * 3      # t3, acc3, pl
                          + echunk * tile * tile        # u3/u2
                          + tile * tile * tile          # pr tile
                          + 2 * tile * tile)            # ga + gb-ish
            if fp <= 220 * 1024:
                cfgs.append((tile, echunk))
    return cfgs


# ----------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------

def cmd_smoke(_args) -> int:
    import importlib.metadata as md
    ok = True
    for p in ("cuda-tile", "nvidia-cuda-tileiras"):
        try:
            print(f"{p}: {md.version(p)}")
        except Exception as exc:  # noqa: BLE001
            print(f"{p}: MISSING ({exc})")
            ok = False
    dev = cp.cuda.Device()
    cc = dev.compute_capability
    drv = cp.cuda.runtime.driverGetVersion()
    print(f"device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}"
          f"  CC {cc}  driver {drv}")
    if drv < 13000:
        print("WARNING: driver < r580-era; Tile IR may not run")

    # 1: vector add
    @ct.kernel
    def vadd(a, b_, c, TS: ct.Constant[int]):
        i = ct.bid(0)
        at = ct.load(a, index=(i,), shape=(TS,), padding_mode=ZERO)
        bt = ct.load(b_, index=(i,), shape=(TS,), padding_mode=ZERO)
        ct.store(c, index=(i,), tile=at + bt)

    n = 1000
    a = cp.random.standard_normal(n)
    bb = cp.random.standard_normal(n)
    c = cp.empty_like(a)
    ct.launch(cp.cuda.get_current_stream(), (math.ceil(n / 128),), vadd,
              (a, bb, c, 128))
    cp.cuda.Device().synchronize()
    v = bool(cp.allclose(c, a + bb))
    print(f"vector-add: {'PASS' if v else 'FAIL'}")
    ok = ok and v

    # 2: f64 mma on a non-divisible array
    @ct.kernel
    def mm(a, b_, c, T: ct.Constant[int]):
        i = ct.bid(0)
        j = ct.bid(1)
        acc = ct.zeros((T, T), F64)
        for k in range(ct.cdiv(70, T)):
            at = ct.load(a, index=(i, k), shape=(T, T), padding_mode=ZERO)
            bt = ct.load(b_, index=(k, j), shape=(T, T), padding_mode=ZERO)
            acc = ct.mma(at, bt, acc)
        ct.store(c, index=(i, j), tile=acc)

    m = 70
    A = cp.random.standard_normal((m, m))
    B = cp.random.standard_normal((m, m))
    C = cp.empty((m, m))
    try:
        ct.launch(cp.cuda.get_current_stream(), (math.ceil(m / 32),) * 2, mm,
                  (A, B, C, 32))
        cp.cuda.Device().synchronize()
        err = float(cp.abs(C - A @ B).max() / cp.abs(A @ B).max())
        v = err < 1e-13
        print(f"f64 mma (70x70, zero-pad): {'PASS' if v else 'FAIL'} "
              f"maxrel={err:.2e}")
        ok = ok and v
    except Exception as exc:  # noqa: BLE001
        # Known blocker (2026-07-31): tileiras 13.3.36 -- the only wheel
        # with sm_86/sm_90 support -- rejects EVERY ct.mma bytecode
        # ("failed to compile Tile IR program", rc 5) for every
        # --gpu-name incl. sm_80/sm_90/sm_100, with cuda-tile 1.5.0 AND
        # 1.6.0rc3; 3D broadcast+reduce fails likewise. Same signature
        # as NVIDIA/cutile-python#72 (closed, unresolved). Non-mma
        # kernels compile and run fine. Re-run this smoke when a newer
        # tileiras ships; the ring kernel below is complete and waiting.
        print(f"f64 mma: BLOCKED ({type(exc).__name__}) -- see docstring "
              "note; experiment gated on a fixed tileiras release")
        ok = False
    print("SMOKE:", "PASS" if ok else "FAIL/BLOCKED")
    return 0 if ok else 1


def cmd_check(args) -> int:
    ok = True
    # --quick: b=36/w=60 only -- the laptop A1000 runs FP64 at 1:64, so
    # the full cross (esp. b=135) belongs on the GH200.
    bs = (36,) if args.quick else (36, 63, 135)
    ws = (60,) if args.quick else (60, 241)
    for b in bs:
        for w in ws:
            for nb in (1, 3):
                phi_l, phi_r, ga, gb = rand_inputs(b, w, nb, seed=b + w + nb)
                ref = reference_ring(phi_l, phi_r, ga, gb, b)
                phis = prep_phis(phi_l, phi_r)
                for (tile, echunk) in tile_configs(b):
                    sr, si = run_ring4m(phis, ga, gb, b, tile, echunk)
                    got = sr + 1j * si
                    err = float(cp.abs(got - ref).max() /
                                cp.abs(ref).max())
                    stat = "PASS" if err <= 1e-12 else "FAIL"
                    ok = ok and err <= 1e-12
                    print(f"b={b:>3} w={w:>3} nb={nb} T={tile:>2} E={echunk} "
                          f"{stat} maxrel={err:.2e}", flush=True)
    print("CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _time_gpu(fn, reps=10, repeats=5):
    times = []
    for _ in range(3):
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


def cmd_bench(args) -> int:
    rows = []
    print(f"{'b':>4} {'w':>4} {'nb':>3} | {'cuBLAS':>8} {'cuTile':>8} "
          f"{'ratio':>6} | {'cfg':>7} {'maxrel':>9}")
    for b in (36, 63, 135):
        for w in (60, 241, 481):
            for nb in (1, 16):
                if nb * w * b * b * 16 * 4 > 30e9:  # input-size guard
                    continue
                phi_l, phi_r, ga, gb = rand_inputs(b, w, nb, seed=1)
                flops = 24.0 * w * b**4 * nb

                from quatrex.phonon.bubble import (phi_perms,
                                                   ring_contract_pre)
                PL, PR, nI, bK2, nJ = phi_perms(phi_l, phi_r, cp)

                def blas():
                    for i in range(nb):
                        ring_contract_pre(PL, PR, nI, bK2, nJ,
                                          ga[i], gb[i], cp)

                t_blas = _time_gpu(blas)
                ref = reference_ring(phi_l, phi_r, ga, gb, b)

                phis = prep_phis(phi_l, phi_r)
                gar, gai = split_g(ga)
                gbr, gbi = split_g(gb)
                sr = cp.empty((nb, w, b, b), dtype=cp.float64)
                si = cp.empty((nb, w, b, b), dtype=cp.float64)
                best = None
                for (tile, echunk) in tile_configs(b):
                    ntile = math.ceil(b / tile)
                    grid = (ntile * ntile, w, nb)

                    def tile_fn():
                        ct.launch(cp.cuda.get_current_stream(), grid,
                                  ring4m, (*phis, gar, gai, gbr, gbi,
                                           sr, si, b, tile, echunk))

                    t = _time_gpu(tile_fn)
                    if best is None or t < best[0]:
                        err = float(cp.abs((sr + 1j * si) - ref).max()
                                    / cp.abs(ref).max())
                        best = (t, tile, echunk, err)
                t_tile, tile, echunk, err = best
                row = dict(b=b, w=w, nb=nb,
                           cublas_tfs=flops / t_blas / 1e12,
                           cutile_tfs=flops / t_tile / 1e12,
                           cfg=f"T{tile}/E{echunk}", maxrel=err)
                rows.append(row)
                print(f"{b:>4} {w:>4} {nb:>3} | "
                      f"{row['cublas_tfs']:8.1f} {row['cutile_tfs']:8.1f} "
                      f"{row['cutile_tfs'] / row['cublas_tfs']:6.2f} | "
                      f"{row['cfg']:>7} {err:9.2e}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    q = sub.add_parser("check")
    q.add_argument("--quick", action="store_true")
    q = sub.add_parser("bench")
    q.add_argument("--json", default=None)
    args = p.parse_args()
    return {"smoke": cmd_smoke, "check": cmd_check,
            "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
