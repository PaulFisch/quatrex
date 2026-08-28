"""Is the phonon bubble worth moving to a GPU, and does batching enable it?

Measured facts this exists to settle, in order.

``phonon/solver/se_finite.py`` -- the DENSE reference kernel the spatial
studies run on -- issues one ``bubble_dense_from_fft`` call per
``(I, J, iq, kind, K1, K1', K2, K2')`` tuple through a ThreadPoolExecutor:
251384 calls for six SCBA iterations on a 16-cell 4-DOF chain. It contains
zero references to ``xp``; ``sse_phonon_phonon.py``, the production kernel,
contains 122. So the study path is numpy-only and cannot reach a GPU at all,
which is the first thing to know before booking GH200 time.

The per-call arrays are ``(n_fft, d, d)`` with ``d`` between 1 and 6 -- far too
small to saturate anything. On the CPU that costs little: a task-batched
rewrite measured 0.9x, i.e. slightly SLOWER, because numpy is already at its
FLOP/memory bound and the Python overhead per task is a few microseconds
against a few hundred of work. Batching is therefore not a CPU optimisation.
It is the ENABLING step for a GPU, where the same 21000 tiny launches per
iteration would be pure overhead and one batched launch is not.

What this script measures, at the sizes the studies actually use:

* the per-task loop on CPU, which is the incumbent;
* the same contraction task-batched on CPU, to confirm it is a wash there;
* the task-batched contraction on the GPU, including a batched FFT;
* effective GFLOP/s for each, and the agreement with the CPU result.

Transfer is reported separately from compute. In a full port the Green
functions never leave the device, so the compute number is the one that
bounds a ported kernel and the transfer number is what a half-port would pay
every iteration.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_BUBBLE = Path(__file__).resolve().parents[2] / "solver" / "bubble.py"


def _load_kernel():
    """``_bubble_contract_chunk`` without importing the ``solver`` package.

    ``solver/__init__`` re-exports the whole SCBA driver, which pulls in
    phonopy; the daint venv has no phonopy and does not need one to time three
    matmuls. Loading the module by path keeps the kernel a single source of
    truth -- this is the tree's contraction, not a copy of it -- while making
    the benchmark runnable anywhere numpy is.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_bubble_bench", _BUBBLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._bubble_contract_chunk


_contract_chunk = _load_kernel()


def contract_per_task(pl, pr, ga, gb, xp, n_fft):
    """The incumbent: one call per task, exactly as ``se_finite`` issues it."""
    out = xp.empty((pl.shape[0], n_fft, pl.shape[1], pr.shape[1]), dtype=complex)
    for t in range(pl.shape[0]):
        s = _contract_chunk(pl[t], pr[t], ga[t], gb[t])
        out[t] = xp.fft.ifft(s, axis=0)
    return out


def contract_batched(pl, pr, ga, gb, xp, n_fft):
    """The same three matmuls with the task axis carried as a batch.

    Identical index structure to ``_bubble_contract_chunk`` -- the frequency
    axis stays the inner batch and the task axis is prepended, so nothing about
    the contraction changes and the two must agree to roundoff.
    """
    nt, nw = ga.shape[0], ga.shape[1]
    nI, bK1, bK2 = pl.shape[1], ga.shape[2], gb.shape[2]
    bK1p, bK2p, nJ = ga.shape[3], gb.shape[3], pr.shape[1]

    t1 = pl.reshape(nt, 1, nI * bK1, bK2) @ gb
    t1 = t1.reshape(nt, nw, nI, bK1, bK2p)
    t2 = t1.transpose(0, 1, 2, 4, 3).reshape(nt, nw, nI * bK2p, bK1) @ ga
    t2 = t2.reshape(nt, nw, nI, bK2p, bK1p)
    s = (t2.reshape(nt, nw, nI, bK2p * bK1p)
         @ pr.reshape(nt, 1, nJ, bK2p * bK1p).transpose(0, 1, 3, 2))
    return xp.fft.ifft(s, axis=1)


def flops(nt, nw, d):
    """Complex mul-add counted as 8 real flops, three matmuls plus the FFT."""
    mm = 3 * nw * (d ** 4) * 8
    fft = d * d * nw * np.log2(max(nw, 2)) * 5
    return nt * (mm + fft)


def bench(d: int, n_fft: int, n_task: int, reps: int = 3) -> None:
    rng = np.random.default_rng(0)

    def mk(*s):
        return (rng.normal(size=s) + 1j * rng.normal(size=s))

    pl, pr = mk(n_task, d, d, d), mk(n_task, d, d, d)
    ga, gb = mk(n_task, n_fft, d, d), mk(n_task, n_fft, d, d)
    fl = flops(n_task, n_fft, d)
    print(f"\n  d={d}  n_fft={n_fft}  tasks={n_task}   "
          f"({fl / 1e9:.1f} GFLOP, {(ga.nbytes + gb.nbytes) / 2**20:.0f} MiB in)")

    def timed(fn, *a):
        fn(*a)
        best = min(_time(fn, *a) for _ in range(reps))
        return best

    def _time(fn, *a):
        t0 = time.perf_counter()
        fn(*a)
        return time.perf_counter() - t0

    ref = contract_per_task(pl, pr, ga, gb, np, n_fft)
    t_per = timed(contract_per_task, pl, pr, ga, gb, np, n_fft)
    print(f"    CPU per-task : {t_per * 1e3:9.1f} ms   "
          f"{fl / t_per / 1e9:7.2f} GFLOP/s   (the incumbent)")

    got = contract_batched(pl, pr, ga, gb, np, n_fft)
    t_bat = timed(contract_batched, pl, pr, ga, gb, np, n_fft)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    print(f"    CPU batched  : {t_bat * 1e3:9.1f} ms   "
          f"{fl / t_bat / 1e9:7.2f} GFLOP/s   "
          f"{t_per / t_bat:5.2f}x   agree {err:.1e}")

    try:
        import cupy as cp
    except Exception as exc:
        print(f"    GPU          : cupy unavailable ({type(exc).__name__}: {exc})")
        return

    dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    print(f"    GPU device   : {dev['name'].decode()}")

    t0 = time.perf_counter()
    g_pl, g_pr = cp.asarray(pl), cp.asarray(pr)
    g_ga, g_gb = cp.asarray(ga), cp.asarray(gb)
    cp.cuda.Stream.null.synchronize()
    t_xfer = time.perf_counter() - t0

    def run():
        contract_batched(g_pl, g_pr, g_ga, g_gb, cp, n_fft)
        cp.cuda.Stream.null.synchronize()

    run()
    t_gpu = min(_time(run) for _ in range(reps))
    g_out = contract_batched(g_pl, g_pr, g_ga, g_gb, cp, n_fft)
    gerr = float(cp.abs(g_out - cp.asarray(ref)).max()
                 / cp.abs(cp.asarray(ref)).max())
    print(f"    GPU batched  : {t_gpu * 1e3:9.1f} ms   "
          f"{fl / t_gpu / 1e9:7.2f} GFLOP/s   "
          f"{t_per / t_gpu:5.2f}x   agree {gerr:.1e}")
    print(f"    GPU +transfer: {(t_gpu + t_xfer) * 1e3:9.1f} ms   "
          f"{t_per / (t_gpu + t_xfer):5.2f}x   "
          f"(transfer {t_xfer * 1e3:.1f} ms -- a full port pays this once, "
          f"a half-port every iteration)")

    def run_per():
        contract_per_task(g_pl, g_pr, g_ga, g_gb, cp, n_fft)
        cp.cuda.Stream.null.synchronize()

    run_per()
    t_gper = min(_time(run_per) for _ in range(1))
    print(f"    GPU per-task : {t_gper * 1e3:9.1f} ms   "
          f"{fl / t_gper / 1e9:7.2f} GFLOP/s   {t_per / t_gper:5.2f}x   "
          f"<-- porting WITHOUT batching")


def thread_scan(dofs, n_fft: int, n_task: int, pools) -> None:
    """How the per-task thread pool scales, which is what ``se_finite`` does.

    Measured because the answer is not the core count. On a 16-core laptop the
    optimum is 1 thread at d=1 and 2-4 at d=4-6, and a 16-thread pool runs at
    HALF the speed of serial at d=4 -- the tasks are small enough that pool
    and GIL traffic outweigh the parallelism. A 256-core node has more memory
    bandwidth to give, so it has to be measured there rather than inferred.
    """
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(0)
    print(f"\n  per-task loop through a thread pool, {n_task} tasks, "
          f"n_fft={n_fft} -- the structure se_finite issues")
    for d in dofs:
        def mk(*s):
            return rng.normal(size=s) + 1j * rng.normal(size=s)

        pl, pr = mk(n_task, d, d, d), mk(n_task, d, d, d)
        ga, gb = mk(n_task, n_fft, d, d), mk(n_task, n_fft, d, d)

        def one(t):
            return np.fft.ifft(
                _contract_chunk(pl[t], pr[t], ga[t], gb[t]), axis=0)

        row = []
        for nthr in pools:
            if nthr == 1:
                t0 = time.perf_counter()
                [one(t) for t in range(n_task)]
            else:
                with ThreadPoolExecutor(nthr) as ex:
                    t0 = time.perf_counter()
                    list(ex.map(one, range(n_task)))
            row.append(time.perf_counter() - t0)
        base = row[0]
        best = min(range(len(row)), key=lambda i: row[i])
        print(f"    d={d}: " + "  ".join(
            f"{n}thr {t * 1e3:7.1f}ms ({base / t:4.2f}x)"
            for n, t in zip(pools, row))
            + f"   BEST {pools[best]} threads")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dof", type=int, nargs="+", default=[1, 4, 6])
    ap.add_argument("--n-fft", type=int, default=481)
    ap.add_argument("--tasks", type=int, default=512)
    ap.add_argument("--thread-scan", action="store_true",
                    help="scan the per-task pool size instead of benchmarking "
                         "the batched kernel")
    ap.add_argument("--pools", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32])
    a = ap.parse_args(argv)
    print(__doc__.split("\n\n")[0])
    if a.thread_scan:
        thread_scan(a.dof, a.n_fft, a.tasks, a.pools)
        return 0
    for d in a.dof:
        bench(d, a.n_fft, a.tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
