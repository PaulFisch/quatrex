"""Micro-benchmark: what the pole solve costs, and whether it still loops.

``phonon/docs/pole_solve_batching.md`` Sec. 0.
Usage::
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/quatrex/phonon"))

from quatrex.core.config import PoleSectorConfig                # noqa: E402
from quatrex.phonon.experimental.pole.pole_sector import PoleSector               # noqa: E402
from test_pole_sector import FMAX, DAMP, W_C, _sparse_indices   # noqa: E402


def _bed(nf: int, sizes: tuple[int, ...], seed: int = 0):
    """A device whose modes ALL sit inside the pole window.

    Not ``test_pole_sector._bed``: its diagonal is a geometric ladder
    ``20 * 1.55**k``, which crosses the top of the grid after about nine modes,
    so the candidate count sticks at nine however large the device is -- and
    the candidate count is exactly what this benchmark has to vary. Here the
    harmonic frequencies are spread linearly across the window, so the batch is
    the number of degrees of freedom.
    """
    rng = np.random.default_rng(seed)
    total = int(sum(sizes))
    omega = np.linspace(0.15 * FMAX, 0.92 * FMAX, total)
    d_ii, d_ij, k = [], [], 0
    for n in sizes:
        m = 0.3 * rng.normal(size=(n, n))
        d_ii.append(m + m.T + np.diag(omega[k:k + n] ** 2))
        k += n
    for i in range(len(sizes) - 1):
        d_ij.append(0.3 * rng.normal(size=(sizes[i], sizes[i + 1])))

    freqs = np.linspace(0.0, FMAX, nf)
    a = DAMP * freqs * np.exp(-((freqs / W_C) ** 2))       # Gamma = a >= 0
    delta = np.einsum("w,ij->wij", -1j * a, np.eye(total))
    return freqs, (d_ii, d_ij, [b.T for b in d_ij]), delta


def context(nf: int, sizes: tuple[int, ...]) -> PoleSector:
    """Drive the sector the way ``PhononSolver._update_pole_sector`` does."""
    freqs, (d_ii, d_ij, d_ji), delta = _bed(nf, sizes)
    sizes = list(sizes)
    d_blocks = {}
    for i in range(len(sizes)):
        d_blocks[(i, i)] = d_ii[i] + 0j
        if i + 1 < len(sizes):
            d_blocks[(i, i + 1)] = d_ij[i] + 0j
            d_blocks[(i + 1, i)] = d_ji[i] + 0j
    rows, cols = _sparse_indices(np.array(sizes))
    sec = PoleSector(PoleSectorConfig(enabled=True), freqs)
    sec.set_operator_context(
        delta=delta[:, rows, cols], d_blocks=d_blocks, obc_left=None,
        obc_right=None, block_sizes=np.array(sizes), rows=rows, cols=cols,
    )
    return sec


def measure(nf: int, sizes: tuple[int, ...], repeats: int) -> dict:
    sec = context(nf, sizes)
    sec.refresh()                                   # warm caches and imports
    best = float("inf")
    for _ in range(repeats):
        s = context(nf, sizes)
        t0 = time.perf_counter()
        state = s.refresh()
        best = min(best, time.perf_counter() - t0)
    s = context(nf, sizes)
    pr = cProfile.Profile()
    pr.enable()
    state = s.refresh()
    pr.disable()
    calls = pstats.Stats(pr).total_calls
    n = state.n_poles + len(state.rejected)
    return {"n": n, "time": best, "calls": calls,
            "per_candidate": calls / max(n, 1), "state": state}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nf", type=int, default=401)
    p.add_argument("--sizes", default="3,3,3",
                   help="block sizes of the device, comma separated")
    p.add_argument("--sweep", default="3,3,3|6,6,6|9,9,9|12,12,12",
                   help="'|'-separated block-size sets for the scaling table")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args()

    sizes = tuple(int(x) for x in args.sizes.split(","))
    r = measure(args.nf, sizes, args.repeats)
    print(f"one refresh, nf={args.nf}, blocks={sizes} ({sum(sizes)} dof)")
    print(f"  {r['n']} candidates   {r['time'] * 1e3:.2f} ms   "
          f"{r['calls']} Python calls   {r['per_candidate']:.0f} per candidate")

    print("\nhot frames")
    s = context(args.nf, sizes)
    pr = cProfile.Profile()
    pr.enable()
    s.refresh()
    pr.disable()
    st = pstats.Stats(pr)
    rows = sorted(st.stats.items(), key=lambda kv: -kv[1][3])[:args.top]
    for (fn, line, name), (cc, nc, tt, ct, _) in rows:
        where = f"{Path(fn).name}:{line}({name})"
        print(f"  {ct * 1e3:8.2f} ms  {nc:6d} calls  {where}")

    print("\nPython calls per candidate vs batch size")
    print("  dof  cand    ms   calls  per cand")
    for spec in args.sweep.split("|"):
        sz = tuple(int(x) for x in spec.split(","))
        m = measure(args.nf, sz, max(1, args.repeats // 2))
        print(f"  {sum(sz):3d}  {m['n']:4d}  {m['time'] * 1e3:6.1f}  "
              f"{m['calls']:6d}  {m['per_candidate']:8.0f}")
    print("\n(a batched solve does constant Python work per Newton step, so the\n"
          " last column must FALL as the batch grows; flat means it still loops)")


if __name__ == "__main__":
    main()
