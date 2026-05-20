#!/usr/bin/env python
"""Profile variants of the 3-phonon bubble contraction.

The bubble is the dominant cost in the dense reference SCBA (cProfile
on d5a showed 96% of wall time in the three einsums of
:func:`phonon.solver.bubble.bubble_dense`). This script times the
contraction at d5a-like sizes for several implementations:

  * **v1** — current three np.einsum calls without ``optimize``
    (legacy behaviour, slow because each operand-pair is walked in
    Python).
  * **v2** — same three np.einsum calls with ``optimize="optimal"``.
  * **v3** — single fused ``opt_einsum.contract`` over all four
    operands, ``optimize="optimal"`` (numpy backend).
  * **v4** — single fused ``opt_einsum.contract_expression`` with a
    pre-computed path (the expression is built once and reused, so
    the path-finding cost is paid only at import).

Each variant is timed with a small warm-up + best-of-N reporting.
Use ``--n-dof / --ne / --runs`` to mimic a different system size or
average over more repeats.

Cluster invocation::

    /home/paul/miniconda3/envs/quatrex-dev/bin/python \\
        phonon/scripts/profile_bubble.py --n-dof 63 --ne 41 --runs 5
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np


def _rng_block(shape, rng):
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )


def _build_inputs(n_dof: int, ne: int, seed: int = 0):
    """Allocate the four operands at the FFT'd convention used by the bubble.

    Returns ``(phi_left, phi_right, Ga_fft, Gb_fft)`` matching the
    indices ``("ace", "Jdb", "wcb", "wed")`` of the contraction.
    """
    rng = np.random.default_rng(seed)
    n_fft = 2 * ne - 1
    phi_left = _rng_block((n_dof, n_dof, n_dof), rng)
    phi_right = _rng_block((n_dof, n_dof, n_dof), rng)
    G = _rng_block((ne, n_dof, n_dof), rng)
    Ga = np.zeros((n_fft, n_dof, n_dof), dtype=complex)
    Ga[:ne] = G
    Gb = Ga.copy()
    Ga_fft = np.fft.fft(Ga, axis=0)
    Gb_fft = np.fft.fft(Gb, axis=0)
    return phi_left, phi_right, Ga_fft, Gb_fft


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


def v1_three_einsum_no_optimize(phi_left, phi_right, Ga_fft, Gb_fft):
    A = np.einsum("ace,wed->wacd", phi_left, Gb_fft)
    B = np.einsum("wacd,wcb->wabd", A, Ga_fft)
    return np.einsum("wabd,Jdb->waJ", B, phi_right)


def v2_three_einsum_optimal(phi_left, phi_right, Ga_fft, Gb_fft):
    A = np.einsum("ace,wed->wacd", phi_left, Gb_fft, optimize="optimal")
    B = np.einsum("wacd,wcb->wabd", A, Ga_fft, optimize="optimal")
    return np.einsum("wabd,Jdb->waJ", B, phi_right, optimize="optimal")


def v3_fused_opt_einsum(phi_left, phi_right, Ga_fft, Gb_fft):
    import opt_einsum
    return opt_einsum.contract(
        "ace,Jdb,wcb,wed->waJ",
        phi_left, phi_right, Ga_fft, Gb_fft,
        optimize="optimal",
    )


def _build_v4_expression(n_dof: int, n_fft: int):
    """Pre-compute the contraction path; returns a callable expression."""
    import opt_einsum
    shape_phi = (n_dof, n_dof, n_dof)
    shape_G = (n_fft, n_dof, n_dof)
    return opt_einsum.contract_expression(
        "ace,Jdb,wcb,wed->waJ",
        shape_phi, shape_phi, shape_G, shape_G,
        optimize="optimal",
    )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _time(label: str, fn: Callable, *args, runs: int = 3) -> float:
    fn(*args)  # warm-up: also surfaces import or path-finding cost
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    best = min(ts)
    median = sorted(ts)[len(ts) // 2]
    print(
        f"  {label:42s} best={best*1000:8.2f} ms  "
        f"median={median*1000:8.2f} ms"
    )
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-dof", type=int, default=63,
                        help="primitive-cell DOF (default %(default)s — d5a)")
    parser.add_argument("--ne", type=int, default=21,
                        help="positive frequency samples (default %(default)s)")
    parser.add_argument("--runs", type=int, default=3,
                        help="best-of-N timing (default %(default)s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-match", action="store_true",
                        help="assert every variant matches v1 within 1e-6")
    args = parser.parse_args()

    n_fft = 2 * args.ne - 1
    inputs = _build_inputs(args.n_dof, args.ne, args.seed)
    phi_left, phi_right, Ga_fft, Gb_fft = inputs

    print(
        f"Bubble profile: n_dof={args.n_dof}, ne={args.ne}, "
        f"n_fft={n_fft}, runs={args.runs}",
        flush=True,
    )
    print(f"  phi shape = {phi_left.shape}, G_fft shape = {Ga_fft.shape}")
    print()

    t1 = _time("v1 three-einsum, no optimize", v1_three_einsum_no_optimize,
               *inputs, runs=args.runs)
    t2 = _time("v2 three-einsum, optimize='optimal'",
               v2_three_einsum_optimal, *inputs, runs=args.runs)
    t3 = _time("v3 opt_einsum.contract (fused)", v3_fused_opt_einsum,
               *inputs, runs=args.runs)

    expr = _build_v4_expression(args.n_dof, n_fft)

    def v4_precomputed(phi_left, phi_right, Ga_fft, Gb_fft):
        return expr(phi_left, phi_right, Ga_fft, Gb_fft)

    t4 = _time("v4 opt_einsum precomputed expression", v4_precomputed,
               *inputs, runs=args.runs)

    print()
    print(
        f"Speedup vs v1:"
        f"  v2={t1/t2:6.2f}x"
        f"  v3={t1/t3:6.2f}x"
        f"  v4={t1/t4:6.2f}x"
    )

    if args.check_match:
        r1 = v1_three_einsum_no_optimize(*inputs)
        for label, fn in (
            ("v2", v2_three_einsum_optimal),
            ("v3", v3_fused_opt_einsum),
            ("v4", v4_precomputed),
        ):
            r = fn(*inputs)
            diff = float(np.max(np.abs(r - r1)))
            print(f"  {label} vs v1 max abs diff: {diff:.2e}")
            assert diff < 1e-6, f"{label} disagrees with v1"


if __name__ == "__main__":
    main()
