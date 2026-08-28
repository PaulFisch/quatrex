"""The cost table of task T5: structured `Sigma` against reblocking.

The competitor is fixed by the measurement in
``phonon/docs/spatial_analytic_tail.md`` Sec. 11: reblocking the device at two
transport cells per BTD block is MORE accurate on the lead current than the
production pin (2.74e-03 against 6.31e-03) while discarding 58.8 % of
``|Sigma|``. Anything structured has to beat that, not the pin.

The accounting here differs from the proposal's Sec. 39, which sets the
augmented block ``d + r+ + r-`` against reblocking's ``2d`` and reads off a
factor. That comparison is of block WIDTHS, and it is not the cost. A block
solve is cubic in the width and linear in the block COUNT, and reblocking
halves the count:

    pin                 N     blocks of  d              ->      N d^3
    m-cell reblock      N/m   blocks of  m d            ->  m^2 N d^3
    augmented RGF       N     blocks of  d + r+ + r-     ->      N (d+2r)^3

So a 2-cell reblock costs 4x the pin, not 2x, and the augmented block has to
come in under ``4^(1/3) d ~ 1.587 d`` to beat it -- i.e. ``r < 0.293 d`` per
side, which is nearly six times harder than the width comparison suggests.

Storage per frequency is quadratic in the width at the same block count, so the
memory column is the gentler one: ``m N d^2`` against ``N (d+2r)^2``.

The iterative alternative (proposal Sec. 15/16: never form the operator, solve
by Krylov on the matrix-free action) is costed separately and from measurement,
not from the width: the Krylov basis was measured to reach ``N_D`` exactly, so
a solve costs ``N_D`` matvecs and the compression buys nothing at solve time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "cluster"
TOLS = (1e-2, 1e-3, 1e-4)

# Sec. 11, same frozen 20-cell chain, dense re-solve against the untruncated D.
EPS_J = {"pin": 6.31e-03, "reblock2": 2.74e-03, "reblock3": 1.18e-02,
         "reblock4": 1.12e-02, "modal-direct": 2.43e-01, "congruence": 3.27e-02}


def rgf_cost(n_blocks: float, width: float) -> float:
    """Leading block-RGF flop count, in units of a single ``d x d`` cube."""
    return n_blocks * width ** 3


def break_even_dof(r_per_side: float, m: int = 2) -> float:
    """Smallest ``d`` at which an augmented RGF undercuts an ``m``-cell reblock.

    ``(d + 2r)^3 <= m^2 d^3``  <=>  ``d >= 2r / (m^(2/3) - 1)``.
    """
    return 2.0 * r_per_side / (m ** (2.0 / 3.0) - 1.0)


def load(paths) -> list[dict]:
    rows = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        if "offdiag_names" not in z.files:
            print(f"  (skipping {p.name}: not a rank-scaling artifact)")
            continue
        names = [str(s) for s in z["offdiag_names"]]
        od = {n: dict(zip(TOLS, z["offdiag"][i])) for i, n in enumerate(names)}
        rows.append(dict(name=str(z["name"]), n=int(z["n_slabs"]),
                         d=int(z["n_dof"]), offdiag=od,
                         caps=dict(zip(names, z["offdiag_caps"])),
                         joint=dict(zip(TOLS, z["joint_all"])),
                         n_live=int(z["n_live"]), n_freq=int(z["n_freq"])))
    return rows


def table(rows, tol: float = 1e-2, key: str = "Sigma^<|b1") -> None:
    print(f"\nSemiseparable rank of {key} at tol={tol:g}, and what it costs")
    print("  bed              N   d | r_Sigma  cap | aug width  | RGF cost, "
          "units of N d^3")
    print("  " + "-" * 76)
    print(f"  {'(pin)':<14s} {'--':>3s} {'--':>3s} |  {'--':>6s} {'--':>4s} | "
          f"{'d':>10s} | {1.0:>8.2f}")
    for m in (2, 3, 4):
        print(f"  {f'({m}-cell reblock)':<14s} {'--':>3s} {'--':>3s} | "
              f"{'--':>6s} {'--':>4s} | {f'{m}d':>10s} | {m ** 2:>8.2f}")
    print("  " + "-" * 76)
    for r in rows:
        rs = r["offdiag"].get(key, r["offdiag"].get("Sigma^<", {})).get(tol, -1)
        if rs < 0:
            print(f"  {r['name']:<14s} {r['n']:3d} {r['d']:3d} | "
                  "device too short for an interior split")
            continue
        cap = r["caps"].get(key, r["caps"].get("Sigma^<", 0))
        w = r["d"] + 2 * rs
        cost = rgf_cost(1.0, w) / rgf_cost(1.0, r["d"])
        flag = "  <-- CAPPED" if rs >= cap else ""
        print(f"  {r['name']:<14s} {r['n']:3d} {r['d']:3d} | {rs:6d} "
              f"{cap:4d} | {w:10d} | {cost:8.2f}{flag}")

    print("\n  Break-even: the DOF count at which an augmented RGF would first")
    print("  undercut a reblock, d >= 2r / (m^(2/3) - 1)")
    print("     r/side |  vs 2-cell  vs 3-cell  vs 4-cell")
    for rr in (2, 4, 6, 8, 10, 14):
        print(f"     {rr:6d} | " + "".join(
            f"{break_even_dof(rr, m):10.0f}" for m in (2, 3, 4)))


def scaling(rows, key: str = "Sigma^<|b1") -> None:
    """Does r_Sigma saturate in N?  Sec. 48.2's stop condition is r ~ N/2."""
    by_d: dict[int, list] = {}
    for r in rows:
        od = r["offdiag"].get(key, r["offdiag"].get("Sigma^<", {}))
        rs = od.get(1e-2, -1)
        if rs >= 0:
            by_d.setdefault(r["d"], []).append(
                (r["n"], rs, r["caps"].get(key, r["caps"].get("Sigma^<", 0))))
    for d, pts in sorted(by_d.items()):
        if len(pts) < 2:
            continue
        pts.sort()
        n = np.array([p[0] for p in pts], float)
        rr = np.array([p[1] for p in pts], float)
        slope = float(np.polyfit(np.log(n), np.log(rr), 1)[0]) if len(pts) > 2 \
            else float(np.log(rr[-1] / rr[0]) / np.log(n[-1] / n[0]))
        print(f"\n  d={d}: r_Sigma vs N -- "
              + ", ".join(f"N={int(a)}:{int(b)}(cap {int(c)})" for a, b, c in pts)
              + f"\n       exponent {slope:.2f}  "
              + ("(saturating)" if slope < 0.35 else
                 "(growing -- Sec. 48.2 stop condition is linear)"))


def main(argv=None) -> int:
    paths = [Path(p) for p in (argv or sys.argv[1:])]
    if not paths:
        paths = sorted((OUT).rglob("*_rank.npz")) + sorted(OUT.rglob("*_fine.npz"))
    if not paths:
        print("no rank npz files found; pass paths explicitly")
        return 1
    rows = [r for r in load(paths) if r is not None]
    for key in ("Sigma^<", "Sigma^<|b1"):
        table(rows, key=key)
        scaling(rows, key=key)
    print("\n  eps(J_L) on the 20-cell chain (Sec. 11), for the accuracy axis:")
    for k, v in EPS_J.items():
        print(f"    {k:<14s} {v:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
