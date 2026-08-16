"""Cost of the decomposed three-phonon SSE (fig:res_decomp_cost, fig:res_decomp_scaling).

Data:
  Kernel timings are literals: they come from the standalone micro-benchmark
  `phonon/studies/_bench_factored_sse.py --verify` run on tortin at
  OMP_NUM_THREADS=1 over the full q mesh (logs: cluster/bench-decomp/run.log,
  cluster/bench-legacy/run.log, and the Gamma/b=63 shape in scratch gamma63.log).
  They are not derivable from any committed run.npz -- the benchmark contracts the
  ring in isolation, which no production run does. Parity vs the dense ring fed the
  identical vertex was ~5e-15 at every point.

  The SCBA points come from phonon/scripts/data/decomposed_sse.csv.

Run:  python phonon/scripts/figures/decomposed_sse_cost.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from matplotlib.ticker import NullFormatter

from phonon.studies import style

CSV = ROOT / "phonon/scripts/data/decomposed_sse.csv"
FIGDIR = ROOT / "document/fig/transport_sweeps"

RANKS = [8, 16, 32, 64, 128]

# --- micro-benchmark, total work at full q (seconds) -------------------------
# film sifilm ns3_nk9: b=6, N_q=81, n_tau=60, 7 pairs, 181 quads
FILM_DENSE = {8: 852.86, 16: 853.18, 32: 854.06, 64: 853.27, 128: 852.52}
FILM_NEW = {8: 0.88, 16: 2.67, 32: 9.36, 64: 38.66, 128: 200.27}
# The legacy O(N_q^2) kernel only ever reached R=8. Its numbers come from a
# SEPARATE benchmark run (cluster/bench-legacy) -- keep the comparison inside
# that run, do not mix it with the FILM_NEW timings above.
LEGACY_RUN = {"dense": 854.75, "legacy": 85.53, "new": 0.94}   # all at R=8
# Gamma-only wire (d5a shape): b=63, N_q=1 -- the regime the factored path could
# not reach at all before the Gamma wiring
WIRE_DENSE = {16: 780.38, 32: 774.73, 64: 795.59}
WIRE_NEW = {16: 0.22, 32: 0.42, 64: 1.43}

# per-iteration phonon-phonon cost on the L10 film (s), from the campaign log
L10_PHPH_PER_ITER = {8: 1.542, 16: 3.063, 32: 14.558, 64: 66.466}


def _rows():
    with CSV.open() as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)

    film = {r: FILM_DENSE[r] / FILM_NEW[r] for r in RANKS}
    wire = {r: WIRE_DENSE[r] / WIRE_NEW[r] for r in sorted(WIRE_NEW)}

    legacy_vs_dense = LEGACY_RUN["dense"] / LEGACY_RUN["legacy"]
    new_vs_legacy = LEGACY_RUN["legacy"] / LEGACY_RUN["new"]

    print("kernel speedup over the dense ring (total work, full q):")
    for r in RANKS:
        print(f"  film R={r:>3}: {film[r]:8.1f}x")
    for r in sorted(wire):
        print(f"  wire R={r:>3}: {wire[r]:8.1f}x")
    print(f"  at R=8, within one run: legacy is {legacy_vs_dense:.0f}x dense, "
          f"new is {new_vs_legacy:.0f}x legacy")

    # NO break-even is quoted. The curve is not a power law: the local exponent
    # runs 1.60 -> 1.81 -> 2.05 -> 2.37 across the measured range, so a single
    # fitted slope averages a steepening curve and any extrapolated R* inherits
    # that. What IS measurable: at the largest rank measured the kernel is still
    # faster than dense.
    lo = np.diff(np.log([FILM_NEW[r] for r in RANKS])) / np.diff(np.log(RANKS))
    print("  local exponent between successive ranks: "
          + ", ".join(f"{e:.2f}" for e in lo))
    print(f"  -> not a power law; no break-even rank is extrapolated.")
    print(f"  at the largest rank measured (R=128) the kernel is still "
          f"{film[128]:.1f}x the dense ring.")

    # ---------------- figure 1: speedup ------------------------------------
    fig, ax = style.doc_figure(frac=0.48, aspect=0.80)
    ax.loglog(RANKS, [film[r] for r in RANKS], "o-", color="C0",
              label=r"film, coupled $q_\perp$ ($b=6$, $N_q=81$)")
    ax.loglog(sorted(wire), [wire[r] for r in sorted(wire)], "s-", color="C2",
              label=r"wire, $\Gamma$ only ($b=63$)")
    ax.loglog([8], [legacy_vs_dense], "^", color="C1",
              label="film, previous kernel")
    ax.axhline(1.0, color="0.4", lw=1.0, ls="--")
    ax.annotate("dense vertex", xy=(128, 1.0), xytext=(60, 1.15),
                color="0.4", fontsize=7.5)
    ax.set_xlabel("CP rank $R$")
    ax.set_ylabel("speed-up over the dense ring")
    ax.set_xticks(RANKS)
    ax.set_xticklabels([str(r) for r in RANKS])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=7.5, loc="lower left")
    style.save(fig, "decomp_kernel_speedup", directory=FIGDIR)

    # ---------------- figure 2: where the SCBA time actually goes -------------
    rows = [r for r in _rows() if r["length"] == "L10" and int(r["rank"]) > 0]
    rr = sorted(int(r["rank"]) for r in rows)
    if not rr:
        print("\n(no SCBA legs in the archive -- skipping decomp_cost_scaling)")
        return
    g = {int(r["rank"]): r for r in rows}
    fnum = lambda r, k: (float(g[r][k]) if g[r].get(k, "") not in ("", "nan")
                         else float("nan"))
    wall = {r: fnum(r, "wall_s") for r in rr}
    ring = {r: fnum(r, "t_ring") for r in rr}
    sse = {r: fnum(r, "t_sse") for r in rr}
    nit = {r: int(float(g[r]["n_iter"])) for r in rr}

    fig, (a0, a1) = style.figure(ncols=2, width=4.4, height=3.4)

    # LEFT: the ring contraction per iteration -- the quantity the R^2 Gram term
    # predicts. Plotting the SSE TOTAL instead hides an R-independent floor (the
    # FFT + fold), which is 71% of the SSE at R=8 and makes the curve look
    # sub-quadratic when the ring itself is super-quadratic.
    rok = [r for r in rr if np.isfinite(ring[r]) and ring[r] > 0 and nit[r]]
    if rok:
        per = [ring[r] / nit[r] for r in rok]
        a0.loglog(rok, per, "o-", color="C0", label="ring contraction")
        a0.loglog(rok, [sse[r] / nit[r] for r in rok], "s--", color="C1",
                  label="whole SSE (ring + FFT + fold)")
        ref = [per[0] * (r / rok[0]) ** 2 for r in rok]
        a0.loglog(rok, ref, ":", color="0.5", label=r"$\propto R^{2}$")
        a0.set_xticks(rok); a0.set_xticklabels([str(r) for r in rok])
    a0.set_xlabel("CP rank $R$")
    a0.set_ylabel("per SCBA iteration (s)")
    a0.legend(fontsize=7)

    # RIGHT: the wall time, split. A rank-independent offset (imports, MPI init,
    # geometry + FC3 load, output write) sits outside the loop at every rank; at
    # low rank it is the MAJORITY of the run.
    off = [wall[r] - sse[r] for r in rr]
    a1.bar([str(r) for r in rr], [sse[r] for r in rr], color="C0",
           label="three-phonon SSE")
    a1.bar([str(r) for r in rr], off, bottom=[sse[r] for r in rr], color="0.75",
           label="everything else (setup, solver, I/O)")
    a1.set_xlabel("CP rank $R$")
    a1.set_ylabel("SCBA wall time (s)")
    a1.set_yscale("log")
    a1.legend(fontsize=7)
    style.save(fig, "decomp_cost_scaling", directory=FIGDIR)

    print(f"\nL10 SCBA -- where the time goes:")
    print(f"{'R':>4} {'iters':>6} {'wall s':>9} {'SSE s':>9} {'SSE/wall':>9} "
          f"{'ring s':>9} {'ring/SSE':>9}")
    for r in rr:
        print(f"{r:>4} {nit[r]:>6} {wall[r]:>9.1f} {sse[r]:>9.1f} "
              f"{sse[r]/wall[r]:>8.1%} {ring[r]:>9.1f} {ring[r]/sse[r]:>8.1%}")
    print("  -> at low rank the SSE is a MINORITY of the run: the end-to-end")
    print("     speed-up is bounded by the rank-independent remainder, not by the")
    print("     kernel. Quote the kernel benchmark for kernel claims.")
    if rok:
        lr = np.diff(np.log([ring[r] / nit[r] for r in rok])) / np.diff(np.log(rok))
        print("  ring-contraction local exponent: "
              + ", ".join(f"{e:.2f}" for e in lr) + "  (R^2 would be 2.00)")


if __name__ == "__main__":
    main()
