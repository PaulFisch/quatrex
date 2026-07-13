"""Cost of the decomposed three-phonon SSE (fig:res_decomp_cost, fig:res_decomp_scaling).

  decomp_kernel_speedup   ring-contraction speedup over the dense vertex vs rank,
                          for the coupled-q film (b=6, N_q=81) and the Gamma-only
                          wire (b=63); the R ~ 130 break-even is marked.
  decomp_cost_scaling     measured per-iteration SSE cost vs rank on the L10 film
                          against the R^2 term of the cost model, plus the
                          end-to-end SCBA wall time.

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

    # Break-even: the rank at which the factored ring would cost the dense ring.
    # The fitted exponent is the quantity of interest -- R* itself is an
    # extrapolation beyond the measured range and is quoted as such.
    lg = np.polyfit(np.log(RANKS), np.log([FILM_NEW[r] for r in RANKS]), 1)
    r_star = float(np.exp((np.log(np.mean(list(FILM_DENSE.values()))) - lg[1]) / lg[0]))
    print(f"  fitted exponent {lg[0]:.2f} (the R^2 Gram term); at the largest rank")
    print(f"  measured (R=128) the kernel is still {film[128]:.1f}x dense, and the")
    print(f"  extrapolated break-even is R* ~ {r_star:.0f}")

    # ---------------- figure 1: speedup ------------------------------------
    fig, ax = style.figure(width=5.0, height=3.6)
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
    ax.legend(fontsize=7.5, loc="lower left")
    style.save(fig, "decomp_kernel_speedup", directory=FIGDIR)

    # ---------------- figure 2: SCBA cost scaling ---------------------------
    rows = [r for r in _rows() if r["length"] == "L10" and int(r["rank"]) > 0]
    rr = sorted(int(r["rank"]) for r in rows)
    wall = {int(r["rank"]): float(r["wall_s"]) for r in rows
            if r["wall_s"] not in ("", "nan")}

    fig, (a0, a1) = style.figure(ncols=2, width=4.4, height=3.4)

    pr = sorted(L10_PHPH_PER_ITER)
    a0.loglog(pr, [L10_PHPH_PER_ITER[r] for r in pr], "o-", color="C0",
              label="measured")
    ref = [L10_PHPH_PER_ITER[pr[0]] * (r / pr[0]) ** 2 for r in pr]
    a0.loglog(pr, ref, "--", color="0.5", label=r"$\propto R^{2}$")
    a0.set_xlabel("CP rank $R$")
    a0.set_ylabel("SSE per SCBA iteration (s)")
    a0.set_xticks(pr); a0.set_xticklabels([str(r) for r in pr])
    a0.legend(fontsize=7.5)

    wr = sorted(wall)
    a1.loglog(wr, [wall[r] for r in wr], "o-", color="C0", label="factored")
    a1.set_xlabel("CP rank $R$")
    a1.set_ylabel("SCBA wall time (s)")
    a1.set_xticks(wr); a1.set_xticklabels([str(r) for r in wr])
    a1.legend(fontsize=7.5)
    style.save(fig, "decomp_cost_scaling", directory=FIGDIR)

    print("\nL10 SCBA (121 ranks):")
    for r in wr:
        print(f"  R={r:>3}: wall {wall[r]:8.1f} s")
    print("\nper-iteration SSE cost ratio (doubling R):")
    for i in range(1, len(pr)):
        print(f"  R {pr[i-1]:>3} -> {pr[i]:>3}: "
              f"{L10_PHPH_PER_ITER[pr[i]] / L10_PHPH_PER_ITER[pr[i-1]]:.2f}x "
              f"(R^2 would be 4.00x)")


if __name__ == "__main__":
    main()
