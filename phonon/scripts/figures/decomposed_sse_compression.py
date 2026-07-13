"""FC3 vertex compression by the CP/INDSCAL fit (fig:res_decomp_compression).

  decomp_compression   left:  INDSCAL fit residual eps_R vs rank, with the 1%
                              line that fixes the operating rank;
                       right: vertex storage vs device length -- the factors are
                              FLAT in device length (the fit is on the bulk FC3),
                              while the dense q-folded vertex grows with it.

The right panel is the structural argument for the decomposition and it is not a
FLOP count: the dense q-fold is one npz per device length holding
(7*n_slabs - 6) x N_q^2 blocks, and every MPI rank deserialises all of them. At
L10 that is 1.56 GB in ~420k arrays -- the L10 dense reference never reached its
first SCBA iteration because of it, while the factored legs converged in minutes.

Data: eps_R and the file sizes are literals from the geometry build log
      (cluster/sifilm-L10-geom/run.log), which reports the INDSCAL fit residual
      per rank; the fits are cached on the BULK FC3 hash, so L3 and L10 get
      byte-identical factor files.

Run:  python phonon/scripts/figures/decomposed_sse_compression.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from matplotlib.ticker import NullFormatter

from phonon.studies import style

FIGDIR = ROOT / "document/fig/transport_sweeps"

RANKS = [8, 16, 32, 64, 128]
EPS_R = {8: 0.1554, 16: 0.0834, 32: 0.0071, 64: 0.0006, 128: 0.0004}
FACTOR_MB = {8: 0.2497, 16: 0.5897, 32: 1.3208, 64: 2.8229, 128: 6.3009}
# dense q-folded vertex, per device length (slabs -> MB)
QFOLD_MB = {3: 365.52, 10: 1559.37}
OPERATING = 32


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, (a0, a1) = style.figure(ncols=2, width=4.4, height=3.4)

    # ---- left: fit residual ------------------------------------------------
    a0.loglog(RANKS, [100 * EPS_R[r] for r in RANKS], "o-", color="C0")
    a0.axhline(1.0, color="C3", lw=1.0, ls="--")
    a0.annotate(r"$1\%$", xy=(8.6, 1.0), xytext=(8.6, 1.3),
                color="C3", fontsize=7.5)
    a0.plot([OPERATING], [100 * EPS_R[OPERATING]], "o", ms=9, mfc="none",
            mec="C3", mew=1.4)
    a0.annotate(f"$R={OPERATING}$", xy=(OPERATING, 100 * EPS_R[OPERATING]),
                xytext=(OPERATING * 1.15, 100 * EPS_R[OPERATING] * 2.4),
                color="C3", fontsize=7.5)
    a0.set_xlabel("CP rank $R$")
    a0.set_ylabel(r"fit residual $\varepsilon_R$ (\%)".replace("\\%", "%"))
    a0.set_xticks(RANKS); a0.set_xticklabels([str(r) for r in RANKS])
    a0.xaxis.set_minor_formatter(NullFormatter())

    # ---- right: storage vs device length -----------------------------------
    lengths = sorted(QFOLD_MB)
    a1.semilogy(lengths, [QFOLD_MB[n] for n in lengths], "o-", color="C3",
                label="dense $q$-folded vertex")
    for r, col in ((8, "C2"), (32, "C0"), (128, "C1")):
        a1.semilogy(lengths, [FACTOR_MB[r]] * len(lengths), "s--",
                    color=col, lw=1.6 if r == OPERATING else 1.1,
                    label=(f"CP factors, $R={r}$"
                           + (" (operating)" if r == OPERATING else "")))
    a1.set_xlabel("device length (transport cells)")
    a1.set_ylabel("vertex storage (MB)")
    a1.set_xticks(lengths)
    a1.legend(fontsize=7)
    style.save(fig, "decomp_compression", directory=FIGDIR)

    print("INDSCAL fit residual and factor storage (the fit is on the BULK FC3,")
    print("so the factor files are identical at every device length):")
    print(f"{'R':>5} {'eps_R':>9} {'factors':>10} "
          f"{'vs L3 dense':>12} {'vs L10 dense':>13}")
    for r in RANKS:
        print(f"{r:>5} {100 * EPS_R[r]:>8.2f}% {FACTOR_MB[r]:>9.2f} MB "
              f"{QFOLD_MB[3] / FACTOR_MB[r]:>11.0f}x "
              f"{QFOLD_MB[10] / FACTOR_MB[r]:>12.0f}x")
    print()
    print(f"dense q-folded vertex: {QFOLD_MB[3]:.0f} MB at L3 -> "
          f"{QFOLD_MB[10]:.0f} MB at L10 "
          f"({QFOLD_MB[10] / QFOLD_MB[3]:.1f}x for {10 / 3:.1f}x the slabs)")
    print(f"at the operating rank R={OPERATING}: eps_R = "
          f"{100 * EPS_R[OPERATING]:.2f}%, factors {FACTOR_MB[OPERATING]:.2f} MB "
          f"= {QFOLD_MB[10] / FACTOR_MB[OPERATING]:.0f}x smaller than the L10 vertex")


if __name__ == "__main__":
    main()
