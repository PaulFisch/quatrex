"""Rank-truncation error of the decomposed SSE (fig:res_decomp_error, fig:res_decomp_amp).

  decomp_rank_error   left:  the ONE-SHOT vertex error -- both self-energies
                             evaluated on the SAME ballistic G, so this is the
                             vertex error propagated through the bubble with no
                             SCBA feedback. Drawn against the theory chapter's
                             bound, 2 eps_R.
                      right: the SELF-CONSISTENT observable errors -- a full SCBA
                             per rank against the dense-vertex run, every
                             observable, with the comparison floor shaded.
  decomp_amplification  observable error / eps_R. This is the result: the vertex
                        -> Sigma map is faithful (amplification ~1), but the
                        Sigma -> observable map suppresses the error by one to
                        three orders of magnitude.

THE COMPARISON FLOOR. The dense q-folded vertex and the factored vertex do not
carry the same FC3 block support: the q-fold keeps 7 transport-offset pairs, the
factors span the full 5x5 window over offsets [-2..2]. The 18 extra pairs hold
6.8e-5 of the vertex amplitude (measured), and Sigma is bilinear in the vertex, so
no error against this reference can fall below ~1.4e-4 however good the fit gets.
R=64 already sits there. Errors at or under the shaded band measure the REFERENCE,
not the rank -- which is why the curves stop descending, and why R=128 is not an
improvement on R=64.

Data: one-shot numbers are literals from phonon/studies/_rank_error_sse.py
      (scratch rank_err.log); the self-consistent ones from
      phonon/scripts/data/decomposed_sse.csv (the L3 accuracy campaign).

Run:  python phonon/scripts/figures/decomposed_sse_error.py
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

EPS_R = {8: 0.1554, 16: 0.0834, 32: 0.0071, 64: 0.0006, 128: 0.0004}
FLOOR = 1.4e-4          # see the module docstring

# One-shot vertex -> Sigma error, both self-energies on the SAME ballistic G
# (phonon/studies/_rank_error_sse.py). Sigma^R is absent by construction: under
# retarded_method="half" its Hermitian part is identically zero, so the retarded
# error is carried by Gamma.
ONESHOT_SIGMA = {8: 2.37e-1, 16: 1.42e-1, 32: 4.83e-3, 64: 2.38e-4, 128: 2.67e-4}
ONESHOT_GAMMA = {8: 1.31e-1, 16: 5.42e-2, 32: 2.94e-3, 64: 3.53e-4, 128: 2.50e-4}

# The observables compared self-consistently, in reporting order.
OBS = [("err_heat", r"heat current $J$", "C0"),
       ("err_G", r"conductance $G$", "C1"),
       ("err_j_wq", r"$j(\omega,q_\perp)$", "C2"),
       ("err_pabs", r"absorption $P(x)$", "C4"),
       ("err_ldos", "LDOS", "C5"),
       ("err_gl", r"$G^{<}$ diag", "C6")]


def _rows(length):
    with CSV.open() as fh:
        return [r for r in csv.DictReader(fh)
                if r["length"] == length and int(r["rank"]) > 0]


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rows = _rows("L3") or _rows("L10")
    length = "L3" if _rows("L3") else "L10"
    if not rows:
        raise SystemExit("no rank legs in the archive")
    ranks = sorted(int(r["rank"]) for r in rows)
    by_rank = {int(r["rank"]): r for r in rows}

    def col(name):
        out = []
        for r in ranks:
            v = by_rank[r].get(name, "")
            out.append(float(v) if v not in ("", "nan") else np.nan)
        return np.array(out)

    # ---------------- figure 1: the two errors ------------------------------
    fig, (a0, a1) = style.figure(ncols=2, width=4.5, height=3.5)

    orr = sorted(ONESHOT_SIGMA)
    a0.loglog(orr, [EPS_R[r] for r in orr], "s--", color="0.55",
              label=r"FC3 residual $\varepsilon_R$")
    a0.loglog(orr, [2 * EPS_R[r] for r in orr], "-", color="0.75", lw=1.0,
              label=r"bound $2\varepsilon_R$")
    a0.loglog(orr, [ONESHOT_SIGMA[r] for r in orr], "o-", color="C3",
              label=r"$\Sigma^{\lessgtr}$")
    a0.loglog(orr, [ONESHOT_GAMMA[r] for r in orr], "^-", color="C0",
              label=r"$\Gamma=i(\Sigma^{>}-\Sigma^{<})$")
    a0.axhspan(0, FLOOR, color="0.85", zorder=0)
    a0.annotate("comparison floor", xy=(40, FLOOR), xytext=(40, FLOOR * 1.35),
                fontsize=7, color="0.35")
    a0.set_xlabel("CP rank $R$")
    a0.set_ylabel("relative error (one-shot, fixed $G$)")
    a0.set_xticks(orr); a0.set_xticklabels([str(r) for r in orr])
    a0.xaxis.set_minor_formatter(NullFormatter())
    a0.legend(fontsize=7, loc="lower left")

    a1.loglog(ranks, [EPS_R[r] for r in ranks], "s--", color="0.55",
              label=r"FC3 residual $\varepsilon_R$")
    for key, lab, c in OBS:
        v = col(key)
        if np.all(np.isnan(v)):
            continue
        a1.loglog(ranks, v, "o-", color=c, label=lab, ms=4)
    a1.set_ylim(0.2 * FLOOR, None)
    a1.axhspan(0, FLOOR, color="0.85", zorder=0)
    a1.annotate("comparison floor", xy=(ranks[0] * 1.1, FLOOR),
                xytext=(ranks[0] * 1.1, FLOOR * 1.5), fontsize=7, color="0.35")
    a1.set_xlabel("CP rank $R$")
    a1.set_ylabel(f"relative error (self-consistent SCBA, {length})")
    a1.set_xticks(ranks); a1.set_xticklabels([str(r) for r in ranks])
    a1.xaxis.set_minor_formatter(NullFormatter())
    a1.legend(fontsize=6.5, loc="lower left", ncol=2)
    style.save(fig, "decomp_rank_error", directory=FIGDIR)

    # ---------------- figure 2: amplification -------------------------------
    fig, ax = style.figure(width=5.0, height=3.6)
    eps = np.array([EPS_R[r] for r in ranks])
    ax.axhline(2.0, color="0.5", ls="--", lw=1.2)
    ax.annotate(r"theory bound $2\varepsilon_R$", xy=(ranks[0], 2.0),
                xytext=(ranks[0] * 1.05, 2.4), fontsize=7.5, color="0.4")
    ax.semilogy(orr, [ONESHOT_SIGMA[r] / EPS_R[r] for r in orr], "o-",
                color="C3", label=r"$\Sigma$ (one-shot vertex error)")
    for key, lab, c in OBS:
        v = col(key)
        if np.all(np.isnan(v)):
            continue
        ax.semilogy(ranks, v / eps, "o-", color=c, label=lab, ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("CP rank $R$")
    ax.set_ylabel(r"amplification: error $/\ \varepsilon_R$")
    allr = sorted(set(ranks) | set(orr))
    ax.set_xticks(allr); ax.set_xticklabels([str(r) for r in allr])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=7, ncol=2)
    style.save(fig, "decomp_amplification", directory=FIGDIR)

    # ---------------- the claim-verification trail --------------------------
    print("ONE-SHOT vertex -> Sigma error (same ballistic G, no SCBA feedback):")
    print(f"{'R':>5} {'eps_R':>9} {'Sigma':>10} {'amplif.':>9} {'Gamma':>10}")
    for r in orr:
        print(f"{r:>5} {EPS_R[r]:>9.2%} {ONESHOT_SIGMA[r]:>10.2e} "
              f"{ONESHOT_SIGMA[r] / EPS_R[r]:>9.2f} {ONESHOT_GAMMA[r]:>10.2e}")
    amps = [ONESHOT_SIGMA[r] / EPS_R[r] for r in orr]
    print(f"  -> the 2*eps_R bound HOLDS at every rank; "
          f"measured amplification {min(amps):.2f}-{max(amps):.2f}")

    print(f"\nSELF-CONSISTENT observable error ({length}, full SCBA per rank):")
    hdr = f"{'R':>5} {'eps_R':>9}" + "".join(f"{l[:11]:>12}" for _, l, _ in OBS)
    print(hdr)
    for i, r in enumerate(ranks):
        line = f"{r:>5} {EPS_R[r]:>9.2%}"
        for key, _, _ in OBS:
            v = col(key)[i]
            line += f"{v:>12.1e}" if np.isfinite(v) else f"{'--':>12}"
        print(line)

    print("\nAMPLIFICATION (observable error / eps_R):")
    for i, r in enumerate(ranks):
        line = f"{r:>5}          "
        for key, _, _ in OBS:
            v = col(key)[i] / EPS_R[r]
            line += f"{v:>12.4f}" if np.isfinite(v) else f"{'--':>12}"
        print(line)
    print("\n<1 means the physics is LESS sensitive than the tensor residual.")
    print(f"Errors below {FLOOR:.0e} measure the reference, not the rank.")


if __name__ == "__main__":
    main()
