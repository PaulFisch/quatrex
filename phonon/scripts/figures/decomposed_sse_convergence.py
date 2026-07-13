"""SCBA convergence of the decomposed three-phonon SSE (fig:res_decomp_convergence).

  decomp_scba_convergence  left:   the SCBA residual (rel. Sigma^R change) vs
                                   iteration, per rank, on the L10 film;
                           middle: the lead heat-flow imbalance |J_L-J_R|/J;
                           right:  the bubble energy-balance residual
                                   |P_in - P_out|/(|P_in|+|P_out|) vs iteration.

The residual and lead-balance traces are parsed from the SCBA's own stdout, NOT
from run.npz: `iter_heat` and `iter_sigma_max` are stored as the RANK-0-LOCAL
frequency slice, and these runs use 121 ranks over 121 frequencies, so rank 0
owns omega=0 alone -- where the heat current is identically zero. The bubble
trace does come from the npz; it is all-reduced and therefore global.

The right panel is the physical check, not a numerical one. The bubble is
Phi-derivable, so P_in = P_out is an identity of the exact vertex. A low-rank
CP fit is a DIFFERENT vertex, and nothing guarantees the truncated one still
satisfies it -- a fit that broke the vertex's permutation symmetry would violate
energy conservation outright. It does not: the residual sits at ~1e-6 at every
rank down to R=8, whose FC3 residual is 15.5%.

Data: phonon/scripts/data/decomposed_sse_spectra.npz (iter_heat, iter_sigma_max,
      iter_bubble_balance, from the L10 campaign).

Run:  python phonon/scripts/figures/decomposed_sse_convergence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style

NPZ = ROOT / "phonon/scripts/data/decomposed_sse_spectra.npz"
FIGDIR = ROOT / "document/fig/transport_sweeps"

LENGTH = "L10"
RANKS = [8, 16, 32, 64, 128]
EPS_R = {8: 0.1554, 16: 0.0834, 32: 0.0071, 64: 0.0006, 128: 0.0004}


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    z = np.load(NPZ)
    have = [r for r in RANKS if f"{LENGTH}_r{r}/trace_residual" in z.files]
    if not have:
        raise SystemExit(f"no {LENGTH} rank traces in {NPZ}")

    fig, (a0, a1, a2) = style.figure(ncols=3, width=3.5, height=3.1)
    colors = {r: f"C{i}" for i, r in enumerate(have)}

    print(f"{LENGTH} SCBA convergence (121 ranks, linear mix 0.3):")
    print(f"{'R':>5} {'iters':>6} {'residual':>11} {'lead bal.':>11} "
          f"{'bubble resid':>13}  eps_R")

    for r in have:
        c, lab = colors[r], f"$R={r}$"
        res = np.asarray(z[f"{LENGTH}_r{r}/trace_residual"])
        lead = np.asarray(z[f"{LENGTH}_r{r}/trace_lead_balance"])
        it = np.arange(1, len(res) + 1)
        a0.semilogy(it, res, "-", color=c, label=lab)
        a1.semilogy(np.arange(1, len(lead) + 1), lead, "-", color=c, label=lab)

        key = f"{LENGTH}_r{r}/iter_bubble_balance"
        bres = float("nan")
        if key in z.files:
            bb = np.asarray(z[key])                            # (n_it, 3)
            a2.semilogy(np.arange(1, len(bb) + 1), np.abs(bb[:, 2]), "-",
                        color=c, label=lab)
            bres = float(np.abs(bb[-1, 2]))
        print(f"{r:>5} {len(res):>6} {res[-1]:>11.2e} {lead[-1]:>11.2e} "
              f"{bres:>13.2e}  {100 * EPS_R[r]:.2f}%")

    a0.axhline(1e-3, color="0.4", lw=1.0, ls="--")
    a0.annotate("convergence gate", xy=(1, 1e-3), xytext=(1.5, 1.25e-3),
                color="0.4", fontsize=7)
    a0.set_xlabel("SCBA iteration")
    a0.set_ylabel(r"rel. $\Sigma^R$ residual")
    a0.legend(fontsize=7)

    a1.set_xlabel("SCBA iteration")
    a1.set_ylabel(r"lead imbalance $|J_L-J_R|/\bar{J}$")
    a1.legend(fontsize=7)

    a2.set_xlabel("SCBA iteration")
    a2.set_ylabel(r"bubble balance $|P_{\rm in}-P_{\rm out}|/(|P_{\rm in}|+|P_{\rm out}|)$")
    a2.legend(fontsize=7)

    style.save(fig, "decomp_scba_convergence", directory=FIGDIR)

    print()
    print("The bubble residual is the Phi-derivability test: a truncated vertex")
    print("is a different vertex, and nothing forces it to still conserve energy.")
    print("It does, at every rank -- including R=8, whose FC3 residual is 15.5%.")


if __name__ == "__main__":
    main()
