"""Conservation figures for the energy-conservation write-up (appendix
``app:conservation`` + the eta=0 results section). Companion to
``eta0_convergence.py``/``eta0_physics.py``; same style and output directory, so
the reruns drop straight into the document includes.

Four figures, every number printed for the LaTeX, each from VALID cached data:

  F1 conservation vs SCBA iteration   eta0_cnt33_conservation_iter
        phonon/scripts/out/prod/cnt33_eta0/L2_anh.npz  (iter_heat,
        iter_bubble_balance, iter_sigma_max) -- the production transport
        iteration: the bubble balance is pinned at machine precision every
        step while the Sigma^R residual and the lead imbalance converge.
  F2/F3 conductance ratio + lead balance vs broadening eta and grid
        eta0_cnt33_ratio_eta
        the matched-eta sweep of tab:cons_ratio (conservation.ratio_eta) with
        the eta->0 extrapolation, plus the 181/241-pt grid points.
  F4 lead conservation + iteration count vs temperature
        eta0_cnt33_conservation_T
        phonon/scripts/out/prod/cnt33_eta0/summary.json.
  F5 discrete-bubble energy balance is exactly conserving (roundoff floor)
        conservation_bubble_replica
        phonon.studies.conservation.replica_check() (float64 vs float128).

Run:  OMP_NUM_THREADS=1 python phonon/scripts/figures/conservation_figs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style

PROD = ROOT / "phonon/scripts/out/prod/cnt33_eta0"
FIGDIR = ROOT / "document/fig/transport_sweeps"

# --- cnt33 (L2, 300 K) matched-eta sweep -- appendix tab:cons_ratio, produced
# by ``phonon.studies.conservation.ratio_eta``. (eta THz, G_anh/G_ball, lead
# balance). The 241-pt entry is the finer-grid check at eta=0.45.
RATIO_ETA = [(0.30, 0.793, 5.1e-6), (0.45, 0.838, 6.9e-6),
             (0.60, 0.872, 7.3e-6), (0.90, 0.920, 3.2e-6)]
RATIO_ETA_241 = (0.45, 0.805, 3.3e-6)
RATIO_ETA0_DIRECT = 0.574   # direct eta=1e-12 production (summary.json, L2/300 K)


def fig_conservation_iter():
    """F1: the three convergence/conservation measures across SCBA iterations."""
    d = np.load(PROD / "L2_anh.npz", allow_pickle=True)
    ih = d["iter_heat"]                       # (n_it, 3) = [J_L, J_dev, J_R]
    bub = d["iter_bubble_balance"][:, 2]      # relative |P_in - P_out|
    sig = d["iter_sigma_max"]                 # (n_it, nfreq) |Sigma^R| per omega
    JL, JR = ih[:, 0], ih[:, 2]
    imbal = np.abs(JL - JR) / (0.5 * (np.abs(JL) + np.abs(JR)) + 1e-300)
    # relative Sigma^R change between consecutive iterates (residual proxy)
    res = (np.linalg.norm(np.diff(sig, axis=0), axis=1)
           / (np.linalg.norm(sig[:-1], axis=1) + 1e-300))
    nit = ih.shape[0]

    fig, axes = style.figure(ncols=1, width=5.0, height=3.6)
    ax = axes[0] if hasattr(axes, "__len__") else axes
    ax.semilogy(np.arange(2, nit + 1), np.maximum(res, 1e-18), "-", color="C0",
                lw=1.4, label=r"rel. $\Sigma^R$ change")
    ax.semilogy(np.arange(1, nit + 1), np.maximum(imbal, 1e-18), "-", color="C3",
                lw=1.3, label=r"lead imbalance $|J_L-J_R|/\bar J$")
    ax.semilogy(np.arange(1, nit + 1), np.maximum(bub, 1e-19), "-", color="C2",
                lw=1.1, label=r"bubble balance $|P_{\rm in}-P_{\rm out}|$")
    ax.axhline(np.median(bub), color="C2", ls=":", lw=0.7)
    ax.set_xlabel("SCBA iteration")
    ax.set_ylabel("convergence / conservation measure")
    ax.set_ylim(1e-18, 5)
    ax.legend(fontsize=7.5, loc="center right")
    style.save(fig, "eta0_cnt33_conservation_iter", directory=FIGDIR)

    print("\n[F1 conservation vs iteration] n_iter={}  final: rel-resid={:.2e}  "
          "lead imbalance={:.2e}  bubble balance(median)={:.2e}".format(
              nit, res[-1], imbal[-1], float(np.median(bub))))


def _fit_intercepts(eta, ratio):
    eta = np.asarray(eta, float); ratio = np.asarray(ratio, float)
    lin = np.polyfit(eta, ratio, 1)            # ratio = a + b eta
    quad = np.polyfit(eta, ratio, 2)           # ratio = a + b eta + c eta^2
    return float(np.polyval(lin, 0.0)), float(np.polyval(quad, 0.0)), lin, quad


def fig_ratio_eta():
    """F2/F3: G_anh/G_ball and lead balance vs eta (and grid), with eta->0."""
    eta = [e for e, _, _ in RATIO_ETA]
    rat = [r for _, r, _ in RATIO_ETA]
    bal = [b for _, _, b in RATIO_ETA]
    lin0, quad0, lin, quad = _fit_intercepts(eta, rat)

    fig, axes = style.figure(ncols=2, width=4.4, height=3.4)
    a = axes[0]
    a.plot(eta, rat, "o", ms=7, color="C0", label=r"matched $\eta$ (181 pt)")
    a.plot(*RATIO_ETA_241[:2], "D", ms=7, mfc="none", color="C1",
           label=r"$\eta{=}0.45$ (241 pt)")
    x = np.linspace(0, 0.95, 60)
    a.plot(x, np.polyval(lin, x), "k--", lw=1.0,
           label=rf"linear $\to{lin0:.2f}$")
    a.plot(x, np.polyval(quad, x), "k:", lw=1.0,
           label=rf"quadratic $\to{quad0:.2f}$")
    a.plot(0, lin0, "k*", ms=12)
    a.plot(0, RATIO_ETA0_DIRECT, "v", ms=8, color="C3",
           label=rf"direct $\eta{{=}}10^{{-12}}$ ({RATIO_ETA0_DIRECT:.2f})")
    a.set_xlabel(r"broadening $\eta$ (THz)")
    a.set_ylabel(r"$G_\mathrm{anh}/G_\mathrm{ball}$")
    a.set_xlim(-0.05, 0.98); a.legend(fontsize=6.5, loc="lower right")

    a = axes[1]
    a.semilogy(eta, bal, "o-", ms=7, color="C0", label="181 pt")
    a.semilogy(*RATIO_ETA_241[:1], RATIO_ETA_241[2], "D", ms=7, mfc="none",
               color="C1", label="241 pt")
    a.set_xlabel(r"broadening $\eta$ (THz)")
    a.set_ylabel(r"lead balance $|J_L-J_R|/\bar J$")
    a.set_ylim(1e-6, 2e-5); a.legend(fontsize=7, loc="upper right")
    style.save(fig, "eta0_cnt33_ratio_eta", directory=FIGDIR)

    print("\n[F2 ratio vs eta]  data eta/ratio={}  eta->0: linear={:.3f}  "
          "quadratic={:.3f}  direct(1e-12)={:.3f}".format(
              list(zip(eta, rat)), lin0, quad0, RATIO_ETA0_DIRECT))
    print("[F3 grid] cnt33 ratio 181pt={:.3f} -> 241pt={:.3f} (eta=0.45); "
          "lead balance flat at ~{:.0e}".format(
              RATIO_ETA[1][1], RATIO_ETA_241[1], np.mean(bal)))


def fig_conservation_T():
    """F4: lead conservation and SCBA iteration count vs temperature."""
    rows = [r for r in json.load(open(PROD / "summary.json"))
            if r.get("sweep") == "temperature" and r.get("anh_converged")]
    rows.sort(key=lambda r: r["t_mean"])
    T = [r["t_mean"] for r in rows]
    cons = [r["lead_conservation"] for r in rows]
    nit = [r["anh_n_iter"] for r in rows]

    fig, axes = style.figure(ncols=1, width=5.0, height=3.6)
    ax = axes[0] if hasattr(axes, "__len__") else axes
    ax.semilogy(T, cons, "o-", color="C0", ms=6,
                label=r"lead balance $|J_L-J_R|/\bar J$")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"lead conservation", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    ax2.plot(T, nit, "s--", color="C3", ms=6, label="SCBA iterations")
    ax2.set_ylabel("SCBA iterations to converge", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.grid(False)
    style.save(fig, "eta0_cnt33_conservation_T", directory=FIGDIR)

    print("\n[F4 conservation vs T]  T/lead-balance/n_iter:",
          [(t, f"{c:.1e}", n) for t, c, n in zip(T, cons, nit)])


def fig_bubble_replica():
    """F5: the discrete bubble energy balance is exactly conserving."""
    from phonon.studies import conservation as cons
    rep = cons.replica_check()
    labels = [r"float64", r"float128"]
    resid = [rep["complex128"]["resid"], rep["clongdouble"]["resid"]]

    fig, axes = style.figure(ncols=1, width=4.4, height=3.4)
    ax = axes[0] if hasattr(axes, "__len__") else axes
    x = np.arange(len(labels))
    ax.bar(x, resid, width=0.5, color=["C0", "C2"])
    for xi, r in zip(x, resid):
        ax.annotate(f"{r:.1e}", (xi, r), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8)
    ax.axhline(1e-6, color="C3", ls="--", lw=0.9)
    ax.annotate(r"production accumulation floor $\sim10^{-6}$", (len(labels) - 1, 1e-6),
                textcoords="offset points", xytext=(-4, -11), ha="right",
                fontsize=7, color="C3")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"$|P_{\rm in}-P_{\rm out}|/|P|$ (replica)")
    ax.set_ylim(1e-20, 3e-6)
    style.save(fig, "conservation_bubble_replica", directory=FIGDIR)

    print("\n[F5 bubble replica]  float64 resid={:.2e}  float128 resid={:.2e}  "
          "(scales with precision -> exactly conserving)".format(*resid))


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("=" * 66 + "\nCONSERVATION FIGURES (verified numbers)\n" + "=" * 66)
    fig_conservation_iter()
    fig_ratio_eta()
    fig_conservation_T()
    fig_bubble_replica()
    print("\nfigures ->", FIGDIR)
