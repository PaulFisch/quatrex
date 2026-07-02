"""Unified d5a eta=0 marginal-fixed-point figure (results/60_eta0).

Replaces the ten orphaned sinw_d5a_eta0_* diagnostics with one three-panel
figure, built ONLY from the saved conv1e10 runs:

  (a) convergence + conservation of the floor-stabilised JFNK family
      (trust-grow bold; warm/t005/ptc01 thin): rel Sigma^R residual floors at
      ~0.016-0.032 (marginal, NOT 1e-10), lead balance ~2.5e-3-1.7e-2, bubble
      balance pinned at machine precision.
  (b) the OBSERVABLE is method-invariant despite the floor: per-iteration lead
      heat current G(it) of the four runs + the independent repro run collapse
      to 2.613-2.640 (1%); the coarser irsub variant reaches 2.70 -> the
      marginal floor limits the absolute G to a ~5% envelope around ~2.65.
  (c) the broadening floor is a removable crutch: annealing Gamma_floor
      2 -> 0 cells over 80 iterations, G moves <5% after warm-up and the
      fixed point survives at Gamma_floor = 0.

Run:  OMP_NUM_THREADS=1 python phonon/scripts/figures/sinw_d5a_eta0_panel.py
Figure -> document/fig/transport_sweeps/sinw_d5a_eta0_panel.{pdf,png}
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style

CONV = ROOT / "phonon/studies/out/conv1e10"
FIGDIR = ROOT / "document/fig/transport_sweeps"

RUNS = [  # (tag, label, bold)
    ("jfnk_tgrow", "JFNK trust-grow", True),
    ("jfnk_warm", "JFNK warm-start", False),
    ("jfnk_t005", "JFNK trust 0.05", False),
    ("jfnk_ptc01", "JFNK PTC", False),
]
REPRO = ("jfnk_tgrow_repro", "trust-grow (repro)")

_PR = re.compile(r"rel Sigma\^R residual ([0-9.eE+-]+); lead balance ([0-9.eE+-]+)")
_PB = re.compile(r"Bubble energy balance: .*resid=([0-9.eE+-]+)")


def trace(tag):
    res, lead, bub = [], [], []
    for ln in (CONV / f"sinw_d5a_L2_eta0_{tag}.log").read_text(errors="ignore").splitlines():
        m = _PR.search(ln)
        if m:
            res.append(float(m.group(1))); lead.append(float(m.group(2)))
        m = _PB.search(ln)
        if m:
            bub.append(float(m.group(1)))
    return map(np.array, (res, lead, bub))


def g_iter(tag):
    """Per-iteration lead heat current on the physical scale of lead_current."""
    z = np.load(CONV / f"sinw_d5a_L2_eta0_{tag}.npz", allow_pickle=True)
    jl = np.abs(z["iter_heat"][:, 0])
    scale = float(np.atleast_1d(z["lead_current"])[0]) / jl[-1]
    return jl * scale


def main():
    fig, axes = style.figure(ncols=3, width=3.35, height=3.0)

    # (a) convergence + conservation
    ax = axes[0]
    for tag, label, bold in RUNS:
        res, lead, bub = trace(tag)
        it = np.arange(1, res.size + 1)
        ax.semilogy(it, res, "-", color="C0", lw=1.6 if bold else 0.8,
                    alpha=1.0 if bold else 0.45,
                    label=r"rel $\Sigma^R$ residual" if bold else None)
        if bold:
            ax.semilogy(it, lead, "-", color="C3", lw=1.2, label="lead balance")
            ax.semilogy(np.arange(1, bub.size + 1), bub, "-", color="C2",
                        lw=1.0, label="bubble balance")
    ax.axhline(0.016, color="C0", ls=":", lw=0.8)
    ax.annotate("marginal floor 0.016--0.032", (0.03, 0.040),
                xycoords=("axes fraction", "data"), fontsize=6.5, color="C0")
    ax.set_xlabel("SCBA iteration")
    ax.set_ylabel("convergence measure")
    ax.set_ylim(1e-18, 30)
    ax.legend(fontsize=6, loc="center right")
    ax.set_title("(a) floored JFNK family", fontsize=8)

    # (b) method invariance of the observable
    ax = axes[1]
    finals = []
    for tag, label, bold in RUNS:
        g = g_iter(tag)
        it = np.arange(1, g.size + 1)
        ax.plot(it, g, "-", lw=1.5 if bold else 1.0, label=label)
        finals.append(g[-1])
    g = g_iter(REPRO[0])
    ax.plot(np.arange(1, g.size + 1), g, "--", lw=1.0, color="0.4",
            label=REPRO[1])
    finals.append(g[-1])
    lo, hi = min(finals), max(finals)
    ax.axhspan(lo, hi, color="C0", alpha=0.12)
    ax.annotate(rf"$G={lo:.2f}$--${hi:.2f}$ ($\approx1\%$)",
                (0.97, hi + 0.06), xycoords=("axes fraction", "data"),
                fontsize=7, ha="right")
    ax.set_xlabel("SCBA iteration")
    ax.set_ylabel(r"lead heat current $G$ (W\,m$^{-2}$K$^{-1}$)"
                  if False else "lead heat current (arb. norm)")
    ax.set_ylim(2.2, 3.4)
    ax.set_xlim(40, 260)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title("(b) observable, method-invariant", fontsize=8)

    # (c) floor anneal: the crutch is removable
    ax = axes[2]
    log = (CONV / "sinw_d5a_L2_eta0_floor_anneal.log").read_text(errors="ignore")
    fl = np.full(100, np.nan)
    for m in re.finditer(r"eta IR floor ramp: it=(\d+) floor_cells=([0-9.]+)", log):
        i = int(m.group(1))
        if i < fl.size:
            fl[i] = float(m.group(2))
    g = g_iter("floor_anneal")
    it = np.arange(1, g.size + 1)
    ax.plot(it, g, "-", color="C0", lw=1.4, label="lead heat current")
    ax.set_xlabel("SCBA iteration")
    ax.set_ylabel("lead heat current (arb. norm)")
    ax.set_ylim(0, 3.4)
    ax2 = ax.twinx()
    ax2.plot(it[: fl.size], fl[: g.size], "-", color="C1", lw=1.1,
             label=r"$\Gamma_{\rm floor}$ (cells)")
    ax2.set_ylabel(r"$\eta$-floor (cells of $d\omega$)", color="C1")
    ax2.tick_params(axis="y", colors="C1")
    ax2.set_ylim(0, 2.2)
    ax.annotate(r"$G$ drifts $<5\%$ as $\Gamma_{\rm floor}\to0$;"
                "\nfixed point survives at zero floor",
                (0.5, 0.30), xycoords="axes fraction", fontsize=6.5,
                ha="center")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower right")
    ax.set_title("(c) floor anneal $2\\to0$", fontsize=8)

    style.save(fig, "sinw_d5a_eta0_panel", directory=FIGDIR)

    print("finals:", np.round(finals, 3), " anneal final:", round(float(g[-1]), 3))


if __name__ == "__main__":
    main()
