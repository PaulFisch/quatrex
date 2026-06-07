"""Build the new single-panel report figures from cached run data.

All inputs are CSVs already on disk (no SCBA/DFT reruns). Outputs go to the
LaTeX figure tree via the shared publication style. Panels are exported
separately; the report groups them with ``subfigure``.

Figures produced:
  * d11a_decomp_ganh.pdf / d11a_decomp_conservation.pdf
        FC3 tensor-decomposition transport quality on the d11a nanowire
        (method x rank): anharmonic conductance and heat-flow conservation
        versus parameter count, against the dense reference.
  * cnt33_temperature_g.pdf / cnt33_temperature_ratio.pdf
        Converged (3,3) CNT anharmonic transport versus temperature.
  * cnt33_cutoff.pdf
        (3,3) CNT self-energy cutoff robustness across the eight
        (sigma, vertex, g) corners.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # phonon/ on path
from finite_analysis.plot_style import (  # noqa: E402
    FIG_HALF,
    FIG_SINGLE,
    METHOD_COLORS,
    REPO,
    finalize,
    set_publication_style,
)
import matplotlib.pyplot as plt  # noqa: E402

OUT = REPO / "phonon/scripts/out"
set_publication_style()
MW = 1.0e6  # W/m^2/K -> MW/m^2/K


def _rows(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# 1. d11a FC3-decomposition transport quality
# ---------------------------------------------------------------------------
def decomposition_figures() -> None:
    rows = _rows(REPO / "phonon/configs/sinw/reaps/sinw100_d11a_vasp_sc4/transport_quality/transport_quality.csv")
    dense = next(r for r in rows if r["method"] == "dense")
    g_dense = float(dense["G_anh_W_per_m2_K"]) / MW
    g_ball = float(dense["G_ball_W_per_m2_K"]) / MW
    methods = ["mSVD", "HOSVD", "CP", "INDSCAL", "Waring"]

    # Panel A: anharmonic conductance vs parameter count.
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.axhline(g_dense, color="k", lw=1.3, label="dense FC3 ($G_\\mathrm{anh}$)")
    ax.axhline(g_ball, color="0.55", lw=1.0, ls=":", label="ballistic ($G_\\mathrm{ball}$)")
    for m in methods:
        mr = [r for r in rows if r["method"] == m]
        x = [float(r["n_params"]) for r in mr]
        y = [float(r["G_anh_W_per_m2_K"]) / MW for r in mr]
        collapse = [r["ballistic_collapse"] == "True" for r in mr]
        unphysical = [float(r["conservation_err"]) > 0.5 for r in mr]
        c = METHOD_COLORS[m]
        ax.plot(x, y, "-", color=c, alpha=0.6, zorder=1)
        for xi, yi, col, un in zip(x, y, collapse, unphysical):
            if un:
                ax.plot(xi, yi, "x", color=c, ms=8, mew=2, zorder=3)
            elif col:
                ax.plot(xi, yi, "o", mfc="white", mec=c, ms=7, zorder=3)
            else:
                ax.plot(xi, yi, "o", color=c, ms=6, zorder=3)
        ax.plot([], [], "o", color=c, label=m)  # legend proxy
    ax.set_xscale("log")
    ax.set_xlabel("number of FC3 parameters")
    ax.set_ylabel(r"$G_\mathrm{anh}\ (\mathrm{MW\,m^{-2}\,K^{-1}})$")
    ax.set_ylim(-30, 200)
    ax.legend(ncol=2, fontsize=8)
    finalize(fig, "d11a_decomp_ganh.pdf")

    # Panel B: heat-flow conservation residual vs parameter count.
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.axhline(float(dense["conservation_err"]), color="k", lw=1.3, label="dense FC3")
    for m in methods:
        mr = [r for r in rows if r["method"] == m]
        x = [float(r["n_params"]) for r in mr]
        y = [max(float(r["conservation_err"]), 1e-9) for r in mr]
        ax.plot(x, y, "o-", color=METHOD_COLORS[m], alpha=0.85, label=m)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of FC3 parameters")
    ax.set_ylabel("heat-flow conservation residual")
    ax.legend(ncol=2, fontsize=8)
    finalize(fig, "d11a_decomp_conservation.pdf")


# ---------------------------------------------------------------------------
# 2. CNT(3,3) converged anharmonic transport vs temperature
# ---------------------------------------------------------------------------
def cnt_temperature_figures() -> None:
    rows = _rows(OUT / "cnt33_tempsweep/summary.csv")
    rows.sort(key=lambda r: float(r["t_mean"]))
    T = np.array([float(r["t_mean"]) for r in rows])
    gb = np.array([float(r["G_ball_W_per_m2_K"]) for r in rows]) / MW
    ga = np.array([float(r["G_anh_W_per_m2_K"]) for r in rows]) / MW

    fig, ax = plt.subplots(figsize=FIG_HALF)
    ax.plot(T, gb, "o-", color="0.55", label=r"$G_\mathrm{ball}$")
    ax.plot(T, ga, "s-", color=METHOD_COLORS["INDSCAL"], label=r"$G_\mathrm{anh}$")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"$G\ (\mathrm{MW\,m^{-2}\,K^{-1}})$")
    ax.legend()
    finalize(fig, "cnt33_temperature_g.pdf")

    fig, ax = plt.subplots(figsize=FIG_HALF)
    ax.plot(T, ga / gb, "D-", color=METHOD_COLORS["mSVD"])
    ax.axhline(1.0, color="0.6", lw=0.9, ls=":")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"$G_\mathrm{anh}/G_\mathrm{ball}$")
    ax.set_ylim(0.6, 1.02)
    finalize(fig, "cnt33_temperature_ratio.pdf")


# ---------------------------------------------------------------------------
# 3. CNT(3,3) cutoff robustness across the eight corners
# ---------------------------------------------------------------------------
def cnt_cutoff_figure() -> None:
    rows = _rows(OUT / "cnt33_cutoff/summary.csv")

    def lab(r: dict) -> str:
        def s(v: str) -> str:
            return r"$\infty$" if v.strip().lower().startswith("inf") else "0"
        return f"$\\sigma${s(r['sigma_cutoff'])} v{s(r['vertex_cutoff'])} g{s(r['g_cutoff'])}"

    labels = [lab(r) for r in rows]
    ga = np.array([float(r["G_anh"]) for r in rows]) / MW
    full = next(r for r in rows if r["sigma_cutoff"].lower().startswith("inf")
                and r["vertex_cutoff"].lower().startswith("inf")
                and r["g_cutoff"].lower().startswith("inf"))
    g_full = float(full["G_anh"]) / MW

    order = np.argsort(ga)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.barh(y, ga[order], color=METHOD_COLORS["HOSVD"], alpha=0.85)
    ax.axvline(g_full, color="k", lw=1.2, ls="--", label="full coupling")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order], fontsize=8)
    ax.set_xlabel(r"$G_\mathrm{anh}\ (\mathrm{MW\,m^{-2}\,K^{-1}})$")
    ax.legend()
    ax.grid(axis="y", visible=False)
    finalize(fig, "cnt33_cutoff.pdf")


if __name__ == "__main__":
    decomposition_figures()
    cnt_temperature_figures()
    cnt_cutoff_figure()
    print("new report figures written to", REPO / "document/fig/transport_sweeps")
