"""Overlay the d5a and d11a transport/cutoff sweeps on shared axes.

Reads the per-wire ``summary.csv`` files written by ``d5_transport_sweep.py``
and ``d5_cutoff_sweep.py`` and produces combined figures with both wires, so
the width dependence of the phonon transport observables is visible directly.

Usage:
  python phonon/scripts/verify/overlay_d5_d11.py transport <d5_dir> <d11_dir> <out_dir>
  python phonon/scripts/verify/overlay_d5_d11.py cutoff    <d5_dir> <d11_dir> <out_dir>
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KIND = sys.argv[1]
D5 = Path(sys.argv[2])
D11 = Path(sys.argv[3])
OUT = Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)

STYLE = {
    "d5a": dict(color="#1f77b4", marker="o", label="d5a (0.027 THz twist)"),
    "d11a": dict(color="#d62728", marker="s", label="d11a (no soft mode)"),
}


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def transport():
    data = {"d5a": load(D5 / "summary.csv"), "d11a": load(D11 / "summary.csv")}

    # G_anh and G_ball vs temperature (length-1 temperature sweep).
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for wire, rows in data.items():
        st = STYLE[wire]
        pts = sorted(
            [r for r in rows if r["sweep"] == "temperature"
             and not _truthy(r.get("ballistic_only"))],
            key=lambda r: fnum(r["t_mean"]),
        )
        if not pts:
            continue
        T = [fnum(r["t_mean"]) for r in pts]
        ax.plot(T, [fnum(r["G_anh_W_per_m2_K"]) for r in pts],
                ls="-", **st)
        ax.plot(T, [fnum(r["G_ball_W_per_m2_K"]) for r in pts],
                ls="--", color=st["color"], marker=st["marker"], alpha=0.45)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"thermal conductance (W m$^{-2}$ K$^{-1}$)")
    ax.set_title("Conductance vs T (solid: anharmonic, dashed: ballistic)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "conductance_vs_T_d5_d11.pdf")
    plt.close(fig)

    # G_anh/G_ball vs temperature -- the anharmonic reduction factor.
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for wire, rows in data.items():
        st = STYLE[wire]
        pts = sorted(
            [r for r in rows if r["sweep"] == "temperature"
             and not _truthy(r.get("ballistic_only"))],
            key=lambda r: fnum(r["t_mean"]),
        )
        if not pts:
            continue
        T = [fnum(r["t_mean"]) for r in pts]
        ratio = [fnum(r["G_anh_W_per_m2_K"]) / fnum(r["G_ball_W_per_m2_K"])
                 for r in pts]
        ax.plot(T, ratio, ls="-", **st)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"$G_\mathrm{anh}/G_\mathrm{ball}$")
    ax.set_title("Anharmonic reduction vs T")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ratio_vs_T_d5_d11.pdf")
    plt.close(fig)

    # G_anh vs length (length sweep at fixed T).
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for wire, rows in data.items():
        st = STYLE[wire]
        pts = sorted(
            [r for r in rows if r["sweep"] == "length"
             and not _truthy(r.get("ballistic_only"))],
            key=lambda r: fnum(r["n_slabs"]),
        )
        if not pts:
            continue
        L = [fnum(r["n_slabs"]) for r in pts]
        ax.plot(L, [fnum(r["G_anh_W_per_m2_K"]) for r in pts], ls="-", **st)
        ax.plot(L, [fnum(r["G_ball_W_per_m2_K"]) for r in pts],
                ls="--", color=st["color"], marker=st["marker"], alpha=0.45)
    ax.set_xlabel("device length (transport cells)")
    ax.set_ylabel(r"thermal conductance (W m$^{-2}$ K$^{-1}$)")
    ax.set_title("Conductance vs length (solid: anh, dashed: ball)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "conductance_vs_length_d5_d11.pdf")
    plt.close(fig)
    print(f"[overlay] wrote transport figures to {OUT}", flush=True)


def cutoff():
    data = {"d5a": load(D5 / "summary.csv"), "d11a": load(D11 / "summary.csv")}
    # G_anh vs vertex_cutoff with the other cutoffs at "Inf" (full).
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for wire, rows in data.items():
        st = STYLE[wire]
        pts = [r for r in rows if str(r.get("sigma_cutoff")) in ("Inf", "inf")
               and str(r.get("g_cutoff")) in ("Inf", "inf")]

        def vkey(r):
            v = str(r.get("vertex_cutoff"))
            return 1e9 if v in ("Inf", "inf") else fnum(v)
        pts = sorted(pts, key=vkey)
        if not pts:
            continue
        xs = list(range(len(pts)))
        labels = [str(r.get("vertex_cutoff")) for r in pts]
        ax.plot(xs, [fnum(r["G_anh"]) for r in pts], ls="-", **st)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
    ax.set_xlabel("vertex cutoff (transport cells; Inf = full)")
    ax.set_ylabel(r"$G_\mathrm{anh}$ (W m$^{-2}$ K$^{-1}$)")
    ax.set_title("Vertex-cutoff convergence of the anharmonic conductance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ganh_vs_vertex_cutoff_d5_d11.pdf")
    plt.close(fig)
    print(f"[overlay] wrote cutoff figures to {OUT}", flush=True)


def _truthy(x):
    return str(x).strip().lower() in ("true", "1", "1.0")


if KIND == "transport":
    transport()
elif KIND == "cutoff":
    cutoff()
else:
    raise SystemExit(f"unknown kind {KIND!r}")
