"""SrTiO3 eta=0 anharmonic transport figure (results/70_strong_srtio3).

Reads ONLY the saved summary of the June-2026 eta=0 production campaign
(phonon/scripts/out/prod/srtio3_eta0/summary.csv): temperature sweep at two
cells (T200/T300 converged, T600 NOT converged -> hollow marker) and the
length ladder at 300 K (L2/L3).

Run:  python phonon/scripts/figures/srtio3_eta0_transport.py
Figure -> document/fig/transport_sweeps/srtio3_eta0_transport.{pdf,png}
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style

SUMMARY = ROOT / "phonon/scripts/out/prod/srtio3_eta0/summary.csv"
FIGDIR = ROOT / "document/fig/transport_sweeps"


def main():
    rows = list(csv.DictReader(open(SUMMARY)))
    for r in rows:
        for k in ("t_mean", "n_slabs", "G_ball_W_per_m2_K", "G_anh", "ratio"):
            r[k] = float(r[k]) if r[k] else float("nan")
        r["conv"] = r["anh_converged"] == "True"

    tsw = sorted((r for r in rows if r["n_slabs"] == 2), key=lambda r: r["t_mean"])
    lsw = sorted((r for r in rows if r["t_mean"] == 300), key=lambda r: r["n_slabs"])

    fig, axes = style.figure(ncols=2, width=3.6, height=2.9)
    ax = axes[0]
    T = [r["t_mean"] for r in tsw]
    ax.plot(T, [r["G_ball_W_per_m2_K"] / 1e9 for r in tsw], "o-", label=r"$G_\mathrm{ball}$")
    for r in tsw:
        mk = dict(marker="o", color="C1") if r["conv"] else dict(
            marker="o", mfc="none", color="C1")
        ax.plot(r["t_mean"], r["G_anh"] / 1e9, ls="", **mk)
    ax.plot([r["t_mean"] for r in tsw if r["conv"]],
            [r["G_anh"] / 1e9 for r in tsw if r["conv"]], "-", color="C1",
            label=r"$G_\mathrm{anh}$")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"$G$ (GW\,m$^{-2}$\,K$^{-1}$)" if False else "G (GW m$^{-2}$ K$^{-1}$)")
    ax.legend()
    for r in tsw:
        dy = -0.12 if r["t_mean"] == 600 else 0.09
        ax.annotate(f"{r['ratio']:.3f}" + ("*" if not r["conv"] else ""),
                    (r["t_mean"], r["G_anh"] / 1e9 + dy), fontsize=7,
                    ha="center", color="C1")
    ax.set_title("two cells; * = not converged", fontsize=8)

    ax = axes[1]
    L = [int(r["n_slabs"]) for r in lsw]
    ax.plot(L, [r["ratio"] for r in lsw], "s-", color="C2")
    for r in lsw:
        ax.annotate(f"{r['ratio']:.3f}", (r["n_slabs"], r["ratio"] + 0.012),
                    fontsize=7, ha="center", color="C2")
    ax.set_xlabel("transport cells")
    ax.set_ylabel(r"$G_\mathrm{anh}/G_\mathrm{ball}$")
    ax.set_xticks(L)
    ax.set_ylim(0.55, 0.80)
    ax.set_title("300 K", fontsize=8)

    style.save(fig, "srtio3_eta0_transport", directory=FIGDIR)


if __name__ == "__main__":
    main()
