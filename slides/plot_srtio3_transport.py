"""SrTiO3 cross-plane transport @300 K: absolute conductances (de-confounded).

The two routes use different effective FC2, so G_ball itself shifts (1053 vs
1179 MW/m2K) -- a ratio G_anh/G_ball would mix that FC2 change into the
"reduction". Here we show the absolute G_ball (tick) and G_anh (dot) per route;
the scattering is the G_ball -> G_anh drop, and the % label is the clean,
same-FC2 reduction.

  - low-disp FC, bubble only:     G_ball 1053, G_anh 1059  (no reduction)
  - low-disp FC, +loop+tadpole:   G_ball 1053, G_anh 1037  (loop+tadpole needed)
  - high-disp eff. FC, bubble:    G_ball 1179, G_anh 1147
  src: cluster/tortin3-tmp/srtio3_scp_T300 (low-disp) + cluster/prod/srtio3 (high-disp)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
C = {"bub": "#CCBB44", "scp": "#AA3377", "ball": "#555555"}


def main():
    scp = json.load(open(REPO / "cluster/tortin3-tmp/srtio3_scp_T300/summary.json"))
    gb_lo = scp["G_ball"] / 1e6
    ga_bub = scp["G_anh_baseline"] / 1e6
    ga_scp = scp["G_anh_scp"] / 1e6
    prod = {r["tag"]: r for r in csv.DictReader(
        open(REPO / "cluster/prod/srtio3/summary.csv"))}["T300"]
    gb_hi = float(prod["G_ball_W_per_m2_K"]) / 1e6
    ga_hi = float(prod["G_anh_W_per_m2_K"]) / 1e6

    routes = [
        ("low-disp\nbubble only", gb_lo, ga_bub, C["bub"]),
        ("low-disp\n+loop+tadpole", gb_lo, ga_scp, C["scp"]),
        ("high-disp\nbubble only", gb_hi, ga_hi, C["bub"]),
    ]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for i, (lbl, gb, ga, col) in enumerate(routes):
        ax.plot([i, i], [gb, ga], color="0.65", lw=2.0, zorder=1)
        ax.plot(i, gb, marker="_", color=C["ball"], ms=26, mew=2.6, zorder=2,
                label=r"$G_{\rm ball}$" if i == 0 else None)
        ax.plot(i, ga, "o", color=col, ms=13, markeredgecolor="k",
                markeredgewidth=0.5, zorder=3,
                label=r"$G_{\rm anh}$" if i == 0 else None)
        ax.annotate(f"{gb:.0f}", (i, gb), textcoords="offset points",
                    xytext=(16, 0), va="center", fontsize=8, color=C["ball"])
        pct = (ga / gb - 1) * 100
        ax.annotate(f"{pct:+.1f}%", (i, ga), textcoords="offset points",
                    xytext=(16, -2), va="center", fontsize=9, fontweight="bold",
                    color=col)

    ax.set_xticks(range(len(routes)))
    ax.set_xticklabels([r[0] for r in routes], fontsize=8.5)
    ax.set_xlim(-0.5, 2.7)
    ax.set_ylim(1010, 1200)
    ax.set_ylabel(r"thermal conductance @300 K [MW m$^{-2}$K$^{-1}$]")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, handletextpad=0.3)
    fig.tight_layout()
    out = HERE / "fig" / "srtio3_transport.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=130)
    print("wrote", out, "| Gball lo/hi:", round(gb_lo), round(gb_hi),
          "| %:", round((ga_bub/gb_lo-1)*100, 1), round((ga_scp/gb_lo-1)*100, 1),
          round((ga_hi/gb_hi-1)*100, 1))


if __name__ == "__main__":
    main()
