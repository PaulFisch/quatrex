"""Si thin film (cross-plane, coupled-q_perp), production data (cluster/si_film).

Single frame: thermal conductance vs thickness at 300 K. vertex_scale = 1.0
(the correct prefactor). Ballistic and anharmonic, for the 2^3 and the large
(converged) supercell force constants; Guo et al. (2020) actual data points
overlaid at their thicknesses (3uc, 5uc; 1uc = 5.4018 A) at 300 K.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SF = HERE.parents[1] / "cluster" / "si_film"
C = {"sc2": "#CCBB44", "big": "#AA3377", "guo": "#4477AA", "grey": "#999999"}

# Guo, Bescond et al., PRB 102, 195412 (2020), Si film cross-plane @300 K
# (anharmonic NEGF, "present approximation"); 1uc = 5.4018 A.
GUO = [(3 * 0.54018, 939.72), (5 * 0.54018, 890.97)]


def rows(name):
    return json.load(open(SF / f"{name}.json"))["rows"]


def main():
    r2 = rows("si_film_2x2x2_ballistic")          # 2^3 ballistic
    rb = rows("si_film_kappa_bigfc3")             # large SC: G_ball + G_anh
    g2 = rows("si_film_kappa_nk8_guo")[0]         # 2^3 anharmonic (L=1.55 nm)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    ax.plot([x["L_nm"] for x in r2], [x["G_ball"] / 1e6 for x in r2], "D-",
            color=C["sc2"], label=r"ballistic, $2^3$ SC")
    ax.plot([x["L_nm"] for x in rb], [x["G_ball"] / 1e6 for x in rb], "o-",
            color=C["big"], label="ballistic, large SC")
    ax.plot([x["L_nm"] for x in rb], [x["G_anh"] / 1e6 for x in rb], "s--",
            color=C["big"], label="anharmonic, large SC")
    ax.plot([g2["L_nm"]], [g2["G_anh"] / 1e6], "D", color=C["sc2"], ms=9,
            markerfacecolor="white", markeredgewidth=1.6,
            label=r"anharmonic, $2^3$ SC")
    ax.plot([t for t, _ in GUO], [v for _, v in GUO], "*", color=C["guo"], ms=15,
            markeredgecolor="k", markeredgewidth=0.4,
            label="Guo 2020, anh. (300 K)")

    ax.set_xlabel("film thickness [nm]")
    ax.set_ylabel(r"thermal conductance [MW m$^{-2}$K$^{-1}$]")
    ax.set_ylim(250, 1100)
    ax.grid(alpha=0.25)
    ax.legend(loc="center right", fontsize=8.5, framealpha=0.95)

    fig.tight_layout()
    out = HERE / "fig" / "si_film_prod.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
