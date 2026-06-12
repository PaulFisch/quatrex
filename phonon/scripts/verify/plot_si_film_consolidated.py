"""Consolidated Si thin-film cross-plane conductance vs thickness:
our 2x2x2-FC and 5x5x5-FC NEGF (ballistic + anharmonic) against
Guo, Bescond & Zhang 2020. Single figure, data-driven (reads the json
outputs), so it can be regenerated as the runs complete.

Guo values are taken from the paper full text (PRB 102, 195412, 2020):
anharmonic conductance 939.72 (3uc) / 890.97 (5uc) MW/m^2K (present
approximation; 1uc=5.4018 A). Guo's ballistic is a BTE dashed line
(kappa_eff=G_ball*d) and is not quoted as a single number, so only
their anharmonic points are plotted.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/usr/scratch/mont-fort11/pfischill/quatrex/phonon/scripts/out/si_film")
FIG = Path("/usr/scratch/mont-fort11/pfischill/quatrex/document/fig/transport_sweeps")
FIG.mkdir(parents=True, exist_ok=True)


def load_rows(name):
    p = OUT / name
    return json.load(open(p))["rows"] if p.exists() else []


# 5x5x5 (long-range VASP-hiPhive) -- ballistic + anharmonic
big = load_rows("si_film_kappa_bigfc3.json")
# 2x2x2 (QE-FD): matched ballistic curve (3 thicknesses) + the single
# CONVERGED anharmonic point (the short-cutoff FC3 diverges the small-eta
# SCBA at other thicknesses; the long-range 5x5x5 FC3 is more robust).
two_ball = load_rows("si_film_2x2x2_ballistic.json")
# All CONVERGED 2x2x2 anharmonic points: the n_slabs=4 reference point plus
# any thicknesses the (re)run converges (filter the diverged garbage --
# the short-cutoff FC3 destabilises the small-eta SCBA at some thicknesses).
_anh = load_rows("si_film_kappa_nk8_guo.json") + load_rows("si_film_kappa_2x2x2_curve.json")
two_anh = sorted({round(r["L_nm"], 2): r for r in _anh
                  if r.get("G_anh", 0) > 0
                  and r.get("conservation", 1.0) < 0.05}.values(),
                 key=lambda r: r["L_nm"])
# Guo 2020 anharmonic (verified from full text)
guo = {1.62: 939.72, 2.70: 890.97}   # 3uc, 5uc

fig, ax = plt.subplots(figsize=(6.2, 4.6))

# 2x2x2
if two_ball:
    L = [r["L_nm"] for r in two_ball]
    ax.plot(L, [r["G_ball"] / 1e6 for r in two_ball], "o-", color="C0",
            label=r"2$\times$2$\times$2 FC, ballistic")
if two_anh:
    ax.plot([r["L_nm"] for r in two_anh], [r["G_anh"] / 1e6 for r in two_anh],
            "s--", color="C0", ms=8, label=r"2$\times$2$\times$2 FC, anharmonic")
two = two_anh

# 5x5x5
if big:
    L = [r["L_nm"] for r in big]
    ax.plot(L, [r["G_ball"] / 1e6 for r in big], "o-", color="C3", label=r"5$\times$5$\times$5 FC, ballistic")
    ax.plot(L, [r["G_anh"] / 1e6 for r in big], "s--", color="C3", label=r"5$\times$5$\times$5 FC, anharmonic")

# Guo anharmonic
ax.plot(list(guo), list(guo.values()), "k*", ms=15, label="Guo et al.\\ 2020 (anharmonic)")

ax.set_xlabel("film thickness $L$ (nm)")
ax.set_ylabel(r"cross-plane conductance $G$ (MW m$^{-2}$K$^{-1}$)")
ax.set_title("Si thin film, cross-plane conductance vs. Guo 2020", fontsize=10)
ax.set_ylim(0, 1150)
ax.legend(fontsize=8, ncol=1, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "si_film_vs_guo.pdf")
print("wrote", FIG / "si_film_vs_guo.pdf")
print("\n  thickness(nm)  2x2x2_ball  2x2x2_anh  5x5x5_ball  5x5x5_anh   Guo_anh")
for src, lab in [(two or two_ball, "2x2x2"), (big, "5x5x5")]:
    for r in src:
        ga = r.get("G_anh")
        print(f"  {lab} {r['L_nm']:.2f}: G_ball={r['G_ball']/1e6:6.1f}"
              + (f"  G_anh={ga/1e6:6.1f}" if ga else "  (ball only)"))
print("  Guo:", {k: v for k, v in guo.items()})
