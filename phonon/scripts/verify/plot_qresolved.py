"""Figures for the q-resolved anharmonic transport work (F23):
  (1) distributed-q self-energy scaling (the scalable periodic axis) vs the flat energy axis;
  (2) Si thin-film cross-plane conductance/kappa vs thickness, with Guo-Bescond-Zhang targets.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, "/usr/scratch/mont-fort11/pfischill/quatrex/phonon")
from finite_analysis.plot_style import (  # noqa: E402
    FIG_HALF, FIG_SINGLE, PALETTE, finalize, set_publication_style,
)

set_publication_style()
_W = Path("/usr/scratch/mont-fort11/pfischill/quatrex/phonon")

# ---- (1) distributed-q scaling (single panel, no title) ----
ranks = [1, 2, 4, 8]
wall = [5.300, 2.827, 1.479, 0.805]      # phph_q_dist_scaling.py, 4x4 mesh
speedup = [wall[0] / w for w in wall]
fig, ax = plt.subplots(figsize=FIG_SINGLE)
ax.plot(ranks, speedup, "o-", color=PALETTE["blue"],
        label="distributed external-$q$ (all-gather internal $G$)")
ax.plot(ranks, ranks, "k:", lw=1.0, label=r"ideal $P\times$")
ax.plot(ranks, [1.0, 0.97, 0.89, 0.85], "s--", color=PALETTE["red"],
        label="energy/stack axis (replicated)")
ax.set_xlabel("MPI ranks on the $q$-communicator")
ax.set_ylabel("self-energy speed-up")
ax.set_xticks(ranks)
ax.legend(fontsize=8.5)
finalize(fig, "phph_q_scaling.pdf")
print(f"q-scaling speed-ups: {[round(s,2) for s in speedup]} (82% at 8 ranks)")

# ---- (2a) Si film conductance vs thickness (native self-energy only; no div4) ----
# Guo, Bescond & Zhang 2020, Sec. II B (anharmonic NEGF conductance, MW/m^2/K)
guo = {1.62: 939.72, 2.70: 890.97}   # 3uc, 5uc  (1uc=5.4018 A)
jfull = _W / "scripts/out/si_film/si_film_kappa_nk8_guo.json"
fig, ax1 = plt.subplots(figsize=FIG_HALF)
rows = json.load(open(jfull))["rows"]
L = [r["L_nm"] for r in rows]
ax1.plot(L, [r["G_ball"] / 1e6 for r in rows], "o-", color=PALETTE["blue"],
         label="ballistic (ours)")
ax1.plot(L, [r["G_anh"] / 1e6 for r in rows], "s-", color=PALETTE["red"],
         label="anharmonic NEGF (ours)")
ax1.plot(list(guo), list(guo.values()), "k*", ms=13, label="Guo et al.\\ 2020")
ax1.set_xlabel("film thickness (nm)")
ax1.set_ylabel(r"$G\ (\mathrm{MW\,m^{-2}\,K^{-1}})$")
ax1.legend(fontsize=8.5)
finalize(fig, "si_film_conductance_a.pdf")
print("\nsi_film native: L(nm)  G_ball  G_anh (MW/m2K)")
for r in rows:
    print(f"  {r['n_slabs']:2d}  {r['L_nm']:.2f}  {r['G_ball']/1e6:7.1f} {r['G_anh']/1e6:7.1f}")

# ---- (2b) ballistic kappa = G*L linear in L (ballistic regime) ----
bpath = _W / "scripts/out/si_film/si_film_ballistic_lconv.json"
fig, ax2 = plt.subplots(figsize=FIG_HALF)
b = json.load(open(bpath))
L = [r["L_nm"] for r in b["rows"]]
kap = [r["kappa_ball"] for r in b["rows"]]
ax2.plot(L, kap, "o-", color=PALETTE["blue"], label=r"$\kappa_\mathrm{ball}=G_\mathrm{ball}L$ (ours)")
ax2.set_xlabel("film thickness (nm)")
ax2.set_ylabel(r"$\kappa\ (\mathrm{W\,m^{-1}\,K^{-1}})$")
ax2.legend(fontsize=8.5)
finalize(fig, "si_film_conductance_b.pdf")
print("[done] wrote phph_q_scaling.pdf, si_film_conductance_{a,b}.pdf")
