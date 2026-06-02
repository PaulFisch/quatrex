"""Figures for the q-resolved anharmonic transport work (F23):
  (1) distributed-q self-energy scaling (the scalable periodic axis) vs the flat energy axis;
  (2) Si thin-film cross-plane conductance/kappa vs thickness, with Guo-Bescond-Zhang targets.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_W = Path("/usr/scratch/mont-fort11/pfischill/quatrex/phonon")
OUTS = [_W / "scripts/out/si_film",
        Path("/usr/scratch/mont-fort11/pfischill/quatrex/document/fig/transport_sweeps")]
for o in OUTS:
    o.mkdir(parents=True, exist_ok=True)

# ---- (1) distributed-q scaling ----
ranks = [1, 2, 4, 8]
wall = [5.300, 2.827, 1.479, 0.805]      # phph_q_dist_scaling.py, 4x4 mesh
speedup = [wall[0] / w for w in wall]
fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.plot(ranks, speedup, "o-", color="#1f77b4", lw=1.8,
        label="distributed external-q (all-gather internal G)")
ax.plot(ranks, ranks, "k:", lw=1.0, label="ideal $P\\times$")
ax.plot(ranks, [1.0, 0.97, 0.89, 0.85], "s--", color="#d62728", lw=1.4,
        label="energy/stack axis (replicated, F22)")
ax.set_xlabel("MPI ranks on the q-communicator")
ax.set_ylabel("self-energy speed-up")
ax.set_title("Distributed q-resolved 3-phonon self-energy\n(coupled in q by momentum conservation)")
ax.set_xticks(ranks); ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
fig.tight_layout()
for o in OUTS:
    fig.savefig(o / "phph_q_scaling.pdf")
plt.close(fig)
print(f"q-scaling speed-ups: {[round(s,2) for s in speedup]} (82% at 8 ranks)")

# ---- (2) Si film conductance vs thickness ----
# Guo, Bescond & Zhang 2020, Sec. II B (anharmonic NEGF conductance, MW/m^2/K)
guo = {1.62: 939.72, 2.70: 890.97}   # 3uc, 5uc  (1uc=5.4018 A)
jfull = _W / "scripts/out/si_film/si_film_kappa_nk8_guo.json"
jnn = _W / "scripts/out/si_film/si_film_kappa_nk8_nn.json"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
plotted_ball = False
jdiv4 = _W / "scripts/out/si_film/si_film_kappa_nk8_div4.json"
for jpath, lab, col in [(jfull, "anh. NEGF, native self-energy (ours)", "#d62728"),
                        (jdiv4, "anh. NEGF, vertex x0.5 (prefactor unresolved)", "#2ca02c")]:
    if not jpath.exists():
        print(f"[warn] {jpath.name} not found yet")
        continue
    rows = json.load(open(jpath))["rows"]
    L = [r["L_nm"] for r in rows]
    if not plotted_ball:
        ax1.plot(L, [r["G_ball"] / 1e6 for r in rows], "o-", color="#1f77b4",
                 label="ballistic (ours)")
        plotted_ball = True
    ax1.plot(L, [r["G_anh"] / 1e6 for r in rows], "s-", color=col, label=lab)
    print(f"\n{jpath.name}: thickness L(nm)  G_ball  G_anh (MW/m2K)  kappa_anh(W/mK)")
    for r in rows:
        print(f"  {r['n_slabs']:2d}  {r['L_nm']:.2f}  {r['G_ball']/1e6:7.1f} {r['G_anh']/1e6:7.1f}"
              f"   {r['kappa_anh']:.2f}")
ax1.plot(list(guo), list(guo.values()), "k*", ms=14,
         label="Guo et al. 2020 (anh. NEGF)")
ax1.set_xlabel("film thickness (nm)"); ax1.set_ylabel("conductance G (MW/m$^2$/K)")
ax1.set_title("Cross-plane conductance vs thickness"); ax1.legend(fontsize=8.5); ax1.grid(alpha=0.3)

# ballistic length scan (kappa = G*L linear in L confirms ballistic regime)
bpath = _W / "scripts/out/si_film/si_film_ballistic_lconv.json"
if bpath.exists():
    b = json.load(open(bpath))
    L = [r["L_nm"] for r in b["rows"]]; kap = [r["kappa_ball"] for r in b["rows"]]
    ax2.plot(L, kap, "o-", color="#1f77b4", label="$\\kappa_{ball}=G_{ball}L$ (ours)")
    ax2.set_xlabel("film thickness (nm)"); ax2.set_ylabel("$\\kappa$ (W/m/K)")
    ax2.set_title("Ballistic $\\kappa$ rises ~linearly in L"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
fig.tight_layout()
for o in OUTS:
    fig.savefig(o / "si_film_conductance.pdf")
plt.close(fig)
print("[done] wrote phph_q_scaling.pdf, si_film_conductance.pdf")
