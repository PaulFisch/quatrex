#!/usr/bin/env python
"""CNT deep-dive result figures (F30/F30b/F31).

Reads the SCBA checkpoints from scripts/out/{cnt33_converge, cnt33_tempsweep,
d5a_tempsweep, cnt80_transport, cnt80_ballistic, cnt33_cutoff} and produces two
PDFs in document/fig/transport_sweeps/:
  cnt_transport.pdf   -- length ladder + (3,3)-vs-(8,0); temperature/low-T; cutoff bars
  (the channel-freezing ratio panel is included in cnt_transport.pdf)
"""
from __future__ import annotations
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/usr/scratch/mont-fort11/pfischill/quatrex/phonon/scripts/out"
FIG = "/usr/scratch/mont-fort11/pfischill/quatrex/document/fig/transport_sweeps"
os.makedirs(FIG, exist_ok=True)


def load(npz):
    d = np.load(npz, allow_pickle=True)
    gb = float(np.atleast_1d(d["thermal_conductance_ballistic"]).ravel()[0])
    ga = float(np.atleast_1d(d["thermal_conductance_anharmonic"]).ravel()[0])
    return gb, ga


def length_series(sub, key="n_slabs"):
    rows = []
    for f in glob.glob(f"{ROOT}/{sub}/checkpoints/length/*.npz"):
        d = np.load(f, allow_pickle=True)
        L = int(np.atleast_1d(d["n_slabs"]).ravel()[0])
        gb, ga = load(f)
        rows.append((L, gb, ga))
    return sorted(rows)


def temp_series(sub):
    rows = []
    for f in glob.glob(f"{ROOT}/{sub}/checkpoints/temperature/*.npz"):
        d = np.load(f, allow_pickle=True)
        T = float(np.atleast_1d(d["t_mean"]).ravel()[0])
        gb, ga = load(f)
        rows.append((T, gb, ga))
    return sorted(rows)


# ---- gather ----
cnt33_L = length_series("cnt33_converge")
cnt80_L1 = load(f"{ROOT}/cnt80_transport/checkpoints/length/L1_T300_dT10_scba.npz")
cnt33_T = temp_series("cnt33_tempsweep")
d5a_T = temp_series("d5a_tempsweep")

# cutoff: map (sigma,vertex,g) -> G_anh
cut = {}
for f in glob.glob(f"{ROOT}/cnt33_cutoff/checkpoints/*/*.npz"):
    name = os.path.basename(f).replace("_dc-interpolate.npz", "")
    _, ga = load(f)
    cut[name] = ga

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

# Panel 1: length ladder + (8,0) L1
Ls = [r[0] for r in cnt33_L]
ratio = [r[2] / r[1] for r in cnt33_L]
ax[0].plot(Ls, ratio, "o-", color="C0", label="(3,3) armchair")
ax[0].plot([1], [cnt80_L1[1] / cnt80_L1[0]], "s", color="C3", ms=9,
           label="(8,0) zigzag (L=1)")
ax[0].axhline(1.0, ls=":", color="gray", lw=0.8)
ax[0].set_xlabel("device length $L$ (cells)")
ax[0].set_ylabel(r"$G_\mathrm{anh}/G_\mathrm{ball}$")
ax[0].set_title("Length ladder (300 K)")
ax[0].set_xticks(Ls)
ax[0].set_ylim(0.6, 1.02)
ax[0].legend(fontsize=8)

# Panel 2: temperature -- G_ball and G_anh (CNT)
Ts = [r[0] for r in cnt33_T]
gb = [r[1] / 1e8 for r in cnt33_T]
ga = [r[2] / 1e8 for r in cnt33_T]
ax[1].plot(Ts, gb, "o-", color="C0", label=r"$G_\mathrm{ball}$")
ax[1].plot(Ts, ga, "s--", color="C1", label=r"$G_\mathrm{anh}$")
ax[1].set_xlabel("temperature (K)")
ax[1].set_ylabel(r"$G$ ($10^8\,\mathrm{W\,m^{-2}K^{-1}}$)")
ax[1].set_title("(3,3) conductance vs $T$ (L=1)")
ax[1].legend(fontsize=8)

# Panel 3: ratio vs T -- channel freezing; CNT vs d5a SiNW
ax[2].plot(Ts, [r[2] / r[1] for r in cnt33_T], "o-", color="C0",
           label="(3,3) CNT")
if d5a_T:
    Td = [r[0] for r in d5a_T]
    ax[2].plot(Td, [r[2] / r[1] for r in d5a_T], "^--", color="C2",
               label="d5a SiNW (soft)")
ax[2].axhline(1.0, ls=":", color="gray", lw=0.8)
ax[2].set_xlabel("temperature (K)")
ax[2].set_ylabel(r"$G_\mathrm{anh}/G_\mathrm{ball}$")
ax[2].set_title(r"Channel freezing: $\to 1$ as $T\to0$")
ax[2].set_ylim(0.6, 1.02)
ax[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(f"{FIG}/cnt_transport.pdf")
fig.savefig(f"{FIG}/cnt_transport.png", dpi=130)
print("wrote cnt_transport.pdf")

# ---- cutoff bar figure ----
order = ["sInf_vInf_gInf", "sInf_vInf_g0", "sInf_v0_gInf", "sInf_v0_g0",
         "s0_vInf_gInf", "s0_vInf_g0", "s0_v0_gInf", "s0_v0_g0"]
labels = {"sInf": r"$\Sigma$:full", "s0": r"$\Sigma$:diag",
          "vInf": "v:full", "v0": "v:diag", "gInf": "G:full", "g0": "G:diag"}
fig2, axc = plt.subplots(figsize=(7.5, 4.2))
names = [n for n in order if n in cut]
vals = [cut[n] / 1e8 for n in names]
def pretty(n):
    s, v, g = n.split("_")
    return f"{labels[s]}\n{labels[v]}\n{labels[g]}"
ref = cut.get("sInf_vInf_gInf", 4.48e8) / 1e8
colors = ["C0" if "g0" not in n else "C3" for n in names]
axc.bar(range(len(names)), vals, color=colors)
axc.axhline(ref, ls="--", color="k", lw=0.8, label=f"full ref {ref:.2f}")
axc.set_xticks(range(len(names)))
axc.set_xticklabels([pretty(n) for n in names], fontsize=7)
axc.set_ylabel(r"$G_\mathrm{anh}$ ($10^8\,\mathrm{W\,m^{-2}K^{-1}}$)")
axc.set_title("(3,3) cutoff hierarchy (n_slabs=2): robust ($\\pm$10%); red = diagonal-G")
axc.set_ylim(3.8, 5.1)
axc.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(f"{FIG}/cnt_cutoff.pdf")
fig2.savefig(f"{FIG}/cnt_cutoff.png", dpi=130)
print("wrote cnt_cutoff.pdf  (cutoff combos:", len(names), ")")
