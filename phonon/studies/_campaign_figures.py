"""Render the campaign-wide observables + convergence figure set.

Consumes campaign_report/<run>_obs.npz (from _campaign_observables.py),
the campaign scheme run.logs, and the pulled physics-run logs. Writes
campaign_report/fig/*.png.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phonon.studies import style

BASE = ROOT / "phonon/studies/out/anderson_test"
REP = BASE / "campaign_report"
FIG = REP / "fig"
FIG.mkdir(exist_ok=True)

C = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc",
     "#ca9161", "#fbafe4", "#949494"]
plt.rcParams.update(style.RC)

O = {p.stem[:-4]: np.load(p) for p in REP.glob("*_obs.npz")}


def residuals(log: Path) -> np.ndarray:
    """rel Sigma^R residuals of the LAST run in a (possibly appended) log."""
    text = log.read_text(errors="replace")
    tail = text[text.rfind("Entering SCBA loop"):]
    return np.array([float(m) for m in re.findall(
        r"rel Sigma\^R residual ([0-9.e+-]+)", tail)])


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("saved", name)


LOGS = REP / "logs"

# ---- S1a: CNT length ladder ------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
for log, lab, c, ls in [
    ("and-cnt-lin", "L2  a=0.2  (conv 222)", C[0], "-"),
    ("cnt-L3-lin", "L3  a=0.2  (conv 347)", C[2], "-"),
    ("cnt-L4-lin", "L4  a=0.2  (fails late)", C[1], "-"),
    ("cnt-L4-m01", "L4  a=0.1  (fails late)", C[1], "--"),
    ("cnt-L10-lin", "L10 a=0.1  (diverges)", C[3], "-"),
    ("cnt-L10-lf", "L10 a=0.1 +lowfreq (diverges)", C[3], "--"),
]:
    r = residuals(LOGS / f"{log}.log")
    if r.size:
        ax.semilogy(r, color=c, ls=ls, label=lab, lw=1.3)
ax.axhline(1e-3, color="#888888", lw=0.8, ls=":")
ax.set_xlabel("SCBA iteration")
ax.set_ylabel(r"rel. $\Sigma^R$ residual")
ax.set_title("CNT(3,3): linear mixing hardens with device length")
ax.set_ylim(1e-4, 1e4)
ax.legend(frameon=False, fontsize=7.5, ncol=2)
save(fig, "s1a_cnt_ladder")

# ---- S1b: L2 scheme comparison ----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), constrained_layout=True)
ax = axes[0]
for scheme, lab, c in [
    ("lin02", "linear 0.2", C[0]),
    ("lin01", "linear 0.1", C[5]),
    ("and_d8", "Anderson d8 (ridge 0)", C[1]),
    ("and_d8_r1e4_guard", "Anderson +ridge+guards", C[3]),
    ("broyden", "Broyden", C[4]),
    ("rpm", "RPM", C[6]),
    ("rre_c12", "RRE c12", C[2]),
]:
    log = BASE / f"mixer_campaign_L2/{scheme}/run.log"
    if log.exists():
        r = residuals(log)
        ax.semilogy(r, color=c, label=lab, lw=1.2)
ax.axhline(1e-3, color="#888888", lw=0.8, ls=":")
ax.set_xlabel("SCBA iteration"); ax.set_ylabel(r"rel. $\Sigma^R$ residual")
ax.set_title("CNT L2: scheme families")
ax.set_ylim(5e-4, 30)
ax.legend(frameon=False, fontsize=7.5)
ax = axes[1]
for scheme, c in [("rre_c10", C[5]), ("rre_c12", C[2]), ("rre_c16", C[0]),
                  ("rre_c20", C[4]), ("rre_c12_b0.3", C[3]),
                  ("rre_c16_b0.3", C[1])]:
    for root in ("rre_sweep_L2", "mixer_campaign_L2"):
        log = BASE / root / scheme / "run.log"
        if log.exists():
            r = residuals(log)
            ax.semilogy(r, color=c, label=scheme.replace("_", " "), lw=1.2)
            break
ax.axhline(1e-3, color="#888888", lw=0.8, ls=":")
ax.set_xlabel("SCBA iteration")
ax.set_title("CNT L2: RRE parameters")
ax.set_ylim(5e-4, 30)
ax.legend(frameon=False, fontsize=7.5)
save(fig, "s1b_l2_schemes")

# ---- S1c: d5a scheme comparison ---------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
r = residuals(LOGS / "and-d5a-lin.log")
ax.semilogy(r, color=C[0], label="linear 0.1 (baseline)", lw=1.2)
for scheme, lab, c in [
    ("and_d8", "Anderson d8 (legacy)", C[1]),
    ("and_d8_r1e4", "Anderson +ridge", C[5]),
    ("and_d8_r1e4_guard", "Anderson +ridge+guards", C[3]),
    ("rre_c8", "RRE c8", C[2]),
    ("rre_c12", "RRE c12", C[4]),
    ("rpm", "RPM", C[6]),
]:
    log = BASE / f"mixer_campaign_d5a_v2/{scheme}/run.log"
    if log.exists():
        ax.semilogy(residuals(log), color=c, label=lab, lw=1.2)
ax.axhline(1e-3, color="#888888", lw=0.8, ls=":")
ax.set_xlabel("SCBA iteration"); ax.set_ylabel(r"rel. $\Sigma^R$ residual")
ax.set_title("d5a SiNW (a=0.1, IR floor): schemes")
ax.set_ylim(1e-4, 30)
ax.legend(frameon=False, fontsize=7.5)
save(fig, "s1c_d5a_schemes")

# ---- S1d: Anderson forensics -------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), constrained_layout=True)
runs = [("mixer_campaign_L2/and_d8_r1e4/run.npz",
         "CNT L2 Anderson+ridge (fails)", C[1]),
        ("mixer_campaign_d5a_v2/and_d8_r1e4_guard/run.npz",
         "d5a Anderson+ridge+guards (49 it)", C[3])]
for path, lab, c in runs:
    d = np.load(BASE / path)
    if "iter_mixer_fnorm" not in d.files:
        continue
    fn = d["iter_mixer_fnorm"]
    cond = d["iter_mixer_cond"]
    gn = d["iter_mixer_gnorm"]
    it = np.arange(fn.size)
    axes[0].semilogy(it, fn, color=c, label=lab, lw=1.1)
    m = np.isfinite(cond)
    axes[1].semilogy(it[m], cond[m], color=c, lw=1.1)
    m = np.isfinite(gn)
    axes[2].semilogy(it[m], gn[m], color=c, lw=1.1)
axes[0].set_ylabel(r"$\|f\|$ (mixer residual)")
axes[1].set_ylabel(r"cond$(\Delta F^H \Delta F)$")
axes[2].set_ylabel(r"$\|\gamma\|$ (LS coefficients)")
for ax in axes:
    ax.set_xlabel("mixer step")
axes[0].legend(frameon=False, fontsize=7)
axes[0].set_title("residual"); axes[1].set_title("LS conditioning")
axes[2].set_title("extrapolation weights")
save(fig, "s1d_forensics")

# ---- S1e: Jacobian mechanism -------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
lam = np.linspace(1, 45, 300)
ax.plot(lam, 2.0 / (1.0 + lam), color=C[0], lw=1.6,
        label=r"stability bound  $\alpha < 2/(1+|\lambda|)$")
pts = [  # (|lambda|, alpha, converged?, label)
    (4.33, 0.2, True, "L2 @0.2 (probe: $\\lambda=-4.33$)"),
    (4.33, 0.3, True, "L2 RRE build @0.3"),
    (20, 0.2, False, "L4 @0.2"),
    (20, 0.1, False, "L4 @0.1"),
    (40, 0.1, False, "L10 @0.1"),
    (40, 0.05, False, "L4 RRE build @0.05"),
]
for lm, a, ok, lab in pts:
    ax.plot([lm], [a], "o" if ok else "x", color=C[2] if ok else C[3],
            markersize=8, mew=2)
    ax.annotate(lab, (lm, a), textcoords="offset points", xytext=(6, 4),
                fontsize=7)
ax.set_xlabel(r"dominant $|\lambda|$ of $J=\partial F/\partial\Sigma$ "
              "(negative, IR-localized)")
ax.set_ylabel(r"damping $\alpha$")
ax.set_title("Measured mechanism: negative IR eigenvalue grows with L")
ax.set_xlim(1, 45); ax.set_ylim(0, 0.45)
ax.legend(frameon=False, fontsize=8, loc="upper right")
save(fig, "s1e_jacobian")

# ---- S2 grids: DOS / transmission / heat spectra -----------------------------
FULL = [("cnt_L2", "CNT L2"), ("cnt_L3", "CNT L3"),
        ("cnt_L4", "CNT L4 (unconv.)"), ("d5a_lin", "d5a SiNW")]

for panel, key_s, key_b, ylab, title in [
    ("s2a_dos", "ldos", "ldos_ball", r"DOS per DOF (THz$^{-1}$)",
     "Density of states: SCBA vs ballistic"),
]:
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), constrained_layout=True)
    for ax, (run, lab) in zip(axes.ravel(), FULL):
        o = O[run]
        f = o["f"]
        ax.plot(f, o[key_b].mean(axis=1), color=C[7], lw=1.1,
                label="ballistic")
        ax.plot(f, o[key_s].mean(axis=1), color=C[0], lw=1.3, label="SCBA")
        ax.set_title(f"{lab}  (sum rule {o['sum_rule'].mean():.3f})",
                     fontsize=9)
        ax.set_xlim(0, f[-1]); ax.set_ylim(bottom=0)
        ax.set_xlabel("frequency (THz)"); ax.set_ylabel(ylab)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    save(fig, panel)

fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), constrained_layout=True)
for ax, (run, lab) in zip(axes.ravel(), FULL):
    o = O[run]
    f = o["f"]
    ax.plot(f, o["T_ball"], color=C[7], lw=1.1, label=r"ballistic $\mathcal{T}$")
    ax.plot(f, o["T_eff_w"][:, 0], color=C[0], lw=1.3,
            label=r"SCBA $\mathcal{T}_{\rm eff}$")
    ax.set_title(lab, fontsize=9)
    ax.set_xlim(0, f[-1]); ax.set_ylim(bottom=0)
    ax.set_xlabel("frequency (THz)"); ax.set_ylabel("transmission")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("Transmission: Caroli vs effective (eq:caroli, eq:T_eff)")
save(fig, "s2b_transmission")

fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), constrained_layout=True)
for ax, (run, lab) in zip(axes.ravel(), FULL):
    o = O[run]
    f = o["f"]
    ax.plot(f, o["j_W_per_THz"][:, 0] * 1e9, color=C[0], lw=1.3,
            label=r"SCBA $j_L(\omega)$")
    ax.plot(f, o["j_W_per_THz"][:, -1] * 1e9, color=C[1], lw=1.0, ls="--",
            label=r"SCBA $j_R(\omega)$")
    # IR plateau k_B dT Tbar(0) (all runs: T = 305/295 K)
    plateau = 1.380649e-23 * 10.0 * float(o["T_ball"][1]) * 1e12
    ax.axhline(plateau * 1e9, color=C[3], ls=":", lw=1.1)
    ax.set_title(lab, fontsize=9)
    ax.set_xlim(0, f[-1]); ax.set_ylim(bottom=0)
    ax.set_xlabel("frequency (THz)"); ax.set_ylabel(r"$j(\omega)$ (nW/THz)")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("Spectral heat current + IR plateau (eq:meir_wingreen, eq:ir_plateau)")
save(fig, "s2c_heat_spectra")

# ---- S2d: T_eff profiles + occupation ---------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), constrained_layout=True)
ax = axes[0]
for run, lab, c in [("cnt_L2", "CNT L2", C[0]), ("cnt_L3", "CNT L3", C[2]),
                    ("cnt_L4", "CNT L4*", C[1]), ("d5a_lin", "d5a", C[3])]:
    t = O[run]["T_eff_slab"]
    x = np.linspace(0, 1, t.size + 2)
    ax.plot(x, np.concatenate(([305], t, [295])), "o-", color=c, label=lab,
            markersize=4)
ax.set_xlabel("position along device (leads at 0, 1)")
ax.set_ylabel(r"$T_{\rm eff}$ (K)")
ax.set_title("Local effective temperature (eq:Teff_local)")
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
o = O["cnt_L3"]
f = o["f"]
nb = o["n_bose_mid"]
for s in range(o["occ_slab"].shape[1]):
    dev = 100 * (o["occ_slab"][1:, s] - nb[1:]) / np.maximum(nb[1:], 1e-30)
    ax.plot(f[1:], dev, color=C[s], lw=1.1, label=f"slab {s+1}")
ax.axhline(0, color="#888888", lw=0.8)
ax.set_xlabel("frequency (THz)")
ax.set_ylabel(r"$(n - n_{300})/n_{300}$ (%)")
ax.set_ylim(-6, 6)
ax.set_title("CNT L3 occupation deviation (eq:neq_occupation)")
ax.legend(frameon=False, fontsize=8)
save(fig, "s2d_teff_occ")

# ---- S2e: MSD ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), constrained_layout=True)
ax = axes[0]
for run, lab, c in [("cnt_L2", "L2", C[0]), ("cnt_L3", "L3", C[2]),
                    ("cnt_L4", "L4*", C[1])]:
    m = O[run]["msd_atom"]
    ax.plot(np.linspace(0, 1, m.size), m, "o-", color=c, markersize=3,
            lw=1.0, label=f"CNT {lab}")
ax.plot(np.linspace(0, 1, O["cnt_L2"]["msd_eq_atom"].size),
        O["cnt_L2"]["msd_eq_atom"], "s--", color=C[7], markersize=3,
        label="isolated mode sum (L2)")
ax.set_xlabel("position along device"); ax.set_ylabel(r"$\langle w^2\rangle$ (amu Å$^2$)")
ax.set_title("CNT mean-square displacement")
ax.legend(frameon=False, fontsize=7.5)
ax = axes[1]
m = O["d5a_lin"]["msd_atom"]
ax.plot(np.arange(1, m.size + 1), m, "o-", color=C[3], markersize=3, lw=1.0,
        label="d5a NEGF")
me = O["d5a_lin"]["msd_eq_atom"]
ax.plot(np.arange(1, me.size + 1), me, "s--", color=C[7], markersize=3,
        label="isolated mode sum")
ax.set_xlabel("atom index"); ax.set_ylabel(r"$\langle w^2\rangle$ (amu Å$^2$)")
ax.set_title("d5a mean-square displacement")
ax.legend(frameon=False, fontsize=7.5)
save(fig, "s2e_msd")

# ---- S2f: ledgers -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), constrained_layout=True)
for ax, run, lab in zip(axes, ("cnt_L3", "cnt_L4", "d5a_lin"),
                        ("CNT L3", "CNT L4*", "d5a")):
    o = O[run]
    fh = o["final_heat"]
    pa = o["slab_pa"]
    tele = fh - np.concatenate(([0.0], np.cumsum(pa)))
    xpos = np.arange(fh.size)
    w = 0.38
    ax.bar(xpos - w / 2, fh, w, color=C[0], label="raw bond current")
    ax.bar(xpos + w / 2, tele, w, color=C[2], label="telescoped")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"I{i}" for i in xpos], fontsize=8)
    lo = min(fh.min(), tele.min())
    ax.set_ylim(lo * 0.97, max(fh.max(), tele.max()) * 1.01)
    ax.set_title(lab, fontsize=9)
    ax.set_ylabel("J (internal)")
axes[0].legend(frameon=False, fontsize=7.5)
fig.suptitle("Bond-current ledger: raw vs telescoped (eq:local_ledger)")
save(fig, "s2f_ledger")

# ---- S3a: kappa(L), r(L) ------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
import json
summ = json.loads((REP / "summary.json").read_text())
Ls = np.array([2, 3, 4])
G = np.array([summ[f"cnt_L{k}"]["G_anh"] for k in Ls]) * 1e9
Gb = summ["cnt_L2"]["G_ball"] * 1e9
c_ang = 2.4595
ax = axes[0]
ax.plot(Ls, G, "o-", color=C[0], label=r"$G_{\rm anh}(L)$")
ax.axhline(Gb, color=C[7], ls="--", lw=1.1, label=r"$G_{\rm ball}$ (L-indep.)")
ax.set_xlabel("device length L (cells)"); ax.set_ylabel("G (nW/K)")
ax.set_xticks(Ls)
ax.set_title("Conductance vs length")
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
kappa = G * Ls * c_ang * 1e-10 * 1e-9  # W/K * m -> per-area-free 1D kappa
ax.plot(Ls, G * Ls, "o-", color=C[2],
        label=r"$\kappa \propto G \cdot L$ (arb.)")
ax.plot(Ls, G / Gb, "s-", color=C[1], label=r"$r(L) = G_{\rm anh}/G_{\rm ball}$")
ax.set_xlabel("device length L (cells)")
ax.set_xticks(Ls)
ax.set_title("Length scaling (L4 point unconverged)")
ax.legend(frameon=False, fontsize=8)
save(fig, "s3a_length")

# ---- S3b: fixed-point agreement -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True)
ax = axes[0]
ref = float(O["cnt_L2"]["final_heat"][0])
l2 = [(k, float(O[k]["final_heat"][0])) for k in sorted(O)
      if k.startswith("L2_")]
names = [k[3:] for k, _ in l2]
vals = [(v - ref) / ref for _, v in l2]
colors = [C[3] if "irfloor" in n else C[0] for n in names]
ax.bar(range(len(vals)), np.array(vals) * 1e6, color=colors)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
ax.set_ylabel(r"$(J_L - J_L^{\rm lin02})/J_L^{\rm lin02}$  (ppm)")
ax.set_title("CNT L2: fixed-point agreement across schemes")
ax = axes[1]
d5 = [("linear 0.1", "d5a_lin"), ("Anderson stall*", "d5a_and"),
      ("guard (conv)", "d5a_guard"), ("RRE c8", "d5a_rre8")]
refd = float(O["d5a_guard"]["final_heat"][0])
vals = [(float(O[k]["final_heat"][0]) - refd) / refd * 100 for _, k in d5]
ax.bar(range(len(d5)), vals, color=[C[0], C[5], C[3], C[2]])
ax.set_xticks(range(len(d5)))
ax.set_xticklabels([n for n, _ in d5], rotation=30, ha="right", fontsize=8)
ax.set_ylabel(r"$\Delta J_L$ vs guard (%)")
ax.set_title("d5a: spread limited by the conservation floor")
save(fig, "s3b_agreement")

# ---- S3c: CNT vs d5a -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), constrained_layout=True)
ax = axes[0]
for run, lab, c in [("cnt_L2", "CNT L2", C[0]), ("d5a_lin", "d5a", C[3])]:
    o = O[run]
    f = o["f"]
    Tb = o["T_ball"]
    Te = o["T_eff_w"][:, 0]
    m = Tb > 0.05
    fmax = f[m][-1]
    ax.plot(f[m] / fmax, Te[m] / Tb[m], color=c, lw=1.2, label=lab)
ax.axhline(1, color="#888888", lw=0.8, ls=":")
ax.set_xlabel(r"$\omega/\omega_{\rm max}$")
ax.set_ylabel(r"$\mathcal{T}_{\rm eff}/\mathcal{T}_{\rm ball}$")
ax.set_ylim(0, 1.2)
ax.set_title("Anharmonic suppression: CNT vs d5a")
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
for run, lab, c in [("cnt_L2", "CNT L2", C[0]), ("d5a_lin", "d5a", C[3])]:
    o = O[run]
    f = o["f"]
    dos = o["ldos"].mean(axis=1)
    m = f < (47 if run.startswith("cnt") else 20)
    ax.plot(f[m] / f[m][-1], dos[m] / dos[m].max(), color=c, lw=1.2,
            label=lab)
ax.set_xlabel(r"$\omega/\omega_{\rm max}$")
ax.set_ylabel("DOS (normalized)")
ax.set_title("SCBA DOS, normalized")
ax.legend(frameon=False, fontsize=8)
save(fig, "s3c_systems")

print("all campaign figures done")
