"""CNT observables atlas: every transport observable across the length ladder.

Seven figures + a summary table over the eta=0 CNT campaigns, organised by
the observables of document/src/theory/20_negf.tex (sub:negf_obs) and the
conservation appendix:

  1 atlas_ladder        G(L) [eq:G_thermal via eq:meir_wingreen], R(L),
                        kappa_eff(L) = G L / A  (pi*d*h tube convention)
  2 atlas_teff          effective transmission T_eff(w) [eq:T_eff] vs exact
                        Bloch mode staircase M(w); spectral deficit
  3 atlas_spectral_current  heat-current integrand + cumulative fraction
  4 atlas_local         slab-resolved T_eff(i) [eq:Teff_local] + LDOS map
                        [eq:dos]
  5 atlas_conservation  energy ledger [eq:local_ledger] reconstruction,
                        D(w) sum rule [eq:sumrule], per-rung conservation
  6 atlas_stability     SCBA lead-current trajectories: g1/g2 divergence
                        vs g3 / Bartlett-taper stability (eta=0)
  7 atlas_tubes_T       conductance ratio r(T) [eq:cond_ratio] and
                        r(length) at eta=0. NB the legacy (8,0) prod runs
                        used eta=0.7 (pre-doctrine) and are EXCLUDED; an
                        eta=0 (8,0) rerun is needed for any cross-tube
                        comparison.

Data: phonon/studies/out/cnt33_gband_length (L8/L10 x g1,g2,g3,g1t; full
conservation keys), phonon/studies/out/cnt33_long_gband3 (L16/L24/L32),
phonon/scripts/out/prod/cnt33_eta0 (L2-L4 + T-sweep anh/ball pairs),
units_parity ballistic reference.

Output: phonon/studies/out/fig/cnt33_atlas/ (png+pdf via style.save).
Run:  python phonon/scripts/figures/cnt33_observables_atlas.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "phonon")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phonon.studies import style  # noqa: E402
from phonon.postproc.units import heat_current_watts  # noqa: E402
from phonon.postproc import local_observables  # noqa: E402

OUT = ROOT / "phonon/studies/out/fig/cnt33_atlas"
GBL = ROOT / "phonon/studies/out/cnt33_gband_length"
LG3 = ROOT / "phonon/studies/out/cnt33_long_gband3"
PROD = ROOT / "phonon/scripts/out/prod/cnt33_eta0"
PROD80 = ROOT / "cluster/prod/cnt80"
BALL = ROOT / "phonon/studies/out/units_parity/cnt33_L4/run.npz"
MAT = ROOT / "phonon/studies/out/anderson_test/cnt33_L4_inputs/dynamical_matrix.mat"

CELL_A = 2.4595                      # (3,3) transport period, Angstrom
D33, D80, SHELL = 4.07e-10, 6.26e-10, 3.35e-10
A_T33 = np.pi * D33 * SHELL          # pi*d*h tube cross-section (m^2)
A_T80 = np.pi * D80 * SHELL
T0, DT = 300.0, 10.0
KB_EV = 8.617333262e-5
THZ_TO_EV = 6.582119569e-16 * 2 * np.pi * 1e12


def bose(freqs_thz, T):
    x = THZ_TO_EV * np.abs(np.asarray(freqs_thz)) / (KB_EV * T)
    out = np.zeros_like(x)
    pos = x > 1e-12
    out[pos] = 1.0 / np.expm1(np.clip(x[pos], None, 500.0))
    return out


def load_run(path):
    d = np.load(path, allow_pickle=True)
    en = np.abs(np.asarray(d["energies"], float))
    cs = np.real(np.asarray(d["current_spectrum"]))
    while cs.ndim > 2:
        cs = cs.mean(axis=1)
    w = (np.asarray(d["frequency_cell_widths"], float)
         if "frequency_cell_widths" in d.files else np.gradient(en))
    tl = float(d["t_left"]) if "t_left" in d.files else T0 + DT / 2
    tr = float(d["t_right"]) if "t_right" in d.files else T0 - DT / 2
    nblk = int(d["nblocks"]) if "nblocks" in d.files else cs.shape[-1] - 1
    # block length: campaign geoms are 18-DOF half cells; prod/parity are
    # 36-DOF full cells. Infer from gr_diag when present.
    ndof_blk = None
    if "gr_diag_imag" in d.files:
        g = np.asarray(d["gr_diag_imag"])
        ndof_blk = g.shape[-1] // nblk
    blk_A = CELL_A * (ndof_blk / 36.0) if ndof_blk else CELL_A
    length_m = nblk * blk_A * 1e-10
    J = heat_current_watts(en, cs, w)
    G = 0.5 * (abs(J[0]) + abs(J[-1])) / (tl - tr)
    dn = bose(en, tl) - bose(en, tr)
    teff = 0.5 * (np.abs(cs[:, 0]) + np.abs(cs[:, -1]))
    teff = np.divide(teff, dn, out=np.zeros_like(teff), where=dn > 1e-30)
    return dict(d=d, en=en, cs=cs, w=w, tl=tl, tr=tr, nblk=nblk,
                length_m=length_m, G=G, teff=teff, dn=dn,
                conv=bool(d["converged"]), n_iter=int(d["n_iter"]))


def mode_count(en):
    from scipy.io import loadmat
    m = loadmat(str(MAT))
    h00 = np.asarray(m["[0, 0, 0]"], complex)
    h01 = np.asarray(m["[0, 0, 1]"], complex)
    ks = np.linspace(0, np.pi, 4001)
    bands = np.zeros((len(ks), h00.shape[0]))
    for i, k in enumerate(ks):
        Hk = h00 + h01 * np.exp(1j * k) + h01.conj().T * np.exp(-1j * k)
        bands[i] = np.sqrt(np.clip(np.linalg.eigvalsh(Hk), 0, None))
    M = np.zeros(len(en))
    for n in range(bands.shape[1]):
        s = np.signbit(bands[:, n][None, :] - en[:, None])
        M += (s[:, 1:] != s[:, :-1]).sum(axis=1)
    return M


# ---------------------------------------------------------------- registry
# CRITICAL bookkeeping (verified 2026-07-31): the two campaigns tile the
# SAME 12-atom cell but partition differently. cnt33_gband_length runs use
# FULL-cell 36-DOF blocks (tag L8 = 8 cells = 1.97 nm); cnt33_long_gband3
# uses HALF-cell 18-DOF blocks (tag L16 = 8 cells = 1.97 nm!). Because
# sse_g_band counts BLOCK off-diagonals, the PHYSICAL bubble support is
# g_band x block length: e.g. "g1t" = +-2.46 A (full) vs +-1.23 A (half),
# "g3" = +-7.4 A (full) vs +-3.7 A (half). Series must never be mixed.
LADDER = [  # (label, path, series); series = <partition>_<treatment>
    ("8c",  GBL / "L8_g1t/run.npz",  "full_g1t"),
    ("10c", GBL / "L10_g1t/run.npz", "full_g1t"),
    ("8c",  GBL / "L8_g3/run.npz",   "full_g3"),
    ("10c", GBL / "L10_g3/run.npz",  "full_g3"),
    ("8c",  LG3 / "L16_g1t/run.npz", "half_g1t"),
    ("12c", LG3 / "L24_g1t/run.npz", "half_g1t"),
    ("16c", LG3 / "L32_g1t/run.npz", "half_g1t"),
    ("8c",  LG3 / "L16_g3/run.npz",  "half_g3"),
    ("12c", LG3 / "L24_g3/run.npz",  "half_g3"),
]
SERIES_STYLE = {
    "full_g1t": dict(marker="o", color="#0173b2", ls="-",
                     label="full-cell taper (+-2.5 A)"),
    "full_g3":  dict(marker="s", color="#029e73", ls="-",
                     label="full-cell g3 (+-7.4 A)"),
    "half_g1t": dict(marker="v", color="#56b4e9", ls="--",
                     label="half-cell taper (+-1.2 A)"),
    "half_g3":  dict(marker="^", color="#de8f05", ls="--",
                     label="half-cell g3 (+-3.7 A)"),
}
#: physical Sigma support in Angstrom per series (for the support panel)
SUPPORT_A = {"full_g1t": 2.46, "full_g3": 7.38,
             "half_g1t": 1.23, "half_g3": 3.69}
PROD_L = [("L2", 2), ("L3", 3), ("L4", 4)]


def fig_ladder(runs, ball):
    fig, axes = style.figure(ncols=3, width=11.8, height=3.3)
    ax_g, ax_s, ax_k = axes

    G_ball = ball["G"]
    ax_g.axhline(G_ball * 1e9, color="0.55", ls="--", lw=1,
                 label=f"ballistic ({G_ball*1e9:.2f} nW/K)")
    for s, kw in SERIES_STYLE.items():
        pts = sorted((r["length_m"] * 1e9, r["G"] * 1e9, r["conv"])
                     for (lab, p, ser), r in runs.items() if ser == s)
        x = [p[0] for p in pts]; y = [p[1] for p in pts]
        ax_g.plot(x, y, kw["ls"], color=kw["color"], lw=1, alpha=0.6)
        for xi, yi, conv in pts:
            ax_g.plot(xi, yi, kw["marker"], color=kw["color"],
                      mfc=kw["color"] if conv else "none", ms=6)
        ax_g.plot([], [], kw["marker"], color=kw["color"], ls=kw["ls"],
                  label=kw["label"])
    for lab, L in PROD_L:
        p = PROD / f"{lab}_anh.npz"
        if p.exists():
            r = load_run(p)
            ax_g.plot(r["length_m"] * 1e9, r["G"] * 1e9, "D",
                      color="#cc78bc", ms=5, mfc="none")
    ax_g.plot([], [], "D", color="#cc78bc", mfc="none",
              label="prod L2-L4 (full, untapered)")
    ax_g.set_xlabel("device length (nm)")
    ax_g.set_ylabel("G (nW/K per tube)")
    ax_g.legend(fontsize=6, loc="upper right")
    ax_g.set_title("lead conductance vs length\n(open symbols = 600-iter cap)")

    # THE partition/support systematic: G at the SAME 1.97 nm device vs
    # physical Sigma support (Sigma truncation counts BLOCKS, not Angstrom).
    pts = [(SUPPORT_A[ser], r["G"] * 1e9, r["conv"], SERIES_STYLE[ser])
           for (lab, p, ser), r in runs.items() if lab == "8c"]
    for sup, G, conv, kw in sorted(pts):
        ax_s.plot(sup, G, kw["marker"], color=kw["color"], ms=7,
                  mfc=kw["color"] if conv else "none")
    ax_s.plot([p[0] for p in sorted(pts)], [p[1] for p in sorted(pts)],
              ":", color="0.6", lw=1)
    ax_s.axhline(G_ball * 1e9, color="0.55", ls="--", lw=1)
    ax_s.set_xlabel(r"bubble support g$_{\rm band}\times$block length (A)")
    ax_s.set_ylabel("G (nW/K), 8-cell device")
    ax_s.set_title("SAME device, growing Sigma support:\n"
                   "NOT support-converged (rising trend)")

    # kappa_eff, per series
    for s, kw in SERIES_STYLE.items():
        pts = sorted((r["length_m"], r["G"]) for (lab, p, ser), r
                     in runs.items() if ser == s)
        ax_k.plot([p[0] * 1e9 for p in pts],
                  [p[1] * p[0] / A_T33 for p in pts],
                  kw["ls"], marker=kw["marker"], color=kw["color"], ms=5,
                  label=kw["label"])
    xb = np.linspace(0, 4.2, 50)
    ax_k.plot(xb, ball["G"] * xb * 1e-9 / A_T33, "--", color="0.55",
              label="ballistic G*L")
    ax_k.set_xlabel("device length (nm)")
    ax_k.set_ylabel(r"$\kappa_{\rm eff}$ = G L / A$_{\pi dh}$  (W/m/K)")
    ax_k.legend(fontsize=6)
    ax_k.set_title("effective conductivity (quasi-ballistic growth)")
    style.save(fig, "atlas_ladder", directory=OUT)


def fig_teff(runs, ball):
    fig, axes = style.figure(ncols=2, width=9.2, height=3.4)
    ax, axd = axes
    en = ball["en"]
    M = mode_count(en)
    ax.step(en, M, where="mid", color="0.3", lw=1.2, label="Bloch mode count M")
    ax.plot(en, ball["teff"], color="0.6", lw=1, label="engine ballistic")
    sel = [("8c", "full_g1t", "#0173b2"), ("10c", "full_g1t", "#56b4e9"),
           ("8c", "half_g1t", "#029e73"), ("12c", "half_g1t", "#de8f05"),
           ("16c", "half_g1t", "#d55e00")]
    for lab, ser, col in sel:
        key = next(k for k in runs if k[0] == lab and k[2] == ser)
        r = runs[key]
        part = "full" if ser.startswith("full") else "half"
        ax.plot(r["en"], r["teff"], color=col, lw=1,
                label=f"{lab} {part} ({r['length_m']*1e9:.1f} nm)")
        d = np.divide(M - r["teff"], M, out=np.zeros_like(M), where=M > 0.5)
        axd.plot(r["en"], d, color=col, lw=1)
    ax.set_xlabel("frequency (THz)"); ax.set_ylabel(r"T$_{\rm eff}$($\omega$)")
    ax.legend(fontsize=6.5); ax.set_xlim(0, 50)
    ax.set_title("effective transmission vs length (taper branch)")
    axd.set_xlabel("frequency (THz)")
    axd.set_ylabel(r"deficit 1 - T$_{\rm eff}$/M")
    axd.set_xlim(0, 50); axd.set_ylim(0, 1.05)
    axd.set_title("spectral transmission deficit")
    style.save(fig, "atlas_teff", directory=OUT)


def fig_spectral_current(runs, ball):
    fig, axes = style.figure(ncols=2, width=9.2, height=3.4)
    ax, axc = axes
    sel = [("ballistic", ball, "0.5"),
           ("8c full taper", runs[("8c", GBL / "L8_g1t/run.npz", "full_g1t")], "#0173b2"),
           ("16c half taper", runs[("16c", LG3 / "L32_g1t/run.npz", "half_g1t")], "#d55e00")]
    HB = 1.0545718e-34 * 2 * np.pi * 1e24    # internal -> W conversion
    for lab, r, col in sel:
        # heat-current spectral density (pW per THz)
        dens = HB * np.abs(r["en"]) * 0.5 * (np.abs(r["cs"][:, 0]) +
                                             np.abs(r["cs"][:, -1])) * 1e12
        ax.plot(r["en"], dens, color=col, lw=1, label=lab)
        cum = np.cumsum(r["w"] * np.abs(r["en"]) * 0.5 *
                        (np.abs(r["cs"][:, 0]) + np.abs(r["cs"][:, -1])))
        axc.plot(r["en"], cum / cum[-1], color=col, lw=1, label=lab)
    ax.set_xlabel("frequency (THz)"); ax.set_ylabel("dJ/dw (pW/THz)")
    ax.legend(fontsize=7); ax.set_xlim(0, 50)
    ax.set_title("spectral heat current (300 K, dT=10 K)")
    axc.set_xlabel("frequency (THz)"); axc.set_ylabel("cumulative fraction of J")
    axc.axhline(0.5, color="0.8", lw=0.6); axc.set_xlim(0, 50)
    axc.legend(fontsize=7); axc.set_title("cumulative heat vs frequency")
    style.save(fig, "atlas_spectral_current", directory=OUT)


def fig_local(runs):
    fig, axes = style.figure(ncols=2, width=9.6, height=3.4)
    axt, axl = axes
    sel = [("8c full", GBL / "L8_g1t/run.npz", "full_g1t", "#0173b2"),
           ("12c half", LG3 / "L24_g1t/run.npz", "half_g1t", "#029e73"),
           ("16c half", LG3 / "L32_g1t/run.npz", "half_g1t", "#d55e00")]
    for lab, path, ser, col in sel:
        key = next(k for k in runs if k[1] == path)
        r = runs[key]
        ndof = np.asarray(r["d"]["gr_diag_imag"]).shape[-1] // r["nblk"]
        obs = local_observables.compute(path, n_dof=ndof)
        Tprof = obs["T_eff"]
        x = (np.arange(r["nblk"]) + 0.5) / r["nblk"]
        axt.plot(x, Tprof, "o-", ms=3, lw=1, color=col,
                 label=f"{lab} ({r['length_m']*1e9:.1f} nm)")
    axt.axhline(305, color="0.7", ls=":", lw=0.8)
    axt.axhline(295, color="0.7", ls=":", lw=0.8)
    axt.text(0.01, 305.1, "T$_L$", fontsize=7, color="0.4")
    axt.text(0.01, 294.4, "T$_R$", fontsize=7, color="0.4")
    axt.set_xlabel("position x/L"); axt.set_ylabel("local T$_{\\rm eff}$ (K)")
    axt.legend(fontsize=7); axt.set_title("slab-resolved effective temperature")

    # LDOS map for L32
    path = LG3 / "L32_g1t/run.npz"
    r = runs[("16c", path, "half_g1t")]
    ndof = np.asarray(r["d"]["gr_diag_imag"]).shape[-1] // r["nblk"]
    obs = local_observables.compute(path, n_dof=ndof)
    ld = obs["ldos"].reshape(len(r["en"]), r["nblk"], ndof).sum(axis=-1)
    im = axl.pcolormesh(np.arange(r["nblk"]) * r["length_m"] * 1e9 / r["nblk"],
                        r["en"], np.log10(np.clip(ld, 1e-6, None)),
                        shading="auto", cmap="magma", rasterized=True)
    fig.colorbar(im, ax=axl, label=r"log$_{10}$ LDOS")
    axl.set_xlabel("position (nm)"); axl.set_ylabel("frequency (THz)")
    axl.set_title("slab-resolved LDOS, L32 taper")
    style.save(fig, "atlas_local", directory=OUT)


def fig_conservation(runs):
    fig, axes = style.figure(ncols=3, width=11.5, height=3.2)
    axJ, axD, axB = axes
    # (a) energy ledger reconstruction, L8 g3 vs g1t
    for tag, col in (("L8_g3", "#de8f05"), ("L8_g1t", "#0173b2")):
        d = np.load(GBL / tag / "run.npz", allow_pickle=True)
        J = np.real(np.asarray(d["last_heat"]))
        pa = np.asarray(d["slab_absorption"])
        if pa.ndim == 2:
            pa = pa[0]
        recon = J[0] + np.concatenate(([0.0], np.cumsum(pa)[: len(J) - 1]))
        k = np.arange(len(J))
        axJ.plot(k, J, "o", ms=4, color=col, label=f"{tag} J_k")
        axJ.plot(k, recon, "-", lw=1, color=col, alpha=0.7)
    axJ.set_xlabel("interface k"); axJ.set_ylabel("heat current (internal)")
    axJ.legend(fontsize=7)
    axJ.set_title("energy ledger: J_k vs J_0 + cumsum P_abs")
    # (b) D(omega) sum rule
    for tag, col in (("L8_g3", "#de8f05"), ("L8_g1t", "#0173b2")):
        obs = local_observables.compute(GBL / tag / "run.npz")
        if "D_omega" in obs:
            axD.plot(obs["freqs_thz"], np.real(obs["D_omega"]), lw=0.8,
                     color=col, label=tag)
    axD.set_xlabel("frequency (THz)"); axD.set_ylabel(r"D($\omega$)")
    axD.legend(fontsize=7); axD.set_title("energy sum rule spectrum")
    # (c) conservation metrics per rung
    tags, js_rel, bb = [], [], []
    for L in (8, 10):
        for g in ("1", "2", "3", "1t"):
            p = GBL / f"L{L}_g{g}/run.npz"
            d = np.load(p, allow_pickle=True)
            en = np.abs(np.asarray(d["energies"], float))
            w = np.asarray(d["frequency_cell_widths"], float)
            cs = np.real(np.asarray(d["current_spectrum"]))
            while cs.ndim > 2:
                cs = cs.mean(axis=1)
            I = np.sum(w * en * 0.5 * (np.abs(cs[:, 0]) + np.abs(cs[:, -1])))
            p_in, p_out = np.asarray(d["bubble_balance_spectrum"])
            Js = np.sum((p_out.real - p_in.real))
            tags.append(f"L{L}g{g}")
            js_rel.append(abs(Js) / max(I, 1e-30))
            ib = np.asarray(d["iter_bubble_balance"])
            bb.append(abs(ib[-1, 2]))
    xpos = np.arange(len(tags))
    axB.bar(xpos - 0.2, js_rel, 0.4, color="#029e73", label="|J_s|/|I|")
    axB.bar(xpos + 0.2, bb, 0.4, color="#cc78bc", label="bubble resid")
    axB.set_yscale("log"); axB.set_xticks(xpos)
    axB.set_xticklabels(tags, rotation=60, fontsize=6)
    axB.legend(fontsize=7); axB.set_title("conservation per rung")
    style.save(fig, "atlas_conservation", directory=OUT)


def fig_stability():
    fig, axes = style.figure(ncols=2, width=9.2, height=3.4)
    for ax, L in zip(axes, (8, 10)):
        for g, col in (("1", "#d55e00"), ("2", "#de8f05"),
                       ("3", "#029e73"), ("1t", "#0173b2")):
            d = np.load(GBL / f"L{L}_g{g}/run.npz", allow_pickle=True)
            ih = np.asarray(d["iter_heat"])
            lead = 0.5 * (np.abs(ih[:, 0]) + np.abs(ih[:, -1]))
            ax.semilogy(np.arange(len(lead)), np.clip(lead, 1e-3, None),
                        color=col, lw=1,
                        label=f"g{g}" + (" (taper)" if g == "1t" else ""))
        ax.set_xlabel("SCBA iteration")
        ax.set_ylabel("lead current (internal, rank-0 slice)")
        ax.set_title(f"L{L}: eta=0 stability by band treatment")
        ax.legend(fontsize=7)
    style.save(fig, "atlas_stability", directory=OUT)


def fig_tubes_T(runs):
    import csv
    fig, axes = style.figure(ncols=2, width=9.2, height=3.4)
    axT, axC = axes
    rows33 = list(csv.DictReader(open(PROD / "summary.csv")))
    rows80 = list(csv.DictReader(open(PROD80 / "summary.csv")))
    # r(T), (3,3) prod L2 temperature sweep
    tv = sorted((float(r["t_mean"]), float(r["ratio"]))
                for r in rows33 if r["sweep"] == "temperature")
    axT.plot([t for t, _ in tv], [r for _, r in tv], "o-", color="#0173b2",
             label="(3,3) L2, eta=0 prod")
    axT.set_xlabel("temperature (K)")
    axT.set_ylabel(r"r = G$_{\rm anh}$/G$_{\rm ball}$")
    axT.legend(fontsize=7)
    axT.set_title("anharmonic suppression vs temperature")

    # NB the legacy cnt80 prod runs used eta=0.7 (pre-doctrine) and are NOT
    # comparable to the eta=0 (3,3) data -- no cross-tube panel until an
    # eta=0 (8,0) rerun exists (geoms are on the cluster).
    del rows80

    # suppression fraction vs length at 300 K, eta=0 only
    ball_G = 1.562e-9  # engine ballistic, W/K (parity run, same grid family)
    lv = sorted((0.245 * {"L2": 2, "L3": 3, "L4": 4}[r["tag"]],
                 float(r["ratio"]))
                for r in rows33 if r["sweep"] == "length")
    axC.plot([x for x, _ in lv], [y for _, y in lv], "D-", color="#cc78bc",
             label="prod full cells, untapered")
    for s, kw in SERIES_STYLE.items():
        pts = sorted((r["length_m"] * 1e9, r["G"] / ball_G, r["conv"])
                     for (lab, p, ser), r in runs.items()
                     if ser == s)
        axC.plot([p[0] for p in pts], [p[1] for p in pts], kw["ls"],
                 marker=kw["marker"], color=kw["color"], ms=5,
                 label=kw["label"])
    axC.set_xlabel("device length (nm)")
    axC.set_ylabel(r"r = G$_{\rm anh}$/G$_{\rm ball}$ at 300 K")
    axC.set_ylim(0, 1.0)
    axC.legend(fontsize=6)
    axC.set_title("anharmonic suppression vs length (eta=0)")
    style.save(fig, "atlas_tubes_T", directory=OUT)


def summary_table(runs, ball):
    print("\n===== CNT (3,3) ladder, physical units (300 K, dT=10 K) =====")
    print(f"{'run':10} {'len(nm)':>8} {'status':>8} {'iter':>5} "
          f"{'G(nW/K)':>8} {'G/G_ball':>9} {'kappa_eff(W/mK)':>16}")
    print(f"{'ballistic':10} {ball['length_m']*1e9:8.2f} {'exact':>8} "
          f"{ball['n_iter']:5d} {ball['G']*1e9:8.3f} {1.0:9.3f} "
          f"{ball['G']*ball['length_m']/A_T33:16.2f}")
    for (lab, p, ser), r in sorted(runs.items(),
                                   key=lambda kv: (kv[0][2], kv[1]["length_m"])):
        st = "conv" if r["conv"] else "CAP"
        print(f"{lab+'_'+ser:10} {r['length_m']*1e9:8.2f} {st:>8} "
              f"{r['n_iter']:5d} {r['G']*1e9:8.3f} {r['G']/ball['G']:9.3f} "
              f"{r['G']*r['length_m']/A_T33:16.2f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ball = load_run(BALL)
    runs = {}
    for lab, p, ser in LADDER:
        runs[(lab, p, ser)] = load_run(p)
    summary_table(runs, ball)
    fig_ladder(runs, ball)
    fig_teff(runs, ball)
    fig_spectral_current(runs, ball)
    fig_local(runs)
    fig_conservation(runs)
    fig_stability()
    fig_tubes_T(runs)
    print(f"\nfigures -> {OUT}")


if __name__ == "__main__":
    main()
