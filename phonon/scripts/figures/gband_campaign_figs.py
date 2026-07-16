"""Wrap-up figures for the g_band=2 (exact bubble kernel) campaign.

Reads the local mirrors (cluster/<name>/run.log, phonon/studies/out/
anderson_test/*) and writes PNG+PDF into
phonon/studies/out/anderson_test/campaign_report/fig/.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CL = ROOT / "cluster"
A = ROOT / "phonon/studies/out/anderson_test"
FIG = A / "campaign_report/fig"
FIG.mkdir(parents=True, exist_ok=True)

RE_RES = re.compile(r"rel Sigma\^R residual ([0-9.e+-]+)")


def residuals(log: Path, segment: int = 0) -> np.ndarray:
    """Residual history; ``segment`` selects a chained-run section."""
    text = log.read_text(errors="ignore")
    if segment:
        parts = text.split("RUN config=")
        text = parts[segment + 1] if len(parts) > segment + 1 else ""
    return np.array([float(m) for m in RE_RES.findall(text)])


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=160)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("saved", name, flush=True)


# ---------------------------------------------------------------- W1
def w1_length_series():
    conv = {2: (222, 44.20), 3: (209, 38.86), 4: (311, 35.22),
            5: (304, 32.44), 6: (241, 30.73), 7: (314, 28.07)}
    div = {8: 63, 10: 119}
    J_BALL = 77.669

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7),
                             constrained_layout=True)
    ax = axes[0]
    series = [("L3", CL / "cnt-L3-gband2/run.log", 0, "tab:purple"),
              ("L4", CL / "cnt-L4-gband2/run.log", 0, "tab:blue"),
              ("L5", CL / "cnt-L5-gband2/run.log", 0, "tab:cyan"),
              ("L6", CL / "cnt-L56-chain/run.log", 1, "tab:green"),
              ("L7", CL / "cnt-L7-gband2/run.log", 0, "tab:olive"),
              ("L8", CL / "cnt-L8-gband2/run.log", 0, "tab:red"),
              ("L10", CL / "cnt-L10-gband2/run.log", 0, "tab:brown")]
    for lab, log, seg, c in series:
        if not log.exists():
            continue
        r = residuals(log, seg)
        ls = "--" if lab in ("L8", "L10") else "-"
        ax.semilogy(r, ls, color=c, lw=1.2, label=lab)
    ax.axhline(1e-3, color="#888888", lw=0.8, ls=":")
    ax.set_ylim(1e-4, 50)
    ax.set_xlabel("SCBA iteration")
    ax.set_ylabel(r"rel. $\Sigma^R$ residual")
    ax.set_title("exact kernel: residual histories", fontsize=10)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    Ls = sorted(conv)
    ax.bar([str(x) for x in Ls], [conv[x][0] for x in Ls],
           color="tab:blue")
    for x, lab in ((str(8), "div. it 63"), (str(10), "div. it 119")):
        ax.bar(x, 450, color="tab:red", alpha=0.35)
        ax.text(x, 300, lab, ha="center", fontsize=8, rotation=90)
    ax.set_xlabel("transport cells $L$")
    ax.set_ylabel("iterations to tolerance")
    ax.set_title("convergence cost vs length", fontsize=10)

    ax = axes[2]
    ax.plot(Ls, [conv[x][1] / J_BALL for x in Ls], "o-",
            color="tab:blue", label="exact kernel")
    legacy = {2: 0.569, 3: 0.501, 4: 0.480}
    ax.plot(list(legacy), list(legacy.values()), "s--", color="tab:gray",
            alpha=0.8, label="masked kernel (superseded)")
    ax.axvspan(7.5, 10.5, color="tab:red", alpha=0.12)
    ax.text(8.9, 0.52, "lattice-\nunstable", ha="center", fontsize=8,
            color="tab:red")
    ax.set_xlabel("transport cells $L$")
    ax.set_ylabel(r"$r = J/J_{\rm ball}$")
    ax.set_title("anharmonic suppression vs length", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("CNT(3,3), 300 K, exact bubble kernel (sse_g_band = 2): "
                 "length series", fontsize=11)
    save(fig, "w1_g2_length_series")


# ---------------------------------------------------------------- W2
def w2_legacy_vs_g2():
    fig, ax = plt.subplots(figsize=(7.6, 3.9), constrained_layout=True)
    lin02 = A / "mixer_campaign_L4_b005/lin02/run.log"
    if lin02.exists():
        r = residuals(lin02)
        ax.semilogy(r, color="tab:gray", lw=1.0,
                    label="masked kernel (g_band=1): 1200 its, best "
                          r"$3.1\times10^{-3}$")
    r2 = residuals(CL / "cnt-L4-gband2/run.log")
    ax.semilogy(r2, color="tab:blue", lw=1.3,
                label="exact kernel (g_band=2): converged at 311")
    ax.axhline(1e-3, color="#888888", lw=0.8, ls=":")
    ax.set_xlabel("SCBA iteration")
    ax.set_ylabel(r"rel. $\Sigma^R$ residual")
    ax.set_ylim(5e-4, 30)
    ax.set_title("CNT L4, linear $\\alpha=0.2$: the sawtooth transient is "
                 "the masked kernel's non-causal gain", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "w2_l4_legacy_vs_g2")


# ---------------------------------------------------------------- W3
def w3_mechanism():
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7),
                             constrained_layout=True)
    ax = axes[0]
    slabs = ["edge 0", "int 1", "int 2", "edge 3"]
    vals = [0.0, -1.57e3, -1.57e3, 0.0]
    ax.bar(slabs, vals, color=["tab:gray", "tab:red", "tab:red",
                               "tab:gray"])
    ax.set_ylabel(r"min eig of $-i(\Sigma^>-\Sigma^<)_{II}$")
    ax.set_title("first Born, masked kernel (L4, production):\n"
                 "interior slabs are non-causal", fontsize=9)
    ax = axes[1]
    x = np.arange(3)
    ax.bar(x - 0.15, [-1.054e4, 0, 0], width=0.3, color="tab:red",
           label="interior slab")
    ax.bar(x + 0.15, [0, 0, 0], width=0.3, color="tab:gray",
           label="edge slab")
    ax.set_xticks(x, ["band 1", "band 2", "full G"])
    ax.set_ylabel("min eig (dense validation)")
    ax.set_title("completing the inner G band restores\n"
                 "positivity exactly (band 2 = full for NN vertex)",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[2]
    bins = ["0.6", "3.1", "9.2", "18.3", "36.7"]
    ax.bar(np.arange(5) - 0.15, [3.49, 2.11, 3.15, 1.00, 5.17], width=0.3,
           color="tab:red", label="interior slab")
    ax.bar(np.arange(5) + 0.15, [0, 0, 0, 0, 0], width=0.3,
           color="tab:gray", label="edge slab (exactly 0)")
    ax.set_xticks(np.arange(5), bins)
    ax.set_xlabel(r"$\omega$ (THz)")
    ax.set_ylabel(r"$|\Delta\Sigma^<_{II}|/|\Sigma^<_{II}|$")
    ax.set_title("the correction is O(1) on interior slabs\n"
                 "(production first Born, band 2 vs band 1)", fontsize=9)
    fig.suptitle("The g_band mechanism: a masked PSD kernel is not PSD "
                 "(Schur), and the mask completes at band 2", fontsize=11)
    save(fig, "w3_gband_mechanism")


# ---------------------------------------------------------------- W4
def w4_ne_scan():
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    data = [(161, "diverged", 77), (181, "sawtooth crawl", 1200),
            (201, "converged", 249), (271, "budget, unconverged", 350),
            (361, "diverged", 34)]
    colors = {"diverged": "tab:red", "converged": "tab:green",
              "sawtooth crawl": "tab:orange",
              "budget, unconverged": "tab:gray"}
    for ne, fate, its in data:
        ax.bar(str(ne), its, color=colors[fate])
        ax.text(str(ne), its + 25, fate, ha="center", fontsize=7.5,
                rotation=0)
    ax.set_xlabel(r"frequency grid points $n_e$")
    ax.set_ylabel("iterations run")
    ax.set_ylim(0, 1400)
    ax.set_title("masked kernel, CNT L4: fate vs grid density "
                 "(alignment lottery)", fontsize=10)
    save(fig, "w4_ne_scan")


# ---------------------------------------------------------------- W5
def w5_l10_forensics():
    fig = plt.figure(figsize=(13.6, 3.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.5, 1, 1))
    ax = fig.add_subplot(gs[0])
    d = np.load(A / "cnt33_L10_linear/run_gband2.npz", allow_pickle=True)
    sm = np.asarray(d["iter_sigma_max"], float)
    en = np.asarray(d["energies"], float)
    m = ax.imshow(np.log10(np.maximum(sm.T, 1e-3)), aspect="auto",
                  origin="lower", cmap="inferno",
                  extent=(0, sm.shape[0], en[0], en[-1]))
    ax.set_ylim(0, 12)
    ax.set_xlabel("SCBA iteration")
    ax.set_ylabel(r"$\omega$ (THz)")
    ax.set_title(r"L10, exact kernel: $\log_{10}\max|\Sigma^<|$ per bin "
                 "-- IR seeds, then cascade", fontsize=9)
    fig.colorbar(m, ax=ax, shrink=0.85)
    ax = fig.add_subplot(gs[1])
    lad = [("plain", 119), ("IR floor\n+$\\alpha$=0.1", 51),
           ("dressed\ncontacts", 23), ("vertex\nramp", 91),
           ("SCP\ntadpole", 139)]
    ax.bar([x[0] for x in lad], [x[1] for x in lad], color="tab:red")
    ax.set_ylabel("divergence iteration")
    ax.set_title("L10 stabilizer ladder: all fail\n(model, not scheme)",
                 fontsize=9)
    ax.tick_params(axis="x", labelsize=7.5)
    ax = fig.add_subplot(gs[2])
    bins = ["0.31", "0.61", "0.92", "1.53"]
    ax.bar(np.arange(4) - 0.15, [-282.2, -217.8, -115.7, -25.8], width=0.3,
           color="tab:orange", label="edge slab")
    ax.bar(np.arange(4) + 0.15, [-91.8, -76.2, -52.4, -24.8], width=0.3,
           color="tab:red", label="interior slab")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(4), bins)
    ax.set_xlabel(r"$\omega$ (THz)")
    ax.set_ylabel(r"min eig of $h_{00}+\mathrm{Re}\,\Sigma^R_{II}$ "
                  r"(THz$^2$)")
    ax.set_title("soft-mode collapse at it 85:\nmodes pushed through "
                 r"$\omega^2=0$", fontsize=9)
    ax.legend(fontsize=8)
    save(fig, "w5_l10_forensics")


# ---------------------------------------------------------------- W6
def w6_contacts():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7),
                             constrained_layout=True)
    ax = axes[0]
    r_ideal = residuals(CL / "cnt-L4-gband2/run.log")
    r_scat = residuals(CL / "cnt-L4-g2scat/run.log")
    ax.semilogy(r_ideal, color="tab:blue", lw=1.1,
                label="ideal reservoirs (converged 311)")
    ax.semilogy(r_scat, color="tab:purple", lw=1.1,
                label=r"dressed contacts (1.06$\times10^{-3}$ at 450)")
    ax.axhline(1e-3, color="#888888", lw=0.8, ls=":")
    ax.set_xlabel("SCBA iteration")
    ax.set_ylabel(r"rel. $\Sigma^R$ residual")
    ax.set_title("L4, exact kernel: contact models", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    x = np.arange(5)
    ideal = [35.22, 32.33, 32.50, 32.40, 35.22]
    dressed = [45.58, 41.06, 40.73, 42.14, 45.59]
    ax.plot(x, ideal, "o-", color="tab:blue", label="ideal reservoirs")
    ax.plot(x, dressed, "s-", color="tab:purple",
            label="dressed contacts (best-conserved)")
    ax.set_xticks(x, ["lead L", "1", "2", "3", "lead R"])
    ax.set_ylabel("interface heat current (W)")
    ax.set_title("contact model moves J by ~25-30%", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("GW-ordering (scattering-dressed) contacts vs ideal "
                 "ballistic reservoirs", fontsize=11)
    save(fig, "w6_contact_model")


# ---------------------------------------------------------------- W7
def w7_mixers():
    fig, ax = plt.subplots(figsize=(7.6, 3.9), constrained_layout=True)
    for lab, log, c in (
            ("linear $\\alpha$=0.2 (converged 311)",
             CL / "cnt-L4-gband2/run.log", "tab:blue"),
            ("RRE c12, build $\\beta$=0.3 (stalls)",
             CL / "cnt-L4-g2rre/run.log", "tab:orange"),
            ("RRE c12, build $\\beta$=0.2 (stalls)",
             CL / "cnt-L4-g2rre-b02/run.log", "tab:red")):
        if not log.exists():
            continue
        r = residuals(log, segment=0)
        # the first g2rre log contains a failed-launch stub; take the
        # longest RUN segment
        if "g2rre" in str(log):
            segs = [residuals(log, segment=i) for i in range(3)]
            segs = [s for s in segs if s.size]
            r = max(segs, key=len) if segs else r
        ax.semilogy(r, color=c, lw=1.1, label=lab)
    ax.axhline(1e-3, color="#888888", lw=0.8, ls=":")
    ax.set_xlabel("SCBA iteration")
    ax.set_ylabel(r"rel. $\Sigma^R$ residual")
    ax.set_title("L4, exact kernel: plain linear still beats RRE",
                 fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "w7_g2_mixers")


w1_length_series()
w2_legacy_vs_g2()
w3_mechanism()
w4_ne_scan()
w5_l10_forensics()
w6_contacts()
w7_mixers()
print("all wrap-up figures done")
