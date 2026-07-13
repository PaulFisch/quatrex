"""Figures for the synthetic grid-convergence study (toy_grid E1-E7)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "out/toy_grid"
FIG = OUT / "fig"
FIG.mkdir(parents=True, exist_ok=True)

R = json.loads((OUT / "results.json").read_text())["rows"]
REF = json.loads((OUT / "results_ref.json").read_text())["rows"]
E6 = json.loads((OUT / "results_e6.json").read_text())["rows"]
E7 = json.loads((OUT / "results_e7.json").read_text())["results"]


def rows(exp, src=None):
    return [r for r in (R if src is None else src) if r.get("exp") == exp]


# ---------------------------------------------------------------- F1
def f1():
    e1 = rows("E1")
    gs = sorted({r["g"] for r in e1})
    titles = {gs[0]: r"$\Gamma_{\rm FB}=0.02$ THz",
              gs[1]: r"$\Gamma_{\rm FB}=0.2$ THz",
              gs[2]: r"$\Gamma_{\rm FB}=2.0$ THz"}
    # grid-converged references
    ref_weak = [r for r in REF if r["exp"] == "E1ref"
                and abs(r["g"] - gs[0]) < 1e9 and r["nfreq"] == 3840]
    ref_mid = [r for r in REF if r["exp"] == "E1ref"
               and abs(r["g"] - gs[1]) < 1e9 and r["nfreq"] == 1920]
    warm = [r for r in REF if r.get("exp") == "E1warm"]
    ref_strong = ([w for w in warm if w["nfreq"] == 1920
                   and w.get("converged")] or warm[-1:] if warm else [])
    refs = {gs[0]: (ref_weak[0]["gamma_em"] if ref_weak else None),
            gs[1]: (ref_mid[0]["gamma_em"] if ref_mid else None),
            gs[2]: (ref_strong[0]["gamma_em"] if ref_strong else None)}

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, g in zip(axes[:3], gs):
        rr = sorted([r for r in e1 if r["g"] == g], key=lambda r: r["nfreq"])
        nf = [r["nfreq"] for r in rr]
        gm = [r["gamma_em"] for r in rr]
        conv = [r["converged"] for r in rr]
        ax.plot(nf, gm, "o-", color="tab:blue", ms=6)
        for x, y, c in zip(nf, gm, conv):
            if not c:
                ax.plot(x, y, "x", color="tab:red", ms=11, mew=2.2)
        # fine-grid refs for this coupling: cold starts vs warm chain
        for expn, col, style in (("E1ref", "tab:green", "s--"),
                                 ("E1warm", "tab:purple", "^-")):
            fr = sorted([r for r in REF if abs(r["g"] - g) < 1e9
                         and r["exp"] == expn], key=lambda r: r["nfreq"])
            if not fr:
                continue
            ax.plot([r["nfreq"] for r in fr], [r["gamma_em"] for r in fr],
                    style, color=col, ms=6, alpha=0.85,
                    label={"E1ref": "cold, fine grid",
                           "E1warm": "warm chain"}[expn])
            for r in fr:
                bad = ((not r.get("converged", True)
                        and r.get("jitter", 0) > 0.5)
                       or r.get("n_it", 99) <= 1)
                if bad and expn == "E1ref":
                    ax.plot(r["nfreq"], r["gamma_em"], "x", color="tab:red",
                            ms=11, mew=2.2)
        if g == gs[2]:
            ax.legend(fontsize=7)
        if refs[g]:
            ax.axhline(refs[g], color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel(r"$n_f$")
        ax.set_title(titles[g], fontsize=10)
        ax.set_ylabel(r"$\Gamma_{\rm em}$ (THz)")
    # collapse: relative error vs dw / Gamma_ref
    ax = axes[3]
    for g, c, lbl in zip(gs, ("tab:blue", "tab:orange", "tab:brown"),
                         ("0.02", "0.2", "2.0")):
        if not refs[g]:
            continue
        rr = sorted([r for r in e1 if r["g"] == g and r["converged"]],
                    key=lambda r: r["nfreq"])
        x = [r["dw"] / refs[g] for r in rr]
        y = [abs(r["gamma_em"] - refs[g]) / refs[g] for r in rr]
        ax.loglog(x, y, "o-", color=c, label=rf"$\Gamma$={lbl}")
    xs = np.array([0.2, 60.0])
    ax.loglog(xs, 0.12 * xs, "k:", lw=0.9)
    ax.text(2.5, 0.6, r"$\propto\Delta\omega/\Gamma$", fontsize=9)
    ax.set_xlabel(r"$\Delta\omega/\Gamma_{\rm em}^{\rm ref}$")
    ax.set_ylabel(r"$|\Gamma_{\rm em}-\Gamma^{\rm ref}|/\Gamma^{\rm ref}$")
    ax.legend(fontsize=8)
    ax.set_title("relative linewidth error", fontsize=10)
    fig.suptitle("E1: iteration converges; the linewidth is grid-converged "
                 r"only for $\Delta\omega \lesssim \Gamma$ "
                 "(crosses: not converged / false convergence)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "toy_f1_linewidth.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- F2
def f2():
    e2 = sorted(rows("E2"), key=lambda r: r["frac"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    ax.plot([r["frac"] for r in e2], [r["gamma_em"] for r in e2], "o-",
            color="tab:blue")
    ax.set_xlabel("pole position within one grid cell (frac)")
    ax.set_ylabel(r"$\Gamma_{\rm em}$ (THz)")
    ax.set_title(r"linewidth vs sub-bin pole position "
                 r"($\Delta\omega/\Gamma\approx 2.9$)", fontsize=10)
    gm = [r["gamma_em"] for r in e2]
    ax.annotate(f"swing {100 * (max(gm) - min(gm)) / np.mean(gm):.0f}%",
                xy=(0.55, 0.1), xycoords="axes fraction", fontsize=9)
    ax = axes[1]
    ax.plot([r["frac"] for r in e2], [r["rate"] for r in e2], "s-",
            color="tab:orange")
    ax.set_xlabel("pole position within one grid cell (frac)")
    ax.set_ylabel("residual contraction rate")
    ax.set_title("convergence rate vs alignment", fontsize=10)
    fig.suptitle("E2: sub-bin alignment of a flat-band pole moves the "
                 "converged linewidth and the convergence rate", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG / "toy_f2_alignment.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- F3
def f3():
    chain = [r for r in E6 if r["tag"].startswith("eps=")]
    cold = [r for r in E6 if r["tag"] == "cold-control"]
    ladder = sorted([r for r in E6 if r["tag"] == "warm-ladder"],
                    key=lambda r: r["nfreq"])
    e5 = sorted([r for r in rows("E5") if r["gamma_t"] == 0.2],
                key=lambda r: r["nfreq"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    x = np.arange(len(chain) + 1)
    vals = [r["gamma_em_B2"] for r in chain] + [cold[0]["gamma_em_B2"]]
    labs = [r["tag"] for r in chain] + ["cold\n(same system)"]
    cols = ["tab:blue"] * len(chain) + ["tab:gray"]
    ax.bar(x, np.maximum(vals, 1e-12), color=cols)
    ax.set_yscale("log")
    ax.set_xticks(x, labs, fontsize=8)
    ax.set_ylabel(r"$\Gamma_{\rm em}(B_2)$ (THz)")
    ax.set_title("contact-broadening continuation vs cold start "
                 "(nf=480): two fixed points", fontsize=10)
    ax = axes[1]
    seed = [r for r in chain if r["tag"] == "eps=0.02"]
    lad = ladder + [dict(nfreq=480, gamma_em_B2=seed[0]["gamma_em_B2"],
                         gamma_em_B1=seed[0]["gamma_em_B1"])]
    lad = sorted(lad, key=lambda r: r["nfreq"])
    nfs = [r["nfreq"] for r in lad]
    gm = [max(abs(r["gamma_em_B2"]), 1e-22) for r in lad]
    ax.plot(nfs, gm, "o-", color="tab:blue", label="warm-laddered branch")
    cold5 = [max(r["gamma_em_B2"], 1e-22) for r in e5]
    ax.plot([r["nfreq"] for r in e5], cold5, "s--", color="tab:gray",
            label="cold start")
    for r in lad:
        if r["gamma_em_B1"] < 0 or r["gamma_em_B2"] > 5:
            ax.annotate("unphysical\n($\\Gamma_{B_1}<0$)",
                        xy=(r["nfreq"], abs(r["gamma_em_B2"])),
                        xytext=(r["nfreq"] * 1.5, 1.0), fontsize=8,
                        arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_f$")
    ax.set_ylabel(r"$\Gamma_{\rm em}(B_2)$ (THz)")
    ax.axhline(0.2, color="k", lw=0.8, ls=":")
    ax.text(33, 0.3, r"$\Gamma$ target", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("scattering branch vs grid: sustained / unphysical / "
                 "deleted", fontsize=10)
    fig.suptitle("E5+E6: a sharp-sharp (flat-band pair) channel is grid- "
                 "and history-selected", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG / "toy_f3_branches.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- F4
def f4():
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=0.8)
    # damped-stability disk for alpha = 0.2: |lam + 4| < 5
    ax.plot(-4 + 5 * np.cos(th), 5 * np.sin(th), "k--", lw=0.8)
    ax.axvline(1.0, color="tab:red", lw=1.2, ls="-.")
    ax.text(1.08, 4.2, r"Re$\,\lambda = 1$: no damping $\alpha$ converges",
            fontsize=9, color="tab:red", rotation=90, va="top")
    ax.text(-8.6, 3.4, r"$|\lambda+4|=5$ ($\alpha=0.2$ stable)", fontsize=8)
    mark = {"weak": ("o", "tab:blue"),
            "strong": ("s", "tab:orange"),
            "sharp-pair-branch": ("D", "tab:red"),
            "sharp-pair-ballistic": ("v", "tab:gray")}
    seen = set()
    for rec in E7:
        m, c = mark[rec["case"]]
        lbl = {"weak": "broad bath, weak (nf=480)",
               "strong": f"broad bath, strong",
               "sharp-pair-branch": "sharp pair, scattering branch",
               "sharp-pair-ballistic": "sharp pair, ballistic branch"}[
                   rec["case"]]
        if rec["case"] == "strong":
            lbl += f" (nf={rec['nfreq']})"
            c = {240: "#ffd0a0", 480: "tab:orange", 960: "#a04000"}[
                rec["nfreq"]]
        for t in rec["top"][:4]:
            ax.plot(t["re"], t["im"], m, color=c, ms=7,
                    label=lbl if lbl not in seen else None)
            seen.add(lbl)
    ax.set_xlabel(r"Re$\,\lambda$")
    ax.set_ylabel(r"Im$\,\lambda$")
    ax.set_title("E7: measured Jacobian spectra of the toy SCBA map\n"
                 "(top Arnoldi eigenvalues at each fixed point / best "
                 "iterate)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "toy_f4_spectra.png", dpi=160)
    plt.close(fig)


f1()
f2()
f3()
f4()
print("figures written to", FIG)
