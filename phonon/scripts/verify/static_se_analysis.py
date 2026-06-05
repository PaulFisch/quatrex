#!/usr/bin/env python
"""Comprehensive analysis + plots of the static-correction study (loop / tadpole
/ bubble), and the PHYSICAL verification of the tadpole breakdown.

Tadpole physics (Paulatto-Errea-Calandra, PRB 91 054304): the optical tadpole
T_O is the GRADIENT that relaxes the internal coordinates; at the relaxed
structure (where <F>=0) it VANISHES. Used as a perturbative self-energy it is
valid only while |Sigma| << omega^2; when |Sigma_T| approaches omega^2 it drives
modes imaginary and the propagator stops conserving current -- exactly the
breakdown criterion |Pi| >~ omega. We verify this directly from the data.

Figures (document/fig/transport_sweeps/):
  static_se_tadpole_breakdown.pdf : ||Sigma_T||, ||Sigma_L|| vs T + imaginary-mode
                                    / non-convergence markers + validity ratio
  static_se_transport.pdf         : G_anh/G_ball and conservation vs T, per mode
  static_se_spectral.pdf          : A(Gamma,omega) bare vs +each correction
  static_se_current.pdf           : spectral heat current J(omega) +/- corrections
  static_se_bubble_shape.pdf      : Re/Im Tr Sigma_B(omega) (bubble self-energy)
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from postproc.spectral import spectral_function_qw, frequencies_from_dynamical

SRC = Path(__file__).resolve().parents[2] / "scripts/out/snapshots"
OUT = _REPO / "document/fig/transport_sweeps"
MODES = ["bubble", "loop", "tadpole", "loop_tadpole"]
MCOL = {"bubble": "k", "loop": "tab:blue", "tadpole": "tab:red",
        "loop_tadpole": "tab:green"}


def load():
    d = {}
    for f in glob.glob(str(SRC / "study_*.npz")):
        z = np.load(f, allow_pickle=True)
        d[(str(z["struct"]), float(z["temp"]), str(z["mode"]))] = z
    return d


def main():
    data = load()
    structs = sorted({k[0] for k in data})
    temps = sorted({k[1] for k in data})

    # --- (b) tadpole breakdown + validity ----------------------------------
    fig, ax = plt.subplots(1, len(structs), figsize=(5.4 * len(structs), 4.4),
                           squeeze=False)
    for c, s in enumerate(structs):
        Ts = [T for T in temps if (s, T, "tadpole") in data]
        a = ax[0][c]
        sigT = [float(data[(s, T, "tadpole")]["sigma_static_norm"]) for T in Ts]
        sigL = [float(data[(s, T, "loop")]["sigma_static_norm"])
                if (s, T, "loop") in data else np.nan for T in Ts]
        # omega^2 of the lowest device mode (validity scale)
        w2min = []
        for T in Ts:
            z = data[(s, T, "tadpole")]
            sb = float(z["soft_bare"]); w2min.append(sb * sb)
        a.semilogy(Ts, sigT, "rs-", label=r"$\|\Sigma_T\|$ (tadpole)")
        a.semilogy(Ts, sigL, "bo-", label=r"$\|\Sigma_L\|$ (loop)")
        a.semilogy(Ts, w2min, "k--", label=r"$\omega^2_{\rm low}$ (validity scale)")
        # mark imaginary-mode / non-converged tadpole points
        for T in Ts:
            z = data[(s, T, "tadpole")]
            if float(z["soft_ren"]) < 0 or not bool(z["converged"]):
                a.scatter([T], [float(z["sigma_static_norm"])], s=160,
                          facecolors="none", edgecolors="red", linewidths=2,
                          zorder=5)
        a.set_title(f"{s}: tadpole breakdown\n(circled = imaginary mode / not conv.)")
        a.set_xlabel("T [K]"); a.set_ylabel(r"magnitude [THz$^2$]")
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "static_se_tadpole_breakdown.pdf", dpi=140, bbox_inches="tight")
    fig.savefig(OUT / "static_se_tadpole_breakdown.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- (a) transport: Ga/Gb + conservation vs T per mode -----------------
    fig, ax = plt.subplots(2, len(structs), figsize=(5.4 * len(structs), 8), squeeze=False)
    for c, s in enumerate(structs):
        Ts = [T for T in temps if (s, T, "bubble") in data]
        for m in MODES:
            ga = [float(data[(s, T, m)]["Ga_over_Gb"]) if (s, T, m) in data else np.nan for T in Ts]
            cons = [float(data[(s, T, m)]["conservation"]) if (s, T, m) in data else np.nan for T in Ts]
            conv = [bool(data[(s, T, m)]["converged"]) if (s, T, m) in data else False for T in Ts]
            ax[0][c].plot(Ts, ga, "o-", color=MCOL[m], label=m)
            ax[1][c].semilogy(Ts, cons, "o-", color=MCOL[m], label=m)
            for T, g, cv in zip(Ts, ga, conv):
                if not cv:
                    ax[0][c].scatter([T], [g], marker="x", color=MCOL[m], s=70, zorder=5)
        ax[0][c].axhline(1.0, color="0.6", lw=0.6)
        ax[0][c].set_title(f"{s}: $G_{{anh}}/G_{{ball}}$ (x = not conv.)")
        ax[0][c].set_xlabel("T [K]"); ax[0][c].set_ylabel(r"$G_{anh}/G_{ball}$"); ax[0][c].legend(fontsize=7); ax[0][c].grid(alpha=0.3)
        ax[1][c].set_title(f"{s}: heat-flow conservation"); ax[1][c].set_xlabel("T [K]")
        ax[1][c].set_ylabel("conservation err"); ax[1][c].legend(fontsize=7); ax[1][c].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "static_se_transport.pdf", dpi=140, bbox_inches="tight")
    fig.savefig(OUT / "static_se_transport.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # --- (a) spectral A(Gamma,omega) + current J(omega) (needs device_D/J) --
    def has(z, key):
        return key in z.files
    for kind, fname, ylab in [("A", "static_se_spectral", r"$A(\Gamma,\omega)$"),
                              ("J", "static_se_current", r"$J(\omega)$ [a.u.]")]:
        Tshow = [T for T in temps if T in (30.0, 300.0)]
        fig, ax = plt.subplots(len(Tshow), len(structs),
                               figsize=(5.4 * len(structs), 3.6 * len(Tshow)), squeeze=False)
        any_plot = False
        for r, T in enumerate(Tshow):
            for c, s in enumerate(structs):
                a = ax[r][c]
                zb = data.get((s, T, "bubble"))
                if zb is None or not has(zb, "device_D"):
                    a.set_axis_off(); continue
                D = np.asarray(zb["device_D"]); freqs = np.asarray(zb["freqs"])
                eta_w = 1.5 * (freqs[1] - freqs[0])
                # band edge for a sensible x-limit (grid extends to the full
                # bubble-convolution range, ~2x the band)
                wmax = 1.08 * float(np.sqrt(np.abs(np.linalg.eigvalsh(
                    0.5 * (D + D.T)).max())))
                if kind == "J" and has(zb, "J_ball"):
                    a.plot(freqs, np.asarray(zb["J_ball"]), "0.6", lw=1.2, label="ballistic")
                for m in MODES:
                    z = data.get((s, T, m))
                    if z is None:
                        continue
                    if kind == "A":
                        if not (np.all(np.isfinite(z["sigma_b"])) and np.all(np.isfinite(z["sigma_static"]))):
                            continue
                        A = spectral_function_qw(D[None], freqs, eta_w,
                                                 sigma_static=np.asarray(z["sigma_static"])[None],
                                                 sigma_b=np.asarray(z["sigma_b"])[None])[0]
                        a.semilogy(freqs, A + 1e-6, color=MCOL[m], lw=0.9, label=m); any_plot = True
                    else:
                        if not has(z, "J_anh"):
                            continue
                        a.plot(freqs, np.asarray(z["J_anh"]), color=MCOL[m], lw=0.9, label=m); any_plot = True
                a.set_title(f"{s}, T={T:.0f} K"); a.set_xlabel(r"$\omega$ [THz]")
                a.set_ylabel(ylab); a.legend(fontsize=6); a.set_xlim(0, wmax)
        if any_plot:
            fig.tight_layout(); fig.savefig(OUT / f"{fname}.pdf", dpi=140, bbox_inches="tight")
            fig.savefig(OUT / f"{fname}.png", dpi=130, bbox_inches="tight")
            print(f"wrote {OUT / (fname + '.pdf')}")
        plt.close(fig)

    # --- (a) bubble self-energy spectral shape Re/Im Tr Sigma_B(omega) ------
    fig, ax = plt.subplots(1, len(structs), figsize=(5.4 * len(structs), 4.2), squeeze=False)
    for c, s in enumerate(structs):
        a = ax[0][c]
        for T in [t for t in temps if t in (30.0, 300.0, 600.0)]:
            z = data.get((s, T, "bubble"))
            if z is None:
                continue
            sb = np.asarray(z["sigma_b"]); fr = np.asarray(z["freqs"])
            trRe = np.array([np.trace(sb[i].real) for i in range(sb.shape[0])])
            trIm = np.array([np.trace(sb[i].imag) for i in range(sb.shape[0])])
            a.plot(fr, trRe, "-", label=f"Re, {T:.0f}K")
            a.plot(fr, trIm, "--", label=f"Im, {T:.0f}K")
        a.set_title(f"{s}: bubble Tr$\\,\\Sigma_B(\\omega)$")
        a.set_xlabel(r"$\omega$ [THz]"); a.set_ylabel(r"Tr$\,\Sigma_B$ [THz$^2$]")
        a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "static_se_bubble_shape.pdf", dpi=140, bbox_inches="tight")
    fig.savefig(OUT / "static_se_bubble_shape.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote tadpole_breakdown, transport, bubble_shape figures")


if __name__ == "__main__":
    main()
