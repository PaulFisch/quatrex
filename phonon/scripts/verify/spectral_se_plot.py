#!/usr/bin/env python
"""Aggregate the spectral_se grid into with-/without-self-energy figures.

For every (structure, T, mode) npz, build the zone-centre spectral function
    A(Gamma, omega) = -1/pi Im Tr [(omega+i eta)^2 I - D - Sigma_static - Sigma_B]^{-1}
WITHOUT the self-energy (bare D) and WITH it, and lay them out as
rows = self-energy mode, cols = temperature, one figure per structure.
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

SRC = Path("/tmp/claude/spectral_se")
OUT = _REPO / "document/fig/transport_sweeps"
MODES = ["bubble_half", "bubble_fft", "scp_fft"]
MODE_LABEL = {"bubble_half": "bubble, no KK (half)",
              "bubble_fft": "bubble + KK (fft)",
              "scp_fft": "bubble + KK + tadpole"}


def A_of(D, grid, eta_w, sb=None, ss=None):
    return spectral_function_qw(
        D[None], grid, eta_w,
        sigma_static=(None if ss is None or not np.any(ss) else ss[None]),
        sigma_b=(None if sb is None else sb[None]))[0]


def main():
    files = sorted(glob.glob(str(SRC / "*.npz")))
    if not files:
        print("no npz yet"); return
    data = {}
    print(f"{'struct':6} {'T':>5} {'mode':14} {'conv':5} {'resid':>9} "
          f"{'cons':>9} {'Ga/Gb':>6}")
    for f in files:
        d = np.load(f, allow_pickle=True)
        key = (str(d["struct"]), float(d["temp"]), str(d["mode"]))
        data[key] = d
        print(f"{key[0]:6} {key[1]:5.0f} {key[2]:14} "
              f"{str(bool(d['converged'])):5} {float(d['resid']):9.2e} "
              f"{float(d['conservation']):9.2e} {float(d['Ga_over_Gb']):6.3f}")

    structs = sorted({k[0] for k in data})
    temps = sorted({k[1] for k in data})
    for s in structs:
        fig, axes = plt.subplots(len(MODES), len(temps),
                                 figsize=(3.4 * len(temps), 2.8 * len(MODES)),
                                 squeeze=False)
        fmax = None
        for r, mode in enumerate(MODES):
            for c, T in enumerate(temps):
                ax = axes[r][c]
                d = data.get((s, T, mode))
                if d is None:
                    ax.set_axis_off(); continue
                fmax = float(d["fmax"])
                freqs = np.asarray(d["freqs"]); D = np.asarray(d["D"])
                sb = np.asarray(d["sigma_b"]); ss = np.asarray(d["sigma_static"])
                eta_w = 1.5 * (freqs[1] - freqs[0])
                ax.semilogy(freqs, A_of(D, freqs, eta_w) + 1e-6, "k-", lw=0.8,
                            label="bare D")
                finite = np.isfinite(sb).all() and np.isfinite(ss).all()
                if finite:
                    ax.semilogy(freqs, A_of(D, freqs, eta_w, sb, ss) + 1e-6,
                                color="tab:red", lw=0.9, label="with $\\Sigma$")
                else:
                    ax.text(0.5, 0.5, "diverged\n(NaN $\\Sigma$)", ha="center",
                            va="center", color="tab:red", fontsize=8,
                            transform=ax.transAxes)
                if not bool(d["converged"]):
                    ax.text(0.97, 0.03, "not conv.", ha="right", va="bottom",
                            color="tab:orange", fontsize=6, transform=ax.transAxes)
                if r == 0:
                    ax.set_title(f"T = {T:.0f} K", fontsize=9)
                if c == 0:
                    ax.set_ylabel(f"{MODE_LABEL[mode]}\n$A(\\Gamma,\\omega)$",
                                  fontsize=7)
                if r == len(MODES) - 1:
                    ax.set_xlabel(r"$\omega$ [THz]", fontsize=8)
                ax.set_xlim(0, fmax)
                ax.tick_params(labelsize=6)
                if r == 0 and c == 0:
                    ax.legend(fontsize=6, loc="upper right")
        fig.suptitle(f"{s}: spectral function with vs without self-energy",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"spectral_se_{s}.{ext}", dpi=130,
                        bbox_inches="tight")
        print(f"wrote {OUT / f'spectral_se_{s}.pdf'}")


if __name__ == "__main__":
    main()
