#!/usr/bin/env python
"""Single-panel overlay of A(Gamma, omega): bare D vs the self-energy toggles,
built from the already-computed spectral_se grid npz (no re-run).

Top row  : A(Gamma,omega) at T=300 K -- bare D / +bubble(no KK) / +bubble(KK) /
           +bubble+tadpole all on ONE axes, so the KK real shift and the tadpole
           shift are directly visible against the bare diagonalisation of D.
Bottom   : q_z-resolved bands bare vs +Sigma_static (the q-independent tadpole
           shift added to D(q_z)); omitted where Sigma_static is NaN (d5a soft mode).
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
import warnings; warnings.filterwarnings("ignore")
from phonon.finite_analysis.loader import load_system
from postproc.spectral import (spectral_function_qw, frequencies_from_dynamical,
                               dynamical_matrix_qpath)

SRC = Path("/tmp/claude/spectral_se")
OUT = _REPO / "document/fig/transport_sweeps"
T_OVERLAY = 300
CFG = {"cnt33": "phonon/configs/cnt/cnt33_vasp.yaml",
       "d5a": "phonon/configs/sinw/sinw100_d5a_vasp_sc4_fc4.yaml",
       "d11a": "phonon/configs/sinw/sinw100_d11a_vasp_sc4.yaml"}
MODES = [("bubble_half", "tab:blue", "+bubble (no KK)"),
         ("bubble_fft", "tab:red", "+bubble (KK)"),
         ("scp_fft", "tab:green", "+bubble+tadpole")]


def load(struct, mode, T=T_OVERLAY):
    f = SRC / f"{struct}_T{T}_{mode}.npz"
    return np.load(f, allow_pickle=True) if f.exists() else None


def A_of(D, grid, eta_w, sb=None, ss=None):
    return spectral_function_qw(
        D[None], grid, eta_w,
        sigma_static=(None if ss is None or not np.any(ss) else ss[None]),
        sigma_b=(None if sb is None else sb[None]))[0]


def main():
    structs = [s for s in CFG if list(glob.glob(str(SRC / f"{s}_*.npz")))]
    fig, axes = plt.subplots(2, len(structs), figsize=(4.6 * len(structs), 8),
                             squeeze=False)
    for c, s in enumerate(structs):
        # reference D + grid from any available mode
        ref = next((load(s, m) for m, _, _ in MODES if load(s, m) is not None), None)
        if ref is None:
            continue
        D = np.asarray(ref["D"]); freqs = np.asarray(ref["freqs"])
        fmax = float(ref["fmax"]); eta_w = 1.5 * (freqs[1] - freqs[0])
        ax = axes[0][c]
        ax.semilogy(freqs, A_of(D, freqs, eta_w) + 1e-6, "k-", lw=1.1,
                    label="bare D")
        ss_finite = None
        for mode, col, lab in MODES:
            d = load(s, mode)
            if d is None:
                continue
            sb = np.asarray(d["sigma_b"]); ss = np.asarray(d["sigma_static"])
            if not (np.all(np.isfinite(sb)) and np.all(np.isfinite(ss))):
                ax.plot([], [], color=col, label=f"{lab} (NaN)")
                continue
            ax.semilogy(np.asarray(d["freqs"]),
                        A_of(D, np.asarray(d["freqs"]), eta_w, sb, ss) + 1e-6,
                        color=col, lw=1.0, label=lab)
            if mode == "scp_fft" and np.any(ss):
                ss_finite = ss
        ax.set_title(f"{s}: A($\\Gamma,\\omega$) at {T_OVERLAY} K")
        ax.set_xlabel(r"$\omega$ [THz]"); ax.set_ylabel(r"$A(\Gamma,\omega)$")
        ax.set_xlim(0, fmax); ax.legend(fontsize=7)

        # bands bare vs +static
        ax2 = axes[1][c]
        ph = load_system(str(_REPO / CFG[s]), validate=False,
                         transport_axis=2).phonon
        nq = 81; qz = np.linspace(0, 0.5, nq)
        Dq = dynamical_matrix_qpath(
            ph, np.column_stack([np.zeros(nq), np.zeros(nq), qz])).real
        qd = qz * 2.0
        bb = frequencies_from_dynamical(Dq)
        for n in range(D.shape[0]):
            ax2.plot(qd, np.abs(bb[:, n]), color="0.5", lw=0.5,
                     label="bare D" if n == 0 else None)
        if ss_finite is not None:
            rb = frequencies_from_dynamical(Dq + ss_finite.real[None])
            for n in range(D.shape[0]):
                ax2.plot(qd, np.abs(rb[:, n]), color="tab:green", lw=0.5, ls="--",
                         label="+tadpole" if n == 0 else None)
        else:
            ax2.text(0.5, 0.5, "tadpole NaN\n(no static shift)", ha="center",
                     va="center", transform=ax2.transAxes, color="tab:green")
        ax2.set_title(f"{s}: bands bare vs +static SE")
        ax2.set_xlabel(r"$q_z\ [\pi/a]$"); ax2.set_ylabel(r"$\omega$ [THz]")
        ax2.set_ylim(0, fmax); ax2.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"spectral_renorm_overlay.{ext}", dpi=140,
                    bbox_inches="tight")
    print(f"wrote {OUT / 'spectral_renorm_overlay.pdf'}")


if __name__ == "__main__":
    main()
