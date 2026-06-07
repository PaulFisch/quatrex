#!/usr/bin/env python
"""Aggregate the static-correction magnitude study into a table + figure.

Answers: is the loop necessary, and how big is each correction (loop Sigma_L,
tadpole Sigma_T, dynamic bubble Sigma_B) vs T, for d5a (soft-mode) and CNT (stiff)?
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, "/usr/scratch/mont-fort11/pfischill/quatrex/phonon")
from finite_analysis.plot_style import set_publication_style  # noqa: E402
set_publication_style()
_REPO = Path(__file__).resolve().parents[3]
SRC = Path("/tmp/claude/se_study")
OUT = _REPO / "document/fig/transport_sweeps"


def load_all():
    d = {}
    for f in glob.glob(str(SRC / "study_*.npz")):
        z = np.load(f, allow_pickle=True)
        d[(str(z["struct"]), float(z["temp"]), str(z["mode"]))] = z
    return d


def main():
    data = load_all()
    if not data:
        print("no study npz yet"); return
    structs = sorted({k[0] for k in data})
    temps = sorted({k[1] for k in data})
    modes = ["bubble", "loop", "tadpole", "loop_tadpole"]
    print(f"{'struct':6} {'T':>4} {'mode':12} {'conv':5} {'cons':>9} {'Ga/Gb':>6} "
          f"{'||Sig_st||':>10} {'maxReB':>8} {'maxImB':>8} {'soft_b->soft_r':>16}")
    for s in structs:
        for T in temps:
            for m in modes:
                z = data.get((s, T, m))
                if z is None:
                    continue
                print(f"{s:6} {T:4.0f} {m:12} {str(bool(z['converged'])):5} "
                      f"{float(z['conservation']):9.2e} {float(z['Ga_over_Gb']):6.3f} "
                      f"{float(z['sigma_static_norm']):10.3f} {float(z['reB']):8.2f} "
                      f"{float(z['imB']):8.2f} "
                      f"{float(z['soft_bare']):7.4f}->{float(z['soft_ren']):.4f}")

    # figure: magnitudes vs T, per structure
    fig, axes = plt.subplots(2, len(structs), figsize=(5.2 * len(structs), 8),
                             squeeze=False)
    for c, s in enumerate(structs):
        Ts = [T for T in temps if (s, T, "bubble") in data]
        def series(mode, field):
            return [float(data[(s, T, mode)][field]) if (s, T, mode) in data else np.nan
                    for T in Ts]
        ax = axes[0][c]
        ax.semilogy(Ts, series("loop", "sigma_static_norm"), "o-", label=r"$\|\Sigma_L\|$ (loop)")
        ax.semilogy(Ts, series("tadpole", "sigma_static_norm"), "s-", label=r"$\|\Sigma_T\|$ (tadpole)")
        ax.semilogy(Ts, series("bubble", "reB"), "^-", label=r"max$|{\rm Re}\,\Sigma_B|$ (shift)")
        ax.semilogy(Ts, series("bubble", "imB"), "v-", label=r"max$|{\rm Im}\,\Sigma_B|$ (linewidth)")
        ax.set_xlabel("T [K]"); ax.set_ylabel(r"magnitude [THz$^2$]")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        # conservation per mode
        ax2 = axes[1][c]
        for m, mk in [("bubble", "o-"), ("loop", "s-"), ("tadpole", "^-"),
                      ("loop_tadpole", "v-")]:
            ax2.semilogy(Ts, series(m, "conservation"), mk, label=m)
        ax2.set_xlabel("T [K]"); ax2.set_ylabel("conservation error")
        ax2.legend(fontsize=7); ax2.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"static_se_study.{ext}", dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT / 'static_se_study.pdf'}")


if __name__ == "__main__":
    main()
