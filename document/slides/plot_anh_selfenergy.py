"""The anharmonic self-energy spectrum -- the most fundamental untapped
anharmonic observable. From se_study sigma_b (full Sigma_B(omega) matrices):

  Im Sigma(omega)  = phonon linewidth (anharmonic broadening / lifetime^-1)
  Re Sigma(omega)  = frequency shift (softening / stiffening)

One panel per transport wire (CNT(3,3), d5a SiNW), mean over the diagonal
(per-mode average). This is the object behind the phono3py linewidth
cross-check and the SCBA transmission suppression.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SE = HERE.parents[1] / "cluster" / "tortin3-tmp" / "se_study"
C = {"im": "#AA3377", "re": "#4477AA", "grey": "#BBBBBB"}

plt.rcParams.update({"font.size": 10, "legend.fontsize": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True, "lines.linewidth": 1.8})


def selfenergy(system):
    z = np.load(SE / f"study_{system}_T300_bubble.npz", allow_pickle=True)
    sb = np.asarray(z["sigma_b"]); fr = np.asarray(z["freqs"])
    im = np.array([np.mean(np.abs(np.diag(sb[i].imag))) for i in range(len(fr))])
    re = np.array([np.mean(np.diag(sb[i].real)) for i in range(len(fr))])
    return fr, im, re


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for ax, (sysn, lbl, wmax) in zip(axes, [("cnt33", "CNT(3,3)", 50), ("d5a", "SiNW d5a", 18)]):
        fr, im, re = selfenergy(sysn)
        m = fr <= wmax
        ax.semilogy(fr[m], im[m], "-", color=C["im"], label=r"$\langle|{\rm Im}\,\Sigma|\rangle$ (linewidth)")
        ax.set_xlabel(r"$\omega$ [THz]")
        ax.set_ylabel(r"$\langle|{\rm Im}\,\Sigma_{ii}|\rangle$ [THz$^2$]", color=C["im"])
        ax.tick_params(axis="y", colors=C["im"])
        ax2 = ax.twinx()
        ax2.plot(fr[m], re[m], "--", color=C["re"], lw=1.5, label=r"$\langle{\rm Re}\,\Sigma\rangle$ (shift)")
        ax2.axhline(0, ls=":", color=C["grey"], lw=1)
        ax2.set_ylabel(r"$\langle{\rm Re}\,\Sigma_{ii}\rangle$ [THz$^2$]", color=C["re"])
        ax2.tick_params(axis="y", colors=C["re"])
        ax.set_title(lbl, fontsize=10)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7)
    fig.tight_layout()
    out = HERE / "fig" / "anh_selfenergy.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
