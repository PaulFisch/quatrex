"""Spectral transmission / spectral heat current from an SCBA checkpoint.

Reads a production checkpoint .npz (the format written by the transport
sweep: keys ``freqs_thz``, ``transmission_ballistic``,
``spectral_heat_current_ballistic``, ``spectral_heat_current``) and writes
fig/spectral_transmission.pdf with two panels:

  top    : ballistic transmission staircase T_ball(omega) (channel count)
  bottom : spectral heat current J(omega), ballistic vs anharmonic --
           the area ratio is the conductance ratio G_anh/G_ball shown on
           the CNT slide.

Usage:
    python plot_spectral_transmission.py <checkpoint.npz> [label]

The CNT checkpoints live on the cluster
(/usr/scratch/.../phonon/scripts/out/cnt33_*/checkpoints/...); copy one over
(each is only a few kB) or mount scratch, then point this at it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ckpt = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else ckpt.stem
    d = np.load(ckpt, allow_pickle=True)

    w = np.asarray(d["freqs_thz"]).ravel()
    t_ball = np.asarray(d["transmission_ballistic"]).ravel()
    j_ball = np.asarray(d["spectral_heat_current_ballistic"]).ravel()
    j_anh = np.asarray(d["spectral_heat_current"]).ravel()

    g_ball = float(np.atleast_1d(d["thermal_conductance_ballistic"]).ravel()[0])
    g_anh = float(np.atleast_1d(d["thermal_conductance_anharmonic"]).ravel()[0])
    ratio = g_anh / g_ball

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.0, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2], "hspace": 0.12},
        constrained_layout=True,
    )

    ax1.plot(w, t_ball, color="0.25", lw=1.5, drawstyle="steps-mid")
    ax1.fill_between(w, 0, t_ball, step="mid", color="0.8")
    ax1.set_ylabel(r"$\mathcal{T}_{\rm ball}(\omega)$")
    ax1.set_title(f"{label}: ballistic channels", fontsize=9)
    ax1.set_ylim(bottom=0)

    ax2.fill_between(w, 0, j_ball, color="#9ecae1", label="ballistic")
    ax2.plot(w, j_anh, color="#cb181d", lw=1.6,
             label=f"anharmonic (SCBA)")
    ax2.set_xlabel(r"$\omega$ (THz)")
    ax2.set_ylabel(r"$J(\omega)$  (a.u.)")
    ax2.legend(fontsize=8, frameon=False, loc="upper right")
    ax2.set_ylim(bottom=0)
    ax2.text(0.03, 0.9,
             rf"$G_{{\rm anh}}/G_{{\rm ball}} = {ratio:.3f}$",
             transform=ax2.transAxes, fontsize=9, va="top")

    # Clip to the phonon band (grid runs to ~2*wmax for anti-aliasing).
    nz = w[t_ball > 1e-6]
    if nz.size:
        ax2.set_xlim(0, 1.08 * float(nz.max()))

    out = HERE / "fig" / "spectral_transmission.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}  (ratio={ratio:.3f}, G_ball={g_ball:.3g} W/m2/K)")


if __name__ == "__main__":
    main()
