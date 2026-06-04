#!/usr/bin/env python
"""Line dispersion of the lowest branches right up against Gamma, to show what
the 'soft mode' actually is: NOT a flat band, but the value of the lowest
non-translational (twist) branch AT q=0. The 3 acoustic branches go to 0 by the
translational ASR; the twist branch bottoms out at a small finite frequency
(d5a 0.0075, CNT 0.0265 THz) and then disperses up like the others.
"""
from __future__ import annotations
import sys
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
from postproc.spectral import dynamical_matrix_qpath, frequencies_from_dynamical

STRUCTS = [
    ("CNT(3,3)", "phonon/configs/cnt/cnt33_vasp.yaml"),
    ("SiNW d5a", "phonon/configs/sinw/sinw100_d5a_vasp_sc4_fc4.yaml"),
]
OUT = _REPO / "document/fig/transport_sweeps"

fig, axes = plt.subplots(1, len(STRUCTS), figsize=(11, 4.6))
for ax, (name, cfg) in zip(np.atleast_1d(axes), STRUCTS):
    bundle = load_system(str(_REPO / cfg), validate=False, transport_axis=2)
    ph = bundle.phonon
    nq = 120
    qz = np.linspace(0.0, 0.06, nq)              # right up against Gamma
    Dq = dynamical_matrix_qpath(
        ph, np.column_stack([np.zeros(nq), np.zeros(nq), qz])).real
    bands = np.abs(frequencies_from_dynamical(Dq))   # (nq, N)
    qd = qz * 2.0                                     # pi/a units
    nshow = 7
    for n in range(nshow):
        ax.plot(qd, bands[:, n], lw=1.1, marker="o", ms=2, markevery=20)
    wG = np.sort(bands[0])
    # the twist = lowest mode at Gamma that is NOT one of the 3 ~0 acoustic
    nonzero = wG[wG > 1e-3]
    twist = nonzero[0] if len(nonzero) else 0.0
    ax.scatter([0], [twist], s=80, facecolors="none", edgecolors="lime",
               linewidths=1.8, zorder=5,
               label=f"twist @ $\\Gamma$ = {twist:.4f} THz")
    ax.scatter([0, 0, 0], wG[:3], s=40, color="red", zorder=5,
               label="3 acoustic @ $\\Gamma$ = 0")
    ax.set_title(f"{name}: branches near $\\Gamma$")
    ax.set_xlabel(r"$q_z\ [\pi/a]$")
    ax.set_ylabel(r"$\omega$ [THz]")
    ax.set_xlim(0, qd[-1])
    ax.set_ylim(-0.02, max(bands[:, nshow - 1].max(), 0.5))
    ax.legend(fontsize=8, loc="upper left")
    print(f"{name}: lowest 5 @ Gamma = {np.array2string(wG[:5], precision=4)} THz; "
          f"twist = {twist:.4f}")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"softmode_dispersion.{ext}", dpi=140, bbox_inches="tight")
print(f"wrote {OUT / 'softmode_dispersion.pdf'}")
