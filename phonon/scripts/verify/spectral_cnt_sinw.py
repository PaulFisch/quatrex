#!/usr/bin/env python
"""Phonon spectral function A(q_z, omega) along the 1D periodic axis for the
CNT(3,3) and the d5a SiNW (the two transport structures), rendered as a heat map
with the dispersion overlaid. This is the A(q,omega) = -1/pi Im Tr G^R object of
postproc.spectral, the same product we computed for bulk-Si SCP -- here for the
two wires so the soft twist (d5a 0.0075 THz, CNT 0.026 THz) and the low-omega
channel structure are visible. Static/eta-broadened (no bubble linewidth, which
on d5a needs the fragile mean_displacement solve -- F34); the anharmonic
broadening is in the transport spectral-current J(omega) overlays.
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
from postproc.spectral import dynamical_matrix_qpath, spectral_function_qw, frequencies_from_dynamical

STRUCTS = [
    ("CNT(3,3)", "phonon/configs/cnt/cnt33_vasp.yaml", 18.0),
    ("SiNW d5a", "phonon/configs/sinw/sinw100_d5a_vasp_sc4_fc4.yaml", 16.0),
]
OUT = _REPO / "document/fig/transport_sweeps"
OUT.mkdir(parents=True, exist_ok=True)

# two columns (full range + low-omega zoom) per structure so the soft twist
# (0.0075-0.0265 THz, otherwise buried at the origin) is actually resolved.
fig, axes = plt.subplots(2, len(STRUCTS), figsize=(11, 8))
for col, (name, cfg, fmax) in enumerate(STRUCTS):
    bundle = load_system(str(_REPO / cfg), validate=False, transport_axis=2)
    ph = bundle.phonon
    nq = 81
    qz = np.linspace(0.0, 0.5, nq)
    q_path = np.column_stack([np.zeros(nq), np.zeros(nq), qz])
    D_q = dynamical_matrix_qpath(ph, q_path)
    bands = frequencies_from_dynamical(D_q)           # (nq, N) signed THz
    qd = qz * 2.0                                     # in units of pi/a (0..1)
    soft = float(np.min(np.abs(bands[0])[np.abs(bands[0]) > 1e-4]))

    # (top) full range -- coarse grid + eta a few grid spacings.
    # (bottom) 0-1.2 THz zoom -- fine grid + tiny eta to resolve the twist.
    for row, (wlo, whi, nw, eta_w) in enumerate([
            (0.02, fmax, 700, 0.5 * (fmax / 700) * 6),
            (0.001, 1.2, 600, 0.004)]):
        ax = axes[row, col]
        grid = np.linspace(wlo, whi, nw)
        A = spectral_function_qw(D_q, grid, eta_w)
        im = ax.pcolormesh(qd, grid, np.log10(A.T + 1e-3), cmap="magma",
                           shading="auto", rasterized=True)
        for n in range(bands.shape[1]):
            ax.plot(qd, np.abs(bands[:, n]), color="cyan", lw=0.35, alpha=0.5)
        ax.set_xlabel(r"$q_z\ [\pi/a]$")
        ax.set_ylabel(r"$\omega$ [THz]")
        ax.set_ylim(wlo if row else 0, whi)
        if row == 1:
            ax.axhline(soft, color="lime", ls=":", lw=1.0)
            ax.set_title(f"{name}  zoom: twist at {soft:.4f} THz (green)")
        else:
            ax.set_title(f"{name}  (full range)")
        fig.colorbar(im, ax=ax, label=r"$\log_{10} A(q_z,\omega)$")
    print(f"{name}: nq={nq}, soft/twist {soft:.4f} THz, "
          f"max band {np.abs(bands).max():.2f} THz")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"spectral_cnt_sinw.{ext}", dpi=140, bbox_inches="tight")
print(f"wrote {OUT / 'spectral_cnt_sinw.pdf'}")
