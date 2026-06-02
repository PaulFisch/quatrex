"""Direct Caroli ballistic conductance curves for d5a and d11a.

Pure harmonic NEGF: T(omega)=Tr(Gamma_L G^R Gamma_R G^A) via Sancho-Rubio
surface Green's functions, with NO third-order force constants and NO bubble
(unlike the SCBA transport sweep, whose ``--ballistic-only`` still assembles
the FC3 vertex). The ballistic transmission is temperature independent, so it
is computed once per (wire, length) and integrated against every temperature
weight for free. Reuses the exact conductance formula of
``transmission_finite`` (dense.py:1351-1376).

Writes ``phonon/scripts/out/ballistic_curves/ballistic.csv``.
"""
import csv
import sys
import time
import warnings
from pathlib import Path

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from phonon_inputs.constants import CONVERSION_THZ2, HBAR_SI, THZ_TO_RAD  # noqa: E402
from phonon_inputs.convention import get_btd_blocks  # noqa: E402
from phonon.finite_analysis.loader import load_system  # noqa: E402
from phonon.solver.grids import bose_full_axis, build_frequency_grid  # noqa: E402
from phonon.solver.leads import (  # noqa: E402
    build_device_hamiltonian, ballistic_transmission_z2)

WIRES = {
    "d5a": "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml",
    "d11a": "phonon/configs/sinw/sinw100_d11a_vasp_sc4.yaml",
    "cnt33": "phonon/configs/cnt/cnt33_vasp.yaml",
}
LENGTHS = [1, 2, 4]
TEMPS = [200, 300, 400, 500, 600]
DELTA_T = 10.0
# fmax covers 2*omega_max (~138 THz) so the integral support is complete.
FREQ_RANGE = (0.01, 140.0, 701)
ETA_FACTOR = 0.3

OUT = _REPO / "phonon/scripts/out/ballistic_curves"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for wire, cfg in WIRES.items():
    b = load_system(_REPO / cfg, validate=False, transport_axis=2)
    ph = b.phonon
    H_00, H_01 = get_btd_blocks(
        ph, (0.0, 0.0), transport_direction="z",
        conversion_factor=CONVERSION_THZ2)
    n_dof = H_00.shape[0]
    lattice = ph.primitive.cell
    a1, a2 = lattice[0], lattice[1]  # perp to transport (z)
    A_c = np.linalg.norm(np.cross(a1, a2)) * 1e-20

    freqs_thz, dw_thz, eta_w, z2_arr, pos_mask, mid = build_frequency_grid(
        FREQ_RANGE, eta_factor=ETA_FACTOR)
    omega_rad = freqs_thz * THZ_TO_RAD

    for n_slabs in LENGTHS:
        t0 = time.time()
        H_D = build_device_hamiltonian(H_00, H_01, n_slabs)
        N_D = n_slabs * n_dof
        H_LD = np.zeros((n_dof, N_D), dtype=complex)
        H_LD[:, :n_dof] = H_01
        H_DR = np.zeros((N_D, n_dof), dtype=complex)
        H_DR[-n_dof:, :] = H_01
        trans = np.zeros(len(freqs_thz))
        for iw, z2 in enumerate(z2_arr):
            trans[iw] = ballistic_transmission_z2(
                z2, H_D, H_00, H_01, H_LD, H_DR)
        wall = time.time() - t0
        for T in TEMPS:
            n_L = bose_full_axis(freqs_thz, T + DELTA_T / 2)
            n_R = bose_full_axis(freqs_thz, T - DELTA_T / 2)
            spec = HBAR_SI * omega_rad * (n_L - n_R) * trans
            J = np.sum(spec[pos_mask]) * dw_thz * 1e12
            G = J / (A_c * DELTA_T)
            rows.append(dict(
                wire=wire, n_slabs=n_slabs, T=T, delta_T=DELTA_T,
                G_ball=G, maxT=float(trans.max()), n_dof=n_dof))
        print(f"[{wire}] L={n_slabs}: maxT={trans.max():.3f}, "
              f"G_ball(300)={[r['G_ball'] for r in rows if r['wire']==wire and r['n_slabs']==n_slabs and r['T']==300][0]:.3e}, "
              f"{wall:.1f}s", flush=True)

with open(OUT / "ballistic.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"[done] wrote {OUT / 'ballistic.csv'} ({len(rows)} rows)", flush=True)
