#\!/usr/bin/env python
"""Ballistic-only (FC3 off, vertex_scale=0) eta sweep on the REAL d5a wire.

If the clean-chain ballistic path is eta-stable (verified) but d5a's G_ball
swings wildly with eta, the swing is structural: a soft/near-zero FC2 mode
makes the low-omega transmission pathological. We look at max T(omega) vs the
physical channel count and at G_ball(eta).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")
import numpy as np
from phonon.finite_analysis.loader import load_system
from phonon.solver.dense import transmission_finite

cfg = _REPO / "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml"
bundle = load_system(cfg, validate=False, transport_axis=2)
fc3 = Path(bundle.meta.get("fc3_path", "")).expanduser().resolve()
ndof = int(bundle.phonon.primitive.masses.shape[0] * 3)
print(f"d5a: {ndof} DOF/slab; fc3={fc3.name}\n")
print(f"{'eta_f':>6} {'G_ball':>13} {'maxT':>9} {'T@f0':>9} {'T@f1':>9} {'T@f2':>9}")
for eta_factor in (9.0, 6.75, 4.5, 3.0, 1.0, 0.3):
    res = transmission_finite(
        bundle.phonon, fc3_hdf5=str(fc3),
        freq_range_thz=(0.01, 18.0, 81), transport_direction="z",
        temperature=300.0, delta_T=10.0, n_slabs=2,
        eta_factor=eta_factor, vertex_scale=0.0, max_scba_iter=1,
        auto_extend_fmax=False, zero_mode_projection=True,
        verbose=False,
    )
    T = np.asarray(res["transmission_ballistic"])
    G = res["thermal_conductance_ballistic"]
    print(f"{eta_factor:>6.2f} {G:>13.5e} {T.max():>9.3f} "
          f"{T[0]:>9.3f} {T[1]:>9.3f} {T[2]:>9.3f}")
