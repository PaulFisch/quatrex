#!/usr/bin/env python
"""Diagnose WHY Anderson diverges on the d5a soft-mode bubble where linear
mixing converges. Theory says safeguarded Anderson should never be worse than
linear (it falls back to the linear step when the extrapolation overshoots), so
a divergence points at the *safeguard policy* -- specifically the new
accelerator KEEPS secant history across residual upticks, which can poison the
least-squares on a strongly nonlinear map. We test:

  linear            : plain damped Picard (baseline, converges)
  and_default       : safeguarded Anderson depth8 beta0.3 cap8 (the diverger)
  and_legacy        : legacy _anderson_mix (restart on ANY uptick)  <-- key test
  and_damped        : safeguarded depth8 beta0.1 cap8
  and_lowdepth_cap  : safeguarded depth2 beta0.3 cap1.5 (near-linear + slight accel)
  and_legacy_d4     : legacy depth4 beta0.2

Each prints the residual TRAJECTORY so the divergence onset is visible.
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import warnings; warnings.filterwarnings("ignore")
from phonon.finite_analysis.loader import load_system
from solver.dense import transmission_finite

VARIANTS = {
    "linear":           dict(solver="linear", mixing=0.3),
    "and_default":      dict(solver="anderson", anderson_depth=8, mixing=0.3,
                             anderson_step_cap=8.0, anderson_safeguard=True),
    "and_legacy":       dict(solver="anderson", anderson_depth=8, mixing=0.3,
                             anderson_step_cap=2.0, anderson_safeguard=False),
    "and_damped":       dict(solver="anderson", anderson_depth=8, mixing=0.1,
                             anderson_step_cap=8.0, anderson_safeguard=True),
    "and_lowdepth_cap": dict(solver="anderson", anderson_depth=2, mixing=0.3,
                             anderson_step_cap=1.5, anderson_safeguard=True),
    "and_legacy_d4":    dict(solver="anderson", anderson_depth=4, mixing=0.2,
                             anderson_step_cap=2.0, anderson_safeguard=False),
}

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True, choices=list(VARIANTS))
ap.add_argument("--nfreq", type=int, default=31)
ap.add_argument("--max-iter", type=int, default=70)
args = ap.parse_args()

b = load_system(str(_REPO / "phonon/configs/sinw/sinw100_d5a_vasp_sc4_fc4.yaml"),
                validate=False, transport_axis=2)
ph = b.phonon; fc3 = b.meta["fc3_path"]
common = dict(fc3_hdf5=fc3, transport_direction="z", n_slabs=1, retarded="fft",
              freq_range_thz=(0.05, 16.0, args.nfreq), eta_factor=0.5,
              temperature=300.0, delta_T=10.0, max_scba_iter=args.max_iter,
              scba_tol=1e-6, conservation_tol=5e-2, enforce_asr=True,
              divergence_guard=False, verbose=False)
r = transmission_finite(ph, **{**common, **VARIANTS[args.variant]})
hist = r.get("convergence_history") or []
traj = " ".join(f"{x:.1e}" for x in hist)
print(f"VARIANT {args.variant}: conv={r.get('scba_converged')} "
      f"n_it={len(hist)} final={r.get('scba_residual'):.2e} "
      f"cons={r.get('heat_flow_conservation'):.2e}")
print(f"  traj: {traj}")
