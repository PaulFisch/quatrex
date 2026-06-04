#!/usr/bin/env python
"""Probe what controls the NEGF-SCBA heat-flow CONSERVATION error.

In a Phi-derivable (Baym-Kadanoff conserving) SCBA the current is conserved
exactly; a residual |J_L-J_R|/J therefore comes from one of:
  (i)   under-converged SCBA fixed point      -> falls with scba_tol
  (ii)  frequency-grid discretisation of the bubble convolution + the soft mode
        (its n_B(omega->0) ~ 1/omega weight)  -> falls with nfreq
  (iii) finite eta broadening                 -> falls with eta
  (iv)  FC3 vertex quality / sum rules         -> changes with the FC3 file
This worker runs ONE (structure, nfreq, eta, tol, fc3) point and prints the
conservation so a launcher can sweep each axis independently. Bubble-only
(no static SE); solver defaults to linear for the bubble (robust on the soft wire).
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

CFGS = {"cnt33": ("phonon/configs/cnt/cnt33_vasp.yaml", 18.0),
        "d5a": ("phonon/configs/sinw/sinw100_d5a_vasp_sc4_fc4.yaml", 16.0),
        "d11a": ("phonon/configs/sinw/sinw100_d11a_vasp_sc4.yaml", 16.0)}

ap = argparse.ArgumentParser()
ap.add_argument("--struct", required=True, choices=list(CFGS))
ap.add_argument("--nfreq", type=int, default=81)
ap.add_argument("--eta", type=float, default=0.5)
ap.add_argument("--tol", type=float, default=1e-4)
ap.add_argument("--max-iter", type=int, default=250)
ap.add_argument("--retarded", default="fft")
ap.add_argument("--fc3", default=None, help="override fc3.hdf5 path")
ap.add_argument("--tag", default="")
args = ap.parse_args()

cfg, fmax = CFGS[args.struct]
b = load_system(str(_REPO / cfg), validate=False, transport_axis=2)
ph = b.phonon
fc3 = args.fc3 or b.meta["fc3_path"]
r = transmission_finite(
    ph, fc3_hdf5=fc3, transport_direction="z", n_slabs=1, retarded=args.retarded,
    freq_range_thz=(0.05, fmax, args.nfreq), eta_factor=args.eta,
    temperature=300.0, delta_T=10.0, max_scba_iter=args.max_iter,
    scba_tol=args.tol, conservation_tol=5e-2, enforce_asr=True,
    solver="linear", mixing=0.3, verbose=False)
print(f"CONS {args.struct} {args.tag} nfreq={args.nfreq} eta={args.eta} "
      f"tol={args.tol:.0e} ret={args.retarded} fc3={Path(fc3).parent.name} :: "
      f"conv={r.get('scba_converged')!s:5s} resid={r.get('scba_residual'):.2e} "
      f"cons={r.get('heat_flow_conservation'):.3e} "
      f"Ga/Gb={r['thermal_conductance_anharmonic']/r['thermal_conductance_ballistic']:.3f}")
