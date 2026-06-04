#!/usr/bin/env python
"""One spectral-function self-energy run (parallel worker).

Runs a single converged SCBA for (structure, temperature, self-energy mode) and
saves the pieces needed to build the zone-centre spectral function
    A(Gamma, omega) = -1/pi Im Tr [(omega+i eta)^2 I - D - Sigma_static - Sigma_B(omega)]^{-1}
both WITH the self-energy and WITHOUT (bare D is saved too). Designed to be
launched many-at-once (each single-slab bubble is single-core-bound, F18).

modes:
  bubble_half : cubic bubble, retarded="half" (broadening only, no KK real shift)
  bubble_fft  : cubic bubble, retarded="fft"  (full causal: broadening + KK)
  scp_fft     : bubble (fft) + self-consistent static tadpole Sigma_T
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
from postproc.spectral import dynamical_matrix_at_q

CFGS = {
    "cnt33": ("phonon/configs/cnt/cnt33_vasp.yaml", 18.0),
    "d5a": ("phonon/configs/sinw/sinw100_d5a_vasp_sc4_fc4.yaml", 16.0),
    "d11a": ("phonon/configs/sinw/sinw100_d11a_vasp_sc4.yaml", 16.0),
}

ap = argparse.ArgumentParser()
ap.add_argument("--struct", required=True, choices=list(CFGS))
ap.add_argument("--temp", type=float, required=True)
ap.add_argument("--mode", required=True, choices=["bubble_half", "bubble_fft", "scp_fft"])
ap.add_argument("--nfreq", type=int, default=81)
ap.add_argument("--eta", type=float, default=0.5)
ap.add_argument("--max-iter", type=int, default=80)
ap.add_argument("--out", required=True)
args = ap.parse_args()

cfg, fmax = CFGS[args.struct]
bundle = load_system(str(_REPO / cfg), validate=False, transport_axis=2)
ph = bundle.phonon
fc3 = str(Path(bundle.meta["fc3_path"]).expanduser().resolve())

kw = dict(
    fc3_hdf5=fc3, transport_direction="z", n_slabs=1,
    freq_range_thz=(0.05, fmax, args.nfreq), eta_factor=args.eta,
    temperature=args.temp, delta_T=10.0, max_scba_iter=args.max_iter,
    scba_tol=1e-3, conservation_tol=5e-2, enforce_asr=True, verbose=False,
)
if args.mode == "bubble_half":
    kw.update(retarded="half")
elif args.mode == "bubble_fft":
    kw.update(retarded="fft")
else:  # scp_fft
    kw.update(retarded="fft", tadpole=True, stage_loop_first=True,
              static_mixing=0.1)

try:
    r = transmission_finite(ph, **kw)
    freqs = np.asarray(r["freqs_thz"])
    sb = np.asarray(r["self_energy_retarded"])           # (nw, N, N)
    ss = r.get("sigma_static")
    ss = None if ss is None else np.asarray(ss)
    phi = r.get("phi_eff")
    D = ((np.asarray(phi) - ss) if (phi is not None and ss is not None)
         else dynamical_matrix_at_q(ph, [0, 0, 0]).real)
    np.savez_compressed(
        args.out, struct=args.struct, temp=args.temp, mode=args.mode,
        freqs=freqs, sigma_b=sb,
        sigma_static=(np.zeros((sb.shape[1],) * 2) if ss is None else ss),
        D=np.asarray(D), fmax=fmax,
        converged=bool(r.get("scba_converged")),
        resid=float(r.get("scba_residual", np.nan)),
        conservation=float(r.get("heat_flow_conservation", np.nan)),
        Ga_over_Gb=float(r["thermal_conductance_anharmonic"]
                         / r["thermal_conductance_ballistic"]),
    )
    print(f"OK {args.struct} T={args.temp:.0f} {args.mode}: "
          f"conv={r.get('scba_converged')} resid={r.get('scba_residual'):.2e} "
          f"cons={r.get('heat_flow_conservation'):.2e}")
except Exception as exc:
    print(f"FAIL {args.struct} T={args.temp:.0f} {args.mode}: "
          f"{type(exc).__name__}: {str(exc)[:80]}")
    sys.exit(1)
