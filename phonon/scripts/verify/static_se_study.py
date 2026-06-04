#!/usr/bin/env python
"""One point of the static-correction magnitude study (parallel worker).

Runs a single (structure, T, mode) SCBA and reports the MAGNITUDE of each
anharmonic correction so the sweep can answer: is the loop necessary, and how
big is it vs the tadpole vs the dynamic bubble?

modes (all retarded='fft'): bubble | loop | tadpole | loop_tadpole
Reports: convergence, conservation, G_anh/G_ball, ||Sigma_static|| (the static
loop+/tadpole shift), max|Re Sigma_B| (bubble frequency shift), max|Im Sigma_B|
(bubble linewidth), and the bare-vs-renormalised soft-mode frequency (the SCP
stiffening of the lowest mode). Saves an npz for aggregation/plots.
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
        "d5a": ("phonon/configs/sinw/sinw100_d5a_vasp_sc4_fc4.yaml", 16.0)}
MODES = {
    "bubble":       dict(),
    "loop":         dict(loop=True),
    "tadpole":      dict(tadpole=True),
    "loop_tadpole": dict(loop=True, tadpole=True),
}

ap = argparse.ArgumentParser()
ap.add_argument("--struct", required=True, choices=list(CFGS))
ap.add_argument("--temp", type=float, required=True)
ap.add_argument("--mode", required=True, choices=list(MODES))
ap.add_argument("--nfreq", type=int, default=61)
ap.add_argument("--eta", type=float, default=0.5)
ap.add_argument("--max-iter", type=int, default=250)
ap.add_argument("--solver", default="linear")
ap.add_argument("--fc3", default=None, help="override fc3.hdf5 (e.g. CNT-with-FC4)")
ap.add_argument("--out", required=True)
args = ap.parse_args()

cfg, fmax = CFGS[args.struct]
b = load_system(str(_REPO / cfg), validate=False, transport_axis=2)
ph = b.phonon
fc3 = args.fc3 or b.meta["fc3_path"]
mk = MODES[args.mode]
kw = dict(fc3_hdf5=fc3, transport_direction="z", n_slabs=1, retarded="fft",
          freq_range_thz=(0.05, fmax, args.nfreq), eta_factor=args.eta,
          temperature=args.temp, delta_T=10.0, max_scba_iter=args.max_iter,
          scba_tol=1e-3, conservation_tol=5e-2, enforce_asr=True,
          solver=args.solver, mixing=0.3, verbose=False)
if mk.get("loop"):
    kw.update(loop=True, fc4_hdf5=fc3)
if mk.get("tadpole"):
    kw.update(tadpole=True)
if mk.get("loop") or mk.get("tadpole"):
    kw.update(stage_loop_first=True, static_mixing=0.1)

try:
    r = transmission_finite(ph, **kw)
    sb = np.asarray(r["self_energy_retarded"])            # (nw, N, N)
    ss = r.get("sigma_static")
    ss = None if ss is None else np.asarray(ss)
    phi = r.get("phi_eff")
    ss_norm = 0.0 if ss is None else float(np.linalg.norm(ss))
    reB = float(np.max(np.abs(sb.real))); imB = float(np.max(np.abs(sb.imag)))
    # bare vs renormalised lowest (soft) mode: sqrt(min eig) of D and D+Sigma_static
    soft_bare = soft_ren = float("nan")
    if phi is not None and ss is not None:
        D = (np.asarray(phi) - ss).real
        D = 0.5 * (D + D.T)
        w2 = np.linalg.eigvalsh(D)
        nz = w2[np.abs(w2) > 1e-6]
        soft_bare = float(np.sign(nz[0]) * np.sqrt(abs(nz[0]))) if len(nz) else float("nan")
        w2r = np.linalg.eigvalsh(0.5 * (np.asarray(phi).real + np.asarray(phi).real.T))
        nzr = w2r[np.abs(w2r) > 1e-6]
        soft_ren = float(np.sign(nzr[0]) * np.sqrt(abs(nzr[0]))) if len(nzr) else float("nan")
    np.savez_compressed(
        args.out, struct=args.struct, temp=args.temp, mode=args.mode,
        freqs=np.asarray(r["freqs_thz"]), sigma_b=sb,
        sigma_static=(np.zeros((sb.shape[1],) * 2) if ss is None else ss),
        converged=bool(r.get("scba_converged")),
        resid=float(r.get("scba_residual", np.nan)),
        conservation=float(r.get("heat_flow_conservation", np.nan)),
        Ga_over_Gb=float(r["thermal_conductance_anharmonic"]
                         / r["thermal_conductance_ballistic"]),
        sigma_static_norm=ss_norm, reB=reB, imB=imB,
        soft_bare=soft_bare, soft_ren=soft_ren)
    print(f"STUDY {args.struct} T={args.temp:.0f} {args.mode:12s} :: "
          f"conv={r.get('scba_converged')!s:5s} cons={r.get('heat_flow_conservation'):.2e} "
          f"Ga/Gb={r['thermal_conductance_anharmonic']/r['thermal_conductance_ballistic']:.3f} "
          f"||Sig_stat||={ss_norm:8.3f} maxReB={reB:7.2f} maxImB={imB:7.2f} "
          f"soft {soft_bare:.4f}->{soft_ren:.4f} THz")
except Exception as e:
    print(f"STUDY {args.struct} T={args.temp:.0f} {args.mode}: FAIL {type(e).__name__}: {str(e)[:60]}")
    sys.exit(1)
