#\!/usr/bin/env python
"""Decisive: does SCBA converge at PHYSICAL eta (where the ballistic baseline
is correct) with a modest vertex (lambda=0.3)? And is G_anh < G_ball?

Runs two small-eta points (eta_f=0.5 and 1.0) at lambda=0.3, n_slabs=2.
auto_extend_fmax=True so the 3-phonon bubble FFT support is unaliased.
"""
from __future__ import annotations
import sys, warnings, time, json
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")
import numpy as np
from phonon.finite_analysis.loader import load_system
from phonon.solver.dense import transmission_finite

cfg = _REPO / "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml"
bundle = load_system(cfg, validate=False, transport_axis=2)
ph = bundle.phonon
fc3 = str(Path(bundle.meta.get("fc3_path", "")).expanduser().resolve())
out = _REPO / "phonon/scripts/out/d5a_smalleta_verify"
out.mkdir(parents=True, exist_ok=True)

for eta_f, lam in [(0.5, 0.3), (1.0, 0.3)]:
    t0 = time.time()
    tag = f"eta{eta_f}_lam{lam}"
    print(f"\n===== {tag} =====", flush=True)
    r = transmission_finite(
        ph, fc3_hdf5=fc3, freq_range_thz=(0.01, 18.0, 81),
        transport_direction="z", temperature=300.0, delta_T=10.0, n_slabs=2,
        eta_factor=eta_f, vertex_scale=lam, max_scba_iter=80, scba_tol=1e-3,
        mixing=0.3, anderson_mixing=True, anderson_depth=8,
        anderson_safeguard=True, zero_mode_projection=True,
        divergence_guard=True, auto_extend_fmax=True, fmax_margin=1.05,
        retarded="fft", dc_handling="interpolate", verbose=True,
    )
    rec = dict(eta_f=eta_f, lam=lam,
               converged=bool(r.get("scba_converged", False)),
               n_iter=int(r.get("n_scba_iterations", 0)),
               resid=float(r.get("scba_residual", float("nan"))),
               G_ball=float(r["thermal_conductance_ballistic"]),
               G_anh=float(r["thermal_conductance_anharmonic"]),
               conservation=float(r.get("heat_flow_conservation", float("nan"))),
               wall_s=time.time() - t0)
    rec["G_anh_over_G_ball"] = rec["G_anh"] / rec["G_ball"] if rec["G_ball"] else float("nan")
    print(f"RESULT {tag}: {json.dumps(rec)}", flush=True)
    with open(out / f"{tag}.json", "w") as fh:
        json.dump(rec, fh, indent=2)
print("\nALL DONE", flush=True)
