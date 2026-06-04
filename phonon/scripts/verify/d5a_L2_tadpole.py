#!/usr/bin/env python
"""d5a at L=2 (n_slabs=2): does the self-consistent tadpole rescue full-KK
convergence as it did at L=1? Uses nearest-neighbour cutoffs (sigma/vertex/g) to
keep the dense multi-slab bubble tractable at 126 DOF.
"""
from __future__ import annotations
import sys, numpy as np
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import warnings; warnings.filterwarnings("ignore")
from phonon.finite_analysis.loader import load_system
from solver.dense import transmission_finite

CFG = str(_REPO / "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml")
bundle = load_system(CFG, validate=False, transport_axis=2)
ph = bundle.phonon
fc3 = str(Path(bundle.meta["fc3_path"]).expanduser().resolve())

common = dict(
    fc3_hdf5=fc3, transport_direction="z", retarded="fft", n_slabs=2,
    freq_range_thz=(0.01, 18.0, 41), eta_factor=1.0,
    temperature=300.0, delta_T=10.0, max_scba_iter=50, scba_tol=1e-3,
    conservation_tol=5e-2, enforce_asr=True,
    sigma_cutoff=0, vertex_cutoff=0, g_cutoff=0,   # diagonal blocks (cheap convergence probe)
    verbose=True,
)


def run(label, **kw):
    r = transmission_finite(ph, **{**common, **kw})
    ss = r.get("sigma_static"); ssn = (np.linalg.norm(ss) if ss is not None else 0.0)
    print(f"  {label:36s}: conv={r.get('scba_converged')!s:5s} "
          f"it={r.get('n_scba_iterations'):>3} resid={r.get('scba_residual'):.2e} "
          f"cons={r.get('heat_flow_conservation'):.2e} "
          f"Ga/Gb={r['thermal_conductance_anharmonic']/r['thermal_conductance_ballistic']:.3f} "
          f"||Sig_static||={ssn:.3f}")
    return r


print("== d5a L=2 (NN cutoffs s1/v1/g1): full-KK fft +/- self-consistent tadpole ==")
run("fft, no tadpole (ref)")
run("fft + self-consistent tadpole", tadpole=True, stage_loop_first=True,
    static_mixing=0.1)
