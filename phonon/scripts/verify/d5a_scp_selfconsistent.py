#!/usr/bin/env python
"""d5a SiNW: SELF-CONSISTENT tadpole SCP (the right physics for a soft mode).

The fixed (harmonic) <uu> tadpole runs away on d5a because the soft twist's
<uu> ~ 1/omega^2 is divergent (=> ||Sigma_static|| ~ 1e4). The point of SCP is
self-consistency: the tadpole stiffens the soft mode -> omega rises -> <uu>
drops -> the self-energy self-limits. That needs <uu> recomputed from the
RENORMALISED modes every iteration -- i.e. the device-G^< path (tadpole=True
WITHOUT static_uu), which the SCBA loop already iterates. Pair it with
retarded='half' (the full-KK 'fft' alone does not converge d5a).
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
    fc3_hdf5=fc3, transport_direction="z",
    freq_range_thz=(0.01, 18.0, 61), eta_factor=0.5,
    temperature=300.0, delta_T=10.0, max_scba_iter=80, scba_tol=1e-3,
    conservation_tol=5e-2, enforce_asr=True, verbose=True,
)


def run(label, **kw):
    r = transmission_finite(ph, **{**common, **kw})
    ss = r.get("sigma_static")
    ssn = (np.linalg.norm(ss) if ss is not None else 0.0)
    print(f"  {label:34s}: conv={r.get('scba_converged')!s:5s} "
          f"it={r.get('n_scba_iterations'):>3} resid={r.get('scba_residual'):.2e} "
          f"cons={r.get('heat_flow_conservation'):.2e} "
          f"Ga/Gb={r['thermal_conductance_anharmonic']/r['thermal_conductance_ballistic']:.3f} "
          f"||Sig_static||={ssn:.3f}")
    return r


print("== d5a SELF-CONSISTENT tadpole (device <uu>, recomputed each iter) ==")
run("half, no tadpole (ref)", retarded="half")
run("half + self-consistent tadpole", retarded="half", tadpole=True,
    stage_loop_first=True, static_mixing=0.1)
