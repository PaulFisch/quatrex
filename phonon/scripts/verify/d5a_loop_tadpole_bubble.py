#!/usr/bin/env python
"""d5a SiNW full SCP: bubble / +tadpole / +loop / +loop+tadpole (n_slabs=1).
Uses the rfe-refit FC4 (loop) + FC3 (bubble/tadpole). Tests whether the quartic
loop -- the strongest, rigorous SCP stiffening -- stabilises the full-KK (fft)
soft-mode SCBA, and how each term renormalises transport. <uu> is self-consistent
(device G^<, recomputed each iter) so the soft mode self-limits.
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

CFG = str(_REPO / "phonon/configs/sinw/sinw100_d5a_vasp_sc4_fc4.yaml")
bundle = load_system(CFG, validate=False, transport_axis=2)
ph = bundle.phonon
fc3 = str(Path(bundle.meta["fc3_path"]).expanduser().resolve())
print(f"d5a fc3/fc4 = {fc3}")

common = dict(
    fc3_hdf5=fc3, transport_direction="z", retarded="fft", n_slabs=1,
    freq_range_thz=(0.01, 18.0, 61), eta_factor=0.5,
    temperature=300.0, delta_T=10.0, max_scba_iter=80, scba_tol=1e-3,
    conservation_tol=5e-2, enforce_asr=True,
    # feed the static SE the BUBBLE-BROADENED G^< (full_G), no pre-staging, and
    # gentle mixing: the soft twist's loop-only <uu> ~ 1/omega^2 diverges and
    # NaNs the static SE otherwise; the bubble broadening tames it.
    loop_propagator="full_G", stage_loop_first=False, static_mixing=0.05,
    verbose=True,
)


def run(label, **kw):
    r = transmission_finite(ph, **{**common, **kw})
    ss = r.get("sigma_static"); ssn = (np.linalg.norm(ss) if ss is not None else 0.0)
    print(f"  {label:22s}: conv={r.get('scba_converged')!s:5s} "
          f"it={r.get('n_scba_iterations'):>3} resid={r.get('scba_residual'):.2e} "
          f"cons={r.get('heat_flow_conservation'):.2e} "
          f"Ga/Gb={r['thermal_conductance_anharmonic']/r['thermal_conductance_ballistic']:.3f} "
          f"||Sig_static||={ssn:.3f}")
    return r


print("== d5a bubble / +tadpole / +loop / +loop+tadpole (fft, self-consistent) ==")
run("bubble only")
run("+tadpole", tadpole=True)
run("+loop", loop=True, fc4_hdf5=fc3)
run("+loop+tadpole", loop=True, tadpole=True, fc4_hdf5=fc3)
