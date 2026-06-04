#!/usr/bin/env python
"""d5a SiNW: does the static tadpole / Kramers-Kronig real part shift the soft
twist mode and help the SCBA converge?

Runs transmission_finite (n_slabs=1, Gamma device) in several configurations and
reports SCBA convergence + the renormalised soft-mode frequency. The loop
(Sigma_L, FC4) is unavailable for d5a (84-atom supercell -> 30 GiB dense FC4
export, over the 8 GiB guard; needs the sparse exporter), so we test the cubic
tadpole (FC3-only) and the bubble retarded construction (KK).
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
T = 300.0

bundle = load_system(CFG, validate=False, transport_axis=2)
ph = bundle.phonon
fc3 = str(Path(bundle.meta["fc3_path"]).expanduser().resolve())
print(f"d5a loaded; fc3={fc3}")

# The tadpole <uu> is sourced self-consistently from the device G^< every SCBA
# iteration (a fixed/frozen <uu> is non-self-consistent and physically wrong --
# it gives a runaway tadpole on the soft twist; removed).
common = dict(
    fc3_hdf5=fc3, transport_direction="z",
    freq_range_thz=(0.01, 18.0, 61), eta_factor=0.5,
    temperature=T, delta_T=10.0, max_scba_iter=50, scba_tol=1e-3,
    conservation_tol=5e-2, enforce_asr=True, verbose=True,
)


def run(label, **kw):
    r = transmission_finite(ph, **{**common, **kw})
    conv = r.get("scba_converged")
    it = r.get("n_scba_iterations")
    res = r.get("scba_residual", float("nan"))
    cons = r.get("heat_flow_conservation", float("nan"))
    gb = r["thermal_conductance_ballistic"]
    ga = r["thermal_conductance_anharmonic"]
    ss = r.get("sigma_static")
    ssn = (np.linalg.norm(ss) if ss is not None else 0.0)
    print(f"  {label:28s}: conv={conv!s:5s} it={it:>3} resid={res:.2e} "
          f"cons={cons:.2e} Ga/Gb={ga/gb:.3f} ||Sig_static||={ssn:.3f}")
    return r


print("\n== d5a SCBA convergence + soft-mode tests (n_slabs=1, 300 K) ==")
run("baseline (retarded=fft)")
run("retarded=half (KK variant)", retarded="half")
run("+tadpole (self-consistent)", tadpole=True, stage_loop_first=True,
    static_mixing=0.1)
run("+tadpole, retarded=half", tadpole=True, stage_loop_first=True,
    static_mixing=0.1, retarded="half")
