"""Numerical check of the bosonic detailed balance on a real device
(theory 20_self_energy.tex, eq:detailed_balance TODO):

    Sigma^>(omega) = exp(hbar omega / kB T) Sigma^<(omega)

holds for the converged 3-phonon bubble in EQUILIBRIUM (delta_T = 0) and
is broken by a thermal bias. Runs the dense d5a reference (minutes on a
laptop) at delta_T = 0 and delta_T = 10 K and reports the relative
residuals from finite_analysis.physical_tests.detailed_balance_residual.

Run: python phonon/scripts/verify/detailed_balance_device.py [--temp 300]
Output: phonon/scripts/verify/detailed_balance_device.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for p in (str(_REPO / "phonon"), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.finite_analysis.loader import load_system  # noqa: E402
from phonon.finite_analysis.physical_tests import (  # noqa: E402
    detailed_balance_residual,
)
from solver.dense import transmission_finite  # noqa: E402

CFG = "phonon/configs/sinw/sinw100_d5a_vasp_sc4_fc4.yaml"
FMAX = 16.0

ap = argparse.ArgumentParser()
ap.add_argument("--temp", type=float, default=300.0)
ap.add_argument("--nfreq", type=int, default=61)
ap.add_argument("--eta", type=float, default=0.5)
args = ap.parse_args()

b = load_system(str(_REPO / CFG), validate=False, transport_axis=2)
kw = dict(fc3_hdf5=b.meta["fc3_path"], transport_direction="z", n_slabs=1,
          retarded="fft", freq_range_thz=(0.05, FMAX, args.nfreq),
          eta_factor=args.eta, temperature=args.temp,
          max_scba_iter=250, scba_tol=1e-3, conservation_tol=5e-2,
          enforce_asr=True, solver="linear", mixing=0.3, verbose=False)

out = {"temperature_K": args.temp, "nfreq": args.nfreq,
       "eta_factor": args.eta, "config": CFG}
for tag, dT in (("equilibrium_dT0", 0.0), ("biased_dT10", 10.0)):
    r = transmission_finite(b.phonon, delta_T=dT, **kw)
    res = detailed_balance_residual(
        np.asarray(r["self_energy_lesser"]),
        np.asarray(r["self_energy_greater"]),
        np.asarray(r["freqs_thz"]), args.temp)
    out[tag] = {k: float(v) for k, v in res.items()}
    out[tag]["scba_converged"] = bool(r.get("scba_converged", False))
    print(f"{tag}: {out[tag]}")

dst = Path(__file__).with_suffix(".json")
dst.write_text(json.dumps(out, indent=2))
print(f"saved {dst}")
