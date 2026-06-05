#!/usr/bin/env python
"""Si nanosheet (thin film) WITH vs WITHOUT the diagrams: q-resolved cross-plane
transport with the dynamic bubble alone vs bubble + the static loop/tadpole
(the SC1 + bubble scheme), now that the static-SE hook handles n_kpts>1.

Uses the high-quality si_big (5x5x5) FC2+FC3 for the bubble vertex and the
sparsely-exported si_big (5x5x5) FC4 for the quartic loop. Reports, per
thickness, the ballistic / +bubble / +bubble+loop+tadpole conductance and the
static renormalisation magnitude.
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
_W = Path(__file__).resolve().parents[1]          # phonon/
for p in (_W.parent, _W):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import warnings; warnings.filterwarnings("ignore")
from scripts.verify.si_film_kappa import load_bulk_si
from solver.dense import transmission_q

ap = argparse.ArgumentParser()
ap.add_argument("--nk", type=int, default=4)
ap.add_argument("--nfreq", type=int, default=81)
ap.add_argument("--n-slabs", type=int, nargs="+", default=[3, 5])
ap.add_argument("--temperature", type=float, default=300.0)
ap.add_argument("--eta-factor", type=float, default=0.1)
ap.add_argument("--max-iter", type=int, default=50)
args = ap.parse_args()

phonon, fc3 = load_bulk_si("reaps/si_big_hiphive")
fc4 = str(_W / "configs/si_primitive/fc3_hiphive_si_big_fc4/fc3.hdf5")
print(f"Si thin film (5x5x5 FC2/FC3 + sparse FC4), nk={args.nk}, T={args.temperature}")

common = dict(
    fc3_hdf5=fc3, q_mesh_transverse=(args.nk, args.nk), retarded="half",
    freq_range_thz=(0.05, 18.0, args.nfreq), eta_factor=args.eta_factor,
    temperature=args.temperature, delta_T=10.0, max_scba_iter=args.max_iter,
    scba_tol=2e-3, conservation_tol=5e-2, enforce_asr=True, solver="linear",
    mixing=0.3, transport_direction="x",
)

for N in args.n_slabs:
    rb = transmission_q(phonon, n_slabs=N, **common)
    Gb = rb["thermal_conductance_ballistic"]; Gbub = rb["thermal_conductance_anharmonic"]
    try:
        rl = transmission_q(phonon, n_slabs=N, loop=True, tadpole=True, fc4_hdf5=fc4,
                            stage_loop_first=True, static_mixing=0.1, **common)
        Gscp = rl["thermal_conductance_anharmonic"]
        ss = rl.get("sigma_static")
        ssn = float(np.linalg.norm(ss)) if ss is not None else 0.0
        scp = (f"+loop+tad G={Gscp:.3e} ({Gscp/Gb:.3f}) ||Sig_stat||={ssn:.2f} "
               f"conv={rl.get('scba_converged')}")
    except Exception as e:
        scp = f"+loop+tad FAILED ({type(e).__name__}: {str(e)[:50]})"
    print(f"  N={N} ({N*0.543/2:.2f} nm): ballistic G={Gb:.3e}  "
          f"+bubble G={Gbub:.3e} (Ga/Gb={Gbub/Gb:.3f})  {scp}")
