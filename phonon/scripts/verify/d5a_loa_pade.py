#\!/usr/bin/env python
"""Correct version of the eta/lambda diagnostic: small-lambda LOA sweep at
PHYSICAL eta (where the ballistic baseline is intact), then Pade-extrapolate
G_anh(lambda^2) -> lambda^2=1 (Bescond LOA). Each small-lambda point converges
cheaply. n_slabs=1, coarse grid, bounded memory."""
import sys, warnings, time, json
from pathlib import Path
import numpy as np
_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
for p in (_REPO, _REPO/"phonon"): sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")
from phonon.finite_analysis.loader import load_system
from phonon.solver.dense import transmission_finite
b = load_system(_REPO/"phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml", validate=False, transport_axis=2)
fc3 = str(Path(b.meta["fc3_path"]).expanduser().resolve())
out = _REPO/"phonon/scripts/out/d5a_loa_pade"; out.mkdir(parents=True, exist_ok=True)
lams = [0.1, 0.2, 0.3, 0.4, 0.5]
pts = []
for lam in lams:
    jf = out/f"lam{lam}.json"
    if jf.exists():
        pts.append(json.load(open(jf))); print(f"[cache] lam={lam}", flush=True); continue
    t0=time.time(); print(f"\n##### lam={lam} #####", flush=True)
    r = transmission_finite(b.phonon, fc3_hdf5=fc3, freq_range_thz=(0.01,18.0,41),
        transport_direction="z", temperature=300.0, delta_T=10.0, n_slabs=1,
        eta_factor=0.25, vertex_scale=lam, max_scba_iter=40, scba_tol=2e-3, mixing=0.3,
        anderson_mixing=True, anderson_depth=6, anderson_safeguard=True,
        zero_mode_projection=True, divergence_guard=True, auto_extend_fmax=True,
        retarded="fft", verbose=False)
    rec=dict(lam=lam, converged=bool(r.get("scba_converged")), n_iter=int(r.get("n_scba_iterations",0)),
             G_ball=float(r["thermal_conductance_ballistic"]), G_anh=float(r["thermal_conductance_anharmonic"]),
             conservation=float(r.get("heat_flow_conservation",float("nan"))), wall_s=time.time()-t0)
    json.dump(rec, open(jf,"w"), indent=2); pts.append(rec)
    print("RESULT "+json.dumps(rec), flush=True)
# Pade/poly extrapolation G_anh(lambda^2) -> 1
x=np.array([p["lam"]**2 for p in pts if p["converged"]]); y=np.array([p["G_anh"] for p in pts if p["converged"]])
gb=np.mean([p["G_ball"] for p in pts])
if len(x)>=3:
    for deg in (1,2):
        c=np.polyfit(x,y,deg); g1=np.polyval(c,1.0)
        print(f"poly deg={deg}: G_anh(lambda^2=1) = {g1:.4e}  (G_ball={gb:.4e}, ratio={g1/gb:.3f})", flush=True)
json.dump({"points":pts,"G_ball_mean":gb}, open(out/"summary.json","w"), indent=2)
print("\nALLDONE", flush=True)
