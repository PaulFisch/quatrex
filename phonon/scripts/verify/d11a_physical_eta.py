#\!/usr/bin/env python
"""Physical-eta SCBA on the well-conditioned d11a wire (no soft mode, 90-struct fit)."""
import sys, warnings, time, json
from pathlib import Path
_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
for p in (_REPO, _REPO/"phonon"): sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")
from phonon.finite_analysis.loader import load_system
from phonon.solver.dense import transmission_finite
b = load_system(_REPO/"phonon/configs/sinw/sinw100_d11a_vasp_sc4.yaml", validate=False, transport_axis=2)
fc3 = str(Path(b.meta["fc3_path"]).expanduser().resolve())
ndof = int(b.phonon.primitive.masses.shape[0]*3)
print(f"d11a: {ndof} DOF/slab", flush=True)
out = _REPO/"phonon/scripts/out/d11a_physical_eta"; out.mkdir(parents=True, exist_ok=True)
for lam in (0.3, 0.6):
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
    rec["ratio"]=rec["G_anh"]/rec["G_ball"] if rec["G_ball"] else None
    print("RESULT "+json.dumps(rec), flush=True); json.dump(rec, open(out/f"lam{lam}.json","w"), indent=2)
print("\nALLDONE", flush=True)
