import sys, warnings
from pathlib import Path
_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
for p in (_REPO, _REPO/"phonon"):
    sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")
from phonon.finite_analysis.loader import load_system
from phonon.solver.dense import transmission_finite
b = load_system(_REPO/"phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml", validate=False, transport_axis=2)
fc3 = str(Path(b.meta["fc3_path"]).expanduser().resolve())
r = transmission_finite(b.phonon, fc3_hdf5=fc3, freq_range_thz=(0.01,18.0,81),
    transport_direction="z", temperature=300.0, delta_T=10.0, n_slabs=1,
    eta_factor=0.5, vertex_scale=0.3, max_scba_iter=4, scba_tol=1e-3, mixing=0.3,
    anderson_mixing=True, zero_mode_projection=True, divergence_guard=True,
    auto_extend_fmax=True, verbose=True)
print("DONE G_anh=%.4e G_ball=%.4e conv=%s iters=%d" % (
    r["thermal_conductance_anharmonic"], r["thermal_conductance_ballistic"],
    r.get("scba_converged"), r.get("n_scba_iterations",0)))
