"""Re-validate the d11a anharmonic SCBA at moderate coupling (projected FC3).

Single converged point: lambda=0.3 (the F15 regime, which converges cleanly,
unlike full lambda=1 single-cell which is strong-coupling-stiff for both wires).
Confirms the d11a anharmonic transport is well-posed now that the memory blocker
is gone. Reports G_anh/G_ball, conservation, convergence.
"""
import json
import sys
import time
import warnings
from pathlib import Path

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

from phonon.finite_analysis.loader import load_system  # noqa: E402
from phonon.solver.dense import transmission_finite  # noqa: E402

lam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
out = _REPO / "phonon/scripts/out/asr_compare"
out.mkdir(parents=True, exist_ok=True)

b = load_system(_REPO / "phonon/configs/sinw/sinw100_d11a_vasp_sc4.yaml",
                validate=False, transport_axis=2)
fc3 = str(Path(b.meta["fc3_path"]).expanduser().resolve())

t0 = time.time()
r = transmission_finite(
    b.phonon, fc3_hdf5=fc3, freq_range_thz=(0.01, 18.0, 31),
    transport_direction="z", temperature=300.0, delta_T=10.0, n_slabs=1,
    eta_factor=0.5, vertex_scale=lam, max_scba_iter=45, scba_tol=2e-3,
    conservation_tol=1e-1, mixing=0.3, anderson_mixing=True, anderson_depth=8,
    enforce_asr=True, zero_mode_projection=True, divergence_guard=True,
    verbose=True)
wall = time.time() - t0
g_ball = float(r["thermal_conductance_ballistic"])
g_anh = float(r["thermal_conductance_anharmonic"])
rec = dict(
    wire="d11a", vertex_scale=lam, enforce_asr=True,
    G_ball=g_ball, G_anh=g_anh, ratio=g_anh / g_ball,
    converged=bool(r.get("scba_converged", False)),
    n_iter=int(r.get("n_scba_iterations", 0)),
    residual=float(r.get("scba_residual", float("nan"))),
    conservation=float(r.get("heat_flow_conservation", float("nan"))),
    wall_s=wall)
with open(out / f"d11a_validate_lam{lam}.json", "w") as f:
    json.dump(rec, f, indent=2)
print("RESULT " + json.dumps(rec), flush=True)
