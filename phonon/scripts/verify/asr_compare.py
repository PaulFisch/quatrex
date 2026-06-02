"""Projected-vs-raw FC3 comparison for the dense SCBA solver.

Runs ``transmission_finite`` twice on the same wire/point -- once with the
new ``enforce_asr`` FC3 ASR projection and once without -- and writes a
side-by-side JSON. This is the decisive validation that the projection
(CLAUDE.md F9) stabilises the small-eta SCBA fixed point without changing
the ballistic baseline.

Usage:
  ... python -u phonon/scripts/verify/asr_compare.py <tag> <config.yaml> <n_slabs>
"""
import sys
import json
import time
import warnings
from pathlib import Path

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
for p in (_REPO, _REPO / "phonon"):
    sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

from phonon.finite_analysis.loader import load_system  # noqa: E402
from phonon.solver.dense import transmission_finite  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "d5a"
config = sys.argv[2] if len(sys.argv) > 2 else \
    "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml"
n_slabs = int(sys.argv[3]) if len(sys.argv) > 3 else 2

out_dir = _REPO / "phonon/scripts/out/asr_compare"
out_dir.mkdir(parents=True, exist_ok=True)

b = load_system(_REPO / config, validate=False, transport_axis=2)
fc3 = str(Path(b.meta["fc3_path"]).expanduser().resolve())

# Physical eta (eta_w ~ 0.11 THz at eta_factor=0.5), full coupling (lambda=1),
# the regime where the un-projected FC3 showed Gamma sign violations on d5a.
COMMON = dict(
    fc3_hdf5=fc3, freq_range_thz=(0.01, 18.0, 41), transport_direction="z",
    temperature=300.0, delta_T=10.0, n_slabs=n_slabs, eta_factor=0.5,
    vertex_scale=1.0, max_scba_iter=35, scba_tol=1e-3, conservation_tol=1e-1,
    mixing=0.3, anderson_mixing=True, anderson_depth=8,
    zero_mode_projection=True, divergence_guard=True, auto_extend_fmax=True,
    verbose=True,
)

results = {}
for label, asr in (("raw", False), ("projected", True)):
    print(f"\n===== {tag} : enforce_asr={asr} =====", flush=True)
    t0 = time.time()
    r = transmission_finite(b.phonon, enforce_asr=asr, **COMMON)
    wall = time.time() - t0
    g_ball = float(r["thermal_conductance_ballistic"])
    g_anh = float(r["thermal_conductance_anharmonic"])
    rec = {
        "enforce_asr": asr,
        "G_ball": g_ball,
        "G_anh": g_anh,
        "ratio": g_anh / g_ball if g_ball else float("nan"),
        "converged": bool(r.get("scba_converged", False)),
        "n_iter": int(r.get("n_scba_iterations", 0)),
        "residual": float(r.get("scba_residual", float("nan"))),
        "conservation": float(r.get("heat_flow_conservation", float("nan"))),
        "wall_s": wall,
    }
    results[label] = rec
    print(f"  -> {json.dumps(rec)}", flush=True)

with open(out_dir / f"{tag}_compare.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[done] wrote {out_dir / (tag + '_compare.json')}", flush=True)
print(json.dumps(results, indent=2), flush=True)
