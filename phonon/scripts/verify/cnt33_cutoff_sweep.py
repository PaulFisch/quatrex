#!/usr/bin/env python
"""(3,3) CNT self-energy cutoff cube: the 8 (sigma, vertex, g) corners at n_slabs=2.

Restores the F30 "cutoff hierarchy" study (lab_notebook_archive.md, section F30)
whose data (phonon/scripts/out/cnt33_cutoff/summary.csv) was purged in 843c3069.
The original driver was phonon/scripts/d5_cutoff_sweep.py pointed at the cnt33
config; this is that driver slimmed to the 8-corner cube and adapted to the
current tree (kwargs verified against solver.dense.transmission on 2026-07-02).

Each corner drives :func:`phonon.solver.dense.transmission_finite` with
  sigma_cutoff  in {None, 0}   -- produced Sigma blocks |I-J| (None = all, 0 = diagonal)
  vertex_cutoff in {None, 0}   -- FC3 vertex slab-distance
  g_cutoff      in {None, 0}   -- G blocks |K-K'| in the inner bubble sum
at fixed n_slabs=2, T=300 K, dT=10 K, dc_handling="interpolate". The corner
order puts the full-coupling reference (sInf_vInf_gInf) FIRST. One npz
checkpoint per corner (same key scheme as the original:
checkpoints/cutoff/s{Inf|0}_v{Inf|0}_g{Inf|0}_dc-interpolate.npz); summary.csv
is rewritten after EVERY corner so partial sweeps are plottable.

Numerics follow the F30 run: freq grid (0.01, 18, 61) auto-extended by the
solver to cover the 3-phonon convolution support (~0.3 THz spacing),
eta_factor 0.7 (eta_w ~ 0.21 THz), safeguarded Anderson, tol 1e-5. Two
deliberate departures from F30 (current-code canon, see
docs/conserving_vertex_findings.md): enforce_asr=False (the ASR projection
corrupts the cnt33 vertex; the raw fit is already ASR-exact) and
zero_mode_projection=False (current solver default). Archived F30 reference
numbers for orientation: G_ball 6.38e8, full G_anh ~4.48e8, corner span
4.09-4.96e8, sInf_vInf_g0 4.39e8 (W/m^2/K).

Usage (sequential, ~20-40 min/corner, ~3-5 h total)::

    cd <repo>
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=8 QUATREX_PHPH_THREADS=2 \
    QUATREX_PHPH_MEMORY_GB=30 nohup python \
        phonon/scripts/verify/cnt33_cutoff_sweep.py --verbose \
        > phonon/scripts/out/cnt33_cutoff/sweep.log 2>&1 &

Figure: phonon/scripts/figures/cnt33_cutoff.py (reads summary.csv).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phonon.finite_analysis.loader import load_system  # noqa: E402
from solver.dense import transmission_finite  # noqa: E402

CONFIG = _REPO / "phonon/configs/cnt/cnt33_vasp.yaml"
OUT_DIR = _REPO / "phonon/scripts/out/cnt33_cutoff"

#: npz fields checkpointed per corner (superset of what the figure needs).
_CHECKPOINT_FIELDS = (
    "freqs_thz",
    "transmission_ballistic",
    "spectral_heat_current_ballistic",
    "spectral_heat_current",
    "heat_current_ballistic",
    "heat_current",
    "thermal_conductance_ballistic",
    "thermal_conductance_anharmonic",
    "heat_flow_conservation",
    "delta_T",
    "n_scba_iterations",
    "convergence_history",
    "scba_converged",
    "scba_residual",
)


def _tag(c: int | None) -> str:
    return "Inf" if c is None else str(int(c))


def corner_key(s: int | None, v: int | None, g: int | None) -> str:
    return f"s{_tag(s)}_v{_tag(v)}_g{_tag(g)}"


def run_corner(bundle, fc3_hdf5: str, s, v, g, args) -> dict:
    t0 = time.time()
    res = transmission_finite(
        bundle.phonon,
        fc3_hdf5=fc3_hdf5,
        transport_direction="z",
        n_slabs=args.n_slabs,
        freq_range_thz=tuple(args.freq_range),
        eta_factor=args.eta_factor,
        temperature=args.t_mean,
        delta_T=args.delta_T,
        max_scba_iter=args.max_scba_iter,
        scba_tol=args.scba_tol,
        conservation_tol=args.conservation_tol,
        mixing=args.mixing,
        solver="anderson",
        anderson_depth=args.anderson_depth,
        sigma_cutoff=s,
        vertex_cutoff=v,
        g_cutoff=g,
        dc_handling="interpolate",
        enforce_asr=False,
        verbose=args.verbose,
    )
    out = {k: res[k] for k in _CHECKPOINT_FIELDS if k in res}
    sigma_R = res.get("self_energy_retarded")
    if sigma_R is not None:
        out["sigma_frobenius_norm"] = float(np.linalg.norm(np.asarray(sigma_R)))
    out["wall_time_seconds"] = time.time() - t0
    return out


def write_summary(rows: list[dict], out_dir: Path) -> None:
    csv_path = out_dir / "summary.csv"
    keys = list(rows[0].keys())
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[summary] wrote {csv_path} ({len(rows)} rows)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--n-slabs", type=int, default=2)
    p.add_argument("--t-mean", type=float, default=300.0)
    p.add_argument("--delta-T", type=float, default=10.0)
    p.add_argument("--freq-range", type=float, nargs=3, default=[0.01, 18.0, 61],
                   metavar=("FMIN", "FMAX", "NPTS"),
                   help="auto-extended by the solver to cover 2*omega_max")
    p.add_argument("--eta-factor", type=float, default=0.7)
    p.add_argument("--max-scba-iter", type=int, default=300)
    p.add_argument("--scba-tol", type=float, default=1e-5)
    p.add_argument("--conservation-tol", type=float, default=1e-1)
    p.add_argument("--mixing", type=float, default=0.3)
    p.add_argument("--anderson-depth", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    ck_dir = out_dir / "checkpoints" / "cutoff"
    ck_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "invocation.json", "w") as f:
        json.dump({"argv": sys.argv,
                   "args": {k: str(v) if isinstance(v, Path) else v
                            for k, v in vars(args).items()}}, f, indent=2)

    print(f"[load ] {args.config}", flush=True)
    bundle = load_system(str(args.config), validate=False, transport_axis=2)
    fc3_hdf5 = str(Path(bundle.meta["fc3_path"]).expanduser().resolve())
    print(f"[load ] fc3.hdf5 : {fc3_hdf5}", flush=True)

    # Full-coupling reference first, then the truncated corners.
    corners = list(product((None, 0), (None, 0), (None, 0)))
    rows: list[dict] = []
    for i, (s, v, g) in enumerate(corners):
        key = corner_key(s, v, g)
        ck = ck_dir / f"{key}_dc-interpolate.npz"
        if ck.exists() and not args.overwrite:
            with np.load(ck, allow_pickle=False) as d:
                point = {k: d[k] for k in d.files}
            print(f"[cache] {key}", flush=True)
        else:
            print(f"[run  ] {i + 1}/8 {key}  (sigma={s}, vertex={v}, g={g})",
                  flush=True)
            point = run_corner(bundle, fc3_hdf5, s, v, g, args)
            np.savez_compressed(
                ck, **{k: np.asarray(v_) for k, v_ in point.items()})
        gb = float(point["thermal_conductance_ballistic"])
        ga = float(point["thermal_conductance_anharmonic"])
        print(f"[done ] {key}  G_ball={gb:.4g} G_anh={ga:.4g} "
              f"conserv={float(point['heat_flow_conservation']):.2e} "
              f"iters={int(point['n_scba_iterations'])} "
              f"resid={float(point['scba_residual']):.2e} "
              f"conv={bool(point['scba_converged'])} "
              f"wall={float(point['wall_time_seconds']):.0f}s", flush=True)
        rows.append({
            "sigma_cutoff": _tag(s), "vertex_cutoff": _tag(v),
            "g_cutoff": _tag(g), "dc_handling": "interpolate",
            "G_ball": gb, "G_anh": ga,
            "conservation": float(point["heat_flow_conservation"]),
            "n_scba_iter": int(point["n_scba_iterations"]),
            "scba_converged": bool(point["scba_converged"]),
            "scba_residual": float(point["scba_residual"]),
            "wall_s": float(point["wall_time_seconds"]),
        })
        write_summary(rows, out_dir)   # incremental: partial sweeps plottable

    print("[done ] cnt33 cutoff cube complete.", flush=True)


if __name__ == "__main__":
    main()
