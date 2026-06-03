#!/usr/bin/env python
"""Broadening (eta) extrapolation for the d5a SiNW anharmonic solver.

The dense SCBA solver evaluates the Green's functions on a finite
frequency grid with a Lorentzian broadening ``eta = eta_factor * d_omega``.
``eta`` must be large enough to resolve the propagator on the grid
(``eta/d_omega ~ 1-2``; see ``verify_discretization``) -- running at
``eta ~ 0`` leaves the propagator spiky between grid points and feeds
noise into the bubble. The physical ``eta -> 0`` answer is therefore
recovered by *extrapolation*: run at a few ``eta_factor`` values where
the solver is well behaved and linearly extrapolate the observable to
``eta_factor = 0``.

This driver runs :func:`phonon.solver.dense.transmission_finite` at
several ``eta_factor`` values, caches each run, and fits

    G(eta_factor) ~ G0 + slope * eta_factor

reporting the intercept ``G0`` as the extrapolated conductance.

Usage (cluster)::

    /home/paul/miniconda3/envs/quatrex-dev/bin/python \\
        phonon/scripts/extrapolate_eta.py \\
        --n-slabs 4 --eta-factors 1.0 1.5 2.0 \\
        --solver anderson+jfnk \\
        --out-dir phonon/scripts/out/eta_extrap
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PHONON_DIR = _REPO_ROOT / "phonon"
if str(_PHONON_DIR) not in sys.path:
    sys.path.insert(0, str(_PHONON_DIR))

from phonon.finite_analysis.loader import load_system  # noqa: E402
from phonon.solver.dense import transmission_finite  # noqa: E402


_CHECKPOINT_FIELDS = (
    "freqs_thz",
    "spectral_heat_current",
    "heat_current",
    "thermal_conductance_anharmonic",
    "thermal_conductance_ballistic",
    "heat_flow_conservation",
    "n_scba_iterations",
    "scba_converged",
    "scba_residual",
)


def _run_one(bundle, fc3_hdf5: Path, eta_factor: float, args) -> dict[str, Any]:
    t0 = time.time()
    res = transmission_finite(
        bundle.phonon,
        fc3_hdf5=str(fc3_hdf5),
        freq_range_thz=tuple(args.freq_range),
        transport_direction=args.transport_direction,
        eta_factor=eta_factor,
        temperature=args.t_mean,
        delta_T=args.delta_T,
        max_scba_iter=args.max_scba_iter,
        scba_tol=args.scba_tol,
        conservation_tol=args.conservation_tol,
        mixing=args.mixing,
        anderson_mixing=True,
        anderson_depth=args.anderson_depth,
        solver=args.solver,
        n_slabs=args.n_slabs,
        verbose=args.verbose,
    )
    out = {k: np.asarray(res[k]) for k in _CHECKPOINT_FIELDS if k in res}
    out["wall_time_seconds"] = np.asarray(time.time() - t0)
    out["eta_factor"] = np.asarray(eta_factor)
    return out


def _load_or_run(ckpt: Path, runner, *, overwrite: bool):
    if ckpt.exists() and not overwrite:
        with np.load(ckpt, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    point = runner()
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ckpt, **point)
    return point


def _linear_extrapolation(eta: np.ndarray, obs: np.ndarray):
    """Least-squares fit ``obs ~ a + b*eta``; return ``(a, b)``."""
    A = np.column_stack([np.ones_like(eta), eta])
    coef, *_ = np.linalg.lstsq(A, obs, rcond=None)
    return float(coef[0]), float(coef[1])


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    repo = _REPO_ROOT
    p.add_argument(
        "--config", type=Path,
        default=repo / "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=repo / "phonon/scripts/out/eta_extrap",
    )
    p.add_argument("--transport-direction", default="z", choices=("x", "y", "z"))
    p.add_argument("--n-slabs", type=int, default=4)
    p.add_argument("--t-mean", type=float, default=300.0)
    p.add_argument("--delta-T", type=float, default=10.0)
    p.add_argument(
        "--freq-range", type=float, nargs=3, default=[0.01, 18.0, 81],
        metavar=("FMIN", "FMAX", "NPTS"),
    )
    p.add_argument(
        "--eta-factors", type=float, nargs="+", default=[1.0, 1.5, 2.0],
        help="eta/d_omega values to run and extrapolate from.",
    )
    p.add_argument("--max-scba-iter", type=int, default=90)
    p.add_argument("--scba-tol", type=float, default=1e-3)
    p.add_argument("--conservation-tol", type=float, default=2e-3)
    p.add_argument("--mixing", type=float, default=0.3)
    p.add_argument("--anderson-depth", type=int, default=8)
    p.add_argument(
        "--solver", default="anderson+jfnk",
        choices=("linear", "anderson", "jfnk", "anderson+jfnk"),
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true", default=False)
    args = p.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] output dir : {out_dir}", flush=True)

    transport_axis = "xyz".index(args.transport_direction)
    bundle = load_system(args.config, validate=False,
                         transport_axis=transport_axis)
    fc3_hdf5 = Path(bundle.meta.get("fc3_path", "")).expanduser().resolve()
    if not fc3_hdf5.exists():
        raise FileNotFoundError(f"fc3.hdf5 not found at {fc3_hdf5}")
    print(f"[load ] fc3.hdf5   : {fc3_hdf5}", flush=True)

    ck_dir = out_dir / "checkpoints"
    points: list[dict[str, Any]] = []
    for eta_factor in sorted(args.eta_factors):
        ckpt = ck_dir / f"eta_{eta_factor:g}.npz"
        print(f"[run ] eta_factor={eta_factor:g}", flush=True)
        point = _load_or_run(
            ckpt,
            lambda ef=eta_factor: _run_one(bundle, fc3_hdf5, ef, args),
            overwrite=args.overwrite,
        )
        g_anh = float(point["thermal_conductance_anharmonic"])
        conv = bool(point.get("scba_converged", np.asarray(True)))
        print(f"[done] eta_factor={eta_factor:g}: G_anh={g_anh:.4g} "
              f"W/m^2K, converged={conv}", flush=True)
        points.append(point)

    eta = np.array([float(p["eta_factor"]) for p in points])
    g_anh = np.array([float(p["thermal_conductance_anharmonic"])
                      for p in points])
    j_anh = np.array([float(p["heat_current"]) for p in points])

    g0, g_slope = _linear_extrapolation(eta, g_anh)
    j0, j_slope = _linear_extrapolation(eta, j_anh)

    summary = {
        "eta_factors": eta.tolist(),
        "G_anh": g_anh.tolist(),
        "heat_current": j_anh.tolist(),
        "G_anh_eta0": g0,
        "G_anh_slope": g_slope,
        "heat_current_eta0": j0,
        "heat_current_slope": j_slope,
        "converged": [bool(p.get("scba_converged", np.asarray(True)))
                      for p in points],
    }
    with open(out_dir / "eta_extrapolation.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[extrap] G_anh(eta->0) = {g0:.4g} W/m^2K  "
          f"(slope {g_slope:+.3g} per eta_factor)", flush=True)
    print(f"[extrap] heat_current(eta->0) = {j0:.4g} W", flush=True)

    # --- plot -----------------------------------------------------------
    mpl.rcParams.update({"font.family": "serif", "axes.grid": True,
                         "grid.alpha": 0.3, "grid.linestyle": "--"})
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(eta, g_anh, "o", color="C0", label="SCBA runs")
    eta_line = np.linspace(0.0, eta.max() * 1.05, 50)
    ax.plot(eta_line, g0 + g_slope * eta_line, "-", color="C0", alpha=0.7,
            label=f"linear fit ($G_0$={g0:.3g})")
    ax.plot([0.0], [g0], "*", color="C3", markersize=14,
            label=r"extrapolated $\eta\to0$")
    ax.set_xlabel(r"$\eta / \Delta\omega$")
    ax.set_ylabel(r"$G_{\mathrm{anh}}$ (W m$^{-2}$ K$^{-1}$)")
    ax.set_title("d5a SiNW: broadening extrapolation")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "eta_extrapolation.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "eta_extrapolation.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[done] wrote {out_dir / 'eta_extrapolation.pdf'}", flush=True)


if __name__ == "__main__":
    main()
