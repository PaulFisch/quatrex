#!/usr/bin/env python
"""Comprehensive cluster diagnostic for the d5a SiNW SCBA.

Runs a suite of solver configurations on the same problem and records
the per-iteration trajectory of every quantity that can break SCBA on
a strongly-anharmonic wire: residual, |Sigma^R|, heat-flow
conservation, causality violations of Gamma_Sigma=i(Sigma^R-Sigma^A),
and the smallest eigenvalue of H_D + Re Sigma^R (negative -> Dyson
denominator singular, the "epic divergence" signature). Configurations
sweep:

  * the new safeguards (Anderson, zero-mode projection, causality
    projection),
  * the solver mode (linear with several damping factors, safeguarded
    Anderson, pure JFNK, anderson+jfnk),
  * a vertex rescaling sweep (Bescond et al., JAP 110, 094517, 2011):
    Sigma scales as lambda^2 in the cubic coupling, so running SCBA at
    lambda in (0,1] -- which converges easily for small lambda -- and
    extrapolating the observable via a Pade fit in lambda^2 produces a
    SCBA-equivalent answer at lambda=1 without iterating an unstable
    fixed point.

Outputs (under ``--out-dir``):

  * ``<run>/info.json``     -- the full settings and final stats.
  * ``<run>/trajectory.csv``-- per-iteration trace (residual,
    |Sigma^R|, conservation, causality, stability eigenvalue, ...).
  * ``<run>/result.npz``    -- spectral heat current, Sigma^R, etc.
  * ``summary.csv``         -- one row per run with the headline numbers.
  * ``trajectory_overlay.pdf`` -- residual + causality + stability vs
    iteration, all configurations on the same panel set.
  * ``vertex_pade.pdf``     -- G_anh vs lambda^2 with the Pade fit.

Usage (cluster)::

    /home/paul/miniconda3/envs/quatrex-dev/bin/python \\
        phonon/scripts/scba_diagnostic.py \\
        --n-slabs 2 \\
        --out-dir phonon/scripts/out/diag \\
        --runs all \\
        --max-scba-iter 80
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phonon.finite_analysis.loader import load_system  # noqa: E402
from phonon.solver.dense import transmission_finite  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration matrix
# ---------------------------------------------------------------------------


@dataclass
class RunSpec:
    name: str
    kwargs: dict = field(default_factory=dict)
    note: str = ""


def _matrix(args) -> list[RunSpec]:
    """Build the configuration matrix. Easy to extend."""
    common = dict(
        eta_factor=args.eta_factor,
        scba_tol=args.scba_tol,
        conservation_tol=args.conservation_tol,
        anderson_depth=args.anderson_depth,
        track_diagnostics=True,
        verbose=False,
    )

    runs = []

    # 1. Baseline: new defaults (anderson safeguarded, zero-mode
    #    projection, auto-extend fmax, divergence guard).
    runs.append(RunSpec(
        "baseline",
        dict(common, solver="anderson", mixing=0.3,
             anderson_mixing=True, anderson_safeguard=True,
             zero_mode_projection=True, divergence_guard=True),
        "anderson safeguarded + zero-mode projection (current default)",
    ))

    # 2. Baseline + causality projection of Sigma^R each iteration.
    runs.append(RunSpec(
        "causality",
        dict(common, solver="anderson", mixing=0.3,
             anderson_mixing=True, anderson_safeguard=True,
             zero_mode_projection=True, divergence_guard=True,
             causality_projection=True),
        "baseline + Gamma_Sigma PSD projection each iter (Pourfath-style)",
    ))

    # 3. Heavy linear damping -- maps the contraction radius.
    for alpha in (0.10, 0.05, 0.02):
        runs.append(RunSpec(
            f"linear_a{alpha:g}",
            dict(common, solver="linear", mixing=alpha,
                 anderson_mixing=False, zero_mode_projection=True,
                 divergence_guard=True),
            f"linear mixing alpha={alpha}",
        ))

    # 4. Pure JFNK -- robust for linearly-unstable fixed points.
    runs.append(RunSpec(
        "jfnk",
        dict(common, solver="jfnk", mixing=0.3,
             zero_mode_projection=True, divergence_guard=False),
        "pure Newton-Krylov (no Anderson warm-up)",
    ))

    # 5. anderson+jfnk: Anderson does best-effort, JFNK takes over.
    runs.append(RunSpec(
        "anderson_jfnk",
        dict(common, solver="anderson+jfnk", mixing=0.3,
             anderson_mixing=True, anderson_safeguard=True,
             zero_mode_projection=True, divergence_guard=True),
        "safeguarded Anderson + JFNK fallback",
    ))

    # 6. Lowest-order Born (no SCBA iteration). The Bescond et al.
    #    "LOA" baseline: one bubble at the bare G, no self-consistency.
    runs.append(RunSpec(
        "one_shot_born",
        dict(common, solver="linear", mixing=0.3,
             anderson_mixing=False, zero_mode_projection=True,
             divergence_guard=True),
        "single bubble at the bare G (LOA, no iteration)",
    ))
    runs[-1].kwargs["max_scba_iter"] = 1  # override

    # 7. Vertex rescaling sweep: Sigma scales as vertex_scale**2.
    #    SCBA at small lambda is easy; Pade in lambda**2 extrapolates
    #    to lambda=1 without iterating an unstable fixed point.
    for lam in args.lambda_sweep:
        runs.append(RunSpec(
            f"lambda_{lam:g}",
            dict(common, solver="anderson+jfnk", mixing=0.3,
                 anderson_mixing=True, anderson_safeguard=True,
                 zero_mode_projection=True, divergence_guard=True,
                 vertex_scale=lam),
            f"vertex rescaled by lambda={lam} (Bescond JAP 2011)",
        ))

    # Apply common overrides.
    for r in runs:
        r.kwargs.setdefault("max_scba_iter", args.max_scba_iter)

    return runs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


_TRAJ_FIELDS = (
    "iter", "resid", "max_abs_Sigma_R", "conservation_err", "J_total",
    "n_violation_points", "max_violation", "mean_violation",
    "omega_at_max_causality",
    "min_eig_HD_plus_ReSigma", "omega_at_min_stability",
)


def _flatten_diag(d):
    """Flatten a single trajectory point's nested causality/stability
    dicts into the flat ``_TRAJ_FIELDS`` row."""
    c = d.get("causality") or {}
    s = d.get("stability") or {}
    return {
        "iter": int(d.get("iter", -1)),
        "resid": float(d.get("resid", float("nan"))),
        "max_abs_Sigma_R": float(d.get("max_abs_Sigma_R", float("nan"))),
        "conservation_err": float(d.get("conservation_err", float("nan"))),
        "J_total": float(d.get("J_total", float("nan"))),
        "n_violation_points": int(c.get("n_violation_points", 0)),
        "max_violation": float(c.get("max_violation", 0.0)),
        "mean_violation": float(c.get("mean_violation", 0.0)),
        "omega_at_max_causality": float(c.get("omega_at_max", 0.0)),
        "min_eig_HD_plus_ReSigma":
            float(s.get("min_eig_HD_plus_ReSigma", float("nan"))),
        "omega_at_min_stability": float(s.get("omega_at_min", 0.0)),
    }


def _save_trajectory_csv(traj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_flatten_diag(d) for d in traj]
    with open(path, "w") as fh:
        fh.write(",".join(_TRAJ_FIELDS) + "\n")
        for r in rows:
            fh.write(",".join(str(r[f]) for f in _TRAJ_FIELDS) + "\n")


def _save_result_npz(res, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = (
        "freqs_thz", "transmission_ballistic",
        "spectral_heat_current_ballistic", "spectral_heat_current",
        "spectral_heat_current_L", "spectral_heat_current_R",
        "heat_current_ballistic", "heat_current",
        "thermal_conductance_ballistic", "thermal_conductance_anharmonic",
        "heat_flow_conservation", "n_scba_iterations",
        "convergence_history", "scba_converged", "scba_residual",
    )
    saveable = {k: np.asarray(res[k]) for k in keep if k in res}
    np.savez_compressed(path, **saveable)


def _run_one(spec: RunSpec, bundle, fc3, args) -> dict[str, Any]:
    t0 = time.time()
    print(f"\n=== {spec.name}  --  {spec.note} ===", flush=True)
    try:
        res = transmission_finite(
            bundle.phonon, fc3_hdf5=fc3,
            freq_range_thz=tuple(args.freq_range),
            transport_direction=args.transport_direction,
            temperature=args.t_mean, delta_T=args.delta_T,
            n_slabs=args.n_slabs,
            sigma_cutoff=None, vertex_cutoff=None, g_cutoff=None,
            dc_handling=args.dc_handling,
            **spec.kwargs,
        )
        ok = True
        err = None
    except Exception as exc:  # noqa: BLE001
        res = None
        ok = False
        err = "".join(traceback.format_exception_only(type(exc), exc))
        print(f"  RUN FAILED: {err}", flush=True)
    wall = time.time() - t0
    return {"spec": spec, "ok": ok, "wall": wall, "result": res, "error": err}


# ---------------------------------------------------------------------------
# Bescond Pade extrapolation from the vertex-scale sweep
# ---------------------------------------------------------------------------


def _pade_extrapolate(lams, vals):
    """Return ``f(lambda=1)`` via a polynomial fit in ``lambda^2``.

    For three or more points we fit ``f(x) = a0 + a1 x + a2 x^2`` and
    evaluate at ``x = 1``; this is the leading Pade [2/0] in
    ``x = lambda^2``. With fewer points we fall back to the highest
    polynomial degree the data supports (linear or constant).
    """
    lams = np.asarray(lams, dtype=float)
    vals = np.asarray(vals, dtype=float)
    finite = np.isfinite(vals)
    if finite.sum() < 1:
        return float("nan"), None
    x = lams[finite] ** 2
    y = vals[finite]
    deg = min(2, len(x) - 1)
    if deg < 0:
        return float("nan"), None
    coef = np.polyfit(x, y, deg)
    extr = float(np.polyval(coef, 1.0))
    return extr, coef.tolist()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_trajectories(runs, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    cmap = mpl.colormaps["tab10"]
    for i, r in enumerate(runs):
        if not r["ok"]:
            continue
        traj = r["result"].get("diagnostics_history", [])
        if not traj:
            continue
        rows = [_flatten_diag(d) for d in traj]
        it = np.array([row["iter"] for row in rows])
        c = cmap(i % 10)

        ax = axes[0, 0]
        y = np.array([row["resid"] for row in rows])
        good = np.isfinite(y) & (y > 0)
        ax.semilogy(it[good], y[good], "-", color=c, lw=1.4,
                    label=r["spec"].name)

        ax = axes[0, 1]
        y = np.array([row["max_abs_Sigma_R"] for row in rows])
        ax.semilogy(it, np.where(y > 0, y, np.nan), "-", color=c, lw=1.4,
                    label=r["spec"].name)

        ax = axes[1, 0]
        y = np.array([row["max_violation"] for row in rows])
        ax.semilogy(it, np.where(y > 0, y, np.nan), "-", color=c, lw=1.4,
                    label=r["spec"].name)

        ax = axes[1, 1]
        y = np.array([row["min_eig_HD_plus_ReSigma"] for row in rows])
        good = np.isfinite(y)
        ax.plot(it[good], y[good], "-", color=c, lw=1.4,
                label=r["spec"].name)

    axes[0, 0].set_xlabel("iteration"); axes[0, 0].set_ylabel("SCF residual")
    axes[0, 0].set_title("SCF residual"); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].set_xlabel("iteration"); axes[0, 1].set_ylabel(r"max$|\Sigma^R|$ (THz$^2$)")
    axes[0, 1].set_title(r"$\Sigma^R$ growth"); axes[0, 1].grid(alpha=0.3)
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel(r"max($-\lambda_{\min}\,\Gamma_\Sigma$) (THz$^2$)")
    axes[1, 0].set_title("Causality violation"); axes[1, 0].grid(alpha=0.3)
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].set_ylabel(r"$\min\lambda(H_D+\mathrm{Re}\,\Sigma^R)$ (THz$^2$)")
    axes[1, 1].set_title("Dynamical stability"); axes[1, 1].grid(alpha=0.3)
    axes[1, 1].axhline(0, color="k", lw=0.6, alpha=0.5)
    axes[0, 0].legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_overlay.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "trajectory_overlay.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_pade(runs, out_dir):
    lam_runs = [r for r in runs if r["ok"] and r["spec"].name.startswith("lambda_")]
    if not lam_runs:
        return None
    lams, G_anh, conv, cons = [], [], [], []
    for r in lam_runs:
        lam = float(r["spec"].kwargs["vertex_scale"])
        res = r["result"]
        lams.append(lam)
        G_anh.append(float(res["thermal_conductance_anharmonic"]))
        conv.append(bool(res.get("scba_converged", False)))
        cons.append(float(res["heat_flow_conservation"]))
    G_anh = np.asarray(G_anh)
    lams = np.asarray(lams)
    conv = np.asarray(conv)

    # Only use converged points for the fit.
    mask = conv & np.isfinite(G_anh)
    extr, coef = _pade_extrapolate(lams[mask], G_anh[mask])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(lams**2, G_anh, "o", color="C0",
            label=r"SCBA at $\lambda$" + " (open=diverged)")
    ax.plot(lams[~mask]**2, G_anh[~mask], "o", color="white",
            markeredgecolor="C0")
    if coef is not None:
        xs = np.linspace(0, 1.05, 50)
        ys = np.polyval(coef, xs)
        ax.plot(xs, ys, "-", color="C0", alpha=0.7,
                label=r"Pad\'e fit (deg $\le$2)")
        ax.plot([1.0], [extr], "*", color="C3", markersize=14,
                label=f"$G_{{anh}}(\\lambda^2{{=}}1)$={extr:.3g}")
    ax.set_xlabel(r"$\lambda^2$")
    ax.set_ylabel(r"$G_\mathrm{anh}$ (W m$^{-2}$ K$^{-1}$)")
    ax.set_title("Bescond rescaling: $G_\\mathrm{anh}$ vs $\\lambda^2$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "vertex_pade.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "vertex_pade.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"extrapolated": extr, "coef": coef, "lambdas": lams.tolist(),
            "G_anh": G_anh.tolist(), "converged": conv.tolist()}


# ---------------------------------------------------------------------------
# Summary + CLI
# ---------------------------------------------------------------------------


def _summary_row(r):
    s = r["spec"]
    res = r["result"]
    if not r["ok"] or res is None:
        return {"name": s.name, "ok": False, "wall_s": r["wall"],
                "error": r.get("error", "")}
    return {
        "name": s.name,
        "ok": True,
        "wall_s": r["wall"],
        "converged": bool(res.get("scba_converged", False)),
        "scba_residual": float(res.get("scba_residual", float("nan"))),
        "n_scba_iterations": int(res.get("n_scba_iterations", 0)),
        "G_ball": float(res["thermal_conductance_ballistic"]),
        "G_anh": float(res["thermal_conductance_anharmonic"]),
        "J_anh_W": float(res["heat_current"]),
        "conservation": float(res["heat_flow_conservation"]),
        "note": s.note,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path,
                   default=_REPO / "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml")
    p.add_argument("--out-dir", type=Path,
                   default=_REPO / "phonon/scripts/out/scba_diag")
    p.add_argument("--transport-direction", default="z", choices=("x", "y", "z"))
    p.add_argument("--n-slabs", type=int, default=2)
    p.add_argument("--t-mean", type=float, default=300.0)
    p.add_argument("--delta-T", type=float, default=10.0)
    p.add_argument("--freq-range", type=float, nargs=3,
                   default=[0.01, 16.0, 32],
                   metavar=("FMIN", "FMAX", "NPTS"),
                   help="will be auto-extended to fmax >= 2*omega_max")
    p.add_argument("--eta-factor", type=float, default=1.0)
    p.add_argument("--dc-handling", default="interpolate",
                   choices=("zero", "interpolate", "keep"))
    p.add_argument("--max-scba-iter", type=int, default=80)
    p.add_argument("--scba-tol", type=float, default=1e-3)
    p.add_argument("--conservation-tol", type=float, default=2e-2)
    p.add_argument("--anderson-depth", type=int, default=8)
    p.add_argument("--lambda-sweep", type=float, nargs="+",
                   default=[0.2, 0.4, 0.6, 0.8])
    p.add_argument("--runs", nargs="+", default=["all"],
                   help="subset by name; 'all' runs everything")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[setup] out_dir: {out}", flush=True)
    print(f"[setup] config:  {args.config}", flush=True)
    with open(out / "invocation.json", "w") as fh:
        json.dump({"argv": sys.argv,
                   "args": {k: (str(v) if isinstance(v, Path) else v)
                            for k, v in vars(args).items()}}, fh, indent=2)

    print("[load ] system bundle ...", flush=True)
    transport_axis = "xyz".index(args.transport_direction)
    bundle = load_system(args.config, validate=False,
                         transport_axis=transport_axis)
    fc3 = Path(bundle.meta.get("fc3_path", "")).expanduser().resolve()
    if not fc3.exists():
        raise FileNotFoundError(f"fc3.hdf5 not found at {fc3}")
    print(f"[load ] fc3.hdf5 : {fc3}", flush=True)

    matrix = _matrix(args)
    if args.runs and args.runs != ["all"]:
        wanted = set(args.runs)
        matrix = [r for r in matrix if r.name in wanted]
    print(f"[plan ] {len(matrix)} configurations: "
          f"{[r.name for r in matrix]}", flush=True)

    results = []
    for spec in matrix:
        run_dir = out / spec.name
        ckpt = run_dir / "result.npz"
        traj_csv = run_dir / "trajectory.csv"
        if ckpt.exists() and traj_csv.exists() and not args.overwrite:
            print(f"=== {spec.name}  (cached, skipping)", flush=True)
            with np.load(ckpt, allow_pickle=False) as data:
                res = {k: data[k] for k in data.files}
            # rehydrate diagnostics from CSV
            traj = []
            with open(traj_csv) as fh:
                header = fh.readline().strip().split(",")
                for line in fh:
                    vals = line.strip().split(",")
                    traj.append({h: float(v) if h != "iter" else int(v)
                                 for h, v in zip(header, vals)})
            res["diagnostics_history"] = traj
            results.append({"spec": spec, "ok": True, "wall": 0.0,
                            "result": res, "error": None})
            continue
        r = _run_one(spec, bundle, str(fc3), args)
        if r["ok"]:
            _save_trajectory_csv(
                r["result"].get("diagnostics_history", []), traj_csv)
            _save_result_npz(r["result"], ckpt)
            with open(run_dir / "info.json", "w") as fh:
                summary = {
                    "spec_name": spec.name,
                    "kwargs": {k: (str(v) if isinstance(v, Path) else v)
                               for k, v in spec.kwargs.items()},
                    "wall_s": r["wall"],
                    "scba_converged": bool(r["result"].get("scba_converged", False)),
                    "scba_residual": float(r["result"].get("scba_residual", float("nan"))),
                    "n_scba_iterations": int(r["result"].get("n_scba_iterations", 0)),
                    "G_anh": float(r["result"]["thermal_conductance_anharmonic"]),
                    "G_ball": float(r["result"]["thermal_conductance_ballistic"]),
                    "conservation": float(r["result"]["heat_flow_conservation"]),
                    "note": spec.note,
                }
                json.dump(summary, fh, indent=2)
        results.append(r)

    print("\n[plot ] writing trajectory overlay...", flush=True)
    _plot_trajectories(results, out)
    print("[plot ] writing Pade extrapolation...", flush=True)
    pade = _plot_pade(results, out)
    if pade is not None:
        with open(out / "pade_extrapolation.json", "w") as fh:
            json.dump(pade, fh, indent=2)
        print(f"[pade ] G_anh(lambda=1) extrapolated = "
              f"{pade['extrapolated']:.4e} W/m^2K", flush=True)

    rows = [_summary_row(r) for r in results]
    with open(out / "summary.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(out / "summary.csv", "w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(
                    str(r.get(k, "")) for k in keys) + "\n")
    print(f"\n[done ] wrote {out / 'summary.csv'} ({len(rows)} rows)",
          flush=True)


if __name__ == "__main__":
    main()
