#!/usr/bin/env python
"""Joint eta-and-lambda SCBA convergence diagnostic.

The dense SCBA loop returns a fixed-point observable G_anh(eta, lambda)
parameterised by two knobs:

  * eta_w : numerical regulator (broadening of the Dyson denominator
    z^2 = (omega + i*eta_w)^2). The physical answer is at eta_w -> 0,
    so a sequence of eta_w values is linearly extrapolated to zero.

  * lambda : vertex rescaling Phi -> lambda*Phi, so Sigma ~ lambda^2.
    The Bescond et al. (JAP 110, 094517, 2011) trick runs SCBA at
    lambda in (0, 1] where convergence is mechanical and Pade-
    extrapolates G_anh(lambda^2) to lambda^2 = 1 (the lowest-order-
    approximation, LOA).

Both routes target the same SCBA fixed point at the same eta_w. The
*combined* diagnostic compares them: if the eta-extrapolated value at
lambda = 1 agrees with the Pade-extrapolated value at lambda^2 = 1
(within a user-set tolerance), the SCBA at lambda = 1 is internally
consistent with its own analytic continuation -- the bubble closure
captures the physics. A persistent disagreement that survives more
lambda points and a higher-order Pade is a physical statement that the
SCBA closure misses higher-order contributions (vertex corrections,
four-phonon channels). See ``phonon_solver.tex`` appendix subsection
"Joint eta-and-lambda convergence diagnostic" for the full discussion.

Usage (cluster, ~20--40 min on d5a n_slabs=2 with the defaults)::

    /home/paul/miniconda3/envs/quatrex-dev/bin/python \\
        phonon/scripts/scba_convergence_diagnostic.py \\
        --n-slabs 2 --t-mean 300 --delta-T 10 \\
        --out-dir phonon/scripts/out/eta_lambda_diag

Outputs (under ``--out-dir``):

  * ``eta_sweep/<eta>/result.npz``, ``trajectory.csv``, ``info.json``
  * ``lambda_sweep/<lambda>/result.npz``, ``trajectory.csv``, ``info.json``
  * ``convergence_assessment.json`` -- verdict + numbers
  * ``eta_lambda_diagnostic.pdf`` -- 4-panel summary plot
  * ``summary.csv`` -- one row per converged run

The driver caches every run; re-invoking without ``--overwrite`` reuses
existing checkpoints, and ``--replot-only`` regenerates plots and the
verdict from cached data without recomputing anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
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
    """One SCBA invocation = a unique (sweep, parameter) pair."""

    sweep: str          # "eta" or "lambda"
    label: str          # e.g. "eta_4.5" or "lambda_0.4"
    eta_factor: float
    vertex_scale: float
    kwargs: dict = field(default_factory=dict)

    def filename_stem(self) -> str:
        return self.label


def _build_matrix(args) -> list[RunSpec]:
    """Build the joint matrix: an eta sweep at lambda=1 + a lambda sweep
    at the safe ``eta_factor`` defined by ``args.lambda_eta_factor``.

    The two sweeps share the same SCBA settings (mixing, safeguards,
    cutoffs disabled) so that the only meaningful axis between them is
    the (eta, lambda) parameter.
    """
    common = dict(
        max_scba_iter=args.max_scba_iter,
        scba_tol=args.scba_tol,
        conservation_tol=args.conservation_tol,
        anderson_depth=args.anderson_depth,
        mixing=args.mixing,
        solver="anderson+jfnk",
        anderson_mixing=True,
        anderson_safeguard=True,
        zero_mode_projection=True,
        divergence_guard=True,
        track_diagnostics=True,
        verbose=False,
    )

    runs: list[RunSpec] = []

    # eta sweep at lambda = 1
    for eta in args.eta_factors:
        runs.append(RunSpec(
            sweep="eta",
            label=f"eta_{eta:g}",
            eta_factor=float(eta),
            vertex_scale=1.0,
            kwargs=dict(common, eta_factor=float(eta), vertex_scale=1.0),
        ))

    # lambda sweep at the safe eta (always converges by construction)
    for lam in args.lambda_values:
        runs.append(RunSpec(
            sweep="lambda",
            label=f"lambda_{lam:g}",
            eta_factor=float(args.lambda_eta_factor),
            vertex_scale=float(lam),
            kwargs=dict(common, eta_factor=float(args.lambda_eta_factor),
                        vertex_scale=float(lam)),
        ))

    return runs


# ---------------------------------------------------------------------------
# Cache scan + runner
# ---------------------------------------------------------------------------


_TRAJ_FIELDS = (
    "iter", "resid", "max_abs_Sigma_R", "conservation_err", "J_total",
    "n_violation_points", "max_violation", "mean_violation",
    "omega_at_max_causality",
    "min_eig_HD_plus_ReSigma", "omega_at_min_stability",
)


def _flatten_diag(d: dict) -> dict[str, Any]:
    """Flatten a per-iter diagnostic dict (live or CSV-loaded).

    Mirrors ``scba_diagnostic.py``: live runs carry nested
    ``causality`` / ``stability`` sub-dicts; CSV reloads are flat.
    """
    c = d.get("causality") or {}
    s = d.get("stability") or {}

    def _pick(flat_key, nested_key, nested_src, default=0.0, cast=float):
        if flat_key in d:
            v = d[flat_key]
        elif nested_src and nested_key in nested_src:
            v = nested_src[nested_key]
        else:
            v = default
        try:
            return cast(v)
        except (TypeError, ValueError):
            return default

    return {
        "iter": _pick("iter", "iter", None, default=-1, cast=int),
        "resid": _pick("resid", "resid", None, default=float("nan")),
        "max_abs_Sigma_R": _pick("max_abs_Sigma_R", "max_abs_Sigma_R", None,
                                  default=float("nan")),
        "conservation_err": _pick("conservation_err", "conservation_err",
                                   None, default=float("nan")),
        "J_total": _pick("J_total", "J_total", None, default=float("nan")),
        "n_violation_points": _pick("n_violation_points",
                                     "n_violation_points", c, cast=int),
        "max_violation": _pick("max_violation", "max_violation", c),
        "mean_violation": _pick("mean_violation", "mean_violation", c),
        "omega_at_max_causality": _pick("omega_at_max_causality",
                                          "omega_at_max", c),
        "min_eig_HD_plus_ReSigma": _pick("min_eig_HD_plus_ReSigma",
                                           "min_eig_HD_plus_ReSigma", s,
                                           default=float("nan")),
        "omega_at_min_stability": _pick("omega_at_min_stability",
                                          "omega_at_min", s),
    }


def _save_trajectory_csv(traj: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_flatten_diag(d) for d in traj]
    with open(path, "w") as fh:
        fh.write(",".join(_TRAJ_FIELDS) + "\n")
        for r in rows:
            fh.write(",".join(str(r[f]) for f in _TRAJ_FIELDS) + "\n")


def _load_trajectory_csv(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        for line in fh:
            vals = line.strip().split(",")
            row: dict[str, Any] = {}
            for h, v in zip(header, vals):
                if v in ("", "nan"):
                    row[h] = float("nan")
                elif h == "iter":
                    row[h] = int(float(v))
                else:
                    row[h] = float(v)
            out.append(row)
    return out


_NPZ_FIELDS = (
    "freqs_thz", "spectral_heat_current", "spectral_heat_current_ballistic",
    "heat_current", "heat_current_ballistic",
    "thermal_conductance_ballistic", "thermal_conductance_anharmonic",
    "heat_flow_conservation", "n_scba_iterations", "scba_converged",
    "scba_residual", "convergence_history",
)


def _save_result_npz(res: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    saveable = {k: np.asarray(res[k]) for k in _NPZ_FIELDS if k in res}
    np.savez_compressed(path, **saveable)


def _scan_cache(out_dir: Path, runs: list[RunSpec]) -> dict[str, list[str]]:
    """Inspect which (run, sweep) checkpoints already exist."""
    cached: list[str] = []
    missing: list[str] = []
    for r in runs:
        result_npz = out_dir / f"{r.sweep}_sweep" / r.label / "result.npz"
        traj_csv = out_dir / f"{r.sweep}_sweep" / r.label / "trajectory.csv"
        if result_npz.exists() and traj_csv.exists():
            cached.append(r.label)
        else:
            missing.append(r.label)
    return {"cached": cached, "missing": missing}


def _run_one(spec: RunSpec, bundle, fc3: str, args) -> dict[str, Any]:
    t0 = time.time()
    print(f"\n=== {spec.sweep}/{spec.label}  (eta_factor={spec.eta_factor:g}, "
          f"lambda={spec.vertex_scale:g}) ===", flush=True)
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
    return {"spec": spec, "ok": ok, "wall": wall,
            "result": res, "error": err}


# ---------------------------------------------------------------------------
# Extrapolation fits
# ---------------------------------------------------------------------------


def _fit_eta_linear(etas: np.ndarray, vals: np.ndarray):
    """Linear fit G(eta) ~ a + b*eta; return ``(a, b, sigma_a, sigma_b, R^2)``.

    Uses ``numpy.polyfit`` with ``cov=True`` so the intercept uncertainty
    comes from the diagonal of the parameter covariance. ``R^2`` is the
    coefficient of determination, useful as a "is the linear assumption
    even appropriate?" check.
    """
    m = np.isfinite(etas) & np.isfinite(vals)
    if m.sum() < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), \
            float("nan")
    x = etas[m].astype(float)
    y = vals[m].astype(float)
    if m.sum() == 2:
        coef = np.polyfit(x, y, 1)
        return float(coef[1]), float(coef[0]), float("nan"), float("nan"), \
            float("nan")
    coef, cov = np.polyfit(x, y, 1, cov=True)
    yhat = np.polyval(coef, x)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return (float(coef[1]), float(coef[0]),
            float(np.sqrt(cov[1, 1])), float(np.sqrt(cov[0, 0])), r2)


def _fit_lambda_pade(lams: np.ndarray, vals: np.ndarray):
    """Polynomial fit G(lambda^2) ~ sum_{k<=deg} c_k * (lambda^2)^k.

    Returns ``(extrapolated_at_lambda_squared_one, sigma, coef, deg, R^2)``.
    ``deg`` is ``min(2, n_points - 1)`` -- ``[2/0]`` Pade as in
    Bescond et al. (2011), reduced when fewer points are available.
    """
    m = np.isfinite(lams) & np.isfinite(vals)
    if m.sum() < 1:
        return float("nan"), float("nan"), None, -1, float("nan")
    x = lams[m].astype(float) ** 2
    y = vals[m].astype(float)
    deg = min(2, len(x) - 1)
    if deg == 0:
        return float(y[0]), float("nan"), [float(y[0])], 0, float("nan")
    if deg == 1:
        coef = np.polyfit(x, y, 1)
        return float(np.polyval(coef, 1.0)), float("nan"), coef.tolist(), 1, \
            float("nan")
    coef, cov = np.polyfit(x, y, 2, cov=True)
    val_at_1 = float(np.polyval(coef, 1.0))
    # Propagate variance to val_at_1 = c2 + c1 + c0:
    # Var = [1, 1, 1] @ cov @ [1, 1, 1]^T
    e = np.array([1.0, 1.0, 1.0])
    var = float(e @ cov @ e)
    sigma = float(np.sqrt(max(var, 0.0)))
    yhat = np.polyval(coef, x)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return val_at_1, sigma, coef.tolist(), 2, r2


def _verdict(rel_gap: float, args) -> str:
    if not np.isfinite(rel_gap):
        return "undetermined"
    if rel_gap <= args.tol_consistent:
        return "consistent"
    if rel_gap <= args.tol_borderline:
        return "borderline"
    return "inconsistent"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_summary(eta_data, lam_data, eta_fit, lam_fit, args, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    # (a) G_anh vs eta_w at lambda = 1
    ax = axes[0, 0]
    if eta_data["points"]:
        etas = np.array([p["eta_w_thz"] for p in eta_data["points"]])
        g = np.array([p["G_anh"] for p in eta_data["points"]])
        conv = np.array([p["converged"] for p in eta_data["points"]])
        ax.plot(etas[conv], g[conv], "o", color="C0", label="converged")
        if (~conv).any():
            ax.plot(etas[~conv], g[~conv], "x", color="C3", label="diverged")
    a, b, sa, _, _ = eta_fit
    if np.isfinite(a):
        x = np.linspace(0, max(0.01, args.eta_factors_thz_max), 50)
        ax.plot(x, a + b * x, "-", color="C0", alpha=0.7,
                label=rf"$G(\eta_w\!\to\!0)={a:.3g}\pm{sa:.2g}$")
        ax.plot([0.0], [a], "*", color="C3", markersize=14)
    ax.set_xlabel(r"$\eta_w$ (THz)")
    ax.set_ylabel(r"$G_\mathrm{anh}$ (W m$^{-2}$ K$^{-1}$)")
    ax.set_title(r"$\eta$-sweep at $\lambda=1$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # (b) G_anh vs lambda^2 at the safe eta
    ax = axes[0, 1]
    if lam_data["points"]:
        lams = np.array([p["lambda"] for p in lam_data["points"]])
        g = np.array([p["G_anh"] for p in lam_data["points"]])
        conv = np.array([p["converged"] for p in lam_data["points"]])
        ax.plot(lams[conv] ** 2, g[conv], "o", color="C1", label="converged")
        if (~conv).any():
            ax.plot(lams[~conv] ** 2, g[~conv], "x", color="C3",
                    label="diverged")
    val, sig, coef, deg, _ = lam_fit
    if coef is not None:
        xs = np.linspace(0, 1.05, 60)
        ys = np.polyval(coef, xs)
        ax.plot(xs, ys, "-", color="C1", alpha=0.7,
                label=rf"Pad\'e deg={deg}")
        ax.plot([1.0], [val], "*", color="C3", markersize=14,
                label=rf"$G(\lambda^2\!=\!1)={val:.3g}\pm{sig:.2g}$")
    ax.set_xlabel(r"$\lambda^2$")
    ax.set_ylabel(r"$G_\mathrm{anh}$ (W m$^{-2}$ K$^{-1}$)")
    ax.set_title(rf"$\lambda$-sweep at $\eta_w={args.lambda_eta_factor*1.0:g}\times d\omega$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Side-by-side: both extrapolated values with their uncertainty
    ax = axes[1, 0]
    if np.isfinite(eta_fit[0]) and np.isfinite(lam_fit[0]):
        labels = [r"$\eta\!\to\!0$", r"$\lambda^2\!=\!1$"]
        vals = [eta_fit[0], lam_fit[0]]
        errs = [eta_fit[2] if np.isfinite(eta_fit[2]) else 0,
                lam_fit[1] if np.isfinite(lam_fit[1]) else 0]
        ax.errorbar([0, 1], vals, yerr=errs, fmt="o", capsize=6,
                    color="C2", markersize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        gap = abs(vals[0] - vals[1])
        rel = gap / abs(vals[0]) if vals[0] else float("inf")
        verdict = _verdict(rel, args)
        ax.set_title(
            rf"$|\Delta|={gap:.3g}$, $|\Delta|/G^{{\eta}}={rel:.2%}$"
            rf"  $\Rightarrow$ \textbf{{{verdict}}}".replace(
                r"\textbf{", "").replace("}", "")
        )
    ax.set_ylabel(r"$G_\mathrm{anh}$ (W m$^{-2}$ K$^{-1}$)")
    ax.grid(alpha=0.3)

    # (d) Spectral overlay
    ax = axes[1, 1]
    eta_pts = [p for p in eta_data["points"]
               if p["converged"] and "spectral" in p]
    lam_pts = [p for p in lam_data["points"]
               if p["converged"] and "spectral" in p
               and abs(p["lambda"] - max(args.lambda_values)) < 1e-9]
    for p in eta_pts:
        ax.plot(p["freqs_thz"], p["spectral"] * 1e12, "-", alpha=0.6,
                label=rf"$\eta_w={p['eta_w_thz']:.2g}$")
    for p in lam_pts:
        ax.plot(p["freqs_thz"], p["spectral"] * 1e12, "--", alpha=0.7,
                label=rf"$\lambda={p['lambda']:g}$")
    ax.set_xlabel(r"$\omega/2\pi$ (THz)")
    ax.set_ylabel(r"$j(\omega)$ (pW/THz)")
    ax.set_title("Spectral heat current overlay")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "eta_lambda_diagnostic.pdf", bbox_inches="tight")
    fig.savefig(out / "eta_lambda_diagnostic.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _aggregate(run_records: list[dict[str, Any]], args, out_dir: Path):
    """Pull headline numbers from each run for the fits + plots."""
    eta_points = []
    lam_points = []
    for rec in run_records:
        if not rec["ok"] or rec["result"] is None:
            continue
        spec: RunSpec = rec["spec"]
        res = rec["result"]
        # eta_w in THz at the run's grid (we compute it from the
        # auto-extended grid because the user-requested d_omega is
        # preserved through ``_ensure_fmax``):
        freqs = np.asarray(res["freqs_thz"])
        # d_omega from a positive-side neighbour pair
        d_omega = float(freqs[1] - freqs[0])
        eta_w = spec.eta_factor * d_omega
        point = {
            "label": spec.label,
            "eta_factor": float(spec.eta_factor),
            "eta_w_thz": eta_w,
            "lambda": float(spec.vertex_scale),
            "G_anh": float(res["thermal_conductance_anharmonic"]),
            "G_ball": float(res["thermal_conductance_ballistic"]),
            "converged": bool(res.get("scba_converged", False)),
            "scba_residual": float(res.get("scba_residual", float("nan"))),
            "n_scba_iter": int(res.get("n_scba_iterations", 0)),
            "conservation": float(res.get("heat_flow_conservation", 0.0)),
            "wall_s": float(rec["wall"]),
            "freqs_thz": np.asarray(res["freqs_thz"]).tolist(),
            "spectral":
                np.asarray(res["spectral_heat_current"]).tolist(),
        }
        if spec.sweep == "eta":
            eta_points.append(point)
        else:
            lam_points.append(point)
    return {"points": eta_points}, {"points": lam_points}


def _write_csv(out: Path, eta_data: dict, lam_data: dict) -> None:
    """One row per run, both sweeps combined."""
    cols = ["sweep", "label", "eta_factor", "eta_w_thz", "lambda",
            "G_anh", "G_ball", "converged", "scba_residual",
            "n_scba_iter", "conservation", "wall_s"]
    with open(out / "summary.csv", "w") as fh:
        fh.write(",".join(cols) + "\n")
        for src, sweep in ((eta_data, "eta"), (lam_data, "lambda")):
            for p in src["points"]:
                row = [sweep] + [str(p.get(c, "")) for c in cols[1:]]
                fh.write(",".join(row) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path,
                   default=_REPO / "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml")
    p.add_argument("--out-dir", type=Path,
                   default=_REPO / "phonon/scripts/out/eta_lambda_diag")
    p.add_argument("--transport-direction", default="z", choices=("x", "y", "z"))
    p.add_argument("--n-slabs", type=int, default=2)
    p.add_argument("--t-mean", type=float, default=300.0)
    p.add_argument("--delta-T", type=float, default=10.0)
    p.add_argument("--freq-range", type=float, nargs=3,
                   default=[0.01, 18.0, 81],
                   metavar=("FMIN", "FMAX", "NPTS"),
                   help="grid spec; auto-extended to fmax >= 2*omega_max")
    p.add_argument("--dc-handling", default="interpolate",
                   choices=("zero", "interpolate", "keep"))

    # eta sweep --------------------------------------------------------------
    p.add_argument(
        "--eta-factors", type=float, nargs="+",
        default=[9.0, 6.75, 4.5, 3.0],
        help="eta_factor = eta_w / d_omega values to sweep at lambda=1. "
             "Default is decreasing from a safely broad value down to "
             "where SCBA marginally converges. Drop --eta-factor 2.25 "
             "in if you also want a divergent / borderline point.",
    )

    # lambda sweep -----------------------------------------------------------
    p.add_argument(
        "--lambda-values", type=float, nargs="+",
        default=[0.3, 0.5, 0.7, 0.9],
        help="lambda values for the Bescond LOA Pade sweep "
             "(vertex_scale, so Sigma scales as lambda^2). Default "
             "stays below 1 so each point converges mechanically.",
    )
    p.add_argument(
        "--lambda-eta-factor", type=float, default=6.75,
        help="The eta_factor used for the lambda sweep -- pick a value "
             "from --eta-factors where SCBA converges cleanly so the "
             "Pade extrapolation is at a regulated SCBA fixed point.",
    )

    # SCBA solver knobs ------------------------------------------------------
    p.add_argument(
        "--max-scba-iter", type=int, default=120,
        help="Lower than scba_diagnostic.py's 200 so the whole "
             "8-point diagnostic finishes in 20-40 min on n_slabs=2.",
    )
    p.add_argument(
        "--scba-tol", type=float, default=1e-3,
        help="Looser than 1e-6 to keep wall-clock affordable; reaches "
             "the geometric-convergence plateau on d5a in 30-50 iters.",
    )
    p.add_argument("--conservation-tol", type=float, default=1e-1)
    p.add_argument("--mixing", type=float, default=0.3)
    p.add_argument("--anderson-depth", type=int, default=8)

    # Verdict tolerances -----------------------------------------------------
    p.add_argument(
        "--tol-consistent", type=float, default=0.05,
        help="|Delta|/G^eta <= this -> verdict 'consistent'.",
    )
    p.add_argument(
        "--tol-borderline", type=float, default=0.20,
        help="|Delta|/G^eta <= this -> 'borderline'; else 'inconsistent'.",
    )

    # Modes ------------------------------------------------------------------
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--replot-only", action="store_true",
                   help="Skip all SCBA compute and re-derive the verdict "
                        "and plots from existing checkpoints.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.eta_factors_thz_max = max(args.eta_factors) * 0.5  # for x-axis range

    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[setup] out_dir: {out}", flush=True)
    print(f"[setup] config : {args.config}", flush=True)
    with open(out / "invocation.json", "w") as fh:
        json.dump({"argv": sys.argv,
                   "args": {k: (str(v) if isinstance(v, Path) else v)
                            for k, v in vars(args).items()}}, fh, indent=2)

    runs = _build_matrix(args)
    cache = _scan_cache(out, runs)
    print(
        f"[cache] {len(cache['cached'])} cached, "
        f"{len(cache['missing'])} to compute "
        f"(overwrite={args.overwrite}, replot_only={args.replot_only})",
        flush=True,
    )
    for r in runs:
        flag = "cached" if r.label in cache["cached"] else "missing"
        print(f"  {r.sweep:6s} {r.label:14s}  {flag}", flush=True)

    need_compute = (not args.replot_only) and (
        len(cache["missing"]) > 0 or args.overwrite
    )
    if need_compute:
        print("[load ] system bundle ...", flush=True)
        transport_axis = "xyz".index(args.transport_direction)
        bundle = load_system(args.config, validate=False,
                             transport_axis=transport_axis)
        fc3 = Path(bundle.meta.get("fc3_path", "")).expanduser().resolve()
        if not fc3.exists():
            raise FileNotFoundError(f"fc3.hdf5 not found at {fc3}")
        print(f"[load ] fc3.hdf5 : {fc3}", flush=True)
    else:
        bundle = None
        fc3 = None
        if not args.replot_only:
            print("[load ] skipping system bundle (all cached)", flush=True)

    # Run / load every requested point.
    run_records: list[dict[str, Any]] = []
    for spec in runs:
        run_dir = out / f"{spec.sweep}_sweep" / spec.label
        ckpt = run_dir / "result.npz"
        traj = run_dir / "trajectory.csv"
        if ckpt.exists() and traj.exists() and not args.overwrite:
            print(f"  [skip] {spec.sweep}/{spec.label} (cached)", flush=True)
            res = _load_npz(ckpt)
            run_records.append({"spec": spec, "ok": True, "wall": 0.0,
                                "result": res, "error": None})
            continue
        if args.replot_only:
            print(f"  [skip] {spec.sweep}/{spec.label} (no cache, "
                  f"--replot-only)", flush=True)
            continue
        rec = _run_one(spec, bundle, str(fc3), args)
        if rec["ok"] and rec["result"] is not None:
            _save_trajectory_csv(
                rec["result"].get("diagnostics_history", []), traj)
            _save_result_npz(rec["result"], ckpt)
            with open(run_dir / "info.json", "w") as fh:
                json.dump({
                    "sweep": spec.sweep,
                    "label": spec.label,
                    "eta_factor": float(spec.eta_factor),
                    "vertex_scale": float(spec.vertex_scale),
                    "wall_s": float(rec["wall"]),
                    "scba_converged": bool(
                        rec["result"].get("scba_converged", False)),
                    "scba_residual": float(
                        rec["result"].get("scba_residual", float("nan"))),
                    "n_scba_iterations": int(
                        rec["result"].get("n_scba_iterations", 0)),
                    "G_anh": float(
                        rec["result"]["thermal_conductance_anharmonic"]),
                    "G_ball": float(
                        rec["result"]["thermal_conductance_ballistic"]),
                    "conservation": float(
                        rec["result"]["heat_flow_conservation"]),
                }, fh, indent=2)
        run_records.append(rec)

    # Aggregate, fit, write verdict + plots.
    print("\n[fit  ] computing extrapolations ...", flush=True)
    eta_data, lam_data = _aggregate(run_records, args, out)

    etas = np.array([p["eta_w_thz"] for p in eta_data["points"]
                     if p["converged"]])
    ge = np.array([p["G_anh"] for p in eta_data["points"]
                   if p["converged"]])
    eta_fit = _fit_eta_linear(etas, ge)

    lams = np.array([p["lambda"] for p in lam_data["points"]
                     if p["converged"]])
    gl = np.array([p["G_anh"] for p in lam_data["points"]
                   if p["converged"]])
    lam_fit = _fit_lambda_pade(lams, gl)

    eta_extr = eta_fit[0]
    lam_extr = lam_fit[0]
    gap = (abs(eta_extr - lam_extr)
           if np.isfinite(eta_extr) and np.isfinite(lam_extr)
           else float("nan"))
    rel_gap = (gap / abs(eta_extr)
               if np.isfinite(eta_extr) and eta_extr != 0
               else float("nan"))
    verdict = _verdict(rel_gap, args)

    assessment = {
        "verdict": verdict,
        "tol_consistent": float(args.tol_consistent),
        "tol_borderline": float(args.tol_borderline),
        "eta_extrapolation": {
            "G_anh_at_eta_zero": eta_extr,
            "slope_per_thz": eta_fit[1],
            "sigma_intercept": eta_fit[2],
            "sigma_slope": eta_fit[3],
            "R2": eta_fit[4],
            "n_converged_points": int(len(etas)),
        },
        "lambda_pade": {
            "G_anh_at_lambda_one": lam_extr,
            "sigma": lam_fit[1],
            "coefficients_lambda2_series": lam_fit[2],
            "polynomial_degree": int(lam_fit[3]),
            "R2": lam_fit[4],
            "n_converged_points": int(len(lams)),
            "fixed_eta_factor": float(args.lambda_eta_factor),
        },
        "gap": {"absolute": gap, "relative_to_eta_extrapolation": rel_gap},
    }
    with open(out / "convergence_assessment.json", "w") as fh:
        json.dump(assessment, fh, indent=2)

    print(
        f"  G(eta->0, lambda=1)  = {eta_extr:.4g} +/- {eta_fit[2]:.2g}",
        flush=True,
    )
    print(
        f"  G(lambda^2=1, eta_sf) = {lam_extr:.4g} +/- {lam_fit[1]:.2g}",
        flush=True,
    )
    print(f"  |Delta|/G_eta = {rel_gap:.2%}  -> verdict: {verdict}",
          flush=True)

    _write_csv(out, eta_data, lam_data)
    _plot_summary(eta_data, lam_data, eta_fit, lam_fit, args, out)

    print(f"\n[done ] wrote {out / 'convergence_assessment.json'}, "
          f"{out / 'eta_lambda_diagnostic.pdf'}, "
          f"{out / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
