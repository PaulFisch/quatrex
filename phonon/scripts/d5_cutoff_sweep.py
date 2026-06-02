#!/usr/bin/env python
"""d5a SiNW SCBA cutoff sweep: vary sigma_cutoff, vertex_cutoff, g_cutoff,
and dc_handling at fixed (T, dT, L) and record how the heat-current,
conductance, Sigma norm, and SCBA convergence depend on each cutoff.

The sweep drives :func:`phonon.solver.dense.transmission_finite` with
every approximation disabled by default (full multi-slab self-energy).
A cartesian product over the requested cutoff axes is built; each point
is cached to ``out/checkpoints/cutoff/<key>.npz`` and ``--replot-only``
regenerates the plots from cache without re-running the SCBA.

Usage (cluster)::

    /home/paul/miniconda3/envs/quatrex-dev/bin/python \\
        phonon/scripts/d5_cutoff_sweep.py \\
        --n-slabs 4 --t-mean 300 --delta-T 10 \\
        --max-scba-iter 90 --scba-tol 1e-3 --anderson-mixing \\
        --out-dir phonon/scripts/out/d5_cutoff
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
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


# =============================================================================
# Plot style
# =============================================================================

PLOT_STYLE: dict[str, Any] = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.6,
    "lines.markersize": 5.5,
}


def _apply_style() -> None:
    mpl.rcParams.update(PLOT_STYLE)


# =============================================================================
# Sweep key
# =============================================================================


_NONE_SENTINEL = -1


def _encode(c: int | None) -> int:
    """Encode an integer cutoff or None into a sortable int.

    ``None`` (no truncation) is stored as -1 so it sorts after every
    finite cutoff. Keeps file keys unambiguous and the sweep tables
    sortable without special-casing None.
    """
    return _NONE_SENTINEL if c is None else int(c)


def _decode(x: int) -> int | None:
    return None if int(x) == _NONE_SENTINEL else int(x)


@dataclass(frozen=True)
class CutoffKey:
    sigma_cutoff: int  # encoded; -1 = None = no truncation
    vertex_cutoff: int
    g_cutoff: int
    dc_handling: str

    @property
    def sigma_cutoff_value(self) -> int | None:
        return _decode(self.sigma_cutoff)

    @property
    def vertex_cutoff_value(self) -> int | None:
        return _decode(self.vertex_cutoff)

    @property
    def g_cutoff_value(self) -> int | None:
        return _decode(self.g_cutoff)

    def filename(self) -> str:
        def _tag(c: int) -> str:
            return "Inf" if c == _NONE_SENTINEL else str(c)

        return (
            f"s{_tag(self.sigma_cutoff)}_v{_tag(self.vertex_cutoff)}_"
            f"g{_tag(self.g_cutoff)}_dc-{self.dc_handling}.npz"
        )

    def label(self) -> str:
        def _tag(c: int) -> str:
            return r"\infty" if c == _NONE_SENTINEL else str(c)

        return (
            f"$\\sigma={_tag(self.sigma_cutoff)},\\ "
            f"V={_tag(self.vertex_cutoff)},\\ "
            f"G={_tag(self.g_cutoff)},\\ "
            f"\\mathrm{{dc}}={self.dc_handling}$"
        )


# =============================================================================
# Run a single point
# =============================================================================


_CHECKPOINT_FIELDS = (
    "freqs_thz",
    "transmission_ballistic",
    "spectral_heat_current_ballistic",
    "spectral_heat_current",
    "spectral_heat_current_L",
    "spectral_heat_current_R",
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


def _run_one(
    *,
    bundle,
    fc3_hdf5: Path,
    key: CutoffKey,
    temperature: float,
    delta_T: float,
    n_slabs: int,
    transport_direction: str,
    freq_range: tuple[float, float, int],
    eta_factor: float,
    max_scba_iter: int,
    scba_tol: float,
    conservation_tol: float,
    mixing: float,
    anderson_mixing: bool,
    anderson_depth: int,
    solver: str | None,
    anderson_safeguard: bool,
    zero_mode_projection: bool,
    gate_on_conservation: bool,
    divergence_guard: bool,
    enforce_asr: bool,
    verbose: bool,
) -> dict[str, Any]:
    t_start = time.time()
    res = transmission_finite(
        bundle.phonon,
        fc3_hdf5=str(fc3_hdf5),
        freq_range_thz=freq_range,
        transport_direction=transport_direction,
        eta_factor=eta_factor,
        temperature=temperature,
        delta_T=delta_T,
        max_scba_iter=max_scba_iter,
        scba_tol=scba_tol,
        conservation_tol=conservation_tol,
        mixing=mixing,
        anderson_mixing=anderson_mixing,
        anderson_depth=anderson_depth,
        solver=solver,
        anderson_safeguard=anderson_safeguard,
        zero_mode_projection=zero_mode_projection,
        gate_on_conservation=gate_on_conservation,
        divergence_guard=divergence_guard,
        n_slabs=n_slabs,
        verbose=verbose,
        sigma_cutoff=key.sigma_cutoff_value,
        vertex_cutoff=key.vertex_cutoff_value,
        g_cutoff=key.g_cutoff_value,
        dc_handling=key.dc_handling,
        enforce_asr=enforce_asr,
    )
    wall = time.time() - t_start

    out = {k: res[k] for k in _CHECKPOINT_FIELDS if k in res}
    sigma_R = res.get("self_energy_retarded")
    if sigma_R is not None:
        sigma_R = np.asarray(sigma_R)
        out["sigma_frobenius_norm"] = float(np.linalg.norm(sigma_R))
        out["re_sigma_frobenius_norm"] = float(np.linalg.norm(sigma_R.real))
        out["im_sigma_frobenius_norm"] = float(np.linalg.norm(sigma_R.imag))
    out["wall_time_seconds"] = wall
    out["enforce_asr"] = bool(enforce_asr)
    return out


def _load_or_run(
    checkpoint_path: Path,
    key: CutoffKey,
    runner,
    *,
    overwrite: bool,
    replot_only: bool,
) -> dict[str, Any] | None:
    if checkpoint_path.exists() and not overwrite:
        with np.load(checkpoint_path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files} | {"_cached": True}
    if replot_only:
        print(
            f"[skip] no cache for {checkpoint_path.name} "
            "and --replot-only is set",
            flush=True,
        )
        return None
    print(f"[run ] {checkpoint_path.name} : {asdict(key)}", flush=True)
    point = runner(key)
    g_ball = float(point["thermal_conductance_ballistic"])
    g_anh = float(point["thermal_conductance_anharmonic"])
    print(
        f"[done] {checkpoint_path.name}  "
        f"({float(point['wall_time_seconds']):.1f} s, "
        f"G_ball={g_ball:.3g}, G_anh={g_anh:.3g})",
        flush=True,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    saveable = {
        k: np.asarray(v) for k, v in point.items() if not k.startswith("_")
    }
    np.savez_compressed(checkpoint_path, **saveable)
    return point


# =============================================================================
# Sweep driver
# =============================================================================


def _resolve_axis(
    arg_values: list[int | str] | None,
    default_values: list[int | None],
) -> list[int | None]:
    """Convert a CLI list of cutoff values to a list of int|None.

    Accepts the strings 'none'/'inf'/'None' to mean "no truncation".
    """
    if arg_values is None:
        return default_values
    out: list[int | None] = []
    for v in arg_values:
        if isinstance(v, str) and v.lower() in ("none", "inf", "infty"):
            out.append(None)
        else:
            out.append(int(v))
    return out


def sweep_cutoffs(
    *,
    bundle,
    fc3_hdf5: Path,
    out_dir: Path,
    args,
) -> list[dict[str, Any]]:
    ck_dir = out_dir / "checkpoints" / (
        "cutoff_asr" if args.enforce_asr else "cutoff_raw")
    ck_dir.mkdir(parents=True, exist_ok=True)

    sigma_axis = _resolve_axis(args.sigma_cutoffs, args.default_sigma)
    vertex_axis = _resolve_axis(args.vertex_cutoffs, args.default_vertex)
    g_axis = _resolve_axis(args.g_cutoffs, args.default_g)
    dc_axis = args.dc_handling

    print(
        f"[sweep] axes: sigma={sigma_axis}, vertex={vertex_axis}, "
        f"g={g_axis}, dc={dc_axis}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    for s_c, v_c, g_c, dc in product(sigma_axis, vertex_axis, g_axis, dc_axis):
        key = CutoffKey(
            sigma_cutoff=_encode(s_c),
            vertex_cutoff=_encode(v_c),
            g_cutoff=_encode(g_c),
            dc_handling=str(dc),
        )

        def runner(k: CutoffKey) -> dict[str, Any]:
            return _run_one(
                bundle=bundle, fc3_hdf5=fc3_hdf5,
                key=k,
                temperature=args.t_mean,
                delta_T=args.delta_T,
                n_slabs=args.n_slabs,
                transport_direction=args.transport_direction,
                freq_range=tuple(args.freq_range),
                eta_factor=args.eta_factor,
                max_scba_iter=args.max_scba_iter,
                scba_tol=args.scba_tol,
                conservation_tol=args.conservation_tol,
                mixing=args.mixing,
                anderson_mixing=args.anderson_mixing,
                anderson_depth=args.anderson_depth,
                solver=args.solver,
                anderson_safeguard=args.anderson_safeguard,
                zero_mode_projection=args.zero_mode_projection,
                gate_on_conservation=args.gate_on_conservation,
                divergence_guard=args.divergence_guard,
                enforce_asr=args.enforce_asr,
                verbose=args.verbose,
            )

        point = _load_or_run(
            ck_dir / key.filename(), key, runner,
            overwrite=args.overwrite, replot_only=args.replot_only,
        )
        if point is None:
            continue
        point["_key"] = key
        results.append(point)

    return results


# =============================================================================
# Plotting
# =============================================================================


def _save_fig(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)


def _axis_label(name: str) -> str:
    return {
        "sigma_cutoff": r"$\sigma$ cutoff $|I-J|$",
        "vertex_cutoff": r"vertex cutoff",
        "g_cutoff": r"$G$ cutoff $|K-K'|$",
    }[name]


def _x_value(c: int) -> float:
    return float(c) if c != _NONE_SENTINEL else float("inf")


def _format_cutoff(c: int) -> str:
    return r"$\infty$" if c == _NONE_SENTINEL else str(c)


def plot_observable_vs_cutoff(
    results: list[dict[str, Any]],
    out_dir: Path,
    *,
    x_field: str,
    y_field: str,
    y_label: str,
    title: str,
    fname: str,
    log_y: bool = False,
) -> None:
    """Group results by (other-cutoffs, dc_handling) and plot y_field vs x_field."""
    if not results:
        return

    plot_dir = out_dir / "plots" / "cutoff"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Group by every key field except x_field.
    other_fields = [
        f for f in ("sigma_cutoff", "vertex_cutoff", "g_cutoff", "dc_handling")
        if f != x_field
    ]
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in results:
        k: CutoffKey = r["_key"]
        gkey = tuple(getattr(k, f) for f in other_fields)
        groups.setdefault(gkey, []).append(r)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    cmap = mpl.colormaps["viridis"]
    n_groups = max(1, len(groups))
    for i, (gkey, group_rs) in enumerate(sorted(groups.items())):
        pts = sorted(group_rs, key=lambda r: _x_value(getattr(r["_key"], x_field)))
        xs = [_x_value(getattr(r["_key"], x_field)) for r in pts]
        ys = [float(np.asarray(r[y_field])) if not hasattr(r[y_field], 'shape')
              or np.asarray(r[y_field]).ndim == 0
              else float(np.asarray(r[y_field]).item()) for r in pts]
        # Replace inf x by a sentinel one unit past the largest finite x.
        finite_xs = [x for x in xs if np.isfinite(x)]
        x_inf_sub = max(finite_xs) + 1 if finite_xs else 0
        xs_plot = [x if np.isfinite(x) else x_inf_sub for x in xs]
        color = cmap(i / n_groups)
        label_parts = []
        for f, v in zip(other_fields, gkey):
            if f == "dc_handling":
                label_parts.append(f"dc={v}")
            else:
                label_parts.append(f"{f[0]}={_format_cutoff(v)}")
        label = ", ".join(label_parts)
        ax.plot(xs_plot, ys, "o-", color=color, label=label)

    # Annotate the inf-substitute on the x-axis.
    if not all(np.isfinite(_x_value(getattr(r["_key"], x_field))) for r in results):
        xtick_locs = ax.get_xticks()
        if x_inf_sub not in xtick_locs:
            xtick_locs = list(xtick_locs) + [x_inf_sub]
            xtick_labels = [
                r"$\infty$" if x == x_inf_sub else str(int(x))
                for x in xtick_locs
            ]
            ax.set_xticks(xtick_locs)
            ax.set_xticklabels(xtick_labels)

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(_axis_label(x_field))
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="best")
    _save_fig(fig, plot_dir / fname)


def plot_convergence_overlay(
    results: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    valid = [
        r for r in results
        if "convergence_history" in r
        and np.asarray(r["convergence_history"]).size > 0
    ]
    if not valid:
        return

    plot_dir = out_dir / "plots" / "cutoff"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cmap = mpl.colormaps["plasma"]
    for i, r in enumerate(valid):
        hist = np.asarray(r["convergence_history"], dtype=float)
        iters = np.arange(2, 2 + hist.size)
        color = cmap(i / max(1, len(valid) - 1))
        key: CutoffKey = r["_key"]
        ax.semilogy(iters, hist, "o-", color=color, label=key.label())
    ax.set_xlabel("SCBA iteration")
    ax.set_ylabel(r"SCF residual $\|G(\Sigma)-\Sigma\| / \|\Sigma\|$")
    ax.set_title("d5a SiNW: SCBA convergence across cutoffs")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _save_fig(fig, plot_dir / "convergence_overlay")


def _autoclip_xlim(curves, tol_rel=1e-4, pad=2.0):
    """Tightest x-window that contains every curve down to ``tol_rel`` of
    its own peak. The bubble convolution grid runs to ``2*omega_max``
    for FFT/KK reasons, but the *observable* ``j(omega) ~ (n_L-n_R)``
    is Bose-suppressed past a few times ``kT/h``; showing the full grid
    just squashes the meaningful range. Returns ``(xmin, xmax)`` or
    ``None`` if no curve has any weight.
    """
    significant = []
    for freqs, vals in curves:
        peak = float(np.max(np.abs(vals)))
        if peak <= 0:
            continue
        mask = np.abs(vals) > tol_rel * peak
        if mask.any():
            significant.append(float(np.asarray(freqs)[mask].max()))
            significant.append(float(np.asarray(freqs)[mask].min()))
    if not significant:
        return None
    return (max(0.0, min(significant) - pad), max(significant) + pad)


def plot_spectral_J_overlay(
    results: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    if not results:
        return
    plot_dir = out_dir / "plots" / "cutoff"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cmap = mpl.colormaps["viridis"]
    curves = []
    for i, r in enumerate(results):
        freqs = np.asarray(r["freqs_thz"])
        j_anh = np.asarray(r["spectral_heat_current"]) * 1e12
        color = cmap(i / max(1, len(results) - 1))
        key: CutoffKey = r["_key"]
        ax.plot(freqs, j_anh, color=color, label=key.label())
        curves.append((freqs, j_anh))
    # Ballistic reference (same for every point).
    j_ball = np.asarray(results[0]["spectral_heat_current_ballistic"]) * 1e12
    freqs_ball = np.asarray(results[0]["freqs_thz"])
    ax.plot(freqs_ball, j_ball, "k--", lw=1.0, alpha=0.7, label="ballistic")
    curves.append((freqs_ball, j_ball))
    xlim = _autoclip_xlim(curves)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(r"Frequency $\omega/2\pi$ (THz)")
    ax.set_ylabel(r"Spectral heat current (pW/THz)")
    ax.set_title("d5a SiNW: spectral $J(\\omega)$ across cutoffs")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _save_fig(fig, plot_dir / "spectral_J_overlay")


def plot_all(results: list[dict[str, Any]], out_dir: Path) -> None:
    if not results:
        print("[warn] no results to plot", flush=True)
        return

    for x_field, fname_suffix in (
        ("sigma_cutoff", "sigma_cutoff"),
        ("vertex_cutoff", "vertex_cutoff"),
        ("g_cutoff", "g_cutoff"),
    ):
        plot_observable_vs_cutoff(
            results, out_dir,
            x_field=x_field,
            y_field="thermal_conductance_anharmonic",
            y_label=r"$G_{\mathrm{anh}}$ (W m$^{-2}$ K$^{-1}$)",
            title=f"d5a SiNW: $G_{{\\mathrm{{anh}}}}$ vs {fname_suffix}",
            fname=f"conductance_vs_{fname_suffix}",
        )
        plot_observable_vs_cutoff(
            results, out_dir,
            x_field=x_field,
            y_field="sigma_frobenius_norm",
            y_label=r"$||\Sigma^R||_F$ (THz$^2$)",
            title=f"d5a SiNW: $\\Sigma^R$ norm vs {fname_suffix}",
            fname=f"sigma_norm_vs_{fname_suffix}",
            log_y=True,
        )
        plot_observable_vs_cutoff(
            results, out_dir,
            x_field=x_field,
            y_field="wall_time_seconds",
            y_label=r"wall time (s)",
            title=f"d5a SiNW: wall time vs {fname_suffix}",
            fname=f"wall_time_vs_{fname_suffix}",
            log_y=True,
        )

    plot_convergence_overlay(results, out_dir)
    plot_spectral_J_overlay(results, out_dir)


# =============================================================================
# Summary CSV
# =============================================================================


def write_summary(
    results: list[dict[str, Any]], out_dir: Path,
) -> None:
    rows = []
    for r in results:
        key: CutoffKey = r["_key"]
        rows.append({
            "sigma_cutoff": "Inf" if key.sigma_cutoff == _NONE_SENTINEL
            else key.sigma_cutoff,
            "vertex_cutoff": "Inf" if key.vertex_cutoff == _NONE_SENTINEL
            else key.vertex_cutoff,
            "g_cutoff": "Inf" if key.g_cutoff == _NONE_SENTINEL
            else key.g_cutoff,
            "dc_handling": key.dc_handling,
            "enforce_asr": bool(r.get("enforce_asr", False)),
            "G_ball": float(r["thermal_conductance_ballistic"]),
            "G_anh": float(r["thermal_conductance_anharmonic"]),
            "J_anh_pW": float(r["heat_current"]) * 1e12,
            "conservation": float(r.get("heat_flow_conservation", 0.0)),
            "n_scba_iter": int(r.get("n_scba_iterations", 0)),
            "scba_converged": bool(r.get("scba_converged", True)),
            "scba_residual": float(r.get("scba_residual", float("nan"))),
            "sigma_norm": float(r.get("sigma_frobenius_norm", 0.0)),
            "wall_s": float(r.get("wall_time_seconds", 0.0)),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w") as f:
        if not rows:
            f.write("# no results\n")
            return
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[summary] wrote {csv_path} ({len(rows)} rows)", flush=True)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    repo = _REPO_ROOT
    p.add_argument(
        "--config", type=Path,
        default=repo / "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml",
        help="d5a YAML config (default: %(default)s).",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=repo / "phonon/scripts/out/d5_cutoff_sweep",
        help="Output directory for checkpoints + plots + summary.",
    )
    p.add_argument(
        "--transport-direction", default="z", choices=("x", "y", "z"),
    )

    # System ---------------------------------------------------------------
    p.add_argument("--n-slabs", type=int, default=4)
    p.add_argument("--t-mean", type=float, default=300.0)
    p.add_argument("--delta-T", type=float, default=10.0)
    p.add_argument(
        "--freq-range", type=float, nargs=3, default=[0.01, 18.0, 81],
        metavar=("FMIN", "FMAX", "NPTS"),
    )
    # eta/d_omega ~ 1 resolves the propagator on the grid; the legacy
    # 0.05 under-resolves it (see verify_discretization). Recover the
    # eta -> 0 limit with phonon/scripts/extrapolate_eta.py.
    p.add_argument("--eta-factor", type=float, default=1.0)

    # SCBA -----------------------------------------------------------------
    # d5a (auto-extended fmax, dynamical zero-mode projection, safeguarded
    # Anderson) shows clean monotone geometric convergence with rate
    # ~0.86 per iteration; reaching the numerical noise floor from a
    # cold start takes ~100--150 iters, so set --max-scba-iter generously
    # and tighten --scba-tol to chase the residual below 1e-3 if you want
    # to confirm a true fixed point.
    p.add_argument("--max-scba-iter", type=int, default=200)
    p.add_argument("--scba-tol", type=float, default=1e-6)
    p.add_argument("--conservation-tol", type=float, default=1e-1)
    p.add_argument("--mixing", type=float, default=0.3)
    p.add_argument(
        "--anderson-mixing", action=argparse.BooleanOptionalAction,
        default=True,
        help="Safeguarded Anderson (default) or --no-anderson-mixing for "
             "linear; on d5a linear at any damping factor in {0.02, 0.05, "
             "0.1, 0.3} stalls above the SCF tolerance.",
    )
    p.add_argument("--anderson-depth", type=int, default=8)
    p.add_argument(
        "--solver", default=None,
        choices=("linear", "anderson", "jfnk", "anderson+jfnk"),
        help="SCBA fixed-point accelerator. Default: derived from "
             "--anderson-mixing (anderson if set, else linear).",
    )
    p.add_argument(
        "--anderson-safeguard", action=argparse.BooleanOptionalAction,
        default=True,
        help="Safeguarded Anderson (default); --no-anderson-safeguard "
             "restores the legacy hard-restart scheme.",
    )
    p.add_argument(
        "--zero-mode-projection", action=argparse.BooleanOptionalAction,
        default=True,
        help="Project the rigid-translation component out of the "
             "self-energy each SCBA iteration (default on).",
    )
    p.add_argument(
        "--gate-on-conservation", action="store_true",
        help="Legacy stop test: require heat-flow conservation in "
             "addition to the SCF residual.",
    )
    p.add_argument(
        "--divergence-guard", action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort early on residual blow-up (default on).",
    )
    p.add_argument(
        "--enforce-asr", action=argparse.BooleanOptionalAction,
        default=True,
        help="ASR-project the FC3 onto the Gamma-translation null space "
             "(both legs) before building the device vertex (default on). "
             "Removes the ~0.8 leg-j/k ASR residual of open-wire FC3 that "
             "stiffens the small-eta SCBA fixed point. Cached separately "
             "from --no-enforce-asr runs.",
    )

    # Cutoff axes ----------------------------------------------------------
    p.add_argument(
        "--sigma-cutoffs", nargs="+", default=None,
        help="Values of sigma_cutoff to sweep. Use 'none' or 'inf' for the "
             "unrestricted case. Default: 0..n_slabs-1 plus None.",
    )
    p.add_argument(
        "--vertex-cutoffs", nargs="+", default=None,
        help="Values of vertex_cutoff to sweep. Default: 0, 1, ..., None.",
    )
    p.add_argument(
        "--g-cutoffs", nargs="+", default=None,
        help="Values of g_cutoff to sweep. Default: 0, 1, ..., None.",
    )
    p.add_argument(
        "--dc-handling", nargs="+", default=["interpolate"],
        choices=("zero", "interpolate", "keep"),
        help="dc_handling values to sweep.",
    )

    # Modes ----------------------------------------------------------------
    p.add_argument("--replot-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true", default=False)

    args = p.parse_args()

    # Build default axes from --n-slabs and supercell extent.
    half_window = 2  # N_super_z // 2 = 4 // 2 = 2 for d5a [1,1,4]
    args.default_sigma = list(range(args.n_slabs)) + [None]
    args.default_vertex = list(range(half_window + 1)) + [None]
    args.default_g = list(range(args.n_slabs)) + [None]
    return args


def main() -> None:
    args = parse_args()
    _apply_style()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] output dir : {out_dir}", flush=True)
    print(f"[setup] config     : {args.config}", flush=True)
    print(
        f"[setup] replot_only={args.replot_only} overwrite={args.overwrite}",
        flush=True,
    )

    invocation = {
        "argv": sys.argv,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    with open(out_dir / "invocation.json", "w") as f:
        json.dump(invocation, f, indent=2)

    if not args.replot_only:
        print("[load ] system bundle ...", flush=True)
        transport_axis = "xyz".index(args.transport_direction)
        bundle = load_system(
            args.config, validate=False, transport_axis=transport_axis,
        )
        fc3_hdf5 = Path(bundle.meta.get("fc3_path", "")).expanduser().resolve()
        if not fc3_hdf5.exists():
            raise FileNotFoundError(
                f"fc3.hdf5 not found at {fc3_hdf5}. Run the d5a reap first."
            )
        print(f"[load ] fc3.hdf5   : {fc3_hdf5}", flush=True)
    else:
        bundle = None
        fc3_hdf5 = None

    print("\n[sweep] cutoff ...", flush=True)
    results = sweep_cutoffs(
        bundle=bundle, fc3_hdf5=fc3_hdf5,
        out_dir=out_dir, args=args,
    )

    plot_all(results, out_dir)
    write_summary(results, out_dir)
    print("\n[done] cutoff sweep complete.", flush=True)


if __name__ == "__main__":
    main()
