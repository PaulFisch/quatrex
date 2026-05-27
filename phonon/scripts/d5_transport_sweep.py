#!/usr/bin/env python
"""d5a SiNW transport sweep over temperature differences and wire lengths.

Drives :func:`phonon.solver.dense.transmission_finite` on the d5a
(Si9H12, 1.0 nm) canonical [100] H-passivated SiNW for two sweeps:

  (a) **Temperature-difference sweep.** Fixed wire length (``n_slabs``),
      vary ``delta_T`` at fixed mean temperature, and optionally vary
      the mean temperature itself. Produces heat-current-vs-ΔT,
      conductance-vs-T_avg, and spectral-current overlays.

  (b) **Wire-length sweep.** Fixed temperatures, vary ``n_slabs``
      (number of repeating transport cells between the leads).
      Produces ballistic-and-anharmonic conductance vs length and
      spectral-current overlays showing the ballistic-to-diffusive
      crossover.

Each sweep point is cached to disk (``checkpoints/<sweep>/<key>.npz``)
so reruns skip completed work; ``--replot-only`` regenerates plots
from cache without recomputing anything. Designed for cluster use:
absolute paths, flush prints, single-process, no MPI.

Usage (cluster)
---------------
::

    /home/paul/miniconda3/envs/quatrex-dev/bin/python \\
        phonon/scripts/d5_transport_sweep.py \\
        --config phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml \\
        --out-dir phonon/scripts/out/d5_transport_sweep \\
        --delta-Ts 5 10 20 50 100 \\
        --t-mean 300 \\
        --lengths 1 2 4 8 \\
        --max-scba-iter 8

Add ``--ballistic-only`` to skip the SCBA loop entirely (a few
seconds per point) and validate the pipeline / plotting end-to-end.
Add ``--replot-only`` to regenerate plots from cached results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")  # headless: no display on the cluster, no Qt/Wayland warnings.

import matplotlib.pyplot as plt  # noqa: E402  (Agg must be set before pyplot)
import numpy as np  # noqa: E402

# --- Import bootstrap --------------------------------------------------------
# The phonon analysis package lives under ``phonon/`` and is not installed; add
# the repo root to ``sys.path`` so ``from phonon.solver ...`` works regardless
# of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# ``phonon_inputs`` lives at ``phonon/phonon_inputs`` and is imported by
# ``phonon.finite_analysis.loader``; expose it without the ``phonon.`` prefix.
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

CMAP_T = mpl.colormaps["viridis"]
CMAP_L = mpl.colormaps["plasma"]


def _apply_style() -> None:
    mpl.rcParams.update(PLOT_STYLE)


# =============================================================================
# Sweep point types
# =============================================================================


@dataclass(frozen=True)
class SweepKeyT:
    """Identifier for a (T_mean, delta_T, n_slabs) temperature-sweep point."""

    t_mean: float
    delta_T: float
    n_slabs: int
    ballistic_only: bool

    def filename(self) -> str:
        ballistic = "ball" if self.ballistic_only else "scba"
        return (
            f"T{self.t_mean:g}_dT{self.delta_T:g}_"
            f"L{self.n_slabs}_{ballistic}.npz"
        )


@dataclass(frozen=True)
class SweepKeyL:
    """Identifier for an (n_slabs, T_mean, delta_T) length-sweep point."""

    n_slabs: int
    t_mean: float
    delta_T: float
    ballistic_only: bool

    def filename(self) -> str:
        ballistic = "ball" if self.ballistic_only else "scba"
        return (
            f"L{self.n_slabs}_T{self.t_mean:g}_dT{self.delta_T:g}_"
            f"{ballistic}.npz"
        )


# =============================================================================
# Run a single point
# =============================================================================


# Fields kept from the transmission_finite return dict in the checkpoint.
# Heavy arrays (per-iteration Σ histories etc.) are dropped so checkpoints
# stay small enough to copy off the cluster.
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
)


def _run_one(
    *,
    bundle,
    fc3_hdf5: Path,
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
    ballistic_only: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Execute one ``transmission_finite`` call.

    When ``ballistic_only=True`` we still call the SCBA driver but cap
    it at zero iterations so the returned ballistic quantities are
    populated without paying the anharmonic cost. The output dict is
    pruned to :data:`_CHECKPOINT_FIELDS`.
    """
    effective_max_iter = 0 if ballistic_only else max_scba_iter

    res = transmission_finite(
        bundle.phonon,
        fc3_hdf5=str(fc3_hdf5),
        freq_range_thz=freq_range,
        transport_direction=transport_direction,
        eta_factor=eta_factor,
        temperature=temperature,
        delta_T=delta_T,
        max_scba_iter=effective_max_iter,
        scba_tol=scba_tol,
        conservation_tol=conservation_tol,
        mixing=mixing,
        anderson_mixing=anderson_mixing,
        anderson_depth=anderson_depth,
        n_slabs=n_slabs,
        verbose=verbose,
    )

    out = {k: res[k] for k in _CHECKPOINT_FIELDS if k in res}
    out["t_mean"] = temperature
    out["n_slabs"] = n_slabs
    out["ballistic_only"] = ballistic_only
    return out


def _load_or_run(
    checkpoint_path: Path,
    key: SweepKeyT | SweepKeyL,
    runner,
    *,
    overwrite: bool,
    replot_only: bool,
) -> dict[str, Any] | None:
    """Return the cached point, or run and cache it.

    Returns ``None`` if ``replot_only`` is set and there's no cache —
    in that case the plot loop just skips the point.
    """
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

    t_start = time.time()
    print(
        f"[run ] {checkpoint_path.name} : "
        f"{asdict(key)}",
        flush=True,
    )
    point = runner(key)
    elapsed = time.time() - t_start
    print(
        f"[done] {checkpoint_path.name}  "
        f"({elapsed:.1f} s, "
        f"G_ball={point['thermal_conductance_ballistic']:.3g} W/(m²K), "
        f"G_anh={point['thermal_conductance_anharmonic']:.3g} W/(m²K))",
        flush=True,
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    saveable = {
        k: np.asarray(v) for k, v in point.items() if not k.startswith("_")
    }
    np.savez_compressed(checkpoint_path, **saveable)
    return point


# =============================================================================
# Sweep drivers
# =============================================================================


def _temperature_keys(args) -> list[SweepKeyT]:
    """Enumerate every (T_mean, ΔT, n_slabs) key for the temperature sweep.

    Used both by the cache pre-scan (to decide whether to load the
    system bundle at all) and by :func:`sweep_temperature` (to drive
    the actual run). Keeping the enumeration in one place guarantees
    the scan and the run see exactly the same set of checkpoints.
    """
    return [
        SweepKeyT(
            t_mean=float(t_mean),
            delta_T=float(delta_T),
            n_slabs=int(args.length_for_t_sweep),
            ballistic_only=bool(args.ballistic_only),
        )
        for t_mean in args.t_means
        for delta_T in args.delta_Ts
    ]


def _length_keys(args) -> list[SweepKeyL]:
    """Enumerate every n_slabs key for the length sweep (see ``_temperature_keys``)."""
    return [
        SweepKeyL(
            n_slabs=int(n_slabs),
            t_mean=float(args.t_for_l_sweep),
            delta_T=float(args.dt_for_l_sweep),
            ballistic_only=bool(args.ballistic_only),
        )
        for n_slabs in args.lengths
    ]


def _scan_cache(out_dir: Path, args) -> dict[str, dict[str, list[str]]]:
    """Inspect which sweep checkpoints already exist on disk.

    Returns ``{sweep: {"cached": [names], "missing": [names]}}`` so the
    caller can decide whether to load the (expensive) system bundle at
    all. ``--overwrite`` is *not* treated as "missing" here -- the goal
    is just to enumerate what is on disk; the run loop in
    ``_load_or_run`` still honours ``--overwrite`` on a per-point basis.
    """
    out: dict[str, dict[str, list[str]]] = {
        "temperature": {"cached": [], "missing": []},
        "length": {"cached": [], "missing": []},
    }
    if not args.skip_temperature:
        ck = out_dir / "checkpoints" / "temperature"
        for k in _temperature_keys(args):
            path = ck / k.filename()
            out["temperature"]["cached" if path.exists() else "missing"].append(
                path.name
            )
    if not args.skip_length:
        ck = out_dir / "checkpoints" / "length"
        for k in _length_keys(args):
            path = ck / k.filename()
            out["length"]["cached" if path.exists() else "missing"].append(
                path.name
            )
    return out


def sweep_temperature(
    *,
    bundle,
    fc3_hdf5: Path,
    out_dir: Path,
    args,
) -> list[dict[str, Any]]:
    """Vary delta_T (and optionally T_mean) at fixed wire length."""
    ck_dir = out_dir / "checkpoints" / "temperature"
    ck_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for key in _temperature_keys(args):
        def runner(k: SweepKeyT) -> dict[str, Any]:
            return _run_one(
                bundle=bundle, fc3_hdf5=fc3_hdf5,
                temperature=k.t_mean, delta_T=k.delta_T,
                n_slabs=k.n_slabs,
                transport_direction=args.transport_direction,
                freq_range=tuple(args.freq_range),
                eta_factor=args.eta_factor,
                max_scba_iter=args.max_scba_iter,
                scba_tol=args.scba_tol,
                conservation_tol=args.conservation_tol,
                mixing=args.mixing,
                anderson_mixing=args.anderson_mixing,
                anderson_depth=args.anderson_depth,
                ballistic_only=k.ballistic_only,
                verbose=args.verbose,
            )

        point = _load_or_run(
            ck_dir / key.filename(), key, runner,
            overwrite=args.overwrite,
            replot_only=args.replot_only,
        )
        if point is None:
            continue
        point["_key"] = key
        results.append(point)
    return results


def sweep_length(
    *,
    bundle,
    fc3_hdf5: Path,
    out_dir: Path,
    args,
) -> list[dict[str, Any]]:
    """Vary n_slabs at fixed (T_mean, delta_T)."""
    ck_dir = out_dir / "checkpoints" / "length"
    ck_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for key in _length_keys(args):
        def runner(k: SweepKeyL) -> dict[str, Any]:
            return _run_one(
                bundle=bundle, fc3_hdf5=fc3_hdf5,
                temperature=k.t_mean, delta_T=k.delta_T,
                n_slabs=k.n_slabs,
                transport_direction=args.transport_direction,
                freq_range=tuple(args.freq_range),
                eta_factor=args.eta_factor,
                max_scba_iter=args.max_scba_iter,
                scba_tol=args.scba_tol,
                conservation_tol=args.conservation_tol,
                mixing=args.mixing,
                anderson_mixing=args.anderson_mixing,
                anderson_depth=args.anderson_depth,
                ballistic_only=k.ballistic_only,
                verbose=args.verbose,
            )

        point = _load_or_run(
            ck_dir / key.filename(), key, runner,
            overwrite=args.overwrite,
            replot_only=args.replot_only,
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


def _band_label(t_mean: float, delta_T: float) -> str:
    return rf"$\bar T={t_mean:g}\,$K, $\Delta T={delta_T:g}\,$K"


def plot_convergence(
    results: list[dict[str, Any]],
    out_dir: Path,
    *,
    sweep_label: str,
    series_label,
    cmap,
) -> None:
    """Plot per-iteration SCBA rel-change for every result that has it.

    ``series_label(result) -> str`` formats the legend entry for one
    result. The plot is skipped if no result carries a non-trivial
    convergence_history (i.e. SCBA was disabled or every point
    converged immediately).
    """
    valid = [
        r for r in results
        if "convergence_history" in r
        and np.asarray(r["convergence_history"]).size > 0
    ]
    if not valid:
        return

    plot_dir = out_dir / "plots" / sweep_label
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for i, r in enumerate(valid):
        hist = np.asarray(r["convergence_history"], dtype=float)
        # Iteration count = len(history) + 1 (the first iter has no
        # rel_change since there's no "previous" Σ yet). Plot rel_change
        # at iters 2, 3, ... to match how scba_loop logs it.
        iters = np.arange(2, 2 + hist.size)
        color = cmap(i / max(1, len(valid) - 1))
        ax.semilogy(iters, hist, "o-", color=color, label=series_label(r))

    tol = None
    # Best-effort: pull the tolerance from the invocation JSON if available.
    inv_path = out_dir / "invocation.json"
    if inv_path.exists():
        try:
            inv = json.loads(inv_path.read_text())
            tol = float(inv["args"].get("scba_tol"))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            tol = None
    if tol is not None:
        ax.axhline(tol, color="k", linestyle=":", linewidth=1.0,
                   label=rf"$\mathrm{{tol}}={tol:g}$")

    ax.set_xlabel("SCBA iteration")
    ax.set_ylabel(r"$|\Delta J / J|$ per iter")
    ax.set_title(f"d5a SiNW: SCBA convergence ({sweep_label} sweep)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save_fig(fig, plot_dir / "scba_convergence")


def plot_temperature_sweep(
    results: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Plot results of the temperature sweep.

    Produces:
      ``temperature/heat_current_vs_delta_T.{pdf,png}`` and
      ``temperature/conductance_vs_t_mean.{pdf,png}`` and
      ``temperature/spectral_current_overlay.{pdf,png}``.
    """
    if not results:
        print("[warn] No temperature-sweep results to plot.", flush=True)
        return

    plot_dir = out_dir / "plots" / "temperature"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- (a1) heat current vs delta_T at each T_mean --------------------------
    by_t_mean: dict[float, list[dict]] = {}
    for r in results:
        by_t_mean.setdefault(float(r["t_mean"]), []).append(r)
    t_means = sorted(by_t_mean)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for i, t_mean in enumerate(t_means):
        pts = sorted(by_t_mean[t_mean], key=lambda r: float(r["delta_T"]))
        dts = np.array([float(r["delta_T"]) for r in pts])
        j_ball = np.array(
            [float(r["heat_current_ballistic"]) for r in pts]
        )
        j_anh = np.array([float(r["heat_current"]) for r in pts])

        color = CMAP_T(i / max(1, len(t_means) - 1))
        ax.plot(dts, j_ball * 1e12, "o-", color=color,
                label=rf"ballistic, $\bar T={t_mean:g}\,$K")
        if np.any(j_anh != 0):
            ax.plot(dts, j_anh * 1e12, "s--", color=color, alpha=0.7,
                    label=rf"SCBA, $\bar T={t_mean:g}\,$K")

    ax.set_xlabel(r"Temperature difference $\Delta T$ (K)")
    ax.set_ylabel(r"Heat current $J$ (pW)")
    ax.set_title("d5a SiNW: heat current vs $\\Delta T$")
    ax.legend(frameon=False, ncol=1, loc="best")
    _save_fig(fig, plot_dir / "heat_current_vs_delta_T")

    # --- (a2) conductance vs T_mean -------------------------------------------
    if len(t_means) >= 2:
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        # Use the smallest delta_T per T_mean to approximate the
        # linear-response conductance.
        for label, key, marker in (
            ("ballistic", "thermal_conductance_ballistic", "o-"),
            ("SCBA", "thermal_conductance_anharmonic", "s--"),
        ):
            ts = []
            gs = []
            for t_mean in t_means:
                pts = sorted(by_t_mean[t_mean],
                             key=lambda r: float(r["delta_T"]))
                ts.append(t_mean)
                gs.append(float(pts[0][key]))
            ts_arr = np.asarray(ts)
            gs_arr = np.asarray(gs)
            if np.allclose(gs_arr, 0):
                continue
            ax.plot(ts_arr, gs_arr, marker, label=label)
        ax.set_xlabel(r"Mean temperature $\bar T$ (K)")
        ax.set_ylabel(r"$G$ (W m$^{-2}$ K$^{-1}$)")
        ax.set_title(
            "d5a SiNW: thermal conductance vs mean temperature "
            f"(at $\\Delta T={min(float(r['delta_T']) for r in results):g}\\,$K)"
        )
        ax.legend(frameon=False)
        _save_fig(fig, plot_dir / "conductance_vs_t_mean")

    # --- (a3) spectral current overlay ----------------------------------------
    # Pick a representative T_mean (the median) and overlay every delta_T.
    rep_t = t_means[len(t_means) // 2]
    rep_pts = sorted(by_t_mean[rep_t], key=lambda r: float(r["delta_T"]))
    if rep_pts:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for i, r in enumerate(rep_pts):
            color = CMAP_T(i / max(1, len(rep_pts) - 1))
            freqs = np.asarray(r["freqs_thz"])
            j_ball = np.asarray(r["spectral_heat_current_ballistic"]) * 1e12
            ax.plot(freqs, j_ball, color=color,
                    label=rf"$\Delta T={float(r['delta_T']):g}\,$K (ball.)")
            j_anh = np.asarray(r["spectral_heat_current"]) * 1e12
            if np.any(j_anh != 0):
                ax.plot(freqs, j_anh, "--", color=color, alpha=0.6)
        ax.set_xlabel(r"Frequency $\omega/2\pi$ (THz)")
        ax.set_ylabel(r"Spectral heat current (pW/THz)")
        ax.set_title(
            f"d5a SiNW: spectral $J(\\omega)$ at $\\bar T={rep_t:g}\\,$K"
        )
        ax.legend(frameon=False, ncol=2, fontsize=8)
        _save_fig(fig, plot_dir / "spectral_current_overlay")


def plot_length_sweep(
    results: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Plot results of the length sweep."""
    if not results:
        print("[warn] No length-sweep results to plot.", flush=True)
        return

    plot_dir = out_dir / "plots" / "length"
    plot_dir.mkdir(parents=True, exist_ok=True)

    results_sorted = sorted(results, key=lambda r: int(r["n_slabs"]))
    lengths = np.array([int(r["n_slabs"]) for r in results_sorted])
    g_ball = np.array(
        [float(r["thermal_conductance_ballistic"]) for r in results_sorted]
    )
    g_anh = np.array(
        [float(r["thermal_conductance_anharmonic"]) for r in results_sorted]
    )
    j_ball = np.array(
        [float(r["heat_current_ballistic"]) for r in results_sorted]
    ) * 1e12  # pW
    j_anh = np.array(
        [float(r["heat_current"]) for r in results_sorted]
    ) * 1e12

    t_mean = float(results_sorted[0]["t_mean"])
    delta_T = float(results_sorted[0]["delta_T"])

    # --- (b1) ballistic + SCBA conductance vs length --------------------------
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(lengths, g_ball, "o-", label="ballistic", color="C0")
    if np.any(g_anh != 0):
        ax.plot(lengths, g_anh, "s--", label="SCBA (anharmonic)", color="C3")
    ax.set_xlabel(r"Wire length $L$ (transport-cell units)")
    ax.set_ylabel(r"$G$ (W m$^{-2}$ K$^{-1}$)")
    ax.set_title(
        "d5a SiNW: thermal conductance vs length "
        f"({_band_label(t_mean, delta_T)})"
    )
    ax.legend(frameon=False)
    _save_fig(fig, plot_dir / "conductance_vs_length")

    # --- (b2) heat current vs length (log-log) --------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.loglog(lengths, j_ball, "o-", label="ballistic", color="C0")
    if np.any(j_anh != 0):
        ax.loglog(lengths, j_anh, "s--", label="SCBA", color="C3")
    ax.set_xlabel(r"Wire length $L$ (transport-cell units)")
    ax.set_ylabel(r"Heat current $J$ (pW)")
    ax.set_title(
        "d5a SiNW: heat current vs length (log-log)"
    )
    ax.legend(frameon=False)
    _save_fig(fig, plot_dir / "heat_current_vs_length_loglog")

    # --- (b3) spectral current overlay ----------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, r in enumerate(results_sorted):
        color = CMAP_L(i / max(1, len(results_sorted) - 1))
        freqs = np.asarray(r["freqs_thz"])
        j_b = np.asarray(r["spectral_heat_current_ballistic"]) * 1e12
        ax.plot(freqs, j_b, color=color, label=rf"$L={int(r['n_slabs'])}$ (ball.)")
        j_a = np.asarray(r["spectral_heat_current"]) * 1e12
        if np.any(j_a != 0):
            ax.plot(freqs, j_a, "--", color=color, alpha=0.6)
    ax.set_xlabel(r"Frequency $\omega/2\pi$ (THz)")
    ax.set_ylabel(r"Spectral heat current (pW/THz)")
    ax.set_title(
        f"d5a SiNW: spectral $J(\\omega)$ vs length "
        f"({_band_label(t_mean, delta_T)})"
    )
    ax.legend(frameon=False, ncol=2, fontsize=8)
    _save_fig(fig, plot_dir / "spectral_current_overlay")


# =============================================================================
# Summary CSV / JSON
# =============================================================================


def write_summary(
    results_T: list[dict[str, Any]],
    results_L: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Dump a compact human-readable summary alongside the npz cache."""

    def row(r: dict[str, Any], sweep: str) -> dict[str, Any]:
        return {
            "sweep": sweep,
            "t_mean": float(r["t_mean"]),
            "delta_T": float(r["delta_T"]),
            "n_slabs": int(r["n_slabs"]),
            "G_ball_W_per_m2_K": float(r["thermal_conductance_ballistic"]),
            "G_anh_W_per_m2_K": float(r["thermal_conductance_anharmonic"]),
            "J_ball_pW": float(r["heat_current_ballistic"]) * 1e12,
            "J_anh_pW": float(r["heat_current"]) * 1e12,
            "heat_flow_conservation": float(r.get("heat_flow_conservation", 0.0)),
            "n_scba_iterations": int(r.get("n_scba_iterations", 0)),
            "ballistic_only": bool(r.get("ballistic_only", False)),
        }

    rows = [row(r, "temperature") for r in results_T]
    rows += [row(r, "length") for r in results_L]

    csv_path = out_dir / "summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        if not rows:
            f.write("# no results\n")
            return
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(repr(r[k]) for k in keys) + "\n")

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
        help="Path to d5a YAML config (default: %(default)s).",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=repo / "phonon/scripts/out/d5_transport_sweep",
        help="Output directory for checkpoints + plots + summary.",
    )
    p.add_argument(
        "--transport-direction", default="z", choices=("x", "y", "z"),
        help="Cartesian axis along which the device extends. d5a is "
             "built along z; only override if you load a different wire.",
    )

    # Temperature sweep ----------------------------------------------------
    p.add_argument(
        "--delta-Ts", type=float, nargs="+",
        default=[5.0, 10.0, 20.0, 50.0, 100.0],
        help="Temperature differences ΔT (K) to sweep.",
    )
    p.add_argument(
        "--t-means", type=float, nargs="+",
        default=[300.0],
        help="Mean temperatures T_avg (K) to sweep.",
    )
    p.add_argument(
        "--length-for-t-sweep", type=int, default=1,
        help="n_slabs to use for the temperature sweep.",
    )

    # Length sweep ---------------------------------------------------------
    p.add_argument(
        "--lengths", type=int, nargs="+", default=[1, 2, 4, 8],
        help="Wire lengths (in transport-cell repeats) to sweep.",
    )
    p.add_argument(
        "--t-for-l-sweep", type=float, default=300.0,
        help="Mean temperature for the length sweep.",
    )
    p.add_argument(
        "--dt-for-l-sweep", type=float, default=10.0,
        help="ΔT for the length sweep.",
    )

    # Solver knobs ---------------------------------------------------------
    p.add_argument(
        "--freq-range", type=float, nargs=3, default=[0.01, 18.0, 121],
        metavar=("FMIN", "FMAX", "NPTS"),
        help="ω grid: min (THz), max (THz), N points.",
    )
    p.add_argument("--eta-factor", type=float, default=0.05)
    p.add_argument("--max-scba-iter", type=int, default=15)
    p.add_argument("--scba-tol", type=float, default=1e-3)
    p.add_argument(
        "--conservation-tol", type=float, default=1e-3,
        help="Heat-flow conservation tolerance (|J_L - J_R|/J). SCBA "
             "continues even after dΣ/Σ < scba_tol until this is hit.",
    )
    p.add_argument("--mixing", type=float, default=0.5)
    p.add_argument(
        "--anderson-mixing", action="store_true",
        help="Use Anderson mixing instead of linear mixing — usually "
             "converges 2-3× faster on the cubic-anharmonic SCBA.",
    )
    p.add_argument("--anderson-depth", type=int, default=5)

    # Modes ----------------------------------------------------------------
    p.add_argument(
        "--ballistic-only", action="store_true",
        help="Skip the SCBA loop; only ballistic quantities are computed.",
    )
    p.add_argument(
        "--replot-only", action="store_true",
        help="Skip all computation; regenerate plots from existing checkpoints.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Recompute even when a checkpoint exists.",
    )
    p.add_argument(
        "--skip-temperature", action="store_true",
        help="Skip the temperature sweep.",
    )
    p.add_argument(
        "--skip-length", action="store_true",
        help="Skip the length sweep.",
    )
    p.add_argument(
        "--verbose", action="store_true", default=False,
        help="Verbose per-call solver output.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _apply_style()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] output dir : {out_dir}", flush=True)
    print(f"[setup] config     : {args.config}", flush=True)
    print(f"[setup] ballistic_only={args.ballistic_only} "
          f"replot_only={args.replot_only} overwrite={args.overwrite}",
          flush=True)

    # Persist the exact CLI invocation so reruns are reproducible.
    invocation = {
        "argv": sys.argv,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    with open(out_dir / "invocation.json", "w") as f:
        json.dump(invocation, f, indent=2)

    # Pre-scan the checkpoint directory before doing any work: lets us
    # report what is already on disk vs. what needs computing, and --
    # if every requested sweep point is already cached -- skip loading
    # the (multi-second) system bundle and fc3.hdf5 entirely. The
    # per-point ``_load_or_run`` cache check still runs as before, so
    # the only thing this short-circuit saves on a fully cached run is
    # the bundle/fc3 read.
    cache_info = _scan_cache(out_dir, args)
    n_cached = sum(len(v["cached"]) for v in cache_info.values())
    n_missing = sum(len(v["missing"]) for v in cache_info.values())
    print(
        f"[cache] {n_cached} cached, {n_missing} to compute "
        f"(overwrite={args.overwrite})",
        flush=True,
    )
    for sweep, info in cache_info.items():
        if info["cached"] or info["missing"]:
            print(
                f"  {sweep}: cached={len(info['cached'])}, "
                f"missing={len(info['missing'])}",
                flush=True,
            )

    need_compute = (
        not args.replot_only and (n_missing > 0 or args.overwrite)
    )
    if need_compute:
        print("[load ] system bundle ...", flush=True)
        transport_axis = "xyz".index(args.transport_direction)
        bundle = load_system(
            args.config, validate=False, transport_axis=transport_axis,
        )
        # Pick the bundled fc3.hdf5 (the loader already validated it).
        fc3_hdf5 = Path(bundle.meta.get("fc3_path", "")).expanduser().resolve()
        if not fc3_hdf5.exists():
            raise FileNotFoundError(
                f"fc3.hdf5 not found at {fc3_hdf5}. "
                "Make sure the d5a reap has run; see plan section "
                "'Step 1 — reap hiphive FC3 (d5a)'."
            )
        print(f"[load ] fc3.hdf5   : {fc3_hdf5}", flush=True)
    else:
        bundle = None
        fc3_hdf5 = None
        if not args.replot_only:
            print(
                "[load ] skipping system bundle (all checkpoints cached; "
                "pass --overwrite to recompute)",
                flush=True,
            )

    # --- (a) temperature sweep -------------------------------------------
    results_T: list[dict[str, Any]] = []
    if not args.skip_temperature:
        print("\n[sweep] temperature ...", flush=True)
        results_T = sweep_temperature(
            bundle=bundle, fc3_hdf5=fc3_hdf5,
            out_dir=out_dir, args=args,
        )
        plot_temperature_sweep(results_T, out_dir)
        plot_convergence(
            results_T, out_dir, sweep_label="temperature",
            series_label=lambda r: (
                rf"$\bar T={float(r['t_mean']):g}\,$K, "
                rf"$\Delta T={float(r['delta_T']):g}\,$K"
            ),
            cmap=CMAP_T,
        )

    # --- (b) length sweep -------------------------------------------------
    results_L: list[dict[str, Any]] = []
    if not args.skip_length:
        print("\n[sweep] length ...", flush=True)
        results_L = sweep_length(
            bundle=bundle, fc3_hdf5=fc3_hdf5,
            out_dir=out_dir, args=args,
        )
        plot_length_sweep(results_L, out_dir)
        plot_convergence(
            results_L, out_dir, sweep_label="length",
            series_label=lambda r: rf"$L={int(r['n_slabs'])}$",
            cmap=CMAP_L,
        )

    write_summary(results_T, results_L, out_dir)
    print("\n[done] all sweeps complete.", flush=True)


if __name__ == "__main__":
    main()
