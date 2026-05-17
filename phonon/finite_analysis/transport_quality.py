"""FC3-approximation quality vs. transport — the headline plot suite.

For each (compression method, rank) cell:

  1. Fit the FC3 with the chosen method via ``phonon_inputs.fc3_compression``.
  2. Reconstruct the compressed FC3 in the solver's ``M_stacked`` shape
     ``(n_dof, dim_sc, dim_sc)``.
  3. Run :func:`phonon.solver.transmission_q` with the reconstruction as
     ``M_stacked_override`` (q-resolved path → safe for SiNW because the
     bubble runs on the primitive-cell DOFs after Γ-projection, not on
     ``n_dof_super``).
  4. Cache the result in ``<out_dir>/cache/{method}_{rank}.npz``.

Each rerun re-reads cached cells and only computes the missing ones —
analogous to the legacy ``plot_quality.py`` checkpoint behaviour. To
force a recompute, pass ``force_recompute=True`` or delete the cache
directory.

Plots produced (all in ``<out_dir>/``):

  * ``transport_quality_frob_vs_params.{png,pdf}`` — Frobenius rel-error
    vs. parameter count, one line per method.
  * ``transport_quality_ganh_vs_params.{png,pdf}`` — anharmonic thermal
    conductance vs. parameter count, with the dense FC3 reference as a
    horizontal band.
  * ``transport_quality_frob_vs_transport_err.{png,pdf}`` — the headline
    scatter: relative G_anh error vs. Frobenius rel-error. Tests whether
    a small Frobenius residual implies a small transport residual.
  * ``transport_quality_spectral_current.{png,pdf}`` — spectral heat
    current vs. ω for the highest rank of each method, dense overlay.
  * ``transport_quality_conservation.{png,pdf}`` — heat-flow conservation
    residual |J_L − J_R| / (|J_L| + |J_R|) per (method, rank).

Tabular outputs:

  * ``transport_quality.json`` — summary statistics.
  * ``transport_quality.csv``  — one row per (method, rank) cell.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phonon_inputs.fc3_compression import (
    FITTERS,
    reconstruct_cp,
    reconstruct_hosvd,
    reconstruct_indscal,
    reconstruct_msvd,
    reconstruct_pcp,
    reconstruct_waring,
    n_params_cp,
    n_params_hosvd,
    n_params_indscal,
    n_params_msvd,
    n_params_pcp,
    n_params_waring,
)

from .loader import SystemBundle


# ---------------------------------------------------------------------------
# Style (Tol-bright palette; matches the historical plot_quality.py)
# ---------------------------------------------------------------------------

METHOD_ORDER = ["mSVD", "HOSVD", "CP", "INDSCAL", "Waring"]
PALETTE: dict[str, str] = {
    "mSVD":     "#4477AA",  # blue
    "HOSVD":    "#228833",  # green
    "CP":       "#CCBB44",  # yellow
    "INDSCAL":  "#66CCEE",  # cyan
    "Waring":   "#AA3377",  # purple
    "PCP":      "#000000",  # black (rarely used; opt-in only)
    "dense":    "#BB5566",  # muted red
}
MARKERS: dict[str, str] = {
    "mSVD":     "o",
    "HOSVD":    "v",
    "CP":       "s",
    "INDSCAL":  "^",
    "Waring":   "D",
    "PCP":      "X",
    "dense":    "*",
}


# ---------------------------------------------------------------------------
# Reconstruction dispatch
# ---------------------------------------------------------------------------


_RECONSTRUCTORS = {
    "mSVD":    reconstruct_msvd,
    "HOSVD":   reconstruct_hosvd,
    "CP":      reconstruct_cp,
    "INDSCAL": reconstruct_indscal,
    "Waring":  reconstruct_waring,
}


def _reconstruct(method: str, res, target, phonon=None) -> np.ndarray:
    if method == "PCP":
        return reconstruct_pcp(res, target, phonon)
    return _RECONSTRUCTORS[method](res, target)


def _count_params(method: str, rank, n_dof: int, dim_sc: int) -> int:
    if method == "mSVD":
        return n_params_msvd(int(rank), n_dof, dim_sc)
    if method == "HOSVD":
        R1, R2 = rank if isinstance(rank, (tuple, list)) else (int(rank), int(rank))
        return n_params_hosvd(int(R1), int(R2), n_dof, dim_sc)
    if method == "CP":
        return n_params_cp(int(rank), n_dof, dim_sc)
    if method == "INDSCAL":
        return n_params_indscal(int(rank), n_dof, dim_sc)
    if method == "Waring":
        return n_params_waring(int(rank), dim_sc)
    if method == "PCP":
        return n_params_pcp(int(rank), dim_sc)
    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Cache + cell computation
# ---------------------------------------------------------------------------


def _rank_tag(rank) -> str:
    if isinstance(rank, (tuple, list)):
        return "_".join(str(int(r)) for r in rank)
    return str(int(rank))


@dataclass
class QualityCell:
    """One ``(method, rank)`` checkpoint."""

    method: str
    rank: Any
    n_params: int
    frob_rel_err: float
    fit_time_s: float
    transport_time_s: float
    G_anh: float
    G_ball: float
    J_anh_total: float
    J_ball_total: float
    conservation_err: float
    n_scba_iter: int
    freqs_thz: np.ndarray
    transmission_ballistic: np.ndarray
    spectral_heat_current: np.ndarray
    error: str | None = None

    def to_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            method=np.array(self.method),
            rank=np.array(self.rank),
            n_params=np.array(self.n_params),
            frob_rel_err=np.array(self.frob_rel_err),
            fit_time_s=np.array(self.fit_time_s),
            transport_time_s=np.array(self.transport_time_s),
            G_anh=np.array(self.G_anh),
            G_ball=np.array(self.G_ball),
            J_anh_total=np.array(self.J_anh_total),
            J_ball_total=np.array(self.J_ball_total),
            conservation_err=np.array(self.conservation_err),
            n_scba_iter=np.array(self.n_scba_iter),
            freqs_thz=self.freqs_thz,
            transmission_ballistic=self.transmission_ballistic,
            spectral_heat_current=self.spectral_heat_current,
            error=np.array(self.error or ""),
        )

    @classmethod
    def from_npz(cls, path: Path) -> "QualityCell":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                method=str(data["method"]),
                rank=data["rank"].tolist(),
                n_params=int(data["n_params"]),
                frob_rel_err=float(data["frob_rel_err"]),
                fit_time_s=float(data["fit_time_s"]),
                transport_time_s=float(data["transport_time_s"]),
                G_anh=float(data["G_anh"]),
                G_ball=float(data["G_ball"]),
                J_anh_total=float(data["J_anh_total"]),
                J_ball_total=float(data["J_ball_total"]),
                conservation_err=float(data["conservation_err"]),
                n_scba_iter=int(data["n_scba_iter"]),
                freqs_thz=np.asarray(data["freqs_thz"]),
                transmission_ballistic=np.asarray(data["transmission_ballistic"]),
                spectral_heat_current=np.asarray(data["spectral_heat_current"]),
                error=(str(data["error"]) or None) if "error" in data.files else None,
            )


def _solver_kwargs(
    *,
    q_mesh_transverse,
    freq_range_thz,
    transport_direction,
    temperature,
    delta_T,
    max_scba_iter,
    scba_tol,
    mixing,
    anderson_mixing,
    eta_factor,
) -> dict:
    return dict(
        q_mesh_transverse=q_mesh_transverse,
        freq_range_thz=freq_range_thz,
        transport_direction=transport_direction,
        eta_factor=eta_factor,
        temperature=temperature,
        delta_T=delta_T,
        max_scba_iter=max_scba_iter,
        scba_tol=scba_tol,
        mixing=mixing,
        anderson_mixing=anderson_mixing,
        n_slabs=1,
        retarded="half",
        verbose=False,
    )


def _run_transport(
    bundle: SystemBundle, M_stacked: np.ndarray, solver_kw: dict,
) -> dict:
    """Run ``solver.transmission_q`` with the supplied M_stacked. Imports
    locally so the analysis module doesn't pay the SCBA import cost
    unless used.
    """
    from solver import transmission_q

    return transmission_q(
        bundle.phonon, M_stacked_override=M_stacked, **solver_kw,
    )


def _compute_cell(
    bundle: SystemBundle,
    method: str,
    rank,
    target_norm: float,
    solver_kw: dict,
    extra_kwargs: dict,
    verbose: bool,
) -> QualityCell:
    """Fit + reconstruct + transport for one (method, rank) cell."""
    target = bundle.fc3_target
    n_dof, dim_sc = target.n_dof, target.dim_sc

    t0 = time.time()
    fitter = FITTERS[method]
    kwargs = dict(extra_kwargs)
    if method == "PCP":
        kwargs.setdefault("phonon", bundle.phonon)
        kwargs.setdefault("fc3_raw", bundle.fc3_raw)
    try:
        if method == "HOSVD":
            R1, R2 = rank if isinstance(rank, (tuple, list)) else (rank, rank)
            res = fitter(target, R1=int(R1), R2=int(R2), **kwargs)
        else:
            res = fitter(target, rank=int(rank), **kwargs)
    except Exception as exc:  # noqa: BLE001
        return QualityCell(
            method=method, rank=rank, n_params=0,
            frob_rel_err=float("nan"), fit_time_s=time.time() - t0,
            transport_time_s=0.0,
            G_anh=float("nan"), G_ball=float("nan"),
            J_anh_total=float("nan"), J_ball_total=float("nan"),
            conservation_err=float("nan"), n_scba_iter=0,
            freqs_thz=np.empty(0), transmission_ballistic=np.empty(0),
            spectral_heat_current=np.empty(0),
            error=f"{type(exc).__name__}: {exc}",
        )
    fit_time = time.time() - t0
    frob_rel = float(res.rel_err)
    n_params = _count_params(method, rank, n_dof, dim_sc)

    if verbose:
        print(f"    fit ok: rel_err={frob_rel:.3e}, n_params={n_params}, "
              f"fit_time={fit_time:.1f} s; running transport ...", flush=True)

    M_stacked = _reconstruct(method, res, target, bundle.phonon)
    if M_stacked.shape != (n_dof, dim_sc, dim_sc):
        # Some reconstructors return the lifted (dim_sc^3) form. Slice down.
        if M_stacked.shape == (dim_sc, dim_sc, dim_sc):
            from phonon_inputs.fc3_compression import _slice_to_ndof
            M_stacked = _slice_to_ndof(M_stacked, target.p2s_map)
        else:
            raise RuntimeError(
                f"{method}: reconstruction shape {M_stacked.shape} "
                f"matches neither (n_dof, dim_sc, dim_sc) nor lifted."
            )

    t1 = time.time()
    try:
        out = _run_transport(bundle, M_stacked, solver_kw)
    except Exception as exc:  # noqa: BLE001
        return QualityCell(
            method=method, rank=rank, n_params=n_params,
            frob_rel_err=frob_rel, fit_time_s=fit_time,
            transport_time_s=time.time() - t1,
            G_anh=float("nan"), G_ball=float("nan"),
            J_anh_total=float("nan"), J_ball_total=float("nan"),
            conservation_err=float("nan"), n_scba_iter=0,
            freqs_thz=np.empty(0), transmission_ballistic=np.empty(0),
            spectral_heat_current=np.empty(0),
            error=f"transport: {type(exc).__name__}: {exc}",
        )
    transport_time = time.time() - t1

    return QualityCell(
        method=method, rank=rank, n_params=n_params,
        frob_rel_err=frob_rel, fit_time_s=fit_time,
        transport_time_s=transport_time,
        G_anh=float(out["thermal_conductance_anharmonic"]),
        G_ball=float(out["thermal_conductance_ballistic"]),
        J_anh_total=float(out["heat_current"]),
        J_ball_total=float(out["heat_current_ballistic"]),
        conservation_err=float(out["heat_flow_conservation"]),
        n_scba_iter=int(out["n_scba_iterations"]),
        freqs_thz=np.asarray(out["freqs_thz"]),
        transmission_ballistic=np.asarray(out["transmission_ballistic"]),
        spectral_heat_current=np.asarray(out["spectral_heat_current"]),
    )


def _compute_dense_reference(
    bundle: SystemBundle, solver_kw: dict, verbose: bool,
) -> QualityCell:
    """Dense FC3 transport (no compression)."""
    t1 = time.time()
    out = _run_transport(bundle, bundle.fc3_target.T, solver_kw)
    return QualityCell(
        method="dense", rank=0,
        n_params=int(bundle.fc3_target.T.size),
        frob_rel_err=0.0,
        fit_time_s=0.0,
        transport_time_s=time.time() - t1,
        G_anh=float(out["thermal_conductance_anharmonic"]),
        G_ball=float(out["thermal_conductance_ballistic"]),
        J_anh_total=float(out["heat_current"]),
        J_ball_total=float(out["heat_current_ballistic"]),
        conservation_err=float(out["heat_flow_conservation"]),
        n_scba_iter=int(out["n_scba_iterations"]),
        freqs_thz=np.asarray(out["freqs_thz"]),
        transmission_ballistic=np.asarray(out["transmission_ballistic"]),
        spectral_heat_current=np.asarray(out["spectral_heat_current"]),
    )


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def run_transport_quality(
    bundle: SystemBundle,
    out_dir: Path,
    *,
    scalar_ranks: Sequence[int] = (2, 4, 8, 16),
    methods: Iterable[str] | None = None,
    q_mesh_transverse: tuple[int, int] = (1, 1),
    freq_range_thz: tuple[float, float, int] = (0.01, 22.0, 121),
    transport_direction: str | None = None,
    temperature: float = 300.0,
    delta_T: float = 10.0,
    eta_factor: float = 0.1,
    max_scba_iter: int = 8,
    scba_tol: float = 1e-3,
    mixing: float = 0.5,
    anderson_mixing: bool = True,
    extra_kwargs: dict | None = None,
    include_pcp: bool = False,
    force_recompute: bool = False,
    verbose: bool = True,
) -> dict:
    """Sweep (method, rank) → fit, reconstruct, transport. Cached per cell.

    Returns a JSON-serialisable summary dict. Plots written to ``out_dir``.
    """
    out_dir = Path(out_dir)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if transport_direction is None:
        transport_direction = "xyz"[bundle.transport_axis]

    methods_list = list(methods) if methods is not None else list(METHOD_ORDER)
    if not include_pcp and "PCP" in methods_list:
        methods_list.remove("PCP")

    extra = dict(extra_kwargs or {})

    solver_kw = _solver_kwargs(
        q_mesh_transverse=q_mesh_transverse,
        freq_range_thz=freq_range_thz,
        transport_direction=transport_direction,
        temperature=temperature, delta_T=delta_T,
        max_scba_iter=max_scba_iter, scba_tol=scba_tol,
        mixing=mixing, anderson_mixing=anderson_mixing,
        eta_factor=eta_factor,
    )

    # --- dense reference --------------------------------------------------
    dense_path = cache_dir / "dense.npz"
    if dense_path.exists() and not force_recompute:
        dense = QualityCell.from_npz(dense_path)
        if verbose:
            print(f"[transport_quality] dense reference: cache hit "
                  f"(G_anh={dense.G_anh:.3e} W/(m²·K))")
    else:
        if verbose:
            print("[transport_quality] computing dense FC3 reference transport ...")
        dense = _compute_dense_reference(bundle, solver_kw, verbose)
        dense.to_npz(dense_path)

    # --- per-(method, rank) sweep ----------------------------------------
    cells: list[QualityCell] = [dense]
    for method in methods_list:
        if method == "HOSVD":
            ranks_for_method = [(int(r), int(r)) for r in scalar_ranks]
        else:
            ranks_for_method = [int(r) for r in scalar_ranks]
        for rank in ranks_for_method:
            tag = f"{method}_r{_rank_tag(rank)}"
            cell_path = cache_dir / f"{tag}.npz"
            if cell_path.exists() and not force_recompute:
                cell = QualityCell.from_npz(cell_path)
                if verbose:
                    rel = abs(cell.G_anh - dense.G_anh) / max(abs(dense.G_anh), 1e-30)
                    print(f"[transport_quality] {tag}: cache hit "
                          f"(rel_err={cell.frob_rel_err:.3e}, "
                          f"ΔG_anh/G={rel:.3e})")
                cells.append(cell)
                continue
            if verbose:
                print(f"[transport_quality] {tag}: computing ...")
            cell = _compute_cell(
                bundle, method, rank, target_norm=bundle.fc3_target.target_norm,
                solver_kw=solver_kw, extra_kwargs=extra.get(method, {}),
                verbose=verbose,
            )
            cell.to_npz(cell_path)
            cells.append(cell)
            if verbose and cell.error is None:
                rel = abs(cell.G_anh - dense.G_anh) / max(abs(dense.G_anh), 1e-30)
                print(f"    -> rel_err={cell.frob_rel_err:.3e}, "
                      f"ΔG_anh/G={rel:.3e}, "
                      f"transport_time={cell.transport_time_s:.1f} s")
            elif cell.error is not None:
                print(f"    !! FAILED: {cell.error}")

    # Cross-section area used for the W/(m²·K) conductance — extracted
    # once here so the CSV and per-wire plot share the same number.
    A_c_m2 = _box_cross_section_m2(bundle, solver_kw["transport_direction"])

    # --- write tabular outputs -------------------------------------------
    _write_csv(cells, out_dir / "transport_quality.csv", dense, A_c_m2)
    summary = _build_summary(cells, dense, A_c_m2)
    (out_dir / "transport_quality.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    # --- plots -----------------------------------------------------------
    _plot_frob_vs_params(cells, dense, out_dir / "transport_quality_frob_vs_params")
    _plot_ganh_vs_params(
        cells, dense, out_dir / "transport_quality_ganh_vs_params", A_c_m2,
    )
    _plot_frob_vs_transport_err(
        cells, dense, out_dir / "transport_quality_frob_vs_transport_err",
    )
    _plot_spectral_current(
        cells, dense, out_dir / "transport_quality_spectral_current",
    )
    _plot_conservation(cells, out_dir / "transport_quality_conservation")
    return summary


# ---------------------------------------------------------------------------
# Tabular outputs
# ---------------------------------------------------------------------------


def _box_cross_section_m2(bundle: SystemBundle, transport_direction: str) -> float:
    """Cross-section perpendicular to transport, in m². Same definition as
    ``solver/dense.py:508–513`` (norm of the cross product of the
    primitive lattice vectors perpendicular to the transport axis) so
    the per-wire pW/K conversion is consistent with the W/(m²·K)
    conductance returned by the solver. For SiNW the box is mostly
    vacuum; this is documented next to the per-wire column.
    """
    lattice = bundle.phonon.primitive.cell
    tidx = "xyz".index(transport_direction)
    perp = [i for i in range(3) if i != tidx]
    a1 = np.asarray(lattice[perp[0]])
    a2 = np.asarray(lattice[perp[1]])
    return float(np.linalg.norm(np.cross(a1, a2))) * 1e-20


def _ballistic_collapse(cell: QualityCell, dense: QualityCell,
                        tol: float = 1e-5) -> bool:
    """True when the SCBA reduced to the ballistic fixed point — i.e. the
    reconstructed FC3 produced a vanishing bubble and G_anh ≈ G_ball.
    A common failure mode of low-rank FC3 fits, surfaced in the plots
    and the CSV so the reader doesn't mistake "flat horizontal line"
    for "low transport error".
    """
    if not (np.isfinite(cell.G_anh) and np.isfinite(cell.G_ball)):
        return False
    denom = max(abs(cell.G_ball), 1e-30)
    return abs(cell.G_anh - cell.G_ball) / denom < tol


def _write_csv(cells: list[QualityCell], path: Path, dense: QualityCell,
               A_c_m2: float) -> None:
    rows = []
    for c in cells:
        rel_G = (
            abs(c.G_anh - dense.G_anh) / max(abs(dense.G_anh), 1e-30)
            if np.isfinite(c.G_anh) else float("nan")
        )
        # Per-wire conductance: G_anh [W/(m²·K)] × A_c [m²] = W/K. Scale
        # to pW/K because absolute values for ~1-nm SiNW are tens of
        # pW/K, much more readable than the 3e7 W/(m²·K) numbers.
        G_anh_pW_per_K = c.G_anh * A_c_m2 * 1e12 if np.isfinite(c.G_anh) else float("nan")
        G_ball_pW_per_K = c.G_ball * A_c_m2 * 1e12 if np.isfinite(c.G_ball) else float("nan")
        rows.append({
            "method": c.method,
            "rank": _rank_tag(c.rank),
            "n_params": c.n_params,
            "frob_rel_err": c.frob_rel_err,
            "fit_time_s": c.fit_time_s,
            "transport_time_s": c.transport_time_s,
            "G_anh_W_per_m2_K": c.G_anh,
            "G_anh_pW_per_K": G_anh_pW_per_K,
            "G_anh_rel_err_vs_dense": rel_G,
            "G_ball_W_per_m2_K": c.G_ball,
            "G_ball_pW_per_K": G_ball_pW_per_K,
            "conservation_err": c.conservation_err,
            "n_scba_iter": c.n_scba_iter,
            "ballistic_collapse": _ballistic_collapse(c, dense),
            "error": c.error or "",
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _build_summary(cells: list[QualityCell], dense: QualityCell,
                   A_c_m2: float) -> dict:
    by_method: dict[str, list[dict]] = {}
    for c in cells:
        if c.method == "dense":
            continue
        rel_G = (
            abs(c.G_anh - dense.G_anh) / max(abs(dense.G_anh), 1e-30)
            if np.isfinite(c.G_anh) else float("nan")
        )
        by_method.setdefault(c.method, []).append({
            "rank": _rank_tag(c.rank),
            "n_params": c.n_params,
            "frob_rel_err": c.frob_rel_err,
            "G_anh_rel_err": rel_G,
            "fit_time_s": c.fit_time_s,
            "transport_time_s": c.transport_time_s,
            "error": c.error,
        })
    return {
        "units": {
            "frob_rel_err": "dimensionless (relative Frobenius)",
            "G_anh_rel_err": "dimensionless (relative thermal conductance)",
            "G_anh": "W/(m²·K)",
            "G_anh_pW_per_K": "pW/K (per wire, box-area convention)",
            "fit_time_s": "s",
            "transport_time_s": "s",
        },
        "cross_section": {
            "A_c_m2": A_c_m2,
            "note": (
                "Box cross-section perpendicular to the transport axis "
                "(includes vacuum padding for isolated wires). Per-wire "
                "pW/K = G_anh × A_c × 1e12. See solver/dense.py:508."
            ),
        },
        "dense": {
            "G_anh": dense.G_anh,
            "G_ball": dense.G_ball,
            "G_anh_pW_per_K": dense.G_anh * A_c_m2 * 1e12,
            "G_ball_pW_per_K": dense.G_ball * A_c_m2 * 1e12,
            "conservation_err": dense.conservation_err,
            "n_scba_iter": dense.n_scba_iter,
            "transport_time_s": dense.transport_time_s,
        },
        "per_method": by_method,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _save(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(base.with_suffix(".png")), dpi=160, bbox_inches="tight")
    fig.savefig(str(base.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


def _method_cells(cells: list[QualityCell], method: str) -> list[QualityCell]:
    return sorted(
        (c for c in cells if c.method == method and c.error is None),
        key=lambda c: c.n_params,
    )


def _setup_axes(ax, *, xlog=False, ylog=False, grid=True) -> None:
    if xlog: ax.set_xscale("log")
    if ylog: ax.set_yscale("log")
    if grid: ax.grid(True, which="both", alpha=0.3, lw=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _plot_frob_vs_params(cells, dense, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plotted = False
    for method in METHOD_ORDER:
        rows = _method_cells(cells, method)
        if not rows:
            continue
        x = [c.n_params for c in rows]
        y = [c.frob_rel_err for c in rows]
        ax.plot(
            x, y, "-", color=PALETTE[method], marker=MARKERS[method],
            markersize=6, lw=1.5, label=method,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel(r"# parameters")
    ax.set_ylabel(r"$\|T - T_{\mathrm{approx}}\|_F / \|T\|_F$")
    ax.set_title("FC3 approximation quality — Frobenius residual")
    _setup_axes(ax, xlog=True, ylog=True)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    _save(fig, out_base)


def _plot_ganh_vs_params(cells, dense, out_base: Path, A_c_m2: float) -> None:
    if not np.isfinite(dense.G_anh):
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    # Primary y-axis: per-wire pW/K (natural unit for a nanowire). The
    # secondary y-axis on the right keeps the W/(m²·K) convention for
    # cross-system comparability with the Si/Ge interface result.
    to_pW_per_K = lambda g: g * A_c_m2 * 1e12  # noqa: E731

    collapsed_x: list[float] = []
    collapsed_y: list[float] = []
    plotted = False
    for method in METHOD_ORDER:
        rows = _method_cells(cells, method)
        if not rows:
            continue
        x = [c.n_params for c in rows]
        y = [to_pW_per_K(c.G_anh) for c in rows]
        ax.plot(
            x, y, "-", color=PALETTE[method], marker=MARKERS[method],
            markersize=6, lw=1.5, label=method,
        )
        for c, yv in zip(rows, y, strict=True):
            if _ballistic_collapse(c, dense):
                collapsed_x.append(c.n_params)
                collapsed_y.append(yv)
        plotted = True
    if not plotted:
        plt.close(fig)
        return

    # Two reference lines: dense G_anh (the target) and dense G_ball
    # (the ballistic fixed point that bad fits collapse onto).
    ax.axhline(
        to_pW_per_K(dense.G_anh), color=PALETTE["dense"], lw=1.6, ls="-",
        label=f"dense $G_{{anh}}$ = {to_pW_per_K(dense.G_anh):.2f} pW/K",
    )
    ax.axhline(
        to_pW_per_K(dense.G_ball), color="0.55", lw=1.2, ls=":",
        label=f"dense $G_{{ball}}$ = {to_pW_per_K(dense.G_ball):.2f} pW/K",
    )
    if collapsed_x:
        ax.scatter(
            collapsed_x, collapsed_y, s=110, facecolors="none",
            edgecolors="red", linewidths=1.4, zorder=5,
            label="ballistic collapse (Σ ≈ 0)",
        )

    ax.set_xlabel(r"# parameters")
    ax.set_ylabel(r"$G_{\mathrm{anh}}$  [pW/K  per wire]")
    ax.set_title("Anharmonic thermal conductance vs. compression budget")
    _setup_axes(ax, xlog=True)
    # Bound the y-axis so the dense_G_ball line is at most ~1.1× the top.
    ymax = 1.1 * to_pW_per_K(dense.G_ball)
    ax.set_ylim(top=ymax, bottom=0)

    # Right-hand axis: same data in W/(m²·K) for cross-system comparison.
    ax_r = ax.twinx()
    ax_r.set_ylim(0, ymax / (A_c_m2 * 1e12))
    ax_r.set_ylabel(r"$G_{\mathrm{anh}}$  [W/(m$^2$·K)]")
    for sp in ("top",):
        ax_r.spines[sp].set_visible(False)

    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    fig.text(
        0.01, -0.02,
        f"Box cross-section A_c = {A_c_m2 * 1e20:.1f} Å² "
        f"({A_c_m2:.2e} m²). Includes vacuum padding for isolated NW.",
        fontsize=7, color="grey",
    )
    fig.tight_layout()
    _save(fig, out_base)


def _plot_frob_vs_transport_err(cells, dense, out_base: Path) -> None:
    """The headline scatter: does Frobenius residual predict transport error?"""
    if not np.isfinite(dense.G_anh):
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    G0 = abs(dense.G_anh)
    plotted = False
    for method in METHOD_ORDER:
        rows = _method_cells(cells, method)
        if not rows:
            continue
        x = [c.frob_rel_err for c in rows]
        y = [abs(c.G_anh - dense.G_anh) / max(G0, 1e-30) for c in rows]
        ax.plot(
            x, y, "-", color=PALETTE[method], marker=MARKERS[method],
            markersize=6, lw=1.5, label=method,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    # Diagonal reference: G error == Frob error.
    lo, hi = ax.get_xlim()
    lim = (max(min(lo, 1e-4), 1e-6), max(hi, 1e0))
    diag = np.geomspace(lim[0], lim[1], 80)
    ax.plot(diag, diag, ":", color="grey", lw=0.8, label="y = x")
    ax.set_xlabel(r"$\|T - T_{\mathrm{approx}}\|_F / \|T\|_F$")
    ax.set_ylabel(r"$|G_{\mathrm{anh}} - G_{\mathrm{anh,dense}}| / G_{\mathrm{anh,dense}}$")
    ax.set_title("Predictive value of Frobenius error for thermal-conductance error")
    _setup_axes(ax, xlog=True, ylog=True)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    _save(fig, out_base)


def _plot_spectral_current(cells, dense, out_base: Path) -> None:
    if dense.freqs_thz.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    # Convert W/THz → pW/THz so the y-axis reads in physical units
    # instead of "1e-22 W/THz" exponent labels.
    ax.plot(
        dense.freqs_thz, dense.spectral_heat_current * 1e12,
        color=PALETTE["dense"], lw=2.0, label="dense",
    )
    for method in METHOD_ORDER:
        rows = _method_cells(cells, method)
        if not rows:
            continue
        c = rows[-1]   # highest rank
        if c.freqs_thz.size == 0:
            continue
        ax.plot(
            c.freqs_thz, c.spectral_heat_current * 1e12,
            color=PALETTE[method], lw=1.2, alpha=0.85,
            label=f"{method} r{_rank_tag(c.rank)}",
        )
    ax.set_xlabel(r"$\omega$  [THz]")
    ax.set_ylabel(r"spectral heat current  [pW/THz]")
    ax.set_title("Spectral current — dense vs. highest-rank approximations")
    # The lowest plotted frequency is Δω, not exact ω=0; mark it so the
    # finite low-ω weight isn't misread as an unphysical Drude term at
    # ω=0 (which is excluded from the integration by pos_mask in
    # solver/dense.py).
    if dense.freqs_thz.size:
        dw = float(dense.freqs_thz[0])
        ax.axvline(dw, color="grey", ls=":", lw=0.7, alpha=0.7)
        ax.text(
            dw, ax.get_ylim()[1] * 0.95,
            f"  $\\Delta\\omega$ = {dw:.2f} THz  (ω=0 excluded)",
            fontsize=7, color="grey", va="top", ha="left",
        )
    _setup_axes(ax)
    ax.legend(fontsize=9, frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, out_base)


def _plot_conservation(cells, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    plotted = False
    bailed_x: list[float] = []
    bailed_y: list[float] = []
    for method in METHOD_ORDER:
        rows = _method_cells(cells, method)
        if not rows:
            continue
        x = [c.n_params for c in rows]
        y = [abs(c.conservation_err) + 1e-16 for c in rows]
        ax.plot(
            x, y, "-", color=PALETTE[method], marker=MARKERS[method],
            markersize=6, lw=1.5, label=method,
        )
        # A 2-iteration SCBA is the "Σ ≈ 0 → ballistic on first
        # iterate" pathology (mixing converges trivially because there
        # is nothing to update). Mark these so the tiny conservation
        # residual isn't mistaken for a good fit.
        for c, yv in zip(rows, y, strict=True):
            if c.n_scba_iter <= 2:
                bailed_x.append(c.n_params)
                bailed_y.append(yv)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    if bailed_x:
        ax.scatter(
            bailed_x, bailed_y, s=100, facecolors="none", edgecolors="red",
            linewidths=1.4, zorder=5,
            label="SCBA bailed at iter ≤ 2 (Σ ≈ 0)",
        )
    ax.set_xlabel(r"# parameters")
    ax.set_ylabel(r"$|J_L - J_R| / (|J_L| + |J_R|)$")
    ax.set_title("Heat-flow conservation residual per fit")
    _setup_axes(ax, xlog=True, ylog=True)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, out_base)
