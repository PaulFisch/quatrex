"""Convergence sweep for the hiphive FC3 fit.

For each subsample size in :attr:`ConvergenceConfig.sizes` and each
``fit_method`` in :attr:`ConvergenceConfig.fit_methods`, refit the
force-constant potential on a deterministic permutation of the master
DFT pool and record:

  * ``rmse_train`` / ``rmse_test`` from ``trainstation.Optimizer``,
  * ``n_parameters`` (sparsity bookkeeping),
  * ``dispersion_max_thz`` and ``n_imaginary_modes`` from a phonopy
    band-structure evaluation on
    :attr:`ConvergenceConfig.dispersion_q_mesh`,
  * ``rotational_residual`` before/after sum-rule projection (if the
    fit's ``HiphiveConfig.rotational_sum_rule`` is not ``"off"``).

A single PDF + PNG plot collects RMSE (log-scale left axis) and
``dispersion_max_thz`` (right axis) across the sweep so the user can
read off the saturation point at a glance.

The harness is callable directly from Python (for tests and notebooks)
and from the CLI via ``python -m phonon_inputs.pipeline ... --convergence-check``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import ConvergenceConfig, HiphiveConfig


@dataclass
class FitResult:
    """One ``(size, fit_method)`` cell of the sweep."""

    size: int
    fit_method: str
    rmse_train: float
    rmse_test: float
    n_parameters: int
    dispersion_max_thz: float
    n_imaginary_modes: int
    rotational_residual_before: float
    rotational_residual_after: float
    fcp_path: str
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def run_convergence_check(
    rattled_structures: list,
    primitive,
    atoms_ideal,
    hh_cfg: HiphiveConfig,
    out_dir: Path,
) -> dict[tuple[int, str], FitResult]:
    """Subsample the master DFT pool and refit at each ``(size, method)``.

    Parameters
    ----------
    rattled_structures
        Master pool of ``ase.Atoms`` with ``arrays["forces"]`` already
        populated (typically produced by the cluster reap path inside
        :func:`hiphive_fc3.reap`).
    primitive
        Primitive cell as :class:`ase.Atoms` or
        :class:`phonopy.structure.atoms.PhonopyAtoms`.
    atoms_ideal
        Undisplaced supercell — must match the rattled structures' cell.
    hh_cfg
        The system's :class:`HiphiveConfig`. Its
        :attr:`HiphiveConfig.convergence` field drives the sweep; if
        ``None`` a default :class:`ConvergenceConfig` is used.
    out_dir
        Output directory; per-size FCPs and the summary plot are written
        here. Created if missing.

    Returns
    -------
    results : dict
        Keyed by ``(size, fit_method)``. Each value is a
        :class:`FitResult`.
    """
    import hiphive
    from hiphive import (
        ClusterSpace, ForceConstantPotential, StructureContainer,
    )
    from hiphive.utilities import prepare_structures
    from trainstation import Optimizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_cfg = hh_cfg.convergence or ConvergenceConfig()
    if not cv_cfg.sizes:
        raise ValueError("ConvergenceConfig.sizes is empty")
    if cv_cfg.pool_size < max(cv_cfg.sizes):
        raise ValueError(
            f"pool_size={cv_cfg.pool_size} < max(sizes)={max(cv_cfg.sizes)}"
        )
    if len(rattled_structures) < max(cv_cfg.sizes):
        raise ValueError(
            f"Only {len(rattled_structures)} rattled structures available; "
            f"need at least max(sizes)={max(cv_cfg.sizes)} to run the sweep."
        )

    print(f"hiphive convergence sweep: sizes={cv_cfg.sizes}, "
          f"fit_methods={list(cv_cfg.fit_methods)}, "
          f"test_fraction={cv_cfg.test_fraction}")
    print(f"  cutoffs={hh_cfg.cutoffs} A, rotational_sum_rule="
          f"{hh_cfg.rotational_sum_rule}")

    # Hiphive's ClusterSpace requires an ``ase.Atoms`` with a populated
    # ``.pbc`` attribute. ``PhonopyAtoms`` lacks it; convert as needed.
    cs_primitive = _to_ase_atoms(primitive)
    cs = ClusterSpace(cs_primitive, list(hh_cfg.cutoffs))
    prepared = prepare_structures(rattled_structures, atoms_ideal)

    rng = np.random.default_rng(cv_cfg.seed)
    pool_perm = rng.permutation(len(prepared))

    results: dict[tuple[int, str], FitResult] = {}
    train_size = max(0.05, min(0.95, 1.0 - cv_cfg.test_fraction))

    for size in cv_cfg.sizes:
        subset = [prepared[i] for i in pool_perm[:size]]
        sc = StructureContainer(cs)
        for s in subset:
            sc.add_structure(s)
        fit_data = sc.get_fit_data()

        for fit_method in cv_cfg.fit_methods:
            label = f"n{size}_{fit_method}"
            fcp_path = out_dir / f"fcp_{label}.fcp"
            error: str | None = None
            try:
                opt = Optimizer(
                    fit_data,
                    fit_method=fit_method,
                    train_size=train_size,
                    **dict(hh_cfg.fit_kwargs),
                )
                opt.train()
                params = opt.parameters
                rot_before, rot_after = _rotational_residual_pair(
                    cs, params, mode=hh_cfg.rotational_sum_rule,
                )
                if hh_cfg.rotational_sum_rule == "post_fit":
                    params = _apply_rotational_sum_rules(cs, params)

                fcp = ForceConstantPotential(cs, params)
                fcp.write(str(fcp_path))

                dmax_thz, n_imag = _dispersion_metrics(
                    fcp, primitive, cv_cfg.dispersion_q_mesh,
                )
                results[(size, fit_method)] = FitResult(
                    size=size,
                    fit_method=fit_method,
                    rmse_train=float(opt.rmse_train),
                    rmse_test=(
                        float(opt.rmse_test)
                        if opt.rmse_test is not None else float("nan")
                    ),
                    n_parameters=int(opt.n_parameters),
                    dispersion_max_thz=float(dmax_thz),
                    n_imaginary_modes=int(n_imag),
                    rotational_residual_before=float(rot_before),
                    rotational_residual_after=float(rot_after),
                    fcp_path=str(fcp_path),
                )
                print(
                    f"  {label}: rmse_train={opt.rmse_train:.3e}, "
                    f"rmse_test={opt.rmse_test if opt.rmse_test else float('nan'):.3e}, "
                    f"n_param={opt.n_parameters}, "
                    f"max_freq={dmax_thz:.2f} THz, n_imag={n_imag}"
                )
            except Exception as exc:
                # RFE / lasso / ardr can all fail at small sample sizes;
                # record nan and continue the sweep rather than abort.
                error = f"{type(exc).__name__}: {exc}"
                results[(size, fit_method)] = FitResult(
                    size=size,
                    fit_method=fit_method,
                    rmse_train=float("nan"),
                    rmse_test=float("nan"),
                    n_parameters=0,
                    dispersion_max_thz=float("nan"),
                    n_imaginary_modes=-1,
                    rotational_residual_before=float("nan"),
                    rotational_residual_after=float("nan"),
                    fcp_path="",
                    error=error,
                )
                print(f"  {label}: FAILED ({error})")

    _write_summary_json(results, out_dir / "convergence_summary.json", hh_cfg)
    plot_convergence(results, out_dir / "convergence_vs_n_structures.png")
    return results


def plot_convergence(
    results: dict[tuple[int, str], FitResult], out_png: Path,
) -> None:
    """Render the RMSE and dispersion-max plot side by side.

    Two y-axes: left = train/test RMSE (log scale), right =
    ``dispersion_max_thz``. One line per ``fit_method``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit_methods = sorted({r.fit_method for r in results.values()})
    sizes = sorted({r.size for r in results.values()})
    if not sizes:
        return

    fig, ax_rmse = plt.subplots(figsize=(8.5, 4.5))
    ax_disp = ax_rmse.twinx()
    cmap = plt.get_cmap("tab10")

    for i, fm in enumerate(fit_methods):
        color = cmap(i % 10)
        rmse_test = []
        rmse_train = []
        dmax = []
        for s in sizes:
            r = results.get((s, fm))
            rmse_test.append(r.rmse_test if r else np.nan)
            rmse_train.append(r.rmse_train if r else np.nan)
            dmax.append(r.dispersion_max_thz if r else np.nan)
        ax_rmse.plot(sizes, rmse_test, "o-", color=color,
                     label=f"{fm} test")
        ax_rmse.plot(sizes, rmse_train, "o--", color=color, alpha=0.55,
                     label=f"{fm} train")
        ax_disp.plot(sizes, dmax, "s:", color=color, alpha=0.6,
                     markersize=4)

    ax_rmse.set_xlabel("n_structures")
    ax_rmse.set_ylabel(r"RMSE force residual  [eV/Å]")
    ax_rmse.set_yscale("log")
    ax_rmse.grid(True, which="both", alpha=0.3)
    ax_disp.set_ylabel(r"max phonon frequency  [THz]  (dashed)")
    ax_rmse.legend(loc="upper right", fontsize=8)
    ax_rmse.set_title("hiphive FC3 convergence vs. n_structures")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(Path(out_png).with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _to_ase_atoms(structure):
    """Coerce a :class:`PhonopyAtoms` (or pass through an :class:`ase.Atoms`)
    into the periodic :class:`ase.Atoms` form expected by hiphive.
    """
    from ase import Atoms
    if isinstance(structure, Atoms):
        if not all(structure.pbc):
            structure = structure.copy()
            structure.pbc = True
        return structure
    # PhonopyAtoms exposes attributes, not getters.
    if hasattr(structure, "symbols") and not callable(structure.symbols):
        return Atoms(
            symbols=list(structure.symbols),
            cell=np.asarray(structure.cell),
            scaled_positions=np.asarray(structure.scaled_positions),
            pbc=True,
        )
    # Fallback for ASE-like objects with getter methods.
    return Atoms(
        symbols=list(structure.get_chemical_symbols()),
        cell=structure.cell,
        scaled_positions=structure.get_scaled_positions(),
        pbc=True,
    )


def _rotational_residual_pair(cs, parameters, *, mode: str) -> tuple[float, float]:
    """Return ``(before, after)`` rotational-residual norms.

    ``before`` is always the residual of the input ``parameters``;
    ``after`` is the residual after the post-fit projection if the
    policy is ``"post_fit"``, otherwise equal to ``before``.
    """
    try:
        from hiphive.core.rotational_constraints import (
            get_rotational_constraint_matrix,
        )
    except ImportError:  # pragma: no cover  (interface noted upstream)
        return float("nan"), float("nan")
    M = get_rotational_constraint_matrix(
        cs, sum_rules=["Huang", "Born-Huang"],
    )
    before = float(np.linalg.norm(M @ parameters))
    if mode == "post_fit":
        after_params = _apply_rotational_sum_rules(cs, parameters)
        after = float(np.linalg.norm(M @ after_params))
    else:
        after = before
    return before, after


def _apply_rotational_sum_rules(cs, parameters):
    """Project ``parameters`` onto the rotational-invariance manifold."""
    from hiphive.core.rotational_constraints import enforce_rotational_sum_rules
    return enforce_rotational_sum_rules(
        cs, parameters, sum_rules=["Huang", "Born-Huang"], alpha=1e-6,
    )


def _dispersion_metrics(fcp, primitive, q_mesh) -> tuple[float, int]:
    """Diagonalise the fitted FC2 on ``q_mesh`` and report max frequency
    and imaginary-mode count.

    Returns
    -------
    dispersion_max_thz : float
        max(|ω|) over the q-mesh, in THz.
    n_imaginary_modes : int
        count of frequencies with ω² < 0 (imaginary modes are reported
        by phonopy with negative ω).
    """
    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError:
        return float("nan"), -1

    # primitive can be PhonopyAtoms or ase.Atoms.
    if not isinstance(primitive, PhonopyAtoms):
        primitive = PhonopyAtoms(
            symbols=list(primitive.get_chemical_symbols()),
            cell=primitive.cell,
            scaled_positions=primitive.get_scaled_positions(),
        )
    ph = Phonopy(primitive, supercell_matrix=np.eye(3, dtype=int))
    # ``fcp.get_force_constants`` expects an ASE-like atoms object with
    # ``get_scaled_positions``; the phonopy Supercell doesn't expose one.
    from ase import Atoms as _Atoms
    sc = ph.supercell
    sc_ase = _Atoms(
        symbols=list(sc.symbols),
        cell=np.asarray(sc.cell),
        scaled_positions=np.asarray(sc.scaled_positions),
        pbc=True,
    )
    fcs = fcp.get_force_constants(sc_ase)
    ph.force_constants = fcs.get_fc_array(order=2)

    ph.run_mesh(list(q_mesh), with_eigenvectors=False)
    mesh = ph.get_mesh_dict()
    freqs = np.asarray(mesh["frequencies"])  # (n_q, n_modes), THz, signed
    n_imag = int(np.sum(freqs < -1e-3))
    return float(np.max(np.abs(freqs))), n_imag


def _write_summary_json(
    results: dict[tuple[int, str], FitResult],
    path: Path,
    hh_cfg: HiphiveConfig,
) -> None:
    summary = {
        "hiphive_config": {
            "supercell": list(hh_cfg.supercell),
            "cutoffs": list(hh_cfg.cutoffs),
            "rotational_sum_rule": hh_cfg.rotational_sum_rule,
        },
        "rows": [
            {
                "size": r.size,
                "fit_method": r.fit_method,
                "rmse_train": r.rmse_train,
                "rmse_test": r.rmse_test,
                "n_parameters": r.n_parameters,
                "dispersion_max_thz": r.dispersion_max_thz,
                "n_imaginary_modes": r.n_imaginary_modes,
                "rotational_residual_before": r.rotational_residual_before,
                "rotational_residual_after": r.rotational_residual_after,
                "fcp_path": r.fcp_path,
                "error": r.error,
            }
            for r in results.values()
        ],
    }
    path.write_text(json.dumps(summary, indent=2, default=str))
