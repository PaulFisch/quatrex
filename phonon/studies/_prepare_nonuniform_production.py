"""Prepare matched production dual-grid cases on the current filesystem.

The script is intended to be launched through ``phonon/scripts/daint.py`` so
the generated run directories live beside the existing Alps artifacts.  It
does no physics: it selects a subset of primary nodes from a saved converged
run, writes ``phonon_energies.npy``, and keeps the original uniform spacing as
the SCBA auxiliary grid.  Thus the A/B changes only primary sampling.

Prepared cases:

* ``c16x2-nu`` from the conserving CNT 8x2 run;
* ``si-l8x2-nu`` from the conserving longer-Si 8x2 run, including a
  per-q-rank interpolation of its saved warm-start self-energy.

No production code or public configuration is modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np

try:  # package import in tests / PYTHONPATH=phonon
    from studies._mixed_structure_review import (
        _frequency_channels, linear_reconstruction, weighted_error)
except ModuleNotFoundError:  # direct ``python phonon/studies/...py`` execution
    from _mixed_structure_review import (  # type: ignore[no-redef]
        _frequency_channels, linear_reconstruction, weighted_error)


REPO = Path(__file__).resolve().parents[2]


def selected_knots(run: Path, tolerance: float) -> np.ndarray:
    """Greedy common P1 grid used by the cross-structure pilot."""
    x, weights, values = _frequency_channels(str(run))
    selected = [0, x.size - 1]
    while len(selected) < x.size:
        ids = np.asarray(sorted(selected), dtype=int)
        approx = linear_reconstruction(x, values, ids)
        if weighted_error(values, approx, weights) <= tolerance:
            return ids
        row_error = weights * np.sum(np.abs(values - approx) ** 2, axis=1)
        row_error[ids] = -1.0
        selected.append(int(np.argmax(row_error)))
    return np.arange(x.size)


def _replace_or_insert(cfg: str, key: str, value: str,
                       section_after: str = "[phonon.solver]") -> str:
    pattern = rf"(?m)^{re.escape(key)} = .*?$"
    if re.search(pattern, cfg):
        return re.sub(pattern, f"{key} = {value}", cfg)
    return cfg.replace(section_after, f"{key} = {value}\n{section_after}", 1)


def prepare_case(base: Path, reference: Path, target: Path, tolerance: float,
                 warm_parts: int = 0, warm_base: Path | None = None) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    ref = np.load(reference)
    old_grid = np.asarray(ref["energies"], float)
    ids = selected_knots(reference, tolerance)
    new_grid = old_grid[ids]
    if not np.allclose(np.diff(old_grid), np.diff(old_grid)[0],
                       rtol=1e-10, atol=1e-12):
        raise ValueError(f"reference grid is not uniform: {reference}")
    aux_dw = float(old_grid[1] - old_grid[0])
    np.save(target / "phonon_energies.npy", new_grid)

    for name in ("dynamical_matrix.mat", "fc3_blocks.hdf5", "kshift.npy",
                 "structure.xyz", "qfold_vertices.npz",
                 "decomposed_vertices.npz"):
        source = base / name
        dest = target / name
        if source.exists() and not dest.exists():
            dest.symlink_to(source.resolve())

    cfg = (base / "quatrex_config.toml").read_text()
    cfg = cfg.replace(str(base.resolve()), str(target.resolve()))
    cfg = re.sub(r"(?m)^energy_window_num = \d+$",
                 f"energy_window_num = {new_grid.size}", cfg)
    cfg = _replace_or_insert(cfg, "frequency_grid", '"file"')
    cfg = _replace_or_insert(cfg, "sse_aux_grid_dw_thz", repr(aux_dw))
    cfg = _replace_or_insert(cfg, "sse_aux_grid_fmax_thz",
                             repr(float(old_grid[-1])))
    (target / "quatrex_config.toml").write_text(cfg)

    warm_written = 0
    if warm_parts:
        if warm_base is None:
            raise ValueError("warm_base is required with warm_parts")
        for rank in range(warm_parts):
            source = Path(f"{warm_base}.rank{rank}.npz")
            if not source.exists():
                raise FileNotFoundError(source)
            snap = np.load(source)
            out = {}
            for key in ("sigma_lesser", "sigma_greater", "sigma_retarded"):
                data = np.asarray(snap[key])
                if data.shape[0] != old_grid.size:
                    raise ValueError(
                        f"{source}:{key} has {data.shape[0]} frequencies, "
                        f"expected {old_grid.size}; q-distributed warm start required")
                flat = data.reshape(data.shape[0], -1)
                interp = np.empty((new_grid.size, flat.shape[1]), data.dtype)
                for j in range(flat.shape[1]):
                    interp[:, j] = (
                        np.interp(new_grid, old_grid, flat[:, j].real) +
                        1j * np.interp(new_grid, old_grid, flat[:, j].imag))
                out[key] = interp.reshape((new_grid.size,) + data.shape[1:])
            np.savez(target / f"sigma_init.rank{rank}.npz", **out)
            warm_written += 1

    return {
        "target": str(target), "uniform_points": int(old_grid.size),
        "primary_points": int(new_grid.size), "tolerance": tolerance,
        "aux_dw_thz": aux_dw, "aux_points": int(old_grid.size),
        "min_primary_spacing_thz": float(np.min(np.diff(new_grid))),
        "max_primary_spacing_thz": float(np.max(np.diff(new_grid))),
        "warm_parts": warm_written,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--tolerance", type=float, default=1e-3)
    ap.add_argument("--case", choices=("all", "cnt", "si"), default="all")
    args = ap.parse_args(argv)
    root = args.repo.resolve()
    rows = []
    if args.case in ("all", "cnt"):
        rows.append(prepare_case(
            root / "cluster/c16x2h", root / "cluster/c16x2h/run.npz",
            root / "cluster/c16x2-nu", args.tolerance))
    if args.case in ("all", "si"):
        rows.append(prepare_case(
            root / "cluster/sifilm8x2",
            root / "cluster/si-l8x2-final/run.npz",
            root / "cluster/si-l8x2-nu", args.tolerance,
            warm_parts=4,
            warm_base=root / "cluster/si-l8x2-final/sigma_best"))
    for row in rows:
        print(row, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
