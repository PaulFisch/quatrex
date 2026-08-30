"""Interpolate a saved self-energy snapshot onto a new frequency grid.

Usage:
  python phonon/studies/regrid_sigma.py --sigma old_sigma.npz \
      --old-grid old/run.npz --new-grid new_grid.npy \
      --out new_sigma.npz [--scale 1.0]

Frequency-distributed snapshots use the same base name on every rank.  They
can be gathered, interpolated and repartitioned without a single-rank solver
run by adding ``--input-parts N --output-parts M``.  For example,
``--sigma sigma_best --input-parts 4 --out sigma_seed --output-parts 4``
reads ``sigma_best.rank0.npz`` through ``rank3`` and writes the corresponding
four ``sigma_seed.rank*.npz`` files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_grid(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        return np.asarray(np.load(path)["energies"], dtype=float)
    return np.asarray(np.load(path), dtype=float).ravel()


_SIGMA_KEYS = ("sigma_lesser", "sigma_greater", "sigma_retarded")


def _rank_path(base: Path, rank: int) -> Path:
    """Return the production distributed-snapshot name for ``rank``."""
    stem = base.with_suffix("") if base.suffix == ".npz" else base
    return stem.with_name(f"{stem.name}.rank{rank}.npz")


def _load_sigma(base: Path, parts: int) -> dict[str, np.ndarray]:
    if parts <= 0:
        snap = np.load(base)
        return {key: np.asarray(snap[key]) for key in _SIGMA_KEYS}

    snapshots = [np.load(_rank_path(base, rank)) for rank in range(parts)]
    return {
        key: np.concatenate([np.asarray(snap[key]) for snap in snapshots],
                            axis=0)
        for key in _SIGMA_KEYS
    }


def _save_sigma(base: Path, state: dict[str, np.ndarray], parts: int) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    if parts <= 0:
        np.savez(base, **state)
        return

    split_state = {key: np.array_split(value, parts, axis=0)
                   for key, value in state.items()}
    for rank in range(parts):
        np.savez(_rank_path(base, rank),
                 **{key: values[rank]
                    for key, values in split_state.items()})


def _interpolate_frequency(
        values: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Piecewise-linearly regrid the leading frequency axis.

    Refinement campaigns often extend a uniform grid without moving any of
    its existing samples.  Detect that case and copy the complete matrix
    slices directly.  Besides being exact, this avoids one ``np.interp`` call
    per q/matrix element for multi-gigabyte distributed Si checkpoints.
    Samples outside the source interval retain the established zero-fill
    convention.
    """
    sig = np.asarray(values)
    old = np.asarray(old, dtype=float).ravel()
    new = np.asarray(new, dtype=float).ravel()
    if sig.shape[0] != old.size:
        raise ValueError(
            f"frequency axis {sig.shape[0]} does not match the old grid "
            f"({old.size} points)")

    indices = np.searchsorted(new, old)
    inside = (new >= old[0] - 1e-12) & (new <= old[-1] + 1e-12)
    nested = (
        np.all(indices < new.size)
        and np.allclose(new[indices], old, rtol=0.0, atol=1e-12)
        and np.count_nonzero(inside) == old.size
    )
    if nested:
        out = np.zeros((new.size,) + sig.shape[1:], dtype=sig.dtype)
        out[indices] = sig
        return out

    flat = sig.reshape(sig.shape[0], -1)
    result = np.empty((new.size, flat.shape[1]), dtype=flat.dtype)
    for j in range(flat.shape[1]):
        result[:, j] = (
            np.interp(new, old, flat[:, j].real, left=0.0, right=0.0)
            + 1j * np.interp(
                new, old, flat[:, j].imag, left=0.0, right=0.0)
        )
    return result.reshape((new.size,) + sig.shape[1:])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sigma", type=Path, required=True,
                   help="QX_SAVE_SIGMA snapshot (.npz)")
    p.add_argument("--input-parts", type=int, default=0,
                   help="gather this many .rankN.npz input slices [0]")
    p.add_argument("--old-grid", type=Path, required=True,
                   help="grid the snapshot lives on (.npy, or run.npz "
                        "with an 'energies' key)")
    p.add_argument("--new-grid", type=Path,
                   help="target grid (.npy or run.npz)")
    p.add_argument("--new-min", type=float,
                   help="uniform target-grid minimum")
    p.add_argument("--new-max", type=float,
                   help="uniform target-grid maximum")
    p.add_argument("--new-points", type=int,
                   help="number of points on a uniform target grid")
    p.add_argument("--scale", type=float, default=1.0,
                   help="extra scale on the loaded Sigma (cf. "
                        "QX_SIGMA_SCALE) [1.0]")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--output-parts", type=int, default=0,
                   help="split output into this many .rankN.npz slices [0]")
    a = p.parse_args()

    if a.input_parts < 0 or a.output_parts < 0:
        p.error("--input-parts and --output-parts must be non-negative")

    old = _load_grid(a.old_grid)
    uniform_args = (a.new_min, a.new_max, a.new_points)
    if a.new_grid is not None and any(value is not None
                                      for value in uniform_args):
        p.error("use either --new-grid or --new-min/--new-max/--new-points")
    if a.new_grid is not None:
        new = _load_grid(a.new_grid)
    elif all(value is not None for value in uniform_args):
        if a.new_points < 2:
            p.error("--new-points must be at least two")
        new = np.linspace(a.new_min, a.new_max, a.new_points)
    else:
        p.error("provide --new-grid or all uniform target-grid arguments")
    snap = _load_sigma(a.sigma, a.input_parts)

    out = {}
    for key in ("sigma_lesser", "sigma_greater", "sigma_retarded"):
        sig = snap[key]
        if sig.shape[0] != old.size:
            raise SystemExit(
                f"{key}: frequency axis {sig.shape[0]} does not match the "
                f"old grid ({old.size} pts); wrong --old-grid?")
        out[key] = a.scale * _interpolate_frequency(sig, old, new)

    _save_sigma(a.out, out, a.output_parts)
    print(f"wrote {a.out}: {old.size} -> {new.size} frequency points "
          f"(scale {a.scale}, input parts {a.input_parts or 1}, "
          f"output parts {a.output_parts or 1}); use "
          f"QX_SIGMA_INIT={a.out} on the new grid.")


if __name__ == "__main__":
    main()
