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

When every rank contains the complete frequency axis and instead distributes
q or block work, use ``--replicated-parts N``.  Each rank file is then
regridded independently and retains its original non-frequency axes.

Production Si runs use a Cartesian communicator: ``stack`` partitions the
frequency axis and ``q`` partitions work while retaining a complete, reduced
Sigma on every q peer.  Use ``--stack-parts S --q-parts Q`` for that layout.
Only the first q replica is read; the transformed stack slices are hard-linked
into the other q replica names.  Periodic convex interpolation between regular
transverse meshes is enabled with, for example,
``--old-q-shape 5,5 --new-q-shape 7,7``.  It is a warm-start transformation,
not a replacement for converging the target q functional.
"""

from __future__ import annotations

import argparse
import os
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


def _parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"q shape must be comma-separated integers, got {value!r}"
        ) from exc
    if not shape or any(size < 1 for size in shape):
        raise argparse.ArgumentTypeError(
            f"q shape entries must be positive, got {value!r}")
    return shape


def _periodic_linear_axis(
        values: np.ndarray, new_size: int, axis: int) -> np.ndarray:
    """Convex periodic linear interpolation on one regular unit interval."""
    old_size = values.shape[axis]
    if old_size == new_size:
        return values
    position = np.arange(new_size, dtype=float) * old_size / new_size
    lower = np.floor(position).astype(np.int64)
    upper = (lower + 1) % old_size
    weight = position - lower
    shape = [1] * values.ndim
    shape[axis] = new_size
    weight = weight.reshape(shape)
    return ((1.0 - weight) * np.take(values, lower, axis=axis)
            + weight * np.take(values, upper, axis=axis))


def _interpolate_periodic_q(
        values: np.ndarray, old_shape: tuple[int, ...],
        new_shape: tuple[int, ...]) -> np.ndarray:
    """Tensor-product convex interpolation of q axes after frequency."""
    if len(old_shape) != len(new_shape):
        raise ValueError(
            f"old/new q dimensionality differs: {old_shape} vs {new_shape}")
    actual = tuple(values.shape[1:1 + len(old_shape)])
    if actual != old_shape:
        raise ValueError(
            f"snapshot q axes {actual} do not match --old-q-shape "
            f"{old_shape}")
    result = values
    for offset, new_size in enumerate(new_shape, start=1):
        result = _periodic_linear_axis(result, new_size, offset)
    return result


def _regrid_state(
        state: dict[str, np.ndarray], old: np.ndarray, new: np.ndarray,
        scale: float, old_q_shape: tuple[int, ...] | None = None,
        new_q_shape: tuple[int, ...] | None = None) -> dict[str, np.ndarray]:
    out = {}
    for key in _SIGMA_KEYS:
        sig = state[key]
        if sig.shape[0] != old.size:
            raise SystemExit(
                f"{key}: frequency axis {sig.shape[0]} does not match the "
                f"old grid ({old.size} pts); wrong --old-grid?")
        value = _interpolate_frequency(sig, old, new)
        if old_q_shape is not None and new_q_shape is not None:
            value = _interpolate_periodic_q(
                value, old_q_shape, new_q_shape)
        out[key] = scale * value
    return out


def _save_cartesian_sigma(
        base: Path, state: dict[str, np.ndarray], stack_parts: int,
        q_parts: int) -> None:
    """Write stack slices and link identical q-work replicas."""
    _save_sigma(base, state, stack_parts)
    for q_rank in range(1, q_parts):
        for stack_rank in range(stack_parts):
            source = _rank_path(base, stack_rank)
            target = _rank_path(base, q_rank * stack_parts + stack_rank)
            if target.exists() or target.is_symlink():
                target.unlink()
            try:
                os.link(source, target)
            except OSError:
                # Some parallel filesystems disable hard links.  Preserve the
                # restart layout at the cost of duplicate storage there.
                import shutil
                shutil.copy2(source, target)


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
    p.add_argument("--replicated-parts", type=int, default=0,
                   help="independently regrid this many .rankN.npz files "
                        "whose frequency axes are complete [0]")
    p.add_argument("--stack-parts", type=int, default=0,
                   help="frequency slices in a Cartesian stack x q layout")
    p.add_argument("--q-parts", type=int, default=0,
                   help="identical q-work replicas in the Cartesian layout")
    p.add_argument("--old-q-shape", type=_parse_shape,
                   help="regular transverse q axes in the source, e.g. 5,5")
    p.add_argument("--new-q-shape", type=_parse_shape,
                   help="regular transverse q axes in the target, e.g. 7,7")
    a = p.parse_args()

    if min(a.input_parts, a.output_parts, a.replicated_parts,
           a.stack_parts, a.q_parts) < 0:
        p.error("part counts must be non-negative")
    if a.replicated_parts and (a.input_parts or a.output_parts
                               or a.stack_parts or a.q_parts):
        p.error("--replicated-parts cannot be combined with "
                "other part-layout arguments")
    if bool(a.stack_parts) != bool(a.q_parts):
        p.error("--stack-parts and --q-parts must be supplied together")
    if a.stack_parts and (a.input_parts or a.output_parts):
        p.error("Cartesian --stack-parts/--q-parts cannot be combined with "
                "--input-parts or --output-parts")
    if bool(a.old_q_shape) != bool(a.new_q_shape):
        p.error("--old-q-shape and --new-q-shape must be supplied together")

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
    if a.replicated_parts:
        for rank in range(a.replicated_parts):
            source = _rank_path(a.sigma, rank)
            target = _rank_path(a.out, rank)
            state = _load_sigma(source, parts=0)
            _save_sigma(
                target, _regrid_state(
                    state, old, new, a.scale,
                    a.old_q_shape, a.new_q_shape), parts=0)
    elif a.stack_parts:
        snap = _load_sigma(a.sigma, a.stack_parts)
        _save_cartesian_sigma(
            a.out,
            _regrid_state(
                snap, old, new, a.scale,
                a.old_q_shape, a.new_q_shape),
            a.stack_parts, a.q_parts)
    else:
        snap = _load_sigma(a.sigma, a.input_parts)
        _save_sigma(
            a.out, _regrid_state(
                snap, old, new, a.scale,
                a.old_q_shape, a.new_q_shape), a.output_parts)
    input_count = (a.stack_parts * a.q_parts if a.stack_parts
                   else a.replicated_parts or a.input_parts or 1)
    output_count = (a.stack_parts * a.q_parts if a.stack_parts
                    else a.replicated_parts or a.output_parts or 1)
    print(f"wrote {a.out}: {old.size} -> {new.size} frequency points "
          f"(scale {a.scale}, input parts "
          f"{input_count}, output parts {output_count}); use "
          f"QX_SIGMA_INIT={a.out} on the new grid.")


if __name__ == "__main__":
    main()
