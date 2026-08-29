#!/usr/bin/env python3
"""Algebraic lower bounds for conventional-cell Si FC3 factor ranks.

An INDSCAL vertex of rank R has matrix rank at most R in either contracted-leg
unfolding.  The truncated SVD error of that unfolding is therefore a strict
lower bound on every possible fit, independent of optimiser, restart count or
initialisation.  This study distinguishes an inadequate fit from a rank that
cannot meet the requested tolerance at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT / "phonon", _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

def unfolding_lower_bounds(
    target: np.ndarray, ranks: tuple[int, ...]
) -> tuple[np.ndarray, dict[int, float]]:
    """Return singular values and unavoidable relative errors at ``ranks``."""
    target = np.asarray(target, dtype=float)
    if target.ndim != 3:
        raise ValueError("target must be a third-order tensor")
    unfolding = target.transpose(1, 0, 2).reshape(target.shape[1], -1)
    singular_values = np.linalg.svd(unfolding, compute_uv=False)
    tail2 = np.r_[np.cumsum(singular_values[::-1] ** 2)[::-1], 0.0]
    norm = float(np.linalg.norm(target))
    bounds = {
        int(rank): float(np.sqrt(tail2[min(rank, len(singular_values))])
                         / max(norm, np.finfo(float).tiny))
        for rank in ranks
    }
    return singular_values, bounds


def main() -> None:
    import h5py
    from studies.engine.build_inputs import _load_bulk_film
    from phonon_inputs.separable import (
        build_realspace_fc3_matrices,
        build_supercell_mapping,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fc3-subdir", required=True)
    parser.add_argument("--tdir", default="x")
    parser.add_argument("--ranks", default="64,128,256,512")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    phonon, fc3_path = _load_bulk_film(args.fc3_subdir)
    nat = len(phonon.primitive.masses)
    prim, frac, slab, reference = build_supercell_mapping(phonon, args.tdir)
    del prim, frac, slab
    with h5py.File(fc3_path, "r") as handle:
        fc3 = handle["fc3"][:]
    stacked = build_realspace_fc3_matrices(
        fc3, nat, phonon.supercell.masses, reference
    )
    dimension = 3 * len(phonon.supercell.masses)
    target = np.asarray(stacked, float).reshape(3 * nat, dimension, dimension)
    ranks = tuple(int(value) for value in args.ranks.split(",") if value)
    singular_values, bounds = unfolding_lower_bounds(target, ranks)

    source_path = Path(fc3_path).resolve()
    try:
        recorded_path = source_path.relative_to(_ROOT)
    except ValueError:
        recorded_path = source_path
    result = {
        "fc3_path": str(recorded_path),
        "fc3_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "transport_direction": args.tdir,
        "target_shape": list(target.shape),
        "target_norm": float(np.linalg.norm(target)),
        "contracted_leg_s2_asymmetry": float(
            np.linalg.norm(target - target.transpose(0, 2, 1))
            / max(float(np.linalg.norm(target)), np.finfo(float).tiny)
        ),
        "unfolding_numerical_rank_1e12": int(
            np.count_nonzero(singular_values
                             > 1e-12 * singular_values[0])
        ),
        "indscal_relative_error_lower_bound": {
            str(rank): error for rank, error in bounds.items()
        },
        "singular_values": singular_values.tolist(),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
