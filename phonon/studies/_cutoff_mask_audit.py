"""Is the H6 box mask active on this run?

``phonon.interaction_cutoff`` is not a vertex knob: ``core/scba.py`` hands it to
``compute_sparsity_pattern``, which (strategy ``"box"``) keeps a pair iff its
separation ALONG THE TRANSPORT DIRECTION is within the cutoff. Every stored
matrix lives on that pattern, ``G`` included, and a box-masked ``G`` is not
positive semidefinite -- measured directly in ``bubble_positivity.md``
Sec. 6.10c, where the violation is already there at iteration 0 with the
self-energy identically zero.

So the question "is this run contaminated by H6" is decided before any physics,
by geometry alone: the mask is inactive iff the cutoff exceeds the largest
transport-direction separation in the device.

This reads a run's config, builds the grid through the solver's own loader and
reports the answer with the fill fraction, rather than reasoning about lattice
constants.

Usage::

    python -m phonon.studies._cutoff_mask_audit cluster/*/quatrex_config.toml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Offline analysis -- it must never need a GPU, and it must give the same
# numbers on every machine. quatrex picks its array module at qttools import
# time from this variable, so it has to be set before the first
# quatrex/qttools import (all of which happen lazily inside the functions
# below). Under the default cupy backend the mask audit dies inside
# compute_sparsity_pattern, an xp routine handed a host grid.
os.environ.setdefault("QTX_ARRAY_MODULE", "numpy")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def audit(config_path: Path) -> dict:
    from quatrex.core.config import parse_config
    from quatrex.device import Device
    from quatrex.core.utils import compute_sparsity_pattern

    cfg = parse_config(config_path)
    # Configs carry the cluster's absolute paths. Re-point input_dir at the
    # local copy: the same-named directory under cluster/, else the config's
    # own directory.
    if not (Path(cfg.input_dir) / "structure.xyz").exists():
        for cand in (REPO / "cluster" / Path(cfg.input_dir).name,
                     config_path.resolve().parent):
            if (cand / "structure.xyz").exists():
                cfg.input_dir = cand
                break
    grid, _, _ = Device.load_structure(cfg)
    grid = np.asarray(grid)
    tdir = cfg.device.transport_direction
    axis = "xyz".index(tdir)
    z = grid[:, axis]
    span = float(z.max() - z.min())
    cutoff = float(cfg.phonon.interaction_cutoff)

    masked = compute_sparsity_pattern(grid, cutoff, transport_direction=tdir)
    dense = compute_sparsity_pattern(grid, span + 1.0, transport_direction=tdir)
    n_masked, n_dense = int(masked.nnz), int(dense.nnz)
    return {
        "config": config_path,
        "cells": int(cfg.device.num_transport_cells),
        "dof": int(grid.shape[0]),
        "tdir": tdir,
        "span": span,
        "cutoff": cutoff,
        "active": cutoff < span,
        "fill": n_masked / n_dense if n_dense else float("nan"),
        "dropped": n_dense - n_masked,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("configs", nargs="+", type=Path)
    p.add_argument("--quiet-clean", action="store_true",
                   help="list only the runs where the mask is active")
    args = p.parse_args(argv)

    print(f"{'run':<22} {'cells':>5} {'dof':>5} {'span [A]':>9} "
          f"{'cutoff':>8} {'fill':>7}  verdict")
    n_bad = 0
    for path in args.configs:
        try:
            r = audit(path)
        except Exception as exc:  # noqa: BLE001 - a bad config is a datum
            print(f"{path.parent.name:<22} {'--':>5} {'--':>5} "
                  f"{'--':>9} {'--':>8} {'--':>7}  ERROR {type(exc).__name__}: {exc}")
            continue
        if r["active"]:
            n_bad += 1
        elif args.quiet_clean:
            continue
        verdict = ("MASK ACTIVE -- H6" if r["active"]
                   else "dense (mask inactive)")
        print(f"{path.parent.name:<22} {r['cells']:>5} {r['dof']:>5} "
              f"{r['span']:>9.3f} {r['cutoff']:>8.1f} {r['fill']:>7.3f}  "
              f"{verdict}")
    print(f"\n{n_bad} of {len(args.configs)} runs have the box mask active.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
