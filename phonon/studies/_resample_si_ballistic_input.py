#!/usr/bin/env python3
"""Build a denser-q Si ballistic input from an existing odd square mesh.

The harmonic matrix in the source input is already stored in transverse
real space, so changing the configured q mesh evaluates the same FC2 Fourier
polynomial.  The SCBA study driver still constructs its (subsequently removed)
phonon-phonon interaction in ``QX_BALLISTIC`` mode.  This utility therefore
resamples the fixed tensor factors by exact trigonometric interpolation as a
lightweight, mesh-compatible placeholder.  No factor is used by the harmonic
solve.

For the 5x5x5 Si force-constant supercell the contracted-leg factors have
transverse support only at translations -2,...,2.  An odd source mesh of at
least five points consequently determines their Fourier polynomial exactly.
The utility reports any Fourier weight outside that support and refuses an
under-resolved source.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil

import numpy as np

from quatrex.phonon.vertex_q_resample import (
    q_difference_map,
    trigonometric_resample,
)


def build_input(source: Path, output: Path, target_nk: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "dynamical_matrix.mat", "fc3_blocks.hdf5", "structure.xyz",
        "phonon_energies.npy",
    ):
        shutil.copy2(source / name, output / name)

    factors = np.load(source / "decomposed_vertices.npz", allow_pickle=True)
    ub, audit_b = trigonometric_resample(factors["UB"], target_nk)
    uc, audit_c = trigonometric_resample(factors["UC"], target_nk)
    meta = dict(factors["meta"].item())
    meta["ballistic_q_resample"] = {"UB": audit_b, "UC": audit_c}
    np.savez_compressed(
        output / "decomposed_vertices.npz",
        format_version=factors["format_version"],
        D=factors["D"], lambdas=factors["lambdas"],
        offsets=factors["offsets"], UB=ub, UC=uc,
        q_diff_map=q_difference_map(target_nk),
        nk_shape=np.asarray([target_nk, target_nk], dtype=np.int64),
        ansatz=factors["ansatz"], meta=np.array(meta, dtype=object),
    )
    shift = 0.5 - 0.5 / target_nk
    np.save(output / "kshift.npy", np.asarray(shift))

    config = (source / "quatrex_config.toml").read_text()
    config = config.replace(source.name, output.name)
    config = re.sub(
        r"kpoint_grid\s*=\s*\[[^\n]+\]",
        f"kpoint_grid = [1, {target_nk}, {target_nk}]", config)
    config = re.sub(
        r"kpoint_shift\s*=\s*\[[^\n]+\]",
        f"kpoint_shift = [0.0, {shift:.10f}, {shift:.10f}]", config)
    (output / "quatrex_config.toml").write_text(config)
    return {"UB": audit_b, "UC": audit_c}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nk", required=True, type=int)
    args = parser.parse_args()
    audit = build_input(args.source, args.output, args.nk)
    for leg, values in audit.items():
        print(
            f"{leg}: source roundtrip "
            f"{values['relative_source_roundtrip']:.3e}, Fourier tail "
            f"{values['relative_fourier_tail']:.3e}")
        if values["relative_source_roundtrip"] > 1e-12:
            raise SystemExit("trigonometric interpolation roundtrip failed")
        if values["relative_fourier_tail"] > 1e-10:
            raise SystemExit("source mesh contains material out-of-support weight")
    print(f"ballistic q={args.nk} input -> {args.output}")


if __name__ == "__main__":
    main()
