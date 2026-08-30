#!/usr/bin/env python3
"""Project a production CP vertex onto exact contracted-leg symmetry.

The input remains a normal ``VertexFactors`` archive.  Every CP component is
paired with its contracted-leg transpose, so a base rank R becomes a final
rank 2R without reconstructing any dense q-folded vertex.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from quatrex.phonon.vertex_factors import (  # noqa: E402
    VertexFactors,
    load_decomposed,
    save_decomposed,
)


def pair_cp_vertex(vertex: VertexFactors) -> VertexFactors:
    """Return the exact S2 projection of a CP ``VertexFactors`` archive."""
    if vertex.ansatz != "CP":
        raise ValueError(f"expected a CP archive, got {vertex.ansatz!r}")
    base_rank = vertex.rank
    base_error = float(vertex.meta.get("rel_err", np.nan))
    base_s2 = float(vertex.meta.get("s2_recon", np.nan))
    paired_error = np.nan
    if np.isfinite(base_error) and np.isfinite(base_s2):
        paired_error = float(
            np.sqrt(max(base_error * base_error - 0.25 * base_s2 * base_s2, 0.0))
        )
    stale_qfold = {
        key: value for key, value in vertex.meta.items()
        if key.startswith("qfold_sample_")
    }
    meta = {
        **{key: value for key, value in vertex.meta.items()
           if not key.startswith("qfold_sample_")},
        "method": "S2CP",
        "rank": 2 * base_rank,
        "paired_base_rank": base_rank,
        "base_cp_rel_err": base_error,
        "base_cp_s2_recon": base_s2,
        "rel_err": paired_error,
        "s2_recon": 0.0,
        "paired_cp_exact_projection": True,
        "base_qfold_sample_diagnostics": stale_qfold,
    }
    return VertexFactors(
        D=np.concatenate((vertex.D, vertex.D), axis=1),
        lambdas=0.5 * np.concatenate((vertex.lambdas, vertex.lambdas)),
        offsets=vertex.offsets.copy(),
        UB=np.concatenate((vertex.UB, vertex.UC), axis=-1),
        UC=np.concatenate((vertex.UC, vertex.UB), axis=-1),
        q_diff_map=vertex.q_diff_map.copy(),
        nk_shape=tuple(vertex.nk_shape),
        ansatz="S2CP",
        meta=meta,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paired = pair_cp_vertex(load_decomposed(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_decomposed(args.output, paired)
    print(
        f"{args.input}: CP rank {paired.meta['paired_base_rank']} -> "
        f"{args.output}: S2CP rank {paired.rank}, "
        f"estimated rel_err={paired.meta['rel_err']:.6g}"
    )


if __name__ == "__main__":
    main()
