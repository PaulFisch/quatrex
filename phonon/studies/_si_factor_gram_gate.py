#!/usr/bin/env python3
"""Reduced dense-reconstruction gate for the Si factor Gram kernel.

The production factor files use a 9x9 mesh and a five-offset factor table,
while the physical vertex occupies only seven coupled offset pairs.  This
script restricts that mesh to its closed 3x3 subgroup and compares the Gram
collapse with a dense ring reconstructed from the same factors.  It isolates
kernel algebra and numerical conditioning from the quality of the FC3 fit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from qttools import xp
from qttools.utils.gpu_utils import get_host
from quatrex.phonon.bubble_factored import contract_tau_q_factored
from quatrex.phonon.vertex_factors import VertexFactors, load_decomposed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from phonon.studies._bench_factored_sse import contract_dense


def _subgroup(vf: VertexFactors, stride: int = 3) -> VertexFactors:
    if tuple(vf.nk_shape) != (9, 9) or 9 % stride:
        raise ValueError("the reduced gate expects the production 9x9 mesh")
    side = 9 // stride
    indices = np.array([
        ix * stride * 9 + iy * stride
        for ix in range(side) for iy in range(side)
    ])
    qdm = np.array([
        [
            ((a // side - b // side) % side) * side
            + (a % side - b % side) % side
            for b in range(side * side)
        ]
        for a in range(side * side)
    ])
    return VertexFactors(
        D=vf.D,
        lambdas=vf.lambdas,
        offsets=vf.offsets,
        UB=vf.UB[:, indices],
        UC=vf.UC[:, indices],
        q_diff_map=qdm,
        nk_shape=(side, side),
        ansatz=vf.ansatz,
        meta=vf.meta,
    )


def _quads(vf: VertexFactors, ncell: int, band: int, pairs):
    support = {
        tuple(int(x) for x in pair)
        for pair in vf.meta.get("support_pairs", [])
    }
    result = {}
    for i, j in pairs:
        entries = []
        for d1, d2 in support:
            k1, k2 = i + d1, i + d2
            if not (0 <= k1 < ncell and 0 <= k2 < ncell):
                continue
            for r1, r2 in support:
                # The right vertex is consumed as Phi[j, k2p, k1p].
                k2p, k1p = j + r1, j + r2
                if not (0 <= k1p < ncell and 0 <= k2p < ncell):
                    continue
                if abs(k1 - k1p) <= band and abs(k2 - k2p) <= band:
                    entries.append((k1, k2, k1p, k2p))
        result[(i, j)] = entries
    return result


def run(path: Path, seed: int = 20260829) -> dict[str, float]:
    vf = _subgroup(load_decomposed(path))
    nq = vf.n_kpts
    ncell, band, d, nt = 5, 3, vf.D.shape[0], 1
    pairs = ((0, 0), (2, 2), (0, 4), (2, 3))
    quads = _quads(vf, ncell, band, pairs)
    links = {
        link
        for entries in quads.values()
        for q in entries
        for link in ((q[0], q[2]), (q[1], q[3]))
    }
    rng = np.random.default_rng(seed)
    variants = {}
    for name in ("l", "g", "lr", "gr"):
        variants[name] = {
            link: (
                rng.standard_normal((nt, nq, d, d))
                + 1j * rng.standard_normal((nt, nq, d, d))
            )
            for link in links
        }

    dense, _ = contract_dense(
        quads, vf, vf.q_diff_map, nq, nq, d, nt, variants
    )
    device_variants = {
        name: {link: xp.asarray(value) for link, value in family.items()}
        for name, family in variants.items()
    }
    gram = contract_tau_q_factored(
        quads,
        np.full(ncell, d),
        vf.nk_shape,
        0,
        nq,
        nq,
        device_variants,
        xp.asarray(vf.D * vf.lambdas[None, :]),
        xp.asarray(vf.UB),
        xp.asarray(vf.UC),
        vf.offset_index(),
        0,
        nt,
        xp,
        bool(np.array_equal(vf.UB, vf.UC)),
        np.complex128,
    )
    errors = {}
    for pair in pairs:
        for side, label in ((0, "lesser"), (1, "greater")):
            reference = dense[pair][side]
            candidate = np.asarray(get_host(gram[pair][side]))
            errors[f"{pair}_{label}"] = float(
                np.linalg.norm(candidate - reference)
                / max(np.linalg.norm(reference), np.finfo(float).tiny)
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factors", type=Path)
    args = parser.parse_args()
    errors = run(args.factors)
    for name, value in errors.items():
        print(f"{name}: {value:.12e}")
    print(f"maximum: {max(errors.values()):.12e}")


if __name__ == "__main__":
    main()
