"""Accuracy/cost study for the adaptive P1 fast convolution transform.

Run::

    PYTHONPATH=phonon python phonon/studies/_adaptive_p1_fct_review.py \
      --json phonon/studies/out/adaptive_p1_fct_review.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from studies import _adaptive_p1_fct as F
from studies._nonuniform_grid_review import LorentzianMixture


def adaptive_mesh(fn, base_h: float, first: int, stop: int,
                  tolerance: float, max_level: int = 16) -> F.DyadicMesh:
    """Bisect until midpoint P1 defect is below a global scaled tolerance."""
    scale_probe = np.linspace(first * base_h, stop * base_h, 4097)
    scale = max(float(np.max(np.abs(fn(scale_probe)))), 1e-300)

    def target(left: float, right: float) -> int:
        level = int(round(np.log2(base_h / (right - left))))
        mid = 0.5 * (left + right)
        q1, q3 = 0.75 * left + 0.25 * right, 0.25 * left + 0.75 * right
        vals = np.asarray(fn(np.array([left, q1, mid, q3, right])))
        linear = np.array([
            vals[0], 0.75 * vals[0] + 0.25 * vals[4],
            0.5 * (vals[0] + vals[4]),
            0.25 * vals[0] + 0.75 * vals[4], vals[4]])
        defect = float(np.max(np.abs(vals - linear)))
        return level + 1 if defect > tolerance * scale and level < max_level else level

    return F.DyadicMesh.refined(base_h, first, stop, target)


def p1_l2_error(field: F.P1Field, exact, interval: tuple[float, float]) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(12)
    numerator = denominator = 0.0
    for level, index in field.mesh.leaves:
        h = field.mesh.base_h / 2**level
        left = max(index * h, interval[0])
        right = min((index + 1) * h, interval[1])
        if right <= left:
            continue
        x = 0.5 * ((right - left) * nodes + right + left)
        coeff = field.levels[level].sample(np.array([index]))[0]
        xi = (x - (index + 0.5) * h) / h
        tail = coeff.shape[1:]
        got = (coeff[0] / np.sqrt(h)
               + np.sqrt(12.0 / h) * xi.reshape(
                   xi.shape + (1,) * len(tail)) * coeff[1])
        want = exact(x)
        numerator += 0.5 * (right - left) * float(
            np.sum(weights * np.abs(got - want)**2))
        denominator += 0.5 * (right - left) * float(
            np.sum(weights * np.abs(want)**2))
    return float(np.sqrt(numerator / max(denominator, 1e-300)))


def run(tolerance: float = 2e-3) -> dict:
    base_h = 0.25
    fmax = 4.0
    rows = []
    for ratio in (1.0, 0.2, 0.04, 0.008, 0.001):
        gamma = ratio * base_h
        worst = None
        for offset in (0.0, 0.25, 0.49):
            centre = 1.0 + offset * base_h
            model = LorentzianMixture(
                centres=np.array([centre - 1.4 * gamma,
                                  centre + 1.4 * gamma]),
                widths=np.array([gamma, 1.15 * gamma]),
                amplitudes=np.array([1.0, 0.63]))
            input_mesh = adaptive_mesh(
                model.eval, base_h, -16, 16, tolerance)
            output_mesh = adaptive_mesh(
                model.bubble, base_h, -32, 32, tolerance)
            field = F.P1Field.from_callable(input_mesh, model.eval)
            ring = {"calls": 0, "modes": 0, "max_modes": 0}

            def counted_product(a, b):
                ring["calls"] += 1
                ring["modes"] += int(a.shape[0])
                ring["max_modes"] = max(ring["max_modes"], int(a.shape[0]))
                return a * b

            t0 = time.perf_counter()
            bubble = F.projected_convolution_combined(
                field, field, output_mesh, counted_product)
            elapsed = time.perf_counter() - t0
            row = {
                "gamma_over_background_h": ratio,
                "offset_over_background_h": offset,
                "input_cells": len(input_mesh.leaves),
                "output_cells": len(output_mesh.leaves),
                "max_input_level": max(input_mesh.levels),
                "max_output_level": max(output_mesh.levels),
                "equivalent_uniform_input_cells": 32 * 2**max(input_mesh.levels),
                "elapsed_seconds": elapsed,
                "ring_calls": ring["calls"],
                "ring_modes": ring["modes"],
                "max_ring_modes": ring["max_modes"],
                "bubble_relative_l2": p1_l2_error(
                    bubble, model.bubble, (-3.5, 3.5)),
            }
            rows.append(row)
            if worst is None or row["bubble_relative_l2"] > worst[
                    "bubble_relative_l2"]:
                worst = row
        print(ratio, json.dumps(worst, sort_keys=True), flush=True)
    return {"tolerance": tolerance, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tolerance", type=float, default=2e-3)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    result = run(args.tolerance)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
