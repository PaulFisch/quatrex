"""Adaptive-P1 convolution against a 15,001-point production Si oracle.

The saved run contains diagonal ``G^R`` and ``G^<`` for every transverse q
and device DOF.  This study constructs one *shared* dyadic mesh from all of
those spectra, plus a shared output mesh from the convolution of their
normalised envelope.  It then compares exact projected convolution on that
mesh with fine uniform FFT oracles for representative real Si channels.

Run::

    PYTHONPATH=phonon python phonon/studies/_si_adaptive_fct_oracle.py \
      --run cluster/sichk_res/run.npz \
      --json phonon/studies/out/si_adaptive_fct_oracle.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import fftconvolve

from studies import _adaptive_p1_fct as F
from studies._adaptive_p1_fct_review import adaptive_mesh, p1_l2_error


class UniformOracle:
    """Vector-valued linear interpolant on one uniform axis."""

    def __init__(self, axis: np.ndarray, values: np.ndarray):
        self.axis = np.asarray(axis, float)
        self.values = np.asarray(values)
        self.step = float(self.axis[1] - self.axis[0])
        if not np.allclose(np.diff(self.axis), self.step, rtol=2e-10,
                           atol=2e-13):
            raise ValueError("oracle axis must be uniform")
        if self.values.shape[0] != self.axis.size:
            raise ValueError("oracle values must match its axis")

    def __call__(self, points):
        x = np.asarray(points, float)
        flat = x.reshape(-1)
        pos = (flat - self.axis[0]) / self.step
        lo = np.floor(pos).astype(int)
        weight = pos - lo
        lo = np.clip(lo, 0, self.axis.size - 2)
        shape = (flat.size,) + (1,) * (self.values.ndim - 1)
        out = ((1.0 - weight).reshape(shape) * self.values[lo]
               + weight.reshape(shape) * self.values[lo + 1])
        outside = (flat < self.axis[0]) | (flat > self.axis[-1])
        if np.any(outside):
            out[outside] = 0.0
        return out.reshape(x.shape + self.values.shape[1:])


def certified_mesh(oracle: UniformOracle, base_h: float, first: int, stop: int,
                   tolerance: float, max_level: int) -> F.DyadicMesh:
    """Dyadic mesh whose P1 defect is checked at every stored oracle point.

    This is an offline reference for a pole-informed detector.  It deliberately
    avoids the five-point indicator's aliasing failure when a very narrow line
    falls between all probes of a coarse cell.
    """
    scale = max(float(np.max(np.abs(oracle.values))), 1e-300)

    def target(left: float, right: float) -> int:
        level = int(round(np.log2(base_h / (right - left))))
        lo = int(np.searchsorted(oracle.axis, left, side="left"))
        hi = int(np.searchsorted(oracle.axis, right, side="right"))
        x = oracle.axis[lo:hi]
        if x.size:
            yl, yr = oracle(np.array([left, right]))
            shape = (x.size,) + (1,) * (oracle.values.ndim - 1)
            t = ((x - left) / (right - left)).reshape(shape)
            linear = (1.0 - t) * yl + t * yr
            defect = float(np.max(np.abs(oracle.values[lo:hi] - linear)))
        else:
            defect = 0.0
        return (level + 1 if defect > tolerance * scale
                and level < max_level else level)

    return F.DyadicMesh.refined(base_h, first, stop, target)


def _full_keldysh(run: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    z = np.load(run)
    omega = np.asarray(z["energies"], float)
    gl = np.asarray(z["gl_diag_imag"], float).reshape(omega.size, -1)
    spectral = 2.0 * np.asarray(z["gr_diag_imag"], float).reshape(
        omega.size, -1)
    # Quatrex's stored occupation-positive convention has Im G^> = Im G^<+A
    # on a diagonal block, with A=-2 Im G^R.
    gg = gl + spectral
    # Do not duplicate omega=0.  Lesser continuation at negative frequency is
    # greater at positive frequency (diagonal: transpose is the identity).
    full_w = np.concatenate((-omega[:0:-1], omega))
    full_l = np.concatenate((gg[:0:-1], gl), axis=0)
    meta = {
        "n_q": int(np.prod(z["gr_diag_imag"].shape[1:-1])),
        "n_dof": int(z["gr_diag_imag"].shape[-1]),
        "source_iterations": int(z["n_iter"]),
        "source_converged": bool(z["converged"]),
        "source_diverged": bool(z["diverged"]),
        "source_eta": float(z["eta"]),
    }
    return full_w, full_l, meta


def _representatives(values: np.ndarray, step: float) -> dict[str, int]:
    scale = np.max(np.abs(values), axis=0)
    norm = values / np.maximum(scale, 1e-300)
    curvature = np.max(np.abs(np.diff(norm, n=2, axis=0)), axis=0)
    weight = np.sum(np.abs(values), axis=0) * step
    active = np.flatnonzero(scale > 1e-12 * np.max(scale))
    typical = active[np.argsort(weight[active])[len(active) // 2]]
    return {
        "sharpest": int(np.argmax(curvature)),
        "largest_weight": int(np.argmax(weight)),
        "typical_weight": int(typical),
    }


def run_study(run_path: Path, tolerances: tuple[float, ...],
              base_h: float, max_level: int,
              normalisation: str = "global",
              mesh_only: bool = False,
              channels: tuple[str, ...] | None = None,
              input_scope: str = "shared",
              detector: str = "oracle") -> dict:
    full_w, full_l, meta = _full_keldysh(run_path)
    fine_h = float(full_w[1] - full_w[0])
    wmax = float(full_w[-1])
    if not np.isclose(wmax / base_h, round(wmax / base_h), atol=1e-10):
        raise ValueError("base_h must tile the Si frequency interval")
    first = -int(round(wmax / base_h))
    stop = -first

    scales = np.max(np.abs(full_l), axis=0)
    active = scales > 1e-12 * np.max(scales)
    if normalisation == "per-channel":
        normalised = full_l[:, active] / np.maximum(scales[active], 1e-300)
    elif normalisation == "global":
        normalised = full_l[:, active] / max(float(np.max(scales)), 1e-300)
    else:
        raise ValueError("normalisation must be global or per-channel")
    shared_oracle = UniformOracle(full_w, normalised)
    envelope = np.max(np.abs(normalised), axis=1)
    envelope_bubble = fftconvolve(envelope, envelope, mode="full") * fine_h
    output_w = np.linspace(-2.0 * wmax, 2.0 * wmax,
                           envelope_bubble.size)
    output_oracle = UniformOracle(output_w, envelope_bubble)
    reps = _representatives(full_l, fine_h)
    if channels is not None:
        unknown = set(channels) - set(reps)
        if unknown:
            raise ValueError(f"unknown representative channels: {unknown}")
        reps = {name: reps[name] for name in channels}

    rows = []
    mesh_builder = certified_mesh if detector == "oracle" else adaptive_mesh
    for tolerance in tolerances:
        t_mesh = time.perf_counter()
        input_mesh = mesh_builder(
            shared_oracle, base_h, first, stop, tolerance, max_level)
        output_mesh = mesh_builder(
            output_oracle, base_h, 2 * first, 2 * stop,
            tolerance, max_level)
        mesh_seconds = time.perf_counter() - t_mesh
        print(
            f"tol={tolerance:g} shared Si mesh: "
            f"input={len(input_mesh.leaves)} output={len(output_mesh.leaves)} "
            f"levels={max(input_mesh.levels)}/{max(output_mesh.levels)} "
            f"build={mesh_seconds:.2f}s", flush=True)
        local_meshes = {}
        for label, channel in reps.items():
            channel_scale = max(float(scales[channel]), 1e-300)
            channel_oracle = UniformOracle(
                full_w, full_l[:, channel] / channel_scale)
            local = mesh_builder(
                channel_oracle, base_h, first, stop, tolerance, max_level)
            local_meshes[label] = local
        if mesh_only:
            rows.append({
                "tolerance": tolerance,
                "channel": "mesh-only",
                "input_cells": len(input_mesh.leaves),
                "input_vertices": input_mesh.vertices.size,
                "output_cells": len(output_mesh.leaves),
                "output_vertices": output_mesh.vertices.size,
                "max_input_level": max(input_mesh.levels),
                "max_output_level": max(output_mesh.levels),
                "mesh_seconds": mesh_seconds,
                "representative_local_input_cells": {
                    name: len(mesh.leaves) for name, mesh in local_meshes.items()},
            })
            continue
        for label, channel in reps.items():
            oracle = UniformOracle(full_w, full_l[:, channel])
            exact_bubble = fftconvolve(
                full_l[:, channel], full_l[:, channel], mode="full") * fine_h
            bubble_oracle = UniformOracle(output_w, exact_bubble)
            field_mesh = (input_mesh if input_scope == "shared"
                          else local_meshes[label])
            field = F.P1Field.from_callable(field_mesh, oracle)
            ring = {"calls": 0, "modes": 0, "max_modes": 0}

            def counted(a, b):
                ring["calls"] += 1
                ring["modes"] += int(a.shape[0])
                ring["max_modes"] = max(ring["max_modes"], int(a.shape[0]))
                return a * b

            t0 = time.perf_counter()
            bubble = F.projected_convolution_combined(
                field, field, output_mesh, counted)
            elapsed = time.perf_counter() - t0
            continuous = F.project_continuous(bubble)
            sample = F.evaluate_continuous(continuous, output_w)
            denom = np.linalg.norm(exact_bubble)
            rows.append({
                "tolerance": tolerance,
                "channel": label,
                "channel_index": channel,
                "input_cells": len(input_mesh.leaves),
                "input_vertices": input_mesh.vertices.size,
                "output_cells": len(output_mesh.leaves),
                "output_vertices": output_mesh.vertices.size,
                "max_input_level": max(input_mesh.levels),
                "max_output_level": max(output_mesh.levels),
                "equivalent_uniform_positive_points": int(
                    round(wmax / (base_h / 2**max(input_mesh.levels))) + 1),
                "mesh_seconds": mesh_seconds,
                "convolution_seconds": elapsed,
                "ring_calls": ring["calls"],
                "ring_modes": ring["modes"],
                "max_ring_modes": ring["max_modes"],
                "local_input_cells": len(local_meshes[label].leaves),
                "input_relative_l2": p1_l2_error(
                    field, oracle, (-wmax, wmax)),
                "bubble_projected_relative_l2": p1_l2_error(
                    bubble, bubble_oracle, (-2 * wmax, 2 * wmax)),
                "bubble_continuous_fine_l2": float(
                    np.linalg.norm(sample - exact_bubble)
                    / max(denom, 1e-300)),
            })
        print(tolerance, rows[-len(reps):], flush=True)
    return {
        "source": str(run_path),
        "fine_points_positive": int((full_w.size + 1) // 2),
        "fine_spacing_thz": fine_h,
        "base_h_thz": base_h,
        "max_level": max_level,
        "normalisation": normalisation,
        "input_scope": input_scope,
        "detector": detector,
        "active_channels": int(np.count_nonzero(active)),
        **meta,
        "rows": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--tolerances", default="0.02,0.005")
    ap.add_argument("--base-h", type=float, default=0.125)
    ap.add_argument("--max-level", type=int, default=9)
    ap.add_argument("--normalisation", choices=("global", "per-channel"),
                    default="global")
    ap.add_argument("--mesh-only", action="store_true")
    ap.add_argument("--channels", default="sharpest,largest_weight,typical_weight")
    ap.add_argument("--input-scope", choices=("shared", "local"),
                    default="shared")
    ap.add_argument("--detector", choices=("oracle", "five-point"),
                    default="oracle")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    result = run_study(
        args.run, tuple(float(x) for x in args.tolerances.split(",")),
        args.base_h, args.max_level, args.normalisation, args.mesh_only,
        tuple(x for x in args.channels.split(",") if x), args.input_scope,
        args.detector)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
