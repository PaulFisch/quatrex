"""Summarise matched uniform/nonuniform production runs.

Run after pulling ``c16x2-nu``, ``c16x2-nu-safe`` and ``si-l8x2-nu``::

    python phonon/studies/_nonuniform_production_review.py \
      --json phonon/studies/out/nonuniform_production_review.json

Uniform historical runs store legacy unweighted frequency sums, whereas file
grids store cell-weighted integrals.  This script normalises the former by
``domega`` before comparing currents.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import tomllib

import numpy as np


REPO = Path(__file__).resolve().parents[2]


def physical_current(run: np.lib.npyio.NpzFile) -> float:
    value = float(run["lead_current"])
    uniform = bool(run["uniform_frequency_grid"])
    if uniform:
        w = np.asarray(run["energies"], float)
        value *= float(w[1] - w[0])
    return value


def _timing(path: Path) -> dict:
    files = sorted(path.glob("*_quatrex_times.out"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        return {}
    text = files[-1].read_text(errors="replace")
    labels = {
        "solver_seconds": "PhononSolver all",
        "ring_seconds": "PhPh SSE: 3 ring contraction all",
        "sse_seconds": "SigmaPhononPhonon all",
        "iteration_seconds": "SCBA: Iteration all",
    }
    out = {"file": str(files[-1])}
    for key, label in labels.items():
        values = [float(x) for x in re.findall(
            re.escape(label) + r" : ([0-9.]+)s", text)]
        if values:
            # Drop first-use compilation when there is more than one sample.
            live = values[1:] if len(values) > 1 else values
            out[key] = float(statistics.median(live))
            out[f"{key}_samples"] = len(live)
    return out


def _aux_grid(path: Path, primary: np.ndarray) -> np.ndarray:
    with (path / "quatrex_config.toml").open("rb") as f:
        cfg = tomllib.load(f)
    ph = cfg.get("phonon", {})
    dw = float(ph.get("sse_aux_grid_dw_thz", 0.0) or 0.0)
    if dw <= 0.0:
        return np.asarray(primary, float)
    top = max(float(primary[-1]),
              float(ph.get("sse_aux_grid_fmax_thz", 0.0) or 0.0))
    return np.arange(int(np.ceil(top / dw - 1e-9)) + 1, dtype=float) * dw


def _aux_points(path: Path, primary: np.ndarray) -> int:
    return int(_aux_grid(path, primary).size)


def moment_intertwining_defect(primary: np.ndarray,
                               auxiliary: np.ndarray) -> tuple[float, float]:
    """Measure ``Omega_a P - P Omega_p`` without forming dense ``P``.

    The energy-adjoint restriction pins one pairing, but exact collision-energy
    conservation additionally needs frequency multiplication to commute with
    reconstruction.  Linear interpolation has two nonzeros per auxiliary row,
    so the Frobenius defect is available in O(N_aux).
    """
    p = np.asarray(primary, float)
    a = np.asarray(auxiliary, float)
    if p.shape == a.shape and np.allclose(p, a, rtol=1e-12, atol=1e-14):
        return 0.0, 0.0
    hi = np.clip(np.searchsorted(p, a, side="left"), 1, p.size - 1)
    lo = hi - 1
    t = np.clip((a - p[lo]) / (p[hi] - p[lo]), 0.0, 1.0)
    c0 = (1.0 - t) * (a - p[lo])
    c1 = t * (a - p[hi])
    q0 = a * (1.0 - t)
    q1 = a * t
    row = np.sqrt(c0 * c0 + c1 * c1)
    den = np.sqrt(np.sum(q0 * q0 + q1 * q1))
    return float(np.linalg.norm(row) / max(float(den), 1e-300)), float(np.max(row))


def case(path: Path, config_path: Path | None = None) -> dict:
    run_path = path / "run.npz"
    if not run_path.exists():
        logs = sorted(path.glob("slurm-*.out"), key=lambda p: p.stat().st_mtime)
        grid_path = path / "phonon_energies.npy"
        if not logs or not grid_path.exists():
            raise FileNotFoundError(run_path)
        text = logs[-1].read_text(errors="replace")
        rows = re.findall(
            r"Phonon: rel Sigma\^R residual ([0-9.eE+-]+); "
            r"lead balance ([0-9.eE+-]+); internal spread ([0-9.eE+-]+); "
            r"lead current ([0-9.eE+-]+)", text)
        if not rows:
            raise ValueError(f"no completed SCBA iteration in {logs[-1]}")
        residual, balance, spread, current = map(float, rows[-1])
        w = np.asarray(np.load(grid_path), float)
        aux = _aux_grid(path if config_path is None else config_path, w)
        comm, comm_max = moment_intertwining_defect(w, aux)
        row = {
            "path": str(path), "log_only": True,
            "primary_points": int(w.size),
            "auxiliary_points": int(aux.size),
            "frequency_moment_intertwining_defect": comm,
            "frequency_moment_intertwining_max_thz": comm_max,
            "uniform_primary": bool(np.allclose(np.diff(w), np.diff(w)[0])),
            "physical_current": current,
            "internal_spread": spread, "lead_balance": balance,
            "last_sigma_residual": residual,
            "converged": False, "diverged": False,
            "iterations": len(rows), "timing": _timing(path),
            "source_log": str(logs[-1]),
        }
        return row

    run = np.load(run_path)
    w = np.asarray(run["energies"], float)
    aux = _aux_grid(path if config_path is None else config_path, w)
    comm, comm_max = moment_intertwining_defect(w, aux)
    row = {
        "path": str(path),
        "primary_points": int(w.size),
        "auxiliary_points": int(aux.size),
        "frequency_moment_intertwining_defect": comm,
        "frequency_moment_intertwining_max_thz": comm_max,
        "uniform_primary": bool(run["uniform_frequency_grid"]),
        "physical_current": physical_current(run),
        "internal_spread": float(run["internal_spread"]),
        "converged": bool(run["converged"]),
        "diverged": bool(run["diverged"]),
        "iterations": int(run["n_iter"]),
        "timing": _timing(path),
    }
    logs = sorted(path.glob("slurm-*.out"), key=lambda p: p.stat().st_mtime)
    if logs:
        text = logs[-1].read_text(errors="replace")
        peaks = re.findall(r"GPU mempool peak \(max over ranks\): ([0-9.]+) GB", text)
        if peaks:
            row["gpu_mempool_peak_gb"] = float(peaks[-1])
    return row


def compare(reference: dict, candidate: dict) -> dict:
    out = dict(candidate)
    ref_current = reference["physical_current"]
    out["current_relative_error"] = (
        candidate["physical_current"] - ref_current) / max(abs(ref_current), 1e-300)
    for key in ("solver_seconds", "ring_seconds", "sse_seconds",
                "iteration_seconds"):
        ref = reference.get("timing", {}).get(key)
        got = candidate.get("timing", {}).get(key)
        if ref is not None and got is not None:
            out[f"{key}_ratio"] = got / ref
    return out


def _relative_paths(row: dict, root: Path) -> dict:
    """Keep committed summaries independent of the local checkout path."""
    out = dict(row)
    for key in ("path", "source_log"):
        if key in out:
            try:
                out[key] = str(Path(out[key]).resolve().relative_to(root))
            except ValueError:
                pass
    if "timing" in out:
        out["timing"] = dict(out["timing"])
        if "file" in out["timing"]:
            try:
                out["timing"]["file"] = str(
                    Path(out["timing"]["file"]).resolve().relative_to(root))
            except ValueError:
                pass
    return out


def run(root: Path = REPO) -> dict:
    specs = {
        "cnt_uniform": (root / "cluster/c16x2h", None),
        "cnt_aggressive": (root / "cluster/c16x2-nu", None),
        "cnt_safe": (root / "cluster/c16x2-nu-safe", None),
        # The final Si continuation intentionally wrote into a separate run
        # directory while retaining the prepared sifilm8x2 input config.
        "si_uniform": (root / "cluster/si-l8x2-final",
                       root / "cluster/sifilm8x2"),
        "si_nonuniform": (root / "cluster/si-l8x2-nu", None),
    }
    missing = [str(p) for p, _ in specs.values()
               if not (p / "run.npz").exists() and not list(p.glob("slurm-*.out"))]
    if missing:
        raise FileNotFoundError("pull/complete production runs first: " + ", ".join(missing))
    rows = {name: case(path, cfg) for name, (path, cfg) in specs.items()}
    cnt_ref = rows["cnt_uniform"]
    si_ref = rows["si_uniform"]
    result = {
        "cnt_uniform": cnt_ref,
        "cnt_aggressive": compare(cnt_ref, rows["cnt_aggressive"]),
        "cnt_safe": compare(cnt_ref, rows["cnt_safe"]),
        "si_uniform": si_ref,
        "si_nonuniform": compare(si_ref, rows["si_nonuniform"]),
    }
    return {name: _relative_paths(row, root) for name, row in result.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=REPO)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    result = run(args.root.resolve())
    for name, row in result.items():
        print(name, json.dumps(row, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
