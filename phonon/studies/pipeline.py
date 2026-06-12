"""Solver-cell pipeline: geometry build, config write, hygienic launch, npz IO.

Wraps the engine scripts in :mod:`phonon.studies.engine` (``build_inputs.py``,
``write_config.py``, ``run.py``, ``launch.sh``) with the node rules that every
investigation must obey:

- single-threaded BLAS, parallelism via MPI ranks and/or the ring thread pool;
- no launch while other solver processes are alive (``assert_node_idle``);
- one full npz snapshot per run, never re-run hours to replot.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENGINE = Path(__file__).resolve().parent / "engine"
OUT = Path(__file__).resolve().parent / "out"
GEOM = ROOT / "phonon/scripts/out/prod/geom"  # shared geometry inputs


def assert_node_idle(allow: int = 0) -> None:
    """Refuse to launch while solver processes are alive on the node."""
    res = subprocess.run(
        ["pgrep", "-u", str(os.getuid()), "-f", r"engine/run\.py|prterun|mpirun"],
        capture_output=True, text=True,
    )
    pids = [p for p in res.stdout.split() if p]
    if len(pids) > allow:
        raise RuntimeError(
            f"node not idle: {len(pids)} solver processes alive ({pids[:8]}); "
            "kill or wait before launching (see phonon/CLAUDE.md hygiene)."
        )


def build_geometry(system: str, work: Path, **kwargs) -> None:
    """Run the geometry/FC3 builder for ``system`` into ``work``."""
    args = [sys.executable, str(ENGINE / "build_inputs.py"),
            "--system", system, "--out", str(work)]
    for key, val in kwargs.items():
        args += [f"--{key.replace('_', '-')}", str(val)]
    subprocess.run(args, check=True)


def write_config(system: str, work: Path, **kwargs) -> Path:
    """Write the cell's quatrex_config.toml; returns its path."""
    args = [sys.executable, str(ENGINE / "write_config.py"),
            "--system", system, "--work", str(work)]
    for key, val in kwargs.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(val, bool):
            if val:
                args.append(flag)
        else:
            args += [flag, str(val)]
    subprocess.run(args, check=True)
    return Path(work) / "quatrex_config.toml"


def launch_cell(config: Path, npz: Path, log: Path, *, nranks: int = 1,
                ring_threads: int = 1, ballistic: bool = False,
                env: dict | None = None, check_idle: bool = True,
                timeout: float | None = None) -> int:
    """Run one solver cell (blocking); returns the exit code.

    ``env`` may carry extra ``QX_*`` overrides (QX_MIX, QX_MAXIT, ...).
    """
    if check_idle:
        assert_node_idle()
    run_env = os.environ.copy()
    run_env.update({
        "QX_CONFIG": str(config),
        "QX_NPZ": str(npz),
        "NRANKS": str(nranks),
        "QX_RING_THREADS": str(ring_threads),
    })
    if ballistic:
        run_env["QX_BALLISTIC"] = "1"
    if env:
        run_env.update({k: str(v) for k, v in env.items()})
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        proc = subprocess.run(["bash", str(ENGINE / "launch.sh")],
                              stdout=fh, stderr=subprocess.STDOUT,
                              env=run_env, timeout=timeout)
    return proc.returncode


def load_run(npz: Path) -> dict:
    """Load a run snapshot as a plain dict of arrays."""
    with np.load(npz, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def lead_heat(npz: Path) -> dict:
    """q-summed lead heat current (lead0/lead1/mean) of a snapshot.

    Prefers the SCBA's all-reduced ``last_heat`` (exact at any stack split);
    falls back to rank-0-local ``final_heat`` (complete only at stack=1,
    e.g. ballistic runs that exit before tracking last_heat).
    """
    data = load_run(npz)
    heat = data.get("last_heat")
    if heat is None or not np.isfinite(np.asarray(heat)).all():
        heat = data.get("final_heat")
    js = np.asarray(heat).reshape(-1)
    return {"lead0": float(abs(js[0])), "lead1": float(abs(js[-1])),
            "mean": float(0.5 * (abs(js[0]) + abs(js[-1])))}


def parse_scba_trace(log: Path) -> dict:
    """Per-iteration traces from a run log: Sigma residual, lead balance,
    bubble energy balance."""
    import re
    pat_res = re.compile(
        r"rel Sigma\^R residual ([0-9.e+-]+); lead balance ([0-9.e+-]+)")
    pat_bal = re.compile(r"Bubble energy balance: .*resid=([0-9.e+-]+)")
    residual, lead, bubble = [], [], []
    for line in Path(log).read_text(errors="ignore").splitlines():
        m = pat_res.search(line)
        if m:
            residual.append(float(m.group(1)))
            lead.append(float(m.group(2)))
        m = pat_bal.search(line)
        if m:
            bubble.append(float(m.group(1)))
    return {"residual": np.array(residual), "lead_balance": np.array(lead),
            "bubble_balance": np.array(bubble)}
