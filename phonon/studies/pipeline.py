"""Solver-cell pipeline: geometry build, config write, hygienic launch, npz IO.

Wraps the engine scripts in :mod:`phonon.studies.engine` (``build_inputs.py``,
``write_config.py``, ``run.py``, ``launch.sh``) with the node rules that every
investigation must obey:

- single-threaded BLAS, parallelism via MPI ranks and/or the ring thread pool;
- no launch while other solver processes are alive (``assert_node_idle``);
- one full npz snapshot per run, never re-run hours to replot.
"""

import os
import re
import subprocess
import time
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


def node_ring_threads(nranks: int = 1, cap: int = 64) -> int:
    """Ring-pool width for a single blocking cell that should fill the node.

    The 3-phonon bubble (~99% of a step) is genuine BLAS work, not Python
    overhead: a fully gemm-batched rewrite is bit-identical and the SAME speed
    single-core (~22 GF/s), and is ~10x SLOWER under the pool (it materialises
    all rings' BS^3 intermediates at once and thrashes cache). The per-ring loop
    instead keeps each chunk's intermediate in cache and scales near-linearly
    with the omega/tau thread pool up to a measured cnt33 sweet spot of ~64
    (w=241: 56x / 1232 GF/s; 128 regresses from tau-fragmentation, also guarded
    by the SSE n_tau//4 floor). So default to ~all cores per rank, capped at
    ``cap``. For SWEEPS use :func:`launch_cells_concurrent` (many cells x
    moderate threads, cells*ring_threads <= core_budget) to fill all 256 cores.
    """
    cores = os.cpu_count() or 8
    return max(1, min(cap, cores // max(1, nranks)))


def numa_sockets() -> list[tuple[int | None, int]]:
    """``[(numa_node_id, n_physical_cores), ...]`` for socket-pinned concurrency.

    A single solver cell is hard-capped at ~one NUMA socket: the per-ring bubble
    is cache-bound (tau-chunking is mandatory; ring-/quad-parallel thrash cache
    and are 10-20x slower) and a single process tops out at ~one socket's worth
    (~1.4 TF/s) because of the GIL + cross-socket NUMA. So the full-node lever is
    NOT more threads in one cell but running ONE cell per socket, each pinned
    NUMA-local. This returns the sockets to pin to. Falls back to a single
    pseudo-socket ``[(None, os.cpu_count())]`` if detection fails (then no
    pinning is applied). Physical cores/socket excludes SMT siblings.
    """
    try:
        info = {}
        for ln in subprocess.run(
                ["lscpu"], capture_output=True, text=True).stdout.splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                info[k.strip()] = v.strip()
        nsock = int(info.get("Socket(s)", "1"))
        cps = int(info.get("Core(s) per socket", str(os.cpu_count() or 1)))
        nodes = list(range(nsock))
        ids = re.findall(
            r"node (\d+) cpus:",
            subprocess.run(["numactl", "--hardware"],
                           capture_output=True, text=True).stdout)
        if len(ids) == nsock:
            nodes = [int(x) for x in ids]
        if nsock >= 1 and cps >= 1:
            return [(n, cps) for n in nodes]
    except Exception:
        pass
    return [(None, os.cpu_count() or 8)]


def _numactl_prefix(node: int | None) -> list[str]:
    """``numactl`` argv prefix pinning CPUs + memory to one NUMA node (empty if
    ``node is None`` or numactl is unavailable)."""
    if node is None:
        return []
    from shutil import which
    if which("numactl") is None:
        return []
    return ["numactl", f"--cpunodebind={node}", f"--membind={node}"]


def launch_cell(config: Path, npz: Path, log: Path, *, nranks: int = 1,
                ring_threads: int | None = None, ballistic: bool = False,
                env: dict | None = None, check_idle: bool = True,
                timeout: float | None = None) -> int:
    """Run one solver cell (blocking); returns the exit code.

    ``env`` may carry extra ``QX_*`` overrides (QX_MIX, QX_MAXIT, ...).
    ``ring_threads=None`` (default) fills the node via :func:`node_ring_threads`.
    """
    if ring_threads is None:
        ring_threads = node_ring_threads(nranks)
    if check_idle:
        assert_node_idle()
    run_env = _cell_env(config, npz, nranks=nranks, ring_threads=ring_threads,
                        ballistic=ballistic, env=env)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        proc = subprocess.run(["bash", str(ENGINE / "launch.sh")],
                              stdout=fh, stderr=subprocess.STDOUT,
                              env=run_env, timeout=timeout)
    return proc.returncode


def _cell_env(config, npz, *, nranks=1, ring_threads=1, ballistic=False,
              env=None):
    run_env = os.environ.copy()
    run_env.update({"QX_CONFIG": str(config), "QX_NPZ": str(npz),
                    "NRANKS": str(nranks), "QX_RING_THREADS": str(ring_threads)})
    if ballistic:
        run_env["QX_BALLISTIC"] = "1"
    if env:
        run_env.update({k: str(v) for k, v in env.items()})
    return run_env


def launch_cells_concurrent(cells, core_budget: int = 200, poll: float = 5.0,
                            pin_numa: bool = True, dry_run: bool = False):
    """Run INDEPENDENT solver cells concurrently, ONE per NUMA socket.

    A single Gamma-only cell is hard-capped at ~one socket (the per-ring bubble
    is cache-bound, so tau-chunking is mandatory and a single GIL'd process tops
    out at ~one socket's ~1.4 TF/s -- 128 threads in one process do NOT beat 64;
    see :func:`numa_sockets`). The full 2-socket / 128-core node is therefore
    filled by running ONE cell per socket, each pinned NUMA-local with
    ``numactl --cpunodebind=S --membind=S`` and given ``ring_threads = cores per
    socket`` (~64). This is the real node-utilisation lever for the sweep-heavy
    workload (eta-extrapolation, convergence grids, the production matrix).

    ``cells`` is a list of dicts: ``{config, npz, log, ballistic(=False),
    env(=None), nranks(=1)}``; per-cell ``ring_threads`` is overridden to fill
    the assigned socket (pass it only to cap lower). Returns ``{npz: returncode}``.
    With ``pin_numa=False`` (or a single detected socket) falls back to the old
    ``core_budget`` packing with no pinning. ``dry_run=True`` prints the planned
    argv + socket map and returns it without launching. One MPI rank per cell.
    """
    sockets = numa_sockets() if pin_numa else [(None, os.cpu_count() or 8)]
    use_sockets = pin_numa and len(sockets) > 1 and sockets[0][0] is not None
    if dry_run:
        plan = []
        for i, c in enumerate(cells):
            node, cps = sockets[i % len(sockets)] if use_sockets else (None, 0)
            rt = cps if use_sockets else c.get("ring_threads", 8)
            plan.append({"npz": str(Path(c["npz"]).name), "numa_node": node,
                         "ring_threads": rt,
                         "argv": _numactl_prefix(node) + ["bash", str(ENGINE / "launch.sh")]})
        for p in plan:
            print(f"[plan] {p['npz']:32s} node={p['numa_node']} ring={p['ring_threads']} "
                  f"argv={' '.join(p['argv'])}", flush=True)
        return plan

    assert_node_idle()
    pending = list(cells)
    active = []   # (Popen, fh, cell, socket_or_None)
    results = {}

    if use_sockets:
        free = list(sockets)                  # available (node, cores) slots
        print(f"[concurrent] {len(sockets)} NUMA sockets, "
              f"{sockets[0][1]} cores each -> one cell/socket", flush=True)
        while pending or active:
            still = []
            for proc, fh, c, sk in active:
                if proc.poll() is None:
                    still.append((proc, fh, c, sk))
                else:
                    fh.close()
                    results[str(c["npz"])] = proc.returncode
                    free.append(sk)
                    print(f"[done] {Path(c['npz']).name} rc={proc.returncode}",
                          flush=True)
            active = still
            while pending and free:
                c = pending.pop(0)
                node, cps = free.pop(0)
                rt = min(int(c["ring_threads"]), cps) if "ring_threads" in c else cps
                Path(c["log"]).parent.mkdir(parents=True, exist_ok=True)
                fh = open(c["log"], "w")
                renv = _cell_env(c["config"], c["npz"], nranks=c.get("nranks", 1),
                                 ring_threads=rt, ballistic=c.get("ballistic", False),
                                 env=c.get("env"))
                argv = _numactl_prefix(node) + ["bash", str(ENGINE / "launch.sh")]
                proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT,
                                        env=renv)
                active.append((proc, fh, c, (node, cps)))
                print(f"[launch] {Path(c['npz']).name} (numa {node}, ring={rt}, "
                      f"{len(pending)} queued)", flush=True)
            if active and (pending or any(p.poll() is None for p, _, _, _ in active)):
                time.sleep(poll)
        return results

    # Fallback: core-budget packing, no NUMA pinning (single socket / pin_numa off)
    while pending or active:
        still = []
        for proc, fh, c, _sk in active:
            if proc.poll() is None:
                still.append((proc, fh, c, _sk))
            else:
                fh.close()
                results[str(c["npz"])] = proc.returncode
                print(f"[done] {Path(c['npz']).name} rc={proc.returncode}",
                      flush=True)
        active = still
        used = sum(c.get("ring_threads", 8) for _, _, c, _ in active)
        launched = True
        while pending and launched:
            launched = False
            c = pending[0]
            need = c.get("ring_threads", 8)
            if not active or used + need <= core_budget:
                pending.pop(0)
                Path(c["log"]).parent.mkdir(parents=True, exist_ok=True)
                fh = open(c["log"], "w")
                renv = _cell_env(c["config"], c["npz"], nranks=c.get("nranks", 1),
                                 ring_threads=need, ballistic=c.get("ballistic", False),
                                 env=c.get("env"))
                proc = subprocess.Popen(["bash", str(ENGINE / "launch.sh")],
                                        stdout=fh, stderr=subprocess.STDOUT, env=renv)
                active.append((proc, fh, c, None))
                used += need
                launched = True
                print(f"[launch] {Path(c['npz']).name} "
                      f"(ring={need}, {used}/{core_budget} cores, "
                      f"{len(pending)} queued)", flush=True)
        if active and (pending or any(p.poll() is None for p, _, _, _ in active)):
            time.sleep(poll)
    return results


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
