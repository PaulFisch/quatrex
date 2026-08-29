"""Launch and track quatrex GPU debug jobs on Alps/daint (GH200).

Policy (Paul, 2026-07-31 -- hard-coded, do not work around):
  - account lp16 ONLY;
  - debug partition by default (30 min limit). NORMAL partition allowed
    (Paul 2026-08-02) with <=2 nodes/job; more needs Paul FIRST, then
    --approved-by-paul. Other partitions: ask Paul first;
  - EVERY partition is charged to daint_normal_ledger.md, debug included
    (Paul 2026-08-10 -- CSCS bills debug like the rest, so leaving it out
    understated the figure). Hard cap 500 committed node-hours
    (nodes x walltime at submission); exceeding it needs Paul FIRST.
    Rows over-charge jobs that finish early, so the ledger drifts above
    real usage; a "Running total from here: **N nh**" line reconciles it
    to the CSCS figure and supersedes everything above it;
  - at most MAX_ACTIVE jobs queued/running at once;
  - every job goes through this script (never ad-hoc sbatch), named
    qx-<name>, workdir <repo>/cluster/<name>/ on daint scratch.

One-time:  python phonon/scripts/daint.py setup
Code sync: commit+push locally, then python phonon/scripts/daint.py sync
Launch:    python phonon/scripts/daint.py launch --name X \\
               --config cluster/X/quatrex_config.toml [--ranks 4] \\
               [--env QX_MAXIT=3 ...]
           python phonon/scripts/daint.py launch --name X -- <command>
Monitor:   status | list | tail --name X [-f] | kill --name X | pull --name X
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

HOST = "daint"
ACCOUNT = "lp16"
PARTITION = "debug"
TIME_LIMIT = "00:30:00"
MAX_ACTIVE = 2          # debug-partition churn limit
MAX_ACTIVE_NORMAL = 5   # Paul 2026-08-02: parallel normal jobs OK, don't overdo
MAX_NODES = 1
SCRATCH = "/capstor/scratch/cscs/pfischil"
REPO = f"{SCRATCH}/quatrex"
VENV = f"{SCRATCH}/quatrex-venv"
UENV = "prgenv-gnu/26.3:v1"
ORIGIN = "git@github.com:PaulFisch/quatrex.git"
BRANCH = "phonon-phonon"


def ssh(cmd: str, timeout: int = 60, check: bool = False) -> str:
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", HOST, cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    if check and r.returncode != 0:
        sys.exit(f"ssh failed ({r.returncode}): {r.stderr.strip()}")
    return r.stdout


def our_jobs() -> list[tuple[str, str, str, str]]:
    """(jobid, name, state, elapsed) for all qx- jobs in the queue."""
    out = ssh('squeue -u $USER -h -o "%i|%j|%T|%M"')
    jobs = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 4 and parts[1].startswith("qx-"):
            jobs.append(tuple(parts))
    return jobs


def cmd_status(_):
    jobs = our_jobs()
    print(f"active qx- jobs: {len(jobs)}/{MAX_ACTIVE}")
    for jid, name, state, elapsed in jobs:
        print(f"  {jid:>9} {name:<24} {state:<10} {elapsed}")
    usage = ssh(
        'sacct -u $USER -A lp16 -S $(date +%Y-%m-01) -X -n '
        '--format=ElapsedRaw,AllocNodes 2>/dev/null'
    )
    sec = 0
    for line in usage.splitlines():
        f = line.split()
        if len(f) == 2 and f[0].isdigit() and f[1].isdigit():
            sec += int(f[0]) * int(f[1])
    print(f"lp16 node-hours this month (this user): {sec / 3600:.2f}")


def cmd_setup(_):
    print("cloning + venv on daint (idempotent) ...")
    print(ssh(
        f"test -d {REPO} || git clone -b {BRANCH} {ORIGIN} {REPO}", 240))
    # cupy wheel flavour must match the uenv CUDA major.
    cuda_major = ssh(
        f"uenv run {UENV} --view=default -- nvcc --version 2>/dev/null "
        "| grep -oP 'release \\K[0-9]+' | head -1", 120).strip() or "12"
    print(f"uenv CUDA major: {cuda_major}")
    print(ssh(
        f"test -d {VENV} || uenv run {UENV} --view=default -- "
        f"python -m venv {VENV}", 300))
    print(ssh(
        f"uenv run {UENV} --view=default -- bash -c '"
        f"source {VENV}/bin/activate && "
        f"pip install -q --upgrade pip && "
        # mpi4py MUST build against the uenv mpicc (cray-mpich); the
        # plain env var MPICC is ignored by mpi4py >= 4.
        f"MPI4PY_BUILD_MPICC=\"mpicc -shared\" "
        f"pip install -q --no-binary=mpi4py mpi4py && "
        f"pip install -q cupy-cuda{cuda_major}x numpy scipy h5py pydantic "
        f"toml numba ase matplotlib pytest pytest-mpi "
        # phonon/solver/__init__ -> dense -> phonon_inputs.convention imports
        # Phonopy at module scope, so the whole studies tree needs it even
        # for a toy chain that never reads a force-constant file.
        f"phonopy'", 1800))
    print(ssh(
        f"uenv run {UENV} --view=default -- bash -c '"
        f"source {VENV}/bin/activate && python -c \""
        f"import cupy; import mpi4py.MPI as M; "
        f"print(cupy.__version__, M.Get_library_version()[:40])"
        f"\"'", 300))
    print("setup done")


def cmd_sync(_):
    print(ssh(f"git -C {REPO} pull --ff-only", 120, check=True))


NORMAL_LEDGER = Path(__file__).resolve().parent / "daint_normal_ledger.md"
# Paul 2026-08-29: raised to 500 (was 400, 300, 200 and 100). HARD upper
# limit, not a target. The ledger charges EVERY partition, debug included --
# debug jobs are billed by CSCS like any other, so leaving them out understated
# the true figure.
NORMAL_NH_CAP = 500.0
NORMAL_MAX_NODES = 2         # per job, unless Paul authorises more


def _walltime_hours(t: str) -> float:
    p = [int(x) for x in t.split("-")[-1].split(":")]
    days = int(t.split("-")[0]) if "-" in t else 0
    while len(p) < 3:
        p.insert(0, 0)
    return days * 24 + p[0] + p[1] / 60 + p[2] / 3600


def _ledger_committed() -> float:
    """Node-hours charged so far.

    Rows charge nodes x walltime AT SUBMISSION (worst case, the only
    figure known before a run starts, so the cap cannot be exceeded by
    construction). Jobs killed early therefore over-charge: on
    2026-08-08 CSCS reported 103 nh actually consumed against 197
    committed here. A "Running total from here: **N nh**" line in the
    ledger reconciles the accumulated figure to measured usage; when
    present, the LAST such line replaces everything above it and only
    rows after it are added.
    """
    if not NORMAL_LEDGER.exists():
        return 0.0
    import re

    total = 0.0
    for line in NORMAL_LEDGER.read_text().splitlines():
        m = re.match(r"Running total from here: \*\*([\d.]+) nh\*\*", line)
        if m:                                # reset: discard everything above
            total = float(m.group(1))
            continue
        if line.startswith("| 2"):           # data rows: | 2026-.. |
            try:
                total += float(line.split("|")[6])
            except (IndexError, ValueError):
                pass
    return total


def _ledger_append(args, job_id: str, nh: float, total: float) -> None:
    import datetime
    if not NORMAL_LEDGER.exists():
        NORMAL_LEDGER.write_text(
            "# daint job ledger (ALL partitions)\n\n"
            f"Hard cap {NORMAL_NH_CAP:.0f} node-hours committed "
            "(nodes x walltime at submission). Paul authorises any "
            "excess BEFORE launch.\n\n"
            "| date | job | name | nodes | walltime | nh | total nh | part |\n"
            "|---|---|---|---|---|---|---|---|\n")
    with NORMAL_LEDGER.open("a") as f:
        f.write(f"| {datetime.date.today()} | {job_id} | {args.name} "
                f"| {args.nodes} | {args.time} | {nh:.2f} "
                f"| {total:.2f} | {getattr(args, 'partition', PARTITION)} |\n")


def _guard(args):
    part = getattr(args, "partition", PARTITION)
    # The cap covers EVERY partition (Paul 2026-08-10): CSCS bills debug
    # jobs too, so charging only "normal" understated the real figure.
    nh = getattr(args, "nodes", 1) * _walltime_hours(args.time)
    committed = _ledger_committed()
    if committed + nh > NORMAL_NH_CAP:
        sys.exit(f"policy: ledger holds {committed:.1f} nh committed; "
                 f"+{nh:.1f} nh would exceed the {NORMAL_NH_CAP:.0f} nh "
                 "cap -- needs Paul's explicit authorization")
    if part == "normal":
        # Paul 2026-08-02: normal partition allowed under the ledgered
        # cap and <=2 nodes/job (more needs his OK).
        if getattr(args, "nodes", 1) > NORMAL_MAX_NODES \
                and not args.approved_by_paul:
            sys.exit(f"policy: normal partition allows <={NORMAL_MAX_NODES} "
                     "nodes/job (Paul's OK + --approved-by-paul for more)")
    elif part != PARTITION and not args.approved_by_paul:
        sys.exit(f"policy: partition != {PARTITION} needs Paul's explicit OK "
                 "first (then pass --approved-by-paul)")
    if part != "normal" and getattr(args, "nodes", 1) > MAX_NODES \
            and not args.approved_by_paul:
        sys.exit(f"policy: >{MAX_NODES} node needs Paul's explicit OK first "
                 "(then pass --approved-by-paul)")
    jobs = our_jobs()
    cap_active = MAX_ACTIVE_NORMAL if part == "normal" else MAX_ACTIVE
    if len(jobs) >= cap_active:
        sys.exit(f"policy: {len(jobs)} qx- jobs already active "
                 f"(max {cap_active} for {part}) -- wait or kill one")
    if any(j[1] == f"qx-{args.name}" for j in jobs):
        sys.exit(f"qx-{args.name} is already queued/running")


def cmd_launch(args):
    _guard(args)
    run_dir = f"{REPO}/cluster/{args.name}"
    env_lines = "\n".join(f"export {shlex.quote(e)}" for e in args.env or [])
    if args.command:
        payload = " ".join(args.command)
    else:
        if not args.config:
            sys.exit("either --config or a trailing '-- <command>' is needed")
        env_lines += f"\nexport QX_CONFIG={REPO}/{args.config}"
        if not any(e.startswith("QX_NPZ=") for e in (args.env or [])):
            env_lines += f"\nexport QX_NPZ={run_dir}/run.npz"
        payload = f"python {REPO}/phonon/studies/engine/run.py"
    script = f"""#!/bin/bash
#SBATCH --job-name=qx-{args.name}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks-per-node={args.ranks}
#SBATCH --gpus-per-task=1
#SBATCH --uenv={UENV}
#SBATCH --view=default
#SBATCH --output={run_dir}/slurm-%j.out
set -u
source {VENV}/bin/activate
export PYTHONPATH={REPO}/src
export QTX_ARRAY_MODULE=cupy
export MPICH_GPU_SUPPORT_ENABLED=1
export QTX_PROFILE_LEVEL=${{QTX_PROFILE_LEVEL:-default}}
{env_lines}
cd {run_dir}
srun --cpu-bind=cores bash -c \\
    'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; exec {payload}'
"""
    ssh(f"mkdir -p {run_dir}", check=True)
    subprocess.run(
        ["ssh", HOST, f"cat > {run_dir}/job.sh"],
        input=script, text=True, check=True,
    )
    out = ssh(f"cd {run_dir} && sbatch job.sh", check=True)
    print(out.strip())
    job_id = out.strip().split()[-1]
    nh = args.nodes * _walltime_hours(args.time)
    total = _ledger_committed() + nh
    _ledger_append(args, job_id, nh, total)
    print(f"  ledger: +{nh:.2f} nh committed "
          f"({total:.2f}/{NORMAL_NH_CAP:.0f} nh, "
          f"{getattr(args, 'partition', PARTITION)}) -> {NORMAL_LEDGER}")
    print(f"  tail: python phonon/scripts/daint.py tail --name {args.name} -f")


def cmd_list(_):
    for jid, name, state, elapsed in our_jobs():
        print(f"{jid:>9} {name:<24} {state:<10} {elapsed}")


def cmd_tail(args):
    run_dir = f"{REPO}/cluster/{args.name}"
    latest = ssh(f"ls -t {run_dir}/slurm-*.out 2>/dev/null | head -1").strip()
    if not latest:
        sys.exit(f"no slurm output yet in {run_dir}")
    flag = "-f" if args.follow else f"-n {args.lines}"
    subprocess.run(["ssh", "-t", HOST, f"tail {flag} {latest}"])


def cmd_kill(args):
    jobs = [j for j in our_jobs() if j[1] == f"qx-{args.name}"]
    if not jobs:
        sys.exit(f"no active job qx-{args.name}")
    for jid, name, *_ in jobs:
        print(f"scancel {jid} ({name})")
        ssh(f"scancel {jid}", check=True)


def cmd_pull(args):
    subprocess.run(
        ["rsync", "-avz", f"{HOST}:{REPO}/cluster/{args.name}/",
         f"cluster/{args.name}/"],
        check=True,
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    sub.add_parser("sync").set_defaults(fn=cmd_sync)
    q = sub.add_parser("launch")
    q.add_argument("--name", required=True)
    q.add_argument("--config", help="repo-relative TOML for engine runs")
    q.add_argument("--ranks", type=int, default=1, help="tasks (GPUs), <=4")
    q.add_argument("--nodes", type=int, default=1)
    q.add_argument("--partition", default=PARTITION)
    q.add_argument("--time", default=TIME_LIMIT)
    q.add_argument("--env", action="append", metavar="K=V")
    q.add_argument("--approved-by-paul", action="store_true",
                   help="required for >1 node or non-debug partitions")
    q.add_argument("command", nargs="*",
                   help="alternative to --config: raw command after --")
    q.set_defaults(fn=cmd_launch)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    q = sub.add_parser("tail")
    q.add_argument("--name", required=True)
    q.add_argument("-f", "--follow", action="store_true")
    q.add_argument("-n", "--lines", type=int, default=40)
    q.set_defaults(fn=cmd_tail)
    q = sub.add_parser("kill")
    q.add_argument("--name", required=True)
    q.set_defaults(fn=cmd_kill)
    q = sub.add_parser("pull")
    q.add_argument("--name", required=True)
    q.set_defaults(fn=cmd_pull)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
