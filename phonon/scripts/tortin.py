#!/usr/bin/env python3
"""Polite launcher for background quatrex runs on the tortin cluster.

Run from the laptop; drives ssh + tmux on the nodes. Policy (see CLAUDE.md):
  - use at most MAX_NODES nodes simultaneously,
  - after claiming a node, at least MIN_FREE_LEFT other idle nodes must remain,
  - never the highest-numbered node,
  - prefer tortin1-tortin8,
  - only long-term-idle nodes (all three load averages < LOAD_MAX).

No local state: tmux is ground truth. Our sessions are named qx-<name>; a
session closes when its command exits, so live qx-* sessions == active runs.
Run logs and outputs live on the shared scratch at <REPO>/cluster/<name>/,
pulled back with the `pull` subcommand into the laptop's gitignored cluster/.

Usage:
  tortin.py status                  # cload + our sessions
  tortin.py pick [-n N]             # which nodes the policy would grant
  tortin.py launch --name X [--node tortinK] [--env K=V ...] \
      (--config path/to.toml [--ranks N] | -- CMD...)
  tortin.py list                    # our qx-* sessions on all nodes
  tortin.py tail --name X [-f]      # tail the run log
  tortin.py kill --name X           # kill session qx-X
  tortin.py pull --name X           # rsync cluster/<X> back to the laptop
"""

import argparse
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "/usr/scratch/mont-fort11/pfischill/quatrex"
CONDA_ENV = "/usr/scratch/mont-fort11/pfischill/conda/envs/quatrex-dev"
LOCAL_REPO = Path(__file__).resolve().parents[2]
MAX_NODES = 4
MIN_FREE_LEFT = 2
LOAD_MAX = 0.5
PREFERRED = set(range(1, 9))
PREFIX = "qx-"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "-o", "LogLevel=ERROR"]
GATEWAY_CANDIDATES = [2, 3, 4, 5, 6, 8, 9, 10]

_CLOAD_RE = re.compile(
    r"^(tortin(\d+))\s+up\s+.*load average:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")


def ssh(node, cmd, timeout=30, check=False):
    try:
        r = subprocess.run(["ssh", *SSH_OPTS, node, cmd],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # A hung node must not take the tooling down (cf. the cload note):
        # report it like a failed command; callers treat the node as
        # unavailable.
        r = subprocess.CompletedProcess(
            ["ssh", node, cmd], returncode=255, stdout="",
            stderr=f"ssh {node}: timeout after {timeout}s")
        if check:
            sys.exit(r.stderr)
        return r
    if check and r.returncode != 0:
        sys.exit(f"ssh {node} failed: {r.stderr.strip() or r.stdout.strip()}")
    return r


_gateway = None


def gateway():
    """First reachable node, used for cload / shared-fs access."""
    global _gateway
    if _gateway is None:
        for k in GATEWAY_CANDIDATES:
            if ssh(f"tortin{k}", "true", timeout=12).returncode == 0:
                _gateway = f"tortin{k}"
                break
        else:
            sys.exit("no tortin node reachable")
    return _gateway


def cload():
    """{node_number: (load1, load5, load15)} for all reachable nodes.

    ``cload`` exits non-zero when any node is down (rup RPC failure) but
    still prints the reachable ones -- parse what came through and treat
    missing nodes as unavailable (they are never launch candidates)."""
    r = ssh(gateway(), "cload tortin", timeout=60)
    loads = {}
    for line in r.stdout.splitlines():
        m = _CLOAD_RE.match(line.strip())
        if m:
            loads[int(m.group(2))] = tuple(float(m.group(i)) for i in (3, 4, 5))
    if not loads:
        sys.exit("could not parse cload output:\n"
                 + r.stdout + r.stderr)
    return loads


def our_sessions(numbers):
    """{node_number: [session names]} of our qx-* tmux sessions."""
    def probe(k):
        r = ssh(f"tortin{k}", "tmux ls -F '#S' 2>/dev/null", timeout=15)
        return k, [s for s in r.stdout.split() if s.startswith(PREFIX)]

    with ThreadPoolExecutor(max_workers=12) as ex:
        return {k: names for k, names in ex.map(probe, numbers) if names}


def classify(loads):
    """Split nodes into (excluded_highest, in_use_by_us, idle, busy)."""
    highest = max(loads)
    candidates = [k for k in sorted(loads) if k != highest]
    used = our_sessions(candidates)
    idle, busy = [], []
    for k in candidates:
        if k in used:
            continue
        (idle if all(l < LOAD_MAX for l in loads[k]) else busy).append(k)
    idle.sort(key=lambda k: (k not in PREFERRED, k))
    return highest, used, idle, busy


def pick(n, loads=None):
    """Nodes the policy grants right now (list of ints, preference order)."""
    loads = loads or cload()
    _, used, idle, _ = classify(loads)
    grant = min(n, MAX_NODES - len(used), len(idle) - MIN_FREE_LEFT)
    return [f"tortin{k}" for k in idle[:max(grant, 0)]], used, idle


def cmd_status(_):
    loads = cload()
    highest, used, idle, busy = classify(loads)
    for k in sorted(loads):
        l1, l5, l15 = loads[k]
        tag = ("EXCLUDED (highest)" if k == highest else
               f"OURS: {','.join(used[k])}" if k in used else
               "idle" if k in idle else "busy")
        pref = "*" if k in PREFERRED and tag == "idle" else " "
        print(f"tortin{k:<3} {l1:7.2f} {l5:7.2f} {l15:7.2f}  {pref} {tag}")
    print(f"\nbudget: {len(used)}/{MAX_NODES} nodes in use, "
          f"{len(idle)} idle candidates (must keep >={MIN_FREE_LEFT} free)")


def cmd_pick(args):
    nodes, used, idle = pick(args.n)
    print(" ".join(nodes) if nodes else "NONE (budget or idle-count exhausted)")
    if len(nodes) < args.n:
        print(f"  (granted {len(nodes)}/{args.n}: {len(used)} in use, "
              f"{len(idle)} idle, keep {MIN_FREE_LEFT} free)", file=sys.stderr)


def cmd_launch(args):
    name = args.name
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        sys.exit("--name must be [A-Za-z0-9._-]+")
    session = PREFIX + name
    loads = cload()

    if args.node:
        node = args.node
        k = int(node.removeprefix("tortin"))
        if k == max(loads):
            sys.exit(f"{node} is the highest-numbered node -- never used")
        granted, used, idle = pick(1, loads)
        if k in used:
            sys.exit(f"{node} already runs our session(s): {used[k]}")
        if k not in idle:
            sys.exit(f"{node} is not idle (load {loads[k]})")
        if not granted:
            sys.exit("budget exhausted (4-node cap or <3 idle nodes would remain)")
    else:
        granted, _, _ = pick(1, loads)
        if not granted:
            sys.exit("no node grantable (budget or idle-count exhausted)")
        node = granted[0]

    # race guard: the load right now, not the cached cload
    r = ssh(node, "cat /proc/loadavg", check=True)
    if any(float(x) >= LOAD_MAX for x in r.stdout.split()[:3]):
        sys.exit(f"{node} got busy since the cload check: {r.stdout.strip()}")
    if ssh(node, f"tmux has-session -t {shlex.quote(session)} 2>/dev/null").returncode == 0:
        sys.exit(f"session {session} already exists on {node}")

    exports = [f"PATH={CONDA_ENV}/bin:$PATH", f"PYTHONPATH={REPO}/src"]
    exports += args.env or []
    if args.config:
        exports.append(f"QX_CONFIG={args.config}")
        exports.append(f"NRANKS={args.ranks}")
        cmd = "bash phonon/studies/engine/launch.sh"
    elif args.cmd:
        cmd = " ".join(args.cmd)
    else:
        sys.exit("give --config or a command after --")

    inner = (f"export {' '.join(exports)}; cd {REPO}; mkdir -p cluster/{name}; "
             f"( {cmd} ) 2>&1 | tee -a cluster/{name}/run.log")
    remote = (f"tmux new-session -d -s {shlex.quote(session)} "
              f"{shlex.quote(f'bash -c {shlex.quote(inner)}')}")
    ssh(node, remote, check=True)
    print(f"launched {session} on {node}")
    print(f"  log:  {REPO}/cluster/{name}/run.log")
    print(f"  tail: {sys.argv[0]} tail --name {name} [-f]")


def cmd_list(_):
    used = our_sessions(range(1, max(cload()) + 1))
    if not used:
        print("no qx-* sessions on any node")
    for k, names in sorted(used.items()):
        print(f"tortin{k}: {' '.join(names)}")


def cmd_tail(args):
    log = f"{REPO}/cluster/{args.name}/run.log"
    flag = "-f" if args.follow else f"-n {args.lines}"
    subprocess.run(["ssh", *SSH_OPTS, "-t" if args.follow else "-T",
                    gateway(), f"tail {flag} {shlex.quote(log)}"])


def cmd_kill(args):
    session = PREFIX + args.name
    used = our_sessions(range(1, max(cload()) + 1))
    for k, names in used.items():
        if session in names:
            ssh(f"tortin{k}", f"tmux kill-session -t {shlex.quote(session)}",
                check=True)
            print(f"killed {session} on tortin{k}")
            return
    sys.exit(f"session {session} not found on any node")


def cmd_pull(args):
    src = f"{gateway()}:{REPO}/cluster/{args.name}/"
    dst = LOCAL_REPO / "cluster" / args.name
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--info=stats1", "-e",
                    "ssh " + " ".join(SSH_OPTS), src, str(dst) + "/"],
                   check=True)
    print(f"pulled to {dst}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="sub", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    q = sub.add_parser("pick")
    q.add_argument("-n", type=int, default=1)
    q.set_defaults(fn=cmd_pick)
    q = sub.add_parser("launch")
    q.add_argument("--name", required=True)
    q.add_argument("--node")
    q.add_argument("--config", help="QX_CONFIG toml (runs the study engine)")
    q.add_argument("--ranks", type=int, default=1, help="NRANKS for launch.sh")
    q.add_argument("--env", action="append", metavar="K=V")
    q.add_argument("cmd", nargs="*", help="command after -- (instead of --config)")
    q.set_defaults(fn=cmd_launch)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    q = sub.add_parser("tail")
    q.add_argument("--name", required=True)
    q.add_argument("-f", "--follow", action="store_true")
    q.add_argument("-n", "--lines", type=int, default=50)
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
