"""Read-only collector for production phonon-SCBA profiler dumps.

Parses one or more ``quatrex_times.json`` (the per-label stats written by
``Profiler.dump_stats``: ``total_call_time``, ``*_per_rank``, and the
``*_after_barrier_*`` comm-wait when ``QTX_PROFILE_COMM_SYNC=1``) plus the
sibling ``run.npz`` snapshot (n_iter, comm grid). Emits:

  * a per-phase breakdown for a single run (compute vs comm-wait, % of SCBA),
  * a scaling CSV across runs (nranks, bcs, qcs, SCBA wall, per-iter wall).

Usage:
    python parse_profile.py RUN_DIR [RUN_DIR ...]      # tables
    python parse_profile.py --csv out.csv RUN_DIR ...  # also write scaling CSV
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

# The per-iteration phases worth surfacing (top-level + phonon-solver + phph).
PHASES = [
    "SCBA: Iteration",
    "PhononSolver: OBC",
    "PhononSolver: Assemble",
    "PhononSolver: Selected Solve",
    "RGF dist: init",
    "RGF dist: Schur",
    "SCBA: stack->nnz transpose",
    "SCBA: stack->nnz transpose back",
    "SCBA: Interactions",
    "Interaction: Phonon-Phonon",
    "SigmaPhononPhonon",
    "SCBA: Symmetrize Sigma",
    "SCBA: Update Sigma",
    "SCBA: Convergence test",
]


def load_run(run_dir):
    d = Path(run_dir)
    jpath = next(iter(sorted(d.glob("quatrex_times.json"))), None) or (d / "quatrex_times.json")
    stats = json.load(open(jpath)) if jpath.exists() else {}
    npz = d / "run.npz"
    snap = dict(np.load(npz, allow_pickle=True)) if npz.exists() else {}
    return stats, snap, jpath


def per_rank(stat, key, field="total_call_time_per_rank"):
    s = stat.get(key)
    return float(s[field]) if s and field in s else None


def print_breakdown(run_dir):
    stats, snap, jpath = load_run(run_dir)
    if not stats:
        print(f"[{run_dir}] no profiler JSON ({jpath})")
        return
    n_iter = int(snap.get("n_iter", 0)) or 1
    bcs = int(snap.get("block_comm_size", 1))
    qcs = int(snap.get("q_comm_size", 1))
    nranks = int(snap.get("nranks", 1))
    total = per_rank(stats, "SCBA: Iteration") or per_rank(stats, "SCBA") or 0.0
    print(f"\n=== {run_dir}  (nranks={nranks} bcs={bcs} qcs={qcs} n_iter={n_iter}) ===")
    print(f"{'phase':<34}{'wall/rank[s]':>13}{'%SCBAit':>9}{'comm-wait[s]':>14}{'/iter[s]':>10}")
    for ph in PHASES:
        t = per_rank(stats, ph)
        if t is None:
            continue
        bar = per_rank(stats, ph, "total_after_barrier_time_per_rank")
        wait = (bar - t) if (bar is not None) else float("nan")
        pct = 100.0 * t / total if total else float("nan")
        print(f"{ph:<34}{t:>13.3f}{pct:>9.1f}{wait:>14.3f}{t / n_iter:>10.3f}")
    # Any other labels not in PHASES (so nothing is silently dropped).
    extra = sorted(set(stats) - set(PHASES) - {"SCBA"})
    if extra:
        print("  (other labels:", ", ".join(extra[:12]) + (" ..." if len(extra) > 12 else ""), ")")


def scaling_rows(run_dirs):
    rows = []
    for rd in run_dirs:
        stats, snap, _ = load_run(rd)
        if not stats:
            continue
        n_iter = int(snap.get("n_iter", 0)) or 1
        it = per_rank(stats, "SCBA: Iteration")
        rows.append(dict(
            run=str(rd), system=str(snap.get("system", "")),
            nranks=int(snap.get("nranks", 1)),
            bcs=int(snap.get("block_comm_size", 1)),
            qcs=int(snap.get("q_comm_size", 1)),
            n_iter=n_iter,
            scba_wall=per_rank(stats, "SCBA") or per_rank(stats, "SCBA: Iteration"),
            iter_wall=it,
            per_iter=(it / n_iter) if it else None,
            sigma_phph=per_rank(stats, "SigmaPhononPhonon"),
            selected_solve=per_rank(stats, "PhononSolver: Selected Solve"),
            obc=per_rank(stats, "PhononSolver: OBC"),
            best_cons=float(snap["best_cons"]) if "best_cons" in snap else None,
        ))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--csv", default=None, help="write a scaling CSV across the run dirs")
    a = p.parse_args()

    for rd in a.run_dirs:
        print_breakdown(rd)

    if a.csv:
        rows = scaling_rows(a.run_dirs)
        if rows:
            with open(a.csv, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
            print(f"\nwrote scaling CSV -> {a.csv} ({len(rows)} rows)")
            # quick speedup view vs the smallest nranks
            base = min(rows, key=lambda r: r["nranks"])
            if base["per_iter"]:
                print(f"{'nranks':>7}{'bcs':>5}{'qcs':>5}{'per_iter[s]':>13}{'speedup':>9}{'eff%':>7}")
                for r in sorted(rows, key=lambda r: (r["nranks"], r["bcs"])):
                    if not r["per_iter"]:
                        continue
                    sp = base["per_iter"] / r["per_iter"]
                    eff = 100.0 * sp / (r["nranks"] / base["nranks"])
                    print(f"{r['nranks']:>7}{r['bcs']:>5}{r['qcs']:>5}"
                          f"{r['per_iter']:>13.3f}{sp:>9.2f}{eff:>7.0f}")


if __name__ == "__main__":
    main()
