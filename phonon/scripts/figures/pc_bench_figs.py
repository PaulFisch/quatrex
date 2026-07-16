"""Preconditioner benchmark analysis: parse the newton-pc-* run logs.

Emits a per-step table (gmres_m, ||R||, trust, precond rank) per arm, the
cumulative-JVP accounting (recycle adds none; fresh adds rank per step),
and a two-panel figure: gmres_m per Newton step, and ||R|| vs cumulative
exact JVPs (the true cost axis; one JVP ~ two bubble evaluations).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
CL = REPO / "cluster"
FIG = REPO / "phonon/studies/out/newton_pc_bench"
FIG.mkdir(parents=True, exist_ok=True)

PAT = re.compile(
    r"newton#(\d+): gmres_m=(\d+) inner_res/\|\|R\|\|=([\d.e+-]+) "
    r"\(tol ([\d.e+-]+)\) \|\|R\|\|=([\d.e+-]+) \|\|delta\|\|=([\d.e+-]+) "
    r"trust=([\d.e+-]+) precond=(\S+)")


def parse(arm: str):
    log = CL / f"newton-pc-{arm}/run.log"
    if not log.exists():
        return []
    steps = []
    for line in log.read_text(errors="replace").splitlines():
        mm = PAT.search(line)
        if mm:
            steps.append(dict(
                n=int(mm.group(1)), m=int(mm.group(2)),
                inner=float(mm.group(3)), R=float(mm.group(5)),
                delta=float(mm.group(6)), trust=float(mm.group(7)),
                pc=mm.group(8)))
    return steps


def main():
    arms = sys.argv[1:] or ["none", "recycle", "fresh"]
    colors = {"none": "tab:gray", "recycle": "tab:blue",
              "fresh": "tab:orange"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8),
                             constrained_layout=True)
    print(f"{'arm':>8} {'steps':>5} {'sum_m':>6} {'+setup':>6} "
          f"{'total_jvp':>9} {'final ||R||':>12} {'median m':>8}")
    for arm in arms:
        steps = parse(arm)
        if not steps:
            print(f"{arm:>8}   (no data)")
            continue
        ms = [s["m"] for s in steps]
        # fresh spends rank JVPs per step on images (rank parsed from
        # the precond tag, e.g. fresh(r=12)).
        setup = 0
        for s in steps:
            if s["pc"].startswith("fresh"):
                setup += int(re.search(r"r=(\d+)", s["pc"]).group(1))
        total = sum(ms) + setup
        med = sorted(ms)[len(ms) // 2]
        print(f"{arm:>8} {len(steps):>5} {sum(ms):>6} {setup:>6} "
              f"{total:>9} {steps[-1]['R']:>12.3e} {med:>8}")
        axes[0].plot([s["n"] for s in steps], ms, "o-",
                     color=colors.get(arm), label=arm)
        cum, xs, rs = 0, [], []
        for s in steps:
            cum += s["m"] + (int(re.search(r"r=(\d+)", s["pc"]).group(1))
                             if s["pc"].startswith("fresh") else 0)
            xs.append(cum)
            rs.append(s["R"])
        axes[1].semilogy(xs, rs, "o-", color=colors.get(arm), label=arm)
    axes[0].set_xlabel("Newton step")
    axes[0].set_ylabel("GMRES dimension m")
    axes[0].set_title("inner-solve cost per step")
    axes[0].legend()
    axes[1].set_xlabel("cumulative exact JVPs")
    axes[1].set_ylabel(r"$\|R\|$")
    axes[1].set_title("residual vs true cost")
    axes[1].legend()
    fig.suptitle("Newton inner-GMRES deflation benchmark "
                 "(CNT L4, from the stall snapshot)")
    fig.savefig(FIG / "pc_bench.png", dpi=160)
    fig.savefig(FIG / "pc_bench.pdf")
    print("saved", FIG / "pc_bench.png")


if __name__ == "__main__":
    main()
