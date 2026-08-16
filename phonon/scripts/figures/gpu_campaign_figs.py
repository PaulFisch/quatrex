"""GPU campaign figures (fig:res_gpu_ceiling, fig:res_gpu_scaling).

Data:
  All numbers are literals from phonon/docs/gpu_campaign_2026-07.md
  committed run.npz; the in-engine rates come from the campaign's slurm
  logs (cluster/l4bench, cluster/l*, cluster/mos2f3, cluster/filmq).

Run:  python phonon/scripts/figures/gpu_campaign_figs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.studies import style

FIGDIR = ROOT / "document/fig/transport_sweeps"

# --- verdict chain (TF/s, FP64) ---------------------------------------------
CHAIN = [
    ("TC peak", 67.0, 67.0),
    ("sustained\nGEMM", 43.0, 54.0),
    ("ring shape\n$b=36$", 13.5, 13.5),
    ("in-engine", 11.4, 12.5),
]

# --- batched ring-shape ceilings (TF/s), doc section 9 ----------------------
W_COLS = [60, 241, 481]
CEIL = {  # b: [w=60, w=241, w=481]
    15: [0.35, 1.36, 2.69],
    18: [0.71, 2.59, 5.48],
    36: [9.8, 12.3, 13.7],
    54: [22.8, 24.7, 25.4],
    63: [27.5, 28.3, 28.2],
}
# in-engine operating points: (b, TF/s, label)
ENGINE_PTS = [
    (36, 11.4, "CNT $\\Gamma$"),
    (18, 2.46, "film, per-task"),
    (18, 6.84, "film, batched"),
    (135, 26.7, "d11a $\\Gamma$"),
]
B135_CEIL = (22.0, 27.0)  # measured b=135 microbench spread

# --- strong scaling, CNT L4 ne=241 (stack axis) -----------------------------
SCALING = [  # (gpus, s/iter, E(p))
    (1, 3.87, 1.00),
    (2, 2.11, 0.92),
    (4, 1.10, 0.88),
    (8, 0.62, 0.78),
]

# --- film iteration attribution (mos2f3, nq=25, ne=121, 1 GH200) ------------
FILM = {
    "per-task": {"ring": 6.47, "obc": 2.90, "rest": 9.53 - 6.47 - 2.90},
    "batched": {"ring": 2.33, "obc": 2.90, "rest": 5.35 - 2.33 - 2.90},
}


def fig_ceiling() -> None:
    fig, (ax_a, ax_b) = style.figure(ncols=2, width=4.4, height=3.3)

    xs = np.arange(len(CHAIN))
    los = np.array([lo for _, lo, _ in CHAIN])
    his = np.array([hi for _, _, hi in CHAIN])
    mid = 0.5 * (los + his)
    ax_a.bar(xs, mid, 0.62, color=style.RC["axes.prop_cycle"].by_key()["color"][0])
    ax_a.errorbar(xs, mid, yerr=np.vstack([mid - los, his - mid]),
                  fmt="none", ecolor="k", elinewidth=1.1, capsize=3)
    for x, hi in zip(xs, his):
        ax_a.annotate(f"{hi:g}" if los[x] == hi else f"{los[x]:g}–{hi:g}",
                      (x, hi), textcoords="offset points", xytext=(0, 4),
                      ha="center", fontsize=8)
    ax_a.set_xticks(xs, [n for n, _, _ in CHAIN])
    ax_a.set_ylabel("FP64 throughput (TF/s)")
    ax_a.set_ylim(0, 74)

    colors = style.RC["axes.prop_cycle"].by_key()["color"]
    bs = sorted(CEIL)
    for i, w in enumerate(W_COLS):
        ax_b.plot(bs, [CEIL[b][i] for b in bs], "o-", color=colors[i],
                  label=f"$w={w}$")
    ax_b.fill_between([130, 140], *B135_CEIL, color=colors[2], alpha=0.25,
                      linewidth=0)
    mk = {"CNT $\\Gamma$": "s", "film, per-task": "v", "film, batched": "^",
          "d11a $\\Gamma$": "D"}
    for b, tf, lab in ENGINE_PTS:
        ax_b.plot([b], [tf], mk[lab], color="k", markersize=5.5, zorder=5)
        ax_b.annotate(lab, (b, tf), textcoords="offset points",
                      xytext=(5, -3), fontsize=7.5)
    ax_b.set_xscale("log")
    ax_b.set_xticks(bs + [135], [str(b) for b in bs] + ["135"])
    ax_b.minorticks_off()
    ax_b.set_xlabel("block size $b$")
    ax_b.set_ylabel("ring throughput (TF/s)")
    ax_b.legend(loc="upper left", title="batch depth")

    style.save(fig, "gpu_ceiling", directory=FIGDIR)

    print("verdict chain (TF/s):",
          " -> ".join(f"{n.replace(chr(10), ' ')} {lo:g}" +
                      ("" if lo == hi else f"-{hi:g}")
                      for n, lo, hi in CHAIN))
    print(f"in-engine fraction of shape ceiling (b=36): "
          f"{11.4 / 13.5:.0%}-{12.5 / 13.5:.0%}")
    print(f"b=63 fraction of TC peak: {CEIL[63][2] / 67:.0%}")
    print(f"film batched vs per-task: {6.84 / 2.46:.2f}x ring rate; "
          f"batched vs b=18 deep-batch ceiling {6.84 / 5.48:.2f}x "
          "(saturated; >1 = deeper effective batch than w=481)")


def fig_scaling_film() -> None:
    fig, (ax_a, ax_b) = style.figure(ncols=2, width=4.4, height=3.3)
    colors = style.RC["axes.prop_cycle"].by_key()["color"]

    gpus = np.array([g for g, _, _ in SCALING])
    s_it = np.array([s for _, s, _ in SCALING])
    ax_a.plot(gpus, s_it, "o-", color=colors[0], label="measured")
    ax_a.plot(gpus, s_it[0] / gpus, ":", color="0.5", label="ideal")
    for g, s, e in SCALING[1:]:
        ax_a.annotate(f"$E={e:.2f}$", (g, s), textcoords="offset points",
                      xytext=(6, 4), fontsize=8)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xticks(gpus, [str(g) for g in gpus])
    ax_a.minorticks_off()
    ax_a.set_xlabel("GH200 GPUs (stack axis)")
    ax_a.set_ylabel("s / iteration")
    ax_a.legend()

    labels = list(FILM)
    bottoms = np.zeros(len(labels))
    for i, part in enumerate(("ring", "obc", "rest")):
        vals = np.array([FILM[k][part] for k in labels])
        ax_b.bar(labels, vals, 0.55, bottom=bottoms, color=colors[i],
                 label={"ring": "dense-$q$ ring", "obc": "OBC",
                        "rest": "rest"}[part])
        bottoms += vals
    for x, k in enumerate(labels):
        tot = sum(FILM[k].values())
        ax_b.annotate(f"{tot:.2f} s", (x, tot), textcoords="offset points",
                      xytext=(0, 4), ha="center", fontsize=8.5)
    ax_b.set_ylabel("s / iteration")
    ax_b.legend()

    style.save(fig, "gpu_scaling_film", directory=FIGDIR)

    print("scaling E(p):", {g: e for g, _, e in SCALING})
    tot = {k: sum(v.values()) for k, v in FILM.items()}
    print(f"film iteration {tot['per-task']:.2f} -> {tot['batched']:.2f} s/it "
          f"({tot['per-task'] / tot['batched']:.1f}x); OBC share after: "
          f"{FILM['batched']['obc'] / tot['batched']:.0%}")


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_ceiling()
    fig_scaling_film()


if __name__ == "__main__":
    main()
