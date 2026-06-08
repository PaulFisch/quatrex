"""Consolidated profiling + strong-scaling results for the production
distributed phonon-phonon SCBA (measured on the 256-core node, 1 thread/rank,
--bind-to core). Writes a CSV + a figure for the writeup.

Per-phase profile (CNT(3,3) L=4, -np 1): the 3-phonon bubble
(SigmaPhononPhonon) is 99.8% of every SCBA iteration; the RGF Dyson solve and
the OBC are <0.1% combined. So the bubble IS the workload, and the end-to-end
strong scaling equals the bubble's. The bubble is ring-contraction-bound (the
x6 folded contractions per quad), operating on each rank's local tau-slice --
which is why it distributes over the energy (stack) axis, overturning the
earlier "energy axis is flat" expectation.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parents[3] / "document/fig/transport_sweeps"
FIG.mkdir(parents=True, exist_ok=True)
OUT = Path(__file__).resolve().parent

# --- measured bubble wall (s/rank), 1 thread/rank --------------------------
# CNT(3,3) L=2 (k=1), stack/energy axis:
CNT_STACK = {1: 207.8, 2: 137.1, 4: 57.0, 8: 22.5, 16: 13.0, 32: 12.3}
# Si film nk=5 n_slabs=3 (k>1, 25 transverse q), stack/energy axis:
FILM_STACK = {1: 309.1, 4: 87.9, 8: 46.8}
# Si film nk=5, dedicated q axis (q_comm, stack=1) -- distributes the 25 ext-q:
FILM_QCOMM = {1: 309.9, 2: 180.7, 4: 198.9}

# Per-phase profile (CNT L=4, -np 1), s/iter and % of the SCBA iteration:
PROFILE = [
    ("3-phonon bubble (SigmaPhononPhonon)", 3741.5, 99.8),
    ("RGF Dyson solve (Selected Solve)", 1.42, 0.04),
    ("OBC (Sancho-Rubio)", 1.84, 0.05),
    ("stack<->nnz transpose", 3.14, 0.08),
    ("symmetrize/update/converge", 1.41, 0.04),
]


def curve(d):
    ks = sorted(d)
    base = d[ks[0]]
    return ks, [base / d[k] for k in ks], [100 * base / d[k] / k for k in ks]


def main():
    # CSV
    with open(OUT / "scaling_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "axis", "ranks", "bubble_s_per_rank", "speedup", "eff_pct"])
        for name, d in [("CNT33_L2", "stack"), ]:
            pass
        for (name, axis, d) in [("CNT33_L2", "stack", CNT_STACK),
                                ("SiFilm_nk5", "stack", FILM_STACK),
                                ("SiFilm_nk5", "qcomm", FILM_QCOMM)]:
            ks, sp, eff = curve(d)
            for k, s, e in zip(ks, sp, eff):
                w.writerow([name, axis, k, d[k], round(s, 2), round(e, 0)])
    print("wrote", OUT / "scaling_results.csv")

    # Figure: (a) per-phase pie-ish bar, (b) strong-scaling curves
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.4))

    labels = [p[0] for p in PROFILE]
    pcts = [p[2] for p in PROFILE]
    ax0.barh(range(len(labels)), pcts, color=["C3", "C0", "C2", "C1", "C7"])
    ax0.set_yticks(range(len(labels)))
    ax0.set_yticklabels(labels, fontsize=8)
    ax0.set_xlabel("% of SCBA iteration wall")
    ax0.set_xscale("log")
    ax0.set_xlim(0.01, 200)
    ax0.set_title("Per-phase profile (CNT L=4, -np 1)", fontsize=10)
    for i, p in enumerate(pcts):
        ax0.text(p * 1.1, i, f"{p:.2f}%", va="center", fontsize=7)

    for (name, d, mk, c) in [("CNT L=2 (stack)", CNT_STACK, "o-", "C0"),
                             ("Si film nk=5 (stack)", FILM_STACK, "s-", "C3"),
                             ("Si film nk=5 (q_comm)", FILM_QCOMM, "^--", "C2")]:
        ks, sp, _ = curve(d)
        ax1.plot(ks, sp, mk, color=c, label=name)
    mx = max(max(CNT_STACK), max(FILM_STACK), max(FILM_QCOMM))
    ax1.plot([1, mx], [1, mx], "k:", lw=0.8, label="ideal")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)
    ax1.set_xlabel("MPI ranks (1 thread/rank)")
    ax1.set_ylabel("bubble speedup")
    ax1.set_title("Strong scaling of the 3-phonon bubble", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "prod_phph_scaling.pdf")
    print("wrote", FIG / "prod_phph_scaling.pdf")


if __name__ == "__main__":
    main()
