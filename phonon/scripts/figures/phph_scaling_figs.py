"""Production-scaling figures for results/90_scaling.tex (consolidated).

Regenerates the four figures of \\cref{fig:res_phph_scaling,fig:res_phph_prod,
fig:res_phph_memory}:

  phph_scaling       -- earlier gather-based kernel: block vs energy/stack axis
                        (Gamma-only conductors); 5.4x at 6 ranks on the block axis.
  phph_q_scaling     -- earlier gather-based kernel: external-q communicator vs the
                        replicated energy/stack axis; 6.6x at 8 ranks.
  prod_phph_scaling  -- production kernel: per-phase profile (bubble = 99.8%) +
                        strong scaling of the bubble (near-ideal stack to ~16 ranks).
  phph_memory        -- dense vs streamed 3-phonon vertex peak memory vs N_q, with a
                        measured tracemalloc point and the 80 GB single-GPU line.

Data provenance
---------------
* phph_scaling: wall times measured by the retired verify/phph_dist_scaling.py
  (NBLK=6, BS=16, NE=96); literals recovered from
  verify/plot_phph_scaling.py at commit 843c3069^.
* phph_q_scaling: wall times measured by the retired verify/phph_q_dist_scaling.py
  (4x4 transverse mesh) and the F22 replicated-stack measurement; literals
  recovered from verify/plot_qresolved.py at commit 843c3069^.
* prod_phph_scaling: bubble wall times read from
  phonon/scripts/data/prod_scaling_results.csv (restored from
  prod/scaling_results.csv at commit 843c3069^; measured on the 256-core node,
  1 thread/rank, --bind-to core). The per-phase profile literals are from the
  same campaign (CNT(3,3) L=4 single-rank SCBA iteration), recovered from
  prod/scaling_results.py at commit 843c3069^.
* phph_memory: analytic O(N_q^2 n_dof^3) / O(N_q n_dof^2 dim_t + n_dof^3) curves;
  the measured point is a tracemalloc run of the dense-vs-streamed Phi build
  executed by this script (deterministic rng seed 0).

Run:  python phonon/scripts/figures/phph_scaling_figs.py
Figures -> document/fig/transport_sweeps/<name>.{pdf,png}
"""
from __future__ import annotations

import csv
import sys
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style

FIGDIR = ROOT / "document/fig/transport_sweeps"
CSV = ROOT / "phonon/scripts/data/prod_scaling_results.csv"


# ----------------------------------------------------------------------
# (1) phph_scaling: earlier gather-based kernel, block vs energy/stack axis.
# Measured by verify/phph_dist_scaling.py (NBLK=6, BS=16, NE=96); recovered
# from plot_phph_scaling.py @ 843c3069^.
# ----------------------------------------------------------------------
def fig_phph_scaling():
    energy_np, energy_t = [1, 2, 4], [4.13, 4.27, 4.64]
    block_np, block_t = [1, 2, 3, 6], [4.13, 2.72, 1.74, 0.77]
    t1 = block_t[0]

    fig, ax = style.figure(width=4.4, height=3.3)
    ax.plot(block_np, [t1 / t for t in block_t], "s-", color="#029e73",
            label="block-parallel (the $(I,J)$ loop)")
    ax.plot(energy_np, [t1 / t for t in energy_t], "o-", color="#d55e00",
            label="energy/stack-parallel (bubble replicated)")
    rr = np.array([1, 2, 3, 4, 6])
    ax.plot(rr, rr, "k:", lw=1.0, label="ideal")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("phph self-energy speed-up")
    ax.set_xticks([1, 2, 3, 4, 6])
    ax.legend()
    style.save(fig, "phph_scaling", directory=FIGDIR)
    print(f"  block speed-up at 6 ranks: {t1 / block_t[-1]:.2f}x "
          f"(caption: 5.4x)")
    print(f"  energy/stack speed-up at 4 ranks: {t1 / energy_t[-1]:.2f}x "
          "(caption: no speed-up)")


# ----------------------------------------------------------------------
# (2) phph_q_scaling: earlier gather-based kernel, external-q communicator.
# Measured by verify/phph_q_dist_scaling.py (4x4 transverse mesh) + the F22
# replicated-stack curve; recovered from plot_qresolved.py @ 843c3069^.
# ----------------------------------------------------------------------
def fig_phph_q_scaling():
    ranks = [1, 2, 4, 8]
    wall = [5.300, 2.827, 1.479, 0.805]
    stack_speedup = [1.0, 0.97, 0.89, 0.85]  # F22: replicated bubble FFT
    speedup = [wall[0] / w for w in wall]

    fig, ax = style.figure(width=4.4, height=3.3)
    ax.plot(ranks, speedup, "o-", color="#0173b2",
            label="distributed external-$q$ (all-gather internal $G$)")
    ax.plot(ranks, stack_speedup, "s--", color="#d55e00",
            label="energy/stack axis (replicated)")
    ax.plot(ranks, ranks, "k:", lw=1.0, label="ideal")
    ax.set_xlabel("MPI ranks on the $q$-communicator")
    ax.set_ylabel("phph self-energy speed-up")
    ax.set_xticks(ranks)
    ax.legend()
    style.save(fig, "phph_q_scaling", directory=FIGDIR)
    print(f"  q-axis speed-up at 8 ranks: {speedup[-1]:.2f}x (caption: 6.6x)")


# ----------------------------------------------------------------------
# (3) prod_phph_scaling: production kernel profile + strong scaling.
# Scaling curves from data/prod_scaling_results.csv; profile literals from
# prod/scaling_results.py @ 843c3069^ (CNT(3,3) L=4, single rank).
# ----------------------------------------------------------------------
PROFILE = [  # (phase, s/iter, % of the SCBA iteration wall)
    ("3-phonon bubble (SigmaPhononPhonon)", 3741.5, 99.8),
    ("RGF Dyson solve (Selected Solve)", 1.42, 0.04),
    ("OBC (Sancho-Rubio)", 1.84, 0.05),
    ("stack$\\leftrightarrow$nnz transpose", 3.14, 0.08),
    ("symmetrize/update/converge", 1.41, 0.04),
]


def fig_prod_phph_scaling():
    series: dict[tuple[str, str], dict[int, float]] = {}
    for r in csv.DictReader(open(CSV)):
        series.setdefault((r["system"], r["axis"]), {})[int(r["ranks"])] = \
            float(r["bubble_s_per_rank"])

    def curve(d):
        ks = sorted(d)
        return ks, [d[ks[0]] / d[k] for k in ks]

    fig, (ax0, ax1) = style.figure(ncols=2, width=4.6, height=3.4)

    labels = [p[0] for p in PROFILE]
    pcts = [p[2] for p in PROFILE]
    ax0.barh(range(len(labels)), pcts,
             color=["#d55e00", "#0173b2", "#029e73", "#de8f05", "#949494"])
    ax0.set_yticks(range(len(labels)))
    ax0.set_yticklabels(labels, fontsize=7.5)
    ax0.invert_yaxis()
    ax0.set_xlabel("% of SCBA iteration wall")
    ax0.set_xscale("log")
    ax0.set_xlim(0.01, 400)
    ax0.set_title("Per-phase profile (CNT $L=4$, single rank)", fontsize=9)
    for i, p in enumerate(pcts):
        ax0.text(p * 1.25, i, f"{p:.2f}%", va="center", fontsize=7)

    for name, key, mk, c in [
            ("CNT $L=2$ (stack)", ("CNT33_L2", "stack"), "o-", "#0173b2"),
            ("Si film $n_k=5$ (stack)", ("SiFilm_nk5", "stack"), "s-", "#d55e00"),
            ("Si film $n_k=5$ ($q$_comm)", ("SiFilm_nk5", "qcomm"), "^--", "#029e73")]:
        ks, sp = curve(series[key])
        ax1.plot(ks, sp, mk, color=c, label=name)
    mx = max(k for d in series.values() for k in d)
    ax1.plot([1, mx], [1, mx], "k:", lw=0.8, label="ideal")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)
    ax1.set_xlabel("MPI ranks (1 thread/rank)")
    ax1.set_ylabel("bubble speedup")
    ax1.set_title("Strong scaling of the 3-phonon bubble", fontsize=9)
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")
    style.save(fig, "prod_phph_scaling", directory=FIGDIR)

    cnt = series[("CNT33_L2", "stack")]
    print(f"  bubble share of the iteration: {PROFILE[0][2]:.1f}% "
          "(caption: 99.8%)")
    print(f"  CNT stack speed-up at 16 ranks: {cnt[1] / cnt[16]:.2f}x "
          "(caption: near-ideal to ~16 ranks)")


# ----------------------------------------------------------------------
# (4) phph_memory: dense vs streamed vertex peak memory vs N_q.
# Analytic curves; measured point via tracemalloc (deterministic).
# ----------------------------------------------------------------------
def _measure_phi_peak(n_dof, N_q, dim_t, stream):
    rng = np.random.default_rng(0)
    T_arr = (rng.standard_normal((N_q, n_dof, dim_t))
             + 1j * rng.standard_normal((N_q, n_dof, dim_t)))
    M_blocks = rng.standard_normal((n_dof, dim_t, dim_t)).astype(complex)
    tracemalloc.start()
    TM = np.einsum("qci,aij->qacj", T_arr, M_blocks)
    T_arr_H = T_arr.conj().transpose(0, 2, 1).copy()
    if not stream:
        Phi_all = np.einsum("qacj,rjd->qracd", TM, T_arr_H)  # noqa: F841
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def fig_phph_memory():
    N_q_meas, n_dof_meas, dim_t_meas = 8, 63, 189
    peak_dense = _measure_phi_peak(n_dof_meas, N_q_meas, dim_t_meas, stream=False)
    peak_stream = _measure_phi_peak(n_dof_meas, N_q_meas, dim_t_meas, stream=True)
    print(f"  measured (n_dof={n_dof_meas}, N_q={N_q_meas}, dim_t={dim_t_meas}): "
          f"dense {peak_dense / 1e6:.1f} MB, streamed {peak_stream / 1e6:.1f} MB, "
          f"ratio {peak_dense / peak_stream:.1f}x")

    # NOTE: the recovered original also drew n_dof=6 curves; they sit three
    # decades below the n_dof=63 story and only compressed the axis, so they
    # were dropped (2026-07 figure review).
    fig, ax = style.figure(width=4.6, height=3.4)
    Nq = np.arange(2, 33)
    n_dof, dim_t = 63, 189
    dense = Nq**2 * n_dof**3 * 16 / 1e9
    stream = (Nq * n_dof**2 * dim_t * 16 + n_dof**3 * 16) / 1e9
    ax.plot(Nq, dense, "-", color="#d55e00", label="dense $\\Phi(q_1,q_2)$")
    ax.plot(Nq, stream, "--", color="#0173b2", label="streamed $\\Phi$")
    ax.axhline(80, color="gray", ls=":", lw=1.0, label="80 GB GPU")
    ax.plot([N_q_meas], [peak_dense / 1e9], "k*", ms=11, label="measured (dense)")
    ax.set_yscale("log")
    ax.set_xlabel("transverse $q$-mesh count $N_q$")
    ax.set_ylabel("peak vertex memory (GB)")
    ax.set_title(f"$n_\\mathrm{{dof}}={n_dof}$, $\\dim T={dim_t}$", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    style.save(fig, "phph_memory", directory=FIGDIR)

    dense32 = 32**2 * 63**3 * 16 / 1e9
    stream32 = (32 * 63**2 * 189 * 16 + 63**3 * 16) / 1e9
    print(f"  analytic n_dof=63 @ N_q=32: dense {dense32:.1f} GB, "
          f"streamed {stream32:.2f} GB, ratio {dense32 / stream32:.1f}x "
          "(caption: few-GB dense, streaming ~an order of magnitude lower, "
          "under the 80 GB line)")


def main():
    for name, fn in [("phph_scaling", fig_phph_scaling),
                     ("phph_q_scaling", fig_phph_q_scaling),
                     ("prod_phph_scaling", fig_prod_phph_scaling),
                     ("phph_memory", fig_phph_memory)]:
        print(f"[{name}]")
        fn()


if __name__ == "__main__":
    main()
