"""Ballistic nanowire figures (results/20_nanowires, fig:res_ballistic_curves).

Regenerates ballistic_vs_T_d5_d11 and ballistic_vs_length_d5_d11 from the
fresh Caroli sweep (phonon/studies/out/ballistic_curves.npz; dense
Sancho-Rubio leads, eta_factor 0.3, 701-point grid, recomputed 2026-07-02).
Review-mandated redesign: the CNT series is dropped (it belongs to
sec:res_cnt and its scale buried both wire claims); the length panel is
normalised and the finite-eta coherence attenuation labelled as such.

Run:  python phonon/scripts/figures/ballistic_wires.py
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

NPZ = ROOT / "phonon/studies/out/ballistic_curves.npz"
FIGDIR = ROOT / "document/fig/transport_sweeps"
STYLES = {"d5a": dict(color="C0", marker="o"), "d11a": dict(color="C1", marker="s")}


def main():
    z = np.load(NPZ, allow_pickle=True)
    wire, ns, T, G = z["wire"], z["n_slabs"], z["T"], z["G_ball"]

    # G(T) at one cell
    fig, ax = style.figure(width=4.6, height=3.4)
    ratio = {}
    for w in ("d5a", "d11a"):
        m = (wire == w) & (ns == 1)
        order = np.argsort(T[m])
        Ts, Gs = T[m][order], G[m][order] / 1e7
        ax.plot(Ts, Gs, "-", label=w, **STYLES[w])
        ratio[w] = dict(zip(Ts, Gs))
    r300 = ratio["d11a"][300] / ratio["d5a"][300]
    ax.annotate(rf"$G_{{\rm d11a}}/G_{{\rm d5a}}={r300:.2f}$ at 300 K",
                (0.97, 0.08), xycoords="axes fraction", ha="right", fontsize=8)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"$G_\mathrm{ball}$ ($10^{7}$ W m$^{-2}$ K$^{-1}$)")
    ax.legend()
    style.save(fig, "ballistic_vs_T_d5_d11", directory=FIGDIR)
    print("ratios d11a/d5a:", {int(t): round(ratio['d11a'][t] / ratio['d5a'][t], 3)
                               for t in sorted(ratio['d5a'])})

    # G(L)/G(1) at 300 K
    fig, ax = style.figure(width=4.6, height=3.4)
    for w in ("d5a", "d11a"):
        m = (wire == w) & (T == 300)
        order = np.argsort(ns[m])
        Ls, Gs = ns[m][order], G[m][order]
        ax.plot(Ls, Gs / Gs[0], "-", label=w, **STYLES[w])
        print(w, "G(L)/G(1) at 300K:", np.round(Gs / Gs[0], 3), "L:", Ls)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4])
    ax.set_xticklabels(["1", "2", "4"])
    ax.set_xlabel("device length (transport cells)")
    ax.set_ylabel(r"$G_\mathrm{ball}(L)\,/\,G_\mathrm{ball}(1)$ (300 K)")
    ax.annotate("decay = finite-$\\eta$ coherence attenuation\n"
                "(the exact $\\eta\\to0$ Caroli $T(\\omega)$ is\n"
                "length-independent)", (0.03, 0.08),
                xycoords="axes fraction", fontsize=7)
    ax.legend()
    style.save(fig, "ballistic_vs_length_d5_d11", directory=FIGDIR)


if __name__ == "__main__":
    main()
