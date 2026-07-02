"""Silicon thin-film ballistic kappa(L)=G*L figure (fig:res_sifilm, panel b).

Regenerates document/fig/transport_sweeps/si_film_conductance_b from the
SURVIVING record. The raw sweep JSONs of the original dense-solver runs
(phonon/scripts/out/si_film/si_film_ballistic{,_lconv}.json) were purged;
the surviving numerical record is the F23 entry of
phonon/docs/lab_notebook_archive.md (Part B, "Si thin-film cross-plane
kappa"), which archives three mesh-converged (nk=8, 121 frequencies,
eta_factor=0.1, 300 K, bulk-Si 2x2x2 FD force constants) ballistic points:

  G_ball = 1034 MW m^-2 K^-1  @ 4 layers ~ 1.55 nm  ("matched-mesh result")
  G_ball =  952 MW m^-2 K^-1  @ 5 uc     = 2.70 nm  ("5uc ballistic G_ball=952")
  G_ball =  928 MW m^-2 K^-1  @ 8 layers = 3.09 nm  (nk=4/8/12/16 -> 970/928/
                                                     921/920; nk=8 value)

Thickness conventions as recorded in document/src/results/40_film_hetero.tex
(1 uc = 5.4018 A; primitive layers ~0.39 nm: 4 layers ~ 1.55 nm, 8 layers
= 3.09 nm). kappa(L) = G*L in W m^-1 K^-1; a thin linear guide through the
first point makes the sub-linear growth visible, and G itself (declining)
is shown on a second axis.

Run:  python phonon/scripts/figures/si_film_conductance.py
Figure -> document/fig/transport_sweeps/si_film_conductance_b.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style

FIGDIR = ROOT / "document/fig/transport_sweeps"

# Archived F23 ballistic points (lab_notebook_archive.md, see docstring).
L_NM = [1.55, 2.70, 3.09]               # film thickness (nm)
G_MW = [1034.0, 952.0, 928.0]           # G_ball (MW m^-2 K^-1)
KAPPA = [g * 1e6 * L * 1e-9 for g, L in zip(G_MW, L_NM)]  # W m^-1 K^-1


def main():
    fig, ax = style.figure(width=4.0, height=3.0)
    # thin linear guide through the first point: kappa = G(1.55 nm) * L
    Lg = [0.0, 3.3]
    ax.plot(Lg, [G_MW[0] * 1e6 * L * 1e-9 for L in Lg], "-", color="0.6",
            lw=0.9, label=r"linear, $G(1.55\,\mathrm{nm})\,L$")
    ax.plot(L_NM, KAPPA, "o-", color="C0",
            label=r"$\kappa=G_\mathrm{ball}L$")
    ax.set_xlabel("film thickness $L$ (nm)")
    ax.set_ylabel(r"$\kappa=G_\mathrm{ball}L$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_xlim(0, 3.3)
    ax.set_ylim(0, 3.5)

    ax2 = ax.twinx()
    ax2.plot(L_NM, G_MW, "s--", color="C1", label=r"$G_\mathrm{ball}$")
    ax2.set_ylabel(r"$G_\mathrm{ball}$ (MW m$^{-2}$ K$^{-1}$)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax2.set_ylim(800, 1060)
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", framealpha=0.9)
    style.save(fig, "si_film_conductance_b", directory=FIGDIR)


if __name__ == "__main__":
    main()
