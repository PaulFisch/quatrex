"""MoS2 cross-plane kappa_z(t) against experiment (fig:res_mos2_kappa).

  mos2_kappa_sood   (a) kappa_z vs film thickness: our eta=0 ballistic
                    NEGF ladder (kappa = G t, thickness-independent
                    G) against the quasi-ballistic band implied by
                    the published TDTR asymptote and bulk spread of
                    Sood et al.; (b) the volumetric resistance
                    R = t/kappa: our ballistic bound is a
                    thickness-independent 6.8 m^2K/GW below the
                    measured ~10 m^2K/GW asymptote.

Data: phonon/scripts/data/film_kappa.csv (distilled by
_extract_film_kappa.py from the cluster/mos2f*b ballistic runs;
eta=0, 5x5 shifted transverse mesh, 262-pt non-uniform grid).
Experimental literals with provenance:
  Sood et al., Nano Lett 19, 2434 (2019), DOI
  10.1021/acs.nanolett.8b05174 -- films 20-240 nm; the volumetric
  thermal resistance "asymptotes to a non-zero value,
  ~10 m^2 K GW^-1"; bulk kappa_z measurements 2.0 (Liu), 2.5
  (Muratore), 4.8 (Jiang) W/mK; their DFT ~5.
  Lindroth & Erhart, PRB 94, 115205 (2016), DOI
  10.1103/PhysRevB.94.115205 -- BTE bulk kappa_z 5.1 W/mK.
The experiment band is the two-parameter quasi-ballistic form
R(t) = R_asym + t/kappa_bulk with R_asym = 10 m^2K/GW and
kappa_bulk in [2.0, 4.8] W/mK -- both parameters published, no fit
of ours.

Run:  python phonon/scripts/figures/mos2_kappa_sood.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.studies import style

DATA = ROOT / "phonon/scripts/data/film_kappa.csv"
FIGDIR = ROOT / "document/fig/transport_sweeps"

R_ASYM = 10.0e-9      # m^2 K / W  (Sood 2019 abstract)
K_BULK = (2.0, 4.8)   # W / m K    (bulk measurement spread)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(DATA.open())
            if r["system"] == "mos2" and r["kind"] == "ballistic"]
    t_nm = np.array([float(r["t_nm"]) for r in rows])
    G = np.array([float(r["G_W_m2K"]) for r in rows])
    order = np.argsort(t_nm)
    t_nm, G = t_nm[order], G[order]
    G_flat = float(np.median(G[-3:]))  # the ladder-grid value

    fig, (ax_a, ax_b) = style.figure(ncols=2, width=4.4, height=3.4)
    colors = style.RC["axes.prop_cycle"].by_key()["color"]

    tt = np.logspace(np.log10(2.0), np.log10(300.0), 200)  # nm
    ax_a.plot(tt, G_flat * tt * 1e-9, color=colors[0], lw=1.4,
              label=r"ballistic NEGF ($\kappa = G_\mathrm{ball}\,t$)")
    ax_a.plot(t_nm, G * t_nm * 1e-9, "o", color=colors[0], markersize=6)
    lo = tt * 1e-9 / (R_ASYM + tt * 1e-9 / K_BULK[0])
    hi = tt * 1e-9 / (R_ASYM + tt * 1e-9 / K_BULK[1])
    ax_a.fill_between(tt, lo, hi, color=colors[1], alpha=0.3, linewidth=0,
                      label="Sood 2019 asymptote + bulk spread")
    ax_a.axhspan(K_BULK[0], K_BULK[1], color="0.85", zorder=0)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("film thickness (nm)")
    ax_a.set_ylabel(r"$\kappa_z$ (W m$^{-1}$K$^{-1}$)")
    ax_a.axvspan(20, 240, color=colors[2], alpha=0.06, zorder=0)
    ax_a.legend(fontsize=7.5, loc="upper left")

    ax_b.axhline(1e9 / G_flat, color=colors[0], lw=1.4,
                 label="ballistic NEGF")
    ax_b.plot(t_nm, t_nm * 1e-9 / (G * t_nm * 1e-9) * 1e9, "o",
              color=colors[0], markersize=6)
    ax_b.axhline(R_ASYM * 1e9, color=colors[1], lw=1.2, ls="--",
                 label="measured asymptote")
    for kb in K_BULK:
        ax_b.plot(tt, (R_ASYM + tt * 1e-9 / kb) * 1e9, color=colors[1],
                  lw=0.8, alpha=0.5)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("film thickness (nm)")
    ax_b.set_ylabel(r"$R = t/\kappa_z$ (m$^2$K GW$^{-1}$)")
    ax_b.set_ylim(0, 40)
    ax_b.axvspan(20, 240, color=colors[2], alpha=0.06, zorder=0)
    ax_b.legend(fontsize=7.5, loc="upper left")

    style.save(fig, "mos2_kappa_sood", directory=FIGDIR)

    print(f"ballistic G (ladder grid): {G_flat:.4e} W/m^2/K -> "
          f"R = {1e9 / G_flat:.2f} m^2K/GW vs Sood asymptote "
          f"{R_ASYM * 1e9:.0f} m^2K/GW "
          f"({1e9 / G_flat / (R_ASYM * 1e9):.0%} of it)")
    for t, g in zip(t_nm, G):
        print(f"  t={t:6.2f} nm  G={g:.4e}  kappa={g * t * 1e-9:.3f} W/mK")
    print(f"kappa_z at 20/100/240 nm (ballistic): "
          f"{G_flat * 20e-9:.2f} / {G_flat * 100e-9:.2f} / "
          f"{G_flat * 240e-9:.2f} W/mK; "
          "experiment reaches the bulk 2-4.8 W/mK band instead -- the "
          "anharmonic resistance the SCBA must supply")


if __name__ == "__main__":
    main()
