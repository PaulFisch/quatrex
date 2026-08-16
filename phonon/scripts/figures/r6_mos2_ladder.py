"""Results 6: the MoS2 cross-plane thickness ladder against experiment.

R(t) = R_c + t/kappa_bulk separates the contact resistance from the bulk
conductivity, which no single thickness can do -- R_c is 80-92% of the total
resistance across this ladder, so a kappa_z read off one film is mostly
interface.

Three series: the harmonic bound (thickness-independent R), the sparsifying
(ARD) vertex ladder at three thicknesses, and the least-squares vertex at two.
The two-point fits use the SAME pair of thicknesses, so the gap between them
is the force-constant fit alone.

Every number comes from phonon.studies._kappa_z_ladder.read_run(), the same
function the text report uses, so the figure cannot drift from the table.

Run:  python phonon/scripts/figures/r6_mos2_ladder.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QTX_ARRAY_MODULE", "numpy")

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT.parent, ROOT.parent / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np                                     # noqa: E402

from phonon.studies import style                       # noqa: E402
from phonon.studies._kappa_z_ladder import read_run    # noqa: E402

CLUSTER = ROOT.parent / "cluster"

#: The sparsifying-vertex ladder and the least-squares rebuild of its two
#: shorter rungs. Both are mask-clean (cutoff > device span); read_run
#: refuses them otherwise.
SERIES = [
    ("sparsifying (ARD)", ["cvM2b", "cvM4e", "cvM6b"], style.C_ANHARMONIC, "o"),
    ("least squares", ["lsM2f", "lsM4f"], style.C_BALLISTIC, "s"),
]

#: Sood et al. 2019: the quasi-ballistic volumetric-resistance asymptote and
#: the spread of bulk cross-plane conductivity measurements.
R_ASYM_M2KGW = 10.0
KAPPA_BULK_EXPT = (2.0, 4.8)

#: The harmonic bound of sub:res_coherent_film, mesh-extrapolated.
R_BALLISTIC_M2KGW = 7.2


def fit(t_nm, r_m2kgw):
    """R(t) = R_c + t/kappa. Returns (kappa [W/m/K], R_c [m2K/GW])."""
    slope, intercept = np.polyfit(np.asarray(t_nm) * 1e-9,
                                  np.asarray(r_m2kgw) * 1e-9, 1)
    return 1.0 / slope, intercept * 1e9


def main() -> int:
    fig, (ax_r, ax_k) = style.doc_figure(ncols=2, aspect=0.40)
    t_line = np.linspace(0.0, 8.5, 100)
    # The right panel runs out to the experimental window, so the fitted
    # curves can be seen approaching their own kappa_bulk -- which is the
    # quantity the measurement constrains, and which the 2-7 nm films do
    # not themselves display.
    t_far = np.logspace(np.log10(1.0), np.log10(400.0), 300)

    # --- the measured anchors, drawn first so the series sit on top ---
    ax_r.axhline(R_ASYM_M2KGW, color=style.C_REFERENCE, lw=1.0, ls="--")
    ax_r.text(0.15, R_ASYM_M2KGW - 0.25, "measured asymptote", fontsize=8,
              color=style.C_REFERENCE, ha="left", va="top")
    ax_r.axhline(R_BALLISTIC_M2KGW, color=style.C_THIRD, lw=1.0, ls=":")
    ax_r.text(0.15, R_BALLISTIC_M2KGW - 0.25, "harmonic bound", fontsize=8,
              color=style.C_THIRD, ha="left", va="top")

    ax_k.axhspan(*KAPPA_BULK_EXPT, color=style.C_REFERENCE, alpha=0.16, lw=0)
    ax_k.text(1.3, np.mean(KAPPA_BULK_EXPT), "measured bulk", fontsize=8,
              color=style.C_REFERENCE, va="center")
    ax_k.axvspan(20.0, 240.0, color=style.C_THIRD, alpha=0.10, lw=0)
    ax_k.text(70.0, 0.13, "experimental\nwindow", fontsize=8,
              color=style.C_THIRD, ha="center", va="bottom")

    for label, names, colour, marker in SERIES:
        rows = [read_run(CLUSTER / n) for n in names]
        rows.sort(key=lambda r: r["t_nm"])
        t = np.array([r["t_nm"] for r in rows])
        R = np.array([r["R_m2KGW"] for r in rows])
        kappa, r_c = fit(t, R)

        ax_r.plot(t, R, marker, color=colour, ms=6, zorder=3,
                  label=f"{label}: $R_c$={r_c:.2f}, "
                        rf"$\kappa$={kappa:.2f}")
        ax_r.plot(t_line, r_c + t_line * 1e-9 / kappa * 1e9, "-",
                  color=colour, lw=1.2, alpha=0.85)
        ax_k.plot(t, t / R, marker, color=colour, ms=6, zorder=3, label=label)
        ax_k.plot(t_far, t_far / (r_c + t_far * 1e-9 / kappa * 1e9),
                  "-", color=colour, lw=1.2, alpha=0.85)
        ax_k.axhline(kappa, color=colour, lw=0.8, ls="--", alpha=0.7)

        print(f"  {label:20s} n={len(rows)}  kappa_bulk={kappa:.4f} W/m/K  "
              f"R_c={r_c:.4f} m2K/GW")
        for r in rows:
            print(f"      t={r['t_nm']:.4f} nm  J={r['j_raw']:9.4f}  "
                  f"R={r['R_m2KGW']:.4f}  bar={r['j_rel_halfrange']:.2%}")

    # the harmonic bound has no bulk term, so its kappa_z grows without limit
    ax_k.plot(t_far, t_far / R_BALLISTIC_M2KGW, ":", color=style.C_THIRD,
              lw=1.0, label="harmonic bound")

    ax_r.set_xlabel("thickness $t$ (nm)")
    ax_r.set_ylabel(r"$R = t/\kappa_z$ (m$^2$K GW$^{-1}$)")
    ax_r.set_xlim(0, 8.5)
    ax_r.set_ylim(6, 17.5)
    ax_r.legend(loc="upper left", framealpha=0.9, fontsize=7.5)

    ax_k.set_xscale("log")
    ax_k.set_xlabel("thickness $t$ (nm)")
    ax_k.set_ylabel(r"$\kappa_z$ (W m$^{-1}$K$^{-1}$)")
    ax_k.set_xlim(1.0, 400.0)
    ax_k.set_ylim(0, 5.2)
    ax_k.legend(loc="upper left", framealpha=0.9, fontsize=7.5)

    style.save(fig, "r6_mos2_ladder", directory=style.DOC_FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
