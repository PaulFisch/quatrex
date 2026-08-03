"""Consolidated Si thin-film cross-plane conductance vs thickness (300 K).

Our ballistic + anharmonic NEGF with the 2x2x2 (QE-FD) and the high-quality
5x5x5 (~4th-neighbour, VASP-hiPhive) bulk-Si force constants, against the
anharmonic-NEGF reference values of Guo, Bescond & Zhang, PRB 102, 195412
(2020).

The raw sweep JSONs (scripts/out/si_film/*.json) are gone; the authoritative
surviving record is phonon/docs/lab_notebook_archive.md, from which every
number below is hard-coded:

  5x5x5 VASP FC3 sweep  (notebook F29, "si_film_kappa_bigfc3"; nk=8,
  121 freq, eta_factor 0.1, native prefactor; heat-flow cons. 5-8e-4):
      3 L (1.16 nm)  G_ball 907  G_anh 470
      5 L (1.93 nm)  G_ball 849  G_anh 378
      8 L (3.09 nm)  G_ball 775  G_anh 296        [MW m^-2 K^-1]

  2x2x2 QE-FD FC3  (notebook F23; nk=8, nfreq=121, eta_factor 0.1):
      ballistic: 1034 at 4 layers ~ 1.55 nm (~3 uc); 952 at 5 uc (2.70 nm);
                 928 at 8 layers (3.09 nm), the nk=8 entry of the 8-layer
                 mesh-convergence series nk=4/8/12/16 -> 970/928/921/920
                 (q-converged ~920).
      anharmonic (native prefactor): 570 at ~1.55 nm ONLY -- the small-eta
                 SCBA with this short-cutoff FC destabilises away from that
                 single thickness, so no series exists (do not invent one).
                 The retracted /4-convention values (831/704) are NOT plotted.

  Guo et al. 2020 (PRB 102, 195412; 1 uc = 5.4018 A):
      anharmonic 939.7 at 3 uc (1.62 nm), 890.97 at 5 uc (2.70 nm);
      ballistic NEGF 1065.81 at 5 uc (their convergence table).

  eta=0 coupled-q re-runs (2026-08-03, the finite-eta supersession):
      read live from phonon/scripts/data/film_kappa.csv (extractor
      _extract_film_kappa.py; runs cluster/sifilm{3,5,8}{b,s} --
      nk=9 shifted mesh, 121 freq, eta=0, converged fixed points
      70/52 iterations at L3/L8, 5x5x5 VASP constants, nk9r rebuilt
      post-min-image build). Slab here = the FCC-primitive transport
      cell (V/A_perp = 3.16 A), so thicknesses are 0.95/1.58/2.53 nm
      -- thinner per layer count than the old finite-eta series.
      The finite-eta curves above stay plotted, greyed, as the
      retracted record.

Review-mandated fixes vs the old plot_si_film_consolidated.py figure:
  (i)  legend moved OUTSIDE the axes (it hid the Guo 890.97 star and the
       last 2x2x2 ballistic point);
  (ii) legend typo "Guo et al.\\ 2020" -> "Guo et al. 2020";
  (iii) no in-figure title;  (iv) phonon.studies.style palette.

Run:  OMP_NUM_THREADS=1 python phonon/scripts/figures/si_film_vs_guo.py
Figure -> document/fig/transport_sweeps/si_film_vs_guo.{pdf,png}
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

# ---- data (MW m^-2 K^-1; thickness in nm) -- sources in the docstring ----
BIG_L = [1.16, 1.93, 3.09]           # 5x5x5 VASP FC3 (notebook F29)
BIG_BALL = [907.0, 849.0, 775.0]
BIG_ANH = [470.0, 378.0, 296.0]

TWO_BALL_L = [1.55, 2.70, 3.09]      # 2x2x2 QE-FD FC3 (notebook F23, nk=8)
TWO_BALL = [1034.0, 952.0, 928.0]
TWO_ANH_L = [1.55]                   # single surviving anharmonic point
TWO_ANH = [570.0]

GUO_ANH_L = [1.62, 2.70]             # Guo PRB 102, 195412 (2020)
GUO_ANH = [939.7, 890.97]
GUO_BALL_L = [2.70]
GUO_BALL = [1065.81]

C = ["#0173b2", "#de8f05", "#029e73", "#d55e00"]   # style palette


def _eta0_rows():
    import csv
    rows = [r for r in csv.DictReader(
        (ROOT / "phonon/scripts/data/film_kappa.csv").open())
        if r["system"] == "si"]
    out = {}
    for kind in ("ballistic", "scba"):
        pts = sorted((float(r["t_nm"]), 1e-6 * float(r["G_W_m2K"]))
                     for r in rows if r["kind"] == kind
                     and (kind == "ballistic" or r["converged"] == "True"))
        out[kind] = ([p[0] for p in pts], [p[1] for p in pts])
    return out


def main():
    fig, ax = style.figure(width=4.8, height=3.4)

    # retracted finite-eta record, greyed (see docstring)
    ax.plot(TWO_BALL_L, TWO_BALL, "o-", color="0.75",
            label=r"finite-$\eta$ $2{\times}2{\times}2$ (retracted)")
    ax.plot(TWO_ANH_L, TWO_ANH, "s", color="0.75", mfc="none", mew=1.4)
    ax.plot(BIG_L, BIG_BALL, "o-", color="0.55",
            label=r"finite-$\eta$ $5{\times}5{\times}5$ (retracted)")
    ax.plot(BIG_L, BIG_ANH, "s--", color="0.55")

    e0 = _eta0_rows()
    ax.plot(*e0["ballistic"], "o-", color=C[0], lw=1.6,
            label=r"$\eta=0$ coupled-$q$, ballistic")
    ax.plot(*e0["scba"], "s-", color=C[3], lw=1.6,
            label=r"$\eta=0$ coupled-$q$, anharmonic")

    ax.plot(GUO_ANH_L, GUO_ANH, "*", color="k", ms=11, ls="none",
            label="Guo et al. 2020, anharmonic")
    ax.plot(GUO_BALL_L, GUO_BALL, "*", color="k", ms=11, mfc="none",
            mew=1.2, ls="none", label="Guo et al. 2020, ballistic")

    ax.set_xlabel("film thickness $L$ (nm)")
    ax.set_ylabel(r"cross-plane conductance $G$ (MW m$^{-2}$K$^{-1}$)")
    ax.set_xlim(0.8, 3.3)
    ax.set_ylim(0, 1150)
    # legend OUTSIDE the axes so it cannot mask any data point (review fix)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    style.save(fig, "si_film_vs_guo", directory=FIGDIR)

    print("\n  plotted values (MW m^-2 K^-1):")
    for kind in ("ballistic", "scba"):
        L, G = e0[kind]
        print(f"  eta0 {kind:9s}: " + "  ".join(
            f"{l:.2f}nm={g:.1f}" for l, g in zip(L, G)))
    if e0["scba"][0]:
        rr = [g / e0["ballistic"][1][0] for g in e0["scba"][1]]
        print("  eta0 reduction: " + "  ".join(
            f"{1 - r:.1%}" for r in rr)
            + "  (Guo's own: 16.4% at 5 uc)")
    for lab, L, G in [("2x2x2 ball", TWO_BALL_L, TWO_BALL),
                      ("2x2x2 anh ", TWO_ANH_L, TWO_ANH),
                      ("5x5x5 ball", BIG_L, BIG_BALL),
                      ("5x5x5 anh ", BIG_L, BIG_ANH),
                      ("Guo   anh ", GUO_ANH_L, GUO_ANH),
                      ("Guo   ball", GUO_BALL_L, GUO_BALL)]:
        print(f"  {lab}: " + "  ".join(f"{l:.2f}nm={g:.1f}"
                                       for l, g in zip(L, G)))


if __name__ == "__main__":
    main()
