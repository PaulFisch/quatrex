"""FC3 tensor-decomposition transport quality on the d11a nanowire (fig:res_decomp).

Two figures (method x rank vs the dense reference), from cached run data only:
  d11a_decomp_ganh          anharmonic conductance vs FC3 parameter count
  d11a_decomp_conservation  heat-flow conservation residual vs parameter count

Data:  phonon/configs/sinw/reaps/sinw100_d11a_vasp_sc4/transport_quality/
       transport_quality.csv
Style: phonon/studies/style.py (unified).  Successor of the retired
       phonon/scripts/verify/plot_report_new_figs.py::decomposition_figures()
       (git 843c3069^), with the review-mandated improvements:
  (a) ranks r=2/4/8/16 annotated on the mSVD and INDSCAL series;
  (b) the negative-G mSVD rank-16 point drawn as an off-scale marker at the
      axis edge with a "G<0" annotation (not silently clipped);
  (c) legend entries for the X (conservation-violating) and open-circle
      (collapsed-to-ballistic) markers;
  (d) dense reference line labeled with its value;
  (e) conservation panel annotates that the ~1e-8 collapsed-to-ballistic
      points are trivially conserving (no anharmonic flow to conserve).

Run:  python phonon/scripts/figures/d11a_decomposition.py
Figures -> document/fig/transport_sweeps/d11a_decomp_{ganh,conservation}.{png,pdf}
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style

CSV = (ROOT / "phonon/configs/sinw/reaps/sinw100_d11a_vasp_sc4"
       / "transport_quality/transport_quality.csv")
FIGDIR = ROOT / "document/fig/transport_sweeps"
MW = 1.0e6  # W/m^2/K -> MW/m^2/K

METHODS = ["mSVD", "HOSVD", "CP", "INDSCAL", "Waring"]
COLORS = {m: f"C{i}" for i, m in enumerate(METHODS)}
ANNOTATED = ("mSVD", "INDSCAL")  # series that get r=... rank labels


def _rows() -> list[dict]:
    with open(CSV) as fh:
        return list(csv.DictReader(fh))


def _rank(r: dict) -> int:
    return int(str(r["rank"]).split("_")[0])


def ganh_figure(rows: list[dict], dense: dict) -> None:
    g_dense = float(dense["G_anh_W_per_m2_K"]) / MW
    g_ball = float(dense["G_ball_W_per_m2_K"]) / MW
    n_dense = float(dense["n_params"])

    fig, ax = style.figure(width=4.6, height=3.5)
    ax.axhline(g_dense, color="k", lw=1.3)
    ax.annotate(f"dense FC3: $G_\\mathrm{{anh}}={g_dense:.1f}$",
                (1.9e6, g_dense + 4), fontsize=7.5, color="k")
    ax.axhline(g_ball, color="0.55", lw=1.0, ls=":")
    ax.annotate(f"ballistic: $G_\\mathrm{{ball}}={g_ball:.1f}$",
                (1.9e6, g_ball + 4), fontsize=7.5, color="0.45")

    ylo, yhi = -20.0, 205.0
    for m in METHODS:
        mr = [r for r in rows if r["method"] == m]
        c = COLORS[m]
        x = [float(r["n_params"]) for r in mr]
        y = [float(r["G_anh_W_per_m2_K"]) / MW for r in mr]
        y_draw = [max(yi, ylo) for yi in y]  # off-scale points pinned to edge
        ax.plot(x, y_draw, "-", color=c, alpha=0.6, zorder=1)
        for r, xi, yi in zip(mr, x, y):
            unphysical = float(r["conservation_err"]) > 0.5 or yi < 0
            collapsed = r["ballistic_collapse"] == "True"
            if yi < ylo:  # (b) off-scale negative G: edge marker + annotation
                ax.plot(xi, ylo, "v", color=c, ms=8, mew=1.5,
                        clip_on=False, zorder=4)
                ax.annotate(f"$r={_rank(r)}$: $G<0$ ({yi:.0f})",
                            (xi * 1.25, ylo + 4), fontsize=7, color=c,
                            ha="left", zorder=4)
                continue  # rank already in the annotation
            elif unphysical:
                ax.plot(xi, yi, "x", color=c, ms=8, mew=2, zorder=3)
            elif collapsed:
                ax.plot(xi, yi, "o", mfc="white", mec=c, ms=7, zorder=3)
            else:
                ax.plot(xi, yi, "o", color=c, ms=5.5, zorder=3)
            if m in ANNOTATED:  # (a) rank labels
                if m == "mSVD":
                    dx, dy, ha = 2, 9, "left"
                else:
                    dx, dy, ha = -5, -12, "right"
                ax.annotate(f"$r={_rank(r)}$", (xi, max(yi, ylo)),
                            textcoords="offset points", xytext=(dx, dy),
                            fontsize=6.5, color=c, ha=ha)
        ax.plot([], [], "o", color=c, label=m)  # legend proxy
    # (c) marker-semantics legend entries
    ax.plot([], [], "x", color="0.3", ms=7, mew=1.8,
            label="violates conservation")
    ax.plot([], [], "o", mfc="white", mec="0.3", ms=6.5,
            label="collapsed to ballistic")
    ax.plot([], [], "v", color="0.3", ms=7, label=r"$G<0$ (off scale)")
    ax.axhline(0.0, color="0.75", lw=0.7)

    ax.set_xscale("log")
    ax.set_xlim(0.8e3, 1.5 * n_dense)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("number of FC3 parameters")
    ax.set_ylabel(r"$G_\mathrm{anh}\ (\mathrm{MW\,m^{-2}\,K^{-1}})$")
    ax.legend(ncol=2, fontsize=7, loc="lower left")
    style.save(fig, "d11a_decomp_ganh", directory=FIGDIR)


def conservation_figure(rows: list[dict], dense: dict) -> None:
    fig, ax = style.figure(width=4.6, height=3.5)
    e_dense = float(dense["conservation_err"])
    ax.axhline(e_dense, color="k", lw=1.3)
    ax.annotate(f"dense FC3: {e_dense:.1e}", (1.15e6, e_dense * 1.35),
                fontsize=7.5, color="k", ha="right")

    collapsed_pts = []
    for m in METHODS:
        mr = [r for r in rows if r["method"] == m]
        c = COLORS[m]
        x = [float(r["n_params"]) for r in mr]
        y = [float(r["conservation_err"]) for r in mr]
        ax.plot(x, y, "-", color=c, alpha=0.6, zorder=1, label=m)
        for r, xi, yi in zip(mr, x, y):
            if float(r["conservation_err"]) > 0.5:
                ax.plot(xi, yi, "x", color=c, ms=8, mew=2, zorder=3)
            elif r["ballistic_collapse"] == "True":
                ax.plot(xi, yi, "o", mfc="white", mec=c, ms=7, zorder=3)
                collapsed_pts.append((xi, yi))
            else:
                ax.plot(xi, yi, "o", color=c, ms=5.5, zorder=3)
    ax.plot([], [], "x", color="0.3", ms=7, mew=1.8,
            label="violates conservation")
    ax.plot([], [], "o", mfc="white", mec="0.3", ms=6.5,
            label="collapsed to ballistic")

    # (e) the ~1e-8 cluster conserves trivially (no anharmonic flow at all)
    cx = max(x for x, _ in collapsed_pts)
    cy = max(y for _, y in collapsed_pts)
    ax.annotate("collapsed to ballistic:\ntrivially conserving",
                xy=(cx, cy), xytext=(2.5e4, 3e-7), fontsize=7, color="0.25",
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.8))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 4)
    ax.set_xlabel("number of FC3 parameters")
    ax.set_ylabel("heat-flow conservation residual")
    ax.legend(ncol=2, fontsize=7, loc="lower right")
    style.save(fig, "d11a_decomp_conservation", directory=FIGDIR)


def main() -> None:
    rows = _rows()
    dense = next(r for r in rows if r["method"] == "dense")
    fits = [r for r in rows if r["method"] != "dense"]

    # console cross-check of the report's claims
    n_dense = float(dense["n_params"])
    print("=" * 64)
    print("d11a FC3 decomposition transport quality (vs report claims)")
    print("=" * 64)
    print(f"dense: G_anh={float(dense['G_anh_W_per_m2_K'])/MW:.2f} "
          f"G_ball={float(dense['G_ball_W_per_m2_K'])/MW:.2f} MW/m^2/K, "
          f"n_params={int(n_dense)}")
    for r in fits:
        if _rank(r) == 16:
            print(f"{r['method']:8s} r=16: G err {float(r['G_anh_rel_err_vs_dense'])*100:6.1f}%, "
                  f"frob {float(r['frob_rel_err'])*100:5.1f}%, "
                  f"params x{n_dense/float(r['n_params']):7.0f} fewer, "
                  f"cons {float(r['conservation_err']):.2e}")
    msvd16 = next(r for r in fits if r["method"] == "mSVD" and _rank(r) == 16)
    ind16 = next(r for r in fits if r["method"] == "INDSCAL" and _rank(r) == 16)
    print(f"mSVD/INDSCAL param ratio at r=16: "
          f"{float(msvd16['n_params'])/float(ind16['n_params']):.0f}x denser")
    print(f"mSVD r=16 G_anh = {float(msvd16['G_anh_W_per_m2_K'])/MW:.1f} "
          f"MW/m^2/K (negative)")

    ganh_figure(rows, dense)
    conservation_figure(rows, dense)


if __name__ == "__main__":
    main()
