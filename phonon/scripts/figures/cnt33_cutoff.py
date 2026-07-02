"""(3,3) CNT cutoff-cube figure (fig:res_cnt_cutoff in document/src/results/30_cnt.tex).

Horizontal G_anh bars for the 8 corners of the (FC3-magnitude sigma,
vertex-range v, Green's-function-range g) cutoff cube at n_slabs=2, against the
full-coupling value (dashed) with its +11/-9% spread band shaded (the review
spec: "All lie within +11/-9% of the full-coupling value").

Data: phonon/scripts/out/cnt33_cutoff/summary.csv, produced by
phonon/scripts/verify/cnt33_cutoff_sweep.py (one row per corner, written
incrementally). Fails with a clear message if the csv is absent or the
full-coupling reference corner has not finished yet; plots partial sweeps
with a warning.

Run:  OMP_NUM_THREADS=1 python phonon/scripts/figures/cnt33_cutoff.py
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
from phonon.studies import style  # noqa: E402

CSV = ROOT / "phonon/scripts/out/cnt33_cutoff/summary.csv"
FIGDIR = ROOT / "document/fig/transport_sweeps"
MW = 1.0e6  # W/m^2/K -> MW/m^2/K
BAND = (0.91, 1.11)  # the reviewed +11/-9% spread about full coupling


def _is_inf(v: str) -> bool:
    return v.strip().lower().startswith("inf")


def _label(r: dict) -> str:
    def s(v: str) -> str:
        return r"\infty" if _is_inf(v) else v.strip()
    return (rf"$\sigma_{{{s(r['sigma_cutoff'])}}}\,"
            rf"v_{{{s(r['vertex_cutoff'])}}}\,g_{{{s(r['g_cutoff'])}}}$")


def main() -> None:
    if not CSV.exists():
        sys.exit(
            f"[cnt33_cutoff] {CSV} not found.\n"
            "Run the 8-corner sweep first:\n"
            "  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=8 QUATREX_PHPH_THREADS=2 \\\n"
            "  QUATREX_PHPH_MEMORY_GB=30 nohup python \\\n"
            "      phonon/scripts/verify/cnt33_cutoff_sweep.py --verbose \\\n"
            "      > phonon/scripts/out/cnt33_cutoff/sweep.log 2>&1 &")
    with open(CSV) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"[cnt33_cutoff] {CSV} is empty -- sweep not started?")

    full = [r for r in rows if _is_inf(r["sigma_cutoff"])
            and _is_inf(r["vertex_cutoff"]) and _is_inf(r["g_cutoff"])]
    if not full:
        sys.exit(
            f"[cnt33_cutoff] {CSV} has {len(rows)} row(s) but no full-coupling "
            "corner (sigma=vertex=g=Inf) yet -- the reference is required. "
            "Wait for the sweep (it runs the full corner first) and re-run.")
    if len(rows) < 8:
        print(f"[cnt33_cutoff] WARNING: only {len(rows)}/8 corners in "
              f"{CSV} -- plotting the partial sweep.", flush=True)
    bad = [r for r in rows if r["scba_converged"].strip() != "True"]
    for r in bad:
        print(f"[cnt33_cutoff] WARNING: corner s{r['sigma_cutoff']}_"
              f"v{r['vertex_cutoff']}_g{r['g_cutoff']} did NOT converge "
              f"(resid {r['scba_residual']}) -- bar shown hollow.", flush=True)

    ga = np.array([float(r["G_anh"]) for r in rows]) / MW
    g_full = float(full[0]["G_anh"]) / MW
    conv = np.array([r["scba_converged"].strip() == "True" for r in rows])
    labels = [_label(r) for r in rows]

    order = np.argsort(ga)
    y = np.arange(len(rows))
    fig, ax = style.figure(width=4.6, height=3.0)
    ax.axvspan(BAND[0] * g_full, BAND[1] * g_full, color="0.88", zorder=0,
               label=r"$+11/{-}9\%$ of full coupling")
    ax.barh(y, ga[order], height=0.62, zorder=2,
            color=["#0173b2" if conv[i] else "none" for i in order],
            edgecolor="#0173b2", linewidth=1.0)
    ax.axvline(g_full, color="k", lw=1.2, ls="--", zorder=3,
               label="full coupling")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order])
    ax.set_xlabel(r"$G_\mathrm{anh}\ (\mathrm{MW\,m^{-2}\,K^{-1}})$")
    ax.set_xlim(0.85 * ga.min(), 1.05 * max(ga.max(), BAND[1] * g_full))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    style.save(fig, "cnt33_cutoff", directory=FIGDIR)

    print(f"\nfull coupling  G_anh = {g_full:.3f} MW/m^2/K "
          f"(G_ball = {float(full[0]['G_ball']) / MW:.3f})")
    lo, hi = ga.min(), ga.max()
    print(f"corner span    {lo:.3f} -- {hi:.3f} MW/m^2/K "
          f"({(lo / g_full - 1) * 100:+.1f}% / {(hi / g_full - 1) * 100:+.1f}%)")
    for i in order[::-1]:
        r = rows[i]
        print(f"  s{r['sigma_cutoff']:>3}_v{r['vertex_cutoff']:>3}_"
              f"g{r['g_cutoff']:>3}  G_anh={ga[i]:7.3f}  "
              f"({(ga[i] / g_full - 1) * 100:+5.1f}%)  "
              f"conserv={float(r['conservation']):.1e}  "
              f"iters={r['n_scba_iter']}")


if __name__ == "__main__":
    main()
