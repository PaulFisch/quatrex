"""Summarize PRODUCTION phonon-transport runs into conductances + the legacy
CSV schema the dense-era plotters consume.

and the run dir holding ``<tag>_ball.npz`` / ``<tag>_anh.npz``.
Usage:
emits ``<out>/summary.csv`` (legacy cols) + ``<out>/summary.json``; ``--plot``
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

HBAR_SI = 1.054571817e-34
THZ_TO_RAD = 2.0 * np.pi * 1e12


def cross_section_area(structure_xyz, tdir):
    """Transverse cell area [m^2] from the extended-XYZ Lattice header (the same
    quantity as dense._cross_section_area)."""
    line2 = Path(structure_xyz).read_text().splitlines()[1]
    m = re.search(r'Lattice="([^"]+)"', line2)
    lat = np.array([float(x) for x in m.group(1).split()]).reshape(3, 3)
    ti = "xyz".index(tdir)
    perp = [i for i in range(3) if i != ti]
    return np.linalg.norm(np.cross(lat[perp[0]], lat[perp[1]])) * 1e-20


def lead_heat(npz):
    """q-summed lead heat current per device interface; lead value = index 0.

    Prefers the SCBA's all-reduced ``last_heat`` (exact at any stack split,
    already q-summed); ``final_heat`` is the rank-0-local frequency slice and
    is only complete for stack=1 runs (kept as fallback for ballistic runs,
    where the SCBA loop exits before tracking last_heat)."""
    d = np.load(npz, allow_pickle=True)
    lh = d.get("last_heat")
    if lh is not None and np.isfinite(np.asarray(lh)).all():
        js = np.asarray(lh).reshape(-1)
        return dict(lead0=float(abs(js[0])), lead1=float(abs(js[-1])),
                    per_interface=js.tolist(),
                    finite=True,
                    converged=bool(d.get("converged", False)),
                    n_iter=int(d.get("n_iter", -1)),
                    internal_spread=float(d.get("internal_spread", np.nan)))
    fh = d.get("final_heat")
    if fh is None:
        return None
    fh = np.asarray(fh)
    js = np.nansum(fh.reshape(-1, fh.shape[-1]), axis=0)
    out = dict(lead0=float(js[0]), lead1=float(js[-1]),
               per_interface=js.tolist(),
               finite=bool(np.isfinite(fh).all()),
               converged=bool(d.get("converged", False)),
               n_iter=int(d.get("n_iter", -1)))
    out["internal_spread"] = float(d.get("internal_spread", np.nan))
    # Heat-key convention: uniform grids store the legacy unweighted sum
    # (needs the * dw of g_const); non-uniform runs already fold the cell
    # widths in, so the keys ARE integrals (skip the dw factor).
    out["uniform_grid"] = bool(d.get("uniform_frequency_grid", True))
    return out


def g_const(cell, run_dir):
    """Analytic dense-matching constant C such that G[W/m^2/K] = C * lead0.

    lead0 is the q-SUMMED heat current (the SCBA sums the transverse mesh),
    while the dense reference and A_c are per primitive cell: divide by the
    mesh size N_q (Gamma runs: 1).
    """
    dw = (cell["fmax"] - cell.get("emin", 0.0)) / (cell["nfreq"] - 1)  # THz
    A_c = cross_section_area(Path(run_dir) / cell["tag_dir"] / "structure.xyz"
                             if cell.get("tag_dir") else
                             Path(cell["work"]) / "structure.xyz", cell["tdir"])
    n_q = int(np.prod(cell.get("nk", (1, 1)))) or 1
    return (dw * 1e12 * HBAR_SI * THZ_TO_RAD) / (A_c * cell["dt"] * n_q), A_c, dw


def summarize(run_dir, out_dir, do_plot):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cells = manifest["cells"] if isinstance(manifest, dict) else manifest
    rows = []
    for c in cells:
        bp = run_dir / f"{c['tag']}_ball.npz"
        ap = run_dir / f"{c['tag']}_anh.npz"
        b = lead_heat(bp) if bp.exists() else None
        a = lead_heat(ap) if ap.exists() else None
        if b is None:
            print(f"[skip] {c['tag']}: no ballistic npz")
            continue
        C, A_c, dw = g_const(c, run_dir)
        # Non-uniform (file-grid) runs store the heat INTEGRAL, not the
        # unweighted sum: drop the dw factor of the conversion for them.
        C_b = C if b.get("uniform_grid", True) else C / dw
        row = dict(tag=c["tag"], system=c["system"], sweep=c["sweep"],
                   t_mean=c["t_mean"], n_slabs=c["n_slabs"],
                   G_ball_W_per_m2_K=C_b * b["lead0"], A_c=A_c, dw_THz=dw)
        if a is not None:
            # The anharmonic conductance is the CONVERGED fixed point. A
            # non-converged run reports its last iterate with
            # anh_converged = False -- it is NOT silently rescued by a
            # cherry-picked best-conserved iterate (that iterate is not a
            # fixed point). Filter on anh_converged when reading the CSV.
            anh0 = a["lead0"]
            C_a = C if a.get("uniform_grid", True) else C / dw
            row["G_anh_W_per_m2_K"] = C_a * anh0
            row["G_anh"] = C_a * anh0
            row["ratio"] = (C_a * anh0) / (C_b * b["lead0"])
            row["anh_converged"] = a["converged"]
            row["anh_n_iter"] = a["n_iter"]
            row["lead_conservation"] = (abs(a["lead0"] - a["lead1"])
                                        / abs(a["lead0"]) if a["lead0"] else np.nan)
            row["eta_dip"] = a.get("internal_spread", np.nan)
        rows.append(row)
        rr = row.get("ratio")
        print(f"{c['tag']:24s} G_ball={row['G_ball_W_per_m2_K']:.3e} "
              + (f"G_anh={row['G_anh_W_per_m2_K']:.3e} ratio={rr:.3f} "
                 f"conv={row.get('anh_converged')} cons={row.get('lead_conservation'):.1e}"
                 if rr is not None else "(ballistic only)"))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["tag", "system", "sweep", "t_mean", "n_slabs",
            "G_ball_W_per_m2_K", "G_anh_W_per_m2_K", "G_anh", "ratio",
            "anh_converged", "anh_n_iter", "lead_conservation", "eta_dip",
            "A_c", "dw_THz"]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump(rows, open(out_dir / "summary.json", "w"), indent=2, default=float)
    print(f"\nwrote {out_dir/'summary.csv'} ({len(rows)} rows)")

    if do_plot and rows:
        _plot(rows, out_dir)
    return rows


def _plot(rows, out_dir):
    from phonon.studies import style

    stem = Path(out_dir).name  # study name -> unique figure names in out/fig
    temp = sorted([r for r in rows if r["sweep"] == "temperature" and "ratio" in r],
                  key=lambda r: r["t_mean"])
    leng = sorted([r for r in rows if r["sweep"] == "length" and "ratio" in r],
                  key=lambda r: r["n_slabs"])
    if temp:
        T = [r["t_mean"] for r in temp]
        fig, (ax1, ax2) = style.figure(ncols=2, width=4.5, height=3.6)
        ax1.plot(T, [r["G_ball_W_per_m2_K"] for r in temp], "o-", label=r"$G_{\rm ball}$")
        ax1.plot(T, [r["G_anh_W_per_m2_K"] for r in temp], "s-", label=r"$G_{\rm anh}$")
        ax1.set_xlabel("T (K)"); ax1.set_ylabel(r"$G$ (W m$^{-2}$K$^{-1}$)"); ax1.legend()
        ax2.plot(T, [r["ratio"] for r in temp], "^-", color="C3")
        ax2.set_xlabel("T (K)"); ax2.set_ylabel(r"$G_{\rm anh}/G_{\rm ball}$")
        style.save(fig, f"{stem}_summary_temperature")
    if leng:
        L = [r["n_slabs"] for r in leng]
        fig, ax = style.figure(width=5.0, height=3.6)
        ax.plot(L, [r["ratio"] for r in leng], "o-")
        ax.set_xlabel("device length $L$ (cells)")
        ax.set_ylabel(r"$G_{\rm anh}/G_{\rm ball}$")
        style.save(fig, f"{stem}_summary_length")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--plot", action="store_true")
    a = p.parse_args(argv)
    summarize(a.run_dir, a.out or a.run_dir, a.plot)


if __name__ == "__main__":
    main()
