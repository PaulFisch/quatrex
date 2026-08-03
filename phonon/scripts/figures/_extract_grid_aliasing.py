"""Distil the energy-grid resolution studies into a committed dataset.

Sources (uncommitted run artifacts):
  cluster/mos2f3nu/run_ballistic.npz          MoS2 film L3 ballistic on the
                                              262-point non-uniform grid
                                              (min spacing 0.020 THz)
  phonon/studies/out/cnt33_L4_nugrid/{nu2,uni}/run.npz
                                              CNT L4 converged A/B: 287-pt
                                              non-uniform vs 361-pt uniform,
                                              identical physics otherwise
  phonon/studies/out/anderson_test/cnt33_L4_linear/run_ne{161,201,271,361}.npz
                                              the uniform-grid ne scan under
                                              fixed linear mixing (the
                                              convergence lottery)
  phonon/studies/out/d5a_gridladder/nf{181,721}/run.npz
                                              d5a uniform ladder (both legs
                                              diverge; density alone does
                                              not cure the d5a fixed point
                                              -- its converged eta=0 record
                                              is the guarded Anderson
                                              scheme of sec:res_campaign)

Writes phonon/scripts/data/grid_aliasing.npz. Uniform-grid lead
currents are multiplied by dw (integral convention) so all quoted
currents are grid-convention-free.

Run:  python phonon/scripts/figures/_extract_grid_aliasing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "phonon/scripts/data/grid_aliasing.npz"


def lead_current_integral(d) -> float:
    lc = float(d["lead_current"])
    if bool(d.get("uniform_frequency_grid", True)):
        lc *= float(np.diff(np.asarray(d["energies"])).mean())
    return lc


def main() -> None:
    data: dict = {}

    b = np.load(ROOT / "cluster/mos2f3nu/run_ballistic.npz")
    e = np.asarray(b["energies"])
    spec = np.asarray(b["current_spectrum"]).sum(axis=-1)[:, 0, -1]
    data["film_e"] = e
    data["film_ball_spec"] = spec

    for leg in ("nu2", "uni"):
        d = np.load(ROOT / f"phonon/studies/out/cnt33_L4_nugrid/{leg}/run.npz")
        data[f"cnt_{leg}_e"] = np.asarray(d["energies"])
        cs = np.asarray(d["current_spectrum"])
        data[f"cnt_{leg}_spec"] = cs[:, 0] if cs.ndim == 2 else cs.sum(-1)[:, 0, -1]
        data[f"cnt_{leg}_I"] = lead_current_integral(d)
        data[f"cnt_{leg}_conv"] = bool(d["converged"])

    rows = []
    for ne in (161, 201, 271, 361):
        d = np.load(ROOT / "phonon/studies/out/anderson_test/cnt33_L4_linear"
                    / f"run_ne{ne}.npz")
        rows.append([ne, int(d["converged"]), int(d["diverged"]),
                     lead_current_integral(d), int(d["n_iter"])])
    data["nescan"] = np.array(rows)

    d5a = []
    for nf in (181, 721):
        d = np.load(ROOT / f"phonon/studies/out/d5a_gridladder/nf{nf}/run.npz")
        d5a.append([nf, int(d["converged"]), int(d["diverged"]),
                    int(d["n_iter"])])
    data["d5a_ladder"] = np.array(d5a)

    np.savez_compressed(OUT, **data)
    print(f"wrote {OUT}")
    for k, v in data.items():
        a = np.asarray(v)
        print(f"  {k}: {a.shape if a.shape else a}")


if __name__ == "__main__":
    main()
