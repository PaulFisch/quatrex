"""Distil the MoS2 film instability record into a committed dataset.

Reduces the uncommitted cluster/ artifacts of the film campaign to
phonon/scripts/data/mos2_spiral.npz for the mos2_spiral.py generator:

  res_long        (95, 3) rel Sigma^R residual, lead balance, internal
                  spread per iteration of the 95-iteration record
                  (cluster/mos2f3long, job 4318325: SCP(300 K) fc2 +
                  cubic vertex, eta=0, linear 0.1 mixing, nu grid 267)
  heat_long       (95, 5, 5) per-iteration heat matrix (rank-summed)
  sigmax_long     (95, 67) per-iteration max|Sigma^<| on rank 0's
                  energy slice (the low-frequency 67 bins of 267)
  energies_lo     (67,) those energies (THz)
  res_<record>    residual series of every other stabiliser probe:
                  mix_a/mix_b/mix_c   three 55-it mixing variants
                                      (jobs 4315556/4315564/4315592;
                                      per-job scheme labels were not
                                      preserved -- treated as an
                                      ensemble, see REVIEW note in
                                      75_mos2.tex)
                  floor_ramp          annealed eta_ir floor 2->0/30
                                      (job 4315641)
                  orbit_mean          orbit-mean restart (4318752)
                  tadpole             SCP tadpole static Sigma (4318792)
                  loop3               tadpole+quartic loop, dressed
                                      <uu> (4319028, diverged)
                  loop4               quartic loop only, dressed <uu>,
                                      bare o4 build (4320390, diverged
                                      at it 28, best residual 0.83)

Run:  python phonon/scripts/figures/_extract_mos2_spiral.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "phonon/scripts/data/mos2_spiral.npz"

RE_RES = re.compile(r"rel Sigma\^R residual ([\d.e+-]+); lead balance "
                    r"([\d.e+-]+)(?:.*?internal spread ([\d.e+-]+))?")

RECORDS = {
    "mix_a": "cluster/mos2f3/slurm-4315556.out",
    "mix_b": "cluster/mos2f3/slurm-4315564.out",
    "mix_c": "cluster/mos2f3/slurm-4315592.out",
    "floor_ramp": "cluster/mos2f3/slurm-4315641.out",
    "orbit_mean": "cluster/mos2f3mr/slurm-4318752.out",
    "tadpole": "cluster/mos2f3tp/slurm-4318792.out",
    "loop3": "cluster/mos2f3o4/slurm-4319028.out",
    "loop4": "cluster/mos2f3o4/slurm-4320390.out",
}


def residual_series(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        m = RE_RES.search(line)
        if m:
            rows.append([float(m.group(1)), float(m.group(2)),
                         float(m.group(3)) if m.group(3) else np.nan])
    return np.array(rows)


def main() -> None:
    data = {}
    long_log = ROOT / "cluster/mos2f3long/slurm-4318325.out"
    data["res_long"] = residual_series(long_log)

    run = np.load(ROOT / "cluster/mos2f3long/run.npz")
    data["heat_long"] = run["iter_heat"].sum(axis=-1)
    data["sigmax_long"] = run["iter_sigma_max"]
    n_lo = run["iter_sigma_max"].shape[1]
    data["energies_lo"] = run["energies"][:n_lo]

    for name, rel in RECORDS.items():
        s = residual_series(ROOT / rel)
        if not len(s):
            print(f"WARNING: no residual lines in {rel}")
            continue
        data[f"res_{name}"] = s

    np.savez_compressed(OUT, **data)
    print(f"wrote {OUT}")
    for k, v in data.items():
        print(f"  {k}: {np.asarray(v).shape}")


if __name__ == "__main__":
    main()
