"""Distil the MoS2 film instability record into a committed dataset.

Reduces the uncommitted cluster/ artifacts of the film campaign to
phonon/scripts/data/mos2_spiral.npz for the mos2_spiral.py generator.

VERTEX PROVENANCE (established 2026-08-03, the decisive fact for how
these series may be read): the early cluster/mos2f3 and mos2f3nu
builds were materialised from the ARDR-era film reap, whose vdW-gap
FC3 is exact zero -- their device vertex holds ONLY the three
slab-diagonal blocks (build_device_fc3_blocks drops hard-zero
blocks), i.e. NO cross-slab three-phonon coupling. The mos2f3scp and
mos2f3o4 builds (and every mos2film_L*_nk5_* build) carry the full
15-block vertex with 12 cross-slab blocks. Runs on the broken build
are therefore probes of a DIFFERENT physical model (intra-slab-only
anharmonicity) -- kept here as the vertex-ablation control, not as
stabiliser evidence.

  res_long        (94, 3) rel Sigma^R residual, lead balance,
                  internal spread per iteration of the 95-iteration
                  record (cluster/mos2f3long, job 4318325: SCP(300 K)
                  fc2, FULL vertex, eta=0, linear 0.1, nu grid)
  heat_long       (95, 5, 5) per-iteration heat matrix (rank-summed)
  sigmax_long     (95, 67) per-iteration max|Sigma^<| on rank 0's
                  low-frequency energy slice
  energies_lo     (67,) those energies (THz)

  FULL-vertex stabiliser probes (the honest ensemble):
  res_orbit_mean  orbit-mean restart (job 4318752, scp build)
  res_tadpole     SCP tadpole static Sigma (4318792, scp build)
  res_loop3       tadpole+quartic loop, dressed <uu> (4319028,
                  o4 build, diverged)
  res_loop4       quartic loop only, dressed <uu> (4320390,
                  o4 build, diverged at 28, best 0.83)

  BROKEN-vertex (diagonal-only) probes -- the ablation control:
  res_abl_a/b/c   three 55-it probes on cluster/mos2f3 (jobs
                  4315556/4315564/4315592; per-run env overrides not
                  preserved; c descends monotonically to 0.087 and
                  was cut by the iteration cap)
  res_abl_floor   annealed eta_ir floor ramp on the same broken
                  build (4315641)
  res_abl_cont    the 2026-08-03 continuation (mos2f3-b2cont, job
                  4322408, current code, same broken build: descends
                  to 0.646 then diverges at 66 -- the ablated model
                  is gentler, not stable)

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
    "orbit_mean": "cluster/mos2f3mr/slurm-4318752.out",
    "tadpole": "cluster/mos2f3tp/slurm-4318792.out",
    "loop3": "cluster/mos2f3o4/slurm-4319028.out",
    "loop4": "cluster/mos2f3o4/slurm-4320390.out",
    "abl_a": "cluster/mos2f3/slurm-4315556.out",
    "abl_b": "cluster/mos2f3/slurm-4315564.out",
    "abl_c": "cluster/mos2f3/slurm-4315592.out",
    "abl_floor": "cluster/mos2f3/slurm-4315641.out",
    "abl_cont": "cluster/mos2f3-b2cont/slurm-4322408.out",
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
        p = ROOT / rel
        if not p.exists():
            print(f"WARNING: missing {rel}")
            continue
        s = residual_series(p)
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
