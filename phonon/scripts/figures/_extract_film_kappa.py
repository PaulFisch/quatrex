"""Distil the film ballistic/SCBA ladders into a committed kappa table.

Data:
  Writes phonon/scripts/data/film_kappa.csv with one row per run:
  Sources: cluster/mos2f{3nu,6b,10b,16b} (ballistic, nu grid),
  cluster/sifilm{3,5,8}{b,s} (ballistic + SCBA legs, uniform 121 grid).

Run:  python phonon/scripts/figures/_extract_film_kappa.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.studies.summarize import cross_section_area

OUT = ROOT / "phonon/scripts/data/film_kappa.csv"
H_SI = 6.62607015e-34

RUNS = [
    # (system, tag, kind, run_dir, npz, tdir)
    ("mos2", "L3", "ballistic", "cluster/mos2f3b2", "run.npz", "z"),
    ("mos2", "L6", "ballistic", "cluster/mos2f6b", "run.npz", "z"),
    ("mos2", "L10", "ballistic", "cluster/mos2f10b", "run.npz", "z"),
    ("mos2", "L16", "ballistic", "cluster/mos2f16b", "run.npz", "z"),
    ("si", "L3", "ballistic", "cluster/sifilm3b", "run.npz", "x"),
    ("si", "L5", "ballistic", "cluster/sifilm5b", "run.npz", "x"),
    ("si", "L8", "ballistic", "cluster/sifilm8b", "run.npz", "x"),
    ("si", "L3", "scba", "cluster/sifilm3s", "run.npz", "x"),
    ("si", "L5", "scba", "cluster/sifilm5s", "run.npz", "x"),
    ("si", "L8", "scba", "cluster/sifilm8s", "run.npz", "x"),
    # QCONV transverse-mesh ladder (2026-08-05, daint jobs 4344979 /
    # 4345441-era reruns 4344975->q7b2, 4345165->q9b2 after the
    # QX_BALLISTIC registry fix): same L3 films, larger q meshes.
    # MoS2 on the 262-pt nu grid: G falls 147.3 -> 141.7 -> 139.6
    # MW/m^2K (nk5/7/9, geometric increments ratio 0.37 -> G_inf ~
    # 138.4, R_inf ~ 7.2 m^2K/GW); Si converged: nk13 = +0.68% vs nk9.
    ("mos2", "L3q7", "ballistic", "cluster/mos2f3q7b2", "run.npz", "z"),
    ("mos2", "L3q9", "ballistic", "cluster/mos2f3q9b2", "run.npz", "z"),
    ("si", "L3q13", "ballistic", "cluster/sifilm3q13b", "run.npz", "x"),
]


def slab_height_m(structure_xyz: Path, tdir: str) -> float:
    """Transport-cell height = cell volume / transverse area (exact for
    any lattice; the diagonal element is wrong for non-orthogonal
    primitives, e.g. the FCC-primitive Si film cell)."""
    line2 = structure_xyz.read_text().splitlines()[1]
    m = re.search(r'Lattice="([^"]+)"', line2)
    lat = np.array([float(x) for x in m.group(1).split()]).reshape(3, 3)
    ti = "xyz".index(tdir)
    perp = [i for i in range(3) if i != ti]
    area = np.linalg.norm(np.cross(lat[perp[0]], lat[perp[1]]))
    return float(abs(np.linalg.det(lat)) / area) * 1e-10


def config_meta(cfg_path: Path) -> tuple[float, int]:
    txt = cfg_path.read_text()
    tl = float(re.search(r"left_temperature\s*=\s*([\d.]+)", txt).group(1))
    tr = float(re.search(r"right_temperature\s*=\s*([\d.]+)", txt).group(1))
    kg = re.search(r"kpoint_grid\s*=\s*\[([^\]]+)\]", txt).group(1)
    n_q = int(np.prod([int(x) for x in kg.split(",")]))
    return abs(tl - tr), n_q


def main() -> None:
    rows = []
    for system, tag, kind, run_dir, npz_name, tdir in RUNS:
        d = ROOT / run_dir
        npz = d / npz_name
        if not npz.exists():
            print(f"skip (missing): {run_dir}/{npz_name}")
            continue
        r = np.load(npz, allow_pickle=True)
        lh = np.asarray(r["last_heat"]).reshape(-1)
        J = float(abs(lh[0]))
        uniform = bool(r.get("uniform_frequency_grid", True))
        e = np.asarray(r["energies"])
        dw = float(np.diff(e).mean())
        struct = d / "structure.xyz"
        if not struct.exists():
            struct = d.parent / "sifilm_nk9r/structure.xyz"
        A_c = cross_section_area(struct, tdir)
        dT, n_q = config_meta(d / "quatrex_config.toml")
        n_slabs = int(r["nblocks"])
        t_m = n_slabs * slab_height_m(struct, tdir)
        G = H_SI * 1e24 * J * (dw if uniform else 1.0) / (A_c * dT * n_q)
        rows.append(dict(
            system=system, tag=tag, kind=kind, n_slabs=n_slabs,
            t_nm=t_m * 1e9, J=J, uniform=uniform, dw_thz=dw,
            A_c_m2=A_c, dT=dT, N_q=n_q, G_W_m2K=G,
            converged=bool(r.get("converged", False)),
            n_iter=int(r.get("n_iter", -1)),
        ))
        print(f"{system} {tag} {kind}: J={J:.4f} A_c={A_c:.3e} "
              f"t={t_m * 1e9:.2f} nm G={G:.4e} W/m^2/K "
              f"R={1e9 / G:.2f} m^2K/GW kappa(t)={G * t_m:.3f} W/mK")

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
