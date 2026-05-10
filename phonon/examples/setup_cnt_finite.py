"""Generate a finite-length CNT(3,3) segment by stacking N unit cells along z.

Reads the existing periodic CNT(3,3) config and replicates it ``--n-cells``
times along z, then adds a c-axis vacuum gap so the resulting structure is
truly finite (no periodicity along the tube axis). DFT/relax/hiphive
settings are inherited from the base config.

Usage:
    python examples/setup_cnt_finite.py --base configs/cnt/cnt33_vasp.yaml --n-cells 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import add_common_args, write_config  # noqa: E402


def make_config(
    base_path: Path, n_cells: int, vacuum_z_A: float = 8.0,
) -> dict:
    cfg = yaml.safe_load(Path(base_path).read_text())
    lat = np.array(cfg["structure"]["lattice"], dtype=float)
    sp = np.array(cfg["structure"]["scaled_positions"], dtype=float)
    syms = list(cfg["structure"]["symbols"])

    cz_unit = lat[2, 2]
    new_cz = n_cells * cz_unit + vacuum_z_A

    new_sp: list[list[float]] = []
    new_syms: list[str] = []
    for k in range(n_cells):
        offset_z = k * cz_unit / new_cz
        for i, s in enumerate(syms):
            x, y, z_unit = sp[i]
            new_sp.append([x, y, offset_z + z_unit * cz_unit / new_cz])
            new_syms.append(s)

    new_lat = lat.copy()
    new_lat[2, 2] = new_cz

    cfg["structure"]["symbols"] = new_syms
    cfg["structure"]["lattice"] = new_lat.tolist()
    cfg["structure"]["scaled_positions"] = new_sp

    tag = f"cnt33_finite_n{n_cells}"
    cfg["relax"]["work_dir"] = f"./relax_{tag}_vasp"
    if "thirdorder" in cfg:
        cfg["thirdorder"]["work_dir"] = f"./fc3_{tag}_vasp"
        cfg["thirdorder"]["supercell"] = [1, 1, 1]
    if "hiphive" in cfg:
        cfg["hiphive"]["work_dir"] = f"./fc3_hiphive_{tag}_vasp"
        cfg["hiphive"]["supercell"] = [1, 1, 1]
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    here = Path(__file__).resolve()
    p.add_argument(
        "--base", type=Path,
        default=here.parent.parent / "configs" / "cnt" / "cnt33_vasp.yaml",
        help="Periodic CNT base config",
    )
    p.add_argument("--n-cells", type=int, default=4)
    p.add_argument("--vacuum-z", type=float, default=8.0)
    add_common_args(
        p, default_out=here.parent.parent / "configs" / "cnt"
        / "cnt33_finite_nN_vasp.yaml",
    )
    args = p.parse_args()

    cfg = make_config(args.base, args.n_cells, args.vacuum_z)
    out = (
        args.out if args.out.name != "cnt33_finite_nN_vasp.yaml"
        else args.out.with_name(f"cnt33_finite_n{args.n_cells}_vasp.yaml")
    )
    write_config(cfg, out)


if __name__ == "__main__":
    main()
