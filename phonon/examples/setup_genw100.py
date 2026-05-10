"""Generate a Ge equivalent of the H-passivated SiNW(100).

Same construction as :mod:`setup_sinw100`, with Si→Ge substitution,
lattice rescaled by ``a_Ge / a_Si``, and Ge–H bond length 1.53 Å.

Usage:
    python examples/setup_genw100.py --diameter 6.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import (  # noqa: E402
    add_common_args, default_hiphive_block, default_relax_block,
    default_thirdorder_block, default_vasp_block, write_config,
)
from _nanowire import bulk_diamond_supercell, carve_wire, passivate  # noqa: E402

A_GE = 5.658
D_GE_H = 1.53


def make_config(diameter_A: float, supercell_z: int = 2) -> dict:
    n_xy = max(3, int(np.ceil(diameter_A / A_GE)) + 2)
    cart, lat = bulk_diamond_supercell(A_GE, n_xy, n_z=1)
    cart = carve_wire(cart, lat, radius_A=diameter_A / 2.0)
    h_cart = passivate(cart, lat, a_lattice=A_GE, d_x_h=D_GE_H)
    cart_all = np.concatenate([cart, h_cart])
    syms = ["Ge"] * cart.shape[0] + ["H"] * h_cart.shape[0]

    new_lat = np.diag([18.0, 18.0, A_GE])
    cart_xy = cart_all.copy()
    cart_xy[:, 0] += 9.0 - lat[0, 0] / 2
    cart_xy[:, 1] += 9.0 - lat[1, 1] / 2
    frac = (cart_xy @ np.linalg.inv(new_lat)) % 1.0

    tag = f"genw100_d{int(diameter_A)}a"
    cfg = {
        "structure": {
            "source": "inline", "symbols": syms,
            "lattice": new_lat.tolist(),
            "scaled_positions": frac.tolist(),
        },
        **default_vasp_block({"Ge": "Ge", "H": "H"}),
        **default_relax_block(f"./relax_{tag}_vasp"),
        **default_hiphive_block(
            f"./fc3_hiphive_{tag}_vasp", [1, 1, supercell_z],
            cutoffs=(5.5, 4.5),
        ),
        **default_thirdorder_block(
            f"./fc3_{tag}_vasp", [1, 1, supercell_z],
        ),
    }
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--diameter", type=float, required=True)
    p.add_argument("--supercell-z", type=int, default=2)
    here = Path(__file__).resolve()
    add_common_args(
        p, default_out=here.parent.parent / "configs" / "genw"
        / "genw100_dXa_vasp.yaml",
    )
    args = p.parse_args()

    cfg = make_config(args.diameter, args.supercell_z)
    out = (
        args.out if args.out.name != "genw100_dXa_vasp.yaml"
        else args.out.with_name(f"genw100_d{int(args.diameter)}a_vasp.yaml")
    )
    write_config(cfg, out)


if __name__ == "__main__":
    main()
