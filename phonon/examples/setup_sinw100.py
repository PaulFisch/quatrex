"""Generate a <100> H-passivated Si nanowire YAML at a chosen radius.

Builds the wire by carving a circular column of radius ``--diameter / 2``
out of bulk Si along the z-axis, then capping every undercoordinated
surface Si along its missing sp3 directions with H at d(Si–H) = 1.48 Å.

Usage:
    python examples/setup_sinw100.py --diameter 6.0
    python examples/setup_sinw100.py --diameter 9.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import (  # noqa: E402  (sys.path bootstrap above)
    add_common_args, default_hiphive_block, default_relax_block,
    default_thirdorder_block, default_vasp_block, write_config,
)
from _nanowire import bulk_diamond_supercell, carve_wire, passivate  # noqa: E402

A_SI = 5.43      # Å, bulk Si lattice constant (PBE eq.)
D_SI_H = 1.48    # Å, Si–H bond length


def make_config(diameter_A: float, supercell_z: int = 2) -> dict:
    n_xy = max(3, int(np.ceil(diameter_A / A_SI)) + 2)
    cart, lat = bulk_diamond_supercell(A_SI, n_xy, n_z=1)
    cart = carve_wire(cart, lat, radius_A=diameter_A / 2.0)
    h_cart = passivate(cart, lat, a_lattice=A_SI, d_x_h=D_SI_H)
    cart_all = np.concatenate([cart, h_cart])
    syms = ["Si"] * cart.shape[0] + ["H"] * h_cart.shape[0]

    new_lat = np.diag([18.0, 18.0, A_SI])
    cart_xy = cart_all.copy()
    cart_xy[:, 0] += 9.0 - lat[0, 0] / 2
    cart_xy[:, 1] += 9.0 - lat[1, 1] / 2
    frac = (cart_xy @ np.linalg.inv(new_lat)) % 1.0

    tag = f"sinw100_d{int(diameter_A)}a"
    cfg = {
        "structure": {
            "source": "inline", "symbols": syms,
            "lattice": new_lat.tolist(),
            "scaled_positions": frac.tolist(),
        },
        **default_vasp_block({"Si": "Si", "H": "H"}),
        **default_relax_block(f"./relax_{tag}_vasp"),
        **default_hiphive_block(
            f"./fc3_hiphive_{tag}_vasp", [1, 1, supercell_z],
        ),
        **default_thirdorder_block(
            f"./fc3_{tag}_vasp", [1, 1, supercell_z],
        ),
    }
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--diameter", type=float, required=True,
                   help="Wire diameter in Å (e.g. 6.0, 9.0)")
    p.add_argument("--supercell-z", type=int, default=2)
    here = Path(__file__).resolve()
    add_common_args(
        p, default_out=here.parent.parent / "configs" / "sinw"
        / "sinw100_dXa_vasp.yaml",
    )
    args = p.parse_args()

    cfg = make_config(args.diameter, args.supercell_z)
    out = (
        args.out if args.out.name != "sinw100_dXa_vasp.yaml"
        else args.out.with_name(f"sinw100_d{int(args.diameter)}a_vasp.yaml")
    )
    write_config(cfg, out)


if __name__ == "__main__":
    main()
