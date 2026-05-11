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
from _nanowire import build_h_passivated_wire  # noqa: E402

A_SI = 5.43      # Å, bulk Si lattice constant (PBE eq.)
D_SI_H = 1.48    # Å, Si–H bond length


def make_config(diameter_A: float, supercell_z: int = 2) -> dict:
    wire = build_h_passivated_wire(
        a_lattice=A_SI, diameter_A=diameter_A, vacuum_A=18.0,
        n_z=1, species="Si", d_x_h=D_SI_H,
    )
    new_lat = wire.get_cell().array
    syms = list(wire.get_chemical_symbols())
    order = sorted(range(len(syms)), key=lambda k: (syms[k] != "Si", k))
    syms = [syms[k] for k in order]
    cart_all = wire.get_positions()[order]
    frac = (cart_all @ np.linalg.inv(new_lat)) % 1.0

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
