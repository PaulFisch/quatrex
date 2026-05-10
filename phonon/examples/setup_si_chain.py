"""Generate a finite Si-chain config + structure (also a test fixture).

Default: 8-atom chain at 2.35 Å spacing, ~Si bulk bond length. Used as
the smallest test fixture for the ``finite_analysis`` package.

Usage:
    python examples/setup_si_chain.py --n-atoms 12 --spacing 2.35
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import (  # noqa: E402
    add_common_args, default_relax_block, default_thirdorder_block,
    default_vasp_block, write_config,
)


def make_config(n_atoms: int, spacing: float, vacuum: float = 15.0) -> dict:
    c_len = n_atoms * spacing
    z_offsets = [(i + 0.5) * spacing / c_len for i in range(n_atoms)]
    tag = f"si_chain_n{n_atoms}"
    return {
        "structure": {
            "source": "inline",
            "symbols": ["Si"] * n_atoms,
            "lattice": [
                [vacuum, 0.0, 0.0],
                [0.0, vacuum, 0.0],
                [0.0, 0.0, c_len],
            ],
            "scaled_positions": [[0.5, 0.5, z] for z in z_offsets],
        },
        **default_vasp_block(
            {"Si": "Si"},
            kpoints_scf=(1, 1, 1),
            sigma=0.05, kpar=1,
            vasp_command="ulimit -s unlimited; mpirun -np 32 vasp_std",
        ),
        **default_relax_block(f"./relax_{tag}_vasp", fc_method="thirdorder"),
        **default_thirdorder_block(
            f"./fc3_{tag}_vasp", [1, 1, 1],
            cutoff_pair_distance=5.0, pw_timeout=3600,
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    here = Path(__file__).resolve()
    p.add_argument("--n-atoms", type=int, default=8)
    p.add_argument("--spacing", type=float, default=2.35,
                   help="z-spacing in Å (default ≈ Si bulk bond length)")
    p.add_argument("--vacuum", type=float, default=15.0,
                   help="x/y vacuum thickness in Å")
    add_common_args(
        p, default_out=here.parent.parent / "configs" / "chain"
        / "si_chain_nN_auto.yaml",
    )
    args = p.parse_args()

    cfg = make_config(args.n_atoms, args.spacing, args.vacuum)
    out = (
        args.out if args.out.name != "si_chain_nN_auto.yaml"
        else args.out.with_name(f"si_chain_n{args.n_atoms}_auto.yaml")
    )
    write_config(cfg, out)


if __name__ == "__main__":
    main()
