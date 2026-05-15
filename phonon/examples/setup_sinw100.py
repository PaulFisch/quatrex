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

# PBE-equilibrium Si lattice constant. The previous default 5.43 Å is the
# room-temperature experimental value; PBE optimises slightly larger.
# Using 5.470 Å with a PBE functional avoids a built-in 0.7 % compressive
# strain on the wire core, which otherwise shifts every bulk-derived mode
# by O(3 cm⁻¹) via the Grüneisen relation. Tested against the PBE-relaxed
# diamond Si reference in `phonon/reaps/si_primitive_vasp/`.
A_SI = 5.470
D_SI_H = 1.48    # Å, Si–H bond length (SiH₄ reference, PBE)


def _suggest_fc2_cutoff(envelope_A: float) -> float:
    """FC2 cutoff covering both wire cross-section and the second-NN z-shell.

    SiNWs have a free transverse direction: any pair of atoms with
    Cartesian separation > cutoffs[0] gets zero FC2 by construction. For
    a wire diameter d, cross-section pairs span up to ~d, so we want
    cutoffs[0] ≳ d + 1 NN shell (~2.4 Å). Cap at 12 Å so the cluster
    space stays tractable; for the d12a wire that lands at 12 Å, which
    still misses a few diametrical pairs but keeps n_parameters under
    O(10 000).
    """
    return min(12.0, max(5.0, envelope_A + 2.4))


def make_config(diameter_A: float, supercell_z: int = 2) -> dict:
    # vacuum_A=None lets the builder size the box so adjacent periodic
    # images are MIN_INTERWIRE_VACUUM_A apart for the actual H-shell
    # radius (was a hard-coded 18.0, which gave d12a only 4 Å vacuum and
    # contaminated the DFT with image overlap).
    wire = build_h_passivated_wire(
        a_lattice=A_SI, diameter_A=diameter_A, vacuum_A=None,
        n_z=1, species="Si", d_x_h=D_SI_H,
    )
    new_lat = wire.get_cell().array
    syms = list(wire.get_chemical_symbols())
    order = sorted(range(len(syms)), key=lambda k: (syms[k] != "Si", k))
    syms = [syms[k] for k in order]
    cart_all = wire.get_positions()[order]
    frac = (cart_all @ np.linalg.inv(new_lat)) % 1.0

    # Use the actual H-shell envelope to pick the FC2 cutoff.
    center_xy = 0.5 * new_lat.diagonal()[:2]
    envelope_A = float(np.linalg.norm(cart_all[:, :2] - center_xy, axis=1).max())
    fc2_cutoff = _suggest_fc2_cutoff(envelope_A)

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
            cutoffs=(fc2_cutoff, 4.0),
            rattle_d_min=1.4,  # Si-H 1.48 Å, H-H 2.4 Å — 1.4 keeps both safe
            # Rattle amplitude reduced 0.04 → 0.03 Å and n_iter 20 → 10 to
            # keep mc-rattle Si-Si pairs above VASP's RWIGS-derived "two
            # ions too close" warning threshold (≈ RWIGS_Si × 2 ≈ 2.6 Å)
            # in most rattled structures. Cumulative RMS displacement
            # ≈ std × √n_iter ≈ 0.095 Å (was ~0.18 Å); worst-case Si-Si
            # ≈ 2.35 − 0.19 ≈ 2.16 Å, still above the warning floor for
            # most structures. 0.03 Å matches the value used in the
            # hiphive Si-thermal-conductivity tutorial.
            rattle_std=0.03,
            rattle_n_iter=10,
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
