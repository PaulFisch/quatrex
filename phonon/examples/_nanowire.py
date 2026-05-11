"""Geometry primitives for H-passivated diamond-cubic nanowires.

Implementation uses ASE for two reasons:
  1. ``ase.build.bulk`` gives a known-correct conventional diamond cell.
  2. ``ase.neighborlist.NeighborList`` does periodic-image-aware neighbour
     detection — necessary for the z-periodic wire, where each surface Si
     has bonds that wrap across the periodic boundary.

The previous hand-rolled implementation missed periodic z-neighbours and
generated Si-H pairs at ~0.87 A for wider wires (any radius such that the
opposing surface Si pair lay along an sp3 vector across the z-image).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import NeighborList, natural_cutoffs


def _diamond_block(a: float, n_xy: int, n_z: int) -> Atoms:
    """Conventional 8-atom diamond cell tiled ``n_xy x n_xy x n_z``.

    Returns an ASE ``Atoms`` object with cubic PBC along all three axes;
    callers pull positions/cell from it but rebuild the cell later for
    vacuum padding and z-periodic wire conventions.
    """
    conv = bulk("Si", "diamond", a=a, cubic=True)
    return conv.repeat((n_xy, n_xy, n_z))


def _carve_column(atoms: Atoms, radius_A: float) -> Atoms:
    """Keep only atoms within ``radius_A`` of the (x, y) centre of the cell."""
    pos = atoms.get_positions()
    cell = atoms.get_cell().array
    cx = 0.5 * cell[0, 0]
    cy = 0.5 * cell[1, 1]
    keep_mask = (pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2 <= radius_A ** 2
    return atoms[keep_mask]


def _passivate_with_nl(
    wire: Atoms,
    *,
    a_lattice: float,
    d_x_h: float,
    species: str = "Si",
) -> Atoms:
    """Cap undercoordinated surface atoms with H along missing sp3 dirs.

    Uses ASE's NeighborList with ``self_interaction=False, bothways=True``
    so neighbours across the periodic z-boundary are detected correctly.
    """
    sp3 = (np.array([
        [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
    ]) / np.sqrt(3.0))
    bond_length = a_lattice * np.sqrt(3.0) / 4.0
    cutoff = 0.6 * bond_length

    nl_cutoffs = [cutoff] * len(wire)
    nl = NeighborList(
        nl_cutoffs, self_interaction=False, bothways=True, skin=0.0,
    )
    nl.update(wire)

    cell = wire.get_cell().array
    positions = wire.get_positions()
    h_positions: list[np.ndarray] = []

    for i, p in enumerate(positions):
        if wire[i].symbol != species:
            continue
        neighbour_idx, offsets = nl.get_neighbors(i)
        neighbour_vecs: list[np.ndarray] = []
        for j, off in zip(neighbour_idx, offsets):
            disp = positions[j] + off @ cell - p
            neighbour_vecs.append(disp / np.linalg.norm(disp))

        for d_unit in sp3:
            occupied = any(
                np.dot(d_unit, nv) > 0.9 for nv in neighbour_vecs
            )
            if not occupied:
                h_positions.append(p + d_unit * d_x_h)

    if not h_positions:
        return wire

    h_arr = np.array(h_positions)
    h_atoms = Atoms("H" * len(h_arr), positions=h_arr, cell=cell, pbc=wire.pbc)
    return wire + h_atoms


def build_h_passivated_wire(
    *,
    a_lattice: float,
    diameter_A: float,
    vacuum_A: float = 18.0,
    n_z: int = 1,
    species: str = "Si",
    d_x_h: float = 1.48,
) -> Atoms:
    """Return an H-passivated <100> diamond nanowire as an ASE Atoms object.

    The cell is set to ``diag(vacuum_A, vacuum_A, n_z * a_lattice)`` with
    ``pbc=(False, False, True)`` — true 1-D periodicity along z.

    The carved column is centred in the vacuum box before passivation so
    that NeighborList sees a single isolated wire (periodic only in z).
    """
    n_xy = max(3, int(np.ceil(diameter_A / a_lattice)) + 2)
    bulk_block = _diamond_block(a_lattice, n_xy=n_xy, n_z=n_z)
    if species != "Si":
        bulk_block.set_chemical_symbols([species] * len(bulk_block))

    carved = _carve_column(bulk_block, radius_A=diameter_A / 2.0)

    new_cell = np.diag([vacuum_A, vacuum_A, n_z * a_lattice])
    pos = carved.get_positions()
    src_cell = carved.get_cell().array
    pos[:, 0] += 0.5 * vacuum_A - 0.5 * src_cell[0, 0]
    pos[:, 1] += 0.5 * vacuum_A - 0.5 * src_cell[1, 1]

    wire = Atoms(
        symbols=[a.symbol for a in carved],
        positions=pos,
        cell=new_cell,
        pbc=(False, False, True),
    )

    wire = _passivate_with_nl(
        wire, a_lattice=a_lattice, d_x_h=d_x_h, species=species,
    )
    return wire


def bulk_diamond_supercell(
    a: float, n_xy: int, n_z: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy helper kept for callers that want raw (positions, lattice).

    Prefer :func:`build_h_passivated_wire` for new code.
    """
    block = _diamond_block(a, n_xy, n_z)
    return block.get_positions(), block.get_cell().array


def carve_wire(
    positions: np.ndarray, lattice: np.ndarray, radius_A: float,
) -> np.ndarray:
    """Legacy helper: keep atoms within radius_A of the (x, y) cell centre."""
    cx, cy = 0.5 * lattice[0, 0], 0.5 * lattice[1, 1]
    keep = (positions[:, 0] - cx) ** 2 + (positions[:, 1] - cy) ** 2 <= radius_A ** 2
    return positions[keep]
