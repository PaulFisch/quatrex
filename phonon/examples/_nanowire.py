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
    radial_dot_min: float = 0.1,
    h_h_clash_A: float = 1.9,
) -> Atoms:
    """Cap undercoordinated surface atoms with H along outward sp3 dirs.

    Two filters keep this physically sensible for narrow wires:

    1. ``radial_dot_min``: an sp3 direction is only filled if it points
       outward from the wire axis (``d_unit . r_hat > 0.1``). Inward
       directions collide with H atoms from the opposite surface and
       lead to H-H pairs < 1.5 A that relax into H2 in DFT.
    2. ``h_h_clash_A``: after placing all candidates we iteratively
       drop the H atom involved in the most sub-threshold pairs until
       no pair lies below ``h_h_clash_A``. Periodic z-neighbours are
       considered via :class:`ase.neighborlist.NeighborList`.
    """
    bond_length = a_lattice * np.sqrt(3.0) / 4.0
    cutoff = 0.6 * bond_length

    nl_cutoffs = [cutoff] * len(wire)
    nl = NeighborList(
        nl_cutoffs, self_interaction=False, bothways=True, skin=0.0,
    )
    nl.update(wire)

    cell = wire.get_cell().array
    positions = wire.get_positions()
    center_xy = 0.5 * np.array([cell[0, 0], cell[1, 1]])

    h_positions: list[np.ndarray] = []
    h_host: list[int] = []                  # parent Si index per H candidate
    si_si_bonds: dict[int, int] = {}        # current Si-Si bond count per Si
    for i, p in enumerate(positions):
        if wire[i].symbol != species:
            continue
        neighbour_idx, offsets = nl.get_neighbors(i)
        neighbour_vecs: list[np.ndarray] = []
        for j, off in zip(neighbour_idx, offsets):
            disp = positions[j] + off @ cell - p
            neighbour_vecs.append(disp / np.linalg.norm(disp))
        si_si_bonds[i] = len(neighbour_vecs)

        r_xy = p[:2] - center_xy
        r_norm = np.linalg.norm(r_xy)
        r_hat = np.array([r_xy[0], r_xy[1], 0.0]) / r_norm if r_norm > 1e-6 \
            else np.array([1.0, 0.0, 0.0])

        missing = _missing_bond_directions(neighbour_vecs)
        for d_unit in missing:
            if np.dot(d_unit, r_hat) < radial_dot_min:
                continue
            h_positions.append(p + d_unit * d_x_h)
            h_host.append(i)

    h_positions, h_host = _resolve_h_clashes(
        np.array(h_positions) if h_positions else np.zeros((0, 3)),
        h_host=h_host,
        cell=cell,
        threshold=h_h_clash_A,
        si_si_bonds=si_si_bonds,
    )
    if len(h_positions) == 0:
        return wire

    h_atoms = Atoms(
        "H" * len(h_positions), positions=h_positions, cell=cell, pbc=wire.pbc,
    )
    return wire + h_atoms


def _missing_bond_directions(
    nv: list[np.ndarray],
) -> list[np.ndarray]:
    """Return the missing sp3 unit vectors completing a tetrahedron.

    ``nv`` is a list of existing neighbour unit vectors (length 0-4).
    The four unit vectors of a perfect tetrahedron sum to zero, so:

      - 0 NN -> default tetrahedron (matches diamond [1,1,1] sublattice)
      - 1 NN -> three vectors equally spaced on the cone at
        ``arccos(-1/3)`` around ``-nv[0]``; we use an arbitrary basis.
      - 2 NN -> two vectors that bisect the plane orthogonal to
        ``nv[0] + nv[1]`` at the tetrahedral half-angle.
      - 3 NN -> ``-(nv[0] + nv[1] + nv[2]) / |.|`` (exact).
      - 4 NN -> empty list.

    Branches with ``k < 3`` happen at corner / edge Si on narrow wires
    where two of the bulk bonds are missing simultaneously.
    """
    k = len(nv)
    if k >= 4:
        return []
    if k == 3:
        out = -sum(nv)
        return [out / np.linalg.norm(out)]
    if k == 0:
        s = 1.0 / np.sqrt(3.0)
        return [
            np.array([s, s, s]), np.array([-s, -s, s]),
            np.array([-s, s, -s]), np.array([s, -s, -s]),
        ]
    if k == 1:
        # Three directions on the cone at angle arccos(-1/3) around -nv[0].
        a = -nv[0]
        # Build a 2D basis orthogonal to a.
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(a @ tmp) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e1 = tmp - (tmp @ a) * a; e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        # Cone: c = cos(theta_tet/2)? Actually: dot with a = -1/3.
        ca = -1.0 / 3.0
        sa = np.sqrt(1 - ca**2)
        out = []
        for phi in (0.0, 2 * np.pi / 3, 4 * np.pi / 3):
            v = ca * (-a) + sa * (np.cos(phi) * e1 + np.sin(phi) * e2)
            # ca*(-a) flips back: missing dirs point AWAY from a (which is -nv[0]).
            # Simpler: dot(missing, nv[0]) = -1/3 for tetrahedron.
            out.append(v / np.linalg.norm(v))
        # Rewrite to enforce dot with nv[0] = -1/3
        # Above derivation is messy; just construct directly:
        n0 = nv[0]
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(n0 @ tmp) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e1 = tmp - (tmp @ n0) * n0; e1 /= np.linalg.norm(e1)
        e2 = np.cross(n0, e1)
        ca = -1.0 / 3.0
        sa = np.sqrt(1 - ca**2)
        return [
            ca * n0 + sa * (np.cos(phi) * e1 + np.sin(phi) * e2)
            for phi in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)
        ]
    # k == 2: two missing directions in the plane orthogonal to (nv[0]+nv[1]).
    s = nv[0] + nv[1]
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-6:
        # Existing bonds are antiparallel: missing pair must be ortho to both.
        ax = np.cross(nv[0], np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(nv[0], np.array([0.0, 1.0, 0.0]))
        ax /= np.linalg.norm(ax)
        ay = np.cross(nv[0], ax); ay /= np.linalg.norm(ay)
        return [ax, -ax, ay, -ay][:2]
    # Bisector of the two missing dirs points opposite to (nv[0]+nv[1]).
    b = -s / s_norm
    # Ortho axis lies in the plane perpendicular to b, also perp to nv[0]-nv[1].
    ortho = np.cross(nv[0], nv[1])
    ortho /= np.linalg.norm(ortho)
    # Tetrahedral half-angle around bisector: cos = sqrt(1/3) (109.47/2).
    cb = np.sqrt(1.0 / 3.0)
    sb = np.sqrt(2.0 / 3.0)
    return [
        cb * b + sb * ortho,
        cb * b - sb * ortho,
    ]


def _resolve_h_clashes(
    h_pos: np.ndarray,
    *,
    h_host: list[int],
    cell: np.ndarray,
    threshold: float,
    si_si_bonds: dict[int, int],
) -> tuple[np.ndarray, list[int]]:
    """Drop H atoms with sub-threshold pairs, preferring over-passivated hosts.

    Score for each H candidate (higher = drop first):
        score = (n_clashes, host_current_coordination)
    where ``host_current_coordination = n_Si_neighbours + n_H_already_kept``.
    This keeps H atoms attached to Si that would otherwise be under-bonded.
    """
    if len(h_pos) == 0:
        return h_pos, list(h_host)
    cz = cell[2, 2]
    pos = h_pos.copy()
    host = list(h_host)
    while True:
        d = pos[:, None, :] - pos[None, :, :]
        d[..., 2] -= np.round(d[..., 2] / cz) * cz
        r = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(r, np.inf)
        clash = r < threshold
        clash_counts = clash.sum(axis=1)
        if clash_counts.max() == 0:
            return pos, host
        # Host coordination including already-kept H atoms on the same Si.
        host_h_count = {h: host.count(h) for h in set(host)}
        host_coord = np.array(
            [si_si_bonds[h] + host_h_count[h] for h in host]
        )
        # Drop the clash-involved H with the highest (clash_count, host_coord).
        candidates = np.flatnonzero(clash_counts == clash_counts.max())
        kill_local = int(np.argmax(host_coord[candidates]))
        kill = int(candidates[kill_local])
        pos = np.delete(pos, kill, axis=0)
        host.pop(kill)


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
