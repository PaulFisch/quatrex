"""Geometry primitives for the H-passivated <100> diamond-cubic nanowires.

Used by ``setup_sinw100.py``, ``setup_genw100.py``, and (indirectly via
``setup_sige_nw.py``) the SiGe alloy generator. These were originally
duplicated as private helpers inside ``setup_sinw100.py``.
"""

from __future__ import annotations

import numpy as np


def bulk_diamond_supercell(
    a: float, n_xy: int, n_z: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Conventional 8-atom diamond-cubic cell tiled into an
    ``(n_xy, n_xy, n_z)`` block.

    Returns ``(positions, lattice)`` in Cartesian Å. Use as the substrate
    that subsequent :func:`carve_wire` / :func:`passivate` calls operate on.
    """
    basis = np.array([
        [0.00, 0.00, 0.00], [0.50, 0.50, 0.00],
        [0.50, 0.00, 0.50], [0.00, 0.50, 0.50],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
    ])
    positions = []
    for ix in range(n_xy):
        for iy in range(n_xy):
            for iz in range(n_z):
                for b in basis:
                    positions.append((b + np.array([ix, iy, iz])))
    positions = np.array(positions) * a
    lattice = np.diag([n_xy * a, n_xy * a, n_z * a])
    return positions, lattice


def carve_wire(
    positions: np.ndarray, lattice: np.ndarray, radius_A: float,
) -> np.ndarray:
    """Keep atoms within ``radius_A`` of the lattice (x, y) center."""
    cx, cy = 0.5 * lattice[0, 0], 0.5 * lattice[1, 1]
    keep = (positions[:, 0] - cx) ** 2 + (positions[:, 1] - cy) ** 2 <= radius_A ** 2
    return positions[keep]


def passivate(
    positions: np.ndarray,
    lattice: np.ndarray,
    *,
    a_lattice: float,
    d_x_h: float,
) -> np.ndarray:
    """Add H atoms along missing sp3 directions of undercoordinated cores.

    For every atom with fewer than four neighbours within an sp3-bond
    sphere, an H is added along each unoccupied sp3 direction at distance
    ``d_x_h``.

    Parameters
    ----------
    positions : (n, 3) ndarray
        Cartesian positions of the core atoms (Si or Ge).
    lattice : (3, 3) ndarray
        Lattice (used for the ``a_lattice``-derived neighbour radius).
    a_lattice : float
        Bulk lattice constant (Å) of the core material; sets the sp3
        nearest-neighbour bond length to ``a_lattice * sqrt(3)/4``.
    d_x_h : float
        X–H bond length (Å); 1.48 for Si–H, 1.53 for Ge–H.
    """
    sp3 = (np.array([
        [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
    ]) / np.sqrt(3.0)) * (a_lattice * np.sqrt(3.0) / 4.0)
    sp3_unit = sp3 / np.linalg.norm(sp3, axis=1, keepdims=True)
    bond_length = a_lattice * np.sqrt(3.0) / 4.0

    def _has_neighbour(p: np.ndarray, direction: np.ndarray) -> bool:
        cand = p + direction * bond_length
        return bool(np.any(np.linalg.norm(positions - cand, axis=1) < 0.5))

    h_positions: list[np.ndarray] = []
    for p in positions:
        for d_unit in sp3_unit:
            if not _has_neighbour(p, d_unit):
                h_positions.append(p + d_unit * d_x_h)
    return np.array(h_positions) if h_positions else np.zeros((0, 3))
