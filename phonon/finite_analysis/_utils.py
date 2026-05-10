"""Pure helpers used across :mod:`finite_analysis`.

No :class:`SystemBundle` dependency, no I/O — just array operations and
small index manipulations that several modules want.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Distance / geometry                                                         #
# --------------------------------------------------------------------------- #


def min_image_distance_matrix(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Pairwise minimum-image distance ``(n, n)`` in Å.

    Periodicity is assumed in all three directions of ``cell``. For finite
    structures padded with vacuum that's still correct because the relevant
    distance is always within one supercell.
    """
    inv = np.linalg.inv(cell)
    frac = positions @ inv
    diff = frac[:, None, :] - frac[None, :, :]
    diff -= np.round(diff)
    cart = diff @ cell
    return np.linalg.norm(cart, axis=-1)


def triplet_diameter(
    d: np.ndarray, i: np.ndarray, j: np.ndarray, k: np.ndarray
) -> np.ndarray:
    """Triplet diameter = max of the three pairwise distances."""
    return np.maximum(np.maximum(d[i, j], d[i, k]), d[j, k])


# --------------------------------------------------------------------------- #
# Index permutations                                                          #
# --------------------------------------------------------------------------- #


def expand_atom_perm_to_dofs(atom_perm: np.ndarray) -> np.ndarray:
    """Lift an atom-level permutation to a 3-DOF index permutation."""
    return (3 * atom_perm[:, None] + np.arange(3)[None, :]).ravel()


# --------------------------------------------------------------------------- #
# Block slicing                                                               #
# --------------------------------------------------------------------------- #


def project_dense_to_blocks(
    sigma_dense: np.ndarray,
    block_sizes: np.ndarray,
    *,
    nn_only: bool = True,
) -> dict[tuple[int, int], np.ndarray]:
    """Slice a dense ``(n_freq, N, N)`` Σ into the block-tridiagonal dict."""
    block_sizes = np.asarray(block_sizes, dtype=int)
    offsets = np.concatenate(([0], np.cumsum(block_sizes)))
    n_blocks = block_sizes.size
    out: dict[tuple[int, int], np.ndarray] = {}
    for I in range(n_blocks):
        for J in range(n_blocks):
            if nn_only and abs(I - J) > 1:
                continue
            out[(I, J)] = np.ascontiguousarray(
                sigma_dense[:, offsets[I]:offsets[I + 1], offsets[J]:offsets[J + 1]]
            )
    return out
