"""Opt-in cutoff policies for the dense phonon solver.

The solver defaults to no approximation: full off-diagonal G
blocks inside the bubble, full FC3 vertex, no magnitude or distance
threshold. The four knobs below can be flipped individually for
cutoff-sensitivity studies:

  * ``diag_G_in_se`` - restrict the inner G dict in the bubble integrand
    to the diagonal (K₁ = K₁', K₂ = K₂') blocks. Recovers the original
    block-tridiagonal NEGF approximation when ``True``.
  * ``fc3_nn_only`` - drop FC3 block triplets with ``|I-J| > 1``,
    ``|I-K| > 1`` or ``|J-K| > 1``
  * ``fc3_distance_cutoff`` — drop FC3 entries whose triplet diameter
    (max of three pairwise atomic distances) exceeds the cutoff in Å.
  * ``fc3_magnitude_threshold`` — drop FC3 entries whose magnitude is
    below ``threshold × max|Φ|``.

The historical ``diag_G_everywhere`` knob (which strips off-diagonals
globally before any computation) is intentionally NOT preserved: it only
made sense for the eigenmode-Lorentzian synthetic G inside the deleted
``finite_analysis/sse_cutoffs.py`` and has no equivalent in the
production solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CutoffPolicy:
    """Opt-in cutoff knobs for the bubble evaluation.

    Defaults = no approximation. Pass an instance into
    :func:`phonon.solver.se_block.compute_sigma_from_blocks` (or the
    corresponding entry on a future SCBA path) to opt into one or more
    of the cutoffs documented in this module's docstring.
    """

    diag_G_in_se: bool = False
    fc3_nn_only: bool = False
    fc3_distance_cutoff: Optional[float] = None
    fc3_magnitude_threshold: Optional[float] = None


def apply_fc3_cutoffs(
    phi_blocks: dict[tuple[int, int, int], np.ndarray],
    block_sizes: np.ndarray,
    *,
    policy: CutoffPolicy,
    distances_atom: np.ndarray | None = None,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Return a copy of ``phi_blocks`` with the policy's FC3 cutoffs applied.

    Independently masks blocks by distance and magnitude when the policy
    sets those fields. ``fc3_nn_only`` filters out any block triplet
    that violates the nearest-neighbour band; ``fc3_distance_cutoff``
    and ``fc3_magnitude_threshold`` apply per-element masks.
    """
    if not phi_blocks:
        return {}

    block_sizes = np.asarray(block_sizes, dtype=int)
    offsets = np.concatenate(([0], np.cumsum(block_sizes)))

    if policy.fc3_magnitude_threshold is not None:
        max_abs = max(np.abs(b).max() for b in phi_blocks.values()) or 1.0
        mag_floor = policy.fc3_magnitude_threshold * max_abs
    else:
        mag_floor = None

    out: dict[tuple[int, int, int], np.ndarray] = {}
    for (I, J, K), block in phi_blocks.items():
        if policy.fc3_nn_only and (
            abs(I - J) > 1 or abs(I - K) > 1 or abs(J - K) > 1
        ):
            continue

        modified = block.copy()
        if policy.fc3_distance_cutoff is not None:
            if distances_atom is None:
                raise ValueError(
                    "distances_atom required for fc3_distance_cutoff"
                )
            i_atoms = (offsets[I] + np.arange(block.shape[0])) // 3
            j_atoms = (offsets[J] + np.arange(block.shape[1])) // 3
            k_atoms = (offsets[K] + np.arange(block.shape[2])) // 3
            d_ij = distances_atom[i_atoms[:, None], j_atoms[None, :]]
            d_ik = distances_atom[i_atoms[:, None], k_atoms[None, :]]
            d_jk = distances_atom[j_atoms[:, None], k_atoms[None, :]]
            diam = np.maximum(
                np.maximum(d_ij[:, :, None], d_ik[:, None, :]),
                d_jk[None, :, :],
            )
            modified = np.where(
                diam > policy.fc3_distance_cutoff, 0.0, modified,
            )

        if mag_floor is not None:
            modified = np.where(np.abs(modified) < mag_floor, 0.0, modified)

        if np.any(modified):
            out[(I, J, K)] = modified
    return out


def diagonalise_g_blocks(
    g_blocks: dict[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    """Drop off-(K, K') blocks of G, keeping only the K = K' diagonals.

    Used to opt into the ``diag_G_in_se`` cutoff: with the inner G
    restricted to its block-diagonal, the bubble integrand recovers the
    original quatrex ``SigmaPhononPhonon._bubble_block`` (K₁ = K₁',
    K₂ = K₂') behaviour.
    """
    return {(I, J): block for (I, J), block in g_blocks.items() if I == J}
