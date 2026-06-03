"""q-resolved device FC3 vertex folding for the unified self-energy kernel.

The self-energy itself is computed by
:func:`phonon.solver.se_finite.compute_phph_self_energy`. This module only
builds the q-folded device vertex it consumes:

* :func:`_qfold_device_blocks` -- the FC3 device blocks for one transverse
  momentum pair (q1, q2), via transverse Bloch phases on the two contracted legs.
* :func:`_build_folded_vertices` -- the full ``{(iq1, iq2): {(I, K, K'): Phi}}``
  map the coupled-q bubble loop needs (built once per SCBA solve; it does not
  depend on G).
* :func:`compute_phph_self_energy_q_dense_multi_slab` -- thin wrapper that builds
  the folded vertices and calls the unified kernel. Kept for direct callers.
"""

from __future__ import annotations

import numpy as np


def _qfold_device_blocks(M_stacked, prim_indices, cell_frac, slab_indices,
                         n_atoms, n_slabs, q1_frac, q2_frac, transport_direction,
                         vertex_cutoff=None):
    """q-folded device FC3 blocks {(I, K, K'): Phi(q1, q2)[a, b, c]}.

    Reuses :func:`fc3_device.build_device_fc3_blocks` on a phase-modified
    M_stacked: the two CONTRACTED legs (b<-q1, c<-q2) carry transverse Bloch
    phases exp(-2 pi i q . R_perp), exactly as
    :func:`separable.build_gathering_matrix`, so summing supercell images
    reproduces T(q1) M T(q2) per device block. The external leg (a) carries no
    phase. q1_frac / q2_frac are 2-component transverse fractions.
    """
    from .fc3_device import build_device_fc3_blocks
    n_dof = 3 * n_atoms
    n_super = len(prim_indices)
    dim_sc = n_super * 3
    tidx = "xyz".index(transport_direction)
    perp = [i for i in range(3) if i != tidx]
    q1 = np.zeros(3); q1[perp[0]], q1[perp[1]] = q1_frac
    q2 = np.zeros(3); q2[perp[0]], q2[perp[1]] = q2_frac
    ph1 = np.exp(-2j * np.pi * cell_frac @ q1)        # (n_super,)
    ph2 = np.exp(-2j * np.pi * cell_frac @ q2)
    Mb = M_stacked.reshape(n_dof, dim_sc, dim_sc).astype(complex)
    # per-supercell-atom phases on the two contracted legs (3 dofs/atom)
    p1 = np.repeat(ph1, 3); p2 = np.repeat(ph2, 3)
    Mq = (Mb * p1[None, :, None]) * p2[None, None, :]
    Mq_stacked = Mq.reshape(n_dof * dim_sc, dim_sc)
    return build_device_fc3_blocks(
        Mq_stacked, prim_indices, slab_indices, n_atoms, n_slabs,
        vertex_cutoff=vertex_cutoff)


def _build_folded_vertices(M_stacked, prim_indices, cell_frac, slab_indices,
                           n_atoms, n_slabs, n_kpts, q_points, q_diff_map,
                           transport_direction, vertex_cutoff=None):
    """q-folded device vertex {(iq1, iq2): {(I, K, K'): Phi}} for every pair the
    coupled-q bubble loop needs.

    Each external q couples internal q' to q2 = q_ext - q'; the left vertex
    carries legs (q', q2) and the right vertex (q2, q'). Independent of G, so the
    driver builds this once per SCBA solve. The q=Gamma pair (0, 0) is always
    included (the unified kernel takes the (I, J) pair index from it).
    """
    folded_pairs = {(0, 0)}
    for iq_ext in range(n_kpts):
        for iqp in range(n_kpts):
            iq2 = int(q_diff_map[iq_ext, iqp])
            folded_pairs.add((iqp, iq2))
            folded_pairs.add((iq2, iqp))
    return {
        (iq1, iq2): _qfold_device_blocks(
            M_stacked, prim_indices, cell_frac, slab_indices, n_atoms, n_slabs,
            q_points[iq1], q_points[iq2], transport_direction,
            vertex_cutoff=vertex_cutoff)
        for (iq1, iq2) in folded_pairs
    }


def compute_phph_self_energy_q_dense_multi_slab(
    g_lesser_blocks_q, g_greater_blocks_q,
    M_stacked, prim_indices, cell_frac, slab_indices,
    n_atoms, n_slabs, n_kpts, q_points, q_diff_map,
    omega_grid_thz, dw_thz, transport_direction="x", *,
    sigma_cutoff=None, g_cutoff=None, vertex_cutoff=None,
    dc_handling="interpolate", symmetry_factor=None,
):
    """FULL off-diagonal q-resolved 3-phonon self-energy.

    Thin wrapper: builds the q-folded device vertices from ``M_stacked`` + the
    supercell mapping, then calls
    :func:`phonon.solver.se_finite.compute_phph_self_energy`.

    G blocks: ``{(K, K'): (n_kpts, n_freq, n_dof, n_dof)}``. Returns
    ``({(I, J): Sigma^<(n_kpts, n_freq, n_dof, n_dof)}, {(I, J): Sigma^>...})``.
    ``sigma_cutoff`` bounds output ``|I - J|`` (``0`` = Guo approximation III),
    ``g_cutoff`` bounds the input G range, ``vertex_cutoff`` the FC3 slab reach.
    """
    from .se_finite import compute_phph_self_energy

    vertices = _build_folded_vertices(
        M_stacked, prim_indices, cell_frac, slab_indices, n_atoms, n_slabs,
        n_kpts, q_points, q_diff_map, transport_direction,
        vertex_cutoff=vertex_cutoff)
    return compute_phph_self_energy(
        g_lesser_blocks_q, g_greater_blocks_q, vertices, n_slabs, n_kpts,
        q_diff_map, omega_grid_thz, dw_thz,
        sigma_cutoff=sigma_cutoff, g_cutoff=g_cutoff,
        dc_handling=dc_handling, symmetry_factor=symmetry_factor)
