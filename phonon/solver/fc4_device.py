"""Device-resolved quartic (FC4) vertex tensor for the loop self-energy.

The quartic loop self-energy ``Sigma_L = 1/2 Phi4 : <uu>`` needs the FC4 only
as a dense device tensor ``Phi4_dev[A, B, C, D]`` (device DOFs), contracted on
the last two legs with the equal-time correlation. This module builds that
tensor from a **compact-reference sparse** FC4 -- the FC4 analogue of
:func:`phonon.solver.fc3_device.build_device_fc3_blocks`.

Storage format (compact-reference sparse)
-----------------------------------------
``fc4_sparse`` is ``{(s1, s2, s3, s4): T[3,3,3,3]}`` in eV/Angstrom^4, where the
FIRST atom ``s1`` is a **slab-0 reference** supercell atom (the image of a
primitive atom in the reference cell) and ``s2, s3, s4`` are arbitrary supercell
atoms. By translation invariance every device element is obtained by anchoring
leg 1 at a reference atom and folding the other three legs onto device slabs by
minimum image -- exactly the ``M_stacked`` convention used for FC3, with one
more leg. Building this compact-reference dict from hiPhive's label-reduced
``get_fc_dict(order=4)`` (permutation expansion + reference filter + dedupe)
lives on the export side; this module only consumes it.

Mass-weighting: each leg is divided by ``sqrt(mass)`` (no THz conversion -- the
loop applies ``CONVERSION_THZ2``), matching
:func:`phonon.solver.static_se.sigma_loop`.
"""

from __future__ import annotations

import numpy as np

from .fc3_device import _minimum_image_offset


def build_device_fc4_tensor(
    fc4_sparse,
    prim_indices,
    slab_indices,
    masses_super,
    n_atoms,
    n_slabs,
    *,
    vertex_cutoff=None,
):
    """Dense mass-weighted device FC4 tensor ``Phi4_dev[A, B, C, D]``.

    Parameters
    ----------
    fc4_sparse : dict
        ``{(s1, s2, s3, s4): T[3,3,3,3]}`` compact-reference FC4 (eV/Angstrom^4);
        ``s1`` a slab-0 reference supercell atom (see module docstring).
    prim_indices, slab_indices : (n_super,) int arrays
        Primitive-atom and transport-slab index of each supercell atom (from
        :func:`phonon.phonon_inputs.separable.build_supercell_mapping`).
    masses_super : (n_super,) array
        Supercell atom masses [amu].
    n_atoms, n_slabs : int
        Primitive-cell atoms and number of device slabs.
    vertex_cutoff : int, optional
        Drop quadruples whose maximum pairwise slab distance exceeds this.

    Returns
    -------
    Phi4_dev : (N_D, N_D, N_D, N_D) real array
        ``N_D = n_slabs * 3 * n_atoms``. Mass-weighted (``1/sqrt(m_a m_b m_c
        m_d)``), no THz conversion. Feed directly to
        :func:`phonon.solver.static_se.sigma_loop`.
    """
    prim_indices = np.asarray(prim_indices)
    slab_indices = np.asarray(slab_indices)
    sqrt_m = np.sqrt(np.asarray(masses_super, dtype=float))
    nd = 3 * n_atoms
    N_D = n_slabs * nd
    n_super_z = int(slab_indices.max()) + 1
    half_window = n_super_z // 2

    Phi4 = np.zeros((N_D, N_D, N_D, N_D), dtype=float)

    fc4_sparse = dict(fc4_sparse)
    for (s1, s2, s3, s4), tens in fc4_sparse.items():
        z1 = int(slab_indices[s1])
        d2 = _minimum_image_offset(int(slab_indices[s2]) - z1, n_super_z)
        d3 = _minimum_image_offset(int(slab_indices[s3]) - z1, n_super_z)
        d4 = _minimum_image_offset(int(slab_indices[s4]) - z1, n_super_z)
        offs = (0, d2, d3, d4)
        spread = max(offs) - min(offs)
        if spread > half_window:
            continue
        if vertex_cutoff is not None and spread > vertex_cutoff:
            continue

        p1 = int(prim_indices[s1])
        p2 = int(prim_indices[s2])
        p3 = int(prim_indices[s3])
        p4 = int(prim_indices[s4])
        mw = 1.0 / (sqrt_m[s1] * sqrt_m[s2] * sqrt_m[s3] * sqrt_m[s4])
        block = np.asarray(tens, dtype=float) * mw

        for I in range(n_slabs):
            J, K, L = I + d2, I + d3, I + d4
            if not (0 <= J < n_slabs and 0 <= K < n_slabs and 0 <= L < n_slabs):
                continue
            a = I * nd + 3 * p1
            b = J * nd + 3 * p2
            c = K * nd + 3 * p3
            d = L * nd + 3 * p4
            Phi4[a:a + 3, b:b + 3, c:c + 3, d:d + 3] += block

    return Phi4


def build_compact_reference_fc4_from_dense(fc4_dense, ref_sc_atoms, *, tol=1e-8):
    """Compact-reference sparse FC4 from a dense supercell FC4 (export helper).

    For each primitive atom ``p`` with slab-0 reference image ``ref_sc_atoms[p]``,
    slice ``fc4_dense[ref, :, :, :]`` and keep the nonzero
    ``(s2, s3, s4)`` quadruples. The dense tensor is fully symmetric and
    translation-invariant (as hiPhive returns it), so this slice is exactly the
    leg-1-anchored compact-reference the device builder consumes -- correct by
    construction, no permutation/translation bookkeeping. Feasible only when the
    supercell is small enough to materialise ``fc4_dense`` (short-ranged FC4);
    big supercells need the sparse ``get_fc_dict`` unfold route (future work).

    Parameters
    ----------
    fc4_dense : ndarray, shape (n_super, n_super, n_super, n_super, 3, 3, 3, 3)
        Dense supercell FC4 in eV/Angstrom^4 (``ForceConstants.get_fc_array``).
    ref_sc_atoms : (n_atoms,) int
        Slab-0 reference supercell atom for each primitive atom.
    tol : float
        Magnitude threshold for keeping a ``(s2, s3, s4)`` block.

    Returns
    -------
    fc4_sparse : dict ``{(s1, s2, s3, s4): T[3,3,3,3]}``
    """
    fc4_dense = np.asarray(fc4_dense)
    out = {}
    for p, s1 in enumerate(np.asarray(ref_sc_atoms, dtype=int)):
        sub = fc4_dense[s1]                       # (n_super, n_super, n_super, 3,3,3,3)
        mag = np.max(np.abs(sub), axis=(3, 4, 5, 6))
        for s2, s3, s4 in np.argwhere(mag > tol):
            out[(int(s1), int(s2), int(s3), int(s4))] = sub[s2, s3, s4]
    return out
