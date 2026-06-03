"""Device-resolved 3-phonon vertex block tensor.

Builds the FC3 vertex ``Phi_{I,K,K'}[a, b, c]`` in primitive-DOF
coordinates, indexed by slab triplets ``(I, K, K')`` of a finite,
multi-slab device.

The supercell FC3 is translation-invariant along the transport
direction, so the kernel at slab ``I`` is a pure shift of the ``I=0``
kernel:

    Phi_{I, K, K'} = Phi_{0, K-I, K'-I}

The shift is read out via minimum-image lookup against the supercell of
width ``N_super_z``: atoms ``|delta| > N_super_z // 2`` slabs apart are not
representable in the supercell FC3, so those triplets are silently zero
(consistent with the finite-device interpretation; the supercell PBC is
NOT replicated onto the device).

Used by :func:`phonon.solver.dense.transmission` to feed the multi-slab
self-energy kernels
:func:`phonon.solver.se_finite.compute_phph_self_energy_finite_multi_slab`
and :func:`phonon.solver.se_q.compute_phph_self_energy_q_dense_multi_slab`.
"""

from __future__ import annotations

import numpy as np


def _minimum_image_offset(delta_raw: int, n_super_z: int) -> int:
    """Project a raw slab offset to its minimum-image equivalent.

    For supercell width ``N_z``, the minimum-image offset lives in
    ``[-N_z // 2, N_z // 2]``. Atoms separated by raw offset ``delta``
    are placed at their closest periodic image when the supercell PBC
    makes the direct distance ambiguous.
    """
    if n_super_z <= 0:
        raise ValueError("n_super_z must be positive")
    delta = int(delta_raw) % n_super_z
    if delta > n_super_z // 2:
        delta -= n_super_z
    return delta


def build_device_fc3_blocks(
    M_stacked: np.ndarray,
    prim_indices: np.ndarray,
    slab_indices: np.ndarray,
    n_atoms: int,
    n_slabs: int,
    *,
    vertex_cutoff: int | None = None,
    return_offsets: bool = False,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Build the device-level FC3 dict ``{(I, K, K'): Phi[n_dof, n_dof, n_dof]}``.

    Parameters
    ----------
    M_stacked
        Mass-weighted FC3 from
        :func:`phonon.phonon_inputs.separable.build_realspace_fc3_matrices`,
        shape ``(n_dof * dim_sc, dim_sc)``. Equivalent to a tensor
        ``M[a, s2*3+alpha, s3*3+beta]`` indexed by primitive DOF ``a``
        and two supercell DOFs.
    prim_indices
        Length-``n_super`` map: which primitive atom each supercell
        atom corresponds to (from
        :func:`phonon.phonon_inputs.separable.build_supercell_mapping`).
    slab_indices
        Length-``n_super`` map: transport-direction slab index in
        ``[0, N_super_z)`` for each supercell atom.
    n_atoms
        Number of primitive-cell atoms.
    n_slabs
        Number of primitive-cell slabs in the device.
    vertex_cutoff
        Maximum block-distance retained. Triplets with
        ``max(|I-K|, |I-K'|, |K-K'|) > vertex_cutoff`` are dropped.
        ``None`` (default) imposes no extra truncation; the result
        contains every triplet the supercell FC3 can resolve
        (``|delta| <= N_super_z // 2``).
    return_offsets
        If True, return the underlying offset dict
        ``{(delta_K, delta_K'): Phi}`` instead of materialising the
        full device dict. Useful in memory-constrained inner loops.

    Returns
    -------
    phi_dev or phi_offsets
        Either the device triplet dict ``{(I, K, K'): Phi}`` or, when
        ``return_offsets=True``, the offset dict.

        Phi blocks in the device dict are **shared references** to the
        underlying offset kernel: callers must not mutate them in
        place. Use ``.copy()`` if mutation is needed.
    """
    n_super = int(prim_indices.shape[0])
    n_dof = n_atoms * 3
    dim_sc = n_super * 3
    n_super_z = int(slab_indices.max()) + 1
    half_window = n_super_z // 2

    M_blocks = M_stacked.reshape(n_dof, dim_sc, dim_sc)

    phi_offsets: dict[tuple[int, int], np.ndarray] = {}
    for s2 in range(n_super):
        d_k = _minimum_image_offset(int(slab_indices[s2]), n_super_z)
        p2 = int(prim_indices[s2])
        b_lo = p2 * 3
        b_hi = b_lo + 3
        for s3 in range(n_super):
            d_kp = _minimum_image_offset(int(slab_indices[s3]), n_super_z)
            p3 = int(prim_indices[s3])
            c_lo = p3 * 3
            c_hi = c_lo + 3
            key = (d_k, d_kp)
            entry = phi_offsets.get(key)
            if entry is None:
                entry = np.zeros(
                    (n_dof, n_dof, n_dof), dtype=M_blocks.dtype
                )
                phi_offsets[key] = entry
            entry[:, b_lo:b_hi, c_lo:c_hi] += (
                M_blocks[:, s2 * 3:(s2 + 1) * 3, s3 * 3:(s3 + 1) * 3]
            )

    phi_offsets = {k: v for k, v in phi_offsets.items() if np.any(v)}

    if return_offsets:
        return phi_offsets

    phi_dev: dict[tuple[int, int, int], np.ndarray] = {}
    for I in range(n_slabs):
        for K in range(n_slabs):
            d_k = K - I
            if abs(d_k) > half_window:
                continue
            for kp in range(n_slabs):
                d_kp = kp - I
                if abs(d_kp) > half_window:
                    continue
                if abs(K - kp) > half_window:
                    continue
                if vertex_cutoff is not None:
                    if max(abs(d_k), abs(d_kp), abs(K - kp)) > vertex_cutoff:
                        continue
                phi = phi_offsets.get((d_k, d_kp))
                if phi is None:
                    continue
                phi_dev[(I, K, kp)] = phi

    return phi_dev
