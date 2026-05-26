"""Rigid-translation (zero-mode) handling for the dense phonon solver.

The 3-phonon bubble self-energy is not guaranteed to respect the
acoustic sum rule: ``Sigma^R(omega)`` generally has a nonzero
projection onto the rigid-translation subspace. Added to the device
dynamical matrix in the Dyson denominator ``z2 I - H_D - Sigma^R`` that
component shifts the (near-)zero ``omega^2`` eigenvalues of the acoustic
modes -- possibly *negative*, which puts a pole of ``G^R`` on the real
axis and makes the SCBA fixed point linearly unstable. A multi-slab
device has ``n_slabs`` low-lying (folded-acoustic) modes, so the effect
is much worse there than for a single slab.

The cure is a discrete acoustic sum rule on the self-energy: project
``Sigma`` onto the subspace orthogonal to rigid translations before it
enters the Dyson solve, so anharmonic scattering can never renormalise
the translational modes.

Self-energies live in the same mass-weighted ``THz^2`` space as the
dynamical matrix, so the translational zero modes are
``t_alpha[l, i, beta] = sqrt(m_i) delta_{alpha beta}`` -- a uniform
*mass-weighted* displacement of every atom of the device in cartesian
direction ``alpha``.
"""

from __future__ import annotations

import numpy as np


def translation_vectors(masses_primitive, n_slabs, n_cart=3):
    """Orthonormal mass-weighted rigid-translation modes of the device.

    Parameters
    ----------
    masses_primitive : (n_atoms,) array
        Per-atom masses of one primitive-cell slab (amu). Every slab of
        the finite device is a copy of this cell.
    n_slabs : int
        Number of slabs in the device.
    n_cart : int
        Cartesian dimensions per atom (3 for a real phonon device; 1
        for the 1-D analytic toy chains).

    Returns
    -------
    T : (N_D, n_cart) real array
        Columns are the orthonormal translation modes;
        ``N_D = n_slabs * n_cart * n_atoms``. DOF ordering is
        atom-major (``index = l*n_dof + n_cart*i + alpha``), matching
        the device Hamiltonian built by
        :func:`phonon.solver.leads.build_device_hamiltonian`.
    """
    masses = np.asarray(masses_primitive, dtype=float)
    if masses.ndim != 1:
        raise ValueError("masses_primitive must be a 1-D per-atom array")
    n_atoms = masses.shape[0]
    n_dof = n_cart * n_atoms
    N_D = n_slabs * n_dof

    sqrt_m = np.sqrt(masses)
    T = np.zeros((N_D, n_cart))
    for alpha in range(n_cart):
        vec = np.zeros(N_D)
        for l in range(n_slabs):
            base = l * n_dof
            # indices n_cart*i + alpha for i = 0 .. n_atoms-1 (atom-major).
            vec[base + alpha:base + n_dof:n_cart] = sqrt_m
        T[:, alpha] = vec / np.linalg.norm(vec)
    return T


def build_translation_projector(masses_primitive, n_slabs, n_cart=3):
    """Projector ``Q = I - T T^T`` onto the translation-free subspace.

    ``Q @ Sigma @ Q`` removes every matrix element of ``Sigma`` that
    couples into or out of the rigid-translation subspace, enforcing a
    discrete acoustic sum rule on the self-energy. ``Q`` is real,
    symmetric and idempotent. ``n_cart`` is the number of cartesian
    dimensions per atom (3 for a real device, 1 for the 1-D toys).
    """
    T = translation_vectors(masses_primitive, n_slabs, n_cart=n_cart)
    N_D = T.shape[0]
    return np.eye(N_D) - T @ T.T


def project_self_energy(sigma, Q, *, in_place=False):
    """Apply ``Q @ Sigma @ Q`` on the last two axes of ``sigma``.

    ``sigma`` has shape ``(..., N_D, N_D)`` (any number of leading batch
    axes -- typically ``(n_kpts, nfreq)``). ``Q`` is the ``(N_D, N_D)``
    projector from :func:`build_translation_projector`. The projection
    is linear, so applying it to ``Sigma^{<,>}`` carries through the
    (linear) ``build_retarded`` reconstruction to ``Sigma^R``.
    """
    projected = Q @ sigma @ Q
    if in_place:
        sigma[...] = projected
        return sigma
    return projected


def build_dynamical_zero_mode_projector(H_00, H_01, n_slabs, *,
                                         threshold_rel=1e-4):
    """Projector onto the complement of *every* near-zero mode of the
    cell dynamical matrix at the zone centre.

    The translation projector :func:`build_translation_projector`
    handles the three rigid-cartesian zero modes that are guaranteed
    by the FC2 acoustic sum rule. A 1-D wire at q=0 generally has a
    fourth near-zero mode -- the rigid rotation about the wire axis
    ("twist") -- which leaves the translation projector untouched but
    feeds the same instability: its Bose-enhanced ``G^<`` at
    ``omega -> 0`` injects an IR singularity into the bubble that
    drives ``Im Sigma^R`` non-causal and the SCBA loop unstable. This
    projector reads the cell-level dynamical matrix
    ``H_00 + H_01 + H_01^dagger`` (the periodic Gamma matrix), picks
    every eigenvector with eigenvalue below ``threshold_rel *
    max(eigvals)``, replicates each one uniformly across the
    ``n_slabs`` device slabs (the q=0 phase), orthonormalises the
    replicated vectors, and returns ``Q = I - V V^T``.

    For a typical wire this yields the same 3 translation modes plus
    the twist, and for a higher-symmetry system any additional
    accidentally-soft cell mode. For a bulk crystal with FC2 ASR clean
    the result reduces to the translation projector.

    Parameters
    ----------
    H_00, H_01 : (n_dof, n_dof) complex
        Periodic dynamical-matrix on-site and coupling blocks in
        THz^2 (as built by :func:`phonon_inputs.convention.get_btd_blocks`).
    n_slabs : int
    threshold_rel : float
        Eigenvalue cutoff, relative to the largest eigenvalue of the
        cell dynamical matrix. Default ``1e-4`` matches the threshold
        used by :func:`verify_zero_modes.check_acoustic_modes`.

    Returns
    -------
    Q : (n_slabs*n_dof, n_slabs*n_dof) real array
        Idempotent projector onto the device subspace orthogonal to
        every replicated cell zero mode.
    """
    H_00 = np.asarray(H_00)
    H_01 = np.asarray(H_01)
    n_dof = H_00.shape[0]
    dyn = H_00 + H_01 + H_01.conj().T
    dyn = 0.5 * (dyn + dyn.conj().T)
    eigvals, eigvecs = np.linalg.eigh(dyn)
    scale = float(eigvals.max().real)
    cutoff = float(threshold_rel) * scale
    zero_idx = np.where(eigvals.real < cutoff)[0]
    if zero_idx.size == 0:
        N_D = n_slabs * n_dof
        return np.eye(N_D)
    V_cell = eigvecs[:, zero_idx]  # (n_dof, n_zero)
    N_D = n_slabs * n_dof
    # Replicate each cell mode uniformly across slabs (q=0 phase) so
    # the device-level mode is the rigid-body extension of the cell
    # mode -- the actual zero mode of the open finite device in the
    # absence of leads.
    V_dev = np.zeros((N_D, V_cell.shape[1]), dtype=V_cell.dtype)
    for k in range(V_cell.shape[1]):
        for l in range(n_slabs):
            V_dev[l * n_dof:(l + 1) * n_dof, k] = V_cell[:, k]
    # Orthonormalise (the replicated vectors are orthogonal as long as
    # the cell modes are, which np.linalg.eigh guarantees).
    norms = np.linalg.norm(V_dev, axis=0)
    V_dev = V_dev / np.where(norms > 0, norms, 1.0)
    Q = np.eye(N_D, dtype=V_dev.dtype) - V_dev @ V_dev.conj().T
    if np.allclose(Q.imag, 0.0):
        Q = Q.real
    return Q


def translation_leakage(sigma, Q):
    """Relative weight of ``sigma`` in the translational subspace.

    Returns ``||sigma - Q sigma Q|| / ||sigma||`` -- the fraction of the
    self-energy that couples to rigid translations. Zero (up to
    round-off) once :func:`project_self_energy` has been applied;
    a diagnostic for ``verify_zero_modes``.
    """
    sig = np.asarray(sigma)
    norm = float(np.linalg.norm(sig))
    if norm == 0.0:
        return 0.0
    return float(np.linalg.norm(sig - Q @ sig @ Q) / norm)
