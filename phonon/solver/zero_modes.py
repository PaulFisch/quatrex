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
