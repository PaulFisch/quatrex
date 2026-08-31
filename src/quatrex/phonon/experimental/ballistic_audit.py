"""Independent identities used to audit a harmonic phonon transport run.

This module is deliberately not part of the solver path.  The study driver
calls it only when ``QX_DIAG_CAROLI=1`` so that enabling the audit cannot
change the Dyson equation, contact construction, or reported current.
"""

from __future__ import annotations

import numpy as np

from qttools import NDArray, xp


def caroli_transmission(
    g_retarded: NDArray,
    sigma_left_retarded: NDArray,
    sigma_right_retarded: NDArray,
) -> NDArray:
    r"""Return :math:`Tr[\Gamma_L G^R \Gamma_R G^A]` per stack point.

    All leading dimensions are batch dimensions.  The last two dimensions
    are the complete finite-device matrix, as they are for a one-block
    grouped device in the production phonon solver.
    """
    gamma_left = 1j * (
        sigma_left_retarded
        - sigma_left_retarded.conj().swapaxes(-2, -1)
    )
    gamma_right = 1j * (
        sigma_right_retarded
        - sigma_right_retarded.conj().swapaxes(-2, -1)
    )
    g_advanced = g_retarded.conj().swapaxes(-2, -1)
    return xp.real(xp.trace(
        gamma_left @ g_retarded @ gamma_right @ g_advanced,
        axis1=-2,
        axis2=-1,
    ))


def caroli_number_current(
    transmission: NDArray,
    left_occupancy: NDArray,
    right_occupancy: NDArray,
) -> NDArray:
    r"""Return the harmonic number-current spectrum ``(nL-nR) T``.

    Occupancies are one-dimensional frequency arrays.  Extra dimensions in
    ``transmission`` are transverse momenta and are broadcast explicitly.
    """
    delta_n = left_occupancy - right_occupancy
    shape = (delta_n.shape[0],) + (1,) * (transmission.ndim - 1)
    return transmission * delta_n.reshape(shape)


def spectrum_error(reference: np.ndarray, value: np.ndarray) -> dict[str, float]:
    """Scale-aware errors for a spectral identity, including active bins."""
    reference = np.asarray(reference, dtype=float)
    value = np.asarray(value, dtype=float)
    difference = value - reference
    scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    norm = float(np.linalg.norm(reference.ravel()))
    l2 = float(np.linalg.norm(difference.ravel()) / max(norm, np.finfo(float).tiny))
    active = np.abs(reference) > max(1e-12 * scale, np.finfo(float).tiny)
    pointwise = (
        float(np.max(np.abs(difference[active]) / np.abs(reference[active])))
        if np.any(active) else 0.0
    )
    return {
        "relative_l2": l2,
        "active_max_relative": pointwise,
        "absolute_max": float(np.max(np.abs(difference))) if difference.size else 0.0,
    }
