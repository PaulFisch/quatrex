# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Exact infrared (Bose-pole) treatment for the phonon self-energy.

The Bose factor :math:`n(\\omega)=1/(e^{\\hbar\\omega/k_BT}-1)\\simeq k_BT/\\hbar\\omega`
carries a :math:`1/\\omega` pole at :math:`\\omega\\to0` that, on a discrete
frequency grid, corrupts the low-frequency self-energy and the injected current
(see ``document/src/appendices/infrared.tex``). This module removes it *exactly*
by singularity subtraction,

.. math::

    n(\\omega) = \\frac{c}{\\omega} + \\tilde n(\\omega), \\qquad c=\\frac{k_BT}{\\hbar},

with :math:`\\tilde n` bounded and analytic (:math:`\\tilde n\\to-\\tfrac12` as
:math:`\\omega\\to0`), and evaluates the residual Cauchy principal value of the
:math:`c/\\omega` part with a node-omitting (subtracted) trapezoidal rule -- the
:math:`\\omega\\to0` analogue of the :math:`1/(\\omega-\\omega')` Kramers--Kronig
subtraction already used in :mod:`phonon.solver.retarded`. No taper or
``omega_reg`` cutoff is introduced; the :math:`\\omega_{\\rm reg}\\to0` limit is
taken analytically.
"""
from __future__ import annotations

import numpy as np

HBAR_EVS = 6.582119569e-16  # eV s
KB_EV = 8.617333262e-5      # eV / K
THZ_TO_RAD = 2.0 * np.pi * 1e12

_DC = 1e-12  # |omega| below this (THz) is treated as the omega=0 node


def bose(omega_thz, temperature):
    """Bose--Einstein occupation ``n(omega)`` (omega in THz, T in K)."""
    w = np.asarray(omega_thz, dtype=float)
    big = np.abs(w) > _DC
    x = np.where(big, HBAR_EVS * THZ_TO_RAD * w / (KB_EV * temperature), 1.0)
    return np.where(big, 1.0 / np.expm1(x), 0.0)


def bose_pole_coeff(temperature):
    """Residue ``c`` of the Bose pole ``n ~ c/omega`` (omega in THz): ``c=kT/hbar``."""
    return KB_EV * temperature / (HBAR_EVS * THZ_TO_RAD)


def bose_split(omega_thz, temperature):
    """Split ``n(omega) = c/omega + n_tilde(omega)`` exactly.

    Returns ``(n_tilde, c)``. ``n_tilde`` is bounded (``-> -1/2`` as
    ``omega->0``) and analytic; ``c = kT/hbar`` is the pole residue (THz units).
    The split is exact: ``n_tilde + c/omega == bose(omega, T)`` off the node.
    """
    w = np.asarray(omega_thz, dtype=float)
    c = bose_pole_coeff(temperature)
    big = np.abs(w) > _DC
    n = bose(w, temperature)
    n_tilde = np.where(big, n - c / np.where(big, w, 1.0), -0.5)
    return n_tilde, c


def pv_origin(q, omega):
    """Cauchy principal value :math:`\\mathcal P\\!\\int q(\\omega')/\\omega'\\,d\\omega'`.

    The pole sits at ``omega'=0``, which must be a node of the uniform grid
    ``omega``. Singularity subtraction (cf.
    :func:`phonon.solver.retarded.retarded_from_lesser_greater`, pole at
    ``omega'=omega``):

    .. math::

        \\mathcal P\\!\\int \\frac{q(\\omega')}{\\omega'}\\,d\\omega'
        = \\int \\frac{q(\\omega')-q(0)}{\\omega'}\\,d\\omega'
        + q(0)\\,\\mathcal P\\!\\int_a^b \\frac{d\\omega'}{\\omega'} .

    The first integrand is regular -- its value at the node is :math:`q'(0)` --
    and is taken by the trapezoidal rule; the second is the exact scalar
    :math:`\\mathcal P\\!\\int_a^b d\\omega'/\\omega' = \\ln(b/|a|)` for
    :math:`a<0<b` (a discrete fallback is used if the pole is at an endpoint).
    ``q`` may be real or complex; with shape ``(nw, ...)`` the transform acts on
    the leading (frequency) axis.
    """
    w = np.asarray(omega, dtype=float)
    q = np.asarray(q)
    nw = w.size
    if nw < 3:
        raise ValueError("pv_origin needs at least 3 grid points")
    dw = w[1] - w[0]
    if not np.allclose(np.diff(w), dw):
        raise ValueError("pv_origin requires a uniform grid")
    i0 = int(np.argmin(np.abs(w)))
    if abs(w[i0]) > 1e-9 * (w[-1] - w[0]):
        raise ValueError("pv_origin requires omega'=0 to be a grid node")

    shape = (nw,) + (1,) * (q.ndim - 1)
    wv = w.reshape(shape)
    q0 = q[i0]
    with np.errstate(divide="ignore", invalid="ignore"):
        reg = (q - q0) / wv
    # regular value at the pole node = q'(0) (central / one-sided difference)
    if 0 < i0 < nw - 1:
        reg[i0] = (q[i0 + 1] - q[i0 - 1]) / (2 * dw)
    elif i0 == 0:
        reg[i0] = (q[1] - q[0]) / dw
    else:
        reg[i0] = (q[-1] - q[-2]) / dw
    regular_integral = np.trapezoid(reg, w, axis=0)

    a, b = float(w[0]), float(w[-1])
    if a < 0.0 < b:
        pv_scalar = np.log(b / (-a))
    else:  # pole at an endpoint -- discrete fallback (cf. retarded.py)
        mask = np.arange(nw) != i0
        pv_scalar = float(np.sum(1.0 / w[mask]) * dw)
    return regular_integral + q0 * pv_scalar


def bose_weighted_pv(spectral, omega, temperature):
    """:math:`\\int n(\\omega')\\,A(\\omega')\\,d\\omega'` via exact Bose-pole subtraction.

    Equal to ``trapz(n_tilde * A) + c * pv_origin(A, omega)``: the bounded
    remainder is integrated by the ordinary rule and the ``c/omega'`` pole by
    :func:`pv_origin`. ``spectral`` (``A``) may be ``(nw,)`` or ``(nw, ...)``.
    """
    A = np.asarray(spectral)
    n_tilde, c = bose_split(omega, temperature)
    shape = (A.shape[0],) + (1,) * (A.ndim - 1)
    regular = np.trapezoid(n_tilde.reshape(shape) * A, np.asarray(omega, float), axis=0)
    return regular + c * pv_origin(A, omega)
