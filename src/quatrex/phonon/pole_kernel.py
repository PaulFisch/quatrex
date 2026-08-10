# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Analytic continuation of the phonon scattering self-energy to complex frequency.

The production retarded self-energy is the real-axis restriction of an explicit
analytic function of a complex frequency :math:`z`. With
:math:`\Delta = \Sigma^> - \Sigma^<` (the RAW, textbook-signed bubble output, which
equals ``sigma_lesser.data - sigma_greater.data`` in the solver's stored,
occupation-positive convention) the solver builds

.. math::
    \Sigma^R_s(\omega) = \tfrac12 \Delta(\omega)
                       + \tfrac{i}{2}\,\mathcal{H}[\Delta](\omega),

where :math:`\mathcal{H}` is :func:`quatrex.core.fft_utils.hilbert_transform`, a
principal value discretised with *exact cell-integrated* weights. Modelling
:math:`\Delta` as cell-wise constant -- exactly the model that kernel already
uses -- and writing the same integral with the **complex** logarithm gives

.. math::
    F(z) = \frac{i}{2\pi}\sum_k \Delta_k
             \left[\log(z-\omega_k+h/2) - \log(z-\omega_k-h/2)\right]
         + \frac{i}{2\pi}\sum_k \bar\Delta_k
             \left[\log(z+\omega_k+h/2) - \log(z+\omega_k-h/2)\right],

with the bosonic mirror amplitude :math:`\bar\Delta_k = \Delta_k(-q)^*` and the
:math:`\omega=0` cell counted once. Two facts make this the whole foundation of
the pole sector:

1. ``F(omega + i0) == 0.5*Delta + 0.5j*hilbert_transform(Delta)`` to machine
   precision. The cell containing :math:`\omega` contributes
   :math:`\log(i0+h/2)-\log(i0-h/2) = -i\pi`, i.e. exactly the
   :math:`\tfrac12\Delta` term -- it is not a separate addition. So evaluating
   :math:`M(z)` off the real axis introduces **no approximation beyond the one
   production already makes**.
2. The jump across the cut is exactly :math:`\Delta`, so the retarded branch
   continued downward onto the resonance (second) sheet is

   .. math::
       \Sigma^{R,\mathrm{II}}(z) = F(z) + \Delta_{\rm an}(z),

   with :math:`\Delta_{\rm an}` a local polynomial continuation of
   :math:`\Delta` (:func:`delta_local_fit`). Poles worth promoting to the pole
   sector have :math:`\gamma \ll h`, which is exactly the regime where that
   local continuation is most accurate; poles with :math:`\gamma \gtrsim h` are
   resolved by the grid and are not promoted.

Everything here is pure array work -- no MPI, no ``DSDBSparse`` -- so it is
unit-testable against the production Hilbert kernel directly. The contraction is
linear in :math:`\Delta`, which is what lets the caller evaluate it in the
``"nnz"`` distribution state with zero communication.

See ``phonon/docs/pole_subtracted_modal_scba.md`` (in particular Secs. 2, 3, 9
and 36) for the formulation.
"""
from __future__ import annotations

import numpy as np

from qttools import NDArray, xp

__all__ = [
    "cell_width",
    "continuation_weights",
    "bosonic_partner",
    "contract_delta",
    "sigma_retarded_at_z",
    "delta_local_fit",
    "lorentz_retarded",
    "lorentz_pair_retarded",
]

# A mirror cell coincides with the omega=0 cell when the grid is anchored at
# zero; the same guard the production kernel uses (fft_utils.py:108).
_DC_ANCHOR = 0.25


def cell_width(energies: NDArray) -> float:
    """Uniform cell width of a frequency grid.

    Parameters
    ----------
    energies : NDArray
        Ascending, uniform frequency grid (THz).

    Returns
    -------
    float
        The spacing ``h``.

    Raises
    ------
    ValueError
        If the grid is not uniform. The cell-wise-constant model of ``Delta``
        (and the production Hilbert kernel it must reproduce) is only defined on
        a uniform grid; with the auxiliary bubble grid enabled the relevant grid
        is the uniform convolution grid, not the primary one.

    """
    e = np.asarray(_host(energies), dtype=float)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("energies must be a 1-D grid with at least two points.")
    h = float(e[1] - e[0])
    if not np.all(np.abs(np.diff(e) - h) <= 1e-9 * abs(h)):
        raise ValueError(
            "pole_kernel requires a uniform frequency grid (the cell-integrated "
            "model of Delta is only defined there). With a non-uniform primary "
            "grid, pass the auxiliary convolution grid instead."
        )
    return h


def _host(a):
    """NumPy view of an array that may live on the device."""
    return a.get() if hasattr(a, "get") else a


def continuation_weights(
    z: NDArray,
    energies: NDArray,
    *,
    order: int = 0,
) -> tuple[NDArray, NDArray]:
    r"""Cell-integrated weights of the analytic continuation, or its derivatives.

    Returns ``(w_pos, w_mir)`` such that, for the ``order``-th derivative,

    .. math::
        \partial_z^{(n)} F(z_p)
          = \sum_k w^{\rm pos}_{pk}\,\Delta_k
          + \sum_k w^{\rm mir}_{pk}\,\bar\Delta_k .

    The derivatives are needed by the bordered-Newton corrector, the residue
    normalisation and the pole conditioning, and are cheapest to ship as extra
    probe slots in the same contraction.

    Parameters
    ----------
    z : NDArray
        Complex frequencies (THz), shape ``(P,)``. Must lie strictly off the
        real axis: on it, the branch of the cell containing ``Re z`` is decided
        by the sign of a floating-point zero. For the retarded real-axis limit
        pass ``omega + 1j*tiny`` with ``tiny > 0``.
    energies : NDArray
        Ascending, uniform, non-negative frequency grid (THz), shape ``(K,)``.
    order : int, optional
        Derivative order in ``z``: 0, 1 or 2. Default 0.

    Returns
    -------
    w_pos : NDArray
        ``(P, K)`` weights against ``Delta``.
    w_mir : NDArray
        ``(P, K)`` weights against the bosonic partner ``Delta(-q)^*``. The
        ``omega = 0`` column is zeroed when the grid is anchored at zero, so
        that cell is counted exactly once.

    """
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1 or 2 (got {order}).")

    h = cell_width(energies)
    half = 0.5 * h

    zz = xp.asarray(z, dtype=xp.complex128).reshape(-1, 1)
    if bool(xp.any(xp.imag(zz) == 0.0)):
        raise ValueError(
            "continuation_weights requires Im z != 0; on the real axis the "
            "branch of the pole cell is set by the sign of a signed zero. Pass "
            "omega + 1j*tiny for the retarded limit."
        )
    ww = xp.asarray(energies, dtype=xp.float64).reshape(1, -1)

    pref = 1j / (2.0 * np.pi)

    def _kernel(u: NDArray) -> NDArray:
        hi, lo = u + half, u - half
        if order == 0:
            return xp.log(hi) - xp.log(lo)
        if order == 1:
            return 1.0 / hi - 1.0 / lo
        return -1.0 / hi**2 + 1.0 / lo**2

    w_pos = pref * _kernel(zz - ww)
    w_mir = pref * _kernel(zz + ww)

    if abs(float(_host(energies)[0])) < _DC_ANCHOR * h:
        # The omega = 0 sample sits on both the positive grid and its mirror.
        w_mir[:, 0] = 0.0

    return w_pos, w_mir


def bosonic_partner(a: NDArray, transverse_shape: tuple = ()) -> NDArray:
    r"""Mirror amplitude :math:`\bar\Delta_k = \Delta_k(-q)^*` of the continuation.

    The exact bosonic continuation of ``a = Sigma^> - Sigma^<`` is
    ``a(q, -omega) = a*(-q, omega)``; on a Gamma-centred mesh (closed under
    ``q -> -q``) that is a conjugation plus a negation of the transverse axes.
    This is the same construction as ``fft_utils.hilbert_transform`` performs at
    its lines 103-107, without the frequency reversal -- :func:`continuation_weights`
    indexes the mirror cells by ``k`` directly rather than through a convolution.

    Parameters
    ----------
    a : NDArray
        ``Delta`` on the positive-frequency grid; leading axis is frequency.
    transverse_shape : tuple, optional
        Sizes of the transverse-momentum axes following the frequency axis.

    Returns
    -------
    NDArray
        The mirror amplitude, same shape as ``a``.

    """
    out = a
    for ax, k in enumerate(transverse_shape, start=1):
        neg = (-xp.arange(int(k))) % int(k)
        out = xp.take(out, neg, axis=ax)
    return xp.conj(out)


def contract_delta(
    a: NDArray, w_pos: NDArray, w_mir: NDArray, *, transverse_shape: tuple = ()
) -> NDArray:
    """Apply continuation weights to ``Delta`` along the leading frequency axis.

    Linear in ``a``, hence evaluable as a single GEMM. The caller is free to
    contract a rank-local slice and reduce afterwards.

    Parameters
    ----------
    a : NDArray
        ``Delta``, shape ``(K, ...)``; leading axis is frequency.
    w_pos, w_mir : NDArray
        Weights from :func:`continuation_weights`, shape ``(P, K)``.
    transverse_shape : tuple, optional
        Transverse-momentum axis sizes, passed to :func:`bosonic_partner`.

    Returns
    -------
    NDArray
        Shape ``(P,) + a.shape[1:]``.

    """
    n_freq = a.shape[0]
    if w_pos.shape[-1] != n_freq:
        raise ValueError(
            f"weights are built for {w_pos.shape[-1]} frequencies but Delta has "
            f"{n_freq}."
        )
    tail = a.shape[1:]
    flat = a.reshape(n_freq, -1)
    mirror = bosonic_partner(a, transverse_shape).reshape(n_freq, -1)
    out = w_pos @ flat + w_mir @ mirror
    return out.reshape((w_pos.shape[0],) + tail)


def local_fit_weights(
    energies: NDArray,
    z: NDArray,
    *,
    order: int = 2,
    window: int = 4,
    deriv: int = 0,
) -> tuple[NDArray, NDArray]:
    r"""``(P, K)`` weights of :math:`\Delta_{\rm an}(z)`, as a linear map on Delta.

    Same object as :func:`delta_local_fit`, expressed the way
    :func:`continuation_weights` is: the least-squares fit is linear in the
    sampled values, and its coefficients depend only on the GRID, so
    ``coeff = pinv(vander) @ src`` and the whole continuation collapses to a
    weight matrix.

    That matters for more than tidiness. It is what makes the pole sector
    work on a distributed frequency axis: every rank can build the weights for
    the global grid, contract only the columns it owns, and one
    ``all_reduce(sum)`` completes the continuation. Neither branch reindexes
    the frequency axis -- the mirror is elementwise (a conjugation plus a
    transverse ``q -> -q``), and the negative-frequency reflection lives in the
    ``z + omega_k`` argument -- so a rank never needs a frequency another rank
    owns.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(w_pos, w_mir)``, to be used exactly like
        :func:`continuation_weights`' output in :func:`contract_delta`.

    """
    h = cell_width(energies)
    e = xp.asarray(energies, dtype=xp.float64)
    n_freq = int(e.shape[0])
    if 2 * window < order + 1:
        raise ValueError(
            f"window={window} gives {2 * window} samples, too few for a "
            f"degree-{order} fit."
        )
    zz = xp.asarray(z, dtype=xp.complex128).reshape(-1)
    n_probe = int(zz.shape[0])
    w_pos = xp.zeros((n_probe, n_freq), dtype=xp.complex128)
    w_mir = xp.zeros((n_probe, n_freq), dtype=xp.complex128)

    for p in range(n_probe):
        zp = complex(zz[p])
        positive = zp.real >= 0.0
        w_c = zp.real if positive else -zp.real
        centre = int(np.clip(round(w_c / h), 0, n_freq - 1))
        lo = int(np.clip(centre - window, 0, max(0, n_freq - 2 * window)))
        hi = min(n_freq, lo + 2 * window)
        lo = max(0, hi - 2 * window)
        idx = xp.arange(lo, hi)
        t = (e[idx] - w_c) / h
        vander = xp.stack([t**m for m in range(order + 1)], axis=1)
        pinv = xp.linalg.pinv(vander.astype(xp.complex128))     # (order+1, 2*window)

        if positive:
            s, ds_dz = (zp.real - w_c + 1j * zp.imag) / h, 1.0 / h
        else:
            s, ds_dz = (-zp.real - w_c - 1j * zp.imag) / h, -1.0 / h
        pw = []
        for m in range(order + 1):
            if m < deriv:
                pw.append(0.0 * s)
                continue
            fall = 1.0
            for j in range(deriv):
                fall *= m - j
            pw.append(fall * s ** (m - deriv) * ds_dz**deriv)
        powers = xp.asarray(pw, dtype=xp.complex128)            # (order+1,)
        row = powers @ pinv                                     # (2*window,)
        target = w_pos if positive else w_mir
        target[p, lo:hi] = row
    return w_pos, w_mir


def delta_local_fit(
    a: NDArray,
    energies: NDArray,
    z: NDArray,
    *,
    order: int = 2,
    window: int = 4,
    deriv: int = 0,
    transverse_shape: tuple = (),
) -> NDArray:
    r"""Local polynomial continuation :math:`\Delta_{\rm an}(z)` of ``Delta``.

    The second-sheet term of :math:`\Sigma^{R,\rm II}(z) = F(z) + \Delta_{\rm an}(z)`.
    ``Delta`` is known only on the grid, so it is continued off the axis by a
    degree-``order`` least-squares polynomial in the scaled variable
    ``(omega - Re z)/h`` over the ``window`` nearest cells on each side, then
    evaluated at ``z``. The error is ``O((|Im z|/h)^(order+1))``, which is
    controlled precisely because the poles this is used for satisfy
    ``gamma << h``.

    ``Delta`` is continued from its bosonically completed axis, so a ``z`` with
    negative real part is served by the mirror branch.

    Parameters
    ----------
    a : NDArray
        ``Delta``, shape ``(K, ...)``.
    energies : NDArray
        Uniform ascending grid (THz), shape ``(K,)``.
    z : NDArray
        Complex frequencies, shape ``(P,)``.
    order : int, optional
        Polynomial degree. Default 2.
    window : int, optional
        Number of cells on each side of ``Re z``. Must give at least
        ``order + 1`` samples. Default 4.
    deriv : int, optional
        Order of the ``z``-derivative of the model to return. The chain rule
        carries a sign on the mirror branch, where the model is a function of
        ``-z``. Default 0.
    transverse_shape : tuple, optional
        Transverse-momentum axis sizes.

    Returns
    -------
    NDArray
        Shape ``(P,) + a.shape[1:]``.

    """
    h = cell_width(energies)
    e = xp.asarray(energies, dtype=xp.float64)
    n_freq = int(e.shape[0])
    if 2 * window < order + 1:
        raise ValueError(
            f"window={window} gives {2 * window} samples, too few for a "
            f"degree-{order} fit."
        )

    zz = xp.asarray(z, dtype=xp.complex128).reshape(-1)
    tail = a.shape[1:]
    flat = a.reshape(n_freq, -1)
    mirror = bosonic_partner(a, transverse_shape).reshape(n_freq, -1)

    out = xp.empty((zz.shape[0], flat.shape[1]), dtype=xp.complex128)
    for p in range(int(zz.shape[0])):
        zp = complex(zz[p])
        # Negative-frequency z is served by the mirrored branch: Delta(-w) = Dbar(w).
        src, w_c = (flat, zp.real) if zp.real >= 0.0 else (mirror, -zp.real)
        centre = int(np.clip(round(w_c / h), 0, n_freq - 1))
        lo = int(np.clip(centre - window, 0, max(0, n_freq - 2 * window)))
        hi = min(n_freq, lo + 2 * window)
        lo = max(0, hi - 2 * window)
        idx = xp.arange(lo, hi)
        t = (e[idx] - w_c) / h
        vander = xp.stack([t**m for m in range(order + 1)], axis=1)
        coeff, *_ = xp.linalg.lstsq(
            vander.astype(xp.complex128), src[lo:hi], rcond=None
        )
        # Fit variable and its derivative w.r.t. z. On the mirror branch the
        # model is a function of -z, so the chain rule flips the sign.
        if zp.real >= 0.0:
            s, ds_dz = (zp.real - w_c + 1j * zp.imag) / h, 1.0 / h
        else:
            s, ds_dz = (-zp.real - w_c - 1j * zp.imag) / h, -1.0 / h
        pw = []
        for m in range(order + 1):
            if m < deriv:
                pw.append(0.0 * s)
                continue
            fall = 1.0
            for j in range(deriv):
                fall *= m - j
            pw.append(fall * s ** (m - deriv) * ds_dz**deriv)
        powers = xp.asarray(pw, dtype=xp.complex128)
        out[p] = powers @ coeff
    return out.reshape((zz.shape[0],) + tail)


def sigma_retarded_at_z(
    a: NDArray,
    energies: NDArray,
    z: NDArray,
    *,
    sheet: str = "II",
    order: int = 0,
    transverse_shape: tuple = (),
    delta_order: int = 2,
    delta_window: int = 4,
) -> NDArray:
    r"""Scattering :math:`\Sigma^R_s(z)` at complex frequency, or its derivative.

    Convenience wrapper: ``F(z)`` from :func:`continuation_weights` plus, on the
    resonance sheet, the local continuation of ``Delta``.

    Parameters
    ----------
    a : NDArray
        ``Delta = Sigma^> - Sigma^<`` (raw sign), shape ``(K, ...)``.
    energies : NDArray
        Uniform ascending grid (THz).
    z : NDArray
        Complex frequencies, shape ``(P,)``, strictly off the real axis.
    sheet : {"I", "II"}, optional
        ``"I"`` is the plain Cauchy branch (correct where the device has no cut,
        i.e. modes in a lead gap); ``"II"`` adds ``Delta_an(z)`` and is the
        continuation of the *retarded* branch through the cut, which is where
        an in-band resonance lives. Default ``"II"``.
    order : int, optional
        Derivative order in ``z``. On sheet II the local model of ``Delta`` is
        differentiated analytically, so ``M'(z)`` is available in closed form
        rather than by finite differences.
    transverse_shape : tuple, optional
        Transverse-momentum axis sizes.
    delta_order, delta_window : int, optional
        Passed to :func:`delta_local_fit`.

    Returns
    -------
    NDArray
        Shape ``(P,) + a.shape[1:]``.

    """
    if sheet not in ("I", "II"):
        raise ValueError(f"sheet must be 'I' or 'II' (got {sheet!r}).")
    w_pos, w_mir = continuation_weights(z, energies, order=order)
    out = contract_delta(a, w_pos, w_mir, transverse_shape=transverse_shape)
    if sheet == "I":
        return out
    return out + delta_local_fit(
        a, energies, z, order=delta_order, window=delta_window, deriv=order,
        transverse_shape=transverse_shape,
    )


def lorentz_retarded(omega: NDArray, centre: complex) -> NDArray:
    r"""Causal retarded function of a single Lorentzian spectral term.

    For :math:`\Delta(\omega) = L_{\Omega,\gamma}(\omega) = 2\gamma/((\omega-\Omega)^2+\gamma^2)`
    the exact Hilbert partner is :math:`\mathcal{H}[L] = 2(\omega-\Omega)/((\omega-\Omega)^2+\gamma^2)`,
    so

    .. math::
        \tfrac12 L + \tfrac{i}{2}\mathcal{H}[L]
          = \frac{i}{\omega - \Omega + i\gamma},

    a single causal pole. This is why an analytic pole-sector contribution to
    ``Delta`` must **not** be passed through the numerical Hilbert transform:
    its Kramers-Kronig partner is known in closed form.

    Parameters
    ----------
    omega : NDArray
        Real frequencies (THz).
    centre : complex
        The pole ``Omega - 1j*gamma`` with ``gamma > 0``.

    Returns
    -------
    NDArray
        ``i / (omega - Omega + i*gamma)``.

    """
    c = complex(centre)
    if c.imag >= 0.0:
        raise ValueError(
            f"a retarded pole must sit in the lower half plane (got {c})."
        )
    # Pole at z = centre = Omega - i*gamma, i.e. i/(omega - Omega + i*gamma).
    return 1j / (xp.asarray(omega, dtype=xp.complex128) - c)


def lorentz_pair_retarded(omega: NDArray, centre: complex) -> NDArray:
    r"""Retarded partner of a Lorentzian **and** its bosonic mirror.

    The production Hilbert transform completes ``Delta`` to negative frequencies
    with ``a(-omega) = a*(omega)`` before transforming. An analytic contribution
    that bypasses that transform must add its own mirror partner at
    :math:`-\Omega`, or the analytic and numerical halves of :math:`\Sigma^R` are
    built from different integrands.

    Parameters
    ----------
    omega : NDArray
        Real frequencies (THz).
    centre : complex
        The pole ``Omega - 1j*gamma``.

    Returns
    -------
    NDArray
        ``i/(omega - Omega + i*gamma) + i/(omega + Omega + i*gamma)``.

    """
    c = complex(centre)
    return lorentz_retarded(omega, c) + lorentz_retarded(omega, complex(-c.real, c.imag))
