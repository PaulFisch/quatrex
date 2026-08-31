# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Analytic continuation of the phonon scattering self-energy to complex z.

The real-axis retarded self-energy is already built as a cell-integrated
Hilbert transform of :math:`\Delta = \Sigma^> - \Sigma^<`; continuing the same
cell-wise model with the complex logarithm gives a function analytic off the
real axis whose boundary value from above reproduces it to machine precision,
so evaluating :math:`M(z)` off the axis adds no approximation production does
not already make. Reaching the second sheet additionally needs
:math:`\Delta` past the cut, supplied by :func:`delta_local_fit`; that fit is
the one uncontrolled step. The report derives this under
"Where the poles are".

Sign convention: :math:`\Delta` is the raw, textbook-signed bubble output,
which is ``sigma_lesser.data - sigma_greater.data`` in the solver's stored,
occupation-positive convention.

Pure array work, no MPI. The contraction is linear in :math:`\Delta`, which is
what lets the caller evaluate it in the ``"nnz"`` state with no communication.
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
    "local_fit_weights",
    "LocalFitPlan",
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
    r"""Cell-integrated weights of the analytic continuation, or its
    derivatives.

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
    r"""Mirror amplitude :math:`\bar\Delta_k = \Delta_k(-q)^*` of the
    continuation.

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
    a: NDArray, w_pos: NDArray, w_mir: NDArray, *, transverse_shape: tuple = (),
    mirror: NDArray | None = None,
) -> NDArray:
    """Apply continuation weights to ``Delta`` along the leading frequency
    axis.

        Parameters
        ----------
        a : NDArray
            ``Delta``, shape ``(K, ...)``; leading axis is frequency.
        w_pos, w_mir : NDArray
            Weights from :func:`continuation_weights`, shape ``(P, K)``.
        transverse_shape : tuple, optional
            Transverse-momentum axis sizes, passed to :func:`bosonic_partner`.
        mirror : NDArray, optional
            A precomputed :func:`bosonic_partner` of ``a``. It does NOT depend on
            ``z``, so a caller that contracts the same ``Delta`` at many probe
            points -- the bordered Newton does it about 6 times per candidate --
            should build it once and pass it here rather than have it rebuilt on
            every call.

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
    if mirror is None:
        mirror = bosonic_partner(a, transverse_shape)
    mirror = mirror.reshape(n_freq, -1)
    out = w_pos @ flat + w_mir @ mirror
    return out.reshape((w_pos.shape[0],) + tail)


class LocalFitPlan:
    r"""The anchor-dependent half of :func:`local_fit_weights`, computed once.

        Parameters
        ----------
        energies : NDArray
            Uniform ascending grid (THz), ``(K,)``.
        anchors : NDArray
            ``(P,)`` real fit anchors, one per probe. A negative anchor selects the
            mirror branch, where the model is a function of ``-z``.
        order : int, optional
            Polynomial degree. Default 2.
        window : int, optional
            Cells on each side of the anchor. Default 4.
    """

    def __init__(self, energies: NDArray, anchors: NDArray, *,
                 order: int = 2, window: int = 4):
        if 2 * window < order + 1:
            raise ValueError(
                f"window={window} gives {2 * window} samples, too few for a "
                f"degree-{order} fit."
            )
        self.h = cell_width(energies)
        e = xp.asarray(energies, dtype=xp.float64)
        self.n_freq = n_freq = int(e.shape[0])
        self.order = int(order)

        a = np.asarray(_host(anchors), dtype=float).reshape(-1)
        self.n_probe = int(a.size)
        self.positive = a >= 0.0
        w_c = np.where(self.positive, a, -a)

        # Same stencil arithmetic as the scalar path; the width collapses to
        # the whole grid only when the grid is shorter than the window.
        wid = min(2 * window, n_freq)
        centre = np.clip(np.round(w_c / self.h), 0, n_freq - 1)
        lo = np.clip(centre - window, 0, n_freq - wid).astype(int)
        self.lo, self.width = lo, wid
        idx = lo[:, None] + np.arange(wid)[None, :]              # (P, wid)
        self.idx = xp.asarray(idx)

        t = (np.asarray(_host(e))[idx] - w_c[:, None]) / self.h  # (P, wid)
        vander = np.stack([t ** m for m in range(order + 1)], axis=2)
        self.pinv = xp.linalg.pinv(
            xp.asarray(vander, dtype=xp.complex128))             # (P, ord+1, wid)
        self.w_c = xp.asarray(w_c)
        self._sign = xp.asarray(np.where(self.positive, 1.0, -1.0))
        self._pos = xp.asarray(self.positive)

    def powers(self, z: NDArray, deriv: int = 0) -> NDArray:
        """``(P, order+1)`` monomials of the fit variable at ``z``."""
        zz = xp.asarray(z, dtype=xp.complex128).reshape(-1)
        if int(zz.shape[0]) != self.n_probe:
            raise ValueError(
                f"plan holds {self.n_probe} anchors, got {int(zz.shape[0])} "
                "probe points."
            )
        # On the mirror branch the model is a function of -z, so the chain
        # rule flips the sign of ds/dz.
        s = (self._sign * xp.real(zz) - self.w_c
             + 1j * self._sign * xp.imag(zz)) / self.h
        ds_dz = self._sign / self.h
        cols = []
        for m in range(self.order + 1):
            if m < deriv:
                cols.append(xp.zeros_like(s))
                continue
            fall = 1.0
            for j in range(deriv):
                fall *= m - j
            cols.append(fall * s ** (m - deriv) * ds_dz ** deriv)
        return xp.stack(cols, axis=1)

    def compact_weights(self, z: NDArray, *, deriv: int = 0):
        """``(row, idx, positive)`` -- the weights WITHOUT the zero padding.

        :meth:`weights` scatters each probe's ``2*window`` coefficients into a
        full ``(P, K)`` row, which is what :func:`contract_delta` wants but is
        mostly zeros: for a few hundred thousand probes -- one per pole PAIR
        per cluster per q -- that array is the whole cost. A caller that
        gathers its own samples wants the compact form.
        """
        row = (self.powers(z, deriv)[:, None, :] @ self.pinv)[:, 0, :]
        return row, self.idx, self._pos

    def weights(self, z: NDArray, *, deriv: int = 0) -> tuple[NDArray, NDArray]:
        """``(w_pos, w_mir)``, each ``(P, K)``, as :func:`contract_delta` wants."""
        row = (self.powers(z, deriv)[:, None, :] @ self.pinv)[:, 0, :]  # (P, wid)
        w_pos = xp.zeros((self.n_probe, self.n_freq), dtype=xp.complex128)
        w_mir = xp.zeros((self.n_probe, self.n_freq), dtype=xp.complex128)
        rows = xp.arange(self.n_probe)[:, None]
        zero = xp.zeros_like(row)
        w_pos[rows, self.idx] = xp.where(self._pos[:, None], row, zero)
        w_mir[rows, self.idx] = xp.where(self._pos[:, None], zero, row)
        return w_pos, w_mir


def local_fit_weights(
    energies: NDArray,
    z: NDArray,
    *,
    order: int = 2,
    window: int = 4,
    deriv: int = 0,
    anchor: float | NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    r"""``(P, K)`` weights of :math:`\Delta_{\rm an}(z)`, as a linear map on
    Delta.

        Returns
        -------
        tuple[NDArray, NDArray]
            ``(w_pos, w_mir)``, to be used exactly like
            :func:`continuation_weights`' output in :func:`contract_delta`.
    """
    zz = xp.asarray(z, dtype=xp.complex128).reshape(-1)
    if anchor is None:
        anchors = xp.real(zz)
    else:
        anchors = xp.broadcast_to(
            xp.asarray(anchor, dtype=xp.float64).reshape(-1), zz.shape)
    plan = LocalFitPlan(energies, anchors, order=order, window=window)
    return plan.weights(zz, deriv=deriv)


def delta_local_fit(
    a: NDArray,
    energies: NDArray,
    z: NDArray,
    *,
    order: int = 2,
    window: int = 4,
    deriv: int = 0,
    transverse_shape: tuple = (),
    anchor: float | None = None,
) -> NDArray:
    r"""Local polynomial continuation :math:`\Delta_{\rm an}(z)` of ``Delta``.

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
        # ``anchor`` pins the branch AND the stencil so the result is analytic
        # in z; see local_fit_weights for the measured 17 % jump without it.
        ref = zp.real if anchor is None else float(anchor)
        src, w_c = (flat, ref) if ref >= 0.0 else (mirror, -ref)
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
        if ref >= 0.0:
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
    anchor: float | None = None,
) -> NDArray:
    r"""Scattering :math:`\Sigma^R_s(z)` at complex frequency, or its
    derivative.

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
        transverse_shape=transverse_shape, anchor=anchor,
    )


def lorentz_retarded(omega: NDArray, centre: complex) -> NDArray:
    r"""Causal retarded function of a single Lorentzian spectral term.

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
