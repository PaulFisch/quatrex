# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Mixed pole-background convolution: four routes, so they can be measured.

The mixed sectors :math:`\Sigma_{SR} + \Sigma_{RS}` need

.. math::
    C(\omega) = \int\!\frac{d\omega'}{2\pi}\,\frac{c}{\omega'-p}\,R(\omega-\omega'),

one narrow pole against a smooth background known on the grid. Substituting
:math:`u = \omega - \omega'` turns it into a resolvent of the background,

.. math::
    C(\omega) = c \int\!\frac{du}{2\pi}\,\frac{R(u)}{z-u},\qquad z = \omega - p,

with :math:`\operatorname{Im} z = \gamma > 0`, i.e. off the real axis.

Unlike the pole-pole channel there is no free exact answer here, because ``R``
is only known on the grid. Four routes, with different costs and failure modes:

``"grid"``
    The discrete convolution, i.e. what the existing linearised bubble does.
    Cheapest and ``N_p``-independent, but it samples the narrow pole on the
    coarse grid, so it inherits the registration lottery it was meant to avoid.
``"cells"``
    Exact for a cell-wise-CONSTANT model of ``R`` -- the same cell-integrated
    log kernel the production Kramers-Kronig transform uses. No narrow
    denominator is ever sampled. Exact in the pole, first order in ``R``.
``"moments"``
    Local polynomial model of ``R`` over a window around the pole, integrated
    against the pole analytically; the rest of the axis by the cell kernel.
    Exact in the pole, order ``p`` in ``R``.
``"rational"``
    Local rational model of ``R``, turning the whole integral into a residue
    sum. Cheapest asymptotically. **Not implemented** -- see
    :func:`_rational_resolvent` for what is missing and why the measured
    comparison does not motivate finishing it.

The comparison driver is ``phonon/studies/_pole_mixed_compare.py``.
"""
from __future__ import annotations

import numpy as np

from qttools import NDArray, xp

__all__ = ["mixed_convolution", "cell_resolvent_weights",
           "mixed_convolution_batched", "bosonic_extend", "METHODS"]

METHODS = ("grid", "cells", "moments", "rational")


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _cell_resolvent(z: complex, r_vals: np.ndarray, w: np.ndarray) -> complex:
    r"""``(1/2pi) * int R(u)/(z-u) du`` for a cell-wise-constant ``R``.

    Each cell contributes ``R_k * [Log(z - w_k + h/2) - Log(z - w_k - h/2)]``,
    the same exact cell integral the production Hilbert kernel uses -- so the
    narrow denominator is integrated, never sampled.
    """
    h = float(w[1] - w[0])
    d = z - w
    return float(1.0) / (2.0 * np.pi) * np.sum(
        r_vals * (np.log(d + 0.5 * h) - np.log(d - 0.5 * h))
    )


def _moment_resolvent(
    z: complex, r_vals: np.ndarray, w: np.ndarray, order: int, window: int
) -> complex:
    r"""Local polynomial model of ``R`` near the pole, integrated analytically.

    Splits the axis at ``|u - Re z| < window * h``. Inside, ``R`` is fitted by a
    degree-``order`` polynomial in ``t = u - Re z`` and integrated against
    ``1/(i gamma - t)`` in closed form; outside, the cell kernel is used, where
    the pole is far and a cell-wise-constant ``R`` is already accurate.
    """
    h = float(w[1] - w[0])
    u0, gam = z.real, z.imag
    near = np.abs(w - u0) <= window * h
    if near.sum() < order + 1:
        return _cell_resolvent(z, r_vals, w)

    far = _cell_resolvent(z, np.where(near, 0.0, r_vals), w)

    t = w[near] - u0
    coeff = np.polyfit(t, r_vals[near], order)[::-1]      # ascending powers
    a = 1j * gam
    edge = float(np.max(np.abs(t))) + 0.5 * h
    acc = 0.0 + 0.0j
    for m, cm in enumerate(coeff):
        # t^m/(a - t) = -sum_{r<m} a^r t^{m-1-r}  -  a^m/(t - a)
        poly = 0.0 + 0.0j
        for r in range(m):
            k = m - 1 - r
            poly -= a**r * (edge ** (k + 1) - (-edge) ** (k + 1)) / (k + 1)
        poly -= a**m * (np.log(edge - a) - np.log(-edge - a))
        acc += cm * poly
    return far + acc / (2.0 * np.pi)


def _rational_resolvent(
    z: complex, r_vals: np.ndarray, w: np.ndarray, n_fit: int, window: int
) -> complex:
    """Local rational model of ``R``, as a residue sum. NOT IMPLEMENTED.

    Raises rather than falling back, because a silent fallback would report the
    cell-kernel result under the rational label -- which is exactly what an
    earlier version of this function did, and it looked like a validated
    method in the comparison table.

    Two things are missing, and the first is a correctness lesson worth keeping:

    * The causality guard was written as "reject a fitted pole in the same half
      plane as ``z``". That rejects EVERY fit: ``R`` is real, so its fitted
      poles come in conjugate pairs and one is always in the upper half plane.
      A upper-half-plane pole is in fact harmless -- closing the contour
      downward it simply does not contribute -- so the guard should test the
      fit's spuriousness (residue size, distance from the sampled window), not
      its half plane.
    * The residue sum over the fitted denominator poles was never written.

    It is not obviously worth completing: the measured comparison
    (``phonon/studies``) shows the moment route is already flat in ``gamma`` at
    ~1e-03 and exact in the pole, so the only thing a rational model could add
    is a lower asymptotic cost per pole -- against a real risk of a spurious
    pole corrupting the causal structure.
    """
    raise NotImplementedError(
        "the rational mixed-convolution route is not implemented: its causality "
        "guard rejects every fit (a real background has conjugate poles) and "
        "the denominator residue sum is missing. Use method='moments', which "
        "is measured flat in gamma and exact in the pole."
    )


def mixed_convolution(
    omega: NDArray,
    pole: complex,
    coeff: complex,
    r_vals: NDArray,
    freqs: NDArray,
    *,
    method: str = "cells",
    order: int = 3,
    window: int = 6,
    n_fit: int = 2,
) -> NDArray:
    r"""``C(omega) = int dw'/2pi [coeff/(w'-pole)] R(omega-w')``.

    Parameters
    ----------
    omega : NDArray
        Output frequencies.
    pole : complex
        The narrow pole, in the lower half plane.
    coeff : complex
        Its coefficient.
    r_vals : NDArray
        ``R`` sampled on ``freqs``.
    freqs : NDArray
        Uniform grid on which ``R`` is known.
    method : {"grid", "cells", "moments", "rational"}
    order, window, n_fit : int
        Model parameters for the local routes.

    Returns
    -------
    NDArray
        ``C`` at each ``omega``.

    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS} (got {method!r})")
    w = np.asarray(_host(freqs), dtype=float)
    r = np.asarray(_host(r_vals))
    om = np.atleast_1d(np.asarray(_host(omega)))
    p, c = complex(pole), complex(coeff)
    h = float(w[1] - w[0])

    if method == "grid":
        # The discrete convolution: the pole is SAMPLED on the grid.
        out = np.empty(om.size, dtype=complex)
        for i, o in enumerate(om):
            f = c / (w - p)
            # Index of the sample at (o - w[k]) in the grid; the offset w[0]
            # matters whenever the grid is not anchored at zero.
            idx = np.rint((o - w - w[0]) / h).astype(int)
            ok = (idx >= 0) & (idx < w.size)
            out[i] = np.sum(f[ok] * r[idx[ok]]) * h / (2.0 * np.pi)
        return out

    out = np.empty(om.size, dtype=complex)
    for i, o in enumerate(om):
        z = o - p
        if method == "cells":
            out[i] = c * _cell_resolvent(z, r, w)
        elif method == "moments":
            out[i] = c * _moment_resolvent(z, r, w, order, window)
        else:
            out[i] = c * _rational_resolvent(z, r, w, n_fit, window)
    return out


def cell_resolvent_weights(z: NDArray, freqs: NDArray) -> NDArray:
    r"""Weights of ``(1/2pi) int R(u)/(z-u) du`` for a cell-wise-constant ``R``.

    Returns ``W`` with ``W[i, k]`` the weight of cell ``k`` at probe ``z[i]``, so
    the resolvent of a whole array of backgrounds is a single matmul
    ``W @ R`` -- the same shape of contraction as the retarded continuation, and
    the reason the mixed sector does not need a Python loop over frequencies.

    No bosonic mirror is applied: ``R`` here is a background sampled on whatever
    grid the caller has, not a one-sided spectral function.
    """
    w = xp.asarray(freqs, dtype=float).reshape(1, -1)
    zz = xp.asarray(z, dtype=xp.complex128).reshape(-1, 1)
    h = float(_host(freqs)[1] - _host(freqs)[0])
    d = zz - w
    return (xp.log(d + 0.5 * h) - xp.log(d - 0.5 * h)) / (2.0 * np.pi)


def bosonic_extend(
    r_vals: NDArray, freqs: NDArray, transverse_shape: tuple | None = None
) -> tuple[NDArray, NDArray]:
    r"""Mirror a one-sided background onto the full frequency axis.

    The mixed convolution :math:`\int d\omega'\, F(\omega') R(\omega-\omega')`
    runs over the WHOLE axis, but the solver only ever holds :math:`R` for
    :math:`\omega \ge 0`. The negative half is not missing information -- it is
    fixed by the bosonic relation :math:`R(-\omega) = R(\omega)^*` (with
    :math:`q \to -q` on the transverse axes) -- but it does have to be supplied,
    because :func:`cell_resolvent_weights` integrates exactly the cells it is
    given and nothing else.

    Omitting it is not a small error. Measured against a wide-axis reference on
    a background built to satisfy the relation exactly, the one-sided integral
    is wrong by **28%**, and mirroring brings it to 2.5e-4. That is the size of
    defect that breaks the Phi-derivable energy balance rather than merely
    degrading it.

    The RR ring never had this problem: the FFT route builds the bosonic fold
    explicitly before transforming. The pole-pole channel closes its pole SET
    instead (:func:`~quatrex.phonon.pole_bubble.bosonic_closure`). This is the
    same physical statement, applied to a sampled leg.

    Parameters
    ----------
    r_vals : NDArray
        ``(n_freq, ...)`` background on a grid starting at (or near) zero.
    freqs : NDArray
        ``(n_freq,)`` non-negative, uniformly spaced.
    transverse_shape : tuple, optional
        Transverse momentum grid. The mirror carries ``q -> -q`` on these axes;
        the plain conjugate is only correct at ``nq == 1``.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(r_full, freqs_full)`` on ``[-w_max, w_max]``, ascending, with the
        zero bin present exactly once.

    """
    w = xp.asarray(freqs, dtype=float)
    r = xp.asarray(r_vals)
    zero_anchored = bool(abs(float(_host(w)[0])) < 1e-12)
    if not zero_anchored:
        raise NotImplementedError(
            "bosonic_extend needs a zero-anchored grid: the mirror is defined "
            f"about omega = 0 and this grid starts at {float(_host(w)[0]):g}."
        )
    mirror = xp.conj(r[:0:-1])
    if transverse_shape:
        # R(-q, -w) = R(q, w)^*: the conjugate alone is the nq == 1 shortcut.
        for ax, k in enumerate(transverse_shape, start=1):
            mirror = xp.take(mirror, (-xp.arange(k)) % k, axis=ax)
    return (xp.concatenate([mirror, r], axis=0),
            xp.concatenate([-w[:0:-1], w]))


def mixed_convolution_batched(
    omega: NDArray,
    pole: complex,
    coeff: complex,
    r_vals: NDArray,
    freqs: NDArray,
) -> NDArray:
    r"""Cell-route mixed convolution for a whole array of backgrounds.

    ``r_vals`` has shape ``(n_freq, ...)``; the result is ``(n_omega, ...)``.
    Used by the SR/RS sector, where ``R`` is the regular Green's function on the
    stored sparsity pattern and the trailing axis is the pattern itself.

    The narrow denominator is integrated over each cell, never sampled -- which
    is the whole point: the measured alternative of sampling it on the grid is
    wrong by a factor 36 at ``gamma/h = 0.008``.
    """
    z = xp.asarray(omega, dtype=xp.complex128) - complex(pole)
    w = cell_resolvent_weights(z, freqs)
    r = xp.asarray(r_vals)
    tail = r.shape[1:]
    return complex(coeff) * (w @ r.reshape(r.shape[0], -1)).reshape(
        (w.shape[0],) + tail
    )
