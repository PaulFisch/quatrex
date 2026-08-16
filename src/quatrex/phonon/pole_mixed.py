# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Mixed pole-background convolution: three routes, so they can be measured.

The mixed sectors :math:`\Sigma_{SR} + \Sigma_{RS}` need

.. math::
    C(\omega) = \int\!\frac{d\omega'}{2\pi}\,\frac{c}{\omega'-p}\,R(\omega-\omega'),

one narrow pole against a smooth background known on the grid. Substituting
:math:`u = \omega - \omega'` turns it into a resolvent of the background,

.. math::
    C(\omega) = c \int\!\frac{du}{2\pi}\,\frac{R(u)}{z-u},\qquad z = \omega - p,

with :math:`\operatorname{Im} z = \gamma > 0`, i.e. off the real axis.

Unlike the pole-pole channel there is no free exact answer here, because ``R``
is only known on the grid. Three routes, with different costs and failure modes:

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
Production uses ``"cells"``.
"""
from __future__ import annotations

import numpy as np

from qttools import NDArray, xp

__all__ = ["mixed_convolution", "cell_resolvent_weights",
           "mixed_convolution_batched", "bosonic_extend", "METHODS"]

METHODS = ("grid", "cells", "moments")


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
    method : {"grid", "cells", "moments"}
    order, window : int
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
        else:
            out[i] = c * _moment_resolvent(z, r, w, order, window)
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
    r_same: NDArray,
    r_other: NDArray,
    freqs: NDArray,
    transpose_index: NDArray | None = None,
    transverse_shape: tuple | None = None,
) -> tuple[NDArray, NDArray]:
    r"""Extend a Keldysh component onto the full frequency axis.

    The mixed convolution
    :math:`\int d\omega' F(\omega') R(\omega-\omega')` runs over the WHOLE
    axis, but the solver only holds :math:`R` for :math:`\omega \ge 0`. The
    negative half is fixed by the bosonic steady-state relation

    .. math:: G^<_{ij}(\mathbf q, -\omega) = G^>_{ji}(-\mathbf q, \omega)

    and its partner with :math:`<` and :math:`>` exchanged. So the negative
    axis of a LESSER component is built from the GREATER one, transposed and
    with :math:`\mathbf q \to -\mathbf q` -- **not** by conjugating the same
    component.

    An earlier version used ``R(-w) = R(w)^*``. That is the correct relation
    for :math:`\Delta = \Sigma^> - \Sigma^<` (which is why
    :func:`~quatrex.phonon.pole_kernel.bosonic_partner` and the production
    Hilbert transform use it) but it is **wrong for a lesser or greater
    component**. Measured on an equilibrium bed
    (:math:`G^< = i n_B A`, :math:`G^> = i(n_B{+}1)A`, :math:`A` odd):

    ========================================  ==========
    identity                                  residual
    ========================================  ==========
    ``G^<(-w) == G^>(w)``                     8.9e-16
    ``G^<(-w) == conj(G^<(w))``               7.65 (244 % relative)
    ``D(-w) == conj(D(w))``, ``D = S^>-S^<``  5.6e-17
    ========================================  ==========

    Omitting the extension entirely is a 28 % error; using the conjugate
    relation is worse than that, because it is wrong in a way that looks
    plausible at equilibrium and at :math:`\Gamma`.

    Parameters
    ----------
    r_same : NDArray
        ``(n_freq, nnz)`` the component being extended, on ``w >= 0``.
    r_other : NDArray
        Its Keldysh partner on the same grid: pass ``G^>`` when extending
        ``G^<``, and ``G^<`` when extending ``G^>``.
    freqs : NDArray
        ``(n_freq,)`` non-negative, uniform, zero-anchored.
    transpose_index : NDArray, optional
        Permutation taking pattern entry ``(i, j)`` to ``(j, i)``, from
        :func:`~quatrex.phonon.pole_audit.transpose_index`. Required whenever
        the pattern is not symmetric under the index swap; omitting it assumes
        the transpose is the identity, which holds only for a diagonal bed.
    transverse_shape : tuple, optional
        Transverse momentum grid; the mirror carries ``q -> -q``.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(r_full, freqs_full)`` on ``[-w_max, w_max]``, ascending, with the
        zero bin present exactly once.

    """
    w = xp.asarray(freqs, dtype=float)
    same = xp.asarray(r_same)
    other = xp.asarray(r_other)
    if other.shape != same.shape:
        raise ValueError(
            f"the Keldysh partner must share the grid and pattern: "
            f"{other.shape} vs {same.shape}."
        )
    if not bool(abs(float(_host(w)[0])) < 1e-12):
        raise NotImplementedError(
            "bosonic_extend needs a zero-anchored grid: the mirror is defined "
            f"about omega = 0 and this grid starts at {float(_host(w)[0]):g}."
        )

    mirror = other[:0:-1]                       # G^>(+w) reversed in w
    if transpose_index is not None:             # ... and index-transposed
        mirror = mirror[:, xp.asarray(transpose_index)]
    if transverse_shape:                        # ... and q -> -q
        for ax, k in enumerate(transverse_shape, start=2):
            mirror = xp.take(mirror, (-xp.arange(int(k))) % int(k), axis=ax)
    return (xp.concatenate([mirror, same], axis=0),
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
