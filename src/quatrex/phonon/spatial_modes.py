# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Complex bands of the DEVICE, and the range they imply.

The frequency half of the pole method asks which lines the grid cannot carry.
This is the spatial question underneath it: how far a mode reaches before it
decays, and therefore how many blocks a self-energy has to keep.

For a nearest-layer structure the modes solve (method proposal Eq. 143)

.. math::
    \left[-H_{10}\lambda^{-1} + (z^2 I - H_{00}) - H_{01}\lambda\right] v = 0,

and under SCBA dressing (Eq. 144) the blocks pick up the scattering self-energy,
:math:`H_{00} \to H_{00} + \Sigma^R_{00}`, :math:`H_{01} \to H_{01} +
\Sigma^R_{01}`. That single substitution is the whole physical content here:

* **Undressed**, an in-band mode has :math:`|\lambda| = 1`. It never decays, so
  there is no block range at which truncating it is accurate -- what a boxcar
  of range ``b`` discards is :math:`\sum_{n>b} 1`, which diverges for every
  ``b`` (``tests/quatrex/phonon/test_spatial_modal.py``).
* **Dressed**, the reciprocal pair splits and the same mode acquires a finite
  :math:`\xi = -1/\ln|\lambda|`, in cells. That is a mean free path, and it is
  the number a spatial truncation must be compared against.

So the question "is ``sse_g_band`` long enough" has a measurable answer rather
than a convention, which matters for a live one: the thesis' band ladder
brackets the long-CNT result by a factor 2.2 between a boxcar upper bound and a
tapered lower bound (``document/src/results/64_gband.tex``), and the CNT series
stops at seven cells because of it.

Diagnostic only. Nothing here enters the Dyson equation or the bubble; it
measures the operator the solver already builds.

The same pencil is what the lead OBC solves, so this reuses
:mod:`qttools.nevp` rather than reimplementing it -- with undressed lead blocks
it must return the lead's own bands, which is the regression that pins the
convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qttools import NDArray, xp

__all__ = ["ModeSet", "bloch_modes", "decay_lengths", "band_range_cells"]


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


@dataclass(frozen=True)
class ModeSet:
    """Complex bands at one frequency.

    Attributes
    ----------
    lam : NDArray
        Bloch factors :math:`\\lambda = e^{ika}`, one per mode.
    vecs : NDArray
        Right mode vectors, ``(n_dof, n_modes)``.
    xi : NDArray
        Decay length in CELLS, :math:`-1/\\ln|\\lambda|`. Positive and finite
        for a decaying mode, ``inf`` for a propagating one, ``0`` for a mode
        that dies within one cell, and ``nan`` for a growing partner, whose
        range is a statement about the other direction.
    """

    lam: NDArray
    vecs: NDArray
    xi: NDArray

    @property
    def decaying(self) -> NDArray:
        """Mask of modes that decay in the ``+`` direction (``|lambda| < 1``)."""
        return np.abs(np.asarray(_host(self.lam))) < 1.0

    @property
    def propagating(self, tol: float = 1e-10) -> NDArray:
        """Mask of modes on the unit circle -- undamped, so unbounded range."""
        return np.abs(np.abs(np.asarray(_host(self.lam))) - 1.0) < 1e-10


def bloch_modes(a_ii, a_ij, a_ji, nevp=None) -> ModeSet:
    r"""Solve :math:`a_{ji}\lambda^{-1} + a_{ii} + a_{ij}\lambda = 0`.

    The block convention is the OBC's, so the blocks are those of the SYSTEM
    matrix -- ``a_ii = z^2 I - H_00 - Sigma^R_00``, ``a_ij = -H_01 -
    Sigma^R_01`` -- and not of the dynamical matrix. Passing dynamical-matrix
    blocks would solve a different pencil and silently return the wrong bands,
    so the caller supplies what the solver already assembled.

    Parameters
    ----------
    a_ii, a_ij, a_ji : NDArray
        One square block each, ``(b, b)``.
    nevp : NEVP, optional
        Defaults to the dense linearisation
        :class:`~qttools.nevp.full.Full`. It takes every root, which is what a
        survey wants; the Beyn contour returns only what its annulus encloses
        and would need a radius chosen in advance.
    """
    if nevp is None:
        from qttools.nevp.full import Full

        nevp = Full()

    blocks = tuple(xp.asarray(b)[xp.newaxis, :, :] for b in (a_ji, a_ii, a_ij))
    if blocks[0].shape[-1] != blocks[0].shape[-2]:
        raise ValueError(
            f"bloch_modes: blocks must be square, got {blocks[0].shape[-2:]}")

    ws, vs = nevp(blocks)
    lam = np.asarray(_host(ws)).ravel()
    vecs = np.asarray(_host(vs))
    vecs = vecs.reshape(vecs.shape[-2], -1) if vecs.ndim > 2 else vecs
    return ModeSet(lam=lam, vecs=vecs, xi=decay_lengths(lam))


def decay_lengths(lam) -> NDArray:
    r"""``xi = -1/ln|lambda|``, in cells.

    ``inf`` on the unit circle: an undamped mode has no range, which is the
    point rather than an edge case. ``nan`` for ``|lambda| > 1``, because a
    growing partner's range describes the opposite direction and reporting a
    negative length there invites it being minimised over.
    """
    mod = np.abs(np.asarray(_host(lam), dtype=complex))
    out = np.full(mod.shape, np.nan)

    grows = mod > 1.0 + 1e-12
    prop = np.abs(mod - 1.0) <= 1e-12
    dead = mod <= 0.0
    decays = ~(grows | prop | dead) & np.isfinite(mod)

    out[prop] = np.inf
    out[dead] = 0.0
    with np.errstate(divide="ignore"):
        out[decays] = -1.0 / np.log(mod[decays])
    return out


def band_range_cells(a_ii, a_ij, a_ji, nevp=None) -> float:
    r"""Block range a truncation must keep: the SLOWEST decaying mode.

    ``inf`` when any mode is still on the unit circle, which is the honest
    answer for an undressed operator -- no band is long enough, and the reply
    is a modal representation rather than a wider mask.

    Compare against ``sse_g_band``, which counts blocks. A band shorter than
    this cuts a mode that has not yet decayed; the discarded weight is then
    ``exp(-b/xi)`` rather than something small.
    """
    modes = bloch_modes(a_ii, a_ij, a_ji, nevp=nevp)
    xi = modes.xi[modes.decaying]
    if np.any(np.isinf(modes.xi)):
        return float("inf")
    return float(np.max(xi)) if xi.size else 0.0
