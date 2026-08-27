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
measures the operator the solver already builds. The representation built ON
these modes -- coefficient fits, tail pruning, geometric sums -- lives in
:mod:`quatrex.phonon.spatial_fit`, so that separation survives.

The same pencil is what the lead OBC solves, so this reuses
:mod:`qttools.nevp` rather than reimplementing it -- with undressed lead blocks
it must return the lead's own bands, which is the regression that pins the
convention.

Two generalisations beyond the quadratic case, both needed by the spatially
analytic tail programme (``phonon/docs/spatial_analytic_tail.md``):

* **Arbitrary pencil degree.** Once the output pin is removed, :math:`\Sigma^R`
  has range :math:`M = 2p+b > 1` and the spatial recurrence is
  :math:`\sum_{n=-M}^{M} a_n \lambda^n = 0`, not a quadratic. Solving the
  quadratic against a degree-:math:`2M` reference measures the wrong thing.
  :func:`bloch_modes_poly` takes the whole coefficient tuple;
  :class:`~qttools.nevp.full.Full` linearises any length.
* **Batching.** ``Full`` batches over a leading axis, so a frequency (or
  frequency x q) sweep is one call rather than a Python loop -- measured 4-8x.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qttools import NDArray, xp

__all__ = [
    "ModeSet",
    "bloch_modes",
    "bloch_modes_poly",
    "decay_lengths",
    "band_range_cells",
    "mode_band_range",
    "nevp_residual",
]


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


@dataclass(frozen=True)
class ModeSet:
    """Complex bands at one frequency, or at a batch of them.

    Attributes
    ----------
    lam : NDArray
        Bloch factors :math:`\\lambda = e^{ika}`, ``(..., n_modes)``.
    vecs : NDArray
        Right mode vectors, ``(..., n_dof, n_modes)``.
    xi : NDArray
        Decay length in CELLS, :math:`-1/\\ln|\\lambda|`. Positive and finite
        for a decaying mode, ``inf`` for a propagating one, ``0`` for a mode
        that dies within one cell, and ``nan`` for a growing partner, whose
        range is a statement about the other direction.
    residual : NDArray or None
        Per-mode NEVP residual :math:`\\|(\\sum_n a_n \\lambda^n) v\\|/\\|v\\|`,
        normalised by :math:`|\\lambda|`, when it was requested. ``Full``
        returns every root of the linearisation including the spurious ones at
        zero and infinity that a rank-deficient coupling produces, and this is
        what separates them. It is reported, never applied: callers mask with
        :meth:`converged`, because the mode COUNT is itself a diagnostic and
        silently dropping roots destroys it.
    """

    lam: NDArray
    vecs: NDArray
    xi: NDArray
    residual: NDArray | None = None

    @property
    def decaying(self) -> NDArray:
        """Mask of modes that decay in the ``+`` direction (``|lambda| < 1``)."""
        return np.abs(np.asarray(_host(self.lam))) < 1.0

    @property
    def propagating(self, tol: float = 1e-10) -> NDArray:
        """Mask of modes on the unit circle -- undamped, so unbounded range."""
        return np.abs(np.abs(np.asarray(_host(self.lam))) - 1.0) < 1e-10

    def converged(self, tol: float = 1e-3) -> NDArray:
        """Mask of modes whose NEVP residual is below ``tol``.

        The default matches ``Spectral.residual_tolerance``. Raises when the
        residual was not computed, rather than returning an all-true mask that
        would read as "everything converged".
        """
        if self.residual is None:
            raise ValueError(
                "ModeSet.converged: no residual was computed; pass "
                "residual=True to bloch_modes/bloch_modes_poly.")
        return np.asarray(_host(self.residual)) < tol


def nevp_residual(a_blocks, lam, vecs, *, normalise: bool = True) -> NDArray:
    r"""Per-mode residual of :math:`\left(\sum_n a_n\lambda^n\right)v = 0`.

    ``a_blocks`` is ordered lowest to highest, so for a tuple of length
    ``2M+1`` the exponents run ``-M ... +M``. Normalised by ``|v|`` because
    the eigenvectors are not normalised, and (by default) by ``|lambda|``, the
    convention of
    :meth:`qttools.boundary_conditions.obc.spectral.Spectral._find_reflected_modes`
    -- without it a root near zero looks converged for free.
    """
    lam = np.asarray(_host(lam))
    vecs = np.asarray(_host(vecs))
    m = len(a_blocks) // 2
    blocks = [np.asarray(_host(a)) for a in a_blocks]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        prod = sum(
            (b[..., :, :] @ vecs) * lam[..., np.newaxis, :] ** n
            for b, n in zip(blocks, range(-m, m + 1))
        )
        res = np.linalg.norm(prod, axis=-2) / np.linalg.norm(vecs, axis=-2)
        if normalise:
            res = res / np.abs(lam)
    return res


def bloch_modes_poly(a_blocks, nevp=None, *, residual: bool = False) -> ModeSet:
    r"""Solve :math:`\left(\sum_{n=-M}^{M} a_n\lambda^n\right)v = 0`.

    Parameters
    ----------
    a_blocks : sequence of NDArray
        ``2M+1`` square blocks ``(..., b, b)``, ordered from the lowest power
        to the highest -- ``(a_{-M}, ..., a_0, ..., a_{+M})``. This is
        :class:`qttools.nevp.nevp.NEVP`'s own ordering, so no reversal happens
        anywhere. Any leading axes are a batch and are preserved.
    nevp : NEVP, optional
        Defaults to the dense linearisation
        :class:`~qttools.nevp.full.Full`. It takes every root, which is what a
        survey wants; the Beyn contour returns only what its annulus encloses
        and would need a radius chosen in advance.
    residual : bool
        Compute :func:`nevp_residual` and carry it on the result.

    Notes
    -----
    A degree-``2M`` pencil of block size ``b`` has ``2Mb`` finite roots, ``Mb``
    of them inside the unit disc for a passive dressed operator. The reciprocal
    pairing :math:`\lambda\bar\lambda = 1` that the quadratic case enjoys does
    NOT survive to higher degree, so nothing here assumes it.
    """
    if nevp is None:
        from qttools.nevp.full import Full

        nevp = Full()

    a_blocks = tuple(a_blocks)
    if len(a_blocks) < 3 or len(a_blocks) % 2 == 0:
        raise ValueError(
            "bloch_modes_poly: need an odd number of coefficient blocks "
            f"(2M+1, M >= 1), got {len(a_blocks)}")

    arrs = tuple(xp.asarray(a) for a in a_blocks)
    if arrs[0].shape[-1] != arrs[0].shape[-2]:
        raise ValueError(
            f"bloch_modes: blocks must be square, got {arrs[0].shape[-2:]}")
    if any(a.shape != arrs[0].shape for a in arrs):
        raise ValueError(
            "bloch_modes_poly: every coefficient block must have the same "
            f"shape; got {[tuple(a.shape) for a in arrs]}")

    batched = arrs[0].ndim > 2
    blocks = arrs if batched else tuple(a[xp.newaxis, :, :] for a in arrs)

    ws, vs = nevp(blocks)
    lam = np.asarray(_host(ws))
    vecs = np.asarray(_host(vs))
    if not batched:
        # Byte-identical to the pre-batching path for a single pencil.
        lam = lam.ravel()
        vecs = vecs.reshape(vecs.shape[-2], -1) if vecs.ndim > 2 else vecs

    res = None
    if residual:
        host_blocks = [np.asarray(_host(a)) for a in arrs]
        res = nevp_residual(host_blocks, lam, vecs)
    return ModeSet(lam=lam, vecs=vecs, xi=decay_lengths(lam), residual=res)


def bloch_modes(a_ii, a_ij, a_ji, nevp=None, *, residual: bool = False) -> ModeSet:
    r"""Solve :math:`a_{ji}\lambda^{-1} + a_{ii} + a_{ij}\lambda = 0`.

    The block convention is the OBC's, so the blocks are those of the SYSTEM
    matrix -- ``a_ii = z^2 I - H_00 - Sigma^R_00``, ``a_ij = -H_01 -
    Sigma^R_01`` -- and not of the dynamical matrix. Passing dynamical-matrix
    blocks would solve a different pencil and silently return the wrong bands,
    so the caller supplies what the solver already assembled.

    Parameters
    ----------
    a_ii, a_ij, a_ji : NDArray
        One square block each, ``(b, b)``, or a batch ``(..., b, b)``.
    nevp : NEVP, optional
        See :func:`bloch_modes_poly`.
    residual : bool
        Carry the per-mode NEVP residual on the result.

    The nearest-layer case of :func:`bloch_modes_poly`; a wider-ranged
    ``Sigma^R`` needs that one.
    """
    return bloch_modes_poly((a_ji, a_ii, a_ij), nevp=nevp, residual=residual)


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


def band_range_cells(a_ii, a_ij, a_ji, nevp=None):
    r"""Block range a truncation must keep: the SLOWEST decaying mode.

    ``inf`` when any mode is still on the unit circle, which is the honest
    answer for an undressed operator -- no band is long enough, and the reply
    is a modal representation rather than a wider mask.

    Compare against ``sse_g_band``, which counts blocks. A band shorter than
    this cuts a mode that has not yet decayed; the discarded weight is then
    ``exp(-b/xi)`` rather than something small.

    Returns a ``float`` for a single pencil and an array over the leading axes
    for a batch. The reduction is over the mode axis only -- a global ``max``
    would collapse a frequency sweep to one number and report the worst
    frequency's range at every frequency.
    """
    modes = bloch_modes(a_ii, a_ij, a_ji, nevp=nevp)
    return mode_band_range(modes)


def mode_band_range(modes: ModeSet):
    """:func:`band_range_cells` on an already-solved :class:`ModeSet`."""
    xi = np.asarray(modes.xi)
    dec = modes.decaying
    if xi.ndim == 1:
        if np.any(np.isinf(xi)):
            return float("inf")
        live = xi[dec]
        return float(np.max(live)) if live.size else 0.0

    masked = np.where(dec, xi, -np.inf)
    out = np.max(masked, axis=-1)
    out = np.where(np.isfinite(out), out, 0.0)
    return np.where(np.any(np.isinf(xi), axis=-1), np.inf, out)
