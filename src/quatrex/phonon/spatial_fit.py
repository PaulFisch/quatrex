# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The spatial modal REPRESENTATION: coefficients, pruning, geometric sums.

:mod:`quatrex.phonon.spatial_modes` answers "what are the complex bands of this
operator". This module answers "and what does the block sequence look like in
them":

.. math::
    G(n) = V\,\mathrm{diag}(\lambda^n)\,C ,

with :math:`V` from the pencil and :math:`C` solved from exact near blocks. The
split is deliberate -- ``spatial_modes`` is diagnostic and touches nothing the
solver uses, while what is here is a representation the bubble can be fed from,
so it does not belong under that docstring.

The one rule that is not obvious, measured in ``phonon/docs/spatial_representation.md`` Sec. 0.5
and re-derived here as an API constraint: **the rank and the fit anchor are one
choice, not two.** Fitting a truncated mode set at ``n = 1, 2`` pushes the
dropped modes' weight onto the survivors -- 1.2e-02 against 6.0e-05 on CNT at
rank 22 of 36 -- because the design matrix still sees them in the data. And a
fit anchored past a mode's range carries no information about it, so short
blocks degrade even at full rank. Hence :func:`modal_fit` always solves for the
FULL coefficient set at the anchor and :func:`prune_by_amplitude` drops modes
afterwards **without refitting the survivors**, ranking by what each mode
actually contributes at the start of the far field rather than by
:math:`|\lambda|`.

Everything is host numpy: these are per-(omega, q) problems of size
``n_dof x n_modes``, solved by ``lstsq``, and the arrays arrive from the pencil
already on the host.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from qttools import NDArray

__all__ = [
    "ModalSeries",
    "modal_fit",
    "tail_amplitudes",
    "prune_by_amplitude",
    "geometric_factor",
    "geometric_block_sum",
]


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


@dataclass(frozen=True)
class ModalSeries:
    r"""``G(n) = V diag(lambda^n) C`` for one ``(omega, q)``.

    Attributes
    ----------
    lam : NDArray
        ``(r,)`` retained Bloch factors.
    vecs : NDArray
        ``(b, r)`` mode vectors, columns matching ``lam``.
    coef : NDArray
        ``(r, b')`` coefficient rows, from :func:`modal_fit`.
    anchor : tuple of int
        The distances the coefficients were solved at. The representation is
        valid AT AND BEYOND ``min(anchor)``; see the module docstring.
    """

    lam: NDArray
    vecs: NDArray
    coef: NDArray
    anchor: tuple[int, ...]

    @property
    def rank(self) -> int:
        return int(self.lam.size)

    def block(self, n: int) -> NDArray:
        """The reconstructed block at separation ``n``."""
        return self.vecs @ (self.lam[:, None] ** n * self.coef)

    def blocks(self, ns) -> NDArray:
        """``(len(ns), b, b')`` reconstructed blocks."""
        ns = np.asarray(ns)
        return np.stack([self.block(int(n)) for n in ns])


def modal_fit(vecs, lam, blocks, anchors, *, rcond=None) -> ModalSeries:
    r"""Solve the FULL coefficient set ``C`` from exact blocks at ``anchors``.

    Parameters
    ----------
    vecs, lam
        ``(b, r)`` mode vectors and ``(r,)`` Bloch factors, from
        :func:`~quatrex.phonon.spatial_modes.bloch_modes`. Pass the retained
        (typically decaying) branch; the caller owns that selection because it
        is a physical choice, not a numerical one.
    blocks : mapping or sequence
        Exact blocks, indexed by separation: ``blocks[n]`` is ``(b, b')``.
    anchors : sequence of int
        Distances to fit at. At least ``ceil(r / b)`` of them are needed for
        the design matrix to have full column rank; fewer is not an error here
        but ``lstsq`` will return a minimum-norm solution that extrapolates
        badly, so it is checked.
    rcond
        Passed to ``numpy.linalg.lstsq``. ``None`` keeps numpy's default.

    Notes
    -----
    No truncation happens here, by design. Truncating the mode set and THEN
    fitting redistributes the dropped weight onto the survivors; the supported
    route is a full fit followed by :func:`prune_by_amplitude`.
    """
    vecs = np.asarray(_host(vecs))
    lam = np.asarray(_host(lam)).ravel()
    anchors = tuple(int(n) for n in anchors)
    if not anchors:
        raise ValueError("modal_fit: need at least one anchor distance")
    b, r = vecs.shape
    if lam.size != r:
        raise ValueError(
            f"modal_fit: {r} mode vectors but {lam.size} Bloch factors")
    if len(anchors) * b < r:
        raise ValueError(
            f"modal_fit: {len(anchors)} anchors x {b} rows = "
            f"{len(anchors) * b} equations cannot determine {r} coefficient "
            "rows; give more anchors or fewer modes")

    design = np.vstack([vecs @ np.diag(lam ** n) for n in anchors])
    rhs = np.vstack([np.asarray(_host(blocks[n])) for n in anchors])
    coef = np.linalg.lstsq(design, rhs, rcond=rcond)[0]
    return ModalSeries(lam=lam, vecs=vecs, coef=coef, anchor=anchors)


def tail_amplitudes(series: ModalSeries, r0: int) -> NDArray:
    r"""``a_mu(R0) = ||V_mu C_mu|| |lambda_mu|^{R0}``, the far-field weight.

    The rank-one term of mode ``mu`` has Frobenius norm
    ``||V[:, mu]|| ||C[mu, :]||``, so this is what the mode is worth at the
    START of the far field -- which is the quantity to prune on. Pruning on
    ``|lambda|`` alone keeps a long-range mode the vertex barely couples to and
    drops a short-range one it couples to strongly.
    """
    w = (np.linalg.norm(series.vecs, axis=0)
         * np.linalg.norm(series.coef, axis=1))
    return w * np.abs(series.lam) ** int(r0)


def prune_by_amplitude(series: ModalSeries, r0: int, *, rank: int | None = None,
                       tol: float | None = None) -> tuple[ModalSeries, NDArray]:
    r"""Drop modes negligible at ``r0``, keeping the surviving coefficients.

    Returns ``(pruned, keep_mask)``. Exactly one of ``rank`` (keep the ``rank``
    largest) or ``tol`` (keep everything above ``tol`` times the largest) must
    be given.

    The survivors' coefficients are **not** refitted. That is the whole point:
    a refit at the same anchor would redistribute the dropped modes' weight, and
    a refit at a farther anchor would lose the information the near blocks
    carried. What is dropped is dropped, and the error is bounded by the
    amplitudes that were dropped -- which is a number the caller can report.
    """
    if (rank is None) == (tol is None):
        raise ValueError("prune_by_amplitude: give exactly one of rank, tol")
    amp = tail_amplitudes(series, r0)
    if rank is not None:
        rank = int(rank)
        if rank < 1 or rank > series.rank:
            raise ValueError(
                f"prune_by_amplitude: rank {rank} outside 1..{series.rank}")
        cut = np.sort(amp)[::-1][rank - 1]
        keep = amp >= cut
        if keep.sum() > rank:                      # ties at the cut
            idx = np.argsort(-amp, kind="stable")[:rank]
            keep = np.zeros_like(amp, dtype=bool)
            keep[idx] = True
    else:
        peak = float(np.max(amp)) if amp.size else 0.0
        keep = amp > float(tol) * peak

    pruned = replace(series, lam=series.lam[keep], vecs=series.vecs[:, keep],
                     coef=series.coef[keep])
    return pruned, keep


def geometric_factor(zeta, k: int, *, switch: float = 1e-6) -> NDArray:
    r"""``sum_{j=0}^{k-1} zeta^j``, stable through ``zeta = 1``.

    Away from one this is ``(1 - zeta^k)/(1 - zeta)``. At ``zeta = 1`` that is
    0/0 and the limit is ``k``; the closed form loses digits well before it
    reaches the singularity, so with ``u = zeta - 1`` the binomial identity

    .. math::
        \sum_{j=0}^{k-1}(1+u)^j = \sum_{i\ge 0} \binom{k}{i+1} u^i

    is used instead below ``|u| < switch``. The switch balances the two error
    sources: the closed form loses ``eps/|u|`` relative accuracy to
    cancellation, so at ``switch = 1e-6`` both branches hold about 1e-10. The
    series' successive terms fall like ``k u / i``, so it needs ``|k u| \ll 1``
    -- checked, not assumed.
    """
    zeta = np.asarray(zeta, dtype=complex)
    k = int(k)
    if k < 0:
        raise ValueError(f"geometric_factor: k must be >= 0, got {k}")
    if k == 0:
        return np.zeros_like(zeta)

    u = zeta - 1.0
    near = np.abs(u) < switch
    out = np.empty_like(zeta)

    far = ~near
    if np.any(far):
        zf = zeta[far] if zeta.ndim else zeta
        out_far = (1.0 - zf ** k) / (1.0 - zf)
        if zeta.ndim:
            out[far] = out_far
        else:
            out = np.asarray(out_far)

    if np.any(near):
        un = u[near] if u.ndim else u
        if np.any(np.abs(un) * k > 0.5):
            raise ValueError(
                "geometric_factor: the series branch needs |k (zeta-1)| << 1; "
                f"got up to {float(np.max(np.abs(un)) * k):.3g}. Lower "
                "`switch` so the closed form is used instead.")
        term = np.full(np.shape(un), float(k), dtype=complex)   # i = 0
        acc = term.copy()
        binom = float(k)
        for i in range(1, 24):
            binom = binom * (k - i) / (i + 1.0)
            term = binom * un ** i
            acc = acc + term
            if np.all(np.abs(term) < 1e-18 * np.abs(acc)):
                break
        if u.ndim:
            out[near] = acc
        else:
            out = np.asarray(acc)
    return out


def geometric_block_sum(series: ModalSeries, n0: int, n1: int) -> NDArray:
    r"""``sum_{n=n0}^{n1-1} G(n)``, without materialising a single block.

    ``V diag(lambda^{n0} sum_{j<n1-n0} lambda^j) C``: ``r`` scalar series and
    two matrix products, whatever the range. Half-open in ``n1``, matching
    ``range(n0, n1)``.
    """
    if n1 < n0:
        raise ValueError(f"geometric_block_sum: n1={n1} < n0={n0}")
    fac = series.lam ** int(n0) * geometric_factor(series.lam, int(n1) - int(n0))
    return series.vecs @ (fac[:, None] * series.coef)
