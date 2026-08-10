# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Evaluating the Dyson operator at complex frequency, in the solver's layout.

To find poles of :math:`G^R = M^R(z)^{-1}` the operator has to be assembled at
complex :math:`z`. Three pieces are needed and each lands naturally in a
different part of the existing data layout:

``z^2 I``
    Trivial. Note ``eta`` is identically zero in this project, so the diagonal
    is literally :math:`z^2` and no broadening convention has to be continued.

``D`` and the contact self-energy
    Frequency-independent and boundary-only respectively; both come from the
    solver's own blocks, evaluated at complex ``z``.

:math:`\Sigma_s^R(z)`
    The one that needs care. It is reconstructed from
    :math:`\Delta = \Sigma^> - \Sigma^<` by the exact cell-integrated
    continuation of :mod:`quatrex.phonon.pole_kernel`, and that contraction runs
    over **all** frequencies.

The distribution follows from that last point. In the ``"nnz"`` state every rank
already owns the full frequency axis and a slice of the sparsity pattern, so the
contraction is one dense GEMM with **zero communication**. A ``dtranspose`` then
puts the probe axis in the stack position, where each rank owns whole matrices
and block access is legal -- which is what the pole solve needs. This reuses the
solver's existing two-state machinery rather than adding a communication path.

``Delta`` itself needs no new state: in the solver's occupation-positive
convention it is exactly ``sigma_lesser.data - sigma_greater.data``, so it is
reconstructed from the MIXED iterate the Dyson operator is built from. Caching
it would let it drift out of step with :math:`\Sigma^R` across the mixer.
"""
from __future__ import annotations

import numpy as np

from qttools import NDArray, xp

from quatrex.phonon.pole_kernel import (
    continuation_weights,
    contract_delta,
    delta_local_fit,
)

__all__ = [
    "delta_from_sigma",
    "probe_sigma_retarded",
    "nnz_to_blocks",
    "assemble_m_blocks",
    "ProbePlan",
]


def delta_from_sigma(sigma_lesser: NDArray, sigma_greater: NDArray) -> NDArray:
    r"""Recover :math:`\Delta = \Sigma^> - \Sigma^<` from the stored buffers.

    The bubble writes ``Sigma`` with a sign flip (the solver stores
    occupation-positive quantities, the opposite of the textbook Keldysh
    feedback convention), so the raw ``sg - sl`` the retarded reconstruction is
    built from equals ``sigma_lesser - sigma_greater`` as stored.

    Getting this backwards is not a small error: it flips the sign of the
    damping and moves every pole to the wrong half plane.
    """
    return sigma_lesser - sigma_greater


def probe_sigma_retarded(
    delta: NDArray,
    freqs: NDArray,
    z: NDArray,
    *,
    orders: tuple[int, ...] = (0, 1),
    sheet: str = "II",
    transverse_shape: tuple = (),
    delta_order: int = 2,
    delta_window: int = 4,
) -> dict[int, NDArray]:
    r"""Contract :math:`\Sigma_s^R(z)` and its derivatives, rank-locally.

    Intended to be called in the ``"nnz"`` distribution state, where ``delta``
    carries the full frequency axis and a slice of the sparsity pattern, so this
    is a single GEMM per derivative order with no communication. All probes and
    orders should be batched into one call.

    Parameters
    ----------
    delta : NDArray
        ``(n_freq, *nk, nnz_local)`` from :func:`delta_from_sigma`.
    freqs : NDArray
        ``(n_freq,)`` uniform frequency grid (the convolution grid).
    z : NDArray
        ``(n_probe,)`` complex probe points, strictly off the real axis.
    orders : tuple[int, ...], optional
        Derivative orders in ``z`` to return. ``(0, 1)`` is what the bordered
        Newton corrector needs; add ``2`` for a quadratic local model.
    sheet : {"I", "II"}, optional
        ``"II"`` adds the local continuation of ``Delta``, which is the term
        that makes an in-band resonance visible at all -- on the first sheet the
        damping has the opposite sign and the pole is simply absent.
    transverse_shape : tuple, optional
        Transverse-momentum axis sizes; the bosonic mirror carries ``q -> -q``.
    delta_order, delta_window : int, optional
        Local model of ``Delta`` for the second-sheet term.

    Returns
    -------
    dict[int, NDArray]
        Derivative order -> ``(n_probe, *nk, nnz_local)``.

    """
    out: dict[int, NDArray] = {}
    for order in orders:
        w_pos, w_mir = continuation_weights(z, freqs, order=order)
        val = contract_delta(delta, w_pos, w_mir, transverse_shape=transverse_shape)
        if sheet == "II":
            val = val + delta_local_fit(
                delta, freqs, z, order=delta_order, window=delta_window,
                deriv=order, transverse_shape=transverse_shape,
            )
        out[order] = val
    return out


def nnz_to_blocks(
    values: NDArray,
    rows: NDArray,
    cols: NDArray,
    block_sizes: NDArray,
    *,
    band: int = 1,
) -> dict[tuple[int, int], NDArray]:
    """Scatter a sparse value vector into dense block-tridiagonal blocks.

    Parameters
    ----------
    values : NDArray
        ``(..., nnz)`` values on the stored pattern.
    rows, cols : NDArray
        ``(nnz,)`` global row and column indices.
    block_sizes : NDArray
        Block partition.
    band : int, optional
        Widest block offset to materialise. Default 1 (block-tridiagonal), which
        is all the Dyson operator ever needs: the scattering self-energy output
        band is pinned at ``|I-J| <= 1``.

    Returns
    -------
    dict[tuple[int, int], NDArray]
        ``(I, J) -> (..., b_I, b_J)`` dense blocks, zero where the pattern is
        empty.

    """
    sizes = np.asarray(_host(block_sizes), dtype=int)
    off = np.concatenate(([0], np.cumsum(sizes)))
    r = np.asarray(_host(rows), dtype=int)
    c = np.asarray(_host(cols), dtype=int)
    br = np.searchsorted(off, r, side="right") - 1
    bc = np.searchsorted(off, c, side="right") - 1

    stack = values.shape[:-1]
    blocks: dict[tuple[int, int], NDArray] = {}
    for i in range(len(sizes)):
        for j in range(max(0, i - band), min(len(sizes), i + band + 1)):
            sel = np.where((br == i) & (bc == j))[0]
            blk = xp.zeros(stack + (int(sizes[i]), int(sizes[j])),
                           dtype=values.dtype)
            if sel.size:
                blk[..., xp.asarray(r[sel] - off[i]), xp.asarray(c[sel] - off[j])] = (
                    values[..., xp.asarray(sel)]
                )
            blocks[(i, j)] = blk
    return blocks


def _host(a):
    return a.get() if hasattr(a, "get") else a


def assemble_m_blocks(
    z: complex,
    d_blocks: dict[tuple[int, int], NDArray],
    sigma_blocks: dict[tuple[int, int], NDArray] | None = None,
    obc_left: NDArray | None = None,
    obc_right: NDArray | None = None,
    *,
    block_sizes: NDArray | None = None,
) -> tuple[list[NDArray], list[NDArray], list[NDArray]]:
    r"""Assemble :math:`M(z) = z^2 I - D - \Sigma_c^R(z) - \Sigma_s^R(z)`.

    Parameters
    ----------
    z : complex
        Probe frequency (THz).
    d_blocks : dict
        Dynamical-matrix blocks ``(I, J) -> (b_I, b_J)``.
    sigma_blocks : dict, optional
        Scattering self-energy blocks at ``z``.
    obc_left, obc_right : NDArray, optional
        Contact self-energy on the first and last diagonal blocks, evaluated at
        ``z``.
    block_sizes : NDArray, optional
        Inferred from ``d_blocks`` when omitted.

    Returns
    -------
    tuple[list, list, list]
        ``(a_ii, a_ij, a_ji)`` ready for
        :class:`quatrex.phonon.btd_linalg.BTDFactorization`.

    """
    if block_sizes is None:
        n = 1 + max(i for i, _ in d_blocks)
        block_sizes = np.array([int(d_blocks[(i, i)].shape[-1]) for i in range(n)])
    sizes = np.asarray(_host(block_sizes), dtype=int)
    n = len(sizes)

    a_ii, a_ij, a_ji = [], [], []
    for i in range(n):
        m = z * z * xp.eye(int(sizes[i]), dtype=xp.complex128) - d_blocks[(i, i)]
        if sigma_blocks is not None:
            m = m - sigma_blocks[(i, i)]
        if i == 0 and obc_left is not None:
            m = m - obc_left
        if i == n - 1 and obc_right is not None:
            m = m - obc_right
        a_ii.append(m)
    for i in range(n - 1):
        up = -d_blocks[(i, i + 1)] + 0j
        lo = -d_blocks[(i + 1, i)] + 0j
        if sigma_blocks is not None:
            up = up - sigma_blocks[(i, i + 1)]
            lo = lo - sigma_blocks[(i + 1, i)]
        a_ij.append(up)
        a_ji.append(lo)
    return a_ii, a_ij, a_ji


class ProbePlan:
    """Bookkeeping for one batch of complex-frequency probes.

    A pole needs its operator and derivatives at the same ``z``, and every pole
    of every cluster wants them in the SAME contraction -- the GEMM is the
    dominant new cost, so it is issued once for all probes and orders together.
    This class keeps the slot layout so callers can address the result.
    """

    def __init__(self, orders: tuple[int, ...] = (0, 1, 2)):
        self.orders = tuple(orders)
        self._z: list[complex] = []
        self._tags: list[object] = []

    def add(self, z: complex, tag: object = None) -> int:
        """Register a probe point; returns its slot index."""
        self._z.append(complex(z))
        self._tags.append(tag)
        return len(self._z) - 1

    @property
    def z(self) -> NDArray:
        return xp.asarray(self._z, dtype=xp.complex128)

    @property
    def tags(self) -> list[object]:
        return list(self._tags)

    def __len__(self) -> int:
        return len(self._z)

    def evaluate(self, delta: NDArray, freqs: NDArray, **kw) -> dict[int, NDArray]:
        """Run the batched contraction for every registered probe."""
        if not self._z:
            return {o: xp.zeros((0,) + delta.shape[1:], dtype=delta.dtype)
                    for o in self.orders}
        return probe_sigma_retarded(delta, freqs, self.z, orders=self.orders, **kw)
