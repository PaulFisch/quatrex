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
    "BlockLayout",
    "delta_from_sigma",
    "nnz_to_blocks",
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


class BlockLayout:
    r"""Precomputed map from the stored sparsity pattern to BTD block views.

    :func:`nnz_to_blocks` answers the same question, but it re-derives the
    answer on every call: two ``np.searchsorted`` passes over ``nnz``, then a
    ``np.where((br == i) & (bc == j))`` scan and a fresh allocation per block
    pair. Inside the bordered Newton that runs once per candidate per step --
    117 times for nine candidates on the local bed -- for an answer that
    depends only on ``(rows, cols, block_sizes)``, which are fixed for a whole
    SCBA iteration.

    So the scan is done once and stored as a single gather. The band blocks are
    laid out end to end in a flat buffer of length ``total``; ``source`` says,
    for each slot of that buffer, which ``nnz`` column feeds it, with a
    sentinel for slots the pattern leaves empty. Applying the layout to
    ``Delta`` is then one fancy-index gather, and the BTD block list is a set
    of **zero-copy reshaped views** of the flat buffer -- a ``(*stack, total)``
    array with unit last stride reshapes to ``(*stack, b_i, b_j)`` without a
    copy, so materialising the blocks costs no kernel at all.

    Attributes
    ----------
    block_sizes : np.ndarray
        The partition, as given.
    total : int
        Length of the flat band buffer, ``sum_{|I-J| <= band} b_I b_J``.
    source : NDArray
        ``(total,)`` index into a value array PADDED with one trailing zero
        column; ``nnz`` marks an empty slot.
    diag : NDArray
        ``(n_dof,)`` flat positions of the operator's diagonal entries, in
        increasing global-index order.
    slices : list[tuple[tuple[int, int], slice, int, int]]
        ``((I, J), slice, b_I, b_J)`` in buffer order.

    """

    def __init__(self, rows, cols, block_sizes, *, band: int = 1):
        sizes = np.asarray(_host(block_sizes), dtype=int)
        off = np.concatenate(([0], np.cumsum(sizes)))
        r = np.asarray(_host(rows), dtype=int)
        c = np.asarray(_host(cols), dtype=int)
        nnz = int(r.size)
        br = np.searchsorted(off, r, side="right") - 1
        bc = np.searchsorted(off, c, side="right") - 1

        self.block_sizes = sizes
        self.nnz = nnz
        self._band = int(band)
        # Sentinel = nnz: the caller appends one zero column, so an empty slot
        # gathers a zero. This reproduces nnz_to_blocks' zero-filled block.
        source = np.full(0, nnz, dtype=np.int64)
        pieces, self.slices, pos = [], [], 0
        for i in range(len(sizes)):
            for j in range(max(0, i - band), min(len(sizes), i + band + 1)):
                bi, bj = int(sizes[i]), int(sizes[j])
                blk = np.full(bi * bj, nnz, dtype=np.int64)
                sel = np.where((br == i) & (bc == j))[0]
                if sel.size:
                    # Last wins on a duplicated (row, col), exactly as the
                    # fancy-index assignment in nnz_to_blocks does.
                    blk[(r[sel] - off[i]) * bj + (c[sel] - off[j])] = sel
                pieces.append(blk)
                self.slices.append(((i, j), slice(pos, pos + bi * bj), bi, bj))
                pos += bi * bj
        source = np.concatenate(pieces) if pieces else source
        self.total = int(pos)
        self.source = xp.asarray(source)

        # Flat positions of the global diagonal, for the z^2 I term.
        diag = np.full(int(off[-1]), -1, dtype=np.int64)
        for (i, j), sl, bi, bj in self.slices:
            if i == j:
                diag[off[i]:off[i + 1]] = sl.start + np.arange(bi) * bj + np.arange(bi)
        if (diag < 0).any():
            raise ValueError("BlockLayout: the band does not cover the diagonal.")
        self.diag = xp.asarray(diag)
        self.corner = {i: sl for (i, j), sl, _, _ in self.slices if i == j}
        self._build_band()

    def gather(self, values: NDArray) -> NDArray:
        """Lay ``(..., nnz)`` values out in band order, zero where absent."""
        if int(values.shape[-1]) != self.nnz:
            raise ValueError(
                f"values carry {int(values.shape[-1])} nnz, the layout was "
                f"built for {self.nnz}."
            )
        pad = xp.zeros(values.shape[:-1] + (1,), dtype=values.dtype)
        return xp.concatenate([values, pad], axis=-1)[..., self.source]

    def blocks(self, flat: NDArray):
        """``(a_ii, a_ij, a_ji)`` views of a ``(*stack, total)`` buffer.

        No copy and no kernel: each block is a reshaped slice.
        """
        if int(flat.shape[-1]) != self.total:
            raise ValueError(
                f"buffer has {int(flat.shape[-1])} entries, the layout needs "
                f"{self.total}."
            )
        stack = flat.shape[:-1]
        a_ii, a_ij, a_ji = {}, {}, {}
        for (i, j), sl, bi, bj in self.slices:
            view = flat[..., sl].reshape(stack + (bi, bj))
            if i == j:
                a_ii[i] = view
            elif j == i + 1:
                a_ij[i] = view
            elif j == i - 1:
                a_ji[j] = view
        n = len(self.block_sizes)
        return ([a_ii[i] for i in range(n)],
                [a_ij[i] for i in range(n - 1)],
                [a_ji[i] for i in range(n - 1)])

    def block_dict(self, flat: NDArray) -> dict[tuple[int, int], NDArray]:
        """The same views, keyed like :func:`nnz_to_blocks`' output."""
        return {ij: flat[..., sl].reshape(flat.shape[:-1] + (bi, bj))
                for ij, sl, bi, bj in self.slices}

    # -- band-tensor form -------------------------------------------------- #
    #
    # ``blocks()`` hands back a LIST, one entry per block, so every consumer
    # walks it in Python and the call count grows with the device length. The
    # band form carries the same data as a single dense tensor
    # ``(..., n_blocks, 3, bmax, bmax)`` -- block row, band offset ``j - i + 1``,
    # and the two intra-block indices -- padded to the largest block and zero
    # filled outside the band and outside a short block. Every operation the
    # legs need on a block-tridiagonal operator is then ONE einsum or matmul
    # over that tensor, with no loop over blocks at all.

    def _build_band(self) -> None:
        sizes = self.block_sizes
        nb = int(len(sizes))
        bmax = int(sizes.max())
        off = np.concatenate(([0], np.cumsum(sizes)))
        n_dof = int(off[-1])
        self.n_blocks, self.bmax, self.n_dof = nb, bmax, n_dof

        # Global row of local row r of block i; n_dof marks padding, which
        # gathers a zero from a caller-appended sentinel row.
        row_index = np.full((nb, bmax), n_dof, dtype=np.int64)
        for i in range(nb):
            row_index[i, :sizes[i]] = np.arange(off[i], off[i + 1])
        self.row_index = xp.asarray(row_index)

        # Which nnz column feeds band slot (i, o, r, s); self.nnz marks empty.
        band = np.full((nb, 3, bmax, bmax), self.nnz, dtype=np.int64)
        # Sentinel = the band size: an nnz entry OUTSIDE the band unbands to
        # zero, matching nnz_to_blocks, which materialises |I-J| <= band only.
        band_of_nnz = np.full(self.nnz, nb * 3 * bmax * bmax, dtype=np.int64)
        flat_src = np.asarray(_host(self.source))
        for (i, j), sl, bi, bj in self.slices:
            o = j - i + 1
            blk = flat_src[sl].reshape(bi, bj)
            band[i, o, :bi, :bj] = blk
            keep = blk < self.nnz
            band_of_nnz[blk[keep]] = np.ravel_multi_index(
                (np.full(int(keep.sum()), i), np.full(int(keep.sum()), o),
                 *np.nonzero(keep)), (nb, 3, bmax, bmax))
        self.band_source = xp.asarray(band)
        self.band_of_nnz = xp.asarray(band_of_nnz)
        self.band_size = nb * 3 * bmax * bmax

        # Block row j = i + o - 1 reached from (i, o); nb marks off the end.
        shift = np.add.outer(np.arange(nb), np.arange(3) - 1)
        self.shift_index = xp.asarray(np.where((shift < 0) | (shift >= nb),
                                               nb, shift))

        # Slot holding the TRANSPOSED entry: (i, o, r, s) -> (i+o-1, 2-o, s, r).
        # Out-of-band slots map to a sentinel, which :meth:`band_transpose`
        # serves a zero from.
        ii, oo, rr, ss = np.meshgrid(np.arange(nb), np.arange(3),
                                     np.arange(bmax), np.arange(bmax),
                                     indexing="ij")
        jj = ii + oo - 1
        ok = (jj >= 0) & (jj < nb)
        tflat = np.full((nb, 3, bmax, bmax), nb * 3 * bmax * bmax,
                        dtype=np.int64)
        tflat[ok] = np.ravel_multi_index(
            (jj[ok], 2 - oo[ok], ss[ok], rr[ok]), (nb, 3, bmax, bmax))
        self.band_transpose_index = xp.asarray(tflat.reshape(-1))

    def band_transpose(self, band: NDArray) -> NDArray:
        r"""The band tensor of :math:`A^{T}` (no conjugation).

        Slot ``(i, o, r, s)`` holds ``A`` at global ``(row(i,r), col(i,o,s))``,
        and its transpose lives at ``(i+o-1, 2-o, s, r)``. One gather.
        """
        flat = band.reshape(band.shape[:-4] + (self.band_size,))
        pad = xp.zeros(flat.shape[:-1] + (1,), dtype=flat.dtype)
        full = xp.concatenate([flat, pad], axis=-1)
        return full[..., self.band_transpose_index].reshape(band.shape)

    def band(self, values: NDArray) -> NDArray:
        """``(..., nnz)`` values as the band tensor ``(..., nb, 3, bmax, bmax)``."""
        pad = xp.zeros(values.shape[:-1] + (1,), dtype=values.dtype)
        full = xp.concatenate([values, pad], axis=-1)
        return full[..., self.band_source.reshape(-1)].reshape(
            values.shape[:-1] + (self.n_blocks, 3, self.bmax, self.bmax))

    def unband(self, band: NDArray) -> NDArray:
        """Inverse of :meth:`band` onto the stored pattern, ``(..., nnz)``."""
        flat = band.reshape(band.shape[:-4] + (self.band_size,))
        pad = xp.zeros(flat.shape[:-1] + (1,), dtype=flat.dtype)
        return xp.concatenate([flat, pad], axis=-1)[..., self.band_of_nnz]

    def to_blocks(self, dense: NDArray) -> NDArray:
        """``(..., n_dof, m)`` laid out per block row, ``(..., nb, bmax, m)``.

        Rows that do not exist in a short block gather a zero from a sentinel
        row appended here, so the padding never carries weight.
        """
        pad = xp.zeros(dense.shape[:-2] + (1, dense.shape[-1]),
                       dtype=dense.dtype)
        full = xp.concatenate([dense, pad], axis=-2)
        return xp.take(full, self.row_index, axis=full.ndim - 2)

    def from_blocks(self, blocked: NDArray) -> NDArray:
        """Inverse of :meth:`to_blocks`, ``(..., nb, bmax, m) -> (..., n_dof, m)``."""
        flat = blocked.reshape(blocked.shape[:-3]
                               + (self.n_blocks * self.bmax,) + blocked.shape[-1:])
        out = xp.zeros(blocked.shape[:-3] + (self.n_dof + 1,) + blocked.shape[-1:],
                       dtype=blocked.dtype)
        out[..., self.row_index.reshape(-1), :] = flat
        return out[..., :self.n_dof, :]

    def band_neighbours(self, blocked: NDArray) -> NDArray:
        """``(..., nb, bmax, m) -> (..., nb, 3, bmax, m)``: block ``i + o - 1``.

        The partner of :meth:`band`: together they turn a block-tridiagonal
        apply into one matmul whose contracted axis is ``(o, s)``.
        """
        pad = xp.zeros(blocked.shape[:-3] + (1,) + blocked.shape[-2:],
                       dtype=blocked.dtype)
        full = xp.concatenate([blocked, pad], axis=-3)
        return xp.take(full, self.shift_index, axis=full.ndim - 3)

    def apply_band(self, band: NDArray, x: NDArray) -> NDArray:
        """``A @ x`` for a band-tensor ``A`` and dense ``x``, in ONE matmul.

        ``band`` is ``(..., nb, 3, bmax, bmax)`` and ``x`` is
        ``(..., n_dof, m)``; the two batch shapes broadcast. The band offset
        and the contracted intra-block index are folded into a single axis of
        length ``3 * bmax``, so the whole block-tridiagonal apply is one
        batched GEMM rather than a Python walk over the blocks.
        """
        nb, bm = self.n_blocks, self.bmax
        a = xp.swapaxes(band, -3, -2).reshape(
            band.shape[:-4] + (nb, bm, 3 * bm))
        v = self.band_neighbours(self.to_blocks(x))
        v = v.reshape(v.shape[:-4] + (nb, 3 * bm, v.shape[-1]))
        return self.from_blocks(a @ v)

    def flatten_blocks(self, blocks: dict[tuple[int, int], NDArray]) -> NDArray:
        """Inverse of :meth:`block_dict` for a dense block dict (e.g. ``D``)."""
        stack = xp.broadcast_shapes(
            *(xp.asarray(b).shape[:-2] for b in blocks.values()))
        out = xp.zeros(stack + (self.total,), dtype=xp.complex128)
        for ij, sl, bi, bj in self.slices:
            b = blocks.get(ij)
            if b is not None:
                b = xp.asarray(b)
                out[..., sl] = xp.broadcast_to(
                    b, stack + b.shape[-2:]).reshape(stack + (bi * bj,))
        return out
