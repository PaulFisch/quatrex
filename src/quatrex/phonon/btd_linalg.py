# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Batched block-tridiagonal LU with reusable factors.

The pole sector needs something the existing solvers do not offer: repeated
linear solves ``M(z) x = b`` (and ``M(z)^H x = b``) against an *arbitrary* dense
right-hand side, with the factorisation amortised across several solves at the
same ``z``.

* :class:`qttools.greens_function_solver.rgf.RGF` computes the block-banded
  *selected inverse* plus the Keldysh pair. It exposes no factors and takes no
  arbitrary RHS.
* :class:`qttools.wave_function_solver.thomas.Thomas` is the right algorithm but
  takes a ``scipy`` CSR matrix, re-derives the block structure on every call
  (``_analyze``), and is not batched over a stack axis.

So this is a small dedicated helper: the same block Thomas recursion, ``xp``-native,
batched over the leading stack axes, storing the pivot inverses and the
elimination factors so that ``solve`` is pure GEMM work. Because the phonon Dyson
operator is block-tridiagonal in *every* configuration -- the scattering
self-energy output band is pinned at ``|I-J| <= 1`` regardless of ``sse_g_band``,
which widens only the inner Green's-function loops -- this covers the whole
production path.

Vectors are flat ``(*stack, n_dof, nrhs)`` arrays; block sizes may vary and are
carried as offsets, so nothing here assumes a uniform partition.
"""
from __future__ import annotations

import numpy as np

from qttools import NDArray, xp
from qttools.kernels import linalg

__all__ = ["BTDFactorization", "btd_matvec", "btd_norm2"]


def _offsets(block_sizes) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(np.asarray(block_sizes, dtype=int))))


def btd_matvec(
    a_ii: list[NDArray], a_ij: list[NDArray], a_ji: list[NDArray], x: NDArray
) -> NDArray:
    """Block-tridiagonal matrix-vector product.

    Parameters
    ----------
    a_ii : list[NDArray]
        Diagonal blocks, ``(*stack, b_i, b_i)``.
    a_ij : list[NDArray]
        Superdiagonal blocks ``M[i, i+1]``, ``(*stack, b_i, b_{i+1})``.
    a_ji : list[NDArray]
        Subdiagonal blocks ``M[i+1, i]``, ``(*stack, b_{i+1}, b_i)``.
    x : NDArray
        ``(*stack, n_dof, nrhs)``.

    Returns
    -------
    NDArray
        ``M @ x``, same shape as ``x``.

    """
    sizes = [int(b.shape[-1]) for b in a_ii]
    off = _offsets(sizes)
    out = xp.zeros_like(x)
    # Accumulate throughout: the sub-diagonal term of block i lands in block
    # i+1, so a plain assignment on the next iteration would clobber it.
    for i in range(len(a_ii)):
        seg = slice(off[i], off[i + 1])
        out[..., seg, :] += a_ii[i] @ x[..., seg, :]
        if i + 1 < len(a_ii):
            nxt = slice(off[i + 1], off[i + 2])
            out[..., seg, :] += a_ij[i] @ x[..., nxt, :]
            out[..., nxt, :] += a_ji[i] @ x[..., seg, :]
    return out


def btd_norm2(
    a_ii: list[NDArray],
    a_ij: list[NDArray],
    a_ji: list[NDArray],
    *,
    n_power: int = 12,
    seed: int = 0,
) -> NDArray:
    r"""Spectral norm :math:`\|M\|_2` by power iteration on :math:`M^H M`.

    A free function rather than a method because the pole conditioning needs
    :math:`\|M'(z)\|_2`, and :math:`M'` may be singular -- it must not be routed
    through a factorisation. Never densifies the operator; a few correct digits
    are all the conditioning indicator needs.

    Parameters
    ----------
    a_ii, a_ij, a_ji : list[NDArray]
        Block-tridiagonal blocks.
    n_power : int, optional
        Power iterations. Default 12.
    seed : int, optional
        Seed of the starting vector, so the result is deterministic.

    Returns
    -------
    NDArray
        ``(*stack,)`` estimate of the spectral norm.

    """
    def dag(m):
        return m.conj().swapaxes(-2, -1)

    n_dof = int(sum(int(b.shape[-1]) for b in a_ii))
    stack = a_ii[0].shape[:-2]
    rng = xp.random.default_rng(seed)
    v = (rng.standard_normal(stack + (n_dof, 1))).astype(a_ii[0].dtype)
    v /= xp.linalg.norm(v, axis=-2, keepdims=True)
    lam = xp.zeros(stack)
    for _ in range(n_power):
        w = btd_matvec(a_ii, a_ij, a_ji, v)
        u = btd_matvec([dag(m) for m in a_ii], [dag(m) for m in a_ji],
                       [dag(m) for m in a_ij], w)
        nrm = xp.linalg.norm(u, axis=-2, keepdims=True)
        lam = nrm[..., 0, 0]
        v = u / xp.where(nrm > 0, nrm, 1.0)
    return xp.sqrt(lam)


class BTDFactorization:
    """Reusable block-tridiagonal factorisation, batched over the stack axes.

    Attributes
    ----------
    block_sizes : np.ndarray
        Sizes of the diagonal blocks.
    n_dof : int
        Total dimension.

    """

    def __init__(self, a_ii, a_ij, a_ji, ipiv, elim):
        self._a_ii, self._a_ij, self._a_ji = a_ii, a_ij, a_ji
        self._ipiv = ipiv          # inv of the i-th Schur pivot
        self._elim = elim          # ipiv[i] @ a_ij[i]
        self.block_sizes = np.array([int(b.shape[-1]) for b in a_ii], dtype=int)
        self._off = _offsets(self.block_sizes)
        self.n_dof = int(self._off[-1])
        self._adjoint: BTDFactorization | None = None

    # -- construction ------------------------------------------------------ #

    @classmethod
    def factorize(
        cls, a_ii: list[NDArray], a_ij: list[NDArray], a_ji: list[NDArray]
    ) -> "BTDFactorization":
        """Eliminate downwards, storing the pivot inverses and elimination factors.

        Parameters
        ----------
        a_ii : list[NDArray]
            Diagonal blocks, ``(*stack, b_i, b_i)``.
        a_ij : list[NDArray]
            Superdiagonal blocks ``M[i, i+1]``; length ``len(a_ii) - 1``.
        a_ji : list[NDArray]
            Subdiagonal blocks ``M[i+1, i]``; length ``len(a_ii) - 1``.

        Returns
        -------
        BTDFactorization

        """
        n = len(a_ii)
        if len(a_ij) != n - 1 or len(a_ji) != n - 1:
            raise ValueError(
                f"expected {n - 1} off-diagonal blocks, got "
                f"{len(a_ij)} super and {len(a_ji)} sub."
            )
        ipiv: list[NDArray] = []
        elim: list[NDArray] = []
        piv = a_ii[0]
        for i in range(n):
            if i > 0:
                piv = a_ii[i] - a_ji[i - 1] @ elim[i - 1]
            inv = linalg.inv(piv)
            ipiv.append(inv)
            if i < n - 1:
                elim.append(inv @ a_ij[i])
        return cls(a_ii, a_ij, a_ji, ipiv, elim)

    # -- solves ------------------------------------------------------------ #

    def solve(self, b: NDArray) -> NDArray:
        r"""Solve :math:`M x = b`.

        Parameters
        ----------
        b : NDArray
            ``(*stack, n_dof, nrhs)``.

        Returns
        -------
        NDArray
            ``x``, same shape as ``b``.

        """
        if int(b.shape[-2]) != self.n_dof:
            raise ValueError(
                f"rhs has {int(b.shape[-2])} rows, operator has {self.n_dof}."
            )
        n, off = len(self._a_ii), self._off
        x = xp.empty_like(b)
        # Forward elimination.
        prev = self._ipiv[0] @ b[..., off[0]:off[1], :]
        x[..., off[0]:off[1], :] = prev
        for i in range(1, n):
            seg = slice(off[i], off[i + 1])
            prev = self._ipiv[i] @ (
                b[..., seg, :] - self._a_ji[i - 1] @ prev
            )
            x[..., seg, :] = prev
        # Backward substitution.
        for i in range(n - 2, -1, -1):
            seg = slice(off[i], off[i + 1])
            nxt = slice(off[i + 1], off[i + 2])
            x[..., seg, :] = x[..., seg, :] - self._elim[i] @ x[..., nxt, :]
        return x

    def adjoint(self) -> "BTDFactorization":
        r"""Factorisation of :math:`M^H` (built and cached on first use).

        ``(M^H)_{i,i} = M_{i,i}^H``, ``(M^H)_{i,i+1} = M_{i+1,i}^H`` and
        ``(M^H)_{i+1,i} = M_{i,i+1}^H`` -- the off-diagonals swap roles.
        """
        if self._adjoint is None:
            def dag(m):
                return m.conj().swapaxes(-2, -1)

            self._adjoint = BTDFactorization.factorize(
                [dag(m) for m in self._a_ii],
                [dag(m) for m in self._a_ji],
                [dag(m) for m in self._a_ij],
            )
        return self._adjoint

    def solve_hermitian(self, b: NDArray) -> NDArray:
        r"""Solve :math:`M^H x = b`, used for the left null vector."""
        return self.adjoint().solve(b)

    # -- operator norms ---------------------------------------------------- #

    def matvec(self, x: NDArray) -> NDArray:
        """Apply the (unfactorised) operator."""
        return btd_matvec(self._a_ii, self._a_ij, self._a_ji, x)

    def norm2(self, n_power: int = 12, seed: int = 0) -> NDArray:
        r"""Spectral norm of the operator; see :func:`btd_norm2`."""
        return btd_norm2(
            self._a_ii, self._a_ij, self._a_ji, n_power=n_power, seed=seed
        )

    # -- helpers ----------------------------------------------------------- #

    def to_dense(self) -> NDArray:
        """Dense operator; for tests and small systems only."""
        stack = self._a_ii[0].shape[:-2]
        m = xp.zeros(stack + (self.n_dof, self.n_dof), dtype=self._a_ii[0].dtype)
        off = self._off
        for i in range(len(self._a_ii)):
            m[..., off[i]:off[i + 1], off[i]:off[i + 1]] = self._a_ii[i]
            if i + 1 < len(self._a_ii):
                m[..., off[i]:off[i + 1], off[i + 1]:off[i + 2]] = self._a_ij[i]
                m[..., off[i + 1]:off[i + 2], off[i]:off[i + 1]] = self._a_ji[i]
        return m
