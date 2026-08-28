# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Bidirectional semiseparable operators, and the directions they need.

Two things the spatial-modal layer was missing, and one measured failure that
follows from both.

**Direction.** :mod:`quatrex.phonon.spatial_modes` classifies a mode by
:math:`|\lambda| < 1` and nothing else; ``decay_lengths`` returns ``nan`` for the
growing partner, on the reasoning that its range "describes the opposite
direction". That is right, and it is exactly the information a two-terminal
device needs and that module throws away. In a finite device the field at an
interior cell carries waves injected from BOTH contacts, and relative to one
anchor the far-contact contribution appears with :math:`|\lambda| > 1`.
Measured on the converged chain, that branch is **26-30 % of the fitted residue
weight** (``phonon/docs/spatial_representation.md`` §11): dropping it loses a
quarter of the amplitude, and continuing it outward diverges.

The repair is not a better fit. It is to stop writing the interior as one
outward sequence:

.. math::
    G_{ij} =
    \begin{cases}
      D_i, & i = j,\\
      U^+_i A^+_{i-1}\cdots A^+_{j+1} V^+_j, & i > j,\\
      U^-_i A^-_{i+1}\cdots A^-_{j-1} V^-_j, & i < j,
    \end{cases}

with the :math:`+` family decaying rightward and the :math:`-` family leftward,
so **no mode is ever propagated in its growing direction**.

**An operator, rather than a sampler.**
:class:`quatrex.phonon.spatial_hankel.Semiseparable` can evaluate a block and
nothing else; its docstring advertises an ``O(N r)`` matvec "through
prefix/suffix recurrences" that does not exist anywhere in the tree.
:class:`SemiSepOperator` implements it:

.. math::
    s_{i+1} = A^+_i s_i + V^+_i x_i, \qquad y^+_i = U^+_i s_i,

and the mirror recurrence downward for the other direction. One application
costs ``O(N r)`` with the diagonal generators a modal representation produces,
against ``O(N^2 b^2)`` for the dense matrix it replaces, and it never forms a
long-range block.

Generators here are DIAGONAL. That is not a restriction in practice: a modal
representation gives :math:`A = \operatorname{diag}(\lambda)` directly, and a
direct sum of diagonal generators -- which is how independent contributions are
accumulated exactly -- is diagonal again. Dense :math:`A_i` would be needed only
for a genuinely inhomogeneous interior, and the interior is where this
representation is used precisely because it is not.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from qttools import NDArray

__all__ = [
    "DirectionalModes",
    "directional_modes",
    "SemiSepOperator",
    "fit_bidirectional",
]


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


@dataclass(frozen=True)
class DirectionalModes:
    r"""The two stable families of a spatial pencil.

    Both are stored as DECAYING factors in their own direction: ``lam_minus``
    holds :math:`1/\lambda` for the roots outside the unit circle, so
    ``lam_minus ** (j - i)`` decays leftward exactly as ``lam_plus ** (i - j)``
    decays rightward. Nothing in this class ever holds a factor of modulus
    greater than one, which is the whole point -- a representation that stores
    the growing partner invites it being propagated.
    """

    lam_plus: NDArray
    vec_plus: NDArray
    lam_minus: NDArray
    vec_minus: NDArray
    residual_plus: NDArray | None = None
    residual_minus: NDArray | None = None
    n_unit: int = 0

    @property
    def rank_plus(self) -> int:
        return int(self.lam_plus.size)

    @property
    def rank_minus(self) -> int:
        return int(self.lam_minus.size)


def directional_modes(a_blocks, nevp=None, *, unit_tol: float = 1e-9,
                      residual_tol: float | None = None,
                      eta_select: float = 1e-6) -> DirectionalModes:
    r"""Split a spatial pencil's roots into rightward- and leftward-decaying.

    ``a_blocks`` is the coefficient tuple of
    :func:`quatrex.phonon.spatial_modes.bloch_modes_poly`, lowest power first.

    Under SCBA dressing no root sits on the unit circle, so :math:`|\lambda|<1`
    and :math:`|\lambda|>1` is a complete and unambiguous split and no
    heuristic is needed. Undressed, an in-band mode has :math:`|\lambda| = 1`
    exactly and the modulus says nothing.

    The rule there is an **infinitesimal retarded damping**: perturb the
    zeroth coefficient block by :math:`+i\eta`, which is
    :math:`z^2 \to (\omega + i\eta)^2`, and keep the root that moves inside the
    unit circle. That is the limit the retarded Green function IS, and it is
    exact -- checked on a ballistic two-terminal chain, where the true ratio
    ``G[i,i-2]/G[i,i-1]`` is reproduced to four digits at every in-band
    frequency.

    A group-velocity partition is the other available rule and it is NOT used,
    because the obvious transcription of it is wrong here.
    ``Spectral._find_reflected_modes`` selects modes travelling INTO the lead,
    the opposite sense from the tail of ``G`` on one side of a source, so
    copying its ``Re dE/dk < 0`` test picks the complex conjugate at every
    in-band frequency -- a factor of the right modulus and the wrong phase,
    which reproduces a plausible decaying tail and is 150 % wrong.

    The count of unit-circle modes is reported as ``n_unit`` so a caller can
    see when the rule was needed at all.

    ``residual_tol`` masks roots whose NEVP residual is above it. Off by
    default: the mode COUNT is itself a diagnostic (a degree-``2M`` pencil of
    block size ``b`` has ``2Mb`` roots, ``Mb`` per direction), and silently
    returning fewer destroys it.
    """
    from quatrex.phonon.spatial_modes import bloch_modes_poly

    modes = bloch_modes_poly(a_blocks, nevp=nevp, residual=True)
    lam = np.asarray(_host(modes.lam)).ravel()
    vecs = np.asarray(_host(modes.vecs))
    if vecs.ndim > 2:
        vecs = vecs.reshape(vecs.shape[-2], -1)
    res = np.asarray(_host(modes.residual)).ravel()

    mod = np.abs(lam)
    on_unit = np.abs(mod - 1.0) <= unit_tol
    plus = mod < 1.0 - unit_tol
    minus = mod > 1.0 + unit_tol

    if on_unit.any() and eta_select > 0.0:
        blocks = [np.asarray(_host(a)) for a in a_blocks]
        mid = len(blocks) // 2
        scale = max(float(np.abs(b).max()) for b in blocks)
        shifted = list(blocks)
        shifted[mid] = (blocks[mid]
                        + 1j * eta_select * scale * np.eye(blocks[mid].shape[-1]))
        lam_eta = np.asarray(_host(
            bloch_modes_poly(tuple(shifted), nevp=nevp).lam)).ravel()
        # keep the EXACT roots; only the direction is read off the perturbed
        # ones, matched by proximity.
        inside = np.zeros(lam.shape, dtype=bool)
        for k in np.where(on_unit)[0]:
            j = int(np.argmin(np.abs(lam_eta - lam[k])))
            inside[k] = np.abs(lam_eta[j]) < 1.0
        plus = plus | (on_unit & inside)
        minus = minus | (on_unit & ~inside)

    keep_p, keep_m = plus, minus
    if residual_tol is not None:
        keep_p = keep_p & (res < residual_tol)
        keep_m = keep_m & (res < residual_tol)

    lam_minus = np.where(np.abs(lam[keep_m]) > 0.0,
                         1.0 / np.where(lam[keep_m] == 0.0, 1.0, lam[keep_m]),
                         0.0)
    return DirectionalModes(
        lam_plus=lam[keep_p], vec_plus=vecs[:, keep_p],
        lam_minus=lam_minus, vec_minus=vecs[:, keep_m],
        residual_plus=res[keep_p], residual_minus=res[keep_m],
        n_unit=int(on_unit.sum()))


def _bcast(arr, n_cells: int, name: str) -> NDArray:
    """Accept a per-cell array or one block broadcast over every cell."""
    a = np.asarray(_host(arr))
    if a.ndim == 2:
        return np.broadcast_to(a[None], (n_cells,) + a.shape).copy()
    if a.shape[0] != n_cells:
        raise ValueError(
            f"SemiSepOperator: {name} has {a.shape[0]} cells, expected "
            f"{n_cells}")
    return a


def _sss_realisation(mat, n_dof: int, rank=None, tol: float = 1e-10):
    r"""Sequentially semiseparable realisation of a strictly-lower triangle.

    For each split ``k`` the corner ``H_k = M[k:, :k]`` has the quasiseparable
    rank by definition. With ``Q_k`` an orthonormal basis of its ROW space and
    the state ``s_k = Q_k x_{<k}``,

    .. math::
        s_{k+1} = A_k s_k + B_k x_k, \qquad y_k = C_k s_k,

    with ``A_k = Q_{k+1}[:, :kb] Q_k^H`` (valid because the row space of
    ``H_{k+1}`` restricted to the first ``kb`` columns is contained in that of
    ``H_k``), ``B_k = Q_{k+1}[:, kb:]`` and ``C_k = M[k, :k] Q_k^H``. Exact when
    every ``r_k`` is kept; truncating them is the compression.

    Returns ``(A, B, C)`` padded to one common rank so the caller has a fixed
    state size, which is what an augmented Dyson block would need.
    """
    mat = np.asarray(_host(mat))
    n = mat.shape[-1] // n_dof
    bases, ranks = [None], [0]
    for k in range(1, n):
        h = mat[k * n_dof:, :k * n_dof]
        u, sv, vh = np.linalg.svd(h, full_matrices=False)
        keep = (int(rank) if rank is not None
                else int(np.sum(sv > tol * (sv[0] if sv.size and sv[0] > 0
                                            else 1.0))))
        keep = max(0, min(keep, vh.shape[0]))
        bases.append(vh[:keep])
        ranks.append(keep)
    r = max(ranks) if ranks else 0

    a = np.zeros((n, r, r), dtype=complex)
    b = np.zeros((n, r, n_dof), dtype=complex)
    c = np.zeros((n, n_dof, r), dtype=complex)
    for k in range(n):
        if ranks[k]:
            c[k, :, :ranks[k]] = mat[k * n_dof:(k + 1) * n_dof,
                                     :k * n_dof] @ bases[k].conj().T
        if k + 1 < n and ranks[k + 1]:
            q_next = bases[k + 1]
            if ranks[k]:
                a[k, :ranks[k + 1], :ranks[k]] = (
                    q_next[:, :k * n_dof] @ bases[k].conj().T)
            b[k, :ranks[k + 1], :] = q_next[:, k * n_dof:(k + 1) * n_dof]
    return a, b, c


@dataclass
class SemiSepOperator:
    r"""``M_ij`` block-diagonal plus two decaying directional tails.

    Attributes
    ----------
    diag : NDArray
        ``(N, b, b)`` explicit near-field blocks, or one block broadcast.
    lam_plus, lam_minus : NDArray
        ``(r+,)`` and ``(r-,)`` DECAYING factors, one per direction.
    u_plus, v_plus : NDArray
        ``(N, b, r+)`` and ``(N, r+, b)``. ``M_ij = U^+_i diag(lam^{i-j-1}) V^+_j``
        for ``i > j``.
    u_minus, v_minus : NDArray
        The same leftward: ``M_ij = U^-_i diag(lam^{j-i-1}) V^-_j`` for ``i < j``.

    The exponent is ``i - j - 1`` and not ``i - j``: the generator product
    ``A_{i-1} ... A_{j+1}`` has ``i - j - 1`` factors, so the nearest
    off-diagonal ``j = i - 1`` carries none. Getting this off by one still
    reproduces a decaying tail and is wrong by one factor of ``lambda``
    everywhere, which is why the block sampler and the matvec are tested
    against each other and both against a dense matrix.
    """

    n_cells: int
    n_dof: int
    diag: NDArray
    a_plus: NDArray      # (N, r+, r+)
    b_plus: NDArray      # (N, r+, b)
    c_plus: NDArray      # (N, b, r+)
    a_minus: NDArray     # (N, r-, r-)
    b_minus: NDArray     # (N, r-, b)
    c_minus: NDArray     # (N, b, r-)

    @classmethod
    def from_dense(cls, mat, n_dof: int, *, rank=None, tol: float = 1e-10):
        """Exact realisation of any matrix at its quasiseparable rank.

        Both triangles by :func:`_sss_realisation`; the upper one is obtained by
        flipping the cell order, so one algorithm serves both directions and a
        sign or transpose slip cannot affect only one of them.
        """
        mat = np.asarray(_host(mat))
        n = mat.shape[-1] // n_dof
        diag = np.stack([mat[i * n_dof:(i + 1) * n_dof,
                             i * n_dof:(i + 1) * n_dof] for i in range(n)])
        ap, bp, cp = _sss_realisation(mat, n_dof, rank=rank, tol=tol)

        perm = np.arange(n * n_dof).reshape(n, n_dof)[::-1].ravel()
        am, bm, cm = _sss_realisation(mat[np.ix_(perm, perm)], n_dof,
                                      rank=rank, tol=tol)
        return cls(n_cells=n, n_dof=n_dof, diag=diag,
                   a_plus=ap, b_plus=bp, c_plus=cp,
                   a_minus=am[::-1], b_minus=bm[::-1], c_minus=cm[::-1])

    @classmethod
    def from_modal(cls, diag, modes: DirectionalModes, v_plus, v_minus, *,
                   n_cells: int):
        """The homogeneous diagonal case, from a mode set and coefficient rows.

        Kept because it is the representation the pencil hands over directly,
        and because comparing it against :meth:`from_dense` on the same bed is
        what shows a homogeneous interior is not enough for a reflecting
        device.
        """
        diag = _bcast(np.asarray(_host(diag)), n_cells, "diag")
        b = diag.shape[-1]

        def gen(lam, u, v):
            r = int(np.asarray(lam).size)
            a = np.broadcast_to(np.diag(np.asarray(lam).astype(complex)),
                                (n_cells, r, r)).copy()
            return (a, _bcast(np.asarray(_host(v)), n_cells, "v"),
                    _bcast(np.asarray(_host(u)), n_cells, "u"))

        ap, bp, cp = gen(modes.lam_plus, modes.vec_plus, v_plus)
        am, bm, cm = gen(modes.lam_minus, modes.vec_minus, v_minus)
        return cls(n_cells=n_cells, n_dof=b, diag=diag,
                   a_plus=ap, b_plus=bp, c_plus=cp,
                   a_minus=am, b_minus=bm, c_minus=cm)

    @property
    def rank(self) -> tuple[int, int]:
        return int(self.a_plus.shape[-1]), int(self.a_minus.shape[-1])

    @property
    def augmented_block(self) -> int:
        """``d + r+ + r-`` -- the local block an augmented Dyson would carry.

        The number to compare against ``m*d`` for an ``m``-cell reblock, which
        is the incumbent this representation has to beat.
        """
        return self.n_dof + sum(self.rank)

    # -- evaluation ------------------------------------------------------- #

    def sample_block(self, i: int, j: int) -> NDArray:
        r"""``M_ij`` by the generator product, without forming the matrix.

        ``C_i A_{i-1} ... A_{j+1} B_j`` for ``i > j``: the product has
        ``i - j - 1`` factors, so the nearest off-diagonal carries none.
        """
        if i == j:
            return self.diag[i]
        # The order is not symmetric and getting it backwards still produces a
        # decaying tail: `apply` unrolls to A_{i-1} ... A_{j+1} B_j, so B_j is
        # innermost and the factors are applied from the SOURCE outwards --
        # ascending for the rightward tail, descending for the leftward one.
        if i > j:
            a, b, c = self.a_plus, self.b_plus, self.c_plus
            rng = range(j + 1, i)
        else:
            a, b, c = self.a_minus, self.b_minus, self.c_minus
            rng = range(j - 1, i, -1)
        m = b[j]
        for k in rng:
            m = a[k] @ m
        return c[i] @ m

    def to_dense(self) -> NDArray:
        n, b = self.n_cells, self.n_dof
        out = np.zeros((n * b, n * b), dtype=complex)
        for i in range(n):
            for j in range(n):
                out[i * b:(i + 1) * b, j * b:(j + 1) * b] = self.sample_block(i, j)
        return out

    def apply(self, x) -> NDArray:
        r"""``y = M x`` by two recurrences, without forming a single tail block.

        ``s_{i+1} = A_i s_i + B_i x_i``, ``y_i += C_i s_i`` forward, and the
        mirror downward. Cost ``O(N r^2 k)`` against ``O(N^2 b^2 k)`` dense, and
        no long-range block is ever materialised.
        """
        x = np.asarray(_host(x))
        flat = x.ndim == 1
        if flat:
            x = x[:, None]
        n, b = self.n_cells, self.n_dof
        if x.shape[0] != n * b:
            raise ValueError(
                f"SemiSepOperator.apply: x has {x.shape[0]} rows, expected "
                f"{n * b}")
        xb = x.reshape(n, b, x.shape[1])
        y = np.einsum("nij,njk->nik", self.diag, xb)

        rp = self.rank[0]
        if rp:
            s = np.zeros((rp, x.shape[1]), dtype=complex)
            for i in range(n):
                y[i] += self.c_plus[i] @ s
                s = self.a_plus[i] @ s + self.b_plus[i] @ xb[i]
        rm = self.rank[1]
        if rm:
            t = np.zeros((rm, x.shape[1]), dtype=complex)
            for i in range(n - 1, -1, -1):
                y[i] += self.c_minus[i] @ t
                t = self.a_minus[i] @ t + self.b_minus[i] @ xb[i]

        y = y.reshape(n * b, x.shape[1])
        return y[:, 0] if flat else y

    # -- algebra ---------------------------------------------------------- #

    def direct_sum(self, other: "SemiSepOperator") -> "SemiSepOperator":
        """Exact sum of two operators, by block-diagonalising the generators.

        The proposal's Eq. (21). Exact, and it grows the rank by the sum --
        which is why an accumulation has to be recompressed rather than only
        appended.
        """
        if (self.n_cells, self.n_dof) != (other.n_cells, other.n_dof):
            raise ValueError("direct_sum: operators have different shapes")

        def blkdiag(p, q):
            n, r1, r2 = p.shape[0], p.shape[-1], q.shape[-1]
            out = np.zeros((n, r1 + r2, r1 + r2), dtype=complex)
            out[:, :r1, :r1] = p
            out[:, r1:, r1:] = q
            return out

        return SemiSepOperator(
            n_cells=self.n_cells, n_dof=self.n_dof,
            diag=self.diag + other.diag,
            a_plus=blkdiag(self.a_plus, other.a_plus),
            b_plus=np.concatenate([self.b_plus, other.b_plus], axis=1),
            c_plus=np.concatenate([self.c_plus, other.c_plus], axis=2),
            a_minus=blkdiag(self.a_minus, other.a_minus),
            b_minus=np.concatenate([self.b_minus, other.b_minus], axis=1),
            c_minus=np.concatenate([self.c_minus, other.c_minus], axis=2))

    def compress(self, rank=None, tol: float = 1e-10) -> "SemiSepOperator":
        """Re-realise at a smaller state size.

        Goes through the dense matrix, so it is a reference implementation and
        not the production route -- an in-place balanced truncation on the
        generators is what a streaming accumulation would need, and it is only
        worth writing once the rank is known to saturate.
        """
        return SemiSepOperator.from_dense(self.to_dense(), self.n_dof,
                                          rank=rank, tol=tol)
