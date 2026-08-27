# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""How many exponentials a spatial block sequence actually needs.

:mod:`quatrex.phonon.spatial_modes` gets the exponents from the OPERATOR -- it
solves the pencil the device already defines. This module gets them from the
DATA, by fitting the sequence itself, and the difference is the point: the
bubble uses :math:`G^{<,>}`, not :math:`G^R`, and the Keldysh object is not the
resolvent of anything. Whether it is even a sum of exponentials in the
separation, and with which exponents, is a measurement.

Three things, in the order an experiment needs them.

**Rank.** For a sequence :math:`g_n = \sum_p A_p \xi_p^n` the block-Hankel
matrix factors as :math:`H = L R` with :math:`L` carrying :math:`\xi_p^i A_p`
and :math:`R` carrying :math:`\xi_p^k I`, so

.. math::
    \operatorname{rank} H = \sum_p \operatorname{rank} A_p \le r\,b .

A SCALAR sequence therefore has Hankel rank equal to the number of exponentials,
and a block sequence has that number times the residue rank -- which is not a
detail: reading a block-Hankel rank as an exponent count overstates it by the
block size. :func:`numerical_rank` reports the matrix rank at a stated
tolerance, and :func:`cluster_exponents` converts it into a count of distinct
exponents with their multiplicities.

**Exponents.** From the shifted pair :math:`(H_0, H_1)`, block-ESPRIT recovers
the :math:`\xi_p` themselves. What they turn out to be decides the production
algebra: the retarded :math:`\lambda_\mu`, the advanced conjugates
:math:`\lambda_\mu^*`, or the products :math:`\lambda_m\lambda_n^*` that a
source-resolved Keldysh derivation predicts are three different answers, and
they are distinguishable to several digits.

**Structure.** A two-index object need not be a function of the separation at
all. The source-resolved form carries a residue
:math:`(\lambda_m\lambda_n^*)^I`, which makes :math:`\Sigma_{IJ}` SEMISEPARABLE
-- :math:`\sum_a u_a\mu_a^I v_a\nu_a^J` -- rather than Toeplitz, so the
state-space ansatz :math:`\Sigma_R = CA^{R-1}B` is the wrong class and a fit to
it will fail for a reason that has nothing to do with rank.
:func:`directional_exponents` measures which of the two it is by running the
pencil along three directions, and :func:`semiseparable_fit` fits the class the
answer indicates. Nothing is lost operationally: the semiseparable class has the
same ``O(N r)`` matvec, through prefix/suffix recurrences.

Nothing here assumes a physical interpretation of the sequence, so the same code
serves :math:`G^R`, :math:`G^{<,>}`, the positivity factor :math:`Y = G^R L`,
and :math:`\Sigma` itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qttools import NDArray

__all__ = [
    "block_hankel",
    "singular_spectrum",
    "numerical_rank",
    "matrix_pencil",
    "cluster_exponents",
    "fit_residues",
    "ExponentialSeries",
    "directional_exponents",
    "semiseparable_fit",
    "Semiseparable",
]


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _as_seq(seq) -> NDArray:
    """``(n, b, b')`` from a list/array/dict of blocks; scalars are ``(n,1,1)``."""
    if isinstance(seq, dict):
        seq = [seq[k] for k in sorted(seq)]
    arr = np.asarray([np.asarray(_host(s)) for s in seq])
    if arr.ndim == 1:
        arr = arr[:, None, None]
    if arr.ndim != 3:
        raise ValueError(f"expected a sequence of blocks, got shape {arr.shape}")
    return arr


def block_hankel(seq, n_rows: int | None = None) -> NDArray:
    r"""``H[i, k] = g_{i+k}`` as one dense matrix.

    ``n_rows`` block rows; the default splits the sequence as evenly as it can,
    which maximises the rank the matrix is able to reveal. A sequence of length
    ``n`` supports at most ``floor((n+1)/2)`` rows before the columns run out.
    """
    arr = _as_seq(seq)
    n, b, bp = arr.shape
    if n < 2:
        raise ValueError(f"block_hankel: need at least 2 blocks, got {n}")
    n_rows = (n + 1) // 2 if n_rows is None else int(n_rows)
    n_cols = n - n_rows + 1
    if n_rows < 1 or n_cols < 1:
        raise ValueError(
            f"block_hankel: {n_rows} rows leaves {n_cols} columns from "
            f"{n} blocks")
    out = np.empty((n_rows * b, n_cols * bp), dtype=complex)
    for i in range(n_rows):
        for k in range(n_cols):
            out[i * b:(i + 1) * b, k * bp:(k + 1) * bp] = arr[i + k]
    return out


def singular_spectrum(seq, n_rows: int | None = None) -> NDArray:
    """Singular values of the block-Hankel matrix, normalised to the largest."""
    sv = np.linalg.svd(block_hankel(seq, n_rows), compute_uv=False)
    return sv / (sv[0] if sv[0] > 0 else 1.0)


def numerical_rank(seq, eps: float = 1e-8, n_rows: int | None = None) -> int:
    r"""``r_eps = min{r : sum_{j>r} sigma_j^2 / sum_j sigma_j^2 < eps^2}``.

    Energy-based rather than a bare threshold on ``sigma_j``, so a sequence with
    a long shallow tail is not credited with a rank it cannot support.
    """
    sv = singular_spectrum(seq, n_rows)
    energy = np.cumsum(sv[::-1] ** 2)[::-1]
    total = float(np.sum(sv ** 2))
    if total <= 0.0:
        return 0
    for r in range(sv.size + 1):
        tail = float(energy[r]) if r < sv.size else 0.0
        if tail / total < eps ** 2:
            return r
    return int(sv.size)


def cluster_exponents(xi, tol: float = 1e-6):
    r"""Group near-degenerate exponents: ``(unique, multiplicity)``.

    Block-ESPRIT returns each exponent with multiplicity ``rank(A_p)``, so a
    ``2 x 2`` sequence with four exponentials hands back eight eigenvalues in
    four coincident pairs. Reporting the raw count as "the number of
    exponentials" overstates the rank by the block size; this is what makes the
    two numbers separately readable.
    """
    xi = np.asarray(_host(xi)).ravel()
    order = np.argsort(-np.abs(xi))
    xi = xi[order]
    uniq: list[complex] = []
    mult: list[int] = []
    for z in xi:
        hit = None
        for i, u in enumerate(uniq):
            if abs(z - u) <= tol * max(1.0, abs(u)):
                hit = i
                break
        if hit is None:
            uniq.append(complex(z))
            mult.append(1)
        else:
            # running mean keeps the representative from drifting to whichever
            # member happened to come first
            uniq[hit] = (uniq[hit] * mult[hit] + z) / (mult[hit] + 1)
            mult[hit] += 1
    return np.asarray(uniq), np.asarray(mult, dtype=int)


@dataclass(frozen=True)
class ExponentialSeries:
    r"""``g_n = sum_p A_p xi_p^n`` fitted from a block sequence.

    Attributes
    ----------
    xi : NDArray
        ``(r,)`` exponents.
    residues : NDArray or None
        ``(r, b, b')`` coefficient blocks, from :func:`fit_residues`.
    spectrum : NDArray
        The normalised Hankel singular values the rank was taken from -- kept
        so a report can show WHY a rank was chosen instead of asserting it.
    """

    xi: NDArray
    residues: NDArray | None
    spectrum: NDArray

    @property
    def rank(self) -> int:
        """Number of recovered eigenvalues -- the MATRIX rank, multiplicities
        included. For the number of distinct exponents use
        :meth:`n_exponents`."""
        return int(self.xi.size)

    def n_exponents(self, tol: float = 1e-6) -> int:
        return int(cluster_exponents(self.xi, tol)[0].size)

    def block(self, n: int) -> NDArray:
        if self.residues is None:
            raise ValueError("ExponentialSeries: no residues were fitted")
        if self.residues.shape[0] == 0:
            return np.zeros(self.residues.shape[1:], dtype=complex)
        return np.tensordot(self.xi ** n, self.residues, axes=(0, 0))

    def rel_error(self, seq) -> NDArray:
        """Per-distance relative reconstruction error."""
        arr = _as_seq(seq)
        out = np.empty(arr.shape[0])
        for n in range(arr.shape[0]):
            den = np.linalg.norm(arr[n])
            out[n] = np.linalg.norm(self.block(n) - arr[n]) / (den + 1e-300)
        return out


def matrix_pencil(seq, rank: int | None = None, *, eps: float = 1e-8,
                  n_rows: int | None = None,
                  drop_below: float = 0.0) -> ExponentialSeries:
    r"""Block-ESPRIT: recover ``xi_p`` from the shift invariance of ``H_0``.

    The signal subspace ``U`` of the block-Hankel matrix satisfies
    ``U_{shift-down} = U_{shift-up} X`` with ``eig(X) = {xi_p}``; the exponents
    therefore come out of the DATA and owe nothing to an operator. ``rank``
    defaults to :func:`numerical_rank` at ``eps``.

    ``drop_below`` discards recovered exponents of modulus below it. Zero by
    default: a tiny exponent is a real (fast-decaying) component, and dropping
    it silently would misreport the rank -- which is the quantity under test.
    """
    arr = _as_seq(seq)
    n, b, _ = arr.shape
    h = block_hankel(arr, n_rows)
    u, sv, _ = np.linalg.svd(h, full_matrices=False)
    spectrum = sv / (sv[0] if sv[0] > 0 else 1.0)

    if rank is None:
        rank = numerical_rank(arr, eps, n_rows)
    rank = int(min(max(rank, 0), u.shape[1], u.shape[0] - b))
    if rank == 0:
        # numerical_rank only reaches zero on an identically-zero sequence, so
        # the fitted model is zero -- carried as an empty residue array of the
        # right shape rather than as None, which would make rel_error raise on
        # a source arm that legitimately contributes nothing here.
        return ExponentialSeries(xi=np.zeros(0, complex),
                                 residues=np.zeros((0, arr.shape[1],
                                                    arr.shape[2]), complex),
                                 spectrum=spectrum)

    us = u[:, :rank]
    xi = np.linalg.eigvals(np.linalg.pinv(us[:-b]) @ us[b:])
    if drop_below > 0.0:
        xi = xi[np.abs(xi) >= drop_below]
    order = np.argsort(-np.abs(xi))
    xi = xi[order]
    return ExponentialSeries(xi=xi, residues=fit_residues(arr, xi),
                             spectrum=spectrum)


def fit_residues(seq, xi, *, rcond=None) -> NDArray:
    r"""Least-squares ``A_p`` in ``g_n = sum_p A_p xi_p^n`` over every ``n``."""
    arr = _as_seq(seq)
    n, b, bp = arr.shape
    xi = np.asarray(_host(xi)).ravel()
    design = xi[None, :] ** np.arange(n)[:, None]          # (n, r)
    rhs = arr.reshape(n, b * bp)
    coef = np.linalg.lstsq(design, rhs, rcond=rcond)[0]     # (r, b*b')
    return coef.reshape(xi.size, b, bp)


def directional_exponents(mat, *, rank: int | None = None, eps: float = 1e-8,
                          n_dof: int = 1, anchor: int | None = None,
                          span: int | None = None) -> dict:
    r"""Run the pencil along ``J``, along ``I``, and along the diagonal.

    A two-index object ``M_{IJ}`` that is a function of ``J - I`` alone gives
    the same exponents in the first two directions and their products on the
    diagonal. One that is semiseparable -- residue ``(mu nu)^I`` -- does not,
    and the three answers separate the cases without any fitting decision.

    ``mat`` is dense ``(..., N_D, N_D)`` with ``n_dof`` per cell; a leading
    frequency axis is contracted away by taking the block Frobenius structure
    at each cell pair, so pass ONE frequency.
    """
    mat = np.asarray(_host(mat))
    if mat.ndim != 2:
        raise ValueError(
            f"directional_exponents: pass one frequency, got shape {mat.shape}")
    n_cells = mat.shape[-1] // n_dof
    anchor = n_cells // 2 if anchor is None else int(anchor)
    span = n_cells - anchor if span is None else int(span)
    span = max(2, min(span, n_cells - anchor))

    def blk(i, j):
        return mat[i * n_dof:(i + 1) * n_dof, j * n_dof:(j + 1) * n_dof]

    along_j = [blk(anchor, anchor + r) for r in range(span)]
    along_i = [blk(anchor + r, anchor) for r in range(span)]
    diag = [blk(anchor + r, anchor + r) for r in range(span)]
    return {
        "along_J": matrix_pencil(along_j, rank, eps=eps),
        "along_I": matrix_pencil(along_i, rank, eps=eps),
        "along_diag": matrix_pencil(diag, rank, eps=eps),
        "anchor": anchor, "span": span,
    }


@dataclass(frozen=True)
class Semiseparable:
    r"""``M_{IJ} = sum_a U_a mu_a^I V_a nu_a^J``, the class the Keldysh algebra
    produces.

    ``Sigma_R = C A^{R-1} B`` -- the Toeplitz/state-space ansatz -- is the
    special case ``mu_a nu_a = 1``. Fitting that class to a source-resolved
    Keldysh self-energy fails for a structural reason and not for want of rank,
    which is why the two are separate objects here.
    """

    mu: NDArray
    nu: NDArray
    coef: NDArray                     # (r_mu, r_nu, b, b')

    def block(self, i: int, j: int) -> NDArray:
        w = np.outer(self.mu ** i, self.nu ** j)
        return np.tensordot(w, self.coef, axes=((0, 1), (0, 1)))

    def rel_error(self, mat, n_dof: int = 1, cells=None) -> float:
        mat = np.asarray(_host(mat))
        n_cells = mat.shape[-1] // n_dof
        cells = range(n_cells) if cells is None else cells
        num = den = 0.0
        for i in cells:
            for j in cells:
                ref = mat[i * n_dof:(i + 1) * n_dof, j * n_dof:(j + 1) * n_dof]
                num += float(np.sum(np.abs(self.block(i, j) - ref) ** 2))
                den += float(np.sum(np.abs(ref) ** 2))
        return float(np.sqrt(num / (den + 1e-300)))


def semiseparable_fit(mat, *, n_dof: int = 1, rank: int | None = None,
                      eps: float = 1e-8, cells=None,
                      rcond=None) -> Semiseparable:
    r"""Fit ``M_{IJ} = sum_{ab} C_{ab} mu_a^I nu_b^J`` on the given cells.

    The exponents come from :func:`directional_exponents` -- ``mu`` from the
    ``I`` direction, ``nu`` from the ``J`` direction -- and only the amplitudes
    are solved, so the fit is linear and has no local minima to fall into.
    """
    mat = np.asarray(_host(mat))
    n_cells = mat.shape[-1] // n_dof
    cells = list(range(n_cells)) if cells is None else list(cells)
    d = directional_exponents(mat, rank=rank, eps=eps, n_dof=n_dof)
    mu = np.asarray(d["along_I"].xi)
    nu = np.asarray(d["along_J"].xi)
    if mu.size == 0 or nu.size == 0:
        raise ValueError("semiseparable_fit: the pencil found no exponents")

    b = n_dof
    rows, rhs = [], []
    for i in cells:
        for j in cells:
            rows.append(np.outer(mu ** i, nu ** j).ravel())
            rhs.append(mat[i * b:(i + 1) * b, j * b:(j + 1) * b].ravel())
    design = np.asarray(rows)
    coef = np.linalg.lstsq(design, np.asarray(rhs), rcond=rcond)[0]
    return Semiseparable(mu=mu, nu=nu,
                         coef=coef.reshape(mu.size, nu.size, b, b))
