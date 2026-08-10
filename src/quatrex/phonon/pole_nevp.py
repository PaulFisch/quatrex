# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Nonlinear eigenvalue solve for the poles of the phonon Green's function.

A pole of :math:`G^R(z) = M^R(z)^{-1}` is a :math:`z_\alpha` with

.. math::
    M(z_\alpha) r_\alpha = 0, \qquad l_\alpha^\dagger M(z_\alpha) = 0,

and, for a simple pole, the residue is :math:`R_\alpha = r_\alpha l_\alpha^\dagger / d_\alpha`
with :math:`d_\alpha = l_\alpha^\dagger M'(z_\alpha) r_\alpha`. Normalising
:math:`l_\alpha^\dagger M' r_\alpha = 1` makes :math:`d_\alpha = 1` and the residue
simply :math:`r_\alpha l_\alpha^\dagger`.

Two routines:

* :func:`beyn_contour` -- initialisation and periodic audit. Contour moments
  :math:`A_k = \frac{1}{2\pi i}\oint z^k M(z)^{-1} V\,dz` give the enclosed pole
  set and its subspace. This is a *new* implementation rather than a reuse of
  :class:`qttools.nevp.beyn.Beyn`: that class solves a **polynomial** pencil in
  the Bloch factor :math:`\lambda` through ``operator_inverse``, and cannot take
  a general matrix function of :math:`z`.
* :func:`bordered_newton` -- the corrector. The bordered system of the design
  note is solved by eliminating the border analytically, so each step costs two
  BTD solves against one factorisation rather than an :math:`(N+1)`-dimensional
  solve.

The operator is supplied as callables returning block-tridiagonal blocks, so the
same code serves the toy beds and the distributed production assembly, and the
expensive part (evaluating :math:`\Sigma_s^R(z)`) stays under the caller's control.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from qttools import NDArray, xp

from quatrex.phonon.btd_linalg import BTDFactorization, btd_matvec, btd_norm2

__all__ = [
    "PoleSolution",
    "bordered_newton",
    "beyn_contour",
    "residue",
    "ellipse_contour",
]

Blocks = tuple[list[NDArray], list[NDArray], list[NDArray]]
BlockFn = Callable[[complex], Blocks]


@dataclass
class PoleSolution:
    """A converged (or abandoned) simple pole.

    Attributes
    ----------
    z : complex
        Pole location (THz).
    r : NDArray
        Right null vector, ``(n_dof,)``, unit norm.
    l : NDArray
        Left null vector, ``(n_dof,)``, normalised so ``l^H M'(z) r == 1``.
    eps_nep : float
        Scaled residual, doc Eq. (51).
    kappa : float
        Conditioning indicator, doc Eq. (54). Large means the residue is
        ill-determined and the mode belongs in a cluster.
    converged : bool
        Whether ``eps_nep`` met the tolerance.
    iterations : int
        Newton steps taken.

    """

    z: complex
    r: NDArray
    l: NDArray
    eps_nep: float
    kappa: float
    converged: bool
    iterations: int

    def residue(self) -> NDArray:
        """Residue matrix ``R = r l^H`` (doc Eq. 50)."""
        return residue(self.r, self.l)


def residue(r: NDArray, l: NDArray) -> NDArray:
    r"""Residue :math:`R_\alpha = r_\alpha l_\alpha^\dagger` of a normalised pair."""
    return xp.outer(r, xp.conj(l))


def _flat_solve(fac: BTDFactorization, v: NDArray) -> NDArray:
    """Solve for a single vector, hiding the (n, nrhs) convention."""
    return fac.solve(v.reshape(-1, 1))[..., 0]


def _matvec(blocks: Blocks, v: NDArray) -> NDArray:
    out = btd_matvec(*blocks, v.reshape(-1, 1))[..., 0]
    if out.ndim > 1:
        # The bordered Newton corrects ONE pole at a time, so whatever leading
        # axes the assembled blocks carry are singleton. Drop them explicitly:
        # a bare reshape would fold a genuine stack axis into the row index and
        # return a plausible-looking vector of the wrong length.
        if int(np.prod(out.shape[:-1])) != 1:
            raise ValueError(
                f"_matvec: blocks carry a non-singleton stack axis "
                f"{tuple(out.shape[:-1])}; the bordered Newton solves one "
                "pole at a time."
            )
        out = out.reshape(-1)
    return out


def bordered_newton(
    m_blocks: BlockFn,
    dm_blocks: BlockFn,
    z0: complex,
    r0: NDArray | None = None,
    *,
    tol: float = 1e-10,
    max_iter: int = 8,
    trust_radius: float = np.inf,
    norm_power: int = 12,
) -> PoleSolution:
    r"""Correct a predicted pole with the bordered Newton iteration.

    The bordered system

    .. math::
        \begin{bmatrix} M(z) & M'(z) r \\ c^\dagger & 0 \end{bmatrix}
        \begin{bmatrix} \delta r \\ \delta z \end{bmatrix}
        = -\begin{bmatrix} M(z) r \\ c^\dagger r - 1 \end{bmatrix}

    is solved by eliminating the border: with :math:`x_1 = M^{-1}(Mr)` and
    :math:`x_2 = M^{-1}(M'r)`,

    .. math::
        \delta z = \frac{c^\dagger r - 1 - c^\dagger x_1}{c^\dagger x_2},
        \qquad \delta r = -x_1 - \delta z\, x_2 .

    Two solves against one factorisation per step, instead of an
    :math:`(N+1)`-dimensional solve.

    Parameters
    ----------
    m_blocks : callable
        ``z -> (a_ii, a_ij, a_ji)`` of ``M(z)``.
    dm_blocks : callable
        ``z -> (a_ii, a_ij, a_ji)`` of ``M'(z)``.
    z0 : complex
        Predicted pole.
    r0 : NDArray, optional
        Predicted right vector; a random unit vector if omitted.
    tol : float, optional
        Acceptance threshold on ``eps_nep`` (doc Eq. 52). Default 1e-10.
    max_iter : int, optional
        Newton steps. Default 8.
    trust_radius : float, optional
        Cap on ``|delta z|`` per step (THz). The caller passes
        ``trust_radius_cells * h`` so a step never leaves the window in which
        the local model of ``Sigma^R(z)`` is valid.
    norm_power : int, optional
        Power iterations for the operator norms in ``eps_nep`` and ``kappa``.

    Returns
    -------
    PoleSolution

    """
    z = complex(z0)
    blocks = m_blocks(z)
    n_dof = int(sum(int(b.shape[-1]) for b in blocks[0]))

    if r0 is None:
        rng = xp.random.default_rng(0)
        r = rng.standard_normal(n_dof) + 1j * rng.standard_normal(n_dof)
    else:
        r = xp.asarray(r0, dtype=xp.complex128).reshape(-1)
    r = r / xp.linalg.norm(r)
    c = xp.array(r, copy=True)

    it = 0
    for it in range(1, max_iter + 1):
        blocks = m_blocks(z)
        dblocks = dm_blocks(z)
        fac = BTDFactorization.factorize(*blocks)

        x1 = _flat_solve(fac, _matvec(blocks, r))
        x2 = _flat_solve(fac, _matvec(dblocks, r))

        denom = complex(xp.vdot(c, x2))
        if denom == 0.0:
            break
        dz = (complex(xp.vdot(c, r)) - 1.0 - complex(xp.vdot(c, x1))) / denom
        if abs(dz) > trust_radius:
            dz *= trust_radius / abs(dz)
        dr = -x1 - dz * x2

        z += dz
        # NOTE: r is deliberately NOT renormalised here. The gauge condition
        # c^H r = 1 is what fixes the scale of the null vector and closes the
        # bordered system; rescaling r inside the loop violates it and the
        # iteration stops converging.
        r = r + dr
        if abs(dz) < 1e-15 * max(1.0, abs(z)):
            break

    r = r / xp.linalg.norm(r)

    # Final residual and the left partner at the accepted z.
    blocks = m_blocks(z)
    dblocks = dm_blocks(z)
    fac = BTDFactorization.factorize(*blocks)
    m_norm = float(xp.asarray(fac.norm2(n_power=norm_power)).reshape(-1)[0])
    resid = float(xp.linalg.norm(_matvec(blocks, r)))
    scale = abs(z) ** 2 + m_norm
    eps_nep = resid / (scale * float(xp.linalg.norm(r)))

    # Left null vector by one step of inverse iteration on M^H, then the
    # normalisation l^H M' r = 1 that makes d_alpha = 1 (doc Eqs. 49-50).
    l = fac.solve_hermitian(c.reshape(-1, 1))[..., 0]
    l = l / xp.linalg.norm(l)
    d = complex(xp.vdot(l, _matvec(dblocks, r)))
    if d != 0.0:
        l = l / np.conj(d)

    # M'(z) is routinely singular (for the phonon operator it is essentially
    # 2z*I plus lead derivatives), so take its norm without factorising it.
    dm_norm = float(xp.asarray(btd_norm2(*dblocks, n_power=norm_power)).reshape(-1)[0])
    kappa = float(xp.linalg.norm(l) * xp.linalg.norm(r) * dm_norm)

    return PoleSolution(
        z=z, r=r, l=l, eps_nep=eps_nep, kappa=kappa,
        converged=bool(eps_nep < tol), iterations=it,
    )


def ellipse_contour(
    centre: complex, semi_re: float, semi_im: float, n_quad: int = 32
) -> tuple[NDArray, NDArray]:
    r"""Quadrature nodes and weights of an elliptical contour.

    The contour must enclose the sought poles and avoid the real axis, on which
    the continuation of :math:`\Sigma_s^R` has its branch cut. A trapezoidal rule
    on a smooth closed curve is spectrally accurate.

    Parameters
    ----------
    centre : complex
        Ellipse centre; normally ``Omega - 1j*gamma_mid`` in the lower half plane.
    semi_re, semi_im : float
        Semi-axes along the real and imaginary directions (THz).
    n_quad : int, optional
        Number of nodes. Default 32.

    Returns
    -------
    nodes : NDArray
        ``(n_quad,)`` complex quadrature points.
    weights : NDArray
        ``(n_quad,)`` values of ``z'(t) dt / (2 pi i)``.

    """
    t = xp.arange(n_quad) * (2.0 * np.pi / n_quad)
    nodes = centre + semi_re * xp.cos(t) + 1j * semi_im * xp.sin(t)
    dz = -semi_re * xp.sin(t) + 1j * semi_im * xp.cos(t)
    weights = dz * (2.0 * np.pi / n_quad) / (2.0 * np.pi * 1j)
    return nodes, weights


def beyn_contour(
    m_blocks: BlockFn,
    nodes: NDArray,
    weights: NDArray,
    *,
    n_probe: int = 8,
    rank_tol: float = 1e-8,
    seed: int = 42,
) -> tuple[NDArray, NDArray]:
    r"""Beyn's contour method for the poles enclosed by a contour.

    Forms the moments :math:`A_k = \frac{1}{2\pi i}\oint z^k M(z)^{-1} V\,dz`
    for :math:`k = 0, 1` with a random probing matrix :math:`V`, and reads the
    enclosed poles off the reduced pencil. Used for initialisation, for periodic
    audit, and as the recovery path when the predictor/corrector loses a pole.

    Parameters
    ----------
    m_blocks : callable
        ``z -> (a_ii, a_ij, a_ji)`` of ``M(z)``.
    nodes, weights : NDArray
        From :func:`ellipse_contour`.
    n_probe : int, optional
        Columns of the probing matrix; must exceed the number of enclosed
        poles. Default 8.
    rank_tol : float, optional
        Relative singular-value threshold for the numerical rank. Default 1e-8.
    seed : int, optional
        Probing-matrix seed. Fixed so the SCBA map stays a deterministic
        function of the self-energy.

    Returns
    -------
    z : NDArray
        ``(m,)`` enclosed poles.
    r : NDArray
        ``(n_dof, m)`` right vectors, unit norm.

    """
    blocks0 = m_blocks(complex(nodes[0]))
    n_dof = int(sum(int(b.shape[-1]) for b in blocks0[0]))

    rng = xp.random.default_rng(seed)
    v = rng.standard_normal((n_dof, n_probe)) + 1j * rng.standard_normal(
        (n_dof, n_probe)
    )

    a0 = xp.zeros((n_dof, n_probe), dtype=xp.complex128)
    a1 = xp.zeros((n_dof, n_probe), dtype=xp.complex128)
    y_scale = 0.0
    for zk, wk in zip(nodes, weights):
        fac = BTDFactorization.factorize(*m_blocks(complex(zk)))
        y = fac.solve(v)
        y_scale = max(y_scale, float(xp.linalg.norm(y)))
        a0 += wk * y
        a1 += wk * complex(zk) * y

    u, s, vh = xp.linalg.svd(a0, full_matrices=False)
    # An ABSOLUTE floor is essential: with no pole enclosed, A_0 is zero up to
    # quadrature error, and a purely relative test against s[0] would promote
    # that noise to a full set of spurious poles.
    floor = rank_tol * max(float(s[0]), y_scale)
    m = int(xp.sum(s > floor)) if float(s[0]) > 0 else 0
    if m == 0:
        return xp.zeros(0, dtype=xp.complex128), xp.zeros((n_dof, 0), dtype=xp.complex128)
    if m == n_probe:
        # The probing matrix saturated: there may be more poles than columns.
        import warnings

        warnings.warn(
            f"beyn_contour: numerical rank saturated the probing matrix "
            f"({m} == n_probe); poles may be missing. Increase n_probe.",
            stacklevel=2,
        )

    u, s, vh = u[:, :m], s[:m], vh[:m]
    b = u.conj().T @ a1 @ vh.conj().T / s
    lam, w = xp.linalg.eig(b)
    r = u @ w
    r = r / xp.linalg.norm(r, axis=0, keepdims=True)
    return lam, r
