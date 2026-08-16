# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Nonlinear eigenvalue solve for the poles of the phonon Green's function.

A pole of :math:`G^R(z) = M^R(z)^{-1}` is a :math:`z_\alpha` with

.. math::
    M(z_\alpha) r_\alpha = 0, \qquad l_\alpha^\dagger M(z_\alpha) = 0,

and, for a simple pole, the residue is :math:`R_\alpha = r_\alpha l_\alpha^\dagger / d_\alpha`
with :math:`d_\alpha = l_\alpha^\dagger M'(z_\alpha) r_\alpha`. Normalising
:math:`l_\alpha^\dagger M' r_\alpha = 1` makes :math:`d_\alpha = 1` and the residue
simply :math:`r_\alpha l_\alpha^\dagger`.

:func:`bordered_newton` is the corrector. The bordered system is solved by
eliminating the border analytically, so each step costs two BTD solves against
one factorisation rather than an :math:`(N+1)`-dimensional solve. Seeds come
from the harmonic re-seed in :func:`~quatrex.phonon.pole_sector.refresh_many`;
there is no contour initialiser.

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
    "PoleSolutions",
    "bordered_newton",
    "bordered_newton_batch",
    "residue",
]

Blocks = tuple[list[NDArray], list[NDArray], list[NDArray]]
BlockFn = Callable[[complex], Blocks]


@dataclass
class PoleSolution:
    """A converged (or abandoned) simple pole.

    Attributes
    ----------
    z : complex
        Pole location (THz), ``z = omega_pole - i * gamma_hwhm`` under the
        ``e^{-i omega t}`` convention this package uses throughout. See
        :attr:`gamma_hwhm` and :attr:`is_passive`.
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
    eps_left : float
        ``||M^H l|| / ||M||``: how well the left vector annihilates ``M^H``.
        ``M(z_alpha)`` is singular at the pole, so the adjoint inverse
        iteration that produces ``l`` becomes worse conditioned the better the
        pole solve gets. This reports that rather than leaving it implicit --
        a large value means the residue ``R = r l^H`` is unreliable even
        though ``eps_nep`` looks fine.
    dz_est : complex
        Estimated remaining FREQUENCY error, ``-l^H M(z) r`` under the
        normalisation ``l^H M'(z) r = 1``, in THz.

        This is the number the physics cares about, and it is not
        ``eps_nep``. That is a scaled matrix residual whose denominator
        ``|z|^2 + ||M||`` runs to ``1e3-1e4 THz^2`` for a phonon operator, so
        a candidate refused at ``eps_nep = 1e-9`` can be sitting within
        ``1e-5`` of its own linewidth. On the CNT bed that gate refused 142 of
        144 candidates. The caller divides this by the smallest scale the pole
        has to be resolved against -- its width, its separation from the next
        pole, the local cell -- to get the acceptance metric.

    """

    z: complex
    r: NDArray
    l: NDArray
    eps_nep: float
    kappa: float
    converged: bool
    iterations: int
    eps_left: float = float("nan")
    dz_est: complex = 0.0 + 0.0j

    @property
    def omega_pole(self) -> float:
        """Resonance frequency ``Re z`` (THz)."""
        return float(self.z.real)

    @property
    def gamma_hwhm(self) -> float:
        r"""HALF width at half maximum, ``-Im z`` (THz).

        Named because the tree carries both conventions in prose and the
        factor of two between them lands in every resolution gate: the exact
        line-weight error is a function of ``h/gamma`` with ``gamma`` the HALF
        width (:meth:`~quatrex.phonon.pole_sector.PoleSector.leg_weight_error`),
        and using the full width there would misjudge a line by the same factor
        of two in the argument of a function that varies exponentially.

        Positive for a passive resonance. A negative value means the root came
        back on the wrong half plane and is a continuation failure, not a mode.
        """
        return -float(self.z.imag)

    @property
    def Gamma_fwhm(self) -> float:
        """FULL width at half maximum, ``-2 Im z`` (THz). Twice
        :attr:`gamma_hwhm`, and never what the gates take."""
        return -2.0 * float(self.z.imag)

    @property
    def is_passive(self) -> bool:
        r"""Whether the pole sits in the lower half plane.

        The time convention is :math:`e^{-i\omega t}`, so a mode that decays
        carries :math:`z = \Omega - i\gamma` with :math:`\gamma > 0`. Under the
        opposite convention every sign here flips and the retarded function
        would be built from the wrong half plane.
        """
        return self.z.imag < 0.0

    def residue(self) -> NDArray:
        """Residue matrix ``R = r l^H`` (doc Eq. 50)."""
        return residue(self.r, self.l)


def residue(r: NDArray, l: NDArray) -> NDArray:
    r"""Residue :math:`R_\alpha = r_\alpha l_\alpha^\dagger` of a normalised pair."""
    return xp.outer(r, xp.conj(l))


def _adjoint_blocks(blocks: Blocks) -> Blocks:
    """Blocks of ``M^H`` from those of ``M``: conjugate-transpose each block
    and swap the super- and sub-diagonals."""
    a_ii, a_ij, a_ji = blocks
    h = lambda b: xp.conj(xp.swapaxes(b, -1, -2))
    return ([h(b) for b in a_ii], [h(b) for b in a_ji], [h(b) for b in a_ij])


def _matvec(blocks: Blocks, v: NDArray) -> NDArray:
    """``M @ v`` for ``v`` of shape ``(*batch, n_dof)``.

    The batch axis is the CANDIDATE axis: the bordered Newton corrects a whole
    seed set at once, so the blocks and the vectors carry one leading axis each
    and this is a single batched GEMM per block.
    """
    return btd_matvec(*blocks, v[..., None])[..., 0]


def _vdot(a: NDArray, b: NDArray) -> NDArray:
    r""":math:`a^\dagger b` along the last axis, batched.

    NOT bit-identical to ``xp.vdot``: BLAS ``zdotc`` and a pairwise sum
    accumulate in different orders and differ in the last ulps (measured ~1e-14
    relative). That is far below every threshold the pole sector gates on --
    ``newton_tol`` is 1e-10 on a scaled residual and ``locate_tol`` is 5 % of a
    linewidth -- and Newton is contracting, so it cannot move the fixed point
    further than roundoff.
    """
    return xp.sum(xp.conj(a) * b, axis=-1)


@dataclass
class PoleSolutions:
    """A batch of :class:`PoleSolution`, held as arrays.

    The bordered Newton solves the whole seed set at once, so its natural
    output is one array per field with a leading candidate axis. Turning that
    into per-candidate objects costs one device-to-host transfer of the small
    scalar fields, done in :meth:`to_list`, instead of the several synchronising
    ``float(...)`` calls the per-candidate solve used to make.
    """

    z: NDArray
    r: NDArray
    l: NDArray
    eps_nep: NDArray
    kappa: NDArray
    converged: NDArray
    iterations: NDArray
    eps_left: NDArray
    dz_est: NDArray

    def __len__(self) -> int:
        return int(self.z.shape[0])

    def to_list(self) -> list[PoleSolution]:
        """One :class:`PoleSolution` per candidate."""
        host = {name: np.asarray(_host(getattr(self, name)))
                for name in ("z", "eps_nep", "kappa", "converged",
                             "iterations", "eps_left", "dz_est")}
        return [
            PoleSolution(
                z=complex(host["z"][k]), r=self.r[k], l=self.l[k],
                eps_nep=float(host["eps_nep"][k]),
                kappa=float(host["kappa"][k]),
                converged=bool(host["converged"][k]),
                iterations=int(host["iterations"][k]),
                eps_left=float(host["eps_left"][k]),
                dz_est=complex(host["dz_est"][k]),
            )
            for k in range(len(self))
        ]


def _host(a):
    return a.get() if hasattr(a, "get") else a


def bordered_newton_batch(
    m_blocks: BlockFn,
    dm_blocks: BlockFn,
    z0: NDArray,
    r0: NDArray | None = None,
    *,
    tol: float = 1e-10,
    max_iter: int = 8,
    trust_radius=np.inf,
    norm_power: int = 12,
    left_iters: int = 3,
) -> PoleSolutions:
    r"""Correct a whole seed set with the bordered Newton iteration.

    Same iteration as :func:`bordered_newton`, with a leading candidate axis on
    every quantity. ``m_blocks`` and ``dm_blocks`` take a ``(P,)`` vector of
    probe points and return blocks shaped ``(P, b_i, b_j)``.

    The ragged control flow of the per-candidate solve is deliberately NOT
    preserved. Candidates converge after different numbers of steps; rather
    than compacting the batch as they drop out -- which buys nothing at these
    sizes and costs synchronisation and index bookkeeping -- all ``P`` run to
    ``max_iter`` under a boolean ``active`` mask, and a finished candidate's
    update is masked to zero so its ``z`` and ``r`` are frozen. ``iterations``
    still records the step at which each candidate would have left the loop,
    not ``max_iter``.

    Parameters
    ----------
    m_blocks, dm_blocks : callable
        ``(P,) -> (a_ii, a_ij, a_ji)`` of ``M(z)`` and ``M'(z)``.
    z0 : NDArray
        ``(P,)`` predicted poles.
    r0 : NDArray, optional
        ``(P, n_dof)`` predicted right vectors; one random unit vector, shared
        by every candidate, if omitted.
    tol : float, optional
        Acceptance threshold on ``eps_nep`` (doc Eq. 52). Default 1e-10.
    max_iter : int, optional
        Newton steps. Default 8.
    trust_radius : float or NDArray, optional
        Per-candidate cap on ``|delta z|`` (THz).
    norm_power : int, optional
        Power iterations for the operator norms in ``eps_nep`` and ``kappa``.
    left_iters : int, optional
        Adjoint inverse iterations for the left null vector.

    Returns
    -------
    PoleSolutions

    """
    z = xp.asarray(z0, dtype=xp.complex128).reshape(-1)
    n_probe = int(z.shape[0])
    blocks = m_blocks(z)
    n_dof = int(sum(int(b.shape[-1]) for b in blocks[0]))

    if r0 is None:
        # ONE start vector for the whole batch, so a candidate's answer does
        # not depend on who it was batched with. Identical to the draw the
        # per-candidate solve made, which reseeded rng(0) for every candidate.
        rng = xp.random.default_rng(0)
        r = rng.standard_normal(n_dof) + 1j * rng.standard_normal(n_dof)
        r = xp.broadcast_to(r, (n_probe, n_dof)) + 0.0
    else:
        r = xp.asarray(r0, dtype=xp.complex128).reshape(n_probe, n_dof)
    r = r / xp.linalg.norm(r, axis=-1, keepdims=True)
    c = xp.array(r, copy=True)

    tr = xp.broadcast_to(
        xp.asarray(trust_radius, dtype=xp.float64).reshape(-1), (n_probe,))
    active = xp.ones(n_probe, dtype=bool)
    # Default: the candidate used its whole budget, exactly as the scalar loop
    # leaves `it` at max_iter when it never breaks.
    iterations = xp.full(n_probe, max_iter, dtype=xp.int64)
    zero = xp.zeros((), dtype=xp.complex128)

    for it in range(1, max_iter + 1):
        blocks = m_blocks(z)
        dblocks = dm_blocks(z)
        fac = BTDFactorization.factorize(*blocks)

        # Both right-hand sides against one factorisation, in ONE solve.
        rhs = xp.stack([_matvec(blocks, r), _matvec(dblocks, r)], axis=-1)
        sol = fac.solve(rhs)
        x1, x2 = sol[..., 0], sol[..., 1]

        denom = _vdot(c, x2)
        dead = denom == 0.0
        iterations = xp.where(active & dead, it, iterations)
        active = active & ~dead

        dz = (_vdot(c, r) - 1.0 - _vdot(c, x1)) / xp.where(dead, 1.0, denom)
        adz = xp.abs(dz)
        dz = xp.where(adz > tr, dz * (tr / xp.where(adz > 0, adz, 1.0)), dz)
        dz = xp.where(active, dz, zero)
        dr = -x1 - dz[:, None] * x2

        z = z + dz
        # NOTE: r is deliberately NOT renormalised here. The gauge condition
        # c^H r = 1 is what fixes the scale of the null vector and closes the
        # bordered system; rescaling r inside the loop violates it and the
        # iteration stops converging.
        r = r + xp.where(active[:, None], dr, zero)

        settled = xp.abs(dz) < 1e-15 * xp.maximum(1.0, xp.abs(z))
        iterations = xp.where(active & settled, it, iterations)
        active = active & ~settled
        if not bool(active.any()):
            break

    r = r / xp.linalg.norm(r, axis=-1, keepdims=True)

    # Final residual and the left partner at the accepted z.
    blocks = m_blocks(z)
    dblocks = dm_blocks(z)
    fac = BTDFactorization.factorize(*blocks)
    m_norm = fac.norm2(n_power=norm_power)
    resid = xp.linalg.norm(_matvec(blocks, r), axis=-1)
    scale = xp.abs(z) ** 2 + m_norm
    eps_nep = resid / (scale * xp.linalg.norm(r, axis=-1))

    # Left null vector by inverse iteration on M^H, then the normalisation
    # l^H M' r = 1 that makes d_alpha = 1 (doc Eqs. 49-50).
    #
    # M(z_alpha) is singular at the pole, so M^{-H} does not exist there and
    # this is adjoint inverse iteration at a slightly unconverged z. That is
    # well posed for the DIRECTION -- the solve amplifies precisely the null
    # component being sought -- but the conditioning worsens as the pole solve
    # improves, so the result is iterated until its own residual stops falling
    # and that residual is reported rather than assumed.
    adj = _adjoint_blocks(blocks)
    l = c
    eps_left = xp.full(n_probe, xp.inf)
    # Latching: once a candidate stops improving it is DONE, exactly as the
    # scalar loop's `break` left it. A plain per-step minimum would let a later
    # iterate revive a candidate the scalar path had already abandoned.
    going = xp.ones(n_probe, dtype=bool)
    for _ in range(left_iters):
        cand = fac.solve_hermitian(l[..., None])[..., 0]
        nrm = xp.linalg.norm(cand, axis=-1)
        going = going & xp.isfinite(nrm) & (nrm != 0.0)
        cand = cand / xp.where(nrm > 0, nrm, 1.0)[:, None]
        # ||M^H l|| / ||M||: how well l annihilates M^H.
        res = xp.linalg.norm(
            xp.conj(_matvec(adj, xp.conj(cand))), axis=-1)
        cand_eps = res / xp.maximum(m_norm, 1e-300)
        going = going & ~(cand_eps > eps_left)
        l = xp.where(going[:, None], cand, l)
        eps_left = xp.where(going, cand_eps, eps_left)
        if not bool(going.any()):
            break

    d = _vdot(l, _matvec(dblocks, r))
    nz = d != 0.0
    l = l / xp.where(nz, xp.conj(d), 1.0)[:, None]

    # One more bordered-Newton step's worth of information, in THz. With the
    # normalisation above, l^H M'(z) r = 1, so the Newton correction is just
    # -l^H M(z) r -- no extra solve. Reported, never applied: applying it
    # would be a ninth iteration taken outside the trust region.
    dz_est = xp.where(nz, -_vdot(l, _matvec(blocks, r)), zero)

    # M'(z) is routinely singular (for the phonon operator it is essentially
    # 2z*I plus lead derivatives), so take its norm without factorising it.
    dm_norm = btd_norm2(*dblocks, n_power=norm_power)
    kappa = (xp.linalg.norm(l, axis=-1) * xp.linalg.norm(r, axis=-1) * dm_norm)

    return PoleSolutions(
        z=z, r=r, l=l, eps_nep=eps_nep, kappa=kappa,
        converged=eps_nep < tol, iterations=iterations, eps_left=eps_left,
        dz_est=dz_est,
    )


def _lift(fn: BlockFn) -> BlockFn:
    """A scalar ``z -> blocks`` callable, seen as a batched one.

    The reference path: it evaluates the operator one probe point at a time and
    stacks the answers, so a batched solve driven through it is arithmetically
    the old per-candidate solve. That is what makes
    :func:`bordered_newton_batch` testable against the code it replaces
    independently of the batched operator assembly.
    """

    def batched(zv: NDArray) -> Blocks:
        zv = xp.asarray(zv, dtype=xp.complex128).reshape(-1)
        out = [fn(complex(zv[k])) for k in range(int(zv.shape[0]))]

        def stack(part: int, b: int) -> NDArray:
            return xp.stack([xp.asarray(o[part][b]).reshape(
                xp.asarray(o[part][b]).shape[-2:]) for o in out])

        return tuple([stack(part, b) for b in range(len(out[0][part]))]
                     for part in range(3))

    return batched


def bordered_newton(
    m_blocks: BlockFn,
    dm_blocks: BlockFn,
    z0: complex,
    r0: NDArray | None = None,
    **kwargs,
) -> PoleSolution:
    r"""Correct a single predicted pole with the bordered Newton iteration.

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

    ``m_blocks`` and ``dm_blocks`` take a SCALAR ``z``. Production drives
    :func:`bordered_newton_batch` directly with natively batched operator
    closures; this wrapper exists for callers that have a scalar operator, and
    is the reference the batched core is verified against.

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
    **kwargs
        Passed to :func:`bordered_newton_batch` (``tol``, ``max_iter``,
        ``trust_radius``, ``norm_power``, ``left_iters``).

    Returns
    -------
    PoleSolution

    """
    batch = bordered_newton_batch(
        _lift(m_blocks), _lift(dm_blocks),
        xp.asarray([complex(z0)], dtype=xp.complex128),
        None if r0 is None else xp.asarray(r0).reshape(1, -1),
        **kwargs,
    )
    return batch.to_list()[0]


