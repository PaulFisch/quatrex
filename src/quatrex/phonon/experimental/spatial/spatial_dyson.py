# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""A causal retarded self-energy ACTION, and a Dyson solve that uses it.

The matrix-free plan assumes the Kramers-Kronig step forces a common spatial
basis: if the spatial realisation changes independently at every frequency then
``A``, ``B``, ``C`` cannot be transformed separately, so one seeks
:math:`\Delta\Sigma(\omega) \approx \sum_a F_a d_a(\omega)` with the
:math:`F_a` frequency independent and transforms only the coefficients.

Measured on a converged chain, that basis is **not compact** -- 78 of 140
operators at 1e-3, a flat singular spectrum, a windowed fallback that is a
repackaging, and a dominant spatial subspace that turns 64 degrees in three grid
steps.

It is not needed. The transform is linear and acts pointwise in ``(i, j)`` along
frequency, so for a **frequency-independent, real** probe

.. math::
    \mathcal H[\Sigma x] = \mathcal H[\Sigma]\, x

exactly. Measured: 3.7e-16 for a real probe and 7.4e-01 for a complex one. The
conjugation matters because the production transform is complex-linear on its
positive branch and CONJUGATE-linear on the bosonic mirror
(``core/fft_utils.py`` takes ``a[::-1].conj()``), so it commutes only with a
real probe. That is not a restriction in practice: the SCBA map is R-linear and
not C-linear (``core/jfnk.py``), so a Krylov solver on it already runs in the
real embedding.

**The cost, which is the whole design question.** One frequency pass -- ``N_w``
structured applications -- yields :math:`\Sigma^R(\omega)x` at EVERY frequency
simultaneously, not at one. So the route is efficient exactly when the Krylov
vectors can be shared across frequencies, and wasteful if each frequency runs
its own solver: per-frequency GMRES would pay ``N_w`` passes per frequency,
while a shared search space pays one pass per basis vector for all of them.

:func:`shared_krylov_dyson` therefore builds ONE search space, applies the
operator to it once per basis vector, and projects the residual per frequency.
Its cost is ``m N_w`` structured applications against ``N_D N_w`` to form
:math:`\Sigma^R` outright -- a factor ``N_D/m``, which is where the saving
lives on a device with many degrees of freedom and nowhere else.
"""

from __future__ import annotations

import numpy as np

from qttools import NDArray

__all__ = [
    "RetardedAction",
    "shared_krylov_dyson",
]


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _build_retarded():
    """The reference solver's Kramers-Kronig reconstruction.

    ``phonon/`` is not a package, so it is reached by path insertion -- the
    same route ``spatial_bubble`` takes. Deliberately the SAME function the
    reference arm uses: a second implementation would make any disagreement
    ambiguous.
    """
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[3] / "phonon")
    if root not in sys.path:
        sys.path.insert(0, root)
    from solver.retarded import build_retarded

    return build_retarded


class RetardedAction:
    r"""``Sigma^R(omega) x`` for every omega at once, causally, from the
    lesser/greater actions.

    Parameters
    ----------
    apply_lesser, apply_greater : callable
        ``x -> (n_freq, n, k)``. Whatever produces them -- a structured
        operator, a dense matrix, a streamed bubble -- is this class's
        business only through these two calls.
    freqs_thz : NDArray
        The frequency axis, passed straight to ``build_retarded``.
    method : str
        ``"fft"`` (the full Kramers-Kronig reconstruction), ``"pv"``, or
        ``"half"``. The same one the reference uses, or the comparison is
        contaminated.
    """

    def __init__(self, apply_lesser, apply_greater, freqs_thz,
                 method: str = "fft", require_real: bool = True):
        self.apply_lesser = apply_lesser
        self.apply_greater = apply_greater
        self.freqs_thz = np.asarray(_host(freqs_thz))
        self.method = method
        self.require_real = require_real
        self.n_passes = 0

    def apply(self, x) -> NDArray:
        r"""``(n_freq, n, k)``: the causal action at every frequency.

        ``x`` must be REAL. The bosonic mirror branch of the transform is
        conjugate-linear, so a complex probe silently returns the transform of
        a different operator -- 74 % wrong, with no warning. The check is not
        defensive coding; it is the one condition the identity rests on.
        """
        x = np.asarray(_host(x))
        if self.require_real and np.iscomplexobj(x) and np.abs(x.imag).max() > 0:
            raise ValueError(
                "RetardedAction.apply: the probe must be real. The Hilbert "
                "transform's bosonic mirror is conjugate-linear, so "
                "H[Sigma x] = H[Sigma] x holds only for a real x (measured: "
                "3.7e-16 real against 7.4e-01 complex).")
        self.n_passes += 1
        build_retarded = _build_retarded()
        al = self.apply_lesser(x)
        ag = self.apply_greater(x)
        return build_retarded(al, ag, self.freqs_thz, method=self.method)


def _from_dense_actions(sigma_lesser, sigma_greater):
    """Action callables backed by dense matrices -- the reference arm."""
    def mk(mat):
        def apply(x):
            return np.einsum("wij,jk->wik", mat, np.asarray(_host(x)))
        return apply

    return mk(sigma_lesser), mk(sigma_greater)


def shared_krylov_dyson(a_diag, sigma_action: RetardedAction, rhs, *,
                        max_basis: int = 40, tol: float = 1e-10,
                        preconditioner=None):
    r"""Solve ``[A(omega) - Sigma^R(omega)] G(omega) = rhs`` at every frequency
    with ONE shared search space.

    ``a_diag`` is ``(n_freq, n, n)``, the local part of the Dyson operator --
    ``omega^2 I - D - Sigma_L^R - Sigma_R^R`` -- which is cheap to apply and
    invert per frequency; ``sigma_action`` supplies the long-range part.

    The search space is shared because one call to
    :meth:`RetardedAction.apply` already produces the action at every
    frequency; growing a separate space per frequency would pay ``N_w`` passes
    per frequency instead of one per basis vector. Each vector is orthogonalised
    globally and the projected problem is solved per frequency by least squares,
    which is the same construction as a global/block GMRES.

    The basis is grown from REAL directions, which the transform requires and
    which costs nothing: the real and imaginary parts of a complex residual are
    each admissible directions, so the space is built over the real embedding.

    Returns ``(x, info)`` with ``x`` of shape ``(n_freq, n, k)``.
    """
    a_diag = np.asarray(_host(a_diag))
    rhs = np.asarray(_host(rhs))
    n_freq, n, _ = a_diag.shape
    if rhs.ndim == 2:
        rhs = np.broadcast_to(rhs[None], (n_freq,) + rhs.shape).copy()
    k = rhs.shape[-1]

    def op(v):
        """Apply the full Dyson operator to one shared real direction.

        ``v`` is ``(n,)``; the result is ``(n_freq, n)``. One call is one
        frequency pass and it serves every frequency at once, which is the
        property the shared basis exists to exploit.
        """
        col = np.asarray(v, dtype=float)[:, None]
        return (np.einsum("wij,jk->wik", a_diag, col)
                - sigma_action.apply(col))[:, :, 0]

    basis: list[NDArray] = []
    images: list[NDArray] = []

    def push(v):
        v = np.asarray(v, dtype=float).ravel()
        for b in basis:
            v = v - b * float(b @ v)
        for b in basis:                       # one reorthogonalisation pass
            v = v - b * float(b @ v)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-12:
            return False
        v = v / nrm
        basis.append(v)
        images.append(op(v))
        return True

    # Seed with the right-hand side's own directions, real and imaginary: both
    # are admissible REAL directions, and the transform requires real ones.
    seed = rhs.mean(axis=0)
    for part in (seed.real, seed.imag):
        for col in range(part.shape[-1]):
            if len(basis) >= max_basis:
                break
            push(part[:, col])
    if not basis:
        push(np.ones(n))

    hist = []
    while True:
        m = len(basis)
        vb = np.stack(basis, axis=1).astype(complex)         # (n, m)
        x = np.zeros((n_freq, n, k), dtype=complex)
        resid = np.zeros((n_freq, n, k), dtype=complex)
        for iw in range(n_freq):
            design = np.stack([images[a][iw] for a in range(m)], axis=1)
            coef, *_ = np.linalg.lstsq(design, rhs[iw], rcond=None)
            x[iw] = vb @ coef
            resid[iw] = rhs[iw] - design @ coef
        rel = float(np.linalg.norm(resid) / (np.linalg.norm(rhs) + 1e-300))
        hist.append(rel)
        if rel < tol or m >= max_basis:
            break
        # Grow along the worst frequency's residual. Both parts are pushed
        # because a complex residual carries two real directions and dropping
        # one halves the space for no saving.
        iw = int(np.argmax(np.linalg.norm(resid, axis=(1, 2))))
        grew = False
        for part in (resid[iw].real, resid[iw].imag):
            for col in range(part.shape[-1]):
                if len(basis) >= max_basis:
                    break
                grew |= push(part[:, col])
        if not grew:
            break
    return x, {"basis": len(basis), "residual": hist[-1], "history": hist,
               "passes": sigma_action.n_passes}
