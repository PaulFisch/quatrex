# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Limited-memory type-I ("good") Broyden root finder for the SCBA fixed point.

This is a quasi-Newton ROOT FINDER for ``F(x) = g(x) - x = 0`` (here
``g(x) = Sigma[G[Sigma]]``), as opposed to a fixed-point *accelerator*: when
the SCBA map's Jacobian ``G' = dg/dx`` has an eigenvalue outside the unit
circle, the fixed-point iteration limit-cycles for ANY scalar damping, yet a
Newton/Broyden step

    x_{n+1} = x_n - B_n F(x_n),   B_n ~ J^{-1},  J = dF/dx = I - G'

converges to the fixed point regardless of the iteration's spectral radius,
because ``J = I - G'`` is NONSINGULAR there (no eigenvalue of ``G'`` equals
1). If even this stalls (or returns an unphysical Sigma), the zero-eta
self-consistent solution is genuinely absent and finite eta is required.

Type-I vs type-II. The plain Anderson(m)
(:class:`quatrex.core.anderson.AndersonMixer`) is Broyden's *second* ("bad")
method -- it projects the secant least-squares onto the residual differences
``dF``. This is the *first* ("good") method, which projects onto the iterate
differences ``dX``: the only change is the multisecant matrix ``dX^H dF`` in
place of ``dF^H dF``. Type-I minimises the change in the Jacobian (not its
inverse) and is the more robust root finder on stiff/unstable problems.

MPI. The iterate x = [Sigma^<, Sigma^>, Sigma^R] is ROW-PARTITIONED across
ranks (disjoint slices; the concatenation over MPI.COMM_WORLD is the global
vector). The secant matrix ``A = dX^H dF`` (m x m) and right-hand side
``rhs = dX^H f`` (m) contract the distributed rows and MUST be
``Allreduce(SUM)``-reduced, or every rank solves a different, globally wrong
quasi-Newton step. A and rhs are packed into a single Allreduce; the m x m
solve is tiny and done replicated (bit-identical inputs -> a globally
consistent step with no further communication). The final
``x + beta f - (dX + beta dF) gamma`` axpy is purely local.
"""

import numpy as np

from quatrex.core.mpi_linalg import (
    allreduce_sum,
    get_comm,
    global_norm,
    real_embedded,
    trust_cap,
)


class BroydenMixer:
    """Limited-memory MPI-aware type-I (good) Broyden update over flattened iterates.

    Drop-in for :class:`quatrex.core.anderson.AndersonMixer` (same ``step``
    signature), selected by ``config.scba.mixing_method = "broyden"``.

    Parameters
    ----------
    depth : int
        History size ``m`` (stored secant pairs). Spans the small unstable
        subspace plus redundancy; 8 is a good default, 6-12 the useful range.
    beta : float
        Initial inverse-Jacobian scale ``B0 = -beta I`` (the bare step is the
        damped fixed-point step ``x + beta f``; the secant corrections refine
        it). Also the damping of the warm-up Picard phase.
    ridge : float
        Tikhonov regularisation of the (generally non-symmetric, possibly
        singular) ``dX^H dF`` multisecant solve, scaled by ``||A||_F``. Kept
        TINY (1e-8) so it does not damp the Newton correction in the
        near-marginal subspace; the rank-revealing ``lstsq`` cutoff
        (``rcond``) drops degenerate stale-secant directions.
    warmup : int
        Run plain damped-Picard ``x + beta f`` for the first ``warmup`` calls
        (still accumulating secant pairs), then engage Broyden. The warm-up
        parks the iterate near the saddle -- where ``J`` is nonsingular and
        the secant history is clean -- before the local quasi-Newton step
        takes over. 0 = Broyden from the start.
    rcond : float
        Relative singular-value cutoff for the truncated-SVD multisecant solve
        (drop ``sigma_i < rcond * sigma_max``).
    trust : float
        Trust-region step cap: the per-iteration update ``||x_new - x||`` is
        limited to ``trust * ||x||`` (global norms). A full quasi-Newton step
        from a far-from-root / stale secant model overshoots the nonlinear
        SCBA map; the cap forces gradual descent until the model is good,
        then deactivates near the root. 0 disables it.
    """

    #: JFNK emits finite-difference PROBE iterates; this mixer never does, so
    #: the driver may always test convergence on its output.
    probing: bool = False

    def __init__(self, depth: int = 8, beta: float = 0.5,
                 ridge: float = 1e-8, warmup: int = 0,
                 rcond: float = 1e-10, trust: float = 0.3,
                 patience: int = 8, progress_thresh: float = 0.9) -> None:
        self.depth = int(depth)
        self.beta = float(beta)
        self.ridge = float(ridge)
        self.warmup = int(warmup)
        self.rcond = float(rcond)
        self.trust = float(trust)
        self.patience = int(patience)
        self.progress_thresh = float(progress_thresh)
        self._x_prev: np.ndarray | None = None
        self._f_prev: np.ndarray | None = None
        self._dx: list[np.ndarray] = []
        self._df: list[np.ndarray] = []
        self._fhist: list[float] = []
        self._it = 0
        self._comm, self._SUM = get_comm()

    @real_embedded
    def step(self, x: np.ndarray, gx: np.ndarray) -> np.ndarray:
        """One Broyden update; ``x`` is the (rank-local) iterate, ``gx = g(x)``."""
        f = gx - x
        if self._x_prev is not None:
            self._dx.append(x - self._x_prev)
            self._df.append(f - self._f_prev)
            if len(self._dx) > self.depth:
                self._dx.pop(0)
                self._df.pop(0)
        self._x_prev = x.copy()
        self._f_prev = f.copy()
        self._it += 1

        # Warm-up (or empty history): plain damped Picard, but keep building the
        # secant history so Broyden engages with a clean, near-the-root buffer.
        if self._it <= self.warmup or not self._dx:
            return x + self.beta * f

        # STALL GATE (optional, patience>0): only take the quasi-Newton step
        # once plain Picard has PLATEAUED. While the residual keeps
        # shrinking, Picard is doing the right thing and the far-from-root
        # secant model would overshoot; a stalled / limit-cycling residual
        # is the signature of a |lambda|>1 mode -- also where the secant
        # buffer is cleanest. Secants keep accumulating above so Broyden
        # engages warm.
        if self.patience > 0:
            self._fhist.append(global_norm(self._comm, self._SUM, f))
            if len(self._fhist) > self.patience + 2:
                self._fhist.pop(0)
            if len(self._fhist) <= self.patience:
                # Not enough post-warmup residual history to demonstrate a
                # plateau -- keep accumulating secants under damped Picard.
                return x + self.beta * f
            if self._fhist[-1] < self.progress_thresh * self._fhist[0]:
                return x + self.beta * f   # Picard still making progress

        dX = np.stack(self._dx, axis=1)  # (n_local, m)
        dF = np.stack(self._df, axis=1)
        m = dX.shape[1]
        # type-I ("good" Broyden): (dX^H dF) gamma = dX^H f  (vs dF^H dF for
        # Anderson/type-II). These are GLOBAL inner products over the distributed
        # rows -> Allreduce(SUM). Pack A (m x m) and rhs (m) into one reduction.
        A = dX.conj().T @ dF
        rhs = dX.conj().T @ f
        # Both contract the distributed rows; pack into one Allreduce(SUM).
        packed = allreduce_sum(
            self._comm, self._SUM,
            np.concatenate([np.ascontiguousarray(A).ravel(), rhs]))
        A = packed[:m * m].reshape(m, m)
        rhs = packed[m * m:]
        # Tiny ridge from the ALREADY-REDUCED A (do not reduce a local norm);
        # the rank-revealing lstsq cutoff drops only degenerate secant
        # directions, so the marginal subspace (large signal in dF) is
        # preserved.
        reg = self.ridge * (float(np.linalg.norm(A)) + 1e-300)
        gamma = np.linalg.lstsq(A + reg * np.eye(m, dtype=A.dtype), rhs,
                                rcond=self.rcond)[0]
        step = self.beta * f - (dX + self.beta * dF) @ gamma
        return x + trust_cap(self._comm, self._SUM, step, x, self.trust)
