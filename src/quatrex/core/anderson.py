# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Plain Anderson acceleration for fixed-point iterations.

Implements Anderson(m) (Anderson, J. ACM 12, 547 (1965)) in the
unconstrained least-squares form recommended by Walker & Ni,
SIAM J. Numer. Anal. 49, 1715 (2011), Alg. AA / Eq. (3.1):

    f_k   = g(x_k) - x_k                       (fixed-point residual)
    gamma = argmin || f_k - dF gamma ||_2      (dF: last m residual diffs)
    x_+   = x_k + beta f_k - (dX + beta dF) gamma

where dX/dF hold the differences of the last ``m`` iterates/residuals
and ``beta`` is the damping (mixing) factor. With an empty history this
is exactly damped fixed-point iteration -- Anderson(0) in the language
of Toth & Kelley, SIAM J. Numer. Anal. 53, 805 (2015), whose analysis
guarantees local r-linear convergence on a contraction provided the
least-squares coefficients stay bounded. No further safeguards are
applied; ``numpy.linalg.lstsq`` resolves a (near-)rank-deficient history
through its SVD cutoff.
"""

import numpy as np

from quatrex.core.mpi_linalg import allreduce_sum, get_comm


class RREMixer:
    """Restarted Reduced-Rank (minimal-residual) Extrapolation for fixed points.

    Locates the fixed point Sigma* as the affine combination
        s = sum_i gamma_i x_i ,   sum_i gamma_i = 1
    of the last ``cycle`` iterates that MINIMISES the residual combination
    ``|| sum_i gamma_i f_i ||`` (f_i = g(x_i) - x_i). This is RRE / minimal-
    polynomial extrapolation (Sidi, *Vector Extrapolation Methods*, 2017): an
    ALGEBRAIC extrapolation that locates an UNSTABLE fixed point which damped /
    Picard / Anderson iteration cannot reach (Jacobian |lambda| > 1) -- the
    cnt33 eta=0 band-edge optical mode on the longer cells. Damped Picard builds
    the short sequence; every ``cycle`` steps it extrapolates and RESTARTS from
    ``s`` (the restart breaks the complex-mode limit cycle that defeats windowed
    Anderson, which the deep-Anderson sweep confirmed plateaus at ~0.08).

    Real coefficients (the SCBA map is real on the complex Sigma): gamma from the
    real Gram ``Re(F^H F)``. Under MPI the Sigma is row-partitioned across ranks,
    so the (cycle x cycle) Gram is Allreduce-summed to the GLOBAL Gram -- the
    extrapolation coefficients must be identical on every rank to cancel the
    global (frequency-coupled) mode, not a per-slice one.
    """

    def __init__(self, cycle: int = 8, beta: float = 0.2, ridge: float = 1e-6) -> None:
        self.cycle = max(3, int(cycle))
        self.beta = float(beta)
        self.ridge = float(ridge)
        self._X: list[np.ndarray] = []
        self._F: list[np.ndarray] = []
        self._comm, self._SUM = get_comm()

    def step(self, x: np.ndarray, gx: np.ndarray) -> np.ndarray:
        """One RRE step; ``x`` is the (rank-local) iterate, ``gx = g(x)``."""
        f = gx - x
        self._X.append(x.copy())
        self._F.append(f.copy())
        if len(self._F) < self.cycle:
            return x + self.beta * f          # build the sequence (damped Picard)

        F = np.stack(self._F, axis=1)          # (n_local, cycle)
        X = np.stack(self._X, axis=1)
        G = np.ascontiguousarray((F.conj().T @ F).real)   # local Gram (cycle x cycle)
        G = allreduce_sum(self._comm, self._SUM, G)        # GLOBAL Gram
        self._X.clear()
        self._F.clear()                        # restart

        n = G.shape[0]
        reg = self.ridge * (float(np.trace(G)) / n + 1e-300)
        try:
            w = np.linalg.solve(G + reg * np.eye(n), np.ones(n))
            wsum = float(w.sum())
            if not np.isfinite(wsum) or abs(wsum) < 1e-12:
                raise np.linalg.LinAlgError
            s = X @ (w / wsum)
            if not np.all(np.isfinite(s)):
                raise np.linalg.LinAlgError
            return s
        except np.linalg.LinAlgError:
            return x + self.beta * f           # degenerate cycle -> damped restart


class AndersonMixer:
    """Anderson(m) accelerator over flattened fixed-point iterates.

    Parameters
    ----------
    depth : int
        History size ``m`` (number of stored residual differences).
    beta : float
        Damping factor applied to the residual (the linear mixing factor).
    """

    def __init__(self, depth: int = 5, beta: float = 0.5, period: int = 1,
                 restart: int = 0, ridge: float = 0.0) -> None:
        self.depth = int(depth)
        self.beta = float(beta)
        # Periodic-Pulay / alternating-Anderson stride (Banerjee, Suryanarayana &
        # Pask, Chem. Phys. Lett. 647, 31 (2016)): extrapolate only every
        # `period`-th step, plain damped linear mixing in between (history still
        # accumulated). The damped steps stop the accelerator from locking onto a
        # near-unit-modulus Jacobian eigenvector and limit-cycling -- the failure
        # mode of plain Anderson/DIIS on a marginal/soft mode (the d5a soft-twist
        # plateau at ~1e-3). period=1 reproduces ordinary Anderson(m).
        self.period = max(1, int(period))
        # `restart`: periodically forget the history (every `restart` steps) so the
        # accelerator cannot lock into a limit cycle on a marginal (Jacobian
        # eigenvalue ~1) mode -- the cnt33 eta=0 causal-Sigma^R band-edge mode,
        # where periodic Pulay alone still oscillates (resid 0.05<->0.49). The
        # canonical DIIS escape (Pratapa & Suryanarayana 2015; Walker & Ni 2011
        # restarting). 0 = never restart.
        self.restart = int(restart)
        # `ridge`: scale-relative Tikhonov regularisation of the least-squares
        # coefficients, suppressing the blow-ups from a near-rank-deficient
        # history (the source of the overshoot spikes). 0 = plain lstsq (SVD).
        self.ridge = float(ridge)
        self._x_prev: np.ndarray | None = None
        self._f_prev: np.ndarray | None = None
        self._dx: list[np.ndarray] = []
        self._df: list[np.ndarray] = []
        self._it = 0

    def step(self, x: np.ndarray, gx: np.ndarray) -> np.ndarray:
        """One Anderson update; ``x`` is the iterate, ``gx = g(x)``."""
        f = gx - x
        # Periodic restart: drop the accumulated history (and the diff baseline)
        # so the next step is a clean damped-linear restart that rebuilds it.
        if self.restart and self._it and (self._it % self.restart == 0):
            self._dx.clear()
            self._df.clear()
            self._x_prev = None
            self._f_prev = None
        if self._x_prev is not None:
            self._dx.append(x - self._x_prev)
            self._df.append(f - self._f_prev)
            if len(self._dx) > self.depth:
                self._dx.pop(0)
                self._df.pop(0)
        self._x_prev = x.copy()
        self._f_prev = f.copy()
        self._it += 1

        # Periodic Pulay: damped linear except every `period`-th step.
        if (not self._dx) or (self._it % self.period != 0):
            return x + self.beta * f

        dX = np.stack(self._dx, axis=1)  # (n, m)
        dF = np.stack(self._df, axis=1)
        if self.ridge > 0.0:
            # Scale-relative normal-equation solve with Tikhonov damping.
            A = dF.conj().T @ dF
            reg = self.ridge * (float(np.trace(A).real) / A.shape[0] + 1e-300)
            gamma = np.linalg.solve(A + reg * np.eye(A.shape[0], dtype=A.dtype),
                                    dF.conj().T @ f)
        else:
            gamma, *_ = np.linalg.lstsq(dF, f, rcond=None)
        return x + self.beta * f - (dX + self.beta * dF) @ gamma
