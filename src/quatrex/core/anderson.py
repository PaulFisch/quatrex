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


class AndersonMixer:
    """Anderson(m) accelerator over flattened fixed-point iterates.

    Parameters
    ----------
    depth : int
        History size ``m`` (number of stored residual differences).
    beta : float
        Damping factor applied to the residual (the linear mixing factor).
    """

    def __init__(self, depth: int = 5, beta: float = 0.5) -> None:
        self.depth = int(depth)
        self.beta = float(beta)
        self._x_prev: np.ndarray | None = None
        self._f_prev: np.ndarray | None = None
        self._dx: list[np.ndarray] = []
        self._df: list[np.ndarray] = []

    def step(self, x: np.ndarray, gx: np.ndarray) -> np.ndarray:
        """One Anderson update; ``x`` is the iterate, ``gx = g(x)``."""
        f = gx - x
        if self._x_prev is not None:
            self._dx.append(x - self._x_prev)
            self._df.append(f - self._f_prev)
            if len(self._dx) > self.depth:
                self._dx.pop(0)
                self._df.pop(0)
        self._x_prev = x.copy()
        self._f_prev = f.copy()

        if not self._dx:
            return x + self.beta * f

        dX = np.stack(self._dx, axis=1)  # (n, m)
        dF = np.stack(self._df, axis=1)
        gamma, *_ = np.linalg.lstsq(dF, f, rcond=None)
        return x + self.beta * f - (dX + self.beta * dF) @ gamma
