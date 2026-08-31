# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Restarted reduced-rank extrapolation."""

import numpy as np

from quatrex.core.mpi_linalg import allreduce_sum, get_comm, real_embedded


class RREMixer:
    """Restarted Reduced-Rank (minimal-residual) Extrapolation for fixed points.

    Locates the fixed point Sigma* as the affine combination
        s = sum_i gamma_i x_i ,   sum_i gamma_i = 1
    of the last ``cycle`` iterates that MINIMISES the residual combination
    ``|| sum_i gamma_i f_i ||`` (f_i = g(x_i) - x_i). This is RRE / minimal-
    polynomial extrapolation: an ALGEBRAIC extrapolation that can locate an
    UNSTABLE fixed point (Jacobian |lambda| > 1) which damped / Picard /
    Anderson iteration cannot reach. Damped Picard builds the short
    sequence; every ``cycle`` steps it extrapolates and RESTARTS from ``s``
    (the restart breaks the complex-mode limit cycle that defeats windowed
    Anderson).

    Real coefficients (the SCBA map is real on the complex Sigma): gamma from the
    real Gram ``Re(F^H F)``. Under MPI the Sigma is row-partitioned across ranks,
    so the (cycle x cycle) Gram is Allreduce-summed to the GLOBAL Gram -- the
    extrapolation coefficients must be identical on every rank to cancel the
    global (frequency-coupled) mode, not a per-slice one.
    """

    probing: bool = False

    def __init__(self, cycle: int = 8, beta: float = 0.2, ridge: float = 1e-6) -> None:
        self.cycle = max(3, int(cycle))
        self.beta = float(beta)
        self.ridge = float(ridge)
        self._X: list[np.ndarray] = []
        self._F: list[np.ndarray] = []
        self._comm, self._SUM = get_comm()

    @real_embedded
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

