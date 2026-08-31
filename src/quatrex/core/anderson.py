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
is exactly damped fixed-point iteration (Anderson(0)). No further
safeguards are applied; ``numpy.linalg.lstsq`` resolves a
(near-)rank-deficient history through its SVD cutoff.
"""

import numpy as np

from quatrex.core.mpi_linalg import allreduce_sum, get_comm, real_embedded



class AndersonMixer:
    """Anderson(m) accelerator over flattened fixed-point iterates.

    Parameters
    ----------
    depth : int
        History size ``m`` (number of stored residual differences).
    beta : float
        Damping factor applied to the residual (the linear mixing factor).
    """

    probing: bool = False

    def __init__(self, depth: int = 5, beta: float = 0.5, period: int = 1,
                 restart: int = 0, ridge: float = 0.0,
                 step_cap: float = 0.0, revert_factor: float = 0.0,
                 stagnation_restart: int = 0,
                 collect_diagnostics: bool = False) -> None:
        self.depth = int(depth)
        self.beta = float(beta)
        self.period = max(1, int(period))
        self.restart = int(restart)
        self.ridge = float(ridge)
        self.step_cap = float(step_cap)
        self.revert_factor = float(revert_factor)
        self.stagnation_restart = int(stagnation_restart)
        self.collect_diagnostics = bool(collect_diagnostics)
        #: per-step diagnostics dicts (kept small; see step())
        self.diagnostics: list[dict] = []
        self._x_prev: np.ndarray | None = None
        self._f_prev: np.ndarray | None = None
        self._dx: list[np.ndarray] = []
        self._df: list[np.ndarray] = []
        self._best_fnorm = np.inf
        self._best_x: np.ndarray | None = None
        self._n_since_best = 0
        self._it = 0
        self._comm, self._SUM = get_comm()

    def _global_norm2(self, v: np.ndarray) -> float:
        """Global squared 2-norm of a row-partitioned vector (collective)."""
        loc = np.array([float(np.vdot(v, v).real)])
        return float(allreduce_sum(self._comm, self._SUM, loc)[0])

    @real_embedded
    def step(self, x: np.ndarray, gx: np.ndarray) -> np.ndarray:
        """One Anderson update; ``x`` is the iterate, ``gx = g(x)``."""
        f = gx - x
        guarded = (self.step_cap > 0.0 or self.revert_factor > 0.0
                   or self.stagnation_restart > 0 or self.collect_diagnostics)
        diag = {"it": self._it, "kind": "linear", "m": len(self._dx),
                "fnorm": np.nan, "cond": np.nan, "gnorm": np.nan,
                "capped": 0, "reverted": 0, "restarted": 0} if guarded else None
        fnorm = None
        if guarded:
            fnorm = np.sqrt(self._global_norm2(f))
            diag["fnorm"] = fnorm
            # Best-iterate tracking + revert / stagnation safeguards.
            if fnorm < self._best_fnorm:
                self._best_fnorm = fnorm
                self._n_since_best = 0
                if self.revert_factor > 0.0:
                    self._best_x = x.copy()
            else:
                self._n_since_best += 1
                if (self.revert_factor > 0.0 and self._best_x is not None
                        and fnorm > self.revert_factor * self._best_fnorm):
                    self._dx.clear(); self._df.clear()
                    self._x_prev = None; self._f_prev = None
                    self._n_since_best = 0
                    self._it += 1
                    diag["kind"] = "revert"; diag["reverted"] = 1
                    self.diagnostics.append(diag)
                    return self._best_x.copy()
                if (self.stagnation_restart > 0
                        and self._n_since_best >= self.stagnation_restart):
                    self._dx.clear(); self._df.clear()
                    self._x_prev = None; self._f_prev = None
                    self._n_since_best = 0
                    diag["restarted"] = 1
        if self.restart and self._it and (self._it % self.restart == 0):
            self._dx.clear()
            self._df.clear()
            self._x_prev = None
            self._f_prev = None
            if guarded:
                diag["restarted"] = 1
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
            if guarded:
                self.diagnostics.append(diag)
            return x + self.beta * f

        dX = np.stack(self._dx, axis=1)  # (n_local, m)
        dF = np.stack(self._df, axis=1)
        m = dF.shape[1]
        A = dF.conj().T @ dF
        rhs = dF.conj().T @ f
        packed = allreduce_sum(
            self._comm, self._SUM,
            np.concatenate([np.ascontiguousarray(A).ravel(), rhs]))
        A = packed[:m * m].reshape(m, m)
        rhs = packed[m * m:]
        if self.ridge > 0.0:
            # Scale-relative Tikhonov damping on the reduced normal equations.
            reg = self.ridge * (float(np.trace(A).real) / m + 1e-300)
            gamma = np.linalg.solve(A + reg * np.eye(m, dtype=A.dtype), rhs)
        else:
            gamma = np.linalg.lstsq(A, rhs, rcond=None)[0]
        if guarded:
            diag["kind"] = "anderson"
            diag["m"] = m
            diag["gnorm"] = float(np.linalg.norm(gamma))
            ev = np.linalg.eigvalsh(A)
            diag["cond"] = float(ev[-1] / max(ev[0], 1e-300))
        x_new = x + self.beta * f - (dX + self.beta * dF) @ gamma
        if self.step_cap > 0.0:
            corr2 = self._global_norm2(x_new - x)
            f2 = fnorm ** 2 if fnorm is not None else self._global_norm2(f)
            if corr2 > (self.step_cap ** 2) * f2:
                diag["kind"] = "capped"; diag["capped"] = 1
                self.diagnostics.append(diag)
                return x + self.beta * f
        if guarded:
            self.diagnostics.append(diag)
        return x_new
