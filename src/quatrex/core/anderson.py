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

from quatrex.core.mpi_linalg import allreduce_sum, get_comm


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
                 restart: int = 0, ridge: float = 0.0,
                 step_cap: float = 0.0, revert_factor: float = 0.0,
                 stagnation_restart: int = 0,
                 collect_diagnostics: bool = False) -> None:
        self.depth = int(depth)
        self.beta = float(beta)
        # Periodic-Pulay / alternating-Anderson stride: extrapolate only every
        # `period`-th step, plain damped linear mixing in between (history
        # still accumulated). The damped steps stop the accelerator from
        # locking onto a near-unit-modulus Jacobian eigenvector and
        # limit-cycling. period=1 reproduces ordinary Anderson(m).
        self.period = max(1, int(period))
        # `restart`: periodically forget the history (every `restart` steps)
        # so the accelerator cannot lock into a limit cycle on a marginal
        # (Jacobian eigenvalue ~1) mode. 0 = never restart.
        self.restart = int(restart)
        # `ridge`: scale-relative Tikhonov regularisation of the least-squares
        # coefficients, suppressing the blow-ups from a near-rank-deficient
        # history. 0 = plain lstsq (SVD).
        self.ridge = float(ridge)
        # Safeguards (ported from the dense _AndersonAccelerator; all off at
        # their zero defaults, in which case step() is bit-identical to the
        # unguarded implementation):
        #   step_cap: reject the Anderson step when its correction exceeds
        #     step_cap x the damped-linear step norm; take the linear step
        #     that iteration (history kept).
        #   revert_factor: when the residual exceeds revert_factor x the best
        #     residual seen, clear the history and return the best iterate.
        #   stagnation_restart: clear the history after N consecutive
        #     non-improving steps (gentler than restarting on every uptick).
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
            # Global residual norm (collective -- called symmetrically on all
            # ranks whenever any safeguard/diagnostic is enabled).
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
        # Periodic restart: drop the accumulated history (and the diff baseline)
        # so the next step is a clean damped-linear restart that rebuilds it.
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
        # The least-squares coefficients contract the distributed rows, so the
        # normal equations (dF^H dF) gamma = dF^H f are GLOBAL inner products:
        # reduce them (one packed Allreduce(SUM)) and solve the replicated
        # (m x m) system so gamma is identical on every rank.
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
            # Step-cap safeguard: the Anderson step may not exceed
            # step_cap x the UNDAMPED residual norm ||f|| (global norms).
            # A clean quasi-Newton step is ~||f||/(1-lambda_max), so the cap
            # is interpretable as the largest 1/(1-lambda) the extrapolation
            # is trusted for, independent of beta. (The dense solver's cap
            # was relative to the damped step beta*||f||; that made the
            # threshold beta-dependent.)
            corr2 = self._global_norm2(x_new - x)
            f2 = fnorm ** 2 if fnorm is not None else self._global_norm2(f)
            if corr2 > (self.step_cap ** 2) * f2:
                diag["kind"] = "capped"; diag["capped"] = 1
                self.diagnostics.append(diag)
                return x + self.beta * f
        if guarded:
            self.diagnostics.append(diag)
        return x_new
