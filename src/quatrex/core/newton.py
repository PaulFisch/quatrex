# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Newton-Krylov mixer with EXACT analytic Jacobian-vector products.

Same Newton system as :mod:`quatrex.core.jfnk` -- ``(J_F - I) delta = -R``
with ``R = F[Sigma] - Sigma`` solved by real-embedded GMRES -- but the
Jacobian-vector products come from :class:`quatrex.core.phonon_jvp.PhononJVP`
(frozen-G Dyson linearisation + the polarisation identity of the quadratic
bubble) instead of finite differences of the map. Two consequences:

* SYNCHRONOUS: a JVP costs two bubble calls in-process, not a full SCBA
  iteration threaded through the driver, so the whole inner GMRES for one
  Newton step runs inside a single ``step()`` call. No probe state machine,
  no ``probing`` flag gymnastics -- every SCBA iteration is a genuine
  iterate and the driver's convergence/conservation gates see only those.

* EXACT: the Arnoldi basis carries no differencing noise, so the
  Eisenstat-Walker forcing sequence can actually be met by the inner solve.

Two-phase globalisation (the DFT-SCF pattern): damped Picard until the
residual has dropped below ``switch_tol`` relative to the first residual
(Newton's quadratic convergence is local), then inexact Newton with
Eisenstat-Walker forcing, an adaptive trust region, and a backtracking
accept/reject test on ``||R||`` -- the trial iterate's residual arrives with
the NEXT ``step()`` call (its map value is that SCBA iteration), so a
rejected step costs exactly one fixed-point sweep.

MPI: iterates are row-partitioned; every control-flow scalar (norms, inner
products, the Hessenberg solve input) is globally reduced, so all ranks
take identical branches around the collective kernel calls inside the JVP.
"""

from __future__ import annotations

import numpy as np

from quatrex.core.mpi_linalg import (
    complex_to_real as _c2r,
    get_comm,
    global_dot,
    global_norm,
    real_to_complex as _r2c,
    trust_cap,
)

__all__ = ["NewtonKrylovMixer"]


class NewtonKrylovMixer:
    """Synchronous exact-Jacobian Newton-Krylov over flat SCBA iterates.

    Parameters
    ----------
    jvp_factory : callable
        Zero-argument callable returning the (cached) ``PhononJVP``
        context. Lazy because the mixer is built before the phonon
        solver exists.
    warmup : int
        Minimum damped-Picard iterations before Newton may engage.
    switch_tol : float
        Engage Newton once ``||R|| <= switch_tol * ||R_first||`` (and the
        warmup count has passed). ``>= 1`` engages right after warmup.
    beta : float
        Picard damping for the warm phase and fallbacks.
    max_krylov : int
        Maximum GMRES dimension per Newton step (= 2x bubble calls each).
    inner_tol : float
        Base relative GMRES tolerance (used when ``forcing='fixed'``).
    forcing : {"ew", "fixed"}
        Eisenstat-Walker forcing or fixed inner tolerance.
    max_newton : int
        Cap on Newton steps; afterwards fall back to damped Picard.
    trust, trust_max : float
        Initial and maximal trust radius (fraction of ``||x||``); adapted
        with hysteresis like the JFNK mixer. ``trust <= 0`` disables.
    newton_damp : float
        Fixed damping of the accepted Newton step.
    backtrack : int
        Maximum step-halvings when a Newton step increased ``||R||``.
        0 disables the accept/reject test.
    verbose : bool
        Rank-0 per-step diagnostics.
    """

    #: No probe iterates are ever emitted; the driver's convergence test
    #: reads this unconditionally.
    probing: bool = False

    def __init__(self, jvp_factory, warmup: int = 5,
                 switch_tol: float = 1e-2, beta: float = 0.2,
                 max_krylov: int = 30, inner_tol: float = 0.1,
                 forcing: str = "ew", max_newton: int = 100,
                 trust: float = 0.5, trust_max: float = 0.0,
                 newton_damp: float = 1.0, backtrack: int = 3,
                 verbose: bool = True) -> None:
        self._jvp_factory = jvp_factory
        self.warmup = int(warmup)
        self.switch_tol = float(switch_tol)
        self.beta = float(beta)
        self.max_krylov = int(max_krylov)
        self.inner_tol = float(inner_tol)
        self.forcing = str(forcing)
        self.max_newton = int(max_newton)
        self.trust = float(trust)
        self.trust_max = max(float(trust_max), float(trust))
        self.newton_damp = float(newton_damp)
        self.backtrack = int(backtrack)
        self.verbose = bool(verbose)

        self._jvp = None
        self._it = 0
        self._newton_it = 0
        self._R_first = None
        self._trust_k = float(trust)
        # Pending accept/reject bookkeeping: the iterate emitted by the
        # last Newton step, its base point, base residual and step.
        self._pending = None    # (xk, delta, Rk_norm, halvings)
        self._comm, self._SUM = get_comm()

    # ---- distributed primitives -----------------------------------------
    def _dot(self, u, v):
        return global_dot(self._comm, self._SUM, u, v)

    def _gnorm(self, v):
        return global_norm(self._comm, self._SUM, v)

    def _log(self, msg: str) -> None:
        if self.verbose and (self._comm is None or self._comm.rank == 0):
            print(msg, flush=True)

    # ---- the mixer entry point -------------------------------------------
    def step(self, x: np.ndarray, gx: np.ndarray) -> np.ndarray:
        self._it += 1
        R = gx - x
        rk = self._gnorm(_c2r(R))
        if self._R_first is None:
            self._R_first = rk

        # Accept/reject the previous Newton step on its actual residual.
        # A non-finite trial residual (diverged map evaluation) counts as
        # rejected -- NaN comparisons would otherwise silently accept it.
        if self._pending is not None:
            xk, delta, Rk, rk_base, halvings = self._pending
            worse = (not np.isfinite(rk)) or rk >= rk_base
            if worse and self.backtrack > 0:
                # Non-finite trials (diverged map) shrink the radius harder.
                shrink = 0.25 if not np.isfinite(rk) else 0.5
                self._trust_k = max(self._trust_k * shrink, 1e-3)
                if halvings < self.backtrack:
                    frac = 0.5 ** (halvings + 1)
                    self._pending = (xk, delta, Rk, rk_base, halvings + 1)
                    self._log(f"  newton: ||R|| {rk_base:.3e} -> {rk:.3e}, "
                              f"backtrack to {frac:.3g}*delta")
                    return xk + frac * delta
                # Exhausted: damped Picard from the BASE point (whose
                # residual we stored). The trust radius has been halved at
                # every rejection, so the next Newton attempt is shorter.
                self._pending = None
                self._log("  newton: backtracking exhausted, damped Picard "
                          "from the base point")
                return xk + self.beta * Rk
            # Accepted: adapt the trust radius with hysteresis. A solidly
            # accepted step recovers the radius (else a collapsed radius
            # after a rejection burst never regrows and the iteration
            # flatlines); a marginal accept keeps it.
            if rk < 0.95 * rk_base:
                self._trust_k = min(self._trust_k * 1.3, self.trust_max)
            elif rk < 0.999 * rk_base:
                self._trust_k = min(self._trust_k * 1.05, self.trust_max)
            self._pending = None

        # Warm phase: damped Picard into the Newton basin.
        in_basin = rk <= self.switch_tol * self._R_first
        if self._it <= self.warmup or not in_basin:
            return x + self.beta * R
        if self._newton_it >= self.max_newton:
            return x + self.beta * R

        return self._newton_step(x, gx, R, rk)

    # ---- one synchronous Newton step ---------------------------------------
    def _newton_step(self, x, gx, R, rk):
        if self._jvp is None:
            self._jvp = self._jvp_factory()
        recon = self._jvp.prepare()

        # Eisenstat-Walker forcing from the previous Newton residual.
        # The base ``inner_tol`` is the CEILING: forcing may only tighten
        # the inner solve as the outer residual falls, never loosen it
        # (a loose direction on a stagnating residual gets rejected by
        # the backtracking test, which shrinks the trust region, which
        # stalls the residual -- a self-sustaining deadlock).
        if self.forcing == "ew" and getattr(self, "_rk_prev", None):
            ew = 0.9 * (rk / self._rk_prev) ** 2
            inner_tol = float(min(self.inner_tol, max(1e-4, ew)))
        else:
            inner_tol = self.inner_tol
        self._rk_prev = rk

        xk_r = _c2r(x)
        b = -_c2r(R)
        beta_g = rk
        m_max = self.max_krylov
        V = [b / beta_g]
        Hg = np.zeros((m_max + 1, m_max), dtype=np.float64)
        g = np.zeros(m_max + 1, dtype=np.float64)
        g[0] = beta_g
        cs, sn = [], []
        inner_res = beta_g
        m = 0
        for j in range(m_max):
            # (J_F - I) v, real embedding; the JVP is R-linear.
            Av = _c2r(self._jvp.apply(_r2c(V[j]))) - V[j]
            for i in range(j + 1):
                h = self._dot(V[i], Av)
                Hg[i, j] = h
                Av = Av - h * V[i]
            hjp = self._gnorm(Av)
            Hg[j + 1, j] = hjp
            for i in range(j):
                a_, b_ = Hg[i, j], Hg[i + 1, j]
                Hg[i, j] = cs[i] * a_ + sn[i] * b_
                Hg[i + 1, j] = -sn[i] * a_ + cs[i] * b_
            a_, b_ = Hg[j, j], Hg[j + 1, j]
            r_ = float(np.hypot(a_, b_))
            c_, s_ = (1.0, 0.0) if r_ == 0.0 else (a_ / r_, b_ / r_)
            cs.append(c_)
            sn.append(s_)
            Hg[j, j] = c_ * a_ + s_ * b_
            Hg[j + 1, j] = 0.0
            g[j + 1] = -s_ * g[j]
            g[j] = c_ * g[j]
            inner_res = abs(g[j + 1])
            m = j + 1
            if inner_res <= inner_tol * beta_g:
                break
            if hjp <= 1e-13 * max(beta_g, 1e-300):
                break
            V.append(Av / hjp)

        Rt = np.triu(Hg[:m, :m])
        try:
            y = np.linalg.solve(Rt, g[:m])
        except np.linalg.LinAlgError:
            y = np.linalg.lstsq(Rt, g[:m], rcond=None)[0]
        delta_r = np.zeros_like(xk_r)
        for i in range(m):
            delta_r = delta_r + y[i] * V[i]
        if not np.all(np.isfinite(delta_r)):
            self._log("  newton: non-finite delta, damped Picard fallback")
            return x + self.beta * R
        step_r = trust_cap(self._comm, self._SUM,
                           self.newton_damp * delta_r, xk_r, self._trust_k)

        self._newton_it += 1
        self._log(f"  newton#{self._newton_it}: gmres_m={m} "
                  f"inner_res/||R||={inner_res / (beta_g + 1e-300):.2e} "
                  f"(tol {inner_tol:.2e}) ||R||={rk:.3e} "
                  f"||delta||={self._gnorm(step_r):.3e} "
                  f"trust={self._trust_k:.2g} recon={recon:.1e}")

        delta = _r2c(step_r)
        if self.backtrack > 0:
            self._pending = (x, delta, R, rk, 0)
        return x + delta
