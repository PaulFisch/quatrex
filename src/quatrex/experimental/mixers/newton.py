# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Newton-Krylov mixer with EXACT analytic Jacobian-vector products.

Same Newton system as :mod:`quatrex.experimental.mixers.jfnk` -- ``(J_F - I) delta = -R``
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
    allreduce_sum,
    complex_to_real as _c2r,
    get_comm,
    global_dot,
    global_norm,
    real_to_complex as _r2c,
    trust_cap,
)

__all__ = ["NewtonKrylovMixer"]


class _LowRankPrecond:
    """Right preconditioner from stored exact Jacobian actions.

    Given r globally-orthonormal real directions Q (rows) and their
    exact map images Yk = J_F Q, the rank-r surrogate K~ = Yk Q^T of
    J_F gives, by the Woodbury identity, the exact inverse of the
    surrogate Newton operator I - K~:

        M^{-1} v = v + Yk^T D (Q v),   D = (I_r - Q Yk^T)^{-1}.

    On the stored subspace the preconditioned operator (J_F - I) M^{-1}
    acts as -I, so GMRES sees the deflated spectrum: the dominant
    outliers collapse onto the bulk cluster. Application cost is one
    r-vector Allreduce and a rank-r update; the small D is replicated
    (bit-identical control flow on all ranks).
    """

    def __init__(self, Q: np.ndarray, Yk: np.ndarray, comm, op_sum):
        self._Q = Q          # (r, 2n) real, globally orthonormal rows
        self._Yk = Yk        # (r, 2n) real, Yk[i] = J_F Q[i]
        self._comm, self._SUM = comm, op_sum
        r = Q.shape[0]
        gram = allreduce_sum(comm, op_sum,
                             np.ascontiguousarray(Q @ Yk.T))
        small = np.eye(r) - gram
        try:
            self._D = np.linalg.inv(small)
        except np.linalg.LinAlgError:
            self._D = np.linalg.pinv(small)

    @property
    def rank(self) -> int:
        return int(self._Q.shape[0])

    def apply(self, v: np.ndarray) -> np.ndarray:
        c = allreduce_sum(self._comm, self._SUM,
                          np.ascontiguousarray(self._Q @ v))
        return v + self._Yk.T @ (self._D @ c)


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
        Maximum step-halvings when a Newton step increased the scale-free
        residual ``||R|| / ||F(x)||``. 0 disables the accept/reject test.
    precond : {"none", "recycle", "fresh"}
        Low-rank right preconditioner for the inner GMRES. ``recycle``
        harvests harmonic Ritz pairs (the near-singular directions) of
        the previous Newton step's Arnoldi relation -- exact operator
        images at ZERO extra JVPs. ``fresh`` builds the basis at the
        current step (residual + recent updates + previous basis) and
        spends ``precond_rank`` exact JVPs on their images. Memory:
        2 x rank extra state vectors.
    precond_rank : int
        Rank of the stored deflation basis.
    verbose : bool
        Rank-0 per-step diagnostics.
    """

    probing: bool = False

    def __init__(self, jvp_factory, warmup: int = 5,
                 switch_tol: float = 1e-2, beta: float = 0.2,
                 max_krylov: int = 30, inner_tol: float = 0.1,
                 forcing: str = "ew", max_newton: int = 100,
                 trust: float = 0.5, trust_max: float = 0.0,
                 newton_damp: float = 1.0, backtrack: int = 3,
                 precond: str = "none", precond_rank: int = 8,
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
        if precond not in ("none", "recycle", "fresh"):
            raise ValueError(f"Unknown precond={precond!r}.")
        self.precond = str(precond)
        self.precond_rank = int(precond_rank)
        self.verbose = bool(verbose)

        self._jvp = None
        self._it = 0
        self._newton_it = 0
        self._R_first = None
        self._trust_k = float(trust)
        self._pending = None
        # Low-rank deflation state.
        self._precond_op: _LowRankPrecond | None = None
        self._prev_deltas: list[np.ndarray] = []   # fresh-basis seeds
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
        gk = self._gnorm(_c2r(gx))
        merit = rk / max(gk, 1e-300)
        if self._R_first is None:
            self._R_first = rk

        if self._pending is not None:
            xk, delta, Rk, rk_base, merit_base, halvings = self._pending
            worse = (not np.isfinite(merit)) or merit >= merit_base
            if worse and self.backtrack > 0:
                # Non-finite trials (diverged map) shrink the radius harder.
                shrink = 0.25 if not np.isfinite(merit) else 0.5
                self._trust_k = max(self._trust_k * shrink, 1e-3)
                if halvings < self.backtrack:
                    frac = 0.5 ** (halvings + 1)
                    self._pending = (xk, delta, Rk, rk_base, merit_base,
                                     halvings + 1)
                    self._log(
                        f"  newton: relative residual {merit_base:.3e} -> "
                        f"{merit:.3e}, backtrack to {frac:.3g}*delta")
                    return xk + frac * delta
                self._pending = None
                self._log("  newton: backtracking exhausted, damped Picard "
                          "from the base point")
                return xk + self.beta * Rk
            if merit < 0.95 * merit_base:
                self._trust_k = min(self._trust_k * 1.3, self.trust_max)
            elif merit < 0.999 * merit_base:
                self._trust_k = min(self._trust_k * 1.05, self.trust_max)
            self._pending = None

        # Warm phase: damped Picard into the Newton basin.
        in_basin = rk <= self.switch_tol * self._R_first
        if self._it <= self.warmup or not in_basin:
            return x + self.beta * R
        if self._newton_it >= self.max_newton:
            return x + self.beta * R

        return self._newton_step(x, gx, R, rk, merit)

    # ---- one synchronous Newton step ---------------------------------------
    def _newton_step(self, x, gx, R, rk, merit):
        if self._jvp is None:
            self._jvp = self._jvp_factory()
        recon = self._jvp.prepare()

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

        step_frac = getattr(self, "_last_step_frac", None)
        stale = step_frac is None or step_frac > 0.05
        if self.precond == "fresh":
            M = self._build_fresh_precond(b / beta_g)
        elif self.precond == "recycle" and not stale:
            M = self._precond_op
        else:
            M = None

        V = [b / beta_g]
        Hg = np.zeros((m_max + 1, m_max), dtype=np.float64)
        H_raw = np.zeros((m_max + 1, m_max), dtype=np.float64)
        g = np.zeros(m_max + 1, dtype=np.float64)
        g[0] = beta_g
        cs, sn = [], []
        inner_res = beta_g
        m = 0
        v_tail = None
        for j in range(m_max):
            # (J_F - I) M^{-1} v, real embedding; the JVP is R-linear.
            z = M.apply(V[j]) if M is not None else V[j]
            Av = _c2r(self._jvp.apply(_r2c(z))) - z
            for i in range(j + 1):
                h = self._dot(V[i], Av)
                Hg[i, j] = h
                H_raw[i, j] = h
                Av = Av - h * V[i]
            hjp = self._gnorm(Av)
            Hg[j + 1, j] = hjp
            H_raw[j + 1, j] = hjp
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
            v_tail = Av / hjp if hjp > 0.0 else None
            if inner_res <= inner_tol * beta_g:
                break
            if hjp <= 1e-13 * max(beta_g, 1e-300):
                break
            V.append(v_tail)

        Rt = np.triu(Hg[:m, :m])
        try:
            y = np.linalg.solve(Rt, g[:m])
        except np.linalg.LinAlgError:
            y = np.linalg.lstsq(Rt, g[:m], rcond=None)[0]
        u_r = np.zeros_like(xk_r)
        for i in range(m):
            u_r = u_r + y[i] * V[i]
        delta_r = M.apply(u_r) if M is not None else u_r
        if not np.all(np.isfinite(delta_r)):
            self._log("  newton: non-finite delta, damped Picard fallback")
            return x + self.beta * R
        step_r = trust_cap(self._comm, self._SUM,
                           self.newton_damp * delta_r, xk_r, self._trust_k)
        xnorm = self._gnorm(xk_r)
        self._last_step_frac = (self._gnorm(step_r) / xnorm
                                if xnorm > 0.0 else None)

        if self.precond == "recycle":
            self._precond_op = self._harvest_ritz(V, v_tail, H_raw, m, M)

        self._newton_it += 1
        pc = "none" if M is None else f"{self.precond}(r={M.rank})"
        self._log(f"  newton#{self._newton_it}: gmres_m={m} "
                  f"inner_res/||R||={inner_res / (beta_g + 1e-300):.2e} "
                  f"(tol {inner_tol:.2e}) ||R||={rk:.3e} "
                  f"||delta||={self._gnorm(step_r):.3e} "
                  f"trust={self._trust_k:.2g} precond={pc} recon={recon:.1e}")

        # Seed directions for the fresh-basis variant.
        if self.precond == "fresh":
            nrm = self._gnorm(step_r)
            if nrm > 0.0 and np.all(np.isfinite(step_r)):
                self._prev_deltas.append(step_r / nrm)
                self._prev_deltas = self._prev_deltas[-3:]

        delta = _r2c(step_r)
        if self.backtrack > 0:
            self._pending = (x, delta, R, rk, merit, 0)
        return x + delta

    # ---- low-rank deflation machinery ---------------------------------------
    def _harvest_ritz(self, V, v_tail, H_raw, m, M):
        """Harmonic Ritz pairs of the (preconditioned) Arnoldi relation,
        mapped back to exact (direction, J_F direction) pairs.

        The relation A~ V_m = V_{m+1} Hbar with A~ = (J_F - I) M^{-1}
        gives, for any small combination u~ = V_m gvec, the exact image
        A (M^{-1} u~) = V_{m+1} Hbar gvec -- so the harvested pairs are
        exact regardless of the preconditioner in effect. Harmonic Ritz
        targets the smallest-|theta| eigenvalues: the near-singular
        directions that stall GMRES.
        """
        r = min(self.precond_rank, m)
        if r < 1:
            return self._precond_op
        H = H_raw[:m, :m]
        e_m = np.zeros(m)
        e_m[m - 1] = 1.0
        h2 = float(H_raw[m, m - 1]) ** 2
        try:
            f = np.linalg.solve(H.T, e_m)
        except np.linalg.LinAlgError:
            f = np.linalg.lstsq(H.T, e_m, rcond=None)[0]
        theta, Gv = np.linalg.eig(H + h2 * np.outer(f, e_m))
        h_last = float(H_raw[m, m - 1])
        quality = np.abs(h_last * Gv[m - 1, :]) / np.maximum(
            np.abs(theta) * np.linalg.norm(Gv, axis=0), 1e-300)
        converged = quality < 0.1
        by_small = [i for i in np.argsort(np.abs(theta)) if converged[i]]
        by_large = by_small[::-1]
        order = []
        for a_, b_ in zip(by_small, by_large):
            order.extend((a_, b_))
        cols: list[np.ndarray] = []
        used: set[int] = set()
        for idx in order:
            if len(cols) >= r:
                break
            if int(idx) in used:
                continue
            used.add(int(idx))
            th, gvec = theta[idx], Gv[:, idx]
            if abs(th.imag) <= 1e-12 * max(abs(th.real), 1e-300):
                cols.append(np.real(gvec))
            else:
                cols.append(np.real(gvec))
                if len(cols) < r:
                    cols.append(np.imag(gvec))
                for idx2 in order:
                    if int(idx2) not in used and np.isclose(
                            theta[idx2], np.conj(th)):
                        used.add(int(idx2))
                        break
        if not cols:
            return self._precond_op
        G = np.array(cols, dtype=np.float64).T      # (m, r_eff)
        r_eff = G.shape[1]

        Vfull = V + ([v_tail] if len(V) == m and v_tail is not None else [])
        n_rows = min(len(Vfull), m + 1)
        Himg = H_raw[:n_rows, :m] @ G               # (n_rows, r_eff)
        U, Yk = [], []
        for jcol in range(r_eff):
            u = np.zeros_like(V[0])
            for i in range(m):
                u = u + G[i, jcol] * V[i]
            if M is not None:
                u = M.apply(u)
            ya = np.zeros_like(V[0])
            for i in range(n_rows):
                ya = ya + Himg[i, jcol] * Vfull[i]
            U.append(u)
            Yk.append(ya + u)          # J_F u = (J_F - I) u + u
        if self._precond_op is not None:
            U += list(self._precond_op._Q)
            Yk += list(self._precond_op._Yk)
        Q_rows, Y_rows = [], []
        for u, yk in zip(U, Yk):
            if len(Q_rows) >= self.precond_rank:
                break
            u = u.copy()
            yk = yk.copy()
            for q, yq in zip(Q_rows, Y_rows):
                h = self._dot(q, u)
                u = u - h * q
                yk = yk - h * yq
            nrm = self._gnorm(u)
            if nrm <= 1e-8:
                continue
            Q_rows.append(u / nrm)
            Y_rows.append(yk / nrm)
        if not Q_rows:
            return self._precond_op
        return _LowRankPrecond(np.array(Q_rows), np.array(Y_rows),
                               self._comm, self._SUM)

    def _build_fresh_precond(self, r0_unit):
        """The proposal's literal variant: orthonormal basis from the
        current residual, recent Newton updates and the previous basis;
        exact images via ``precond_rank`` fresh JVPs."""
        cands = [r0_unit] + list(self._prev_deltas)
        if self._precond_op is not None:
            cands += list(self._precond_op._Q)
        Q_rows = []
        for u in cands:
            if len(Q_rows) >= self.precond_rank:
                break
            u = u.copy()
            for q in Q_rows:
                u = u - self._dot(q, u) * q
            nrm = self._gnorm(u)
            if nrm <= 1e-10:
                continue
            Q_rows.append(u / nrm)
        if not Q_rows:
            return None
        Y_rows = [_c2r(self._jvp.apply(_r2c(q))) for q in Q_rows]
        op = _LowRankPrecond(np.array(Q_rows), np.array(Y_rows),
                             self._comm, self._SUM)
        self._precond_op = op
        return op
