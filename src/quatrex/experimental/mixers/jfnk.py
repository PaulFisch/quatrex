# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Jacobian-free Newton-Krylov (JFNK) for the STRONGLY-unstable SCBA fixed point.

The anharmonic-phonon SCBA is one fixed-point map ``Sigma = F[Sigma] = B[G[Sigma]]``.
When the map Jacobian ``J_F = dF/dSigma`` has a SMALL number of eigenvalues
with ``|lambda| >> 1``, every contraction-based scheme -- Picard, Anderson,
RRE -- diverges, and the DMD-subspace Newton
(:class:`quatrex.experimental.mixers.rpm.RPMMixer`) also fails because it must explicitly
identify that rank-noisy, high-``|lambda|`` outlier subspace.

JFNK never identifies the unstable subspace. It applies NEWTON to the residual
``R(Sigma) = F[Sigma] - Sigma`` -- ``J delta = -R`` with ``J = J_F - I`` -- and
solves the Newton system with GMRES, which only needs Jacobian-VECTOR products,
formed matrix-free by a finite difference of the map::

    J v = (F[Sigma + eps v] - F[Sigma]) / eps - v .

The Jacobian spectrum here is ideal for GMRES: a tight cluster near ``-1`` (the
contractive bulk ``|lambda(J_F)| < 1`` -> ``lambda(J) ~ -1``) plus a few large
real/complex outliers. GMRES converges in roughly ``n_outliers`` + a few
iterations regardless of how large the outliers are.

Real embedding (correctness). ``F`` is NOT complex-analytic in ``Sigma`` -- it
conjugates (``Sigma^R`` via Kramers-Kronig/Hilbert, ``G^A = (G^R)^H``, the bubble
leg structure), so ``dF/dSigma`` is REAL-linear, not complex-linear
(``J(i v) != i J(v)``). Complex GMRES would build a corrupt Krylov space, so the
whole Newton-Krylov solve runs in the real embedding ``[Re Sigma, Im Sigma]``
(length ``2n``); only the probe handed back to the map is re-complexified.

Embedding in the SCBA loop (one map eval per iteration). The driver evaluates
``F`` once per SCBA iteration and hands the mixer ``(x, gx=F[x])``. JFNK runs as a
STATE MACHINE that emits exactly one next-iterate per call and carries the
GMRES/Newton state across iterations:

  * ``warmup``  : a few damped-Picard steps to fall into the fixed point's basin
                  (the unstable modes have not yet blown up early on).
  * ``arnoldi`` : each call consumes ``F[x_k + eps v_j]`` to form ``J v_j``, takes
                  one Arnoldi/Givens step, and emits the next probe
                  ``x_k + eps v_{j+1}`` -- until the GMRES residual meets the
                  (Eisenstat-Walker) inner tolerance or ``max_krylov``.
  * accept      : assemble ``delta = V y``, emit the damped, trust-capped Newton
                  iterate ``x_{k+1} = x_k + lambda * delta``; its map value
                  ``F[x_{k+1}]`` (next call) opens the next Newton step.

MPI. The iterate is row-partitioned over ``MPI.COMM_WORLD`` (disjoint slices). All
inner products / norms that contract the distributed dimension -- the Arnoldi
``<v_i, w>``, ``||.||``, the trust cap -- are ``Allreduce(SUM)`` of scalars; the
tiny ``(m x m)`` Hessenberg solve is replicated and bit-identical, so the Krylov
basis and ``x_new = x_k + V y`` stay globally consistent with no extra comms.
Same ``step(x, gx)`` signature as the other mixers (drop-in,
``config.scba.mixing_method = "jfnk"``).
"""

import numpy as np

from quatrex.core.mpi_linalg import get_comm, global_dot, global_norm, trust_cap


def _c2r(z: np.ndarray) -> np.ndarray:
    """Complex (n,) -> real (2n,) embedding ``[Re z, Im z]``."""
    return np.concatenate([z.real, z.imag]).astype(np.float64)


def _r2c(r: np.ndarray) -> np.ndarray:
    """Real (2n,) -> complex (n,): inverse of :func:`_c2r`."""
    n = r.size // 2
    return (r[:n] + 1j * r[n:]).astype(np.complex128)


class JFNKMixer:
    """Jacobian-free Newton-Krylov mixer over flattened, row-partitioned iterates.

    Parameters
    ----------
    warmup : int
        Damped-Picard steps before engaging Newton-Krylov (basin capture).
    beta : float
        Picard damping used in warmup and any fallback step.
    max_krylov : int
        Maximum GMRES (Arnoldi) dimension per Newton step = max map evals per
        Newton step. ``n_unstable + a few`` suffices; 30 leaves margin.
    inner_tol : float
        Base relative GMRES tolerance ``||J delta + R|| <= inner_tol ||R||``.
    forcing : {"ew", "fixed"}
        Inexact-Newton forcing. "ew" (Eisenstat-Walker type) tightens the inner
        tolerance as the outer residual falls; "fixed" keeps ``inner_tol``.
    max_newton : int
        Cap on outer Newton steps (each ~ ``GMRES iters`` map evals).
    eps : float
        Relative finite-difference step ``eps_used = eps * (1 + ||x_k||)``.
    trust : float
        Trust-region cap: ``||delta|| <= trust * ||x_k||`` (global). 0 disables.
    newton_damp : float
        Newton step damping ``x_{k+1} = x_k + newton_damp * delta`` (after cap).
    verbose : bool
        Rank-0 per-Newton-step diagnostics.
    """

    #: True while the emitted iterate is a finite-difference PROBE rather than
    #: an accepted Newton iterate. The driver must not test convergence on it.
    probing: bool = False

    def __init__(self, warmup: int = 10, beta: float = 0.3,
                 max_krylov: int = 30, inner_tol: float = 0.1,
                 forcing: str = "ew", max_newton: int = 60, eps: float = 1e-7,
                 trust: float = 0.5, newton_damp: float = 1.0,
                 ptc: float = 0.0, trust_max: float = 0.0,
                 verbose: bool = True) -> None:
        self.warmup = int(warmup)
        self.beta = float(beta)
        self.max_krylov = int(max_krylov)
        self.inner_tol = float(inner_tol)
        self.forcing = str(forcing)
        self.max_newton = int(max_newton)
        self.eps = float(eps)
        self.trust = float(trust)
        # A fixed 1e-3 lower bound used to *increase* a deliberately smaller
        # requested trust radius after the first rejected trial.  Scale the
        # floor from the requested radius instead: ordinary/default runs keep
        # an effectively identical safeguard, while near-root continuation
        # can legitimately ask for 1e-4 or smaller Newton steps.
        self._trust_floor = min(1e-3, max(1e-12, 1e-3 * self.trust))
        # The radius is allowed to GROW from the (small, safe) initial
        # ``trust`` up to ``trust_max`` as the residual descends.
        # ``trust_max <= trust`` => no growth.
        self.trust_max = max(float(trust_max), float(trust))
        self.newton_damp = float(newton_damp)
        # Pseudo-transient continuation / Levenberg-Marquardt shift: solve
        # (J + mu I) delta = -R instead of J delta = -R, with mu annealed to
        # 0 as the outer residual falls (mu_k = ptc * ||R_k||/||R_0||). The
        # shift lifts the near-zero (marginal) eigenvalues of J off the
        # origin so GMRES no longer stalls on the near-null-space; -> 0 at
        # the root recovers pure Newton.
        self.ptc = float(ptc)
        self._mu = 0.0
        self.verbose = bool(verbose)

        self._it = 0
        self._phase = "warmup"
        # Newton state (real embedding)
        self._xk_r = None       # accepted iterate, real (2n,)
        self._Fxk_r = None      # F(xk), real (2n,)
        self._Rk_norm = None
        self._Rk_merit = None   # absolute infinity norm of the root residual
        self._R0_norm = None
        self._Rprev_norm = None
        self._newton_it = 0
        self._trust_k = float(trust)
        # GMRES state (real)
        self._V: list[np.ndarray] = []
        self._Hg = None
        self._g = None
        self._cs: list[float] = []
        self._sn: list[float] = []
        self._j = 0
        self._eps_k = 1.0
        self._beta_g = 1.0
        self._inner_tol_k = float(inner_tol)
        self._pending_v = None
        # The iterate emitted after a completed GMRES solve is a TRIAL.  Its
        # actual map residual arrives on the next mixer call.  Keep the base
        # state so a non-descent trial can be rejected without accepting a
        # bad point into the next Krylov linearisation.
        self._trial_pending = False
        self._trial_rejections = 0
        self._comm, self._SUM = get_comm()

    # ---- MPI-correct distributed primitives (row-partitioned real vectors) ---
    # The iterate is in the real embedding, so the inner product is the plain
    # real dot; ``global_dot`` (vdot.real) reduces to exactly that for real input.
    def _dot(self, u: np.ndarray, v: np.ndarray) -> float:
        return global_dot(self._comm, self._SUM, u, v)

    def _gnorm(self, v: np.ndarray) -> float:
        return global_norm(self._comm, self._SUM, v)

    def _inf_residual(self, x: np.ndarray, gx: np.ndarray) -> float:
        """Global infinity norm of the fixed-point root residual.

        The production stopping test divides this numerator by the current
        self-energy scale.  That moving denominator is unsuitable for a line
        search: a trial can reduce every absolute residual while the ratio
        rises because ``||F(x)||_inf`` fell faster.  Descent of the numerator
        is the actual root-merit condition; the unchanged driver still applies
        its relative tolerance when deciding convergence.
        """
        local = np.array([
            float(np.max(np.abs(gx - x), initial=0.0)),
        ], dtype=np.float64)
        if self._comm is not None and self._comm.size > 1:
            from mpi4py import MPI
            reduced = np.empty_like(local)
            self._comm.Allreduce(local, reduced, op=MPI.MAX)
            local = reduced
        return float(local[0])

    def _cap(self, step: np.ndarray, x: np.ndarray, radius: float) -> np.ndarray:
        return trust_cap(self._comm, self._SUM, step, x, radius)

    def _log(self, msg: str) -> None:
        if self.verbose and (self._comm is None or self._comm.rank == 0):
            print(msg, flush=True)

    # ---- the state machine ---------------------------------------------------
    def step(self, x: np.ndarray, gx: np.ndarray) -> np.ndarray:
        """One mixer call: ``x`` is the iterate just evaluated, ``gx = F[x]``
        (both complex, flattened, row-local). Returns the next iterate (complex)."""
        self._it += 1
        self.probing = False

        # WARM-UP: damped Picard to fall into the basin (the unstable modes are
        # still small in the first few iterations).
        if self._it <= self.warmup:
            if self._it == self.warmup:
                self._log(f"  JFNK: warmup done at it={self._it}, "
                          f"||R||={self._gnorm(_c2r(gx - x)):.3e}"
                          f" -> engaging Newton-Krylov")
            return x + self.beta * (gx - x)

        # Globalise Newton with the same residual-merit test used by the
        # synchronous exact-JVP solver.  Previously a worsening trial was
        # accepted and only shrank the radius at the *next* base point.  On a
        # strongly unstable fixed point that loses the basin the trust region
        # was intended to preserve.  A rejected trial is discarded; reopen
        # GMRES at the stored base with a smaller radius.
        if self._trial_pending:
            trial_norm = self._gnorm(_c2r(gx - x))
            base_norm = float(self._Rk_norm)
            trial_merit = self._inf_residual(x, gx)
            base_merit = float(self._Rk_merit)
            self._trial_pending = False
            if self.trust > 0.0 and (
                (not np.isfinite(trial_merit)) or trial_merit >= base_merit
            ):
                shrink = 0.25 if not np.isfinite(trial_merit) else 0.5
                self._trust_k = max(self._trust_k * shrink, self._trust_floor)
                self._trial_rejections += 1
                self._log(
                    f"  JFNK reject: ||R||_inf {base_merit:.3e} -> "
                    f"{trial_merit:.3e}; ||R|| {base_norm:.3e} -> "
                    f"{trial_norm:.3e}; trust -> {self._trust_k:.2g}"
                )
                # The exact base map value is already stored.  Emit the first
                # finite-difference probe of the retry directly, avoiding a
                # redundant evaluation of F(x_k).
                return self._open_newton_step(
                    _r2c(self._xk_r), _r2c(self._Fxk_r)
                )
            self._trial_rejections = 0

        # Out of Newton budget -> gentle Picard (the best-conserved iterate is
        # already captured by the SCBA driver).
        if self._newton_it >= self.max_newton:
            return x + self.beta * (gx - x)

        if self._phase in ("warmup", "newton_base"):
            return self._open_newton_step(x, gx)
        if self._phase == "arnoldi":
            return self._arnoldi_consume(gx)
        return x + self.beta * (gx - x)

    def _open_newton_step(self, x: np.ndarray, gx: np.ndarray) -> np.ndarray:
        """Begin a Newton step at accepted iterate ``x`` (``gx=F[x]``): set up
        GMRES for ``J delta = -R`` in the real embedding, return the first probe."""
        xk_r = _c2r(x)
        Fxk_r = _c2r(gx)
        R = Fxk_r - xk_r
        self._xk_r = xk_r
        self._Fxk_r = Fxk_r
        rk = self._gnorm(R)
        self._Rprev_norm = self._Rk_norm
        self._Rk_norm = rk
        self._Rk_merit = self._inf_residual(x, gx)
        if self._R0_norm is None:
            self._R0_norm = rk

        # Trust-region adaptation from the previous Newton step's progress.
        # Asymmetric, with hysteresis: shrink HARD on any worsening (overshoot
        # of the unstable mode), but GROW on any solid monotone descent up to
        # ``trust_max``. Only when the trust region is enabled -- adapting a
        # disabled one (trust = 0) would silently switch it back on.
        if self.trust > 0.0 and self._Rprev_norm is not None:
            if rk > self._Rprev_norm:            # step made it worse -> shrink
                self._trust_k = max(self._trust_k * 0.5, self._trust_floor)
            elif rk < 0.95 * self._Rprev_norm:   # strong progress -> grow quickly
                self._trust_k = min(self._trust_k * 1.3, self.trust_max)
            elif rk < 0.999 * self._Rprev_norm:  # accepted descent -> recover gently
                self._trust_k = min(self._trust_k * 1.05, self.trust_max)

        # Eisenstat-Walker forcing: tighten the inner solve as R falls.
        if self.forcing == "ew" and self._Rprev_norm:
            ew = 0.9 * (rk / self._Rprev_norm) ** 2
            self._inner_tol_k = float(min(0.5, max(1e-3, ew)))
        else:
            self._inner_tol_k = self.inner_tol

        # Pseudo-transient / LM shift, annealed with the outer residual (SER).
        self._mu = (self.ptc * rk / (self._R0_norm + 1e-300)) if self.ptc > 0 else 0.0

        if rk == 0.0:                            # already a fixed point
            self._phase = "newton_base"
            return x

        # GMRES init: A = J + mu I, b = -R, x0 = 0  ->  r0 = -R, v1 = r0/||r0||.
        self._eps_k = self.eps * (1.0 + self._gnorm(xk_r))
        self._beta_g = rk
        v1 = (-R) / rk
        self._V = [v1]
        m = self.max_krylov
        self._Hg = np.zeros((m + 1, m), dtype=np.float64)
        self._g = np.zeros(m + 1, dtype=np.float64)
        self._g[0] = rk
        self._cs = []
        self._sn = []
        self._j = 0
        self._pending_v = v1
        self._newton_it += 1
        self._phase = "arnoldi"
        self.probing = True
        return _r2c(xk_r + self._eps_k * v1)

    def _arnoldi_consume(self, gx_probe: np.ndarray) -> np.ndarray:
        """Consume ``F[x_k + eps v_j]`` -> ``J v_j``, advance GMRES one step,
        emit the next probe or (on inner convergence) the Newton iterate."""
        j = self._j
        v = self._pending_v
        # Matrix-free real (J + mu I) v = (F(xk+eps v)-F(xk))/eps - (1-mu) v.
        Av = (_c2r(gx_probe) - self._Fxk_r) / self._eps_k - (1.0 - self._mu) * v

        # Modified Gram-Schmidt against the basis (global inner products).
        for i in range(j + 1):
            h = self._dot(self._V[i], Av)
            self._Hg[i, j] = h
            Av = Av - h * self._V[i]
        hjp = self._gnorm(Av)
        self._Hg[j + 1, j] = hjp

        # Apply stored Givens rotations to the new Hessenberg column.
        for i in range(j):
            a, b = self._Hg[i, j], self._Hg[i + 1, j]
            self._Hg[i, j] = self._cs[i] * a + self._sn[i] * b
            self._Hg[i + 1, j] = -self._sn[i] * a + self._cs[i] * b
        # New Givens rotation to annihilate H[j+1, j].
        a, b = self._Hg[j, j], self._Hg[j + 1, j]
        r = float(np.hypot(a, b))
        if r == 0.0:
            cs, sn = 1.0, 0.0
        else:
            cs, sn = a / r, b / r
        self._cs.append(cs)
        self._sn.append(sn)
        self._Hg[j, j] = cs * a + sn * b
        self._Hg[j + 1, j] = 0.0
        self._g[j + 1] = -sn * self._g[j]
        self._g[j] = cs * self._g[j]
        inner_res = abs(self._g[j + 1])

        converged = inner_res <= self._inner_tol_k * self._beta_g
        breakdown = hjp <= 1e-13 * max(self._beta_g, 1e-300)
        last = (j + 1 >= self.max_krylov)
        if converged or breakdown or last:
            return self._assemble_and_step(j + 1, inner_res)

        v_next = Av / hjp
        self._V.append(v_next)
        self._j += 1
        self._pending_v = v_next
        self.probing = True
        return _r2c(self._xk_r + self._eps_k * v_next)

    def _assemble_and_step(self, m: int, inner_res: float) -> np.ndarray:
        """Solve the ``m x m`` triangular Hessenberg system, build
        ``delta = V y``, return the damped, trust-capped Newton iterate (complex)."""
        R = np.triu(self._Hg[:m, :m])
        try:
            y = np.linalg.solve(R, self._g[:m])
        except np.linalg.LinAlgError:
            y = np.linalg.lstsq(R, self._g[:m], rcond=None)[0]
        delta = np.zeros_like(self._xk_r)
        for i in range(m):
            delta = delta + y[i] * self._V[i]
        if not np.all(np.isfinite(delta)):
            self._phase = "newton_base"
            return _r2c(self._xk_r + self.beta * (self._Fxk_r - self._xk_r))
        step = self._cap(self.newton_damp * delta, self._xk_r, self._trust_k)
        x_new_r = self._xk_r + step
        self._log(f"  JFNK newton#{self._newton_it}: gmres_m={m} "
                  f"inner_res/||R||={inner_res / (self._beta_g + 1e-300):.2e} "
                  f"||R||={self._Rk_norm:.3e} ||delta||={self._gnorm(step):.3e} "
                  f"trust={self._trust_k:.2g} mu={self._mu:.2e}")
        self._phase = "newton_base"
        self._trial_pending = True
        return _r2c(x_new_r)

