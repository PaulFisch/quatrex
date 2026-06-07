"""Safeguarded Anderson (Pulay/DIIS) acceleration for the SCBA fixed point.

The bare linear self-energy mixing ``Sigma <- (1-a) Sigma_prev + a Sigma_new``
oscillates on the strong-coupling SCBA fixed point of soft-mode structures
(e.g. the d5a SiNW): the residual is a near-period-2 mode that damped linear
mixing cannot kill. Anderson acceleration uses a short history of residual
secants to extrapolate the fixed point and breaks that oscillation.

This is a distributed-aware port of the dense reference
``phonon/solver/dense.py::_AndersonAccelerator``. The only change required for
the energy x block partitioned production self-energy is that every inner
product (the residual norm and the least-squares normal equations) is a GLOBAL
reduction: each rank holds a disjoint slice of the self-energy vector, so the
true dot product is the sum of the local dots over all ranks. We therefore
build the (depth x depth) Gram matrix ``dF^H dF`` and rhs ``dF^H f`` from
all-reduced local contributions and solve the small system locally -- this
reproduces the serial truncated-SVD least-squares to the regularisation
tolerance while needing only two small all-reduces per step.

Safeguards (identical to the dense reference): history kept across residual
upticks (Anderson may be non-monotone); ill-conditioned secants filtered by a
regularised solve; an over-long extrapolation replaced by the damped linear
step for that iteration; restart on genuine stagnation; revert-to-best on a
real divergence (so safeguarded Anderson is never worse than damped linear).
"""

from __future__ import annotations

import numpy as np

try:  # global reduction over the full (energy x block) rank grid
    from mpi4py.MPI import COMM_WORLD as _WORLD
    from mpi4py.MPI import SUM as _SUM
except Exception:  # pragma: no cover - mpi4py always present in this env
    _WORLD = None
    _SUM = None


class AndersonMixer:
    """Safeguarded distributed Anderson/Pulay mixer of a flat complex state.

    ``step(x_in, x_out)`` returns ``(x_mixed, fnorm)`` where ``x_in`` is the
    previous iterate, ``x_out`` the fixed-point map applied to it, and
    ``fnorm`` the GLOBAL residual norm. ``x_in``/``x_out`` are this rank's local
    slice of the global state vector.
    """

    def __init__(self, depth=5, beta=0.5, *, comm=None, cond_max=1e8,
                 step_cap=2.0, stagnation=5, revert_factor=2.0):
        self.depth = int(depth)
        self.beta = float(beta)
        self._comm = comm if comm is not None else _WORLD
        self.cond_max = float(cond_max)
        self.step_cap = float(step_cap)
        self.stagnation = int(stagnation)
        self.revert_factor = float(revert_factor)
        # Per-entry residual weights (block re-scaling); set by the caller after
        # the first step. None => unit weights. Used in BOTH the norm and the
        # least-squares fit.
        self.weights = None
        self.x_hist: list = []
        self.f_hist: list = []
        self.best_fnorm = float("inf")
        self.best_x = None
        self.n_since_best = 0

    # --- global inner products (sum of local dots over all ranks) ----------
    def _gsum(self, value):
        if self._comm is not None and self._comm.size > 1:
            return self._comm.allreduce(value, op=_SUM)
        return value

    def _gdot(self, a, b):
        """Global <a, b> = sum_ranks conj(a)·b."""
        return self._gsum(np.vdot(a, b))

    def fnorm(self, f):
        g = f if self.weights is None else self.weights * f
        return float(np.sqrt(max(self._gdot(g, g).real, 0.0)))

    def _truncate(self):
        while len(self.x_hist) > self.depth + 2:
            self.x_hist.pop(0)
            self.f_hist.pop(0)

    def step(self, x_in, x_out):
        x_in = np.asarray(x_in)
        f_k = np.asarray(x_out) - x_in
        fnorm = self.fnorm(f_k)

        if fnorm < self.best_fnorm:
            self.best_fnorm = fnorm
            self.best_x = np.array(x_in, copy=True)
            self.n_since_best = 0
        else:
            self.n_since_best += 1
            if (self.revert_factor > 0.0 and self.best_x is not None
                    and fnorm > self.revert_factor * self.best_fnorm):
                self.x_hist.clear()
                self.f_hist.clear()
                self.n_since_best = 0
                return np.array(self.best_x, copy=True), fnorm
        if self.n_since_best >= self.stagnation:
            self.x_hist.clear()
            self.f_hist.clear()
            self.n_since_best = 0

        self.x_hist.append(x_in)
        self.f_hist.append(f_k)

        x_lin = x_in + self.beta * f_k
        m = len(self.f_hist)
        if m < 2:
            return x_lin, fnorm

        n_use = min(m - 1, self.depth)
        dF = np.column_stack([
            self.f_hist[-n_use + j] - self.f_hist[-n_use + j - 1]
            for j in range(n_use)])
        dX = np.column_stack([
            self.x_hist[-n_use + j] - self.x_hist[-n_use + j - 1]
            for j in range(n_use)])

        # Block-weighted least squares min ||W(dF gamma - f)||: the weights
        # (1/sqrt(||block||) per (I,J) self-energy block) put every block on an
        # equal footing so the large diagonal blocks do not dominate the fit and
        # over-extrapolate the small off-diagonal ones. ESSENTIAL for the
        # soft-mode SCBA (the dense reference relies on it). The weights enter
        # only the fit; the step is applied to the unweighted dX, dF.
        if self.weights is not None:
            dF_w = dF * self.weights[:, None]
            f_w = f_k * self.weights
        else:
            dF_w, f_w = dF, f_k
        try:
            if self._comm is None or self._comm.size == 1:
                # Serial: the dense reference's direct truncated-SVD lstsq.
                gamma, *_ = np.linalg.lstsq(dF_w, f_w, rcond=1.0 / self.cond_max)
            else:
                # Distributed: the same least squares via the GLOBAL normal
                # equations (dF_w^H dF_w) gamma = dF_w^H f_w from all-reduced
                # local contractions, solved by a truncated pseudo-inverse of
                # the Hermitian Gram (its eigenvalues are the squared singular
                # values of dF_w). A small per-step communication.
                gram = self._gsum(dF_w.conj().T @ dF_w)
                rhs = self._gsum(dF_w.conj().T @ f_w)
                evals, evecs = np.linalg.eigh(0.5 * (gram + gram.conj().T))
                keep = evals > (evals.max() / self.cond_max if evals.size else 0.0)
                if not np.any(keep):
                    self._truncate()
                    return x_lin, fnorm
                gamma = evecs[:, keep] @ (
                    (evecs[:, keep].conj().T @ rhs) / evals[keep])
        except np.linalg.LinAlgError:
            self._truncate()
            return x_lin, fnorm

        x_and = x_lin - (dX + self.beta * dF) @ gamma

        lin_step = self.fnorm(x_lin - x_in) + 1e-300
        and_step = self.fnorm(x_and - x_in)
        x_mixed = x_lin if and_step > self.step_cap * lin_step else x_and
        self._truncate()
        return x_mixed, fnorm
