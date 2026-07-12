# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Recursive Projection Method (RPM) for the iteration-unstable SCBA fixed point.

A fixed-point iteration ``x <- g(x)`` that diverges/limit-cycles because a
FEW eigenvalues of the map Jacobian ``G' = dg/dx`` have ``|lambda| >= 1`` is
stabilised by performing NEWTON on the small unstable invariant subspace
``P = ZZ^H`` and ordinary PICARD on the contractive complement ``Q = I - P``:

    x_{n+1} = g(x) + Z (I_k - H)^{-1} Z^H (x - g(x))
            = g(x) - Z (I_k - H)^{-1} (Z^H f),     f = g(x) - x,

where ``Z`` (n x k, k small) spans the unstable subspace and ``H = Z^H G' Z`` is
its k x k restricted Jacobian. At ``k = 0`` this is plain Picard; the correction
multiplies ``f`` so the true fixed point (``f = 0``) is exactly preserved.
``(I_k - H)`` is nonsingular precisely because no ``lambda(G') = 1`` -- the same
condition that makes the full ``I - G'`` invertible -- and it is a TINY k x k
solve done replicated on every rank.

Where :class:`quatrex.core.broyden.BroydenMixer` builds the inverse-Jacobian
action implicitly from a rolling secant window, RPM holds the unstable
directions in an explicit, persistently re-identified subspace -- more robust
when a complex pair sits on the unit circle and the secant window proves too
forgetful.

Identification (matrix-free, no extra map evaluations). The unstable
directions are the increment directions that do not contract under Picard. A
sliding buffer of increments ``dX = x_i - x_{i-1}`` and ``dG = g(x_i) -
g(x_{i-1}) ~ G' dX`` is kept and, each step past the warm-up, ``Z`` is
extracted from the leading singular vectors of the GLOBAL Gram ``dX^H dX``
(DMD on the increment buffer). A rotating complex pair appears as TWO
comparable singular values, so a sigma-threshold keeps the conjugate pair
automatically (the 2x2 ``H`` then carries the rotation). ``H`` is the
projected-secant fit ``Z^H dG = H (Z^H dX)``.

MPI. The iterate is ROW-PARTITIONED across ranks (disjoint slices, concatenation
over ``MPI.COMM_WORLD`` = the global vector). Every quantity that contracts the
distributed dimension -- the Gram ``dX^H dX`` (p x p), the projections
``Z^H dX``, ``Z^H dG``, ``Z^H f`` -- is an ``Allreduce(SUM)`` over small matrices;
the eigendecomposition, the ``H`` fit and the ``(I_k - H)^{-1}`` solve are
replicated and bit-identical, so ``Z`` and ``x_new`` stay globally consistent
with no further communication. ``x_new = g(x) - Z @ (small vector)`` is local.
"""

import numpy as np

from quatrex.core.mpi_linalg import (
    get_comm,
    global_gram,
    global_norm,
    real_embedded,
    trust_cap,
)


class RPMMixer:
    """Recursive Projection Method mixer over flattened, row-partitioned iterates.

    Drop-in for :class:`quatrex.core.anderson.AndersonMixer` (same ``step``
    signature), selected by ``config.scba.mixing_method = "rpm"``.

    Parameters
    ----------
    max_subspace : int
        Cap on the unstable-subspace dimension ``k`` (Newton on ``k`` modes,
        Picard on the complement). A single complex pair needs ``k = 2``;
        6 leaves margin.
    beta : float
        Damping of the Picard step (warm-up and the ``k = 0`` / fallback paths).
    ridge : float
        Tiny Tikhonov ridge on the ``(I_k - H)`` solve, scaled by its norm.
    warmup : int
        Damped-Picard iterations before engaging RPM (lets Picard contract the
        stable subspace so the increment buffer is dominated by the unstable
        modes -- the clean-buffer prerequisite for the ``H`` estimate).
    buffer : int
        Sliding increment-buffer length ``p`` (>= max_subspace + a few).
    eig_tol : float
        Relative singular-value cutoff for admitting a direction into ``Z``.
    """

    #: JFNK emits finite-difference PROBE iterates; this mixer never does, so
    #: the driver may always test convergence on its output.
    probing: bool = False

    def __init__(self, max_subspace: int = 6, beta: float = 0.3,
                 ridge: float = 1e-8, warmup: int = 25, buffer: int = 12,
                 eig_tol: float = 1e-3, trust: float = 0.3,
                 margin: float = 0.02, phi_max: float = 50.0,
                 patience: int = 8, progress_thresh: float = 0.9) -> None:
        self.max_subspace = int(max_subspace)
        self.beta = float(beta)
        self.ridge = float(ridge)
        self.warmup = int(warmup)
        self.buffer = max(int(buffer), self.max_subspace + 2)
        self.eig_tol = float(eig_tol)
        self.trust = float(trust)
        self.margin = float(margin)
        self.phi_max = float(phi_max)
        self.patience = int(patience)
        self.progress_thresh = float(progress_thresh)
        self._fhist: list[float] = []
        self._x_prev: np.ndarray | None = None
        self._gx_prev: np.ndarray | None = None
        self._dx: list[np.ndarray] = []
        self._dg: list[np.ndarray] = []
        self._it = 0
        self._comm, self._SUM = get_comm()

    def _gram(self, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Global ``U^H V`` (contracts the distributed rows) via Allreduce(SUM)."""
        return global_gram(self._comm, self._SUM, U, V)

    @real_embedded
    def step(self, x: np.ndarray, gx: np.ndarray) -> np.ndarray:
        """One RPM step; ``x`` is the (rank-local) iterate, ``gx = g(x)``."""
        f = gx - x
        if self._x_prev is not None:
            self._dx.append(x - self._x_prev)
            self._dg.append(gx - self._gx_prev)
            if len(self._dx) > self.buffer:
                self._dx.pop(0)
                self._dg.pop(0)
        self._x_prev = x.copy()
        self._gx_prev = gx.copy()
        self._it += 1

        # Track the global residual norm for the stall gate below.
        fnorm = global_norm(self._comm, self._SUM, f)
        self._fhist.append(fnorm)
        if len(self._fhist) > self.patience + 2:
            self._fhist.pop(0)

        # Warm-up / insufficient history: plain damped Picard.
        if self._it <= self.warmup or len(self._dx) < 2:
            return x + self.beta * f

        # STALL GATE: only engage the Newton correction when plain Picard has
        # PLATEAUED. While the residual keeps shrinking, Picard is doing the
        # right thing and the DMD H-estimate from the still-moving iterate is
        # unreliable (a blind Newton there hallucinates "unstable" modes).
        # Newton is needed ONLY for a mode that DEFEATS Picard (|lambda|>1),
        # whose signature is exactly a stalled/limit-cycling residual -- and
        # that is also where the increment buffer is cleanest.
        if self.patience > 0:
            if len(self._fhist) <= self.patience:
                # Not enough residual history to demonstrate a plateau --
                # keep accumulating increments under damped Picard.
                return x + self.beta * f
            ratio = fnorm / (self._fhist[0] + 1e-300)
            if ratio < self.progress_thresh:
                return x + self.beta * f   # Picard is still making progress

        try:
            dX = np.stack(self._dx, axis=1)   # (n_local, p)
            dG = np.stack(self._dg, axis=1)   # (n_local, p), dG ~ G' dX
            # Unstable subspace from the GLOBAL Gram of the increment buffer.
            M = self._gram(dX, dX)            # (p, p) Hermitian PSD, global
            w, V = np.linalg.eigh(M)
            w = np.clip(w.real, 0.0, None)
            order = np.argsort(w)[::-1]
            w, V = w[order], V[:, order]
            sig = np.sqrt(w)
            if sig[0] <= 0.0:
                return x + self.beta * f
            keep = int(np.count_nonzero(sig > self.eig_tol * sig[0]))
            k = min(self.max_subspace, keep)
            if k < 1:
                return x + self.beta * f
            V, sig = V[:, :k], sig[:k]
            Z = dX @ (V / sig)                # (n_local, k), Z^H Z = I_k (global)
            # Restricted Jacobian H = Z^H G' Z from the projected secants:
            #   Z^H dG = H (Z^H dX)  =>  H = (Z^H dG) pinv(Z^H dX).
            Ahat = self._gram(Z, dX)          # (k, p)
            Bhat = self._gram(Z, dG)          # (k, p)
            H = Bhat @ np.linalg.pinv(Ahat, rcond=1e-10)   # (k, k)
            Zf = self._gram(Z, f.reshape(-1, 1))[:, 0]     # (k,) global Z^H f
            # SAFEGUARDED MODAL NEWTON: the map can have a near-marginal mode
            # (lambda ~ 1, I - G' near-singular) on which a blind Newton
            # blows up but plain Picard converges (slowly). So Newton ONLY on
            # the genuinely UNSTABLE modes (|lambda(H)| > 1 + margin), Picard
            # on the rest. In the eigenbasis H = W diag(lam) W^{-1}, the RPM
            # correction (I_k-H)^{-1}-I_k has eigenvalues
            # phi(lam) = lam/(1-lam); it is zeroed on contractive/marginal
            # modes and |phi| is capped (the trust region then caps the
            # assembled step).
            try:
                lam, W = np.linalg.eig(H)
                Winv = np.linalg.inv(W)
            except np.linalg.LinAlgError:
                return x + self.beta * f
            unstable = np.abs(lam) > 1.0 + self.margin
            phi = np.where(unstable, lam / (1.0 - lam), 0.0 + 0.0j)
            mag = np.abs(phi)
            phi = np.where(mag > self.phi_max, phi / (mag + 1e-300) * self.phi_max,
                           phi)
            C = (W * phi) @ Winv               # W diag(phi) W^{-1}  (k, k)
            corr = C @ Zf                      # (k,)
            x_new = gx + Z @ corr
            if not np.all(np.isfinite(x_new)):
                return x + self.beta * f
            x_new = x + trust_cap(self._comm, self._SUM, x_new - x, x, self.trust)
            if (self._comm is None or self._comm.rank == 0) and self._it % 10 == 0:
                rho_H = float(np.max(np.abs(lam)))
                print(f"  RPM it={self._it}: k={k} |lam(H)|max={rho_H:.3f} "
                      f"n_unstable={int(np.count_nonzero(unstable))}", flush=True)
            return x_new
        except (np.linalg.LinAlgError, ValueError):
            return x + self.beta * f
