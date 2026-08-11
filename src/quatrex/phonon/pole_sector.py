# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Driver for the pole-subtracted SCBA sector.

Ties together the pieces that each have their own module: the complex-frequency
continuation of the scattering self-energy
(:mod:`~quatrex.phonon.pole_kernel`), the nonlinear eigenvalue solve
(:mod:`~quatrex.phonon.pole_nevp`), continuation across SCBA iterations
(:mod:`~quatrex.phonon.pole_tracker`), the cluster Keldysh matrix
(:mod:`~quatrex.phonon.pole_keldysh`) and the analytic pole-pole bubble
(:mod:`~quatrex.phonon.pole_bubble`).

One update does:

1. rebuild ``Delta`` from the mixed self-energy buffers (never cached -- it must
   not drift out of step with the ``Sigma^R`` the Dyson operator was built from);
2. seed from the previous iterate through the first-order predictor, or from a
   contour scan when tracking has been lost;
3. correct each pole with bordered Newton and accept on the scaled residual;
4. apply the promotion criteria -- resolution, isolation, conditioning, band-edge
   distance and vertex-weighted importance -- with hysteresis, and only at an
   epoch boundary;
5. group the survivors into coherent clusters and project the Keldysh source
   onto each.

The result is a set of clusters that the SSE subtracts from its bubble legs and
adds back analytically. The split is exact -- ``G_S + G_R`` is the untouched
``G`` -- so this changes the representation, not the diagram.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qttools import NDArray, xp

from quatrex.phonon.pole_bubble import bosonic_closure
from quatrex.phonon.pole_keldysh import (
    PoleCluster,
    coherence_metric,
    occupation_matrix,
    project_source,
)
from quatrex.phonon.pole_kernel import sigma_retarded_at_z
from quatrex.phonon.pole_nevp import PoleSolution, bordered_newton
from quatrex.phonon.pole_probe import assemble_m_blocks, nnz_to_blocks
from quatrex.phonon.pole_tracker import PoleTracker, cluster_poles, predict_shift

__all__ = ["PoleSectorState", "PoleSector"]


@dataclass
class PoleSectorState:
    """What one update produced, and why.

    ``source_lesser``/``source_greater`` are the projected Keldysh sources
    ``S = V^dagger Sigma_tot V``, one small matrix per cluster; ``g_pp_*`` are
    the pole-sector legs already reduced to the stored sparsity pattern and
    summed over clusters, ready to be subtracted from the bubble legs.
    """

    clusters: list[PoleCluster] = field(default_factory=list)
    legs: list[PoleCluster] = field(default_factory=list)
    """Clusters CLOSED under the bosonic partner map, as the bubble needs them.

    Kept separate from ``clusters`` because the two have different jobs: the
    tracker matches against the solved set, while every leg the bubble builds
    must come from a set closed under ``z -> -z^*`` or the fold
    ``Sigma^<(q,-w) = Sigma^>(-q,w)^T`` does not hold. Sources are indexed
    alongside THIS list, not ``clusters``."""
    source_lesser: list = field(default_factory=list)
    source_greater: list = field(default_factory=list)
    g_pp_lesser: object = None
    g_pp_greater: object = None
    solutions: list[PoleSolution] = field(default_factory=list)
    rejected: list[tuple[complex, str]] = field(default_factory=list)
    coherence: list[float] = field(default_factory=list)
    iteration: int = 0

    @property
    def n_poles(self) -> int:
        return sum(c.n_poles for c in self.clusters)

    def report(self) -> str:
        """One-line-per-cluster summary for the iteration log."""
        lines = [f"pole sector: iteration {self.iteration}, "
                 f"{len(self.clusters)} cluster(s), {self.n_poles} pole(s)"]
        for c, eps in zip(self.clusters, self.coherence):
            om = np.asarray(_host(c.omega)).ravel()
            ga = np.asarray(_host(c.gamma)).ravel()
            lines.append(
                "  " + ", ".join(f"{o:.4f}-{g:.2e}i" for o, g in zip(om, ga))
                + f"   eps_coh={eps:.3f}"
            )
        for z, why in self.rejected:
            lines.append(f"  rejected {z.real:.4f}-{-z.imag:.2e}i: {why}")
        return "\n".join(lines)


def _host(a):
    return a.get() if hasattr(a, "get") else a


class PoleSector:
    """Stateful driver over SCBA iterations.

    Parameters
    ----------
    config : PoleSectorConfig
        The ``phonon.pole_sector`` block.
    freqs : NDArray
        The uniform convolution grid the continuation is built on.
    band_edges : NDArray, optional
        Known contact band edges / branch points (THz). Poles closer than
        ``edge_factor`` half-widths to one are refused: a band edge is a branch
        point, not a simple pole, and forcing one into a single-pole fit is the
        documented failure mode.

    """

    def __init__(self, config, freqs: NDArray, band_edges: NDArray | None = None,
                 *, global_freqs: NDArray | None = None, freq_offset: int = 0,
                 reduce=None):
        """
        Parameters
        ----------
        freqs : NDArray
            The frequencies THIS rank owns.
        global_freqs : NDArray, optional
            The whole frequency axis. The continuation sums over all of it, so
            the weights are built here even though only the local columns are
            contracted. Defaults to ``freqs`` (serial).
        freq_offset : int, optional
            Index of ``freqs[0]`` within ``global_freqs``.
        reduce : callable, optional
            ``arr -> summed arr`` across the frequency communicator. Defaults
            to the identity (serial).

        """
        self.cfg = config
        self.freqs = xp.asarray(freqs, dtype=float)
        self.global_freqs = (self.freqs if global_freqs is None
                             else xp.asarray(global_freqs, dtype=float))
        self._freq_offset = int(freq_offset)
        self._reduce = (lambda a: a) if reduce is None else reduce
        self.h = float(_host(self.global_freqs)[1] - _host(self.global_freqs)[0])
        self.band_edges = (None if band_edges is None
                           else np.asarray(_host(band_edges), dtype=float))
        self.tracker = PoleTracker(
            cluster_factor=config.cluster_factor,
            angle_tol=config.subspace_angle_tol,
            rescan_iterations=config.rescan_iterations,
            epoch_iterations=config.epoch_iterations,
        )
        self.state = PoleSectorState()
        # Promoted poles are remembered by LOCATION, not by object identity.
        # bordered_newton builds a fresh PoleSolution every iteration, so a set
        # of id(sol) can never match across iterations -- and CPython recycles
        # ids of freed objects, so it could match arbitrarily. That left the
        # hysteresis permanently disengaged: every pole was judged at the
        # strict q_in instead of the lenient q_out once promoted, and
        # membership churned exactly as this config field warns.
        self._promoted_z: list[complex] = []
        # Fit anchor for the current pole solve. Pinning it makes M(z) a
        # genuine analytic function of z for the whole Newton iteration;
        # deriving the stencil from Re z instead makes M only PIECEWISE
        # holomorphic, with a measured 17 % jump at each stencil boundary.
        self._fit_anchor: float | None = None

    # -- window ------------------------------------------------------------ #

    def window(self, low_freq_mask: float = 0.0) -> tuple[float, float]:
        """Resolved pole-search window (THz).

        ``omega_min_thz = 0`` resolves to ``max(4*dw, mask + 2*dw)``: below that
        the quasiparticle picture does not apply, and the continuation has no cut
        where the self-energy has been masked to zero.
        """
        lo = self.cfg.omega_min_thz or max(4.0 * self.h, low_freq_mask + 2.0 * self.h)
        hi = self.cfg.omega_max_thz or float(_host(self.freqs)[-1])
        return float(lo), float(hi)

    # -- criteria ---------------------------------------------------------- #

    def resolution_score(self, gamma: float) -> float:
        r"""``q_omega = gamma / (p_Gamma * h)`` (doc Eq. 125).

        Below one the mode is under-resolved by the grid and is a promotion
        candidate; above it the grid already carries the line.
        """
        return gamma / (self.cfg.samples_per_halfwidth * self.h)

    def _was_promoted(self, z: complex) -> bool:
        """Was a pole at this location in the sector last iteration?

        Identity across SCBA iterations is carried by position: a pole moves by
        at most the predictor shift between iterates, so anything within a
        cluster width of a previously promoted pole is the same mode. This is
        what makes the ``q_in``/``q_out`` hysteresis actually apply.
        """
        if not self._promoted_z:
            return False
        tol = self.cfg.cluster_factor * self.h
        return any(abs(complex(z) - p) <= tol for p in self._promoted_z)

    def screen(self, sol: PoleSolution, was_promoted: bool) -> str | None:
        """Reason to refuse a pole, or ``None`` to accept it.

        Hysteresis is applied here: a mode already in the sector is only demoted
        once it is comfortably resolved, so membership cannot oscillate.
        """
        if not sol.converged:
            return f"eps_nep={sol.eps_nep:.2e} above tolerance"
        gamma = -sol.z.imag
        if gamma <= 0.0:
            return "pole is not in the lower half plane"
        lo, hi = self.window()
        if not (lo <= sol.z.real <= hi):
            return f"outside the pole window [{lo:.3g}, {hi:.3g}]"
        q = self.resolution_score(gamma)
        threshold = self.cfg.q_out if was_promoted else self.cfg.q_in
        if q >= threshold:
            return f"grid-resolved (q_omega={q:.3g} >= {threshold})"
        if sol.kappa > self.cfg.condition_reject:
            return f"ill-conditioned (kappa={sol.kappa:.2e})"
        if self.band_edges is not None and self.band_edges.size:
            d = float(np.min(np.abs(self.band_edges - sol.z.real)))
            if d < self.cfg.edge_factor * gamma:
                return f"within {d / gamma:.2g} half-widths of a band edge"
        return None

    # -- update ------------------------------------------------------------ #

    def solve_poles(
        self, m_blocks, dm_blocks, seeds: list[complex],
        seed_vectors: list[NDArray] | None = None,
    ) -> list[PoleSolution]:
        """Correct a list of seeds with the bordered Newton iteration."""
        out = []
        saved = self._fit_anchor
        try:
            for k, z0 in enumerate(seeds):
                r0 = None if seed_vectors is None else seed_vectors[k]
                # Pin the fit centre and the contact sample to THIS seed for
                # the whole Newton solve, so the operator Newton differentiates
                # is the operator it evaluates.
                self._fit_anchor = float(complex(z0).real)
                out.append(
                    bordered_newton(
                        m_blocks, dm_blocks, z0, r0,
                        tol=self.cfg.newton_tol,
                        max_iter=self.cfg.newton_max_iterations,
                        trust_radius=self.cfg.trust_radius_cells * self.h,
                    )
                )
        finally:
            self._fit_anchor = saved
        return out

    def predict(self, dsigma_at: dict[int, NDArray]) -> list[complex]:
        """Warm-start seeds from the previous iterate (doc Eq. 43)."""
        seeds = []
        for k, sol in enumerate(self.state.solutions):
            ds = dsigma_at.get(k)
            shift = 0.0 if ds is None else predict_shift(sol.l, sol.r, ds)
            seeds.append(sol.z + shift)
        return seeds

    def build_clusters(
        self,
        solutions: list[PoleSolution],
        sigma_lesser: NDArray | None = None,
        sigma_greater: NDArray | None = None,
        omega: NDArray | None = None,
    ) -> PoleSectorState:
        """Group accepted poles into coherent clusters and project the source."""
        accepted, rejected = [], []
        for sol in solutions:
            why = self.screen(sol, self._was_promoted(sol.z))
            (rejected if why else accepted).append(
                (sol.z, why) if why else sol
            )
        if len(accepted) > self.cfg.max_poles:
            accepted.sort(key=lambda s: -abs(np.asarray(_host(s.r))).max())
            for sol in accepted[self.cfg.max_poles:]:
                rejected.append((sol.z, "over max_poles"))
            accepted = accepted[: self.cfg.max_poles]

        clusters, coherence = [], []
        if accepted:
            z = xp.asarray([s.z for s in accepted], dtype=xp.complex128)
            u = xp.stack([s.r for s in accepted], axis=1)
            v = xp.stack([s.l for s in accepted], axis=1)
            for gi, grp in enumerate(cluster_poles(z, self.cfg.cluster_factor)):
                idx = xp.asarray(grp)
                cl = PoleCluster(
                    z=xp.take(z, idx), u=xp.take(u, idx, axis=1),
                    v=xp.take(v, idx, axis=1), label=f"c{gi}",
                )
                clusters.append(cl)
                eps = float("nan")
                if sigma_lesser is not None and omega is not None:
                    s = project_source(sigma_lesser, cl.v)
                    eps = coherence_metric(occupation_matrix(omega, cl, s))
                coherence.append(eps)

        self.state = PoleSectorState(
            clusters=clusters, solutions=accepted, rejected=rejected,
            coherence=coherence, iteration=self.state.iteration + 1,
        )
        self._promoted_z = [complex(s.z) for s in accepted]
        return self.state

    # -- operator context --------------------------------------------------- #

    def set_operator_context(
        self, *, delta, d_blocks, obc_left, obc_right, block_sizes, rows, cols
    ) -> None:
        r"""Store everything :math:`M(z)` needs for this SCBA iteration.

        ``delta`` is ``Sigma^> - Sigma^<`` on the full frequency grid; the rest
        is the operator's frequency-independent part plus the contact blocks.
        The contacts are held at their real-axis value: continuing the lead
        self-energy off the axis is the outgoing-sheet work, and on the physical
        sheet -- which is where the currently supported pole classes live, in a
        contact gap or weakly coupled -- the contact contribution is flat over a
        window narrower than its own scale.
        """
        d = xp.asarray(delta)
        if d.ndim > 2:
            nq = int(np.prod(d.shape[1:-1]))
            if nq != 1:
                raise NotImplementedError(
                    f"pole_sector: coupled-q ({nq} q-points) is not wired yet; "
                    "the pole set is per-q and the vertex fold has to follow it."
                )
            d = d.reshape(d.shape[0], d.shape[-1])
        self._delta = d
        self._d_blocks = d_blocks
        # The contact blocks arrive on the WHOLE frequency grid,
        # ``(n_freq, b, b)``. M(z) is a single matrix, so they have to be
        # sampled at one frequency; holding them flat at the grid point
        # nearest Re z is exactly the approximation this docstring states.
        # Passing the full array instead silently assembles M at every
        # frequency at once and hands the bordered Newton a stack it cannot
        # interpret -- which is what happened on the first production run.
        self._obc = tuple(
            None if o is None else xp.asarray(o) for o in (obc_left, obc_right)
        )
        self._block_sizes = np.asarray(_host(block_sizes), dtype=int)
        self._rows, self._cols = rows, cols

    def continue_sigma(self, z: NDArray, order: int = 0) -> NDArray:
        r"""``Sigma_s^{R,II}(z)`` (or its ``order``-th derivative) on the pattern.

        Distributed by construction. Both terms of the continuation --
        ``F(z)`` and the second-sheet ``Delta_an(z)`` -- are LINEAR in Delta
        with coefficients that depend only on the grid, and neither reindexes
        the frequency axis. So the weights are built for the global grid, each
        rank contracts only the columns it owns, and one sum-reduction
        completes the answer. No transposition of the distributed buffer is
        needed, and in the serial case the reduction is the identity, so this
        is bit-identical to the undistributed path.
        """
        from quatrex.phonon.pole_kernel import (
            contract_delta, continuation_weights, local_fit_weights,
        )

        zz = xp.asarray(z, dtype=xp.complex128).reshape(-1)
        w_pos, w_mir = continuation_weights(zz, self.global_freqs, order=order)
        f_pos, f_mir = local_fit_weights(
            self.global_freqs, zz, order=self.cfg.delta_fit_order,
            window=self.cfg.delta_fit_window_cells, deriv=order,
            anchor=self._fit_anchor,
        )
        lo = self._freq_offset
        hi = lo + int(self._delta.shape[0])
        val = contract_delta(self._delta,
                             w_pos[:, lo:hi] + f_pos[:, lo:hi],
                             w_mir[:, lo:hi] + f_mir[:, lo:hi])
        return self._reduce(val)

    def _sigma_blocks_at(self, z: complex, order: int):
        """Scattering self-energy blocks at complex ``z``."""
        val = self.continue_sigma(xp.asarray([z]), order=order)
        return nnz_to_blocks(val, self._rows, self._cols, self._block_sizes,
                             band=1)

    def _obc_at(self, z: complex):
        """Contact blocks sampled at the grid frequency nearest ``Re z``.

        A leading frequency axis is reduced here and nowhere else, so
        ``assemble_m_blocks`` always sees one matrix per block.
        """
        out = []
        for o in self._obc:
            if o is None or o.ndim < 3:
                out.append(o)
                continue
            # Pinned to the same anchor as the fit: sampling the contact at
            # the point nearest Re z would reintroduce a jump into M(z).
            ref = z.real if self._fit_anchor is None else self._fit_anchor
            k = int(np.argmin(np.abs(np.asarray(_host(self.freqs)) - ref)))
            out.append(o[k])
        return out[0], out[1]

    def operator(self):
        """``(m_blocks, dm_blocks)`` closures over the stored context."""

        def m_blocks(z):
            sig = {k: v[0] for k, v in self._sigma_blocks_at(z, 0).items()}
            obc_l, obc_r = self._obc_at(z)
            return assemble_m_blocks(
                z, self._d_blocks, sig, obc_left=obc_l, obc_right=obc_r,
                block_sizes=self._block_sizes,
            )

        def dm_blocks(z):
            sig = {k: v[0] for k, v in self._sigma_blocks_at(z, 1).items()}
            n = len(self._block_sizes)
            a_ii = [
                2.0 * z * xp.eye(int(self._block_sizes[i]), dtype=xp.complex128)
                - sig[(i, i)] for i in range(n)
            ]
            a_ij = [-sig[(i, i + 1)] for i in range(n - 1)]
            a_ji = [-sig[(i + 1, i)] for i in range(n - 1)]
            return a_ii, a_ij, a_ji

        return m_blocks, dm_blocks

    def harmonic_seeds(self) -> list[complex]:
        r"""Initial pole guesses from the harmonic spectrum and the golden rule.

        The real part comes from the eigenvalues of the frequency-independent
        part. The imaginary part must NOT be a fixed guess: the linewidths this
        sector exists for are orders of magnitude below the grid spacing, and a
        seed that overestimates them by a factor of a few hundred starts the
        corrector outside the pole's basin. Instead use the quasiparticle
        estimate obtained by taking the imaginary part of
        :math:`z^2 - \Omega_0^2 - \Sigma^R = 0` at :math:`z = \Omega - i\gamma`,

        .. math:: \gamma \simeq -\frac{\operatorname{Im}\Sigma^R(\Omega)}{2\Omega},

        which needs no Hilbert transform: with :math:`\Delta = -i\Gamma` and
        :math:`\Gamma` real, :math:`\operatorname{Im}\Sigma^R = \tfrac12
        \operatorname{Im}\Delta` exactly -- the Kramers-Kronig half is purely
        real. So :math:`\gamma \simeq -\operatorname{Im}
        (v^\dagger \Delta v)/(4\Omega)`, read straight off the grid.
        """
        n_dof = int(np.sum(self._block_sizes))
        off = np.concatenate(([0], np.cumsum(self._block_sizes)))
        dense = np.zeros((n_dof, n_dof), dtype=complex)
        for (i, j), b in self._d_blocks.items():
            dense[off[i]:off[i + 1], off[j]:off[j + 1]] = np.asarray(_host(b))
        lam, vec = np.linalg.eigh(0.5 * (dense + dense.conj().T))

        w = np.asarray(_host(self.freqs), dtype=float)
        delta = np.asarray(_host(self._delta))
        rows = np.asarray(_host(self._rows), dtype=int)
        cols = np.asarray(_host(self._cols), dtype=int)

        lo, hi = self.window()
        seeds: list[complex] = []
        for k, l in enumerate(lam.real):
            if l <= 0.0:
                continue
            om = float(np.sqrt(l))
            if not (lo <= om <= hi):
                continue
            # Delta at Omega, linearly interpolated from the grid.
            idx = int(np.clip(np.searchsorted(w, om), 1, w.size - 1))
            f = (om - w[idx - 1]) / (w[idx] - w[idx - 1])
            vals = (1.0 - f) * delta[idx - 1] + f * delta[idx]
            mat = np.zeros((n_dof, n_dof), dtype=complex)
            mat[rows, cols] = vals
            v = vec[:, k]
            gamma = -float(np.imag(np.vdot(v, mat @ v))) / (4.0 * om)
            gamma = max(gamma, 1e-12)
            seeds.append(complex(om, -gamma))
        return seeds

    def refresh(self) -> PoleSectorState:
        """One SCBA iteration's worth of pole tracking.

        Predictor -> corrector -> subspace match -> contour fallback, in that
        order. The seed comes from the first-order response of the previous
        poles to the change in the scattering self-energy (doc Eq. 43), which
        costs one projection; the harmonic/golden-rule estimate is used only on
        the first iteration or after tracking has been lost.

        Sector membership is held fixed within an adaptation epoch: an
        approximate implementation is not invariant under repartitioning, so a
        mode that changes sector every iteration changes the fixed-point map
        itself.
        """
        m_blocks, dm_blocks = self.operator()

        seeds, vectors = self._seed()
        sols = self.solve_poles(m_blocks, dm_blocks, seeds, vectors)

        # A failed correction is a tracking failure, not a reason to carry a
        # pole: re-seed from the harmonic spectrum and try once more.
        if self.state.solutions and not any(s.converged for s in sols):
            self.tracker.rescan_reasons.append("no pole survived the corrector")
            seeds, vectors = self.harmonic_seeds(), None
            sols = self.solve_poles(m_blocks, dm_blocks, seeds, vectors)

        state = self.build_clusters(sols)
        self._track(state)
        self._prev_delta = xp.array(self._delta, copy=True)
        return state

    def _seed(self):
        """Warm-start seeds, or the harmonic estimate when there is no history."""
        prev = self.state.solutions
        if not prev or self.tracker.needs_rescan():
            return self.harmonic_seeds(), None
        dsigma = self._delta_sigma_at([s.z for s in prev])
        seeds = []
        for k, sol in enumerate(prev):
            shift = 0.0 if dsigma is None else predict_shift(sol.l, sol.r, dsigma[k])
            if abs(shift) > self.cfg.trust_radius_cells * self.h:
                shift *= self.cfg.trust_radius_cells * self.h / abs(shift)
            seeds.append(sol.z + shift)
        return seeds, [s.r for s in prev]

    def _delta_sigma_at(self, z_list):
        r"""``Delta Sigma_s^R(z)`` between consecutive iterates, as dense blocks.

        The predictor needs the CHANGE in the self-energy, which is linear in
        the change in ``Delta`` -- so it is the same continuation applied to the
        difference, and costs one extra contraction rather than a re-solve.
        """
        prev = getattr(self, "_prev_delta", None)
        if prev is None or prev.shape != self._delta.shape:
            return None
        diff = self._delta - prev
        if float(xp.abs(diff).max()) == 0.0:
            return None
        out = []
        for z in z_list:
            blocks = self._sigma_blocks_at_from(diff, complex(z), 0)
            n = len(self._block_sizes)
            off = np.concatenate(([0], np.cumsum(self._block_sizes)))
            dense = xp.zeros((int(off[-1]),) * 2, dtype=xp.complex128)
            for (i, j), b in blocks.items():
                dense[off[i]:off[i + 1], off[j]:off[j + 1]] = b[0]
            out.append(dense)
        return out

    def _sigma_blocks_at_from(self, delta, z: complex, order: int):
        """:meth:`_sigma_blocks_at` against an explicit ``Delta``.

        Routed through the same distributed contraction, so the predictor sees
        the whole frequency axis too. Continuing a rank-local slice here would
        make the predicted shift depend on the rank decomposition.
        """
        saved, self._delta = self._delta, delta
        try:
            val = self.continue_sigma(xp.asarray([z]), order=order)
        finally:
            self._delta = saved
        return nnz_to_blocks(val, self._rows, self._cols, self._block_sizes,
                             band=1)

    def _track(self, state: PoleSectorState) -> None:
        """Match this iteration's clusters to the previous ones by subspace.

        Individual eigenvectors rotate arbitrarily near a crossing while the
        invariant subspace stays smooth, so identity is carried by principal
        angles rather than by frequency order.
        """
        if not state.clusters:
            return
        z = xp.concatenate([c.z for c in state.clusters])
        r = xp.concatenate([c.u for c in state.clusters], axis=1)
        l = xp.concatenate([c.v for c in state.clusters], axis=1)
        if self.tracker.clusters and not self.tracker.needs_rescan():
            self.tracker.update(z, r, l)
        else:
            self.tracker.adopt(z, r, l)
            self.tracker.iteration += 1

    # -- consumption ------------------------------------------------------- #

    def bubble_clusters(self) -> list[PoleCluster]:
        """Clusters as the bubble needs them: closed under the bosonic partner.

        The fold ``Sigma^<(q, -w) = Sigma^>(-q, w)^T`` only holds if the pole set
        used to build the legs contains every partner, so the closure is applied
        here rather than being left to the caller to remember.
        """
        return [bosonic_closure(c) for c in self.state.clusters]
