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
from quatrex.phonon.pole_tracker import (
    PoleTracker, cluster_poles, match_cost, match_poles, predict_shift,
)

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
    source_fit: list = field(default_factory=list)
    """Measured source variation across each cluster's pole window, the
    quantity ``source_fit_tol`` gates on."""
    source_lesser: list = field(default_factory=list)
    source_greater: list = field(default_factory=list)
    c_lesser: list = field(default_factory=list)
    c_greater: list = field(default_factory=list)
    """``(c_sr, c_rs, c_ss)`` per leg, from
    :func:`~quatrex.phonon.pole_congruence.background_coefficients`. Empty
    unless ``pole_sector.leg`` is a congruence route. ``c_ss`` duplicates
    ``source_*`` by construction -- both are ``V^dagger Sigma_tot V`` -- and
    the other two are what the mixed sectors need in place of the frozen
    Keldysh remainder."""
    pf_lesser: list = field(default_factory=list)
    pf_greater: list = field(default_factory=list)
    """``(zeta, p_row, q_col)`` per leg: the congruence flattened to ``2 Np``
    simple poles with rank-1 residues, which is what makes the analytic
    convolution affordable. Only under ``leg == "congruence_analytic"``."""
    residue_sum: list = field(default_factory=list)
    """``max |sum_p p_p q_p^T|`` per leg -- the ``1/w`` tail coefficient of
    the analytic leg, which the bosonic closure must drive to zero."""
    mixed_fit: list = field(default_factory=list)
    """``(eps_fit, eps_reg)`` per leg for the mixed coefficient ``c_rs``.

    Replaces a variation measure that asked the wrong question. Under the
    principal-part split ``c(omega)/(omega-z) = c(z)/(omega-z) +
    [c(omega)-c(z)]/(omega-z)`` the second term has a removable singularity,
    so ``c`` moving across the window is not an error. What can be an error is
    the local model's own residual (``eps_fit``) and whether the grid carries
    the regular remainder (``eps_reg``)."""
    g_pp_lesser: object = None
    g_pp_greater: object = None
    solutions: list[PoleSolution] = field(default_factory=list)
    rejected: list[tuple[complex, str]] = field(default_factory=list)
    coherence: list[float] = field(default_factory=list)
    iteration: int = 0
    _h_for_report: float = 0.0
    """Grid spacing, carried only so ``population()`` can report h/gamma."""

    @property
    def n_poles(self) -> int:
        return sum(c.n_poles for c in self.clusters)

    def coverage_chain(self) -> list[tuple[str, int]]:
        """How many candidates survive each stage of promotion, in order.

        Doc Sec. 28. A single yield ("2/144") says the sector is not carrying
        much; it does not say whether that is because the modes are already
        resolved, because the root solve failed, or because the representation
        was refused. Those have completely different fixes, and the chain
        separates them.

        The ``important`` stage is NOT implemented: ``weight_min`` exists in
        the config and nothing reads it, so no candidate is ever refused for
        carrying too little spectral or vertex-weighted weight. It is listed
        with its input count so the gap is visible rather than silently
        absent.
        """
        stages = [("in window", ("outside the pole window",
                                 "pole is not in the lower half plane")),
                  ("unresolved", ("grid-resolved",)),
                  ("important", ()),                   # not implemented
                  ("root solved", ("eps_z", "eps_nep")),
                  ("representation valid", ("ill-conditioned",
                                            "half-widths of a band edge")),
                  ("active", ("over max_poles",))]
        lost = {name: 0 for name, _ in stages}
        for _, why in self.rejected:
            for name, keys in stages:
                if any(k in why for k in keys):
                    lost[name] += 1
                    break
        n = self.n_poles + len(self.rejected)
        chain = [("candidates", n)]
        for name, _ in stages:
            n -= lost[name]
            chain.append((name, n))
        return chain

    def population(self) -> dict[str, float]:
        r"""Is there a population of NARROW, ISOLATED modes to extract at all?

        Two ratios decide whether the sector can do anything, and both are
        properties of the physics rather than of the solver:

        ``h_over_gamma``
            How badly the grid mis-weights the line. A ``dw``-weighted sum of
            point samples carries 98-102 % of a Lorentzian's total weight at
            ``h/gamma = 1.35`` and between 15 % and 660 % at ``h/gamma = 20``
            -- so below roughly 3 there is nothing for an exact cell average
            to recover.
        ``gamma_over_spacing``
            Whether a simple pole exists to be found. Above about one half the
            line overlaps its neighbour, no isolated pole is well defined, and
            a bordered Newton reports exactly that by failing to localise.

        On the CNT bed at 300 K the refused candidates have median
        ``h/gamma = 1.35`` and median ``gamma/spacing = 2.67``, with 85 %
        overlapping. That is not a screening failure: the bed has no narrow
        isolated resonances, and the low yield is the correct answer.
        """
        z = [complex(s.z) for s in self.solutions]
        z += [complex(a) for a, _ in self.rejected]
        if not z:
            return {}
        om = np.asarray([x.real for x in z])
        ga = np.asarray([max(-x.imag, 1e-300) for x in z])
        d = np.abs(om[:, None] - om[None, :])
        np.fill_diagonal(d, np.inf)
        nn = np.maximum(d.min(axis=1), 1e-300)
        return {"h_over_gamma": float(np.median(self._h_for_report / ga)),
                "gamma_over_spacing": float(np.median(ga / nn)),
                "frac_overlapping": float((ga / nn > 0.5).mean())}

    def report(self) -> str:
        """One-line-per-cluster summary for the iteration log.

        The header carries the promotion YIELD, not just the count. On the CNT
        bed it reads ``2/144``: 142 candidates are refused on ``eps_nep``
        alone, and that -- not any property of the sector's quadrature -- is
        why the route moves the answer only in the fourth digit and why
        pole-cell pairs carry under 0.005 % of the ring's weight. A bare "2
        pole(s)" reads like a small system; "2/144" reads like a threshold.
        """
        seen = self.n_poles + len(self.rejected)
        why = {}
        for _, reason in self.rejected:
            key = reason.split("=")[0].split(":")[0].strip()
            why[key] = why.get(key, 0) + 1
        tail = ("" if not why else "  refused: " + ", ".join(
            f"{k} x{v}" for k, v in sorted(why.items(), key=lambda kv: -kv[1])))
        lines = [f"pole sector: iteration {self.iteration}, "
                 f"{len(self.clusters)} cluster(s), "
                 f"{self.n_poles}/{seen} pole(s) promoted{tail}",
                 "  coverage: " + " -> ".join(
                     f"{k} {v}" for k, v in self.coverage_chain())]
        pop = self.population()
        if pop:
            lines.append(
                f"  population: median h/gamma={pop['h_over_gamma']:.2f} "
                f"(needs >~3 for the grid to be wrong), median "
                f"gamma/spacing={pop['gamma_over_spacing']:.2f}, "
                f"{100 * pop['frac_overlapping']:.0f}% overlapping "
                f"(needs <0.5 for an isolated pole to exist)")
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
        self._promoted: list[tuple[complex, NDArray]] = []
        # Two modes count as the same one only if their eigenvectors are not
        # nearly orthogonal. Half is a deliberate midpoint: it admits the
        # smooth rotation a pole undergoes between SCBA iterations and refuses
        # the swap that happens at a crossing.
        self._match_min_overlap = 0.5
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

    def locate_error(self, sol: PoleSolution, separation: float) -> float:
        r"""``eps_z = |dz_est| / min(gamma, separation, h)`` -- doc Eq. 49.

        The acceptance metric in the units the physics is in. ``eps_nep`` is a
        scaled matrix residual whose denominator ``|z|^2 + ||M||`` is
        ``1e3-1e4 THz^2`` here, so it answers a different question: on the CNT
        bed it refused 142 of 144 candidates, with residuals from 4.8e-10 to
        2.8e-02, while a candidate at 1e-9 is within about ``1e-5`` of its own
        linewidth.

        The three scales are the three ways a mislocated pole does damage. Its
        own width, because a residue displaced by more than ``gamma`` is the
        wrong residue. Its separation from the next pole, because beyond that
        the simple-pole split assigns weight to the wrong mode. The local
        cell, because a displacement larger than ``h`` moves the line into a
        different bin, which is the registration error the sector exists to
        remove.
        """
        gamma = max(-float(np.imag(sol.z)), 0.0)
        scale = min(x for x in (gamma, float(separation), self.h) if x > 0.0)
        return abs(complex(sol.dz_est)) / max(scale, 1e-300)

    @staticmethod
    def separations(solutions: list[PoleSolution]) -> list[float]:
        """Distance from each pole to its nearest neighbour in the set (THz)."""
        z = np.asarray([complex(s.z) for s in solutions])
        if z.size < 2:
            return [float("inf")] * z.size
        d = np.abs(z[:, None] - z[None, :])
        np.fill_diagonal(d, np.inf)
        return [float(x) for x in d.min(axis=1)]

    def _match_previous(self, solutions: list[PoleSolution]) -> list[bool]:
        """Which of these were in the sector last iteration.

        Identity across SCBA iterations is carried by an optimal ASSIGNMENT on
        both displacement and eigenvector overlap
        (:func:`~quatrex.phonon.pole_tracker.match_poles`), not by position
        alone. Position is sufficient for well-separated poles and fails
        exactly where it matters: at degeneracies, avoided crossings, pole
        splitting and satellite generation, an eigenvector rotates smoothly
        while its frequency ordering does not.

        Displacement is measured in grid cells so the threshold is the same
        ``cluster_factor`` that defines "the same mode" elsewhere.
        """
        if not self._promoted or not solutions:
            return [False] * len(solutions)

        z_old = xp.asarray([p[0] for p in self._promoted])
        r_old = xp.stack([xp.asarray(p[1]).reshape(-1)
                          for p in self._promoted], axis=1)
        z_new = xp.asarray([complex(s.z) for s in solutions])
        r_new = xp.stack([xp.asarray(s.r).reshape(-1) for s in solutions], axis=1)

        # The optimal assignment uses the combined cost; acceptance then
        # tests the two ingredients SEPARATELY. Their sum is too permissive:
        # an orthogonal eigenvector contributes only 1.0, which sits under
        # cluster_factor, so a summed threshold would carry identity straight
        # through a crossing -- the very case position alone already fails.
        assign = match_poles(z_old, r_old, z_new, r_new, s_z=self.h)
        zo = np.asarray(_host(z_old))
        zn = np.asarray(_host(z_new))
        ro = np.asarray(_host(r_old))
        rn = np.asarray(_host(r_new))
        ro = ro / np.linalg.norm(ro, axis=0, keepdims=True)
        rn = rn / np.linalg.norm(rn, axis=0, keepdims=True)
        overlap = np.abs(np.conj(ro).T @ rn)

        flags = [False] * len(solutions)
        for old_i, new_j in enumerate(assign):
            if new_j < 0:
                continue
            near = abs(zo[old_i] - zn[new_j]) <= self.cfg.cluster_factor * self.h
            aligned = overlap[old_i, new_j] >= self._match_min_overlap
            if near and aligned:
                flags[int(new_j)] = True
        return flags

    def leg_weight_error(self, gamma: float) -> float:
        r"""Worst-case relative error in a line's represented weight.

        .. math::
            E^{\max}_{\rm leg}(r) = \coth(\pi/r) - 1
                                   = \frac{2}{e^{2\pi/r} - 1},
            \qquad r = h/\gamma,

        the maximum of the exact trapezoidal sum
        :math:`\sinh(2\pi/r)/(\cosh(2\pi/r) - \cos(2\pi x))` over the
        sub-cell offset ``x``, attained with the line on a node. Small means
        the grid already carries the line however it happens to fall.
        """
        g = float(gamma)
        if g <= 0.0:
            return float("inf")
        r = self.h / g
        if r <= 0.0:
            return 0.0
        arg = 2.0 * np.pi / r
        return float("inf") if arg > 700.0 else 2.0 / np.expm1(arg)

    def screen(self, sol: PoleSolution, was_promoted: bool,
               separation: float = float("inf")) -> str | None:
        """Reason to refuse a pole, or ``None`` to accept it.

        Hysteresis is applied here: a mode already in the sector is only demoted
        once it is comfortably resolved, so membership cannot oscillate.
        """
        if getattr(self.cfg, "accept", "locate") == "locate":
            eps_z = self.locate_error(sol, separation)
            if not np.isfinite(eps_z) or eps_z > self.cfg.locate_tol:
                return (f"eps_z={eps_z:.2e} above locate_tol "
                        f"(eps_nep={sol.eps_nep:.2e})")
        elif not sol.converged:
            return f"eps_nep={sol.eps_nep:.2e} above tolerance"
        gamma = -sol.z.imag
        if gamma <= 0.0:
            return "pole is not in the lower half plane"
        lo, hi = self.window()
        if not (lo <= sol.z.real <= hi):
            return f"outside the pole window [{lo:.3g}, {hi:.3g}]"
        tol = float(getattr(self.cfg, "leg_weight_tol", 0.0) or 0.0)
        if tol > 0.0:
            # Exact: how much of the line's weight the grid can misrepresent,
            # worst case over where it falls between nodes. See leg_weight_tol.
            err = self.leg_weight_error(gamma)
            if err <= tol:
                return (f"grid-resolved (worst line-weight error {err:.3g} "
                        f"<= {tol:g})")
        else:
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

    def trust_radius(self, z0: complex, seeds: list[complex], k: int) -> float:
        r"""Newton trust radius for one seed, in THz -- doc Eq. 50.

        A pole is a property of :math:`M(z)`, not of the storage grid, so the
        search region is set by the scales that can actually mislead the solve:
        the nearest competing seed (step past its midpoint and the iteration
        can converge onto the neighbour's pole) and the nearest contact band
        edge (a branch point, where the local model of ``Sigma^R(z)`` stops
        being a simple-pole model at all).

        ``trust_radius_cells * h`` survives only as a FLOOR. As the radius
        itself it made grid refinement shrink the physical pole search, so a
        grid ladder would find fewer poles on its fine rungs for a reason
        entirely unrelated to the poles.
        """
        scales = [abs(complex(z0) - complex(s))
                  for j, s in enumerate(seeds) if j != k]
        if self.band_edges is not None and self.band_edges.size:
            scales.append(
                float(np.min(np.abs(self.band_edges - complex(z0).real))))
        floor = self.cfg.trust_radius_cells * self.h
        if not scales:
            return float("inf")
        return max(floor, self.cfg.trust_factor * min(scales))

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
                        trust_radius=self.trust_radius(z0, seeds, k),
                    )
                )
        finally:
            self._fit_anchor = saved
        return out

    def audit(self, m_blocks, dm_blocks, seeds: list[complex],
              seed_vectors: list[NDArray] | None = None) -> list[dict]:
        """Solve the candidates and report, WITHOUT allocating a sector.

        Doc Sec. 27. Root finding and sector allocation fail for unrelated
        reasons, and mixing them makes a low yield uninterpretable: a pole
        missing because Newton did not reach it and a pole missing because
        the representation was refused need opposite fixes. This stops after
        the solve, so the root-finding question can be answered on its own.

        Returns one row per candidate with the location, both acceptance
        metrics, the conditioning, and the refusal reason it would receive.
        """
        sols = self.solve_poles(m_blocks, dm_blocks, seeds, seed_vectors)
        seps = self.separations(sols)
        promoted = self._match_previous(sols)
        rows = []
        for k, sol in enumerate(sols):
            rows.append({
                "z": complex(sol.z),
                "gamma": float(-np.imag(complex(sol.z))),
                "separation": float(seps[k]),
                "q_omega": self.resolution_score(
                    max(-float(np.imag(complex(sol.z))), 0.0)),
                "eps_z": self.locate_error(sol, seps[k]),
                "eps_nep": float(sol.eps_nep),
                "eps_left": float(sol.eps_left),
                "kappa": float(sol.kappa),
                "iterations": int(sol.iterations),
                "trust_radius": self.trust_radius(seeds[k], seeds, k),
                "refused": self.screen(sol, promoted[k], seps[k]),
            })
        return rows

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
        promoted = self._match_previous(solutions)
        seps = self.separations(solutions)
        for k, sol in enumerate(solutions):
            why = self.screen(sol, promoted[k], seps[k])
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
            _h_for_report=self.h,
        )
        self._promoted = [(complex(s.z), s.r) for s in accepted]
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

        if getattr(self.cfg, "extraction_only", False):
            # Census only: report what the solve found and hand back an EMPTY
            # state, so the ring runs pole-free and the numbers cost nothing
            # but the solve. Deliberately BEFORE build_clusters -- allocating
            # and then discarding would still project sources and could still
            # hit the memory path this mode exists to stay clear of.
            self._report_census(self.audit(m_blocks, dm_blocks, seeds, vectors))
            self.state = PoleSectorState(iteration=self.state.iteration + 1)
            self._prev_delta = xp.array(self._delta, copy=True)
            return self.state

        state = self.build_clusters(sols)
        self._track(state)
        self._prev_delta = xp.array(self._delta, copy=True)
        return state

    @staticmethod
    def _report_census(rows: list[dict]) -> None:
        """The extraction-only summary, as a distribution rather than a count.

        A count says how many poles the sector would carry; the distributions
        say whether the bed HAS anything to carry. ``q_omega`` below one is the
        grid failing to resolve the line -- the condition the whole method is
        for -- and ``gamma/separation`` above about a half means neighbouring
        lines overlap, so no isolated simple pole exists for a bordered Newton
        to find however good the solver is.
        """
        if not rows:
            print("  pole census: no candidates", flush=True)
            return

        def _q(vals, ps=(0, 25, 50, 75, 100)):
            v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
            if v.size == 0:
                return "n/a"
            return "  ".join(f"{np.percentile(v, p):.3g}" for p in ps)

        gam = [r["gamma"] for r in rows]
        overlap = [r["gamma"] / r["separation"] if r["separation"] > 0
                   else np.inf for r in rows]
        n_unres = sum(1 for r in rows if r["q_omega"] < 1.0)
        n_iso = sum(1 for o in overlap if o < 0.5)
        why: dict[str, int] = {}
        for r in rows:
            k = (r["refused"] or "accepted").split("=")[0].split(":")[0].strip()
            why[k] = why.get(k, 0) + 1
        print(f"  pole census: {len(rows)} candidates; "
              f"{n_unres} under-resolved (q_omega < 1), "
              f"{n_iso} isolated (gamma/sep < 0.5)", flush=True)
        print(f"    q_omega       min/p25/med/p75/max  {_q([r['q_omega'] for r in rows])}",
              flush=True)
        print(f"    gamma/sep     min/p25/med/p75/max  {_q(overlap)}", flush=True)
        print(f"    eps_z         min/p25/med/p75/max  {_q([r['eps_z'] for r in rows])}",
              flush=True)
        print(f"    gamma [THz]   min/p25/med/p75/max  {_q(gam)}", flush=True)
        print("    outcome: " + ", ".join(f"{k} x{v}" for k, v in
                                          sorted(why.items(), key=lambda kv: -kv[1])),
              flush=True)

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
