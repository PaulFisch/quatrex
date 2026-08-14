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
2. seed from the previous iterate through the first-order predictor, or from the
   harmonic/golden-rule estimate when tracking has been lost;
3. correct each pole with bordered Newton and accept on the scaled residual;
4. apply the promotion criteria -- resolution, conditioning and the location
   error -- with hysteresis, and only at an epoch boundary;
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
from quatrex.phonon.pole_kernel import (
    LocalFitPlan, bosonic_partner, contract_delta, continuation_weights,
    local_fit_weights,
)
from quatrex.phonon.pole_nevp import (
    PoleSolution, _lift, _matvec, _vdot, bordered_newton_batch,
)
from quatrex.phonon.pole_probe import BlockLayout, nnz_to_blocks
from quatrex.phonon.pole_tracker import (
    PoleTracker, cluster_poles, match_poles,
)

__all__ = ["PoleSectorState", "PoleSector", "PoleQBatch", "refresh_many"]


def _report_rank() -> int:
    """Rank for the purpose of PRINTING, or 0 when there is no communicator.

    This module is otherwise free of any communicator: the continuation is a
    pure function of the gathered ``Delta`` and every rank computes the same
    poles. The census is the one thing here that writes to stdout, so it is
    the one thing that needs to know. Imported lazily and defaulting to 0, so
    a serial caller or a test never pays for MPI.
    """
    try:
        from qttools.comm import comm
        return int(comm.rank)
    except Exception:                                       # noqa: BLE001
        return 0


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
    n_seeded: int = 0
    """Candidates the corrector was handed this iteration."""
    eps_z_accepted: list = field(default_factory=list)
    eps_z_refused: list = field(default_factory=list)
    """``eps_z`` for each candidate, split by the decision.

    Whether the refused ones sit just over ``locate_tol`` or orders above it
    is the difference between a threshold that is merely too tight and poles
    the corrector genuinely failed to locate. Widening the hysteresis gap
    helps only in the first case."""
    n_matched: int = 0
    """Of those, how many the tracker recognised as ALREADY in the sector.

    The hysteresis in :meth:`PoleSector.screen` can only act on these. A pole
    that leaves because it was never matched left for a different reason than
    one refused on its own merits, and the two need opposite fixes -- so the
    count is reported rather than inferred from the promoted total."""
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

        NEVER SUPPLIED IN PRODUCTION. Both solver call sites construct the
        sector without it, so the ``edge_factor`` gate and the band-edge term of
        :meth:`trust_radius` are inert on every real run; only tests pass it.
        Supplying it means deriving the lead band edges from the contact
        dispersion, and doing so would change which poles are promoted -- so it
        is a deliberate measurement, not a wiring oversight to fix in passing.

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
        # Fit anchors for the current pole solve, one per candidate. Pinning
        # them makes M(z) a genuine analytic function of z for the whole Newton
        # iteration; deriving the stencil from Re z instead makes M only
        # PIECEWISE holomorphic, with a measured 17 % jump at each stencil
        # boundary. Set for the duration of solve_poles and restored after.
        self._fit_anchors: NDArray | None = None
        self._fit_plan: LocalFitPlan | None = None
        self._delta_raw: NDArray | None = None
        self._delta_mirror: NDArray | None = None
        self._delta_layouts = None

    # -- window ------------------------------------------------------------ #

    def window(self, low_freq_mask: float = 0.0) -> tuple[float, float]:
        """Resolved pole-search window (THz).

        ``omega_min_thz = 0`` resolves to ``max(4*dw, mask + 2*dw)``: below that
        the quasiparticle picture does not apply, and the continuation has no cut
        where the self-energy has been masked to zero.

        The top comes from the GLOBAL axis, not this rank's slice. Every rank
        assembles the same operator and is supposed to solve the same poles
        (see ``PhononSolver._pole_frequency_context``); taking the default from
        ``self.freqs[-1]`` gave each rank a different search window, so on a
        distributed run the ranks screened against different bounds and could
        promote different sets. Serial runs are unaffected -- ``global_freqs``
        IS ``freqs`` there.
        """
        lo = self.cfg.omega_min_thz or max(4.0 * self.h, low_freq_mask + 2.0 * self.h)
        hi = self.cfg.omega_max_thz or float(_host(self.global_freqs)[-1])
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

    def leg_weight_error_finite(self, gamma: float, centre: float) -> float:
        r"""Relative error in a line's represented weight over the FINITE grid.

        .. math::
            E_{\rm finite} =
              \frac{|W^{[a,b]}_{\rm point} - W^{[a,b]}_{\rm exact}|}
                   {W^{[a,b]}_{\rm exact} + \epsilon},

        with :math:`W_{\rm point} = \frac{h}{2\pi}\sum_{\omega_n\in[a,b]}
        L_\gamma(\omega_n)` and :math:`W_{\rm exact} = \int_a^b
        \frac{d\omega}{2\pi} L_\gamma(\omega)`, for the unit-weight Lorentzian
        :math:`L_\gamma(\omega) = 2\gamma/((\omega-\Omega)^2+\gamma^2)`.

        :meth:`leg_weight_error` is the infinite-grid statement and knows
        nothing about where the line sits. It is the right question for a mode
        in the middle of the band and the wrong one near a support boundary,
        where a Lorentzian tail runs off the end of the grid: there the
        trapezoidal sum can be accurate about a weight that is itself only a
        fraction of the line. This measures the quantity the ring actually
        integrates, over the support it actually has.

        The two agree in the regime that matters. Where the line is
        under-resolved and sits well inside the support, the sum departs from
        the integral by the pole's own discretisation error and this returns
        :math:`|W_\infty(r,x) - 1|`, whose worst case over the sub-cell offset
        ``x`` is :meth:`leg_weight_error`; with the line on a node the two
        agree to better than a percent.

        They do NOT agree once the grid resolves the line. Truncation cancels
        in the ratio -- numerator and denominator run over the same ``[a,b]``
        -- so what is left is the Euler-Maclaurin endpoint term,
        :math:`O(h^2 f'(b) - h^2 f'(a))`, which does not vanish with the pole.
        On a 20 THz grid at ``h = 0.125`` a line of half width 0.4 reads
        3.4e-07 here against 3.7e-09 from the infinite formula: both say
        "resolved", by different mechanisms. Read this gate as a statement
        about a line the grid cannot carry, not as a refinement of one it can.

        ``centre`` is the resonance frequency :math:`\Omega = \Re z`; ``gamma``
        is the HALF width (see
        :attr:`~quatrex.phonon.pole_nevp.PoleSolution.gamma_hwhm`).
        """
        g = float(gamma)
        if g <= 0.0:
            return float("inf")
        w = np.asarray(_host(self.global_freqs), dtype=float)
        if w.size < 2:
            return float("inf")
        c = float(centre)
        # The support is the union of the cells, so it runs half a spacing
        # past the outermost NODES -- the same convention the ring integrates
        # under, where every sample owns a cell of width h.
        a, b = w[0] - 0.5 * self.h, w[-1] + 0.5 * self.h
        point = (self.h / np.pi) * float(np.sum(g / ((w - c) ** 2 + g * g)))
        exact = (np.arctan((b - c) / g) - np.arctan((a - c) / g)) / np.pi
        if exact <= 0.0:
            return float("inf")
        return float(abs(point - exact) / exact)

    def screen(self, sol: PoleSolution, was_promoted: bool,
               separation: float = float("inf"),
               *, hard_only: bool = False) -> str | None:
        """Reason to refuse a pole, or ``None`` to accept it.

        Hysteresis is applied here: a mode already in the sector is only demoted
        once it is comfortably resolved, so membership cannot oscillate.

        ``hard_only`` keeps just the gates a pole cannot be carried in spite
        of -- it is not in the lower half plane, or it has left the window --
        and drops the quality gates. Used while membership is frozen for an
        epoch, where the question is not "is this pole worth carrying" (that
        was settled at the epoch boundary) but "is it still a pole at all".
        """
        gamma_ = -sol.z.imag
        if gamma_ <= 0.0:
            return "pole is not in the lower half plane"
        lo_, hi_ = self.window()
        if not (lo_ <= sol.z.real <= hi_):
            return f"outside the pole window [{lo_:.3g}, {hi_:.3g}]"
        if hard_only:
            return None
        if getattr(self.cfg, "accept", "locate") == "locate":
            eps_z = self.locate_error(sol, separation)
            # Hysteresis, same rule as q_in/q_out below: strict to enter,
            # lenient to stay. A single threshold here closes a feedback loop
            # through Sigma and the promoted set limit-cycles with period two
            # (measured on Si: 620 <-> 460 poles, residual stuck at O(1)).
            tol_z = (float(getattr(self.cfg, "locate_tol_out",
                                   self.cfg.locate_tol))
                     if was_promoted else self.cfg.locate_tol)
            if not np.isfinite(eps_z) or eps_z > tol_z:
                return (f"eps_z={eps_z:.2e} above "
                        f"{'locate_tol_out' if was_promoted else 'locate_tol'}"
                        f"={tol_z:g} (eps_nep={sol.eps_nep:.2e})")
        elif not sol.converged:
            return f"eps_nep={sol.eps_nep:.2e} above tolerance"
        gamma = gamma_          # both hard gates ran above, before hard_only
        tol = float(getattr(self.cfg, "leg_weight_tol", 0.0) or 0.0)
        if tol > 0.0:
            # Exact: how much of the line's weight the grid can misrepresent,
            # worst case over where it falls between nodes. See leg_weight_tol.
            # Hysteresis runs the other way here than for eps_z: this gate
            # refuses a pole the grid ALREADY resolves, so staying in the
            # sector means a smaller threshold, not a larger one.
            if was_promoted:
                out = float(getattr(self.cfg, "leg_weight_tol_out", 0.0) or 0.0)
                tol = out if out > 0.0 else tol / 3.0
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

    def trust_radii(self, seeds) -> NDArray:
        """``trust_radius`` for a whole seed set at once, ``(P,)``.

        Same formula as :meth:`trust_radius`, written over the seed array so
        the pole solve does not pay a Python loop per candidate. The two are
        pinned against each other in the suite.
        """
        z = np.asarray([complex(s) for s in seeds])
        if z.size == 0:
            return xp.zeros(0)
        d = np.abs(z[:, None] - z[None, :])
        np.fill_diagonal(d, np.inf)
        nearest = d.min(axis=1) if z.size > 1 else np.full(z.size, np.inf)
        if self.band_edges is not None and self.band_edges.size:
            edge = np.abs(self.band_edges[None, :] - z.real[:, None]).min(axis=1)
            nearest = np.minimum(nearest, edge)
        floor = self.cfg.trust_radius_cells * self.h
        # No competing scale at all -> unbounded, as the scalar form returns
        # inf rather than the floor when its `scales` list comes out empty.
        return xp.asarray(np.where(np.isfinite(nearest),
                                   np.maximum(floor,
                                              self.cfg.trust_factor * nearest),
                                   np.inf))

    # -- update ------------------------------------------------------------ #

    def solve_poles(
        self, m_blocks, dm_blocks, seeds: list[complex],
        seed_vectors: list[NDArray] | None = None,
        *, batched: bool = False,
    ) -> list[PoleSolution]:
        """Correct a whole seed set with the bordered Newton iteration.

        One batched call, not one call per seed. Each candidate keeps its OWN
        pinned fit anchor -- that is what makes M(z) holomorphic over its solve
        -- so the anchors travel as a vector alongside the seeds rather than as
        a single value set and reset around each candidate.

        ``batched`` says whether the operator callables already take a VECTOR
        of probe points, as :meth:`operator` produces. It defaults to False, so
        a caller holding a scalar ``z -> blocks`` closure -- the toy beds, and
        anything outside this class -- still works: such an operator is lifted
        by evaluating it once per probe point, which is arithmetically the
        per-candidate solve this method replaced, and is the reference the
        batched path is verified against.
        """
        if len(seeds) == 0:
            return []
        if not batched:
            m_blocks, dm_blocks = _lift(m_blocks), _lift(dm_blocks)
        z0 = xp.asarray([complex(s) for s in seeds], dtype=xp.complex128)
        r0 = (None if seed_vectors is None
              else xp.stack([xp.asarray(v).reshape(-1) for v in seed_vectors]))
        saved = (self._fit_anchors, self._fit_plan)
        try:
            self._set_fit_anchors(xp.real(z0))
            batch = bordered_newton_batch(
                m_blocks, dm_blocks, z0, r0,
                tol=self.cfg.newton_tol,
                max_iter=self.cfg.newton_max_iterations,
                trust_radius=self.trust_radii(seeds),
            )
        finally:
            self._fit_anchors, self._fit_plan = saved
        return batch.to_list()

    def audit(self, m_blocks, dm_blocks, seeds: list[complex],
              seed_vectors: list[NDArray] | None = None,
              *, batched: bool = False) -> list[dict]:
        """Solve the candidates and report, WITHOUT allocating a sector.

        Doc Sec. 27. Root finding and sector allocation fail for unrelated
        reasons, and mixing them makes a low yield uninterpretable: a pole
        missing because Newton did not reach it and a pole missing because
        the representation was refused need opposite fixes. This stops after
        the solve, so the root-finding question can be answered on its own.

        Returns one row per candidate with the location, both acceptance
        metrics, the conditioning, and the refusal reason it would receive.
        """
        sols = self.solve_poles(m_blocks, dm_blocks, seeds, seed_vectors,
                                batched=batched)
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
        eps_ok, eps_no = [], []
        # Membership frozen: the epoch decided who is in, and re-deciding
        # every iteration is what puts a floor under rel Sigma. Positions
        # still track Sigma; only the SET is held.
        frozen = (bool(getattr(self.cfg, "freeze_membership", False))
                  and bool(self._promoted)
                  and self.tracker.membership_frozen())
        for k, sol in enumerate(solutions):
            if frozen:
                why = (self.screen(sol, True, seps[k], hard_only=True)
                       if promoted[k] else "not in the frozen epoch set")
            else:
                why = self.screen(sol, promoted[k], seps[k])
            (eps_no if why else eps_ok).append(
                float(self.locate_error(sol, seps[k])))
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
            n_seeded=len(solutions), n_matched=int(sum(promoted)),
            eps_z_accepted=eps_ok, eps_z_refused=eps_no,
            _h_for_report=self.h,
        )
        self._promoted = [(complex(s.z), s.r) for s in accepted]
        return self.state

    # -- operator context --------------------------------------------------- #

    def set_operator_context(
        self, *, delta, d_blocks, obc_left, obc_right, block_sizes, rows, cols,
        layout: BlockLayout | None = None,
    ) -> None:
        r"""Store everything :math:`M(z)` needs for this SCBA iteration.

        ``delta`` is ``Sigma^> - Sigma^<`` on the full frequency grid; the rest
        is the operator's frequency-independent part plus the contact blocks.
        The contacts are held at their real-axis value: continuing the lead
        self-energy off the axis is the outgoing-sheet work, and on the physical
        sheet -- which is where the currently supported pole classes live, in a
        contact gap or weakly coupled -- the contact contribution is flat over a
        window narrower than its own scale.

        Everything that does NOT depend on ``z`` is built here, once. The
        bordered Newton evaluates the operator about six times per candidate
        and used to rebuild all of it every time: the bosonic mirror of
        ``Delta`` (identical on every call, ``a`` being ``Delta``), the scatter
        from the stored sparsity pattern into dense blocks, and the dense
        ``D``. See :class:`~quatrex.phonon.pole_probe.BlockLayout`.
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
        self._n_dof = int(np.sum(self._block_sizes))
        self._set_layout(layout)

    def _set_layout(self, layout: BlockLayout | None = None) -> None:
        """Build the Delta-independent half of the operator (see above).

        ``layout`` lets a caller with many q share one: the map from the stored
        sparsity pattern to blocks depends on ``(rows, cols, block_sizes)``
        alone, which every q has in common, so building it per q is the same
        scan repeated ``nq`` times.
        """
        self._layout = layout if layout is not None else BlockLayout(
            self._rows, self._cols, self._block_sizes, band=1)
        d_lay = self._layout.flatten_blocks(self._d_blocks)
        # Drop leading singletons: the dynamical matrix carries a singleton
        # where the Keldysh buffers carry frequency, and the operator's own
        # batch axis is the probe axis.
        self._d_lay = d_lay.reshape(-1, self._layout.total)
        if self._d_lay.shape[0] == 1:
            self._d_lay = self._d_lay[0]

    # ``Delta`` is held behind a property so that everything DERIVED from it --
    # the bosonic mirror and the two block-layout gathers -- cannot survive a
    # change to it. They are pure functions of Delta and rebuilding them per
    # Newton step is the waste this refactor removes, but a stale copy is the
    # exact failure this module refuses to risk: Delta is rebuilt from the
    # mixed buffers every iteration precisely so it cannot drift out of step
    # with the Sigma^R the Dyson operator was built from. Assigning _delta
    # anywhere -- set_operator_context, the predictor's temporary swap, a test
    # perturbing it in place -- invalidates the cache.
    @property
    def _delta(self) -> NDArray:
        return self._delta_raw

    @_delta.setter
    def _delta(self, value) -> None:
        self._delta_raw = xp.asarray(value)
        self._delta_mirror = None
        self._delta_layouts = None

    def _mirror(self) -> NDArray:
        """The bosonic partner of Delta; needs no operator context."""
        if self._delta_mirror is None:
            self._delta_mirror = bosonic_partner(self._delta_raw)
        return self._delta_mirror

    def _delta_derived(self):
        """``(delta_lay, mirror_lay)``, Delta gathered into the block layout."""
        if self._delta_layouts is None:
            self._delta_layouts = (self._layout.gather(self._delta_raw),
                                   self._layout.gather(self._mirror()))
        return self._delta_layouts

    # -- the anchor-pinned fit ---------------------------------------------- #

    def _set_fit_anchors(self, anchors) -> None:
        """Pin the local-fit stencil for a whole batched Newton solve.

        One anchor per candidate. Pinning makes ``M(z)`` a genuine analytic
        function of ``z`` for that candidate's whole solve; deriving the
        stencil from ``Re z`` instead makes ``M`` only PIECEWISE holomorphic,
        with a measured 17 % jump at each stencil boundary. The stencil and its
        pseudo-inverse depend on nothing else, so they are built once here
        rather than at every Newton step.
        """
        if anchors is None:
            self._fit_anchors = self._fit_plan = None
            return
        self._fit_anchors = xp.asarray(anchors, dtype=xp.float64).reshape(-1)
        self._fit_plan = LocalFitPlan(
            self.global_freqs, self._fit_anchors,
            order=self.cfg.delta_fit_order,
            window=self.cfg.delta_fit_window_cells,
        )

    def _weights(self, zz: NDArray, order: int) -> tuple[NDArray, NDArray]:
        """``(w_pos, w_mir)`` on this rank's frequency columns, ``(P, n_local)``."""
        w_pos, w_mir = continuation_weights(zz, self.global_freqs, order=order)
        plan = getattr(self, "_fit_plan", None)
        if plan is None:
            f_pos, f_mir = local_fit_weights(
                self.global_freqs, zz, order=self.cfg.delta_fit_order,
                window=self.cfg.delta_fit_window_cells, deriv=order,
            )
        else:
            if plan.n_probe != int(zz.shape[0]):
                raise ValueError(
                    f"the fit anchors are pinned for {plan.n_probe} candidates "
                    f"but the operator was asked for {int(zz.shape[0])} probe "
                    "points; the anchor must accompany its candidate."
                )
            f_pos, f_mir = plan.weights(zz, deriv=order)
        lo = self._freq_offset
        hi = lo + int(self._delta.shape[0])
        return w_pos[:, lo:hi] + f_pos[:, lo:hi], w_mir[:, lo:hi] + f_mir[:, lo:hi]

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
        zz = xp.asarray(z, dtype=xp.complex128).reshape(-1)
        w_pos, w_mir = self._weights(zz, order)
        val = contract_delta(self._delta, w_pos, w_mir, mirror=self._mirror())
        return self._reduce(val)

    def _continue_flat(self, zz: NDArray, order: int) -> NDArray:
        """``Sigma_s^{R,II}`` already laid out in band-block order, ``(P, total)``.

        Same contraction as :meth:`continue_sigma`, against ``Delta`` gathered
        once into the block layout, so the per-step scatter into dense blocks
        disappears: the result IS the block buffer.
        """
        w_pos, w_mir = self._weights(zz, order)
        delta_lay, mirror_lay = self._delta_derived()
        return self._reduce(w_pos @ delta_lay + w_mir @ mirror_lay)

    def _sigma_blocks_at(self, z, order: int):
        """Scattering self-energy blocks at complex ``z``."""
        val = self.continue_sigma(xp.asarray(z).reshape(-1), order=order)
        return nnz_to_blocks(val, self._rows, self._cols, self._block_sizes,
                             band=1)

    def _obc_at(self, zz: NDArray):
        """Contact blocks sampled at the grid frequency nearest each anchor.

        A leading frequency axis is reduced here and nowhere else, so the
        assembled operator always sees one matrix per block per candidate.
        """
        anchors = getattr(self, "_fit_anchors", None)
        # Pinned to the same anchor as the fit: sampling the contact at the
        # point nearest Re z would reintroduce a jump into M(z).
        ref = xp.real(zz) if anchors is None else anchors
        out = []
        for o in self._obc:
            if o is None or o.ndim < 3:
                out.append(o)
                continue
            k = xp.argmin(xp.abs(self.freqs[None, :] - ref[:, None]), axis=1)
            out.append(o[k])
        return out[0], out[1]

    def operator(self):
        """``(m_blocks, dm_blocks)`` closures over the stored context.

        Both take a VECTOR of probe points and return block-tridiagonal blocks
        with a leading candidate axis. The blocks are reshaped views of one
        flat buffer, so the whole assembly is two GEMMs and a handful of
        elementwise kernels regardless of how many candidates or blocks there
        are.
        """
        lay = self._layout
        last = len(self._block_sizes) - 1

        def _flat(zz):
            return xp.zeros((int(zz.shape[0]), lay.total), dtype=xp.complex128)

        def m_blocks(z):
            zz = xp.asarray(z, dtype=xp.complex128).reshape(-1)
            # Assembled in the same order as the scalar path built it,
            # z^2 I - D - Sigma_s - Sigma_c, so the arithmetic matches.
            flat = _flat(zz)
            flat[..., lay.diag] = (zz * zz)[:, None]
            flat = flat - self._d_lay - self._continue_flat(zz, 0)
            obc_l, obc_r = self._obc_at(zz)
            for obc, i in ((obc_l, 0), (obc_r, last)):
                if obc is not None:
                    sl = lay.corner[i]
                    flat[..., sl] -= xp.asarray(obc).reshape(
                        xp.asarray(obc).shape[:-2] + (-1,))
            return lay.blocks(flat)

        def dm_blocks(z):
            zz = xp.asarray(z, dtype=xp.complex128).reshape(-1)
            flat = _flat(zz)
            flat[..., lay.diag] = 2.0 * zz[:, None]
            return lay.blocks(flat - self._continue_flat(zz, 1))

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
        # Over the whole spectrum at once. The loop this replaces built a dense
        # (n_dof, n_dof) Delta per MODE -- n_dof of them per q, so n_dof * n_q
        # per SCBA iteration -- to take one quadratic form from each.
        keep = np.nonzero((lam.real > 0.0)
                          & (np.sqrt(np.maximum(lam.real, 0.0)) >= lo)
                          & (np.sqrt(np.maximum(lam.real, 0.0)) <= hi))[0]
        if keep.size == 0:
            return []
        om = np.sqrt(lam.real[keep])                                # (K,)

        # Delta at Omega, linearly interpolated from the grid.
        idx = np.clip(np.searchsorted(w, om), 1, w.size - 1)
        f = (om - w[idx - 1]) / (w[idx] - w[idx - 1])
        vals = ((1.0 - f)[:, None] * delta[idx - 1]
                + f[:, None] * delta[idx])                          # (K, nnz)

        # v^H Delta v on the stored pattern, without densifying: the quadratic
        # form is a weighted sum over the nonzeros.
        v = vec[:, keep]                                            # (n_dof, K)
        quad = np.sum(np.conj(v[rows, :]).T * vals * v[cols, :].T, axis=1)
        gamma = np.maximum(-np.imag(quad) / (4.0 * om), 1e-12)
        return [complex(o, -g) for o, g in zip(om, gamma)]

    def refresh(self) -> PoleSectorState:
        """One SCBA iteration's worth of pole tracking.

        Predictor -> corrector -> subspace match -> harmonic re-seed, in that
        order. The seed comes from the first-order response of the previous
        poles to the change in the scattering self-energy (doc Eq. 43), which
        costs one projection; the harmonic/golden-rule estimate is used on the
        first iteration, when a rescan offers new candidates, and as the
        fallback when no pole survived the corrector.

        There is no contour fallback. :func:`~quatrex.phonon.pole_nevp.
        beyn_contour` exists and is tested, but nothing in ``src/`` calls it and
        ``contour_quad_points`` reaches no code; the fallback that actually runs
        is the harmonic re-seed in :func:`refresh_many`.

        Sector membership is held fixed within an adaptation epoch: an
        approximate implementation is not invariant under repartitioning, so a
        mode that changes sector every iteration changes the fixed-point map
        itself.

        The one-q case of :func:`refresh_many`, so that the single-q path and
        the q-batched path are the same code and every test of this method also
        tests that one.
        """
        return refresh_many([self])[0]

    # -- the three phases refresh_many drives ------------------------------- #

    def _begin_refresh(self):
        """Seeds and predicted eigenvectors for this iteration."""
        return self._seed()

    def _end_refresh(self, sols: list[PoleSolution], seeds, vectors,
                     m_blocks, dm_blocks) -> PoleSectorState:
        """Screen, cluster and track what the corrector returned."""
        if getattr(self.cfg, "extraction_only", False):
            # Census only: report what the solve found and hand back an EMPTY
            # state, so the ring runs pole-free and the numbers cost nothing
            # but the solve. Deliberately BEFORE build_clusters -- allocating
            # and then discarding would still project sources and could still
            # hit the memory path this mode exists to stay clear of.
            self._report_census(self._census_rows(sols, seeds))
            self.state = PoleSectorState(iteration=self.state.iteration + 1)
            self._prev_delta = xp.array(self._delta, copy=True)
            return self.state

        state = self.build_clusters(sols)
        self._track(state)
        self._prev_delta = xp.array(self._delta, copy=True)
        return state

    def _census_rows(self, sols: list[PoleSolution], seeds) -> list[dict]:
        """:meth:`audit`'s report for an ALREADY solved candidate set.

        Carries the four quantities the pole/QNM audit (Sec. 38) asks for
        beyond what the gates themselves need: ``chi`` for the simple-versus-
        cluster decision, ``E_finite`` for the support boundary, the anharmonic
        sensitivity for what SETS the width, and ``passive`` so a root that came
        back on the wrong half plane is counted as a continuation failure rather
        than folded into the refusal histogram as one more rejected mode.
        """
        seps = self.separations(sols)
        promoted = self._match_previous(sols)
        try:
            sens = self.sensitivities(sols)
        except (AttributeError, ValueError):
            # No operator context (the manually driven route): the location
            # census is still worth having without it.
            sens = [complex("nan")] * len(sols)
        rows = []
        for k, sol in enumerate(sols):
            gamma = float(-np.imag(complex(sol.z)))
            sep = float(seps[k])
            rows.append({
                "z": complex(sol.z),
                "gamma": gamma,
                "separation": sep,
                "chi": gamma / sep if sep > 0 else float("inf"),
                "q_omega": self.resolution_score(max(gamma, 0.0)),
                "leg_weight_error": self.leg_weight_error(gamma),
                "E_finite": self.leg_weight_error_finite(
                    gamma, float(np.real(complex(sol.z)))),
                "gamma_sens_anh": -sens[k].imag,
                "passive": bool(np.imag(complex(sol.z)) < 0.0),
                "eps_z": self.locate_error(sol, sep),
                "eps_nep": float(sol.eps_nep),
                "eps_left": float(sol.eps_left),
                "kappa": float(sol.kappa),
                "iterations": int(sol.iterations),
                "trust_radius": self.trust_radius(seeds[k], seeds, k),
                "refused": self.screen(sol, promoted[k], sep),
            })
        return rows

    @staticmethod
    def _report_census(rows: list[dict]) -> None:
        """The extraction-only summary, as a distribution rather than a count.

        Rank 0 only. Every rank assembles the same operator and solves the same
        poles by construction, so without the guard all of them write the same
        report to one stdout and the lines interleave mid-word -- the census of
        job 4479538 came back with rows like ``gamma [THz]    gamma [THz]
        min/p25/...`` and was unparseable. ``_census_over_q`` already guards the
        ``q (...)`` header it prints, which is what made the mismatch visible:
        14 headers against 179 bodies.

        A count says how many poles the sector would carry; the distributions
        say whether the bed HAS anything to carry. ``q_omega`` below one is the
        grid failing to resolve the line -- the condition the whole method is
        for -- and ``gamma/separation`` above about a half means neighbouring
        lines overlap, so no isolated simple pole exists for a bordered Newton
        to find however good the solver is.
        """
        if _report_rank() != 0:
            return
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
        n_bad = sum(1 for r in rows if not r["passive"])
        tail = (f"; {n_bad} CONTINUATION FAILURES (upper half plane)"
                if n_bad else "")
        print(f"  pole census: {len(rows)} candidates; "
              f"{n_unres} under-resolved (q_omega < 1), "
              f"{n_iso} isolated (gamma/sep < 0.5){tail}", flush=True)
        print(f"    q_omega       min/p25/med/p75/max  {_q([r['q_omega'] for r in rows])}",
              flush=True)
        print(f"    gamma/sep     min/p25/med/p75/max  {_q(overlap)}", flush=True)
        print(f"    eps_z         min/p25/med/p75/max  {_q([r['eps_z'] for r in rows])}",
              flush=True)
        print(f"    gamma [THz]   min/p25/med/p75/max  {_q(gam)}", flush=True)
        print(f"    E_leg^max     min/p25/med/p75/max  "
              f"{_q([r['leg_weight_error'] for r in rows])}", flush=True)
        print(f"    E_finite      min/p25/med/p75/max  "
              f"{_q([r['E_finite'] for r in rows])}", flush=True)
        print(f"    gamma_sens/gamma  min/p25/med/p75/max  "
              f"{_q([r['gamma_sens_anh'] / r['gamma'] for r in rows if r['gamma'] > 0])}",
              flush=True)
        print("    outcome: " + ", ".join(f"{k} x{v}" for k, v in
                                          sorted(why.items(), key=lambda kv: -kv[1])),
              flush=True)

    def _seed(self):
        """Warm-start seeds, PLUS the harmonic estimate when an audit is due.

        A rescan ADDS candidates; it must never replace the held set. Replacing
        it makes the sector a period-two oscillator, and the mechanism is pure
        control flow -- no physics, no threshold, in it:

        * ``_track`` calls ``tracker.update`` on a warm iteration and
          ``tracker.adopt`` on a rescan. ``update`` arms a rescan whenever the
          cluster COUNT or any cluster SIZE changes; ``adopt`` disarms it.
        * membership moving is what changes those counts, and membership moves
          every iteration, so a warm iteration always arms the next rescan.
        * the rescan then discarded the held poles and re-seeded from the
          harmonic spectrum, which re-found them all.

        So the sector alternated between everything the harmonic spectrum
        offers and whatever survived screening it, locked at period two. On Si
        (81 q, ``h = 0.25``) that was 650 <-> 485 poles for 150 iterations,
        and because the ring convolves the cell average of the reconstruction,
        an alternating pole set makes ``Sigma`` alternate too -- pinning
        ``rel Sigma`` at 2.5e-01 where the pole-free arm reaches 9.3e-04. The
        floor was the discontinuity, not slow convergence.

        The acceptance hysteresis in :meth:`screen` cannot reach this: the
        oscillation is in which seeds the corrector is handed, decided before
        any pole is screened.
        """
        prev = self.state.solutions
        if not prev:
            return self.harmonic_seeds(), None
        z = xp.asarray([complex(s.z) for s in prev], dtype=xp.complex128)
        r = xp.stack([xp.asarray(s.r).reshape(-1) for s in prev])
        l = xp.stack([xp.asarray(s.l).reshape(-1) for s in prev])
        shift = self._predicted_shifts(z, l, r)
        if shift is None:
            shift = xp.zeros_like(z)
        cap = self.cfg.trust_radius_cells * self.h
        mag = xp.abs(shift)
        shift = xp.where(mag > cap, shift * (cap / xp.where(mag > 0, mag, 1.0)),
                         shift)
        seeds = [complex(x) for x in np.asarray(_host(z + shift))]
        vecs = [s.r for s in prev]
        if not (getattr(self.cfg, "audit_every_iteration", True)
                or self.tracker.needs_rescan()):
            return seeds, vecs
        # Audit: offer the harmonic candidates the held set does not already
        # cover. "Already covered" is the same cluster_factor * h that defines
        # "the same mode" everywhere else, so a re-found pole is not duplicated
        # into its own neighbour and then split by the clusterer.
        reach = self.cfg.cluster_factor * self.h
        extra = [w for w in self.harmonic_seeds()
                 if min(abs(w - t) for t in seeds) > reach]
        if not extra:
            return seeds, vecs
        # The same unit vector the unbatched solve would have drawn, so an
        # added candidate starts exactly where a cold solve would start it.
        rng = xp.random.default_rng(0)
        n_dof = int(np.sum(self._block_sizes))
        default = rng.standard_normal(n_dof) + 1j * rng.standard_normal(n_dof)
        return seeds + extra, vecs + [default] * len(extra)

    def _predicted_shifts(self, z, l, r):
        r"""``delta z_alpha = l^H Delta Sigma_s^R(z_alpha) r`` for every pole.

        The predictor needs the CHANGE in the self-energy, which is linear in
        the change in ``Delta`` -- so it is the same continuation applied to the
        difference, and costs one extra contraction rather than a re-solve.

        Contracted for the whole previous pole set at once, and applied through
        the block-tridiagonal operator rather than a dense ``(n_dof, n_dof)``
        per pole: ``Delta Sigma`` has the band the stored pattern has, so the
        dense form was materialising zeros.
        """
        prev = getattr(self, "_prev_delta", None)
        if prev is None or prev.shape != self._delta.shape:
            return None
        diff = self._delta - prev
        if float(xp.abs(diff).max()) == 0.0:
            return None
        saved = self._delta
        try:
            # The setter invalidates the derived gathers, so the contraction
            # below really is against the difference and not against a cached
            # Delta -- see the property.
            self._delta = diff
            flat = self._continue_flat(
                xp.asarray(z, dtype=xp.complex128).reshape(-1), 0)
        finally:
            self._delta = saved
        blocks = self._layout.blocks(flat)
        return _vdot(l, _matvec(blocks, r))

    def sensitivities(self, sols: list[PoleSolution]) -> list[complex]:
        r"""``dz/dlambda`` for the anharmonic channel -- audit Eq. (10).

        Scaling one self-energy component by :math:`\lambda_j` gives
        :math:`M(z,\lambda_j) = M(z,0) - \lambda_j \Sigma_j^R(z)`, and implicit
        differentiation of :math:`l^\dagger M r = 0` gives

        .. math::
            \frac{dz_\alpha}{d\lambda_j}
              = \frac{l_\alpha^\dagger \Sigma_j^R(z_\alpha) r_\alpha}
                     {l_\alpha^\dagger M'(z_\alpha) r_\alpha},

        whose denominator is 1 under the normalisation
        :func:`~quatrex.phonon.pole_nevp.bordered_newton_batch` already applies.
        The imaginary part, :math:`\gamma^{\rm sens}_{\alpha,j} = -\Im\,
        dz_\alpha/d\lambda_j`, is how much of the pole's half width that channel
        accounts for.

        Only ``j = anharmonic`` is available. The two contact channels need
        :math:`\Sigma_c(z)`, and the operator holds the contacts at a real
        anchor with no ``z`` dependence at all (see
        :meth:`set_operator_context`), so their derivative is identically zero
        by construction rather than small -- reporting it would be reporting the
        approximation, not the physics.

        These are diagnostics. They are exact first derivatives of the pole
        location, but the channels are not independent and the widths they
        return do not have to sum to :math:`\gamma`.

        Reuses the contraction :meth:`_predicted_shifts` uses, against the full
        ``Delta`` rather than its change: block-tridiagonal, one batched pass
        over the whole pole set, no dense ``(n_dof, n_dof)`` per pole.
        """
        if not sols:
            return []
        z = xp.asarray([complex(s.z) for s in sols], dtype=xp.complex128)
        r = xp.stack([xp.asarray(s.r).reshape(-1) for s in sols])
        l = xp.stack([xp.asarray(s.l).reshape(-1) for s in sols])
        flat = self._continue_flat(z, 0)
        val = _vdot(l, _matvec(self._layout.blocks(flat), r))
        return [complex(x) for x in np.asarray(_host(val)).reshape(-1)]

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


class PoleQBatch:
    r"""One :math:`M(z)` assembly shared by several independent q.

    A q-resolved device has one pole problem PER q -- ``M_q(z) = z^2 I - D(q) -
    Sigma^R_q(z)`` -- and the sets are unrelated, so the driver keeps a
    :class:`PoleSector` per q with its own tracker, promoted set and epoch
    counter. What the q have in common is the SHAPE of the work: the same
    block layout, the same frequency grid, the same number of Newton steps. So
    the solve is shared even though nothing else is.

    Batching within a q already removed the per-candidate Python loop. What is
    left after that is a loop over q, and on a device like Si -- 81 q, three
    6x6 blocks -- the per-q work is far too small to occupy a GPU: the whole
    operator assembly is two GEMMs on matrices of a few hundred entries. This
    class stacks the q, so the candidate axis the bordered Newton sees is
    ``nq * n_probe`` and every kernel launch does ``nq`` times more work.

    The candidate axis is FLATTENED rather than kept as ``(nq, P)``, so
    :func:`~quatrex.phonon.pole_nevp.bordered_newton_batch` and
    :mod:`~quatrex.phonon.btd_linalg` need no notion of q at all -- they see one
    batch of independent operators, which is what they already handle.

    Parameters
    ----------
    sectors : list[PoleSector]
        The per-q sectors, each with its operator context already set. They
        must share a frequency grid and a block layout; that is checked.
    n_probe : int
        Candidates per q. Shorter seed sets are padded by the caller.

    """

    def __init__(self, sectors: list["PoleSector"], n_probe: int):
        if not sectors:
            raise ValueError("PoleQBatch needs at least one sector.")
        first = sectors[0]
        for sec in sectors[1:]:
            if sec._layout is not first._layout:
                raise ValueError(
                    "the q sectors do not share a block layout; pass the same "
                    "BlockLayout to every set_operator_context.")
        self.sectors = list(sectors)
        self.nq = len(sectors)
        self.n_probe = int(n_probe)
        self.layout = first._layout
        self.cfg = first.cfg
        self.freqs = first.freqs
        self.global_freqs = first.global_freqs
        self._freq_offset = first._freq_offset
        self._reduce = first._reduce
        self._n_local = int(first._delta.shape[0])

        lay = self.layout
        self._delta_lay = xp.stack([s._delta_derived()[0] for s in sectors])
        self._mirror_lay = xp.stack([s._delta_derived()[1] for s in sectors])
        self._d_lay = xp.stack([
            xp.broadcast_to(s._d_lay.reshape(-1, lay.total)[0], (lay.total,))
            for s in sectors])
        self._obc = tuple(
            None if sectors[0]._obc[side] is None
            else xp.stack([xp.asarray(s._obc[side]) for s in sectors])
            for side in (0, 1))
        self._plan: LocalFitPlan | None = None

    # -- the pinned fit ----------------------------------------------------- #

    def set_anchors(self, anchors: NDArray) -> None:
        """Pin one fit stencil per (q, candidate); ``anchors`` is ``(nq, P)``."""
        self._anchors = xp.asarray(anchors, dtype=xp.float64).reshape(-1)
        self._plan = LocalFitPlan(
            self.global_freqs, self._anchors,
            order=self.cfg.delta_fit_order,
            window=self.cfg.delta_fit_window_cells,
        )

    def _weights(self, zf: NDArray, order: int) -> tuple[NDArray, NDArray]:
        """``(w_pos, w_mir)`` shaped ``(nq, P, n_local)`` for the batched GEMM."""
        w_pos, w_mir = continuation_weights(zf, self.global_freqs, order=order)
        f_pos, f_mir = self._plan.weights(zf, deriv=order)
        lo = self._freq_offset
        hi = lo + self._n_local
        shape = (self.nq, self.n_probe, hi - lo)
        return ((w_pos[:, lo:hi] + f_pos[:, lo:hi]).reshape(shape),
                (w_mir[:, lo:hi] + f_mir[:, lo:hi]).reshape(shape))

    def _continue_flat(self, zf: NDArray, order: int) -> NDArray:
        """``Sigma_s^{R,II}`` for every (q, candidate), ``(nq * P, total)``."""
        w_pos, w_mir = self._weights(zf, order)
        sig = (xp.matmul(w_pos, self._delta_lay)
               + xp.matmul(w_mir, self._mirror_lay))
        return self._reduce(sig).reshape(self.nq * self.n_probe,
                                         self.layout.total)

    def _obc_at(self):
        """Contact blocks at each candidate's anchor, ``(nq * P, b, b)``."""
        out = []
        for o in self._obc:
            if o is None or o.ndim < 4:
                # (nq, b, b) already, or absent: broadcast over the candidates.
                out.append(None if o is None
                           else xp.repeat(o, self.n_probe, axis=0))
                continue
            n_freq = int(o.shape[1])
            ref = self._anchors.reshape(self.nq, self.n_probe)
            k = xp.argmin(xp.abs(self.freqs[None, None, :] - ref[..., None]),
                          axis=-1)
            flat = (xp.arange(self.nq)[:, None] * n_freq + k).reshape(-1)
            out.append(o.reshape((self.nq * n_freq,) + o.shape[2:])[flat])
        return out[0], out[1]

    def operator(self):
        """``(m_blocks, dm_blocks)`` over the flattened ``(nq * P,)`` axis."""
        lay = self.layout
        last = len(self.sectors[0]._block_sizes) - 1
        d_lay = xp.repeat(self._d_lay, self.n_probe, axis=0)

        def m_blocks(z):
            zf = xp.asarray(z, dtype=xp.complex128).reshape(-1)
            flat = xp.zeros((zf.shape[0], lay.total), dtype=xp.complex128)
            flat[..., lay.diag] = (zf * zf)[:, None]
            flat = flat - d_lay - self._continue_flat(zf, 0)
            obc_l, obc_r = self._obc_at()
            for obc, i in ((obc_l, 0), (obc_r, last)):
                if obc is not None:
                    flat[..., lay.corner[i]] -= obc.reshape(obc.shape[0], -1)
            return lay.blocks(flat)

        def dm_blocks(z):
            zf = xp.asarray(z, dtype=xp.complex128).reshape(-1)
            flat = xp.zeros((zf.shape[0], lay.total), dtype=xp.complex128)
            flat[..., lay.diag] = 2.0 * zf[:, None]
            return lay.blocks(flat - self._continue_flat(zf, 1))

        return m_blocks, dm_blocks


def _pad_seeds(seeded, n_probe: int):
    """Pad every q's seed list to ``n_probe``, and say which slots are real.

    Candidate counts differ per q: ``harmonic_seeds`` keeps only the modes
    inside the pole window, and the predictor carries only the poles that
    survived last iteration. The batch needs a rectangle.

    Padding REPEATS a real seed rather than inventing one. A made-up ``z``
    could sit on a pole of the operator, and the batched ``inv`` behind the BTD
    factorisation raises for the whole batch if any single matrix is singular
    -- so one meaningless slot would take out every q. A duplicate is
    guaranteed to be as well conditioned as the seed it copies.
    """
    seeds, vectors, valid = [], [], []
    for sd, vec in seeded:
        n = len(sd)
        pad = n_probe - n
        seeds.append(list(sd) + [sd[-1]] * pad)
        vectors.append(None if vec is None else list(vec) + [vec[-1]] * pad)
        valid.append(n)
    return seeds, vectors, valid


def refresh_many(sectors: list["PoleSector"]) -> list[PoleSectorState]:
    """One SCBA iteration's worth of pole tracking for several q at once.

    Seeding, screening, clustering and tracking stay per q -- a mode at one q
    has no relation to one at another, and a shared tracker would either fuse
    two unrelated modes or churn membership every iteration. Only the SOLVE is
    shared, because that is the part whose cost is launch latency rather than
    arithmetic. See :class:`PoleQBatch`.
    """
    if not sectors:
        return []
    seeded = [sec._begin_refresh() for sec in sectors]
    live = [k for k, (sd, _) in enumerate(seeded) if len(sd)]
    states: list[PoleSectorState | None] = [None] * len(sectors)
    for k in range(len(sectors)):
        if k not in live:
            # Nothing in the window at this q: no solve, but the iteration
            # still has to advance so the epoch and the tracker stay in step.
            states[k] = sectors[k]._end_refresh([], [], None, None, None)
    if not live:
        return states

    sols = _solve_batched([sectors[k] for k in live],
                          [seeded[k] for k in live])

    # A failed correction is a tracking failure, not a reason to carry a pole:
    # re-seed from the harmonic spectrum and try once more. Only the q that
    # lost everything are re-solved, and they are re-solved together.
    again = [i for i, k in enumerate(live)
             if sectors[k].state.solutions and not any(s.converged
                                                       for s in sols[i])]
    if again:
        for i in again:
            sectors[live[i]].tracker.rescan_reasons.append(
                "no pole survived the corrector")
        reseed = [(sectors[live[i]].harmonic_seeds(), None) for i in again]
        keep = [i for i, r in zip(again, reseed) if len(r[0])]
        if keep:
            redo = _solve_batched([sectors[live[i]] for i in keep],
                                  [reseed[again.index(i)] for i in keep])
            for slot, out in zip(keep, redo):
                sols[slot] = out
                seeded[live[slot]] = reseed[again.index(slot)]

    for i, k in enumerate(live):
        sd, vec = seeded[k]
        states[k] = sectors[k]._end_refresh(sols[i], sd, vec, None, None)
    return states


def _solve_batched(sectors, seeded) -> list[list[PoleSolution]]:
    """The shared corrector: one bordered Newton for every (q, candidate)."""
    n_probe = max(len(sd) for sd, _ in seeded)
    seeds, vectors, valid = _pad_seeds(seeded, n_probe)
    n_dof = int(np.sum(sectors[0]._block_sizes))

    # Trust radii come from the UNPADDED seed lists: a duplicated seed sits at
    # zero distance from its original, and the nearest-competing-seed rule
    # would then collapse the real candidate's search region to the floor.
    radii = xp.stack([
        xp.concatenate([xp.asarray(sec.trust_radii(sd[:n])),
                        xp.full(n_probe - n, xp.inf)])
        for sec, sd, n in zip(sectors, seeds, valid)])

    z0 = xp.asarray([[complex(x) for x in sd] for sd in seeds],
                    dtype=xp.complex128)
    r0 = None
    if any(v is not None for v in vectors):
        # Where a q has no predicted vector, use the same random unit vector
        # the unbatched solve would have drawn for it.
        rng = xp.random.default_rng(0)
        default = rng.standard_normal(n_dof) + 1j * rng.standard_normal(n_dof)
        r0 = xp.stack([
            xp.stack([xp.asarray(v).reshape(-1) for v in vec])
            if vec is not None else xp.broadcast_to(default, (n_probe, n_dof))
            for vec in vectors]).reshape(-1, n_dof)

    batch = PoleQBatch(sectors, n_probe)
    batch.set_anchors(xp.real(z0))
    m_blocks, dm_blocks = batch.operator()
    cfg = sectors[0].cfg
    out = bordered_newton_batch(
        m_blocks, dm_blocks, z0.reshape(-1), r0,
        tol=cfg.newton_tol, max_iter=cfg.newton_max_iterations,
        trust_radius=radii.reshape(-1),
    ).to_list()
    return [out[q * n_probe:q * n_probe + n]
            for q, n in enumerate(valid)]
