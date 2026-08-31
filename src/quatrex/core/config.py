# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the configuration classes for the quatrex package."""

import os
import re
import subprocess
import tomllib
import warnings
from pathlib import Path
from typing import Literal

import numba as nb
import numpy as np
from mpi4py.MPI import COMM_WORLD as mpi_comm_world
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from qttools import xp
from qttools.comm import comm
from qttools.datastructures import DSDBCOO, DSDBSparse
from qttools.profiling import Profiler
from quatrex.electrostatics.geometry_config import GeometryConfig, parse_geometry_config

profiler = Profiler()


class SCSPConfig(BaseModel):
    """Parameters controlling the self-consistent Schrödinger-Poisson loop.

    For more information on the self-consistent Schrödinger-Poisson
    loop, see the [section on
    electrostatics](../methodology/electrostatics.md) in the user guide.

    """

    model_config = ConfigDict(extra="forbid")

    min_iterations: PositiveInt = 1
    """The minimum number of Schrödinger-Poisson iterations to perform."""

    max_iterations: PositiveInt = 100
    """The maximum number of Schrödinger-Poisson iterations to perform."""

    convergence_tol: PositiveFloat = 1e-3
    r"""The convergence tolerance for the potential in the
    Schrödinger-Poisson loop.

    This is defined as the infinity norm of the difference between the
    potential in the current iteration and the previous iteration.

    $$
        \lVert V_{n} - V_{n-1} \rVert_{\infty} < \texttt{convergence_tol}
    $$

    """

    # Parameters for potential mixing.
    mixer: Literal["under-relaxation", "diis"] = "under-relaxation"
    """The mixing scheme to use for the self-consistent solution of the
    Poisson equation.

    - `"under-relaxation"`: Simple under-relaxation scheme where the new
      potential is a weighted average of the previous potential and the
      newly computed potential. The weight is given by the
      `mixing_factor` parameter.
    - `"diis"`: Direct inversion in the iterative subspace (DIIS) method
      which constructs the new potential as a linear combination of the
      previous potentials and the newly computed potential. The
      coefficients of the linear combination are determined by
      minimizing the residuals of the previous potentials.

    """

    mixing_factor: PositiveFloat = Field(default=0.75, le=1.0)
    """Under-relaxation factor for the under-relaxation mixer. Should be
    between 0 and 1.

    """

    adaptive_mixing: bool = False
    """Whether to adaptively adjust the mixing factor based on the
    convergence behavior.

    If `True`, the mixing factor is adjusted based on the convergence
    behavior. If the residual between two potential iterations is larger
    than the previous iteration, the mixing factor is reduced by 50%. If
    it is smaller, the mixing factor is increased by 10%.

    """

    max_history: PositiveInt = 3
    """Maximum number of previous potentials and residuals to store for
    the DIIS extrapolation.

    Only used if [`mixer`](#mixer) is set to "diis".

    """

    epsilon: PositiveFloat = 1e-5
    """Regularization parameter for the least-squares problem in the
    DIIS method to ensure numerical stability.

    Only used if [`mixer`](#mixer) is set to "diis".

    """

    extrapolation_interval: PositiveInt = 1
    """Number of iterations between DIIS extrapolation steps.

    For example, if set to 3, the mixer will perform two
    under-relaxation steps followed by a DIIS extrapolation step, and
    then repeat this cycle. If set to 1 (the default), the Pulay mixing
    is performed at every iteration.

    Only used if [`mixer`](#mixer) is set to "diis".

    """


class QTBMConfig(BaseModel):
    """Parameters for the quantum transmitting boundary method (QTBM).

    !!! note
        Only used in simulations where
        [`formalism`](quatrex#formalism) is set to "wf".

    """

    model_config = ConfigDict(extra="forbid")

    max_batch_size: PositiveInt = 10
    """The maximum number of energies that are batched together when
    computing open boundary conditions (OBCs) in the QTBM solver.

    This can be used to reduce the memory footprint of the QTBM solver,
    at the cost of increased computation time.

    """

    low_rank_obc: bool = False
    """Whether to use reduced rank for the boundary self-energies.

    If set to True, boundary self-energies are moved to the right-hand-side of
    the linear system, which greatly reduces fill-in during factorization.

    The system matrix becomes Hermitian or even real symmetric in gamma-only
    simulations. Therefore, the `low_rank_obc` parameter can only be used in
    combination with direct solvers that can exploit the symmetry, i.e.,
    [`direct_solver`](solver#direct_solver)="cudss" on GPU,
    [`direct_solver`](solver#direct_solver)="pardiso" on CPU, and
    [`direct_solver`](solver#direct_solver)="thomas" on both CPU and GPU.

    !!! warning
        Potentially the system matrix can become very ill-conditioned if an
        energy is too close to a Van Hove singularity. Without low rank, the
        OBCs regularize the system matrix.

    """

    atom_resolved_outputs: bool = False
    """Whether to output atomic-resolved observables instead of
    orbital-resolved observables"""


class PoleSectorConfig(BaseModel):
    """Pole-subtracted SCBA: carry narrow resonances of G analytically.

    A uniform real-frequency grid must resolve the SHARPEST linewidth in the
    problem, and a grid that does not is not merely imprecise: the discrete
    three-phonon convolution of two sub-grid modes overshoots by orders of
    magnitude depending on where the bins fall (measured: 6.4x too large at
    dw/gamma = 20 on the elementary pair convolution), and an under-resolved
    run reports its own grid spacing as the physics
    (``phonon/docs/grid_audit.md``).

    This sector removes those modes from the grid instead. Poles of
    ``G^R = M^R(z)^{-1}`` are found by a nonlinear eigenvalue solve, subtracted
    from the bubble legs, convolved in closed form, and their retarded partner
    reconstructed analytically rather than through the Hilbert transform. The
    self-energy that the pole solve needs is SMOOTH -- it is a convolution -- so
    a grid far too coarse to resolve the line still samples it perfectly well.

    The construction is a change of representation, not of the diagram: the
    split ``G = G_S + G_R`` is exact, and all four bubble sectors are retained.
    See ``phonon/docs/pole_scba_implemented.md``. Research use only;
    ``enabled = False`` is legacy (bit-identical)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    """Master switch. False = legacy (bit-identical): no probe is assembled and
    no call site is entered."""

    sectors: Literal["rr", "rr_ss", "rr_ss_sr"] = "rr_ss_sr"
    """Which bubble sectors the hybrid evaluates, as a staging control.
    ``"rr_ss_sr"`` is the complete method: the existing FFT ring on the
    remainder (RR), the analytic pole-pole convolution (SS), and the mixed
    pole-background terms (SR + RS) through the existing linearised bubble.
    ``"rr_ss"`` and ``"rr"`` DROP physical three-phonon processes and exist only
    to measure the size of what they drop -- neither is a production setting."""

    leg: Literal["congruence", "congruence_analytic", "keldysh"] = "congruence"
    """WHICH Green's function the pole split is applied to."""

    omega_min_thz: NonNegativeFloat = 0.0
    """Lower edge of the pole search (THz). Below it the quasiparticle picture
    does not apply: the Bose factor carries its own 1/omega pole, the acoustic
    resolvent is near-singular, and the lead-driven translation channel lives
    there. 0 = auto, resolved to ``max(4*dw, sse_low_freq_mask_thz + 2*dw)``."""
    omega_max_thz: NonNegativeFloat = 0.0
    """Upper edge of the pole search (THz). 0 = the top of the frequency grid."""
    max_poles: PositiveInt = 16
    """Cap on simultaneously tracked poles. The coherent pole-pole bubble scales
    as the fourth power of this, so it is a cost control, not a fidelity one."""

    sheet: Literal["physical", "outgoing"] = "physical"
    """Which sheet the contact self-energy is continued onto. ``"physical"``
    keeps the |lambda| < 1 selection and is correct for modes in a lead gap or
    quasi-bound states with negligible contact broadening. ``"outgoing"``
    continues the lead Bloch roots from the real axis and is what an IN-BAND
    resonance requires; it needs the spectral OBC, which exposes the modes."""

    samples_per_halfwidth: PositiveFloat = 2.0
    """``p_Gamma``: samples per half-width below which a mode counts as under-
    resolved and is promoted."""
    q_in: PositiveFloat = 1.0
    """Promote when the resolution score falls below this."""
    q_out: PositiveFloat = 2.0
    """Demote only above this. The gap to ``q_in`` is hysteresis: a mode that
    changes sector every iteration makes the fixed-point map discontinuous."""

    cluster_factor: PositiveFloat = 3.0
    """Single-linkage clustering radius, in units of the summed half-widths."""
    condition_reject: PositiveFloat = 1e5
    """Above this, refuse to promote at all -- near a defective point the
    simple-pole expansion itself fails."""
    band_edges: Literal["none", "lead"] = "none"
    """Where ``edge_factor`` gets the branch points it refuses poles near."""
    edge_factor: PositiveFloat = 5.0
    """Refuse promotion within this many half-widths of a contact band edge.
    Band edges are branch points, not simple poles; forcing one into a
    single-pole fit is the failure mode of Sec. 41."""

    newton_tol: PositiveFloat = 1e-10
    """Acceptance threshold on the scaled nonlinear-eigenvalue residual."""
    accept: Literal["locate", "residual"] = "locate"
    """Which quantity decides that a pole was found."""
    locate_tol: PositiveFloat = 0.05
    """Acceptance threshold on ``eps_z`` under ``accept="locate"``."""
    freeze_membership: bool = False
    """Hold sector MEMBERSHIP fixed except on ``epoch_iterations`` boundaries."""
    audit_every_iteration: bool = True
    """Offer the full harmonic candidate set EVERY iteration, not only on a
    tracker rescan."""
    locate_tol_out: PositiveFloat = 0.15
    """Demotion threshold on ``eps_z``: a pole already in the sector is only
    dropped above THIS, not above ``locate_tol``."""
    newton_max_iterations: PositiveInt = 8
    """Bordered-Newton steps per pole per SCBA iteration."""
    trust_radius_cells: PositiveFloat = 0.25
    """FLOOR on the Newton trust radius, in grid cells."""
    trust_factor: PositiveFloat = 0.5
    """Physical trust radius as a fraction of the nearest competing scale."""
    delta_fit_order: NonNegativeInt = 2
    """Degree of the local polynomial continuation of Sigma^> - Sigma^<, which
    is the second-sheet term."""
    delta_fit_window_cells: PositiveInt = 4
    """Half-width of that fit window, in grid cells."""
    rescan_iterations: PositiveInt = 10
    """Force a full harmonic rescan every N SCBA iterations even when tracking looks
    healthy."""
    subspace_angle_tol: PositiveFloat = 0.35
    """Largest principal angle (rad) still accepted as the same cluster.
    Subspaces are tracked, not individual modes: near a crossing the
    eigenvectors rotate arbitrarily while the invariant subspace is smooth."""
    epoch_iterations: PositiveInt = 5
    """Hold sector membership fixed for this many SCBA iterations. An
    approximate implementation is not invariant under repartitioning, so a mode
    that jumps sector every iteration changes the fixed-point map itself."""

    source_fit_tol: PositiveFloat = 0.1
    """Relative fit residual above which the cluster is DEMOTED rather than
    approximated. A source that is not smooth across its own pole window has no
    business being carried analytically."""
    mixed_scale: float = 1.0
    """DIAGNOSTIC scale on the injected ``Sigma_SR + Sigma_RS``."""
    cell_average: bool = True
    """Emit the analytic sectors as CELL AVERAGES rather than point samples."""

    psd_check: bool = False
    """Per-iteration positivity gate on the reconstructed total (never on
    individual sectors -- only their sum is constrained). Nothing in the solver
    checks this today, and the sector is the first thing that can break it
    structurally."""
    leg_weight_tol: NonNegativeFloat = 0.05
    """Worst-case line-weight error above which a mode counts as unresolved."""

    leg_weight_tol_out: NonNegativeFloat = 0.0
    """Demotion threshold for ``leg_weight_tol``; 0 means "use
    ``leg_weight_tol / 3``"."""
    bubble_correction: Literal["none", "local_covariance"] = "none"
    """Replace the ring's cell-mean product on ACTIVE cell pairs by the exact
    finite-cell integral."""
    covariance_sigma_min: NonNegativeFloat = 0.0
    """Cell activity floor for ``bubble_correction``, relative to the largest
    ``sigma_k`` on the axis. 0 corrects every cell that carries poles."""
    q_stride: PositiveInt = 1
    """Solve every ``q_stride``-th transverse q. 1 = all of them."""
    q_max: NonNegativeInt = 0
    """Hard cap on how many q are solved (0 = no cap), applied after
    ``q_stride``."""
    q_batch: NonNegativeInt = 0
    """How many q share one bordered-Newton solve. 0 = all of them."""
    extraction_only: bool = False
    """Run the pole SOLVE and print the census, then allocate NO sector."""

    @model_validator(mode="after")
    def check_pole_sector_consistency(self) -> Self:
        """Internal consistency of the pole block."""
        if not self.enabled:
            return self
        if self.q_out <= self.q_in:
            raise ValueError(
                f"pole_sector: q_out ({self.q_out}) must exceed q_in "
                f"({self.q_in}); the gap IS the hysteresis that stops a mode "
                "changing sector every iteration."
            )
        if self.locate_tol_out <= self.locate_tol:
            raise ValueError(
                f"pole_sector: locate_tol_out ({self.locate_tol_out}) must "
                f"exceed locate_tol ({self.locate_tol}); the gap IS the "
                "hysteresis that stops a mode changing sector every "
                "iteration."
            )
        if (self.leg_weight_tol > 0.0 and self.leg_weight_tol_out > 0.0
                and self.leg_weight_tol_out >= self.leg_weight_tol):
            raise ValueError(
                f"pole_sector: leg_weight_tol_out "
                f"({self.leg_weight_tol_out}) must be BELOW leg_weight_tol "
                f"({self.leg_weight_tol}); this gate refuses a pole the grid "
                "already resolves, so hysteresis lowers the threshold."
            )
        if 2 * self.delta_fit_window_cells < self.delta_fit_order + 1:
            raise ValueError(
                f"pole_sector: delta_fit_window_cells "
                f"({self.delta_fit_window_cells}) gives "
                f"{2 * self.delta_fit_window_cells} samples, too few for a "
                f"degree-{self.delta_fit_order} fit."
            )
        if self.omega_max_thz and self.omega_max_thz <= self.omega_min_thz:
            raise ValueError(
                f"pole_sector: omega_max_thz ({self.omega_max_thz}) must exceed "
                f"omega_min_thz ({self.omega_min_thz})."
            )
        return self


class ExperimentalMixerConfig(BaseModel):
    """Experimental SCBA root-finders for the iteration-UNSTABLE eta->0 fixed
    point, kept out of the shared SCBA surface. Damped/Anderson mixing
    accelerates a contractive iteration, whereas broyden/rpm/rre/jfnk LAND a
    fixed point whose Jacobian has ``|lambda| > 1`` (which mixing cannot
    reach). Research use only."""

    model_config = ConfigDict(extra="forbid")

    rre_cycle: PositiveInt = 8
    """For ``mixing_method = "rre"``: restart cycle length (number of iterates
    per reduced-rank-extrapolation step). Cf. ``RREMixer`` in
    ``quatrex/core/anderson.py``."""
    rre_ridge: NonNegativeFloat = 1e-6
    """For ``mixing_method = "rre"``: Tikhonov ridge on the (rank-deficient by
    construction, once the residuals go collinear) Gram solve, scaled by its
    mean diagonal. 0 = unregularised."""

    broyden_warmup_iters: NonNegativeInt = 0
    """For ``mixing_method = "broyden"`` / ``"rpm"``: run plain damped-LINEAR
    mixing for the first N SCBA iterations (still accumulating secant history),
    then engage the quasi-Newton / projection step. 0 = engage from the
    start."""
    broyden_ridge: NonNegativeFloat = 1e-8
    """For ``mixing_method = "broyden"`` / ``"rpm"``: Tikhonov ridge on the small
    multisecant / restricted-Jacobian solve, scaled by the matrix norm. Kept
    TINY so it does not damp the Newton correction in the near-marginal
    subspace."""
    broyden_trust: NonNegativeFloat = 0.3
    """For ``mixing_method = "broyden"`` / ``"rpm"``: trust-region step cap --
    the per-iteration update ``||Sigma_new - Sigma||`` is limited to
    ``broyden_trust * ||Sigma||``. Guards against quasi-Newton overshoot far
    from the root; 0 disables it."""
    rpm_max_subspace: PositiveInt = 6
    """For ``mixing_method = "rpm"``: cap on the dimension k of the unstable
    invariant subspace on which Newton is performed (Picard on the contractive
    complement). A single complex pair needs k = 2; 6 leaves margin. Cf.
    ``RPMMixer`` in ``quatrex/core/rpm.py``."""

    # --- Jacobian-free Newton-Krylov (mixing_method = "jfnk") -----------------
    # GMRES on the matrix-free Newton system J delta = -R (finite-difference
    # J*v products), run in the real embedding [Re Sigma, Im Sigma]; lands
    # strongly-unstable fixed points where the subspace-tracking RPM fails.
    # Cf. ``quatrex/core/jfnk.py``.
    jfnk_warmup_iters: NonNegativeInt = 10
    """For ``mixing_method = "jfnk"``: damped-LINEAR steps to fall into the
    fixed point's basin before engaging Newton-Krylov."""
    jfnk_max_krylov: PositiveInt = 30
    """For ``mixing_method = "jfnk"``: maximum GMRES (Arnoldi) dimension per Newton
    step = max map evaluations per Newton step. ``n_unstable + a few`` suffices."""
    jfnk_inner_tol: NonNegativeFloat = 0.1
    """For ``mixing_method = "jfnk"``: base relative GMRES tolerance for the inner
    Newton solve. With ``jfnk_forcing = "ew"`` it is tightened adaptively as the
    outer residual falls (inexact Newton, Eisenstat-Walker)."""
    jfnk_forcing: Literal["ew", "fixed"] = "ew"
    """For ``mixing_method = "jfnk"``: inner-tolerance forcing. ``"ew"`` =
    Eisenstat-Walker (tighten with the outer residual, quadratic-ish near the
    root); ``"fixed"`` = constant ``jfnk_inner_tol``."""
    jfnk_max_newton: PositiveInt = 60
    """For ``mixing_method = "jfnk"``: cap on outer Newton steps (each ~ a GMRES
    cycle of map evaluations)."""
    jfnk_eps: PositiveFloat = 1e-7
    """For ``mixing_method = "jfnk"``: relative finite-difference step for the
    matrix-free Jacobian-vector product, ``eps_used = jfnk_eps * (1 + ||Sigma||)``."""
    jfnk_trust: NonNegativeFloat = 0.5
    """For ``mixing_method = "jfnk"``: INITIAL trust-region cap on the Newton
    step, ``||delta|| <= jfnk_trust * ||Sigma||`` (global). Adapts down on a
    step that raises the residual, up on good progress (toward
    ``jfnk_trust_max``); 0 disables. Start it SMALL (e.g. 0.05) so the early,
    far-from-root Newton steps cannot overshoot."""
    jfnk_trust_max: NonNegativeFloat = 0.0
    """For ``mixing_method = "jfnk"``: MAXIMUM trust radius the adaptive growth
    may reach (``<= 0`` -> use ``jfnk_trust``, i.e. no growth above the
    initial). Set ``jfnk_trust_max > jfnk_trust`` to let the radius grow as
    the residual descends monotonically."""
    jfnk_newton_damp: PositiveFloat = 1.0
    """For ``mixing_method = "jfnk"``: damping of the (already trust-capped) Newton
    step, ``Sigma_{k+1} = Sigma_k + jfnk_newton_damp * delta``. < 1 globalises a
    far-from-root start."""
    jfnk_ptc: NonNegativeFloat = 0.0
    """For ``mixing_method = "jfnk"``: pseudo-transient / Levenberg-Marquardt
    shift -- solve ``(J + mu I) delta = -R`` with
    ``mu = jfnk_ptc * ||R_k||/||R_0||`` (annealed to 0 as the residual falls,
    recovering pure Newton at the root). The shift lifts the near-zero
    (marginal) eigenvalues of ``J = J_F - I`` off the origin so the inner
    GMRES no longer stalls on the near-null-space. 0 = pure Newton."""

    # --- exact-Jacobian Newton-Krylov (mixing_method = "newton") --------------
    # Same Newton system as jfnk, but the Jacobian-vector products are the
    # EXACT analytic linearisation (frozen-G Dyson identity + polarisation
    # identity of the quadratic bubble; cf. ``quatrex/core/phonon_jvp.py``),
    # computed synchronously inside one mixer call -- no finite differences,
    # no probe iterates. Phonon-only; requires a stationary map (no ramps,
    # no Buttiker probe, no SCP tadpole, bare-lead contacts, rgf solver).
    newton_warmup_iters: NonNegativeInt = 5
    """For ``mixing_method = "newton"``: minimum damped-Picard iterations
    before Newton may engage (basin capture)."""
    newton_switch_tol: NonNegativeFloat = 1e-2
    """For ``mixing_method = "newton"``: engage Newton once the residual has
    dropped below ``newton_switch_tol * ||R_first||`` (two-phase
    globalisation). ``>= 1`` engages immediately after the warmup count."""
    newton_max_krylov: PositiveInt = 30
    """For ``mixing_method = "newton"``: maximum GMRES dimension per Newton
    step. Each Krylov vector costs two bubble evaluations (the polarisation
    identity), all inside one SCBA iteration."""
    newton_inner_tol: NonNegativeFloat = 0.1
    """For ``mixing_method = "newton"``: base relative GMRES tolerance (used
    directly when ``newton_forcing = "fixed"``)."""
    newton_forcing: Literal["ew", "fixed"] = "ew"
    """For ``mixing_method = "newton"``: Eisenstat-Walker forcing ("ew") or
    fixed inner tolerance."""
    newton_max_newton: PositiveInt = 100
    """For ``mixing_method = "newton"``: cap on Newton steps; afterwards the
    mixer falls back to damped Picard."""
    newton_trust: NonNegativeFloat = 0.5
    """For ``mixing_method = "newton"``: initial trust-region cap
    ``||delta|| <= newton_trust * ||Sigma||``; adapts with hysteresis
    (shrinks on a rejected step, grows toward ``newton_trust_max``).
    0 disables."""
    newton_trust_max: NonNegativeFloat = 0.0
    """For ``mixing_method = "newton"``: maximum adaptive trust radius
    (``<= 0`` -> no growth above ``newton_trust``)."""
    newton_damp: PositiveFloat = 1.0
    """For ``mixing_method = "newton"``: fixed damping of the accepted
    (trust-capped) Newton step."""
    newton_backtrack: NonNegativeInt = 3
    """For ``mixing_method = "newton"``: maximum step-halvings when a Newton
    step increased ``||R||`` (the trial residual arrives with the next SCBA
    iteration, so each halving costs one fixed-point sweep). 0 disables the
    accept/reject test."""
    newton_recon_check_tol: PositiveFloat = 1e-8
    """For ``mixing_method = "newton"``: relative tolerance of the frozen-G
    reconstruction self-check run at every Newton step (dense reassembled
    G^{<,>} against the solver's RGF output). A failure aborts instead of
    silently corrupting the Krylov space."""
    newton_precond: Literal["none", "recycle", "fresh"] = "none"
    """For ``mixing_method = "newton"``: low-rank deflation right
    preconditioner for the inner GMRES. ``"recycle"`` harvests harmonic
    Ritz pairs (the near-singular directions) from the previous Newton
    step's Arnoldi relation -- exact operator images at zero extra kernel
    cost; ``"fresh"`` builds the basis at the current step and spends
    ``newton_precond_rank`` exact JVPs on their images (the literal
    low-rank Schur surrogate); ``"none"`` disables. Memory: 2 x rank
    extra state vectors."""
    newton_precond_rank: PositiveInt = 8
    """For ``mixing_method = "newton"``: rank of the stored deflation
    basis (number of (direction, image) pairs)."""
    newton_jvp_form: Literal["bilinear", "polarization"] = "bilinear"
    """For ``mixing_method = "newton"``: evaluation route of the exact
    bubble derivative. ``"bilinear"`` is the mixed-leg cross
    B(dG, G) + B(G, dG) through ``compute_linearized`` (no subtraction of
    large terms -- uniformly exact to rounding); ``"polarization"`` is
    S(G+dG) - S(G) - S(dG) through three calls to the unmodified
    production kernel (kept as the independent cross-check; loses digits
    on very small or nearly-annihilated directions). The bilinear route
    requires the symmetry fast paths (``sse_greater_from_lesser``,
    ``sse_hermitian_pairs``) off; with either enabled it falls back to
    polarization with a notice."""


class SCBAConfig(BaseModel):
    """Parameters for the self-consistent Born approximation (SCBA)
    loop.

    This is the main loop that computes the self-energies and Green's
    functions in simulations where [`formalism`](quatrex#formalism)
    is set to "negf".

    See the [section on NEGF](../methodology/negf.md) in the user guide
    for more information on the SCBA loop.

    """

    model_config = ConfigDict(extra="forbid")

    min_iterations: PositiveInt = 1
    """The minimum number of SCBA iterations to perform.

    This must be greater than or equal to 1.

    !!! warning
        This parameter currently has no effect.

    """

    max_iterations: PositiveInt = 100
    """The maximum number of SCBA iterations to perform."""

    convergence_tol: PositiveFloat = 1e-5
    """The convergence tolerance for the SCBA iterations.

    !!! warning
        This parameter currently has no effect.

    """

    mixing_factor: PositiveFloat = Field(default=0.1, le=1.0)
    """Damped self-energy mixing factor: for ``mixing_method = "linear"``
    the update is ``Sigma <- (1-a) Sigma_prev + a Sigma_new``; for
    ``"anderson"`` it is the damping ``beta`` of the accelerated step."""

    mixing_method: Literal[
        "linear", "anderson", "broyden", "rre", "rpm", "jfnk", "newton"
    ] = "linear"
    """Self-energy fixed-point mixer. ``"linear"`` is plain damped mixing;
    ``"anderson"`` is Anderson(m) acceleration (helps a convergent iteration;
    NOT a cure for a marginal / non-existent fixed point) -- cf.
    ``quatrex/core/anderson.py``. ``"broyden"`` (type-I "good" Broyden root
    finder), ``"rpm"`` (Recursive Projection Method), ``"rre"`` and
    ``"jfnk"`` can LAND an iteration-UNSTABLE fixed point (Jacobian
    |lambda|>1) that damped/Anderson mixing cannot reach -- cf.
    ``quatrex/core/broyden.py``, ``quatrex/core/rpm.py`` and
    ``quatrex/core/jfnk.py``. ``"newton"`` is Newton-Krylov with the EXACT
    analytic Jacobian-vector product (phonon-only; cf.
    ``quatrex/core/newton.py`` and ``quatrex/core/phonon_jvp.py``)."""

    anderson_depth: PositiveInt = 5
    """History size m for ``mixing_method = "anderson"`` (number of stored
    residual differences). Memory cost: 2*m copies of the full Sigma."""
    anderson_period: PositiveInt = 1
    """Periodic-Pulay stride: apply the Anderson extrapolation only every
    ``anderson_period``-th iteration, plain damped linear mixing in between.
    >1 breaks the marginal-mode limit cycle that stalls plain Anderson on
    soft-mode systems. 1 = ordinary Anderson(m)."""
    anderson_warmup_iters: NonNegativeInt = 0
    """For ``mixing_method = "anderson"``: run plain damped LINEAR mixing for
    the first ``anderson_warmup_iters`` SCBA iterations, then switch to
    Anderson. Avoids the limit cycle of a cold-started Anderson on the early
    non-linear transient. 0 = Anderson from the start."""
    anderson_restart: NonNegativeInt = 0
    """For ``mixing_method = "anderson"``: forget the Anderson history every N
    steps (0 = never). Breaks marginal-mode limit cycles that periodic Pulay
    alone cannot escape."""
    anderson_ridge: NonNegativeFloat = 0.0
    """For ``mixing_method = "anderson"``: scale-relative Tikhonov
    regularisation of the least-squares coefficients (suppresses overshoot
    from a near-rank-deficient history). 0 = plain SVD lstsq. Also reused as
    the Gram ridge for ``mixing_method = "rre"``."""
    anderson_step_cap: NonNegativeFloat = 0.0
    """Safeguard: reject the Anderson step when its correction norm exceeds
    ``anderson_step_cap`` x the UNDAMPED residual norm ||f||, taking the
    damped-linear step that iteration (history kept). Interpretable as the
    largest 1/(1-lambda) the extrapolation is trusted for, independent of
    the mixing factor. 0 = off; typical 10-50."""
    anderson_revert_factor: NonNegativeFloat = 0.0
    """Safeguard: when the residual exceeds ``anderson_revert_factor`` x the
    best residual seen, clear the history and return the best iterate.
    0 = off; typical 3.0-10.0."""
    anderson_stagnation_restart: NonNegativeInt = 0
    """Safeguard: clear the Anderson history after N consecutive
    non-improving steps (gentler than restarting on every uptick). 0 = off;
    typical 5."""
    mixer_diagnostics: bool = False
    """Collect per-iteration mixer diagnostics (global residual norm, LS
    conditioning, |gamma|, step kind / safeguard flags) on the mixer object;
    the study engine persists them into the run npz. Adds one small
    collective per iteration when enabled."""

    experimental_mixer: ExperimentalMixerConfig = Field(
        default_factory=ExperimentalMixerConfig)
    """Knobs for the experimental eta=0 root-finders (broyden / rpm / rre / jfnk),
    selected via ``mixing_method``. Grouped here to keep the experimental
    fixed-point machinery out of the shared SCBA surface; see
    :class:`ExperimentalMixerConfig`."""

    abort_residual: NonNegativeFloat = 1e3
    """Divergence guard: abort the SCBA once the relative self-energy
    residual exceeds this after iteration 3 (an exploded update never
    recovers). 0 disables."""

    output_interval: PositiveInt = 1
    """The interval at which to output observables during the SCBA iterations.

    !!! warning
        This parameter currently has no effect.

    """

    coulomb_screening: bool = False
    """Whether to include screened Coulomb interactions."""

    photon: bool = False
    """Whether to include electron-photon interactions."""

    phonon: bool = False
    """Whether to include electron-phonon interactions."""

    symmetric: bool = True
    """Whether to exploit symmetry in NEGF calculations.

    All lesser and greater quantitiese are skew-Hermitian, allowing us
    to only store and compute the upper triangular part of the matrices.

    The retarded quantities can be decomposed into a Hermitian and
    skew-Hermitian part, which also allows memory and computation
    savings.

    This can reduce the memory footprint and computation time by a
    significant factor, especially for large systems.

    """

    align_self_energy_to_complex_axes: bool = True
    r"""Whether to discard certain parts of the self-energy.

    This is an approximation that affects the self-energy in the
    following way:

    - The real parts of the lesser/greater self-energy are discarded.
    - The imaginary part of the retarded self-energy from any previous
      computation is discarded.

    $$
    \begin{equation}
    \mathbf{\Sigma}^R_{AH} = \frac{1}{2i} (
    \mathbf{\Sigma}^> - \mathbf{\Sigma}^< )
    \label{eq:retarded_self_energy_from_lesser_greater}
    \end{equation}
    $$

    This happens before the anti-Hermitian part of the retarded
    self-energy is computed from the lesser and greater parts as in
    Equation $\ref{eq:retarded_self_energy_from_lesser_greater}$.

    """


class ElectrostaticsConfig(BaseModel):
    """Parameters for the electrostatics calculations."""

    model_config = ConfigDict(extra="forbid")

    orbital_basis: Literal["point-charge"] = "point-charge"
    """The orbital basis to use to transform between the real-space and
    orbital-space representations of the potential and charge density.

    Currently, only the "point-charge" basis is supported. Each orbital
    is represented as a point charge located at the corresponding atomic
    position.

    """

    solving_scheme: Literal["root-finding", "direct"] = "root-finding"
    """The scheme to solve the non-linear Poisson equation.

    - `"root-finding"`: Solves the Poisson equation using an iterative
      predictor-corrector scheme where the charge density response is
      computed from the potential using a density model and the Poisson
      equation is solved iteratively until convergence.
    - `"direct"`: Solves the Poisson equation directly using a linear
      solver. Due to the non-linearity of the Schrödinger-Poisson
      problem, this scheme is not recommended and should only be used
      with very cautious mixing and a good initial guess.

    """

    max_iterations: PositiveInt = 20
    """The maximum number of inner iterations for the root-finding scheme.

    Only used if [`solving_scheme`](#solving_scheme) is set to
    "root-finding".

    """

    convergence_tol: PositiveFloat = 1e-3
    """The convergence tolerance for the root-finding scheme.

    This is defined as the infinity norm of the potential update in the
    root-finding scheme.

    Only used if [`solving_scheme`](#solving_scheme) is set to
    "root-finding".

    """

    density_model: Literal["single-band", "omen"] = "single-band"
    """The density model to use for the root-finding scheme.

    - `"single-band"`: Uses a simple single-band density model where the
      charge density is computed from the potential using a single-band
      approximation.
    - `"omen"`: Uses the density model from the OMEN code. This is
      almost identical to the single-band model with `density_model_dim
      = 2`. However, it uses slightly different physical constants,
      i.e., not the CODATA values used everywhere else in `quatrex`.

    Only used if [`solving_scheme`](#solving_scheme) is set to
    "root-finding".

    """

    density_model_dim: Literal[1, 2, 3] = 2
    """The dimensionality of the system to use for the single-band
    density model.

    The density model does not have to match the actual dimensionality
    of the system. For example, a 2D density model might actually work
    best for systems of all dimensionalities.

    Only used if [`solving_scheme`](#solving_scheme) is set to
    "root-finding" and [`density_model`](#density_model) is set to
    "single-band".

    """

    initial_guess: Literal["zero", "constraints", "file"] = "zero"
    """The strategy to generate the initial guess for the potential.

    - `"zero"`: Uses a zero potential as the initial guess.
    - `"constraints"`: Solves a linear Poisson equation with the
        potential constraints to generate the initial guess. This is
        expected to work best at regimes close to equilibrium where the
        potential does not vary too much.
    - `"file"`: Loads the initial guess from a file. The file should be
        located in the [`input_dir`](quatrex#input_dir) and named
        `potential.npy`.

    """

    default_epsilon_r: PositiveFloat = 1.0
    """The default relative permittivity to use for the Poisson solver.

    This is used as a fallback for regions that do not have a specified
    relative permittivity.

    """

    electron_affinity: float | None = None
    """The electron affinity of the semiconductor channel.

    This is used to align the voltage levels of any gates to the
    semiconductor channel levels in SCSP runs. If not set, the voltages
    of the gates are taken as absolute values without any alignment,
    i.e. they are directly used as the Dirichlet boundary conditions for
    the Poisson equation.

    """


class MemoizerConfig(BaseModel):
    """Parameters for memoizing wrappers.

    The memoizers store and reuse previously computed results
    to speed up the fixed-point iterations in OBC and Lyapunov solvers.

    See the [section on open boundary conditions](../methodology/obc.md)
    in the user guide for more information on the

    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["auto", "cache", "force", "force-after-first", "off"] = "auto"
    """The memoization mode to determine when to do fixed-point iterations.

    - "auto": Automatically decides whether to use memoization based on the
        specified tolerances. Only useful if all ranks memoize.
    - "cache": Accept the cached solution outright when every local entry
        is converged after one refinement step; full re-solve otherwise.
        The right mode for iteration-invariant boundary systems (eta = 0,
        fixed leads, no scattering contacts): the OBC input is identical
        every SCBA iteration, so from the second iteration the cache is
        exact and the per-iteration OBC cost drops to one refinement step.
        Input changes invalidate the cache and trigger the full solve.
        Rank-local decision, no collectives. Unlike "auto", a converged cache
        is never discarded and an unconverged one is never returned.
    - "force": Always use memoization.
    - "force-after-first": Use memoization after the first SCBA iteration.
    - "off": Never use memoization.
    """

    num_ref_iterations: PositiveInt = Field(default=2, ge=2)
    """The number of fixed-point iterations to perform.

    This must be greater than or equal to 2. The first iteration is used
    to estimate the residuals, and a second fixed-point iteration is
    performed to get the residuals after the memoization is applied.

    """

    relative_tol: PositiveFloat = 2e-1
    """The relative tolerance on fixed-point residuals for memoization.

    !!! note
        Only used if [`mode`](#mode) is set to `"auto"`.

    """

    absolute_tol: PositiveFloat = 1e-6
    """The absolute tolerance on fixed-point residuals for memoization.

    !!! note
        Only used if [`mode`](#mode) is set to `"auto"`.

    """

    warning_threshold: PositiveFloat = 1e-1
    """The threshold for issuing a memoization warning.

    If the memoized functions residual is above this value after the
    fixed-point iterations, a warning is issued. This is to alert the
    user that the memoization may not be accurate enough and that the
    results may be unreliable.

    """

    agreement_threshold: float = Field(default=0.999, ge=0, le=1)
    """The threshold for agreement between ranks for memoization.

    The default value of 0.999 means 99.9% of the ranks must agree to
    use memoization.

    !!! note
        Only used if [`mode`](#mode) is set to `"auto"`.

    """


class SolverConfig(BaseModel):
    """Options for the system solver."""

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["rgf", "inv"] = "rgf"
    """The algorithm to use for the system solver.

    - `"rgf"`: Uses the recursive Green's function (RGF) algorithm to
      compute the Green's functions. This is the default.

    - `"inv"`: Uses a direct matrix inversion to compute the Green's
      functions. This is mainly useful for debugging and testing, as it
      is not efficient for realistically sized systems.

    """

    max_batch_size: PositiveInt = 100
    """The maximum number of energies that are batched together when
    computing the Green's functions in the system solver.

    This can be used to reduce the memory footprint of the system
    solver, at the cost of increased computation time.

    """

    compute_current: bool | None = None
    """Whether to compute the current via the Meir-Wingreen formula.

    Supported by the `"rgf"` and `"inv"` algorithms. If not set, it
    defaults to `True` for `"rgf"` and `False` for `"inv"`.

    If `True`, `"rgf"` computes the current between each layer and
    from/to the leads. `"inv"` computes only the two lead currents and
    fills the internal interfaces with `NaN`.

    This way of computing the current is usually preferable as it is
    independet of any interaction cutoffs, since it is computed from
    the temporarily densified Green's functions and self-energies.

    !!! note
        This is parameter is only used in the electron solver. The
        Coulomb screening solver does not compute currents, so this
        parameter is ignored for the Coulomb screening solver.

    """

    direct_solver: Literal[
        "superlu",
        "mumps",
        "cudss",
        "pardiso",
        "thomas",
        "auto",
    ] = "auto"
    """The direct solver to use in `wf` simulations.

    If set to `"auto"`, the solver is automatically chosen based on the
    matrix type and the available direct solver libraries.

    In runs with [`low_rank_obc`](qtbm#low_rank_obc) = true, the system
    matrix will be Hermitian or even real and symmetric in gamma-only
    simulations. In those cases, libraries that can exploit the symmetry
    are preferred, i.e., cuDSS on GPU and PARDISO on CPU.

    On GPU, SuperLU is the only fallback option if cuDSS is not
    available. On CPU, if PARDISO is not available, the fallback options
    are MUMPS and then SuperLU.

    The Thomas solver involves a straight-forward tiling of the system
    matrix into blocks without reordering. It is therefore important
    that the Hamiltonian is ordered in a way that results in a
    block-tridiagonal structure.

    """

    @model_validator(mode="after")
    def set_compute_current(self) -> Self:
        """Sets the `compute_current` parameter based on the algorithm."""
        if self.compute_current is None:
            self.compute_current = self.algorithm == "rgf"

        if self.compute_current and self.algorithm not in ("rgf", "inv"):
            raise ValueError(
                "Current computation is only supported for the RGF and Inv "
                "algorithms."
            )

        return self


class OBCConfig(BaseModel):
    r"""Options for open-boundary conditions (OBCs).

    The OBC solvers compute the surface Green's functions of the
    contacts. The retarded surface Green's function satisfies the
    following recursion relation:

    $$
    \mathbf{g}^R = \left[\mathbf{m}_{0} - \mathbf{m}_{-1} \mathbf{g}^R
    \mathbf{m}_{+1} \right]^{-1},
    $$

    where $\mathbf{m}_{0}$ is the contact Hamiltonian,
    $\mathbf{m}_{-1}$ is the coupling from the device to the contact,
    and $\mathbf{m}_{+1}$ is the coupling from the contact to the
    device. In the NEGF framework, the system matrix $\mathbf{m}$
    includes scattering self-energies.

    More information on the boundary conditions can be found in the user
    guide [section on open boundary conditions](../methodology/obc.md).

    """

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["sancho-rubio", "spectral"] = "spectral"
    """The algorithm to use when solving the OBC recursion relation.

    - `"sancho-rubio"`: Uses the Sancho-Rubio iterative
      scheme[^sancho-rubio] to compute the surface Green's functions.
      This method achieves exponential convergence compared to the
      linear convergence of fixed-point iterations.

    - `"spectral"`: Uses the specified `nevp_solver` to compute
      eigenpairs of the polynomial contact eigenvalue problem and uses
      them to construct the surface Green's functions. This is generally
      more efficient method when combined with a contour integral NEVP
      solver (`"beyn"`), but can require more parameter tuning.

    [^sancho-rubio]: M. P. Lopez Sancho, et al., 1985 J. Phys. F: Met.
        Phys. 15 851, https://doi.org/10.1088/0305-4608/15/4/009

    """

    nevp_solver: Literal["beyn", "full"] = "beyn"
    r"""The NEVP solver to use for the spectral OBC algorithm.

    The contact eigenvalue problem is a polynomial eigenvalue problem of
    the form:

    $$
    \sum \limits_{n=-b}^{+b} \lambda^{n} \hat{\mathbf{m}}_{n} \mathbf{v}
    = 0,
    $$

    where $b$ is the number of [`block_sections`](#block_sections), and
    $\hat{\mathbf{m}}_{n}$ are potentially reduced coupling matrices.

    From selected eigenvalues $\lambda = e^{i k}$ and eigenvectors
    $\mathbf{v}$, the surface Green's functions can be constructed.

    - `"beyn"`: Uses the Beyn's contour integral method[^beyn] to solve
      the NEVP and find the eigenpairs within a specified contour in the
      complex plane. Also see the [`r_o`](#r_o), [`r_i`](#r_i),
      [`m_0`](#m_0), and [`num_quad_points`](#num_quad_points)
      parameters for configuration of the contour integral method.

    - `"full"`: Uses a full dense eigensolver to solve for all
      eigenvalues, linearizing the original polynomial problem. This
      results in a doubled problem size which is also not reduced by
      block sectioning or exploiting periodicity.

    !!! note
        Only used if [`algorithm`](#algorithm) is set to `"spectral"`.

    [^beyn]: W.-J. Beyn, An integral method for solving nonlinear
        eigenvalue problems, Linear Algebra and its Applications, 2012,
        https://doi.org/10.1016/j.laa.2011.03.030.

    """

    block_sections: PositiveInt = 1
    """The number of unit cell blocks along transport direction.

    !!! note
        This is automatically determined in QTBM calculations. Thus it
        only has an effect in NEGF calculations.

    In NEGF calculations, one needs to define block-sizes that lead to a
    block-tridiagonal tiling of the system matrix. These *transport
    blocks* are sometimes constructed from multiple unit cells.

    With the [`block_sections`](#block_sections) parameter, one can
    specify how many unit cells are merged into a single transport
    block. This is then used when [`nevp_solver`](#nevp_solver) is set
    to `"beyn"` to reduce the size of the contact NEVP.

    For example, if the transport cell is constructed from two unit
    cells along the transport direction, setting `block_sections = 2`
    will halve the size of the NEVP. The contact transport blocks need
    to be sorted accordingly.

    """

    min_decay: PositiveFloat = 1e-3
    r"""The minimum rate by which a mode must decay to be considered
    evanescent.

    The decay rate is computed as $\|\mathrm{Im}(k)\|$ where $k$ is the
    complex wavevector of the mode.

    This is used to classify the modes obtained from the spectral OBC
    solver into propagating modes and evanescent modes. Modes with decay
    rates below this threshold are considered propagating.

    """

    max_decay: PositiveFloat | None = None
    r"""The maximum rate a mode can decay while still being considered
    relevant for the surface Green's functions.

    The decay rate is computed as $\|\mathrm{Im}(k)\|$ where $k$ is the
    complex wavevector of the mode.

    Very rapidly decaying modes do not contribute to the surface Green's
    functions and can be neglected. These modes should be filtered out
    as including them can lead to numerical instabilities.

    If `max_decay` is not set, it is computed from the [outer contour
    radius](#r_o) as `1.5 * log(r_o)`.

    """

    num_ref_iterations: PositiveInt = 2
    """The number of fixed-point iterations used to refine the surface
    Green's functions.

    This is needed to improve the accuracy of the surface Green's
    functions, especially if not enough eigenpairs are considered.

    !!! note
        Only used if [`algorithm`](#algorithm) is set to `"spectral"`.

    """

    min_propagation: PositiveFloat = 1e-2
    r"""The minimum group velocity propagation/decay ratio for a mode to
    be considered.

    This ratio is determined by dividing the real part of the group
    velocity by the imaginary part of the group velocity:

    $$
    \mathrm{Re}(\frac{dE}{dk}) / \mathrm{Im}(\frac{dE}{dk}).
    $$

    """

    residual_tolerance: PositiveFloat = 1e-3
    r"""The tolerance on the residual of an eigenpair.

    The eigenpair residuals are computed as by inserting the eigenvalues
    and eigenvectors back into the polynomial eigenvalue problem.

    $$
    \text{residual} = \lvert \sum \limits_{n=-b}^{b} \lambda^{b}
    \mathbf{M}_{n} \vec{v} \rvert.
    $$

    Modes exceeding this tolerance are considered spurious and are
    discarded.

    !!! note
        Only used if [`algorithm`](#algorithm) is set to `"spectral"`.

    """

    residual_normalization: bool = True
    """Whether to consider relative residuals instead of absolute
    residuals when filtering eigenpairs.

    This is useful to avoid that large eigenvalues will have larger
    absolute residuals than small eigenvalues.

    """

    warning_threshold: PositiveFloat = 1e-1
    r"""The threshold for issuing a warning about the surface Green's
    functions recursion residual.

    This residual is computed as

    $$
    \lvert \mathbf{g}^R - \left[\mathbf{M}_{0} - \mathbf{M}_{-1}
    \mathbf{g}^R \mathbf{M}_{+1} \right]^{-1} \rvert / \lvert
    \mathbf{g}^R \rvert
    $$

    !!! note
        This parameter is only used if the
        [`formalism`](quatrex#formalism) = `"wf"`. Otherwise, the
        memoizer is responsible for residual checking and issuing
        warnings.

    """

    eta_decay: PositiveFloat = 1e-12
    """Small value to separate very slowly decaying modes from perfectly
    propagating ones.

    Modes that are very close to the unit circle could get misclassified
    via the [`min_decay`](#min_decay) and
    [`min_propagation`](#min_propagation) conditions, i.e., when their
    decay rate is smaller than [`min_decay`](#min_decay) but their
    propagation/decay ratio is not pronounced enough. Modes with decay
    rates smaller than this value are considered as perfectly
    propagating modes, even if the propagation/decay ratio is not above
    the [`min_propagation`](#min_propagation) threshold.

    """

    # Parameters for iterative OBC algorithms.
    max_iterations: PositiveInt = 100
    """The maximum number of iterations for the Sancho-Rubio method.

    A warning is issued if the method does not converge within this
    number of iterations.

    """

    convergence_tol: PositiveFloat = 1e-6
    """The convergence tolerance for the Sancho-Rubio method.

    This is the Frobenius norm of the update matrices `alpha` and `beta`
    in the Sancho-Rubio method. Note that the norm is taken over the
    entire energy batch.

    """

    # Parameters for subspace NEVP solvers.
    r_o: PositiveFloat = Field(default=10.0, gt=1)
    """The outer radius of the contour in the complex plane for the
    contour nevp methods (`"beyn"`).

    This parameter should not be too large to avoid having too many
    eigenpairs inside the contour. It should also not be too small to
    avoid missing important eigenpairs. If an eigenpair is very close to
    the contour, it can lead to numerical instabilities.

    """

    r_i: PositiveFloat = Field(default=0.8, gt=0, lt=1)
    """The inner radius of the contour in the complex plane for the
    contour methods.

    This must be less than one to capture propagating modes, but should
    not be too small to avoid including too many decaying modes.

    """

    m_0: PositiveInt = 10
    """The subspace guess in the contour methods.

    The guess has to be larger than the expected number of eigenvalues
    inside the contour. If too small, the method will fail. If too
    large, the method will be less efficient.

    """

    num_quad_points: PositiveInt = 20
    """The number of quadrature points for the contour integrals."""

    memoizer: MemoizerConfig = MemoizerConfig()
    """Options for memoizing the surface Green's functions."""

    @model_validator(mode="after")
    def set_max_decay(self) -> Self:
        """Sets the max decay if not already set."""
        if self.max_decay is None:
            self.max_decay = 1.5 * np.log(self.r_o)

        return self

    @model_validator(mode="after")
    def scale_contour_radii(self) -> Self:
        """Scales the contour radii based on block_sections."""
        self.r_o **= 1 / self.block_sections
        self.r_i **= 1 / self.block_sections

        return self


class LyapunovConfig(BaseModel):
    r"""Parameters for solving the (discrete-time) Lyapunov equation.

    The discrete-time Lyapunov equation (also called Stein equation)
    arises in the computation of lesser boundary conditions.

    This is a matrix equation of the form

    $$
    \mathbf{A} \mathbf{X} \mathbf{A}^{\dagger} - \mathbf{X} =
    -\mathbf{Q}
    $$

    """

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["spectral", "doubling"] = "spectral"
    r"""The Lyapunov solver algorithm to be used.

    - `"spectral"`: Uses eigenvalue decomposition to solve the Lyapunov
      equation. This method is somewhat expensive since a full
      eigendecomposition is required.

    - `"doubling"`: Uses iterative doubling to solve the Lyapunov
      equation. This method should converge exponentially, but is
      theoretically unstable if $\mathbf{A}$ has eigenvalues outside the
      unit circle. It is therefore generally recommended to use
      `"spectral"` in conjuntion with the memoizer, which will only call
      the actual Lyapunov solver when the residuals are above the
      specified tolerances.

    """

    reduce_sparsity: bool = True
    r"""Whether to exploit the sparsity of $\mathbf{A}$ to accelerate
    the Lyapunov solver.

    This is done by removing zero rows and columns from $\mathbf{A}$,
    solving the reduced Lyapunov equation, and then expanding the
    solution back to the original system's size.

    """

    assume_constant_sparsity: bool = False
    r"""Whether to assume that the sparsity pattern of $\mathbf{A}$
    remains constant between calls to the Lyapunov solver. This is only
    relevant when the Lyapunov solver is called during the SCBA
    iterations. In practice, this should always be the case.

    If set to `True`, the sparsity pattern is only computed once during
    the first SCBA iteration and reused for subsequent iterations.

    !!! warning

        There is currently a bug and this parameter should always be set
        to `False`.

    """

    # Parameters for iterative Lyapunov algorithms.
    max_iterations: PositiveInt = 100
    """The maximum number of iterations for the `"doubling"` algorithm."""

    relative_tol: PositiveFloat = 1e-4
    """The relative convergence tolerance for the `"doubling"` algorithm."""

    absolute_tol: PositiveFloat = 1e-8
    """The absolute tolerance for the `"doubling"` algorithm."""

    # Parameter for spectral Lyapunov solver.
    num_ref_iterations: PositiveInt = Field(default=2, ge=1)
    """The number of fixed-point iterations used to refine the solution
    of the spectral Lyapunov solver.

    """

    memoizer: MemoizerConfig = MemoizerConfig()
    """Options for memoizing the solution of the Lyapunov equation."""


class ContactConfig(BaseModel):
    """Configuration for a contact.

    !!! warning

        Many contact parameters are currently only used in the
        `"wf"` formalism.

    """

    model_config = ConfigDict(extra="forbid")

    name: str
    """A unique name for the contact."""

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The origin of the contact region in Å.

    This is used to automatically determine the orbitals that belong to
    this contact.

    !!! warning

        This parameter is currently only used in the `"wf"` formalism.


    """

    lattice_vectors: list[list[float]] = Field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    """The lattice vectors of the contact cell in Å.

    In `"wf"` simulations this is used to automatically determine the
    orbitals that belong to this contact.

    The volume of the contact cell is also used to determine the Fermi
    level of the contact from its doping density and the density of
    states of its band structure.

    """

    direction: Literal["a", "b", "c"] | None = None
    """The direction from contact to the device.

    This is used to find periodic images of the contact in transport
    direction.

    !!! warning

        This parameter is currently only used in the `"wf"` formalism.

    """

    fermi_level: float | None = None
    """The Fermi level of the contact.

    If not set, the Fermi level is automatically determined from the
    band structure of the contact via the `mid_gap_energy` parameter.

    When set explicitly, this may lead to physically inconsistent
    results, especially in the context of Schrödinger-Poisson
    simulations.

    """

    mid_gap_energy: float | None = None
    """An energy lying somewhere in the band gap of the contact.

    This is used to separate conduction from valence band states, which
    is necessary to automatically determine a contact's Fermi level, and
    to compute the excess carrier density that is used in computing the
    electrostatic potential in the Poisson solver.

    This is also necessary when band-edge tracking is enabled in
    `"negf"` simulations, since the band edges, and their initial
    distance to the Fermi level, are determined via the mid-gap energy.

    """

    num_kpoints_transport: int = 50
    """Number of k-points to use for contact band structure calculation.

    This is used when automatically determining the Fermi level of the
    contact from its band structure. The k-point grid along the
    transverse directions are determined from the `kpoint_grid`
    parameter.

    """

    temperature: NonNegativeFloat = 300.0  # K
    """The temperature of the contact."""

    voltage: float = 0.0
    """The voltage applied to the contact.

    At least one contact needs to be grounded (i.e. have zero voltage)
    to serve as a reference for the other voltages.

    The voltage and the Fermi level of the contact are used to determine
    its chemical potential.

    """

    @model_validator(mode="after")
    def to_array(self) -> Self:
        """Transforms origin and size to arrays."""
        self.origin = np.array(self.origin, dtype=float)
        self.lattice_vectors = np.array(self.lattice_vectors, dtype=float)
        return self


class ElectronConfig(BaseModel):
    """Options for the electronic subsystem solver."""

    model_config = ConfigDict(extra="forbid")

    solver: SolverConfig = SolverConfig()
    """Parameters concerning the system solver."""

    obc: OBCConfig = OBCConfig()
    """Parameters concerning the open boundary conditions."""

    eta_obc: NonNegativeFloat = 0  # eV
    """Small imaginary value to add to the energy when computing the
    OBCs.

    Including this small broadening can help stabilize the convergence
    of the iterative sancho-rubio OBC solver near van Hove
    singularities.

    """

    eta: NonNegativeFloat = 1e-12  # eV
    """Small imaginary value to add to the energy when computing the
    Green's functions.

    """

    left_contact: ContactConfig | None = None
    """Configuration for the left contact.

    This must be provided for any `"negf"` simulation.

    !!! note

        In `"wf"` simulations, the left and right contacts are not used.

    """

    right_contact: ContactConfig | None = None
    """Configuration for the right contact.

    This must be provided for any `"negf"` simulation.

    !!! note

        In `"wf"` simulations, the left and right contacts are not used.

    """

    band_edge_tracking: bool = False
    """Whether to track the band edges during the SCBA iterations.

    This is setting is only useful if the considered interactions result
    in energy renormalization in the electronic subsystem, which is
    primarily the screened Coulomb interaction.

    If set to `True`, the band edges are tracked during the SCBA
    iterations by computing the eigenvalues of the Hamiltonian
    renormalized with the current self-energy.

    The Fermi levels are then set to be a fixed distance from the band
    edges, which is determined by the initial Fermi level and band
    edges. For example, if the initial Fermi level is 0.5 eV above the
    conduction band edge, the Fermi level is always set to be 0.5 eV
    above the conduction band edge during all SCBA iterations.

    """

    energy_window_min: float | None = None
    """The minimum energy of the energy grid used for electronic
    quantities."""

    energy_window_max: float | None = None
    """The maximum energy of the energy grid used for electronic
    quantities."""

    energy_window_num: PositiveInt | None = None
    """The number of energy points in the energy grid used for electronic
    quantities.

    Either `energy_window_num` or
    [`energy_window_num_per_rank`](#energy_window_num_per_rank) can be
    set to determine the total number of energy points.

    """
    energy_window_num_per_rank: PositiveInt | None = None
    """The number of energy points per rank in the energy grid used for
    electronic quantities.

    Either [`energy_window_num`](#energy_window_num) or
    `energy_window_num_per_rank`) can be set to determine the total
    number of energy points.

    """

    flatband: bool | None = None
    """Whether the system is in flatband conditions.

    If not set, it is automatically determined from the left and
    right Fermi levels. If the Fermi levels are equal, it is assumed
    to be in flatband conditions.

    """

    dos_peak_limit: PositiveFloat = 100.0
    """The maximum derivative of the density of states (DOS) with
    respect to energy.

    At energy points where the DOS derivative exceeds this value, the
    electronic quantities are set to zero to stabilize the convergence
    of the SCBA iterations.

    This is especially a problem during the first few SCBA iterations
    when the self-energies are not yet fully developed and can lead to
    very sharp features in the DOS.

    """

    filtering_iteration_limit: PositiveInt = 1
    """The maximum number of SCBA iterations during which the DOS peak
    filtering is applied.

    This is because the DOS peak filtering is mainly needed during the
    first few SCBA iterations when the self-energies are not yet fully
    developed and can lead to very sharp features in the DOS.

    """

    max_batch_size: PositiveInt | None = None
    """The maximum number of energies to batch together in the solution
    of the electronic subsystem.

    This controls how many energies are treated together when computing
    boundary conditions and electron Green's functions. If not set, all
    energies are computed at once.

    This can help mitigate memory bottlenecks.

    """

    @model_validator(mode="after")
    def check_mid_gap_energy_band_edge_tracking(self) -> Self:
        """Checks that the mid-gap-energy is set if band edge tracking is enabled."""
        if self.band_edge_tracking:
            if (
                self.left_contact is not None
                and self.left_contact.mid_gap_energy is None
            ):
                raise ValueError(
                    "When band edge tracking is enabled, the `mid_gap_energy` of the left contact must be set."
                )
            if (
                self.right_contact is not None
                and self.right_contact.mid_gap_energy is None
            ):
                raise ValueError(
                    "When band edge tracking is enabled, the `mid_gap_energy` of the right contact must be set."
                )

        return self

    @model_validator(mode="after")
    def verify_energies(self) -> Self:
        """Verifies the energy window settings."""

        if (
            self.energy_window_min is not None
            or self.energy_window_max is not None
            or self.energy_window_num is not None
            or self.energy_window_num_per_rank is not None
        ):

            if (self.energy_window_min is None) and (self.energy_window_max is None):
                raise ValueError(
                    "When the energy grid is not read from file, should set both `energy_window_min` and `energy_window_max`."
                )

            if (
                self.energy_window_num is not None
                and self.energy_window_num_per_rank is not None
            ):
                raise ValueError(
                    "Should **exclusively** set electron `energy_window_num` or `energy_window_num_per_rank` in the config."
                )

        return self


class CoulombScreeningConfig(BaseModel):
    """Options for the Coulomb screening solver."""

    model_config = ConfigDict(extra="forbid")

    interaction_cutoff: PositiveFloat = 10.0  # Angstrom
    """The cutoff distance for the screened Coulomb interaction
    self-energy.

    Self-energy matrix elements corresponding to pairs of orbitals that
    are further apart than this distance are not computed. A higher
    cutoff can lead to more accurate results, but also increases the
    computation time. The optimal value depends on the system and the
    desired accuracy.

    """

    solver: SolverConfig = SolverConfig()
    """Parameters concernig the system solver."""

    obc: OBCConfig = OBCConfig()
    """Parameters concerning the open boundary conditions."""

    lyapunov: LyapunovConfig = LyapunovConfig()
    """Parameters concerning the Lyapunov solver."""

    epsilon_r: PositiveFloat = 1.0
    """The relative permittivity of the system.

    The Coulomb matrix is scaled by this value. It is primarily useful
    as a way to scale the strength of the Coulomb interaction and to
    better fit the model to experimental results.

    """

    # How many blocks should be merged into a single block.
    num_connected_blocks: Literal["auto"] | PositiveInt = "auto"
    r"""The number of connected blocks to merge into a single block.

    The computation of the effective lesser/greater polarization
    involves a "sandwich" multiplication (congruence transform) of the
    form

    $$
    \mathbf{L}^{\lessgtr} = \mathbf{V} \mathbf{P}^{\lessgtr} \mathbf{V},
    $$

    where $\mathbf{V}$ is the Coulomb matrix and $\mathbf{P}^{\lessgtr}$
    is the lesser/greater polarization. Since all of these matrices are
    banded, the resulting effective polarization $\mathbf{L}^{\lessgtr}$
    can have a much larger bandwidth.

    The block-tridiagonal tiling of the system matrix used in the OBC is
    therefore larger than the transport blocks used in the electron
    solver. The `num_connected_blocks` parameter determines how many of
    the original transport blocks are merged into a single block for the
    Coulomb screening solver. If set to `"auto"`, the number of
    connected blocks is automatically determined based on the
    `interaction_cutoff` and the geometry of the system.

    """

    dos_peak_limit: PositiveFloat = 100.0
    """The maximum derivative of the density of states (DOS) with
    respect to energy.

    At energy points where the DOS derivative exceeds this value, the
    Coulomb screening quantities are set to zero to stabilize the
    convergence of the SCBA iterations.

    """

    filtering_iteration_limit: PositiveInt = 1
    """The maximum number of SCBA iterations during which the DOS peak
    filtering is applied.

    This is because the DOS peak filtering is mainly needed during the
    first few SCBA iterations when the self-energies are not yet fully
    developed and can lead to very sharp features in the DOS.

    """

    align_polarization_to_complex_axes: bool = True
    r"""Whether to discard certain parts of the polarization.

    This affects the polarization in the following way:

    - The real parts of the lesser/greater polarization are discarded.
    - The imaginary part of the retarded polarization from anyprevious
      computation is zeroed.

    This happens before the anti-Hermitian part of the retarded
    polarization is computed from the lesser and greater parts as

    $$
    \mathbf{P}^R_{AH} = \frac{1}{2i} ( \mathbf{P}^> - \mathbf{P}^< )
    $$

    """

    include_energy_renormalization: Literal["self-energy", "polarization", "both"] = (
        "self-energy"
    )
    r"""Whether to compute the Hermitian part of the retarded
    polarization and/or self-energy.

    Possible values are `"self-energy"`, `"polarization"`, and `"both"`.

    The full retarded interaction quantities are general complex-valued
    matrices, where the Hermitian part is computed from the
    skew-Hermitian part using the Kramers-Kronig relations:

    $$
    \mathbf{X}^{R} = \frac{1}{2} (\mathbf{X}^{>} - \mathbf{X}^{<}) +
    \frac{1}{2\pi} \mathrm{p.v.} \int_{-\infty}^{\infty}  dE' \,
    \frac{\mathbf{X}^{>} - \mathbf{X}^{<}}{E^{'} - E}
    $$

    The Hermitian part only leads to only a shift in the energy, so it
    is often neglected:

    $$
    \mathbf{X}^{R} \approx \frac{1}{2} (\mathbf{X}^{>} - \mathbf{X}^{<})
    $$

    The default is to only include the skew-Hermitian part in the
    Coulomb screening self-energy and not in the polarization.

    The Hermitian part is computed using a Hilbert transform. For the
    Coulomb screening self-energy, this Hilbert transform can lead to
    errors at the edges of the energy window. The
    [`apply_hilbert_correction`](#apply_hilbert_correction) option can
    be used to apply a correction to the Hilbert transform to mitigate
    these errors.

    """

    apply_hilbert_correction: bool = False
    """Whether to apply the corrections for the edges of the energy
    window to the Hilbert transform when computing the retarded
    self-energy.

    Computing the correction is slightly more expensive.

    """

    @model_validator(mode="after")
    def check_hilbert_correction_applicable(self) -> Self:
        """Checks if the Hilbert correction can be applied."""
        if (
            self.apply_hilbert_correction
            and self.include_energy_renormalization not in ["self-energy", "both"]
        ):
            raise ValueError(
                "Hilbert correction can only be applied if the real part of the self-energy is included."
            )

        return self

    max_batch_size: PositiveInt | None = None
    """The maximum number of energies to batch together in the solution
    of the screened Coulomb interaction.

    This controls how many energies are treated together when computing
    boundary conditions and screened Coulomb interactions. If not set,
    all energies are computed at once.

    This can help mitigate memory bottlenecks.

    """


class PhotonConfig(BaseModel):
    """Parameters for photons and electron-photon interactions.

    !!! warning
        The photon solver is not implemented yet. The parameters in this
        section are not used and may be subject to change in the future.

    """

    model_config = ConfigDict(extra="forbid")

    interaction_cutoff: PositiveFloat = 10.0  # Angstrom

    solver: SolverConfig = SolverConfig()
    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()


class PhononConfig(BaseModel):
    """Parameters for phonons and electron-phonon interactions."""

    model_config = ConfigDict(extra="forbid")

    interaction_cutoff: PositiveFloat = 10.0  # Angstrom
    """The cutoff distance for the electron-phonon interaction
    self-energy.

    !!! note
        Currently, only the `"pseudo-scattering"` model / deformation
        potential interaction is implemented, which does not produce
        any self-energy matrix elements besides the diagonal ones.

    """

    interaction_cutoff_taper: Literal["none", "triangular"] = "none"
    """Shape of the spatial interaction cutoff applied to the phonon SSE."""

    solver: SolverConfig = SolverConfig()
    """Parameters concerning the system solver."""

    obc: OBCConfig = OBCConfig()
    """Parameters concerning the open boundary conditions."""

    lyapunov: LyapunovConfig = LyapunovConfig()
    """Parameters concerning the Lyapunov solver."""

    model: Literal["pseudo-scattering", "negf"] = "pseudo-scattering"
    r"""Which model to use for the electron-phonon interaction.

    Currently, only a monochromatic `"pseudo-scattering"` model is
    implemented.

    In this model, the electron-phonon interaction is modeled as

    $$
    \Sigma^{\lessgtr}(E) = D^2 \left[ (N_{ph} + 1) G^{\lessgtr}(E - \hbar
    \omega) + N_{ph} G^{\lessgtr}(E + \hbar \omega) \right],
    $$

    where $D$ is the [`deformation_potential`](#deformation_potential),
    $\hbar \omega$ is the [`phonon_energy`](#phonon_energy), and
    $N_{ph}$ is the phonon occupation number given by the Bose-Einstein
    distribution at the specified [`temperature`](#temperature).

    """

    phonon_energy: NonNegativeFloat | None = None
    """The energy of the phonon mode in eV."""

    deformation_potential: NonNegativeFloat | None = None
    """The deformation potential of the phonon mode in eV."""

    temperature: PositiveFloat = 300.0  # K
    """The temperature of the system in Kelvin."""

    left_temperature: PositiveFloat | None = None
    """Temperature of the left phonon reservoir. Defaults to `temperature`."""
    right_temperature: PositiveFloat | None = None
    """Temperature of the right phonon reservoir. Defaults to `temperature`."""

    # --- 3-phonon (anharmonic) scattering ----------------------------
    fc3_path: Path | None = None
    """Path to the FC3 source consumed by ``SigmaPhononPhonon``."""

    qfold_path: Path | None = None
    """Path to the q-folded device vertices for transversely-periodic
    (``kpoint_grid`` with k>1) anharmonic transport."""

    decomposed_vertices_path: Path | None = None
    """Path to the TENSOR-DECOMPOSED coupled-q device vertex factors."""

    sse_vertex_rank: int = 0
    """Truncate the decomposed vertex to the leading ``rank`` components."""

    decomposed_kernel: Literal["gram", "reconstruct"] = "gram"
    """How the SSE consumes the decomposed vertex."""

    retarded_method: Literal["half", "fft"] = "fft"
    """How to reconstruct ``Sigma^R`` from ``Sigma^{<,>}``."""

    phonon_phonon_truncation_warn: NonNegativeFloat = 0.01
    """Frobenius-norm threshold for the FC3 nearest-neighbour-truncation
    warning (cf. ``fc3_loader.fc3_to_phi_blocks``)."""

    bubble_balance_check: bool = False
    """Per-iteration Phi-derivable energy-balance diagnostic of the 3-phonon
    bubble: P_in = sum hbar*w*Tr[Sigma^< G^>] must equal P_out (conserving
    identity). Off by default: it keeps the Green's-function data alive through
    the nnz->stack back-transpose, which costs one extra all-to-all per
    iteration at stack > 1."""

    sse_low_freq_mask_thz: NonNegativeFloat = 0.0
    """Zero the bubble legs and outputs on all |omega| < this (THz). The
    frequency grid stays anchored at zero (the FFT convolution and the
    bosonic fold require it), so this is the working equivalent of
    starting the window above the acoustic region: masked bins keep their
    ballistic Dyson/transport content but contribute no three-phonon
    scattering. 0 = legacy (only the omega = 0 bin is masked)."""

    pole_sector: PoleSectorConfig = PoleSectorConfig()
    """Pole-subtracted SCBA (see :class:`PoleSectorConfig`). Disabled by
    default = legacy (bit-identical)."""

    frequency_grid: Literal["window", "file"] = "window"
    """Source of the phonon frequency grid. ``"window"`` (legacy): the
    uniform ``linspace`` from the electron ``energy_window_*`` fields.
    ``"file"``: the grid is read verbatim from
    ``<input_dir>/phonon_energies.npy`` -- it may be NON-UNIFORM
    (ascending, non-negative). A non-uniform grid requires the auxiliary
    bubble grid (``sse_aux_grid_dw_thz > 0``): the FFT convolution and
    the bosonic fold only exist on a uniform, zero-anchored grid."""

    sse_aux_grid_dw_thz: NonNegativeFloat = 0.0
    """Spacing (THz) of the AUXILIARY uniform, zero-anchored grid on which
    the 3-phonon bubble convolution, the bosonic fold and the
    Kramers-Kronig Hilbert transform are evaluated. When > 0 the
    Green's-function legs are linearly interpolated from the primary
    (possibly non-uniform) frequency grid onto the auxiliary grid, the
    FFT pipeline runs there unchanged, and Sigma^{<,>,R} is sampled back
    onto the primary grid. Linear interpolation preserves the sign of
    -i G^{<,>} >= 0 (a convex combination), so the interpolated bubble
    stays dissipative. 0 = legacy (bubble on the primary grid, which
    must then be uniform and start at 0)."""

    sse_aux_grid_fmax_thz: NonNegativeFloat = 0.0
    """Upper edge (THz) of the auxiliary bubble grid (only used when
    ``sse_aux_grid_dw_thz > 0``). The 3-phonon bubble has support up to
    2*omega_max and the Kramers-Kronig integral is support-complete only
    on a grid reaching it, so set this >= 2*omega_max; the PRIMARY grid
    (Dyson solves) can then stop just above omega_max -- the
    [omega_max, 2*omega_max] half consumes no Dyson solves. 0 = span the
    primary grid (no extension beyond its top)."""

    sse_aux_restrict: Literal["adjoint", "sample"] = "adjoint"
    """How Sigma returns from the auxiliary bubble grid to the primary grid.
    ``"adjoint"`` is the adjoint of the leg interpolation with respect to the
    energy measure and keeps the Phi-derivable energy balance to roundoff;
    ``"sample"`` is pointwise, sharper at resonance peaks and not conserving.
    The two coincide when the grids do, up to the masked omega = 0 bin."""

    low_freq_mixing_thz: NonNegativeFloat = 0.0
    """Frequency-dependent SCBA mixing: self-energy bins with |omega| < this
    (THz) are mixed with ``low_freq_mixing_factor`` instead of the global
    ``scba.mixing_factor``. 0 = off (uniform mixing). This DAMPS the IR
    (Bose-divergent) marginal mode at the lowest frequency bins WITHOUT
    removing the low-omega anharmonic scattering: the mode sits near the
    unit circle (|lambda|~1) and a small mixing factor pulls it inside
    (|1+a(lambda-1)|<1)."""
    low_freq_mixing_factor: NonNegativeFloat = 0.02
    """Gentle SCBA mixing factor applied to the |omega| < ``low_freq_mixing_thz``
    bins (see there). Small (~0.01-0.03) to damp the IR marginal mode; the rest
    of the spectrum keeps ``scba.mixing_factor``."""

    sse_release_leg_blocks: bool = False
    """Free the densified G leg blocks once the batched coupled-q kernel has
    stacked them."""

    sse_perm_cache_share: Literal["off", "auto"] = "off"
    """Share pre-permuted vertex pairs across the block row in the dense
    q-folded ring."""

    sse_tau_chunk_bytes: PositiveInt = 256 * 1024 * 1024
    """Memory cap on one tau chunk of the decomposed (``"gram"``) SSE kernel."""

    sse_ring_threads: NonNegativeInt = 0
    """Width of the omega/tau ring-contraction thread pool (bit-identical
    results for any width). 0 = keep the QUATREX_PHPH_RING_THREADS env
    default (1)."""

    sse_ring_min_w: PositiveInt | None = None
    """Minimum omega/tau batch per pool split; None = keep the
    QUATREX_PHPH_RING_MIN_W env default (48). NOTE: this governs only the
    one-shot :func:`quatrex.phonon.bubble.ring_contract` wrapper (dense
    reference / audits); the production SSE splits its tau batch itself,
    see ``sse_tau_min_chunk``."""

    sse_tau_min_chunk: PositiveInt = 4
    """Minimum tau points per ring-pool task in the production SSE stage-3
    split (the pool width is capped at ``n_tau // sse_tau_min_chunk``).
    The default 4 reproduces the legacy split; larger chunks (32-48) give
    each task fatter GEMMs and less dispatch/allocation churn. Results are
    bit-identical for any value."""

    sse_pool_scope: Literal["tau", "pair_tau"] = "tau"
    """Task decomposition of the SSE stage-3 thread pool. ``"tau"``
    (legacy): tasks are tau chunks, each sweeping all owned (I, J) pairs.
    ``"pair_tau"``: tasks are (pair, tau-chunk) tiles, so fat tau chunks
    (``sse_tau_min_chunk`` large) still fill the pool. Bit-identical."""

    sse_ring_workspaces: bool = False
    """Reuse thread-local T/U GEMM workspaces inside the ring contraction
    (``out=`` matmuls) instead of allocating fresh temporaries per call.
    Bit-identical; avoids allocator churn/contention at wide pools."""

    sse_greater_from_lesser: bool = False
    """Reconstruct the cross terms of Sigma^> from the Sigma^< ring pass by the
    exact bosonic tau-domain identity: 4 instead of 6 ring contractions per
    vertex quad. Exact by construction up to summation order. Supported on the
    dense kernels only -- Gamma, and coupled-q fed explicit q-folded vertices,
    whose reality the identity relies on -- and refused with the decomposed
    vertex. Verify with ``sse_fold_verify_iterations``."""

    sse_fold_verify_iterations: NonNegativeInt = 0
    """With ``sse_greater_from_lesser``: for the first N compute() calls run
    the LEGACY 6-ring path, additionally accumulate the cross terms, and
    report the max-abs/rel mismatch of the reconstruction identity per output
    pair (rank 0; single-rank runs only -- multi-rank runs skip the gate with
    a warning). At nq > 1 the checked identity includes the external q -> -q
    map. The legacy result is shipped during these iterations (with
    ``sse_hermitian_pairs`` the halving is also suspended during them)."""

    sse_hermitian_pairs: bool = False
    """Contract only the upper-triangle output pairs (I <= J) and fill the
    (J, I) blocks via the anti-Hermiticity Sigma_JI(w) = -Sigma_IJ(w)^dagger
    (conjugate-transpose in the stack state + tau reversal in the nnz state).
    Exact only to the extent the input G is anti-Hermitian (solver precision,
    NOT construction-exact) -- verify with a run-level A/B and watch the
    bubble-balance residual. Single-rank runs only. Diagonal blocks are
    always contracted fully (no projection is applied)."""

    obc_scattering_contacts: bool = False
    """Dress the lead open-boundary problem with the device's boundary
    scattering self-energy each iteration: the OBC is computed AFTER
    Sigma^R is folded into the system matrix, so the periodic lead
    superblocks inherit the boundary slab's Sigma^R under the
    bulk-periodicity assumption, and the contact injection becomes the
    fluctuation-dissipation pair of the DRESSED escape rate at the lead
    temperatures. This is the ordering the ELECTRON (GW) solver has
    always used (assemble first, OBC second); the phonon solver
    deliberately reversed it to keep ideal ballistic reservoirs. The
    reservoirs then carry the same anharmonic dissipation as the device
    (no artificial contact broadening needed). Default off: bare
    harmonic reservoirs, scattering enters the device Dyson only."""

    sse_g_band: int = Field(default=3, ge=1, le=3)
    """Inner Green's-function block band ``|K - K'|`` kept in the bubble
    contraction. Band 1 masks the kernel to the RGF block-tridiagonal G, and a
    masked positive semi-definite form is not positive semi-definite, so
    interior slabs acquire non-causal gain. Band 2 makes the diagonal Sigma
    blocks exact and causal, band 3 the first off-diagonals as well. Extends
    the shared G/Sigma sparsity pattern by the corresponding off-diagonal
    blocks, and is clamped at use to ``n_blocks - 1``. Distributed use
    needs every block rank to own at least ``sse_g_band + 1`` blocks."""

    sse_microblock_dof: NonNegativeInt = 0
    """Primitive FC3 block size inside a grouped Dyson block.  Zero keeps the
    established one-FC3-block-per-Dyson-block path.  A positive value enables
    the exact primitive-microblock contraction; every Dyson block size must be
    divisible by it.  Bulk Si uses six displacement DOFs per primitive cell."""

    sse_microblock_g_band: NonNegativeInt = 0
    """Green-function range in primitive microblocks for the microblock SSE.
    It is independent of ``sse_g_band``, which is measured in grouped Dyson
    blocks.  The generated primitive self-energy range is determined from the
    actual FC3 support and must fit inside the same/adjacent grouped blocks;
    otherwise construction fails rather than dropping an output shell."""

    sse_g_band_taper: Literal["none", "bartlett"] = "none"
    """Positive semi-definite taper of the inner-G band mask, against the
    boxcar's non-causal gain. ``"bartlett"`` weights each inner G link by
    ``w_d = 1 - d/(sse_g_band+1)`` and the Sigma output blocks by the same
    weight, which is the Phi-derivable pair and so conserves energy. It
    restores end-to-end positivity only at ``sse_g_band = 1``: the output
    band is pinned at ``|I-J| <= 1``, whose tapered Toeplitz symbol needs
    ``w_1 <= 1/2``. At wider bands it fixes the legs and not Sigma, and the
    code warns. Price at band 1 is half the off-diagonal coherence.
    ``"none"`` is the legacy boxcar. Not supported with
    ``decomposed_kernel="gram"``."""

    sse_dense_q_batched: bool = True
    """Coupled-q dense ring: flatten the (q', quad) task axis into
    strided-batched GEMMs (per (I, J), grouped by ring shape) instead of
    one Python task per (q-pair, quad). Same math; the scatter-add
    reduction order differs from the task loop at rounding level
    (~1e-12). The task loop measured ~200 us fixed cost per ring call --
    the film ring ran at 4-10% of peak, ~85% non-GEMM time. False
    restores the legacy per-task loop bit-exactly."""

    sse_ring_dtype: Literal["complex128", "complex64"] = "complex128"
    """Precision of the ring-contraction GEMMs (the SSE hot path). With
    "complex64" the vertex factors and inner-G band links are cast to
    single precision, so every per-quad ring runs as batched CGEMM,
    while the accumulation ACROSS quads (and everything else: FFTs,
    fold, Dyson, OBC) stays complex128. Gamma-only dense path only (the
    coupled-q and factored kernels ignore it). Cheaper in time and leg
    memory; the price is a ~1e-6-relative Sigma and a degraded bubble
    energy-balance residual -- measure before trusting a production
    number. "complex128" is the legacy path, bit-identical."""

    heat_flow_conservation_tol: PositiveFloat = 1e-2
    """Convergence tolerance for the anharmonic phonon SCBA: the relative
    lead balance of the (hbar-omega-weighted) Meir-Wingreen HEAT current.
    3-phonon processes do NOT conserve phonon NUMBER (1<->2 splitting/
    merging), so only the ENERGY current is conserved and SCBA convergence
    means the heat flow matches across the leads. The reported current is
    the converged fixed point; a run that fails this gate is reported as
    non-converged with its last iterate (no most-conserved iterate is
    cherry-picked -- over a non-converged trajectory it is not a fixed
    point, and it is typically an early near-ballistic step)."""

    sigma_convergence_tol: PositiveFloat = 1e-3
    """Relative self-energy residual tolerance
    (``||Sigma_new - Sigma_old||_inf / ||Sigma||_inf``) for the anharmonic
    phonon SCBA fixed point, applied IN ADDITION to
    ``heat_flow_conservation_tol``. Heat-flow conservation alone is
    necessary but not sufficient: at large broadening ``eta`` the heat flow
    conserves before Sigma reaches self-consistency, so a heat-flow-only
    stop accepts an under-scattered, non-converged Sigma. If Sigma
    oscillates (a limit cycle) the run is reported as non-converged."""

    bubble_balance_tol: NonNegativeFloat = 0.0
    """Optional third convergence gate on the Phi-derivable bubble energy
    balance ``|P_in - P_out| / |P_in|`` (requires ``bubble_balance_check``).
    0 disables (residual + lead heat balance only)."""

    scp_tadpole: bool = False
    """Optional self-consistent-phonon (SCP) cubic tadpole static self-energy
    (``model == "negf"``). The cubic tadpole ``Sigma_T = Phi3 : <u>`` is a
    STATIC real self-energy that stiffens the soft modes (the finite-T SCP
    renormalisation), stabilising the dynamic bubble; it is recomputed
    every SCBA iteration from the current device ``G^<`` and added to the
    dynamical matrix via ``Sigma^R``. Needs only FC3 (the quartic loop,
    which would need FC4, is omitted). Default OFF; cf.
    ``quatrex/phonon/static_self_energy.py``."""

    scp_tadpole_term: bool = True
    """Include the cubic tadpole Phi3:<u> in the static SCP
    self-energy. False for centrosymmetric crystals (<u> = 0 by
    inversion; the numerical tadpole is noise amplified through
    Phi_eff^+ and destabilises the map) -- keeps the quartic loop."""

    scp_uu_source: Literal["g", "dressed"] = "g"
    """Equal-time <uu> source for the static SCP terms: "g" =
    the NEGF G^< quadrature (legacy; ill-conditioned at eta=0 on
    IR-resolved grids), "dressed" = SCP closed form on the
    dressed harmonic model Phi_eff at the mean lead temperature
    (exact in equilibrium, robust)."""

    scp_uu_min_thz: NonNegativeFloat = 0.0
    """<uu> quadrature floor (THz): exclude bins below this from
    the equal-time <uu> integral (eta=0 IR tails below the lowest
    device mode can dominate and flip Tr<uu> negative). 0 = legacy
    full integral. Set just below the lowest on-mesh mode."""

    scp_loop: bool = False
    """Quartic (SCP) loop static self-energy Sigma_L = 1/2 Phi4 : <uu>
    on top of the cubic tadpole (requires ``scp_tadpole``): the SCP
    stiffening counterterm that grows with <uu> -- restores the
    negative feedback the cubic-only bubble lacks on soft-mode
    structures. Device FC4 blocks from ``scp_fc4_path`` (default:
    fc4_blocks.hdf5 next to the FC3 file; produced by the fc4 reap)."""

    scp_fc4_path: str | None = None

    scp_static_mixing: PositiveFloat = 0.1
    """Linear mixing factor for the self-consistent static (tadpole)
    self-energy across SCBA iterations. Gentler than the dynamic-Sigma
    mixing (the static SE drives the soft-mode stiffening and overshoots
    if mixed too fast). Only used when ``scp_tadpole``."""

    scp_floor_thz: NonNegativeFloat = 0.5  # THz
    """Absolute frequency floor (THz) for the regularised ``Phi_eff``
    pseudo-inverse in the tadpole ``mean_displacement`` solve: modes below
    this are dropped from the inverse (only the stiff optical modes relax),
    preventing the soft mode's ~1/omega^2 amplification from blowing the
    solve up. Only used when ``scp_tadpole``."""

    @model_validator(mode="after")
    def set_lead_temperatures(self) -> Self:
        """Falls the lead temperatures back to the device `temperature`."""
        if self.left_temperature is None:
            self.left_temperature = self.temperature
        if self.right_temperature is None:
            self.right_temperature = self.temperature

        return self

    @model_validator(mode="after")
    def check_phonon_energy_or_deformation_potential(self):
        """Validate model-specific required parameters."""
        if self.model == "pseudo-scattering" and (
            self.phonon_energy is None or self.deformation_potential is None
        ):
            raise ValueError(
                "'phonon_energy' and 'deformation_potential' must be set "
                "for model='pseudo-scattering'."
            )
        if self.model == "negf" and self.fc3_path is None:
            raise ValueError(
                "'fc3_path' must be set for model='negf'."
            )

        return self

    @model_validator(mode="after")
    def check_pole_sector_preconditions(self) -> Self:
        """Hard gates for the pole sector against the rest of the phonon setup.

        These are refusals, not warnings. Each one is a configuration under
        which the sector would produce a confidently wrong answer rather than a
        noisy one.
        """
        ps = self.pole_sector
        if not ps.enabled:
            return self

        if self.retarded_method != "fft":
            raise ValueError(
                "pole_sector requires retarded_method='fft'. With 'half' the "
                "Kramers-Kronig real part of Sigma^R is never built, so the "
                "operator whose poles are being sought is not causal and its "
                "roots are not resonances."
            )
        if ps.omega_min_thz and self.sse_low_freq_mask_thz:
            if ps.omega_min_thz <= self.sse_low_freq_mask_thz:
                raise ValueError(
                    f"pole_sector: omega_min_thz ({ps.omega_min_thz}) must sit "
                    f"ABOVE sse_low_freq_mask_thz "
                    f"({self.sse_low_freq_mask_thz}). Sigma is identically zero "
                    "below the mask, so the continuation has no cut there while "
                    "the device Green's function still does."
                )
        if ps.sheet == "outgoing" and self.obc.algorithm != "spectral":
            raise ValueError(
                f"pole_sector: sheet='outgoing' requires obc.algorithm="
                f"'spectral' (got {self.obc.algorithm!r}); the outgoing-sheet "
                "continuation tracks the lead Bloch roots, which only the "
                "spectral solver exposes."
            )
        if getattr(ps, "leg", "congruence") == "congruence":
            # sectors is inert on this route: the correction goes into the
            # ring's leg, not into analytic terms beside it, so there is no
            # sector to switch off. Saying so beats letting the setting read
            # as if it still selected something.
            if ps.sectors != "rr_ss_sr":
                warnings.warn(
                    f"pole_sector.sectors={ps.sectors!r} is ignored when "
                    "leg='congruence': the pole enters as a cell-average "
                    "correction to the ring's leg and no analytic sector is "
                    "added beside it.",
                    stacklevel=2,
                )
        elif ps.sectors != "rr_ss_sr":
            warnings.warn(
                f"pole_sector.sectors={ps.sectors!r} DROPS physical "
                "three-phonon processes and is a staging setting, not a "
                "production one. The complete method is 'rr_ss_sr'.",
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def check_vertex_source_exclusivity(self):
        """Dense q-folded dict and tensor-decomposed factors are two
        representations of the SAME coupled-q vertex -- exactly one may be
        configured."""
        if self.qfold_path is not None and self.decomposed_vertices_path is not None:
            raise ValueError(
                "qfold_path and decomposed_vertices_path are mutually "
                "exclusive coupled-q vertex sources; configure one."
            )
        if self.sse_vertex_rank and self.decomposed_vertices_path is None:
            raise ValueError(
                "sse_vertex_rank truncates the decomposed vertex; it "
                "requires decomposed_vertices_path."
            )
        if self.sse_vertex_rank < 0:
            raise ValueError("sse_vertex_rank must be >= 0.")
        micro_dof = int(self.sse_microblock_dof)
        micro_band = int(self.sse_microblock_g_band)
        if bool(micro_dof) != bool(micro_band):
            raise ValueError(
                "sse_microblock_dof and sse_microblock_g_band must either "
                "both be zero (disabled) or both be positive."
            )
        return self


class OutputConfig(BaseModel):
    """Options for the output of `quatrex` calculations.

    !!! warning
        The output options are not yet fully implemented and may be
        subject to change in the future. They are currently not used in
        QTBM calculations.

    """

    model_config = ConfigDict(extra="forbid")

    # Only the spectral currents are saved by default.
    device_currents: bool = True
    """Whether to save the device currents.

    This will output both the spectral device current between transport
    cells computed from the lesser Green's function and, if configured,
    the Meir-Wingreen device current.

    """

    potential: bool = False
    """Whether to save the potential.

    !!! warning
        This option is unused.

    """

    electron_ldos: bool = False
    """Whether to save the spectral electron local density of states
    (LDOS).

    This will output an energy and orbital resolved LDOS computed from
    the retarded Green's function.

    """

    electron_density: bool = False
    """Whether to save the electron density.

    This will output the energy-resolved electron density computed from
    the lesser Green's function.

    """
    hole_density: bool = False
    """Whether to save the hole density.

    This will output the energy-resolved hole density computed from the
    greater Green's function.

    """

    polarization_density: bool = False
    """Whether to save the polarization density.

    This will output the energy-resolved polarization densities computed
    from the lesser and greater polarizations.

    !!! note
        This is primarily a debugging option.

    """

    coulomb_screening_density: bool = False
    """Whether to save the Coulomb screening density.

    This will output the energy-resolved Coulomb screening densities
    computed from the lesser and greater screened Coulomb interactions.

    !!! note
        This is primarily a debugging option.

    """

    self_energy_density: bool = False
    """Whether to save the self-energy density.

    This will output the energy-resolved self-energy densities computed
    from the lesser, greater, and retarded self-energies.

    """

    profiling_path: Path | None = None
    """The file to save the timing results to.

    The timing results are saved in the format specified by
    [`profiling_save_format`](#profiling_save_format).

    If [`save_profiling_results`](#save_profiling_results) is `True`,
    and the `profiling_path` is not set, the file name is inferred from
    the SLURM output file if running in a SLURM context. Otherwise, the
    default name `quatrex_times.out` is used.

    """

    save_profiling_results: bool = False
    """Whether to save the timing results to a file."""

    profiling_save_format: Literal["pickle", "json"] = "json"
    """The format to save the timing results in.

    The timing results are saved in either `pickle` or `json` format.
    The default is `json`. `pickle`-serialized files will contain a
    dictionary with the timing results.

    """

    @model_validator(mode="after")
    def set_profiling_parameters(self) -> Self:
        if self.profiling_path is None:
            self.profiling_path = Path("quatrex_times.out")
            if "SLURM_JOB_ID" in os.environ:
                try:
                    jid = os.environ.get("SLURM_JOB_ID")
                    if not jid:
                        raise ValueError("SLURM_JOB_ID is not set.")
                    info = subprocess.check_output(
                        ["scontrol", "show", "job", jid]
                    ).decode()

                    slurm_out = re.search(r"StdOut=(\S+)", info).group(1)
                    slurm_out_base, _ = os.path.splitext(slurm_out)

                    if os.path.exists(slurm_out):
                        self.profiling_path = Path(
                            slurm_out_base + "_quatrex_times.out"
                        )

                except Exception:
                    pass

        assert self.profiling_path is not None, "profiling_path should be set here."

        return self


class DeviceConfig(BaseModel):
    """Configuration for the simulated device.

    !!! warning
        The contacts configuration in this table is only used in QTBM
        calculations, since we allow more than two contacts in QTBM.

    """

    model_config = ConfigDict(extra="forbid")

    construct_from_unit_cell: bool = False
    """Whether to construct a device from its unit cell geometry and
    electronic structure.

    If this is set to `True`, the Hamiltonian read from the input file
    is assumed to be the tight-binding-like Hamiltonian of a single unit
    cell. The simulated device structure is then constructed by
    repeating the unit cell along the transport direction, as specified
    by `num_transport_cells`, and including the neighboring cells as
    configured by `neighbor_cell_cutoff`.

    """

    geometry: GeometryConfig
    """The geometry configuration of the device.

    This contains a defintion of all regions in the device, such as
    doping, material constants and gates.

    """

    # --- Device geometry ---------------------------------------------
    neighbor_cell_cutoff: (
        tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt] | None
    ) = None
    """The number of neighbor cells to consider along each lattice
    direction.

    If set to `None`, all neighbor cells present in the Hamiltonian
    input file are considered. A `neighbor_cell_cutoff` of zero means
    that only the unit cell itself is considered.

    Along the transport direction, at least one neighboring cell must be
    included if [`construct_from_unit_cell`](#construct_from_unit_cell)
    is `True`. If
    [`construct_from_unit_cell`](#construct_from_unit_cell) is `False`,
    including neighboring cells in transport direction is not allowed,
    since the device should already be upscaled in that case.

    If more neighbor cells are requested than present in the input
    Hamiltonian, a `ValueError` is raised.

    """

    num_transport_cells: PositiveInt = 1
    """The number of transport cells to include in the simulation.

    !!! note

        This parameter is only used if `construct_from_unit_cell` is
        `True`.

    """

    transport_direction: Literal["x", "y", "z"]
    """The direction along which the transport occurs.

    !!! note
        Currently, only axis-aligned transport directions are supported.

    """

    block_size: PositiveInt | list[PositiveInt] | None = None
    """The block size to use for the device Hamiltonian.

    This block size is used in NEGF calculations, where it determines
    the block-tridiagonal tiling of all quantities.

    If a single integer is given, a constant block size is assumed.
    Alternatively, a list of block sizes can be given to specify the
    size of each block along transport direction.

    The `block_size` parameter cannot be used in conjunction with
    [`construct_from_unit_cell`](#construct_from_unit_cell) = `True`
    since the block sizes are determined from the unit cell and the
    `neighbor_cell_cutoff` in that case.

    If [`construct_from_unit_cell`](#construct_from_unit_cell) = `False`
    in NEGF simulations, the block size must be given.

    """

    contacts: list[ContactConfig] = Field(default_factory=list)
    """The contacts of the device.

    !!! warning
        The contacts configuration in this table is only used in QTBM
        calculations, since we allow more than two contacts in QTBM.

    """

    # --- Heterogeneous device (L|D|R) --------------------------------
    num_left_cells: PositiveInt | None = None
    """Number of left-lead cells (tiled from ``left_matrix``).

    When set together with ``num_right_cells``, the device is built as
    three regions: left lead | device | right lead.  The remaining
    ``num_transport_cells - num_left_cells - num_right_cells`` cells in
    the middle are the device region and use the default matrix file.

    !!! note

        Only used when ``construct_from_unit_cell`` is ``True``.

    """

    num_right_cells: PositiveInt | None = None
    """Number of right-lead cells (tiled from ``right_matrix``)."""

    left_matrix: str | None = None
    """Matrix file name (without ``.mat``) for left-lead cells.

    Must be located in ``input_dir``.  If ``None``, the default
    ``matrix_name`` is used for all cells (homogeneous device).
    """

    right_matrix: str | None = None
    """Matrix file name (without ``.mat``) for right-lead cells."""

    num_orbitals_per_atom: dict[str, int] = {"X": 1}
    """The number of orbitals per atom type.

    This mapping is used to connect the atomistic geometry with the
    corresponding operator matrix elements.

    Currently, this is primarily used when configuring contacts via
    their real-space extents in QTBM calculations. It is also used to
    map a given potential vector to the corresponding orbitals in the
    Hamiltonian.

    The keys can be any string, that matches the atom types in the
    structure file. The default is a single atom type "X" with one
    orbital per atom, which is useful when dealing with Wannier orbitals
    that are not atom-centered.

    """

    kpoint_grid: tuple[PositiveInt, PositiveInt, PositiveInt] = (1, 1, 1)
    """The kpoint grid on which to compute transport quantities.

    This is a Monkhorst-Pack grid, which is used to sample the Brillouin
    zone transverse to the transport direction. The k-point grid is
    specified as a tuple of three integers, which correspond to the
    number of k-points along the x, y, and z directions, respectively.
    The k-point grid must be 1 along the transport direction, since the
    periodicity along that direction is broken.

    """
    kpoint_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The kpoint shift to apply to the Monkhorst-Pack grid."""

    @model_validator(mode="after")
    def to_tuple(self) -> Self:
        """Transforms list to tuple."""
        if self.neighbor_cell_cutoff is not None:
            self.neighbor_cell_cutoff = tuple(self.neighbor_cell_cutoff)
        self.kpoint_grid = tuple(self.kpoint_grid)
        return self

    @model_validator(mode="after")
    def check_kpoint_grid(self) -> Self:
        """Checks that the k-point grid is 1 along the transport direction."""

        ind = "xyz".index(self.transport_direction)
        if self.kpoint_grid[ind] != 1:
            raise ValueError(
                f"Along the transport direction ('{self.transport_direction}'), the k-point grid must be 1."
            )

        return self

    @model_validator(mode="after")
    def check_connecting_cells(self) -> Self:
        """Checks that num_connecting_cells is not zero in transport direction."""

        if self.neighbor_cell_cutoff is None:
            return self

        ind = "xyz".index(self.transport_direction)
        if not self.construct_from_unit_cell:
            if self.neighbor_cell_cutoff[ind] != 0:
                raise ValueError(
                    f"Along the transport direction ('{self.transport_direction}'),"
                    "no neighboring cells should be included if `construct_from_unit_cell` is False."
                )
        else:
            if self.neighbor_cell_cutoff[ind] < 1:
                raise ValueError(
                    f"At least one neighboring cell in transport direction "
                    f"('{self.transport_direction}') must be included."
                )

        return self

    @model_validator(mode="after")
    def check_heterogeneous_device(self) -> Self:
        """Validates heterogeneous L|D|R device configuration."""
        has_left = self.num_left_cells is not None
        has_right = self.num_right_cells is not None
        has_left_mat = self.left_matrix is not None
        has_right_mat = self.right_matrix is not None

        # All-or-nothing: if any is set, all four must be set.
        any_set = has_left or has_right or has_left_mat or has_right_mat
        all_set = has_left and has_right and has_left_mat and has_right_mat
        if any_set and not all_set:
            raise ValueError(
                "Heterogeneous device requires all four fields: "
                "num_left_cells, num_right_cells, left_matrix, right_matrix."
            )

        if all_set:
            if not self.construct_from_unit_cell:
                raise ValueError(
                    "Heterogeneous device (left_matrix/right_matrix) "
                    "requires construct_from_unit_cell = true."
                )
            n_device = (
                self.num_transport_cells
                - self.num_left_cells
                - self.num_right_cells
            )
            if n_device < 1:
                raise ValueError(
                    f"num_left_cells ({self.num_left_cells}) + "
                    f"num_right_cells ({self.num_right_cells}) must be "
                    f"less than num_transport_cells ({self.num_transport_cells})."
                )

        return self


class LyapunovComputeConfig(BaseModel):
    """Configuration concerning the solution of the Lyapunov equation."""

    model_config = ConfigDict(extra="forbid")

    eig_compute_location: Literal["numpy", "cupy", "nvmath"] = "numpy"
    """Backend to use for computing eigenvalues.

    The spectral Lyapunov solver requires the computation of eigenvalues
    of a general dense matrix. This parameter determines whether to use
    NumPy, CuPy, or NVMath for this computation. The default is NumPy.

    """

    use_pinned_memory: bool = True
    """Whether to use pinned memory when transferring data in the
    spectral Lyapunov solver."""


class NEVPConfig(BaseModel):
    """Configurations concerning the solution of NEVPs."""

    model_config = ConfigDict(extra="forbid")

    eig_compute_location: Literal["numpy", "cupy", "nvmath"] = "numpy"
    """Backend to use for computing eigenvalues.

    This parameter determines whether to use NumPy, CuPy, or NVMath for
    computing eigenvalues in the NEVP solvers. The default is NumPy.

    """

    # Parameters for contour NEVP solvers.
    project_compute_location: Literal["numpy", "cupy"] = "numpy"
    """Backend to use for computing the projection matrices.

    When using contour-based NEVP solvers, one needs to project the
    non-linear system onto a linear subspace. This can either be done
    using QR decomposition or by computing a singular value
    decomposition (SVD), which is controlled by the [`use_qr`](#use_qr)
    parameter.

    The `project_compute_location` parameter determines whether to use
    NumPy or CuPy for this computation. The default is NumPy.

    """

    use_pinned_memory: bool = True
    """Whether to use pinned memory when transferring data in the NEVP
    solvers."""

    use_qr: bool = False
    """Whether to use QR decomposition or SVD for the projection.

    When using contour-based NEVP solvers, one needs to project the
    non-linear system onto a linear subspace. This can either be done
    using QR decomposition or by computing a singular value
    decomposition (SVD). The `use_qr` parameter determines which method
    to use. The default is to use SVD, but QR decomposition can be
    significantly faster than SVD.

    """

    contour_batch_size: PositiveInt | None = None
    """The batch size to use for the contour NEVP solvers.

    The contour NEVP solvers require performing quadrature of an
    operator over a contour in the complex plane. Since this can lead to
    memory bottlenecks, the quadrature can be performed in batches. The
    `contour_batch_size` parameter determines the number of quadrature
    points to use in each batch. If set to `None`, the entire quadrature
    is performed in a single batch.

    """

    num_threads_contour: PositiveInt = 1024
    """The number of GPU threads to use for computing the operator
    inverses in the contour NEVP solvers.

    Only used if the GPU is available and the contour NEVP solvers are
    used.

    """

    # Parameters for full NEVP solvers.
    reduce_sparsity: bool = False
    """Whether to reduce the sparsity of the matrices in the full NEVP
    solver.

    The matrices arising in the full NEVP solver can contain some zero
    rows and columns, which can be removed to reduce the size of the
    eigenvalue problem.

    """


class BandEdgeConfig(BaseModel):
    """Parameters concerning the eigenvalue-based band-edge tracking."""

    model_config = ConfigDict(extra="forbid")

    use_eigvalsh: bool = True
    r"""Whether to use eigvalsh when computing the band edges.

    The non-linear eigenvalue problem

    $$ \left[\mathbf{H} + \mathbf{\Sigma}^R(E)\right] \boldsymbol{\psi}
    = E \boldsymbol{\psi}, $$

    which needs to be solved to compute the band edges is in principle a
    general eigenvalue problem. However, since we only care about real
    eigenvalues and the energy renormalization due to the Hermitian part
    of $\Sigma^R$, we can just solve the Hermitian part of the problem
    using `eigvalsh`. This is significantly faster than solving the full
    non-linear eigenvalue problem, but it is an approximation if
    scattering is included.

    Only relevant if [`band_edge_tracking`](electron/#band_edge_tracking) =
    `True`.

    """

    eigvalsh_compute_location: Literal["numpy", "cupy"] = "numpy"
    """Location where to compute the eigenvalues.

    The eigenvalues can be computed either on the CPU using NumPy or on
    the GPU using CuPy. The default is to use NumPy.

    Only relevant if [`band_edge_tracking`](electron/#band_edge_tracking) =
    `True`.

    """

    use_pinned_memory: bool = True
    """Whether to use pinned memory when transferring data in the
    band-edge tracking computation.

    Only relevant if [`band_edge_tracking`](electron/#band_edge_tracking) =
    `True`.

    """

    block_sections: PositiveInt = 1
    """The number of block sections to use when computing the band
    edges."""

    num_ref_iterations: PositiveInt = 2
    """The number of refinement iterations to use when computing the
    band edges.

    The location of the band edges need to be refined iteratively, since
    the self-energy depends on the energy.

    """

    @field_validator("use_eigvalsh", mode="after")
    @classmethod
    def check_use_eigvalsh(cls, value) -> bool:
        if not value:
            raise NotImplementedError(
                "Only use_eigvalsh=True is supported at the moment."
            )
        return value

    @field_validator("eigvalsh_compute_location", mode="after")
    @classmethod
    def check_eigvalsh_location(cls, value) -> Literal["numpy", "cupy"]:
        if value == "cupy" and xp.__name__ != "cupy":
            warnings.warn(
                "eigvalsh_compute_location is set to 'cupy' but cupy is not available. Falling back to 'numpy'.",
                UserWarning,
            )
            return "numpy"
        elif value == "numpy" and xp.__name__ == "cupy":
            warnings.warn(
                "eigvalsh_compute_location is set to 'numpy' but cupy is available. Consider setting it to 'cupy' for better performance.",
                UserWarning,
            )

        return value


class ConvolveConfig(BaseModel):
    """Parameters concerning the FFT convolution."""

    model_config = ConfigDict(extra="forbid")

    # NOTE: should be calculate from the number of energy points, ranks,
    # and nnz.
    batch_size: PositiveInt | None = None
    """The batch size to use for the FFT convolution.

    Since the performing FFT can lead to memory bottlenecks, the
    convolution can be performed in batches. The `batch_size` parameter
    determines the number of matrix elements to compute in each batch.
    If set to `None`, the entire convolution is performed in a single
    batch.

    """


class CommConfig(BaseModel):
    """Parameters concerning the communication backends.

    The communication backend in `quatrex` has two subcommicator groups:
    One between energy points and one between matrix blocks.

    For both `block` and `stack` subcommunicators, the following
    communication operations can be performed:

    - `all_to_all`
    - `all_gather`
    - `all_reduce`
    - `bcast`
    - `send_recv`

    The communication backend can be set to either `"host_mpi"`,
    `"device_mpi"`, or `"nccl"` for each of these operations.

    """

    model_config = ConfigDict(extra="forbid")

    block_comm_size: PositiveInt = 1
    """The number of ranks over which to disctribute matrix blocks.

    SCBA supports spatial domain distribution. The matrix blocks can be
    distributed over multiple ranks, which can be useful for extremely
    large systems. The `block_comm_size` parameter determines the number
    of ranks over which to distribute the matrix blocks.

    If set to 1 (the default), the matrix blocks are not distributed
    over multiple ranks.

    """

    block_all_to_all: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for block all-to-all."""
    block_all_gather: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for block all-gather."""
    block_all_reduce: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for block all-reduce."""
    block_bcast: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for block broadcast."""
    block_send_recv: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for block send-receive."""

    stack_all_to_all: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for stack all-to-all."""
    stack_all_gather: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for stack all-gather."""
    stack_all_reduce: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for stack all-reduce."""
    stack_bcast: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for stack broadcast."""
    stack_send_recv: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for stack send-receive."""

    # Transverse-momentum (q-point) communicator: a third axis alongside
    # block x stack, used to distribute the external q of the q-resolved
    # phonon-phonon self-energy. Default 1 leaves block/stack unchanged.
    q_comm_size: PositiveInt = 1

    q_all_to_all: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    q_all_gather: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    q_all_reduce: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    q_bcast: Literal["host_mpi", "device_mpi", "nccl"] | None = None


class ComputeConfig(BaseModel):
    """Top level configuration for all performance and compute options."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    dsdbsparse_type: DSDBSparse = DSDBCOO
    """The type of sparse matrix to use for the DSDBSparse matrices.

    !!! warning
        Currently, only `DSDBCOO` is supported. A CSR type had been
        implemented, but it is no longer fully supported.

    """
    numba_threading_layer: Literal["workqueue", "omp", "tbb"] = "workqueue"
    """The threading layer to use for Numba.

    We recommend using the default `"workqueue"` threading layer in
    numba, as we have had issues with correctly limiting the number of
    threads when using the `"omp"` threading layer.

    """

    threadpool_api: Literal["blas", "openmp"] | None = None
    """The threadpool API to use for threadpoolctl.

    - If "blas", it will only limit BLAS supported libraries.
    - If "openmp", it will only limit OpenMP supported libraries.
        Note that it can affect the number of threads used by the BLAS libraries
        if they rely on OpenMP.
    - If None, this will apply to all threadpoolctl supported libraries.

    """

    numba_num_threads: PositiveInt | None = None
    """The number of threads to use for Numba."""

    blas_num_threads: PositiveInt | Literal["sequential_blas_under_openmp"] | None = (
        None
    )
    """The number of threads to use for BLAS."""

    convolve: ConvolveConfig = ConvolveConfig()
    """Parameters concerning the FFT convolution in scattering interactions."""
    nevp: NEVPConfig = NEVPConfig()
    """Parameters concerning the solution of non-linear eigenvalue problems."""
    lyapunov: LyapunovComputeConfig = LyapunovComputeConfig()
    """Parameters concerning the solution of Lyapunov equations."""
    band_edge: BandEdgeConfig = BandEdgeConfig()
    """Parameters concerning the eigenvalue-based band-edge tracking."""
    comm: CommConfig = CommConfig()
    """Parameters concerning the communication backends."""

    @field_validator("dsdbsparse_type", mode="before")
    @classmethod
    def set_dsdbsparse(cls, value) -> DSDBSparse:
        """Converts the string value to the corresponding DSDBSparse object."""
        if value == "DSDBCOO":
            return DSDBCOO
        raise ValueError(f"Invalid value '{value}' for dbsparse")


class QuatrexConfig(BaseModel):
    """Top-level simulation configuration."""

    model_config = ConfigDict(extra="forbid")

    # --- Simulation parameters ---------------------------------------
    device: DeviceConfig
    """The device configuration."""

    formalism: Literal["wf", "negf"]
    """The transport formalism to use.

    There are two supported formalisms:

    - `"wf"`: Wavefunction formalism
    - `"negf"`: Non-equilibrium Green's function formalism

    !!! warning "Inconsistent input formats"

        Currently, the input formats for the two formalisms are not
        consistent.

    """
    simulation_type: Literal["electron", "phonon"] = "electron"
    """Which subsystem carries the transport: the electronic one (with
    optional scattering interactions) or the phononic one (thermal
    transport, with optional 3-phonon scattering)."""

    scsp: SCSPConfig | None = None
    """Parameters for the self-consistent Schrödinger-Poisson loop."""

    scba: SCBAConfig = SCBAConfig()
    """Parameters for the self-consistent Born approximation loop."""

    qtbm: QTBMConfig = QTBMConfig()
    """Parameters for the quantum transmitting boundary method."""

    electrostatics: ElectrostaticsConfig = ElectrostaticsConfig()
    """Parameters for the electrostatics calculations."""

    electron: ElectronConfig
    """Parameters for the electronic system."""

    phonon: PhononConfig | None = None
    """Parameters for the phonon system."""
    coulomb_screening: CoulombScreeningConfig | None = None
    """Parameters for the Coulomb screening."""
    photon: PhotonConfig | None = None
    """Parameters for the photon system."""

    # --- Directory paths ----------------------------------------------
    config_dir: Path
    simulation_dir: Path = Path(".")
    """The directory where the simulation is run."""
    input_dir: Path | None = None
    """The directory where the input files are located."""
    output_dir: Path | None = None
    """The directory where the output files are saved."""

    # --- Output options -----------------------------------------------
    outputs: OutputConfig = OutputConfig()
    """Parameters for the output of `quatrex` calculations."""

    # --- Compute options ----------------------------------------------
    compute: ComputeConfig = ComputeConfig()
    """Parameters for the performance and compute options."""

    @model_validator(mode="after")
    def resolve_config_path(self) -> Self:
        """Resolves the config directory path."""
        self.config_dir = Path(self.config_dir).resolve()
        return self

    @model_validator(mode="after")
    def resolve_simulation_dir(self):
        """Resolves the simulation directory path."""
        self.simulation_dir = (self.config_dir / self.simulation_dir).resolve()
        return self

    @model_validator(mode="after")
    def set_output_dir(self):
        """Resolves the simulation directory path."""
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
            if self.output_dir.is_absolute():
                self.output_dir = self.output_dir.resolve()
                return self

            self.output_dir = (self.config_dir / self.output_dir).resolve()
            return self

        self.output_dir = self.simulation_dir / "outputs/"
        return self

    @model_validator(mode="after")
    def set_input_dir(self) -> Path:
        """Returns the input directory path."""
        if self.input_dir is not None:
            self.input_dir = Path(self.input_dir)
            if self.input_dir.is_absolute():
                self.input_dir = self.input_dir.resolve()
                return self

            self.input_dir = (self.config_dir / self.input_dir).resolve()
            return self

        self.input_dir = self.simulation_dir / "inputs/"
        return self

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        """Validates the input file paths."""

        if (
            self.electron.energy_window_min is None
            and self.electron.energy_window_max is None
            and self.electron.energy_window_num is None
            and self.electron.energy_window_num_per_rank is None
        ):
            if not (self.input_dir / "electron_energies.npy").resolve().is_file():
                raise ValueError(
                    f"Energy grid not specified and file '{(self.input_dir / 'electron_energies.npy').resolve()}' does not exist."
                )

        # TODO: extend this to other paths, not only energies

        return self

    @model_validator(mode="after")
    def check_device_block_size(self) -> Self:
        """Checks that block size is consistent with other parameters."""

        if self.formalism == "wf":
            # NOTE: Block sizes are not used in the wavefunction
            # formalism.
            return self

        if self.device.construct_from_unit_cell and self.device.block_size is not None:
            raise ValueError(
                "block_size cannot be used in conjunction with construct_from_unit_cell=True."
            )

        if not self.device.construct_from_unit_cell and self.device.block_size is None:
            raise ValueError(
                "block_size must be given when construct_from_unit_cell=False."
            )

        return self

    @model_validator(mode="after")
    def check_phonon_self_energy_symmetry(self) -> Self:
        """Rejects symmetric storage for the phonon self-energy.

        `scba.symmetric` tags the retarded self-energy buffer as
        `"hermitian"`, which discards the anti-Hermitian part on read. The
        phonon solver consumes that buffer as the FULL retarded self-energy
        (Hermitian + damping), so the anharmonic linewidth would be silently
        annihilated.
        """
        if self.simulation_type == "phonon" and self.scba.symmetric:
            raise ValueError(
                "scba.symmetric is not supported for simulation_type='phonon': "
                "the phonon retarded self-energy is not Hermitian and its "
                "damping part would be discarded."
            )

        return self

    @model_validator(mode="after")
    def check_device_contact_voltages(self) -> Self:
        """Checks that at least one contact exists and is grounded."""
        # TODO: Contacts should be unified between the two formalisms.
        if self.simulation_type == "phonon":
            # Phonon transport defines its leads via temperatures, not
            # electronic contacts.
            return self
        if self.formalism == "negf":
            if (
                self.electron.left_contact is None
                or self.electron.right_contact is None
            ):
                raise ValueError("Both left and right contacts must be defined.")
            contacts = [self.electron.left_contact, self.electron.right_contact]
        elif self.formalism == "wf":
            contacts = self.device.contacts
        else:
            raise ValueError(f"Invalid formalism '{self.formalism}'.")

        if len(contacts) < 2:
            raise ValueError("At least two contacts must be defined.")

        if not any(contact.voltage == 0 for contact in contacts):
            raise ValueError(
                "At least one contact must be grounded (i.e. have zero voltage)."
            )

        return self

    @model_validator(mode="after")
    def check_either_fermi_or_midgap(self) -> Self:
        """Checks that either the Fermi level or the mid-gap energy is set."""
        if self.simulation_type == "phonon":
            return self
        if self.formalism == "negf":
            contacts = [self.electron.left_contact, self.electron.right_contact]
        elif self.formalism == "wf":
            contacts = self.device.contacts
        else:
            raise ValueError(f"Invalid formalism '{self.formalism}'.")

        for contact in contacts:
            if contact.fermi_level is None and contact.mid_gap_energy is None:
                raise ValueError(
                    "Either `fermi_level` or `mid_gap_energy` must be set."
                )
            if contact.mid_gap_energy is None and self.formalism == "wf":
                raise ValueError(
                    "In the 'wf' formalism, `mid_gap_energy` must be set for each contact."
                )

        return self

    @model_validator(mode="after")
    def check_contact_direction(self) -> Self:
        """Checks that the contact direction is set in "wf" formalism."""

        if self.formalism == "negf":
            # NOTE: The contact direction is not used in the NEGF
            # formalism.
            return self

        for contact in self.device.contacts:
            if contact.direction is None:
                raise ValueError(
                    "The `direction` parameter of each contact must be "
                    "set in the 'wf' formalism."
                )

        return self


def parse_config(config_file: Path) -> QuatrexConfig:
    """Reads the TOML config file.

    Only rank 0 process reads the config file. It is then broadcasted to
    the other processes. Each process then parses the config into a
    `QuatrexConfig` object.

    Parameters
    ----------
    config_file : Path
        Path to the TOML configuration file.

    Returns
    -------
    QuatrexConfig
        The parsed configuration object.

    """
    config = None
    if mpi_comm_world.rank == 0:
        config_file = Path(config_file).resolve()

        with open(config_file, "rb") as f:
            config = tomllib.load(f)

        if "simulation_dir" in config:
            simulation_dir = config["simulation_dir"]
            if not os.path.isabs(simulation_dir):
                parent_dir = os.path.dirname(os.path.abspath(config_file))
                simulation_dir = Path(os.path.join(parent_dir, simulation_dir))
                config["simulation_dir"] = simulation_dir

        config["config_dir"] = config_file.parent

    config = mpi_comm_world.bcast(config, root=0)

    # Resolve the geometry config.
    config["device"]["geometry"] = parse_geometry_config(config["device"])

    return QuatrexConfig(**config)


def _setup_profiler(config: QuatrexConfig) -> None:
    """Sets up the profiler based on the given configuration.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration object containing the profiling settings.

    """

    if not config.outputs.profiling_path.is_absolute():
        config.outputs.profiling_path = (
            config.config_dir / config.outputs.profiling_path
        ).resolve()

    # Saving will strip the extension
    profiler.set_parameters(
        print_path=config.outputs.profiling_path,
        save_path=config.outputs.profiling_path,
        save_format=config.outputs.profiling_save_format,
    )


def _setup_comm(comm_config: CommConfig) -> None:
    """Sets up the communication backend.

    Parameters
    ----------
    comm_config : CommConfig
        The communication configuration containing the communication settings.

    """
    default_backend = "host_mpi" if xp.__name__ == "cupy" else "device_mpi"

    block_comm_config = {
        "all_to_all": comm_config.block_all_to_all or default_backend,
        "all_gather": comm_config.block_all_gather or default_backend,
        "all_reduce": comm_config.block_all_reduce or default_backend,
        "bcast": comm_config.block_bcast or default_backend,
        "send_recv": comm_config.block_send_recv or default_backend,
    }

    stack_comm_config = {
        "all_to_all": comm_config.stack_all_to_all or default_backend,
        "all_gather": comm_config.stack_all_gather or default_backend,
        "all_reduce": comm_config.stack_all_reduce or default_backend,
        "bcast": comm_config.stack_bcast or default_backend,
        "send_recv": comm_config.stack_send_recv or default_backend,
    }

    q_comm_config = {
        "all_to_all": comm_config.q_all_to_all or default_backend,
        "all_gather": comm_config.q_all_gather or default_backend,
        "all_reduce": comm_config.q_all_reduce or default_backend,
        "bcast": comm_config.q_bcast or default_backend,
        "send_recv": default_backend,
    }

    comm.configure(
        block_comm_size=comm_config.block_comm_size,
        block_comm_config=block_comm_config,
        stack_comm_config=stack_comm_config,
        override=True,
        q_comm_size=comm_config.q_comm_size,
        q_comm_config=q_comm_config,
    )


def _setup_threading(compute_config: ComputeConfig):
    """Sets up the threading layer.

    Parameters
    ----------
    compute_config : ComputeConfig
        The compute configuration containing the threading settings.

    """

    # TODO: set the number of threads automatically based on the available cores
    # problems is that we do not know yet how many energy points there will be
    # has to be after unifying the configs
    # NOTE: here we could now do this tuening
    if compute_config.numba_num_threads is None:
        compute_config.numba_num_threads = 1
    if compute_config.blas_num_threads is None:
        compute_config.blas_num_threads = 1

    nb.set_num_threads(compute_config.numba_num_threads)
    nb.config.THREADING_LAYER = compute_config.numba_threading_layer

    if compute_config.numba_num_threads == 1 and compute_config.blas_num_threads in [
        "sequential_blas_under_openmp",
        1,
    ]:
        if comm.rank == 0:
            warnings.warn(
                "The CPU code will run sequentially which may impact performance.",
                UserWarning,
            )


def setup_context(config: QuatrexConfig) -> None:
    """Sets up the simulation context based on the given configuration.

    This includes setting up the profiler, the communication backend,
    and the threading layer.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration object containing the settings for the
        simulation context.

    """
    _setup_profiler(config)
    _setup_comm(config.compute.comm)
    _setup_threading(config.compute)
