# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Configuration for the experimental analytic-pole treatment."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self


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
    there. 0 = auto, resolved to ``4*dw``."""
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


