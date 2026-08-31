# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Configuration for experimental SCBA root finders."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)


class ExperimentalMixerConfig(BaseModel):
    """Experimental SCBA root-finders for the iteration-UNSTABLE eta->0 fixed
    point, kept out of the shared SCBA surface. Damped/Anderson mixing
    accelerates a contractive iteration, whereas broyden/rpm/rre/jfnk LAND a
    fixed point whose Jacobian has ``|lambda| > 1`` (which mixing cannot
    reach). Research use only."""

    model_config = ConfigDict(extra="forbid")

    rre_cycle: PositiveInt = 8
    """For ``mixing_method = "rre"``: restart cycle length (number of iterates
    per reduced-rank-extrapolation step)."""
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
