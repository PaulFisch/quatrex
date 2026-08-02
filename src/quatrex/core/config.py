# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

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
    """Options for the self-consistent Schrödinger-Poisson loop."""

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

    \[
        \lVert V_{n} - V_{n-1} \rVert_{\infty} < \texttt{convergence_tol}
    \]

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

    Only used if `mixer` is set to "diis".

    """

    epsilon: PositiveFloat = 1e-5
    """Regularization parameter for the least-squares problem in the
    DIIS method to ensure numerical stability.

    Only used if `mixer` is set to "diis".

    """

    extrapolation_interval: PositiveInt = 1
    """Number of iterations between DIIS extrapolation steps.

    For example, if set to 3, the mixer will perform two
    under-relaxation steps followed by a DIIS extrapolation step, and
    then repeat this cycle. If set to 1 (the default), the Pulay mixing
    is performed at every iteration.

    Only used if `mixer` is set to "diis".

    """


class QTBMConfig(BaseModel):
    """Options for the quantum transmitting boundary method (QTBM)."""

    model_config = ConfigDict(extra="forbid")

    max_batch_size: PositiveInt = 10
    """The maximum number of energies per OBC batch."""

    low_rank_obc: bool = False
    """Whether to use reduced rank for the boundary self-energies.
    
    If set to True, boundary self-energies are moved to the
    right-hand-side of linear system, which greatly reduces fill-in
    during factorization.

    The system matrix becomes Hermitian or even real symmetric in
    gamma-only simulations. Therefore, the low_rank_obc parameter can
    only be used in combination with direct solvers that can exploit the
    symmetry, i.e., `direct_solver="cudss"` on GPU,
    `direct_solver="pardiso"` on CPU, and `direct_solver="thomas"` on
    both CPU and GPU.

    """


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
    """Options for the self-consistent Born approximation."""

    model_config = ConfigDict(extra="forbid")

    min_iterations: PositiveInt = 1
    max_iterations: PositiveInt = 100
    convergence_tol: PositiveFloat = 1e-5

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

    coulomb_screening: bool = False
    photon: bool = False
    phonon: bool = False

    symmetric: bool = False

    align_self_energy_to_complex_axes: bool = True
    r"""Whether to discard parts of the self-energy.

    This affects the self-energy in the following way:
    - The real parts of the lesser/greater self-energy are discarded.
    - The imaginary part of the retarded self-energy from any previous
    computation is zeroed.

    This happens before the imaginary part of the retarded self-energy
    is computed from the lesser and greater parts as
    $$\mathrm{Im}\left[\mathbf{\Sigma}^R\right] =
    \frac{\mathbf{\Sigma}^> - \mathbf{\Sigma}^<}{2i}$$.

    """


class ElectrostaticsConfig(BaseModel):
    """Options for the Poisson solver."""

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

    Only used if `solving_scheme` is set to "root-finding".

    """

    convergence_tol: PositiveFloat = 1e-3
    """The convergence tolerance for the root-finding scheme.

    This is defined as the infinity norm of the potential update in the
    root-finding scheme.

    Only used if `solving_scheme` is set to "root-finding".

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

    Only used if `solving_scheme` is set to "root-finding".

    """

    density_model_dim: Literal[1, 2, 3] = 2
    """The dimensionality of the system to use for the single-band
    density model.

    The density model does not have to match the actual dimensionality
    of the system. For example, a 2D density model might actually work
    best for systems of all dimensionalities.

    Only used if `solving_scheme` is set to "root-finding" and
    `density_model` is set to "single-band".

    """

    initial_guess: Literal["zero", "constraints", "file"] = "zero"
    """The strategy to generate the initial guess for the potential.

    - `"zero"`: Uses a zero potential as the initial guess.
    - `"constraints"`: Solves a linear Poisson equation with the
        potential constraints to generate the initial guess. This is
        expected to work best at regimes close to equilibrium where the
        potential does not vary too much.
    - `"file"`: Loads the initial guess from a file. The file should be
        located in the `input_dir` and named `potential.npy`.

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
    """Options for memoizing wrappers.

    The memoizers store and reuse previously computed results
    to speed up the fixed-point iterations in OBC and Lyapunov solvers.

    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["auto", "force", "force-after-first", "off"] = "auto"
    """The memoization mode to determine when to do fixed-point iterations.

    - "auto": Automatically decides whether to use memoization based on the
        specified tolerances. Only useful if all ranks memoize.
    - "force": Always use memoization.
    - "force-after-first": Use memoization after the first SCBA iteration.
    - "off": Never use memoization.
    """

    num_ref_iterations: PositiveInt = Field(default=2, ge=2)
    """The number of fixed-point iterations to perform."""

    relative_tol: PositiveFloat = 2e-1
    """The relative tolerance for the fixed-point iterations.

    Only used if `mode` is set to "auto".
    """

    absolute_tol: PositiveFloat = 1e-6
    """The absolute tolerance for the fixed-point iterations.

    Only used if `mode` is set to "auto".
    """

    warning_threshold: PositiveFloat = 1e-1
    """The threshold for issuing a warning if the memoized functions
        residual is above this value after the fixed-point iterations.
    """

    agreement_threshold: float = Field(default=0.999, ge=0, le=1)


class SolverConfig(BaseModel):
    """Options for the system solver."""

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["rgf", "inv"] = "rgf"

    # The maximum number of energies per batch.
    max_batch_size: PositiveInt = 100

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
        This is parameter is only used in the Electron Solver. The
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

    In runs with `low_rank_obc = true`, the system matrix will be
    Hermitian or even real and symmetric in gamma-only simulations. In
    those cases, libraries that can exploit the symmetry are preferred,
    i.e., cuDSS on GPU and PARDISO on CPU.

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
    r"""Options for open-boundary condition (OBC) solvers.

    The OBC solvers compute the surface Green's functions of the contacts.
    The surface Green's functions is the solution of the non-linear equation:

    $$ \mathbf{g} = [\mathbf{M}_{0} - \mathbf{M}_{-1} g \mathbf{M}_{1} ]^{-1} $$
    """

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["sancho-rubio", "spectral"] = "spectral"
    """The OBC algorithm to use.

    - "sancho-rubio": Uses the Sancho-Rubio iterative scheme to compute the
        surface Green's functions. This method achieves exponential convergence
        compared to the linear convergence of fixed-point iterations.
    - "spectral": Uses a spectral NEVP solver to compute eigenpair and uses
        them to construct the surface Green's functions. This is generally more
        efficient method when combined with a contour integral NEVP solver,
        but requires more parameter tuning.
    """

    nevp_solver: Literal["beyn", "full"] = "beyn"
    r"""The NEVP solver to use for the spectral OBC algorithm.

    - "beyn": Uses the Beyn's contour integral method to solve the NEVP to
        find the eigenpairs within a specified contour in the complex plane.

    - "full": Uses a full dense eigensolver to solve for all eigenvalues by linearizing
        the problem. This results in a doubled problem size which is also not reduced by
        block sectioning / periodicity.

    The following NEVP problem is solved:

    $$ \sum \limits_{n=-b}^{b} \lambda^{n} \hat{\mathbf{M}}_{n} \vec{v} = 0 $$

    where b goes from -block_sections to +block_sections and
    $\hat{\mathbf{M}}_{n}$ are potentially reduced coupling matrices.

    Only used if `algorithm` is set to "spectral".
    """

    # Parameters for spectral OBC algorithms.
    block_sections: PositiveInt = 1
    """The periodicity of the blocks along the transport direction.

    Used in the spectral method with beyn to reduce the size of the NEVP.
    For example, if the supercell is constructed from 2 unit cells along the
    transport direction, setting this parameter to 2 will halve the size of the NEVP.

    Contact blocks need to be sorted accordingly.
    """

    min_decay: PositiveFloat = 1e-3
    """The minimum decay rate where to differentiate between propagating and evanescent modes."""

    max_decay: PositiveFloat | None = None
    """The maximum decay rate for evanescent modes.

    Very large modes do not contribute to the surface Green's functions and
    can be neglected. Very large modes can also lead to numerical instabilities.

    If not set, it is computed as 1.5 * log(r_o).
    """

    num_ref_iterations: PositiveInt = 2
    r"""The number of fixed-point iterations used to refine the surface Green's functions.

    $$ \mathbf{g}_{n+1} = [\mathbf{M}_{0} - \mathbf{M}_{-1} \mathbf{g}_{n} \mathbf{M}_{1} ]^{-1} $$

    This is needed to improve the accuracy of the surface Green's functions
    if not enough eigenpairs are considered.

    Only used if `algorithm` is set to "spectral".
    """

    min_propagation: PositiveFloat = 1e-2
    r"""The minimum propagation speed for propagating modes.

    The propagation speed is computed as:

    $$ abs(real(\frac{dE}{dk})) / abs(imag(\frac{dE}{dk})) $$

    """

    residual_tolerance: PositiveFloat = 1e-3
    r"""The tolerance for the residual of the eigenpairs.

    The residuals are computed as:

    $$ \lvert \sum \limits_{n=-b}^{b} \lambda^{b} \mathbf{M}_{n} \vec{v} \rvert $$

    Modes above this tolerance are considered wrong and are not used.

    Only used if `algorithm` is set to "spectral".
    """

    residual_normalization: bool = True
    """Whether to normalize the residuals by the norm of the eigenvalue.

    This is useful to avoid that large eigenvalues have large residuals
    and small eigenvalues have small residuals.
    """

    warning_threshold: PositiveFloat = 1e-1
    r"""The threshold for issuing a warning if the surface Green's functions
    residual is above this value.

    The residual is computed as:

    $$ \lvert \mathbf{g} - [\mathbf{M}_{0} - \mathbf{M}_{-1} \mathbf{g} \mathbf{M}_{1} ]^{-1} \rvert / \lvert \mathbf{g} \rvert $$

    This parameter is only used if the `formalism` is `wf`. Otherwise, the memoizer
    is responsible for residual checking and warnings.
    """

    eta_decay: PositiveFloat = 1e-12
    """Small value to separate very slow decaying modes from
        non-decaying ones in the spectral OBC solver.

    Modes that are very close to the unit contour could be misclassified
    with 'min_decay' and 'min_propagation' conditions i.e.
    when their decay is smaller than 'min_decay' but they are not propagating fast enough.
    The not fast enough propagating ones with decay smaller than 'eta_decay' are
    considered as well decaying modes.
    """

    # Parameters for iterative OBC algorithms.
    max_iterations: PositiveInt = 100
    """The maximum number of iterations for the Sancho-Rubio method."""

    convergence_tol: PositiveFloat = 1e-6
    """The convergence tolerance for the Sancho-Rubio method."""

    # Parameters for subspace NEVP solvers.
    r_o: PositiveFloat = 10.0
    """The outer radius of the contour in the complex plane for the contour methods.

    This parameter should not be too large to avoid having too many eigenpairs
    inside the contour. It should also not be too small to avoid missing important
    eigenpairs. If a eigenpair is too close to the contour,
    it can lead to numerical instabilities.
    """

    r_i: PositiveFloat = 0.8
    """The inner radius of the contour in the complex plane for the contour methods.

    This parameter should be chosen to be <1 to capture propagating modes, but
    not too small to avoid including too many modes.
    """

    m_0: PositiveInt = 10
    """The subspace guess in the contour methods.

    The guess has to be larger than the expected number of eigenvalues
    inside the contour. If too small, the method will fail. If too large, the method
    will be not/less efficient.
    """

    num_quad_points: PositiveInt = 20
    """The number of quadrature points for the contour integrals."""

    # Parameters for reusing surface Green's functions from previous
    # SCBA iterations.
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
    r"""Options for solving the Lyapunov equation.
    The discrete Lyapunov equation arises in the computation of
    the boundary conditions for quantities such as W.

    The discrete Lyapunov equation has the form:

    $$ \mathbf{A} \mathbf{X} \mathbf{A}^{\dagger} - \mathbf{X} = - \mathbf{Q} $$

    """

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["spectral", "doubling"] = "spectral"
    r"""The Lyapunov solver algorithm used

    - "spectral": Uses the eigenvalue decomposition to solve the Lyapunov equation.
        This method is more expensive since a full eigendecomposition is required.
    - "doubling": Uses the doubling method to iteratively solve the Lyapunov equation.
        This method should exponentially converge, but it is not stable if $\mathbf{A}$
        has eigenvalues outside the unit circle.

    """

    reduce_sparsity: bool = True
    r"""Whether to use the sparsity of $\mathbf{A}$ to accelerate the Lyapunov solver.

    This is done by removing zero rows and columns from $\mathbf{A}$, solving the reduced
    Lyapunov equation, and then expanding the solution back to the original size.
    """

    assume_constant_sparsity: bool = False
    r"""Whether to assume that the sparsity pattern of $\mathbf{A}$ is constant
    during the SCBA iterations.
    In practice, this should be the case, but not guaranteed.

    If set to True, the sparsity pattern is only computed once during
    the first SCBA iteration and reused for subsequent iterations.
    """

    # Parameters for iterative Lyapunov algorithms.
    max_iterations: PositiveInt = 100
    """The maximum number of iterations for the doubling method."""

    relative_tol: PositiveFloat = 1e-4
    """The relative tolerance for the doubling method."""

    absolute_tol: PositiveFloat = 1e-8
    """The absolute tolerance for the doubling method."""

    # Parameter for spectral Lyapunov solver.
    num_ref_iterations: PositiveInt = Field(default=2, ge=1)
    """The number of fixed-point iterations used to refine the solution
        of the solution of the spectral Lyapunov solver.

    This is not used in the doubling method. Additionally, the number of iterations in
    the memoizer is also independent of this parameter.
    """

    memoizer: MemoizerConfig = MemoizerConfig()
    """Options for memoizing the Lyapunov solver."""


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
    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()

    eta_obc: NonNegativeFloat = 0  # eV
    eta: NonNegativeFloat = 1e-12  # eV

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
    above the conduction band edge during the SCBA iterations.

    """

    energy_window_min: float | None = None
    energy_window_max: float | None = None
    energy_window_num: PositiveInt | None = None
    energy_window_num_per_rank: PositiveInt | None = None

    flatband: bool | None = None

    dos_peak_limit: PositiveFloat = 100.0

    filtering_iteration_limit: PositiveInt = 1

    max_batch_size: PositiveInt | None = None
    """The maximum number of energies to batch together in the solution
    of the electronic subsystem.

    This controls how many energies are treated together when computing boundary
    conditions and electron Green's functions. If not set, all energies are
    computed at once.

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

    solver: SolverConfig = SolverConfig()
    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()

    temperature: PositiveFloat = 300.0  # K

    epsilon_r: PositiveFloat = 1.0

    left_temperature: PositiveFloat | None = None
    right_temperature: PositiveFloat | None = None

    # How many blocks should be merged into a single block.
    num_connected_blocks: Literal["auto"] | PositiveInt = "auto"

    dos_peak_limit: PositiveFloat = 100.0

    filtering_iteration_limit: PositiveInt = 1

    align_polarization_to_complex_axes: bool = True
    r"""Whether to discard parts of the polarization.

    This affects the polarization in the following way:
    - The real parts of the lesser/greater polarization are discarded.
    - The imaginary part of the retarded polarization from previous
    computation is zeroed.

    This happens before the imaginary part of the retarded polarization
    is computed from the lesser and greater parts as
    $$\mathrm{Im}\left[\mathbf{P}^R\right] = \frac{\mathbf{P}^> -
    \mathbf{P}^<}{2i}$$.

    """

    include_energy_renormalization: Literal["self-energy", "polarization", "both"] = (
        "self-energy"
    )
    r"""Whether to compute the real part of the retarded polarization and/or self-energy.

    Possible values are `"self-energy"`, `"polarization"`, and `"both"`.

    The full retarded interaction quantities are complex-valued, where
    the real part is computed from the imaginary part using the
    Kramers-Kronig relations:

    $$\mathbf{X}^{R} = \frac{\mathbf{X}^{>} - \mathbf{X}^{<}}{2} +
    \frac{1}{2\pi} \mathrm{p.v.} \int_{-\infty}^{\infty}  dE' \,
    \frac{\mathbf{X}^{>} - \mathbf{X}^{<}}{E^{'} - E}$$

    The real part only leads to only a shift in the energy, so it is
    often neglected:

    $$\mathbf{X}^{R} \approx \frac{\mathbf{X}^{>} - \mathbf{X}^{<}}{2}$$

    The default is to only include the real part in the Coulomb
    screening self-energy and not in the polarization.

    The real part is computed using a Hilbert transform. For the Coulomb
    screening self-energy, this Hilbert transform can lead to errors at
    the edges of the energy window. The `apply_hilbert_correction`
    option can be used to apply a correction to the Hilbert transform to
    mitigate these errors.

    """

    apply_hilbert_correction: bool = False
    """Whether to apply the corrections for the edges of the energy window
    to the hilbert transform when computing the retarded self-energy.

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

    This controls how many energies are treated together when computing boundary
    conditions and screened Coulomb interactions. If not set, all energies are
    computed at once.

    This can help mitigate memory bottlenecks.

    """


class PhotonConfig(BaseModel):
    """Options for the optical degrees of freedom."""

    model_config = ConfigDict(extra="forbid")

    interaction_cutoff: PositiveFloat = 10.0  # Angstrom

    solver: SolverConfig = SolverConfig()
    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()


class PhononConfig(BaseModel):
    """Options for the thermal degrees of freedom."""

    model_config = ConfigDict(extra="forbid")

    interaction_cutoff: PositiveFloat = 10.0  # Angstrom

    solver: SolverConfig = SolverConfig()
    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()

    eta_obc: NonNegativeFloat = 0  # THz^2 (constant imaginary OBC shift)
    eta: NonNegativeFloat = 1e-12  # THz (frequency-linear damping 2*eta*|omega|)
    eta_ramp_iterations: NonNegativeInt = 0
    """Anneal the broadening DOWN over the first N SCBA iterations: the solver's
    eta goes linearly from ``eta`` (iteration 0, while Sigma^R is still ~0) to
    ``eta_final`` by iteration N, then stays at ``eta_final``. Lets the anharmonic
    Sigma^R take over the broadening as it develops (the eta=0 limit). 0 = off
    (constant eta)."""
    eta_final: NonNegativeFloat = 0.0  # THz: target broadening at the end of the ramp
    eta_obc_ramp_iterations: NonNegativeInt = 0
    """Anneal the CONTACT broadening ``eta_obc`` DOWN over the first N SCBA
    iterations: eta_obc goes linearly from ``eta_obc`` (iteration 0, large
    enough to converge the cell cold) to ``eta_obc_final`` by iteration N,
    then holds. 0 = off (constant eta_obc)."""
    eta_obc_final: NonNegativeFloat = 0.0  # THz^2: target contact broadening at ramp end

    model: Literal["pseudo-scattering", "negf"] = "pseudo-scattering"
    phonon_energy: NonNegativeFloat | None = None
    deformation_potential: NonNegativeFloat | None = None
    temperature: PositiveFloat = 300.0  # K

    band_edge_tracking: Literal["dos-peaks", "eigenvalues"] | None = None
    filtering_iteration_limit: PositiveInt = 1

    left_temperature: PositiveFloat | None = None
    """Temperature of the left phonon reservoir. Defaults to `temperature`."""
    right_temperature: PositiveFloat | None = None
    """Temperature of the right phonon reservoir. Defaults to `temperature`."""

    # --- 3-phonon (anharmonic) scattering ----------------------------
    fc3_path: Path | None = None
    """Path to the FC3 source consumed by ``SigmaPhononPhonon``.

    Required when ``model == "negf"``. Format: HDF5 produced by the
    ``phonon_inputs`` pipeline (block-sparse ``/fc3_blocks`` or dense
    ``/fc3``).
    """

    qfold_path: Path | None = None
    """Path to the q-folded device vertices for transversely-periodic
    (``kpoint_grid`` with k>1) anharmonic transport.

    A ``.npz`` written by :func:`quatrex.phonon.qfold.save_qfold`
    (built offline from the real-space FC3 + transverse Bloch phases via
    ``phonon.solver.se_q``). Holds ``{(iq1, iq2): {(I,K,Kp): Phi}}`` plus
    the ``q_diff_map``. Required when any ``kpoint_grid`` entry is > 1 and
    ``model == "negf"``; ignored for the Gamma-only (k==1) device.
    """

    decomposed_vertices_path: Path | None = None
    """Path to the TENSOR-DECOMPOSED coupled-q device vertex factors.

    A ``.npz`` written by
    :func:`quatrex.phonon.vertex_factors.save_decomposed` (built offline
    from a CP/INDSCAL factorisation of the bulk FC3 via
    ``phonon/phonon_inputs/fc3_factor_device.py``). The exact per-leg
    factorisation of the same q-folded blocks ``qfold_path`` holds densely
    -- O(n_off * N_q * n_dof * R) instead of O(N_q^2) dense block dicts.
    Mutually exclusive with ``qfold_path``.
    """

    sse_vertex_rank: int = 0
    """Truncate the decomposed vertex to the leading ``rank`` components.

    ``0`` (default) keeps the full stored rank. Columns are weight-sorted
    at export, so this makes an R-sweep a config-only knob against one
    high-rank factor file. Requires ``decomposed_vertices_path``.
    """

    decomposed_kernel: Literal["gram", "reconstruct"] = "gram"
    """How the SSE consumes the decomposed vertex.

    ``"gram"`` (default): the factored contraction
    (``quatrex.phonon.bubble_factored``). The quad sum collapses onto two summed
    Grams and the transverse-momentum sum runs as an FFT, which the dense vertex
    cannot do, so this is asymptotically cheaper at every rank the fit needs.
    ``"reconstruct"``: materialise the rank-local slice of the dense q-folded
    dict from the factors and run the dense contraction. Keeps the storage win
    but none of the arithmetic one; retained as a fallback and as the oracle the
    parity test compares against.
    """

    retarded_method: Literal["half", "fft"] = "fft"
    """How to reconstruct ``Sigma^R`` from ``Sigma^{<,>}``.

    - ``"half"``: ``Sigma^R = (Sigma^> - Sigma^<) / 2``.
    - ``"fft"``: also add the bosonic Hilbert correction
      ``i/2 * H[Sigma^> - Sigma^<]`` (uses the same FFT kernel as
      ``coulomb_screening/polarization.py``).
    """

    phonon_phonon_truncation_warn: NonNegativeFloat = 0.01
    """Frobenius-norm threshold for the FC3 nearest-neighbour-truncation
    warning (cf. ``fc3_loader.fc3_to_phi_blocks``)."""

    eta_ir_floor_cells: NonNegativeFloat = 0.0
    """Sub-grid soft-mode broadening floor for the eta=0 SCBA, in grid cells
    (0 = off). Adds a DC-CONCENTRATED constant broadening to the Dyson
    denominator,
        z^2(omega) += i * Gamma_floor * omega_c^2/(omega^2 + omega_c^2),
    Gamma_floor = (eta_ir_floor_cells*dw)^2 [THz^2], omega_c = 2*dw, damping
    only the lowest (unresolved, ~zero-heat) bins that are otherwise
    unregularised at eta=0 (G^R ~ 1/omega^2 at the acoustic soft modes).
    Grid-consistent (Gamma_floor -> 0 as dw -> 0); not applied to the OBC."""
    eta_ir_floor_final_cells: NonNegativeFloat = 0.0
    """Target for the in-SCBA anneal of ``eta_ir_floor_cells`` (grid cells).
    With ``eta_ir_floor_ramp_iterations`` > 0 the soft-mode floor is ramped
    linearly from its start value down to this over the ramp, then held."""
    eta_ir_floor_ramp_iterations: int = 0
    """Number of SCBA solves over which to anneal ``eta_ir_floor_cells`` down to
    ``eta_ir_floor_final_cells`` (0 = off, hold the floor constant)."""
    buttiker_probe: bool = False
    """Optional self-consistent Buttiker DEPHASING probe on the eta-broadening
    channel (default OFF). The numerical broadening ``eta`` adds a damping
    ``Gamma_eta = 4*eta*omega`` to G^R with NO matching fluctuation, which
    violates the fluctuation-dissipation balance and, under a thermal bias,
    injects a spurious energy current; when True, a matching fluctuation
    ``Sigma_probe^{<,>} = i*Gamma_eta*(n_p + 0/1)`` is added to the device
    source, with ``n_p = G^< / (G^> - G^<)`` updated self-consistently each
    SCBA iteration so the LOCAL probe current vanishes at every energy.
    NOTE: this injects elastic DEPHASING (physics, not a pure regularizer) --
    for the pure coherent+anharmonic conductance use eta->0 extrapolation
    instead. Single-block (block_comm_size==1) only."""

    bubble_balance_check: bool = False
    """Per-iteration Phi-derivable energy-balance diagnostic of the 3-phonon
    bubble: P_in = sum hbar*w*Tr[Sigma^< G^>] must equal P_out (conserving
    identity). Off by default: it keeps the Green's-function data alive through
    the nnz->stack back-transpose, which costs one extra all-to-all per
    iteration at stack > 1."""

    sse_vertex_scale: PositiveFloat = 1.0
    """Uniform 3-phonon vertex scale lambda (Sigma scales as lambda^2).
    lambda < 1 = reduced-coupling runs for extrapolation and for soft-mode
    structures whose full-coupling bubble-only SCBA is unstable."""

    sse_low_freq_mask_thz: NonNegativeFloat = 0.0
    """Zero the bubble legs and outputs on all |omega| < this (THz). The
    frequency grid stays anchored at zero (the FFT convolution and the
    bosonic fold require it), so this is the working equivalent of
    starting the window above the acoustic region: masked bins keep their
    ballistic Dyson/transport content but contribute no three-phonon
    scattering. 0 = legacy (only the omega = 0 bin is masked)."""

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
    """How Sigma comes back from the auxiliary bubble grid onto the
    primary grid. ``"adjoint"`` (default): the adjoint of the leg
    interpolation w.r.t. the ENERGY measure w*|omega|,
    R = (W O)^-1 P^T (dw O_aux) -- the hbar*omega-weighted pairing
    sum_m w_m om_m Tr[Sigma(w_m) G(w_m)] then equals the aux-grid
    pairing EXACTLY, so the dual-grid bubble keeps the Phi-derivable
    ENERGY balance (and the lead heat balance J_L - J_R = P_in - P_out)
    to roundoff. A pointwise sample breaks it at the interpolation-error
    level, concentrated where the primary grid is coarsest; the net
    current is a small difference of large fluxes, so the leak dominates
    the lead balance long before it is visible in Sigma itself.
    ``"sample"``: pointwise linear sampling of the aux-grid Sigma
    (sharper at resonance peaks, not conserving). Identical when the
    grids coincide (up to the masked omega = 0 bin)."""

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

    sse_tau_chunk_bytes: PositiveInt = 256 * 1024 * 1024
    """Memory cap on one tau chunk of the decomposed (``"gram"``) SSE kernel.

    The kernel's Gram tables are ``(N_q, n_tau, R, R)``, so the working set grows
    with the tau slice AND with the square of the rank: on a single rank at
    R = 128 the full local tau axis is tens of GB. Tau is therefore split so that
    one chunk's Gram stays under this cap, independently of the thread pool (the
    GPU path has none). Results do not depend on the chunk size."""

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
    """Reconstruct the cross terms of Sigma^> from the Sigma^< ring pass via
    the exact bosonic tau-domain identity (the ji-transposed, tau-reversed
    cross terms of pair (J, I) are the absorption terms of pair (I, J)):
    4 instead of 6 ring contractions per vertex quad, and the reversed-lesser
    leg is never built. Construction-exact -- independent of any property of
    G -- up to summation order (~1e-13 rel). Gamma-only (nq == 1); verify
    with ``sse_fold_verify_iterations``."""

    sse_fold_verify_iterations: NonNegativeInt = 0
    """With ``sse_greater_from_lesser``: for the first N compute() calls run
    the LEGACY 6-ring path, additionally accumulate the cross terms, and
    report the max-abs/rel mismatch of the reconstruction identity per output
    pair (rank 0; single-rank runs only -- multi-rank runs skip the gate with
    a warning). The legacy result is shipped during these iterations (with
    ``sse_hermitian_pairs`` the halving is also suspended during them)."""

    sse_hermitian_pairs: bool = False
    """Contract only the upper-triangle output pairs (I <= J) and fill the
    (J, I) blocks via the anti-Hermiticity Sigma_JI(w) = -Sigma_IJ(w)^dagger
    (conjugate-transpose in the stack state + tau reversal in the nnz state).
    Exact only to the extent the input G is anti-Hermitian (solver precision,
    NOT construction-exact) -- verify with a run-level A/B and watch the
    bubble-balance residual. Single-rank runs only. Diagonal blocks are
    always contracted fully (no projection is applied)."""

    sse_ramp_iterations: NonNegativeInt = 0
    """Adiabatic switch-on of the 3-phonon bubble: scale the scattering
    self-energy by ``min(1, it/N)`` over the first N SCBA iterations
    (0 = off). Stabilises soft-mode structures whose full-coupling SCBA
    overshoots into unphysical gain states under plain damped iteration."""

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
    """Inner Green's-function block band |K - K'| kept in the bubble
    contraction. With 1 the RGF block-tridiagonal G masks the bubble
    kernel G(x)G to that band -- a masked positive-semidefinite form is
    NOT positive-semidefinite (Schur product with the indefinite
    tridiagonal-ones mask), so interior slabs (>= 3 transport cells)
    acquire non-causal gain components of Sigma. With 2, the solver
    additionally produces the second off-diagonal G^{<,>} blocks and the
    contraction keeps all links the nearest-neighbour vertex span needs:
    the diagonal Sigma blocks become exact and causal. With 3, the third
    off-diagonal G^{<,>} blocks are produced too, so the first off-diagonal
    Sigma blocks become exact and causal as well. Extends the shared
    G/Sigma sparsity pattern by the corresponding off-diagonal blocks
    (Sigma's extra blocks stay structurally zero). Single block-rank only.

    Default 3 (2026-08-01): the L16-L32 CNT ladder showed the band-1
    boxcar's anomalous gain GROWS with length while the Bartlett-tapered
    band-1 run underweights off-diagonal coherence ~2x -- both
    incorrect; the full band-3 run is the reference. The value is
    clamped at use to n_blocks - 1 (a band wider than the device has
    off-diagonals is meaningless), so short devices keep their exact
    full-band behaviour."""

    sse_g_band_taper: Literal["none", "bartlett"] = "none"
    """PSD taper of the inner-G band mask. The boxcar band truncation is a
    Schur product with the indefinite band-ones matrix; it destroys the
    positive-semidefiniteness of the bubble kernel and injects non-causal
    gain (the sse_g_band=1 instability). "bartlett" weights every inner G
    link by w_d = 1 - d/(sse_g_band+1) (d = |K-K'|) and the Sigma output
    blocks by the same w_{|I-J|}: the taper matrix is PSD (Fejer kernel),
    so by the Schur product theorem -+i Sigma^{<,>} stays PSD -- causal at
    ANY band -- and using the same taper on G and Sigma is the
    Phi-derivable pair (Phi[M o G]), so Baym-Kadanoff energy conservation
    is retained. Price: off-diagonal coherence is underweighted (band-1:
    factor 1/2 per G link and on the off-diagonal Sigma blocks). "none" is
    the legacy boxcar, bit-identical to previous behaviour. Not supported
    with decomposed_kernel="gram" (per-quad weights do not factor through
    the Gram collapse)."""

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
    """Optional self-consistent-phonon (SCP) cubic tadpole static
    self-energy (``model == "negf"``). The cubic tadpole
    ``Sigma_T = Phi3 : <u>`` is a STATIC real self-energy that stiffens the
    soft modes (the finite-T SCP renormalisation), stabilising the dynamic
    bubble; it is recomputed every SCBA iteration from the current device
    ``G^<`` and added to the dynamical matrix via ``Sigma^R``. Needs only
    FC3 (the quartic loop, which would need FC4, is omitted). Default OFF;
    cf. ``quatrex/phonon/static_self_energy.py``.

    NOTE: the implementation assembles dense device-level arrays (FC3
    tensor + Phi_eff eigensolve); intended for single / few-cell devices."""

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
        return self


class OutputConfig(BaseModel):
    """Options for the output."""

    model_config = ConfigDict(extra="forbid")

    # Only the spectral currents are saved by default.
    device_currents: bool = True

    potential: bool = False

    electron_ldos: bool = False
    electron_density: bool = False
    hole_density: bool = False

    polarization_density: bool = False
    coulomb_screening_density: bool = False

    self_energy_density: bool = False

    profiling_path: Path | None = None
    """The files to print and save the timing results to.

    For printing, the full name with extension is used while for saving
    the extension give by `profiling_save_format` is used.

    If None, the file is tried to be infered from the SLURM output file,
    else the default quatrex_times.out is used.
    """

    save_profiling_results: bool = False
    """If the timing stats should be saved."""

    profiling_save_format: Literal["pickle", "json"] = "json"
    """The format to save the timing results in."""

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
    model_config = ConfigDict(extra="forbid")

    construct_from_unit_cell: bool = False

    geometry: GeometryConfig
    """The geometry configuration of the device.

    This contains a defintion of all regions in the device, such as
    doping, material constants and gates.

    """

    # --- Device geometry ---------------------------------------------
    neighbor_cell_cutoff: (
        tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt] | None
    ) = None
    """The number of neighbor cells to consider along each lattice direction.

    If set to `None`, all neighbor cells are considered. A
    `neighbor_cell_cutoff` of zero means that only the unit cell itself
    is considered.

    Along the transport direction, at least one neighboring cell must be
    included if `construct_from_unit_cell` is `True`. If
    `construct_from_unit_cell` is `False`, no neighboring cells should be
    included along the transport direction.

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

    block_size: PositiveInt | list[PositiveInt] | None = None
    """The block size to use for the device Hamiltonian.

    If a single integer is given, a constant block size is assumed.
    Alternatively, a list of block sizes can be given to specify the
    size of each block along transport direction.

    This cannot be used in conjunction with
    `construct_from_unit_cell=True` since the block sizes are determined
    from the unit cell and the `neighbor_cell_cutoff`.

    On the other hand, if `construct_from_unit_cell=False`, the block
    size must be given.

    """

    contacts: list[ContactConfig] = Field(default_factory=list)

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

    kpoint_grid: tuple[PositiveInt, PositiveInt, PositiveInt] = (1, 1, 1)
    kpoint_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)

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
    """Configuration concerning the Lyapunov solvers."""

    model_config = ConfigDict(extra="forbid")

    eig_compute_location: Literal["numpy", "cupy", "nvmath"] = "numpy"
    use_pinned_memory: bool = True


class NEVPConfig(BaseModel):
    """All configurations concerning the solution of NEVPs."""

    model_config = ConfigDict(extra="forbid")

    eig_compute_location: Literal["numpy", "cupy", "nvmath"] = "numpy"

    # Parameters for contour NEVP solvers.
    project_compute_location: Literal["numpy", "cupy"] = "numpy"
    use_pinned_memory: bool = True

    use_qr: bool = False
    contour_batch_size: PositiveInt | None = None
    num_threads_contour: PositiveInt = 1024

    # Parameters for full NEVP solvers.
    reduce_sparsity: bool = False


class BandEdgeConfig(BaseModel):
    """Parameters concerning the eigenvalue-based band-edge tracking."""

    model_config = ConfigDict(extra="forbid")

    use_eigvalsh: bool = True
    """Whether to use eigvalsh or eig to compute the eigenvalues to
    determine the band edges. The eigvalsh function is more efficient,
    but is an approximation if scattering is included.

    Only used if the band edge tracking is set to "eigenvalues".
    """

    eigvalsh_compute_location: Literal["numpy", "cupy"] = "numpy"
    """Location where to compute the eigenvalues.

    Only used if the band edge tracking is set to "eigenvalues".
    """

    use_pinned_memory: bool = True
    """Whether to use pinned memory for eigenvalue computations.

    Only used if the band edge tracking is set to "eigenvalues".
    """

    block_sections: PositiveInt = 1

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
    """All configurations concerning the fft convolution."""

    model_config = ConfigDict(extra="forbid")

    # NOTE: should be calculate from the number of energy points, ranks,
    # and nnz.
    batch_size: PositiveInt | None = None


class CommConfig(BaseModel):
    """All configurations concerning the communication."""

    model_config = ConfigDict(extra="forbid")

    block_comm_size: PositiveInt = 1

    block_all_to_all: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    block_all_gather: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    block_all_reduce: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    block_bcast: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    block_send_recv: Literal["host_mpi", "device_mpi", "nccl"] | None = None

    stack_all_to_all: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    stack_all_gather: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    stack_all_reduce: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    stack_bcast: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    stack_send_recv: Literal["host_mpi", "device_mpi", "nccl"] | None = None

    # Transverse-momentum (q-point) communicator: a third axis alongside
    # block x stack, used to distribute the external q of the q-resolved
    # phonon-phonon self-energy. Default 1 leaves block/stack unchanged.
    q_comm_size: PositiveInt = 1

    q_all_to_all: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    q_all_gather: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    q_all_reduce: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    q_bcast: Literal["host_mpi", "device_mpi", "nccl"] | None = None


class ComputeConfig(BaseModel):
    """All configurations concerning computational details."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    dsdbsparse_type: DSDBSparse = DSDBCOO
    numba_threading_layer: Literal["workqueue", "omp", "tbb"] = "workqueue"
    threadpool_api: Literal["blas", "openmp", "tbb"] | None = None
    numba_num_threads: PositiveInt | None = None
    blas_num_threads: PositiveInt | Literal["sequential_blas_under_openmp"] | None = (
        None
    )

    convolve: ConvolveConfig = ConvolveConfig()
    nevp: NEVPConfig = NEVPConfig()
    lyapunov: LyapunovComputeConfig = LyapunovComputeConfig()
    band_edge: BandEdgeConfig = BandEdgeConfig()
    comm: CommConfig = CommConfig()

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
    formalism: Literal["wf", "negf"]
    """The transport formalism to use.

    There are two supported formalisms:

    - "wf": Wavefunction formalism
    - "negf": Non-equilibrium Green's function formalism

    !!! warning "Input formats"

        Currently, the input formats for the two formalisms are not
        consistent.

    """
    simulation_type: Literal["electron", "phonon"] = "electron"
    """Which subsystem carries the transport: the electronic one (with
    optional scattering interactions) or the phononic one (thermal
    transport, with optional 3-phonon scattering)."""

    scsp: SCSPConfig | None = None
    scba: SCBAConfig = SCBAConfig()
    qtbm: QTBMConfig = QTBMConfig()
    electrostatics: ElectrostaticsConfig = ElectrostaticsConfig()

    electron: ElectronConfig

    phonon: PhononConfig | None = None
    coulomb_screening: CoulombScreeningConfig | None = None
    photon: PhotonConfig | None = None

    # --- Directory paths ----------------------------------------------
    config_dir: Path
    simulation_dir: Path = Path("./quatrex/")
    input_dir: Path | None = None
    output_dir: Path | None = None

    # --- Output options -----------------------------------------------
    outputs: OutputConfig = OutputConfig()

    # --- Compute options ----------------------------------------------
    compute: ComputeConfig = ComputeConfig()

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

            if (
                contact.fermi_level is not None
                and contact.mid_gap_energy is not None
                and not (self.electron.band_edge_tracking or self.scsp is not None)
            ):
                raise ValueError(
                    "Both `fermi_level` and `mid_gap_energy` cannot be set "
                    "simultaneously, unless band edge tracking is active "
                    "or the Schrödinger-Poisson solver is enabled."
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
