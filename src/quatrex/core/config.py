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
    point (cnt33 / SiNW d5a soft-mode saddle), kept out of the shared SCBA
    surface. These are NOT standard NEGF accelerators: damped/Anderson mixing
    accelerates a contractive iteration, whereas broyden/rpm/rre/jfnk LAND a
    fixed point whose Jacobian has ``|lambda| > 1`` (which mixing provably cannot
    reach). They target an η=0 self-energy fixed point whose existence on the
    soft modes is itself marginal -- use only for that research path."""

    model_config = ConfigDict(extra="forbid")

    rre_cycle: PositiveInt = 8
    """For ``mixing_method = "rre"``: restart cycle length (number of iterates per
    reduced-rank-extrapolation step). Locates an UNSTABLE fixed point that damped /
    Anderson mixing cannot reach (the cnt33 eta=0 band-edge mode on longer cells);
    cf. ``RREMixer`` in ``quatrex/core/anderson.py``."""

    broyden_warmup_iters: NonNegativeInt = 0
    """For ``mixing_method = "broyden"`` / ``"rpm"``: run plain damped-LINEAR
    mixing for the first N SCBA iterations (still accumulating secant history),
    then engage the quasi-Newton / projection step. The iteration-unstable map
    limit-cycles in a BOUNDED neighbourhood of the saddle, so the warm-up parks
    the iterate there -- where the Jacobian I-G' is nonsingular and the secant
    buffer is clean -- before the local root finder takes over. 0 = engage from
    the start."""
    broyden_ridge: NonNegativeFloat = 1e-8
    """For ``mixing_method = "broyden"`` / ``"rpm"``: Tikhonov ridge on the small
    multisecant / restricted-Jacobian solve, scaled by the matrix norm. Kept TINY
    so it does not damp the Newton correction in the near-marginal band-edge
    subspace (a fat ridge biases the step back toward limit-cycling Picard)."""
    broyden_trust: NonNegativeFloat = 0.3
    """For ``mixing_method = "broyden"`` / ``"rpm"``: trust-region step cap -- the
    per-iteration update ``||Sigma_new - Sigma||`` is limited to
    ``broyden_trust * ||Sigma||``. A full quasi-Newton step from a far-from-root /
    stale secant model overshoots the nonlinear SCBA map (residual spikes > 2);
    the cap forces gradual descent until the model is good, then deactivates near
    the root. 0 disables it."""
    rpm_max_subspace: PositiveInt = 6
    """For ``mixing_method = "rpm"``: cap on the dimension k of the unstable
    invariant subspace on which Newton is performed (Picard on the contractive
    complement). The cnt33 instability is a single complex band-edge pair, so the
    effective k is ~2; 6 leaves margin. cf. ``RPMMixer`` in ``quatrex/core/rpm.py``."""

    # --- Jacobian-free Newton-Krylov (mixing_method = "jfnk") -----------------
    # JFNK lands a STRONGLY-unstable fixed point (|lambda(J_F)| ~ 100s, several
    # unstable modes) where the subspace-tracking RPM fails: GMRES on the matrix-
    # free Newton system J delta = -R needs only finite-difference J*v products
    # and converges on a cluster-plus-few-outliers spectrum. The Krylov solve runs
    # in the real embedding [Re Sigma, Im Sigma] (the map is real- not complex-
    # linear). cf. ``quatrex/core/jfnk.py``. The SiNW d5a Si-H bending eta=0 saddle.
    jfnk_warmup_iters: NonNegativeInt = 10
    """For ``mixing_method = "jfnk"``: damped-LINEAR steps to fall into the fixed
    point's basin before engaging Newton-Krylov (the unstable modes have not yet
    blown up in the first ~10 iterations)."""
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
    """For ``mixing_method = "jfnk"``: INITIAL trust-region cap on the Newton step,
    ``||delta|| <= jfnk_trust * ||Sigma||`` (global). Adapts down on a step that
    raises the residual, up on good progress (toward ``jfnk_trust_max``). 0
    disables. Start it SMALL (e.g. 0.05) so the early, far-from-root Newton steps
    cannot overshoot the marginal flexural mode (the d5a blow-up); the radius then
    breathes up as descent is demonstrated."""
    jfnk_trust_max: NonNegativeFloat = 0.0
    """For ``mixing_method = "jfnk"``: MAXIMUM trust radius the adaptive growth may
    reach (``<= 0`` -> use ``jfnk_trust``, i.e. no growth above the initial). Set
    ``jfnk_trust_max > jfnk_trust`` to let the radius accelerate as the residual
    descends monotonically -- the cure for the d5a marginal-mode crawl where the
    Newton step is permanently pinned at a small fixed trust boundary."""
    jfnk_newton_damp: PositiveFloat = 1.0
    """For ``mixing_method = "jfnk"``: damping of the (already trust-capped) Newton
    step, ``Sigma_{k+1} = Sigma_k + jfnk_newton_damp * delta``. < 1 globalises a
    far-from-root start."""
    jfnk_ptc: NonNegativeFloat = 0.0
    """For ``mixing_method = "jfnk"``: pseudo-transient / Levenberg-Marquardt shift
    -- solve ``(J + mu I) delta = -R`` with ``mu = jfnk_ptc * ||R_k||/||R_0||``
    (annealed to 0 as the residual falls, recovering pure Newton at the root). The
    shift lifts the near-zero (marginal, ``Gamma_anh ~ dw``) eigenvalues of
    ``J = J_F - I`` off the origin so the inner GMRES no longer stalls on the
    near-null-space. 0 = pure Newton (no shift); ~1 for the marginal d5a saddle."""


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

    mixing_method: Literal["linear", "anderson", "broyden", "rre", "rpm", "jfnk"] = "linear"
    """Self-energy fixed-point mixer. ``"linear"`` is plain damped mixing;
    ``"anderson"`` is plain Anderson(m) acceleration (Anderson 1965;
    Walker & Ni, SIAM J. Numer. Anal. 49, 1715 (2011), Alg. AA in the
    unconstrained least-squares form) -- cf. ``quatrex/core/anderson.py``.
    Acceleration helps a convergent (contractive) iteration; it is NOT a
    cure for a marginal / non-existent fixed point. ``"broyden"`` is the
    MPI-aware type-I "good" Broyden ROOT FINDER and ``"rpm"`` the Recursive
    Projection Method (Shroff & Keller 1993); both LAND an iteration-UNSTABLE
    fixed point (Jacobian |lambda|>1, the cnt33 eta=0 band-edge mode on long
    cells) that damped/Anderson/RRE provably cannot reach -- cf.
    ``quatrex/core/broyden.py`` and ``quatrex/core/rpm.py``."""

    anderson_depth: PositiveInt = 5
    """History size m for ``mixing_method = "anderson"`` (number of stored
    residual differences). Memory cost: 2*m copies of the full Sigma."""
    anderson_period: PositiveInt = 1
    """Periodic-Pulay stride: apply the Anderson extrapolation only every
    ``anderson_period``-th iteration, plain damped linear mixing in between
    (Banerjee et al. 2016). >1 breaks the marginal-mode limit cycle that stalls
    plain Anderson on soft-mode systems (d5a). 1 = ordinary Anderson(m)."""
    anderson_warmup_iters: NonNegativeInt = 0
    """For ``mixing_method = "anderson"``: run plain damped LINEAR mixing for the
    first ``anderson_warmup_iters`` SCBA iterations, then switch to Anderson.
    Anderson limit-cycles when started cold on the eta=0 causal-Sigma^R map (the
    early non-linear transient), but converges fast once the iterate is near the
    fixed point where the map is well-linearized. 0 = Anderson from the start."""
    anderson_restart: NonNegativeInt = 0
    """For ``mixing_method = "anderson"``: forget the Anderson history every N
    steps (0 = never). Breaks the marginal-mode limit cycle that periodic Pulay
    alone cannot escape on the eta=0 causal-Sigma^R band-edge mode (resid
    oscillates 0.05<->0.49)."""
    anderson_ridge: NonNegativeFloat = 0.0
    """For ``mixing_method = "anderson"``: scale-relative Tikhonov regularisation
    of the least-squares coefficients (suppresses the overshoot spikes from a
    near-rank-deficient history). 0 = plain SVD lstsq. Also reused as the Gram
    ridge for ``mixing_method = "rre"``."""

    experimental_mixer: ExperimentalMixerConfig = Field(
        default_factory=ExperimentalMixerConfig)
    """Knobs for the experimental eta=0 root-finders (broyden / rpm / rre / jfnk),
    selected via ``mixing_method``. Grouped here to keep the experimental
    fixed-point machinery out of the shared SCBA surface; see
    :class:`ExperimentalMixerConfig`."""

    abort_residual: NonNegativeFloat = 1e3
    """Divergence guard: abort the SCBA once the relative self-energy
    residual exceeds this after iteration 3 (an exploded update never
    recovers). 0 disables. QX_ABORT_RESIDUAL overrides."""

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

    This is only supported for the `"rgf"` algorithm. If not set, it is
    automatically determined based on the algorithm. (i.e. `True` for
    `"rgf"` and `False` for `"inv"`)

    If `True`, the current is computed between each layer and from/to the
    leads. This way of computing the current is usually preferable as it
    is independet of any interaction cutoffs, since it is computed from
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
        # Both "rgf" and "inv" support the Meir-Wingreen boundary current.
        # ("inv" is the small-eta-stable path: its dense inverse pivots
        # through the near-singular Dyson matrix that NaNs the RGF recursion.)
        if self.compute_current is None:
            self.compute_current = True

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
    """Anneal the CONTACT broadening ``eta_obc`` DOWN over the first N SCBA iterations:
    eta_obc goes linearly from ``eta_obc`` (iteration 0, large enough to converge the
    cell cold) to ``eta_obc_final`` by iteration N, then holds. The MPI-compatible
    in-run analogue of the eta_obc warm-start chain for the eta=0 fixed point on longer
    cells (warm-start files are single-rank only). 0 = off (constant eta_obc)."""
    eta_obc_final: NonNegativeFloat = 0.0  # THz^2: target contact broadening at ramp end

    model: Literal["pseudo-scattering", "negf"] = "pseudo-scattering"
    phonon_energy: NonNegativeFloat | None = None
    deformation_potential: NonNegativeFloat | None = None
    temperature: PositiveFloat = 300.0  # K

    band_edge_tracking: Literal["dos-peaks", "eigenvalues"] | None = None
    filtering_iteration_limit: PositiveInt = 1

    left_temperature: PositiveFloat | None = None
    right_temperature: PositiveFloat | None = None

    # --- 3-phonon (anharmonic) scattering ----------------------------
    fc3_path: Path | None = None
    """Path to the FC3 source consumed by ``SigmaPhononPhonon``.

    Required when ``model == "negf"``. Format: HDF5 produced by the
    ``phonon_inputs`` pipeline (phono3py / hiphive / DFPT). The Phase-2
    sparse-block writer in ``phonon_inputs/quatrex_writer.py`` will
    extend this to consume an on-disk block-sparse Phi.
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

    sse_zero_bands_thz: list[list[float]] = []
    """DIAGNOSTIC frequency ablation: hard-zero the scattering self-energy
    (bubble input G legs, output Sigma^<>, post-Hilbert Sigma^R) in the given
    ``[lo, hi]`` THz windows. NOT a physical treatment -- it deletes real
    two-phonon spectral weight like any hard mask (spectral-deformation
    audit); use it to BISECT which spectral region seeds an eta=0 runaway.
    """

    decomposed_kernel: Literal["gram", "reconstruct"] = "reconstruct"
    """How the SSE consumes the decomposed vertex.

    ``"reconstruct"`` (default): materialise the rank-local slice of the
    dense q-folded dict from the factors once at first compute and run the
    dense contraction at full speed -- the factored win is memory + build
    time. ``"gram"``: the skinny-Gram factored contraction
    (``quatrex.phonon.bubble_factored``); fewer flops than dense only at
    small rank or large block sizes (it is memory-bound in R^2 and loses to
    the dense path beyond R ~ 16 on small-block films).
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
    (0 = off). At eta=0 the device retarded G^R = [omega^2 + 2i*eta*omega - D -
    Sigma^R]^{-1} is UNREGULARISED at the acoustic soft modes (D->0): the
    frequency-proportional damping 2i*eta*omega vanishes as omega->0, so G^R ~
    1/omega^2 blows up and the SCBA diverges. d5a's twist modes (~0.01 THz) sit
    far below the first grid bin -> unresolved, transport-irrelevant (they carry
    ~zero heat), but they destabilise the iteration. This adds a DC-CONCENTRATED
    constant broadening to the Dyson denominator,
        z^2(omega) += i * Gamma_floor * omega_c^2/(omega^2 + omega_c^2),
    Gamma_floor = (eta_ir_floor_cells*dw)^2 [THz^2], omega_c = 2*dw, so only the
    lowest few (unresolved) bins are damped and the resolved low-omega physics
    (and the IR occupation plateau, see ``sse_ir_subtraction``) is untouched.
    Grid-consistent: Gamma_floor -> 0 as dw -> 0. Stabiliser for eta=0; does NOT
    crush the heat current like the omega^2 occupation taper."""
    eta_ir_floor_final_cells: NonNegativeFloat = 0.0
    """Target for the in-SCBA anneal of ``eta_ir_floor_cells`` (grid cells). With
    ``eta_ir_floor_ramp_iterations`` > 0 the soft-mode floor is ramped linearly
    from its start value down to this over the ramp, then held. Tests whether the
    floor is removable upon convergence (anneal -> 0 holds) or load-bearing (the
    fixed point re-diverges as the floor -> 0)."""
    eta_ir_floor_ramp_iterations: int = 0
    """Number of SCBA solves over which to anneal ``eta_ir_floor_cells`` down to
    ``eta_ir_floor_final_cells`` (0 = off, hold the floor constant)."""
    sse_low_freq_cutoff_thz: NonNegativeFloat = 0.0
    """Low-frequency cutoff for the 3-phonon SSE (THz, 0 = off). Modes below
    the cutoff are excluded from the bubble (masked on the INPUT Green's
    functions fed to the convolution -- the Green's functions themselves are
    untouched for Dyson/observables) AND the resulting Sigma is not applied
    below the cutoff: transport below it stays purely BALLISTIC. The
    omega=0 bin is always excluded regardless."""
    ir_taper_cells: NonNegativeFloat = 0.0
    """IR occupancy-taper width in GRID CELLS (0 = off). The Bose occupancy
    n(omega) ~ kT/(hbar*omega) diverges as omega->0; sampled at the first grid
    bin it injects a ~1/dw spike that makes the eta=0 SCBA Sigma^R limit-cycle
    with an UNPHYSICAL IR linewidth Gamma ~ 1/omega (the acoustic sum rule forces
    Gamma ~ omega^2 -> 0). The lead occupancy is multiplied by the smooth taper
    ``t(omega) = omega^2 / (omega^2 + (ir_taper_cells * dw)^2)`` -- regularizing
    the unresolved sharp eta=0 IR poles with a minimal effective width
    ``omega_reg = ir_taper_cells * dw``. ``t ~ (omega/omega_reg)^2`` as omega->0
    (exact ASR onset) and ``t -> 1`` smoothly at large omega (no kink, so the IR
    instability is removed rather than relocated to the first un-tapered bin).
    Applied identically to both leads, so every G^< leg inherits it consistently
    and the Phi-derivable bubble energy balance is preserved. Tied to the grid
    spacing dw: the tapered band [0, ir_taper_cells*dw] shrinks as the grid
    refines, so the converged observable is taper-independent (a grid-consistent
    IR regularization of the unresolved Bose pole, NOT a fixed-frequency cutoff
    that would delete real low-omega channels)."""
    sse_ir_subtraction: bool = False
    """Replace the ``ir_taper_cells`` omega^2 IR taper with the physically-exact
    IR treatment at the LEAD OCCUPATION level (phonon/solver.py). The lead
    injection is Sigma^<_lead = i Gamma(omega) n(omega); the lead broadening
    Gamma(omega) is ODD (Gamma(0)=0, ~omega for acoustic), so Gamma*n is FINITE
    as omega->0 even though n ~ kT/(hbar*omega) diverges. The omega^2 taper
    (omega_reg = ir_taper_cells*dw) therefore OVER-regularizes and unphysically
    crushes the low-omega heat current (destroying the quantised plateau
    omega*I(omega)->const; see document/src/appendices/infrared.tex). With this
    flag the FULL physical Bose occupation is kept (Gamma~omega tames the pole;
    only the omega=0 bin is clipped), applied identically to both leads so the
    device G stays consistent and the Phi-derivable bubble energy balance is
    preserved (~1e-15). NB the bubble itself is left as the bare conserving
    convolution -- the device G^< has no 1/omega pole to subtract (the bosonic
    fold + bounded spectral A cancel it). Off by default; sets ir_taper_cells=0.
    Restores the low-omega physics but does NOT by itself stabilise the eta=0
    SCBA iteration (a separate fixed-point problem)."""
    band_limit_sse: bool = False
    """Restrict the 3-phonon SSE to the phonon band support: mask the bubble
    INPUT G and OUTPUT Sigma at frequencies with no (resolved) spectral weight,
    A(omega)=i(G^>-G^<) < ``spectral_support_tol`` x max. 'Only scatter where
    there are states.' This is the GENERIC, automatic replacement for a hand-set
    cutoff: it auto-locates the band edges at BOTH ends (the empty above-band
    grid region that makes eta=0 diverge for the CNT, and the sub-grid soft band
    for d5a), with no per-system tuning. Physically exact (Sigma=0 where A=0).
    Recommended for the eta->0 limit."""
    spectral_support_tol: NonNegativeFloat = 1e-4
    """Threshold (relative to the per-frequency spectral-weight peak) below which
    a frequency is treated as outside the band support by ``band_limit_sse``."""
    band_support_margin_thz: NonNegativeFloat = 0.0
    """With ``band_limit_sse``: HARMONIC spectral-support masking (0 = off, use
    only the single above-band-top mask). The bulk dispersion
    D(k)=D0+D1 e^{ik}+D1^H e^{-ik} is diagonalized over k in [0,pi]; the union of
    all eigenfrequencies is the phonon spectral support. A grid bin farther than
    ``band_support_margin_thz`` from EVERY band frequency is empty (no states) and
    is masked out of the SSE -- on BOTH sides AND in interior gaps. This is the
    correct treatment for GAPPED / SPARSE spectra (e.g. SiNW: Si band 0-16 THz +
    discrete Si-H surface modes 18-64 THz with empty bins between them) where the
    eta->0 FFT convolution leaks Sigma into the empty bins with no spectral weight
    to damp it -> divergence. The single band-top mask misses these interior
    empties. Computed from the dynamical matrix only (G-independent, so robust to
    the multi-cell corner-block padding that defeats a live A(omega) threshold)
    and frozen. Set the margin to ~ the mode linewidth + a few grid spacings
    (e.g. 1-2 THz) so real modes keep a scattering window but the gaps are masked."""
    sse_freeze_occupation: NonNegativeFloat = 0.0
    """With ``band_limit_sse``: also mask SSE bins whose thermal Bose occupation
    ``n(omega, T_hot)`` is below this threshold (0 = off). These modes are
    thermally FROZEN -- they carry negligible heat (heat ~ hbar*omega*n) -- so
    masking them out of the SSE is transport-exact while removing their
    contribution to the eta=0 iteration. Targets isolated, sharp, gapped
    high-frequency modes that barely self-broaden at eta=0 (e.g. the SiNW Si-H
    STRETCH island ~61-64 THz: n~1e-5 at 300 K, separated from the main band by a
    ~30 THz gap, so almost no 3-phonon decay channels). ``T_hot`` is the warmer
    lead temperature, so a mode is frozen only if frozen at the hot contact too.
    Typical: 1e-3 (masks omega > ~43 THz at 300 K). Applied as an omega-diagonal
    mask on the SSE input/output (conserving), like the band-limit."""
    spectral_sharp_cap: NonNegativeFloat = 0.0
    """With ``band_limit_sse``: also mask NEAR-SINGULAR (sharp) modes -- bins where
    the spectral weight A(omega) exceeds this multiple of the in-band MEDIAN (a
    sub-grid-linewidth resonance spikes A~1/Gamma). 0 = off. These bins are zeroed
    from the bubble for THAT iteration (treated ballistic) and re-admitted once the
    self-energy broadens them enough to drop below the cap -- a dynamic 'trim the
    near-singular modes, reintroduce as they heal' that stabilises the eta->0 SCBA
    on long devices where the anharmonic linewidth alone is sub-grid. ~20-50."""

    buttiker_probe: bool = False
    """Optional self-consistent Buttiker DEPHASING probe on the eta-broadening
    channel (default OFF). The numerical broadening ``eta`` adds a damping
    ``Gamma_eta = 4*eta*omega`` to G^R with NO matching fluctuation -- a
    "damping without fluctuation" that violates the Baym-Kadanoff
    (fluctuation-dissipation) balance and, under a thermal bias, injects a
    spurious energy current (the finite-eta lead-balance floor). When True, a
    matching fluctuation ``Sigma_probe^{<,>} = i*Gamma_eta*(n_p + 0/1)`` is
    added to the device source, with the per-DOF, per-omega occupation
    ``n_p = G^< / (G^> - G^<)`` updated self-consistently each SCBA iteration so
    the LOCAL probe current vanishes at every energy. This restores exact
    energy-current conservation at finite eta -- but it injects elastic
    DEPHASING (it is physics, not a pure regularizer; Miao et al., APL 108,
    113107 (2016); Roy & Dhar, PRB 75, 195110 (2007)). For the pure
    coherent+anharmonic conductance use eta->0 extrapolation instead; use the
    probe when an incoherent channel is physically intended. Single-block
    (Gamma-only / coupled-q with block_comm_size==1) only."""

    bubble_balance_check: bool = True
    """Per-iteration Phi-derivable energy-balance diagnostic of the 3-phonon
    bubble: P_in = sum hbar*w*Tr[Sigma^< G^>] must equal P_out (conserving
    identity). Requires keeping the Green's-function data through the
    nnz->stack back-transpose (free at stack=1; one extra all-to-all per
    iteration at stack>1)."""

    sse_cutoff_zero_g: bool = False
    """With ``sse_low_freq_cutoff_thz`` > 0: additionally zero the lead
    occupancies below the cutoff, killing G^< (and hence ALL transport)
    there -- the hard "totally zeroed" treatment. Default False = the
    masked treatment (transport below the cutoff stays ballistic)."""

    sse_vertex_scale: PositiveFloat = 1.0
    """Uniform 3-phonon vertex scale lambda (Sigma scales as lambda^2).
    lambda < 1 = reduced-coupling runs for LOA-style extrapolation and for
    soft-mode structures whose full-coupling bubble-only SCBA is unstable
    (cf. the dense reference's ``vertex_scale``; d5a F10 used 0.3)."""

    low_freq_mixing_thz: NonNegativeFloat = 0.0
    """Frequency-dependent SCBA mixing: self-energy bins with |omega| < this
    (THz) are mixed with ``low_freq_mixing_factor`` instead of the global
    ``scba.mixing_factor``. 0 = off (uniform mixing). This DAMPS the IR
    (Bose-divergent, n(omega)~kT/hbar.omega) marginal mode at the lowest
    frequency bins -- which limit-cycles the eta=0 SCBA Sigma^R -- WITHOUT
    removing the low-omega anharmonic scattering (unlike
    ``sse_low_freq_cutoff_thz``), so the iteration converges to the CORRECT
    conductance. The IR mode sits on the unit circle (|lambda|~1); a small
    mixing factor pulls it inside (|1+a(lambda-1)|<1)."""
    low_freq_mixing_factor: NonNegativeFloat = 0.02
    """Gentle SCBA mixing factor applied to the |omega| < ``low_freq_mixing_thz``
    bins (see there). Small (~0.01-0.03) to damp the IR marginal mode; the rest
    of the spectrum keeps ``scba.mixing_factor``."""

    sse_ring_threads: NonNegativeInt = 0
    """Width of the omega/tau ring-contraction thread pool (bit-identical
    results for any width). 0 = keep the QUATREX_PHPH_RING_THREADS env
    default (1)."""

    sse_ring_min_w: PositiveInt | None = None
    """Minimum omega/tau batch per pool split; None = keep the
    QUATREX_PHPH_RING_MIN_W env default (48)."""

    sse_ramp_iterations: NonNegativeInt = 0
    """Adiabatic switch-on of the 3-phonon bubble: scale the scattering
    self-energy by ``min(1, it/N)`` over the first N SCBA iterations
    (0 = off). Stabilises soft-mode structures whose full-coupling SCBA
    overshoots into unphysical gain states under plain damped iteration."""

    heat_flow_conservation_tol: PositiveFloat = 1e-2
    """Convergence tolerance for the anharmonic phonon SCBA: the relative
    spread of the (hbar-omega-weighted) Meir-Wingreen HEAT current across
    the device interfaces. This is the physically-correct criterion
    (Guo-Bescond-Zhang 2020 / Luisier): the 3-phonon processes do NOT
    conserve phonon NUMBER (1<->2 splitting/merging), so only the ENERGY
    current is conserved, and SCBA convergence means the heat flow is the
    same across all interfaces (~1%). The Sigma residual oscillates on
    soft-mode structures and must NOT be used. The most-conserved (best)
    iterate's heat current is captured even if the iteration later drifts."""

    sigma_convergence_tol: PositiveFloat = 1e-3
    """Relative Sigma^R residual tolerance for the anharmonic phonon SCBA
    fixed point, applied IN ADDITION to ``heat_flow_conservation_tol``.

    Default 1e-3 (0.1%): 1e-2 is too loose to call a fixed point. Linear mixing
    contracts the residual geometrically (~x0.6 / 5 iters), so 1e-3 is reached
    in ~55 iters and 1e-4 in ~75.

    Convergence requires a GENUINE fixed point, not a transient: the scattering
    self-energy must be self-consistent -- the relative residual
    ``||Sigma_new - Sigma_old||_inf / ||Sigma||_inf`` below this -- AND the heat
    flow conserved. Heat-flow conservation alone is necessary but not
    sufficient: at large broadening ``eta`` the heat flow conserves (its
    elastic part dominates) before Sigma reaches self-consistency, so a
    heat-flow-only stop accepts an under-scattered, non-converged Sigma. If
    Sigma oscillates (a limit cycle) the run does NOT converge and needs a
    continuation strategy (vertex-scale warm starts / annealing) -- we report
    it as non-converged rather than passing off the best-conserved transient
    as the answer."""

    bubble_balance_tol: NonNegativeFloat = 0.0
    """Optional third convergence gate on the Phi-derivable bubble energy
    balance ``|P_in - P_out| / |P_in|`` (requires ``bubble_balance_check``).
    0 disables (legacy: residual + lead heat balance only)."""

    scp_tadpole: bool = False
    """Optional self-consistent-phonon (SCP) cubic tadpole static
    self-energy (``model == "negf"``).

    The dynamic 3-phonon bubble destabilises the SCBA on soft-mode
    structures (the Bose-enhanced ``G^<`` IR singularity). The cubic
    tadpole ``Sigma_T = Phi3 : <u>`` is a STATIC real self-energy that
    *stiffens* the soft mode (raises its frequency) -- the physically
    correct finite-T renormalisation (SCP / SSCHA; Paulatto-Errea-Calandra
    2015) -- so the bubble becomes stable. It is recomputed every SCBA
    iteration from the current device ``G^<`` (hence self-consistent and
    self-limiting) and added to the dynamical matrix via ``Sigma^R``.
    Needs only FC3 (the quartic loop, which would need FC4, is omitted).
    Default OFF. Cf. ``quatrex/phonon/static_self_energy.py``.

    NOTE: the current implementation assembles dense device-level arrays
    (FC3 tensor + Phi_eff eigensolve); intended for single / few-cell
    devices (the soft-mode regime). Large multi-cell / distributed use
    needs the band-sparse variant (Phi_eff solve via the RGF)."""

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
    def check_regularizer_exclusivity(self):
        """Guard the mutually-exclusive eta=0 IR / mask regularizers: each
        treats the same physics differently, so stacking them double-counts
        and is almost always a config error."""
        if self.sse_ir_subtraction and self.ir_taper_cells > 0.0:
            raise ValueError(
                "sse_ir_subtraction and ir_taper_cells are mutually-exclusive "
                "IR occupation treatments: set ir_taper_cells = 0 when using "
                "the exact Bose subtraction."
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
