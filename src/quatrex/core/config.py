# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import os
import re
import subprocess
import tomllib
import warnings
from math import isclose
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

profiler = Profiler()


class SCSPConfig(BaseModel):
    """Options for the self-consistent Schrödinger-Poisson loop."""

    model_config = ConfigDict(extra="forbid")

    min_iterations: PositiveInt = 1
    max_iterations: PositiveInt = 100
    convergence_tol: PositiveFloat = 1e-5

    mixing_factor: PositiveFloat = Field(default=0.1, le=1.0)


class QTBMConfig(BaseModel):
    """Options for the quantum transmitting boundary method (QTBM)."""

    model_config = ConfigDict(extra="forbid")

    # The maximum number of energies per batch.
    max_batch_size: PositiveInt = 10


class SCBAConfig(BaseModel):
    """Options for the self-consistent Born approximation."""

    model_config = ConfigDict(extra="forbid")

    min_iterations: PositiveInt = 1
    max_iterations: PositiveInt = 100
    convergence_tol: PositiveFloat = 1e-5

    mixing_factor: PositiveFloat = Field(default=0.1, le=1.0)

    mixing_method: Literal["linear", "anderson"] = "linear"
    """Self-energy fixed-point mixer. ``"linear"`` (default) is the bare
    damped mixing ``Sigma <- (1-a) Sigma_prev + a Sigma_new``. ``"anderson"``
    is safeguarded Anderson/Pulay (DIIS) acceleration that uses a short
    residual history to extrapolate the fixed point; it accelerates a
    WELL-CONDITIONED SCBA and, with the revert-to-best safeguard, is not
    worse than damped linear. ``mixing_factor`` is the Anderson relaxation
    ``beta``. Cf. ``quatrex/core/anderson.py``.

    NOTE: Anderson is NOT a cure for a marginal / non-existent fixed point.
    On the strong-coupling soft-mode SCBA (e.g. the d5a SiNW at full
    coupling, whose omega->0 modes make the fixed point ill-defined) no
    mixer converges -- that regime needs the LOA-Pade analytic continuation
    from moderate coupling (a separate technique), not acceleration. There
    Anderson reverts to its best iterate (>= linear, but not convergent)."""

    anderson_depth: PositiveInt = 8
    """History depth (number of residual secants) for ``mixing_method =
    "anderson"``. Larger resolves more of the oscillation but costs more
    least-squares conditioning."""

    anderson_revert: NonNegativeFloat = 2.0
    """Revert-to-best safeguard for Anderson. If the residual exceeds
    ``anderson_revert`` x the best seen, the (poisoned) history is cleared
    and the iterate reverted to the best one -- the "never worse than
    linear" guarantee. Set to 0 to DISABLE (keep-history, aggressive: needed
    to break a strong oscillation, but then not guaranteed monotone). The
    dense reference converges the soft-mode CNT/SiNW with 0."""

    anderson_step_cap: PositiveFloat = 3.0
    """Max ratio of the Anderson extrapolation step to the damped-linear
    step; an over-long step is replaced by the linear one for that
    iteration (overshoot safeguard). Looser (larger) helps a strongly
    oscillating fixed point but risks instability."""

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


class PoissonConfig(BaseModel):
    """Options for the Poisson solver."""

    model_config = ConfigDict(extra="forbid")

    model: Literal["point-charge", "orbital"] = "point-charge"
    max_iterations: PositiveInt = 100
    convergence_tol: PositiveFloat = 1e-5
    mixing_factor: PositiveFloat = Field(default=0.1, le=1.0)

    rho_shift: NonNegativeFloat = 1e-8
    cg_tol: PositiveFloat = 1e-5
    cg_max_iter: PositiveInt = 100

    num_orbitals_per_atom: dict[str, int] = Field(default_factory=dict)


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

    direct_solver: Literal["superlu", "mumps", "cudss"] = "superlu"

    @model_validator(mode="after")
    def set_compute_current(self) -> Self:
        """Sets the `compute_current` parameter based on the algorithm."""
        if self.compute_current is None:
            if self.algorithm == "rgf":
                self.compute_current = True
            else:
                self.compute_current = False

        # Both "rgf" and "inv" support the Meir-Wingreen boundary current.
        # ("inv" is the small-eta-stable path: its dense inverse pivots
        # through the near-singular Dyson matrix that NaNs the RGF recursion.)

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

class ElectronConfig(BaseModel):
    """Options for the electronic subsystem solver."""

    model_config = ConfigDict(extra="forbid")

    solver: SolverConfig = SolverConfig()
    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()

    eta_obc: NonNegativeFloat = 0  # eV
    eta: NonNegativeFloat = 1e-12  # eV

    fermi_level: float | None = None
    conduction_band_edge: float | None = None
    valence_band_edge: float | None = None

    left_fermi_level: float | None = None
    right_fermi_level: float | None = None

    band_edge_tracking: Literal["dos-peaks", "eigenvalues"] | None = None

    temperature: PositiveFloat = 300.0  # K

    left_temperature: PositiveFloat | None = None
    right_temperature: PositiveFloat | None = None

    energy_window_min: float | None = None
    energy_window_max: float | None = None
    energy_window_num: PositiveInt | None = None
    energy_window_num_per_rank: PositiveInt | None = None

    flatband: bool | None = None

    dos_peak_limit: PositiveFloat = 100.0

    filtering_iteration_limit: PositiveInt = 1

    @model_validator(mode="after")
    def set_left_right_fermi_levels(self) -> Self:
        """Sets the left and right Fermi levels if not already set."""
        if (self.left_fermi_level is None) != (self.right_fermi_level is None):
            warnings.warn(
                "Either both left and right Fermi levels must be set or neither."
            )

        if self.left_fermi_level is None and self.right_fermi_level is None:
            if self.fermi_level is None:
                warnings.warn("Fermi level must be set.")

            self.left_fermi_level = self.fermi_level
            self.right_fermi_level = self.fermi_level

        return self

    @model_validator(mode="after")
    def set_left_right_temperatures(self) -> Self:
        """Sets the left and right temperatures if not already set."""
        if (self.left_temperature is None) != (self.right_temperature is None):
            raise ValueError(
                "Either both left and right temperatures must be set or neither."
            )

        if self.left_temperature is None and self.right_temperature is None:
            self.left_temperature = self.temperature
            self.right_temperature = self.temperature

        return self

    @model_validator(mode="after")
    def set_flatband(self) -> Self:
        """Sets the flatband flags if not already set."""
        if self.left_fermi_level is not None or self.right_fermi_level is not None:
            if self.flatband is None:
                if isclose(self.left_fermi_level, self.right_fermi_level):
                    self.flatband = True
                else:
                    self.flatband = False

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

    eta_obc: NonNegativeFloat = 0  # eV
    eta: NonNegativeFloat = 1e-12  # eV

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

    sigma_convergence_tol: PositiveFloat = 1e-2
    """Relative Sigma^R residual tolerance for the anharmonic phonon SCBA
    fixed point, applied IN ADDITION to ``heat_flow_conservation_tol``.

    Convergence requires a GENUINE fixed point, not a transient: the scattering
    self-energy must be self-consistent -- the relative residual
    ``||Sigma_new - Sigma_old||_inf / ||Sigma||_inf`` below this -- AND the heat
    flow conserved. Heat-flow conservation alone is necessary but not
    sufficient: at large broadening ``eta`` the heat flow conserves (its
    elastic part dominates) before Sigma reaches self-consistency, so a
    heat-flow-only stop accepts an under-scattered, non-converged Sigma. If
    Sigma oscillates (a limit cycle) the run does NOT converge and the mixing
    must be fixed (Anderson / smaller linear) -- we report it as non-converged
    rather than passing off the best-conserved transient as the answer."""

    zero_mode_projection: bool = False
    """Optional rigid-body (q=0) zero-mode projection of the 3-phonon
    self-energy (``model == "negf"``).

    When True, the per-cell rigid modes -- the 3 Cartesian translations
    plus any near-zero rotational quasi-Goldstone (e.g. the axial twist
    of a 1-D wire) -- are projected out of *every* band block of
    ``Sigma^{<,>}`` (two-sided, ``Q Sigma Q``) each iteration. These q=0
    modes carry no heat (they are global rigid-body symmetries), but
    their divergent Bose occupation ``2n+1 ~ 2 kT / (hbar omega)`` as
    ``omega -> 0`` injects an IR singularity into the bubble that
    destabilises the SCBA on soft structures (e.g. the d5a SiNW twist).
    The finite-q heat carriers are orthogonal to the uniform projected
    subspace and are untouched, so transport physics is preserved.
    Default OFF. Cf. ``phonon/solver/zero_modes.py`` and
    ``build_cell_zero_mode_projector``."""

    zero_mode_floor_thz: NonNegativeFloat = 0.1  # THz
    """Absolute frequency floor for ``zero_mode_projection``: cell modes
    with frequency below this (i.e. eigenvalue of the cell Gamma-matrix
    below ``zero_mode_floor_thz**2``) are treated as rigid and projected
    out. The floor is ABSOLUTE (not relative to the stiffest mode) so a
    high-frequency mode like a Si-H stretch cannot inflate the cutoff and
    over-project real low-omega heat carriers. Every projected mode's
    frequency is logged at solver init."""

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


class ContactConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fermi_level: float
    name: str
    type: Literal["ohmic"] = "ohmic"
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lattice_vectors: list[list[float]] = Field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    direction: Literal["a", "b", "c"]

    @model_validator(mode="after")
    def to_array(self) -> Self:
        """Transforms origin and size to arrays."""
        self.origin = np.array(self.origin, dtype=float)
        self.lattice_vectors = np.array(self.lattice_vectors, dtype=float)
        return self


class DeviceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    construct_from_unit_cell: bool = False

    # --- Device geometry ---------------------------------------------
    neighbor_cell_cutoff: (
        tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt] | None
    ) = None
    """The number of neighbor cells to consider along each lattice direction.

    !!! note

        Currently, this parameter is only used if
        `construct_from_unit_cell` is `True`.

    If set to `None`, all neighbor cells are considered. A
    `neighbor_cell_cutoff` of zero means that only the unit cell itself
    is considered. Along the transport direction, at least one
    neighboring cell must be included.

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

    orthogonal_basis: bool = True
    """Whether the basis set is orthogonal.

    This affects how the overlap matrix is handled.
    In the case of `True`, the overlap matrix is identity.

    !!! warning

        Currently, `False` is not supported since
        the code does not correctly handle overlap matrices in the case
        of kpoints.

    """

    @model_validator(mode="after")
    def to_tuple(self) -> Self:
        """Transforms list to tuple."""
        if self.neighbor_cell_cutoff is not None:
            self.neighbor_cell_cutoff = tuple(self.neighbor_cell_cutoff)
        self.kpoint_grid = tuple(self.kpoint_grid)
        return self

    @model_validator(mode="after")
    def check_connecting_cells(self) -> Self:
        """Checks that num_connecting_cells is not zero in transport direction."""
        if not self.construct_from_unit_cell:
            return self

        if self.neighbor_cell_cutoff is None:
            return self

        ind = "xyz".index(self.transport_direction)
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

    stack_all_to_all: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    stack_all_gather: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    stack_all_reduce: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    stack_bcast: Literal["host_mpi", "device_mpi", "nccl"] | None = None

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
    simulation_type: Literal["electron", "phonon"] = "electron"
    """The transport formalism to use.

    There are two supported formalisms:

    - "wf": Wavefunction formalism
    - "negf": Non-equilibrium Green's function formalism

    !!! warning "Input formats"

        Currently, the input formats for the two formalisms are not
        consistent.

    """
    scsp: SCSPConfig = SCSPConfig()
    scba: SCBAConfig = SCBAConfig()
    qtbm: QTBMConfig = QTBMConfig()
    poisson: PoissonConfig = PoissonConfig()

    electron: ElectronConfig  | None = None
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
    }

    stack_comm_config = {
        "all_to_all": comm_config.stack_all_to_all or default_backend,
        "all_gather": comm_config.stack_all_gather or default_backend,
        "all_reduce": comm_config.stack_all_reduce or default_backend,
        "bcast": comm_config.stack_bcast or default_backend,
    }

    q_comm_config = {
        "all_to_all": comm_config.q_all_to_all or default_backend,
        "all_gather": comm_config.q_all_gather or default_backend,
        "all_reduce": comm_config.q_all_reduce or default_backend,
        "bcast": comm_config.q_bcast or default_backend,
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
