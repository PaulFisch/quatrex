# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import os
from dataclasses import dataclass, field

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_any_location, get_host
from qttools.utils.mpi_utils import distributed_load, get_section_sizes
from quatrex.core.config import QuatrexConfig
from quatrex.core.interaction import (
    CoulombScreeningInteraction,
    Interaction,
    PhononPhononInteraction,
    PseudoScatteringPhononInteraction,
    build_interactions,
)
from quatrex.core.observables import current_conservation, density, device_current
from quatrex.core.transport import TransportSolver
from quatrex.core.utils import compute_num_connected_blocks, compute_sparsity_pattern
from quatrex.coulomb_screening import CoulombScreeningSolver, PCoulombScreening
from quatrex.device import Device
from quatrex.device.inputs import (
    assemble_matrix,
    create_coordinate_grid,
    distributed_read_xyz,
    get_block_sizes,
    load_matrix,
)
from quatrex.electron import (
    ElectronSolver,
    SigmaPhoton,
)
from quatrex.grid import get_electron_energies
from quatrex.phonon import PhononSolver
from quatrex.photon import PhotonSolver, PiPhoton

profiler = Profiler()


class SCBAData:
    """Data container class for the SCBA.

    Parameters
    ----------
    config : QuatrexConfig
        The Quatrex configuration.

    """

    def __init__(self, config: QuatrexConfig, electron_energies: NDArray) -> None:
        """Initializes the SCBA data."""
        # Load orbital positions, energy vector and block-sizes.

        grid, __, atomic_species = Device.load_structure(config)
        self.orbitals_per_atom = [
            config.device.num_orbitals_per_atom.get(s, 1) for s in atomic_species
        ]
        # Kept so the phonon SSE can weight the interaction cutoff by the
        # transport-axis separation (phonon.interaction_cutoff_taper); this is
        # the same `grid` compute_sparsity_pattern masks with, so the taper and
        # the stored pattern are guaranteed to agree.
        self.grid = grid

        block_sizes = get_block_sizes(config, grid)

        kpoint_grid = config.device.kpoint_grid
        # Find the maximum interaction cutoff.
        max_interaction_cutoff = 0.0
        if config.scba.coulomb_screening:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                config.coulomb_screening.interaction_cutoff,
            )
        if config.scba.photon:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                config.photon.interaction_cutoff,
            )
        if config.scba.phonon:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                config.phonon.interaction_cutoff,
            )
        if max_interaction_cutoff == 0.0:
            raise NotImplementedError(
                "At least one interaction must be enabled in the SCBA."
                "Ballistic transport is not properly supported yet."
            )

        if comm.rank == 0:
            print(f"Max Interaction Cutoff: {max_interaction_cutoff}", flush=True)

        with profiler.profile_range(
            label="SCBA: Sparsity Pattern", level="default", comm=comm
        ):
            # Determine the local slice of the data.
            # NOTE: This is arrow-wise partitioning.
            # TODO: Allow more options, e.g., block row-wise partitioning.
            section_sizes, __ = get_section_sizes(len(block_sizes), comm.block.size)
            section_offsets = np.hstack(([0], np.cumsum(section_sizes)))
            block_offsets = np.hstack(([0], np.cumsum(block_sizes)))
            start_idx = block_offsets[section_offsets[comm.block.rank]]
            end_idx = block_offsets[section_offsets[comm.block.rank + 1]]

            self.sparsity_pattern = compute_sparsity_pattern(
                grid,
                max_interaction_cutoff,
                transport_direction=config.device.transport_direction,
                start_idx=start_idx,
                end_idx=end_idx,
            )

            # sse_g_band > 1: the bubble contraction keeps inner G links
            # beyond the tridiagonal band, so the solver must produce (and
            # the shared pattern must hold) the second off-diagonal blocks.
            # Sigma's extra blocks stay structurally zero (the SSE output
            # band is unchanged).
            g_band = 1
            if getattr(config, "simulation_type", "") == "phonon":
                g_band = min(
                    int(getattr(config.phonon, "sse_g_band", 1) or 1),
                    len(block_sizes) - 1,
                )
            if g_band > 1:
                from qttools import sparse as _sparse

                # Distributed: emit the extra band blocks only within
                # this rank's arrow-wise window (min block index inside
                # the rank's section), matching compute_sparsity_pattern
                # above. At block.size == 1 the window spans all blocks
                # and this reduces to the legacy full loop.
                sec_lo = int(section_offsets[comm.block.rank])
                sec_hi = int(section_offsets[comm.block.rank + 1])
                block_offsets_ = np.hstack(([0], np.cumsum(block_sizes)))
                rows_, cols_ = [], []
                for bi in range(len(block_sizes)):
                    for bj in range(len(block_sizes)):
                        if not (1 < abs(bi - bj) <= g_band):
                            continue
                        if not (sec_lo <= min(bi, bj) < sec_hi):
                            continue
                        r = np.arange(block_offsets_[bi],
                                      block_offsets_[bi + 1])
                        c = np.arange(block_offsets_[bj],
                                      block_offsets_[bj + 1])
                        rr, cc = np.meshgrid(r, c, indexing="ij")
                        rows_.append(rr.ravel())
                        cols_.append(cc.ravel())
                if rows_:
                    n_ = int(block_offsets_[-1])
                    # backend-native arrays: qttools sparse is cupyx
                    # under cupy, which rejects numpy index arrays
                    band_pattern = _sparse.coo_matrix(
                        (
                            xp.ones(sum(len(r) for r in rows_)),
                            (xp.asarray(np.concatenate(rows_)),
                             xp.asarray(np.concatenate(cols_))),
                        ),
                        shape=(n_, n_),
                    )
                    ext = (
                        self.sparsity_pattern.astype(np.float64)
                        + band_pattern
                    ).tocoo()
                    ext.data[:] = 1.0
                    self.sparsity_pattern = ext

        dsdbsparse_type = config.compute.dsdbsparse_type

        def _zeros_like(dsdbsparse: DSDBSparse) -> DSDBSparse:
            # These buffers live for the whole SCBA (no allocate/free
            # choreography) and must start zeroed (Sigma = 0 at iteration 0).
            out = dsdbsparse_type.empty_like(dsdbsparse)
            out.allocate_data()
            out.data[:] = 0.0
            return out

        # G^R is only needed between the solve and the observables, so it
        # follows the allocate/free choreography in SCBA.run() -- it must not
        # be resident while the interactions allocate their own buffers.
        self.g_retarded = dsdbsparse_type.from_sparray(
            sparray=self.sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=electron_energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
            allocate=False,
        )

        self.g_lesser = dsdbsparse_type.from_sparray(
            sparray=self.sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=electron_energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
            symmetry="skew-hermitian" if config.scba.symmetric else None,
        )
        self.g_lesser.data[:] = 0.0
        self.g_greater = _zeros_like(self.g_lesser)

        self.sigma_lesser_prev = _zeros_like(self.g_lesser)
        self.sigma_lesser = _zeros_like(self.g_lesser)
        self.sigma_greater_prev = _zeros_like(self.g_lesser)
        self.sigma_greater = _zeros_like(self.g_lesser)

        self.sigma_retarded_hermitian_prev = _zeros_like(self.g_lesser)
        self.sigma_retarded_hermitian = _zeros_like(self.g_lesser)
        if config.scba.symmetric:
            self.sigma_retarded_hermitian.symmetry = "hermitian"
            self.sigma_retarded_hermitian_prev.symmetry = "hermitian"

        if config.scba.coulomb_screening:
            # NOTE: The polarization has the same sparsity pattern as
            # the electronic system (the interactions are local in real
            # space). However, we need to change the block sizes of the
            # screened Coulomb interaction. These buffers follow the
            # allocate/free choreography in the interaction compute.
            self.p_retarded_hermitian = dsdbsparse_type.empty_like(self.g_lesser)
            self.p_lesser = dsdbsparse_type.empty_like(self.g_lesser)
            self.p_greater = dsdbsparse_type.empty_like(self.g_lesser)

            if config.scba.symmetric:
                self.p_retarded_hermitian.symmetry = "hermitian"

            num_connected_blocks = config.coulomb_screening.num_connected_blocks
            if num_connected_blocks == "auto":
                num_connected_blocks = compute_num_connected_blocks(
                    self.sparsity_pattern, block_sizes
                )

            if comm.rank == 0:
                print(f"Number of connected blocks: {num_connected_blocks}", flush=True)

            # TODO: This only works for constant block sizes.
            coulomb_screening_block_sizes = (
                block_sizes[: len(block_sizes) // num_connected_blocks]
                * num_connected_blocks
            )

            self.w_lesser = dsdbsparse_type.from_sparray(
                sparray=self.sparsity_pattern.astype(xp.complex128),
                block_sizes=coulomb_screening_block_sizes,
                global_stack_shape=electron_energies.shape
                + tuple([k for k in kpoint_grid if k > 1]),
                symmetry="skew-hermitian" if config.scba.symmetric else None,
                allocate=False,
            )
            self.w_greater = dsdbsparse_type.empty_like(self.w_lesser)

        # TODO: The interactions with photons and phonons are not yet
        # implemented.
        if config.scba.photon:
            raise NotImplementedError


@dataclass
class Observables:
    """Observable quantities for the SCBA."""

    # --- Electrons ----------------------------------------------------
    electron_ldos: NDArray = None
    electron_density: NDArray = None
    hole_density: NDArray = None
    electron_current: dict = field(default_factory=dict)

    valence_band_edges: NDArray = None
    conduction_band_edges: NDArray = None

    excess_charge_density: NDArray = None

    electron_electron_scattering_rate: NDArray = None
    electron_photon_scattering_rate: NDArray = None
    electron_phonon_scattering_rate: NDArray = None

    sigma_lesser_density: NDArray = None
    sigma_greater_density: NDArray = None

    # --- Coulomb screening --------------------------------------------
    w_lesser_density: NDArray = None
    w_greater_density: NDArray = None

    p_lesser_density: NDArray = None
    p_greater_density: NDArray = None

    # --- Photons ------------------------------------------------------
    pi_photon_retarded_density: NDArray = None
    pi_photon_lesser_density: NDArray = None
    pi_photon_greater_density: NDArray = None

    d_photon_retarded_density: NDArray = None
    d_photon_lesser_density: NDArray = None
    d_photon_greater_density: NDArray = None

    photon_current_density: NDArray = None

    # --- Phonons ------------------------------------------------------
    pi_phonon_retarded_density: NDArray = None
    pi_phonon_lesser_density: NDArray = None
    pi_phonon_greater_density: NDArray = None
    d_phonon_retarded_density: NDArray = None
    d_phonon_lesser_density: NDArray = None
    d_phonon_greater_density: NDArray = None

    thermal_current: NDArray = None


class SCBA(TransportSolver):
    """Self-consistent Born approximation (SCBA) solver.

    The SCBA loop uses two registries:
    * self.subsystems: the physical Green's-function subsystems
      (currently electron or phonon)
    * self.interactions: a list of SSEs (Coulomb screening, phonon-phonon,
      ...). Each interaction reads from one or more subsystems' Green's
      functions and writes its contribution

    Parameters
    ----------
    config : QuatrexConfig
        Quatrex configuration object.

    """

    def __init__(self, config: QuatrexConfig) -> None:
        """Initializes an SCBA instance."""
        self.config = config

        self.observables = Observables()
        electron_energies = xp.zeros((comm.size,))
        self.data = SCBAData(config, electron_energies=electron_energies)  # dummy data
        self.mixing_factor = self.config.scba.mixing_factor

        # Optional quasi-Newton mixing. Damped/Anderson mixing accelerates a
        # contractive iteration; broyden/rpm/rre/jfnk LAND a fixed point whose
        # Jacobian has |lambda| > 1, which mixing cannot reach. Anderson with a
        # linear warm-up is built later, once the iterate is past the cold
        # start (see _update_sigma).
        self._anderson_mixer = None
        if self.config.scba.mixing_method != "linear" and not (
            self.config.scba.mixing_method == "anderson"
            and self.config.scba.anderson_warmup_iters > 0
        ):
            self._anderson_mixer = self._build_mixer()

        # Anharmonic-phonon convergence by HEAT-FLOW conservation. The
        # reported result is the CONVERGED fixed point (last iterate with
        # converged == True); a run that does not converge is reported as
        # such, with its last iterate. No "best-conserved iterate" is
        # tracked: over a non-converged trajectory the most-conserved
        # iterate is not a fixed point (it is typically an early, still
        # near-ballistic step, which conserves trivially because Sigma has
        # not developed), so headlining it misrepresents a non-result as
        # the answer.
        self._scba_iteration = 0
        self._last_heat_current = None
        self._diverged = False

        # ----- Particles ----------------------------------------------
        self.energies = get_electron_energies(config)

        # File-based (possibly NON-UNIFORM) phonon frequency grid: with
        # phonon.frequency_grid = "file" the grid is read verbatim from
        # phonon_energies.npy and overrides the electron energy window.
        # The bubble then needs the auxiliary uniform grid
        # (phonon.sse_aux_grid_dw_thz > 0) unless the file is uniform.
        if (
            config.simulation_type == "phonon"
            and getattr(config.phonon, "frequency_grid", "window") == "file"
        ):
            energies_path = self.config.input_dir / "phonon_energies.npy"
            grid = np.asarray(get_host(distributed_load(energies_path)))
            if grid.ndim != 1 or grid.size < 2:
                raise ValueError(
                    f"phonon_energies.npy ({energies_path}) must hold a 1D "
                    "frequency grid with at least 2 points."
                )
            if float(grid[0]) < 0.0 or np.any(np.diff(grid) <= 0.0):
                raise ValueError(
                    "phonon.frequency_grid='file' requires a strictly "
                    "ascending, non-negative grid in phonon_energies.npy."
                )
            # Fail fast (before a full Dyson sweep) on the combination the
            # SSE will reject anyway: a non-uniform grid needs the
            # auxiliary bubble grid.
            if (
                config.scba.phonon
                and config.phonon.model == "negf"
                and float(getattr(config.phonon,
                                  "sse_aux_grid_dw_thz", 0.0) or 0.0) <= 0.0
            ):
                df = float(grid[1] - grid[0])
                if abs(float(grid[0])) > 1e-9 * df or bool(
                    np.max(np.abs(np.diff(grid) - df)) > 1e-9 * df
                ):
                    raise ValueError(
                        "phonon_energies.npy holds a non-uniform (or "
                        "non-zero-anchored) grid, but the bubble FFT runs "
                        "on the primary grid (sse_aux_grid_dw_thz = 0). "
                        "Set phonon.sse_aux_grid_dw_thz > 0."
                    )
            self.energies = xp.asarray(grid)
            if comm.rank == 0:
                spacings = np.diff(grid)
                print(
                    f"Phonon frequency grid from file: {grid.size} pts on "
                    f"[{grid[0]:.4g}, {grid[-1]:.4g}] THz, spacing "
                    f"[{spacings.min():.4g}, {spacings.max():.4g}] THz.",
                    flush=True,
                )

        min_energy = self.energies[0]
        max_energy = self.energies[-1]
        num_energies = len(self.energies)
        energy_resolution = self.energies[1] - self.energies[0]
        num_energies_per_rank = num_energies // comm.stack.size
        if comm.rank == 0:
            print(
                f"Energy window: {min_energy} to {max_energy} eV with {num_energies} grid points.",
                flush=True,
            )
            print(f"Resolution is {energy_resolution:.6f} eV.", flush=True)
            print(
                f"comm.stack size: {comm.stack.size}, comm.block size: {comm.block.size}",
                flush=True,
            )
            print(
                f"Each comm.block has {num_energies_per_rank} grid points.", flush=True
            )

        # ----- Subsystem solvers ----------------------------------------
        self.subsystems: dict[str, object] = {}
        if config.simulation_type == "electron":
            self.electron_solver = ElectronSolver(
                self.config,
                self.energies,
                sparsity_pattern=self.data.sparsity_pattern,
            )
            self.subsystems["electron"] = self.electron_solver
        elif config.simulation_type == "phonon":
            self.phonon_solver = PhononSolver(
                self.config,
                self.energies,
                sparsity_pattern=self.data.sparsity_pattern,
            )
            self.subsystems["phonon"] = self.phonon_solver
        else:
            raise NotImplementedError(
                f"simulation_type={config.simulation_type!r} not yet "
                "supported."
            )

        # ----- Interactions  --------------------------------------------
        self._coulomb_screening_interaction: Interaction | None = None
        self._phonon_phonon_interaction: Interaction | None = None
        self._pseudo_scattering_phonon_interaction: Interaction | None = None

        if self.config.scba.coulomb_screening:
            # Load the Coulomb matrix.
            coulomb_matrix, __ = assemble_matrix(
                config=config,
                matrix_name="coulomb_matrix",
                sparsity_pattern=self.data.sparsity_pattern,
                shift_kpoints=True,
            )

            # Make sure the Coulomb matrix is hermitian.
            # TODO: Check that this is correct for kpoints.
            if not coulomb_matrix.symmetry:
                coulomb_matrix.symmetrize()
            coulomb_matrix._data /= config.coulomb_screening.epsilon_r

            energies_path = self.config.input_dir / "coulomb_screening_energies.npy"
            if os.path.isfile(energies_path):
                self.coulomb_screening_energies = distributed_load(energies_path)
            else:
                self.coulomb_screening_energies = (
                    self.energies - self.energies[0]
                )
                # Remove the zero energy to avoid division by zero.
                self.coulomb_screening_energies += 1e-6

            self._coulomb_screening_interaction = CoulombScreeningInteraction(
                config=self.config,
                electron_energies=self.energies,
                coulomb_screening_energies=self.coulomb_screening_energies,
                coulomb_matrix=coulomb_matrix,
                sparsity_pattern=self.data.sparsity_pattern,
            )

        if self.config.scba.photon:
            energies_path = self.config.input_dir / "photon_energies.npy"
            self.photon_energies = distributed_load(energies_path)
            self.pi_photon = PiPhoton(...)
            self.photon_solver = PhotonSolver(self.config, self.photon_energies)
            self.sigma_photon = SigmaPhoton(...)

        if self.config.scba.phonon:
            if self.config.phonon.model == "negf":
                energies_path = self.config.input_dir / "phonon_energies.npy"
                self.phonon_energies = distributed_load(energies_path)
                # The SSE MUST live on the grid the Green's functions live on
                # (the solver grid): its bubble prefactor and the SCP <uu>
                # carry the grid spacing d-omega. A stale phonon_energies.npy
                # that disagrees with the configured window would silently
                # misscale Sigma, so pass the solver grid and only warn
                # about the mismatch.
                solver_freqs = np.asarray(
                    get_host(self.phonon_solver.local_frequencies))
                npy_freqs = np.asarray(get_host(self.phonon_energies))
                # Compare against the GLOBAL configured window -- NOT the
                # rank-local slice (len(global)/stack points). With
                # frequency_grid = "file" the npy IS the solver grid, so
                # there is nothing to reconcile.
                if getattr(config.phonon, "frequency_grid",
                           "window") == "file":
                    global_freqs = np.asarray(get_host(self.energies))
                else:
                    el = self.config.electron
                    global_freqs = np.linspace(
                        float(el.energy_window_min),
                        float(el.energy_window_max),
                        int(el.energy_window_num),
                    )
                if npy_freqs.shape != global_freqs.shape or not np.allclose(
                    npy_freqs, global_freqs
                ):
                    if comm.rank == 0:
                        print(
                            "WARNING: phonon_energies.npy "
                            f"({npy_freqs.shape[0]} pts, dw="
                            f"{float(npy_freqs[1] - npy_freqs[0]):.4g}) does "
                            "not match the configured solver energy grid "
                            f"({global_freqs.shape[0]} pts, dw="
                            f"{float(global_freqs[1] - global_freqs[0]):.4g})"
                            "; using the solver grid for the scattering "
                            "self-energy.",
                            flush=True,
                        )
                self._phonon_phonon_interaction = PhononPhononInteraction(
                    config=self.config,
                    phonon_energies=solver_freqs,
                    block_sizes=self.data.g_lesser.block_sizes,
                    dynamical_matrix=self.phonon_solver.dynamical_matrix,
                    orbital_grid=getattr(self.data, "grid", None),
                )

            elif self.config.phonon.model == "pseudo-scattering":
                self._pseudo_scattering_phonon_interaction = (
                    PseudoScatteringPhononInteraction(
                        config=self.config,
                        electron_energies=self.energies,
                    )
                )

        self.data = SCBAData(
            config, electron_energies=self.energies
        )  # real data

        self.interactions: list[Interaction] = build_interactions(self)

    def _build_mixer(self):
        """Constructs the mixer selected by `scba.mixing_method`."""
        from quatrex.core.anderson import AndersonMixer, RREMixer
        from quatrex.core.broyden import BroydenMixer
        from quatrex.core.jfnk import JFNKMixer
        from quatrex.core.rpm import RPMMixer

        scba = self.config.scba
        xm = scba.experimental_mixer

        if scba.mixing_method == "anderson":
            return AndersonMixer(
                depth=scba.anderson_depth,
                beta=self.mixing_factor,
                period=scba.anderson_period,
                restart=scba.anderson_restart,
                ridge=scba.anderson_ridge,
                step_cap=scba.anderson_step_cap,
                revert_factor=scba.anderson_revert_factor,
                stagnation_restart=scba.anderson_stagnation_restart,
                collect_diagnostics=scba.mixer_diagnostics,
            )
        if scba.mixing_method == "broyden":
            return BroydenMixer(
                depth=scba.anderson_depth,
                beta=self.mixing_factor,
                ridge=xm.broyden_ridge,
                warmup=xm.broyden_warmup_iters,
                trust=xm.broyden_trust,
            )
        if scba.mixing_method == "rpm":
            return RPMMixer(
                max_subspace=xm.rpm_max_subspace,
                beta=self.mixing_factor,
                ridge=xm.broyden_ridge,
                warmup=xm.broyden_warmup_iters,
                trust=xm.broyden_trust,
            )
        if scba.mixing_method == "rre":
            return RREMixer(
                cycle=xm.rre_cycle,
                beta=self.mixing_factor,
                ridge=xm.rre_ridge,
            )
        if scba.mixing_method == "jfnk":
            return JFNKMixer(
                warmup=xm.jfnk_warmup_iters,
                beta=self.mixing_factor,
                max_krylov=xm.jfnk_max_krylov,
                inner_tol=xm.jfnk_inner_tol,
                forcing=xm.jfnk_forcing,
                max_newton=xm.jfnk_max_newton,
                eps=xm.jfnk_eps,
                trust=xm.jfnk_trust,
                trust_max=xm.jfnk_trust_max,
                newton_damp=xm.jfnk_newton_damp,
                ptc=xm.jfnk_ptc,
            )
        if scba.mixing_method == "newton":
            from quatrex.core.newton import NewtonKrylovMixer

            # The JVP context is built lazily on the first Newton step:
            # the mixer is constructed before the phonon solver (and the
            # interactions) exist.
            return NewtonKrylovMixer(
                jvp_factory=self._get_phonon_jvp,
                warmup=xm.newton_warmup_iters,
                switch_tol=xm.newton_switch_tol,
                beta=self.mixing_factor,
                max_krylov=xm.newton_max_krylov,
                inner_tol=xm.newton_inner_tol,
                forcing=xm.newton_forcing,
                max_newton=xm.newton_max_newton,
                trust=xm.newton_trust,
                trust_max=xm.newton_trust_max,
                newton_damp=xm.newton_damp,
                backtrack=xm.newton_backtrack,
                precond=xm.newton_precond,
                precond_rank=xm.newton_precond_rank,
            )

        raise ValueError(f"Unknown mixing method '{scba.mixing_method}'.")

    def _get_phonon_jvp(self):
        """Construct (once) and return the exact-JVP context for the
        ``"newton"`` mixer. Deferred to first use so the phonon solver and
        the phonon-phonon interaction exist."""
        jvp = getattr(self, "_phonon_jvp", None)
        if jvp is None:
            from quatrex.core.phonon_jvp import PhononJVP

            jvp = PhononJVP(
                self,
                recon_check_tol=(
                    self.config.scba.experimental_mixer
                    .newton_recon_check_tol),
                jvp_form=(
                    self.config.scba.experimental_mixer.newton_jvp_form),
            )
            self._phonon_jvp = jvp
        return jvp

    def _stash_sigma(self) -> None:
        """Stash the current into the previous self-energy buffers."""
        self.data.sigma_lesser_prev.data[:] = self.data.sigma_lesser.data
        self.data.sigma_greater_prev.data[:] = self.data.sigma_greater.data
        self.data.sigma_retarded_hermitian_prev.data[:] = (
            self.data.sigma_retarded_hermitian.data
        )

        self.data.sigma_retarded_hermitian.data[:] = 0.0
        self.data.sigma_lesser.data[:] = 0.0
        self.data.sigma_greater.data[:] = 0.0

    @profiler.profile(label="SCBA: Symmetrize Sigma", level="default", comm=comm)
    def _symmetrize_sigma(self) -> None:
        # Symmetrization.
        if not self.config.scba.symmetric:
            self.data.sigma_lesser.symmetrize(xp.subtract)
            self.data.sigma_greater.symmetrize(xp.subtract)
            # Make the self-energy Hermitian
            # This is done before adding the skew hermitian part coming
            # from the lesser and greater self-energies
            self.data.sigma_retarded_hermitian.symmetrize(xp.add)

        if self.config.scba.align_self_energy_to_complex_axes:
            self.data.sigma_lesser._data.real = 0
            self.data.sigma_greater._data.real = 0
            # Make sure that the imaginary part comes only from
            # sigma_greater - sigma_lesser.
            self.data.sigma_retarded_hermitian._data.imag = 0

    @profiler.profile(label="SCBA: Update Sigma", level="default", comm=comm)
    def _update_sigma(self) -> None:
        """Updates the self-energy: damped linear or quasi-Newton mixing."""
        # Linear-warmup -> Anderson hand-off. _scba_iteration is already
        # incremented (in _has_converged) when the first update runs, so the
        # strict ">" gives exactly anderson_warmup_iters linear updates.
        if (self._anderson_mixer is None
                and self.config.scba.mixing_method == "anderson"
                and self._scba_iteration > self.config.scba.anderson_warmup_iters):
            self._anderson_mixer = self._build_mixer()
            if comm.rank == 0:
                print(f"mixer: linear -> Anderson(m={self.config.scba.anderson_depth}, "
                      f"period={self.config.scba.anderson_period}, "
                      f"beta={self.mixing_factor}) at iteration "
                      f"{self._scba_iteration}", flush=True)
        if self._anderson_mixer is not None:
            self._update_sigma_anderson()
            return
        # Frequency-dependent mixing factor: gentle on the IR-divergent low-omega
        # bins (which limit-cycle the eta=0 Sigma^R), normal elsewhere. Scalar
        # when the feature is off.
        a = self._freq_mixing_factor()
        self.data.sigma_lesser.data[:] = (
            (1 - a) * self.data.sigma_lesser_prev.data
            + a * self.data.sigma_lesser.data
        )
        self.data.sigma_greater.data[:] = (
            (1 - a) * self.data.sigma_greater_prev.data
            + a * self.data.sigma_greater.data
        )
        self.data.sigma_retarded_hermitian.data[:] = (
            (1 - a) * self.data.sigma_retarded_hermitian_prev.data
            + a * self.data.sigma_retarded_hermitian.data
        )

    def _freq_mixing_factor(self):
        """Per-omega SCBA mixing factor (cached). Returns the scalar
        ``mixing_factor`` when frequency-dependent mixing is off
        (``phonon.low_freq_mixing_thz <= 0``), else a broadcastable
        ``(n_local_freq, 1, ...)`` array that is ``low_freq_mixing_factor`` on
        the |omega| < cutoff bins and ``mixing_factor`` elsewhere -- damping the
        IR (Bose) marginal mode without removing the low-omega scattering."""
        cached = getattr(self, "_freq_mix", None)
        if cached is not None:
            return cached
        w_c = float(self.config.phonon.low_freq_mixing_thz)
        if w_c <= 0.0 or not hasattr(self, "phonon_solver"):
            self._freq_mix = self.mixing_factor
            return self._freq_mix
        a_low = float(self.config.phonon.low_freq_mixing_factor)
        wloc = xp.abs(xp.asarray(self.phonon_solver.local_frequencies,
                                 dtype=float).real)
        a = xp.where(wloc < w_c, a_low, self.mixing_factor)
        data = self.data.sigma_retarded_hermitian.data
        if a.shape[0] == data.shape[0]:
            a = a.reshape((a.shape[0],) + (1,) * (data.ndim - 1))
        else:  # state/shape mismatch -> safe fallback to uniform mixing
            a = self.mixing_factor
        self._freq_mix = a
        if comm.rank == 0:
            n_low = int(get_host(xp.asarray(wloc < w_c).sum()))
            print(f"freq-dependent mixing: rank0 has {n_low} bins < {w_c} THz "
                  f"at factor {a_low} (rest {self.mixing_factor})", flush=True)
        return self._freq_mix

    def _update_sigma_anderson(self) -> None:
        """Anderson(m) step over the concatenated
        ``[Sigma^<, Sigma^>, Sigma^R]`` state: the iterate is the previous
        (mixed) self-energy, the fixed-point map value is the raw SSE
        output computed from it."""
        bufs = (self.data.sigma_lesser, self.data.sigma_greater,
                self.data.sigma_retarded_hermitian)
        prev = (self.data.sigma_lesser_prev, self.data.sigma_greater_prev,
                self.data.sigma_retarded_hermitian_prev)
        to_host = (
            (lambda a: get_any_location(a, "numpy", use_pinned_memory=True))
            if xp.__name__ == "cupy"
            else np.asarray
        )
        x = np.concatenate([to_host(m.data).ravel() for m in prev])
        gx = np.concatenate([to_host(m.data).ravel() for m in bufs])
        x_new = self._anderson_mixer.step(x, gx)
        offset = 0
        for m in bufs:
            size = m.data.size
            m.data[:] = xp.asarray(x_new[offset:offset + size].reshape(m.data.shape))
            offset += size

    @profiler.profile(label="SCBA: Convergence test", level="default", comm=comm)
    def _has_converged(self) -> bool:
        """Checks if the SCBA has converged."""
        # A JFNK finite-difference probe is x_k + eps*v, not an iterate of the
        # SCBA map: its residual is that of x_k up to O(eps), so testing
        # convergence on it would accept a deliberately perturbed self-energy.
        if self._anderson_mixer is not None and self._anderson_mixer.probing:
            self._scba_iteration += 1
            return False

        # Infinity norm of the self-energy update.
        diff = (
            self.data.sigma_retarded_hermitian.data
            - self.data.sigma_retarded_hermitian_prev.data
        )
        local_max_diff = get_host(xp.max(xp.abs(diff)))
        max_diff = np.empty_like(local_max_diff)
        global_comm.Allreduce(local_max_diff, max_diff, op=MPI.MAX)

        if os.environ.get("QX_DIAG_OMEGA") and "phonon" in self.subsystems:
            self._report_residual_by_frequency(diff)

        current_diff = 0.0
        if "electron" in self.subsystems:
            meir_wingreen_current = self.observables.electron_current.get(
                "meir-wingreen", [0, 0]
            )
            i_left = xp.real(meir_wingreen_current[..., 0])
            i_right = xp.real(meir_wingreen_current[..., -1])

            dE = self.energies[1] - self.energies[0]
            current_diff = xp.abs(xp.sum(i_left) * dE - xp.sum(i_right) * dE)

        current_conservation_abs, current_conservation_rel = current_conservation(
            self.data.g_lesser,
            self.data.g_greater,
            self.data.sigma_lesser,
            self.data.sigma_greater,
        )

        if comm.rank == 0:
            print(f"Maximum Self-Energy Update: {max_diff}", flush=True)
            print(f"Contact Current Difference: {current_diff}", flush=True)
            print(f"Current Conservation abs: {current_conservation_abs}", flush=True)
            print(f"Current Conservation rel: {current_conservation_rel}", flush=True)

        self._scba_iteration += 1

        # Anharmonic phonon convergence requires a GENUINE fixed point: the
        # scattering self-energy must reach self-consistency (relative Sigma
        # residual small) AND the heat flow must be conserved. NB the
        # current_conservation above is the NUMBER-current G-R balance,
        # which 3-phonon processes violate by design -- the physical
        # criterion is the hbar-omega-weighted HEAT flow.
        if (self.config.scba.phonon
                and self.config.phonon.model == "negf"):
            # Relative residual ||Sigma_new - Sigma_old||_inf / ||Sigma||_inf
            # over ALL components (lesser, greater, retarded-Hermitian): the
            # Sigma^R part alone is blind to a common lesser+greater drift.
            local_diff = local_max_diff
            local_smag = get_host(
                xp.max(xp.abs(self.data.sigma_retarded_hermitian.data)))
            for cur, prev in (
                (self.data.sigma_lesser, self.data.sigma_lesser_prev),
                (self.data.sigma_greater, self.data.sigma_greater_prev),
            ):
                local_diff = np.maximum(
                    local_diff, get_host(xp.max(xp.abs(cur.data - prev.data))))
                local_smag = np.maximum(
                    local_smag, get_host(xp.max(xp.abs(cur.data))))
            max_diff_all = np.empty_like(local_diff)
            global_comm.Allreduce(local_diff, max_diff_all, op=MPI.MAX)
            smag = np.empty_like(local_smag)
            global_comm.Allreduce(local_smag, smag, op=MPI.MAX)
            rel_sigma = float(max_diff_all) / (float(smag) + 1e-300)
            self._last_rel_sigma = rel_sigma
            # Divergence guard: an exploded update never recovers, so abort
            # instead of burning the remaining iterations.
            abort_at = self.config.scba.abort_residual
            if (abort_at > 0 and self._scba_iteration > 3
                    and rel_sigma > abort_at):
                self._diverged = True
                if comm.rank == 0:
                    print(f"SCBA ABORTED: rel Sigma^R residual {rel_sigma:.3e}"
                          f" > {abort_at:g} at iteration "
                          f"{self._scba_iteration} (diverged).", flush=True)
                return True
            heat, balance, spread = self._phonon_heat_flow_conservation()
            if heat is not None:
                # Last-iterate heat: the actual current SCBA state. This is
                # the reported current when converged; on a non-converged
                # run it is transparently the last iterate, not a
                # cherry-picked one.
                self._last_heat_current = heat.copy()
                self._last_heat_spread = spread
                if comm.rank == 0:
                    # lead balance is a SIGN gate, not a magnitude one: both
                    # currents count positive left-to-right, so a steady state
                    # has heat[0] == heat[-1] and balance -> 0, while
                    # heat[-1] == -heat[0] gives EXACTLY 2. That is the device
                    # emitting into both leads -- anti-damping -- and it is a
                    # different failure from "somewhat unbalanced". Say so,
                    # because reading 2.0 as a large imbalance costs days.
                    _sign = ("  [SIGN INVERSION: emitting into both leads]"
                             if balance > 1.5 else "")
                    # The lead current itself, on the same line: `balance`
                    # and `spread` are both normalised by it, so without it
                    # a run killed by the wall reports how self-consistent
                    # it was but not WHAT it computed. That cost two 24 nh
                    # MoS2 rungs (lsM4c/lsM4w, 2026-08-16), whose currents
                    # existed at every iteration and were never printed.
                    # Appended at the END of the line on purpose: several
                    # log parsers anchor on "residual ...; lead balance ..."
                    # (e.g. phonon/studies/_psd_trace_report.py), so a new
                    # field may only go where it cannot split that pair.
                    _lead_J = 0.5 * (abs(float(heat[0]))
                                     + abs(float(heat[-1])))
                    print(f"Phonon: rel Sigma^R residual {rel_sigma:.4e}; "
                          f"lead balance {balance:.4e}; "
                          f"internal spread {spread:.4e}; "
                          f"lead current {_lead_J:.6e}{_sign}", flush=True)
                # NOTE: G survives the back-transpose only when
                # bubble_balance_check keeps it; on discarded G the traces
                # would evaluate to a spurious machine-perfect 0 == 0.
                bubble_resid = None
                if self.config.phonon.bubble_balance_check:
                    bal = self._phonon_bubble_energy_balance()
                    if bal is not None:
                        p_in, p_out, bres = bal
                        bubble_resid = bres
                        if not hasattr(self, "_bubble_balance_history"):
                            self._bubble_balance_history = []
                        self._bubble_balance_history.append(
                            (p_in.real, p_out.real, bres))
                        if comm.rank == 0:
                            # resid is a RATIO and therefore blind to a
                            # blow-up: it read 1.9e-14 while P_in grew from
                            # 2.6e+05 to 4.4e+32. |P_in| is printed beside it
                            # for exactly that reason, and the convergence
                            # criterion is rel Sigma^R above, not this.
                            print(f"Bubble energy balance: P_in={p_in.real:.6e} "
                                  f"P_out={p_out.real:.6e} resid={bres:.3e}",
                                  flush=True)
                # Conservation gate = LEAD balance |J_L - J_R| / |J|: in
                # steady state the in/out lead currents must match. The
                # max-min spread over ALL interfaces additionally contains
                # the eta-absorption dip of the internal interfaces (finite
                # broadening soaks up flux inside the device), so it is
                # reported as a diagnostic only, not gated on.
                # Optional additional gate on the Phi-derivable bubble
                # balance; 0 disables.
                bb_tol = self.config.phonon.bubble_balance_tol
                bubble_ok = (
                    bb_tol <= 0.0
                    or (bubble_resid is not None and bubble_resid < bb_tol)
                )
                if (self._scba_iteration >= self.config.scba.min_iterations
                        and rel_sigma < self.config.phonon.sigma_convergence_tol
                        and balance < self.config.phonon.heat_flow_conservation_tol
                        and bubble_ok):
                    self._converged = True
                    return True

        return False

    def _report_residual_by_frequency(self, diff: NDArray) -> None:
        """Reports where in omega the Sigma^R residual concentrates (rank 0).

        Pinpoints whether a limit cycle lives at the band edge, at a low-omega
        zero mode, or is broadband. Opt-in via the QX_DIAG_OMEGA env var.
        """
        solver = self.subsystems["phonon"]
        per_frequency = get_host(
            xp.abs(diff).reshape(diff.shape[0], -1).max(axis=1)
        )
        frequencies = np.abs(np.asarray(solver.local_frequencies, dtype=float))
        if comm.stack.size > 1:
            per_frequency = comm.stack.all_gather_v(
                np.ascontiguousarray(per_frequency), 0
            )
            frequencies = comm.stack.all_gather_v(
                np.ascontiguousarray(frequencies), 0
            )

        if comm.rank != 0:
            return

        peak = int(np.argmax(per_frequency))
        print(
            f"SigmaR residual by omega: peak at {frequencies[peak]:.2f} THz, "
            f"value {per_frequency[peak]:.3e}",
            flush=True,
        )

    @staticmethod
    def _phonon_hw_weights(solver) -> NDArray:
        """``|omega|`` weights of the phonon energy integrals, multiplied
        by the per-bin quadrature cell widths when the frequency grid is
        NON-UNIFORM. On a uniform grid the legacy unweighted sum is kept
        bit-for-bit (the constant ``dw`` cancels in every conservation
        ratio); on a non-uniform grid the unweighted sum is not an
        integral and the conservation gates would compare misweighted
        quantities."""
        w = xp.abs(xp.asarray(solver.local_frequencies, dtype=float).real)
        if not getattr(solver, "uniform_frequency_grid", True):
            w = w * xp.asarray(solver.local_frequency_weights, dtype=float)
        return w

    def _phonon_bubble_energy_balance(self):
        """Bubble energy-balance diagnostic: the in- and out-scattering
        energy integrals

            P_in  = sum_w  hbar*w * Tr[Sigma^<(w) G^>(w)]
            P_out = sum_w  hbar*w * Tr[Sigma^>(w) G^<(w)]

        must be EQUAL for the Phi-derivable 3-phonon bubble evaluated with
        the same iterand G on all legs. The trace uses the TRANSPOSE pairing
        Sigma_ij G_ji; a residual at roundoff is healthy, O(0.1) means the
        fold/vertex/grid breaks the reflection or permutation symmetry.

        Returns (P_in, P_out, resid) or None if unavailable.
        NOTE: must be called between the SSE evaluation and the mixing step,
        where data.sigma_* is the raw new Sigma[G^n] and data.g_* is G^n --
        a mixed-iterate pairing need not balance.
        """
        if comm.block.size > 1:
            return None  # rows/cols are block-window-local; not wired up
        g_l, g_g = self.data.g_lesser, self.data.g_greater
        s_l, s_g = self.data.sigma_lesser, self.data.sigma_greater
        rows = getattr(g_l, "rows", None)
        cols = getattr(g_l, "cols", None)
        if rows is None or cols is None:
            return None
        if getattr(self, "_bal_perm", None) is None:
            r = np.asarray(get_host(rows)).astype(np.int64)
            c = np.asarray(get_host(cols)).astype(np.int64)
            lut = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(r, c))}
            perm = np.array([lut.get((int(j), int(i)), -1)
                             for i, j in zip(r, c)], dtype=np.int64)
            ok = perm >= 0
            if not ok.all() and comm.rank == 0:
                print(f"Bubble balance: {int((~ok).sum())}/{ok.size} nnz "
                      "without transpose partner excluded.", flush=True)
            self._bal_perm = (xp.asarray(np.where(ok, perm, 0)),
                              xp.asarray(ok.astype(np.float64)))
        perm, ok = self._bal_perm
        solver = self.subsystems.get("phonon")
        w = self._phonon_hw_weights(solver)

        def weighted(sig, g):
            sd = sig.data.reshape(sig.data.shape[0], -1, sig.data.shape[-1])
            gd = g.data.reshape(g.data.shape[0], -1, g.data.shape[-1])
            gt = gd[..., perm] * ok  # G_ji paired with Sigma_ij
            tr = xp.sum(sd * gt, axis=(1, 2))  # trace per local omega
            return np.asarray(get_host(w * tr))  # per local omega

        spec_in = weighted(s_l, g_g)
        spec_out = weighted(s_g, g_l)
        # Cache the per-omega spectra (rank-local frequency slice) for the
        # engine's bubble_balance_spectrum output key.
        self._bubble_balance_spectra = (spec_in, spec_out)
        p_in = complex(spec_in.sum())
        p_out = complex(spec_out.sum())
        if comm.stack.size > 1:
            buf = np.array([p_in, p_out], dtype=complex)
            recv = np.empty_like(buf)
            comm.stack.all_reduce(buf, recv, op="sum")
            p_in, p_out = complex(recv[0]), complex(recv[1])
        resid = abs(p_in - p_out) / max(abs(p_in) + abs(p_out), 1e-300)
        return p_in, p_out, resid

    def _phonon_slab_absorption(self):
        """Per-SLAB scattering energy absorption -- the block-resolved bubble
        balance,

            P_abs(k) = sum_w hbar*w * Tr_k[Sigma_s^>(w) G^<(w)
                                           - Sigma_s^<(w) G^>(w)],

        the local (slab-k) energy sink of the 3-phonon self-energy. It
        connects adjacent interface heat currents by energy continuity, and
        telescoping over the device reproduces the global P_in - P_out. Same
        transpose pairing (Sigma_ij G_ji), hbar*w weighting and one-sided
        grid as :meth:`_phonon_bubble_energy_balance`, binned by the ROW
        block of each nnz entry. Returns a complex (2, n_blocks) array
        ([0] row-binned, [1] block-diagonal-only; real part is the physical
        power) or None if unavailable.
        """
        if comm.block.size > 1:
            return None
        g_l, g_g = self.data.g_lesser, self.data.g_greater
        s_l, s_g = self.data.sigma_lesser, self.data.sigma_greater
        rows = getattr(g_l, "rows", None)
        cols = getattr(g_l, "cols", None)
        if rows is None or cols is None:
            return None
        # Reuse (or build) the transpose-pairing permutation of the balance.
        if getattr(self, "_bal_perm", None) is None:
            if self._phonon_bubble_energy_balance() is None:
                return None
        perm, ok = self._bal_perm
        if getattr(self, "_slab_onehot", None) is None:
            r = np.asarray(get_host(rows)).astype(np.int64)
            c = np.asarray(get_host(cols)).astype(np.int64)
            offs = np.concatenate(
                ([0], np.cumsum(np.asarray(get_host(g_l.block_sizes)))))
            slab_r = np.searchsorted(offs, r, side="right") - 1
            slab_c = np.searchsorted(offs, c, side="right") - 1
            n_blocks = offs.size - 1
            # Row-binned attribution: the restricted trace Tr_k[Sigma G]
            # of eq. P_abs (row of Sigma in slab k, all columns). The
            # block-DIAGONAL-only variant (Sigma_kk G_kk) is kept as an
            # experimental alternative in slot [1].
            onehot = np.zeros((r.size, n_blocks))
            onehot[np.arange(r.size), slab_r] = 1.0
            onehot_diag = np.where((slab_r == slab_c)[:, None], onehot, 0.0)
            self._slab_onehot = (xp.asarray(onehot), xp.asarray(onehot_diag))
        onehot, onehot_diag = self._slab_onehot
        solver = self.subsystems.get("phonon")
        w = self._phonon_hw_weights(solver)

        def weighted_by_slab(sig, g):
            sd = sig.data.reshape(sig.data.shape[0], -1, sig.data.shape[-1])
            gd = g.data.reshape(g.data.shape[0], -1, g.data.shape[-1])
            gt = gd[..., perm] * ok          # G_ji paired with Sigma_ij
            per_nnz = xp.sum(w[:, None, None] * (sd * gt), axis=(0, 1))
            return per_nnz @ onehot, per_nnz @ onehot_diag

        og_row, og_diag = weighted_by_slab(s_g, g_l)   # Sigma^> G^<
        il_row, il_diag = weighted_by_slab(s_l, g_g)   # Sigma^< G^>
        p_abs = np.stack([np.asarray(get_host(og_row - il_row)),
                          np.asarray(get_host(og_diag - il_diag))])
        if comm.stack.size > 1:
            recv = np.empty_like(p_abs)
            comm.stack.all_reduce(np.ascontiguousarray(p_abs), recv, op="sum")
            p_abs = recv
        # (2, n_blocks): [0] = row-binned (sums to the global balance),
        # [1] = block-diagonal-only (experimental alternative attribution).
        return p_abs

    def _phonon_heat_flow_conservation(self):
        """Heat-flow conservation of the anharmonic phonon SCBA.

        Returns ``(heat_per_interface, lead_balance, rel_spread)`` or
        ``(None, inf, inf)`` if the current is unavailable; all-reduced over
        the energy (stack) partition. ``lead_balance`` is
        ``|J_lead0 - J_leadN| / |mean|`` -- the physical steady-state
        criterion, insensitive to the eta-absorption dip of the internal
        interfaces; ``rel_spread`` is the max-min spread over all interfaces
        (contains the eta dip; diagnostic only)."""
        solver = self.subsystems.get("phonon")
        mw = getattr(solver, "meir_wingreen_current", None)
        if mw is None:
            return None, float("inf"), float("inf")
        mw = xp.asarray(mw)
        w = self._phonon_hw_weights(solver).reshape(
            (-1,) + (1,) * (mw.ndim - 1))
        heat_e = xp.real(xp.sum(w * mw, axis=0))    # (*nk, interface)
        # Sum over any transverse-q axes: under 3-phonon scattering the
        # PER-q heat current is not conserved (the momentum convolution
        # exchanges energy between q), only the q-summed energy current is.
        # The q-axis is local (not stack-partitioned), so this sum is exact.
        while heat_e.ndim > 1:
            heat_e = xp.sum(heat_e, axis=0)
        local_heat = get_host(heat_e)               # per interface
        heat = np.array(local_heat, copy=True)
        if comm.stack.size > 1:
            recv = np.empty_like(heat)
            comm.stack.all_reduce(np.ascontiguousarray(local_heat), recv, op="sum")
            heat = recv
        # The "inv" solver and the distributed RGF leave the internal
        # interfaces NaN, so the denominator uses the leads and the spread is
        # NaN-aware. Both lead currents count positive left-to-right, so a
        # steady state has heat[0] == heat[-1]; differencing them WITHOUT
        # taking absolute values first is what makes the gate sensitive to a
        # sign inversion (anti-damping emits into both leads).
        denom = 0.5 * (abs(float(heat[0])) + abs(float(heat[-1]))) + 1e-300
        spread = float((np.nanmax(heat) - np.nanmin(heat)) / denom)
        balance = float(abs(float(heat[0]) - float(heat[-1])) / denom)
        return heat, balance, spread

    @profiler.profile(label="SCBA: Interactions", level="default", comm=comm)
    def _compute_interactions(self) -> None:
        """Iterate the interaction registry, accumulating into sigma_*
        """
        for interaction in self.interactions:
            interaction.compute(self)

    @profiler.profile(label="SCBA: G observables", level="default", comm=comm)
    def _compute_electron_observables(self) -> None:
        """Computes electron observables."""
        if self.config.outputs.electron_ldos:
            self.observables.electron_ldos = -density(
                self.data.g_retarded,
                self.electron_solver.overlap,
            ) / (2 * xp.pi)
            self.observables.electron_ldos *= 2  # Spin
        if self.config.outputs.electron_density:
            self.observables.electron_density = density(
                self.data.g_lesser,
                self.electron_solver.overlap,
            ) / (2 * xp.pi)
            self.observables.electron_density *= 2  # Spin
        if self.config.outputs.hole_density:
            self.observables.hole_density = -density(
                self.data.g_greater,
                self.electron_solver.overlap,
            ) / (2 * xp.pi)
            self.observables.hole_density *= 2  # Spin

        if self.config.outputs.device_currents:
            self.observables.electron_current["device"] = device_current(
                self.data.g_lesser, self.electron_solver.hamiltonian
            )
            if self.config.electron.solver.compute_current:

                local_current = self.electron_solver.meir_wingreen_current
                meir_wingreen_current = comm.stack.all_gather_v(
                    local_current,
                    axis=0,
                    mask=self.data.g_lesser._stack_padding_mask,
                )

                self.observables.electron_current["meir-wingreen"] = (
                    meir_wingreen_current
                )

        if self.config.outputs.self_energy_density:
            self.observables.sigma_lesser_density = density(
                self.data.sigma_lesser,
                self.electron_solver.overlap,
            ) / (2 * xp.pi)
            self.observables.sigma_greater_density = -density(
                self.data.sigma_greater,
                self.electron_solver.overlap,
            ) / (2 * xp.pi)

    @profiler.profile(label="SCBA: W observables", level="default", comm=comm)
    def _compute_coulomb_screening_observables(self) -> None:

        # NOTE: The overlap is maybe missing here (it is not used)
        if self.config.outputs.polarization_density:
            self.observables.p_lesser_density = density(self.data.p_lesser) / (
                2 * xp.pi
            )
            self.observables.p_greater_density = -density(self.data.p_greater) / (
                2 * xp.pi
            )

        if self.config.outputs.coulomb_screening_density:
            self.observables.w_lesser_density = density(self.data.w_lesser) / (
                2 * xp.pi
            )
            self.observables.w_greater_density = -density(self.data.w_greater) / (
                2 * xp.pi
            )

    def _compute_excess_charge_densities(self):
        """Computes the charge density from the local density of states.

        Returns
        -------
        excess_electron_density : NDArray
            The excess electron density computed from the local density
            of states.
        excess_hole_density : NDArray
            The excess hole density computed from the local density of
            states.

        """
        if (
            self.observables.electron_density is None
            or self.observables.hole_density is None
        ):
            raise ValueError(
                "Electron and hole densities must be computed "
                "before computing excess charge densities."
            )

        # NOTE: Use the mid-gap-energy of a reference contact.
        if self.electron_solver.left_voltage == 0.0:
            mid_gap_energy = self.electron_solver.left_mid_gap_energy
        elif self.electron_solver.right_voltage == 0.0:
            mid_gap_energy = self.electron_solver.right_mid_gap_energy
        else:
            raise NotImplementedError(
                "Cannot determine mid-gap energy for excess charge density calculation. "
                "At least one contact must be grounded (zero voltage)."
            )

        mid_gap_energy = self.electron_solver.potential + mid_gap_energy

        electron_density = self.observables.electron_density.copy()
        hole_density = self.observables.hole_density.copy()

        mask = self.energies[:, None] > mid_gap_energy
        electron_density[~mask] = 0
        hole_density[mask] = 0

        excess_electron_density = np.trapezoid(electron_density, self.energies, axis=0)
        excess_hole_density = np.trapezoid(hole_density, self.energies, axis=0)

        return excess_electron_density, excess_hole_density

    def set_potential(self, potential: NDArray):
        """Sets the potential for the SCBA calculation.

        Parameters
        ----------
        potential : NDArray
            The new potential values to be set in the system matrix.

        """
        if potential.shape[0] != np.sum(self.data.orbitals_per_atom):
            potential = np.repeat(potential, self.data.orbitals_per_atom)

        self.electron_solver.potential = potential

    def get_charge_density(self) -> NDArray:
        """Gets the charge density.

        This computes the excess charge density from the spectral
        electron and hole densities.

        Returns
        -------
        charge_density : NDArray
            The computed charge density for the device.

        """
        electron_density, hole_density = self._compute_excess_charge_densities()
        charge_density = electron_density - hole_density

        # From orbital to atom resolved charge density.
        orbital_offsets = np.hstack(([0], np.cumsum(self.data.orbitals_per_atom)))
        return np.add.reduceat(charge_density, orbital_offsets[:-1])

    @profiler.profile(label="SCBA: Write outputs", level="default", comm=comm)
    def _write_iteration_outputs(self, iteration: int):
        """Writes output for the current iteration on rank zero."""

        if comm.rank != 0:
            return

        outputs = {}

        if self.config.outputs.electron_ldos:
            outputs[f"electron_ldos_{iteration}.npy"] = self.observables.electron_ldos
        if self.config.outputs.electron_density:
            outputs[f"electron_density_{iteration}.npy"] = (
                self.observables.electron_density
            )
        if self.config.outputs.hole_density:
            outputs[f"hole_density_{iteration}.npy"] = self.observables.hole_density

        if self.config.outputs.device_currents and "electron" in self.subsystems:
            outputs[f"device_current_{iteration}.npy"] = (
                self.observables.electron_current["device"]
            )
            if self.config.electron.solver.compute_current:
                outputs[f"meir_wingreen_current_{iteration}.npy"] = (
                    self.observables.electron_current["meir-wingreen"]
                )

        if self.config.scba.coulomb_screening:
            if self.config.outputs.polarization_density:
                outputs.update(
                    {
                        f"p_lesser_density_{iteration}.npy": self.observables.p_lesser_density,
                        f"p_greater_density_{iteration}.npy": self.observables.p_greater_density,
                    }
                )
            if self.config.outputs.coulomb_screening_density:
                outputs.update(
                    {
                        f"w_lesser_density_{iteration}.npy": self.observables.w_lesser_density,
                        f"w_greater_density_{iteration}.npy": self.observables.w_greater_density,
                    }
                )

        if self.config.outputs.self_energy_density:
            outputs.update(
                {
                    f"sigma_lesser_density_{iteration}.npy": self.observables.sigma_lesser_density,
                    f"sigma_greater_density_{iteration}.npy": self.observables.sigma_greater_density,
                }
            )

        print(f"Writing output for iteration {iteration}...", flush=True)

        if not os.path.exists(self.config.output_dir):
            os.mkdir(self.config.output_dir)

        for filename, data in outputs.items():
            xp.save(self.config.output_dir / filename, data)

    @profiler.profile(label="SCBA", level="default", comm=comm)
    def run(self) -> None:
        """Runs the SCBA to convergence."""
        print("Entering SCBA loop...", flush=True) if comm.rank == 0 else None

        # Phonon post-processing reads G^R off the final iterate, so that path
        # holds the buffer; the electron path frees it before the interactions
        # allocate theirs.
        keep_g_retarded = "phonon" in self.subsystems

        for i in range(self.config.scba.max_iterations):
            print(f"Iteration {i}", flush=True) if comm.rank == 0 else None

            with profiler.profile_range(
                label="SCBA: Iteration", level="default", comm=comm
            ):
                self.data.g_retarded.allocate_data()
                self.data.g_retarded.data[:] = 0.0
                for solver in self.subsystems.values():
                    solver.solve(
                        self.data.sigma_lesser,
                        self.data.sigma_greater,
                        self.data.sigma_retarded_hermitian,
                        out=(
                            self.data.g_lesser,
                            self.data.g_greater,
                            self.data.g_retarded,
                        ),
                    )
                # Electron-specific observables; phonon transport reads the
                # heat current from PhononSolver.meir_wingreen_current instead.
                if "electron" in self.subsystems:
                    self._compute_electron_observables()

                if not keep_g_retarded:
                    self.data.g_retarded.free_data()

                # Stash current into previous self-energy buffer.
                self._stash_sigma()

                with profiler.profile_range(
                    label="SCBA: stack->nnz transpose", level="default", comm=comm
                ):
                    # Transpose to nnz distribution.
                    # NOTE: While computing all interactions, we only ever need
                    # to access the Green's function and the self-energies in
                    # their nnz-distributed state.
                    for m in (self.data.g_lesser, self.data.g_greater):
                        m.dtranspose(discard=False)  # This must not be discarded.
                        assert m.distribution_state == "nnz"
                    for m in (
                        self.data.sigma_lesser,
                        self.data.sigma_greater,
                        self.data.sigma_retarded_hermitian,
                    ):
                        m.dtranspose(discard=True)  # These can be safely discarded.
                        assert m.distribution_state == "nnz"

                self._compute_interactions()

                with profiler.profile_range(
                    label="SCBA: stack->nnz transpose back", level="default", comm=comm
                ):
                    # G must survive the back-transpose for the whole phonon
                    # path, not just for the bubble-balance diagnostic: the
                    # slab absorption and the post-hoc G^< diagonals are also
                    # read off the final iterate. Discarding it here does not
                    # fail, it silently zeroes those observables.
                    keep_g = bool(
                        self.config.scba.phonon
                        and self.config.phonon.model == "negf"
                    )
                    for m in (self.data.g_lesser, self.data.g_greater):
                        m.dtranspose(discard=not keep_g)
                        assert m.distribution_state == "stack"
                    for m in (
                        self.data.sigma_lesser,
                        self.data.sigma_greater,
                        self.data.sigma_retarded_hermitian,
                    ):
                        m.dtranspose(discard=False)  # This must not be discarded.
                        assert m.distribution_state == "stack"

            # The anharmonic phonon path keeps the raw SSE output: the
            # skew-Hermitian projection of Sigma^{<,>} breaks the
            # Phi-derivable bubble energy balance. The SSE writes only the
            # Hermitian part of Sigma^R ("half": nothing; "fft": the KK real
            # part), so the skew part is assembled here -- PhononSolver
            # consumes the buffer as the full retarded self-energy (the
            # electron solver, by contrast, adds the skew part internally).
            if (self.config.scba.phonon
                    and self.config.phonon.model == "negf"):
                # "half" retarded rule in the solver's occupation-positive
                # convention (sigma^{<,>} = +i n(+1) Gamma_s, like the lead
                # injection): the damping skew part is (sigma^< - sigma^>)/2
                # = -i Gamma_s / 2.
                self.data.sigma_retarded_hermitian.data += 0.5 * (
                    self.data.sigma_lesser.data - self.data.sigma_greater.data
                )
            else:
                self._symmetrize_sigma()

            if self._has_converged():
                if comm.rank == 0:
                    if self._diverged:
                        print(f"SCBA diverged at iteration {i} "
                              f"(reporting best-conserved iterate).", flush=True)
                    else:
                        print(f"SCBA converged after {i} iterations.", flush=True)
                break

            # Update self-energy for next iteration with mixing factor.
            self._update_sigma()

            if xp.__name__ == "cupy":
                free_memory, total_memory = xp.cuda.Device().mem_info
                usage = np.array((total_memory - free_memory) / total_memory)
                average_usage = np.empty(1)
                max_usage = np.empty(1)
                global_comm.Allreduce(usage, average_usage, op=MPI.SUM)
                global_comm.Allreduce(usage, max_usage, op=MPI.MAX)
                average_usage /= comm.size

                if comm.rank == 0:
                    print(
                        f"Rank-average device memory usage: {average_usage[0] * 100:.4f}%",
                        flush=True,
                    )
                    print(
                        f"Max device memory usage: {max_usage[0] * 100:.4f}%",
                        flush=True,
                    )

            if i % self.config.scba.output_interval == 0:
                self._write_iteration_outputs(i)

        else:  # Did not break, i.e. max_iterations reached.
            if comm.rank == 0:
                print(f"SCBA did not converge after {i} iterations.")
