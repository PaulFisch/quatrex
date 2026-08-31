# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the core classes for the self-consistent Born approximation (SCBA) solver."""

import os
from dataclasses import dataclass, field

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, sparse, xp
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
        block_sizes = get_block_sizes(config, grid)

        kpoint_grid = config.device.kpoint_grid
        phonon_transport = config.simulation_type == "phonon"
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

        if comm.rank == 0 and not phonon_transport:
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

            if phonon_transport:
                from quatrex.phonon.microblocks import grouped_band_indices

                micro_dof = int(config.phonon.sse_microblock_dof)
                g_band = min(
                    1 if micro_dof else int(config.phonon.sse_g_band),
                    len(block_sizes) - 1,
                )
                rows_, cols_ = grouped_band_indices(
                    block_sizes, band=g_band,
                    start_block=int(section_offsets[comm.block.rank]),
                    end_block=int(section_offsets[comm.block.rank + 1]))
                self.sparsity_pattern = sparse.coo_matrix(
                    (xp.ones(rows_.size, dtype=xp.float32),
                     (xp.asarray(rows_), xp.asarray(cols_))),
                    shape=(int(np.sum(block_sizes)),) * 2)
            else:
                self.sparsity_pattern = compute_sparsity_pattern(
                    grid,
                    max_interaction_cutoff,
                    transport_direction=config.device.transport_direction,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )

        dsdbsparse_type = config.compute.dsdbsparse_type

        def _zeros_like(dsdbsparse: DSDBSparse) -> DSDBSparse:
            out = dsdbsparse_type.empty_like(dsdbsparse)
            out.allocate_data()
            out.data[:] = 0.0
            return out

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

        self._anderson_mixer = None
        if self.config.scba.mixing_method != "linear" and not (
            self.config.scba.mixing_method == "anderson"
            and self.config.scba.anderson_warmup_iters > 0
        ):
            self._anderson_mixer = self._build_mixer()

        self._scba_iteration = 0
        self._last_heat_current = None
        self._diverged = False

        # ----- Particles ----------------------------------------------
        self.energies = get_electron_energies(config)

        # A file-based phonon grid overrides the electron energy window.
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
            if coulomb_matrix.symmetry is None:
                coulomb_matrix.symmetrize("hermitian")
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
                solver_freqs = np.asarray(
                    get_host(self.phonon_solver.local_frequencies))
                npy_freqs = np.asarray(get_host(self.phonon_energies))
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
        """Construct the configured SCBA mixer."""
        from quatrex.core.anderson import AndersonMixer

        scba = self.config.scba
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

        from quatrex.experimental.mixers.factory import build_mixer

        return build_mixer(
            scba.mixing_method,
            scba.experimental_mixer,
            self.mixing_factor,
            scba.anderson_depth,
            self._get_phonon_jvp,
        )

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

    def _sigma_checkpoint_buffers(self):
        """Return the state from which the next SCBA step should start."""
        if self._converged or self._diverged:
            return (
                self.data.sigma_lesser_prev,
                self.data.sigma_greater_prev,
                self.data.sigma_retarded_hermitian_prev,
            )
        return (
            self.data.sigma_lesser,
            self.data.sigma_greater,
            self.data.sigma_retarded_hermitian,
        )

    @profiler.profile(label="SCBA: Symmetrize Sigma", level="default", comm=comm)
    def _symmetrize_sigma(self) -> None:
        # Symmetrization.
        if not self.config.scba.symmetric:
            self.data.sigma_lesser.symmetrize("skew-hermitian")
            self.data.sigma_greater.symmetrize("skew-hermitian")
            # Make the self-energy Hermitian
            # This is done before adding the skew hermitian part coming
            # from the lesser and greater self-energies
            self.data.sigma_retarded_hermitian.symmetrize("hermitian")

        if self.config.scba.align_self_energy_to_complex_axes:
            self.data.sigma_lesser._data.real = 0
            self.data.sigma_greater._data.real = 0
            # Make sure that the imaginary part comes only from
            # sigma_greater - sigma_lesser.
            self.data.sigma_retarded_hermitian._data.imag = 0

    @profiler.profile(label="SCBA: Update Sigma", level="default", comm=comm)
    def _update_sigma(self) -> None:
        """Updates the self-energy: damped linear or quasi-Newton mixing."""
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
        a = self.mixing_factor
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

        if (self.config.scba.phonon
                and self.config.phonon.model == "negf"):
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
                self._last_heat_current = heat.copy()
                self._last_heat_spread = spread
                if comm.rank == 0:
                    _sign = ("  [SIGN INVERSION: emitting into both leads]"
                             if balance > 1.5 else "")
                    _lead_J = 0.5 * (abs(float(heat[0]))
                                     + abs(float(heat[-1])))
                    print(f"Phonon: rel Sigma^R residual {rel_sigma:.4e}; "
                          f"lead balance {balance:.4e}; "
                          f"internal spread {spread:.4e}; "
                          f"lead current {_lead_J:.6e}{_sign}", flush=True)
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
                            print(f"Bubble energy balance: P_in={p_in.real:.6e} "
                                  f"P_out={p_out.real:.6e} resid={bres:.3e}",
                                  flush=True)
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
        frequencies = np.abs(np.asarray(
            get_host(solver.local_frequencies), dtype=float
        ))
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
        """``|omega| d omega`` weights of the phonon energy integrals.

        The constant cell width cancels from a conservation *ratio* on a
        uniform grid, but not from the reported current or conductance.  The
        old unweighted sum consequently scaled as ``1/dw`` under frequency
        refinement.  Applying the stored cell measure on every grid makes the
        absolute observable a quadrature while leaving all same-grid balance
        ratios unchanged.
        """
        w = xp.abs(xp.asarray(solver.local_frequencies, dtype=float).real)
        widths = xp.asarray(solver.local_frequency_weights, dtype=float)
        return w * widths

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
        return p_abs

    def _phonon_heat_flow_conservation(self):
        """Heat-flow conservation of the anharmonic phonon SCBA.

        Returns ``(heat_per_interface, lead_balance, rel_spread)`` or
        ``(None, inf, inf)`` if the current is unavailable; all-reduced over
        the energy (stack) partition. ``lead_balance`` is
        ``|J_lead0 - J_leadN| / |mean|`` -- the physical steady-state
        criterion. ``rel_spread`` is the max-min spread over all RGF
        interfaces, or NaN when the distributed solver omits them. It can
        contain physical redistribution between the harmonic and interaction
        channels. It is a diagnostic, not a conservation criterion."""
        solver = self.subsystems.get("phonon")
        mw = getattr(solver, "meir_wingreen_current", None)
        if mw is None:
            return None, float("inf"), float("inf")
        mw = xp.asarray(mw)
        w = self._phonon_hw_weights(solver).reshape(
            (-1,) + (1,) * (mw.ndim - 1))
        heat_e = xp.real(xp.sum(w * mw, axis=0))    # (*nk, interface)
        while heat_e.ndim > 1:
            heat_e = xp.sum(heat_e, axis=0)
        local_heat = get_host(heat_e)               # per interface
        heat = np.array(local_heat, copy=True)
        if comm.stack.size > 1:
            recv = np.empty_like(heat)
            comm.stack.all_reduce(np.ascontiguousarray(local_heat), recv, op="sum")
            heat = recv
        balance, spread = self._phonon_heat_flow_metrics(heat)
        return heat, balance, spread

    @staticmethod
    def _phonon_heat_flow_metrics(heat: NDArray) -> tuple[float, float]:
        """Return lead balance and the complete-interface spread."""
        heat = np.asarray(heat)
        denom = 0.5 * (abs(float(heat[0])) + abs(float(heat[-1]))) + 1e-300
        balance = float(abs(float(heat[0]) - float(heat[-1])) / denom)
        spread = (
            float(np.ptp(heat) / denom)
            if np.all(np.isfinite(heat))
            else float("nan")
        )
        return balance, spread

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

            if (self.config.scba.phonon
                    and self.config.phonon.model == "negf"):
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
