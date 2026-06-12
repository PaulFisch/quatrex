# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np

from qttools.datastructures import DSDBSparse
from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.utils.mpi_utils import distributed_load, get_local_slice, get_section_sizes
from qttools.utils.stack_utils import scale_stack
from qttools.profiling import Profiler
from qttools.greens_function_solver.solver import OBCBlocks

from quatrex.bandstructure.band_edges import (
    find_band_edges,
    find_dos_peaks,
    find_renormalized_eigenvalues,
)
from quatrex.core.statistics import bose_einstein
from qttools.toeplitz.toeplitz import get_periodic_superblocks
from quatrex.device.inputs import load_matrix
from quatrex.core.config import QuatrexConfig
from quatrex.core.subsystem import SubsystemSolver

profiler = Profiler()


# TODO this is a duplicate from electron/solver.py that belongs in neither
def _btd_subtract(a: DSDBSparse, b: DSDBSparse) -> None:
    """Subtracts b from a on the block-tridiagonal.

    This is an in-place operation, i.e. a is modified.

    Parameters
    ----------
    a : DSDBSparse
        The matrix to subtract from.
    b : DSDBSparse
        The matrix to subtract.

    """
    a_ = a.stack[...]
    b_ = b.stack[...]
    for i in range(a.num_local_blocks):
        j = i + 1
        a_.blocks[i, i] -= b_.blocks[i, i]

        if j >= a.num_local_blocks and comm.block.rank == comm.block.size - 1:
            # The last rank does not have these blocks.
            continue

        a_.blocks[i, j] -= b_.blocks[i, j]
        a_.blocks[j, i] -= b_.blocks[j, i]


class PhononSolver(SubsystemSolver):
    """Solves the phonon dynamics.

    Parameters
    -----------
    config : QuatrexConfig
            The quatrex simulation configuration.
    energies : np.ndarray
        The energies at which to solve.

    """
    system = "phonon"

    def __init__(
        self,
        config: QuatrexConfig,
        frequecies: NDArray,
        sparsity_pattern: sparse.coo_matrix,
    ) -> None:
        """Initializes the Phonon solver."""
        super().__init__(config, frequecies)

        self.local_frequencies = get_local_slice(frequecies, comm.stack)

        # Load the dynamical matrix
        self.dynamical_matrix, dynamical_matrix_sparsity_pattern = load_matrix(
            config=config,
            matrix_name="dynamical_matrix",
            sparsity_pattern=None,
            shift_kpoints=False
        )

        # Make sure that the system matrix sparsity is a superset of
        # self-energy and Dynaimcal Matrix sparsity.
        sparsity_pattern += dynamical_matrix_sparsity_pattern

        del dynamical_matrix_sparsity_pattern
        self.block_sizes = self.dynamical_matrix.block_sizes

        # TODO(paul) For phonons we never have non-I overlap?
        self.overlap_sparray = sparse.eye(
            self.dynamical_matrix.shape[-2],
            format="coo",
            dtype=self.dynamical_matrix.dtype
        )

        # Allocate memory for the system matrix.
        self.system_matrix = config.compute.dsdbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape
                               + tuple([int(k) for k in config.device.kpoint_grid if k > 1]),
        )
        self.system_matrix.free_data()  # Free any previously allocated data
        del sparsity_pattern

        self.block_offsets = np.hstack(([0], np.cumsum(self.block_sizes)))
        # Check that the provided block sizes match the Hamiltonian.
        if self.block_sizes.sum() != self.dynamical_matrix.shape[-2]:
            raise ValueError(
                "Block sizes do not match Hamiltonian. "
                f"{self.block_sizes.sum()} != {self.dynamical_matrix.shape[-2]}"
            )

        self.compute_meir_wingreen_current = config.phonon.solver.compute_current

        self.left_temperature = config.phonon.left_temperature
        self.right_temperature = config.phonon.right_temperature

        # frequencies are the linear angular-frequency
        # grid in THz (uniform spacing, as the bubble FFT requires). The Bose
        # occupation uses hbar*omega with omega = 2*pi*1e12 * f[THz].
        hbar_omega_eV = 6.582119569e-16 * (2 * np.pi * 1e12) * np.abs(
            self.local_frequencies
        )
        self.left_occupancies = bose_einstein(hbar_omega_eV, self.left_temperature)
        self.right_occupancies = bose_einstein(hbar_omega_eV, self.right_temperature)
        # Regularize the omega=0 sampling point
        self.left_occupancies = xp.where(
            xp.isfinite(self.left_occupancies), self.left_occupancies, 0.0)
        self.right_occupancies = xp.where(
            xp.isfinite(self.right_occupancies), self.right_occupancies, 0.0)
        # Hard low-frequency cutoff (sse_cutoff_zero_g): no lead injection
        # below the SSE cutoff -> G^< = 0 there -> zero current below the
        # cutoff in BOTH ballistic and anharmonic runs (the "totally
        # zeroed" treatment, vs. the default masked/ballistic-below one).
        _cut = float(getattr(config.phonon, "sse_low_freq_cutoff_thz", 0.0))
        if _cut > 0.0 and bool(getattr(config.phonon, "sse_cutoff_zero_g", False)):
            _m = xp.abs(xp.asarray(self.local_frequencies)) < _cut
            self.left_occupancies = xp.where(_m, 0.0, self.left_occupancies)
            self.right_occupancies = xp.where(_m, 0.0, self.right_occupancies)

        self.eta = config.phonon.eta
        self.eta_obc = config.phonon.eta_obc

        self.obc_blocks = OBCBlocks(num_blocks=self.system_matrix.num_local_blocks)
        self.block_sections = config.phonon.obc.block_sections

        self.band_edge_tracking = config.phonon.band_edge_tracking

        self.call_count = 0
        self.filtering_iteration_limit = config.phonon.filtering_iteration_limit

    @profiler.profile(label="PhononSolver: OBC", level="default", comm=comm)
    def _compute_obc(self) -> None:
        """Computes open boundary conditions."""
        if comm.block.rank == 0:
            s_00 = 1j * self.eta_obc * xp.eye(
                self.block_sizes[0], dtype=self.dynamical_matrix.dtype)

            m_10, m_00, m_01 = get_periodic_superblocks(
                a_ii=self.system_matrix.blocks[0, 0],
                a_ji=self.system_matrix.blocks[1, 0],
                a_ij=self.system_matrix.blocks[0, 1],
                block_sections=self.block_sections,
            )

            g_00, *__ = self.obc(
                (m_00 + s_00, m_01, m_10),
                contact="left",
            )
            # Apply the retarded boundary self-energy.
            sigma_00 = m_10 @ g_00 @ m_01
            if len(self.system_matrix.global_stack_shape) == 1:
                # Gamma-only (real-symmetric D): the exact contact Sigma^R is
                # complex-SYMMETRIC, but the NEVP eigenvector construction
                # breaks it at ~1e-2, which propagates into G (measured
                # G-asymmetry 1.8%) and breaks the bosonic no-transpose fold
                # of the SSE. Project back onto the symmetric subspace.
                sigma_00 = 0.5 * (sigma_00 + sigma_00.swapaxes(-2, -1))
            self.obc_blocks.retarded[0] = sigma_00
            gamma_00 = 1j * (sigma_00 - sigma_00.conj().swapaxes(-2, -1))

            # Compute and apply the lesser boundary self-energy.
            self.obc_blocks.lesser[0] = 1j * scale_stack(
                gamma_00.copy(), self.left_occupancies
            )
            # Compute and apply the greater boundary self-energy.
            self.obc_blocks.greater[0] = 1j * scale_stack(
                gamma_00.copy(), self.left_occupancies + 1
            )
        if comm.block.rank == comm.block.size - 1:
            s_nn = 1j * self.eta_obc * xp.eye(
                self.block_sizes[0], dtype=self.dynamical_matrix.dtype)

            n = self.system_matrix.num_local_blocks - 1
            m = n - 1

            m_mn, m_nn, m_nm = get_periodic_superblocks(
                # Twist it, flip it, ...
                a_ii=xp.flip(self.system_matrix.blocks[n, n], axis=(-2, -1)),
                a_ji=xp.flip(self.system_matrix.blocks[m, n], axis=(-2, -1)),
                a_ij=xp.flip(self.system_matrix.blocks[n, m], axis=(-2, -1)),
                block_sections=self.block_sections,
            )
            # ... bop it.
            m_nn = xp.flip(m_nn, axis=(-2, -1))
            m_nm = xp.flip(m_nm, axis=(-2, -1))
            m_mn = xp.flip(m_mn, axis=(-2, -1))
            g_nn, *__ = self.obc(
                # Twist it, flip it, ...
                (
                    xp.flip(m_nn + s_nn, axis=(-2, -1)),
                    xp.flip(m_nm, axis=(-2, -1)),
                    xp.flip(m_mn, axis=(-2, -1)),
                ),
                contact="right",
            )
            # ... bop it.
            g_nn = xp.flip(g_nn, axis=(-2, -1))

            # NOTE: Here we could possibly do peak/discontinuity detection
            # on the surface Green's function DOS (not same as actual DOS).

            # Apply the retarded boundary self-energy.
            sigma_nn = m_mn @ g_nn @ m_nm
            if len(self.system_matrix.global_stack_shape) == 1:
                sigma_nn = 0.5 * (sigma_nn + sigma_nn.swapaxes(-2, -1))

            self.obc_blocks.retarded[-1] = sigma_nn

            gamma_nn = 1j * (sigma_nn - sigma_nn.conj().swapaxes(-2, -1))

            self.obc_blocks.lesser[-1] = 1j * scale_stack(
                gamma_nn.copy(), self.right_occupancies
            )

            self.obc_blocks.greater[-1] = 1j * scale_stack(
                gamma_nn.copy(), self.right_occupancies + 1
            )

    @profiler.profile(label="PhononSolver: Assemble", level="default", comm=comm)
    def _assemble_system_matrix(self, sse_retarded: DSDBSparse) -> None:
        """Assembles the system matrix.

        Parameters
        ----------
        sse_retarded : DSDBSparse
            The retarded scattering self-energy.

        """
        self.system_matrix.allocate_data()
        self.system_matrix.data = 0.0

        # (omega + i*eta)^2 * I  (THz^2). The phonon Dyson equation is
        # [ (omega+i eta)^2 - D - Sigma ] G = I, with D the dynamical matrix
        # and Sigma the scattering self-energy, all in THz^2
        self.system_matrix.fill_diagonal(1.0)
        if getattr(self.config.phonon, "broadening_form", "squared") == "linear":
            # omega^2 + 2i*eta*omega: frequency-proportional damping WITHOUT
            # the -eta^2 band-edge shift of (omega+i*eta)^2 (which makes
            # omega <~ eta artificially evanescent; see config docstring).
            z2 = (self.local_frequencies ** 2
                  + 2j * self.eta * xp.abs(self.local_frequencies))
        else:
            z2 = (self.local_frequencies + 1j * self.eta) ** 2
        scale_stack(self.system_matrix.data, z2)

        _btd_subtract(self.system_matrix, sse_retarded)
        _btd_subtract(self.system_matrix, self.dynamical_matrix)

    @profiler.profile(label="PhononSolver: Selected Solve", level="default", comm=comm)
    def _selected_solve(
        self,
        sse_lesser: DSDBSparse,
        sse_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...]
    ) -> None:
        """Perform selected solve for the phonon Green's function."""
        if comm.block.size > 1:
            # NOTE: mirror the single-block branch -- the distributed RGF
            # also returns the (block-all-reduced) lead heat current when
            # asked. Without this the block-parallel path leaves
            # ``meir_wingreen_current`` unset, so the heat-flow convergence
            # criterion (the only valid one for the anharmonic phonon SCBA)
            # never fires and no conductance can be extracted.
            self.meir_wingreen_current = self.solver_dist.selected_solve(
                a=self.system_matrix,
                sigma_lesser=sse_lesser,
                sigma_greater=sse_greater,
                obc_blocks=self.obc_blocks,
                out=out,
                return_retarded=True,
                return_current=self.compute_meir_wingreen_current,
            )
        else:
            self.meir_wingreen_current = self.solver.selected_solve(
                a=self.system_matrix,
                sigma_lesser=sse_lesser,
                sigma_greater=sse_greater,
                obc_blocks=self.obc_blocks,
                out=out,
                return_retarded=True,
                return_current=self.compute_meir_wingreen_current,
            )

    @profiler.profile(label="PhononSolver: Filter", level="default", comm=comm)
    def _filter_peaks(self, out: tuple[DSDBSparse, ...]) -> None:
        """Filters out peaks in the output Green's functions"""
        pass

    @profiler.profile(label="PhononSolver: DOS Peaks", level="default", comm=comm)
    def _track_dos_peaks(self, out: tuple[DSDBSparse, ...]) -> None:
        """Tracks dos peaks in the output Green's functions"""
        pass
        # _, _, g_retarded = out
        # left_band_edges = np.empty((2,), dtype=float)
        # right_band_edges = np.empty((2,), dtype=float)
        #
        # if comm.block.rank == 0:
        #     g_00 = g_retarded.blocks[0, 0]
        #     local_left_dos = -xp.mean(
        #         xp.diagonal(g_00, axis1=-2, axis2=-1).imag, axis=-1
        #     )
        #
        #     left_dos = comm.stack.all_gather_v(
        #         local_left_dos,
        #         axis=0,
        #         mask=g_retarded._stack_padding_mask,
        #     )
        #
        #     e_0_left = find_dos_peaks(left_dos, self.energies)
        #     left_band_edges = np.array(
        #         find_band_edges(e_0_left, self.left_mid_gap_frequency)
        #     )
        #
        # if comm.block.rank == comm.block.size - 1:
        #     n = g_retarded.num_local_blocks - 1
        #     g_nn = g_retarded.blocks[n, n]
        #     local_right_dos = -xp.mean(
        #         xp.diagonal(g_nn, axis1=-2, axis2=-1).imag, axis=-1
        #     )
        #
        #     right_dos = comm.stack.all_gather_v(
        #         local_right_dos,
        #         axis=0,
        #         mask=g_retarded._stack_padding_mask,
        #     )
        #
        #     e_0_right = find_dos_peaks(right_dos, self.energies)
        #     right_band_edges = np.array(
        #         find_band_edges(e_0_right, self.right_mid_gap_frequency)
        #     )
        #
        # comm.block.bcast(left_band_edges, root=0, backend="device_mpi")
        # comm.block.bcast(
        #     right_band_edges, root=comm.block.size - 1, backend="device_mpi"
        # )
        #
        # self._update_fermi_levels(left_band_edges, right_band_edges)

    @profiler.profile(label="PhononSolver", level="default", comm=comm)
    def solve(
        self,
        sse_lesser: DSDBSparse,
        sse_greater: DSDBSparse,
        sse_retarded: DSDBSparse,
        out: tuple[DSDBSparse, ...],
    ):
        """Solves for the phonon Green's function.

        Parameters
        ----------
        sse_lesser : DSDBSparse
            The lesser self-energy.
        sse_greater : DSDBSparse
            The greater self-energy.
        sse_retarded : DSDBSparse
            The retarded self-energy.
        out : tuple[DSDBSparse, ...]
            The output matrices. The order is (lesser, greater,
            retarded).

        """

        self._assemble_system_matrix(sse_retarded)

        # TODO Band ege Tracking

        self._compute_obc()
        self._selected_solve(sse_lesser, sse_greater, out)

        self.system_matrix.free_data()
        if self.call_count < self.filtering_iteration_limit:
            self._filter_peaks(out)

        if self.band_edge_tracking == "dos-peaks":
            self._track_dos_peaks(out)

        self.call_count += 1