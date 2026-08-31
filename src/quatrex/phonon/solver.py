# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the Phonon solver class."""

import numpy as np
from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.profiling import Profiler
from qttools.toeplitz.toeplitz import get_periodic_superblocks
from qttools.utils.mpi_utils import distributed_load, get_local_slice, get_section_sizes
from qttools.utils.stack_utils import scale_stack

from quatrex.core.config import QuatrexConfig
from quatrex.core.statistics import bose_einstein
from quatrex.core.subsystem import SubsystemSolver
from quatrex.device.inputs import load_matrix, load_periodic_transport_couplings
from quatrex.phonon.experimental.pole.runtime import PoleRuntimeMixin
from quatrex.phonon.units import thz_to_ev

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


def validate_obc_block_sections(num_blocks: int, block_comm_size: int) -> None:
    """Checks that the boundary ranks own enough blocks for the contact OBC.

    Both contacts build their periodic superblocks from a diagonal block
    and its immediate off-diagonals, so a boundary rank needs two blocks.
    Rank 0 survives on one: its ``local_block_sizes`` and the nnz it holds
    run to the end of the device, so ``blocks[1, 0]`` still reads the right
    values. The LAST rank does not -- with a single block it addresses
    ``num_local_blocks - 2 = -1``, a negative local index that no
    DSDBSparse can serve, and the run dies inside ``_compute_obc`` with an
    opaque ``IndexError``.

    The electron and Coulomb solvers index their boundary blocks the same
    way and carry the same requirement; only the phonon path checks it.

    Parameters
    ----------
    num_blocks : int
        Number of blocks in the device.
    block_comm_size : int
        Size of the block communicator.

    Raises
    ------
    ValueError
        If the last block rank would own fewer than two blocks. Derived
        from the section sizes alone, so every rank raises identically.

    """
    if block_comm_size <= 1:
        return
    sections, __ = get_section_sizes(num_blocks, block_comm_size)
    if int(sections[-1]) < 2:
        raise ValueError(
            f"block_comm_size={block_comm_size} sections {num_blocks} "
            f"blocks as {list(map(int, sections))}, leaving the last rank "
            f"{int(sections[-1])} block(s); the right contact OBC needs "
            ">= 2 there. Use a longer device or reduce block_comm_size."
        )


class PhononSolver(PoleRuntimeMixin, SubsystemSolver):
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
        frequencies: NDArray,
        sparsity_pattern: sparse.coo_matrix,
    ) -> None:
        """Initializes the Phonon solver."""
        super().__init__(config, frequencies)

        if float(np.min(frequencies)) < 0.0:
            raise ValueError(
                "The phonon solver requires a non-negative frequency grid: "
                "the negative axis is handled by the bosonic fold in the SSE."
            )
        self.local_frequencies = get_local_slice(frequencies, comm.stack)
        # Per-bin quadrature cell widths + uniformity flag: every heat/energy
        # integral uses these.  A uniform grid supplies the constant dw; it
        # cancels from balance ratios but is required for an absolute current.
        from quatrex.grid.energies import (
            frequency_cell_widths, is_uniform_grid)
        self.local_frequency_weights = get_local_slice(
            frequency_cell_widths(frequencies), comm.stack)
        self.uniform_frequency_grid = is_uniform_grid(frequencies)
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

        # With one finite Dyson block the two periodic FC2 couplings are not
        # present in the finite matrix, but remain available in the unit-cell
        # input.  Retain them for the exact two-contact OBC.  The minus sign is
        # the Dyson convention A = omega^2 I - D - Sigma^R.
        self._single_block_periodic = None
        self._single_block_contacts = None
        if len(self.block_sizes) == 1:
            d_10, d_01 = load_periodic_transport_couplings(
                config, "dynamical_matrix"
            )
            stack_shape = (len(self.local_frequencies),) + d_10.shape
            self._single_block_periodic = (
                xp.broadcast_to(-d_10[None, ...], stack_shape),
                xp.broadcast_to(-d_01[None, ...], stack_shape),
            )

        # TODO(paul) For phonons we never have non-I overlap?
        self.overlap_sparray = sparse.eye(
            self.dynamical_matrix.shape[-2],
            format="coo",
            dtype=self.dynamical_matrix.dtype
        )

        # The system matrix is allocated per solve and freed again, so it does
        # not sit alongside the interaction buffers.
        self.system_matrix = config.compute.dsdbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape
            + tuple([int(k) for k in config.device.kpoint_grid if k > 1]),
            allocate=False,
        )
        del sparsity_pattern

        self.block_offsets = np.hstack(([0], np.cumsum(self.block_sizes)))
        # Check that the provided block sizes match the Hamiltonian.
        if self.block_sizes.sum() != self.dynamical_matrix.shape[-2]:
            raise ValueError(
                "Block sizes do not match Hamiltonian. "
                f"{self.block_sizes.sum()} != {self.dynamical_matrix.shape[-2]}"
            )

        self.compute_meir_wingreen_current = config.phonon.solver.compute_current

        # sse_g_band = k: the SSE bubble consumes the G^{<,>} blocks out to
        # the k-th off-diagonal, so the selected solve must produce them. The
        # RGF takes this as an integer off-diagonal band (1 = block-tridiagonal
        # only, 2 = + second off-diagonal, 3 = + third). Clamped to the
        # widest off-diagonal the device has.
        micro_dof = int(
            getattr(config.phonon, "sse_microblock_dof", 0) or 0)
        self._gf_band = min(
            (1 if micro_dof else
             int(getattr(config.phonon, "sse_g_band", 1) or 1)),
            len(self.block_sizes) - 1,
        )
        # Experimental rational-state sidecar.  It is installed explicitly by
        # a fixed-basis SCBA representation (or a frozen-state validation),
        # never inferred from the sampled Sigma buffers.  Keeping the default
        # ``None`` makes the established RGF path bit-identical.
        self._auxiliary_channel = None
        self._auxiliary_rgf = None
        self._solver_max_batch_size = int(config.phonon.solver.max_batch_size)

        # GW-style self-consistent contacts: compute the OBC AFTER Sigma^R
        # is folded into the system matrix, dressing the periodic lead
        # superblocks with the boundary slab's scattering self-energy.
        self._obc_scattering_contacts = bool(
            getattr(config.phonon, "obc_scattering_contacts", False)
        )

        self.left_temperature = config.phonon.left_temperature
        self.right_temperature = config.phonon.right_temperature

        # The Bose occupation uses hbar*omega.
        hbar_omega_eV = thz_to_ev(np.abs(self.local_frequencies))
        self.left_occupancies = bose_einstein(hbar_omega_eV, self.left_temperature)
        self.right_occupancies = bose_einstein(hbar_omega_eV, self.right_temperature)
        # Regularize the omega=0 sampling point
        self.left_occupancies = xp.where(
            xp.isfinite(self.left_occupancies), self.left_occupancies, 0.0)
        self.right_occupancies = xp.where(
            xp.isfinite(self.right_occupancies), self.right_occupancies, 0.0)
        # NOTE: the full physical Bose occupation is kept at all omega > 0:
        # the lead broadening Gamma(omega) is odd (Gamma(0) = 0, ~omega for
        # acoustic modes), so Gamma*n stays finite as omega -> 0 even though
        # n ~ kT/(hbar*omega) diverges. Only the omega = 0 bin is clipped
        # above (its injection is ~0 since Gamma(0) = 0).

        self.obc_blocks = OBCBlocks(num_blocks=self.system_matrix.num_local_blocks)
        self.block_sections = config.phonon.obc.block_sections

        validate_obc_block_sections(
            self.system_matrix.num_blocks, comm.block.size
        )

        # Pole-subtracted SCBA sector. The pole set is a deterministic function
        # of the mixed self-energy, so it is recomputed every iteration and only
        # warm-started from the previous one -- it is deliberately NOT part of
        # the mixed state (that would make the Anderson/RRE least-squares
        # rank-deficient and invalidate the exact Newton JVP).
        _ps = getattr(config.phonon, "pole_sector", None)
        self._pole_enabled = bool(_ps is not None and _ps.enabled)
        self._pole_cfg = _ps
        self._pole = None            # PoleSector, built lazily
        self.pole_state = None       # last PoleSectorState, read by the interaction
        self._pole_q = None          # coupled-q: one PoleSector per q
        self.pole_q_states = []      # coupled-q: (q index, state) per solved q
        self.psd_report = {}         # last positivity gate result, if enabled
        self._psd_sigma = None       # Sigma buffers for the gate, if enabled
        # A congruence is PSD exactly, so anything above roundoff on the
        # normalised eigenvalue is structural rather than numerical. The
        # normalisation is global, so this is a single scale-free number.
        self._psd_tol = 1e-10

    @profiler.profile(label="PhononSolver: OBC", level="default", comm=comm)
    def _compute_obc(self) -> None:
        """Computes open boundary conditions."""
        one_block = self.system_matrix.num_blocks == 1
        left_contact = None
        if comm.block.rank == 0:
            if one_block:
                m_10, m_01 = self._single_block_periodic
                m_00 = self.system_matrix.blocks[0, 0]
            else:
                m_10, m_00, m_01 = get_periodic_superblocks(
                    a_ii=self.system_matrix.blocks[0, 0],
                    a_ji=self.system_matrix.blocks[1, 0],
                    a_ij=self.system_matrix.blocks[0, 1],
                    block_sections=self.block_sections,
                )
            g_00, *__ = self.obc(
                (m_00, m_01, m_10),
                contact="left",
            )
            # Apply the retarded boundary self-energy.
            sigma_00 = m_10 @ g_00 @ m_01
            if len(self.system_matrix.global_stack_shape) == 1:
                # Gamma-only (real-symmetric D): the exact contact Sigma^R is
                # complex-SYMMETRIC, but the NEVP eigenvector construction
                # breaks the symmetry, which propagates into G and breaks
                # the bosonic fold of the SSE. Project back onto the
                # symmetric subspace.
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
            left_contact = (
                self.obc_blocks.retarded[0],
                self.obc_blocks.lesser[0],
                self.obc_blocks.greater[0],
            )
        if comm.block.rank == comm.block.size - 1:
            n = self.system_matrix.num_local_blocks - 1
            m = n - 1

            if one_block:
                # This is exactly the result of the established flip/get/flip
                # construction with a_ji=m_01 and a_ij=m_10 at
                # block_sections=1.
                m_mn = self._single_block_periodic[1]
                m_nn = self.system_matrix.blocks[0, 0]
                m_nm = self._single_block_periodic[0]
            else:
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
                    xp.flip(m_nn, axis=(-2, -1)),
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

            if one_block:
                right_contact = (
                    self.obc_blocks.retarded[-1],
                    self.obc_blocks.lesser[-1],
                    self.obc_blocks.greater[-1],
                )
                if left_contact is None:
                    raise RuntimeError(
                        "one-block OBC requires block_comm_size=1"
                    )
                self._single_block_contacts = (left_contact, right_contact)
                # The Green function sees the sum of both reservoirs on the
                # same device block.  Keep the individual triples above for
                # the two separate Meir-Wingreen lead currents.
                self.obc_blocks.retarded[0] = (
                    left_contact[0] + right_contact[0]
                )
                self.obc_blocks.lesser[0] = (
                    left_contact[1] + right_contact[1]
                )
                self.obc_blocks.greater[0] = (
                    left_contact[2] + right_contact[2]
                )

    @profiler.profile(label="PhononSolver: Assemble", level="default", comm=comm)
    def _assemble_system_matrix(self) -> None:
        """Assembles the HARMONIC system matrix (no scattering self-energy).

        The scattering self-energy is subtracted in :meth:`solve` AFTER the
        open boundary conditions are computed: the contacts are ideal
        (harmonic, ballistic) reservoirs, so the lead surface Green's
        function must be built from the bare blocks. Including the device's
        scattering Sigma^R in the OBC input both extends the device
        scattering periodically into the semi-infinite lead (unphysical)
        and destabilises the spectral NEVP once Sigma grows.
        """
        self.system_matrix.allocate_data()
        self.system_matrix.data = 0.0

        self.system_matrix.fill_diagonal(1.0)
        scale_stack(self.system_matrix.data, self.local_frequencies ** 2)

        _btd_subtract(self.system_matrix, self.dynamical_matrix)

    @profiler.profile(label="PhononSolver: Selected Solve", level="default", comm=comm)
    def _selected_solve(
        self,
        sse_lesser: DSDBSparse,
        sse_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...]
    ) -> None:
        """Perform selected solve for the phonon Green's function."""
        if self._auxiliary_channel is not None:
            if comm.block.size > 1:
                raise NotImplementedError(
                    "the local auxiliary RGF requires block_comm_size == 1; "
                    "the distributed augmented recurrence is not implemented")
            if self._auxiliary_rgf is None:
                from quatrex.phonon.experimental.auxiliary_scba import (
                    GlobalAuxiliaryWoodbury,
                    LocalAuxiliaryChannel,
                    LocalAuxiliaryRGF,
                )

                cls = (LocalAuxiliaryRGF
                       if isinstance(self._auxiliary_channel,
                                     LocalAuxiliaryChannel)
                       else GlobalAuxiliaryWoodbury)
                self._auxiliary_rgf = cls(
                    self._auxiliary_channel if cls is LocalAuxiliaryRGF
                    else self._auxiliary_channel,
                    self.local_frequencies,
                    max_batch_size=self._solver_max_batch_size,
                    n_offdiagonals=self._gf_band)
            self.meir_wingreen_current = self._auxiliary_rgf.selected_solve(
                self.system_matrix, sse_lesser, sse_greater, out,
                obc_blocks=self.obc_blocks, return_retarded=True,
                return_current=self.compute_meir_wingreen_current)
            return
        extra_kw = (
            {"n_offdiagonals": self._gf_band} if self._gf_band >= 2 else {}
        )
        one_block_contacts = self._single_block_contacts
        solver_current = (
            self.compute_meir_wingreen_current
            and one_block_contacts is None
        )
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
                return_current=solver_current,
                **extra_kw,
            )
        else:
            self.meir_wingreen_current = self.solver.selected_solve(
                a=self.system_matrix,
                sigma_lesser=sse_lesser,
                sigma_greater=sse_greater,
                obc_blocks=self.obc_blocks,
                out=out,
                return_retarded=True,
                return_current=solver_current,
                **extra_kw,
            )
        if self.compute_meir_wingreen_current and one_block_contacts is not None:
            self.meir_wingreen_current = self._one_block_lead_current(
                out[0].blocks[0, 0], out[1].blocks[0, 0],
                one_block_contacts,
            )

    @staticmethod
    def _one_block_lead_current(
        g_lesser: NDArray,
        g_greater: NDArray,
        contacts: tuple[tuple[NDArray, NDArray, NDArray],
                        tuple[NDArray, NDArray, NDArray]],
    ) -> NDArray:
        """Two separate lead currents when both reservoirs touch one block.

        The selected solve uses the sum of the two contact self-energies.  A
        lead current must instead use the injection of that lead alone.  The
        signs and ordering are the same as the two end-point expressions in
        :class:`qttools.greens_function_solver.rgf.RGF`.
        """
        left, right = contacts
        current = xp.zeros(g_lesser.shape[:-2] + (2,), dtype=g_lesser.dtype)
        current[..., 0] = xp.trace(
            left[2] @ g_lesser - g_greater @ left[1],
            axis1=-2, axis2=-1,
        )
        current[..., 1] = -xp.trace(
            right[2] @ g_lesser - g_greater @ right[1],
            axis1=-2, axis2=-1,
        )
        return current

    def set_auxiliary_channel(self, channel) -> None:
        r"""Install a fixed-basis rational self-energy for the next solves.

        ``channel`` must be a
        :class:`quatrex.phonon.experimental.auxiliary_scba.LocalAuxiliaryChannel`, or
        ``None`` to restore the ordinary RGF.  This method deliberately does
        not project a sampled self-energy or mix two changing pole sets.  A
        self-consistent caller must hold the basis fixed over an SCBA epoch and
        pass the rational coefficients through the same mixer as the smooth
        grid component.  Installing an independently updated sidecar would be
        a non-conserving change of the fixed-point map.
        """
        if channel is not None:
            from quatrex.phonon.experimental.auxiliary_scba import (
                LocalAuxiliaryChannel, RationalKeldyshChannel)

            if not isinstance(channel, (LocalAuxiliaryChannel,
                                        RationalKeldyshChannel)):
                raise TypeError(
                    "channel must be a LocalAuxiliaryChannel, a global "
                    "RationalKeldyshChannel, or None")
            if isinstance(channel, LocalAuxiliaryChannel) and not np.array_equal(
                    channel.block_sizes, self.block_sizes):
                raise ValueError(
                    "auxiliary channel block sizes do not match the solver")
            if isinstance(channel, RationalKeldyshChannel) and \
                    channel.n_dof != int(np.sum(self.block_sizes)):
                raise ValueError(
                    "global auxiliary channel does not span the solver DOFs")
        self._auxiliary_channel = channel
        self._auxiliary_rgf = None

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

        self._assemble_system_matrix()

        if self._obc_scattering_contacts:
            # GW-style self-consistent contacts: fold Sigma^R into the
            # system matrix FIRST, so the periodic lead superblocks (built
            # from the boundary blocks) carry the boundary slab's
            # scattering self-energy, and the contact injection is the
            # fluctuation-dissipation pair of the dressed escape rate.
            # At iteration 0 (Sigma = 0) this degenerates to bare leads.
            _btd_subtract(self.system_matrix, sse_retarded)
            self._compute_obc()
        else:
            # OBC from the bare harmonic blocks (ideal-reservoir
            # contacts); the scattering self-energy enters the device
            # Dyson only.
            self._compute_obc()
            _btd_subtract(self.system_matrix, sse_retarded)
        self._selected_solve(sse_lesser, sse_greater, out)

        # Must run BEFORE free_data(): the pole solve reads the assembled
        # operator's blocks.
        self._update_pole_sector(sse_lesser, sse_greater, out[2], out[0])
        self._psd_sigma = (sse_lesser, sse_greater)
        self._check_positivity(out)
        self._psd_sigma = None
        n_freq = int(self.local_frequencies.shape[0])
        self._psd_sigma_lesser = sse_lesser.data.reshape(n_freq, -1)
        self._report_subcell(out)
        self._psd_sigma_lesser = None

        self.system_matrix.free_data()
