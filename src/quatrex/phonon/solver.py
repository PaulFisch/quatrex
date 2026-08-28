# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import warnings

import numpy as np
from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.profiling import Profiler
from qttools.toeplitz.toeplitz import get_periodic_superblocks
from qttools.utils.gpu_utils import get_host
from qttools.utils.mpi_utils import distributed_load, get_local_slice, get_section_sizes
from qttools.utils.stack_utils import scale_stack

from quatrex.core.config import QuatrexConfig
from quatrex.core.statistics import bose_einstein
from quatrex.core.subsystem import SubsystemSolver
from quatrex.device.inputs import load_matrix
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


def _q_block(arr, idx):
    """One q of a contact block, whose transverse axes follow frequency.

    ``arr`` is ``(n_freq,) + nk + (b, b)``. Indexing it with a bare ``idx``
    would take frequency first, which is the same off-by-one that made the
    dynamical matrix return a stack where a block was wanted.
    """
    if arr is None:
        return None
    return arr[(slice(None),) + tuple(idx)]


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
        frequencies: NDArray,
        sparsity_pattern: sparse.coo_matrix,
    ) -> None:
        """Initializes the Phonon solver."""
        super().__init__(config, frequencies)

        if float(np.min(frequencies)) < 0.0:
            raise ValueError(
                "The phonon solver requires a non-negative frequency grid: "
                "lead occupancies and damping are built as even functions "
                "of omega (the negative axis is handled by the bosonic fold "
                "in the SSE, not by the grid)."
            )
        self.local_frequencies = get_local_slice(frequencies, comm.stack)
        # Per-bin quadrature cell widths + uniformity flag: the heat/energy
        # integrals (SCBA conservation gates, engine snapshots) weight by
        # these on a NON-UNIFORM grid; on a uniform grid they are the
        # constant dw and the legacy unweighted sums are kept.
        from quatrex.grid.energies import (
            frequency_cell_widths, is_uniform_grid)
        self.local_frequency_weights = get_local_slice(
            frequency_cell_widths(frequencies), comm.stack)
        self.uniform_frequency_grid = is_uniform_grid(frequencies)
        self._ir_floor_diag = None

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
        self._gf_band = min(
            int(getattr(config.phonon, "sse_g_band", 1) or 1),
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

        # frequencies are the linear-frequency grid in THz (uniform unless
        # the SSE runs on the auxiliary bubble grid, sse_aux_grid_dw_thz);
        # the Bose occupation needs hbar*omega.
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

        self.eta = config.phonon.eta
        self.eta_obc = config.phonon.eta_obc
        self._eta_ir_floor_c = float(getattr(config.phonon,
                                             "eta_ir_floor_cells", 0.0))
        # Optional in-SCBA anneal of the sub-grid soft-mode floor: ramp
        # eta_ir_floor_cells DOWN from its start value to
        # eta_ir_floor_final_cells over eta_ir_floor_ramp_iterations solves,
        # then hold.
        self._eta_ir_floor_c0 = self._eta_ir_floor_c
        self._eta_ir_floor_final = float(getattr(config.phonon,
                                                 "eta_ir_floor_final_cells", 0.0))
        self._eta_ir_floor_ramp_n = int(getattr(config.phonon,
                                                "eta_ir_floor_ramp_iterations", 0))
        self._eta_ir_floor_it = 0
        # Optional in-SCBA broadening anneal: ramp eta DOWN from eta0 (iteration
        # 0, while Sigma^R~0) to eta_final over eta_ramp_iterations solves, then
        # hold. Lets the anharmonic Sigma^R take over the broadening (eta=0 limit).
        self._eta0 = float(self.eta)
        self._eta_final = float(getattr(config.phonon, "eta_final", 0.0))
        self._eta_ramp_n = int(getattr(config.phonon, "eta_ramp_iterations", 0))
        self._eta_it = 0
        # Optional in-SCBA CONTACT-broadening ramp: ramp eta_obc DOWN from
        # eta_obc0 (large enough to converge the cell cold) to eta_obc_final
        # over eta_obc_ramp_iterations solves, then hold.
        self._eta_obc0 = float(self.eta_obc)
        self._eta_obc_final = float(getattr(config.phonon, "eta_obc_final", self.eta_obc))
        self._eta_obc_ramp_n = int(getattr(config.phonon, "eta_obc_ramp_iterations", 0))
        self._eta_obc_it = 0

        # Optional self-consistent Buttiker dephasing probe on the eta channel
        # (config.phonon.buttiker_probe): adds the matching fluctuation
        # Sigma_probe^{<,>} = i*4*eta*omega*(n_p + 0/1) that the bare eta
        # broadening lacks, with n_p self-consistent for zero local current.
        self._buttiker = bool(getattr(config.phonon, "buttiker_probe", False))
        self._probe_np = None      # (*stack, n_dof) occupation, lazy
        self._probe_diag = None    # (diag nnz positions, their DOF), lazy
        self._probe_added = None   # cached (probe_l, probe_g) for restore

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
        if comm.block.rank == 0:
            m_10, m_00, m_01 = get_periodic_superblocks(
                a_ii=self.system_matrix.blocks[0, 0],
                a_ji=self.system_matrix.blocks[1, 0],
                a_ij=self.system_matrix.blocks[0, 1],
                block_sections=self.block_sections,
            )
            s_00 = 1j * self.eta_obc * xp.eye(
                m_00.shape[-1], dtype=self.dynamical_matrix.dtype)
            if self._ir_floor_diag is not None:
                # The IR floor regularises the DEVICE Dyson only; strip it
                # from the lead blocks so the reservoirs stay bare harmonic.
                s_00 = s_00 - self._ir_floor_diag.reshape(
                    (-1,) + (1,) * (m_00.ndim - 1)
                ) * xp.eye(m_00.shape[-1], dtype=self.dynamical_matrix.dtype)

            g_00, *__ = self.obc(
                (m_00 + s_00, m_01, m_10),
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
        if comm.block.rank == comm.block.size - 1:
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
            s_nn = 1j * self.eta_obc * xp.eye(
                m_nn.shape[-1], dtype=self.dynamical_matrix.dtype)
            if self._ir_floor_diag is not None:
                s_nn = s_nn - self._ir_floor_diag.reshape(
                    (-1,) + (1,) * (m_nn.ndim - 1)
                ) * xp.eye(m_nn.shape[-1], dtype=self.dynamical_matrix.dtype)
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

    def _apply_eta_ramp(self) -> None:
        """In-SCBA broadening anneal (no-op unless ``eta_ramp_iterations>0``).

        Linearly ramps ``self.eta`` from ``eta0`` (iteration 0) to ``eta_final``
        over ``eta_ramp_iterations`` solves, then holds it there. Called once per
        :meth:`solve` (= once per SCBA iteration); the first solve keeps the full
        eta0 while Sigma^R is still ~0, and as Sigma^R develops the broadening is
        turned off, approaching the eta -> eta_final limit.
        """
        if self._eta_ramp_n <= 0:
            return
        frac = min(1.0, self._eta_it / float(self._eta_ramp_n))
        self.eta = self._eta0 + (self._eta_final - self._eta0) * frac
        self._eta_it += 1
        if comm.rank == 0:
            print(f"eta ramp: it={self._eta_it - 1} eta={self.eta:.4g} "
                  f"({self._eta0:.4g} -> {self._eta_final:.4g} over "
                  f"{self._eta_ramp_n})", flush=True)

    def _apply_eta_ir_floor_ramp(self) -> None:
        """In-SCBA anneal of the sub-grid soft-mode floor (no-op unless
        eta_ir_floor_ramp_iterations>0). Linearly ramps eta_ir_floor_cells from
        its start to eta_ir_floor_final_cells over the ramp, then holds. Recomputed
        every solve; _assemble_system_matrix reads the current self._eta_ir_floor_c.
        """
        if self._eta_ir_floor_ramp_n <= 0:
            return
        frac = min(1.0, self._eta_ir_floor_it / float(self._eta_ir_floor_ramp_n))
        self._eta_ir_floor_c = (self._eta_ir_floor_c0
                                + (self._eta_ir_floor_final - self._eta_ir_floor_c0) * frac)
        self._eta_ir_floor_it += 1
        if comm.rank == 0:
            print(f"eta IR floor ramp: it={self._eta_ir_floor_it - 1} "
                  f"floor_cells={self._eta_ir_floor_c:.4g} "
                  f"({self._eta_ir_floor_c0:.4g} -> {self._eta_ir_floor_final:.4g} "
                  f"over {self._eta_ir_floor_ramp_n})", flush=True)

    def _apply_eta_obc_ramp(self) -> None:
        """In-SCBA contact-broadening ramp (no-op unless eta_obc_ramp_iterations>0).

        Linearly ramps ``self.eta_obc`` from ``eta_obc0`` (iteration 0, large
        enough to converge the cell cold) to ``eta_obc_final`` over
        ``eta_obc_ramp_iterations`` solves, then holds. The contact
        regularisation is strong while Sigma is still developing and relaxes
        toward the (small) target.
        """
        if self._eta_obc_ramp_n <= 0:
            return
        frac = min(1.0, self._eta_obc_it / float(self._eta_obc_ramp_n))
        self.eta_obc = self._eta_obc0 + (self._eta_obc_final - self._eta_obc0) * frac
        self._eta_obc_it += 1
        if comm.rank == 0:
            print(f"eta_obc ramp: it={self._eta_obc_it - 1} eta_obc={self.eta_obc:.4g} "
                  f"({self._eta_obc0:.4g} -> {self._eta_obc_final:.4g} over "
                  f"{self._eta_obc_ramp_n})", flush=True)

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

        # (omega + i*eta)^2 * I  (THz^2). The phonon Dyson equation is
        # [ (omega+i eta)^2 - D - Sigma ] G = I, with D the dynamical matrix
        # and Sigma the scattering self-energy, all in THz^2
        self.system_matrix.fill_diagonal(1.0)
        # omega^2 + 2i*eta*omega: frequency-proportional damping WITHOUT the
        # -eta^2 band-edge shift of (omega+i*eta)^2 (which makes omega <~ eta
        # artificially evanescent and suppresses the acoustic plateau).
        z2 = (self.local_frequencies ** 2
              + 2j * self.eta * xp.abs(self.local_frequencies))
        # Sub-grid soft-mode broadening floor (eta_ir_floor_cells): the
        # 2i*eta*omega damping vanishes as omega->0, leaving the acoustic
        # soft modes (D->0) unregularised at eta=0. Add a DC-concentrated
        # constant broadening i*Gamma_floor*lowpass(omega) that damps only
        # the lowest (unresolved, ~zero-heat) bins; Gamma_floor->0 as dw->0
        # (grid-consistent). NOT applied to the OBC (ideal leads).
        _ir_floor_c = self._eta_ir_floor_c
        self._ir_floor_diag = None
        if _ir_floor_c > 0.0 and self.energies.size > 1:
            # On a non-uniform grid this is the width of the FIRST cell --
            # exactly the near-DC resolution the sub-grid soft-mode floor
            # is meant to track (uniform grids: the usual dw).
            _dw = float(abs(get_host(self.energies[1] - self.energies[0])))
            _w = xp.asarray(self.local_frequencies, dtype=float)
            _gamma = (_ir_floor_c * _dw) ** 2          # THz^2
            _wc2 = (2.0 * _dw) ** 2
            self._ir_floor_diag = 1j * _gamma * _wc2 / (_w ** 2 + _wc2)
            z2 = z2 + self._ir_floor_diag
            if comm.rank == 0:
                print(f"eta IR floor ON: Gamma_floor={_gamma:.4g} THz^2 "
                      f"(eta_ir_floor_cells={_ir_floor_c:g}, omega_c=2*dw="
                      f"{2*_dw:.3g} THz) on the unresolved soft modes.", flush=True)
        scale_stack(self.system_matrix.data, z2)

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
                from quatrex.phonon.auxiliary_scba import (
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
                return_current=self.compute_meir_wingreen_current,
                **extra_kw,
            )

    def set_auxiliary_channel(self, channel) -> None:
        r"""Install a fixed-basis rational self-energy for the next solves.

        ``channel`` must be a
        :class:`quatrex.phonon.auxiliary_scba.LocalAuxiliaryChannel`, or
        ``None`` to restore the ordinary RGF.  This method deliberately does
        not project a sampled self-energy or mix two changing pole sets.  A
        self-consistent caller must hold the basis fixed over an SCBA epoch and
        pass the rational coefficients through the same mixer as the smooth
        grid component.  Installing an independently updated sidecar would be
        a non-conserving change of the fixed-point map.
        """
        if channel is not None:
            from quatrex.phonon.auxiliary_scba import (
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

    def _probe_indices(self, sse_lesser: DSDBSparse):
        """Lazily cache the diagonal nnz positions and their device-DOF index
        (where rows == cols), for the Buttiker probe diagonal."""
        if self._probe_diag is not None:
            return self._probe_diag
        rows = getattr(sse_lesser, "rows", None)
        cols = getattr(sse_lesser, "cols", None)
        if rows is None or cols is None:
            self._buttiker = False
            warnings.warn("buttiker_probe disabled: sparsity has no rows/cols.",
                          stacklevel=2)
            return None
        r = np.asarray(get_host(rows)); c = np.asarray(get_host(cols))
        diag = np.where(r == c)[0]
        self._probe_diag = (xp.asarray(diag), xp.asarray(r[diag]))
        return self._probe_diag

    def _apply_buttiker_probe(self, sse_lesser, sse_greater) -> None:
        """Add Sigma_probe^{<,>} = i*4*eta*|omega|*(n_p + 0/1) to the device
        source on the diagonal (no-op until n_p exists, i.e. from iteration 1)."""
        self._probe_added = None
        if not self._buttiker or self._probe_np is None:
            return
        if comm.block.size > 1:
            self._buttiker = False
            warnings.warn("buttiker_probe requires block_comm_size==1; disabled.",
                          stacklevel=2)
            return
        idx = self._probe_indices(sse_lesser)
        if idx is None:
            return
        diag, dof = idx
        w4 = 4.0 * self.eta * xp.abs(xp.asarray(self.local_frequencies))
        np_diag = xp.take(self._probe_np, dof, axis=-1)      # (*stack, n_diag)
        probe_l = scale_stack(1j * np_diag.copy(), w4)
        probe_g = scale_stack(1j * (np_diag + 1.0), w4)
        sse_lesser.data[..., diag] += probe_l
        sse_greater.data[..., diag] += probe_g
        self._probe_added = (diag, probe_l, probe_g)

    def _restore_buttiker_probe(self, sse_lesser, sse_greater) -> None:
        """Remove the probe diagonal so the SCBA's tracked Sigma stays clean."""
        if self._probe_added is None:
            return
        diag, probe_l, probe_g = self._probe_added
        sse_lesser.data[..., diag] -= probe_l
        sse_greater.data[..., diag] -= probe_g
        self._probe_added = None

    def _update_buttiker_probe(self, out: tuple[DSDBSparse, ...]) -> None:
        """Refresh the per-DOF, per-omega probe occupation from the device G:
        n_p = (-i G^<)_dd / (-i (G^> - G^<))_dd  (zero local current per energy)."""
        if not self._buttiker or comm.block.size > 1:
            return
        g_lesser, g_greater = out[0], out[1]
        idx = self._probe_indices(g_lesser)
        if idx is None:
            return
        diag, dof = idx
        gl = g_lesser.data[..., diag]
        gg = g_greater.data[..., diag]
        spectral = (-1j * (gg - gl)).real
        nloc = (-1j * gl).real / xp.where(spectral > 1e-30, spectral, 1e-30)
        nloc = xp.clip(nloc, 0.0, None)
        if self._probe_np is None:
            stack = g_lesser.data.shape[:-1]
            ndof = int(get_host(dof).max()) + 1
            self._probe_np = xp.zeros(stack + (ndof,), dtype=float)
        # diagonal DOFs are unique -> direct scatter (light damping for stability)
        new = xp.array(self._probe_np, copy=True)
        new[..., dof] = nloc
        self._probe_np = 0.5 * self._probe_np + 0.5 * new

    def _band_edges_for(self, index_slice=()):
        """Lead band edges (THz) for one q, or ``None`` when the gate is off.

        Homogenised from the DYNAMICAL matrix by the same
        ``get_periodic_superblocks`` the OBC applies to the system matrix, so
        the edges are the branch points of the contact the operator carries.
        Taking them from the system matrix instead would mean removing the
        ``z^2 I`` the phonon Dyson operator adds, at a frequency that would
        then have to be chosen.

        Frequency independent and cached per q: ``D`` does not move during the
        SCBA, so this runs once per q per run.
        """
        from quatrex.phonon.pole_sector import lead_band_edges

        if getattr(self._pole_cfg, "band_edges", "none") != "lead":
            return None
        key = tuple(int(i) for i in index_slice)
        cache = getattr(self, "_band_edge_cache", None)
        if cache is None:
            cache = self._band_edge_cache = {}
        if key in cache:
            return cache[key]

        blocks = self._pole_blocks(self.dynamical_matrix,
                                   index_slice=index_slice)
        if (1, 0) not in blocks:
            # A single-block device has no periodic layer to homogenise.
            cache[key] = None
            return None
        d_10, d_00, d_01 = get_periodic_superblocks(
            a_ii=blocks[(0, 0)], a_ji=blocks[(1, 0)], a_ij=blocks[(0, 1)],
            block_sections=self.block_sections,
        )

        def _one(b):
            a = xp.asarray(b)
            return a.reshape(a.shape[-2:])      # drop any leading singletons

        edges = lead_band_edges(_one(d_00), _one(d_01), _one(d_10))
        if comm.rank == 0 and not cache:
            print(f"  pole sector: {edges.size} lead band edges "
                  f"[{edges.min():.3f}, {edges.max():.3f}] THz feed edge_factor="
                  f"{self._pole_cfg.edge_factor:g}", flush=True)
        cache[key] = edges
        return edges

    def _pole_blocks(self, matrix, index_slice=()):
        """Dense block-tridiagonal view of a DSDBSparse for one stack element.

        ``index_slice`` addresses the TRANSVERSE q axes, which sit at the END
        of the stack shape. The dynamical matrix carries a leading singleton
        where the Keldysh buffers carry frequency, so a bare ``(i, j)`` lands on
        the wrong axes: on a ``(1, 9, 9)`` stack it consumed the singleton and
        one q index, leaving ``(9, 6, 6)`` where a block was wanted, and every
        ``i > 0`` raised "Index 1 is out of bounds for axis 0 with size 1".
        Padding on the LEFT puts the q indices where they belong whatever the
        buffer's leading rank is.
        """
        if index_slice:
            rank = len(getattr(matrix, "global_stack_shape", ()) or ())
            pad = max(0, rank - len(index_slice))
            index_slice = (0,) * pad + tuple(index_slice)
        view = matrix.stack[index_slice] if index_slice else matrix.stack[...]
        n = matrix.num_local_blocks
        out = {}
        for i in range(n):
            for j in range(max(0, i - 1), min(n, i + 2)):
                out[(i, j)] = view.blocks[i, j]
        return out

    @profiler.profile(label="PhononSolver: Positivity", level="default", comm=comm)
    def _check_positivity(self, out: tuple) -> None:
        """Positivity gate on the RECONSTRUCTED TOTAL ``G^{<,>}``.

        ``bubble_positivity.md`` Thm 1-2: the solver stores ``-i G^{<,>} >= 0``,
        and the bubble is a congruence of it. The gate is on the total and
        never on a sector -- ``G_PP`` and ``G_BB`` are separately PSD, but
        ``G_PP + G_PB + G_BP`` is not, so a per-sector check would report
        violations that are not there.

        Off by default (``pole_sector.psd_check``): it costs a batched
        eigen-decomposition per block window. This closes
        ``bubble_positivity.md``'s open item "a production positivity gate
        behind a flag".
        """
        cfg = getattr(self.config.phonon, "pole_sector", None)
        if cfg is None or not getattr(cfg, "psd_check", False):
            return
        from quatrex.phonon.pole_audit import psd_residual

        # Sigma is checked as well as G, and it is the ROOT check:
        # G^< = G^R Sigma^< G^A is a congruence, so a PSD Sigma cannot produce
        # a non-PSD G. If G^< fails, Sigma^< must have failed first, and
        # reporting only G would send the search to the wrong place.
        n_freq = int(self.local_frequencies.shape[0])
        # BOTH lesser and greater carry sign -1. This solver uses the
        # occupation-positive convention sigma^{<,>} = +i n(+1) Gamma, so
        # -i sigma^< = n Gamma >= 0 AND -i sigma^> = (n+1) Gamma >= 0
        # (bubble_positivity.md: "the solver stores -i G^{<,>} >= 0").
        # The textbook convention has +i G^> >= 0, and borrowing it here made
        # the gate report worst = -1.000 -- uniformly negative -- on the
        # pole-free baseline, which is what a flipped sign looks like rather
        # than a physics result.
        targets = [("g_lesser", out[0], -1.0), ("g_greater", out[1], -1.0)]
        if self._psd_sigma is not None:
            sl, sg = self._psd_sigma
            targets = [("sigma_lesser", sl, -1.0),
                       ("sigma_greater", sg, -1.0)] + targets
        for name, buf, sign in targets:
            rep = psd_residual(
                buf.data.reshape(n_freq, -1), buf.rows, buf.cols,
                self.block_sizes, sign=sign,
            )
            self.psd_report[name] = rep
            if comm.rank == 0:
                flag = "VIOLATION" if rep["worst"] < -self._psd_tol else "ok"
                print(f"  positivity {name:15s} worst={rep['worst']:+.3e} "
                      f"at w[{rep['omega_index']}]  {flag}", flush=True)

    def _pole_frequency_context(self, local_freqs) -> dict:
        """Global grid, local offset and reducer for the pole continuation.

        The continuation sums over the WHOLE frequency axis, which
        ``comm.stack`` splits. Both of its terms are linear in ``Delta`` with
        grid-only coefficients and neither reindexes frequency, so each rank
        can contract its own columns and a single sum-reduction completes the
        result -- no transposition of the distributed buffer, and no
        distributed root finding: every rank ends up with the same operator
        and solves the same poles.

        Serial runs get the identity reducer, so the path is bit-identical to
        the undistributed one rather than merely equivalent.
        """
        if comm.stack.size <= 1:
            return {}

        # comm.stack.all_gather is IN-PLACE -- all_gather(sendbuf, recvbuf)
        # returning None -- and every rank must send the same count, which the
        # frequency split does not guarantee. So gather the counts first, then
        # pad to the max and trim on the way out. Calling it as if it returned
        # a value raised "missing 1 required positional argument: 'recvbuf'"
        # on every rank, which made every multi-rank pole run dead on arrival.
        n_ranks = comm.stack.size
        sizes = np.empty(n_ranks, dtype=np.int64)
        comm.stack.all_gather(np.array([local_freqs.size], dtype=np.int64),
                              sizes)
        sizes = np.asarray(get_host(sizes), dtype=np.int64).ravel()
        offset = int(sizes[:comm.stack.rank].sum())

        n_max = int(sizes.max())
        send = np.zeros(n_max, dtype=float)
        send[:local_freqs.size] = np.asarray(
            get_host(local_freqs), dtype=float).ravel()
        recv = np.empty(n_max * n_ranks, dtype=float)
        comm.stack.all_gather(send, recv)
        recv = np.asarray(get_host(recv), dtype=float).reshape(n_ranks, n_max)
        global_freqs = np.concatenate(
            [recv[r, :int(sizes[r])] for r in range(n_ranks)])

        def _reduce(arr):
            send = xp.ascontiguousarray(arr)
            recv = xp.empty_like(send)
            comm.stack.all_reduce(send, recv, op="sum")
            return recv

        return {"global_freqs": global_freqs, "freq_offset": offset,
                "reduce": _reduce}

    @profiler.profile(label="PhononSolver: Pole sector", level="default", comm=comm)
    def _update_pole_sector(self, sse_lesser, sse_greater,
                            g_retarded=None, g_lesser=None) -> None:
        """Refresh the pole set from the current (mixed) self-energy.

        Runs in the ``"stack"`` distribution state, right after the selected
        solve and before the system matrix is released. Rebuilds
        ``Delta = Sigma^> - Sigma^<`` from the buffers the Dyson operator was
        just built from -- never from a cache, which could drift out of step
        with ``Sigma^R`` across the mixer -- continues it to complex frequency,
        and corrects the previous iterate's poles.

        Staging note: the frequency axis is split across ``comm.stack``, while
        the continuation sums over ALL frequencies. The distributed form
        contracts in the ``"nnz"`` state, where every rank owns the whole axis,
        and is deferred; until it lands this refuses a split stack rather than
        silently continuing an incomplete self-energy.
        """
        if not self._pole_enabled:
            return

        from quatrex.phonon.pole_probe import delta_from_sigma
        from quatrex.phonon.pole_sector import PoleSector

        if comm.block.size > 1:
            raise NotImplementedError(
                "pole_sector: block-distributed devices are not supported yet "
                "(the pole solve needs the whole operator on one rank)."
            )

        freqs = np.asarray(get_host(self.local_frequencies), dtype=float)
        if self._pole is None:
            self._pole = PoleSector(
                self._pole_cfg, freqs,
                **self._pole_frequency_context(freqs),
            )

        delta = delta_from_sigma(sse_lesser.data, sse_greater.data)
        if float(xp.abs(delta).max()) == 0.0:
            # Iteration 0: no scattering yet, so every pole sits on the real
            # axis and none is a resonance. Nothing to promote.
            self.pole_state = None
            return

        # A q-resolved device has one pole problem PER q: M_q(z) = z^2 I -
        # D(q) - Sigma^R_q(z), and the sets are unrelated. Allocating a sector
        # from them additionally needs the vertex fold (the bubble at q sums
        # over q' and q - q'), which is not built -- but the CENSUS needs none
        # of that, and the census is what says whether coupled-q is worth
        # building at all. So extraction-only walks the q axis; the allocating
        # path still refuses it in set_operator_context.
        nq = 1
        if delta.ndim > 2:
            nq = int(np.prod(delta.shape[1:-1]))
        if nq > 1 and getattr(self._pole_cfg, "extraction_only", False):
            self._census_over_q(delta, sse_lesser)
            self.pole_state = None
            return
        if nq > 1:
            self._update_pole_sector_q(delta, sse_lesser, sse_greater,
                                       g_retarded, g_lesser)
            return

        self._pole.set_operator_context(
            band_edges=self._band_edges_for(),
            delta=delta,
            d_blocks=self._pole_blocks(self.dynamical_matrix),
            obc_left=(self.obc_blocks.retarded[0]
                      if self.obc_blocks.retarded[0] is not None else None),
            obc_right=(self.obc_blocks.retarded[-1]
                       if self.obc_blocks.retarded[-1] is not None else None),
            block_sizes=self.block_sizes,
            rows=sse_lesser.rows,
            cols=sse_lesser.cols,
        )
        with profiler.profile_range("PhononSolver: Pole solve", "default",
                                    comm):
            self.pole_state = self._pole.refresh()
        with profiler.profile_range("PhononSolver: Pole legs", "default",
                                    comm):
            if getattr(self._pole_cfg, "leg", "congruence") == "congruence":
                # Same batched path as the q-resolved route, with one q, so
                # both are the same code and every single-q test covers it.
                self._pole_layout = getattr(self._pole, "_layout", None)
                state = self.pole_state
                if self._pole_layout is not None and state is not None \
                        and state.clusters:
                    leg_l, leg_g = self._build_pole_legs(
                        sse_lesser, sse_greater, g_retarded, [state],
                        [self._pole], np.array([0], dtype=int))
                    if leg_l is not None:
                        state.g_pp_lesser = leg_l[0].reshape(
                            sse_lesser.data.shape)
                        state.g_pp_greater = leg_g[0].reshape(
                            sse_greater.data.shape)
                    self._report_pole_registration(
                        [state], g_lesser, state.g_pp_lesser,
                        np.array([0], dtype=int))
            else:
                self._build_pole_keldysh(sse_lesser, sse_greater, g_retarded,
                                         g_lesser)
        if comm.rank == 0 and self.pole_state is not None:
            print(self.pole_state.report(), flush=True)

    def _q_indices(self, nq: int, shape):
        """Which q to solve, honouring ``q_stride`` and ``q_max``."""
        sel = list(range(0, nq, max(1, int(getattr(self._pole_cfg,
                                                   "q_stride", 1)))))
        cap = int(getattr(self._pole_cfg, "q_max", 0) or 0)
        if cap:
            sel = sel[:cap]
        return [(iq, tuple(int(i) for i in np.unravel_index(iq, shape)))
                for iq in sel]

    def _update_pole_sector_q(self, delta, sse_lesser, sse_greater,
                              g_retarded, g_lesser) -> None:
        """Coupled-q, Stage 1: one pole problem per q, one leg per q.

        ``leg="congruence"`` adds no analytic sector -- its whole action is to
        modify the leg the ring convolves, and the ring performs the q fold
        itself, downstream. So the sector is per-q and UNCOUPLED here, and the
        vertex fold the guard in ``set_operator_context`` speaks of is needed
        only where sectors are restored analytically (there
        ``Sigma_q = sum_q' B[G_q', G_{q-q'}]`` pairs pole sets across q). Those
        routes stay refused, by that guard, which still sees only a slice.

        One ``PoleSector`` PER q. The tracker, the promoted set and the epoch
        counter are per-q: a mode at one q has no relation to one at another,
        so a shared tracker would match them across q and either fuse two
        unrelated modes or churn membership every iteration -- exactly what the
        hysteresis exists to prevent.
        """
        from quatrex.phonon.pole_probe import BlockLayout
        from quatrex.phonon.pole_sector import PoleSector, refresh_many

        if getattr(self._pole_cfg, "leg", "congruence") != "congruence":
            raise NotImplementedError(
                f"pole_sector: leg={self._pole_cfg.leg!r} restores analytic "
                "sectors beside the ring, and those pair pole sets from "
                "DIFFERENT q (Sigma_q sums over q' and q-q'). That fold is not "
                "built. Use leg='congruence' on a q-resolved device, or "
                "extraction_only=True for a census."
            )

        freqs = np.asarray(get_host(self.local_frequencies), dtype=float)
        shape = tuple(int(k) for k in delta.shape[1:-1])
        nq = int(np.prod(shape))
        d_flat = delta.reshape(delta.shape[0], nq, delta.shape[-1])
        todo = self._q_indices(nq, shape)

        if self._pole_q is None:
            self._pole_q = {}
        acc_l = xp.zeros_like(sse_lesser.data)
        acc_g = xp.zeros_like(sse_greater.data)
        states, promoted = [], 0

        # ONE block layout for every q. The map from the stored sparsity
        # pattern to dense blocks depends on (rows, cols, block_sizes) alone,
        # which no q changes, so building it per q repeats the same scan nq
        # times -- and it is what lets the sectors share a batched solve.
        layout = BlockLayout(sse_lesser.rows, sse_lesser.cols,
                             self.block_sizes, band=1)
        sectors = []
        for iq, idx in todo:
            sec = self._pole_q.get(iq)
            if sec is None:
                sec = PoleSector(self._pole_cfg, freqs,
                                 **self._pole_frequency_context(freqs))
                self._pole_q[iq] = sec
            sec.set_operator_context(
                band_edges=self._band_edges_for(idx),
                delta=d_flat[:, iq, :],
                d_blocks=self._pole_blocks(self.dynamical_matrix,
                                           index_slice=idx),
                obc_left=_q_block(self.obc_blocks.retarded[0], idx),
                obc_right=_q_block(self.obc_blocks.retarded[-1], idx),
                block_sizes=self.block_sizes,
                rows=sse_lesser.rows,
                cols=sse_lesser.cols,
                layout=layout,
            )
            sectors.append(sec)

        # The q are independent, so their correctors are one batch. q_batch
        # caps how many share a solve, for memory only -- the answer does not
        # depend on it.
        chunk = int(getattr(self._pole_cfg, "q_batch", 0) or 0) or len(sectors)
        solved = []
        with profiler.profile_range("PhononSolver: Pole solve", "default",
                                    comm):
            for lo in range(0, len(sectors), chunk):
                solved.extend(refresh_many(sectors[lo:lo + chunk]))

        for (iq, idx), st in zip(todo, solved):
            states.append((idx, st))
            if st is not None and st.clusters:
                promoted += st.n_poles

        # One leg build for EVERY q and cluster. The q are independent and the
        # clusters within a q are independent, so this is one batched pass, not
        # a Python loop of n_q x n_clusters. See PhononSolver._build_pole_legs.
        self._pole_layout = layout
        with profiler.profile_range("PhononSolver: Pole legs", "default", comm):
            leg_l, leg_g = self._build_pole_legs(
                sse_lesser, sse_greater, g_retarded, solved, sectors,
                np.array([iq for iq, _ in todo], dtype=int))
        if leg_l is not None:
            for k, (iq, idx) in enumerate(todo):
                st = solved[k]
                if st is None or not st.clusters:
                    continue
                sel = (slice(None),) + idx
                st.g_pp_lesser = leg_l[k].reshape(acc_l[sel].shape)
                st.g_pp_greater = leg_g[k].reshape(acc_g[sel].shape)
                acc_l[sel] = st.g_pp_lesser
                acc_g[sel] = st.g_pp_greater
        self._report_pole_registration(solved, g_lesser, acc_l,
                                       np.array([iq for iq, _ in todo],
                                                dtype=int))

        # One state for the consumers, carrying the STACKED channel. Its
        # clusters are the union so the interaction's emptiness test is right;
        # its per-q detail stays in `q_states` for reporting.
        total = states[0][1] if states and states[0][1] is not None else None
        if total is None or not any(
                st is not None and st.clusters for _, st in states):
            self.pole_state = None
        else:
            total = next(st for _, st in states if st is not None
                         and st.clusters)
            total.g_pp_lesser, total.g_pp_greater = acc_l, acc_g
            self.pole_state = total
        self.pole_q_states = states
        if comm.rank == 0:
            n_live = sum(1 for _, st in states
                         if st is not None and st.clusters)
            skipped = nq - len(todo)
            tail = f", {skipped} q skipped (q_stride/q_max)" if skipped else ""
            # A count alone cannot say WHY the set moved. "Refused" and "never
            # found" need opposite fixes, and the hysteresis in
            # PoleSector.screen can only act on a candidate the tracker
            # matched -- so report seeded, matched and the refusal histogram
            # summed over q, not just the promoted total.
            seeded = sum(st.n_seeded for _, st in states if st is not None)
            matched = sum(st.n_matched for _, st in states if st is not None)
            why: dict[str, int] = {}
            for _, st in states:
                if st is None:
                    continue
                for _z, reason in st.rejected:
                    key = reason.split("=")[0].split("(")[0].strip()
                    why[key] = why.get(key, 0) + 1
            hist = "  ".join(f"{k} x{v}" for k, v in
                             sorted(why.items(), key=lambda kv: -kv[1]))
            eno = [v for _, st in states if st is not None
                   for v in st.eps_z_refused if np.isfinite(v)]
            eok = [v for _, st in states if st is not None
                   for v in st.eps_z_accepted if np.isfinite(v)]
            def _pct(v):
                if not v:
                    return "n/a"
                q = np.percentile(np.asarray(v, dtype=float), (50, 90, 100))
                return "  ".join(f"{x:.3g}" for x in q)
            print(f"  pole sector: {len(todo)}/{nq} q solved, {n_live} with "
                  f"poles, {promoted} pole(s) total{tail}", flush=True)
            print(f"    seeded {seeded}, matched-as-promoted {matched}, "
                  f"accepted {promoted}, refused {seeded - promoted}"
                  + (f"  [{hist}]" if hist else ""), flush=True)
            print(f"    eps_z (med/p90/max) accepted {_pct(eok)} | "
                  f"refused {_pct(eno)}   tol {self._pole_cfg.locate_tol:g}"
                  f" in / {self._pole_cfg.locate_tol_out:g} out", flush=True)

    def _census_over_q(self, delta, sse_lesser) -> None:
        """Extraction-only census on a q-resolved device, one q at a time.

        Each q gets its own ``M_q``, so each gets its own solve. Nothing is
        allocated and no vertex is touched, which is why this can run where the
        allocating path cannot.
        """
        shape = tuple(int(k) for k in delta.shape[1:-1])
        nq = int(np.prod(shape))
        d_flat = delta.reshape(delta.shape[0], nq, delta.shape[-1])
        for iq in range(nq):
            # plain ints: np.unravel_index yields np.int64, which both
            # prints as 'np.int64(0)' and makes the log ungreppable.
            idx = tuple(int(i) for i in np.unravel_index(iq, shape))
            try:
                self._pole.set_operator_context(
                    band_edges=self._band_edges_for(idx),
                    delta=d_flat[:, iq, :],
                    d_blocks=self._pole_blocks(self.dynamical_matrix,
                                               index_slice=idx),
                    obc_left=_q_block(self.obc_blocks.retarded[0], idx),
                    obc_right=_q_block(self.obc_blocks.retarded[-1], idx),
                    block_sizes=self.block_sizes,
                    rows=sse_lesser.rows,
                    cols=sse_lesser.cols,
                )
                if comm.rank == 0:
                    print(f"  q {idx}:", flush=True)
                self._pole.refresh()
            except (AttributeError, NameError, ImportError):
                # Never "this q is hard" -- these are programming errors, and
                # absorbing them into the per-q report turns a broken build
                # into a survey that quietly visits nothing. Measured: a
                # missing local import made every q read as "census failed".
                raise
            except Exception as exc:                       # noqa: BLE001
                # One q failing must not lose the other 24: the census is a
                # survey, and a q that cannot be solved is itself a datum.
                if comm.rank == 0:
                    print(f"  q {idx}: census failed ({type(exc).__name__}: "
                          f"{exc})", flush=True)

    @staticmethod
    def _q_stack(buf, iq, tail: int):
        """``(Q, n_freq, *tail)`` view of a stacked buffer for the chosen q.

        Every array the sector touches carries the transverse axis in the
        middle, ``(n_freq,) + nk + tail``. This is the batched form of the
        per-q slice the leg builder used to take one q at a time.
        """
        if buf is None:
            return None
        a = xp.asarray(buf)
        n_freq = int(a.shape[0])
        flat = a.reshape((n_freq, -1) + a.shape[a.ndim - tail:])
        return xp.moveaxis(flat[:, iq], 0, 1)

    def _build_pole_legs(self, sse_lesser, sse_greater, g_retarded,
                         states, sectors, iq) -> tuple:
        """The congruence leg for EVERY q and cluster in one pass.

        The per-cluster routines in :mod:`~quatrex.phonon.pole_congruence` say
        what the leg is, one cluster at a time, and are what this is verified
        against. Driving production through them cost a Python loop of
        ``n_q * n_clusters`` iterations over routines that themselves looped
        over pole columns and pole pairs -- 6.85 million calls and 33 s per
        SCBA iteration on Si, against a bubble of 7.4 s. See
        :mod:`~quatrex.phonon.pole_legs`.

        Only the ``congruence`` route comes through here. ``congruence_analytic``
        and the superseded ``keldysh`` route flatten each cluster into its own
        partial fractions, which is per-cluster by construction and is not a
        production setting; they keep the reference path.
        """
        from quatrex.phonon.pole_legs import (
            ClusterBatch, ClusterViews, CoefficientViews,
            congruence_legs, source_fit,
        )

        n_dof = int(np.sum(self.block_sizes))
        per_q = []
        for st, sec in zip(states, sectors):
            legs = sec.bubble_clusters() if st is not None and st.clusters else []
            if st is not None:
                st.legs = legs
            per_q.append(legs)
        if not any(per_q):
            return None, None

        # The batch trades memory for launches: on Si it peaks around 1.5 GB
        # against a 44 MB self-energy, because every intermediate carries the
        # q, cluster, frequency and pole axes at once. That is the right trade
        # on a GPU and the wrong one if it does not fit, so the q axis is cut
        # to a budget. One chunk whenever it fits -- and it is REPORTED when it
        # does not, because a silently chunked run looks like an unchunked one.
        chunk = int(getattr(self._pole_cfg, "q_batch", 0) or 0)
        if not chunk:
            n_p = max((int(c.z.shape[0]) for legs in per_q for c in legs),
                      default=1)
            n_m = max((len(legs) for legs in per_q), default=1)
            per_q_bytes = 16 * 34 * int(self.local_frequencies.shape[0]) * \
                max(n_dof, 1) * n_p * n_m
            chunk = max(1, self._POLE_LEG_BUDGET // max(per_q_bytes, 1))
        if chunk < len(states):
            if comm.rank == 0:
                print(f"  pole sector: leg build split into "
                      f"{-(-len(states) // chunk)} chunks of {chunk} q "
                      f"(memory budget)", flush=True)
            out_l, out_g = [], []
            for lo in range(0, len(states), chunk):
                hi = min(lo + chunk, len(states))
                part = self._build_pole_legs_chunk(
                    sse_lesser, sse_greater, g_retarded, states[lo:hi],
                    per_q[lo:hi], iq[lo:hi], n_dof)
                if part[0] is None:
                    w = int(self.local_frequencies.shape[0])
                    nnz = int(self._pole_layout.nnz)
                    part = (xp.zeros((hi - lo, w, nnz), dtype=xp.complex128),
                            xp.zeros((hi - lo, w, nnz), dtype=xp.complex128))
                out_l.append(part[0])
                out_g.append(part[1])
            return xp.concatenate(out_l), xp.concatenate(out_g)
        return self._build_pole_legs_chunk(sse_lesser, sse_greater, g_retarded,
                                           states, per_q, iq, n_dof)

    # Peak working set the leg build may allocate before it cuts the q axis.
    _POLE_LEG_BUDGET = 4 << 30

    def _build_pole_legs_chunk(self, sse_lesser, sse_greater, g_retarded,
                               states, per_q, iq, n_dof) -> tuple:
        """One memory-sized slice of :meth:`_build_pole_legs`."""
        from quatrex.phonon.pole_legs import (
            ClusterBatch, ClusterViews, CoefficientViews,
            congruence_legs, source_fit,
        )

        if not any(per_q):
            return None, None
        batch = ClusterBatch.from_clusters(per_q, n_dof)
        omega = xp.asarray(self.local_frequencies, dtype=float)
        widths = xp.asarray(self.local_frequency_weights, dtype=float)
        gr = self._q_stack(g_retarded.data, iq, 1)
        last_block = len(self.block_sizes) - 1

        out = {}
        for tag, buf, lo, hi in (
            ("l", sse_lesser, self.obc_blocks.lesser[0], self.obc_blocks.lesser[-1]),
            ("g", sse_greater, self.obc_blocks.greater[0], self.obc_blocks.greater[-1]),
        ):
            corners = ((self._q_stack(lo, iq, 2), 0),
                       (self._q_stack(hi, iq, 2), last_block))
            out[tag] = congruence_legs(
                batch, self._pole_layout, self._q_stack(buf.data, iq, 1), gr,
                corners, omega, widths)

        # source_fit_tol as a real gate: carrying a source analytically
        # presumes it is smooth across its own pole window, and this is the
        # measured statement of that. Reported, never silently applied -- an
        # asymptotic error estimate cannot see a source with structure of its
        # own.
        fit_l = source_fit(batch, out["l"][1], omega)
        fit_g = source_fit(batch, out["g"][1], omega)
        fit = xp.maximum(fit_l, fit_g)
        fit_host = np.asarray(get_host(fit))

        # Per-q bookkeeping, as VIEWS. Slicing the padded results per cluster
        # is a Python call and a device slice each, which is exactly the cost
        # this path removes -- and the consumers are diagnostics that usually
        # never run. ClusterViews defers it to whoever actually indexes.
        sizes = [[int(cl.z.shape[0]) for cl in legs] for legs in per_q]
        for k, (st, legs) in enumerate(zip(states, per_q)):
            if st is None or not legs:
                continue
            st.source_lesser = ClusterViews(out["l"][1][k], sizes[k])
            st.source_greater = ClusterViews(out["g"][1][k], sizes[k])
            st.c_lesser = CoefficientViews([c[k] for c in out["l"][2]],
                                           sizes[k])
            st.c_greater = CoefficientViews([c[k] for c in out["g"][2]],
                                            sizes[k])
            st.source_fit = list(fit_host[k, :len(legs)])

        # source_fit_tol is REPORTED, never applied -- an asymptotic error
        # estimate cannot see a source with structure of its own. The offenders
        # are found without a walk over the clusters; normally there are none.
        if comm.rank == 0:
            over = np.argwhere(fit_host > self._pole_cfg.source_fit_tol)
            for k, m in over:
                if k < len(per_q) and m < len(per_q[k]):
                    print(f"  pole sector: cluster {per_q[k][m].label} source "
                          f"varies by {fit_host[k, m]:.2e} across its window "
                          f"(tol {self._pole_cfg.source_fit_tol:.2e}); the "
                          "analytic source model is not justified there",
                          flush=True)
        return out["l"][0].sum(axis=1), out["g"][0].sum(axis=1)

    def _report_pole_registration(self, states, g_lesser, leg, iq) -> None:
        """Where the promoted poles sit INSIDE their cells, over the whole set.

        This is the control parameter of the bubble's registration error, and
        nothing measured it before. An exactly cell-averaged leg still places
        all of a line's weight at the cell CENTRE, so the combination frequency
        Re(z_a + z_b) is displaced by up to a full cell and the ring splits the
        peak between two bins. Measured
        (test_cell_averaged_legs_do_not_fix_the_bubble_registration),
        ring/exact at the combination frequency:

            offset 0.00   1.004      offset 0.25   1.79
            offset 0.50   0.536

        and it gets WORSE with h/gamma, not better. The congruence route's
        accuracy is therefore an accident of registration unless this is small.

        Reported once for the whole q axis rather than once per q: it is a
        property of the promoted SET.
        """
        if comm.rank != 0 or leg is None:
            return
        from quatrex.phonon.pole_audit import pole_pair_weight

        w_host = np.asarray(get_host(self.local_frequencies), dtype=float)
        hw = np.asarray(get_host(self.local_frequency_weights), dtype=float)
        z = np.concatenate(
            [np.asarray(get_host(cl.z)).ravel()
             for st in states if st is not None
             for cl in (st.legs or [])] or [np.zeros(0, dtype=complex)])
        z = z[np.real(z) >= 0.0]
        if z.size == 0:
            return
        k = np.argmin(np.abs(w_host[None, :] - np.real(z)[:, None]), axis=1)
        pole_cells = np.zeros(w_host.size, dtype=bool)
        pole_cells[k] = True
        good = hw[k] > 0.0
        off_worst = float(np.max(np.abs(np.real(z)[good] - w_host[k[good]])
                                 / hw[k[good]])) if good.any() else 0.0

        # ... and how much of the ring's weight the offset can actually move.
        # The registration error is order one ONLY on cell pairs (k, m-k) with
        # both ends in a pole cell -- one displaced leg against a resolved
        # partner costs O((delta/Gamma)^2) instead -- so the error in Sigma is
        # bounded by this fraction times ~0.8. A scalar-norm proxy for the
        # vertex contraction, hence an upper bound: a small value is
        # conclusive, a large one is a reason to measure properly.
        # The RING's leg, G^< - g_pp -- not Sigma, and not the raw G^<: the
        # whole point is what the convolution is fed.
        if g_lesser is not None:
            gl = (self._q_stack(g_lesser.data, iq, 1)
                  - xp.asarray(leg).reshape(iq.size, w_host.size, -1))
            norms = np.linalg.norm(np.asarray(get_host(gl)), axis=2).max(axis=0)
            low = max(1e-6, float(getattr(
                self.config.phonon, "sse_low_freq_mask_thz", 0.0) or 0.0))
            pw = pole_pair_weight(norms, pole_cells, freqs=w_host,
                                  skip=np.abs(w_host) < low)
        else:
            pw = {"mean": float("nan"), "worst": float("nan"),
                  "omega": float("nan")}
        print(f"  pole registration: worst sub-cell offset {off_worst:.3f} "
              f"cells (0 = on a grid point, 0.5 = on a cell boundary); "
              f"pole-cell PAIRS carry {100 * pw['mean']:.3g}% of the ring's "
              f"weight, up to {100 * pw['worst']:.3g}% at "
              f"w={pw['omega']:.2f}", flush=True)

    def _build_pole_keldysh(self, sse_lesser, sse_greater,
                            g_retarded=None, g_lesser=None, q_idx=(),
                            sector=None) -> None:
        """Project the Keldysh source and reduce ``G_PP`` onto the pattern.

        Done here, in the ``"stack"`` state, because this is where the contact
        blocks live: ``Sigma_tot`` is the scattering self-energy that entered
        the Dyson solve PLUS both contacts, and it is the contact part that
        drives the device. Everything stays on the stored sparsity pattern --
        the dense intermediate would be ``(n_omega, n_dof, n_dof)``.
        """
        from quatrex.phonon.pole_bridge import (
            add_contact_source, pole_keldysh_pf_sparse, project_source_sparse,
            source_at_poles, source_variation,
        )
        from quatrex.phonon.pole_congruence import (
            apply_sparse, background_coefficients, coefficients_at_poles,
            fit_residual, partial_fraction_legs, pf_leg_sample,
            remainder_resolution, residue_sum, sector_cell_average,
            sector_grid_sample,
        )

        state = self.pole_state
        if state is None or not state.clusters:
            return
        freqs = xp.asarray(self.local_frequencies, dtype=float)
        rows, cols = sse_lesser.rows, sse_lesser.cols

        def _q(a):
            """This q's slice of a stacked array; identity when nq == 1.

            Every array the sector touches carries the transverse axis in the
            middle, ``(n_freq,) + nk + (nnz,)``, and the pole problem is
            independent per q -- so slicing here is the whole of coupled-q for
            this route. The contact blocks carry it too, and dropping them is
            not a small error: they are what drives the device.
            """
            if not q_idx or a is None:
                return a
            return a[(slice(None),) + tuple(q_idx)]

        sl = _q(sse_lesser.data).reshape(freqs.shape[0], -1)
        sg = _q(sse_greater.data).reshape(freqs.shape[0], -1)
        last = int(np.sum(self.block_sizes[:-1]))

        # The pole set must be closed under z -> -z^*: every resonance at
        # +Omega has a partner at -Omega, and the NEP finds only the positive
        # members because the search window is positive. Leaving it unclosed
        # flatters the rr_ss staging setting and costs the complete method
        # four orders.
        #
        # The clusters come from the sector that SOLVED this q, i.e.
        # self._pole_q[iq]. self._pole exists (built before the q dispatch)
        # but has no operator context and returns an empty closure.
        state.legs = (sector or self._pole).bubble_clusters()
        acc_l = acc_g = None
        _leg = getattr(self._pole_cfg, "leg", "congruence")
        analytic = _leg == "congruence_analytic"
        congruence = _leg.startswith("congruence")
        if congruence:
            if g_retarded is None:
                raise ValueError(
                    "pole_sector.leg='congruence' needs G^R: the retarded "
                    "background B^R_k = G^R_k - U D(w_k) V^dagger is what the "
                    "regular leg is made of.")
            n_dof = int(np.sum(self.block_sizes))
            gr = _q(g_retarded.data).reshape(freqs.shape[0], -1)
            corners_l = ((_q(self.obc_blocks.lesser[0]), 0),
                         (_q(self.obc_blocks.lesser[-1]), last))
            corners_g = ((_q(self.obc_blocks.greater[0]), 0),
                         (_q(self.obc_blocks.greater[-1]), last))
            cell_widths = xp.asarray(self.local_frequency_weights, dtype=float)
        for cl in state.legs:
            s_l = project_source_sparse(sl, rows, cols, cl.v)
            s_g = project_source_sparse(sg, rows, cols, cl.v)
            for corner_l, corner_g, off in (
                (_q(self.obc_blocks.lesser[0]),
                 _q(self.obc_blocks.greater[0]), 0),
                (_q(self.obc_blocks.lesser[-1]),
                 _q(self.obc_blocks.greater[-1]), last),
            ):
                if corner_l is not None:
                    s_l = add_contact_source(s_l, corner_l, cl.v, off)
                if corner_g is not None:
                    s_g = add_contact_source(s_g, corner_g, cl.v, off)
            # source_fit_tol as a real gate: carrying a source analytically
            # presumes it is smooth across its own pole window, and this is
            # the measured statement of that. Above the tolerance the cluster
            # is reported rather than silently approximated -- an asymptotic
            # error estimate cannot see a source with structure of its own.
            eps_fit = max(source_variation(s_l, freqs, cl),
                          source_variation(s_g, freqs, cl))
            state.source_fit.append(eps_fit)
            if eps_fit > self._pole_cfg.source_fit_tol and comm.rank == 0:
                print(f"  pole sector: cluster {cl.label} source varies by "
                      f"{eps_fit:.2e} across its window (tol "
                      f"{self._pole_cfg.source_fit_tol:.2e}); the analytic "
                      "source model is not justified there", flush=True)
            state.source_lesser.append(s_l)
            state.source_greater.append(s_g)
            if congruence:
                # Split the RETARDED function and let the Keldysh components
                # follow, so the leg the ring keeps is B^R_k Sigma B^A_k -- a
                # congruence of a PSD source, hence PSD -- instead of the
                # indefinite frozen remainder. See pole_congruence.py.
                for src, corn, c_out, g_out in (
                    (sl, corners_l, state.c_lesser, "l"),
                    (sg, corners_g, state.c_greater, "g"),
                ):
                    sv = apply_sparse(src, rows, cols, cl.v, n_dof,
                                      corners=corn)
                    co = background_coefficients(
                        cl, freqs, sv,
                        apply_sparse(gr, rows, cols, sv, n_dof))
                    c_out.append(co)
                    if analytic:
                        # The sectors are put back as ANALYTIC terms, so what
                        # the ring gives up is the analytic leg's own sample --
                        # not a cell-average correction. G_reg = G - G_S is
                        # exact for any G_S, but only if the leg subtracted
                        # here and the leg the sectors restore are literally
                        # the same function.
                        frozen = coefficients_at_poles(cl, freqs, co)
                        zeta, p_row, q_col = partial_fraction_legs(cl, frozen)
                        (state.pf_lesser if g_out == "l"
                         else state.pf_greater).append((zeta, p_row, q_col))
                        state.residue_sum.append(residue_sum(p_row, q_col))
                        # source_fit_tol gates c_ss and nothing else. The
                        # mixed coefficient c_rs = G^R_k Sigma V - U D_k c_ss
                        # is frozen by the same approximation and carries the
                        # whole retarded background, so it needs its own
                        # measurement -- but of the right thing. Its VARIATION
                        # across the window is not an error: the principal
                        # part is c(z)/(w-z) exactly and the variation lands
                        # in [c(w)-c(z)]/(w-z), which is regular. What can go
                        # wrong is the local model's own residual, and whether
                        # the grid carries that regular remainder.
                        state.mixed_fit.append((
                            fit_residual(co[1], freqs, cl),
                            remainder_resolution(co[1], frozen[1], freqs, cl),
                        ))
                        smp = pf_leg_sample(zeta, p_row, q_col, freqs,
                                            rows, cols)
                    else:
                        # What the ring must give up is the difference between
                        # the POINT sample it would otherwise use and the CELL
                        # AVERAGE the bubble's dw-weighted sum actually wants.
                        # Subtracting it leaves the ring holding <G~^{<,>}>_k,
                        # an average of PSD matrices, hence PSD whatever the
                        # pole model does.
                        smp = (sector_grid_sample(cl, freqs, co, rows, cols)
                               - sector_cell_average(cl, freqs, co, rows,
                                                     cols, cell_widths))
                    if g_out == "l":
                        acc_l = smp if acc_l is None else acc_l + smp
                    else:
                        acc_g = smp if acc_g is None else acc_g + smp
                continue
            # PARTIAL-FRACTION form, not U D^R S(w) D^A U^dag. The analytic
            # sectors split every leg into simple poles, which carries only a
            # rational source, so the resolved form is a DIFFERENT function
            # whenever S varies with frequency (measured 7e-3 apart at a
            # 2%/THz slope, against 7e-16 for a constant source).
            #
            # G_reg = G - G_PP is exact for ANY G_PP, so the sector sum holds
            # iff the leg subtracted here and the leg the sectors put back are
            # literally the same object. Using the resolved form on one side
            # and partial fractions on the other is what broke the SPATIAL
            # balance while leaving the scalar P_in = P_out nearly intact.
            g_l = pole_keldysh_pf_sparse(
                freqs, cl, source_at_poles(s_l, freqs, cl), rows, cols)
            g_g = pole_keldysh_pf_sparse(
                freqs, cl, source_at_poles(s_g, freqs, cl), rows, cols)
            acc_l = g_l if acc_l is None else acc_l + g_l
            acc_g = g_g if acc_g is None else acc_g + g_g
        if acc_l is None or acc_g is None:
            # No leg carried a pole, so there is nothing to remove from the
            # ring. Leave the channel unset rather than reshaping None: an
            # empty sector must be a no-op, not a crash.
            state.g_pp_lesser = state.g_pp_greater = None
            return
        state.g_pp_lesser = acc_l.reshape(_q(sse_lesser.data).shape)
        state.g_pp_greater = acc_g.reshape(_q(sse_greater.data).shape)
        if congruence and comm.rank == 0:
            # Where each promoted pole sits inside its cell, in cells, worst
            # over the set: 0 is registered on a grid point, 0.5 on a cell
            # boundary. This is the control parameter of the bubble's
            # registration error. A cell-averaged leg still places a line's
            # weight at the cell CENTRE, so the combination frequency is
            # displaced by up to a full cell and the ring splits the peak
            # between two bins -- and it worsens with h/gamma. The congruence
            # route's accuracy is an accident of registration unless this is
            # small.
            from quatrex.phonon.pole_audit import pole_pair_weight

            w_host = np.asarray(get_host(freqs), dtype=float)
            hw = np.asarray(get_host(self.local_frequency_weights), dtype=float)
            off_worst = 0.0
            pole_cells = np.zeros(w_host.size, dtype=bool)
            for cl in state.legs:
                for z in np.asarray(get_host(cl.z)):
                    x = float(np.real(z))
                    if x < 0.0:
                        continue
                    k = int(np.argmin(np.abs(w_host - x)))
                    pole_cells[k] = True
                    hk = float(hw[k]) if k < hw.size else 0.0
                    if hk > 0.0:
                        off_worst = max(off_worst, abs(x - w_host[k]) / hk)
            # ... and how much of the ring's weight the offset can actually
            # move. The registration error is order one ONLY on cell pairs
            # (k, m-k) with both ends in a pole cell -- one displaced leg
            # against a resolved partner costs O((delta/Gamma)^2) instead --
            # so the error in Sigma is bounded by this fraction times ~0.8.
            # A scalar-norm proxy for the vertex contraction, hence an upper
            # bound: a small value is conclusive, a large one is a reason to
            # measure properly.
            # The RING's leg, G^< - g_pp -- not Sigma, and not the raw G^<:
            # the whole point is what the convolution is fed.
            if g_lesser is not None:
                gl = (_q(g_lesser.data).reshape(w_host.size, -1)
                      - acc_l.reshape(w_host.size, -1))
                low = max(1e-6, float(getattr(
                    self.config.phonon, "sse_low_freq_mask_thz", 0.0) or 0.0))
                pw = pole_pair_weight(
                    np.linalg.norm(np.asarray(get_host(gl)), axis=1),
                    pole_cells, freqs=w_host, skip=np.abs(w_host) < low)
            else:
                pw = {"mean": float("nan"), "worst": float("nan"),
                      "omega": float("nan")}
            print(f"  pole registration: worst sub-cell offset "
                  f"{off_worst:.3f} cells (0 = on a grid point, 0.5 = on a "
                  f"cell boundary); pole-cell PAIRS carry "
                  f"{100 * pw['mean']:.3g}% of the ring's weight, up to "
                  f"{100 * pw['worst']:.3g}% at w={pw['omega']:.2f}",
                  flush=True)
        if analytic and comm.rank == 0:
            # eps_tail is the coefficient of the leg's 1/w tail. The analytic
            # leg is global -- the pole-pole sector integrates it over the
            # whole axis -- and the true G decays like 1/w^2, so a spurious
            # 1/w is not cosmetic. It vanishes only for a bosonically closed
            # set whose freezing preserved the cancellation.
            #
            # eps_fit is the local model's residual against its own samples;
            # eps_reg is whether the grid integrates the regular remainder.
            # Separate errors; one number cannot stand for both.
            tail = max(state.residue_sum) if state.residue_sum else 0.0
            fits = [a for a, _ in state.mixed_fit] or [0.0]
            regs = [b for _, b in state.mixed_fit] or [0.0]
            flag = ("" if max(regs) <= self._pole_cfg.source_fit_tol
                    else f"  eps_reg ABOVE source_fit_tol "
                         f"({self._pole_cfg.source_fit_tol:.2e})")
            print(f"  pole analytic leg: eps_tail={tail:.3e}  "
                  f"eps_fit={max(fits):.3e}  eps_reg={max(regs):.3e}{flag}",
                  flush=True)

    def _report_subcell(self, out) -> None:
        """Is the reconstruction physical BETWEEN grid points?

        The sectors act on ``G~_h(w) = P(w) + R_k``, not on ``G``. That equals
        ``G`` at the cell centres and nowhere else, and ``R_k = G - P`` is a
        DIFFERENCE of PSD objects. Offline, a crude pole model gives
        ``lambda_min = -1.000`` five percent of a cell off centre while the
        true ``G`` stays at ``+2.3e-02``; an EXACT residue keeps it healthy.
        So this measures whether the production pole model is good enough, and
        it is measured rather than assumed.

        Report only: the threshold at which a pole should be refused is not
        yet established, and gating on a guess would hide the answer.
        """
        from quatrex.phonon.pole_audit import psd_residual, subcell_positivity
        from quatrex.phonon.pole_bridge import (
            pole_keldysh_pf_sparse, source_at_poles,
        )
        from quatrex.phonon.pole_keldysh import pole_retarded

        cfg = getattr(self.config.phonon, "pole_sector", None)
        if cfg is None or not getattr(cfg, "psd_check", False):
            return
        state = self.pole_state
        if state is None or not state.legs:
            return
        if out[0].data.ndim > 2:
            # Coupled-q: every gate below reshapes to (n_freq, nnz) against one
            # rows/cols pair, and a q axis breaks that silently -- it would
            # still produce a number. A per-q subcell gate is its own work.
            if comm.rank == 0:
                print("  ring leg positivity: skipped (q-resolved; the gate is "
                      "per-q and not written)", flush=True)
            return

        if getattr(cfg, "leg", "congruence") == "congruence":
            # The ring convolves G^{<,>}_k - g_pp, which on the cell-average
            # route is an average of PSD matrices: PSD by construction, so
            # this gate checks the implementation (a sign, an index, a cell
            # width), not the maths. Three things it must do: report lesser
            # and greater separately, since a single min hides which failed;
            # EXCLUDE the bins the ring masks (unmasked, the near-singular
            # G^>(0) acoustic bin makes the gate read exactly -1.000 on the
            # pole-free baseline); and print the same gate on the uncorrected
            # leg as a control, since agreement means it is not measuring the
            # pole sector at all.
            #
            # Deliberately NOT taken on congruence_analytic: there the leg is
            # the remainder G - G_S, a difference of PSD objects, which may be
            # indefinite with nothing wrong. Only the total -i Sigma^{<,>} is
            # constrained.
            rows, cols = out[0].rows, out[0].cols
            n_freq = int(self.local_frequencies.shape[0])
            low = max(1e-6, float(
                getattr(self.config.phonon, "sse_low_freq_mask_thz", 0.0) or 0.0))
            w_host = np.asarray(get_host(self.local_frequencies), dtype=float)
            skip = xp.asarray(np.abs(w_host) < low)
            # ... and a second reading restricted to the CELLS THE POLES
            # TOUCH. The global worst is whatever the baseline's own worst bin
            # is, and the correction is localised, so the global number goes
            # blind against its control by construction. This one cannot.
            near = np.zeros(n_freq, dtype=bool)
            for cl in state.legs:
                for z in np.asarray(get_host(cl.z)):
                    x = float(np.real(z))
                    if x < 0.0:
                        continue
                    k = int(np.argmin(np.abs(w_host - x)))
                    near[max(0, k - 2):min(n_freq, k + 3)] = True
            skip_far = xp.asarray((np.abs(w_host) < low) | ~near)
            rep_all = {}
            for name, got, pp in (("lesser", out[0], state.g_pp_lesser),
                                  ("greater", out[1], state.g_pp_greater)):
                raw = got.data.reshape(n_freq, -1)
                for tag, leg in ((name, raw - pp.reshape(n_freq, -1)),
                                 (f"{name}_control", raw)):
                    rep_all[tag] = psd_residual(
                        leg, rows, cols, self.block_sizes, sign=-1.0, skip=skip)
                    rep_all[f"{tag}_poles"] = psd_residual(
                        leg, rows, cols, self.block_sizes, sign=-1.0,
                        skip=skip_far)
            self.psd_report["ring_leg"] = rep_all
            if comm.rank == 0:
                for name in ("lesser", "greater"):
                    a, b = rep_all[name], rep_all[f"{name}_control"]
                    # RELATIVE, not absolute: the two agreeing to a part in
                    # 1e3 already means the worst bin is one the pole
                    # correction does not touch, and the gate is reporting the
                    # baseline's own worst bin whatever the sector does.
                    scale = max(abs(a["worst"]), abs(b["worst"]), 1e-300)
                    same = ("  [== control: gate is blind]"
                            if abs(a["worst"] - b["worst"]) <= 1e-3 * scale
                            else "")
                    pa = rep_all[f"{name}_poles"]
                    pb = rep_all[f"{name}_control_poles"]
                    print(f"  ring leg positivity {name:8s} "
                          f"worst={a['worst']:+.3e} at w[{a['omega_index']}]"
                          f"   pole-off control={b['worst']:+.3e}{same}"
                          f"   | in pole cells {pa['worst']:+.3e} vs "
                          f"{pb['worst']:+.3e}", flush=True)
            return
        rows, cols = out[0].rows, out[0].cols
        freqs = xp.asarray(self.local_frequencies, dtype=float)
        g_l = out[0].data.reshape(freqs.shape[0], -1)
        w = np.asarray(get_host(freqs), dtype=float)
        centres = np.array([int(np.argmin(np.abs(w - float(np.real(z)))))
                            for cl in state.legs for z in np.asarray(
                                get_host(cl.z)) if float(np.real(z)) >= 0.0])
        if centres.size == 0:
            return

        def _pole_at(omega):
            acc = None
            for cl, s_l in zip(state.legs, state.source_lesser):
                v = pole_keldysh_pf_sparse(
                    omega, cl, source_at_poles(s_l, freqs, cl), rows, cols)
                acc = v if acc is None else acc + v
            return acc

        rep = subcell_positivity(
            g_l, state.g_pp_lesser.reshape(freqs.shape[0], -1), _pole_at,
            freqs, rows, cols, self.block_sizes, centres=centres, window=1)
        self.psd_report["subcell"] = rep

        # ... and the same measure for the CONGRUENCE reconstruction, which
        # rebuilds the Keldysh component from the retarded split instead of
        # freezing the Keldysh remainder. Offline the two differ completely
        # (-1.000 against +1.3e-05); this is the in-situ comparison that says
        # whether the redesign is worth its cost on a real device.
        cong = None
        try:
            from quatrex.phonon.pole_audit import subcell_congruence

            def _pole_ret_at(omega):
                acc = None
                for cl in state.legs:
                    v = pole_retarded(omega, cl)
                    acc = v if acc is None else acc + v
                return acc[:, rows, cols]

            cong = subcell_congruence(
                out[2].data.reshape(freqs.shape[0], -1),
                self._psd_sigma_lesser, _pole_ret_at, freqs, rows, cols,
                centres=centres)
            self.psd_report["subcell_congruence"] = cong
        except NotImplementedError as exc:
            if comm.rank == 0:
                print(f"  subcell congruence: skipped ({exc})", flush=True)

        if comm.rank == 0:
            tail = ("" if cong is None
                    else f"   congruence worst={cong['worst']:+.3e}")
            print(f"  subcell positivity: worst={rep['worst']:+.3e} at "
                  f"w[{rep['worst_centre']}], at-centres="
                  f"{rep['at_centres']:+.3e}{tail}", flush=True)

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

        self._apply_eta_ramp()
        self._apply_eta_obc_ramp()
        self._apply_eta_ir_floor_ramp()
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
        self._apply_buttiker_probe(sse_lesser, sse_greater)   # no-op if off
        self._selected_solve(sse_lesser, sse_greater, out)
        self._restore_buttiker_probe(sse_lesser, sse_greater)
        self._update_buttiker_probe(out)

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
