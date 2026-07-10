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

        # frequencies are the linear-frequency grid in THz (uniform spacing,
        # as the bubble FFT requires); the Bose occupation needs hbar*omega.
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
        # Optional in-SCBA ANNEAL of the sub-grid soft-mode floor: ramp
        # eta_ir_floor_cells DOWN from its start value to eta_ir_floor_final_cells
        # over eta_ir_floor_ramp_iterations solves, then hold. Tests whether the
        # floor is only a crutch to REACH the basin (anneal->0 holds) or is
        # load-bearing (re-diverges as floor->0).
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
        # Optional in-SCBA CONTACT-broadening ramp: ramp eta_obc DOWN from eta_obc0
        # (large enough to converge the cell cold) to eta_obc_final over
        # eta_obc_ramp_iterations solves, then hold. The MPI-compatible in-run
        # analogue of the eta_obc warm-start chain for the eta=0 fixed point on the
        # longer cells (warm-start files are single-rank only). lead(eta_obc_final)
        # -> lead(0) via the ~L-independent universal bias factor.
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

        self.band_edge_tracking = config.phonon.band_edge_tracking

        self.call_count = 0
        self.filtering_iteration_limit = config.phonon.filtering_iteration_limit

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

        Linearly ramps ``self.eta_obc`` from ``eta_obc0`` (iteration 0, large enough
        to converge the cell cold) to ``eta_obc_final`` over ``eta_obc_ramp_iterations``
        solves, then holds. The contact NEVP regularisation is strong while Sigma is
        still developing and relaxes toward the (small) target -- the in-run,
        MPI-compatible analogue of the eta_obc warm-start chain. The converged
        lead(eta_obc_final) maps to lead(0) via the ~L-independent universal factor.
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
        and intermittently breaks the spectral NEVP once Sigma grows
        (recursion errors ~1 -> residual spikes in the SCBA).
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
        # 2i*eta*omega damping vanishes as omega->0, leaving the acoustic soft
        # modes (D->0) unregularised at eta=0 (G^R ~ 1/omega^2 -> SCBA diverges).
        # Add a DC-CONCENTRATED constant broadening i*Gamma_floor*lowpass(omega)
        # that damps only the lowest (unresolved, ~zero-heat) bins; the resolved
        # low-omega physics and the IR plateau are untouched. Gamma_floor->0 as
        # dw->0 (grid-consistent). NOT applied to the OBC (ideal leads).
        _ir_floor_c = self._eta_ir_floor_c
        self._ir_floor_diag = None
        if _ir_floor_c > 0.0 and self.energies.size > 1:
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

    @profiler.profile(label="PhononSolver: Filter", level="default", comm=comm)
    def _filter_peaks(self, out: tuple[DSDBSparse, ...]) -> None:
        """Filters out peaks in the output Green's functions"""
        pass

    @profiler.profile(label="PhononSolver: DOS Peaks", level="default", comm=comm)
    def _track_dos_peaks(self, out: tuple[DSDBSparse, ...]) -> None:
        """Tracks dos peaks in the output Green's functions (no-op for phonons:
        there is no Fermi-level / band-edge renormalisation to track here)."""
        pass

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

        # TODO Band ege Tracking

        # OBC from the bare harmonic blocks (ideal-reservoir contacts);
        # the scattering self-energy enters the device Dyson only.
        self._compute_obc()
        _btd_subtract(self.system_matrix, sse_retarded)
        self._apply_buttiker_probe(sse_lesser, sse_greater)   # no-op if off
        self._selected_solve(sse_lesser, sse_greater, out)
        self._restore_buttiker_probe(sse_lesser, sse_greater)
        self._update_buttiker_probe(out)

        self.system_matrix.free_data()
        if self.call_count < self.filtering_iteration_limit:
            self._filter_peaks(out)

        if self.band_edge_tracking == "dos-peaks":
            self._track_dos_peaks(out)

        self.call_count += 1