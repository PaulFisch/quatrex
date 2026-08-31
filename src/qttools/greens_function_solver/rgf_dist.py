# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes the distributed selected inversion solver."""

import numpy as np

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import DSDBSparse, _DStackView
from qttools.greens_function_solver import _serinv
from qttools.greens_function_solver.solver import GFSolver, OBCBlocks
from qttools.kernels import linalg
from qttools.profiling import Profiler
from qttools.utils.solvers_utils import get_batches

profiler = Profiler()


def _dag(x: NDArray) -> NDArray:
    return x.conj().swapaxes(-2, -1)


def _left_connected_chain(
    a: _DStackView,
    sigma_lesser: _DStackView,
    sigma_greater: _DStackView,
    obc_blocks: OBCBlocks,
    stack_slice: slice,
    xr_diag_blocks: list[NDArray],
    xl_diag_blocks: list[NDArray],
    xg_diag_blocks: list[NDArray],
) -> tuple[list[NDArray], list[NDArray], list[NDArray]]:
    """Left-connected auxiliaries g^L_i, g^{L,<}_i, g^{L,>}_i for every
    local block row (the single-node RGF forward-sweep quantities).

    Rank 0 gets them for free: after ``downward_schur`` (with
    ``invert_last_block=False``) the diag stacks hold the INVERTED
    left-connected g^L and the sandwiched g^{L,<,>} for all but the last
    local block, whose slot holds the un-inverted Schur complement --
    stash copies (they are overwritten by the reduced mapback and the
    selinv) and finish the last block by hand.

    Ranks > 0 run the same forward recursion over their local rows,
    seeded by a pipelined point-to-point receive of the previous rank's
    last-row chain values plus the cross-boundary coupling blocks (which
    live in the previous rank's arrow-wise strip). The pipeline
    serialises one forward sweep across the block ranks -- acceptable at
    the small block_comm_size this solver targets.
    """
    n = a.num_local_blocks
    bcomm = comm.block._mpi_comm
    rank, size = comm.block.rank, comm.block.size

    if rank == 0:
        glr = [xr_diag_blocks[i].copy() for i in range(n - 1)]
        gll = [xl_diag_blocks[i].copy() for i in range(n - 1)]
        glg = [xg_diag_blocks[i].copy() for i in range(n - 1)]
        # Last block: Schur complement, un-inverted; sources accumulated
        # but not yet sandwiched.
        g_last = linalg.inv(xr_diag_blocks[-1])
        g_last_dag = _dag(g_last)
        glr.append(g_last)
        gll.append(g_last @ xl_diag_blocks[-1] @ g_last_dag)
        glg.append(g_last @ xg_diag_blocks[-1] @ g_last_dag)
    else:
        # Receive the seed: previous rank's last-row chain + the
        # cross-boundary blocks A_{f,f-1}, A_{f-1,f}, Sigma^{<,>}_{f-1,f}
        # (f = this rank's first global block row).
        glr_p, gll_p, glg_p, a_dn, a_up, sl_up, sg_up = bcomm.recv(
            source=rank - 1, tag=71
        )
        glr, gll, glg = [], [], []
        for j in range(n):
            obc_r = obc_blocks.retarded[j]
            a_jj = (
                a.blocks[j, j]
                if obc_r is None
                else a.blocks[j, j] - obc_r[stack_slice]
            )
            obc_l = obc_blocks.lesser[j]
            sl_jj = (
                sigma_lesser.blocks[j, j]
                if obc_l is None
                else sigma_lesser.blocks[j, j] + obc_l[stack_slice]
            )
            obc_g = obc_blocks.greater[j]
            sg_jj = (
                sigma_greater.blocks[j, j]
                if obc_g is None
                else sigma_greater.blocks[j, j] + obc_g[stack_slice]
            )
            a_dn_glr = a_dn @ glr_p
            xr_j = linalg.inv(a_jj - a_dn_glr @ a_up)
            xr_j_dag = _dag(xr_j)
            t_l = a_dn_glr @ sl_up
            t_g = a_dn_glr @ sg_up
            a_dn_dag = _dag(a_dn)
            xl_j = xr_j @ (
                sl_jj + a_dn @ gll_p @ a_dn_dag + _dag(t_l) - t_l
            ) @ xr_j_dag
            xg_j = xr_j @ (
                sg_jj + a_dn @ glg_p @ a_dn_dag + _dag(t_g) - t_g
            ) @ xr_j_dag
            glr.append(xr_j)
            gll.append(xl_j)
            glg.append(xg_j)
            if j + 1 < n:
                glr_p, gll_p, glg_p = xr_j, xl_j, xg_j
                a_dn = a.blocks[j + 1, j]
                a_up = a.blocks[j, j + 1]
                sl_up = sigma_lesser.blocks[j, j + 1]
                sg_up = sigma_greater.blocks[j, j + 1]

    if rank + 1 < size:
        # Seed the next rank: my last-row chain values and the
        # cross-boundary coupling blocks I own in my arrow-wise strip.
        bcomm.send(
            (
                glr[-1], gll[-1], glg[-1],
                a.blocks[n, n - 1],           # A_{f', last}
                a.blocks[n - 1, n],           # A_{last, f'}
                sigma_lesser.blocks[n - 1, n],
                sigma_greater.blocks[n - 1, n],
            ),
            dest=rank + 1,
            tag=71,
        )

    return glr, gll, glg


def _offdiag_post_pass(
    a: _DStackView,
    sigma_lesser: _DStackView,
    sigma_greater: _DStackView,
    xl_out: _DStackView,
    xg_out: _DStackView,
    xr_out: _DStackView,
    glr: list[NDArray],
    gll: list[NDArray],
    glg: list[NDArray],
    n_offdiagonals: int,
) -> None:
    """Write exact Keldysh block bands with a right-to-left pipeline."""
    n = a.num_local_blocks
    bcomm = comm.block._mpi_comm
    rank, size = comm.block.rank, comm.block.size
    first = a.num_blocks - len(a.local_block_sizes)
    valid_first_band = min(n, a.num_blocks - first - 1)
    xl_previous = [xl_out.blocks[i, i + 1] for i in range(valid_first_band)]
    xg_previous = [xg_out.blocks[i, i + 1] for i in range(valid_first_band)]
    xr_previous = [xr_out.blocks[i + 1, i] for i in range(valid_first_band)]
    padding = [None] * (n - valid_first_band)
    xl_previous += padding
    xg_previous += padding
    xr_previous += padding

    for distance in range(2, min(n_offdiagonals, a.num_blocks - 1) + 1):
        request = None
        if rank > 0:
            request = bcomm.isend(
                (xl_previous[0], xg_previous[0], xr_previous[0]),
                dest=rank - 1,
                tag=72,
            )
        halo = bcomm.recv(source=rank + 1, tag=72) if rank + 1 < size else None
        if request is not None:
            request.wait()

        xl_current = [None] * n
        xg_current = [None] * n
        xr_current = [None] * n
        for i in range(n):
            if first + i + distance >= a.num_blocks:
                continue
            xl_next, xg_next, xr_next = (
                (xl_previous[i + 1], xg_previous[i + 1], xr_previous[i + 1])
                if i + 1 < n else halo
            )
            j = i + 1
            a_down_dag = _dag(a.blocks[j, i])
            xa = _dag(xr_next)
            g_a = glr[i] @ a.blocks[i, j]
            xl_value = (
                -g_a @ xl_next
                + (glr[i] @ sigma_lesser.blocks[i, j] - gll[i] @ a_down_dag)
                @ xa
            )
            xg_value = (
                -g_a @ xg_next
                + (glr[i] @ sigma_greater.blocks[i, j] - glg[i] @ a_down_dag)
                @ xa
            )
            xr_current[i] = -xr_next @ a.blocks[j, i] @ glr[i]
            xl_current[i], xg_current[i] = xl_value, xg_value
            xl_out.blocks[i, i + distance] = xl_value
            xg_out.blocks[i, i + distance] = xg_value
            if xl_out.symmetry is None:
                xl_out.blocks[i + distance, i] = -_dag(xl_value)
            if xg_out.symmetry is None:
                xg_out.blocks[i + distance, i] = -_dag(xg_value)
        xl_previous, xg_previous, xr_previous = (
            xl_current, xg_current, xr_current
        )


class RGFDist(GFSolver):
    """Distributed selected inversion solver.

    Parameters
    ----------
    max_batch_size : int, optional
        Maximum batch size to use when inverting the matrix, by default
        100.

    """

    def __init__(self, max_batch_size: int = 100) -> None:
        """Initializes the selected inversion solver."""
        self.max_batch_size = max_batch_size

    def selected_inv(
        self,
        a: DSDBSparse,
        out: DSDBSparse,
        obc_blocks: OBCBlocks | None = None,
    ) -> None:
        """Performs selected inversion of a block-tridiagonal matrix.

        Parameters
        ----------
        a : DSDBSparse
            Matrix to invert.
        out : DSDBSparse
            Preallocated output matrix.
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.

        """

        # Initialize temporary buffers.
        reduced_system = _serinv.ReducedSystem()

        # Initialize dense temporary buffers for the diagonal blocks and
        # the upper and lower auxiliary buffer blocks.
        x_diag_blocks: list[NDArray | None] = [None] * a.num_local_blocks
        buffer_lower: list[NDArray | None] = [None] * a.num_local_blocks
        buffer_upper: list[NDArray | None] = [None] * a.num_local_blocks

        if obc_blocks is None:
            obc_blocks = OBCBlocks(num_blocks=a.num_local_blocks)

        batch_sizes, batch_offsets = get_batches(a.shape[0], self.max_batch_size)

        for i in range(len(batch_sizes)):
            stack_slice = slice(int(batch_offsets[i]), int(batch_offsets[i + 1]))

            a_ = a.stack[stack_slice]
            out_ = out.stack[stack_slice]

            if comm.block.rank == 0:
                # Direction: downward Schur-complement
                _serinv.downward_schur(
                    a_,
                    x_diag_blocks,
                    obc_blocks,
                    stack_slice=stack_slice,
                    invert_last_block=False,
                )
            elif comm.block.rank == comm.block.size - 1:
                # Direction: upward Schur-complement
                _serinv.upward_schur(
                    a_,
                    x_diag_blocks,
                    obc_blocks,
                    stack_slice=stack_slice,
                    invert_last_block=False,
                )
            else:
                # Permuted Schur-complement
                _serinv.permuted_schur(
                    a_,
                    x_diag_blocks,
                    buffer_lower,
                    buffer_upper,
                    obc_blocks,
                    stack_slice=stack_slice,
                )

            # Construct the reduced system.
            if np.all(a.block_sizes == a.block_sizes[0]):
                gather_reduced_system = reduced_system.gather_constant_block_size
            else:
                # If the block sizes are not the same, we need to use pickle.
                gather_reduced_system = reduced_system.gather

            gather_reduced_system(a_, x_diag_blocks, buffer_upper, buffer_lower)
            # Perform selected-inversion on the reduced system.
            reduced_system.solve()
            # Scatter the result to the output matrix.
            reduced_system.scatter(x_diag_blocks, buffer_upper, buffer_lower, out_)

            if comm.block.rank == 0:
                # Direction: upward sell-inv
                _serinv.downward_selinv(a_, x_diag_blocks, out_)
            elif comm.block.rank == comm.block.size - 1:
                # Direction: downward sell-inv
                _serinv.upward_selinv(a_, x_diag_blocks, out_)
            else:
                # Permuted Sell-inv
                _serinv.permuted_selinv(
                    a_, x_diag_blocks, buffer_lower, buffer_upper, out_
                )

    def selected_solve(
        self,
        a: DSDBSparse | _DStackView,
        sigma_lesser: DSDBSparse | _DStackView,
        sigma_greater: DSDBSparse | _DStackView,
        out: tuple[DSDBSparse, ...] | tuple[_DStackView, ...],
        obc_blocks: OBCBlocks | None = None,
        return_retarded: bool = False,
        return_current: bool = False,
        n_offdiagonals: int = 1,
    ) -> None | NDArray:
        r"""Performs selected inversion of a block-tridiagonal matrix.

        Can optionally solve the quadratic system associated with the
        lesser and greater right-hand-sides.

        Parameters
        ----------
        a : DSDBSparse
            Matrix to invert.
        sigma_lesser : DSDBSparse
            Lesser matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        sigma_greater : DSDBSparse
            Greater matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        out : tuple[DSDBSparse, ...]
            Preallocated output matrices
        obc_blocks : dict[int, OBCBlocks], optional
            OBC blocks for lesser, greater and retarded Green's
            functions, by default None.
        return_retarded : bool, optional
            Wether the retarded Green's function should be returned
            along with lesser and greater, by default False
        return_current : bool, optional
            Whether to compute and return the current for each layer via
            the Meir-Wingreen formula. By default False. Note that this
            is currently only partially supported, and only the boundary
            currents are computed correctly.
        n_offdiagonals : int, optional
            Highest off-diagonal block order of X^{<,>} to compute and
            write, including skew-hermitian mirrors. Values wider than the
            matrix are clipped. Bands above one require ``return_retarded``
            and an output pattern containing all requested blocks. By default
            1.

        Returns
        -------
        None | NDArray
            If `return_current` is True, returns the
            current for each layer.

        """
        if n_offdiagonals < 1:
            raise ValueError("n_offdiagonals must be positive.")
        if n_offdiagonals > 1:
            if not return_retarded:
                raise ValueError(
                    "n_offdiagonals > 1 requires return_retarded=True (the "
                    "off-diagonal post-pass reads the fully-connected X^R "
                    "off-diagonals from the output)."
                )

        with profiler.profile_range(
            label="RGF dist: init", level="default", comm=comm.block
        ):

            if obc_blocks is None:
                obc_blocks = OBCBlocks(num_blocks=sigma_lesser.num_local_blocks)

            if return_current:
                # Allocate a buffer for the current. This includes current
                # between each layer and from/to the leads (in total
                # num_blocks + 1).
                current = xp.zeros(
                    (*sigma_lesser.local_stack_shape, sigma_lesser.num_blocks + 1),
                    dtype=sigma_lesser.dtype,
                )
                # TODO: Only boundary currents are currently supported.
                # Invalidate the remaining layers by setting them to
                # xp.nan.
                current[..., 1:-1] = xp.nan

            xl_out, xg_out, *xr_out = out
            if return_retarded:
                if len(xr_out) != 1:
                    raise ValueError("Invalid number of output matrices.")
                xr_out = xr_out[0]

            if xl_out.symmetry not in [None, "skew-hermitian"]:
                raise ValueError(
                    "Invalid symmetry for lesser Green's function. "
                    "Expected None or 'skew-hermitian'."
                )
            if xg_out.symmetry not in [None, "skew-hermitian"]:
                raise ValueError(
                    "Invalid symmetry for greater Green's function. "
                    "Expected None or 'skew-hermitian'."
                )

            batch_sizes, batch_offsets = get_batches(
                sigma_lesser.local_stack_shape[0], self.max_batch_size
            )

        for i in range(len(batch_sizes)):

            # Initialize temporary buffers.
            reduced_system = _serinv.ReducedSystem(selected_solve=True)

            xr_diag_blocks: list[NDArray | None] = [
                None
            ] * sigma_lesser.num_local_blocks
            xr_buffer_lower: list[NDArray | None] = [
                None
            ] * sigma_lesser.num_local_blocks
            xr_buffer_upper: list[NDArray | None] = [
                None
            ] * sigma_lesser.num_local_blocks

            xl_diag_blocks: list[NDArray | None] = [
                None
            ] * sigma_lesser.num_local_blocks
            xl_buffer_lower = None
            xl_buffer_upper: list[NDArray | None] = [
                None
            ] * sigma_lesser.num_local_blocks

            xg_diag_blocks: list[NDArray | None] = [
                None
            ] * sigma_lesser.num_local_blocks
            xg_buffer_lower = None
            xg_buffer_upper: list[NDArray | None] = [
                None
            ] * sigma_lesser.num_local_blocks

            stack_slice = slice(int(batch_offsets[i]), int(batch_offsets[i + 1]))

            a_ = a.stack[stack_slice]
            sigma_lesser_ = sigma_lesser.stack[stack_slice]
            sigma_greater_ = sigma_greater.stack[stack_slice]

            xl_out_ = xl_out.stack[stack_slice]
            xg_out_ = xg_out.stack[stack_slice]
            xr_out_ = xr_out.stack[stack_slice] if return_retarded else None

            with profiler.profile_range(
                label="RGF dist: Schur", level="default", comm=comm.block
            ):

                if comm.block.rank == 0:
                    # Direction: downward Schur-complement
                    _serinv.downward_schur(
                        a=a_,
                        xr_diag_blocks=xr_diag_blocks,
                        # Lesser quantities.
                        sigma_lesser=sigma_lesser_,
                        xl_diag_blocks=xl_diag_blocks,
                        # Greater quantities.
                        sigma_greater=sigma_greater_,
                        xg_diag_blocks=xg_diag_blocks,
                        # OBC and settings.
                        obc_blocks=obc_blocks,
                        stack_slice=stack_slice,
                        invert_last_block=False,
                        selected_solve=True,
                    )
                elif comm.block.rank == comm.block.size - 1:
                    # Direction: upward Schur-complement
                    _serinv.upward_schur(
                        a=a_,
                        xr_diag_blocks=xr_diag_blocks,
                        # Lesser quantities.
                        sigma_lesser=sigma_lesser_,
                        xl_diag_blocks=xl_diag_blocks,
                        # Greater quantities.
                        sigma_greater=sigma_greater_,
                        xg_diag_blocks=xg_diag_blocks,
                        # OBC and settings.
                        obc_blocks=obc_blocks,
                        stack_slice=stack_slice,
                        invert_last_block=False,
                        selected_solve=True,
                    )
                else:
                    # Permuted Schur-complement
                    _serinv.permuted_schur(
                        a=a_,
                        xr_diag_blocks=xr_diag_blocks,
                        xr_buffer_lower=xr_buffer_lower,
                        xr_buffer_upper=xr_buffer_upper,
                        # Lesser quantities.
                        sigma_lesser=sigma_lesser_,
                        xl_diag_blocks=xl_diag_blocks,
                        xl_buffer_lower=xl_buffer_lower,
                        xl_buffer_upper=xl_buffer_upper,
                        # Greater quantities.
                        sigma_greater=sigma_greater_,
                        xg_diag_blocks=xg_diag_blocks,
                        xg_buffer_lower=xg_buffer_lower,
                        xg_buffer_upper=xg_buffer_upper,
                        # OBC and settings.
                        obc_blocks=obc_blocks,
                        stack_slice=stack_slice,
                        selected_solve=True,
                    )

            if n_offdiagonals > 1:
                with profiler.profile_range(
                    label="RGF dist: left-connected chain",
                    level="default",
                    comm=comm.block,
                ):
                    glr, gll, glg = _left_connected_chain(
                        a_,
                        sigma_lesser_,
                        sigma_greater_,
                        obc_blocks,
                        stack_slice,
                        xr_diag_blocks,
                        xl_diag_blocks,
                        xg_diag_blocks,
                    )

            with profiler.profile_range(
                label="RGF dist: Reduce gather", level="default", comm=comm.block
            ):
                # Construct the reduced system.
                if np.all(a.block_sizes == a.block_sizes[0]):
                    gather_reduced_system = reduced_system.gather_constant_block_size
                else:
                    # If the block sizes are not the same, we need to use pickle.
                    gather_reduced_system = reduced_system.gather

                gather_reduced_system(
                    a=a_,
                    xr_diag_blocks=xr_diag_blocks,
                    xr_buffer_lower=xr_buffer_lower,
                    xr_buffer_upper=xr_buffer_upper,
                    # Lesser quantities.
                    sigma_lesser=sigma_lesser_,
                    xl_diag_blocks=xl_diag_blocks,
                    xl_buffer_lower=xl_buffer_lower,
                    xl_buffer_upper=xl_buffer_upper,
                    # Greater quantities.
                    sigma_greater=sigma_greater_,
                    xg_diag_blocks=xg_diag_blocks,
                    xg_buffer_lower=xg_buffer_lower,
                    xg_buffer_upper=xg_buffer_upper,
                )

            # Perform selected-inversion on the reduced system.
            with profiler.profile_range(
                label="RGF dist: Reduce solve", level="default", comm=comm.block
            ):
                reduced_system.solve()

            with profiler.profile_range(
                label="RGF dist: Reduce scatter", level="default", comm=comm.block
            ):
                # Scatter the result to the output matrix.
                reduced_system.scatter(
                    xr_diag_blocks=xr_diag_blocks,
                    xr_buffer_lower=xr_buffer_lower,
                    xr_buffer_upper=xr_buffer_upper,
                    xr_out=xr_out_,
                    return_retarded=return_retarded,
                    # Lesser quantities.
                    xl_diag_blocks=xl_diag_blocks,
                    xl_buffer_lower=xl_buffer_lower,
                    xl_buffer_upper=xl_buffer_upper,
                    xl_out=xl_out_,
                    # Greater quantities.
                    xg_diag_blocks=xg_diag_blocks,
                    xg_buffer_lower=xg_buffer_lower,
                    xg_buffer_upper=xg_buffer_upper,
                    xg_out=xg_out_,
                )

            with profiler.profile_range(
                label="RGF dist: Selinv", level="default", comm=comm.block
            ):

                if comm.block.rank == 0:
                    # Direction: upward sell-inv
                    _serinv.downward_selinv(
                        a=a_,
                        xr_diag_blocks=xr_diag_blocks,
                        xr_out=xr_out_,
                        # Lesser quantities.
                        sigma_lesser=sigma_lesser_,
                        xl_diag_blocks=xl_diag_blocks,
                        xl_out=xl_out_,
                        # Greater quantities.
                        sigma_greater=sigma_greater_,
                        xg_diag_blocks=xg_diag_blocks,
                        xg_out=xg_out_,
                        selected_solve=True,
                        return_retarded=return_retarded,
                    )
                elif comm.block.rank == comm.block.size - 1:
                    # Direction: downward sell-inv
                    _serinv.upward_selinv(
                        a=a_,
                        xr_diag_blocks=xr_diag_blocks,
                        xr_out=xr_out_,
                        # Lesser quantities.
                        sigma_lesser=sigma_lesser_,
                        xl_diag_blocks=xl_diag_blocks,
                        xl_out=xl_out_,
                        # Greater quantities.
                        sigma_greater=sigma_greater_,
                        xg_diag_blocks=xg_diag_blocks,
                        xg_out=xg_out_,
                        selected_solve=True,
                        return_retarded=return_retarded,
                    )
                else:
                    # Permuted Sell-inv
                    _serinv.permuted_selinv(
                        a=a_,
                        xr_diag_blocks=xr_diag_blocks,
                        xr_buffer_lower=xr_buffer_lower,
                        xr_buffer_upper=xr_buffer_upper,
                        xr_out=xr_out_,
                        # Lesser quantities.
                        sigma_lesser=sigma_lesser_,
                        xl_diag_blocks=xl_diag_blocks,
                        # xl_buffer_lower=xl_buffer_lower,
                        xl_buffer_upper=xl_buffer_upper,
                        xl_out=xl_out_,
                        # Greater quantities.
                        sigma_greater=sigma_greater_,
                        xg_diag_blocks=xg_diag_blocks,
                        # xg_buffer_lower=xg_buffer_lower,
                        xg_buffer_upper=xg_buffer_upper,
                        xg_out=xg_out_,
                        selected_solve=True,
                        return_retarded=return_retarded,
                    )

            if n_offdiagonals > 1:
                with profiler.profile_range(
                    label="RGF dist: off-diagonal post-pass",
                    level="default",
                    comm=comm.block,
                ):
                    _offdiag_post_pass(
                        a_,
                        sigma_lesser_,
                        sigma_greater_,
                        xl_out_,
                        xg_out_,
                        xr_out_,
                        glr,
                        gll,
                        glg,
                        n_offdiagonals,
                    )

            if return_current:
                if comm.block.rank == 0:
                    current[stack_slice, ..., 0] = xp.trace(
                        obc_blocks.greater[0][stack_slice] @ xl_diag_blocks[0]
                        - xg_diag_blocks[0] @ obc_blocks.lesser[0][stack_slice],
                        axis1=-2,
                        axis2=-1,
                    )
                if comm.block.rank == comm.block.size - 1:
                    # NOTE: Negative sign is needed to get the current flowing
                    # in the correct direction (positive from left to right).
                    current[stack_slice, ..., -1] = -xp.trace(
                        obc_blocks.greater[-1][stack_slice] @ xl_diag_blocks[-1]
                        - xg_diag_blocks[-1] @ obc_blocks.lesser[-1][stack_slice],
                        axis1=-2,
                        axis2=-1,
                    )

        if return_current:
            # Now we need to allreduce the current across the block
            # communicator to get the total current for each layer.
            # NOTE: We use allreduce instead of allgather since every
            # rank allocates the full current
            total_current = xp.empty_like(current)
            comm.block.all_reduce(current, total_current, op="sum")

            return total_current
