# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

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
    """Emit the selected 2nd (and 3rd) off-diagonal X^{<,>} blocks.

    Uniform local post-pass over this rank's rows, identical for all
    three selinv variants: the k-th off-diagonal identities (the same
    upward-propagation used by the single-node RGF backward sweep,
    ``rgf.py``) read ONLY fully-connected first off-diagonals already
    written to the outputs, the left-connected auxiliaries from
    :func:`_left_connected_chain`, and local A / Sigma blocks:

        X^x_{i,m} = -g^L_i A_{i,i+1} X^x_{i+1,m}
                    + (g^L_i Sigma^x_{i,i+1} - g^{L,x}_i A_{i+1,i}^dag)
                      (X^R_{m,i+1})^dag,
        X^R_{i+3,i+1} = -X^R_{i+3,i+2} A_{i+2,i+1} g^L_{i+1}.

    Rows whose operands cross into the next partition use a single
    point-to-point halo (received before those rows are computed); the
    boundary-crossing OUTPUT blocks are owned by this (left) rank via
    the arrow-wise partitioning, so no write ownership changes.
    """
    n = a.num_local_blocks
    bcomm = comm.block._mpi_comm
    rank, size = comm.block.rank, comm.block.size
    k3 = n_offdiagonals >= 3

    def _emit(i, m_l, m_g, xa):
        """Write X^{<,>}_{i, i+d} (and mirrors) from the identity."""
        a_ij = a.blocks[i, i + 1]
        a_ji_dag = _dag(a.blocks[i + 1, i])
        sl_ij = sigma_lesser.blocks[i, i + 1]
        sg_ij = sigma_greater.blocks[i, i + 1]
        gA = glr[i] @ a_ij
        xl_val = -gA @ m_l + (glr[i] @ sl_ij - gll[i] @ a_ji_dag) @ xa
        xg_val = -gA @ m_g + (glr[i] @ sg_ij - glg[i] @ a_ji_dag) @ xa
        return xl_val, xg_val

    def _write(i, d, xl_val, xg_val):
        xl_out.blocks[i, i + d] = xl_val
        xg_out.blocks[i, i + d] = xg_val
        if xl_out.symmetry is None:
            xl_out.blocks[i + d, i] = -_dag(xl_val)
        if xg_out.symmetry is None:
            xg_out.blocks[i + d, i] = -_dag(xg_val)

    # ---- step 1: halo-independent k=2 rows (i + 2 <= n: operands local
    # or already written to this rank's outputs by selinv / mapback).
    # The LAST rank has no next partition: its rows clip at the global
    # matrix end (i + 2 <= n - 1), exactly like the single-node sweep.
    d2_l: dict[int, NDArray] = {}
    d2_g: dict[int, NDArray] = {}
    top2 = n - 2 if rank + 1 < size else n - 3
    for i in range(top2, -1, -1):
        xl_val, xg_val = _emit(
            i,
            xl_out.blocks[i + 1, i + 2],
            xg_out.blocks[i + 1, i + 2],
            _dag(xr_out.blocks[i + 2, i + 1]),
        )
        _write(i, 2, xl_val, xg_val)
        d2_l[i], d2_g[i] = xl_val, xg_val

    # ---- step 2: single neighbour halo. My top-row content flows UP to
    # rank-1; I receive the mirror package from rank+1. Everything sent
    # is halo-independent (my rows 0..2), so no pipeline forms.
    halo = None
    req = None
    if rank > 0:
        package = {
            "xl_01": xl_out.blocks[0, 1],
            "xg_01": xg_out.blocks[0, 1],
            "xr_10": xr_out.blocks[1, 0],
        }
        if k3:
            package.update(
                xl_02=d2_l[0], xg_02=d2_g[0],
                xr_21=xr_out.blocks[2, 1],
                a_10=a.blocks[1, 0],
                glr_0=glr[0],
            )
        req = bcomm.isend(package, dest=rank - 1, tag=72)
    if rank + 1 < size:
        halo = bcomm.recv(source=rank + 1, tag=72)
    if req is not None:
        req.wait()

    # ---- step 3: boundary k=2 row (i = n-1; operand rows f=n, f+1 live
    # in the next partition). The output block (n-1, n+1) and its mirror
    # are in this rank's arrow-wise strip.
    if halo is not None:
        xl_val, xg_val = _emit(
            n - 1, halo["xl_01"], halo["xg_01"], _dag(halo["xr_10"])
        )
        _write(n - 1, 2, xl_val, xg_val)
        d2_l[n - 1], d2_g[n - 1] = xl_val, xg_val

    if not k3:
        return

    # ---- step 4: halo-independent k=3 rows (operand rows <= n, all
    # local: X_{i+1,i+3} is a step-1 k=2 result, X^R rows <= n are
    # selinv/mapback outputs, A_{i+2,i+1} is local for i <= n-3). The
    # last rank clips at the global matrix end (i + 3 <= n - 1).
    top3 = n - 3 if rank + 1 < size else n - 4
    for i in range(top3, -1, -1):
        # For i = n-3 the retarded operand is the boundary coupling
        # X^R_{n,n-1} written to this rank's output by the reduced-system
        # mapback; all other operands are interior selinv outputs.
        xr_d3 = (
            -xr_out.blocks[i + 3, i + 2] @ a.blocks[i + 2, i + 1] @ glr[i + 1]
        )
        xl_val, xg_val = _emit(i, d2_l[i + 1], d2_g[i + 1], _dag(xr_d3))
        _write(i, 3, xl_val, xg_val)

    # ---- step 5: boundary k=3 rows n-2 and n-1.
    if halo is not None:
        # Row n-2: X_{n-1,n+1} is my own boundary k=2 (step 3);
        # X^R_{n+1,n} is the halo's xr_10; A_{n,n-1} is in my strip.
        xr_d3 = -halo["xr_10"] @ a.blocks[n, n - 1] @ glr[n - 1]
        xl_val, xg_val = _emit(
            n - 2, d2_l[n - 1], d2_g[n - 1], _dag(xr_d3)
        )
        _write(n - 2, 3, xl_val, xg_val)
        # Row n-1: X_{n,n+2} is the neighbour's own k=2 at its row 0;
        # X^R_{n+2,n+1}, A_{n+1,n} and g^L_n come from the halo.
        xr_d3 = -halo["xr_21"] @ halo["a_10"] @ halo["glr_0"]
        xl_val, xg_val = _emit(
            n - 1, halo["xl_02"], halo["xg_02"], _dag(xr_d3)
        )
        _write(n - 1, 3, xl_val, xg_val)


class RGFDist(GFSolver):
    """Distributed selected inversion solver.

    Parameters
    ----------
    solve_lesser : bool, optional
        Whether to solve the quadratic system associated with the lesser right-hand-side,
        by default False.
    solve_greater : bool, optional
        Whether to solve the quadratic system associated with the greater right-hand-side,
        by default False.
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
        Bl and Bg matrices in the equation AXA^T = B.

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
            write (1 = block-tridiagonal only, the default; 2 also
            writes X^{<,>}_{i,i+2}; 3 also writes X^{<,>}_{i,i+3}),
            together with their skew-hermitian mirrors. Requires
            ``return_retarded=True`` (the post-pass reads the
            fully-connected X^R off-diagonals from the output),
            every comm.block rank to own at least ``n_offdiagonals + 1``
            block rows, and the output sparsity pattern to contain the
            extra off-diagonal blocks -- including the boundary-crossing
            ones in each rank's arrow-wise strip (writes to absent
            blocks drop silently). Only 1, 2 or 3 are supported. By
            default 1.

        Returns
        -------
        None | NDArray
            If `return_current` is True, returns the
            current for each layer.

        """
        if n_offdiagonals not in (1, 2, 3):
            raise ValueError(
                f"n_offdiagonals must be 1, 2, or 3 (got {n_offdiagonals})."
            )
        if n_offdiagonals > 1:
            if not return_retarded:
                raise ValueError(
                    "n_offdiagonals > 1 requires return_retarded=True (the "
                    "off-diagonal post-pass reads the fully-connected X^R "
                    "off-diagonals from the output)."
                )
            if sigma_lesser.num_local_blocks < n_offdiagonals + 1:
                raise ValueError(
                    f"n_offdiagonals={n_offdiagonals} requires every "
                    f"comm.block rank to own at least {n_offdiagonals + 1} "
                    f"block rows (this rank owns "
                    f"{sigma_lesser.num_local_blocks}); reduce "
                    "block_comm_size."
                )

        with profiler.profile_range(label="RGF dist: init", level="default", comm=comm):

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

            if obc_blocks is None:
                obc_blocks = OBCBlocks(num_blocks=sigma_lesser.num_local_blocks)

            if return_current:
                # Allocate a buffer for the current. This includes current
                # between each layer and from/to the leads (in total
                # num_blocks + 1).
                current = xp.zeros(
                    (*sigma_lesser.shape[:-2], sigma_lesser.num_blocks + 1),
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
                sigma_lesser.shape[0], self.max_batch_size
            )

        for i in range(len(batch_sizes)):
            stack_slice = slice(int(batch_offsets[i]), int(batch_offsets[i + 1]))

            a_ = a.stack[stack_slice]
            sigma_lesser_ = sigma_lesser.stack[stack_slice]
            sigma_greater_ = sigma_greater.stack[stack_slice]

            xl_out_ = xl_out.stack[stack_slice]
            xg_out_ = xg_out.stack[stack_slice]
            xr_out_ = xr_out.stack[stack_slice] if return_retarded else None

            with profiler.profile_range(
                label="RGF dist: Schur", level="debug", comm=comm
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
                # Left-connected auxiliaries for the off-diagonal
                # post-pass: stash them NOW -- the reduced-system mapback
                # and the selinv overwrite the diag stacks.
                with profiler.profile_range(
                    label="RGF dist: left-connected chain",
                    level="debug",
                    comm=comm,
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
                label="RGF dist: Reduce gather", level="debug", comm=comm
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
                label="RGF dist: Reduce solve", level="debug", comm=comm
            ):
                reduced_system.solve()

            with profiler.profile_range(
                label="RGF dist: Reduce scatter", level="debug", comm=comm
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
                label="RGF dist: Selinv", level="debug", comm=comm
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
                    level="debug",
                    comm=comm,
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
                # Per-batch boundary traces: the diagonal blocks are the
                # current batch's, so the OBC blocks and the output slots
                # must be sliced to the same stack window.
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
