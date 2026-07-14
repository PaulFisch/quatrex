# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, xp
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.greens_function_solver.solver import GFSolver, OBCBlocks
from qttools.kernels import linalg
from qttools.utils.solvers_utils import get_batches


class RGF(GFSolver):
    """Selected inversion solver based on the Schur complement.

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
        # Initialize dense temporary buffers for the diagonal blocks.
        x_diag_blocks: list[NDArray | None] = [None] * a.num_blocks

        if obc_blocks is None:
            obc_blocks = OBCBlocks(num_blocks=a.num_blocks)

        # Get list of batches to perform
        batches_sizes, batches_slices = get_batches(a.shape[0], self.max_batch_size)

        x = out

        for b in range(len(batches_sizes)):
            stack_slice = slice(int(batches_slices[b]), int(batches_slices[b + 1]), 1)

            a_ = a.stack[stack_slice]
            x_ = x.stack[stack_slice]

            # See if there is an OBC block for the current layer.
            obc = obc_blocks.retarded[0]
            a_00 = (
                a_.blocks[0, 0] if obc is None else a_.blocks[0, 0] - obc[stack_slice]
            )

            x_diag_blocks[0] = linalg.inv(a_00)

            # Forwards sweep.
            for i in range(a.num_blocks - 1):
                j = i + 1

                # See if there is an OBC block for the current layer.
                obc = obc_blocks.retarded[j]
                a_jj = (
                    a_.blocks[j, j]
                    if obc is None
                    else a_.blocks[j, j] - obc[stack_slice]
                )

                x_diag_blocks[j] = linalg.inv(
                    a_jj - a_.blocks[j, i] @ x_diag_blocks[i] @ a_.blocks[i, j]
                )

            # We need to write the last diagonal block to the output.
            x_.blocks[a.num_blocks - 1, a.num_blocks - 1] = x_diag_blocks[-1]

            # Backwards sweep.
            for i in range(a.num_blocks - 2, -1, -1):
                j = i + 1

                x_ii = x_diag_blocks[i]
                x_jj = x_diag_blocks[j]
                a_ij = a_.blocks[i, j]

                x_ji = -x_jj @ a_.blocks[j, i] @ x_ii
                x_.blocks[j, i] = x_ji
                x_.blocks[i, j] = -x_ii @ a_ij @ x_jj

                # NOTE: Cursed Python multiple assignment syntax.
                x_.blocks[i, i] = x_diag_blocks[i] = x_ii - x_ii @ a_ij @ x_ji

    def selected_solve(
        self,
        a: DSDBSparse,
        sigma_lesser: DSDBSparse,
        sigma_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
        obc_blocks: OBCBlocks | None = None,
        return_retarded: bool = False,
        return_current: bool = False,
        second_offdiagonals: bool = False,
    ) -> None | NDArray:
        r"""Produces elements of the solution to the congruence equation.

        This method produces selected elements of the solution to the
        relation:

        \[
            X^{\lessgtr} = A^{-1} \Sigma^{\lessgtr} A^{-\dagger}
        \]

        Parameters
        ----------
        a : DSDBSparse
            Matrix to invert.
        sigma_lesser : DSDBSparse
            Lesser matrix. This matrix is expected to be skew-hermitian,
            i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        sigma_greater : DSDBSparse
            Greater matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        out : tuple[DSDBSparse, ...]
            Preallocated output matrices.
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.
        return_retarded : bool, optional
            Wether the retarded Green's function should be returned
            along with lesser and greater, by default False
        return_current : bool, optional
            Whether to compute and return the current for each layer via
            the Meir-Wingreen formula. By default False.
        second_offdiagonals : bool, optional
            Additionally compute the second off-diagonal blocks
            X^{<,>}_{i,i+2} (and their skew-hermitian mirrors) and write
            them to the output. The output pattern must contain these
            blocks (writes to absent blocks drop silently). The
            block-tridiagonal outputs are bit-identical to the default
            path. By default False.

        Returns
        -------
        None | NDArray
            If `return_current` is True, returns the current for each
            layer.

        """
        # Initialize empty lists for the dense diagonal blocks.
        xr_diag_blocks: list[NDArray | None] = [None] * sigma_lesser.num_blocks
        xl_diag_blocks: list[NDArray | None] = [None] * sigma_lesser.num_blocks
        xg_diag_blocks: list[NDArray | None] = [None] * sigma_lesser.num_blocks

        if obc_blocks is None:
            obc_blocks = OBCBlocks(num_blocks=sigma_lesser.num_blocks)

        if return_current:
            # Allocate a buffer for the current. This includes current
            # between each layer and from/to the leads (in total
            # num_blocks + 1).
            current = xp.zeros(
                (*sigma_lesser.shape[:-2], sigma_lesser.num_blocks + 1),
                dtype=sigma_lesser.dtype,
            )

        # Get list of batches to perform
        batches_sizes, batches_slices = get_batches(
            sigma_lesser.shape[0], self.max_batch_size
        )

        # xr will be the third element of the tuple.
        xl, xg, *xr = out
        if return_retarded:
            if len(xr) != 1:
                raise ValueError("Invalid number of output matrices.")
            xr = xr[0]

        if xl.symmetry not in [None, "skew-hermitian"]:
            raise ValueError(
                "Invalid symmetry for lesser Green's function. "
                "Expected None or 'skew-hermitian'."
            )
        if xg.symmetry not in [None, "skew-hermitian"]:
            raise ValueError(
                "Invalid symmetry for greater Green's function. "
                "Expected None or 'skew-hermitian'."
            )

        # Perform the selected solve by batches.
        for b in range(len(batches_sizes)):
            stack_slice = slice(int(batches_slices[b]), int(batches_slices[b + 1]), 1)

            a_ = a.stack[stack_slice]
            sigma_lesser_ = sigma_lesser.stack[stack_slice]
            sigma_greater_ = sigma_greater.stack[stack_slice]

            xl_ = xl.stack[stack_slice]
            xg_ = xg.stack[stack_slice]
            if return_retarded:
                xr_ = xr.stack[stack_slice]

            # Check if there are OBC blocks for the current layer.
            obc_r = obc_blocks.retarded[0]
            a_jj = (
                a_.blocks[0, 0]
                if obc_r is None
                else a_.blocks[0, 0] - obc_r[stack_slice]
            )
            obc_l = obc_blocks.lesser[0]
            sl_jj = (
                sigma_lesser_.blocks[0, 0]
                if obc_l is None
                else sigma_lesser_.blocks[0, 0] + obc_l[stack_slice]
            )
            obc_g = obc_blocks.greater[0]
            sg_jj = (
                sigma_greater_.blocks[0, 0]
                if obc_g is None
                else sigma_greater_.blocks[0, 0] + obc_g[stack_slice]
            )

            xr_jj = linalg.inv(a_jj)
            xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)
            xr_diag_blocks[0] = xr_jj
            xl_diag_blocks[0] = xr_jj @ sl_jj @ xr_jj_dagger
            xg_diag_blocks[0] = xr_jj @ sg_jj @ xr_jj_dagger

            # Forwards sweep.
            for i in range(a.num_blocks - 1):
                j = i + 1

                # Check if there are OBC blocks for the current layer.
                obc_r = obc_blocks.retarded[j]
                a_jj = (
                    a_.blocks[j, j]
                    if obc_r is None
                    else a_.blocks[j, j] - obc_r[stack_slice]
                )
                obc_l = obc_blocks.lesser[j]
                sl_jj = (
                    sigma_lesser_.blocks[j, j]
                    if obc_l is None
                    else sigma_lesser_.blocks[j, j] + obc_l[stack_slice]
                )
                obc_g = obc_blocks.greater[j]
                sg_jj = (
                    sigma_greater_.blocks[j, j]
                    if obc_g is None
                    else sigma_greater_.blocks[j, j] + obc_g[stack_slice]
                )

                # Get the blocks that are used multiple times.
                a_ji = a_.blocks[j, i]
                xr_ii = xr_diag_blocks[i]

                # Precompute the transposes that are used multiple times.
                a_ji_dagger = a_ji.conj().swapaxes(-2, -1)

                # Precompute some terms that are used multiple times.
                a_ji_xr_ii = a_ji @ xr_ii

                xr_jj = linalg.inv(a_jj - a_ji_xr_ii @ a_.blocks[i, j])
                xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)
                xr_diag_blocks[j] = xr_jj

                a_ji_xr_ii_sx_ij = a_ji_xr_ii @ sigma_lesser_.blocks[i, j]

                xl_diag_blocks[j] = (
                    xr_jj
                    @ (
                        sl_jj
                        + a_ji @ xl_diag_blocks[i] @ a_ji_dagger
                        + a_ji_xr_ii_sx_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sx_ij
                    )
                    @ xr_jj_dagger
                )

                a_ji_xr_ii_sx_ij = a_ji_xr_ii @ sigma_greater_.blocks[i, j]

                xg_diag_blocks[j] = (
                    xr_jj
                    @ (
                        sg_jj
                        + a_ji @ xg_diag_blocks[i] @ a_ji_dagger
                        + a_ji_xr_ii_sx_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sx_ij
                    )
                    @ xr_jj_dagger
                )

            # We need to write the last diagonal blocks to the output.
            xl_.blocks[a.num_blocks - 1, a.num_blocks - 1] = 0.5 * (
                xl_diag_blocks[-1] - xl_diag_blocks[-1].conj().swapaxes(-2, -1)
            )
            xg_.blocks[a.num_blocks - 1, a.num_blocks - 1] = 0.5 * (
                xg_diag_blocks[-1] - xg_diag_blocks[-1].conj().swapaxes(-2, -1)
            )
            if return_retarded:
                xr_.blocks[a.num_blocks - 1, a.num_blocks - 1] = xr_diag_blocks[-1]

            # Backwards sweep.
            compute_d2 = second_offdiagonals and a.num_blocks >= 3
            # Rolling stashes from the previous backward step (row i+1):
            # X^{<,>}_{i+1,i+2} and the fully-connected first
            # off-diagonals X^R_{i+1,i+2} / X^R_{i+2,i+1}.
            prev_xl_up = prev_xg_up = prev_xr_dn = None
            for i in range(a.num_blocks - 2, -1, -1):
                j = i + 1

                # Get the blocks that are used multiple times.
                xr_ii = xr_diag_blocks[i]
                xr_jj = xr_diag_blocks[j]
                a_ij = a_.blocks[i, j]
                a_ji = a_.blocks[j, i]
                xl_ii = xl_diag_blocks[i]
                xl_jj = xl_diag_blocks[j]
                xg_ii = xg_diag_blocks[i]
                xg_jj = xg_diag_blocks[j]
                sigma_lesser_ij = sigma_lesser_.blocks[i, j]
                sigma_greater_ij = sigma_greater_.blocks[i, j]

                # Precompute the transposes that are used multiple times.
                xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)

                # Precompute the terms that are used multiple times.
                xr_ii_a_ij = xr_ii @ a_ij
                a_ij_dagger_xr_ii_dagger = xr_ii_a_ij.conj().swapaxes(-2, -1)
                xr_jj_a_ji = xr_jj @ a_ji
                a_ji_dagger_xr_jj_dagger = xr_jj_a_ji.conj().swapaxes(-2, -1)
                xr_ii_a_ij_xr_jj = xr_ii_a_ij @ xr_jj
                xr_jj_dagger_a_ij_dagger_xr_ii_dagger = (
                    xr_ii_a_ij_xr_jj.conj().swapaxes(-2, -1)
                )
                xr_ii_a_ij_xr_jj_a_ji = xr_ii_a_ij @ xr_jj_a_ji

                temp_1x = (
                    xr_ii_a_ij_xr_jj_a_ji @ xl_ii
                    - xr_ii @ sigma_lesser_ij @ xr_jj_dagger_a_ij_dagger_xr_ii_dagger
                )
                temp_1x -= temp_1x.conj().swapaxes(-2, -1)
                temp_2x = xr_ii_a_ij @ xl_jj

                xl_ij = (
                    -temp_2x
                    - xl_ii @ a_ji_dagger_xr_jj_dagger
                    + xr_ii @ sigma_lesser_ij @ xr_jj_dagger
                )

                xl_.blocks[i, j] = xl_ij
                if xl_.symmetry is None:
                    xl_.blocks[j, i] = -xl_ij.conj().swapaxes(-2, -1)

                xl_diag_blocks[i] = xl_ii + temp_2x @ a_ij_dagger_xr_ii_dagger + temp_1x
                xl_.blocks[i, i] = 0.5 * (
                    xl_diag_blocks[i] - xl_diag_blocks[i].conj().swapaxes(-2, -1)
                )

                temp_1x = (
                    xr_ii_a_ij_xr_jj_a_ji @ xg_ii
                    - xr_ii @ sigma_greater_ij @ xr_jj_dagger_a_ij_dagger_xr_ii_dagger
                )
                temp_1x -= temp_1x.conj().swapaxes(-2, -1)
                temp_2x = xr_ii_a_ij @ xg_jj

                xg_ij = (
                    -temp_2x
                    - xg_ii @ a_ji_dagger_xr_jj_dagger
                    + xr_ii @ sigma_greater_ij @ xr_jj_dagger
                )

                xg_.blocks[i, j] = xg_ij
                if xg_.symmetry is None:
                    xg_.blocks[j, i] = -xg_ij.conj().swapaxes(-2, -1)

                xg_diag_blocks[i] = xg_ii + temp_2x @ a_ij_dagger_xr_ii_dagger + temp_1x
                xg_.blocks[i, i] = 0.5 * (
                    xg_diag_blocks[i] - xg_diag_blocks[i].conj().swapaxes(-2, -1)
                )

                if compute_d2:
                    # Fully-connected first lower off-diagonal of X^R at
                    # this step (xr_ii is still the left-connected g^L_i):
                    #   X^R_{i+1,i} = -X^R_{i+1,i+1} A_{i+1,i} g^L_i
                    xr_dn_i = -xr_jj_a_ji @ xr_ii
                    if i + 2 < a.num_blocks:
                        # Second off-diagonals by upward propagation with
                        # the left-connected auxiliaries (g^L_i = xr_ii,
                        # g^{L,x}_i = xl_ii / xg_ii); same identity as the
                        # first off-diagonal with the (i+1, i+1) column
                        # objects replaced by the (i+1, i+2) ones:
                        #   X^x_{i,i+2} = -g^L_i A_{i,i+1} X^x_{i+1,i+2}
                        #     + (g^L_i Sigma^x_{i,i+1}
                        #        - g^{L,x}_i A_{i+1,i}^dag) X^A_{i+1,i+2}
                        # with X^A_{i+1,i+2} = (X^R_{i+2,i+1})^dag.
                        a_ji_dag = a_ji.conj().swapaxes(-2, -1)
                        xa_d2 = prev_xr_dn.conj().swapaxes(-2, -1)
                        xl_id2 = (
                            -xr_ii_a_ij @ prev_xl_up
                            + (xr_ii @ sigma_lesser_ij - xl_ii @ a_ji_dag)
                            @ xa_d2
                        )
                        xl_.blocks[i, i + 2] = xl_id2
                        if xl_.symmetry is None:
                            xl_.blocks[i + 2, i] = -xl_id2.conj().swapaxes(
                                -2, -1
                            )
                        xg_id2 = (
                            -xr_ii_a_ij @ prev_xg_up
                            + (xr_ii @ sigma_greater_ij - xg_ii @ a_ji_dag)
                            @ xa_d2
                        )
                        xg_.blocks[i, i + 2] = xg_id2
                        if xg_.symmetry is None:
                            xg_.blocks[i + 2, i] = -xg_id2.conj().swapaxes(
                                -2, -1
                            )
                    prev_xl_up, prev_xg_up = xl_ij, xg_ij
                    prev_xr_dn = xr_dn_i

                if return_current:
                    a_ji_dagger = a_ji.conj().swapaxes(-2, -1)
                    a_ji_xr_ii = a_ji @ xr_ii
                    a_ji_xr_ii_sx_ij = a_ji_xr_ii @ sigma_lesser_ij
                    sigma_lesser_tilde = (
                        a_ji @ xl_ii @ a_ji_dagger
                        + a_ji_xr_ii_sx_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sx_ij
                    )
                    a_ji_xr_ii_sx_ij = a_ji_xr_ii @ sigma_greater_ij
                    sigma_greater_tilde = (
                        a_ji @ xg_ii @ a_ji_dagger
                        + a_ji_xr_ii_sx_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sx_ij
                    )
                    current[stack_slice, ..., j] = xp.trace(
                        sigma_greater_tilde @ xl_diag_blocks[j]
                        - xg_diag_blocks[j] @ sigma_lesser_tilde,
                        axis1=-2,
                        axis2=-1,
                    )

                xr_diag_blocks[i] = xr_ii + xr_ii_a_ij_xr_jj_a_ji @ xr_ii
                if return_retarded:
                    xr_.blocks[i, i] = xr_diag_blocks[i]

            # The diagonal blocks are overwritten every batch, so the lead
            # currents must be taken inside the loop, against the OBC blocks
            # sliced to the same energies.
            if return_current:
                current[stack_slice, ..., 0] = xp.trace(
                    obc_blocks.greater[0][stack_slice] @ xl_diag_blocks[0]
                    - xg_diag_blocks[0] @ obc_blocks.lesser[0][stack_slice],
                    axis1=-2,
                    axis2=-1,
                )
                # NOTE: Negative sign gives current flowing left -> right.
                current[stack_slice, ..., -1] = -xp.trace(
                    obc_blocks.greater[-1][stack_slice] @ xl_diag_blocks[-1]
                    - xg_diag_blocks[-1] @ obc_blocks.lesser[-1][stack_slice],
                    axis1=-2,
                    axis2=-1,
                )

        if return_current:
            return current
