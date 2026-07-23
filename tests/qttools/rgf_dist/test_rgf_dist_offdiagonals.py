# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
"""Parity of the DISTRIBUTED RGF off-diagonal X^{<,>} blocks vs dense.

The distributed selected solve can emit the 2nd and 3rd off-diagonal
X^{<,>} blocks (``n_offdiagonals``) via the left-connected chain +
uniform post-pass + boundary halo. Unlike ``test_rgf_dist`` (which masks
the dense reference to the solver's nonzeros, hiding unwritten blocks),
this test asserts every block out to the requested band EXPLICITLY
against the unmasked dense congruence solution -- an unwritten block
fails loudly.

Run with:  mpirun -np 2 pytest ... (two boundary partitions)
           mpirun -np 3 pytest ... (adds an interior/arrowhead partition,
                                    exercising pipeline + halo mid-chain)
The block communicator spans ALL ranks (block_comm_size = world size).
"""
import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBCOO
from qttools.greens_function_solver.rgf_dist import RGFDist

# Per-rank block layouts. Every rank owns >= 4 blocks (the k=3
# constraint) and the minimum block size is 5 so the 11-wide scalar band
# used to seed A stays inside the block-tridiagonal envelope.
BLOCK_SIZES = [
    pytest.param(np.array([6] * 4), id="constant-block-size"),
    pytest.param(np.array([5, 6, 7, 5]), id="mixed-block-size"),
]
GLOBAL_STACK_SHAPES = [
    pytest.param((4,), id="1D-stack"),
    pytest.param((5, 2), id="2D-stack"),
]


def setup_module():
    """Block communicator over ALL ranks (block_comm_size = world size)."""
    if xp.__name__ == "cupy":
        _default_config = {k: "host_mpi" for k in
                           ("all_to_all", "all_gather", "all_reduce", "bcast")}
    else:
        _default_config = {k: "device_mpi" for k in
                           ("all_to_all", "all_gather", "all_reduce", "bcast")}
    comm.configure(
        block_comm_size=global_comm.size,
        block_comm_config=_default_config,
        stack_comm_config=_default_config,
        override=True,
    )


def _random_banded(shape, skew_hermitian=False):
    """Random 11-diagonal sparray, optionally skew-hermitianised."""
    m = sparse.diags(xp.random.rand(11), xp.arange(-5, 6), shape=shape).tocsr()
    m.data = xp.random.rand(len(m.data))
    m += sparse.diags(xp.random.rand(11) * 1j, xp.arange(-5, 6), shape=shape).tocsr()
    if skew_hermitian:
        m = m - m.conj().T
    else:
        m += sparse.diags([20 * (1 + 1j) * xp.random.rand() * 1j], [0], shape=shape).tocsr()
    return m


def _block_band_pattern(block_sizes: NDArray, band: int, dtype):
    """Dense-ones structure on every block (bi, bj) with |bi-bj| <= band."""
    offs = np.hstack(([0], np.cumsum(block_sizes)))
    n = int(block_sizes.sum())
    dense = xp.zeros((n, n), dtype=dtype)
    nb = len(block_sizes)
    for bi in range(nb):
        for bj in range(nb):
            if abs(bi - bj) <= band:
                dense[offs[bi]:offs[bi + 1], offs[bj]:offs[bj + 1]] = 1.0
    return sparse.csr_matrix(dense)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("block_sizes", BLOCK_SIZES)
@pytest.mark.parametrize("global_stack_shape", GLOBAL_STACK_SHAPES)
@pytest.mark.parametrize("n_offdiagonals", [2, 3])
@pytest.mark.parametrize("max_batch_size", [100, 1])
def test_rgf_dist_offdiagonals(
    block_sizes: NDArray,
    global_stack_shape: tuple,
    n_offdiagonals: int,
    max_batch_size: int,
):
    """Distributed 2nd/3rd off-diagonal X^{<,>} match the dense solution."""
    global_block_sizes = np.tile(block_sizes, comm.block.size)
    n = int(global_block_sizes.sum())
    shape = (n, n)

    a_sparray = sl_sparray = sg_sparray = None
    if global_comm.rank == 0:
        a_sparray = _random_banded(shape)
        sl_sparray = _random_banded(shape, skew_hermitian=True)
        sg_sparray = _random_banded(shape, skew_hermitian=True)
    a_sparray = global_comm.bcast(a_sparray, root=0)
    sl_sparray = global_comm.bcast(sl_sparray, root=0)
    sg_sparray = global_comm.bcast(sg_sparray, root=0)

    def _make(sparray):
        coo = sparray.tocoo()
        coo.sum_duplicates()
        coo.eliminate_zeros()
        return DSDBCOO.from_sparray(
            sparray=coo,
            block_sizes=global_block_sizes,
            global_stack_shape=global_stack_shape,
        )

    a_dsdb = _make(a_sparray)
    sl_dsdb = _make(sl_sparray)
    sg_dsdb = _make(sg_sparray)

    # Output pattern: full block band out to n_offdiagonals, so every
    # off-diagonal write (incl. the boundary-crossing blocks) lands.
    out_sparray = _block_band_pattern(
        global_block_sizes, n_offdiagonals, a_sparray.dtype
    )
    outs = [
        DSDBCOO.from_sparray(
            sparray=out_sparray,
            block_sizes=global_block_sizes,
            global_stack_shape=global_stack_shape,
        )
        for _ in range(3)
    ]
    xl_out, xg_out, xr_out = outs
    for m in outs:
        m.data[:] = 0.0

    RGFDist(max_batch_size=max_batch_size).selected_solve(
        a=a_dsdb,
        sigma_lesser=sl_dsdb,
        sigma_greater=sg_dsdb,
        out=(xl_out, xg_out, xr_out),
        return_retarded=True,
        n_offdiagonals=n_offdiagonals,
    )

    Xl_rgf = xl_out.to_dense()
    Xg_rgf = xg_out.to_dense()

    Xr_ref = xp.linalg.inv(a_dsdb.to_dense())
    Xl_ref = Xr_ref @ sl_dsdb.to_dense() @ Xr_ref.conj().swapaxes(-2, -1)
    Xg_ref = Xr_ref @ sg_dsdb.to_dense() @ Xr_ref.conj().swapaxes(-2, -1)

    # Assert EVERY block out to the requested band, explicitly and
    # unmasked -- an unwritten block shows up as zeros vs the reference.
    offs = np.hstack(([0], np.cumsum(global_block_sizes)))
    nb = len(global_block_sizes)
    for bi in range(nb):
        for bj in range(nb):
            if abs(bi - bj) > n_offdiagonals:
                continue
            sl = (Ellipsis, slice(offs[bi], offs[bi + 1]),
                  slice(offs[bj], offs[bj + 1]))
            assert xp.allclose(Xl_rgf[sl], Xl_ref[sl], atol=1e-10), (
                f"X^< block ({bi},{bj}) wrong "
                f"(|d|={abs(bi - bj)}, k={n_offdiagonals})"
            )
            assert xp.allclose(Xg_rgf[sl], Xg_ref[sl], atol=1e-10), (
                f"X^> block ({bi},{bj}) wrong "
                f"(|d|={abs(bi - bj)}, k={n_offdiagonals})"
            )


@pytest.mark.mpi(min_size=2)
def test_rgf_dist_offdiagonals_validation():
    """Bad n_offdiagonals / missing retarded / thin partitions raise."""
    block_sizes = np.array([5] * 4)
    global_block_sizes = np.tile(block_sizes, comm.block.size)
    n = int(global_block_sizes.sum())
    a_sparray = None
    if global_comm.rank == 0:
        a_sparray = _random_banded((n, n))
    a_sparray = global_comm.bcast(a_sparray, root=0)
    coo = a_sparray.tocoo()
    coo.sum_duplicates()
    mk = lambda: DSDBCOO.from_sparray(
        sparray=coo, block_sizes=global_block_sizes, global_stack_shape=(2,)
    )
    a, sl, sg = mk(), mk(), mk()
    xl, xg, xr = mk(), mk(), mk()
    solver = RGFDist()
    with pytest.raises(ValueError, match="must be 1, 2, or 3"):
        solver.selected_solve(a, sl, sg, out=(xl, xg, xr),
                              return_retarded=True, n_offdiagonals=4)
    with pytest.raises(ValueError, match="return_retarded"):
        solver.selected_solve(a, sl, sg, out=(xl, xg),
                              return_retarded=False, n_offdiagonals=2)
