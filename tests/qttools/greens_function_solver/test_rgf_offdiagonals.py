# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
"""Parity of the RGF off-diagonal G^{<,>} blocks against the dense solution.

The single-node RGF ``selected_solve`` can additionally produce the second
(``n_offdiagonals=2``) and third (``n_offdiagonals=3``) off-diagonal blocks
X^{<,>}_{i,i+2} / X^{<,>}_{i,i+3} on top of the block-tridiagonal band. These
feed the ``sse_g_band`` bubble contraction, so they must equal the exact dense
congruence solution X = A^{-1} Sigma A^{-dagger} on every requested block.

This closes a coverage gap: the default path had a parity test
(``test_selected_solve``) but the off-diagonal extension did not.
"""
import numpy as np
import pytest

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBCOO, DSDBCSR
from qttools.greens_function_solver import RGF


@pytest.fixture(autouse=True, scope="module")
def configure_comm():
    """Single block rank (the off-diagonal path requires block_comm_size=1)."""
    if xp.__name__ == "cupy":
        _cfg = {k: "host_mpi" for k in
                ("all_to_all", "all_gather", "all_reduce", "bcast")}
    else:
        _cfg = {k: "device_mpi" for k in
                ("all_to_all", "all_gather", "all_reduce", "bcast")}
    comm.configure(block_comm_size=1, block_comm_config=_cfg,
                   stack_comm_config=_cfg, override=True)


# >= 4 blocks so the third off-diagonal (i, i+3) exists; both a constant and a
# mixed block-size layout so the recursion is exercised on non-square coupling
# blocks too.
BLOCK_SIZES = [
    pytest.param(np.array([3] * 6), id="constant-block-size"),
    pytest.param(np.array([2, 4, 3, 2, 4]), id="mixed-block-size"),
]


def _random_block(m: int, n: int) -> NDArray:
    coo = sparse.random(int(m), int(n), density=0.6,
                        format="coo").astype(xp.complex128)
    coo.data += 1j * xp.random.uniform(size=coo.nnz)
    return coo.toarray()


def _bt_dense(block_sizes: NDArray) -> NDArray:
    """Random, diagonally dominant block-tridiagonal matrix."""
    off = np.hstack(([0], np.cumsum(block_sizes)))
    nb = len(block_sizes)
    size = int(np.sum(block_sizes))
    arr = xp.zeros((size, size), dtype=xp.complex128)
    for i in range(nb):
        arr[off[i]:off[i + 1], off[i]:off[i + 1]] = (
            _random_block(block_sizes[i], block_sizes[i])
            + xp.identity(int(block_sizes[i]), dtype=xp.complex128))
        if i > 0:
            arr[off[i]:off[i + 1], off[i - 1]:off[i]] = _random_block(
                block_sizes[i], block_sizes[i - 1])
            arr[off[i - 1]:off[i], off[i]:off[i + 1]] = _random_block(
                block_sizes[i - 1], block_sizes[i])
    for i in range(size):
        arr[i, i] = (1 + 1j) + complex(xp.sum(arr[i, :]))
    return arr


def _band_pattern(block_sizes: NDArray, band: int) -> sparse.coo_matrix:
    """Dense-ones structure on every block (bi, bj) with |bi - bj| <= band."""
    off = np.hstack(([0], np.cumsum(block_sizes)))
    nb = len(block_sizes)
    size = int(np.sum(block_sizes))
    struct = xp.zeros((size, size), dtype=xp.complex128)
    for bi in range(nb):
        for bj in range(nb):
            if abs(bi - bj) <= band:
                struct[off[bi]:off[bi + 1], off[bj]:off[bj + 1]] = 1.0
    return sparse.coo_matrix(struct)


def _band_mask(block_sizes: NDArray, band: int) -> NDArray:
    off = np.hstack(([0], np.cumsum(block_sizes)))
    nb = len(block_sizes)
    size = int(np.sum(block_sizes))
    mask = xp.zeros((size, size), dtype=bool)
    for bi in range(nb):
        for bj in range(nb):
            if abs(bi - bj) <= band:
                mask[off[bi]:off[bi + 1], off[bj]:off[bj + 1]] = True
    return mask


@pytest.mark.parametrize("block_sizes", BLOCK_SIZES)
@pytest.mark.parametrize("n_offdiagonals", [1, 2, 3])
@pytest.mark.parametrize("dsdbsparse_type", [DSDBCOO, DSDBCSR])
@pytest.mark.parametrize("max_batch_size", [1, 100])
def test_rgf_offdiagonals(block_sizes, n_offdiagonals, dsdbsparse_type,
                          max_batch_size):
    """RGF X^{<,>} match the dense A^{-1} Sigma A^{-dagger} out to band k."""
    global_stack_shape = (2,)
    bt = _bt_dense(block_sizes)

    coo_A = sparse.coo_matrix(bt)
    coo_Bl = sparse.coo_matrix(bt)
    coo_Bl += -coo_Bl.conj().T                      # skew-hermitian
    coo_Bg = sparse.coo_matrix(bt)
    coo_Bg += -coo_Bg.conj().T

    # Dense reference (full off-diagonals): X = A^{-1} Sigma A^{-dagger}.
    ref_Xr = xp.linalg.inv(bt)
    ref_Xl = ref_Xr @ xp.asarray(coo_Bl.toarray()) @ ref_Xr.conj().T
    ref_Xg = ref_Xr @ xp.asarray(coo_Bg.toarray()) @ ref_Xr.conj().T

    A = dsdbsparse_type.from_sparray(
        sparray=coo_A, block_sizes=block_sizes,
        global_stack_shape=global_stack_shape)
    Bl = dsdbsparse_type.from_sparray(
        sparray=coo_Bl, block_sizes=block_sizes,
        global_stack_shape=global_stack_shape)
    Bg = dsdbsparse_type.from_sparray(
        sparray=coo_Bg, block_sizes=block_sizes,
        global_stack_shape=global_stack_shape)

    # Output buffers carry the band-k block pattern so the off-diagonal
    # writes land (writes to absent blocks drop silently), exactly as the
    # production sparsity extension (scba: |bi-bj| <= sse_g_band) provides.
    band_struct = _band_pattern(block_sizes, n_offdiagonals)
    Xl = dsdbsparse_type.from_sparray(
        sparray=band_struct, block_sizes=block_sizes,
        global_stack_shape=global_stack_shape)
    Xg = dsdbsparse_type.from_sparray(
        sparray=band_struct, block_sizes=block_sizes,
        global_stack_shape=global_stack_shape)
    Xl.data = 0.0
    Xg.data = 0.0

    RGF(max_batch_size=max_batch_size).selected_solve(
        A, Bl, Bg, out=[Xl, Xg], n_offdiagonals=n_offdiagonals)

    mask = _band_mask(block_sizes, n_offdiagonals)
    ref_l = xp.broadcast_to(ref_Xl, (*global_stack_shape, *ref_Xl.shape))
    ref_g = xp.broadcast_to(ref_Xg, (*global_stack_shape, *ref_Xg.shape))
    assert xp.allclose(Xl.to_dense() * mask, ref_l * mask,
                       atol=1e-9, rtol=1e-7), (
        f"lesser off-diagonals wrong at n_offdiagonals={n_offdiagonals}")
    assert xp.allclose(Xg.to_dense() * mask, ref_g * mask,
                       atol=1e-9, rtol=1e-7), (
        f"greater off-diagonals wrong at n_offdiagonals={n_offdiagonals}")


def test_rgf_rejects_unsupported_band():
    """n_offdiagonals outside {1,2,3} is a clear error, not silent garbage."""
    block_sizes = np.array([3] * 5)
    bt = _bt_dense(block_sizes)
    coo = sparse.coo_matrix(bt)
    coo_B = coo + (-coo.conj().T)
    A = DSDBCOO.from_sparray(sparray=coo, block_sizes=block_sizes,
                             global_stack_shape=(1,))
    Bl = DSDBCOO.from_sparray(sparray=coo_B, block_sizes=block_sizes,
                              global_stack_shape=(1,))
    Bg = DSDBCOO.from_sparray(sparray=coo_B, block_sizes=block_sizes,
                              global_stack_shape=(1,))
    Xl = DSDBCOO.empty_like(Bl)
    Xl.allocate_data()
    Xl.data = 0.0
    Xg = DSDBCOO.empty_like(Bl)
    Xg.allocate_data()
    Xg.data = 0.0
    with pytest.raises(ValueError):
        RGF(max_batch_size=100).selected_solve(
            A, Bl, Bg, out=[Xl, Xg], n_offdiagonals=4)
