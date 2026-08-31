# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Pins the batched block-tridiagonal factorisation against dense LU."""
import numpy as np
import pytest

from quatrex.phonon.experimental.pole.btd_linalg import (
    BTDFactorization,
    btd_matvec,
    remap_full_block_snapshot,
)


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _blocks(rng, sizes, stack=()):
    n = len(sizes)
    a_ii = [
        (rng.normal(size=stack + (sizes[i], sizes[i]))
         + 1j * rng.normal(size=stack + (sizes[i], sizes[i]))
         + 4.0 * np.eye(sizes[i]))
        for i in range(n)
    ]
    a_ij = [
        rng.normal(size=stack + (sizes[i], sizes[i + 1]))
        + 1j * rng.normal(size=stack + (sizes[i], sizes[i + 1]))
        for i in range(n - 1)
    ]
    a_ji = [
        rng.normal(size=stack + (sizes[i + 1], sizes[i]))
        + 1j * rng.normal(size=stack + (sizes[i + 1], sizes[i]))
        for i in range(n - 1)
    ]
    return a_ii, a_ij, a_ji


@pytest.mark.parametrize("sizes", [(3, 3, 3), (2, 4, 3, 5)])
@pytest.mark.parametrize("stack", [(), (3,), (2, 3)])
def test_solve_matches_dense(sizes, stack):
    rng = np.random.default_rng(0)
    fac = BTDFactorization.factorize(*_blocks(rng, sizes, stack))
    n = fac.n_dof
    b = rng.normal(size=stack + (n, 2)) + 1j * rng.normal(size=stack + (n, 2))

    x = _h(fac.solve(b))
    ref = np.linalg.solve(_h(fac.to_dense()), b)
    assert np.abs(x - ref).max() / np.abs(ref).max() < 1e-10


@pytest.mark.parametrize("sizes", [(3, 3, 3), (2, 4, 3)])
def test_solve_hermitian_matches_dense(sizes):
    rng = np.random.default_rng(1)
    fac = BTDFactorization.factorize(*_blocks(rng, sizes, (2,)))
    n = fac.n_dof
    b = rng.normal(size=(2, n, 1)) + 1j * rng.normal(size=(2, n, 1))

    x = _h(fac.solve_hermitian(b))
    dense = _h(fac.to_dense())
    ref = np.linalg.solve(dense.conj().swapaxes(-2, -1), b)
    assert np.abs(x - ref).max() / np.abs(ref).max() < 1e-10


def test_matvec_matches_dense():
    rng = np.random.default_rng(2)
    a_ii, a_ij, a_ji = _blocks(rng, (3, 4, 2), (2,))
    fac = BTDFactorization.factorize(a_ii, a_ij, a_ji)
    x = rng.normal(size=(2, fac.n_dof, 3)) + 1j * rng.normal(size=(2, fac.n_dof, 3))
    got = _h(btd_matvec(a_ii, a_ij, a_ji, x))
    ref = _h(fac.to_dense()) @ x
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-12


def test_full_snapshot_remaps_between_block_orderings():
    rng = np.random.default_rng(22)
    nc, d = 8, 2
    dense = rng.normal(size=(3, nc * d, nc * d))
    values = dense.reshape(3, nc, d, nc, d).transpose(
        0, 1, 3, 2, 4).reshape(3, -1)
    rows = np.array([0, 3, 7, 10, 15, 4])
    cols = np.array([2, 3, 1, 15, 0, 9])
    got = remap_full_block_snapshot(values, d, rows, cols)
    np.testing.assert_array_equal(got, dense[:, rows, cols])


def test_snapshot_remap_refuses_a_banded_source():
    with pytest.raises(ValueError, match="full square"):
        remap_full_block_snapshot(np.zeros((4, 17)), 2, [0], [0])


def test_solve_round_trips_through_matvec():
    rng = np.random.default_rng(3)
    fac = BTDFactorization.factorize(*_blocks(rng, (4, 4, 4, 4), (2,)))
    b = rng.normal(size=(2, fac.n_dof, 1)) + 1j * rng.normal(size=(2, fac.n_dof, 1))
    resid = _h(fac.matvec(fac.solve(b))) - b
    assert np.abs(resid).max() / np.abs(b).max() < 1e-10


def test_norm2_matches_dense_svd():
    rng = np.random.default_rng(4)
    fac = BTDFactorization.factorize(*_blocks(rng, (3, 3, 3), (2,)))
    got = _h(fac.norm2(n_power=60))
    ref = np.linalg.svd(_h(fac.to_dense()), compute_uv=False)[..., 0]
    assert np.abs(got - ref).max() / ref.max() < 1e-3


def test_block_count_mismatch_raises():
    rng = np.random.default_rng(5)
    a_ii, a_ij, a_ji = _blocks(rng, (3, 3, 3))
    with pytest.raises(ValueError, match="off-diagonal"):
        BTDFactorization.factorize(a_ii, a_ij[:1], a_ji)


def test_rhs_size_mismatch_raises():
    rng = np.random.default_rng(6)
    fac = BTDFactorization.factorize(*_blocks(rng, (3, 3)))
    with pytest.raises(ValueError, match="rows"):
        fac.solve(np.zeros((5, 1), dtype=complex))


def test_matvec_accepts_blocks_carrying_a_singleton_stack_axis():
    """The production assembly hands blocks a probe axis the vector lacks."""
    b, nb = 3, 3
    rng = np.random.default_rng(0)
    a_ii = [rng.standard_normal((1, b, b)) + 0j for _ in range(nb)]
    a_ij = [rng.standard_normal((1, b, b)) + 0j for _ in range(nb - 1)]
    a_ji = [rng.standard_normal((1, b, b)) + 0j for _ in range(nb - 1)]
    x = rng.standard_normal((nb * b, 1)) + 0j

    n = nb * b
    dense = np.zeros((n, n), dtype=complex)
    off = [i * b for i in range(nb + 1)]
    for i in range(nb):
        dense[off[i]:off[i + 1], off[i]:off[i + 1]] = a_ii[i][0]
    for i in range(nb - 1):
        dense[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = a_ij[i][0]
        dense[off[i + 1]:off[i + 2], off[i]:off[i + 1]] = a_ji[i][0]

    got = _h(btd_matvec(a_ii, a_ij, a_ji, x))
    assert got.shape == (1, n, 1)
    assert np.abs(got[0] - dense @ x).max() < 1e-12


def test_bordered_newton_matvec_carries_the_candidate_axis():
    """The stack axis is the CANDIDATE axis and must survive the matvec."""
    from quatrex.phonon.experimental.pole.pole_nevp import _matvec

    b, nb, npole = 2, 2, 4
    rng = np.random.default_rng(1)
    a_ii = [rng.standard_normal((npole, b, b)) + 0j for _ in range(nb)]
    a_ij = [rng.standard_normal((npole, b, b)) + 0j for _ in range(nb - 1)]
    a_ji = [rng.standard_normal((npole, b, b)) + 0j for _ in range(nb - 1)]
    v = rng.standard_normal((npole, nb * b)) + 0j

    got = _h(_matvec((a_ii, a_ij, a_ji), v))
    assert got.shape == (npole, nb * b)
    for k in range(npole):
        dense = np.zeros((nb * b, nb * b), dtype=complex)
        for i in range(nb):
            dense[i * b:(i + 1) * b, i * b:(i + 1) * b] = _h(a_ii[i])[k]
        for i in range(nb - 1):
            dense[i * b:(i + 1) * b, (i + 1) * b:(i + 2) * b] = _h(a_ij[i])[k]
            dense[(i + 1) * b:(i + 2) * b, i * b:(i + 1) * b] = _h(a_ji[i])[k]
        assert np.abs(got[k] - dense @ _h(v)[k]).max() < 1e-12
