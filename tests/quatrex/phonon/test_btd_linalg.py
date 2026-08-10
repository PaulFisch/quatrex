# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Pins the batched block-tridiagonal factorisation against dense LU."""
import numpy as np
import pytest

from quatrex.phonon.btd_linalg import BTDFactorization, btd_matvec


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
