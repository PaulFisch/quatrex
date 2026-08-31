"""Reference tests for the conserving SSS tail study."""

import numpy as np

from studies import _conserving_spatial_tail_review as S
from studies._spatial_hierarchy_review import block_band


def test_sparse_extended_solve_matches_exact_sss_schur_complement():
    rng = np.random.default_rng(1)
    n, d = 14, 2
    scalar = S._random_semisep(n, 2, 3).to_dense()
    far = np.kron(scalar, np.array([[1.0, 0.2], [-0.1j, 0.7]]))
    a = 5.0*np.eye(n*d) + far - block_band(far, d, 1)
    op = S.BandSemiSepOperator.from_dense(
        a, d, half_band=1, tol=1e-13)
    rhs = rng.normal(size=n*d) + 1j*rng.normal(size=n*d)
    got = op.solve(rhs)
    want = np.linalg.solve(op.to_dense(), rhs)
    assert S.relative_error(got, want) < 2e-12
    assert S.relative_error(op.to_dense(), a) < 2e-12


def test_retained_near_band_is_exact_under_tail_truncation():
    rng = np.random.default_rng(2)
    n, d = 16, 3
    a = rng.normal(size=(n*d, n*d)) + 1j*rng.normal(size=(n*d, n*d))
    op = S.BandSemiSepOperator.from_dense(a, d, half_band=2, tol=0.2)
    assert np.allclose(block_band(op.to_dense(), d, 2),
                       block_band(a, d, 2), atol=2e-14, rtol=0.0)
    x = rng.normal(size=n*d) + 1j*rng.normal(size=n*d)
    assert S.relative_error(op.apply(x), op.to_dense() @ x) < 2e-13


def test_antihermitian_carrier_shares_both_directions_exactly():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(30, 7)) + 1j*rng.normal(size=(30, 7))
    sigma = -1j * (z @ z.conj().T)
    op = S.BandSemiSepOperator.from_antihermitian(
        sigma, 2, half_band=1, tol=1e-3)
    got = -1j * op.to_dense()
    assert S.relative_error(got.conj().T, -got) < 2e-14
    assert np.allclose(block_band(got, 2, 1), block_band(sigma, 2, 1),
                       atol=2e-12, rtol=0.0)


def test_scalar_bubble_tail_has_exact_kronecker_closure():
    a = S._random_semisep(17, 2, 8)
    b = S._random_semisep(17, 3, 9)
    got = S.scalar_hadamard_product(a, b)
    assert got.rank == (6, 6)
    assert S.relative_error(got.to_dense(),
                            a.to_dense() * b.to_dense()) < 5e-14


def test_collision_covector_costs_at_most_one_state_and_is_exact():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(80, 25)) + 1j*rng.normal(size=(80, 25))
    c = rng.normal(size=80) + 1j*rng.normal(size=80)
    got, basis = S.moment_preserving_projection(x, 7, c)
    assert basis.shape[1] <= 8
    defect = np.linalg.norm(c.conj() @ (got - x))
    scale = max(np.linalg.norm(c.conj() @ x), 1e-300)
    assert defect / scale < 2e-14


def test_factor_compression_preserves_positivity_by_congruence():
    rng = np.random.default_rng(11)
    z = rng.normal(size=(50, 20)) + 1j*rng.normal(size=(50, 20))
    zr, carrier = S.positive_factor_projection(z, 6)
    assert zr.shape == (50, 6)
    assert np.linalg.eigvalsh(carrier).min() > -2e-12
    assert np.allclose(carrier, zr @ zr.conj().T)


def test_direct_analytic_generators_avoid_dense_tail_extraction_penalty():
    rng = np.random.default_rng(12)
    n, d = 18, 5
    tail = S._random_block_semisep(n, d, 3, 13)
    td = tail.to_dense()
    target_near = 6.0*np.eye(n*d, dtype=complex)
    base = target_near - block_band(td, d, 1)
    op = S.BandSemiSepOperator(S.sparse.csr_matrix(base), tail, 1)
    got = op.to_dense()
    assert np.allclose(block_band(got, d, 1), target_near,
                       atol=2e-14, rtol=0.0)
    rhs = rng.normal(size=n*d) + 1j*rng.normal(size=n*d)
    assert S.relative_error(op.solve(rhs), np.linalg.solve(got, rhs)) < 2e-12
