"""Reference tests for the private hierarchical self-energy study."""

import numpy as np

from studies import _spatial_hierarchy_review as S


def _rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


def test_exact_near_band_and_planted_far_rank_are_recovered():
    rng = np.random.default_rng(2)
    n, d = 12, 2
    near = np.zeros((n*d, n*d), complex)
    for i in range(n):
        for j in range(max(0, i - 1), min(n, i + 2)):
            near[i*d:(i+1)*d, j*d:(j+1)*d] = (
                rng.normal(size=(d, d)) + 1j*rng.normal(size=(d, d)))
    u = rng.normal(size=(n*d, 2)) + 1j*rng.normal(size=(n*d, 2))
    v = rng.normal(size=(n*d, 2)) + 1j*rng.normal(size=(n*d, 2))
    far = u @ v.conj().T
    far = far - S.block_band(far, d, 1)
    a = near + far
    op = S.HODLROperator.from_dense(a, d, half_band=1, leaf_cells=2,
                                    tol=1e-12)
    got = op.to_dense()
    assert _rel(got, a) < 2e-12
    assert np.allclose(S.block_band(got, d, 1), S.block_band(a, d, 1),
                       atol=2e-15, rtol=0.0)


def test_hierarchical_apply_matches_its_dense_realisation():
    a = S.propagating_proxy(16, 3)
    op = S.HODLROperator.from_dense(a, 3, half_band=1, leaf_cells=2,
                                    tol=1e-4, hermitian=True)
    rng = np.random.default_rng(4)
    x = rng.normal(size=a.shape[0]) + 1j*rng.normal(size=a.shape[0])
    assert _rel(op.apply(x), op.to_dense() @ x) < 2e-13


def test_antihermitian_compression_preserves_structure_exactly():
    sigma = -1j * S.propagating_proxy(14, 2)
    op = S.HODLROperator.from_antihermitian(
        sigma, 2, half_band=1, leaf_cells=2, tol=1e-3)
    got = op.antihermitian_dense()
    assert _rel(got.conj().T, -got) < 2e-14
    assert _rel(got, sigma) < 2e-3


def test_sparse_extended_solve_matches_compressed_dense_operator():
    a = 2.0*np.eye(40) + 0.04*S.propagating_proxy(20, 2)
    op = S.HODLROperator.from_dense(a, 2, half_band=1, leaf_cells=2,
                                    tol=1e-4, hermitian=True)
    rng = np.random.default_rng(5)
    rhs = rng.normal(size=40) + 1j*rng.normal(size=40)
    got = op.solve(rhs)
    want = np.linalg.solve(op.to_dense(), rhs)
    assert _rel(got, want) < 2e-11


def test_congruence_control_is_positive_and_reports_its_storage():
    a = S.propagating_proxy(12, 2)
    op = S.HODLROperator.from_dense(a, 2, half_band=1, leaf_cells=2,
                                    tol=1e-3, hermitian=True)
    positive = S.PositiveFactor.from_operator(op)
    got = positive.to_dense()
    assert np.linalg.eigvalsh(got).min() > -1e-12
    assert positive.stored_scalars == positive.factor.size
    assert _rel(positive.apply(np.ones(24)), got @ np.ones(24)) < 2e-14


def test_unitary_phase_extraction_cannot_change_svd_rank():
    a = S.propagating_proxy(18, 3)
    demod = S.phase_demodulate(a, 3, np.array([0.2, 0.7, 1.3]))
    for tol in S.TOLS:
        assert S.quasiseparable_rank(a, 3, tol, band=1) == \
            S.quasiseparable_rank(demod, 3, tol, band=1)
