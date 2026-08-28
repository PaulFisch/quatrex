"""Reduced checks for the long-Si-film spatial compression gate."""

import numpy as np

from studies import _si_long_film_spatial_gate as S


def _encode_full_blocks(matrix, n_cells, cell_dof):
    return matrix.reshape(n_cells, cell_dof, n_cells, cell_dof).transpose(
        0, 2, 1, 3).reshape(-1)


def test_full_dsdb_block_order_round_trip():
    n, d = 4, 2
    matrix = np.arange((n * d) ** 2).reshape(n * d, n * d)
    encoded = _encode_full_blocks(matrix, n, d)
    np.testing.assert_array_equal(S.decode_full_blocks(encoded, n, d), matrix)


def test_hodlr_keeps_near_band_and_hermitian_structure():
    rng = np.random.default_rng(42)
    n, d = 8, 2
    factors = rng.standard_normal((n * d, 3)) + 1j * rng.standard_normal(
        (n * d, 3))
    carrier = factors @ factors.conj().T
    near = S.cell_mask(n, d, lambda i, j: abs(i - j) <= 1)
    approximation, levels, storage = S.hodlr_residual(carrier, n, d, 1e-10)
    np.testing.assert_allclose(approximation[near], carrier[near], rtol=0, atol=0)
    np.testing.assert_allclose(approximation, approximation.conj().T,
                               rtol=0, atol=1e-12)
    assert S._rel(approximation, carrier) < 1e-10
    assert 0 in levels
    assert storage["symmetry_aware"] < carrier.size


def test_analysis_uses_quatrex_lesser_sign_and_reports_reblock_baseline():
    rng = np.random.default_rng(7)
    n, d = 8, 1
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    carrier = a @ a.conj().T
    report = S.analyse_matrix(1j * carrier, n, d)
    assert report["antihermiticity_error"] < 1e-12
    assert report["reference_psd_floor"] >= -1e-12
    assert report["storage"]["two_cell_reblock_explicit"] > report[
        "storage"]["hard_band1_explicit"]
    for row in report["hodlr"].values():
        assert row["additional_normalised_negativity"] <= 0.0

