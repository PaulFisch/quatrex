import numpy as np

from phonon.postproc.cnt_ladder_physics import (
    apparent_mfp,
    lead_spectrum,
    linewidth_sector_matrix,
    modal_bubble_properties,
    spectral_quantiles,
)


def test_lead_spectrum_uses_contact_magnitudes():
    current = np.array([[2.0, 99.0, -4.0], [6.0, 0.0, -2.0]])
    np.testing.assert_allclose(lead_spectrum(current), [3.0, 4.0])


def test_spectral_quantiles_and_apparent_mfp():
    quantiles = spectral_quantiles(
        np.arange(4.0), np.ones(4), np.ones(4), (0.5, 0.75))
    np.testing.assert_allclose(quantiles, [1.0, 2.0])
    np.testing.assert_allclose(apparent_mfp([4.0, 8.0], [0.5, 0.8]),
                               [4.0, 32.0])


def test_linewidth_sector_matrix_normalizes_receiving_rows():
    coupling = np.array([[1.0, 1.0, 2.0],
                         [2.0, 2.0, 4.0],
                         [3.0, 3.0, 6.0]])
    sectors = linewidth_sector_matrix(coupling, [1.0, 2.0, 3.0],
                                      [0.0, 2.5, 4.0])
    np.testing.assert_allclose(sectors.sum(axis=1), 1.0)
    np.testing.assert_allclose(sectors[0], [0.5, 0.5])


def test_modal_bubble_properties_for_diagonal_mode():
    freqs = np.array([1.0, 2.0, 3.0])
    dynamical = np.diag([4.0, 9.0])
    sigma = np.zeros((3, 2, 2), dtype=complex)
    sigma[:, 0, 0] = 0.8 - 0.4j
    sigma[:, 1, 1] = 1.2 - 1.2j
    omega, shift, width = modal_bubble_properties(freqs, dynamical, sigma)
    np.testing.assert_allclose(omega, [2.0, 3.0])
    np.testing.assert_allclose(shift, [0.2, 0.2])
    np.testing.assert_allclose(width, [0.1, 0.2])
