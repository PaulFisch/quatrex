import numpy as np

from quatrex.phonon.ballistic_audit import (
    caroli_number_current,
    caroli_transmission,
    spectrum_error,
)


def test_caroli_scalar_and_batched_matrix_oracles():
    frequency = 4
    qpoint = 3
    rng = np.random.default_rng(7)
    g = rng.normal(size=(frequency, qpoint, 2, 2))
    g = g + 1j * rng.normal(size=g.shape)
    gamma_l = np.broadcast_to(np.diag([0.7, 0.2]), g.shape)
    gamma_r = np.broadcast_to(np.diag([0.1, 0.9]), g.shape)
    sigma_l = -0.5j * gamma_l
    sigma_r = -0.5j * gamma_r

    got = caroli_transmission(g, sigma_l, sigma_r)
    expected = np.empty((frequency, qpoint))
    for iw in range(frequency):
        for iq in range(qpoint):
            expected[iw, iq] = np.trace(
                gamma_l[iw, iq] @ g[iw, iq]
                @ gamma_r[iw, iq] @ g[iw, iq].conj().T
            ).real
    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=1e-14)

    nl = np.array([3.0, 2.0, 1.0, 0.5])
    nr = np.array([2.5, 1.8, 0.9, 0.4])
    np.testing.assert_allclose(
        caroli_number_current(got, nl, nr),
        expected * (nl - nr)[:, None],
    )


def test_spectrum_error_ignores_only_numerically_inactive_bins():
    reference = np.array([0.0, 1.0, 2.0, 1e-15])
    value = np.array([1e-16, 1.0, 2.0 * (1 + 1e-6), 1e-13])
    error = spectrum_error(reference, value)
    assert error["relative_l2"] < 1e-6
    assert np.isclose(error["active_max_relative"], 1e-6)
