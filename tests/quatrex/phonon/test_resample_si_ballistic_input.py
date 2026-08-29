import numpy as np

from quatrex.phonon.vertex_q_resample import (
    q_difference_map,
    trigonometric_resample,
)


def test_trigonometric_factor_resampling_is_exact_for_planted_support():
    rng = np.random.default_rng(11)
    n0, n1 = 9, 17
    coeff = np.zeros((2, n0, n0, 3, 4), dtype=complex)
    translations = np.rint(np.fft.fftfreq(n0) * n0).astype(int)
    active = np.abs(translations) <= 2
    coeff[:, active[:, None] & active[None, :]] = (
        rng.normal(size=coeff[:, active[:, None] & active[None, :]].shape)
        + 1j * rng.normal(size=coeff[:, active[:, None] & active[None, :]].shape))
    source = np.fft.fft2(coeff, axes=(1, 2)).reshape(2, n0 * n0, 3, 4)
    got, audit = trigonometric_resample(source, n1)

    q = np.arange(n1) / n1
    phase = np.exp(-2j * np.pi * q[:, None] * translations[None, :])
    expected = np.einsum(
        "ar,orsdk,bs->oabdk", phase, coeff, phase, optimize=True,
    ).reshape(2, n1 * n1, 3, 4)
    np.testing.assert_allclose(got, expected, rtol=2e-14, atol=2e-14)
    assert audit["relative_source_roundtrip"] < 1e-14
    assert audit["relative_fourier_tail"] < 1e-14


def test_q_difference_map_uses_c_order_modular_difference():
    got = q_difference_map(3)
    assert got.shape == (9, 9)
    assert got[0, 1] == 2
    assert got[0, 3] == 6
    assert got[8, 0] == 8
