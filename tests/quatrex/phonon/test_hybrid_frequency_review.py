"""Reference tests for the private hybrid grid--rational study."""

import numpy as np
from scipy.integrate import quad_vec

from studies import _hybrid_frequency_review as H


def _rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


def test_passive_cluster_is_psd_and_partial_fractions_are_exact():
    cluster = H.proxy_cluster(0.04, 0.49, 0.5, 0.02)
    omega = np.linspace(-3.0, 4.0, 81)
    direct = cluster.spectrum(omega)
    rational = cluster.partial_fractions().eval(omega)
    assert _rel(rational, direct) < 2e-13
    assert min(np.linalg.eigvalsh(a).min() for a in direct) > -1e-12
    lesser = cluster.keldysh(omega)
    assert _rel(lesser.conj().transpose(0, 2, 1), -lesser) < 2e-13
    assert _rel(1j * lesser, direct) < 2e-13


def test_rational_cluster_bubble_matches_independent_real_axis_quadrature():
    cluster = H.proxy_cluster(0.4, 0.25, 1.0, 0.02)
    rational = H.cluster_bubble(cluster)
    omega = 2.15
    centres = sorted(cluster.z.real.tolist()
                     + (omega - cluster.z.real).tolist())
    edges = [-np.inf] + centres + [np.inf]
    want = np.zeros((cluster.n_dof, cluster.n_dof), complex)

    def f(u):
        return cluster.spectrum(u) @ cluster.spectrum(omega - u) / H.TWO_PI

    for lo, hi in zip(edges[:-1], edges[1:]):
        part, _ = quad_vec(f, lo, hi, epsabs=2e-11, epsrel=2e-11, limit=800)
        want += part
    assert _rel(rational.eval(omega), want) < 2e-10
    assert np.all(np.imag(rational.poles[:4]) < 0.0)
    assert np.all(np.imag(rational.poles[4:]) > 0.0)


def test_rational_cell_moments_match_quadrature():
    cluster = H.proxy_cluster(0.02, 0.25, 2.0, 0.0)
    rational = cluster.partial_fractions()
    centres = np.array([-1.0, 0.0, 1.25])
    h = 0.25
    m0, m1 = rational.cell_moments(centres, h)
    x, w = np.polynomial.legendre.leggauss(96)
    for k, centre in enumerate(centres):
        u = centre + 0.5 * h * x
        a = rational.eval(u)
        want0 = np.einsum("s,sij->ij", 0.5 * h * w, a)
        want1 = np.einsum("s,sij->ij", 0.5 * h * w * (u - centre), a)
        assert _rel(m0[k], want0) < 2e-11
        assert _rel(m1[k], want1) < 2e-10


def test_fft_toeplitz_mixed_term_is_the_exact_linear_cell_integral():
    cluster = H.proxy_cluster(0.08, 0.25, 0.5, 0.0)
    grid = np.arange(-4, 5) * 0.25
    rng = np.random.default_rng(12)
    smooth = rng.normal(size=(grid.size, 2, 2)) + 1j * rng.normal(
        size=(grid.size, 2, 2))
    derivative = 0.2 * (rng.normal(size=smooth.shape)
                        + 1j * rng.normal(size=smooth.shape))
    got = H.mixed_product_integration(cluster, grid, smooth, derivative)
    x, w = np.polynomial.legendre.leggauss(120)
    h = grid[1] - grid[0]
    want = np.zeros_like(got)
    for m, omega in enumerate(grid):
        for j, centre in enumerate(grid):
            t = 0.5 * h * x
            b = smooth[j] + t[:, None, None] * derivative[j]
            a = cluster.spectrum(omega - centre - t)
            term = np.einsum("sik,skj->sij", a, b)
            term += np.einsum("sik,skj->sij", b, a)
            want[m] += np.einsum("s,sij->ij", 0.5 * h * w, term) / H.TWO_PI
    assert _rel(got, want) < 2e-10


def test_output_line_is_carried_rationally_when_it_is_subgrid():
    cluster = H.proxy_cluster(0.001, 0.49, 0.5, 0.0)
    out = H.cluster_bubble(cluster)
    lower = out.poles[np.imag(out.poles) < 0.0]
    assert lower.size > 0
    assert np.max(-np.imag(lower)) < 0.001 * 0.25 * 2.4
    assert out.auxiliary_dimension > 0
