"""Local finite-cell replacement of the ring's rectangle rule.

The ring integrates the bubble as a rectangle rule on point samples, which is
wrong by ``O(h/gamma)`` in any cell holding an unresolved pole and wrong by a
factor that depends on where in the cell the pole falls. `pole_local` replaces
those cell-pair contributions with closed-form finite-interval integrals of a
local model -- see `phonon/docs/pole_scba_divergence.md` Sec. 10.

Every accuracy test here also asserts that the UNCORRECTED rectangle rule fails
on the same bed. Without that the bed could be one where nothing is wrong and
the correction is trivially right.
"""

import numpy as np
import pytest

from quatrex.phonon import pole_local as L

_GL = np.polynomial.legendre.leggauss(48)


def _quad(f, a, b, n_panels=600):
    """Composite Gauss-Legendre, the reference every closed form is checked on."""
    x, w = _GL
    edges = np.linspace(a, b, n_panels + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    u = (0.5 * (hi - lo) * x[None] + 0.5 * (hi + lo)).ravel()
    wt = (0.5 * (hi - lo) * w[None]).ravel()
    return np.einsum("s,s...->...", wt, f(u))


def _mats(rng, count, n_dof):
    return np.stack([rng.normal(size=(n_dof, n_dof))
                     + 1j * rng.normal(size=(n_dof, n_dof))
                     for _ in range(count)])


def _leg(rng, centre, zeta, bg_order, n_dof=3):
    return L.LocalLeg(_mats(rng, bg_order + 1, n_dof), centre,
                      np.asarray(zeta),
                      _mats(rng, len(zeta), n_dof) if len(zeta)
                      else np.zeros((0, n_dof, n_dof), complex))


def _bed(gamma, n=41, h=0.25, offsets=(0.11, -0.30), seed=11, n_dof=3):
    """A rational spectrum with two narrow poles and two wide ones."""
    rng = np.random.default_rng(seed)
    freqs = np.arange(n) * h
    cells = (13, 27)
    narrow = np.array([freqs[cells[0]] + offsets[0] * h - 1j * gamma * h,
                       freqs[cells[1]] + offsets[1] * h - 1j * gamma * h])
    wide = np.array([freqs[8] - 2.0j * h, freqs[33] - 3.0j * h])
    zeta = np.concatenate([narrow, wide])
    res = _mats(rng, len(zeta), n_dof)
    const = _mats(rng, 1, n_dof)[0]

    def g(s):
        s = np.asarray(s, complex)
        return const[None] + np.einsum(
            "sp,pij->sij", 1.0 / (s[:, None] - zeta[None]), res)

    return freqs, h, g, cells, narrow, res[:2]


def _ring_and_exact(freqs, h, g, n_dof=3):
    """The ring's rectangle sum and the exact integral of the SAME cells."""
    n = len(freqs)
    samples = g(freqs)
    ring = np.zeros((n, n_dof, n_dof), complex)
    exact = np.zeros_like(ring)
    for m in range(n):
        for k in range(n):
            l = m - k
            if not 0 <= l < n:
                continue
            ring[m] += h * (samples[k] @ samples[l]) / (2 * np.pi)
            exact[m] += _quad(
                lambda u: np.einsum("sij,sjk->sik", g(u), g(freqs[m] - u)),
                freqs[k] - h / 2, freqs[k] + h / 2, n_panels=300) / (2 * np.pi)
    return samples, ring, exact


def _rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


# ---------------------------------------------------------------- primitives

@pytest.mark.parametrize("order", [0, 1, 3])
def test_resolvent_moments_match_quadrature(order):
    zeta = np.array([1.05 - 0.004j, 0.8 - 1.7j, 1.2 + 0.02j])
    a, b, c = 0.9, 1.1, 1.0
    got = L.resolvent_moments(zeta, a, b, c, order)
    for j in range(order + 1):
        want = _quad(
            lambda s: (s[:, None] - c) ** j / (s[:, None] - zeta[None]), a, b)
        assert np.abs(got[j] - want).max() < 1e-11 * np.abs(want).max()


def test_cross_polynomial_moments_match_quadrature():
    a, b, c1, c2 = -0.3, 0.7, 0.1, 2.4
    got = L.cross_polynomial_moments(a, b, c1, c2, 3, 3)
    for i in range(4):
        for j in range(4):
            want = _quad(lambda s: (s - c1) ** i * (s - c2) ** j, a, b)
            assert abs(got[i, j] - want) < 1e-11 * max(abs(want), 1e-12)


def test_resolvent_moments_reject_a_real_pole():
    with pytest.raises(ValueError, match="real axis"):
        L.resolvent_moments(np.array([1.0 + 0j]), 0.0, 2.0, 1.0, 0)


@pytest.mark.parametrize("gamma", [0.4, 0.02, 0.001])
@pytest.mark.parametrize("n_k,n_l", [(1, 0), (0, 2), (2, 2)])
@pytest.mark.parametrize("bg_k,bg_l", [(0, 0), (2, 1)])
def test_pair_terms_exact_against_dense_quadrature(gamma, n_k, n_l, bg_k, bg_l):
    """The closed form is the integral, at every width and composition."""
    rng = np.random.default_rng(4)
    h, w_k, omega = 0.4, 1.0, 3.0
    a, b = w_k - h / 2, w_k + h / 2
    w_l = omega - w_k
    z_k = np.array([w_k + 0.1 * h - 1j * gamma * h, w_k - 0.2 * h - 4j * h])[:n_k]
    z_l = np.array([w_l - 0.05 * h - 1j * gamma * h, w_l + 0.3 * h - 3j * h])[:n_l]
    leg_k, leg_l = _leg(rng, w_k, z_k, bg_k), _leg(rng, w_l, z_l, bg_l)
    want = _quad(lambda u: np.einsum("sij,sjk->sik", leg_k.eval(u),
                                     leg_l.eval(omega - u)),
                 a, b, n_panels=3000) / (2 * np.pi)
    got = L.contract(L.pair_terms(leg_k, leg_l, a, b, omega))
    assert _rel(got, want) < 1e-11


def test_pair_correction_is_exact_minus_rectangle():
    rng = np.random.default_rng(6)
    h, w_k, omega = 0.4, 1.0, 3.0
    a, b = w_k - h / 2, w_k + h / 2
    w_l = omega - w_k
    leg_k = _leg(rng, w_k, [w_k - 0.008j], 2)
    leg_l = _leg(rng, w_l, [w_l + 0.3 - 1.1j], 2)
    g_k, g_l = leg_k.eval(np.array([w_k]))[0], leg_l.eval(np.array([w_l]))[0]
    want = (L.contract(L.pair_terms(leg_k, leg_l, a, b, omega))
            - h * (g_k @ g_l) / (2 * np.pi))
    got = L.contract(L.pair_correction(leg_k, leg_l, a, b, omega, g_k, g_l, h))
    assert _rel(got, want) < 1e-12


# ------------------------------------------------------------- reconstruction

def test_deflation_flattens_the_pole_cell():
    """The raw sample spikes at the pole; the deflated one does not."""
    freqs, h, g, cells, zeta, res = _bed(0.02)
    raw = np.abs(g(freqs)).max(axis=(1, 2))
    smooth = np.abs(L.deflate(freqs, g(freqs), zeta, res)).max(axis=(1, 2))
    k = cells[0]
    assert raw[k] > 6 * raw[k - 3]
    assert smooth[k] < 1.5 * smooth[k - 3]


def test_empty_pole_set_leaves_the_ring_untouched():
    freqs, h, g, cells, _, _ = _bed(0.02)
    empty = np.zeros((0,), complex), np.zeros((0, 3, 3), complex)
    delta, report = L.correct_spectrum(freqs, g(freqs), [], *empty)
    assert report["n_corrected"] == 0
    assert np.abs(delta).max() == 0.0


# ------------------------------------------------------------------- accuracy

@pytest.mark.parametrize("gamma", [0.1, 0.02, 0.005, 0.001])
def test_error_stays_bounded_as_the_pole_narrows(gamma):
    """The requirement a sharp-resonance method has to meet.

    The rectangle rule's error grows without bound as ``gamma/h -> 0``; the
    correction's does not. Both halves are asserted, because a bed on which the
    ring did fine would make the second half meaningless.
    """
    freqs, h, g, cells, zeta, res = _bed(gamma)
    samples, ring, exact = _ring_and_exact(freqs, h, g)
    delta, _ = L.correct_spectrum(freqs, samples, cells, zeta, res, rho_min=0.0)
    ring_err, corrected_err = _rel(ring, exact), _rel(ring + delta, exact)
    assert ring_err > 0.5, "the bed does not exercise the failure mode"
    assert corrected_err < 6e-3
    assert corrected_err < ring_err / 50


@pytest.mark.parametrize("offset", [0.0, 0.25, 0.49])
def test_registration_lottery_is_removed(offset):
    """Moving the pole inside its cell swings the ring, not the correction."""
    freqs, h, g, cells, zeta, res = _bed(0.02, offsets=(offset, -offset))
    samples, ring, exact = _ring_and_exact(freqs, h, g)
    delta, _ = L.correct_spectrum(freqs, samples, cells, zeta, res, rho_min=0.0)
    assert _rel(ring, exact) > 0.3
    assert _rel(ring + delta, exact) < 1e-3


def test_widening_the_corrected_set_reaches_the_poles_tail():
    """Radius 0 leaves the tail to the rectangle rule; radius 1 does not."""
    freqs, h, g, cells, zeta, res = _bed(0.02)
    samples, ring, exact = _ring_and_exact(freqs, h, g)
    errs = []
    for radius in (0, 1, 2):
        delta, _ = L.correct_spectrum(freqs, samples, cells, zeta, res,
                                      radius=radius, rho_min=0.0)
        errs.append(_rel(ring + delta, exact))
    assert errs[0] > 10 * errs[1], "the tail correction is not doing anything"
    assert errs[1] < 3 * errs[2], "radius 1 should be close to converged"


def test_edge_cells_degrade_the_stencil_rather_than_go_uncorrected():
    freqs, h, g, cells, zeta, res = _bed(0.02)
    samples, ring, exact = _ring_and_exact(freqs, h, g)
    high, _ = L.correct_spectrum(freqs, samples, cells, zeta, res,
                                 poly_order=4, rho_min=0.0)
    low, report = L.correct_spectrum(freqs, samples, cells, zeta, res,
                                     poly_order=0, rho_min=0.0)
    assert report["n_degraded"] == 0
    assert _rel(ring + high, exact) < 2 * _rel(ring + low, exact)


# ----------------------------------------------------------------- the gate

def test_rho_out_ignores_pairings_across_the_half_planes():
    """A pole and its conjugate sum to a real number and make no output pole."""
    z = np.array([3.0 - 0.01j, 3.0 + 0.01j])
    rho, _ = L.output_resolution(z[:, None], z[None, :], 0.5)
    assert np.isinf(rho[0, 1]) and np.isinf(rho[1, 0])
    assert np.allclose(np.diag(rho), 2 * 0.02 / 0.5)


def test_rho_gate_refuses_pole_pole_pairs_but_keeps_pole_background():
    freqs, h, g, cells, zeta, res = _bed(0.02)
    samples = g(freqs)
    _, loose = L.correct_spectrum(freqs, samples, cells, zeta, res, rho_min=0.0)
    _, tight = L.correct_spectrum(freqs, samples, cells, zeta, res, rho_min=1.0)
    assert loose["n_refused_rho"] == 0
    assert tight["n_refused_rho"] > 0
    # the first-order pole-background corrections survive the gate
    assert tight["n_corrected"] > 0.5 * loose["n_corrected"]


def test_non_uniform_grid_is_refused_not_approximated():
    freqs = np.array([0.0, 0.1, 0.3, 0.6])
    g = np.zeros((4, 2, 2), complex)
    with pytest.raises(ValueError, match="uniform"):
        L.correct_spectrum(freqs, g, [1], np.array([0.1 - 0.01j]),
                           np.zeros((1, 2, 2), complex))
