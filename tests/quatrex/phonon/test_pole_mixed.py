# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""The mixed pole-background convolution: three routes, measured.

Background with an exact closed form, so the comparison is against the true
answer rather than against another approximation: for
``R(u) = sum_j A_j/((u-c_j)^2+d_j^2)``,

    (1/2pi) int R(u)/(z-u) du  =  -i * sum over R's LOWER-half-plane poles,

because closing the contour downward leaves only those (the pole at ``u = z``
sits in the upper half plane, since ``Im z = gamma > 0``).
"""
import numpy as np
import pytest

from quatrex.phonon.pole_mixed import mixed_convolution

BG = ((1.0, 6.0, 3.0), (0.6, 14.0, 5.0))
PROBES = np.array([4.0, 9.0, 15.0])


def _r(u):
    return sum(a / ((u - c) ** 2 + d**2) for a, c, d in BG)


def _exact(omega, p, c=1.0 + 0j):
    z = omega - p
    return c * sum(-1j * (a / (-2j * d)) / (z - (c0 - 1j * d))
                   for a, c0, d in BG)


def _bed(nf=801, width=40.0):
    w = np.linspace(-width, width, nf)
    return w, _r(w)


def _err(method, gamma, **kw):
    w, r = _bed()
    p = 10.0 - 1j * gamma
    ref = np.array([_exact(o, p) for o in PROBES])
    got = mixed_convolution(PROBES, p, 1.0 + 0j, r, w, method=method, **kw)
    return float(np.abs(got - ref).max() / np.abs(ref).max())


def test_grid_route_fails_below_the_grid_spacing():
    """Sampling the pole on the grid collapses exactly where the sector is needed.

    Above ``gamma ~ h`` the discrete convolution is fine; below it the error
    grows without bound, because the narrow factor is never sampled near its
    peak. This is the same registration failure the pole-pole channel removes,
    and it is why the mixed terms cannot simply be left on the grid.
    """
    coarse = _err("grid", 2.0)
    assert coarse < 1e-2, f"grid should be fine at gamma/h = 20 ({coarse:.2e})"
    for gamma, floor in ((0.02, 0.1), (0.004, 1.0), (0.0008, 10.0)):
        e = _err("grid", gamma)
        assert e > floor, f"grid at gamma/h={gamma / 0.1:.3f} gave only {e:.2e}"


def test_cell_route_is_stable_at_every_linewidth():
    """Integrating the pole over each cell never samples the narrow denominator."""
    for gamma in (2.0, 0.1, 0.004, 0.0008):
        assert _err("cells", gamma) < 1e-2


def test_moment_route_is_flat_in_the_linewidth():
    """Exact in the pole and order-p in the background: no gamma dependence left."""
    errs = [_err("moments", g, order=5, window=10)
            for g in (2.0, 0.5, 0.1, 0.02, 0.004, 0.0008)]
    assert max(errs) < 3e-3, errs
    # The last four rungs span a factor 125 in gamma and must not drift.
    tail = errs[-4:]
    assert max(tail) / min(tail) < 1.2, f"moment error drifts with gamma: {tail}"


def test_moment_route_beats_the_cell_route_when_the_pole_is_narrow():
    """The cell model of R is what limits 'cells'; the moment model removes it."""
    g = 0.0008
    assert _err("moments", g, order=5, window=10) < 0.5 * _err("cells", g)


def test_rational_route_is_refused_not_silently_downgraded():
    """An unimplemented method must raise, not return another method's answer."""
    w, r = _bed()
    with pytest.raises(NotImplementedError, match="not implemented"):
        mixed_convolution(PROBES, 10.0 - 0.01j, 1.0 + 0j, r, w, method="rational")


def test_unknown_method_raises():
    w, r = _bed()
    with pytest.raises(ValueError, match="method must be"):
        mixed_convolution(PROBES, 10.0 - 0.01j, 1.0 + 0j, r, w, method="nope")
