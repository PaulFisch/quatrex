# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Pins the complex-frequency continuation against the production Hilbert kernel.

The pole sector rests on one identity: the retarded scattering self-energy the
solver already builds,

    Sigma^R = 0.5*Delta + 0.5j*hilbert_transform(Delta),   Delta = Sigma^> - Sigma^<,

is the real-axis restriction of an explicit analytic function of complex z (see
:mod:`quatrex.phonon.pole_kernel`). If that identity does not hold to roundoff,
then evaluating M(z) off the real axis is a new approximation rather than a
change of representation, and the whole sector loses its justification. These
tests are therefore the acceptance gate for Phase 0.
"""
import numpy as np
import pytest

from quatrex.core.fft_utils import hilbert_transform
from quatrex.phonon.pole_kernel import (
    bosonic_partner,
    cell_width,
    contract_delta,
    continuation_weights,
    delta_local_fit,
    lorentz_pair_retarded,
    lorentz_retarded,
    sigma_retarded_at_z,
)

# Strictly positive, but far below any physical linewidth: it selects the
# retarded branch of the pole cell without perturbing the value.
TINY = 1e-30


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _grid(ne, fmax=20.0):
    return np.linspace(0.0, fmax, ne)


def _delta(rng, freqs, tail=()):
    """A smooth, compactly supported, complex Delta with the right decay."""
    shape = (freqs.size,) + tail
    base = (
        np.exp(-((freqs - 0.35 * freqs[-1]) / (0.07 * freqs[-1])) ** 2)
        + 0.4 * np.exp(-((freqs - 0.65 * freqs[-1]) / (0.10 * freqs[-1])) ** 2)
    )
    out = np.empty(shape, dtype=complex)
    for idx in np.ndindex(tail):
        amp = rng.normal(size=2).view(complex)[0] if tail else 1.0 + 0.3j
        out[(slice(None),) + idx] = base * amp
    if not tail:
        out = (base * (1.0 + 0.3j)).astype(complex)
    return out


def _delta_random(rng, freqs, tail=()):
    """An arbitrary Delta, including real weight at omega = 0.

    The continuation identity is algebraic -- exact for any cell-wise-constant
    Delta -- so a random draw tests it harder than a smooth one, and it is the
    only way to exercise the DC-cell bookkeeping.
    """
    shape = (freqs.size,) + tail
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _production_sigma_r(a, freqs, transverse_shape=()):
    return 0.5 * a + 0.5j * hilbert_transform(a, freqs, transverse_shape=transverse_shape)


# --------------------------------------------------------------------------- #
# 1. The identity the whole sector rests on.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ne", [101, 201, 501])
@pytest.mark.parametrize("kind", ["smooth", "random"])
def test_continuation_matches_production_hilbert(ne, kind):
    """F(omega + i0) reproduces 0.5*Delta + 0.5j*H[Delta] to roundoff."""
    rng = np.random.default_rng(0)
    freqs = _grid(ne)
    a = _delta(rng, freqs) if kind == "smooth" else _delta_random(rng, freqs)

    ref = _production_sigma_r(a, freqs)
    got = sigma_retarded_at_z(a, freqs, freqs + 1j * TINY, sheet="I")

    scale = np.abs(_h(ref)).max()
    err = np.abs(_h(got) - _h(ref)).max() / scale
    assert err < 1e-12, f"relative error {err:.3e}"


def test_continuation_matches_production_hilbert_with_transverse_axes():
    """The bosonic mirror carries q -> -q, exactly as the production kernel does."""
    rng = np.random.default_rng(7)
    freqs = _grid(151)
    nk = (3, 2)
    a = _delta(rng, freqs, tail=nk + (4,))

    ref = _production_sigma_r(a, freqs, transverse_shape=nk)
    w_pos, w_mir = continuation_weights(freqs + 1j * TINY, freqs)
    got = contract_delta(a, w_pos, w_mir, transverse_shape=nk)

    scale = np.abs(_h(ref)).max()
    err = np.abs(_h(got) - _h(ref)).max() / scale
    assert err < 1e-12, f"relative error {err:.3e}"


def test_dc_cell_is_counted_once():
    """Dropping the omega=0 mirror column is what makes the identity hold at DC.

    Restoring that column must break the match, otherwise the test above would
    pass for the wrong reason.
    """
    rng = np.random.default_rng(3)
    freqs = _grid(121)
    a = _delta_random(rng, freqs)

    w_pos, w_mir = continuation_weights(freqs + 1j * TINY, freqs)
    assert np.allclose(_h(w_mir)[:, 0], 0.0)

    ref = _production_sigma_r(a, freqs)
    bad_pos, bad_mir = continuation_weights(freqs + 1j * TINY, freqs)
    # Undo the DC guard by hand.
    h = cell_width(freqs)
    pref = 1j / (2.0 * np.pi)
    zz = (freqs + 1j * TINY).reshape(-1, 1)
    bad_mir[:, 0] = (
        pref * (np.log(zz[:, 0] + 0.0 + h / 2) - np.log(zz[:, 0] + 0.0 - h / 2))
    )
    bad = contract_delta(a, bad_pos, bad_mir)
    err = np.abs(_h(bad) - _h(ref)).max() / np.abs(_h(ref)).max()
    assert err > 1e-3, "double-counting the DC cell should be visible"


# --------------------------------------------------------------------------- #
# 2. The branch cut, i.e. what makes the resonance sheet reachable.
# --------------------------------------------------------------------------- #

def test_branch_cut_jump_is_delta():
    """F(w+i0) - F(w-i0) == Delta(w): the second-sheet term is exactly Delta."""
    rng = np.random.default_rng(11)
    freqs = _grid(2001)
    a = _delta(rng, freqs)

    probe = np.array([3.0, 7.0, 9.0, 13.0, 17.0])
    up = sigma_retarded_at_z(a, freqs, probe + 1j * TINY, sheet="I")
    dn = sigma_retarded_at_z(a, freqs, probe - 1j * TINY, sheet="I")

    exact = np.interp(probe, freqs, a.real) + 1j * np.interp(probe, freqs, a.imag)
    err = np.abs(_h(up) - _h(dn) - exact).max() / np.abs(exact).max()
    assert err < 1e-6, f"jump mismatch {err:.3e}"


def test_second_sheet_is_continuous_across_the_axis():
    """Sigma^{R,II} continues the retarded branch: it agrees with F(w+i0) on the axis."""
    rng = np.random.default_rng(13)
    freqs = _grid(801)
    a = _delta(rng, freqs)
    probe = np.array([6.0, 8.5, 12.0])

    above = sigma_retarded_at_z(a, freqs, probe + 1j * TINY, sheet="I")
    below = sigma_retarded_at_z(a, freqs, probe - 1j * 1e-9, sheet="II")

    err = np.abs(_h(above) - _h(below)).max() / np.abs(_h(above)).max()
    # Bounded by the local polynomial model of Delta, not by the continuation.
    assert err < 1e-4, f"second sheet is discontinuous at the axis: {err:.3e}"


# --------------------------------------------------------------------------- #
# 3. Derivatives, used by the bordered Newton corrector.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("order", [1, 2])
def test_derivative_weights_match_finite_difference(order):
    rng = np.random.default_rng(5)
    freqs = _grid(301)
    a = _delta(rng, freqs)

    z = np.array([7.3 - 0.05j, 11.1 - 0.2j])
    d = 1e-5

    ana = sigma_retarded_at_z(a, freqs, z, sheet="I", order=order)
    lo = sigma_retarded_at_z(a, freqs, z - d, sheet="I", order=order - 1)
    hi = sigma_retarded_at_z(a, freqs, z + d, sheet="I", order=order - 1)
    fd = (_h(hi) - _h(lo)) / (2 * d)

    err = np.abs(_h(ana) - fd).max() / np.abs(fd).max()
    assert err < 1e-6, f"derivative order {order} mismatch {err:.3e}"


# --------------------------------------------------------------------------- #
# 4. The analytic Lorentzian partner, used to bypass the Hilbert transform.
# --------------------------------------------------------------------------- #

def test_lorentz_retarded_matches_the_numerical_transform():
    """0.5*L + 0.5j*H[L] == i/(w - Omega + i gamma), on a grid fine enough to resolve L."""
    Omega, gamma = 8.0, 0.35
    freqs = np.linspace(0.0, 60.0, 24001)
    L = 2 * gamma / ((freqs - Omega) ** 2 + gamma**2)

    ref = _production_sigma_r(L.astype(complex), freqs)
    got = lorentz_pair_retarded(freqs, complex(Omega, -gamma))

    # Compare where the numerical transform is not dominated by its own tail
    # truncation (the analytic form has infinite support, the grid does not).
    sel = (freqs > 1.0) & (freqs < 40.0)
    err = np.abs(_h(got)[sel] - _h(ref)[sel]).max() / np.abs(_h(ref)[sel]).max()
    assert err < 5e-3, f"analytic Lorentzian partner mismatch {err:.3e}"


def test_lorentz_mirror_partner_is_required():
    """Dropping the -Omega mirror leaves a visible, systematic error."""
    Omega, gamma = 8.0, 0.35
    freqs = np.linspace(0.0, 60.0, 24001)
    L = 2 * gamma / ((freqs - Omega) ** 2 + gamma**2)
    ref = _production_sigma_r(L.astype(complex), freqs)

    sel = (freqs > 1.0) & (freqs < 40.0)
    with_mirror = lorentz_pair_retarded(freqs, complex(Omega, -gamma))
    without = lorentz_retarded(freqs, complex(Omega, -gamma))

    e_with = np.abs(_h(with_mirror)[sel] - _h(ref)[sel]).max()
    e_without = np.abs(_h(without)[sel] - _h(ref)[sel]).max()
    assert e_without > 10 * e_with, (
        f"mirror partner made no difference: {e_without:.3e} vs {e_with:.3e}"
    )


def test_lorentz_retarded_rejects_an_upper_half_plane_pole():
    with pytest.raises(ValueError, match="lower half plane"):
        lorentz_retarded(np.linspace(0, 10, 11), complex(5.0, +0.1))


# --------------------------------------------------------------------------- #
# 5. The local continuation of Delta (the second-sheet term).
# --------------------------------------------------------------------------- #

def test_delta_local_fit_reproduces_grid_samples():
    rng = np.random.default_rng(17)
    freqs = _grid(401)
    a = _delta(rng, freqs)

    probe = freqs[[40, 120, 260, 350]] + 1j * TINY
    got = delta_local_fit(a, freqs, probe, order=3, window=4)
    ref = a[[40, 120, 260, 350]]

    err = np.abs(_h(got) - ref).max() / np.abs(ref).max()
    # A least-squares model, deliberately not an interpolant: Delta comes out of
    # the SCBA and is noisy, so smoothing is wanted. Its error is the truncation
    # of the degree-p fit, checked for order in the next test.
    assert err < 1e-4, f"local fit misses its own nodes by {err:.3e}"


@pytest.mark.parametrize("order", [1, 3])
def test_delta_local_fit_converges_with_the_grid(order):
    """The fit error must fall like h^(order+1); that is what bounds the sheet term."""
    errs = []
    for ne in (201, 401, 801):
        rng = np.random.default_rng(17)
        freqs = _grid(ne)
        a = _delta(rng, freqs)
        k = ne // 3
        got = delta_local_fit(a, freqs, np.array([freqs[k] + 1j * TINY]),
                              order=order, window=4)
        errs.append(abs(_h(got)[0] - a[k]) / abs(a[k]))
    rate = np.log2(errs[0] / errs[-1]) / 2.0
    assert rate > order + 0.5, f"observed order {rate:.2f} for a degree-{order} fit"


def test_delta_local_fit_uses_the_mirror_for_negative_frequency():
    """Delta(-w) = Delta(w)* -- the fit must serve the bosonic partner there."""
    rng = np.random.default_rng(19)
    freqs = _grid(401)
    a = _delta(rng, freqs)

    w = freqs[200]
    pos = delta_local_fit(a, freqs, np.array([w + 1j * TINY]), order=3)
    neg = delta_local_fit(a, freqs, np.array([-w + 1j * TINY]), order=3)

    assert np.abs(_h(neg)[0] - np.conj(_h(pos)[0])) < 1e-8 * abs(_h(pos)[0])


# --------------------------------------------------------------------------- #
# 6. Refusals.
# --------------------------------------------------------------------------- #

def test_real_z_raises():
    freqs = _grid(51)
    with pytest.raises(ValueError, match="Im z"):
        continuation_weights(freqs, freqs)


def test_nonuniform_grid_raises():
    freqs = np.concatenate([np.linspace(0, 5, 20), np.linspace(5.5, 20, 30)])
    with pytest.raises(ValueError, match="uniform"):
        continuation_weights(np.array([1.0 - 0.1j]), freqs)


def test_weight_shape_mismatch_raises():
    freqs = _grid(51)
    w_pos, w_mir = continuation_weights(np.array([1.0 - 0.1j]), freqs)
    with pytest.raises(ValueError, match="frequencies"):
        contract_delta(np.zeros(50, dtype=complex), w_pos, w_mir)


def test_bosonic_partner_negates_transverse_axes():
    rng = np.random.default_rng(23)
    a = rng.normal(size=(4, 3, 2)) + 1j * rng.normal(size=(4, 3, 2))
    got = _h(bosonic_partner(a, (3,)))
    for iq in range(3):
        assert np.allclose(got[:, iq], np.conj(a[:, (-iq) % 3]))


def test_pinned_anchor_makes_the_local_fit_continuous():
    """``M(z)`` must be holomorphic, not piecewise holomorphic."""
    import numpy as np

    from quatrex.phonon.pole_kernel import delta_local_fit

    rng = np.random.default_rng(0)
    e = np.linspace(0.0, 20.0, 81)
    a = rng.normal(size=(81, 3)) + 1j * rng.normal(size=(81, 3))
    probes = np.linspace(9.35, 9.40, 11)

    def _steps(anchor):
        v = [delta_local_fit(a, e, np.array([w - 0.02j]), order=2, window=4,
                             anchor=anchor)[0, 0] for w in probes]
        return np.abs(np.diff(_h(np.array(v))))

    free = _steps(None)
    pinned = _steps(9.30)
    # Unanchored: one step is an order of magnitude above the rest.
    assert free.max() / np.median(free) > 10.0
    # Pinned: the sweep is smooth, so max and median agree.
    assert pinned.max() / np.median(pinned) < 1.5
    assert pinned.max() < 0.1 * free.max()


def test_pinned_anchor_keeps_the_derivative_exact():
    """Pinning must not break ``dM/dz``: the fit stays a polynomial in
    ``(z - anchor)/h``, so the analytic derivative is still the true one."""
    import numpy as np

    from quatrex.phonon.pole_kernel import sigma_retarded_at_z

    rng = np.random.default_rng(1)
    e = np.linspace(0.0, 20.0, 81)
    a = rng.normal(size=(81, 3)) + 1j * rng.normal(size=(81, 3))
    z0, anchor = 9.10 - 0.02j, 9.10
    prev = None
    for eps in (1e-3, 1e-4, 1e-5):
        kw = dict(sheet="II", anchor=anchor)
        f1 = _h(sigma_retarded_at_z(a, e, np.array([z0 + eps]), order=0, **kw))[0, 0]
        f0 = _h(sigma_retarded_at_z(a, e, np.array([z0 - eps]), order=0, **kw))[0, 0]
        an = _h(sigma_retarded_at_z(a, e, np.array([z0]), order=1, **kw))[0, 0]
        err = abs((f1 - f0) / (2 * eps) - an) / abs(an)
        if prev is not None:
            assert err < prev / 50.0, "must converge as eps^2"
        prev = err
    assert prev < 1e-7
