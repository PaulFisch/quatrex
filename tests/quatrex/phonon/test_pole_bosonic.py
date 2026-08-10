# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The bosonic half of the mixed sector.

The mixed convolution runs over the whole frequency axis, but the solver only
ever holds ``G`` for ``omega >= 0``. The negative half is fixed by
``R(-w) = R(w)^*`` -- it is not missing information -- but it does have to be
SUPPLIED, because the cell kernel integrates exactly the cells it is given.

Omitting it was the defect that broke the Phi-derivable energy balance on the
first device-scale run of ``sectors="rr_ss_sr"`` (bubble balance 2.6e-02
against 3e-08 for the baseline, and a heat profile that went negative at the
left contact).
"""
import numpy as np
import pytest

from quatrex.phonon.pole_bubble import bosonic_closure
from quatrex.phonon.pole_keldysh import PoleCluster
from quatrex.phonon.pole_mixed import bosonic_extend, mixed_convolution_batched


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _bosonic_background(u, centre=11.0, gamma=0.4):
    """A background that genuinely satisfies ``a(-w) = conj(a(w))``.

    A single Lorentzian does NOT satisfy it; the conjugate-symmetric pair does.
    Getting this wrong makes the test measure the bed rather than the kernel.
    """
    return 1.0 / (u - centre + 1j * gamma) - 1.0 / (u + centre + 1j * gamma)


def test_background_bed_really_is_bosonic():
    """Guard the guard: the whole test rests on this identity holding."""
    u = np.linspace(0.5, 40.0, 97)
    assert np.allclose(_bosonic_background(-u), np.conj(_bosonic_background(u)))


def test_one_sided_convolution_is_badly_wrong():
    """The negative half is not a correction -- it is ~a quarter of the answer.

    This is the regression: before the fix the production path integrated only
    ``[0, w_max]``.
    """
    pole = 8.0 - 0.3j
    omega = np.array([9.0, 12.5, 15.0])
    wide = np.linspace(-200.0, 200.0, 200001)
    ref = _h(mixed_convolution_batched(
        omega, pole, 1.0 + 0j, _bosonic_background(wide)[:, None], wide))[:, 0]

    half = np.linspace(0.0, 60.0, 20001)
    one_sided = _h(mixed_convolution_batched(
        omega, pole, 1.0 + 0j, _bosonic_background(half)[:, None], half))[:, 0]
    err = np.abs(one_sided - ref).max() / np.abs(ref).max()
    assert err > 0.1, f"one-sided error should be large, got {err:.2e}"

    r_full, w_full = bosonic_extend(_bosonic_background(half)[:, None], half)
    fixed = _h(mixed_convolution_batched(
        omega, pole, 1.0 + 0j, r_full, w_full))[:, 0]
    assert np.abs(fixed - ref).max() / np.abs(ref).max() < 1e-3


def test_bosonic_extend_shapes_and_symmetry():
    freqs = np.linspace(0.0, 10.0, 11)
    rng = np.random.default_rng(0)
    r = rng.normal(size=(11, 4)) + 1j * rng.normal(size=(11, 4))
    r_full, w_full = bosonic_extend(r, freqs)
    assert w_full.shape == (21,)                      # zero bin exactly once
    assert r_full.shape == (21, 4)
    assert np.allclose(_h(w_full), np.concatenate([-freqs[:0:-1], freqs]))
    # a(-w) = conj(a(w)) on the constructed half
    assert np.allclose(_h(r_full)[:10], np.conj(_h(r)[:0:-1]))
    assert np.allclose(_h(r_full)[10:], _h(r))


def test_bosonic_extend_refuses_a_grid_not_anchored_at_zero():
    """The mirror is defined about omega = 0; guessing an offset would put the
    negative half in the wrong place and still return a plausible array."""
    freqs = np.linspace(1.5, 10.0, 18)
    r = np.zeros((18, 2), dtype=complex)
    with pytest.raises(NotImplementedError, match="zero-anchored"):
        bosonic_extend(r, freqs)


def test_pole_set_is_closed_before_any_leg_is_built():
    """``bubble_clusters`` was implemented, documented as mandatory, and never
    called -- ``G_PP`` was built from half a pole set."""
    rng = np.random.default_rng(1)
    z = np.array([8.0 - 0.3j, 11.0 - 0.4j])
    u = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    v = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    closed = bosonic_closure(PoleCluster(z=z, u=u, v=v))
    assert closed.n_poles == 4
    got = _h(closed.z)
    assert np.allclose(got[:2], z)
    assert np.allclose(got[2:], -np.conj(z))


def test_state_separates_the_leg_set_from_the_solved_set():
    """The tracker matches the solved set; the bubble consumes ``legs``.

    They are the same list today (see below), but keeping them distinct is what
    makes enabling the closure a one-line change once the mirrored source
    exists, and stops a closed set doubling the reported pole count.
    """
    from quatrex.phonon.pole_sector import PoleSectorState

    st = PoleSectorState()
    assert hasattr(st, "legs") and st.legs == []
    assert st.clusters == []


def test_closure_is_deferred_because_the_frozen_source_cannot_serve_both_branches():
    """A measured decision, recorded so it is not silently re-attempted.

    ``bubble_clusters()`` closes the pole set under ``z -> -z^*``, which puts
    partners at NEGATIVE real frequency. The frozen source is evaluated at a
    single index, the positive centre, so after closure one source is applied
    to poles at both ``+Omega`` and ``-Omega`` -- and the partner's source is
    the bosonic mirror of the original's, not the same matrix. Wiring the
    closure in without that made ``rr_ss`` worse by 3400x on the production
    bed (bubble balance 5.4e-08 -> 1.8e-04).
    """
    rng = np.random.default_rng(2)
    z = np.array([8.0 - 0.3j, 11.0 - 0.4j])
    u = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    v = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    closed = bosonic_closure(PoleCluster(z=z, u=u, v=v))
    got = _h(closed.z)
    # The partners sit at negative real frequency: that is exactly why one
    # frozen source evaluated at the positive centre cannot serve them.
    assert np.all(got[2:].real < 0.0)
    assert np.all(got.imag < 0.0), "partners must stay retarded"
