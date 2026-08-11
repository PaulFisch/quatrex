# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""End-to-end assembly of ``Sigma^R``, sign conventions included.

The review calls this a release blocker, and it earns the name: one test
catches sign errors, swapped ``<``/``>``, a duplicated ``1/2 Delta``, a
duplicated Hilbert term, and a wrongly signed analytic injection.

Two conventions are in play and the tests below name which is which.

*Theory*: ``Delta = Sigma^> - Sigma^<`` and
``Sigma^R = 1/2 Delta + i/2 H[Delta]``, consistent with
``Sigma^R - Sigma^A = Sigma^> - Sigma^<``.

*Stored*: the solver keeps the occupation-positive convention
``sigma^{<,>} = +i n(+1) Gamma``, i.e. ``sigma_stored = -sigma_theory``, so
``-i sigma^<`` and ``-i sigma^>`` are BOTH positive semidefinite and
``Sigma^R - Sigma^A = sigma_stored^< - sigma_stored^>``. The two differ by an
overall sign, and conflating them is what the Keldysh gate exists to catch.
"""
import numpy as np
import pytest

from quatrex.core.fft_utils import hilbert_transform


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


# --------------------------------------------------------------------------
# The bed, and the check that it is the right bed
# --------------------------------------------------------------------------

def _bosonic_pole_pair(w, centre=9.0, gamma=0.4, weight=1.0):
    r"""``Sigma^R`` of a bosonic resonance, as a conjugate-symmetric pole pair.

    ``Sigma^R(-w) = Sigma^R(w)^*`` -- the relation the production Hilbert
    kernel assumes for ``Delta``. A single pole does NOT satisfy it, and
    neither does the pole DIFFERENCE: it has to be the sum. Getting this wrong
    makes the transform look broken when it is the bed that is broken, which
    is exactly how an earlier version of this check went astray.
    """
    return weight * (1j / (w - centre + 1j * gamma)
                     + 1j / (w + centre + 1j * gamma))


def test_the_bed_has_the_symmetry_the_transform_assumes():
    """Guard the guard, with the wrong construction as a negative control."""
    t = np.linspace(0.5, 30.0, 121)
    good = _bosonic_pole_pair(-t) - np.conj(_bosonic_pole_pair(t))
    assert np.abs(good).max() < 1e-14, "the pole SUM is conjugate-symmetric"

    bad = (1j / (-t - 9.0 + 0.4j) - 1j / (-t + 9.0 + 0.4j))
    bad_ref = np.conj(1j / (t - 9.0 + 0.4j) - 1j / (t + 9.0 + 0.4j))
    assert np.abs(bad - bad_ref).max() > 1.0, \
        "the pole DIFFERENCE must visibly fail, or the guard proves nothing"


# --------------------------------------------------------------------------
# The numerical assembly
# --------------------------------------------------------------------------

def test_retarded_assembly_reproduces_an_exact_pole(monkeypatch):
    r"""``1/2 Delta + i/2 H[Delta]`` returns the ``Sigma^R`` it came from.

    Not to machine precision: ``hilbert_transform`` models ``Delta`` as
    cell-wise constant, and a Lorentzian is not, so there is a genuine
    quadrature error. What must hold is that it CONVERGES under refinement --
    a sign or double-counting error would not.
    """
    prev = None
    for ne in (4096, 16384, 65536):
        w = np.linspace(0.0, 240.0, ne)
        s_r = _bosonic_pole_pair(w)
        delta = s_r - np.conj(s_r)                    # = Sigma^R - Sigma^A
        got = 0.5 * delta + 0.5j * _h(
            hilbert_transform(delta[:, None], w))[:, 0]
        sel = (w > 1.0) & (w < 40.0)
        err = np.abs(got[sel] - s_r[sel]).max() / np.abs(s_r[sel]).max()
        if prev is not None:
            assert err < prev, f"must improve with refinement: {err:.3e} vs {prev:.3e}"
        prev = err
    assert prev < 2e-2


def test_the_opposite_sign_is_catastrophically_wrong():
    """Negative control for the whole convention: feeding ``-Delta``."""
    w = np.linspace(0.0, 240.0, 16384)
    s_r = _bosonic_pole_pair(w)
    delta = s_r - np.conj(s_r)
    got = -0.5 * delta + 0.5j * _h(hilbert_transform(-delta[:, None], w))[:, 0]
    sel = (w > 1.0) & (w < 40.0)
    err = np.abs(got[sel] - s_r[sel]).max() / np.abs(s_r[sel]).max()
    assert err > 1.0, f"a flipped Delta must be obviously wrong, got {err:.3e}"


# --------------------------------------------------------------------------
# The analytic injection, which is exact
# --------------------------------------------------------------------------

def test_analytic_injection_plus_the_global_half_is_the_full_retarded():
    r"""The SS sector must inject the Kramers-Kronig half ALONE.

    ``core/scba.py`` adds ``0.5 * (sigma_stored^< - sigma_stored^>)`` globally
    to the retarded buffer, and the stored components are ``-1x`` the bubble
    output. So for the total to come out as the true ``Sigma^R_SS``, the
    injected piece must be

        kk_half = Sigma^R_SS - 1/2 (Sigma^>_SS - Sigma^<_SS).

    Supplying the full ``Sigma^R_SS`` instead double counts the half term and
    silently breaks causality. This identity is exact, so it is tested at
    roundoff rather than at a tolerance.
    """
    rng = np.random.default_rng(0)
    ne = 64
    acc_l = rng.normal(size=(ne, 5)) + 1j * rng.normal(size=(ne, 5))
    acc_g = rng.normal(size=(ne, 5)) + 1j * rng.normal(size=(ne, 5))
    acc_r = rng.normal(size=(ne, 5)) + 1j * rng.normal(size=(ne, 5))

    kk_half = acc_r - 0.5 * (acc_g - acc_l)          # what interaction.py injects
    # what core/scba.py adds, in STORED variables (stored = -raw)
    global_half = 0.5 * ((-acc_l) - (-acc_g))
    scale = np.abs(acc_r).max()
    assert np.abs((kk_half + global_half) - acc_r).max() < 1e-14 * scale

    # Negative control: injecting the full retarded double counts the half,
    # and the surplus is the whole global term, not a rounding difference.
    doubled = acc_r + global_half
    assert np.abs(doubled - acc_r).max() > 0.1 * scale


def test_pole_sum_retarded_is_the_lower_half_plane_part():
    r"""``Sigma^R_SS`` keeps the LHP poles of ``Delta_SS`` and drops the rest.

    Checked against the closed form for a single Lorentzian:
    ``1/2 L + i/2 H[L] = i/(w - Omega + i gamma)``.
    """
    from quatrex.phonon.pole_kernel import lorentz_retarded

    w = np.linspace(0.0, 40.0, 2001)
    centre, gamma = 9.0, 0.4
    got = _h(lorentz_retarded(w, complex(centre, -gamma)))
    exact = 1j / (w - centre + 1j * gamma)
    assert np.abs(got - exact).max() < 1e-13
