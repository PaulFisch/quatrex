# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Unit tests shared by every SCBA mixer.

The mixers are pure ``(x, g(x)) -> x_new`` maps, so they can be driven against
synthetic fixed-point problems with a known Jacobian spectrum.
"""

from __future__ import annotations

import numpy as np
import pytest

from quatrex.core.anderson import AndersonMixer, RREMixer
from quatrex.core.broyden import BroydenMixer
from quatrex.core.jfnk import JFNKMixer
from quatrex.core.mpi_linalg import complex_to_real, real_to_complex
from quatrex.core.rpm import RPMMixer

N = 40


def _mixers(beta: float = 0.5) -> dict:
    """Every mixer, configured to engage quickly on a small problem."""
    return {
        "anderson": AndersonMixer(depth=8, beta=beta),
        "rre": RREMixer(cycle=8, beta=beta),
        "broyden": BroydenMixer(depth=8, beta=beta, warmup=0, patience=0),
        "rpm": RPMMixer(max_subspace=6, beta=beta, warmup=5, patience=0, trust=0),
        "jfnk": JFNKMixer(warmup=5, beta=beta, verbose=False, trust=0),
    }


MIXERS = list(_mixers())


def _real_linear_map(seed: int = 0):
    """An R-linear, NON-analytic map ``F(z) = A z + B conj(z) + c``.

    This is the structure of the SCBA map: ``G^A = (G^R)^H``, the
    Kramers-Kronig transform and the bosonic fold all conjugate, so
    ``dF/dSigma`` is real-linear only, NOT complex-analytic.

    The map is built from its real-embedded Jacobian, whose spectrum is chosen
    to have two eigenvalues above 1 (the rest contractive). Damped Picard
    multiplies eigenvalue ``lam`` by ``1 - beta + beta * lam``, which exceeds 1
    for every ``beta > 0`` when ``lam > 1``, so the fixed point is unreachable
    by ANY damping -- the regime the quasi-Newton mixers exist for.
    """
    rng = np.random.default_rng(seed)

    eigenvalues = np.concatenate(
        [[1.5, 1.3], rng.uniform(-0.5, 0.5, size=2 * N - 2)]
    )
    basis, __ = np.linalg.qr(rng.standard_normal((2 * N, 2 * N)))
    jacobian = basis @ np.diag(eigenvalues) @ basis.T

    # Split the real-embedded Jacobian back into the analytic (A) and
    # anti-analytic (B) halves; B != 0 is what makes the map non-analytic.
    m11, m12 = jacobian[:N, :N], jacobian[:N, N:]
    m21, m22 = jacobian[N:, :N], jacobian[N:, N:]
    a = 0.5 * ((m11 + m22) + 1j * (m21 - m12))
    b = 0.5 * ((m11 - m22) + 1j * (m21 + m12))
    assert np.linalg.norm(b) > 1.0  # genuinely non-analytic

    c = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    x_star = real_to_complex(
        np.linalg.solve(np.eye(2 * N) - jacobian, complex_to_real(c))
    )

    return (lambda z: a @ z + b @ z.conj() + c), x_star


def _drive(mixer, g, x0, num_iterations: int, tol: float = 1e-9) -> np.ndarray:
    """Drives `mixer` to convergence and returns the last accepted iterate.

    Mirrors `SCBA.run`: the residual gate is tested only on ACCEPTED iterates
    (JFNK emits finite-difference probes ``x_k + eps*v`` through the same
    channel, flagged by `mixer.probing`), and the loop stops once converged.
    """
    x = accepted = x0
    for _ in range(num_iterations):
        x = mixer.step(x, g(x))
        if not np.all(np.isfinite(x)):
            break
        if mixer.probing:
            continue
        accepted = x
        if np.linalg.norm(g(x) - x) < tol:
            break

    return accepted


@pytest.mark.parametrize("name", MIXERS)
def test_lands_unstable_fixed_point_of_real_linear_map(name):
    """Every mixer must land the fixed point of the R-linear SCBA-like map.

    Damped Picard cannot: the map is iteration-unstable by construction. A
    mixer that fits COMPLEX least-squares coefficients linearises the wrong
    (complex-analytic) map and loses many digits here, so this is the test that
    pins the real embedding.
    """
    g, x_star = _real_linear_map()

    picard = _drive(AndersonMixer(depth=0, beta=0.5), g, np.zeros(N, complex), 300)
    assert np.linalg.norm(g(picard) - picard) > 1.0  # Picard diverges

    x = _drive(_mixers()[name], g, np.zeros(N, complex), 300)
    assert np.iscomplexobj(x)
    assert np.linalg.norm(x - x_star) < 1e-6


@pytest.mark.parametrize("name", MIXERS)
def test_fixed_point_is_invariant(name):
    """At the fixed point the residual vanishes, so the mixer must not move."""
    x = np.arange(N, dtype=complex) + 1j
    x_new = _mixers()[name].step(x, x.copy())

    np.testing.assert_allclose(x_new, x, atol=1e-12)


@pytest.mark.parametrize("name", MIXERS)
def test_first_step_is_damped_linear(name):
    """With no history the first step is the bare damped-linear step, so an
    already-easy problem is never made worse on iteration 0."""
    beta = 0.3
    g, __ = _real_linear_map()
    x0 = np.zeros(N, complex)

    x1 = _mixers(beta=beta)[name].step(x0, g(x0))

    np.testing.assert_allclose(x1, x0 + beta * (g(x0) - x0), rtol=1e-10)


def test_real_embedding_round_trip():
    """`complex_to_real` and `real_to_complex` are mutual inverses."""
    z = np.arange(N, dtype=complex) + 1j * np.arange(N)

    np.testing.assert_array_equal(real_to_complex(complex_to_real(z)), z)


def test_jfnk_rejects_a_non_descent_trial_before_relinearising():
    """A Newton overshoot must retry from its base, not become the next base.

    For ``R(x)=x**3-2*x+2`` the Newton step from x=1 lands at x=0,
    increasing ``|R|`` from one to two.  The state-machine JFNK sees that
    residual on the call after emitting the trial.
    """
    def fixed_point_map(z):
        return z + z**3 - 2.0 * z + 2.0

    mixer = JFNKMixer(
        warmup=0,
        max_krylov=1,
        inner_tol=0.0,
        forcing="fixed",
        eps=1e-7,
        trust=2.0,
        trust_max=2.0,
        verbose=False,
    )
    base = np.array([1.0 + 0.0j])

    probe = mixer.step(base, fixed_point_map(base))
    assert mixer.probing
    trial = mixer.step(probe, fixed_point_map(probe))
    assert not mixer.probing
    assert np.linalg.norm(fixed_point_map(trial) - trial) > 1.5

    retry_probe = mixer.step(trial, fixed_point_map(trial))

    assert mixer.probing
    assert mixer._trust_k == pytest.approx(1.0)
    # The retry is a finite-difference probe about x=1, not about the rejected
    # x=0 trial.
    assert abs(retry_probe[0] - base[0]) < 1e-4
    assert abs(retry_probe[0] - trial[0]) > 0.5


def test_jfnk_rejection_does_not_enlarge_a_small_requested_trust_radius():
    """A near-root continuation may need a radius below the old 1e-3 floor."""
    mixer = JFNKMixer(
        warmup=0,
        max_krylov=1,
        inner_tol=0.0,
        forcing="fixed",
        eps=1e-7,
        trust=1e-4,
        trust_max=1e-4,
        verbose=False,
    )

    # Seed the state immediately before a rejected trial.  A smooth Newton
    # direction with a 1e-4 cap is locally descending, so manufacturing the
    # rejection through a scalar polynomial does not actually exercise this
    # branch.  Here the trial residual is explicitly twice the stored base
    # residual, while the stored real vectors let the retry reopen GMRES at
    # the base point exactly as it does in production.
    mixer._trial_pending = True
    mixer._Rk_norm = 1.0
    mixer._Rk_merit = 1.0
    mixer._xk_r = np.array([1.0, 0.0])
    mixer._Fxk_r = np.array([2.0, 0.0])
    mixer.step(np.array([1.0 + 0.0j]), np.array([3.0 + 0.0j]))

    assert mixer._trust_k == pytest.approx(5e-5)
