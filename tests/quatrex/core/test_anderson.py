"""Unit tests for the Anderson/Pulay SCBA mixer."""

from __future__ import annotations

import numpy as np

from quatrex.core.anderson import AndersonMixer


def _oscillating_map(n=40, seed=0):
    """A complex affine map ``g(x)=A x + b`` whose damped-linear iteration
    OSCILLATES (an eigenvalue of A near -1), mimicking the period-2 SCBA
    instability. Returns ``(g, x_star)``."""
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(n, n) + 1j * rng.randn(n, n))
    eig = np.concatenate([
        np.array([-0.95, -0.9, 0.85]),
        0.3 * (rng.randn(n - 3) + 1j * rng.randn(n - 3)),
    ])
    A = Q @ np.diag(eig) @ Q.conj().T
    b = rng.randn(n) + 1j * rng.randn(n)
    x_star = np.linalg.solve(np.eye(n) - A, b)
    return (lambda x: A @ x + b), x_star


def test_anderson_converges_oscillating_map_where_linear_fails():
    g, x_star = _oscillating_map()
    n = x_star.size

    # Damped linear mixing on an eigenvalue ~ -0.95 does NOT converge.
    x = np.zeros(n, complex)
    lin_err = []
    for _ in range(80):
        x = 0.5 * x + 0.5 * g(x)
        lin_err.append(np.linalg.norm(x - x_star))
    assert min(lin_err) > 1e-2  # linear stalls/oscillates

    # Anderson converges to the fixed point.
    mix = AndersonMixer(depth=10, beta=0.5)
    x = np.zeros(n, complex)
    and_err = []
    for _ in range(80):
        x = mix.step(x, g(x))
        and_err.append(np.linalg.norm(x - x_star))
    assert and_err[-1] < 1e-8
    assert and_err[-1] < lin_err[-1] * 1e-4  # vastly better than linear


def test_anderson_first_step_is_damped_linear():
    """With no history the first step must equal the bare damped-linear step
    (so a converged/easy problem is never made worse on iter 0)."""
    g, _ = _oscillating_map(n=12, seed=1)
    mix = AndersonMixer(depth=5, beta=0.3)
    x0 = np.zeros(12, complex)
    x1 = mix.step(x0, g(x0))
    np.testing.assert_allclose(x1, x0 + 0.3 * (g(x0) - x0), rtol=1e-12)


def test_anderson_fixed_point_is_invariant():
    """At the fixed point (x_out == x_in) the residual is zero and the mixer
    returns the fixed point unchanged."""
    mix = AndersonMixer(depth=5, beta=0.5)
    x = np.arange(8, dtype=complex) + 1j
    xm = mix.step(x, x.copy())
    assert np.linalg.norm(x.copy() - x) == 0.0
    np.testing.assert_allclose(xm, x, rtol=1e-12)
