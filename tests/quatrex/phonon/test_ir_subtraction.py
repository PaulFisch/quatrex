# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Tests for the exact infrared (Bose-pole) singularity subtraction.

The Cauchy principal-value quadrature :func:`quatrex.phonon.ir_subtraction.pv_origin`
is checked against ``scipy.integrate.quad(weight="cauchy")`` (the exact PV), on
symmetric and asymmetric grids, including its convergence order and the role of
the analytic ``ln(b/|a|)`` term. The Bose split and the subtracted Bose-weighted
integral are checked against independent references. See
``document/src/appendices/infrared.tex``.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from quatrex.phonon.ir_subtraction import (
    bose,
    bose_pole_coeff,
    bose_split,
    bose_weighted_pv,
    pv_origin,
)


def _cauchy_ref(fn, a, b):
    """Exact PV integral of fn(x)/x over [a, b] (pole at 0)."""
    val, _ = integrate.quad(fn, a, b, weight="cauchy", wvar=0.0, limit=200)
    return val


def _grid_through_zero(a, b, n):
    """Uniform grid on [a, b] with n points; assert 0 is a node."""
    w = np.linspace(a, b, n)
    assert np.min(np.abs(w)) < 1e-12, "0 must be a grid node"
    return w


# --------------------------------------------------------------------------
# pv_origin: Cauchy principal value vs scipy
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fn, label",
    [
        (lambda x: np.exp(0.7 * x), "exp"),
        (lambda x: np.cos(3.0 * x) + x**2, "cos+quad"),
        (lambda x: 1.0 / ((x - 0.3) ** 2 + 0.5), "lorentzian"),
        (lambda x: np.exp(-(x**2) / 2.0), "gaussian"),
    ],
)
def test_pv_origin_symmetric_vs_scipy(fn, label):
    w = _grid_through_zero(-4.0, 4.0, 2001)
    ref = _cauchy_ref(fn, -4.0, 4.0)
    val = pv_origin(fn(w), w)
    assert abs(val - ref) < 1e-3 * max(1.0, abs(ref)), (label, val, ref)


def test_pv_origin_asymmetric_vs_scipy():
    # pole interior but grid NOT symmetric -> the analytic ln(b/|a|) term matters
    w = _grid_through_zero(-2.0, 6.0, 401)
    fn = lambda x: np.exp(0.4 * x) + 0.5  # noqa: E731  (q(0) != 0)
    ref = _cauchy_ref(fn, -2.0, 6.0)
    val = pv_origin(fn(w), w)
    assert abs(val - ref) < 5e-3 * abs(ref), (val, ref)


def test_subtraction_beats_bare_punctured_on_asymmetric_grid():
    """The q(0)*ln(b/|a|) term is what a bare node-omitting sum misses."""
    w = _grid_through_zero(-2.0, 6.0, 401)
    fn = lambda x: np.exp(0.4 * x) + 0.5  # noqa: E731  (q(0) != 0)
    q = fn(w)
    ref = _cauchy_ref(fn, -2.0, 6.0)
    dw = w[1] - w[0]
    nz = np.abs(w) > 1e-12
    bare = float(np.sum(q[nz] / w[nz]) * dw)  # no subtraction, omit the node
    subtracted = pv_origin(q, w)
    assert abs(subtracted - ref) < abs(bare - ref) / 20.0, (subtracted, bare, ref)


def test_pv_origin_second_order_convergence():
    fn = lambda x: np.exp(0.7 * x)  # noqa: E731
    ref = _cauchy_ref(fn, -4.0, 4.0)
    errs = []
    for n in (251, 501, 1001, 2001):
        w = _grid_through_zero(-4.0, 4.0, n)
        errs.append(abs(pv_origin(fn(w), w) - ref))
    # halving h (doubling N) should cut the error ~4x; require >=2.5x per step
    for k in range(len(errs) - 1):
        assert errs[k + 1] < errs[k] / 2.5, errs


def test_pv_origin_handles_complex_and_vector():
    w = _grid_through_zero(-3.0, 3.0, 1001)
    # complex, 2-component "leg": each component an independent PV
    q = np.stack([np.exp(0.5 * w), np.cos(2.0 * w)], axis=1) + 1j * np.stack(
        [np.sin(w), np.exp(-(w**2))], axis=1
    )
    out = pv_origin(q, w)
    assert out.shape == (2,)
    for j in range(2):
        ref = _cauchy_ref(lambda x, j=j: np.real(
            (np.exp(0.5 * x) if j == 0 else np.cos(2.0 * x))), -3.0, 3.0)
        assert abs(out[j].real - ref) < 1e-3 * max(1.0, abs(ref))


# --------------------------------------------------------------------------
# Bose split
# --------------------------------------------------------------------------

def test_bose_split_exact_and_bounded():
    w = np.linspace(0.05, 50.0, 600)
    T = 300.0
    n_tilde, c = bose_split(w, T)
    assert abs(c - bose_pole_coeff(T)) < 1e-15
    # exact reconstruction off the node
    assert np.allclose(n_tilde + c / w, bose(w, T), rtol=1e-12, atol=1e-12)
    # remainder is bounded and -> -1/2 as omega -> 0
    assert np.all(np.abs(n_tilde) < 1.0)
    nt_small, _ = bose_split(np.array([1e-3]), T)
    assert abs(nt_small[0] + 0.5) < 5e-3


# --------------------------------------------------------------------------
# Bose-weighted integral via subtraction
# --------------------------------------------------------------------------

def test_bose_weighted_pv_matches_reference():
    w = _grid_through_zero(-6.0, 6.0, 4001)
    T = 300.0
    c = bose_pole_coeff(T)
    A = lambda x: 1.0 / ((x - 0.4) ** 2 + 1.0)  # noqa: E731 smooth, A(0) != 0

    def n_tilde_scalar(x):  # bounded remainder n - c/x, -> -1/2 at x = 0
        if abs(x) < 1e-12:
            return -0.5
        return float(bose(np.array([x]), T)[0]) - c / x

    # reference: regular part (n - c/x)*A by ordinary quad + c * PV[A/x]
    reg_ref, _ = integrate.quad(
        lambda x: n_tilde_scalar(x) * A(x), -6.0, 6.0, limit=200,
    )
    sing_ref = c * _cauchy_ref(A, -6.0, 6.0)
    ref = reg_ref + sing_ref
    val = bose_weighted_pv(A(w), w, T)
    assert abs(val - ref) < 2e-3 * abs(ref), (val, ref)
