from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PATH = (
    Path(__file__).resolve().parents[3]
    / "phonon/studies/_si_ballistic_mode_count.py"
)
SPEC = importlib.util.spec_from_file_location("si_ballistic_mode_count", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_scalar_chain_mode_count() -> None:
    # D(k) = 2 - 2 cos(2 pi k), hence nu spans [0, 2] and one mode has
    # positive group velocity at every interior frequency.
    transformed = {
        -1: np.array([[-1.0]]),
        0: np.array([[2.0]]),
        1: np.array([[-1.0]]),
    }
    bands = MODULE.bloch_bands(transformed, np.linspace(-0.5, 0.5, 4097))
    frequencies = np.array([0.25, 0.75, 1.25, 1.75, 2.25])
    np.testing.assert_array_equal(
        MODULE.positive_mode_count(bands, frequencies),
        np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
    )


def test_positive_variation_current_converges_to_frequency_integral() -> None:
    transformed = {
        -1: np.array([[-1.0]]),
        0: np.array([[2.0]]),
        1: np.array([[-1.0]]),
    }
    bands = MODULE.bloch_bands(transformed, np.linspace(-0.5, 0.5, 8193))
    current = MODULE.positive_variation_current(bands, 305.0, 295.0)
    frequency = np.linspace(1.0e-8, 2.0, 100_001)
    occupation = MODULE.bose(frequency, 305.0) - MODULE.bose(frequency, 295.0)
    reference = np.trapezoid(
        MODULE.PLANCK * MODULE.THZ * frequency * occupation,
        frequency * MODULE.THZ,
    )
    assert abs(current / reference - 1.0) < 2.0e-7


def test_frequency_weighted_bose_difference_has_finite_zero_limit() -> None:
    frequency = np.array([0.0, 1.0e-8, 1.0])
    weighted = MODULE.frequency_weighted_occupation_difference(
        frequency, 305.0, 295.0
    )
    assert np.all(np.isfinite(weighted))
    assert abs(weighted[0] / weighted[1] - 1.0) < 1.0e-8
