# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Broad plus narrow: does the gate select the mode the method is for?

Audit Sec. 36. The claim the whole sector rests on is "treat selected poles
analytically and the rest normally". A bed with one broad, strongly coupled mode
and one narrow, weakly radiating mode coupled to it is the smallest system where
that claim can be wrong in both directions at once -- by promoting the broad mode
the grid already carries, or by leaving the narrow one on a grid that cannot
carry it.

The two modes interfere, so the narrow feature does not sit on a flat background
and is not a symmetric Lorentzian. That asymmetry is what distinguishes this bed
from two independent resonances, and it is asserted rather than assumed.

The operator is ``M(z) = z^2 I + i z G - D`` with a NON-scalar ``G``: a scalar
damping gives every mode the same width and there is no broad/narrow split to
find.
"""
import numpy as np
import pytest

from quatrex.core.config import PoleSectorConfig
from quatrex.phonon.pole_sector import PoleSector

G_DAMP = np.diag([1.6, 0.004])          # broad, narrow
LAM = np.array([64.0, 70.0])            # squared frequencies
COUPLING = 1.1
D = np.array([[LAM[0], COUPLING], [COUPLING, LAM[1]]])


def _m(z):
    return (z * z) * np.eye(2) + 1j * z * G_DAMP - D


def _poles():
    """Exact roots of ``det M(z)``, by companion linearisation of the pencil."""
    n = 2
    a = np.block([[np.zeros((n, n)), np.eye(n)], [D, -1j * G_DAMP]])
    b = np.block([[np.eye(n), np.zeros((n, n))],
                  [np.zeros((n, n)), np.eye(n)]])
    w = np.linalg.eigvals(np.linalg.solve(b, a))
    # Physical half only: every root has a negative-frequency mirror, and
    # picking the widest over both halves would compare a mode with its own
    # partner rather than with the other mode.
    return np.sort_complex(w[(w.imag < 0.0) & (w.real > 0.0)])


def _spectral(w):
    return np.array([-2.0 * np.imag(np.trace(np.linalg.inv(_m(x)))) for x in w])


def _broad_and_narrow():
    z = _poles()
    order = np.argsort(-z.imag)          # smallest width first
    return z[order[-1]], z[order[0]]     # broad, narrow


def test_the_bed_really_carries_two_very_different_widths():
    """Without this the rest of the file is testing one resonance twice."""
    broad, narrow = _broad_and_narrow()
    assert -broad.imag > 0.5
    assert -narrow.imag < 0.01
    assert (-broad.imag) / (-narrow.imag) > 100.0
    # ... and they are genuinely distinct modes, not a split of one.
    assert abs(broad.real - narrow.real) > 5.0 * (-narrow.imag)


def test_the_narrow_feature_is_asymmetric_which_a_lone_lorentzian_is_not():
    r"""Fano interference: the broad mode supplies a background whose phase
    slides through the narrow resonance, so the spectral function does not
    come back to the same level on the two sides."""
    _, narrow = _broad_and_narrow()
    gn = -narrow.imag
    asym = []
    for d in (30.0 * gn, 40.0 * gn, 60.0 * gn):
        left, right = _spectral(np.array([narrow.real - d, narrow.real + d]))
        asym.append(abs(left - right) / max(left, right))
    assert min(asym) > 0.25
    # It grows with distance, because the background phase keeps sliding while
    # the narrow line has already died away.
    assert asym[0] < asym[1] < asym[2]


def test_the_gate_promotes_the_narrow_mode_and_leaves_the_broad_one_on_the_grid():
    """The selection claim, on the bed built to break it."""
    broad, narrow = _broad_and_narrow()
    h = 0.05                                   # gamma_narrow << h << gamma_broad
    assert -narrow.imag < h < -broad.imag
    sec = PoleSector(PoleSectorConfig(), np.arange(0.0, 30.0 + h, h))

    assert sec.leg_weight_error(-broad.imag) < 1e-12
    assert sec.leg_weight_error(-narrow.imag) > 1.0


@pytest.mark.parametrize("h", [0.05, 0.1, 0.25])
def test_the_grid_misrepresents_the_narrow_line_by_the_predicted_factor(h):
    r"""The gate is not merely ordering the modes, it is quantitative."""
    _, narrow = _broad_and_narrow()
    gn, wn = -narrow.imag, narrow.real
    sec = PoleSector(PoleSectorConfig(), np.arange(0.0, 30.0 + h, h))

    # Put the line exactly on a node, where the worst case is attained.
    centre = round(wn / h) * h
    w = np.arange(-4000.0, 4000.0 + h, h) + centre
    represented = (h / np.pi) * np.sum(gn / ((w - centre) ** 2 + gn * gn))
    measured = abs(represented - 1.0)

    assert measured == pytest.approx(sec.leg_weight_error(gn), rel=1e-3)
    assert measured > 0.5, "the bed stopped being under-resolved"


def test_a_broad_mode_is_carried_to_better_than_a_part_per_billion():
    """The other half of the selection claim: nothing is gained by promoting a
    mode the grid already integrates correctly, and the sector must be able
    to say so rather than promoting everything and leaning on the solve."""
    broad, _ = _broad_and_narrow()
    gb = -broad.imag
    h = 0.05
    centre = round(broad.real / h) * h
    w = np.arange(-4000.0, 4000.0 + h, h) + centre
    represented = (h / np.pi) * np.sum(gb / ((w - centre) ** 2 + gb * gb))

    a, b = w[0] - 0.5 * h, w[-1] + 0.5 * h
    exact = (np.arctan((b - centre) / gb) - np.arctan((a - centre) / gb)) / np.pi
    assert abs(represented - exact) / exact < 1e-9
    assert abs(represented - 1.0) == pytest.approx(1.0 - exact, rel=1e-6)
