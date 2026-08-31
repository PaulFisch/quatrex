# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Where the simple-pole picture stops paying -- audit Sec. 37.

The threshold that decides whether a group of poles is carried as independent
simple poles or as one coherent cluster is 0.5 in
:math:`\chi = \gamma/\Delta^{\rm pole}`, and it has been a hard-coded number.
This sweeps a pair of modes through the transition and measures the two errors
that matter, so the number is calibrated rather than asserted:

``eps_pole``
    coherent cluster against the exact dense resolvent. Audit Sec. 14 claims
    that overlap does not remove the poles -- as long as the roots stay
    distinct, the representation is still exact. If that is right this stays
    flat across the whole sweep.

``eps_occ``
    independent scalar occupations against the same coherent cluster. This is
    what a simple-pole treatment actually throws away, and it is the quantity
    the threshold should be set by.

The bed is the closed-form pencil ``z^2 + i g z - lambda`` of
``test_pole_keldysh``, whose poles, vectors and residues are all analytic, so
neither error is contaminated by a solver.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from quatrex.phonon.experimental.pole.pole_keldysh import (                       # noqa: E402
    pole_keldysh, project_source,
)
from test_pole_keldysh import (                                 # noqa: E402
    _cluster_from_d, _direct_keldysh, _h, _sigma_lesser, _two_mode_d,
)

# delta -> chi = gamma / |z1 - z2|, spanning well-separated to deeply merged.
SPLITS = (4e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3)


def _measure(delta):
    d = _two_mode_d(delta)
    cl = _cluster_from_d(d)
    z = np.asarray(_h(cl.z))
    sep = float(abs(z[0] - z[1]))
    gamma = float(np.mean(np.asarray(_h(cl.gamma))))
    centre = float(np.asarray(_h(cl.omega))[0])

    w = np.linspace(centre - 10 * gamma, centre + 10 * gamma, 201)
    sig = _sigma_lesser(w, 2)
    src = np.asarray(_h(project_source(sig, _h(cl.v))))

    full = np.asarray(_h(pole_keldysh(w, cl, src)))
    diag = np.stack([np.diag(np.diag(s)) for s in src])
    scalar = np.asarray(_h(pole_keldysh(w, cl, diag)))
    exact = _direct_keldysh(w, d, sig)

    return {
        "chi": gamma / sep,
        "eps_pole": float(np.abs(full - exact).max() / np.abs(exact).max()),
        "eps_occ": float(np.abs(full - scalar).max() / np.abs(full).max()),
    }


@pytest.fixture(scope="module")
def sweep():
    return [_measure(d) for d in SPLITS]


def test_the_sweep_actually_crosses_the_transition(sweep):
    """Without this the calibration is being read off one side of the curve."""
    chi = [r["chi"] for r in sweep]
    assert chi == sorted(chi), "the sweep must be monotone in chi"
    assert min(chi) < 0.1 and max(chi) > 5.0


def test_overlap_does_not_break_the_pole_representation(sweep):
    r"""Audit Sec. 14: while the roots stay distinct, large :math:`\chi` does
    not mathematically remove the poles."""
    eps = [r["eps_pole"] for r in sweep]
    assert max(eps) < 2e-2
    assert max(eps) / min(eps) < 2.0, (
        "the cluster representation degraded with overlap, which would "
        "contradict the claim it is exact while the roots stay distinct")


def test_scalar_occupations_degrade_monotonically_and_then_saturate(sweep):
    """What overlap actually costs. The error rises through the transition and
    flattens once the lines are fully merged: past that point the modes are no
    longer separately meaningful and there is nothing further to lose."""
    rising = [r["eps_occ"] for r in sweep if r["chi"] < 1.5]
    flat = [r["eps_occ"] for r in sweep if r["chi"] >= 1.5]
    assert len(rising) >= 4 and len(flat) >= 2

    assert rising == sorted(rising), "scalar error must worsen as poles merge"
    assert rising[0] < 0.05                   # separated: scalar is fine
    assert rising[-1] > 0.2                   # nearly merged: scalar is not

    # Past saturation it is flat, not monotone: the residual wiggle is at the
    # 1e-4 level and asserting strict ordering there would pin noise.
    assert max(flat) - min(flat) < 1e-2
    assert min(flat) > 0.25


def test_the_hard_coded_half_is_where_the_scalar_picture_has_already_failed(
        sweep):
    r"""Calibration of the 0.5 in ``chi``."""
    below = [r["eps_occ"] for r in sweep if r["chi"] < 0.5]
    above = [r["eps_occ"] for r in sweep if r["chi"] >= 0.5]
    assert below and above

    assert max(below) < 0.2, (
        "below the threshold the scalar picture should still be usable")
    assert min(above) > 0.15, (
        "above the threshold it should already be costly")

    crossing = [r["chi"] for r in sweep if r["eps_occ"] > 0.1]
    assert 0.2 < min(crossing) < 1.0, (
        f"the 10 % crossing moved to chi={min(crossing):.3g}; the 0.5 "
        f"threshold is no longer calibrated to this bed")
