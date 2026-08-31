# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Pole-width sensitivity to the anharmonic channel -- audit Eq. (10).

Scaling one self-energy component by :math:`\lambda_j` moves the pole by

.. math::
    \frac{dz_\alpha}{d\lambda_j}
      = \frac{l^\dagger \Sigma_j^R(z_\alpha) r}{l^\dagger M'(z_\alpha) r},

and the denominator is 1 under the normalisation the solve already applies. The
imaginary part is how much of the pole's half width that channel accounts for,
which is the number that would separate a radiating mode from an anharmonically
broadened one.

Only the anharmonic channel is available. The operator holds the contacts at a
real anchor with no ``z`` dependence, so their derivative is identically zero by
construction -- see ``PoleSector.set_operator_context``.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from quatrex.core.config import PoleSectorConfig                    # noqa: E402
from quatrex.phonon.experimental.pole.pole_sector import PoleSector, _solve_batched   # noqa: E402
from test_pole_sector import _bed, _h, _sparse_indices              # noqa: E402


def _sector(scale=1.0, nf=401):
    r"""The ``_context_run`` bed with ``Delta`` scaled by ``lambda``."""
    freqs, d, delta, sizes = _bed(nf)
    d_ii, d_ij, d_ji = d
    sizes = list(sizes)
    blocks = {}
    for i in range(len(sizes)):
        blocks[(i, i)] = d_ii[i] + 0j
        if i + 1 < len(sizes):
            blocks[(i, i + 1)] = d_ij[i] + 0j
            blocks[(i + 1, i)] = d_ji[i] + 0j
    rows, cols = _sparse_indices(np.array(sizes))
    sec = PoleSector(PoleSectorConfig(enabled=True), freqs)
    sec.set_operator_context(
        delta=scale * delta[:, rows, cols], d_blocks=blocks,
        obc_left=None, obc_right=None,
        block_sizes=np.array(sizes), rows=rows, cols=cols,
    )
    return sec


def _solved(sec, seeded=None):
    seeded = sec._begin_refresh() if seeded is None else seeded
    return _solve_batched([sec], [seeded])[0], seeded


def test_sensitivity_is_the_first_derivative_of_the_pole_location():
    """Finite difference against a re-solve at the perturbed coupling."""
    sec = _sector()
    sols, seeded = _solved(sec)
    assert len(sols) > 3, "bed found too few poles to be a test"
    z0 = np.array([complex(s.z) for s in sols])
    sens = np.array(sec.sensitivities(sols))

    errors = []
    for eps in (1e-3, 1e-4):
        moved, _ = _solved(_sector(scale=1.0 + eps), seeded)
        dz = np.array([complex(s.z) for s in moved]) - z0
        pred = eps * sens
        errors.append(float(np.median(np.abs(dz - pred) / np.abs(pred))))

    assert errors[0] < 1e-5
    # First order: a decade smaller step, a decade smaller error.
    assert errors[1] < 0.25 * errors[0]


def test_the_anharmonic_channel_accounts_for_the_whole_width_without_contacts():
    r"""``gamma^sens = -Im dz/dlambda`` against ``gamma = -Im z``."""
    sec = _sector()
    sols, _ = _solved(sec)
    sens = sec.sensitivities(sols)
    for sol, ds in zip(sols, sens):
        assert -ds.imag == pytest.approx(sol.gamma_hwhm, rel=1e-3)


def test_sensitivity_is_empty_safe():
    assert _sector().sensitivities([]) == []


def test_sensitivity_uses_the_full_self_energy_not_its_change():
    """The predictor contracts against ``Delta - Delta_prev``; the sensitivity
    contracts against ``Delta``. They are the same kernel and must not be
    confused: with no previous iterate the predictor has nothing to say, while
    the sensitivity is defined from the first solve onward."""
    sec = _sector()
    sols, _ = _solved(sec)
    assert getattr(sec, "_prev_delta", None) is None
    z = np.array([complex(s.z) for s in sols])
    r = np.stack([np.asarray(_h(s.r)).reshape(-1) for s in sols])
    l = np.stack([np.asarray(_h(s.l)).reshape(-1) for s in sols])
    assert sec._predicted_shifts(z, l, r) is None
    assert np.all(np.abs(np.array(sec.sensitivities(sols))) > 0.0)
