# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Unit conventions for the phonon NEGF stack.

Internal unit system (matches the standalone reference at
``phonon/phonon_inputs/anharmonic.py``):

* Frequency axis is uniform in **THz**.
* Dynamical matrix and self-energies live in **THz^2**.
* FC3 is mass-weighted by ``CONVERSION_FC3_THZ`` so that
  ``Phi_{abc}`` is in **THz^2/sqrt(amu*A^2)** -- i.e. the bubble
  ``Phi G G Phi`` lands directly in THz^2.

This keeps intermediate magnitudes around unity (``THz^2 ~ O(1-100)``)
rather than the eV-natural ~1e-30 scale.

Thin helpers are exposed for tests / regression scripts that need
to bridge with code working in eV.
"""

from __future__ import annotations

import numpy as np

from qttools import NDArray, xp

HBAR_EV = 6.582119569e-16   # eV*s
EV_TO_J = 1.602176634e-19   # J/eV
HBAR_SI = HBAR_EV * EV_TO_J  # J*s
THZ_TO_RAD = 2.0 * np.pi * 1e12  # rad/s per THz


def thz_to_ev(omega_thz: NDArray) -> NDArray:
    """Convert a plain (rad-free) frequency in THz to hbar*omega in eV."""
    return xp.asarray(omega_thz) * (HBAR_EV * THZ_TO_RAD)


def bubble_prefactor_thz(dw_thz: float) -> complex:
    """Prefactor of the 3-phonon bubble in the THz^2 unit system.

    Matches ``phonon_inputs/anharmonic.py:_compute_phph_self_energy_finite``:
    ``0.5j * HBAR_SI * dw_thz / (2*pi)``. The bubble result is in **THz^2**.
    """
    return 0.5j * HBAR_SI * dw_thz / (2.0 * np.pi)
