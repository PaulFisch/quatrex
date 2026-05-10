# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Unit conventions for the phonon NEGF stack.

Internal unit system (matches the standalone reference at
``phonon/phonon_inputs/anharmonic.py``):

* Frequency axis is uniform in **THz**.
* Dynamical matrix and self-energies live in **THz²**.
* FC3 is mass-weighted by ``CONVERSION_FC3_THZ`` so that
  ``Phi_{abc}`` is in **THz²/sqrt(amu·Å²)** — i.e. the bubble
  ``Phi · G · G · Phi`` lands directly in THz².

This keeps intermediate magnitudes around unity (``THz² ~ O(1-100)``)
instead of the eV-natural ``~10⁻³⁰`` that was burying finite-difference
signal in floating-point noise.

Two thin helpers are exposed for tests / regression scripts that need
to bridge with code working in eV.
"""

from __future__ import annotations

import numpy as np

from qttools import NDArray

# Re-export the SI / phonopy constants used by the standalone bubble so
# downstream callers don't have to reach into ``phonon_inputs``.
HBAR_EV = 6.582119569e-16   # eV·s
EV_TO_J = 1.602176634e-19   # J/eV
HBAR_SI = HBAR_EV * EV_TO_J  # J·s
THZ_TO_RAD = 2.0 * np.pi * 1e12  # rad/s per THz


def ev_to_thz(omega_ev: NDArray) -> NDArray:
    """Convert ω in eV (i.e. ℏω) to ω in THz (rad-free, plain frequency)."""
    return np.asarray(omega_ev) / (HBAR_EV * THZ_TO_RAD)


def thz_to_ev(omega_thz: NDArray) -> NDArray:
    """Inverse of :func:`ev_to_thz`."""
    return np.asarray(omega_thz) * (HBAR_EV * THZ_TO_RAD)


def ev2_to_thz2(value_ev2: NDArray) -> NDArray:
    """Convert ω² in eV² to ω² in THz² (used for D, Σ)."""
    return np.asarray(value_ev2) / (HBAR_EV * THZ_TO_RAD) ** 2


def thz2_to_ev2(value_thz2: NDArray) -> NDArray:
    """Inverse of :func:`ev2_to_thz2`."""
    return np.asarray(value_thz2) * (HBAR_EV * THZ_TO_RAD) ** 2


def bubble_prefactor_thz(dw_thz: float) -> complex:
    """Prefactor of the 3-phonon bubble in the THz² unit system.

    Matches ``phonon_inputs/anharmonic.py:_compute_phph_self_energy_finite``
    (line 659): ``0.5j * HBAR_SI * dw_thz / (2π)``. The bubble result is
    in **THz²**.
    """
    return 0.5j * HBAR_SI * dw_thz / (2.0 * np.pi)
