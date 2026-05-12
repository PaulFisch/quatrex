"""Γ-only phonon-phonon self-energy driver.

Thin wrapper around the shared bubble kernel :func:`quatrex.phonon.bubble.bubble_dense`,
encoding the dense reference solver's convention:

  * symmetric ω-grid of length ``n_freq``,
  * FFT length ``n_fft = 2 n_freq - 1``,
  * ω = 0 sample zeroed before FFT (singular Bose at ω = 0),
  * output slice ``[mid : mid + n_freq]`` with ``mid = n_freq // 2``.
"""

from __future__ import annotations

import numpy as np

from phonon_inputs.constants import HBAR_SI
from quatrex.phonon.bubble import bubble_dense


def compute_phph_self_energy_finite(
    G_lesser, G_greater, Phi, omega_grid_thz, dw_thz,
):
    """Phonon-phonon self-energy for a finite device (no transverse q).

    Returns ``(Σ^<, Σ^>)`` only; Σ^R is rebuilt from the mixed pair in
    the SCBA loop via :func:`phonon.solver.retarded.build_retarded`.
    """
    n_freq = len(omega_grid_thz)
    n_fft = 2 * n_freq - 1
    mid = n_freq // 2
    freq_sl = slice(mid, mid + n_freq)
    prefactor = 0.5j * HBAR_SI * dw_thz / (2 * np.pi)

    sig_l = bubble_dense(
        phi_left=Phi, phi_right=Phi,
        G_a=G_lesser, G_b=G_lesser,
        n_fft=n_fft, prefactor=prefactor,
        out_slice=freq_sl, zero_freq_idx=mid,
    )
    sig_g = bubble_dense(
        phi_left=Phi, phi_right=Phi,
        G_a=G_greater, G_b=G_greater,
        n_fft=n_fft, prefactor=prefactor,
        out_slice=freq_sl, zero_freq_idx=mid,
    )
    return sig_l, sig_g
