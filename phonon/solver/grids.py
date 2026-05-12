"""Frequency grid and Bose distribution for the dense phonon solver.

These helpers establish the symmetric ω-grid used by every kernel under
:mod:`phonon.solver`:

  freqs_thz = [-fmax, ..., -Δω, 0, Δω, ..., fmax]

with grid spacing Δω = fmax / nfreq_pos. The ω = 0 sample sits at
``mid = nfreq_pos`` and is included for FFT index arithmetic; it is
excluded from every physical integral via ``pos_mask`` and zeroed out
of the bubble integrand at the caller level.
"""

from __future__ import annotations

import numpy as np

from phonon_inputs.constants import HBAR_SI, KB_SI, THZ_TO_RAD


def build_frequency_grid(freq_range_thz, eta_w_thz=None, eta_factor=0.05):
    """Build a uniform symmetric frequency grid for FFT convolution.

    Parameters
    ----------
    freq_range_thz : (fmin, fmax, nfreq_pos)
        ``fmin`` is advisory (Δω is set to ``fmax / nfreq_pos``).
        ``fmax`` is the upper edge of the positive axis. ``nfreq_pos`` is
        the number of positive-frequency bins (excluding 0).
    eta_w_thz, eta_factor
        Lorentzian half-width applied to ω² inside ``z2_arr``. If
        ``eta_w_thz`` is ``None`` it defaults to ``eta_factor * Δω``.

    Returns
    -------
    freqs_thz : (2*nfreq_pos + 1,) ndarray
    dw_thz : float
    eta_w_thz : float
    z2_arr : complex ndarray — ``(ω + i η_w)²``
    pos_mask : bool ndarray — ``True`` for ω > 0 (excludes 0)
    mid : int — index of the ω = 0 sample
    """
    _fmin, fmax, nfreq_pos = freq_range_thz
    nfreq_pos = int(nfreq_pos)
    if nfreq_pos < 2:
        raise ValueError("nfreq_pos must be >= 2")

    dw_thz = fmax / nfreq_pos
    freqs_pos = np.arange(1, nfreq_pos + 1) * dw_thz
    freqs_thz = np.concatenate((-freqs_pos[::-1], [0.0], freqs_pos))
    mid = nfreq_pos

    if eta_w_thz is None:
        eta_w_thz = eta_factor * dw_thz

    z2_arr = (freqs_thz + 1j * eta_w_thz) ** 2
    pos_mask = freqs_thz > 0.0
    return freqs_thz, dw_thz, eta_w_thz, z2_arr, pos_mask, mid


def bose_full_axis(freqs_thz, T):
    """Bose–Einstein occupation on the full symmetric axis.

    Returns 0 at ω = 0 as a placeholder; the ω = 0 sample is excluded
    from all physical integrals via ``pos_mask``.
    """
    x = HBAR_SI * freqs_thz * THZ_TO_RAD / (KB_SI * T)
    n = np.empty_like(x, dtype=float)

    zero = np.abs(freqs_thz) < 1e-30
    small = (~zero) & (np.abs(x) < 1e-6)
    regular = (~zero) & (~small)

    n[regular] = 1.0 / np.expm1(x[regular])
    n[small] = 1.0 / x[small] - 0.5 + x[small] / 12.0
    n[zero] = 0.0
    return n


def boson_contact_self_energies_from_gamma(Gamma, freqs_thz, T):
    """Build contact ``Σ^<``, ``Σ^>`` from broadening ``Γ``.

    Σ^< = −i n_B Γ,  Σ^> = −i (n_B + 1) Γ.

    The ω = 0 sample is handled via the ``n_B(0) = 0`` placeholder; it is
    never included in physical integrals.
    """
    n = bose_full_axis(freqs_thz, T)
    Sigma_l = -1j * n[:, None, None] * Gamma
    Sigma_g = -1j * (n[:, None, None] + 1.0) * Gamma
    return Sigma_l, Sigma_g
