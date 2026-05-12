"""Retarded self-energy reconstruction from the Σ^{<,>} pair.

Three reconstruction methods are supported:

  * ``"half"`` — Σ^R = ½(Σ^> − Σ^<). Drops the Hilbert transform.
    Useful when only the broadening (imaginary part) is needed.
  * ``"pv"``   — singularity-subtracted principal-value integral.
    Captures both Im Σ^R and Re Σ^R; O(nfreq²) per matrix entry.
  * ``"fft"``  — FFT Hilbert transform along the frequency axis.
    O(nfreq log nfreq); preferred for large grids.

The factor multiplying the (n_B / n_B+1) sum in the eigenmode expansion
is ``-2j`` (no extra π); see ``document/src/theory.tex`` lines 873–874.
The Lorentzian broadening absorbs the ``1/π`` from Im G^R = -π Σ_n …
into ``L_η(ω - ω_n) = (η/π) / (ω² + η²)``, cancelling the explicit π.
"""

from __future__ import annotations

import numpy as np


def hilbert_transform_axis(f, axis=1):
    """Hilbert transform along *axis* via FFT (sign-multiplier method)."""
    N = f.shape[axis]
    Ff = np.fft.fft(f, axis=axis)

    h = np.zeros(N)
    if N % 2 == 0:
        h[1:N // 2] = 1
        h[N // 2 + 1:] = -1
    else:
        h[1:(N + 1) // 2] = 1
        h[(N + 1) // 2:] = -1

    shape = [1] * f.ndim
    shape[axis] = N
    return np.fft.ifft(Ff * (-1j * np.array(h).reshape(shape)), axis=axis)


def retarded_from_lesser_greater(delta, omega_grid_thz):
    """Build Σ^R from Δ = Σ^> − Σ^< via Kramers-Kronig.

    Σ^R(ω) = ½ Δ(ω) + (i/2π) PV∫ Δ(ω') / (ω − ω') dω'

    Singularity subtraction:

        PV∫ Δ(ω')/(ω−ω') dω'
          = ∫ [Δ(ω') − Δ(ω)]/(ω−ω') dω' + Δ(ω) · PV∫ 1/(ω−ω') dω'

    The first integral is regular and is evaluated with trapezoid
    quadrature, filling the diagonal sample with the finite-difference
    derivative −Δ'(ω). The second term uses the analytic PV integral

        PV∫_{ω_min}^{ω_max} dω'/(ω − ω') = ln|(ω − ω_min)/(ω − ω_max)|

    falling back to the discrete sum at the endpoints ω_min, ω_max where
    the analytic form diverges logarithmically.

    Parameters
    ----------
    delta : (n_freq, nd, nd) — Σ^> − Σ^<
    omega_grid_thz : (n_freq,)
    """
    n_freq = len(omega_grid_thz)
    sig_r = 0.5 * delta.astype(complex)
    dw = omega_grid_thz[1] - omega_grid_thz[0] if n_freq > 1 else 1.0
    w_min = omega_grid_thz[0]
    w_max = omega_grid_thz[-1]

    for i in range(n_freq):
        wi = omega_grid_thz[i]

        diff = wi - omega_grid_thz
        reg = np.empty_like(delta)
        nz = diff != 0
        reg[nz] = (delta[nz] - delta[i][None]) / diff[nz, None, None]

        if i == 0:
            deriv = (delta[1] - delta[0]) / dw
        elif i == n_freq - 1:
            deriv = (delta[-1] - delta[-2]) / dw
        else:
            deriv = (delta[i + 1] - delta[i - 1]) / (2 * dw)
        reg[i] = -deriv

        regular_integral = np.trapezoid(reg, omega_grid_thz, axis=0)

        eps_edge = 1e-12 * (w_max - w_min)
        if abs(wi - w_min) < eps_edge or abs(wi - w_max) < eps_edge:
            mask = nz
            pv_scalar = np.sum(1.0 / diff[mask]) * dw
        else:
            pv_scalar = np.log(abs((wi - w_min) / (wi - w_max)))

        pv_integral = regular_integral + delta[i] * pv_scalar
        sig_r[i] += 0.5j / np.pi * pv_integral

    return sig_r


def build_retarded(sig_l, sig_g, omega_grid_thz, method="pv"):
    """Build Σ^R from Σ^< and Σ^> using the specified method.

    Parameters
    ----------
    sig_l, sig_g : (..., n_freq, nd, nd)
        Lesser/greater self-energies. Leading dimensions are preserved.
    method : {"pv", "fft", "half"}
    """
    delta = (sig_g - sig_l).astype(complex)
    if method == "pv":
        leading = delta.shape[:-3]
        flat = delta.reshape(-1, *delta.shape[-3:])
        sig_r = np.empty(flat.shape, dtype=complex)
        for i in range(flat.shape[0]):
            sig_r[i] = retarded_from_lesser_greater(flat[i], omega_grid_thz)
        return sig_r.reshape(leading + delta.shape[-3:])
    if method == "fft":
        freq_axis = len(delta.shape) - 3
        return 0.5 * delta + 0.5j * hilbert_transform_axis(delta, axis=freq_axis)
    if method == "half":
        return 0.5 * delta
    raise ValueError(
        f"Unknown retarded method: {method!r}. Use 'pv', 'fft', or 'half'."
    )
