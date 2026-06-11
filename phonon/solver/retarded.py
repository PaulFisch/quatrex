"""Retarded self-energy reconstruction from the Sigma^{<,>} pair.

Three reconstruction methods are supported:

  * ``"half"`` -- Sigma^R = 1/2(Sigma^> - Sigma^<). Drops the Hilbert transform.
    Useful when only the broadening (imaginary part) is needed.
  * ``"pv"``   -- singularity-subtracted principal-value integral.
    Captures both Im Sigma^R and Re Sigma^R; O(nfreq^2) per matrix entry.
  * ``"fft"``  -- zero-padded FFT Hilbert transform along the frequency
    axis. O(nfreq log nfreq); preferred for large grids. The padding is
    essential: the bare sign-multiplier FFT computes the *periodic*
    Hilbert transform, which carries a resolution-independent ~1 %
    error for the finite-support Sigma^{<,>} fed here (the periodic wrap
    injects a spurious discontinuity). See :func:`hilbert_transform_axis`.

The factor multiplying the (n_B / n_B+1) sum in the eigenmode expansion
is ``-2j`` (no extra pi); see ``document/src/theory.tex`` lines 873-874.
The Lorentzian broadening absorbs the ``1/pi`` from Im G^R = -pi Sigma_n ...
into ``L_eta(omega - omega_n) = (eta/pi) / (omega^2 + eta^2)``, cancelling the explicit pi.
"""

from __future__ import annotations

import numpy as np


def hilbert_transform_axis(f, axis=1, pad_factor=8):
    """Hilbert transform along *axis* via the zero-padded FFT.

    The bare sign-multiplier FFT computes the *circular* (periodic)
    Hilbert transform. For a finite-support signal -- such as the
    self-energy difference Sigma^> - Sigma^< -- the periodic wrap injects a
    spurious discontinuity at the grid boundary and the result carries
    a ~1 % error that does **not** shrink as the grid is refined.

    Zero-padding the signal by ``pad_factor`` before the FFT turns the
    circular convolution into a (truncated) linear convolution over the
    original window, so the result converges to the aperiodic
    (infinite-range) Hilbert transform. ``pad_factor=8`` brings the
    interior error to ~1e-5, matching the O(nfreq^2) ``"pv"`` reference;
    ``pad_factor=1`` recovers the legacy periodic behaviour.

    Parameters
    ----------
    f : ndarray
        Signal; the transform is taken along ``axis``.
    axis : int
        Frequency axis.
    pad_factor : int
        Zero-pad the axis to ``pad_factor`` times its length before the
        FFT, then crop back. Must be >= 1.
    """
    if pad_factor < 1:
        raise ValueError(f"pad_factor must be >= 1, got {pad_factor}")
    N = f.shape[axis]
    if pad_factor > 1:
        pad_width = [(0, 0)] * f.ndim
        pad_width[axis] = (0, (pad_factor - 1) * N)
        f_work = np.pad(f, pad_width)
    else:
        f_work = f
    M = f_work.shape[axis]
    Ff = np.fft.fft(f_work, axis=axis)

    h = np.zeros(M)
    if M % 2 == 0:
        h[1:M // 2] = 1
        h[M // 2 + 1:] = -1
    else:
        h[1:(M + 1) // 2] = 1
        h[(M + 1) // 2:] = -1

    shape = [1] * f.ndim
    shape[axis] = M
    out = np.fft.ifft(Ff * (-1j * np.array(h).reshape(shape)), axis=axis)

    sl = [slice(None)] * f.ndim
    sl[axis] = slice(0, N)
    return out[tuple(sl)]


def retarded_from_lesser_greater(delta, omega_grid_thz):
    """Build Sigma^R from Delta = Sigma^> - Sigma^< via Kramers-Kronig.

    Sigma^R(omega) = 1/2 Delta(omega) + (i/2pi) PVintegral Delta(omega') / (omega - omega') domega'

    Singularity subtraction:

        PVintegral Delta(omega')/(omega-omega') domega'
          = integral [Delta(omega') - Delta(omega)]/(omega-omega') domega' + Delta(omega) . PVintegral 1/(omega-omega') domega'

    The first integral is regular and is evaluated with trapezoid
    quadrature, filling the diagonal sample with the finite-difference
    derivative -Delta'(omega). The second term uses the analytic PV integral

        PVintegral_{omega_min}^{omega_max} domega'/(omega - omega') = ln|(omega - omega_min)/(omega - omega_max)|

    falling back to the discrete sum at the endpoints omega_min, omega_max where
    the analytic form diverges logarithmically.

    Parameters
    ----------
    delta : (n_freq, nd, nd) -- Sigma^> - Sigma^<
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


def hilbert_transform_bosonic(delta, omega_grid_thz, axis):
    """(1/pi) PV transform of the BOSONICALLY CONTINUED delta.

    The positive-only grid is continued to negative frequencies via
    ``delta(-w) = delta(w)*`` (exact for ``Sigma^> - Sigma^<``) before the
    principal-value integral -- the window-only transform misses the
    negative-frequency image entirely. Discretisation uses exact
    cell-integrated log PV weights (the pole cell contributes zero), the
    same scheme as the (audited) production
    ``quatrex.core.fft_utils.hilbert_transform``.
    """
    w = np.asarray(omega_grid_thz, dtype=float)
    ne = w.size
    de = float(w[1] - w[0])
    w0 = float(w[0])

    d = np.moveaxis(np.asarray(delta, dtype=complex), axis, 0)
    tail_shape = d.shape[1:]
    d2 = d.reshape(ne, -1)

    j = np.arange(-(ne - 1), ne, dtype=float)
    absj = np.abs(j)
    safe = np.where(absj > 0, absj, 1.0)
    pos_k = np.where(absj > 0, np.log((safe + 0.5) / (safe - 0.5)) * np.sign(j), 0.0)

    m = np.arange(0, 2 * ne - 1, dtype=float)
    num = 2.0 * w0 + m * de + 0.5 * de
    den = 2.0 * w0 + m * de - 0.5 * de
    mir_k = np.where(den > 0,
                     np.log(np.where(den > 0, num, 1.0)
                            / np.where(den > 0, den, 1.0)), 0.0)

    n_conv = ne + (2 * ne - 1)
    K = np.fft.fft(pos_k, n_conv)
    M = np.fft.fft(mir_k, n_conv)
    D = np.fft.fft(d2, n_conv, axis=0)
    out = np.fft.ifft(D * K[:, None], axis=0)[ne - 1: 2 * ne - 1]

    dm = d2[::-1].conj()
    if abs(w0) < 0.25 * de:
        dm = dm.copy()
        dm[-1] = 0.0  # the w=0 cell is already covered by the main kernel
    Dm = np.fft.fft(dm, n_conv, axis=0)
    out = out + np.fft.ifft(Dm * M[:, None], axis=0)[ne - 1: 2 * ne - 1]

    out = (out / np.pi).reshape((ne,) + tail_shape)
    return np.moveaxis(out, 0, axis)


def build_retarded(sig_l, sig_g, omega_grid_thz, method="pv"):
    """Build Sigma^R from Sigma^< and Sigma^> using the specified method.

    Parameters
    ----------
    sig_l, sig_g : (..., n_freq, nd, nd)
        Lesser/greater self-energies. Leading dimensions are preserved.
    method : {"pv", "fft", "half"}
        "fft" includes the bosonic negative-frequency mirror (2026-06-10
        audit fix); "pv" remains the window-only O(N^2) reference.
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
        return 0.5 * delta + 0.5j * hilbert_transform_bosonic(
            delta, omega_grid_thz, axis=freq_axis)
    if method == "half":
        return 0.5 * delta
    raise ValueError(
        f"Unknown retarded method: {method!r}. Use 'pv', 'fft', or 'half'."
    )
