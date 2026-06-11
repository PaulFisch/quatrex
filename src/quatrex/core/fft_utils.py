# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Shared FFT helpers used by the bosonic scattering self-energies.

`hilbert_transform` is shared between `coulomb_screening.polarization` and
`phonon.sse_phonon_phonon` to reconstruct the retarded component from the
lesser/greater pair under the bosonic full-axis symmetry
``a(-omega) = a*(omega)`` (where ``a = P^> - P^<`` or
``Sigma^> - Sigma^<``).
"""

import math

from qttools import NDArray, xp
from qttools.fft import fft_convolve


def hilbert_transform(a: NDArray, energies: NDArray) -> NDArray:
    r"""Standard Hilbert transform along the leading energy axis.

    Returns

    .. math::
        H[a](\omega_n) = \frac{1}{\pi}\,\mathrm{PV}\!\!
        \int_{-\omega_{max}}^{\omega_{max}}
        \frac{a_{full}(\omega')}{\omega_n - \omega'}\,d\omega',

    where the integrand is continued to negative frequencies via the
    bosonic symmetry :math:`a(-\omega) = a^*(\omega)` (which holds for
    ``a = P^> - P^<`` or ``Sigma^> - Sigma^<`` because
    :math:`[P^{\lessgtr}_{ij}(\omega)]^{\dagger} = -P^{\gtrless}_{ij}(-\omega)`).

    The retarded component then follows as
    ``X^R = 0.5*(X^> - X^<) + 0.5j * hilbert_transform(X^> - X^<)``,
    since :math:`X^R(\omega) = \frac{i}{2\pi}\int
    \frac{\Delta(\omega')}{\omega-\omega'+i0}\,d\omega'
    = \tfrac12\Delta(\omega) + \tfrac{i}{2} H[\Delta](\omega)`.

    The principal value is discretized with *exact cell-integrated*
    weights: each grid cell :math:`[\omega_k - d\omega/2,
    \omega_k + d\omega/2]` contributes
    :math:`\ln\lvert(\omega_n-\omega_k+d\omega/2) /
    (\omega_n-\omega_k-d\omega/2)\rvert`, and the cell containing the
    pole contributes zero (its PV vanishes against a locally constant
    integrand). This is the dimensionally complete transform: callers
    must NOT multiply by any additional ``d\omega`` quadrature weight.

    Parameters
    ----------
    a : NDArray
        The array to transform, sampled on the positive-frequency grid.
        The first axis is the energy axis.
    energies : NDArray
        The (uniform, ascending) energies corresponding to the first
        axis of ``a``.

    Returns
    -------
    NDArray
        The Hilbert transform of the bosonically continued ``a`` along
        the energy axis, on the same positive-frequency grid.
    """
    ne = int(energies.shape[0])
    de = float(energies[1] - energies[0])
    w0 = float(energies[0])

    def _expand(kernel: NDArray) -> NDArray:
        return xp.expand_dims(kernel, tuple(range(1, a.ndim)))

    # --- positive-frequency cells: antisymmetric log PV kernel ----------
    # K(j) = ln((|j|+1/2)/(|j|-1/2)) * sign(j) for j != 0, K(0) = 0.
    j = xp.arange(-(ne - 1), ne, dtype=xp.float64)
    absj = xp.abs(j)
    safe = xp.where(absj > 0, absj, 1.0)
    pos_kernel = xp.where(
        absj > 0, xp.log((safe + 0.5) / (safe - 0.5)) * xp.sign(j), 0.0
    )
    # conv(a, K)[n + ne - 1] = sum_k a_k K(n - k)
    b = fft_convolve(a, _expand(pos_kernel))[ne - 1 : 2 * ne - 1]

    # --- negative-frequency cells (bosonic mirror) -----------------------
    # Cell centered at -omega_k: denominator omega_n + omega_k
    # = 2*w0 + (n+k)*de.  Exact cell integral
    # ln((2*w0 + m*de + de/2)/(2*w0 + m*de - de/2)) with m = n + k; the PV
    # of a pole-straddling cell (possible only at m = 0 for w0 ~ 0) is 0.
    m = xp.arange(0, 2 * ne - 1, dtype=xp.float64)
    num = 2.0 * w0 + m * de + 0.5 * de
    den = 2.0 * w0 + m * de - 0.5 * de
    mir_kernel = xp.where(den > 0, xp.log(xp.where(den > 0, num, 1.0) /
                                          xp.where(den > 0, den, 1.0)), 0.0)
    a_mirror = a[::-1].conj()
    if abs(w0) < 0.25 * de:
        # The omega=0 sample sits on both the positive grid and its mirror;
        # count its cell once (it is already covered by the kernel above).
        a_mirror = a_mirror.copy()
        a_mirror[-1] = 0.0
    # conv(a_mirror, M)[n + ne - 1] = sum_k a*_k M(n + k)
    b += fft_convolve(a_mirror, _expand(mir_kernel))[ne - 1 : 2 * ne - 1]

    return b / math.pi
