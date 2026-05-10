# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Shared FFT helpers used by the bosonic scattering self-energies.

`hilbert_transform` is shared between `coulomb_screening.polarization` and
`phonon.sse_phonon_phonon` to reconstruct the retarded component from the
lesser/greater pair under the bosonic full-axis symmetry
``a(-omega) = a*(omega)`` (where ``a = P^> - P^<`` or
``Sigma^> - Sigma^<``).
"""

from qttools import NDArray, xp
from qttools.fft import fft_convolve


def hilbert_transform(a: NDArray, energies: NDArray) -> NDArray:
    r"""Hilbert transform along the leading energy axis.

    Assumes the bosonic symmetry of the input, i.e.
    :math:`[P^{\lessgtr}_{ij}(\omega)]^{\dagger} = -P^{\gtrless}_{ij}(-\omega)`,
    which becomes :math:`a(-\omega) = a^*(\omega)` for ``a = P^> - P^<``.

    Parameters
    ----------
    a : NDArray
        The array to transform. The first axis is the energy axis.
    energies : NDArray
        The energies corresponding to the first axis of ``a``.

    Returns
    -------
    NDArray
        The Hilbert transform of ``a`` along the energy axis.
    """
    de = energies[1] - energies[0]
    eta = de / 2  # singularity regulator (Cauchy principal value)
    energy_differences = (
        xp.expand_dims(energies - energies[0], tuple(range(1, a.ndim))) + eta
    )
    ne = energies.size

    hilbert_kernel = 1 / energy_differences
    b = fft_convolve(a, hilbert_kernel)[:ne]
    b += fft_convolve(a[::-1].conj(), hilbert_kernel)[-ne:]
    hilbert_kernel = -hilbert_kernel[::-1]
    b += fft_convolve(a, hilbert_kernel)[-ne:]

    return b
