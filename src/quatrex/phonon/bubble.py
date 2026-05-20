"""FFT bubble kernel for the 3-phonon scattering self-energy.

The bubble integrand is

    Sigma^{x}_{IJ}(omega) = (i hbar / 2) * sum_{c,d,e,f}
        phi_left_{I,c,e}
        * G^{x}_{e,d}(omega) * G^{x}_{c,f}(omega)
        * phi_right_{J,d,f}

evaluated as a frequency convolution via zero-padded FFTs along the
omega axis. The prefactor is supplied by the caller (``0.5j * hbar * d_omega / (2 * pi)``).
"""

from __future__ import annotations

import numpy as np


def bubble_dense(
    *,
    phi_left,
    phi_right,
    G_a,
    G_b,
    n_fft: int,
    prefactor,
    out_slice: slice | None = None,
    zero_freq_idx: int | None = None,
    xp=None,
):
    """FFT 3-phonon bubble for one (I, J) block.

    Parameters
    ----------
    phi_left
        Vertex tensor at the I-block, shape ``(nI, bK1, bK2)``,
        indices ``(a, c, e)``.
    phi_right
        Vertex tensor at the J-block, shape ``(nJ, bK2, bK1)``,
        indices ``(J, d, b)``.
    G_a
        Green's function on the c-f link, shape ``(ne, bK1, bK1')``.
        ``bK1' == bK1`` for diagonal blocks (the production sparse path
        always passes diagonal G); off-diagonal blocks with
        ``bK1' != bK1`` are supported by the cutoff-sweep audit on the
        dense side.
    G_b
        Green's function on the e-d link, shape ``(ne, bK2, bK2')``.
        ``bK2' == bK2`` for diagonal blocks. The dense reference passes
        ``G_b is G_a``.
    n_fft
        FFT length along the omega axis. Use ``2 * ne - 1`` for a
        non-aliased linear convolution.
    prefactor
        Multiplied into the output (encodes the SSE prefactor
        ``i hbar d_omega / (4 pi)`` in the caller's units).
    out_slice
        Slice applied to the IFFT result. Default ``slice(0, ne)``
        matches the production solver (positive-only frequency
        convention). The dense reference passes
        ``slice(ne // 2, ne // 2 + ne)`` for its symmetric grid.
    zero_freq_idx
        If not ``None``, zeroes G at this omega index before FFT.
        Used by the dense reference to suppress the singular Bose
        occupation at omega = 0. The production solver passes
        ``None`` and filters DC at the caller level if needed.
    xp
        Array module (``numpy`` or ``cupy``). Defaults to ``numpy``.

    Returns
    -------
    sigma_block : ``(len(out_slice), nI, nJ)`` complex array
        One of ``Sigma^<`` or ``Sigma^>`` for the (I, J) block pair.
    """
    if xp is None:
        xp = np

    # Sizes from the *G arrays*, not from phi_left. Off-diagonal G(K,K')
    # blocks have bK != bK' on the trailing axis; clipping the pad shape
    # to bK from phi_left would silently truncate those.
    ne, bK1, bK1p = G_a.shape
    _, bK2, bK2p = G_b.shape
    if out_slice is None:
        out_slice = slice(0, ne)

    if zero_freq_idx is not None:
        same = G_b is G_a
        G_a = G_a.copy()
        G_a[zero_freq_idx] = 0.0
        if same:
            G_b = G_a
        else:
            G_b = G_b.copy()
            G_b[zero_freq_idx] = 0.0

    Ga_pad = xp.zeros((n_fft, bK1, bK1p), dtype=complex)
    Ga_pad[:ne] = G_a
    Gb_pad = xp.zeros((n_fft, bK2, bK2p), dtype=complex)
    Gb_pad[:ne] = G_b

    Ga_fft = xp.fft.fft(Ga_pad, axis=0)
    Gb_fft = xp.fft.fft(Gb_pad, axis=0)

    # Fused 4-operand contraction via opt_einsum (auto-detects the
    # array backend from the operands). On d5a sizes this is roughly
    # 40-100x faster than the three sequential np.einsum calls without
    # ``optimize``.
    import opt_einsum
    S_hat = opt_einsum.contract(
        "ace,Jdb,wcb,wed->waJ",
        phi_left, phi_right, Ga_fft, Gb_fft,
        optimize="optimal",
    )

    return prefactor * xp.fft.ifft(S_hat, axis=0)[out_slice]
