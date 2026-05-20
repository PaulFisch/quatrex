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

    # Three-matmul kernel — same arithmetic as the four-operand
    # einsum but routes everything through BLAS by reshaping
    # ``(a, c)`` and ``(a, d)`` so the shared ``w`` axis is a clean
    # batch dimension that batched matmul can handle. opt_einsum
    # falls back to slow ``c_einsum`` for the shared-w step.
    nI = phi_left.shape[0]
    nJ = phi_right.shape[0]
    n_w = Ga_fft.shape[0]

    phi_L_r = phi_left.reshape(nI * bK1, bK2)
    T1 = phi_L_r @ Gb_fft  # (w, a*c, d)
    T1 = T1.reshape(n_w, nI, bK1, bK2p)  # (w, a, c, d)

    T1_t = T1.transpose(0, 1, 3, 2)  # (w, a, d, c)
    T1_t_r = T1_t.reshape(n_w, nI * bK2p, bK1)
    T2 = T1_t_r @ Ga_fft  # (w, a*d, b)
    T2 = T2.reshape(n_w, nI, bK2p, bK1p)  # (w, a, d, b)

    T2_r = T2.reshape(n_w, nI, bK2p * bK1p)
    phi_R_r = phi_right.reshape(nJ, bK2p * bK1p)
    S_hat = T2_r @ phi_R_r.T  # (w, a, J)

    return prefactor * xp.fft.ifft(S_hat, axis=0)[out_slice]
