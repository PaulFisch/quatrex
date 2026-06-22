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

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_RING_THREADS = max(1, int(os.environ.get("QUATREX_PHPH_RING_THREADS", "1")))
_RING_POOL = ThreadPoolExecutor(max_workers=_RING_THREADS) if _RING_THREADS > 1 else None
# Only worth the split + concatenate overhead when the batch is large enough.
_RING_MIN_W = int(os.environ.get("QUATREX_PHPH_RING_MIN_W", "48"))


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

    S_hat = ring_contract(phi_left, phi_right, Ga_fft, Gb_fft, xp=xp)

    return prefactor * xp.fft.ifft(S_hat, axis=0)[out_slice]


def ring_contract(phi_left, phi_right, Ga_fft, Gb_fft, *, xp=None):
    """The per-frequency 3-phonon ring contraction.

    Operates on already-transformed Green's functions Ga_fft on the
    (c, b) = (K1, K1') link and Gb_fft on the (e, d) = (K2, K2') link, each
    shaped (w, bK, bK') with w the leading (frequency/τ) batch axis. Returns
    ``S_hat`` shaped (w, nI, nJ).

    When ``QUATREX_PHPH_RING_THREADS>1`` and on the CPU (numpy) backend, the
    embarrassingly-parallel w batch is split across a thread pool (single-thread
    BLAS per chunk) -- the per-w matmuls are too small for BLAS threading but the
    batch parallelises near-linearly. Bit-identical to the serial result.
    """
    if xp is None:
        xp = np

    n_w = Ga_fft.shape[0]
    if _RING_POOL is not None and xp is np and n_w >= _RING_MIN_W:
        nt = min(_RING_THREADS, n_w)
        bnds = [(i * n_w // nt, (i + 1) * n_w // nt) for i in range(nt)]
        parts = list(_RING_POOL.map(
            lambda b: _ring_contract_serial(
                phi_left, phi_right, Ga_fft[b[0]:b[1]], Gb_fft[b[0]:b[1]], xp),
            bnds))
        return xp.concatenate(parts, axis=0)
    return _ring_contract_serial(phi_left, phi_right, Ga_fft, Gb_fft, xp)


def _ring_contract_serial(phi_left, phi_right, Ga_fft, Gb_fft, xp):
    """Serial ring contraction over the full w batch (see ``ring_contract``).

    Transpose-free evaluation of
        S[w,a,J] = phi_L[a,c,e] Ga[w,c,b] Gb[w,e,d] phi_R[J,d,b]
    via  T[w,(a,e),b] = PL[(a,e),c] @ Ga[w,c,b]      (PL = phi_L perm (a,e,c))
         U[w,e,(b,J)] = Gb[w,e,d] @ PR[d,(b,J)]      (PR = phi_R perm (d,b,J))
         S[w,a,J]     = T[w,a,(e,b)] @ U[w,(e,b),J]
    """
    nI, bK1, bK2 = phi_left.shape
    nJ = phi_right.shape[0]
    n_w = Ga_fft.shape[0]
    bK1p = Ga_fft.shape[2]
    bK2p = Gb_fft.shape[2]

    # (a,c,e) -> [(a,e), c] and (J,d,b) -> [d, (b,J)]; w-independent.
    PL = xp.ascontiguousarray(phi_left.transpose(0, 2, 1)).reshape(nI * bK2, bK1)
    PR = xp.ascontiguousarray(phi_right.transpose(1, 2, 0)).reshape(bK2p, bK1p * nJ)

    T = PL @ Ga_fft            # (w, nI*bK2, bK1p)  == (w, a, e, b)
    U = Gb_fft @ PR            # (w, bK2,  bK1p*nJ) == (w, e, b, J)
    return T.reshape(n_w, nI, bK2 * bK1p) @ U.reshape(n_w, bK2 * bK1p, nJ)
