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
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_RING_THREADS = max(1, int(os.environ.get("QUATREX_PHPH_RING_THREADS", "1")))
_RING_POOL = ThreadPoolExecutor(max_workers=_RING_THREADS) if _RING_THREADS > 1 else None
# Only worth the split + concatenate overhead when the batch is large enough.
_RING_MIN_W = int(os.environ.get("QUATREX_PHPH_RING_MIN_W", "48"))
# Thread-local T/U GEMM workspaces (config sse_ring_workspaces): recycles the
# two per-call temporaries, which contend in the allocator at wide pool widths.
_RING_WORKSPACES = os.environ.get("QUATREX_PHPH_RING_WORKSPACES", "0") == "1"
_TLS = threading.local()


def configure_ring_pool(threads: int = 0, min_w: int | None = None,
                        workspaces: bool | None = None) -> None:
    """Reconfigure the omega/tau thread pool; ``threads = 0`` leaves it as is.

    The per-w GEMMs are independent, so the result does not depend on the pool
    width. The QUATREX_PHPH_RING_* env vars supply the initial defaults.
    """
    global _RING_THREADS, _RING_POOL, _RING_MIN_W, _RING_WORKSPACES
    if threads > 0 and threads != _RING_THREADS:
        if _RING_POOL is not None:
            _RING_POOL.shutdown(wait=True)
        _RING_THREADS = int(threads)
        _RING_POOL = (
            ThreadPoolExecutor(max_workers=_RING_THREADS)
            if _RING_THREADS > 1 else None
        )
    if min_w is not None:
        _RING_MIN_W = int(min_w)
    if workspaces is not None:
        _RING_WORKSPACES = bool(workspaces)


def _workspace(key: str, shape: tuple, dtype) -> np.ndarray:
    """Per-thread reusable scratch of at least the requested size,
    viewed to the exact shape."""
    ws = getattr(_TLS, "ws", None)
    if ws is None:
        ws = _TLS.ws = {}
    size = int(np.prod(shape))
    buf = ws.get(key)
    if buf is None or buf.size < size or buf.dtype != dtype:
        buf = ws[key] = np.empty(size, dtype=dtype)
    return buf[:size].reshape(shape)


def ring_pool() -> tuple[ThreadPoolExecutor | None, int]:
    """Current (pool, width) -- read at call time, not import time."""
    return _RING_POOL, _RING_THREADS


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

    # Accept host arrays on any backend (asarray is a free view under
    # numpy); preserve the G_b-is-G_a aliasing the zero_freq path checks.
    same_input = G_b is G_a
    G_a = xp.asarray(G_a)
    G_b = G_a if same_input else xp.asarray(G_b)
    phi_left = xp.asarray(phi_left)
    phi_right = xp.asarray(phi_right)

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
    shaped (w, bK, bK') with w the leading (frequency/tau) batch axis. Returns
    ``S_hat`` shaped (w, nI, nJ).

    On the CPU backend with a pool configured, the w batch is split across the
    thread pool: the per-w matmuls are too small for BLAS threading, but the
    batch parallelises near-linearly.
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

    The permute and the three gemms live in :func:`phi_perms` /
    :func:`ring_contract_pre`; this is the one-shot (un-cached phi) wrapper.
    """
    PL, PR, nI, bK2, nJ = phi_perms(phi_left, phi_right, xp)
    return ring_contract_pre(PL, PR, nI, bK2, nJ, Ga_fft, Gb_fft, xp)


def phi_perms(phi_left, phi_right, xp):
    """Pre-permute the (fixed) phi factors for :func:`ring_contract_pre`.

    PL and PR are w-independent and the FC3 vertex is constant across SCBA
    iterations, so they are built once instead of per call. Returns
    ``(PL, PR, nI, bK2, nJ)``.
    """
    nI, bK1, bK2 = phi_left.shape
    nJ = phi_right.shape[0]
    bK2p, bK1p = phi_right.shape[1], phi_right.shape[2]
    PL = xp.ascontiguousarray(phi_left.transpose(0, 2, 1)).reshape(nI * bK2, bK1)
    PR = xp.ascontiguousarray(phi_right.transpose(1, 2, 0)).reshape(bK2p, bK1p * nJ)
    return PL, PR, nI, bK2, nJ


def ring_contract_pre(PL, PR, nI, bK2, nJ, Ga_fft, Gb_fft, xp):
    """Ring contraction with PRE-permuted phi factors (see :func:`phi_perms`).

    Same three gemms as :func:`_ring_contract_serial`, without the per-call
    transpose copies.
    """
    n_w = Ga_fft.shape[0]
    bK1p = Ga_fft.shape[2]
    if _RING_WORKSPACES and xp is np:
        dt = np.result_type(PL, Ga_fft, Gb_fft, PR)
        T = np.matmul(PL, Ga_fft,
                      out=_workspace("T", (n_w, PL.shape[0], bK1p), dt))
        U = np.matmul(Gb_fft, PR,
                      out=_workspace("U", (n_w, Gb_fft.shape[1], PR.shape[1]),
                                     dt))
    else:
        T = PL @ Ga_fft
        U = Gb_fft @ PR
    return T.reshape(n_w, nI, bK2 * bK1p) @ U.reshape(n_w, bK2 * bK1p, nJ)
