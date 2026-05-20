"""Shared FFT bubble kernel for the 3-phonon scattering self-energy.

The bubble integrand is

    Sigma^{x}_{IJ}(omega) = (i hbar / 2) * sum_{c,d,e,f}
        phi_left_{I,c,e}
        * G^{x}_{e,d}(omega) * G^{x}_{c,f}(omega)
        * phi_right_{J,d,f}

evaluated as a frequency convolution via zero-padded FFTs along the
omega axis. The prefactor is supplied by the caller (typically
``0.5j * hbar * d_omega / (2 * pi)``).

This file holds the canonical implementation for the dense reference
solver (``phonon.solver.dense``) and any local script importing
``solver.bubble``. The installed production package keeps an
independent copy at :mod:`quatrex.phonon.bubble` so the block-sparse
SCBA driver does not depend on the local ``phonon/`` directory being on
``sys.path``. The two implementations are byte-identical; agreement is
locked by :mod:`tests.quatrex.phonon.test_bubble_kernel`.
"""

from __future__ import annotations

import numpy as np


def precompute_g_fft(G, *, n_fft, zero_freq_idx=None, dc_handling="zero",
                     xp=None):
    """Return the zero-padded FFT of one G block, ready for ``bubble_dense``.

    The dense reference and the multi-slab driver both call
    :func:`bubble_dense` many times per SCBA iteration with the same
    G(K,K') blocks; precomputing the FFT here lets callers amortise
    that cost across every bubble it appears in. The ``zero_freq_idx``
    + ``dc_handling`` regularisation is applied here (with the same
    semantics as inside ``bubble_dense``) so the returned tensor
    represents the actual G fed into the convolution.

    Returns a complex ``(n_fft, ne_K, ne_Kp)`` array.
    """
    if xp is None:
        xp = np
    ne, bK, bKp = G.shape
    if zero_freq_idx is not None and dc_handling != "keep":
        if dc_handling not in ("zero", "interpolate"):
            raise ValueError(
                f"Unknown dc_handling={dc_handling!r}. "
                "Use 'zero', 'interpolate', or 'keep'."
            )
        G = G.copy()
        if dc_handling == "zero":
            G[zero_freq_idx] = 0.0
        else:  # interpolate
            if zero_freq_idx <= 0 or zero_freq_idx >= ne - 1:
                raise ValueError(
                    f"dc_handling='interpolate' needs zero_freq_idx in "
                    f"(0, ne-1); got {zero_freq_idx} with ne={ne}."
                )
            G[zero_freq_idx] = 0.5 * (
                G[zero_freq_idx - 1] + G[zero_freq_idx + 1]
            )
    G_pad = xp.zeros((n_fft, bK, bKp), dtype=complex)
    G_pad[:ne] = G
    return xp.fft.fft(G_pad, axis=0)


def _bubble_contract_batched_matmul(phi_left, phi_right,
                                    G_a_fft, G_b_fft, xp=np):
    """Three-matmul kernel for ``S[w,a,J] = phi_L[ace] * G_b[wed] *
    G_a[wcb] * phi_R[Jdb]``.

    Profiling on the 4-operand fused ``opt_einsum.contract`` showed
    that the chosen path falls back to ``c_einsum`` (slow Python
    walker) for the contractions that share the ``w`` batch axis
    between ``G_a`` and ``G_b`` — ``tensordot`` cannot handle shared
    broadcast indices. The kernel below does the same arithmetic but
    routes everything through BLAS ``matmul`` (with explicit
    reshapes), which gives ~5-10x more speedup on d5a sizes.
    """
    nI = phi_left.shape[0]
    nJ = phi_right.shape[0]
    n_w = G_a_fft.shape[0]
    # G_a indices (w, c, b); shape (n_w, bK1, bK1p).
    bK1 = G_a_fft.shape[1]
    bK1p = G_a_fft.shape[2]
    # G_b indices (w, e, d); shape (n_w, bK2, bK2p).
    bK2 = G_b_fft.shape[1]
    bK2p = G_b_fft.shape[2]
    # phi_left indices (a, c, e); shape (nI, bK1, bK2).
    # phi_right indices (J, d, b); shape (nJ, bK2p, bK1p).

    # Step 1: T1[w, a, c, d] = sum_e phi_L[a, c, e] * G_b[w, e, d]
    #   reshape phi_L -> (a*c, e), batched matmul (1, a*c, e) @ (w, e, d).
    phi_L_r = phi_left.reshape(nI * bK1, bK2)
    T1 = phi_L_r @ G_b_fft  # (w, a*c, d)
    T1 = T1.reshape(n_w, nI, bK1, bK2p)  # (w, a, c, d)

    # Step 2: T2[w, a, d, b] = sum_c T1[w, a, c, d] * G_a[w, c, b]
    # Move c to inner axis of the LHS: (w, a, c, d) -> (w, a, d, c)
    T1_t = T1.transpose(0, 1, 3, 2)
    T1_t_r = T1_t.reshape(n_w, nI * bK2p, bK1)
    T2 = T1_t_r @ G_a_fft  # (w, a*d, b)
    T2 = T2.reshape(n_w, nI, bK2p, bK1p)  # (w, a, d, b)

    # Step 3: S[w, a, J] = sum_{b, d} T2[w, a, d, b] * phi_R[J, d, b]
    # Both LHS and RHS already in (d, b) order — flatten the trailing
    # pair directly, no transpose.
    T2_r = T2.reshape(n_w, nI, bK2p * bK1p)
    phi_R_r = phi_right.reshape(nJ, bK2p * bK1p)
    return T2_r @ phi_R_r.T  # (w, a, J)


def bubble_dense_from_fft(
    *,
    phi_left,
    phi_right,
    G_a_fft,
    G_b_fft,
    ne: int,
    prefactor,
    out_slice: slice | None = None,
    xp=None,
):
    """Bubble integrand from pre-FFT'd G blocks.

    Skips the input zero-pad + forward FFT (the caller is expected to
    have done that once via :func:`precompute_g_fft`) and runs only
    the fused contraction + inverse FFT. Used by the multi-slab driver
    in :mod:`phonon.solver.se_finite` to amortise the FFT cost across
    every bubble that touches the same G block.

    Returns ``prefactor * IFFT(S_hat)[out_slice]`` of shape
    ``(len(out_slice), nI, nJ)``.
    """
    if xp is None:
        xp = np
    if out_slice is None:
        out_slice = slice(0, ne)

    S_hat = _bubble_contract_batched_matmul(
        phi_left, phi_right, G_a_fft, G_b_fft, xp=xp,
    )
    return prefactor * xp.fft.ifft(S_hat, axis=0)[out_slice]


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
    dc_handling: str = "zero",
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
        ``bK1' == bK1`` when ``G_a`` is a diagonal block (the production
        sparse path and the dense reference both pass diagonal G blocks
        by default); off-diagonal blocks with ``bK1' != bK1`` are
        supported by the cutoff-sweep audit.
    G_b
        Green's function on the e-d link, shape ``(ne, bK2, bK2')``.
        ``bK2' == bK2`` for diagonal G blocks. The dense reference
        passes ``G_b is G_a``.
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
        Location of the omega = 0 sample in the input grid. When this
        is ``None``, the input is treated as positive-only (no DC
        sample to worry about) and ``dc_handling`` is a no-op. When
        not ``None``, the value of ``dc_handling`` decides how the DC
        sample of G is treated before the FFT — see below.
    dc_handling
        How to regularise the omega = 0 sample of G before the FFT.
        Only consulted when ``zero_freq_idx`` is not ``None``.

          * ``"zero"`` (default; legacy dense-reference behaviour) —
            sets ``G[zero_freq_idx] = 0`` to suppress the singular
            Bose occupation at omega = 0. Drops the genuine DC
            contribution from G as a side effect.
          * ``"interpolate"`` — replaces ``G[zero_freq_idx]`` by the
            midpoint of its two omega neighbours. Captures the linear
            behaviour of G near DC without amplifying the singular
            Bose factor (Bose-weighted lesser/greater G already vanish
            at omega = 0 in the broadened propagator).
          * ``"keep"`` — leaves ``G[zero_freq_idx]`` untouched. Closest
            to the strict SCBA expression; relies on finite ``eta``
            broadening to suppress the DC pole.
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

    if zero_freq_idx is not None and dc_handling != "keep":
        if dc_handling not in ("zero", "interpolate"):
            raise ValueError(
                f"Unknown dc_handling={dc_handling!r}. "
                "Use 'zero', 'interpolate', or 'keep'."
            )
        same = G_b is G_a
        G_a = G_a.copy()
        if dc_handling == "zero":
            G_a[zero_freq_idx] = 0.0
        else:  # interpolate
            if zero_freq_idx <= 0 or zero_freq_idx >= ne - 1:
                raise ValueError(
                    f"dc_handling='interpolate' needs zero_freq_idx in "
                    f"(0, ne-1); got {zero_freq_idx} with ne={ne}."
                )
            G_a[zero_freq_idx] = 0.5 * (
                G_a[zero_freq_idx - 1] + G_a[zero_freq_idx + 1]
            )
        if same:
            G_b = G_a
        else:
            G_b = G_b.copy()
            if dc_handling == "zero":
                G_b[zero_freq_idx] = 0.0
            else:
                G_b[zero_freq_idx] = 0.5 * (
                    G_b[zero_freq_idx - 1] + G_b[zero_freq_idx + 1]
                )

    Ga_pad = xp.zeros((n_fft, bK1, bK1p), dtype=complex)
    Ga_pad[:ne] = G_a
    Gb_pad = xp.zeros((n_fft, bK2, bK2p), dtype=complex)
    Gb_pad[:ne] = G_b

    Ga_fft = xp.fft.fft(Ga_pad, axis=0)
    Gb_fft = xp.fft.fft(Gb_pad, axis=0)

    S_hat = _bubble_contract_batched_matmul(
        phi_left, phi_right, Ga_fft, Gb_fft, xp=xp,
    )
    return prefactor * xp.fft.ifft(S_hat, axis=0)[out_slice]
