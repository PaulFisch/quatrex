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

    # Fused 4-operand contraction. opt_einsum picks an optimal
    # pairwise contraction path; on d5a sizes this is ~40-100x faster
    # than three sequential np.einsum calls without ``optimize``.
    # ``opt_einsum.contract`` auto-detects the array backend (numpy,
    # cupy, etc.) from the operand types.
    import opt_einsum
    S_hat = opt_einsum.contract(
        "ace,Jdb,wcb,wed->waJ",
        phi_left, phi_right, Ga_fft, Gb_fft,
        optimize="optimal",
    )

    return prefactor * xp.fft.ifft(S_hat, axis=0)[out_slice]
