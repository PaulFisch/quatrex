"""FFT bubble kernel for the 3-phonon scattering self-energy.

The bubble integrand is

    Sigma^{x}_{IJ}(omega) = (i hbar / 2) * sum_{c,d,e,f}
        phi_left_{I,c,e}
        * G^{x}_{e,d}(omega) * G^{x}_{c,f}(omega)
        * phi_right_{J,d,f}
"""

from __future__ import annotations

import numpy as np


def precompute_g_fft(G, *, n_fft, zero_freq_idx=None, dc_handling="zero",
                     xp=None):
    """Return the zero-padded FFT of one G block.

    The dense reference driver calls bubble_dense many times per SCBA
    iteration with the same G(K,K') blocks; precomputing the FFT here
    lets callers amortise that cost across every bubble it appears in.

    The zero_freq_idx + dc_handling regularisation is applied here.

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


def _bubble_contract_chunk(phi_left, phi_right, G_a_fft, G_b_fft):
    """Three-matmul kernel over a single contiguous block of the
    frequency axis.
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
    phi_L_r = phi_left.reshape(nI * bK1, bK2)
    T1 = phi_L_r @ G_b_fft  # (w, a*c, d)
    T1 = T1.reshape(n_w, nI, bK1, bK2p)  # (w, a, c, d)

    # Step 2: T2[w, a, d, b] = sum_c T1[w, a, c, d] * G_a[w, c, b]
    T1_t = T1.transpose(0, 1, 3, 2)
    T1_t_r = T1_t.reshape(n_w, nI * bK2p, bK1)
    T2 = T1_t_r @ G_a_fft  # (w, a*d, b)
    T2 = T2.reshape(n_w, nI, bK2p, bK1p)  # (w, a, d, b)

    # Step 3: S[w, a, J] = sum_{b, d} T2[w, a, d, b] * phi_R[J, d, b]
    T2_r = T2.reshape(n_w, nI, bK2p * bK1p)
    phi_R_r = phi_right.reshape(nJ, bK2p * bK1p)
    return T2_r @ phi_R_r.T  # (w, a, J)


def bubble_chunk_peak_bytes_per_w(
    *, nI: int, nJ: int, bK1: int, bK1p: int, bK2: int, bK2p: int,
    itemsize: int = 16,
) -> int:
    """Transient bytes the matmul kernel holds per frequency sample.

    The chunked kernel processes w_chunk frequencies at a time;
    multiply this by the chunk length to bound a chunk's peak. The
    four (n_w, n_dof^3)-shaped intermediates dominate.
    """
    big = nI * max(bK1, bK2p) * max(bK1p, bK2p) * itemsize
    small = nI * nJ * itemsize
    return int(4 * big + 2 * small)


def _bubble_contract_batched_matmul(phi_left, phi_right,
                                    G_a_fft, G_b_fft, xp=np,
                                    max_bytes: int | None = None):
    """kernel for S[w,a,J] = phi_L[ace] * G_b[wed] * G_a[wcb] * phi_R[Jdb].

    max_bytes bounds the transient memory: the frequency axis is
    used as batch dimension, so when the full contraction would exceed max_bytes, the kernel slices
    the w axis into chunks that each fit.
    """
    nI = phi_left.shape[0]
    nJ = phi_right.shape[0]
    n_w = G_a_fft.shape[0]
    bK1 = G_a_fft.shape[1]
    bK1p = G_a_fft.shape[2]
    bK2 = G_b_fft.shape[1]
    bK2p = G_b_fft.shape[2]

    if max_bytes is None:
        return _bubble_contract_chunk(phi_left, phi_right, G_a_fft, G_b_fft)

    per_w = bubble_chunk_peak_bytes_per_w(
        nI=nI, nJ=nJ, bK1=bK1, bK1p=bK1p, bK2=bK2, bK2p=bK2p,
    )
    w_chunk = max(1, min(n_w, int(max_bytes // max(per_w, 1))))
    if w_chunk >= n_w:
        return _bubble_contract_chunk(phi_left, phi_right, G_a_fft, G_b_fft)

    S_hat = xp.empty((n_w, nI, nJ), dtype=complex)
    for w0 in range(0, n_w, w_chunk):
        w1 = min(w0 + w_chunk, n_w)
        S_hat[w0:w1] = _bubble_contract_chunk(
            phi_left, phi_right, G_a_fft[w0:w1], G_b_fft[w0:w1],
        )
    return S_hat


def _bubble_contract_task_batched(phi_left, phi_right, G_a_fft, G_b_fft):
    """``_bubble_contract_chunk`` with a leading TASK axis.

    Same three matmuls and the same index structure; the frequency axis stays
    the inner batch and the task axis is prepended, so the result must agree
    with looping ``_bubble_contract_chunk`` to roundoff (and does, exactly, on
    numpy -- see ``test_bubble_kernel``).

    This is not a CPU optimisation. Measured on a 16-cell chain it is 2.5x at
    ``d = 1``, where the per-task arrays are ``(n_fft, 1, 1)`` and the Python
    overhead is everything, and 0.68-0.90x at ``d = 4-6``, where numpy is
    already at its FLOP/memory bound. It exists because a GPU cannot use the
    per-task form at all: on a GH200 the per-task path runs at 0.09-1.3x of the
    CPU (it is 21000 launches of a few microseconds' work each), while this one
    runs at 13x, 5.5x, 72x, 111x for ``d = 1, 2, 4, 6``. Batching is the
    difference between a GPU that helps and one that does not.

    Shapes, with ``t`` the task axis:
    ``phi_left (t, nI, bK1, bK2)``, ``phi_right (t, nJ, bK2p, bK1p)``,
    ``G_a_fft (t, n_w, bK1, bK1p)``, ``G_b_fft (t, n_w, bK2, bK2p)``
    -> ``(t, n_w, nI, nJ)``.
    """
    nt, n_w = G_a_fft.shape[0], G_a_fft.shape[1]
    nI, bK1, bK2 = phi_left.shape[1], G_a_fft.shape[2], G_b_fft.shape[2]
    bK1p, bK2p, nJ = G_a_fft.shape[3], G_b_fft.shape[3], phi_right.shape[1]

    # Step 1: T1[t, w, a, c, d] = sum_e phi_L[t, a, c, e] G_b[t, w, e, d]
    t1 = phi_left.reshape(nt, 1, nI * bK1, bK2) @ G_b_fft
    t1 = t1.reshape(nt, n_w, nI, bK1, bK2p)

    # Step 2: T2[t, w, a, d, b] = sum_c T1[t, w, a, c, d] G_a[t, w, c, b]
    t2 = t1.transpose(0, 1, 2, 4, 3).reshape(nt, n_w, nI * bK2p, bK1) @ G_a_fft
    t2 = t2.reshape(nt, n_w, nI, bK2p, bK1p)

    # Step 3: S[t, w, a, J] = sum_{b, d} T2[t, w, a, d, b] phi_R[t, J, d, b]
    return (t2.reshape(nt, n_w, nI, bK2p * bK1p)
            @ phi_right.reshape(nt, 1, nJ, bK2p * bK1p).transpose(0, 1, 3, 2))


def bubble_task_batch_bytes(*, n_fft, nI, nJ, bK1, bK1p, bK2, bK2p,
                            itemsize: int = 16) -> int:
    """Transient bytes the task-batched kernel holds PER TASK.

    The two ``(n_fft, nI, d, d)`` intermediates dominate; the caller divides
    its budget by this to pick a task-chunk length, the way the per-task
    kernel divides by :func:`bubble_chunk_peak_bytes_per_w`.
    """
    big = n_fft * nI * max(bK1, bK2p) * max(bK1p, bK2p) * itemsize
    io = n_fft * (bK1 * bK1p + bK2 * bK2p + nI * nJ) * itemsize
    # FIVE d^3-sized transients live at once, not two: T1, the copy the
    # transpose+reshape forces before the second matmul, T2, and the two the
    # `prefactor * ifft(...)` expression makes. Counting two of them sized a
    # chunk that asked a GH200 for 24-40 GB in one allocation.
    return int(5 * big + 2 * io)


def bubble_dense_from_fft_task_batched(
    *,
    phi_left,
    phi_right,
    G_a_fft,
    G_b_fft,
    ne: int,
    prefactor,
    out_slice: slice | None = None,
    max_bytes: int | None = None,
    xp=None,
):
    """:func:`bubble_dense_from_fft` over a batch of tasks at once.

    Every input carries a leading task axis; the return is
    ``(n_task, len(out_slice), nI, nJ)``. ``max_bytes`` chunks the TASK axis
    rather than the frequency axis, because the point of the batch is to keep
    the frequency axis whole -- a batched FFT over ``(t, n_fft, nI, nJ)`` is
    one call where the per-task form is ``t`` of them.
    """
    if xp is None:
        xp = np
    if out_slice is None:
        out_slice = slice(0, ne)

    n_task = G_a_fft.shape[0]
    chunk = n_task
    if max_bytes is not None:
        per_task = bubble_task_batch_bytes(
            n_fft=G_a_fft.shape[1], nI=phi_left.shape[1],
            nJ=phi_right.shape[1], bK1=G_a_fft.shape[2],
            bK1p=G_a_fft.shape[3], bK2=G_b_fft.shape[2], bK2p=G_b_fft.shape[3],
        )
        chunk = max(1, min(n_task, int(max_bytes // max(per_task, 1))))

    n_out = len(range(*out_slice.indices(G_a_fft.shape[1])))
    out = xp.empty((n_task, n_out, phi_left.shape[1], phi_right.shape[1]),
                   dtype=complex)
    for t0 in range(0, n_task, chunk):
        t1 = min(t0 + chunk, n_task)
        s_hat = _bubble_contract_task_batched(
            phi_left[t0:t1], phi_right[t0:t1],
            G_a_fft[t0:t1], G_b_fft[t0:t1])
        spec = xp.fft.ifft(s_hat, axis=1)
        del s_hat
        out[t0:t1] = prefactor * spec[:, out_slice]
        del spec
        if xp is not np:
            # cupy caches freed blocks; without this the pool grows to the
            # high-water mark of every chunk at once and the next allocation
            # fails with memory that is free but not returned.
            xp.get_default_memory_pool().free_all_blocks()
    return out


def bubble_dense_from_fft(
    *,
    phi_left,
    phi_right,
    G_a_fft,
    G_b_fft,
    ne: int,
    prefactor,
    out_slice: slice | None = None,
    max_bytes: int | None = None,
    xp=None,
):
    """Bubble integrand from pre-FFT'd G blocks.

    Skips the input zero-pad + forward FFT (the caller is expected to
    have done that once via precompute_g_fft) and runs only
    the fused contraction + inverse FFT. Used by the multi-slab driver
    in :mod:`phonon.solver.se_finite` to amortise the FFT cost across
    every bubble that touches the same G block.

    ``max_bytes`` bounds the transient memory of the contraction by
    chunking the frequency axis (see
    :func:`_bubble_contract_batched_matmul`). ``None`` disables
    chunking.

    Returns ``prefactor * IFFT(S_hat)[out_slice]`` of shape
    ``(len(out_slice), nI, nJ)``.
    """
    if xp is None:
        xp = np
    if out_slice is None:
        out_slice = slice(0, ne)

    S_hat = _bubble_contract_batched_matmul(
        phi_left, phi_right, G_a_fft, G_b_fft, xp=xp,
        max_bytes=max_bytes,
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
        ``bK1' == bK1`` when ``G_a`` is a diagonal block;
         off-diagonal blocks with ``bK1' != bK1`` are supported
    G_b
        Green's function on the e-d link, shape ``(ne, bK2, bK2')``.
        ``bK2' == bK2`` for diagonal G blocks.
    n_fft
        FFT length along the omega axis.
    prefactor
        Multiplied into the output
    out_slice
        Slice applied to the IFFT result. Default ``slice(0, ne)``
    zero_freq_idx
        Location of the omega = 0 sample in the input grid. When this
        is ``None``, the input is treated as positive-only
        and ``dc_handling`` is a no-op. When not ``None``,
        the value of ``dc_handling`` decides how the DC
        sample of G is treated before the FFT -- see below.
    dc_handling
        How to regularise the omega = 0 sample of G before the FFT.
        Only consulted when ``zero_freq_idx`` is not ``None``.

          * ``"zero"`` sets ``G[zero_freq_idx] = 0`` to suppress the singular
            Bose occupation at omega = 0.
          * ``"interpolate"`` -- replaces ``G[zero_freq_idx]`` by the
            midpoint of its two omega neighbours.
          * ``"keep"`` -- leaves ``G[zero_freq_idx]`` untouched.
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
