# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Tensor-decomposed (factored) three-phonon ring contraction.

With the folded device vertex factorised per leg (see
:mod:`quatrex.phonon.vertex_factors`), the dense ring collapses to a Hadamard
product of two skinny Grams sandwiched by the external leg. Two exact collapses
make it asymptotically optimal: the quad sum factorises, because the two lines
depend on disjoint index pairs and the quad set is their Cartesian product, so
the Grams are summed before the Hadamard; and the internal momentum sum is a
circular convolution, evaluated by FFT.

Convention: the row factor is conjugated, which is the conjugated-left-vertex
convention at factor level (``D`` and ``lam`` are real). No ``g = g^T``
assumption is made anywhere.
"""

from __future__ import annotations

from qttools import NDArray
from qttools.fft import fft_circular_convolve

# fold-term table: Sigma^< and Sigma^> as (a-line variant, b-line variant)
FOLD_L = (("l", "l"), ("l", "gr"), ("gr", "l"))
FOLD_G = (("g", "g"), ("g", "lr"), ("lr", "g"))
_VARIANTS = ("l", "g", "lr", "gr")


def gram_stack(u_row: NDArray, g_q: NDArray, u_col: NDArray, xp) -> NDArray:
    """P[iq, w, r, s] = conj(u_row[iq])^T @ g_q[w, iq] @ u_col[iq].

    u_row/u_col: (nq, b, R) (b') ; g_q: (w, nq, b, b')  ->  (nq, w, R, R).
    Batched skinny zgemms: (nq,1,R,b) @ (nq,w,b,b') @ (nq,1,b',R).
    """
    g = xp.swapaxes(g_q, 0, 1)                              # (nq, w, b, b')
    left = xp.conj(xp.swapaxes(u_row, -2, -1))[:, None]     # (nq, 1, R, b)
    right = u_col[:, None]                                  # (nq, 1, b', R)
    return (left @ g) @ right                               # (nq, w, R, R)


class GramTables:
    """Gram tables for ONE output pair (I, J).

    Key: (variant, (K, Kp), d_row, d_col, role) -> (nq, w, R, R). For INDSCAL
    (shared contracted legs) the role is dropped from the key.

    The key determines (I, J) -- ``I = K - d_row``, ``J = Kp - d_col`` -- so a
    table built for one pair is dead for every other pair. The cache is therefore
    scoped to a single pair; holding it across pairs would only grow the peak.
    """

    def __init__(self, g_dicts, UB, UC, off_pos, lo, hi, xp, shared_legs):
        self._g = g_dicts     # {"l": gl_q, "g": gg_q, "lr": glr_q, "gr": ggr_q}
        self._UB, self._UC = UB, UC     # (n_off, nq, b, R)
        self._pos = off_pos
        self._lo, self._hi = lo, hi
        self._xp = xp
        self._shared = shared_legs
        self._cache: dict = {}

    def get(self, variant, link, d_row, d_col, role) -> NDArray:
        key = (variant, link, d_row, d_col, "s" if self._shared else role)
        P = self._cache.get(key)
        if P is None:
            g_q = self._g[variant][link][self._lo:self._hi]   # (w, nq, b, b')
            if role == "a":   # rows: leg B (q'-leg of PhiL); cols: leg C
                u_row = self._UB[self._pos[d_row]]            # (nq, b, R)
                u_col = self._UC[self._pos[d_col]]
            else:             # b-line: rows leg C, cols leg B
                u_row = self._UC[self._pos[d_row]]
                u_col = self._UB[self._pos[d_col]]
            P = gram_stack(u_row, g_q, u_col, self._xp)
            self._cache[key] = P
        return P


def _convolve_q(a: NDArray, b: NDArray, nk_shape: tuple, xp) -> NDArray:
    """Circular convolution over the transverse mesh: sum_{q'} a[q'] * b[Q - q'].

    ``a``/``b`` are (nq, w, R, R) with the flat momentum index in C order
    (iq = ix*nky + iy), which is the native FFT ordering of the Gamma-centered
    mesh -- no shift or roll is needed. At the Gamma point (nq == 1) the
    convolution is the identity and the FFT is skipped.
    """
    if not nk_shape:
        return a * b

    shape = tuple(nk_shape) + a.shape[1:]
    axes = tuple(range(len(nk_shape)))
    out = fft_circular_convolve(a.reshape(shape), b.reshape(shape), axes=axes)

    return out.reshape(a.shape)


def contract_tau_q_factored(
    quads_by_pair: dict,
    block_sizes,
    nk_shape: tuple,
    q_lo: int,
    q_hi: int,
    nq: int,
    g_dicts: dict,
    Dt: NDArray,
    UB: NDArray,
    UC: NDArray,
    off_pos: dict,
    lo: int,
    hi: int,
    xp,
    shared_legs: bool,
    dtype,
) -> dict:
    """Sigma^{<,>}(I, J, q_ext) for the tau slice [lo:hi], factored kernel.

    Mirrors ``_contract_tau_q``'s contract: returns {(I, J): (out_l, out_g)}
    with out_* shaped (hi-lo, nq, bs_I, bs_J); entries outside [q_lo, q_hi) stay
    zero (summed over comm.q downstream).
    """
    nt = hi - lo
    R = Dt.shape[1]
    DtT = Dt.T.copy()
    res = {}

    for (I, J), quads in quads_by_pair.items():
        # The quad set is the product of the a-line and b-line link sets, so the
        # Grams are summed over each line separately (see the module docstring).
        a_links = sorted({(K1, K1p) for (K1, _K2, K1p, _K2p) in quads})
        b_links = sorted({(K2, K2p) for (_K1, K2, _K1p, K2p) in quads})
        actual = set(quads)
        product = {
            (K1, K2, K1p, K2p)
            for K1, K1p in a_links for K2, K2p in b_links
        }
        if actual != product:
            raise ValueError(
                "The factored Gram collapse requires Cartesian FC3 offset "
                f"support for output pair {(I, J)}; got {len(actual)} quads "
                f"but its independent link product contains {len(product)}. "
                "Use decomposed_kernel='reconstruct' or a Cartesian support "
                "mask."
            )

        grams = GramTables(g_dicts, UB, UC, off_pos, lo, hi, xp, shared_legs)
        sa, sb = {}, {}
        for variant in _VARIANTS:
            sa[variant] = sum(
                grams.get(variant, (K, Kp), K - I, Kp - J, role="a")
                for (K, Kp) in a_links
            )
            sb[variant] = sum(
                grams.get(variant, (K, Kp), K - I, Kp - J, role="b")
                for (K, Kp) in b_links
            )
        del grams

        # The 3-term bosonic fold, merged 6 -> 4 products on the shared a-line
        # factor. One Hadamard (one convolution) per term, not one per quad.
        h_lesser = _convolve_q(sa["l"], sb["l"] + sb["gr"], nk_shape, xp)
        h_lesser += _convolve_q(sa["gr"], sb["l"], nk_shape, xp)
        h_greater = _convolve_q(sa["g"], sb["g"] + sb["lr"], nk_shape, xp)
        h_greater += _convolve_q(sa["lr"], sb["g"], nk_shape, xp)

        bs_i = int(block_sizes[I])
        bs_j = int(block_sizes[J])
        out_l = xp.zeros((nt, nq, bs_i, bs_j), dtype=dtype)
        out_g = xp.zeros((nt, nq, bs_i, bs_j), dtype=dtype)
        for iq_ext in range(q_lo, q_hi):
            out_l[:, iq_ext] = (Dt @ h_lesser[iq_ext]) @ DtT
            out_g[:, iq_ext] = (Dt @ h_greater[iq_ext]) @ DtT

        res[(I, J)] = (out_l, out_g)

    return res
