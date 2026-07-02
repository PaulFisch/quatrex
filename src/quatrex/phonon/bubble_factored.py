# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Tensor-decomposed (factored) coupled-q three-phonon ring contraction.

Replaces the dense per-(q', q2) vertex-pair triple-GEMM loop of
``SigmaPhononPhonon._contract_tau_q`` by the exact factored form. With the
folded device vertex factorised per leg (see ``quatrex.phonon.vertex_factors``),

    Phi~(q1,q2)[(I,K,K')][a,b,c] = sum_r lam_r D[a,r] UB[K-I][q1][b,r]
                                                     UC[K'-I][q2][c,r],

the dense ring  S[w,a,j] = conj(PhiL)[a,c,e] Ga[w,c,b] Gb[w,e,d] PhiR[j,d,b]
(with PhiL = Phi~(q',q2)[(I,K1,K2)], PhiR = Phi~(q2,q')[(J,K2p,K1p)] and the
audited LEFT-vertex conjugation, sse_phonon_phonon.py:906-930) collapses
exactly to

    S[w] = Dt @ ( P_a[w] * P_b[w] ) @ Dt.T,        Dt = D · diag(lam)

    P_a[w,r,s] = conj(UB[K1-I][q'])^T · Ga[w,q'] · UC[K1p-J][q']     (a-line)
    P_b[w,r,s] = conj(UC[K2-I][q2])^T · Gb[w,q2] · UB[K2p-J][q2]     (b-line)

i.e. one skinny Gram per (g-variant, band link, row/col transport offsets,
momentum) instead of one BS^3 triple-GEMM per (vertex pair, quad). The row
factor is conjugated -- that IS the conjugated-left-vertex convention at
factor level (D and lam are real). The 3+3 absorption-fold terms combine the
same gl/gg/glr/ggr buffers as the dense path; the bosonic fold, DC-zeroing
and masks are inherited upstream and NOT re-derived here.

The q'-convolution H[iq_ext] = sum_iqp P_a[iqp] * P_b[q_diff_map[iq_ext,iqp]]
is evaluated as a vectorised gather + reduction over stacked-in-q Gram tables
(the explicit-sum reference path; a per-(r,s) transverse FFT is a future hook
for nk >= 13).

For the S2-symmetric INDSCAL ansatz UB is UC, so the a/b-line leg roles
coincide and one Gram table serves both lines. For the CP fallback the roles
differ and tables are keyed by role. No ``g = g^T`` assumption is made
anywhere.
"""

from __future__ import annotations

import numpy as np

from qttools import NDArray

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
    """Per-tau-chunk cache of q-stacked Gram tables.

    Key: (variant, (K, Kp), d_row, d_col, role) -> (nq, w, R, R).
    For INDSCAL (shared contracted legs) the role is dropped from the key.
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


def contract_tau_q_factored(
    quads_by_pair: dict,
    block_sizes,
    q_diff_map: NDArray,
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

    Mirrors ``_contract_tau_q``'s contract: returns
    {(I, J): (out_l, out_g)} with out_* shaped (hi-lo, nq, bs_I, bs_J);
    entries outside [q_lo, q_hi) stay zero (summed over comm.q downstream).
    """
    grams = GramTables(g_dicts, UB, UC, off_pos, lo, hi, xp, shared_legs)
    nt = hi - lo
    R = Dt.shape[1]
    DtT = Dt.T.copy()
    res = {}
    for (I, J), quads in quads_by_pair.items():
        bs_I = int(block_sizes[I])
        bs_J = int(block_sizes[J])
        out_l = xp.zeros((nt, nq, bs_I, bs_J), dtype=dtype)
        out_g = xp.zeros((nt, nq, bs_I, bs_J), dtype=dtype)
        Hl = xp.zeros((q_hi - q_lo, nt, R, R), dtype=dtype)
        Hg = xp.zeros((q_hi - q_lo, nt, R, R), dtype=dtype)
        for (K1, K2, K1p, K2p) in quads:
            a_key = ((K1, K1p), K1 - I, K1p - J)
            b_key = ((K2, K2p), K2 - I, K2p - J)
            Pa = {v: grams.get(v, *a_key, role="a") for v in _VARIANTS}
            Pb = {v: grams.get(v, *b_key, role="b") for v in _VARIANTS}
            # Fold terms combined by shared a-line factor (6 -> 4 products):
            #   Hl = Pa^l  o (Pb^l + Pb^gr) + Pa^gr o Pb^l
            #   Hg = Pa^g  o (Pb^g + Pb^lr) + Pa^lr o Pb^g
            # The b-line sums are quad-local; the gather (fancy index over
            # the q axis) is the traffic bottleneck, so fewer products =
            # proportionally less memory-bound work.
            Pb_l_gr = Pb["l"] + Pb["gr"]
            Pb_g_lr = Pb["g"] + Pb["lr"]
            for ie, iq_ext in enumerate(range(q_lo, q_hi)):
                idx = q_diff_map[iq_ext]        # iqp -> iq2 = q_ext - q'
                Hl[ie] += xp.einsum("qwrs,qwrs->wrs", Pa["l"], Pb_l_gr[idx])
                Hl[ie] += xp.einsum("qwrs,qwrs->wrs", Pa["gr"], Pb["l"][idx])
                Hg[ie] += xp.einsum("qwrs,qwrs->wrs", Pa["g"], Pb_g_lr[idx])
                Hg[ie] += xp.einsum("qwrs,qwrs->wrs", Pa["lr"], Pb["g"][idx])
        for ie, iq_ext in enumerate(range(q_lo, q_hi)):
            out_l[:, iq_ext] = (Dt @ Hl[ie]) @ DtT
            out_g[:, iq_ext] = (Dt @ Hg[ie]) @ DtT
        res[(I, J)] = (out_l, out_g)
    return res
