# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""What the cell-averaged ring misses: the subcell covariance.

The ring evaluates the bubble from cell means. Splitting each leg into a mean
plus a zero-mean fluctuation, the reflection that maps one cell onto another on
a uniform grid makes both mixed mean/fluctuation terms integrate to zero, so
the exact cell integral is the ring's term plus a pure fluctuation-fluctuation
correction. That cancellation is what makes the correction cheap: the ring
already computes the first term and nothing has to be recomputed or removed,
unlike the subtract-a-pole-model-and-add-sectors routes, where the leg taken
out and the leg put back must agree over the same support under the same
quadrature.
"""
from __future__ import annotations

import numpy as np

from qttools import NDArray, xp

__all__ = [
    "cell_resolvent_mean",
    "centred_gram",
    "cell_variance",
    "covariance_kernel",
]


def cell_resolvent_mean(zeta: NDArray, centre: float, h: float) -> NDArray:
    r"""``d_p = (1/h) * int_{I} du/(u - zeta_p)`` over ``I`` centred on ``centre``.

    .. math::
        d_p = \frac{\log(c + h/2 - \zeta_p) - \log(c - h/2 - \zeta_p)}{h}

    The poles sit strictly off the real axis, so the two logarithms lie on the
    same side of the cut and their difference has no branch ambiguity -- the
    same statement :func:`~quatrex.phonon.experimental.pole.pole_congruence.cell_weights` relies
    on, and the reason this is a closed form rather than a quadrature.
    """
    z = xp.asarray(zeta, dtype=xp.complex128)
    hh = float(h)
    if hh <= 0.0:
        raise ValueError("cell_resolvent_mean: h must be positive")
    return (xp.log(centre + 0.5 * hh - z) - xp.log(centre - 0.5 * hh - z)) / hh


def centred_gram(zeta: NDArray, centre: float, h: float) -> NDArray:
    r"""``H_pq = (1/h) int_I phi_p(u) conj(phi_q(u)) du`` for the centred
    basis."""
    z = xp.asarray(zeta, dtype=xp.complex128)
    hh = float(h)
    d = cell_resolvent_mean(z, centre, hh)
    gap = z[:, None] - xp.conj(z)[None, :]
    lo = 1.0 / (centre - 0.5 * hh - z)
    hi = 1.0 / (centre + 0.5 * hh - z)
    double = ((lo - hi) / hh)[:, None] * xp.ones_like(gap)
    degenerate = xp.abs(gap) < 1e-300
    safe = xp.where(degenerate, 1.0, gap)
    uncentred = xp.where(degenerate, double,
                         (d[:, None] - xp.conj(d)[None, :]) / safe)
    return uncentred - d[:, None] * xp.conj(d)[None, :]


def cell_variance(residues: NDArray, zeta: NDArray, centre: float,
                  h: float) -> float:
    r"""``sigma_k^2 = (1/h) int_I ||delta G_k(u)||_F^2 du``."""
    r = xp.asarray(residues, dtype=xp.complex128)
    r = r.reshape(r.shape[0], -1)
    gram = centred_gram(zeta, centre, h)
    inner = r @ xp.conj(r).T                      # <R_p, R_q>_F
    val = float(xp.real(xp.sum(gram * inner)))
    if val < -1e-8 * float(xp.abs(inner).max() + 1.0):
        raise ValueError(
            f"cell_variance: negative variance {val:.3e}; the centred basis or "
            "its Gram matrix is wrong (H must be PSD).")
    return float(max(val, 0.0))


def covariance_kernel(
    zeta_k: NDArray, zeta_l: NDArray, centre_k: float, centre_l: float,
    h: float, omega: NDArray,
) -> NDArray:
    r"""``K_pq(Omega)`` for one cell pair -- the ring's error, per residue
    pair."""
    from quatrex.phonon.experimental.pole.pole_bubble import pair_convolution

    zk = xp.asarray(zeta_k, dtype=xp.complex128)
    zl = xp.asarray(zeta_l, dtype=xp.complex128)
    a, b = centre_k - 0.5 * h, centre_k + 0.5 * h
    j = pair_convolution(zk[:, None], zl[None, :], omega, window=(a, b))
    d_k = cell_resolvent_mean(zk, centre_k, h)
    d_l = cell_resolvent_mean(zl, centre_l, h)
    box = (float(h) / (2.0 * xp.pi)) * d_k[:, None] * d_l[None, :]
    return j - box[None, :, :]


def spectrum_correction(
    freqs: NDArray, cells, phi_blocks: dict, block_sizes: NDArray,
    rows: NDArray, cols: NDArray, h: float, prefactor: complex | None = None,
    chunk_bytes: int = 1 << 28,
) -> tuple[NDArray, dict]:
    r"""The total correction to add to the ring's output, on the stored
    pattern."""
    from quatrex.phonon.experimental.pole.pole_bridge import analytic_prefactor, modal_vertex_blocks

    if prefactor is None:
        prefactor = analytic_prefactor()
    w = np.asarray(_host(freqs), dtype=float)
    n_w, nnz = int(w.size), int(np.asarray(_host(rows)).size)
    out = xp.zeros((n_w, nnz), dtype=xp.complex128)
    if not cells:
        return out, {"pairs": 0, "applied": 0, "out_of_range": 0}

    r_idx, c_idx = xp.asarray(rows), xp.asarray(cols)
    w0, hh = float(w[0]), float(h)
    n_p = max(int(np.asarray(_host(c[1])).size) for c in cells)
    applied = dropped = 0
    for centre_k, zeta_k, p_k, q_k in cells:
        for centre_l, zeta_l, p_l, q_l in cells:
            om = float(centre_k) + float(centre_l)
            m = int(round((om - w0) / hh))
            if m < 0 or m >= n_w:
                dropped += 1
                continue
            kern = covariance_kernel(zeta_k, zeta_l, float(centre_k),
                                     float(centre_l), hh,
                                     xp.asarray([w[m]]))[0]
            vl = modal_vertex_blocks(phi_blocks, block_sizes, p_k,
                                     conjugate=False, v=p_l)
            vr = modal_vertex_blocks(phi_blocks, block_sizes, q_l,
                                     conjugate=False, v=q_k)
            step = max(1, int(chunk_bytes // max(1, 32 * n_p * n_p)))
            for lo in range(0, nnz, step):
                hi = min(lo + step, nnz)
                out[m, lo:hi] += prefactor * xp.einsum(
                    "pq,kpq,kqp->k", kern,
                    xp.take(vl, r_idx[lo:hi], axis=0),
                    xp.take(vr, c_idx[lo:hi], axis=0))
            applied += 1
    return out, {"pairs": len(cells) ** 2, "applied": applied,
                 "out_of_range": dropped}


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)
