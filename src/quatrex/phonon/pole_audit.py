# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Correctness gates for the pole-subtracted SCBA sector.

Four checks, each one a configuration under which the hybrid would return a
confidently wrong answer rather than a noisy one:

1. **Sector sum** -- ``B(G, G) == B(G_S,G_S) + B(G_S,G_R) + B(G_R,G_S) +
   B(G_R,G_R)``. The split ``G = G_S + G_R`` is exact and the bubble is
   bilinear, so the four sectors must reassemble the undecomposed answer when
   all four are evaluated with the SAME quadrature. This is the test that a
   term is neither dropped nor double counted; it is the reason
   ``sectors="rr"`` and ``"rr_ss"`` are staging settings and not physics.
2. **Keldysh identity** -- ``Sigma^R - Sigma^A == Sigma^< - Sigma^>`` (the
   solver's occupation-positive stored convention, ``core/scba.py``: the skew
   part of ``Sigma^R`` is assembled as ``(Sigma^< - Sigma^>)/2``). Purely
   algebraic, so it must sit at roundoff. It fails exactly when an injected
   analytic ``Sigma^R`` carries more than the Kramers-Kronig half, or when an
   analytic ``Sigma^{<,>}`` is not a congruence.
3. **Positivity** -- ``-i G^{<,>} >= 0`` on the RECONSTRUCTED TOTAL, never on a
   sector. ``G_PP`` and ``G_BB`` are separately congruences and separately
   PSD, but ``G_PP + G_PB + G_BP`` is not; only the full sum is
   (``bubble_positivity.md`` Thm 1-2).
4. **Energy balance** -- the bubble in/out identity, which is only testable
   PRE-mixing (``QX_BBCHECK=1``); the saved arrays are post-mixing and measure
   the SCBA residual instead (``phonon_conservation_measurement_trap``).

Every function is pure and backend-agnostic; nothing here holds state or
touches the distribution.
"""
from __future__ import annotations

import numpy as np

from qttools import NDArray, xp


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def transpose_index(rows: NDArray, cols: NDArray) -> NDArray:
    """Permutation ``t`` with ``(rows[t[k]], cols[t[k]]) == (cols[k], rows[k])``.

    The adjoint of a matrix stored on a structurally symmetric pattern is a
    permutation of the values plus a conjugation, so no densification is
    needed. Raises if the pattern is not structurally symmetric -- in that case
    the adjoint is not representable and the Keldysh identity cannot be
    evaluated on the pattern at all.
    """
    r = np.asarray(_host(rows), dtype=np.int64)
    c = np.asarray(_host(cols), dtype=np.int64)
    n = int(max(r.max(), c.max())) + 1
    key = r * n + c
    tkey = c * n + r
    order = np.argsort(key, kind="stable")
    pos = np.searchsorted(key[order], tkey)
    if np.any(pos >= key.size) or np.any(key[order][np.minimum(pos, key.size - 1)] != tkey):
        raise ValueError(
            "transpose_index: the stored pattern is not structurally "
            "symmetric, so Sigma^A is not representable on it."
        )
    return order[pos]


def keldysh_identity(
    sigma_retarded: NDArray,
    sigma_lesser: NDArray,
    sigma_greater: NDArray,
    rows: NDArray,
    cols: NDArray,
) -> dict[str, float]:
    r"""``eps_KI`` and the two failure modes it decomposes into.

        Parameters
        ----------
        sigma_retarded, sigma_lesser, sigma_greater : NDArray
            ``(n_omega, nnz)`` values on the stored pattern.
        rows, cols : NDArray
            ``(nnz,)`` global indices.

        Returns
        -------
        dict
            ``eps_ki`` (the identity residual -- the only one that sees a pure
            magnitude error such as a double-counted retarded half),
            ``eps_delta_skew`` (non-anti-Hermiticity of ``Sigma^< - Sigma^>``,
            i.e. an analytic ``Sigma^{<,>}`` that is not a congruence) and
            ``eps_kk_hermitian`` (non-Hermiticity of the recovered KK part), each
            relative to ``||Sigma^< - Sigma^>||_F``.
    """
    t = xp.asarray(transpose_index(rows, cols))
    delta = sigma_lesser - sigma_greater
    scale = float(xp.linalg.norm(delta))
    if scale == 0.0:
        return {"eps_ki": 0.0, "eps_delta_skew": 0.0, "eps_kk_hermitian": 0.0}

    adjoint = xp.conj(sigma_retarded[:, t])
    ki = float(xp.linalg.norm(sigma_retarded - adjoint - delta)) / scale

    d_skew = float(xp.linalg.norm(delta + xp.conj(delta[:, t]))) / scale
    kk = sigma_retarded - 0.5 * delta
    kk_h = float(xp.linalg.norm(kk - xp.conj(kk[:, t]))) / scale
    return {"eps_ki": ki, "eps_delta_skew": d_skew, "eps_kk_hermitian": kk_h}


def psd_residual(
    values: NDArray,
    rows: NDArray,
    cols: NDArray,
    block_sizes: NDArray,
    sign: float = -1.0,
    window: int = 2,
    skip: NDArray | None = None,
) -> dict[str, float]:
    r"""Worst normalised eigenvalue of ``sign * i * G`` over the diagonal
    blocks.

        Parameters
        ----------
        values : NDArray
            ``(n_omega, nnz)`` ``G^<`` or ``G^>`` on the stored pattern.
        sign : float
            ``-1`` for ``G^<`` (``-i G^< >= 0``), ``+1`` for ``G^>``.
        skip : NDArray, optional
            ``(n_omega,)`` boolean; ``True`` drops that bin from BOTH the search
            and the normalisation. Pass the bubble's own ``conv_mask``. Without it
            the gate is decided by bins the ring never integrates: on the CNT bed
            ``G^>`` at ``w = 0`` -- the near-singular acoustic bin the ring zeroes
            -- carries the largest eigenvalue in the whole window AND the most
            negative one, so ``worst`` saturates at exactly ``-1.000`` on the
            POLE-FREE baseline and reports the same ``-1.000`` for every variant
            of the pole sector. A gate that reads the same with the feature on and
            off is measuring the feature not at all.

        Returns
        -------
        dict
            ``worst`` (most negative normalised eigenvalue; 0 if none),
            ``scale`` (the global normalisation) and ``omega_index`` of the worst.
    """
    sizes = np.asarray(_host(block_sizes), dtype=int)
    off = np.concatenate(([0], np.cumsum(sizes)))
    r = np.asarray(_host(rows), dtype=np.int64)
    c = np.asarray(_host(cols), dtype=np.int64)

    if window < 1:
        raise ValueError("psd_residual: window must be >= 1")
    keep = np.arange(int(values.shape[0]))
    if skip is not None:
        m = np.asarray(_host(skip), dtype=bool).ravel()
        keep = keep[~m[:keep.size]]
        if keep.size == 0:
            return {"worst": 0.0, "scale": 0.0, "omega_index": -1}
        values = values[xp.asarray(keep)]
    evs = []
    for i in range(max(1, sizes.size - window + 1)):
        lo, hi = int(off[i]), int(off[min(i + window, sizes.size)])
        sel = np.flatnonzero((r >= lo) & (r < hi) & (c >= lo) & (c < hi))
        if sel.size == 0:
            continue
        b = hi - lo
        dense = xp.zeros((values.shape[0], b, b), dtype=values.dtype)
        dense[:, xp.asarray(r[sel] - lo), xp.asarray(c[sel] - lo)] = \
            values[:, xp.asarray(sel)]
        herm = sign * 1j * dense
        herm = 0.5 * (herm + xp.conj(xp.swapaxes(herm, -1, -2)))
        evs.append(xp.linalg.eigvalsh(herm))          # (n_omega, b)

    if not evs:
        return {"worst": 0.0, "scale": 0.0, "omega_index": -1}

    scale = max(float(xp.abs(ev).max()) for ev in evs)
    if scale == 0.0:
        return {"worst": 0.0, "scale": 0.0, "omega_index": -1}

    worst, worst_w = 0.0, -1
    for ev in evs:
        m = float(ev.min()) / scale
        if m < worst:
            worst = m
            worst_w = int(keep[int(_host(xp.argmin(ev.min(axis=-1))))])
    return {"worst": worst, "scale": scale, "omega_index": worst_w}


def pole_pair_weight(
    leg_norms: NDArray, pole_cells: NDArray, freqs: NDArray | None = None,
    skip: NDArray | None = None,
) -> dict[str, float]:
    r"""What fraction of the ring's weight comes from POLE-CELL PAIRS.

        Parameters
        ----------
        leg_norms : NDArray
            ``(n_omega,)`` per-bin norm of the leg the ring convolves.
        pole_cells : NDArray
            ``(n_omega,)`` boolean, True where a promoted pole sits.
        freqs : NDArray, optional
            Only used to report where the worst bin is.
        skip : NDArray, optional
            ``(n_omega,)`` boolean over OUTPUT bins, excluded from ``worst`` and
            ``mean``. Pass the ring's own mask: a single pole pairs with itself at
            the DIFFERENCE frequency ``omega = 0``, where the fraction is ~1 by
            construction and where the ring sets ``Sigma`` to zero anyway, so an
            unmasked report is pinned there and says nothing.

        Returns
        -------
        dict
            ``worst`` (largest pair fraction over output bins), ``omega_index``
            and ``omega`` of that bin, and ``mean`` (weighted by total ring
            weight, i.e. the fraction of ALL the ring's weight that sits on pole
            cell pairs).
    """
    g = np.abs(np.asarray(_host(leg_norms), dtype=float)).ravel()
    p = np.asarray(_host(pole_cells), dtype=bool).ravel()
    n = g.size
    if n == 0 or not p.any():
        return {"worst": 0.0, "omega_index": -1, "omega": float("nan"),
                "mean": 0.0}
    gp = g * p
    k = np.arange(n)
    tot = np.empty(n)
    par = np.empty(n)
    for m in range(n):
        j = np.abs(m - k)                  # bosonic mirror onto omega >= 0
        tot[m] = float(g[k] @ g[j])
        par[m] = float(gp[k] @ gp[j])
    frac = par / np.where(tot > 0.0, tot, 1.0)
    live = np.ones(n, dtype=bool)
    if skip is not None:
        live = ~np.asarray(_host(skip), dtype=bool).ravel()[:n]
    if not live.any():
        return {"worst": 0.0, "omega_index": -1, "omega": float("nan"),
                "mean": 0.0}
    i = int(np.flatnonzero(live)[np.argmax(frac[live])])
    w = (float("nan") if freqs is None
         else float(np.asarray(_host(freqs), dtype=float).ravel()[i]))
    return {"worst": float(frac[i]), "omega_index": i, "omega": w,
            "mean": float(par[live].sum() / (tot[live].sum() or 1.0))}


def sector_sum_residual(
    total: NDArray, sectors: dict[str, NDArray]
) -> dict[str, float]:
    """Relative residual of ``sum(sectors) - total``, plus each sector's weight.

    The weights matter as much as the residual: a sector-sum test passes
    vacuously if three of the four sectors are numerically zero, so the report
    carries the fraction of the total each one contributes.
    """
    acc = None
    weights = {}
    for name, s in sectors.items():
        acc = s if acc is None else acc + s
        weights[f"weight_{name}"] = float(xp.linalg.norm(s))
    scale = float(xp.linalg.norm(total))
    out = {"residual": float(xp.linalg.norm(acc - total)) / (scale or 1.0),
           "scale": scale}
    for k, v in weights.items():
        out[k] = v / (scale or 1.0)
    return out


def subcell_positivity(
    g_full: NDArray,
    g_pole: NDArray,
    pole_at: callable,
    freqs: NDArray,
    rows: NDArray,
    cols: NDArray,
    block_sizes: NDArray,
    centres: NDArray,
    n_sub: int = 17,
    sign: float = -1.0,
    window: int = 2,
) -> dict[str, float]:
    r"""Is the reconstructed hybrid function physical BETWEEN grid points?

        Parameters
        ----------
        g_full, g_pole : NDArray
            ``(n_omega, nnz)`` the stored Green's function and the pole part
            evaluated on the SAME grid, so ``R_k = g_full - g_pole``.
        pole_at : callable
            ``omega -> (n_omega, nnz)``, the pole part at arbitrary frequency.
        centres : NDArray
            Grid indices of the cells to probe -- normally the promoted poles'.
        n_sub : int
            Sub-cell samples per cell, spanning the full width.

        Returns
        -------
        dict
            ``worst`` (most negative normalised eigenvalue over all probes),
            ``worst_centre`` (the cell index where it occurred) and
            ``at_centres`` (the same measure evaluated only AT the centres, which
            should be healthy -- if it is not, the failure is upstream).
    """
    w = np.asarray(_host(freqs), dtype=float)
    if w.size < 2:
        return {"worst": 0.0, "worst_centre": -1, "at_centres": 0.0}
    h = float(w[1] - w[0])
    idx = np.atleast_1d(np.asarray(_host(centres), dtype=int))
    if idx.size == 0:
        return {"worst": 0.0, "worst_centre": -1, "at_centres": 0.0}

    remainder = xp.asarray(g_full) - xp.asarray(g_pole)

    worst, worst_at = 0.0, -1
    at_centres = 0.0
    offsets = np.linspace(-0.5, 0.5, int(n_sub))
    for k in idx:
        k = int(np.clip(k, 0, w.size - 1))
        r_k = remainder[k][None, :]
        probes = xp.asarray(w[k] + offsets * h)
        recon = pole_at(probes) + r_k                  # (n_sub, nnz)
        rep = psd_residual(recon, rows, cols, block_sizes,
                           sign=sign, window=window)
        if rep["worst"] < worst:
            worst, worst_at = rep["worst"], k
        centre_rep = psd_residual(recon[int(n_sub) // 2][None, :], rows, cols,
                                  block_sizes, sign=sign, window=window)
        at_centres = min(at_centres, centre_rep["worst"])

    return {"worst": worst, "worst_centre": worst_at, "at_centres": at_centres}


def subcell_congruence(
    g_retarded: NDArray,
    sigma: NDArray,
    pole_retarded_at: callable,
    freqs: NDArray,
    rows: NDArray,
    cols: NDArray,
    centres: NDArray,
    n_sub: int = 17,
    sign: float = -1.0,
    max_dof: int = 512,
) -> dict[str, float]:
    r"""Subcell positivity of the CONGRUENCE reconstruction.

        Returns
        -------
        dict
            ``worst`` and ``worst_centre``, as :func:`subcell_positivity`.
    """
    r = np.asarray(_host(rows), dtype=np.int64)
    c = np.asarray(_host(cols), dtype=np.int64)
    n_dof = int(max(r.max(), c.max())) + 1
    if n_dof > max_dof:
        raise NotImplementedError(
            f"subcell_congruence densifies the full {n_dof}x{n_dof} operator "
            f"and refuses above {max_dof}; the congruence couples the whole "
            "device, so a block-window restriction would not be the same "
            "object."
        )
    w = np.asarray(_host(freqs), dtype=float)
    h = float(w[1] - w[0])
    idx = np.atleast_1d(np.asarray(_host(centres), dtype=int))

    def _dense(vals):
        out = xp.zeros((vals.shape[0], n_dof, n_dof), dtype=xp.complex128)
        out[:, xp.asarray(r), xp.asarray(c)] = vals
        return out

    worst, worst_at = 0.0, -1
    offsets = np.linspace(-0.5, 0.5, int(n_sub))
    for k in idx:
        k = int(np.clip(k, 0, w.size - 1))
        gr_k = _dense(xp.asarray(g_retarded)[k][None, :])[0]
        pr_k = _dense(pole_retarded_at(xp.asarray([w[k]])))[0]
        rem = gr_k - pr_k                                   # frozen RETARDED
        sig = _dense(xp.asarray(sigma)[k][None, :])[0]

        probes = xp.asarray(w[k] + offsets * h)
        pr = _dense(pole_retarded_at(probes))               # (n_sub, N, N)
        g_ret = pr + rem[None]
        recon = g_ret @ sig[None] @ xp.conj(xp.swapaxes(g_ret, -1, -2))
        herm = sign * 1j * recon
        herm = 0.5 * (herm + xp.conj(xp.swapaxes(herm, -1, -2)))
        ev = xp.linalg.eigvalsh(herm)
        scale = float(xp.abs(ev).max())
        if scale == 0.0:
            continue
        m = float(ev.min()) / scale
        if m < worst:
            worst, worst_at = m, k
    return {"worst": worst, "worst_centre": worst_at}
