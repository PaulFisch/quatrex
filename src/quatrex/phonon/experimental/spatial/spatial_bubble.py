# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Feeding the cubic bubble from a spatial modal representation of the legs.

Two constructions, and the second is only worth building if the first says the
missing blocks matter.

**Decompression.** Replace the Green-function blocks beyond the stored band by
a modal reconstruction and leave everything else alone -- same vertex, same
frequency convolution, same kernel. This answers "if the far blocks were free,
would the wider bubble be different", which is the question the proposal's
Sec. 20 asks and the only one that needs answering before any algebra is
designed. Two routes to the far blocks, both provided because they fail
differently:

* ``_spatial_tail_tails.sigma_decompressed`` fits the :math:`G^{<,>}`
  block sequence itself
  (block-ESPRIT, :mod:`quatrex.phonon.experimental.spatial.spatial_hankel`). Cheapest, and it does
  not preserve the matrix sign structure.
* ``_spatial_tail_tails.sigma_congruence`` continues :math:`G^R` modally and
  rebuilds
  :math:`\tilde G^{<,>} = \tilde G^R \Sigma_{\rm tot}^{<,>}\tilde G^{A}` with
  :math:`\tilde G^A = (\tilde G^R)^\dagger` exactly, so positivity is
  structural rather than checked afterwards.

**The analytic contraction.** With the vertex translation-invariant,
:math:`\Phi_{I,K,K'} = \Psi(K-I, K'-I)`, and the legs modal, the tail
factorises per modal pair into an outer product times :math:`\zeta_{pq}^R`:

.. math::
    \mathcal V^L_{pq}[\mu] &= \sum_{\alpha\beta}
        \xi_p^{-\alpha}\eta_q^{-\beta}\,
        \overline{\Psi(\alpha,\beta)}[\mu,c,e]\, v_p[c]\, w_q[e] \\
    \mathcal V^R_{pq}[\nu] &= \sum_{\gamma\delta}
        \xi_p^{\delta}\eta_q^{\gamma}\,
        \Psi(\gamma,\delta)[\nu,d,b]\, d_q[d]\, c_p[b] \\
    \Sigma_R^{pq}(\Omega) &= \mathrm{pref}\int\!d\omega\;
        \mathcal V^L_{pq}\otimes\mathcal V^R_{pq}\,
        [\xi_p(\omega)\eta_q(\Omega-\omega)]^R .

Three things about that formula are easy to get wrong and are handled here.
The cell offsets are :math:`\alpha=K_1-I`, :math:`\beta=K_2-I`,
:math:`\gamma=K_2'-J`, :math:`\delta=K_1'-J`, so the leg separations are
:math:`R+\delta-\alpha` and :math:`R+\gamma-\beta` -- checked against
``phonon/solver/bubble.py``'s own index convention, where ``a,b,c,d`` are DOF
indices and not cells. The left vertex is CONJUGATED
(``se_finite.py``), and the modal factors are not conjugated with it. And
:math:`\zeta^R` does NOT leave the frequency integral, because
:math:`\xi_p=\xi_p(\omega)` while :math:`\eta_q=\eta_q(\Omega-\omega)`; the
recurrence of the proposal's Eq. (49) acts on the integrand.

That last point is why :func:`analytic_tail` is built as the ORDINARY kernel
driven with rank-one modal legs rather than as new contraction code. The
algebra is then the same algebra, the frequency convolution is the same FFT,
and what is being tested is the representation and not a reimplementation.
"""

from __future__ import annotations

import numpy as np

from qttools import NDArray

__all__ = [
    "modal_far_blocks",
    "pair_vertex_projection",
    "pair_importance",
    "analytic_tail",
]


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def modal_far_blocks(series, band: int, r_max: int) -> dict:
    r"""``{r: G(r)}`` for ``band < r <= r_max``, from a modal series.

    The near blocks are deliberately NOT produced: the hybrid of the proposal's
    Eq. (44) is piecewise -- numerical inside the band, modal outside -- and not
    an additive correction on top of the numerical part. Mixing the two would
    double count.
    """
    return {r: series.block(r) for r in range(int(band) + 1, int(r_max) + 1)}


def pair_vertex_projection(psi, xi, v, c, eta, w, d):
    r"""``(V^L, V^R)`` for one modal pair, the proposal's Eq. (47) factors.

    Parameters
    ----------
    psi : mapping
        ``{(alpha, beta): Psi[n_dof, n_dof, n_dof]}``, the translation-invariant
        vertex kernel. ``alpha`` indexes the SECOND tensor slot and ``beta`` the
        third, matching ``Phi_{I,K,K'} = Psi(K-I, K'-I)``.
    xi, v, c
        Bloch factor, mode vector ``(n_dof,)`` and coefficient row ``(n_dof,)``
        of the mode on the first leg, at ``omega``.
    eta, w, d
        The same for the second leg, at ``Omega - omega``.

    Returns
    -------
    v_left, v_right : NDArray
        ``(n_dof,)`` each. Their outer product is the ``R``-independent factor
        of the pair's contribution; the whole ``R`` dependence is
        ``(xi eta)^R``.
    """
    v, c = np.asarray(_host(v)).ravel(), np.asarray(_host(c)).ravel()
    w, d = np.asarray(_host(w)).ravel(), np.asarray(_host(d)).ravel()
    nd = v.size
    v_left = np.zeros(nd, dtype=complex)
    v_right = np.zeros(nd, dtype=complex)
    for (a, b), block in psi.items():
        blk = np.asarray(_host(block))
        # left: conj(Psi(alpha, beta))[mu, c, e] v[c] w[e], with xi^-alpha eta^-beta
        v_left += (xi ** (-a)) * (eta ** (-b)) * (np.conj(blk) @ w) @ v
        # right: Psi(gamma, delta)[nu, d, b] d[d] c[b], with xi^delta eta^gamma
        v_right += (xi ** b) * (eta ** a) * (blk @ c) @ d
    return v_left, v_right


def pair_importance(v_left, v_right, zeta, r0: int, n_cells: int | None = None):
    r"""``I_pq(R0) = ||C_pq|| |zeta|^{R0} / (1 - |zeta|)``, the screening weight.

    The proposal's Eq. (50), and the reason for it: a mode with
    ``|lambda| ~ 1`` is still irrelevant if the FC3 barely projects onto it, so
    pairs are screened by what they contribute to the tail and not by how slowly
    they decay.

    ``|zeta| >= 1`` has no geometric sum. ``n_cells`` then gives the finite
    device sum, which is what the proposal's Sec. 23 says to use there; without
    it the weight is reported as ``inf``, which is honest and not usable.
    """
    amp = float(np.linalg.norm(v_left) * np.linalg.norm(v_right))
    z = abs(complex(zeta))
    if z < 1.0:
        return amp * z ** int(r0) / (1.0 - z)
    if n_cells is None:
        return float("inf")
    return amp * float(np.sum([z ** r for r in range(int(r0), int(n_cells))]))


def analytic_tail(psi, series_a, series_b, r_values, *, freqs_thz,
                  prefactor, pairs=None, n_fft=None, out_slice=None,
                  importance_floor: float = 0.0, n_cells=None):
    r"""``{R: Sigma_R(Omega)}`` in the pure-tail region, by the modal-pair sum.

    ``series_a[iw]`` / ``series_b[iw]`` are per-frequency
    :class:`~quatrex.phonon.experimental.spatial.spatial_fit.ModalSeries` for the two legs, or
    ``None`` where the pencil had nothing usable at that frequency (those
    samples contribute zero and are counted in the return).

    The contraction is delegated to ``phonon.solver.bubble.bubble_dense_from_fft``
    with RANK-ONE legs, one call per ``(p, q, alpha, beta, gamma, delta, R)``.
    That is deliberately the same kernel the explicit route uses: what is under
    test is the representation, not a second implementation of the ring, and a
    reimplementation would make a disagreement ambiguous.

    ``importance_floor`` drops modal pairs whose :func:`pair_importance` is
    below that fraction of the largest, which is screening by FC3-weighted tail
    amplitude rather than by ``|lambda|``.
    """
    import sys
    from pathlib import Path

    _phonon = str(Path(__file__).resolve().parents[3] / "phonon")
    if _phonon not in sys.path:
        sys.path.insert(0, _phonon)
    from solver.bubble import bubble_dense_from_fft, precompute_g_fft

    freqs_thz = np.asarray(freqs_thz)
    ne = freqs_thz.size
    n_fft = 2 * ne - 1 if n_fft is None else int(n_fft)
    mid = ne // 2
    out_slice = slice(mid, mid + ne) if out_slice is None else out_slice
    nd = next(s for s in series_a if s is not None).vecs.shape[0]

    offsets = sorted(psi)
    out = {int(r): np.zeros((ne, nd, nd), dtype=complex) for r in r_values}
    n_pairs = n_kept = 0

    # Per-frequency mode data, padded to a common rank so the loops are square.
    def unpack(series):
        r = max((s.rank for s in series if s is not None), default=0)
        lam = np.zeros((ne, r), dtype=complex)
        vec = np.zeros((ne, nd, r), dtype=complex)
        cof = np.zeros((ne, r, nd), dtype=complex)
        live = np.zeros((ne, r), dtype=bool)
        for iw, s in enumerate(series):
            if s is None:
                continue
            k = s.rank
            lam[iw, :k] = s.lam
            vec[iw, :, :k] = s.vecs
            cof[iw, :k] = s.coef
            live[iw, :k] = True
        return lam, vec, cof, live, r

    lam_a, vec_a, cof_a, live_a, ra = unpack(series_a)
    lam_b, vec_b, cof_b, live_b, rb = unpack(series_b)
    if ra == 0 or rb == 0:
        return out, {"pairs": 0, "kept": 0}

    todo = [(p, q) for p in range(ra) for q in range(rb)] if pairs is None \
        else list(pairs)

    # Screening, on the frequency carrying the largest leg amplitude.
    weights = {}
    iw0 = int(np.argmax(np.abs(lam_a).max(axis=1)))
    r0 = int(min(r_values))
    for (p, q) in todo:
        if not (live_a[iw0, p] and live_b[iw0, q]):
            weights[(p, q)] = 0.0
            continue
        vl, vr = pair_vertex_projection(
            psi, lam_a[iw0, p], vec_a[iw0, :, p], cof_a[iw0, p],
            lam_b[iw0, q], vec_b[iw0, :, q], cof_b[iw0, q])
        weights[(p, q)] = pair_importance(
            vl, vr, lam_a[iw0, p] * lam_b[iw0, q], r0, n_cells)
    peak = max([w for w in weights.values() if np.isfinite(w)] or [0.0])
    kept = [pq for pq in todo
            if not np.isfinite(weights[pq]) or weights[pq] > importance_floor * peak]
    n_pairs, n_kept = len(todo), len(kept)

    fft_cache: dict = {}

    def leg_fft(side, idx, expo):
        key = (side, idx, expo)
        hit = fft_cache.get(key)
        if hit is not None:
            return hit
        lam, vec, cof = ((lam_a, vec_a, cof_a) if side == "a"
                         else (lam_b, vec_b, cof_b))
        leg = ((vec[:, :, idx] * (lam[:, idx] ** expo)[:, None])[:, :, None]
               * cof[:, idx][:, None, :])
        fft_cache[key] = out_ = precompute_g_fft(
            leg, n_fft=n_fft, zero_freq_idx=mid, dc_handling="interpolate")
        return out_

    for (p, q) in kept:
        for r in r_values:
            r = int(r)
            for (al, be) in offsets:
                phi_l = np.conj(np.asarray(_host(psi[(al, be)])))
                for (ga, de) in offsets:
                    phi_r = np.asarray(_host(psi[(ga, de)]))
                    fa = leg_fft("a", p, r + de - al)
                    fb = leg_fft("b", q, r + ga - be)
                    out[r] += bubble_dense_from_fft(
                        phi_left=phi_l, phi_right=phi_r,
                        G_a_fft=fa, G_b_fft=fb, ne=ne,
                        prefactor=prefactor, out_slice=out_slice)
        fft_cache.clear()   # the exponents do not carry across (p, q)
    return out, {"pairs": n_pairs, "kept": n_kept, "weights": weights}
