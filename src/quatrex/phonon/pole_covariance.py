# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""What the cell-averaged ring misses: the subcell covariance.

The ring evaluates the bubble from cell means. Write the local hybrid function
on cell :math:`I_k` as a mean plus a zero-mean fluctuation,

.. math::
    \tilde G_k(u) = \bar G_k + \delta G_k(u), \qquad
    \frac{1}{h}\int_{I_k}\delta G_k(u)\,du = 0 .

On a uniform grid with output :math:`\Omega_m = m h`, the reflection
:math:`v = \Omega_m - u` maps :math:`I_k` exactly onto :math:`I_l`,
:math:`l = m - k`, so both mixed mean/fluctuation terms integrate to zero and

.. math::
    \int_{I_k}\frac{du}{2\pi} B[\tilde G_k(u), \tilde G_l(\Omega_m - u)]
      = \underbrace{\frac{h}{2\pi} B(\bar G_k, \bar G_l)}_{\text{the ring}}
      + \underbrace{\Delta I_{kl}}_{\text{this module}} ,

.. math::
    \Delta I_{kl} = \int_{I_k}\frac{du}{2\pi}
                    B[\delta G_k(u), \delta G_l(\Omega_m - u)] .

That exact cancellation of the cross terms is what makes the correction cheap:
the ring already computes the first term, and nothing else has to be recomputed
or removed. It is the difference from the subtract-a-pole-model-and-add-sectors
routes, where the leg taken out and the leg put back have to be the same
function over the same support under the same quadrature -- three separate ways
to be wrong, none of which arises here.

**The basis.** With the local congruence partial-fractioned to
:math:`\tilde G_k(u) = C_k + \sum_p R_{kp}/(u - \zeta_{kp})`, the centred pole
functions

.. math::
    \phi_{kp}(u) = \frac{1}{u - \zeta_{kp}} - d_{kp}, \qquad
    d_{kp} = \frac{1}{h}\int_{I_k}\frac{du}{u - \zeta_{kp}}

have zero cell mean by construction, and :math:`\delta G_k = \sum_p R_{kp}
\phi_{kp}`. The constant :math:`C_k` drops out entirely -- it is part of the
mean -- which is why no background model is needed for the leading correction.

**The kernel.** Expanding the two centred factors and using
:math:`\int_{I_k} du/(\Omega - u - \zeta_q) = h\,d_{lq}` (the same reflection),

.. math::
    K_{pq} = J_{[a,b]}(\zeta_{kp}, \zeta_{lq}; \Omega_m)
             - \frac{h}{2\pi} d_{kp} d_{lq},

with :math:`J` the finite-interval two-pole integral already in
:func:`~quatrex.phonon.pole_bubble.pair_convolution` under ``window=``.

**Stability.** :math:`s = \Omega - \zeta_p - \zeta_q` does vanish, and both
places it can are ordinary here.

For two poles in the SAME half plane :math:`\operatorname{Im} s =
\gamma_p + \gamma_q > 0` bounds :math:`|s|` below, and the logarithmic form
holds to 2e-16 down to :math:`\gamma/h = 10^{-10}`. But a flattened family is
:math:`[z, \bar z]` by construction, so it always contains MIXED pairings, and
for those :math:`\zeta_p + \zeta_q = 2\operatorname{Re} z` is real:
:math:`\operatorname{Im} s = \gamma_p - \gamma_q = 0` and :math:`s` passes
exactly through zero as :math:`\Omega` sweeps.
:func:`~quatrex.phonon.pole_bubble.pair_convolution` therefore carries the
review's series branch after all, switched on :math:`|s|` against
:math:`\min_u |u - \zeta_p|` -- which for a pole inside its own cell is
:math:`\gamma_p`, not the distance to the cell endpoints.

The same degeneracy makes :math:`\zeta_p = \bar\zeta_q` the rule rather than
the exception in :func:`centred_gram`, where the partial fraction is 0/0 and
the limit is the elementary double-pole integral.

**Cost.** Every residue here is rank one, :math:`R_p = x_p y_p^\dagger`, so

.. math::
    B(R_p, R_q)_{\mu\nu} \propto
        \Big[\sum_{ab}\Phi_{\mu ab} x_{p,a} x_{q,b}\Big]
        \Big[\sum_{cd}\Phi_{\nu cd} \bar y_{q,c} \bar y_{p,d}\Big],

one projected pair vertex on each side. The pole algebra is quadratic in the
flattened residue count, not quartic; a quartic cost would be a redundant modal
index rather than the method.
"""
from __future__ import annotations

import numpy as np

from qttools import NDArray, xp

__all__ = [
    "cell_resolvent_mean",
    "centred_gram",
    "cell_variance",
    "covariance_kernel",
    "pair_covariance",
]


def cell_resolvent_mean(zeta: NDArray, centre: float, h: float) -> NDArray:
    r"""``d_p = (1/h) * int_{I} du/(u - zeta_p)`` over ``I`` centred on ``centre``.

    .. math::
        d_p = \frac{\log(c + h/2 - \zeta_p) - \log(c - h/2 - \zeta_p)}{h}

    The poles sit strictly off the real axis, so the two logarithms lie on the
    same side of the cut and their difference has no branch ambiguity -- the
    same statement :func:`~quatrex.phonon.pole_congruence.cell_weights` relies
    on, and the reason this is a closed form rather than a quadrature.
    """
    z = xp.asarray(zeta, dtype=xp.complex128)
    hh = float(h)
    if hh <= 0.0:
        raise ValueError("cell_resolvent_mean: h must be positive")
    return (xp.log(centre + 0.5 * hh - z) - xp.log(centre - 0.5 * hh - z)) / hh


def centred_gram(zeta: NDArray, centre: float, h: float) -> NDArray:
    r"""``H_pq = (1/h) int_I phi_p(u) conj(phi_q(u)) du`` for the centred basis.

    .. math::
        H_{pq} = \frac{d_p - \bar d_q}{\zeta_p - \bar\zeta_q} - d_p \bar d_q ,

    the uncentred moment by partial fractions minus the product of means.

    ``H`` is a Gram matrix of the functions :math:`\phi_p` under the cell inner
    product, so it is Hermitian PSD up to roundoff. That is not decoration: it
    is the cheapest available check that the centred basis was built correctly,
    and :func:`cell_variance` is only a variance because of it.
    """
    z = xp.asarray(zeta, dtype=xp.complex128)
    hh = float(h)
    d = cell_resolvent_mean(z, centre, hh)
    gap = z[:, None] - xp.conj(z)[None, :]
    # zeta_p == conj(zeta_q) is the RULE here, not an exception: a flattened
    # family is [z, conj(z)] by construction, so every pole meets its own
    # conjugate and the partial fraction is 0/0 for that pair. The limit is the
    # double-pole integral,
    #
    #     (1/h) int_I du/(u - zeta)^2 = (1/h)[1/(c - h/2 - zeta)
    #                                         - 1/(c + h/2 - zeta)],
    #
    # which is elementary. Refusing instead -- as an earlier version did --
    # rejects every real pole set the sector can produce.
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
    r"""``sigma_k^2 = (1/h) int_I ||delta G_k(u)||_F^2 du``.

    .. math::
        \sigma_k^2 = \sum_{pq} H_{pq}\,\langle R_p, R_q\rangle_F

    How much subcell structure the cell actually carries, in the norm the
    Cauchy-Schwarz bound uses:

    .. math:: \|\Delta I_{kl}\|_F \le \frac{h}{2\pi}\,\beta\,\sigma_k\sigma_l .

    This is the screening quantity. A binary pole-cell / not-a-pole-cell rule
    corrects cells whose fluctuation is numerically irrelevant and misses cells
    one spacing away whose fluctuation is not -- the tail of an unresolved line
    reaches further than the cell holding it.

    ``residues`` is ``(P,) + G.shape`` or ``(P, n)`` flattened; only the
    Frobenius inner products are used.
    """
    r = xp.asarray(residues, dtype=xp.complex128)
    r = r.reshape(r.shape[0], -1)
    gram = centred_gram(zeta, centre, h)
    inner = r @ xp.conj(r).T                      # <R_p, R_q>_F
    # |delta G|^2 = sum_pq <R_p, R_q>_F phi_p conj(phi_q), so the Gram pairs
    # with inner UNCONJUGATED. Conjugating it here gave 7.30e+03 against a
    # quadrature 7.80e+03 -- a 6 % error that looks like a tolerance issue and
    # is not one.
    val = float(xp.real(xp.sum(gram * inner)))
    # Hermitian PSD in exact arithmetic; a small negative is roundoff on a
    # near-null basis, a large one is a defect in the basis.
    if val < -1e-8 * float(xp.abs(inner).max() + 1.0):
        raise ValueError(
            f"cell_variance: negative variance {val:.3e}; the centred basis or "
            "its Gram matrix is wrong (H must be PSD).")
    return float(max(val, 0.0))


def covariance_kernel(
    zeta_k: NDArray, zeta_l: NDArray, centre_k: float, centre_l: float,
    h: float, omega: NDArray,
) -> NDArray:
    r"""``K_pq(Omega)`` for one cell pair -- the ring's error, per residue pair.

    .. math::
        K_{pq} = J_{[a,b]}(\zeta_{kp}, \zeta_{lq}; \Omega)
                 - \frac{h}{2\pi}\,d_{kp}\,d_{lq}

    The second term is what the ring already put there: the product of the two
    cell means of the resolvents. Subtracting it is what makes this a
    CORRECTION rather than a replacement, so nothing has to be removed from the
    FFT output and the identity holds cell pair by cell pair.

    All four half-plane pairings are kept. Their cancellation is a whole-axis
    contour statement; on one finite cell the mixed pairings are a genuine part
    of the integral, and the residue form's zero would be wrong here.

    Returns ``(n_omega, P_k, P_l)``.
    """
    from quatrex.phonon.pole_bubble import pair_convolution

    zk = xp.asarray(zeta_k, dtype=xp.complex128)
    zl = xp.asarray(zeta_l, dtype=xp.complex128)
    a, b = centre_k - 0.5 * h, centre_k + 0.5 * h
    j = pair_convolution(zk[:, None], zl[None, :], omega, window=(a, b))
    d_k = cell_resolvent_mean(zk, centre_k, h)
    d_l = cell_resolvent_mean(zl, centre_l, h)
    box = (float(h) / (2.0 * xp.pi)) * d_k[:, None] * d_l[None, :]
    return j - box[None, :, :]


def pair_covariance(
    res_k: NDArray, res_l: NDArray, zeta_k: NDArray, zeta_l: NDArray,
    centre_k: float, centre_l: float, h: float, omega: NDArray,
    bilinear=None,
) -> NDArray:
    r"""``Delta I_kl(Omega) = sum_pq K_pq B(R_kp, R_lq)``, densely.

    The reference form, for the bed and the tests: it contracts dense residues
    through ``bilinear`` (the plain matrix product by default). A device path
    keeps the residues rank one and contracts the projected pair vertices
    instead -- see the module docstring -- but both evaluate the same sum, so
    this is the oracle that path is checked against.
    """
    k = covariance_kernel(zeta_k, zeta_l, centre_k, centre_l, h, omega)
    rk = xp.asarray(res_k, dtype=xp.complex128)
    rl = xp.asarray(res_l, dtype=xp.complex128)
    if bilinear is None:
        blocks = xp.einsum("pij,qjl->pqil", rk, rl)
    else:
        blocks = xp.stack([xp.stack([bilinear(x, y) for y in rl]) for x in rk])
    return xp.einsum("wpq,pqij->wij", k, blocks)


def spectrum_correction(
    freqs: NDArray, cells, phi_blocks: dict, block_sizes: NDArray,
    rows: NDArray, cols: NDArray, h: float, prefactor: complex | None = None,
    chunk_bytes: int = 1 << 28,
) -> tuple[NDArray, dict]:
    r"""The total correction to add to the ring's output, on the stored pattern.

    ``cells`` is the ACTIVE set on the EXTENDED frequency axis: one entry per
    cell that carries enough subcell structure to matter, as
    ``(centre, zeta, p_row, q_col)`` with rank-one residues
    :math:`R_p = p_p q_p^{\mathsf T}`.

    Both convolution channels come out of one loop. The ring integrates over
    the whole axis, so a cell at :math:`-\omega_l` is as much a partner as one
    at :math:`+\omega_l`; putting the negative cells in ``cells`` with their
    own (partner-component, transposed) families makes the SUM channel
    :math:`\omega_k + \omega_l` and the DIFFERENCE channel
    :math:`\omega_k - \omega_l` the same statement. Treating only the sum would
    silently drop every :math:`\Omega_a - \Omega_b` process, which is half the
    convolution.

    Each active PAIR writes to exactly one output bin, ``m`` with
    :math:`\omega_m = \omega_k + \omega_l`, so the cost is quadratic in the
    number of active cells and independent of how narrow the lines are. Pairs
    landing outside the stored output range are counted in the report rather
    than dropped silently -- they are corrections the run is not applying.

    Returns ``(correction, report)``; ``correction`` is ``(n_omega, nnz)``.
    """
    from quatrex.phonon.pole_bridge import analytic_prefactor, modal_vertex_blocks

    if prefactor is None:
        prefactor = analytic_prefactor()
    w = np.asarray(_host(freqs), dtype=float)
    n_w, nnz = int(w.size), int(np.asarray(_host(rows)).size)
    out = xp.zeros((n_w, nnz), dtype=xp.complex128)
    if not cells:
        return out, {"pairs": 0, "applied": 0, "out_of_range": 0}

    r_idx, c_idx = xp.asarray(rows), xp.asarray(cols)
    w0, hh = float(w[0]), float(h)
    n_p = int(np.asarray(_host(cells[0][1])).size)
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
            # The two legs are different cells, so the vertex is projected onto
            # a MIXED pair of families -- p_row from k on alpha, p_row from l
            # on beta, and the mirrored pairing on the right.
            vl = modal_vertex_blocks(phi_blocks, block_sizes, p_k,
                                     conjugate=False, v=p_l)
            vr = modal_vertex_blocks(phi_blocks, block_sizes, q_l,
                                     conjugate=False, v=q_k)
            # CHUNKED over the pattern. vl[rows] is (nnz, P, P): at a
            # production nnz and P = 2 N_p that is tens of gigabytes per pair,
            # which is the same materialisation that took the sector kernels
            # to a 290 GB allocation.
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
