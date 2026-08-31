# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Local finite-cell replacement of the ring's rectangle rule.

**Status: NOT WIRED.** Nothing in ``src/`` or the engine calls
:func:`correct_spectrum`, and no config field reaches it; the module is
exercised only by ``tests/quatrex/phonon/test_pole_local.py``. Wiring it means
adding a ``pole_sector`` flag (default off, per the project's
options-not-silent-defaults rule), passing the promoted cells and the flattened
``(zeta, residues)`` through to :mod:`quatrex.phonon.sse_phonon_phonon`, and
adding the correction to the raw bubble output BEFORE ``delta`` is formed so
the Kramers-Kronig transform covers it -- the same placement the mixed sector
needs.

That has deliberately not been done, because on the only production CNT bed
there is nothing for it to correct: the converged ``cnt33_L4_linear``
linewidths run from ``Gamma_tot/dw = 1.573`` upward with ZERO of 144 modes
below one grid spacing (``phonon/docs/cnt_observations.md`` Sec. 6). Wiring it
now would ship a path that provably does nothing on the bed it would be judged
on. It is kept because the kernels are verified and because a bed WITH narrow
modes is the case it was built for.

The ring evaluates the bubble as a rectangle rule on point samples,

.. math::
    \Sigma(\Omega_m) \simeq \frac{h}{2\pi}\sum_k
        B\big[G(\omega_k),\, G(\omega_{m-k})\big],

which is wrong by :math:`O(h/\gamma)` in any cell holding a pole narrower than
the spacing, and wrong in a way that depends on where the pole falls inside its
cell. Measured on one cell against dense quadrature, pole cell against a smooth
partner:

======== ============ ================= ==========
gamma/h  rectangle    cell-avg product  exact
======== ============ ================= ==========
0.400    5.14e-01     3.29e-01          2.5e-18
0.100    1.77e+00     5.29e-01          4.7e-16
0.020    9.41e+00     6.06e-01          1.8e-16
0.005    3.82e+01     6.21e-01          2.2e-16
0.001    1.92e+02     6.26e-01          2.2e-16
======== ============ ================= ==========

This module replaces those finitely many cell-pair contributions,

.. math::
    \Sigma_{\text{corrected}} = \Sigma_{\text{ring}}
        + \sum_{k\ \text{pole},\ l} \big[I^{\text{exact}}_{kl}
                                       - I^{\text{rect}}_{kl}\big],

rather than subtracting a pole model from the leg and adding sectors back. The
difference matters. A global split needs the subtracted and restored legs to be
the same function over the same support with the same quadrature semantics, and
each of those is a separate way to be wrong; here both halves of every term are
integrals of one model over one interval, so the only approximation left is the
local model itself.

The leg inside a pole cell is the congruence
:math:`\tilde G^R \Sigma_k \tilde G^A`, which is exactly a constant plus simple
poles once ``pole_congruence.partial_fraction_legs`` has flattened it -- the
review's Eq. (9), evaluated per cell instead of once. Outside a pole cell the
leg is modelled by a polynomial in the stored grid samples, which converges
only when the cell is more than one spacing from its own nearest pole:

===== ========= ========= ========= =========
d/h   const     linear    quadratic cubic
===== ========= ========= ========= =========
0.3   5.77e-01  3.00e-01  1.54e-01  7.79e-02
1.0   3.04e-01  7.74e-02  1.91e-02  4.67e-03
2.0   1.53e-01  1.95e-02  2.40e-03  2.92e-04
5.0   6.13e-02  3.10e-03  1.53e-04  7.40e-06
===== ========= ========= ========= =========

That is the design rule: rational against rational when both cells hold poles,
rational against polynomial otherwise, and the promotion criterion is what
guarantees a non-pole cell sits at ``d/h >~ 1``.

What this does NOT fix is output resolution. A corrected pair puts structure at
:math:`\Omega = \Re(\zeta_p + \zeta_q)` of width :math:`\gamma_p + \gamma_q`,
and integrating the input exactly does not make that structure representable on
the output grid. :func:`output_resolution` measures it; see the module note in
``phonon/docs/pole_scba_divergence.md`` Sec. 10.
"""
from __future__ import annotations

from dataclasses import dataclass

from qttools import NDArray, xp

from quatrex.phonon.experimental.pole.pole_bubble import pair_convolution

__all__ = [
    "LocalLeg",
    "deflate",
    "background_from_samples",
    "resolvent_moments",
    "cross_polynomial_moments",
    "pair_terms",
    "pair_correction",
    "contract",
    "build_legs",
    "correct_spectrum",
    "output_resolution",
]


@dataclass(frozen=True)
class LocalLeg:
    r"""One cell's leg: a polynomial background plus shared analytic poles."""

    bg: NDArray
    centre: float
    zeta: NDArray
    residues: NDArray

    def __post_init__(self):
        if self.zeta.shape[0] != self.residues.shape[0]:
            raise ValueError(
                f"zeta has {self.zeta.shape[0]} poles but residues has "
                f"{self.residues.shape[0]}.")

    @property
    def order(self) -> int:
        """Degree of the background polynomial."""
        return int(self.bg.shape[0]) - 1

    def eval(self, s: NDArray) -> NDArray:
        """The leg at ``s``, shape ``(len(s),) + bg.shape[1:]``."""
        s = xp.asarray(s, dtype=xp.complex128)
        t = s - self.centre
        powers = xp.stack([t**j for j in range(self.bg.shape[0])])
        out = xp.einsum("js,j...->s...", powers, self.bg)
        if self.zeta.shape[0]:
            d = 1.0 / (s[:, None] - self.zeta[None, :])
            out = out + xp.einsum("sp,p...->s...", d, self.residues)
        return out


def deflate(
    freqs: NDArray, g: NDArray, zeta: NDArray, residues: NDArray
) -> NDArray:
    r"""Remove the promoted poles from the stored samples, on the whole grid.

    ``g - sum_p residues[p] / (freqs - zeta[p])``. The result is what the
    per-cell backgrounds are interpolated from, and it is smooth across the
    pole cells and their neighbours because the structure that was not smooth
    has been taken out analytically.
    """
    if xp.asarray(zeta).shape[0] == 0:
        return xp.asarray(g, dtype=xp.complex128)
    w = xp.asarray(freqs, dtype=xp.complex128)[:, None]
    d = 1.0 / (w - xp.asarray(zeta)[None, :])
    return xp.asarray(g, dtype=xp.complex128) - xp.einsum(
        "sp,p...->s...", d, xp.asarray(residues))


def background_from_samples(
    samples: NDArray, centre: float, h: float
) -> NDArray:
    r"""Taylor coefficients about ``centre`` from an odd centred stencil.

    ``samples`` are the DEFLATED leg at ``centre + (i - (n-1)/2) h``. Odd
    lengths 1, 3, 5 give degree 0, 2, 4; a centred stencil of even degree
    carries the same leading error as the odd degree above it, so nothing is
    gained by asking for one.
    """
    n = int(samples.shape[0])
    if n % 2 == 0:
        raise ValueError(f"need an odd, centred stencil; got {n} samples.")
    offsets = (xp.arange(n, dtype=float) - (n - 1) / 2.0) * float(h)
    vander = xp.stack([offsets**j for j in range(n)], axis=1)
    flat = xp.asarray(samples, dtype=xp.complex128).reshape(n, -1)
    coeffs = xp.linalg.solve(vander.astype(xp.complex128), flat)
    return coeffs.reshape(samples.shape)


def resolvent_moments(
    zeta: NDArray, a: float, b: float, centre: float, order: int
) -> NDArray:
    r"""``R_j = int_a^b (s - centre)^j / (s - zeta) ds`` for ``j = 0 ..
    order``."""
    z = xp.asarray(zeta, dtype=xp.complex128)
    if bool(xp.any(xp.imag(z) == 0.0)):
        raise ValueError(
            "resolvent_moments: a pole lies on the real axis, where the "
            "integral is a principal value and the Log difference is "
            "ambiguous. Promoted poles are strictly off axis.")
    t1, t2 = float(a) - float(centre), float(b) - float(centre)
    alpha = z - float(centre)
    out = [xp.log(float(b) - z) - xp.log(float(a) - z)]
    for j in range(1, int(order) + 1):
        out.append(alpha * out[-1] + (t2**j - t1**j) / j)
    return xp.stack(out)


def cross_polynomial_moments(
    a: float, b: float, c1: float, c2: float, order1: int, order2: int
) -> NDArray:
    r"""``Q_ij = int_a^b (s - c1)^i (s - c2)^j ds``.

    With :math:`\tau = s - c_1` and :math:`\delta = c_1 - c_2`, expanding
    :math:`(\tau + \delta)^j` binomially leaves plain power integrals.
    Returns ``(order1 + 1, order2 + 1)``.
    """
    from math import comb
    t1, t2 = float(a) - float(c1), float(b) - float(c1)
    delta = float(c1) - float(c2)
    pw = [(t2 ** (k + 1) - t1 ** (k + 1)) / (k + 1)
          for k in range(int(order1) + int(order2) + 1)]
    out = xp.zeros((int(order1) + 1, int(order2) + 1), dtype=xp.complex128)
    for i in range(int(order1) + 1):
        for j in range(int(order2) + 1):
            out[i, j] = sum(comb(j, r) * delta ** (j - r) * pw[i + r]
                            for r in range(j + 1))
    return out


def pair_terms(
    leg_k: LocalLeg, leg_l: LocalLeg, a: float, b: float, omega: float
) -> tuple[NDArray, NDArray, NDArray]:
    r"""Exact :math:`\int_{[a,b]} \frac{du}{2\pi} B[G_k(u), G_l(\omega-u)]`."""
    two_pi = 2.0 * xp.pi
    omega = float(omega)
    ok, ol = leg_k.order, leg_l.order
    # partner poles re-expressed in u: 1/(v - xi) = -1/(u - (omega - xi))
    zeta_l_u = omega - xp.asarray(leg_l.zeta)
    res_l_u = -xp.asarray(leg_l.residues)
    # partner background in u: (v - c_l)^j = (-1)^j (u - (omega - c_l))^j
    centre_l_u = omega - leg_l.centre
    sign_l = xp.asarray([(-1.0) ** j for j in range(ol + 1)])

    cs, xs, ys = [], [], []

    # background x background
    q = cross_polynomial_moments(a, b, leg_k.centre, centre_l_u, ok, ol)
    for i in range(ok + 1):
        for j in range(ol + 1):
            cs.append(sign_l[j] * q[i, j] / two_pi)
            xs.append(leg_k.bg[i]); ys.append(leg_l.bg[j])

    # background_k x poles_l
    if zeta_l_u.shape[0]:
        r = resolvent_moments(zeta_l_u, a, b, leg_k.centre, ok)
        for i in range(ok + 1):
            for qq in range(zeta_l_u.shape[0]):
                cs.append(r[i, qq] / two_pi)
                xs.append(leg_k.bg[i]); ys.append(res_l_u[qq])

    # poles_k x background_l
    if leg_k.zeta.shape[0]:
        r = resolvent_moments(leg_k.zeta, a, b, centre_l_u, ol)
        for j in range(ol + 1):
            for pp in range(leg_k.zeta.shape[0]):
                cs.append(sign_l[j] * r[j, pp] / two_pi)
                xs.append(leg_k.residues[pp]); ys.append(leg_l.bg[j])

    # poles_k x poles_l -- the finite-cell kernel, already 1/(2 pi)-normalised
    if leg_k.zeta.shape[0] and leg_l.zeta.shape[0]:
        kern = pair_convolution(
            xp.asarray(leg_k.zeta)[:, None], xp.asarray(leg_l.zeta)[None, :],
            xp.asarray([omega]), window=(float(a), float(b)))[0]
        for pp in range(leg_k.zeta.shape[0]):
            for qq in range(leg_l.zeta.shape[0]):
                cs.append(kern[pp, qq])
                xs.append(leg_k.residues[pp]); ys.append(leg_l.residues[qq])

    return (xp.asarray(cs, dtype=xp.complex128),
            xp.stack(xs), xp.stack(ys))


def pair_correction(
    leg_k: LocalLeg, leg_l: LocalLeg, a: float, b: float, omega: float,
    g_k_point: NDArray, g_l_point: NDArray, h: float,
) -> tuple[NDArray, NDArray, NDArray]:
    r"""The replacement ``I_exact - I_rect`` for one cell pair, as a term list.

    ``g_k_point`` and ``g_l_point`` must be the SAME point samples the ring
    convolved, not the local models evaluated at the cell centres. Those differ
    in a pole cell by exactly the error being removed, and using the model here
    instead would leave the ring's contribution silently in place. It is the
    requirement :func:`~quatrex.phonon.experimental.pole.pole_congruence.pf_leg_sample` states for
    the global route -- the leg taken out and the leg put back have to be one
    function.
    """
    c, x, y = pair_terms(leg_k, leg_l, a, b, omega)
    rect = xp.asarray([-float(h) / (2.0 * xp.pi)], dtype=xp.complex128)
    return (xp.concatenate([c, rect]),
            xp.concatenate([x, xp.asarray(g_k_point)[None]]),
            xp.concatenate([y, xp.asarray(g_l_point)[None]]))


def contract(terms, bilinear=None) -> NDArray:
    """Sum a term list under a bilinear form; default is the matrix product.

    The default exists for the synthetic bed and the tests. A device path
    passes the projected cubic vertex.
    """
    c, x, y = terms
    if bilinear is None:
        return xp.einsum("t,tij,tjk->ik", c, x, y)
    return sum(ci * bilinear(xi, yi) for ci, xi, yi in zip(c, x, y))


def _swap(terms):
    """Exchange the two arguments of the bilinear form in a term list."""
    c, x, y = terms
    return c, y, x


def _centred_background(smooth, freqs, idx: int, h: float, poly_order: int):
    """Highest centred stencil that fits at ``idx``, and the half-width used.

    Near a grid edge the requested degree is reduced rather than the term
    dropped. Dropping is strictly worse: it leaves the ring's rectangle value
    in place, which is the error being removed, so a degree-0 background beats
    no correction at every width. Measured on the bed, skipping instead of
    degrading made degree 4 look WORSE than degree 0 (9.0e-02 against 4.1e-02)
    purely through the 8 terms it declined to touch.
    """
    n = int(smooth.shape[0])
    half = min((int(poly_order) + 1) // 2, int(idx), n - 1 - int(idx))
    sl = slice(int(idx) - half, int(idx) + half + 1)
    return background_from_samples(smooth[sl], float(freqs[idx]), h), half


def build_legs(
    freqs: NDArray, g: NDArray, zeta: NDArray, residues: NDArray,
    cells, poly_order: int = 2,
) -> dict:
    """One :class:`LocalLeg` per requested cell, off a deflated background.

    ``cells`` is any iterable of grid indices. Near a grid edge the stencil is
    narrowed rather than the cell dropped, see :func:`_centred_background`.
    """
    freqs = xp.asarray(freqs, dtype=float)
    h = float(freqs[1] - freqs[0])
    smooth = deflate(freqs, g, zeta, residues)
    out = {}
    for k in cells:
        bg, _ = _centred_background(smooth, freqs, int(k), h, poly_order)
        out[int(k)] = LocalLeg(bg, float(freqs[k]),
                               xp.asarray(zeta), xp.asarray(residues))
    return out


def correct_spectrum(
    freqs: NDArray,
    g: NDArray,
    pole_cells,
    zeta: NDArray,
    residues: NDArray,
    *,
    bilinear=None,
    poly_order: int = 2,
    radius: int = 1,
    rho_min: float = 0.0,
) -> tuple[NDArray, dict]:
    r"""The total correction to add to the ring's output.

        Parameters
        ----------
        freqs : NDArray
            ``(n_omega,)`` uniform grid; the ring's own axis.
        g : NDArray
            ``(n_omega,) + G.shape`` stored point samples, exactly as convolved.
        pole_cells : iterable
            Grid indices of the promoted cells.
        zeta, residues : NDArray
            The promoted pole set, shared by every cell.
        radius : int
            Cells on each side of a promoted cell to correct as well. The
            rectangle rule fails across an unresolved pole's TAIL, not only in the
            cell that holds it: a neighbouring cell sits one spacing from the pole,
            where the leg still varies by order one across the cell. Measured on
            the bed at ``gamma/h = 0.02``, the residual error against the exact
            cell integrals was

            ======= ========= =======
            radius  error     gain
            ======= ========= =======
            0       4.07e-02  30
            1       1.48e-03  819
            2       7.45e-04  1624
            3       7.10e-04  1705
            ======= ========= =======

            so 1 buys a factor 27 over correcting the pole cell alone and 2 has
            converged. The cost is linear in it.
        rho_min : float
            Output-resolution floor for pole-POLE cell pairs, see
            :func:`output_resolution`. DEFAULT 0, i.e. off, and the measurement is
            why.

            Refusing an unresolved pair leaves the ring's rectangle value in place,
            and that value is worst at exactly the outputs the gate protects: on
            the bed the refused set is the combination frequencies
            ``Omega = omega_p + omega_q`` and their immediate neighbours, where
            declining to correct costs a factor 7.34 while correcting costs
            2.7e-06. Away from them the gate changes nothing (1.99e-05 against
            2.72e-06). So the gate is strictly worse pointwise and buys nothing
            measurable.

            What ``rho_out`` measures is real, but it is not visible in a pointwise
            comparison: the corrected value is accurate AT each grid point even at
            ``rho_out = 0.08``; what is missing is the peak BETWEEN samples, which
            costs weight when the stored ``Sigma`` is later integrated or handed to
            Dyson. Refusing the correction does not restore that weight either.
            Treat ``report["rho_out"]`` as a diagnostic on whether the output grid
            can carry the answer, and reach for a rational output representation
            (review Sec. 24) when it says no -- not for this veto, which is kept
            only so the ablation can be rerun.

        Returns
        -------
        delta : NDArray
            Same shape as ``g``; add it to the ring's self-energy.
        report : dict
            ``n_corrected``, ``n_refused_rho``, ``n_degraded``, the full
            ``rho_out`` pole-pair matrix, and its minimum ``rho_worst``.
    """
    freqs = xp.asarray(freqs, dtype=float)
    n = int(freqs.shape[0])
    if n < 2:
        raise ValueError("correct_spectrum needs at least two frequencies.")
    h = float(freqs[1] - freqs[0])
    if not bool(xp.allclose(xp.diff(freqs), h, rtol=1e-10, atol=0.0)):
        raise ValueError(
            "correct_spectrum assumes the uniform ring grid; a non-uniform "
            "axis needs the cell edges partitioned by exact pair overlap "
            "first, which is not implemented.")
    promoted = sorted(int(k) for k in pole_cells)
    pole_cells = sorted({k + d for k in promoted
                         for d in range(-int(radius), int(radius) + 1)
                         if 0 <= k + d < n})
    legs = build_legs(freqs, g, zeta, residues, pole_cells, poly_order)
    want = (int(poly_order) + 1) // 2
    smooth = deflate(freqs, g, zeta, residues)

    delta = xp.zeros_like(xp.asarray(g, dtype=xp.complex128))
    n_ok, n_rho, n_deg = 0, 0, 0
    zt = xp.asarray(zeta)
    # cell independent: every cell carries the same pole set after deflation
    rho_pairs, _ = output_resolution(zt[:, None], zt[None, :], h)
    rho_worst = float(xp.min(rho_pairs)) if zt.shape[0] else float("inf")

    for m in range(n):
        touched = {k for k in pole_cells if 0 <= m - k < n}
        touched |= {m - j for j in pole_cells if 0 <= m - j < n}
        for k in sorted(touched):
            l = m - k
            inner, outer, swap = (k, l, False) if k in legs else (l, k, True)
            if outer in legs:
                if rho_worst < float(rho_min):
                    n_rho += 1
                    continue
                leg_out = legs[outer]
            else:
                bg, used = _centred_background(smooth, freqs, outer, h,
                                               poly_order)
                if used < want:
                    n_deg += 1
                leg_out = LocalLeg(bg, float(freqs[outer]), zt,
                                   xp.asarray(residues))
            a, b = float(freqs[inner]) - h / 2, float(freqs[inner]) + h / 2
            terms = pair_correction(legs[inner], leg_out, a, b, float(freqs[m]),
                                    g[inner], g[outer], h)
            if swap:
                terms = _swap(terms)
            delta[m] = delta[m] + contract(terms, bilinear)
            n_ok += 1

    return delta, {
        "n_corrected": n_ok,
        "n_refused_rho": n_rho,
        "n_degraded": n_deg,
        "rho_out": rho_pairs,
        "rho_worst": rho_worst,
    }


def output_resolution(
    zeta_p: NDArray, zeta_q: NDArray, h: float, omega_0: float = 0.0
) -> tuple[NDArray, NDArray]:
    r"""Whether a corrected pair's OUTPUT structure fits the output grid."""
    zp = xp.asarray(zeta_p, dtype=xp.complex128)
    zq = xp.asarray(zeta_q, dtype=xp.complex128)
    total = zp + zq
    same = (xp.imag(zp) * xp.imag(zq)) > 0.0
    rho = xp.where(
        same, 2.0 * xp.abs(xp.imag(zp) + xp.imag(zq)) / float(h), xp.inf)
    centre = xp.real(total)
    nearest = xp.round((centre - float(omega_0)) / float(h)) * float(h) + omega_0
    return rho, (centre - nearest) / float(h)
