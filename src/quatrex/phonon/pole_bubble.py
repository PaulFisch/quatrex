# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Analytic pole-pole three-phonon bubble, evaluated without a frequency grid.

The pole-pole sector convolves two rational functions, so it has a closed form.
Partial-fractioning the modal Keldysh matrix turns every leg into a sum of
simple poles, one per half plane, and the elementary convolution of two simple
poles has only three cases: two retarded poles give a pole at the sum of their
locations, which is where the three-phonon sum and difference structures come
from; two advanced poles give the conjugate; a mixed pair vanishes.
"""
from __future__ import annotations

import numpy as np

from qttools import NDArray, xp

from quatrex.phonon.pole_keldysh import PoleCluster

__all__ = [
    "pair_convolution",
    "leg_partial_fractions",
    "modal_convolution",
    "ss_self_energy",
    "retarded_from_pole_sum",
    "bosonic_closure",
    "modal_vertex",
]


def pair_convolution(
    p: NDArray, q: NDArray, omega: NDArray, cell: float | None = None,
    window: tuple[float, float] | None = None,
) -> NDArray:
    r"""Elementary convolution :math:`J(p,q;\omega)` of two simple poles.

    Parameters
    ----------
    p, q : NDArray
        Pole locations, broadcastable against each other.
    omega : NDArray
        ``(n_omega,)`` real frequencies; prepended as a leading axis.

    window : tuple, optional
        ``(a, b)``: integrate over ``[a, b]`` instead of the whole axis,

        .. math::
            J_{[a,b]} = \frac{\log(b-p) - \log(a-p)
                              - \log(\omega-b-q) + \log(\omega-a-q)}
                             {2\pi\,(\omega - p - q)} .

        This is the production kernel of the LOCAL route
        (:mod:`quatrex.phonon.pole_local`), where ``[a, b]`` is one grid cell
        and the result replaces what the ring's rectangle rule put there.
        Verified against dense composite Gauss-Legendre on one cell, pole cell
        against a smooth partner, as relative error of the pair contribution:

        ======== ============ ================ ==========
        gamma/h  rectangle    cell-avg product  this
        ======== ============ ================ ==========
        0.400    5.14e-01     3.29e-01          2.5e-18
        0.100    1.77e+00     5.29e-01          4.7e-16
        0.020    9.41e+00     6.06e-01          1.8e-16
        0.005    3.82e+01     6.21e-01          2.2e-16
        0.001    1.92e+02     6.26e-01          2.2e-16
        ======== ============ ================ ==========

        The rectangle rule diverges as ``1/gamma`` and the product of cell
        averages saturates at an O(1) error (the missing piece is the subcell
        covariance); this form is exact at every width. Moving the pole from
        cell centre to edge at ``gamma/h = 0.02`` swings the rectangle error
        9.41 -> 0.27 -- the registration lottery -- while this stays at 1e-16.

        Note that ALL FOUR half-plane pairings are kept here. The mixed ones
        vanish only under whole-axis contour closure; on a finite interval
        they are a genuine part of the integral, and the residue form's zero
        is what would be wrong.

        As a GLOBAL replacement for the residue form it is instead a
        diagnostic, measuring the cost of the four sectors not acting on one
        common function: the residue form integrates the analytic leg over the
        whole axis while the mixed sectors and the ring only ever see the
        stored window. On a CNT-shaped pole pair (``z = 3 - 0.1i``,
        ``5 - 0.2i``, ``omega = 7``), truncating at ``+-100`` moves a
        same-half-plane pairing by 3.3e-03 relative and gives the
        opposite-half-plane pairing -- which the residue form sets to exactly
        zero -- a magnitude 3.3e-03 of it; at ``+-60`` both are 5.5e-03. So
        that inconsistency is a sub-percent effect at a realistic window, and
        it is NOT what makes the analytic route diverge.

        Mutually exclusive with ``cell`` -- the cell average of the log form
        has no elementary antiderivative.

    Returns
    -------
    NDArray
        ``(n_omega,) + broadcast(p, q).shape``.

    """
    if window is not None and cell:
        raise ValueError(
            "pair_convolution: window and cell are mutually exclusive; the "
            "cell average of the finite-window log form is not elementary.")
    p = xp.asarray(p, dtype=xp.complex128)
    q = xp.asarray(q, dtype=xp.complex128)
    w = xp.asarray(omega, dtype=xp.complex128).reshape(
        (-1,) + (1,) * max(p.ndim, q.ndim)
    )
    if window is not None:
        a, b = (float(window[0]), float(window[1]))
        s_val = w - p - q
        aa, bb = a - p, b - p
        # Two forms of the same integral, split on |s| against the distances
        # from the pole to the cell ENDPOINTS.
        #
        # s does vanish. It is NOT bounded below by gamma_p + gamma_q, which
        # only holds when both poles are in the same half plane: a flattened
        # family is [z, conj(z)] by construction, and for such a MIXED pairing
        # p + q = 2 Re z is real, so Im s = gamma_p - gamma_q = 0 and s passes
        # exactly through zero as Omega sweeps. The log form then divides a
        # cancelling bracket by it; the previous code returned exactly 0.0
        # there, which is silently wrong rather than loud.
        #
        # Writing Omega - u - q = s - (u - p) puts the bracket in the form
        # log1p(-s/A) - log1p(-s/B), whose term-by-term expansion is
        #   J = (1/2pi) sum_n s^n/(n+1) [B^-(n+1) - A^-(n+1)],
        # convergent for |s| < min(|A|, |B|) and exact at s = 0.
        # Convergence needs |s| < min_u |u - p| over the WHOLE path, not just
        # at the endpoints: for a pole INSIDE the cell -- which is the case the
        # sector exists for -- that minimum is |Im p|, and using the endpoint
        # distance instead selects the series where it diverges (measured:
        # +1.27 against a true -98.73).
        inside = ((xp.real(p) >= a) & (xp.real(p) <= b))
        reach = xp.where(inside, xp.abs(xp.imag(p)),
                         xp.minimum(xp.abs(aa), xp.abs(bb)))
        small = xp.abs(s_val) < 0.25 * reach
        num = (xp.log(bb) - xp.log(aa)
               - xp.log(s_val - bb) + xp.log(s_val - aa))
        den = xp.where(s_val == 0, 1.0, s_val)
        far = num / (2.0 * xp.pi * den) * (s_val != 0)

        near = xp.zeros_like(far)
        term_a, term_b = xp.ones_like(aa) / aa, xp.ones_like(bb) / bb
        for n in range(24):
            near = near + (s_val ** n / (n + 1.0)) * (term_b - term_a)
            term_a, term_b = term_a / aa, term_b / bb
        return xp.where(small, near / (2.0 * xp.pi), far)

    both_lower = (xp.imag(p) < 0) & (xp.imag(q) < 0)
    both_upper = (xp.imag(p) > 0) & (xp.imag(q) > 0)
    sign = xp.where(both_lower, -1j, xp.where(both_upper, 1j, 0.0))

    if cell:
        # CELL AVERAGE, not a point sample. The grid solver treats every array
        # as piecewise constant over its cell and integrates with weight dw;
        # handing it the value AT omega_m instead re-imports exactly the
        # registration error this sector exists to remove, at the interface.
        #
        # Measured, summing c/(w-p) over a grid and comparing with the exact
        # integral: point samples are wrong by 16.5 % at gamma/h = 0.4 and
        # 286 % at 0.08, while the cell average is exact to 1e-16 at every
        # width. Promoted poles have gamma/h < q_in = 1 by construction, so
        # the sector always operates in the regime where this matters.
        half = 0.5 * float(cell)
        pq = p + q
        return sign * (xp.log(w + half - pq) - xp.log(w - half - pq)) / float(cell)

    denom = w - p - q
    return sign / xp.where(denom == 0, 1.0, denom) * (denom != 0)


def leg_partial_fractions(
    cluster: PoleCluster, source: NDArray, *, source_b: NDArray | None = None,
    centre: float | None = None, scale: float | None = None,
) -> tuple[NDArray, NDArray]:
    r"""Split every modal leg into two simple poles.

    :math:`F_{\alpha\beta} = S_{\alpha\beta}/((\omega-z_\alpha)(\omega-z_\beta^*))
    = c/(\omega-z_\alpha) - c/(\omega-z_\beta^*)` with
    :math:`c = S_{\alpha\beta}/(z_\alpha - z_\beta^*)`.

    Parameters
    ----------
    cluster : PoleCluster
    source : NDArray
        ``(Np, Np)`` frozen projected source (doc Eq. 91).

    Returns
    -------
    poles : NDArray
        ``(Np, Np, 2)`` -- ``[..., 0]`` is ``z_alpha`` (lower half plane),
        ``[..., 1]`` is ``conj(z_beta)`` (upper).
    coeffs : NDArray
        ``(Np, Np, 2)`` matching coefficients ``(+c, -c)``.

    Raises
    ------
    ValueError
        If a pole coincides with a partner's conjugate, where the simple-pole
        split is undefined and a higher-order principal part is needed. The
        design note calls for a cluster/rational fallback there rather than a
        scalar pole label.

    """
    z = cluster.z
    s = xp.asarray(source, dtype=xp.complex128)
    za = z[:, None]
    zb = xp.conj(z)[None, :]
    gap = za - zb
    if bool(xp.any(xp.abs(gap) < 1e-300)):
        raise ValueError(
            "a pole coincides with a partner's conjugate; the simple-pole "
            "split is undefined there (defective/exceptional cluster)."
        )

    if source_b is not None:
        # Per-pole sources: the EXACT residues. Writing
        # S(w)/((w-z_a)(w-zb_b)), the residue at z_a is S(z_a)/gap and the one
        # at conj(z_b) is S(conj(z_b))/gap -- each leg carries the source at
        # ITS OWN pole, not one value shared by the whole cluster.
        #
        # This is what the frozen approximation gets wrong once a cluster holds
        # more than one pole, and badly wrong once it is closed under
        # z -> -z^*: the partner sits at -Omega while the frozen value is taken
        # at +Omega. A polynomial model in (w - Omega_c)/h cannot fix that
        # either, because it would EXTRAPOLATE across the whole band; the
        # source at a negative-frequency pole is the bosonic mirror of the
        # positive one, not a continuation of it.
        sb = xp.asarray(source_b, dtype=xp.complex128)
        if sb.shape != s.shape:
            raise ValueError(
                f"source_b must match source: {sb.shape} vs {s.shape}.")
        poles = xp.stack([xp.broadcast_to(za, gap.shape),
                          xp.broadcast_to(zb, gap.shape)], axis=-1)
        return poles, xp.stack([s / gap, -sb / gap], axis=-1)

    if s.ndim == 2:                       # frozen source: S constant in omega
        c = s / gap
    elif s.ndim == 3:                     # polynomial source: S = sum_m S_m t^m
        if centre is None or scale is None:
            raise ValueError(
                "a polynomial source needs the centre and scale of its fit "
                "variable t = (omega - centre)/scale."
            )
        n_moments = int(s.shape[0])
        if n_moments > 2:
            # t^m/((t-t_a)(t-t_b)) leaves a polynomial quotient of degree m-2,
            # which is NOT a simple pole and convolves to moments of the
            # background rather than to a resolvent. Refusing beats silently
            # dropping it: the dropped piece is the smooth part of the source,
            # so the answer would look reasonable and be wrong.
            raise NotImplementedError(
                f"a source of order {n_moments - 1} needs the degree-"
                f"{n_moments - 2} polynomial remainder of the partial-fraction "
                "split, which is not implemented. Orders 0 and 1 are exact "
                "here (the quotient is empty)."
            )
        ta = (za - complex(centre)) / float(scale)
        tb = (zb - complex(centre)) / float(scale)
        # Residue at z_alpha carries t_a^m, at conj(z_beta) carries t_b^m; both
        # over the SAME gap, so m = 0 reduces to the frozen expression exactly.
        ca = sum(s[m] * ta**m for m in range(n_moments)) / gap
        cb = sum(s[m] * tb**m for m in range(n_moments)) / gap
        poles = xp.stack([xp.broadcast_to(za, gap.shape),
                          xp.broadcast_to(zb, gap.shape)], axis=-1)
        return poles, xp.stack([ca, -cb], axis=-1)
    else:
        raise ValueError(
            f"source must be (Np, Np) or (n_moments, Np, Np), got {s.shape}."
        )

    poles = xp.stack([xp.broadcast_to(za, gap.shape),
                      xp.broadcast_to(zb, gap.shape)], axis=-1)
    coeffs = xp.stack([c, -c], axis=-1)
    return poles, coeffs


def _split_leg(cluster, src):
    """Partial-fraction one leg, accepting a per-pole source pair.

    ``src`` is either a single ``(Np, Np)`` matrix (the frozen source, kept for
    the analytic tests and for backwards compatibility) or a
    ``(S_at_row_pole, S_at_col_pole)`` pair from
    :func:`~quatrex.phonon.pole_bridge.source_at_poles`, which carries the
    exact residues.
    """
    if isinstance(src, tuple):
        return leg_partial_fractions(cluster, src[0], source_b=src[1])
    return leg_partial_fractions(cluster, src)


def modal_convolution(
    omega: NDArray,
    cluster: PoleCluster,
    source_a: NDArray,
    source_b: NDArray,
    *,
    retarded_only: bool = False,
    cell: float | None = None,
) -> NDArray:
    r"""The four-index modal convolution of doc Eq. (117).

    .. math::
        C_{\alpha\delta\beta\gamma}(\omega)
          = \int\!\frac{d\omega'}{2\pi}
            F^{(a)}_{\alpha\delta}(\omega')\,F^{(b)}_{\beta\gamma}(\omega-\omega')

    Parameters
    ----------
    omega : NDArray
        ``(n_omega,)`` real frequencies.
    cluster : PoleCluster
    source_a, source_b : NDArray
        ``(Np, Np)`` frozen sources of the two legs.
    retarded_only : bool, optional
        Keep only the pairings of two RETARDED poles. Those are exactly the
        terms whose combined pole ``p + q`` lands in the lower half plane, so
        this returns the causal part of the pole sum -- i.e. the Kramers-Kronig
        partner of the full result, in closed form and with no Hilbert
        transform. See :func:`retarded_from_pole_sum` for the same statement in
        its general form. Default ``False``.

    Returns
    -------
    NDArray
        ``(n_omega, Np, Np, Np, Np)`` indexed ``[w, alpha, delta, beta, gamma]``.

    """
    pa, ca = _split_leg(cluster, source_a)             # (Np,Np,2)
    pb, cb = _split_leg(cluster, source_b)

    # Sum over the 2x2 pole pairings; only like-half-plane pairs survive. Index
    # 0 of the partial fraction is z_alpha (retarded), index 1 is conj(z_beta)
    # (advanced), so retarded_only keeps the (0, 0) pairing alone.
    out = None
    pairs = [(0, 0)] if retarded_only else [(j, k) for j in range(2) for k in range(2)]
    for j, k in pairs:
        if True:
            p = pa[..., j][:, :, None, None]           # (a, d, 1, 1)
            q = pb[..., k][None, None, :, :]           # (1, 1, b, g)
            amp = (ca[..., j][:, :, None, None]
                   * cb[..., k][None, None, :, :])
            term = amp[None] * pair_convolution(p, q, omega, cell=cell)
            out = term if out is None else out + term
    return out


def modal_vertex(phi: NDArray, u: NDArray) -> NDArray:
    r"""Project the cubic vertex onto the modal basis, doc Eq. (116).

    .. math:: \bar\Phi_{\mu,\alpha\beta} = \sum_{ab}\Phi_{\mu ab}u_{a\alpha}u_{b\beta}

    Parameters
    ----------
    phi : NDArray
        ``(n_dof, n_dof, n_dof)`` cubic vertex.
    u : NDArray
        ``(n_dof, Np)`` right modal vectors.

    Returns
    -------
    NDArray
        ``(n_dof, Np, Np)``.

    """
    return xp.einsum("mab,aA,bB->mAB", phi, u, u)


def ss_self_energy(
    omega: NDArray,
    cluster: PoleCluster,
    source_a: NDArray,
    source_b: NDArray,
    phi_left: NDArray,
    phi_right: NDArray,
    prefactor: complex,
) -> NDArray:
    r"""Pole-pole self-energy :math:`\Sigma_{SS}(\omega)`, doc Eq. (117).

    The index placement mirrors the production ring
    (``phonon/docs/bubble_positivity.md`` Sec. 1): the second vertex is
    contracted with its leg pair **transposed**, ``Phi_R[J, d, b]``, which is the
    ordering that makes the contraction a congruence and carries the positivity
    statement. Passing the vertices in any other order silently destroys it.

    Parameters
    ----------
    omega : NDArray
        ``(n_omega,)`` real frequencies.
    cluster : PoleCluster
    source_a, source_b : NDArray
        ``(Np, Np)`` frozen projected sources of the two internal lines.
    phi_left : NDArray
        ``(n_dof, n_dof, n_dof)`` -- ``Phi_L[a, c, e]``.
    phi_right : NDArray
        ``(n_dof, n_dof, n_dof)`` -- ``Phi_R[J, d, b]``.
    prefactor : complex
        The bubble prefactor, ``i*hbar/2`` in the doc's normalisation; the
        production value is :func:`quatrex.phonon.units.bubble_prefactor_thz`
        divided by the grid spacing it carries (no ``dw`` here -- this term
        never touches a quadrature).

    Returns
    -------
    NDArray
        ``(n_omega, n_dof, n_dof)``.

    """
    u = cluster.u
    vl = xp.einsum("ace,cA,eB->aAB", phi_left, u, u)
    vr = xp.einsum("Jdb,dG,bD->JGD", phi_right, xp.conj(u), xp.conj(u))
    c = modal_convolution(omega, cluster, source_a, source_b)
    return prefactor * xp.einsum("aAB,JGD,wADBG->waJ", vl, vr, c)


def retarded_from_pole_sum(
    omega: NDArray, poles: NDArray, coeffs: NDArray
) -> NDArray:
    r"""Causal retarded partner of a self-energy given as a sum of simple poles.

    For :math:`\Delta(\omega) = \sum_j c_j/(\omega-p_j)`,

    .. math:: \Sigma^R(\omega) = \sum_{j:\ \operatorname{Im}p_j<0}
              \frac{c_j}{\omega-p_j},

    obtained by closing the Kramers-Kronig contour. The upper-half-plane poles
    of :math:`\Delta` drop out entirely. This replaces the numerical Hilbert
    transform for the analytic sector (doc Sec. 36) -- passing an analytic pole
    term through the discrete transform would both cost resolution and
    reintroduce the grid dependence the sector exists to remove.

    Parameters
    ----------
    omega : NDArray
        ``(n_omega,)`` real frequencies.
    poles : NDArray
        ``(...,)`` pole locations.
    coeffs : NDArray
        ``(...,)`` matching coefficients, broadcastable with ``poles``.

    Returns
    -------
    NDArray
        ``(n_omega,) + poles.shape``.

    """
    p = xp.asarray(poles, dtype=xp.complex128)
    c = xp.asarray(coeffs, dtype=xp.complex128)
    w = xp.asarray(omega, dtype=xp.complex128).reshape((-1,) + (1,) * p.ndim)
    keep = xp.imag(p) < 0
    return xp.where(keep, c / (w - p), 0.0)


def bosonic_closure(cluster: PoleCluster) -> PoleCluster:
    r"""Close a pole set under the bosonic partner map :math:`z \mapsto -z^*`.

    Retarded bosonic symmetry pairs every positive-frequency resonance with a
    negative-frequency partner. The bubble's fold
    :math:`\Sigma^<_{ij}(q,-\omega) = \Sigma^>_{ji}(-q,\omega)` only holds if the
    pole set used to build the legs is closed under that map, so this must be
    applied before the sector is contracted.

    The residue transformation is *not* a sign flip on the real part: it carries
    the conjugation inherited from :math:`G^R(-\omega) \leftrightarrow G^R(\omega)^*`
    (doc Sec. 3.2 warns about exactly this). For a real dynamical matrix at the
    Gamma point that is ``r -> conj(r)``, ``l -> conj(l)``; a q-resolved caller
    must additionally negate the transverse momentum, which is why this helper
    refuses to guess and takes the Gamma-only case only.

    Parameters
    ----------
    cluster : PoleCluster

    Returns
    -------
    PoleCluster
        With ``2 Np`` poles: the originals followed by their partners.

    """
    z = xp.concatenate([cluster.z, -xp.conj(cluster.z)])
    u = xp.concatenate([cluster.u, xp.conj(cluster.u)], axis=1)
    v = xp.concatenate([cluster.v, xp.conj(cluster.v)], axis=1)
    return PoleCluster(z=z, u=u, v=v, label=cluster.label + "+partner")
