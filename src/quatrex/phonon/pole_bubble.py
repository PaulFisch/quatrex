# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Analytic pole-pole three-phonon bubble, evaluated without a frequency grid.

The pole-pole sector of the cubic bubble,

.. math::
    \Sigma_{SS}^{\lessgtr}(\omega) = \frac{i\hbar}{2}
      \sum_{\alpha\beta\gamma\delta}
      \bar\Phi_{\mu,\alpha\beta}\,\bar\Phi^*_{\mu',\gamma\delta}
      \int\!\frac{d\omega'}{2\pi}
      F^{\lessgtr}_{\alpha\delta}(\omega')\,F^{\lessgtr}_{\beta\gamma}(\omega-\omega'),

is a convolution of *rational* functions, so it has a closed form. Writing the
modal Keldysh matrix of :mod:`quatrex.phonon.pole_keldysh`,

.. math::
    F^{\lessgtr}_{\alpha\beta}(\omega)
      = \frac{S^{\lessgtr}_{\alpha\beta}}{(\omega-z_\alpha)(\omega-z_\beta^*)},

in partial fractions turns every leg into a sum of **simple** poles, one in each
half plane. The elementary convolution of two simple poles then has only three
cases:

.. math::
    J(p,q;\omega) = \int\!\frac{d\omega'}{2\pi}
        \frac{1}{(\omega'-p)\,(\omega-\omega'-q)}
      = \begin{cases}
        -i/(\omega-p-q) & \operatorname{Im}p<0,\ \operatorname{Im}q<0\\
        +i/(\omega-p-q) & \operatorname{Im}p>0,\ \operatorname{Im}q>0\\
        0 & \text{mixed.}
        \end{cases}

The first line is the doc's Eq. (30): the convolution of two retarded poles is a
pole at the sum of their locations, which is where the three-phonon
:math:`\Omega_\alpha \pm \Omega_\beta` structures come from. The mixed case
vanishing is what makes the sum short -- only two of every four pole pairings
survive.

Two consequences worth stating separately:

* **No grid is involved.** The result is exact at any frequency, at a cost set
  by the number of pole pairs rather than by the sharpest linewidth. This is the
  whole point: a discrete convolution of two Lorentzians of half-width
  :math:`\gamma` needs :math:`\Delta\omega \ll \gamma` to be even approximately
  right, and at :math:`\Delta\omega/\gamma \sim 50` it does not merely lose
  weight, it *gains* it by two orders of magnitude depending on where the bins
  fall.
* **The retarded partner is free.** Because
  :math:`\Sigma^R(z) = \frac{i}{2\pi}\int \Delta(\omega')/(z-\omega')\,d\omega'`
  and :math:`\Delta_{SS}` is a sum of simple poles, closing the contour leaves

  .. math:: \Sigma^R_{SS}(\omega) = \sum_{j:\ \operatorname{Im}p_j<0}
            \frac{c_j}{\omega-p_j},

  i.e. **keep the lower-half-plane poles of** :math:`\Delta` **and drop the
  rest**. Manifestly causal, and it must not be passed through the numerical
  Hilbert transform (doc Sec. 36). As a check, a Lorentzian
  :math:`\Delta = L_{\Omega,\gamma}` has LHP coefficient :math:`+i` at
  :math:`\Omega-i\gamma`, reproducing :math:`i/(\omega-\Omega+i\gamma)`.

The pole set fed here must be closed under the bosonic partner map
:math:`z \mapsto -z^*` (doc Sec. 3.2), or the fold
:math:`\Sigma^<(-\omega) = \Sigma^>(\omega)^T` will not hold; see
:func:`bosonic_closure`.
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


def pair_convolution(p: NDArray, q: NDArray, omega: NDArray) -> NDArray:
    r"""Elementary convolution :math:`J(p,q;\omega)` of two simple poles.

    Parameters
    ----------
    p, q : NDArray
        Pole locations, broadcastable against each other.
    omega : NDArray
        ``(n_omega,)`` real frequencies; prepended as a leading axis.

    Returns
    -------
    NDArray
        ``(n_omega,) + broadcast(p, q).shape``.

    """
    p = xp.asarray(p, dtype=xp.complex128)
    q = xp.asarray(q, dtype=xp.complex128)
    w = xp.asarray(omega, dtype=xp.complex128).reshape(
        (-1,) + (1,) * max(p.ndim, q.ndim)
    )
    both_lower = (xp.imag(p) < 0) & (xp.imag(q) < 0)
    both_upper = (xp.imag(p) > 0) & (xp.imag(q) > 0)
    sign = xp.where(both_lower, -1j, xp.where(both_upper, 1j, 0.0))
    denom = w - p - q
    return sign / xp.where(denom == 0, 1.0, denom) * (denom != 0)


def leg_partial_fractions(
    cluster: PoleCluster, source: NDArray
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
    c = s / gap
    poles = xp.stack([xp.broadcast_to(za, gap.shape),
                      xp.broadcast_to(zb, gap.shape)], axis=-1)
    coeffs = xp.stack([c, -c], axis=-1)
    return poles, coeffs


def modal_convolution(
    omega: NDArray,
    cluster: PoleCluster,
    source_a: NDArray,
    source_b: NDArray,
    *,
    retarded_only: bool = False,
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
    pa, ca = leg_partial_fractions(cluster, source_a)   # (Np,Np,2)
    pb, cb = leg_partial_fractions(cluster, source_b)

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
            term = amp[None] * pair_convolution(p, q, omega)
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
