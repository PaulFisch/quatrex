# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The leg the sectors should act on: a congruence, not a frozen remainder.

The pole split is applied to the RETARDED function, and the Keldysh components
follow from it:

.. math::
    \tilde G^R(\omega) = B^R_k + U D^R(\omega) V^\dagger, \qquad
    B^R_k = G^R(\omega_k) - U D^R(\omega_k) V^\dagger,

.. math::
    \tilde G^{\lessgtr}(\omega)
        = \tilde G^R(\omega)\, \Sigma^{\lessgtr}_k\, \tilde G^A(\omega),

with :math:`D^R_{\alpha\beta} = \delta_{\alpha\beta}/(\omega - z_\alpha)` and
:math:`B^R_k`, :math:`\Sigma_k` the stored grid samples, frozen across cell
``k``.

Two properties, and both are the reason for the rewrite.

:math:`-i\tilde G^{\lessgtr} \succeq 0` at **every** frequency, because it is a
congruence :math:`\tilde G^R (-i\Sigma) \tilde G^{R\dagger}` of a PSD matrix.
It needs no accuracy from the pole model: a residue wrong by a factor of two
gives a worse approximation but never a wrong sign. The superseded
reconstruction :math:`P^{\lessgtr}(\omega) + [G^{\lessgtr}(\omega_k) -
P^{\lessgtr}(\omega_k)]` inverts at a **20 %** residue error
(``test_pole_subcell.py``), and that inversion is the anti-damping.

The reconstruction is EXACT at the cell centre. Grouped the other way,
:math:`\tilde G^R(\omega) = G^R(\omega_k) + U[D^R(\omega) - D^R(\omega_k)]
V^\dagger`, the correction vanishes at :math:`\omega_k`, so an empty pole set
reproduces the grid solver bit-for-bit rather than approximately.

Expanding the congruence gives the four bubble sectors as an identity,

.. math::
    \tilde G^{\lessgtr} =
      \underbrace{B^R_k \Sigma B^A_k}_{RR}
    + \underbrace{U D^R(\omega)\,[V^\dagger \Sigma B^A_k]}_{SR}
    + \underbrace{[B^R_k \Sigma V]\, D^A(\omega) U^\dagger}_{RS}
    + \underbrace{U D^R(\omega)\,[V^\dagger \Sigma V]\, D^A(\omega)
                  U^\dagger}_{SS},

each in a category the existing quadratures already handle -- ``SS`` by
analytic residues, ``SR``/``RS`` by the cell resolvent, ``RR`` by the FFT
ring. The redesign changes the COEFFICIENTS the sectors carry, not the
quadratures, which review Sec. 9 showed were mutually consistent all along.

Against the superseded code the differences are that the regular leg is the
retarded background :math:`B^R_k` -- whose ring :math:`B^R_k \Sigma B^A_k` is
PSD by congruence, where :math:`G^{\lessgtr} - G_{PP}^{\lessgtr}` was
indefinite -- and that the sample subtracted from the ring must cover ``SR``
and ``RS`` too, not only ``SS``. Those two terms are FIRST order in the pole
while the retained one is second, so away from resonance they dominate exactly
what was being dropped.

See ``phonon/docs/pole_scba_divergence.md`` Sec. 3.3b.
"""
from __future__ import annotations

import numpy as np

from qttools import NDArray, xp

from quatrex.phonon.pole_keldysh import PoleCluster

__all__ = [
    "apply_sparse",
    "sector_terms",
    "background_coefficients",
    "sector_grid_sample",
    "sector_cell_average",
    "cell_weights",
    "reconstruct",
]


def apply_sparse(
    values: NDArray,
    rows: NDArray,
    cols: NDArray,
    v: NDArray,
    n_dof: int,
    corners: tuple = (),
) -> NDArray:
    r"""``out[w, i, a] = sum_j A[w, i, j] v[j, a]`` for ``A`` on the pattern.

    The congruence needs two such applies per Keldysh component,
    :math:`\Sigma V` and :math:`G^R (\Sigma V)`, and nothing else touches the
    full operator. Done one pole-column at a time so the intermediate is the
    size of the stored self-energy rather than ``N_p`` times it.

    Parameters
    ----------
    values : NDArray
        ``(n_omega, nnz)`` on the stored pattern.
    rows, cols : NDArray
        ``(nnz,)`` global indices.
    v : NDArray
        ``(n_dof, Np)`` dense columns, or ``(n_omega, n_dof, Np)`` when the
        right-hand side is itself frequency dependent -- which it is for the
        second apply, ``G^R_k (\Sigma V)``.
    n_dof : int
        Rows of the operator.
    corners : tuple
        ``(block, offset)`` pairs for operators held as dense corners rather
        than on the pattern -- the lead self-energies. Omitting them drops the
        injection that drives the device, and then ``G^R Sigma G^A`` is not
        ``G^<`` at all.

    """
    r, c = xp.asarray(rows), xp.asarray(cols)
    n_w, n_p = int(values.shape[0]), int(v.shape[-1])
    out = xp.zeros((n_w, n_dof, n_p), dtype=xp.complex128)
    for a in range(n_p):
        va = v[..., a]                                        # (n_dof,) | (w, n_dof)
        # (nnz,) broadcasts against (n_omega, nnz); (n_omega, nnz) is already
        # the frequency-dependent case, and both cost one self-energy.
        contrib = values * xp.take(va, c, axis=va.ndim - 1)
        tgt = out[:, :, a].T                                  # (n_dof, n_omega) view
        if xp is np:
            np.add.at(tgt, r, contrib.T)
        else:                                                 # no complex add.at
            import cupyx
            cupyx.scatter_add(tgt.real, r, contrib.T.real)
            cupyx.scatter_add(tgt.imag, r, contrib.T.imag)
    for block, off in corners:
        if block is None:
            continue
        b = int(block.shape[-1])
        vb = v[off:off + b] if v.ndim == 2 else v[:, off:off + b]
        out[:, off:off + b, :] += xp.einsum(
            "wij,ja->wia" if v.ndim == 2 else "wij,wja->wia", block, vb)
    return out


def background_coefficients(
    cluster: PoleCluster,
    omega: NDArray,
    sigma_v: NDArray,
    g_sigma_v: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    r"""Coefficients of the three analytic sectors, per frequency.

    The caller supplies the two contractions that need the full operator --
    :math:`\Sigma V` and :math:`G^R_k \Sigma V`, each one sparse apply against
    ``Np`` dense columns. Everything else is :math:`O(N_p^2)`.

    ``B^R_k`` is never formed. It enters only through

    .. math::
        B^R_k \Sigma V = G^R_k \Sigma V - U D^R(\omega_k)\,[V^\dagger \Sigma V],

    and the ``SR`` bracket follows from the ``RS`` one because
    :math:`\Sigma^{\lessgtr\dagger} = -\Sigma^{\lessgtr}` -- which is exactly
    the statement that :math:`-i\Sigma^{\lessgtr}` is Hermitian, the convention
    this solver runs in.

    Parameters
    ----------
    cluster : PoleCluster
        Poles ``z``, right vectors ``u``, left vectors ``v``.
    omega : NDArray
        ``(n_omega,)`` cell centres.
    sigma_v : NDArray
        ``(n_omega, n_dof, Np)``, :math:`\Sigma V`.
    g_sigma_v : NDArray
        ``(n_omega, n_dof, Np)``, :math:`G^R(\omega_k) \Sigma V`.

    Returns
    -------
    c_sr : NDArray
        ``(n_omega, Np, n_dof)``, :math:`V^\dagger \Sigma B^A_k`.
    c_rs : NDArray
        ``(n_omega, n_dof, Np)``, :math:`B^R_k \Sigma V`.
    c_ss : NDArray
        ``(n_omega, Np, Np)``, :math:`V^\dagger \Sigma V`.

    """
    z = xp.asarray(cluster.z)
    w = xp.asarray(omega, dtype=xp.complex128)
    d = 1.0 / (w[:, None] - z[None, :])                       # (n_omega, Np)
    c_ss = xp.einsum("ia,wib->wab", xp.conj(cluster.v), sigma_v)
    c_rs = g_sigma_v - xp.einsum("ia,wa,wab->wib", cluster.u, d, c_ss)
    c_sr = -xp.conj(xp.swapaxes(c_rs, 1, 2))
    return c_sr, c_rs, c_ss


def sector_terms(
    cluster: PoleCluster,
    omega: NDArray,
    coefficients: tuple[NDArray, NDArray, NDArray],
    rows: NDArray,
    cols: NDArray,
    probe: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    r"""``SR``, ``RS`` and ``SS`` on the stored pattern.

    ``probe`` is the frequency at which the POLE factors are evaluated; the
    coefficients stay frozen on their own cell. Passing ``None`` evaluates at
    the cell centres, which is the sample that must be removed from the ring's
    legs; passing a sub-cell probe gives the reconstruction between grid
    points, which is what the audit measures.
    """
    z = xp.asarray(cluster.z)
    wp = xp.asarray(omega if probe is None else probe, dtype=xp.complex128)
    d = 1.0 / (wp[:, None] - z[None, :])                      # (n_omega, Np)
    c_sr, c_rs, c_ss = coefficients
    r, c = xp.asarray(rows), xp.asarray(cols)
    ur = xp.take(cluster.u, r, axis=0)                        # (nnz, Np)
    uc = xp.conj(xp.take(cluster.u, c, axis=0))

    sr = xp.einsum("ka,wa,wak->wk", ur, d, xp.take(c_sr, c, axis=2))
    rs = xp.einsum("wka,wa,ka->wk", xp.take(c_rs, r, axis=1), xp.conj(d), uc)
    ss = xp.einsum("ka,wa,wab,wb,kb->wk", ur, d, c_ss, xp.conj(d), uc)
    return sr, rs, ss


def sector_grid_sample(
    cluster: PoleCluster,
    omega: NDArray,
    coefficients: tuple[NDArray, NDArray, NDArray],
    rows: NDArray,
    cols: NDArray,
) -> NDArray:
    r"""``SR + RS + SS`` AT the cell centres, on the stored pattern.

    This is what must be removed from the ring's legs, so that what the ring
    keeps is :math:`B^R_k \Sigma B^A_k` -- the PSD congruence -- rather than
    the indefinite Keldysh remainder. The superseded code removed the ``SS``
    sample alone, dropping two terms that are FIRST order in the pole.
    """
    return sum(sector_terms(cluster, omega, coefficients, rows, cols))


def reconstruct(
    cluster: PoleCluster,
    omega_k: float,
    probe: NDArray,
    g_retarded_k: NDArray,
    sigma_k: NDArray,
) -> NDArray:
    r"""The reconstruction densely, for audits and tests.

    ``G~^R(w) = G^R_k + U [D(w) - D(w_k)] V^H``, then ``G~^R Sigma G~^A``. The
    correction vanishes at ``w = omega_k``, so the leg is the untouched grid
    sample there.
    """
    z = np.asarray(_h(cluster.z))
    u, v = np.asarray(_h(cluster.u)), np.asarray(_h(cluster.v))
    w = np.atleast_1d(np.asarray(_h(probe)))
    gk, sk = np.asarray(_h(g_retarded_k)), np.asarray(_h(sigma_k))
    dk = 1.0 / (float(omega_k) - z)
    out = np.empty((w.size,) + gk.shape, dtype=complex)
    for i, wi in enumerate(w):
        g = gk + (u * (1.0 / (wi - z) - dk)) @ v.conj().T
        out[i] = g @ sk @ g.conj().T
    return out


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def cell_weights(
    cluster: PoleCluster, omega: NDArray, h
) -> tuple[NDArray, NDArray]:
    r"""Cell averages of the pole factors over ``[w_k - h/2, w_k + h/2]``.

    .. math::
        \langle D_a \rangle_k = \frac{1}{h}\big[
            \mathrm{Log}(\omega_k + h/2 - z_a)
          - \mathrm{Log}(\omega_k - h/2 - z_a)\big],

    .. math::
        \langle D_a \bar D_b \rangle_k =
            \frac{\langle D_a\rangle_k - \overline{\langle D_b\rangle_k}}
                 {z_a - \bar z_b},

    the second by partial fractions. The poles sit off the real axis, so the
    difference of Logs has no branch ambiguity: the ``2 pi i`` cancels.

    Returns ``(d1, d2)`` of shapes ``(n_omega, Np)`` and
    ``(n_omega, Np, Np)``.
    """
    z = xp.asarray(cluster.z)
    w = xp.asarray(omega, dtype=xp.complex128)[:, None]
    # per-bin widths, because the frequency grid need not be uniform
    hh = xp.asarray(h, dtype=float).reshape(-1, 1)
    if bool(xp.all(hh <= 0.0)):
        d1 = 1.0 / (w - z[None, :])
        return d1, d1[:, :, None] * xp.conj(d1)[:, None, :]
    d1 = (xp.log(w + 0.5 * hh - z[None, :])
          - xp.log(w - 0.5 * hh - z[None, :])) / hh
    gap = z[:, None] - xp.conj(z)[None, :]                    # (Np, Np)
    # z_a - conj(z_b) has imaginary part -(gamma_a + gamma_b) < 0, so it never
    # vanishes: the retarded poles are strictly in the lower half plane and
    # their conjugates strictly in the upper one.
    d2 = (d1[:, :, None] - xp.conj(d1)[:, None, :]) / gap[None]
    return d1, d2


def sector_cell_average(
    cluster: PoleCluster,
    omega: NDArray,
    coefficients: tuple[NDArray, NDArray, NDArray],
    rows: NDArray,
    cols: NDArray,
    h,
) -> NDArray:
    r"""``<SR + RS + SS>_k`` -- the analytic sectors averaged over their cell.

    The bubble integrates its legs with weight ``dw``, so what it wants is the
    cell average, and for an under-resolved pole that differs from the point
    sample by order one (measured 8.2e-01 relative on the ``h = 20 gamma``
    bed). Averaging the RECONSTRUCTION rather than resampling it keeps the
    result PSD: an average of PSD matrices is PSD, so no accuracy is demanded
    of the pole model for the sign to come out right.
    """
    c_sr, c_rs, c_ss = coefficients
    d1, d2 = cell_weights(cluster, omega, h)
    r, c = xp.asarray(rows), xp.asarray(cols)
    ur = xp.take(cluster.u, r, axis=0)                        # (nnz, Np)
    uc = xp.conj(xp.take(cluster.u, c, axis=0))
    sr = xp.einsum("ka,wa,wak->wk", ur, d1, xp.take(c_sr, c, axis=2))
    rs = xp.einsum("wka,wa,ka->wk", xp.take(c_rs, r, axis=1), xp.conj(d1), uc)
    ss = xp.einsum("ka,wab,wab,kb->wk", ur, d2, c_ss, uc)
    return sr + rs + ss
