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
    "coefficients_at_poles",
    "residue_sum",
    "fit_residual",
    "remainder_resolution",
    "partial_fraction_legs",
    "pf_leg_sample",
    "pf_self_energy",
    "pf_mixed_self_energy",
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


def _pattern_chunk(n_omega: int, n_p: int, nnz: int,
                   budget_bytes: int = 1 << 28) -> int:
    r"""Pattern rows per chunk, sized so the working set fits ``budget_bytes``.

    The natural way to write these contractions materialises
    ``(n_omega, N_p, nnz)`` -- ``take(c_sr, cols, axis=2)`` and its ``c_rs``
    partner. At the two poles the CNT bed promotes that is invisible; at a few
    dozen it is hundreds of gigabytes, and raising the Newton budget on that
    bed died with

        OutOfMemoryError: allocating 290,488,467,456 bytes

    so the route could never be asked to carry the pole count the physics
    needs. Chunking over the pattern makes the peak
    ``O(n_omega * N_p * chunk)`` with ``chunk`` set here rather than by the
    device size.

    The default budget is 256 MB of complex128, which keeps the chunk count in
    the hundreds for a production pattern rather than the thousands.
    """
    per = max(1, 16 * n_omega * max(n_p, 1))
    return int(max(1, min(nnz, budget_bytes // per)))


def sector_terms(
    cluster: PoleCluster,
    omega: NDArray,
    coefficients: tuple[NDArray, NDArray, NDArray],
    rows: NDArray,
    cols: NDArray,
    probe: NDArray | None = None,
    chunk_bytes: int = 1 << 28,
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
    n_w, n_p, nnz = int(d.shape[0]), int(d.shape[1]), int(r.shape[0])
    sr = xp.empty((n_w, nnz), dtype=xp.complex128)
    rs = xp.empty_like(sr)
    ss = xp.empty_like(sr)
    step = _pattern_chunk(n_w, n_p, nnz, budget_bytes=chunk_bytes)
    for lo in range(0, nnz, step):
        hi = min(lo + step, nnz)
        ur = xp.take(cluster.u, r[lo:hi], axis=0)             # (chunk, Np)
        uc = xp.conj(xp.take(cluster.u, c[lo:hi], axis=0))
        sr[:, lo:hi] = xp.einsum(
            "ka,wa,wak->wk", ur, d, xp.take(c_sr, c[lo:hi], axis=2))
        rs[:, lo:hi] = xp.einsum(
            "wka,wa,ka->wk", xp.take(c_rs, r[lo:hi], axis=1), xp.conj(d), uc)
        ss[:, lo:hi] = xp.einsum(
            "ka,wa,wab,wb,kb->wk", ur, d, c_ss, xp.conj(d), uc)
    return sr, rs, ss


def sector_grid_sample(
    cluster: PoleCluster,
    omega: NDArray,
    coefficients: tuple[NDArray, NDArray, NDArray],
    rows: NDArray,
    cols: NDArray,
    chunk_bytes: int = 1 << 28,
) -> NDArray:
    r"""``SR + RS + SS`` AT the cell centres, on the stored pattern.

    This is what must be removed from the ring's legs, so that what the ring
    keeps is :math:`B^R_k \Sigma B^A_k` -- the PSD congruence -- rather than
    the indefinite Keldysh remainder. The superseded code removed the ``SS``
    sample alone, dropping two terms that are FIRST order in the pole.
    """
    return sum(sector_terms(cluster, omega, coefficients, rows, cols,
                            chunk_bytes=chunk_bytes))


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
    chunk_bytes: int = 1 << 28,
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
    n_w, n_p, nnz = int(d1.shape[0]), int(d1.shape[1]), int(r.shape[0])
    out = xp.empty((n_w, nnz), dtype=xp.complex128)
    step = _pattern_chunk(n_w, n_p, nnz, budget_bytes=chunk_bytes)
    for lo in range(0, nnz, step):
        hi = min(lo + step, nnz)
        ur = xp.take(cluster.u, r[lo:hi], axis=0)             # (chunk, Np)
        uc = xp.conj(xp.take(cluster.u, c[lo:hi], axis=0))
        out[:, lo:hi] = (
            xp.einsum("ka,wa,wak->wk", ur, d1,
                      xp.take(c_sr, c[lo:hi], axis=2))
            + xp.einsum("wka,wa,ka->wk",
                        xp.take(c_rs, r[lo:hi], axis=1), xp.conj(d1), uc)
            + xp.einsum("ka,wab,wab,kb->wk", ur, d2, c_ss, uc))
    return out


def partial_fraction_legs(
    cluster: PoleCluster,
    coefficients: tuple[NDArray, NDArray, NDArray],
) -> tuple[NDArray, NDArray, NDArray]:
    r"""Flatten the congruence into simple poles with rank-1 residues.

    .. math::
        \tilde G(\omega) - B^R \Sigma B^A
            = \sum_p \frac{p_p\, q_p^{\mathsf T}}{\omega - \zeta_p},
        \qquad p = 1 \dots 2N_p,

    by partial-fractioning ``SS``'s double pole and collecting every term that
    shares a pole. For :math:`\zeta_p = z_a` the row factor is :math:`u_a` and
    everything else lands in the column factor; for :math:`\zeta_p = \bar z_b`
    it is the other way round:

    .. math::
        q_a = c^{SR}_{a,:} + \sum_b \frac{c^{SS}_{ab}}{z_a - \bar z_b}\,
              \bar u_b, \qquad
        y_b = c^{RS}_{:,b} - \sum_a \frac{c^{SS}_{ab}}{z_a - \bar z_b}\, u_a .

    This is what makes the analytic convolution affordable. A leg with an OPEN
    (non-modal) index would force the cubic vertex to be re-contracted at every
    frequency; here both families are fixed vectors, so the vertex is projected
    onto them once per iteration exactly as it is onto ``U`` today, and the
    bubble of two such legs is the existing pole-pole algebra over ``2 Np``
    poles instead of ``Np``.

    The coefficients must already be FROZEN -- one value per pole, not one per
    cell -- for the families to be frequency independent. That is the same
    approximation :func:`~quatrex.phonon.pole_bridge.source_at_poles` makes for
    the source, and it is better justified here: the poles have been removed
    from :math:`B^R` by construction, so it is the smooth part being sampled.

    Parameters
    ----------
    cluster : PoleCluster
    coefficients : tuple
        ``(c_sr, c_rs, c_ss)`` of shapes ``(Np, n_dof)``, ``(n_dof, Np)`` and
        ``(Np, Np)`` -- :func:`background_coefficients` output with the
        frequency axis already reduced to the poles.

    Returns
    -------
    zeta : NDArray
        ``(2 Np,)`` poles, the first ``Np`` retarded and the rest their
        conjugates.
    p_row, q_col : NDArray
        ``(n_dof, 2 Np)`` row and column families.

    """
    c_sr, c_rs, c_ss = (xp.asarray(a, dtype=xp.complex128)
                        for a in coefficients)
    z = xp.asarray(cluster.z)
    u, zb = cluster.u, xp.conj(z)
    gap = z[:, None] - zb[None, :]                            # (Np, Np)
    if bool(xp.any(xp.abs(gap) < 1e-300)):
        raise ValueError(
            "a pole coincides with a partner's conjugate; the simple-pole "
            "split is undefined there (defective/exceptional cluster).")
    w = c_sr + xp.einsum("ab,jb->aj", c_ss / gap, xp.conj(u))
    y = c_rs - xp.einsum("ab,ja->jb", c_ss / gap, u)
    zeta = xp.concatenate([z, zb])
    p_row = xp.concatenate([u, y], axis=1)
    q_col = xp.concatenate([xp.swapaxes(w, 0, 1), xp.conj(u)], axis=1)
    return zeta, p_row, q_col


def pf_leg_sample(
    zeta: NDArray,
    p_row: NDArray,
    q_col: NDArray,
    omega: NDArray,
    rows: NDArray,
    cols: NDArray,
) -> NDArray:
    r"""The partial-fraction leg on the stored pattern.

    Used for BOTH sides of the decomposition: this sample is what the ring
    gives up, and the analytic convolution of the same object is what is put
    back. ``G_reg = G - G_S`` is exact for any ``G_S``, but only if the leg
    subtracted and the leg restored are literally the same function -- using
    the per-cell coefficients on one side and the pole-frozen ones on the
    other is what once broke the spatial balance while leaving the scalar
    ``P_in = P_out`` nearly intact.
    """
    w = xp.asarray(omega, dtype=xp.complex128)[:, None]
    d = 1.0 / (w - xp.asarray(zeta)[None, :])                 # (n_omega, 2Np)
    pr = xp.take(p_row, xp.asarray(rows), axis=0)             # (nnz, 2Np)
    qc = xp.take(q_col, xp.asarray(cols), axis=0)
    return xp.einsum("kp,wp,kp->wk", pr, d, qc)


def pf_self_energy(
    omega: NDArray,
    zeta: NDArray,
    vl: NDArray,
    vr: NDArray,
    rows: NDArray,
    cols: NDArray,
    prefactor: complex | None = None,
    retarded_only: bool = False,
    cell=None,
    chunk: int = 1 << 17,
) -> NDArray:
    r"""Analytic bubble of two partial-fraction legs, on the stored pattern.

    .. math::
        \Sigma(\omega)_{\mu\nu} = \mathcal{P} \sum_{pq}
            \bar\Phi^{L}_{\mu, pq}\, \bar\Phi^{R}_{\nu, qp}\,
            \int\!\frac{d\omega'}{2\pi}
            \frac{1}{\omega' - \zeta_p}\,\frac{1}{\omega-\omega' - \zeta_q}

    -- the same algebra as the pole-pole sector, over ``2 Np`` simple poles
    with unit coefficients instead of ``Np`` poles carrying a source matrix.
    The residue coefficients live in the vertex projections, because the
    residues are rank one: ``vl`` is the cubic vertex projected onto the ROW
    family and ``vr`` onto the COLUMN family, each through
    :func:`~quatrex.phonon.pole_bridge.modal_vertex_blocks`.

    ``retarded_only`` keeps the pairings of two poles in the lower half plane,
    whose combined pole ``zeta_p + zeta_q`` also lies there. That is the causal
    part in closed form, with no Hilbert transform.

    Chunked over the pattern: ``vl[rows]`` is ``(nnz, 2Np, 2Np)``, four times
    the pole-pole sector's footprint, and at ``max_poles = 16`` that is 1024
    complex numbers per stored entry.
    """
    from quatrex.phonon.pole_bridge import analytic_prefactor
    from quatrex.phonon.pole_bubble import pair_convolution

    if prefactor is None:
        prefactor = analytic_prefactor()
    z = xp.asarray(zeta, dtype=xp.complex128)
    c = pair_convolution(z[:, None], z[None, :], omega, cell=cell)
    if retarded_only:
        keep = (xp.imag(z) < 0.0)
        c = c * (keep[None, :, None] & keep[None, None, :])
    r, cc = xp.asarray(rows), xp.asarray(cols)
    n = int(r.shape[0])
    out = xp.empty((c.shape[0], n), dtype=xp.complex128)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[:, s:e] = xp.einsum(
            "kpq,kqp,wpq->wk", xp.take(vl, r[s:e], axis=0),
            xp.take(vr, cc[s:e], axis=0), c)
    return prefactor * out


def pf_mixed_self_energy(
    omega: NDArray,
    zeta: NDArray,
    p_row: NDArray,
    q_col: NDArray,
    g_reg: NDArray,
    g_partner: NDArray,
    freqs: NDArray,
    phi_blocks: dict,
    block_sizes: NDArray,
    rows: NDArray,
    cols: NDArray,
    prefactor: complex | None = None,
) -> NDArray:
    r"""``Sigma_SR + Sigma_RS`` for a partial-fraction leg.

    The same block triple product as
    :func:`~quatrex.phonon.pole_bridge.mixed_self_energy_blocked`, with one
    change: the pole leg's row and column modes are no longer independent
    indices :math:`(\alpha, \delta)` over a single family, but a single index
    :math:`p` over TWO families, because every residue here is rank one. The
    double loop over ``(Np, Np)`` therefore collapses to a single loop over
    ``2 Np``, and the vertex is projected onto ``p_row`` on the left and
    ``q_col`` on the right rather than onto ``u`` and ``conj(u)``.

    ``q_col`` already carries whatever conjugation each pole needs -- ``conj u``
    at :math:`\bar z_b`, the ``SR`` bracket at :math:`z_a` -- so both
    projections are taken unconjugated.
    """
    from quatrex.phonon.pole_audit import transpose_index
    from quatrex.phonon.pole_bridge import (
        analytic_prefactor, block_offsets, blocks_from_pattern,
        mixed_vertex_block_dict, pattern_from_blocks,
    )
    from quatrex.phonon.pole_mixed import bosonic_extend, mixed_convolution_batched

    if prefactor is None:
        prefactor = analytic_prefactor()
    # The convolution runs over the WHOLE axis while the solver holds G only
    # for omega >= 0; the negative half comes from the PARTNER component,
    # transposed, not from conjugating this one.
    g_ext, f_ext = bosonic_extend(
        g_reg, g_partner, freqs, transpose_index=transpose_index(rows, cols))

    z = np.asarray(_h(zeta))
    n_p, n_omega = int(z.size), int(omega.shape[0])
    m_blocks = [
        blocks_from_pattern(
            mixed_convolution_batched(omega, complex(z[p]), 1.0 + 0.0j,
                                      g_ext, f_ext),
            rows, cols, block_sizes)
        for p in range(n_p)
    ]

    off = block_offsets(block_sizes)
    r = np.asarray(_h(rows), dtype=np.int64)
    c = np.asarray(_h(cols), dtype=np.int64)
    out_pairs = {(int(i), int(j)) for i, j in zip(
        np.searchsorted(off, r, side="right") - 1,
        np.searchsorted(off, c, side="right") - 1)}

    vd = mixed_vertex_block_dict
    total = None
    for legs in ((0, 1), (1, 0)):            # Sigma_SR, then Sigma_RS
        bl = vd(phi_blocks, block_sizes, p_row, leg=legs[0], conjugate=False)
        br = vd(phi_blocks, block_sizes, q_col, leg=legs[1], conjugate=False)
        acc: dict[tuple[int, int], NDArray] = {}
        for (i_out, j_out) in out_pairs:
            got = None
            for p in range(n_p):
                for (a, b), mab in m_blocks[p].items():
                    left, right = bl.get((i_out, a)), br.get((j_out, b))
                    if left is None or right is None:
                        continue
                    term = xp.einsum("ik,wkl,jl->wij",
                                     left[..., p], mab, right[..., p])
                    got = term if got is None else got + term
            if got is not None:
                acc[(i_out, j_out)] = got
        part = pattern_from_blocks(acc, rows, cols, block_sizes, n_omega,
                                   dtype=xp.complex128)
        total = part if total is None else total + part
    return prefactor * total


def coefficients_at_poles(
    cluster: PoleCluster,
    freqs: NDArray,
    coefficients: tuple[NDArray, NDArray, NDArray],
    order: int = 2,
    window: int = 4,
) -> tuple[NDArray, NDArray, NDArray]:
    r"""Freeze the sector coefficients at their own poles.

    The partial-fraction families are only frequency independent if the
    coefficients are, so they must be continued off the real axis exactly as
    the projected source is. ``c_ss`` goes through
    :func:`~quatrex.phonon.pole_bridge.source_at_poles`, which shares one value
    between the two residues of a pair -- that is what keeps ``c_a + c_b = 0``
    and the pole-pole leg decaying like ``1/w^2``.

    ``c_sr`` and ``c_rs`` have no pair structure: each belongs to a SINGLE
    pole, ``c_sr[a]`` to ``z_a`` and ``c_rs[:, b]`` to ``conj(z_b)``, so each
    is evaluated at its own. Their asymptotics are governed instead by
    ``sum_a u_a v_a^H = 0``, the sum rule the bosonically closed pole set
    satisfies, and :func:`residue_sum` is what measures it.
    """
    from quatrex.phonon.pole_bridge import source_at_poles
    from quatrex.phonon.pole_kernel import delta_local_fit

    c_sr, c_rs, c_ss = coefficients
    z = np.asarray(_h(cluster.z))
    npp = int(z.size)
    ss = source_at_poles(c_ss, freqs, cluster, order=order, window=window)

    def _at(flat, targets, shape):
        return xp.stack([
            delta_local_fit(flat, freqs, xp.asarray([t]), order=order,
                            window=window, anchor=float(np.real(t))
                            ).reshape(shape)
            for t in targets])

    n_dof = int(c_sr.shape[-1])
    sr = _at(c_sr.reshape(c_sr.shape[0], -1), [complex(x) for x in z],
             (npp, n_dof))
    rs = _at(xp.swapaxes(c_rs, 1, 2).reshape(c_rs.shape[0], -1),
             [complex(np.conj(x)) for x in z], (npp, n_dof))
    # each pole keeps only its own row/column
    sr = xp.stack([sr[a, a] for a in range(npp)])
    rs = xp.swapaxes(xp.stack([rs[b, b] for b in range(npp)]), 0, 1)
    return sr, rs, ss


def fit_residual(
    values: NDArray, freqs: NDArray, cluster: PoleCluster,
    order: int = 2, window: int = 4,
) -> float:
    r"""``eps_fit`` -- how well the local model reproduces its own samples.

    Doc Eq. 38. The analytic route needs the coefficient AT the pole, which no
    amount of grid data gives directly; it comes from a local polynomial model,
    and this is that model's residual against the data it was fitted to.

    Distinct from :func:`remainder_resolution`, and the two must not be
    conflated: a perfect fit whose regularized remainder the grid cannot carry
    is useless, and so is a resolvable remainder built on a bad fit.
    """
    from quatrex.phonon.pole_kernel import delta_local_fit

    w = np.asarray(_h(freqs), dtype=float)
    z = np.asarray(_h(cluster.z))
    v = np.asarray(_h(values))
    flat = v.reshape(v.shape[0], -1)
    worst = 0.0
    for b in range(int(z.size)):
        anchor = float(np.real(z[b]))
        k0 = int(np.argmin(np.abs(w - abs(anchor))))
        lo, hi = max(0, k0 - window), min(w.size, k0 + window + 1)
        if hi - lo < 2:
            continue
        model = np.asarray(_h(delta_local_fit(
            xp.asarray(flat), xp.asarray(w), xp.asarray(w[lo:hi] + 0j),
            order=order, window=window, anchor=anchor)))
        num = float(np.linalg.norm(model - flat[lo:hi]))
        den = float(np.linalg.norm(flat[lo:hi]))
        worst = max(worst, num / (den + 1e-300))
    return worst


def remainder_resolution(
    values: NDArray, frozen: NDArray, freqs: NDArray, cluster: PoleCluster,
    order: int = 2, window: int = 4, refine: int = 16,
) -> float:
    r"""``eps_reg,int`` -- is the REGULARIZED remainder carried by the grid?

    This replaces the coefficient-variation gate, which asked the wrong
    question. For :math:`F = c(\omega)/(\omega - z)` with locally analytic
    ``c``, the principal part is exactly :math:`c(z)/(\omega - z)` and

    .. math::
        F(\omega) - \frac{c(z)}{\omega - z}
            = \frac{c(\omega) - c(z)}{\omega - z}

    has a REMOVABLE singularity, tending to :math:`c'(z)`. So ``c`` varying
    across the pole window is not an error and not grounds for refusing the
    pole -- the variation lands in a regular function, and the only question is
    whether the grid can integrate that function. Measuring the variation
    instead (the retired ``coefficient_variation``) reported 0.908 on a bed
    where nothing was wrong with the principal-part split.

    Measured as doc Eq. 37: the grid's own midpoint rule over the pole's
    window against a ``refine``-times denser quadrature of the same interval,
    both applied to the local model of the remainder.
    """
    from quatrex.phonon.pole_kernel import delta_local_fit

    w = np.asarray(_h(freqs), dtype=float)
    z = np.asarray(_h(cluster.z))
    v = np.asarray(_h(values))                       # (n_omega, n_dof, Np)
    fr = np.asarray(_h(frozen))                      # (n_dof, Np)
    worst = 0.0
    for b in range(int(z.size)):
        zb = complex(np.conj(z[b]))
        anchor = float(np.real(z[b]))
        k0 = int(np.argmin(np.abs(w - abs(anchor))))
        lo, hi = max(0, k0 - window), min(w.size, k0 + window + 1)
        if hi - lo < 3:
            continue
        data = xp.asarray(v[:, :, b])                # (n_omega, n_dof)
        c_at_pole = fr[:, b].reshape(1, -1)

        def _rem(grid):
            model = np.asarray(_h(delta_local_fit(
                data, xp.asarray(w), xp.asarray(grid + 0j),
                order=order, window=window, anchor=anchor)))
            return (model.reshape(grid.size, -1) - c_at_pole) / (
                grid[:, None] - zb)

        g_c = w[lo:hi]
        g_f = np.linspace(g_c[0], g_c[-1], (hi - lo - 1) * refine + 1)
        # The grid's own rule -- a dw-weighted sum -- against a dense
        # trapezoid of the same interval. The gap IS the unresolved part.
        i_c = np.asarray(_rem(g_c)).sum(axis=0) * float(w[1] - w[0])
        i_f = np.trapezoid(np.asarray(_rem(g_f)), g_f, axis=0)
        den = float(np.linalg.norm(i_f))
        worst = max(worst, float(np.linalg.norm(i_c - i_f)) / (den + 1e-300))
    return worst


def residue_sum(p_row: NDArray, q_col: NDArray) -> float:
    r"""``max |sum_p p_p q_p^T|`` -- the coefficient of the leg's ``1/w`` tail.

    The analytic leg is a GLOBAL function, so this is not cosmetic. The true
    ``G`` decays like ``1/w^2``; a spurious ``1/w`` once made ``G_PP`` 17x too
    large at ``w = 1e2`` and cost four orders. It vanishes exactly when
    ``sum_a u_a v_a^H = 0``, which the bosonically closed pole set satisfies
    because the residue at ``-Omega`` cancels the one at ``+Omega``. A
    truncated or unclosed set does not, and this is the number that says so.
    """
    return float(xp.abs(p_row @ xp.swapaxes(q_col, 0, 1)).max())
