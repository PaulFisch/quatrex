# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Sigma_SR + Sigma_RS against a brute-force evaluation of the same ring.

The kernel integrates the narrow pole over each cell while the brute force
samples it, so the two agree only where sampling is accurate -- i.e. at
``gamma >> h``. That is exactly the regime in which to test INDEX BOOKKEEPING,
which is what can actually be wrong: which vertex leg carries the modal index,
and whether ``Sigma_RS`` is a genuinely different contraction from
``Sigma_SR`` (it is).

The measured advantage of the cell route at small ``gamma`` is established
separately in ``test_pole_mixed.py``.
"""
import numpy as np
import pytest

from quatrex.phonon.pole_bridge import (
    analytic_prefactor,
    mixed_self_energy_sparse,
    mixed_vertex_blocks,
)
from quatrex.phonon.pole_audit import transpose_index
from quatrex.phonon.pole_mixed import bosonic_extend
from quatrex.phonon.pole_keldysh import PoleCluster

SIZES = np.array([2, 2])
N_DOF = int(SIZES.sum())


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _pattern():
    off = np.concatenate(([0], np.cumsum(SIZES)))
    rows, cols = [], []
    for i in range(len(SIZES)):
        for j in range(max(0, i - 1), min(len(SIZES), i + 2)):
            for a in range(off[i], off[i + 1]):
                for b in range(off[j], off[j + 1]):
                    rows.append(a)
                    cols.append(b)
    return np.array(rows), np.array(cols)


def _bed(gamma, seed=0):
    rng = np.random.default_rng(seed)
    off = np.concatenate(([0], np.cumsum(SIZES)))
    phi = {}
    for i in range(2):
        for k1 in range(2):
            for k2 in range(2):
                phi[(i, k1, k2)] = rng.normal(size=(2, 2, 2))
    z = np.array([8.0 - 1j * gamma, 12.0 - 1j * gamma])
    u = rng.normal(size=(N_DOF, 2)) + 1j * rng.normal(size=(N_DOF, 2))
    v = rng.normal(size=(N_DOF, 2)) + 1j * rng.normal(size=(N_DOF, 2))
    cl = PoleCluster(z=z, u=u, v=v)
    a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    src = a @ a.conj().T
    return phi, cl, src, off


def _dense_gs(w, cl, src):
    """G_S(w) = U F(w) U^dagger, dense."""
    z = _h(cl.z)
    f = src[None] / ((w[:, None, None] - z[None, :, None])
                     * (w[:, None, None] - np.conj(z)[None, None, :]))
    u = _h(cl.u)
    return np.einsum("ia,wab,jb->wij", u, f, np.conj(u))


def _brute_ring(phi, off, a_leg, b_leg, w, rows, cols):
    """Direct evaluation of Phi_L[a,c,e] A[c,b] B[e,d] Phi_R[J,d,b]."""
    n, h = N_DOF, w[1] - w[0]
    dense = np.zeros((n, n, n))
    for (i, k1, k2), blk in phi.items():
        dense[off[i]:off[i + 1], off[k1]:off[k1 + 1], off[k2]:off[k2 + 1]] = blk
    # Discrete convolution over the grid. ``w`` must span the FULL axis: the
    # ring integral runs over all omega', and restricting it to omega' >= 0
    # drops roughly a quarter of the answer. The original version of this
    # reference was one-sided, which meant it agreed with a one-sided kernel --
    # two identical mistakes matching, and the defect that broke energy
    # conservation on the first device-scale run.
    conv = np.zeros((w.size, n, n, n, n), dtype=complex)   # [w, c, b, e, d]
    for iw, om in enumerate(w):
        idx = np.rint((om - w - w[0]) / h).astype(int)
        ok = (idx >= 0) & (idx < w.size)
        conv[iw] = np.einsum("kcb,ked->cbed", a_leg[ok], b_leg[idx[ok]]) * h
    out = np.einsum("ace,Jdb,wcbed->waJ", dense, dense, conv) / (2 * np.pi)
    return analytic_prefactor() * out[:, rows, cols]


def test_mixed_sectors_match_the_brute_force_ring():
    """Index bookkeeping against a direct ring, on the FULL frequency axis.

    The kernel is handed the background on ``[0, w_max]`` only -- exactly what
    the solver holds -- and must reconstruct the negative half from
    ``R(-w) = R(w)^*``. The brute force integrates the full axis directly. So
    this now tests the reconstruction as well as the leg conventions.

    The background is built conjugate-symmetric on purpose: a single Lorentzian
    does NOT satisfy the bosonic relation, and with a non-bosonic bed the test
    would measure the bed rather than the kernel.
    """
    gamma = 0.5
    h = 0.1
    w_pos = np.arange(0.0, 24.0 + 1e-9, h)         # gamma/h = 5: sampling is fine
    w_full = np.concatenate([-w_pos[:0:-1], w_pos])
    phi, cl, src, off = _bed(gamma)
    rows, cols = _pattern()
    rng = np.random.default_rng(3)

    # REAL: the scalar Lorentzian pair below is conjugate-symmetric, but a
    # complex matrix factor would break a(-w) = conj(a(w)) again.
    mat = rng.normal(size=(N_DOF, N_DOF))
    mat = mat + mat.T                     # symmetric, so the transpose is trivial

    def _pair(x):
        """A physical (G^<, G^>) pair: G^< = i n_B A, G^> = i(n_B+1) A, A odd.

        Built from the physics, NOT from the relation under test. An earlier
        version used a conjugate-symmetric background, which satisfies the
        WRONG mirror and therefore validated the wrong code.
        """
        kt = 25.0
        n_b = 1.0 / np.expm1(np.where(np.abs(x) < 1e-9, 1e-9, x) / kt)
        a = (4.0 / ((x - 9.0) ** 2 + 16.0) - 4.0 / ((x + 9.0) ** 2 + 16.0))
        return (np.einsum("w,ij->wij", 1j * n_b * a, mat),
                np.einsum("w,ij->wij", 1j * (n_b + 1.0) * a, mat))

    gl_full, gg_full = _pair(w_full)
    gl_neg, _ = _pair(-w_pos[1:])
    _, gg_pos = _pair(w_pos[1:])
    assert np.abs(gl_neg - gg_pos).max() < 1e-10, \
        "the bed must satisfy G^<(-w) = G^>(w)"

    g_reg_full = gl_full
    # The kernel gets ONLY the non-negative half plus the PARTNER, as
    # production does; it must rebuild the negative axis itself.
    gl_pos_v, gg_pos_v = (x[:, rows, cols] for x in _pair(w_pos))
    got = _h(mixed_self_energy_sparse(
        w_pos, cl, src, gl_pos_v, gg_pos_v, w_pos, phi, SIZES, rows, cols))

    gs_full = _dense_gs(w_full, cl, src)
    ref = (_brute_ring(phi, off, gs_full, g_reg_full, w_full, rows, cols)
           + _brute_ring(phi, off, g_reg_full, gs_full, w_full, rows, cols))
    # Compare on the positive half, away from the window edges.
    keep = (w_full > 4.0) & (w_full < 20.0)
    sel_pos = (w_pos > 4.0) & (w_pos < 20.0)
    err = (np.abs(got[sel_pos] - ref[keep]).max()
           / np.abs(ref[keep]).max())
    assert err < 5e-2, f"mixed sectors vs brute-force ring: {err:.3e}"


def test_rs_is_not_a_doubling_of_sr():
    """The two mixed sectors are genuinely different contractions."""
    from quatrex.phonon.pole_bridge import _mixed_one_sector

    gamma = 0.5
    w = np.arange(0.0, 24.0 + 1e-9, 0.1)
    phi, cl, src, off = _bed(gamma)
    rows, cols = _pattern()
    rng = np.random.default_rng(4)
    g_reg = np.einsum("w,k->wk", 1.0 / ((w - 9.0) ** 2 + 16.0),
                      rng.normal(size=rows.size) + 0j)

    kw = dict(freqs=w, rows=rows, cols=cols, g_partner=g_reg)
    mv = mixed_vertex_blocks
    sr = _h(_mixed_one_sector(
        w, cl, src, g_reg,
        bl=mv(phi, SIZES, cl.u, leg=0, conjugate=False),
        br=mv(phi, SIZES, cl.u, leg=1, conjugate=True), **kw))
    rs = _h(_mixed_one_sector(
        w, cl, src, g_reg,
        bl=mv(phi, SIZES, cl.u, leg=1, conjugate=False),
        br=mv(phi, SIZES, cl.u, leg=0, conjugate=True), **kw))

    diff = np.abs(sr - rs).max() / np.abs(sr).max()
    assert diff > 1e-2, (
        f"Sigma_RS came out equal to Sigma_SR ({diff:.3e}) -- the leg "
        "projections are not independent"
    )


def test_vertex_projection_reduces_the_requested_leg():
    phi, cl, src, off = _bed(0.5)
    u = _h(cl.u)
    dense = np.zeros((N_DOF, N_DOF, N_DOF))
    for (i, k1, k2), blk in phi.items():
        dense[off[i]:off[i + 1], off[k1]:off[k1 + 1], off[k2]:off[k2 + 1]] = blk

    got0 = _h(mixed_vertex_blocks(phi, SIZES, cl.u, leg=0, conjugate=False))
    assert np.abs(got0 - np.einsum("mab,aA->mbA", dense, u)).max() < 1e-12
    got1 = _h(mixed_vertex_blocks(phi, SIZES, cl.u, leg=1, conjugate=False))
    assert np.abs(got1 - np.einsum("mab,bA->maA", dense, u)).max() < 1e-12
    gotc = _h(mixed_vertex_blocks(phi, SIZES, cl.u, leg=1, conjugate=True))
    assert np.abs(gotc - np.einsum("mab,bA->maA", dense, np.conj(u))).max() < 1e-12


def test_oversized_pattern_is_refused():
    phi, cl, src, off = _bed(0.5)
    rows, cols = _pattern()
    w = np.arange(0.0, 4.0, 0.1)
    g_reg = np.zeros((w.size, rows.size), dtype=complex)
    with pytest.raises(NotImplementedError, match="exceeds the"):
        mixed_self_energy_sparse(w, cl, src, g_reg, g_reg, w, phi, SIZES,
                                 rows, cols, max_nnz=2)


def test_invalid_leg_raises():
    phi, cl, src, off = _bed(0.5)
    with pytest.raises(ValueError, match="leg must be"):
        mixed_vertex_blocks(phi, SIZES, cl.u, leg=2, conjugate=False)


def test_production_kernel_matches_an_explicit_ring():
    """The production path against an explicit ring, end to end.

    Everything else in this campaign tested the DECOMPOSITION (sector sum) or
    the kernel against another form of itself. This pins
    ``mixed_self_energy_blocked`` -- the routine the interaction actually calls
    -- against a direct evaluation of ``B(G_S,G_R) + B(G_R,G_S)``, with ``G_S``
    built as exactly the partial-fraction object the kernel represents.

    Run at ``gamma/h = 40`` so the discrete ring is itself accurate; the
    kernel's advantage at small ``gamma`` is measured in ``test_pole_mixed.py``
    and is not what is under test here.
    """
    from quatrex.phonon.pole_bridge import (analytic_prefactor,
                                            mixed_self_energy_blocked)

    sizes = np.array([2, 2])
    off = np.concatenate(([0], np.cumsum(sizes)))
    n = int(sizes.sum())
    rows, cols = _pattern()
    rng = np.random.default_rng(0)
    phi = {(i, k1, k2): rng.normal(size=(2, 2, 2))
           for i in range(2) for k1 in range(2) for k2 in range(2)}

    h = 0.05
    w_pos = np.arange(0.0, 60.0 + 1e-9, h)
    w = np.concatenate([-w_pos[:0:-1], w_pos])       # zero exactly on the grid
    z = np.array([8.0 - 2.0j, 12.0 - 2.0j])
    u = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
    cl = PoleCluster(z=z, u=u, v=u)
    s_a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    s_b = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))

    za, zb = z[:, None], np.conj(z)[None, :]
    gap = za - zb
    f = ((s_a / gap)[None] / (w[:, None, None] - za[None])
         - (s_b / gap)[None] / (w[:, None, None] - zb[None]))
    gs = np.einsum("ia,wab,jb->wij", u, f, np.conj(u))
    # SYMMETRIC: the bosonic relation carries an index transpose
    # (G^<_ij(-w) = G^>_ji(w)), so an asymmetric bed would make the reference
    # ambiguous about which of the two the negative axis should hold.
    mat = rng.normal(size=(n, n)); mat = mat + mat.T
    lor = 1.0 / (w - 9.0 + 6.0j) - 1.0 / (w + 9.0 + 6.0j)
    gr = np.einsum("w,ij->wij", lor, mat)

    dense = np.zeros((n, n, n))
    for (i, k1, k2), blk in phi.items():
        dense[off[i]:off[i + 1], off[k1]:off[k1 + 1], off[k2]:off[k2 + 1]] = blk

    def _ring(a_leg, b_leg):
        conv = np.zeros((w.size, n, n, n, n), dtype=complex)
        for iw, om in enumerate(w):
            idx = np.rint((om - w - w[0]) / h).astype(int)
            ok = (idx >= 0) & (idx < w.size)
            conv[iw] = np.einsum("kcb,ked->cbed", a_leg[ok], b_leg[idx[ok]]) * h
        return analytic_prefactor() * np.einsum(
            "ace,Jdb,wcbed->waJ", dense, dense, conv)[:, rows, cols] / (2 * np.pi)

    ref = _ring(gs, gr) + _ring(gr, gs)
    npos = w_pos.size
    # The kernel is handed the positive half plus the PARTNER, and must
    # rebuild the negative axis. The partner is by definition the background
    # at negative frequencies, indexed so that partner[j] = gr(-w_pos[j]).
    gr_pat = gr[:, rows, cols]
    gr_partner = gr_pat[npos - 1::-1]
    _chk, _ = bosonic_extend(gr_pat[-npos:], gr_partner, w_pos,
                             transpose_index=transpose_index(rows, cols))
    assert np.abs(_h(_chk) - gr_pat).max() < 1e-12, \
        "the partner must reconstruct the full-axis background exactly"
    got = _h(mixed_self_energy_blocked(
        w_pos, cl, (s_a, s_b), gr_pat[-npos:], gr_partner,
        w_pos, phi, sizes, rows, cols))
    sel = (w_pos > 3.0) & (w_pos < 17.0)
    keep = (w > 3.0) & (w < 17.0)
    err = np.abs(got[sel] - ref[keep]).max() / np.abs(ref[keep]).max()
    assert err < 1e-2, f"production kernel vs explicit ring: {err:.3e}"
