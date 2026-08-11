# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The four sectors are an identity of the retarded split, on a real pattern.

``test_pole_subcell.py`` establishes the identity on a 2x2 bed by hand. This
checks that the production coefficients -- which never form ``B^R_k``, and get
the ``SR`` bracket from the ``RS`` one by anti-Hermiticity of ``Sigma`` --
reproduce it for several poles, several dof, and a nontrivial sparsity pattern.
"""
import numpy as np
import pytest

from quatrex.phonon.pole_congruence import (
    background_coefficients, reconstruct, sector_grid_sample, sector_terms,
)
from quatrex.phonon.pole_keldysh import PoleCluster

N_DOF, N_P, N_W = 6, 3, 5


def _bed(seed=7):
    rng = np.random.default_rng(seed)

    def cx(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    z = rng.uniform(4.0, 10.0, N_P) - 1j * rng.uniform(0.01, 0.2, N_P)
    cl = PoleCluster(z=z, u=cx(N_DOF, N_P), v=cx(N_DOF, N_P), label="bed")
    w = np.linspace(3.0, 11.0, N_W)
    gk = cx(N_W, N_DOF, N_DOF)
    # -i Sigma Hermitian PSD  <=>  Sigma^dagger = -Sigma, which is the
    # relation background_coefficients uses to get the SR bracket for free.
    a = cx(N_W, N_DOF, N_DOF)
    psd = a @ np.conj(np.swapaxes(a, 1, 2))
    sig = 1j * psd
    assert np.abs(np.conj(np.swapaxes(sig, 1, 2)) + sig).max() < 1e-10
    return cl, w, gk, sig


def _coeffs(cl, w, gk, sig):
    v = np.asarray(cl.v)
    sv = sig @ v
    return background_coefficients(cl, w, sv, gk @ sv)


@pytest.mark.parametrize("seed", [7, 11])
def test_sectors_sum_to_the_congruence(seed):
    """``RR + SR + RS + SS`` is the congruence at ANY probe, not only at the
    cell centres -- that is what makes the reconstruction PSD off-centre."""
    cl, w, gk, sig = _bed(seed)
    co = _coeffs(cl, w, gk, sig)
    rows, cols = (np.repeat(np.arange(N_DOF), N_DOF),
                  np.tile(np.arange(N_DOF), N_DOF))
    u, v, z = np.asarray(cl.u), np.asarray(cl.v), np.asarray(cl.z)

    for frac in (0.0, 0.17, -0.4):
        probe = w + frac * (w[1] - w[0])
        sr, rs, ss = sector_terms(cl, w, co, rows, cols, probe=probe)
        for i in range(N_W):
            # RR = B^R_k Sigma B^A_k, formed densely only here
            b = gk[i] - (u * (1.0 / (w[i] - z))) @ v.conj().T
            rr = b @ sig[i] @ b.conj().T
            got = rr + (sr[i] + rs[i] + ss[i]).reshape(N_DOF, N_DOF)
            want = reconstruct(cl, w[i], probe[i], gk[i], sig[i])[0]
            assert np.abs(got - want).max() < 1e-8 * np.abs(want).max()


def test_grid_sample_is_what_the_ring_must_give_up():
    """At the centre the reconstruction is the untouched ring, so the sample
    removed from the ring's legs is exactly ``G^< - B^R_k Sigma B^A_k``."""
    cl, w, gk, sig = _bed()
    co = _coeffs(cl, w, gk, sig)
    rows, cols = (np.repeat(np.arange(N_DOF), N_DOF),
                  np.tile(np.arange(N_DOF), N_DOF))
    smp = sector_grid_sample(cl, w, co, rows, cols)
    u, v, z = np.asarray(cl.u), np.asarray(cl.v), np.asarray(cl.z)
    for i in range(N_W):
        g_lesser = gk[i] @ sig[i] @ gk[i].conj().T
        b = gk[i] - (u * (1.0 / (w[i] - z))) @ v.conj().T
        rr = b @ sig[i] @ b.conj().T
        assert np.abs(smp[i].reshape(N_DOF, N_DOF)
                      - (g_lesser - rr)).max() < 1e-8 * np.abs(g_lesser).max()


def test_reconstruction_is_psd_across_the_whole_cell():
    """The property the redesign exists for. ``-i G~^<`` is a congruence of a
    PSD matrix, so it is PSD at every probe -- with no accuracy demanded of the
    pole model, which the superseded form needed to 20 percent."""
    cl, w, gk, sig = _bed()
    h = w[1] - w[0]
    for i in range(N_W):
        for frac in np.linspace(-0.5, 0.5, 21):
            g = reconstruct(cl, w[i], w[i] + frac * h, gk[i], sig[i])[0]
            herm = -1j * g
            herm = 0.5 * (herm + herm.conj().T)
            ev = np.linalg.eigvalsh(herm)
            assert ev.min() / max(abs(ev).max(), 1e-300) > -1e-12


def test_sample_vanishes_without_poles_shifting_the_leg():
    """An empty correction reproduces the grid solver bit-for-bit: at the
    centre the pole supplies sub-cell structure and nothing else."""
    cl, w, gk, sig = _bed()
    co = _coeffs(cl, w, gk, sig)
    rows, cols = (np.repeat(np.arange(N_DOF), N_DOF),
                  np.tile(np.arange(N_DOF), N_DOF))
    sr, rs, ss = sector_terms(cl, w, co, rows, cols, probe=w)
    dense = [(sr[i] + rs[i] + ss[i]).reshape(N_DOF, N_DOF) for i in range(N_W)]
    for i in range(N_W):
        want = gk[i] @ sig[i] @ gk[i].conj().T
        got = reconstruct(cl, w[i], w[i], gk[i], sig[i])[0]
        assert np.abs(got - want).max() < 1e-9 * np.abs(want).max()
        assert np.abs(dense[i]).max() > 0.0


def test_apply_sparse_matches_a_dense_product():
    """The one primitive that touches the full operator. Duplicate row indices
    must accumulate, and the dense contact corners must be included -- without
    them ``G^R Sigma G^A`` is not ``G^<``."""
    from quatrex.phonon.pole_congruence import apply_sparse

    rng = np.random.default_rng(3)
    rows = np.array([0, 1, 1, 3, 4, 5, 2, 2])
    cols = np.array([1, 1, 4, 0, 5, 5, 2, 3])
    vals = rng.normal(size=(N_W, rows.size)) + 1j * rng.normal(size=(N_W, rows.size))
    v = rng.normal(size=(N_DOF, N_P)) + 1j * rng.normal(size=(N_DOF, N_P))
    corner = rng.normal(size=(N_W, 2, 2)) + 1j * rng.normal(size=(N_W, 2, 2))

    dense = np.zeros((N_W, N_DOF, N_DOF), dtype=complex)
    np.add.at(dense.transpose(1, 2, 0), (rows, cols), vals.T)
    dense[:, 4:6, 4:6] += corner

    got = apply_sparse(vals, rows, cols, v, N_DOF, corners=((corner, 4),))
    assert np.abs(np.asarray(got) - dense @ v).max() < 1e-12

    # ... and with a frequency-dependent right-hand side, which is what the
    # second apply G^R_k (Sigma V) needs.
    vw = np.stack([v * (1.0 + 0.3 * k) for k in range(N_W)])
    got = apply_sparse(vals, rows, cols, vw, N_DOF, corners=((corner, 4),))
    assert np.abs(np.asarray(got) - dense @ vw).max() < 1e-12


def test_cell_average_matches_quadrature():
    """The analytic cell weights against brute-force Gauss-Legendre."""
    from quatrex.phonon.pole_congruence import sector_cell_average, sector_terms

    cl, w, gk, sig = _bed()
    co = _coeffs(cl, w, gk, sig)
    rows, cols = (np.repeat(np.arange(N_DOF), N_DOF),
                  np.tile(np.arange(N_DOF), N_DOF))
    h = float(w[1] - w[0])
    got = np.asarray(sector_cell_average(cl, w, co, rows, cols, h))

    x, wt = np.polynomial.legendre.leggauss(160)
    ref = np.zeros_like(got)
    for xi, wi in zip(x, wt):
        probe = w + 0.5 * h * xi
        ref += 0.5 * wi * sum(
            np.asarray(t) for t in sector_terms(cl, w, co, rows, cols, probe))
    assert np.abs(got - ref).max() < 1e-9 * np.abs(ref).max()


def test_cell_averaged_leg_is_psd():
    """What the ring will consume. An average of PSD matrices is PSD, so the
    corrected leg cannot anti-damp however wrong the pole model is."""
    from quatrex.phonon.pole_congruence import sector_cell_average

    cl, w, gk, sig = _bed()
    co = _coeffs(cl, w, gk, sig)
    rows, cols = (np.repeat(np.arange(N_DOF), N_DOF),
                  np.tile(np.arange(N_DOF), N_DOF))
    h = float(w[1] - w[0])
    avg = np.asarray(sector_cell_average(cl, w, co, rows, cols, h))
    z, u, v = np.asarray(cl.z), np.asarray(cl.u), np.asarray(cl.v)
    for i in range(N_W):
        b = gk[i] - (u * (1.0 / (w[i] - z))) @ v.conj().T
        leg = b @ sig[i] @ b.conj().T + avg[i].reshape(N_DOF, N_DOF)
        herm = -1j * leg
        herm = 0.5 * (herm + herm.conj().T)
        ev = np.linalg.eigvalsh(herm)
        assert ev.min() / abs(ev).max() > -1e-10


def test_correction_vanishes_as_the_grid_resolves_the_line():
    """The pole channel is a DISCRETISATION correction, so it must go to zero
    on a grid fine enough to hold the line. If it did not, it would be adding
    physics the ring already has, and the two would double-count."""
    from quatrex.phonon.pole_congruence import (
        sector_cell_average, sector_grid_sample,
    )

    cl, w, gk, sig = _bed()
    co = _coeffs(cl, w, gk, sig)
    rows, cols = (np.repeat(np.arange(N_DOF), N_DOF),
                  np.tile(np.arange(N_DOF), N_DOF))
    smp = np.asarray(sector_grid_sample(cl, w, co, rows, cols))
    scale = np.abs(smp).max()
    prev = None
    for h in (0.4, 0.1, 0.025):
        avg = np.asarray(sector_cell_average(cl, w, co, rows, cols, h))
        err = np.abs(smp - avg).max() / scale
        if prev is not None:
            # second order in the cell width: halving h quarters the gap, so
            # a factor-4 refinement should gain about 16x
            assert err < prev / 8.0, f"h={h}: {err:.3e} vs {prev:.3e}"
        prev = err
    assert prev < 1e-3


def test_partial_fraction_legs_reproduce_the_sectors():
    """The flattening is algebra, not a model: with the SAME coefficients the
    2Np simple poles must give back SR + RS + SS at every frequency."""
    from quatrex.phonon.pole_congruence import (
        partial_fraction_legs, pf_leg_sample, sector_terms,
    )

    cl, w, gk, sig = _bed()
    co = _coeffs(cl, w, gk, sig)
    rows, cols = (np.repeat(np.arange(N_DOF), N_DOF),
                  np.tile(np.arange(N_DOF), N_DOF))
    for i in range(N_W):                      # freeze on cell i, probe anywhere
        frozen = (co[0][i], co[1][i], co[2][i])
        zeta, pr, qc = partial_fraction_legs(cl, frozen)
        assert zeta.shape[0] == 2 * N_P
        probe = np.array([w[0] - 0.7, w[2], w[-1] + 1.3, 0.5 * (w[1] + w[2])])
        got = np.asarray(pf_leg_sample(zeta, pr, qc, probe, rows, cols))
        # the same three sectors, evaluated directly with cell i's coefficients
        cof = tuple(np.repeat(c[i:i + 1], probe.size, axis=0) for c in co)
        want = sum(np.asarray(t) for t in
                   sector_terms(cl, probe, cof, rows, cols, probe=probe))
        assert np.abs(got - want).max() < 1e-9 * np.abs(want).max()


def _phi_bed(seed=5, sizes=(2, 2)):
    """A block-structured cubic vertex, and its dense equivalent."""
    rng = np.random.default_rng(seed)
    off = np.concatenate(([0], np.cumsum(sizes)))
    n = int(off[-1])
    blocks, dense = {}, np.zeros((n, n, n), dtype=complex)
    for i in range(len(sizes)):
        for k1 in range(len(sizes)):
            for k2 in range(len(sizes)):
                b = (rng.normal(size=(sizes[i], sizes[k1], sizes[k2]))
                     + 1j * rng.normal(size=(sizes[i], sizes[k1], sizes[k2])))
                blocks[(i, k1, k2)] = b
                dense[off[i]:off[i + 1], off[k1]:off[k1 + 1],
                      off[k2]:off[k2 + 1]] = b
    return blocks, dense, np.array(sizes), n


def test_pair_convolution_is_the_residue_formula():
    """Independent check of the closed form the analytic sector rests on.

    Closing the contour upward, the integrand ``1/(w'-p) 1/(w-w'-q)`` encloses
    only ``w' = w - q``, and only when ``p`` and ``q`` sit in the SAME half
    plane; otherwise both poles fall on the same side and the integral is zero.
    """
    from quatrex.phonon.pole_bubble import pair_convolution

    w = np.linspace(-3.0, 7.0, 11)
    lo = np.array([2.0 - 0.3j, 5.0 - 0.1j])
    for p in lo:
        for q in lo:
            got = np.asarray(pair_convolution(p, q, w))
            assert np.abs(got - (-1j / (w - p - q))).max() < 1e-12
            # conjugate pair: both in the upper half plane
            got = np.asarray(pair_convolution(np.conj(p), np.conj(q), w))
            assert np.abs(got - (1j / (w - np.conj(p) - np.conj(q)))).max() < 1e-12
            # mixed half planes: nothing is enclosed
            assert np.abs(np.asarray(pair_convolution(p, np.conj(q), w))).max() < 1e-12


def test_pf_self_energy_matches_a_dense_contraction():
    """The one index structure that cannot be checked by inspection.

    Builds the same object densely -- vertex, leg families and the residue
    formula -- and compares. A transposed vertex leg or a swapped (p, q) would
    survive every other test in this file.
    """
    from quatrex.phonon.pole_bridge import modal_vertex_blocks
    from quatrex.phonon.pole_congruence import pf_self_energy

    blocks, phi, sizes, n = _phi_bed()
    rng = np.random.default_rng(9)

    def cx(*s):
        return rng.normal(size=s) + 1j * rng.normal(size=s)

    npq = 4
    zeta = np.concatenate([cx(npq // 2).real + 2.0 - 0.2j * np.arange(1, 3)])
    zeta = np.concatenate([zeta, np.conj(zeta)])
    pr, qc = cx(n, zeta.size), cx(n, zeta.size)
    rows, cols = (np.repeat(np.arange(n), n), np.tile(np.arange(n), n))
    w = np.linspace(1.0, 9.0, 7)

    vl = np.asarray(modal_vertex_blocks(blocks, sizes, pr, conjugate=False))
    vr = np.asarray(modal_vertex_blocks(blocks, sizes, qc, conjugate=False))
    got = np.asarray(pf_self_energy(w, zeta, vl, vr, rows, cols,
                                    prefactor=1.0)).reshape(-1, n, n)

    # dense reference, built from the residue formula rather than the code
    j = np.where(
        (np.imag(zeta)[:, None] < 0) == (np.imag(zeta)[None, :] < 0),
        np.where(np.imag(zeta)[:, None] < 0, -1j, 1j)
        / (w[:, None, None] - zeta[None, :, None] - zeta[None, None, :]),
        0.0)
    ref = np.einsum("mce,ndb,cp,bp,eq,dq,wpq->wmn",
                    phi, phi, pr, qc, pr, qc, j)
    assert np.abs(got - ref).max() < 1e-9 * np.abs(ref).max()


def test_leg_tail_is_the_residue_sum_and_the_closure_kills_it():
    """The analytic leg is a GLOBAL function, so its tail is not cosmetic.

    ``sum_p p_p q_p^T`` IS the coefficient of the ``1/w`` tail. The true
    phonon ``G`` decays like ``1/w^2``, and a spurious ``1/w`` is what once
    made ``G_PP`` 17x too large at ``w = 1e2`` and cost four orders
    (``source_at_poles``). It vanishes exactly when ``sum_a u_a v_a^H = 0``,
    which is the sum rule the BOSONICALLY CLOSED pole set satisfies: the
    residue at ``-Omega`` cancels the one at ``+Omega``.

    That closure is not optional decoration -- it is what makes the analytic
    sector's tail legitimate.
    """
    from quatrex.phonon.pole_congruence import (
        background_coefficients, partial_fraction_legs, pf_leg_sample,
    )

    rng = np.random.default_rng(4)

    def cx(*s):
        return rng.normal(size=s) + 1j * rng.normal(size=s)

    n, om, gam = 4, 7.0, 0.05
    c, d = cx(n, 1), cx(n, 1)
    big = np.array([1e3, 1e4, 1e5])
    rows, cols = np.repeat(np.arange(n), n), np.tile(np.arange(n), n)
    w = np.linspace(1.0, 20.0, 24)
    a = cx(w.size, n, n)
    sig = 1j * (a @ np.conj(np.swapaxes(a, 1, 2)))
    gk = cx(w.size, n, n)

    def tail(z, u, v):
        cl = PoleCluster(z=z, u=u, v=v, label="t")
        co = background_coefficients(cl, w, sig @ v, gk @ (sig @ v))
        zeta, p, q = partial_fraction_legs(cl, tuple(x[0] for x in co))
        leg = np.asarray(pf_leg_sample(zeta, p, q, big, rows, cols))
        return np.abs(p @ q.T).max(), np.abs(leg).max(axis=1)

    # open: one pole, nothing to cancel against
    s_open, l_open = tail(np.array([om - 1j * gam]), c, d)
    # closed under z -> -conj(z), residues equal and opposite
    s_cl, l_cl = tail(np.array([om - 1j * gam, -om - 1j * gam]),
                      np.hstack([c, c]), np.hstack([d, -d]))

    # the residue sum IS the 1/w coefficient, to a few percent at w = 1e5
    assert abs(l_open[-1] * big[-1] - s_open) < 0.02 * s_open
    # ... and the closure removes it: the tail steepens from 1/w to 1/w^2
    assert s_cl < 1e-10 * s_open
    assert (l_cl[-1] * big[-1] ** 2) / (l_cl[0] * big[0] ** 2) < 1.05
    # w*|leg| is flat for the open set and falls like 1/w for the closed one
    assert (l_open[0] * big[0]) / (l_open[-1] * big[-1]) < 1.05
    assert (l_cl[0] * big[0]) / (l_cl[-1] * big[-1]) > 50.0


def test_pf_mixed_sectors_match_the_brute_force_ring():
    """The new mixed kernel against a direct evaluation of the same ring.

    Reuses the bed of ``test_pole_mixed_sectors``: a physical ``(G^<, G^>)``
    pair built from the physics rather than from the mirror under test, and a
    kernel handed only the non-negative half so it must rebuild the negative
    axis itself. The pole set is closed under ``z -> -conj(z)`` so the flat
    leg has no spurious ``1/w`` tail to contaminate the comparison.
    """
    import importlib.util
    import pathlib

    from quatrex.phonon.pole_congruence import pf_mixed_self_energy

    # the sibling module is not a package; load it by path
    _p = pathlib.Path(__file__).with_name("test_pole_mixed_sectors.py")
    _sp = importlib.util.spec_from_file_location("_pms", _p)
    m = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(m)

    # The reference must span well past the poles: it truncates the ring
    # integral at its grid edge while the kernel does the pole part
    # analytically over the whole axis. Measured on this bed, comparing on
    # [4, 20]: w_max = 24 gives 1.0e-01, 40 gives 1.4e-03, 64 gives 2.3e-04.
    # The disagreement is the REFERENCE's, and it converges away.
    h = 0.1
    w_pos = np.arange(0.0, 40.0 + 1e-9, h)
    w_full = np.concatenate([-w_pos[:0:-1], w_pos])
    phi, _, _, off = m._bed(0.5)
    rows, cols = m._pattern()
    rng = np.random.default_rng(17)
    nd = m.N_DOF

    # closed pole set: residues at -Omega cancel those at +Omega
    a = rng.normal(size=(nd, 2)) + 1j * rng.normal(size=(nd, 2))
    b = rng.normal(size=(nd, 2)) + 1j * rng.normal(size=(nd, 2))
    zc = np.array([8.0 - 0.5j, 12.0 - 0.5j])
    zeta = np.concatenate([zc, -np.conj(zc)])
    p_row = np.hstack([a, a])
    q_col = np.hstack([b, -b])
    assert np.abs(p_row @ q_col.T).max() < 1e-12

    mat = rng.normal(size=(nd, nd))
    mat = mat + mat.T

    def _pair(x):
        kt = 25.0
        n_b = 1.0 / np.expm1(np.where(np.abs(x) < 1e-9, 1e-9, x) / kt)
        aa = (4.0 / ((x - 9.0) ** 2 + 16.0) - 4.0 / ((x + 9.0) ** 2 + 16.0))
        return (np.einsum("w,ij->wij", 1j * n_b * aa, mat),
                np.einsum("w,ij->wij", 1j * (n_b + 1.0) * aa, mat))

    gl_full, _ = _pair(w_full)
    gl_pos_v, gg_pos_v = (x[:, rows, cols] for x in _pair(w_pos))

    got = np.asarray(pf_mixed_self_energy(
        w_pos, zeta, p_row, q_col, gl_pos_v, gg_pos_v, w_pos,
        phi, m.SIZES, rows, cols))

    d = 1.0 / (w_full[:, None] - zeta[None, :])
    gs_full = np.einsum("ip,wp,jp->wij", p_row, d, q_col)
    ref = (m._brute_ring(phi, off, gs_full, gl_full, w_full, rows, cols)
           + m._brute_ring(phi, off, gl_full, gs_full, w_full, rows, cols))

    keep = (w_full > 4.0) & (w_full < 20.0)
    sel = (w_pos > 4.0) & (w_pos < 20.0)
    err = np.abs(got[sel] - ref[keep]).max() / np.abs(ref[keep]).max()
    assert err < 5e-3, f"pf mixed sectors vs brute-force ring: {err:.3e}"
