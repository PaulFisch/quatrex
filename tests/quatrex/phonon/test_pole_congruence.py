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


def test_interaction_class_keeps_its_pole_hook():
    """A module-level helper inserted into the class body once truncated it,
    moving _inject_pole_sector out of the class. Every local test passed --
    none of them call it -- and three device runs died on the attribute."""
    from quatrex.core.interaction import PhononPhononInteraction

    for name in ("compute", "_inject_pole_sector"):
        assert callable(getattr(PhononPhononInteraction, name, None)), name


# --- how the analytic route's two halves reach the bubble -------------------- #

class _Buf:
    def __init__(self, data, rows=None, cols=None):
        self.data = data
        self.rows, self.cols = rows, cols


def _analytic_harness(monkeypatch, mixed_scale=1.0, low_freq_mask=0.0):
    """Drive ``_pole_analytic_sectors`` with stubbed kernels.

    The kernels are covered by their own tests; what is under test here is the
    ASSEMBLY -- which hook each half goes through, and what the mixed sector is
    handed. Both are invisible to a kernel test and both cost a device run.
    """
    from quatrex.core import interaction as I
    from quatrex.phonon import pole_congruence as PC

    n_w, nnz = 6, 4
    freqs = np.linspace(0.0, 5.0, n_w)
    rows = cols = np.arange(nnz)
    shape = (n_w, nnz)
    ones = np.ones(shape, dtype=complex)

    seen = {}
    monkeypatch.setattr(PC, "pf_self_energy",
                        lambda *a, retarded_only=False, **k:
                        (3.0 if retarded_only else 1.0) * ones)

    def _fake_mixed(omega, zeta, p_row, q_col, g_reg, g_partner, *a, **k):
        seen.setdefault("reg", []).append(np.asarray(g_reg).copy())
        return 10.0 * ones

    monkeypatch.setattr(PC, "pf_mixed_self_energy", _fake_mixed)
    monkeypatch.setattr(I, "data_rows_cols", lambda scba: (rows, cols))
    monkeypatch.setattr(
        "quatrex.phonon.pole_bridge.modal_vertex_blocks",
        lambda *a, **k: np.zeros((nnz, 2, 2), dtype=complex))

    class _PS:
        cell_average, mixed_scale = True, 1.0
    _PS.mixed_scale = mixed_scale

    class _SSP:
        phi_blocks, block_sizes = {}, np.array([nnz])
        _low_freq_mask = low_freq_mask

        def set_pole_self_energy(self, l, g, r):
            seen["ss"] = (np.asarray(l), np.asarray(g), np.asarray(r))

        def set_pole_mixed(self, l, g):
            seen["mx"] = (np.asarray(l), np.asarray(g))

    class _Solver:
        local_frequencies = freqs

    class _Cfg:
        class phonon:
            pole_sector = _PS()

    class _Data:
        sigma_lesser = _Buf(np.zeros(shape, dtype=complex), rows, cols)
        g_lesser = _Buf(2.0 * ones)
        g_greater = _Buf(5.0 * ones)

    class _Scba:
        phonon_solver, config, data = _Solver(), _Cfg(), _Data()

    class _State:
        g_pp_lesser, g_pp_greater = 1.0 * ones, 1.0 * ones
        zz = (np.array([1.0 - 0.1j]), np.zeros((nnz, 1), dtype=complex),
              np.zeros((nnz, 1), dtype=complex))
        pf_lesser, pf_greater = [zz], [zz]

    I._pole_analytic_sectors(_Scba(), _State(), _SSP())
    return seen


def test_analytic_mixed_sector_goes_through_the_hilbert_hook(monkeypatch):
    """``SR + RS`` must reach ``set_pole_mixed``, not ``set_pole_self_energy``.

    ``set_pole_self_energy`` lands AFTER ``delta = sigma^> - sigma^<`` is
    formed, so anything routed there is invisible to the Kramers-Kronig
    transform and must supply its own causal partner. ``SS`` can (the
    two-retarded pairing); the mixed sector cannot, because one of its legs is
    the numerical background. Folding it into the ``SS`` accumulator leaves
    Sigma^R without the dispersive half of a term Sigma^{<,>} has in full --
    a fluctuation-dissipation break, measured as ``lead balance = 2.0000``
    (job 4398805).
    """
    seen = _analytic_harness(monkeypatch)
    assert "mx" in seen, "the mixed sector never reached set_pole_mixed"
    assert np.allclose(seen["mx"][0], 10.0), "mixed payload not the mixed term"
    # ... and the SS hook carries SS ALONE: 1.0, not 1.0 + 10.0.
    assert np.allclose(seen["ss"][0], 1.0), (
        "set_pole_self_energy carries the mixed term too; its retarded partner "
        "(the two-retarded pairing) is only the pole-pole one")
    # kk_half = acc_r - 0.5*(acc_g - acc_l), with acc_r = rr_g - rr_l = 0 here
    # because both components stub to the same value: the point is that it is
    # built from the SS accumulators, which now exclude mx.
    assert np.allclose(seen["ss"][2], 0.0)


def test_analytic_mixed_sector_masks_the_background_leg(monkeypatch):
    """The same low-frequency mask the ring applies to its own legs.

    The omega = 0 bin carries the near-singular acoustic peak and the ring
    excludes it; an unmasked background leg makes the two sectors integrate
    different data and injects that peak into Sigma. Measured on the
    ``rr_ss_sr`` route: Sigma^> non-PSD by 0.15, strictly LINEAR in the
    injected mixed term.
    """
    seen = _analytic_harness(monkeypatch, low_freq_mask=1.5)
    regs = seen["reg"]
    assert regs, "the mixed kernel was never called"
    for r in regs:
        assert np.abs(r[:2]).max() == 0.0, "background leg not masked below 1.5"
        assert np.abs(r[2:]).max() > 0.0, "the whole leg was masked"


def test_analytic_route_honours_mixed_scale(monkeypatch):
    """``mixed_scale`` is the bisection knob that separates 'too large' from
    'wrongly signed'. Ignoring it on this route makes it silently inert."""
    a = _analytic_harness(monkeypatch, mixed_scale=1.0)["mx"][0]
    b = _analytic_harness(monkeypatch, mixed_scale=0.0)["mx"][0]
    assert np.abs(a).max() > 0.0
    assert np.abs(b).max() == 0.0


# --- finite support: how far apart are the four sectors' quadratures? -------- #

def test_finite_window_kernel_matches_quadrature_and_its_residue_limit():
    """The four sectors do not integrate over the same axis, and this bounds it.

    ``pf_self_energy`` uses the residue kernel, which integrates the analytic
    leg over ``(-inf, inf)``; the mixed sectors and the ring only ever see the
    stored window. The finite-window kernel is the same integral over
    ``[a, b]``, so the gap between them IS the inconsistency.
    """
    from scipy.integrate import quad

    from quatrex.phonon.pole_bubble import pair_convolution

    w = np.array([7.0])

    def kern(p, q, window=None):
        return pair_convolution(np.array([p]), np.array([q]), w,
                                window=window)[0, 0]

    def numeric(p, q, a, b):
        f = lambda u: 1.0 / ((u - p) * (w[0] - u - q))
        re = quad(lambda u: f(u).real, a, b, limit=400)[0]
        im = quad(lambda u: f(u).imag, a, b, limit=400)[0]
        return (re + 1j * im) / (2 * np.pi)

    pairs = [(3 - 0.1j, 5 - 0.2j),      # both lower
             (3 + 0.1j, 5 + 0.2j),      # both upper
             (3 - 0.1j, -5 + 0.2j)]     # mixed -- residue form gives 0
    # One common scale: the mixed pairing's residue value is exactly zero, so
    # a relative tolerance on it would be a tolerance on nothing.
    scale = abs(kern(*pairs[0]))
    for p, q in pairs:
        got, ref = kern(p, q, (-60.0, 60.0)), numeric(p, q, -60.0, 60.0)
        assert abs(got - ref) <= 1e-10 * scale, (p, q, got, ref)
        # ... and it must reproduce the residue answer as the window opens
        assert abs(kern(p, q, (-1e7, 1e7)) - kern(p, q)) < 1e-6 * scale, (p, q)

    # The number that matters: at a realistic omega_max this is sub-percent,
    # so finite support is NOT what makes the analytic route diverge.
    same = kern(*pairs[0])
    for R, tol in ((60.0, 6e-3), (100.0, 4e-3)):
        assert abs(kern(*pairs[0], window=(-R, R)) - same) / abs(same) < tol
        assert abs(kern(*pairs[2], window=(-R, R))) / abs(same) < tol


def test_window_and_cell_average_are_refused_together():
    """The cell average of the log form has no elementary antiderivative, so
    silently ignoring one of the two would emit a kernel that is neither."""
    from quatrex.phonon.pole_bubble import pair_convolution

    with pytest.raises(ValueError, match="mutually exclusive"):
        pair_convolution(np.array([1 - 1j]), np.array([2 - 1j]),
                         np.array([0.0]), cell=0.5, window=(-10.0, 10.0))


def test_analytic_sectors_sum_to_the_bubble_of_the_same_hybrid():
    """Review Sec. 24, the gate that was missing entirely.

    ``SS + SR + RS + RR`` must reproduce ``B(Ghat, Ghat)`` where

        Ghat(w) = G_S(w) + R_h(w)

    is the hybrid the sector kernels actually assume -- the analytic leg plus
    the piecewise-constant remainder -- and NOT the physical ``G``. Comparing
    against ``G`` would fold representation error into the same number as
    implementation error, and the whole point of the gate is to separate them.

    ``B(Ghat, Ghat) = B(G_S,G_S) + B(G_S,R_h) + B(R_h,G_S) + B(R_h,R_h)`` is
    an identity of a bilinear form, so any residual here is a defect: a
    transposed vertex leg, a swapped ``(p, q)``, a dropped conjugation, or the
    ring giving up more than the sectors put back -- which is the failure this
    whole construction started from.
    """
    import importlib.util
    import pathlib

    from quatrex.phonon.pole_audit import sector_sum_residual
    from quatrex.phonon.pole_congruence import (
        pf_mixed_self_energy, pf_self_energy,
    )

    _p = pathlib.Path(__file__).with_name("test_pole_mixed_sectors.py")
    _sp = importlib.util.spec_from_file_location("_pms", _p)
    m = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(m)

    h = 0.1
    w_pos = np.arange(0.0, 40.0 + 1e-9, h)
    w_full = np.concatenate([-w_pos[:0:-1], w_pos])
    phi, _, _, off = m._bed(0.5)
    rows, cols = m._pattern()
    rng = np.random.default_rng(11)
    nd = m.N_DOF

    # Closed pole set: the residue sum vanishes, so the analytic leg decays
    # like 1/w^2 and the residue kernel's infinite tail is negligible against
    # the reference's finite grid (test_finite_window_kernel... bounds it).
    a = rng.normal(size=(nd, 2)) + 1j * rng.normal(size=(nd, 2))
    b = rng.normal(size=(nd, 2)) + 1j * rng.normal(size=(nd, 2))
    zc = np.array([8.0 - 0.5j, 12.0 - 0.5j])
    zeta = np.concatenate([zc, -np.conj(zc)])
    p_row, q_col = np.hstack([a, a]), np.hstack([b, -b])
    assert np.abs(p_row @ q_col.T).max() < 1e-12, "unclosed set"

    mat = rng.normal(size=(nd, nd))
    mat = mat + mat.T

    def _pair(x):
        kt = 25.0
        n_b = 1.0 / np.expm1(np.where(np.abs(x) < 1e-9, 1e-9, x) / kt)
        aa = 4.0 / ((x - 9.0) ** 2 + 16.0) - 4.0 / ((x + 9.0) ** 2 + 16.0)
        return (np.einsum("w,ij->wij", 1j * n_b * aa, mat),
                np.einsum("w,ij->wij", 1j * (n_b + 1.0) * aa, mat))

    reg_full, _ = _pair(w_full)
    reg_l_v, reg_g_v = (x[:, rows, cols] for x in _pair(w_pos))

    d = 1.0 / (w_full[:, None] - zeta[None, :])
    gs_full = np.einsum("ip,wp,jp->wij", p_row, d, q_col)
    hybrid = gs_full + reg_full

    total = m._brute_ring(phi, off, hybrid, hybrid, w_full, rows, cols)
    rr = m._brute_ring(phi, off, reg_full, reg_full, w_full, rows, cols)

    from quatrex.phonon.pole_bridge import modal_vertex_blocks
    vl = modal_vertex_blocks(phi, m.SIZES, p_row, conjugate=False)
    vr = modal_vertex_blocks(phi, m.SIZES, q_col, conjugate=False)
    ss = np.asarray(pf_self_energy(w_pos, zeta, vl, vr, rows, cols))
    mx = np.asarray(pf_mixed_self_energy(
        w_pos, zeta, p_row, q_col, reg_l_v, reg_g_v, w_pos,
        phi, m.SIZES, rows, cols))

    # Compare where the reference is not dominated by its own grid edges.
    keep = (w_full > 4.0) & (w_full < 20.0)
    sel = (w_pos > 4.0) & (w_pos < 20.0)
    rep = sector_sum_residual(np.asarray(total)[keep],
                              {"ss": ss[sel], "mixed": mx[sel],
                               "rr": np.asarray(rr)[keep]})
    # Every sector must actually carry weight, or the sum passes vacuously.
    for k in ("weight_ss", "weight_mixed", "weight_rr"):
        assert rep[k] > 1e-3, f"{k} is numerically absent: {rep[k]:.2e}"
    assert rep["residual"] < 1e-2, (
        f"analytic sectors do not reassemble their own hybrid: {rep}")


# --- what the cell-averaged leg does NOT fix ------------------------------- #

def _registration_bed(gamma, h, shift, w_max=800.0):
    """Two Lorentzian lines at ``+-w0``, exactly cell-averaged, run through a
    midpoint-rule ring, against the exact cell-averaged convolution.

    ``shift`` moves ``w0`` off its cell centre, in cells. Everything else is
    exact: the legs are analytic cell averages, and so is the reference.
    """
    n = int(2 * w_max / h) // 2 * 2 + 1
    wk = (np.arange(n) - n // 2) * h            # w = 0 IS a cell centre
    avg = lambda wc, g: 2 * (np.arctan((wc + h / 2) / g)
                             - np.arctan((wc - h / 2) / g)) / h
    w0 = 10.0 + shift * h
    leg = avg(wk - w0, gamma) + avg(wk + w0, gamma)
    ring = np.convolve(leg, leg)[np.arange(n) + n // 2] * (h / (2 * np.pi))
    # L_g * L_g = L_2g, so the exact answer is the cell average of the three
    # lines at -2w0, 0, +2w0 (the middle one doubled by the two cross terms).
    exact = (avg(wk - 2 * w0, 2 * gamma) + 2 * avg(wk, 2 * gamma)
             + avg(wk + 2 * w0, 2 * gamma))
    i = int(np.argmin(np.abs(wk - 2 * w0)))
    return ring[i] / exact[i], float(np.abs(ring - exact).max()
                                     / np.abs(exact).max())


@pytest.mark.parametrize("h_over_gamma", [20, 200])
@pytest.mark.parametrize("shift,lo,hi", [(0.00, 0.99, 1.01),
                                         (0.25, 1.70, 2.00),
                                         (0.50, 0.50, 0.55)])
def test_cell_averaged_legs_do_not_fix_the_bubble_registration(
        h_over_gamma, shift, lo, hi):
    """An EXACT leg average still gets the combination line in the wrong bin.

    The congruence route makes ``<G~>_k`` exact and stops there. That is not
    the same as integrating the product: the cell average puts all of a line's
    weight at its cell CENTRE, so a resonance sitting a quarter cell off has
    its combination frequency ``Re(z_a + z_b)`` displaced by half a cell, and
    the ring splits the peak between two bins.

    The control parameter is the pole's SUB-CELL POSITION, not ``h/gamma``.
    With the pole centred the ring is exact to 0.4 % at ``h = 20 gamma`` and
    improves as ``h`` grows; a quarter cell off it is 79 % high and gets WORSE
    with ``h``, asymptoting to a factor 2. Pole placement is set by the
    physics, so the congruence route's accuracy here is an accident of
    registration -- which is the reason to resolve the pole inside the
    convolution rather than only in the leg weight.
    """
    gamma = 0.05
    ratio, _ = _registration_bed(gamma, h_over_gamma * gamma, shift)
    assert lo < ratio < hi, (
        f"h/gamma={h_over_gamma} shift={shift}: ratio {ratio:.4f} "
        f"outside [{lo}, {hi}]")


def test_registration_error_is_dominated_by_pole_cell_PAIRS():
    """... and it is an ``|P|^2`` object, not an ``|P| * N`` one.

    Displacing a line to its cell centre costs the convolution
    ``O((delta/Gamma_other)^2)`` when the OTHER leg is resolved -- 2 % here,
    at the worst possible placement -- but an order-one splitting when both
    legs are displaced, because then the combination line moves a full cell
    (46 % at the same placement, measured by the test above). Two orders
    apart, so a correction that replaces the boxcar on just the cell PAIRS
    with both ends in ``P`` recovers essentially all of it, and there are
    ``|P|^2`` such pairs rather than ``|P| * N``.
    """
    gamma, h, w_max = 0.05, 1.0, 800.0
    n = int(2 * w_max / h) // 2 * 2 + 1
    wk = (np.arange(n) - n // 2) * h
    avg = lambda wc, g: 2 * (np.arctan((wc + h / 2) / g)
                             - np.arctan((wc - h / 2) / g)) / h
    ring = lambda a, b: np.convolve(a, b)[np.arange(n) + n // 2] * (h / (2 * np.pi))
    rel = lambda got, ex: float(np.abs(got - ex).max() / np.abs(ex).max())

    w0 = 10.0 + 0.5 * h                     # worst-case placement, both cases
    narrow = avg(wk - w0, gamma) + avg(wk + w0, gamma)
    broad = avg(wk, 3.0)                    # gamma >> h: fully resolved

    one = rel(ring(narrow, broad),
              avg(wk - w0, gamma + 3.0) + avg(wk + w0, gamma + 3.0))
    both = rel(ring(narrow, narrow),
               avg(wk - 2 * w0, 2 * gamma) + 2 * avg(wk, 2 * gamma)
               + avg(wk + 2 * w0, 2 * gamma))
    assert one < 0.05, f"one leg in a pole cell: {one:.3e}"
    assert both > 0.4, f"both legs in pole cells: {both:.3e}"
    assert both > 20 * one, (
        f"the two cases must be orders apart, or restricting the correction "
        f"to pole-cell PAIRS is not justified: {both:.3e} vs {one:.3e}")


# --- conjugate-pole residue symmetry, review Sec. 21 ------------------------ #

def _flat_bed(closed, n_dof=6, n_p=2, n_w=41, seed=4):
    """Coefficients -> freeze -> flatten, on a smooth anti-Hermitian source."""
    from quatrex.phonon.pole_congruence import (
        background_coefficients, coefficients_at_poles, partial_fraction_legs,
    )

    rng = np.random.default_rng(seed)
    cx = lambda *s: rng.normal(size=s) + 1j * rng.normal(size=s)
    w = np.linspace(4.0, 14.0, n_w)
    zc = np.array([8.0 - 0.3j, 11.0 - 0.25j])[:n_p]
    z = np.concatenate([zc, -np.conj(zc)]) if closed else zc
    cl = PoleCluster(z=z, u=cx(n_dof, z.size), v=cx(n_dof, z.size), label="ah")

    # -i Sigma PSD  =>  Sigma^dagger = -Sigma, and smooth across the window so
    # the local fit is a fair one.
    a0, a1 = cx(n_dof, n_dof), cx(n_dof, n_dof)
    sig = np.stack([1j * ((a0 + 0.03 * x * a1)
                          @ np.conj((a0 + 0.03 * x * a1).T)) for x in w])
    gk = np.stack([cx(n_dof, n_dof) for _ in w])
    sv = np.einsum("wij,ja->wia", sig, np.asarray(cl.v))
    co = background_coefficients(cl, w, sv, np.einsum("wij,wja->wia", gk, sv))
    frozen = coefficients_at_poles(cl, w, co)
    return cl, co, frozen, partial_fraction_legs(cl, frozen)


@pytest.mark.parametrize("closed", [False, True])
def test_flattened_residues_keep_the_conjugate_pole_antisymmetry(closed):
    r"""``R_{conj z} = -R_z^dagger``, and it survives the freeze.

    On the real axis the Keldysh leg must be anti-Hermitian, so a simple-pole
    expansion has to pair each residue with minus the adjoint of its
    conjugate partner's. The concern (review Sec. 21) is that ``c_sr`` and
    ``c_rs`` are fitted INDEPENDENTLY, at different poles, so nothing
    obviously enforces it.

    It is enforced, and by two things that are easy to break by accident:

    * ``conj(c_ss[b,a]) = -c_ss[a,b]`` survives ``source_at_poles`` because it
      shares ``0.5*(P(z_a) + P(conj z_b))`` between the pair, evaluated from
      ONE fit with a common anchor. A per-pole value would not.
    * ``c_sr(z_a) = -c_rs(conj z_a)^dagger`` survives ``coefficients_at_poles``
      because both fits run through the same REAL design matrix, so the fitted
      coefficient matrices inherit ``A_n = -B_n^dagger`` from the data.

    Change the anchor, the window, or the weighting on one side and this
    breaks silently, which is why it is measured rather than argued.
    """
    cl, co, frozen, (zeta, p_row, q_col) = _flat_bed(closed)

    # the two identities the result rests on, before and after freezing
    assert np.abs(np.conj(np.swapaxes(co[2], 1, 2)) + co[2]).max() < 1e-10
    assert np.abs(np.conj(frozen[2].T) + frozen[2]).max() < 1e-9, (
        "the pair-shared source freeze lost c_ss^dagger = -c_ss")
    assert np.abs(frozen[0] + np.conj(frozen[1].T)).max() < 1e-10, (
        "the two independent fits lost c_sr = -c_rs^dagger")

    n = int(np.asarray(cl.z).size)
    res = np.einsum("ip,jp->pij", p_row, q_col)
    eps = max(np.linalg.norm(res[n + a] + np.conj(res[a].T))
              / (np.linalg.norm(res[a]) + np.linalg.norm(res[n + a]))
              for a in range(n))
    assert eps < 1e-12, f"eps_AH,res = {eps:.3e}"


@pytest.mark.parametrize("closed", [False, True])
def test_flattened_leg_is_anti_hermitian_on_the_real_axis(closed):
    """The same statement where it is actually consumed: ``-i G_S`` Hermitian.

    The residue test above can pass while the assembled leg does not, if a
    pole were paired with the wrong partner in ``zeta``.
    """
    _, _, _, (zeta, p_row, q_col) = _flat_bed(closed)
    w = np.linspace(5.0, 13.0, 17)
    d = 1.0 / (w[:, None] - zeta[None, :])
    gs = np.einsum("ip,wp,jp->wij", p_row, d, q_col)
    eps = (np.abs(gs + np.conj(np.swapaxes(gs, 1, 2))).max()
           / np.abs(gs).max())
    assert eps < 1e-12, f"eps_AH(w) = {eps:.3e}"


def test_pole_pair_weight_bounds_where_the_registration_error_can_live():
    """How much of the ring's weight sits on cell pairs with BOTH ends poled.

    That is the only place the registration error is order one, so the error
    in ``Sigma`` is bounded by this fraction times ~0.8. The measure has to
    get three things right or the bound is not a bound.
    """
    from quatrex.phonon.pole_audit import pole_pair_weight

    n = 41
    w = np.arange(n, dtype=float)
    mask0 = w < 0.5                                # the ring's own w = 0 mask

    # (i) one pole cell against an otherwise flat leg. It pairs with itself
    #     at the DIFFERENCE frequency 0 and at the SUM frequency 2*w0; the
    #     former is the bin the ring zeroes, so it must not decide `worst`.
    g = np.ones(n)
    g[10] = 50.0
    p = np.zeros(n, dtype=bool)
    p[10] = True
    assert pole_pair_weight(g, p, freqs=w)["omega_index"] == 0
    rep = pole_pair_weight(g, p, freqs=w, skip=mask0)
    assert rep["omega_index"] == 20, rep           # 10 + 10
    assert rep["worst"] > 0.9
    assert 0.0 < rep["mean"] < 0.5

    # (ii) the negative half must be folded back, or every DIFFERENCE
    #      frequency pairing is silently dropped -- roughly half the
    #      convolution, and exactly the half that carries Omega_a - Omega_b.
    g = np.ones(n)
    g[[10, 16]] = 50.0
    p = np.zeros(n, dtype=bool)
    p[[10, 16]] = True
    rep = pole_pair_weight(g, p, freqs=w, skip=mask0)
    k = np.arange(n)
    gp = g * p
    frac6 = float(gp[k] @ gp[np.abs(6 - k)]) / float(g[k] @ g[np.abs(6 - k)])
    assert frac6 > 0.1, f"difference-frequency pairing dropped: {frac6:.3e}"
    assert rep["worst"] >= frac6 - 1e-12

    # (iii) no poles -> no bound to report, and no division by zero
    empty = pole_pair_weight(g, np.zeros(n, dtype=bool), freqs=w)
    assert empty["mean"] == 0.0 and empty["omega_index"] == -1

    # (iv) every cell poled -> the fraction is exactly 1, so the bound is
    #      vacuous and says so rather than reading small by accident
    allp = pole_pair_weight(g, np.ones(n, dtype=bool), freqs=w, skip=mask0)
    assert abs(allp["mean"] - 1.0) < 1e-12
    assert abs(allp["worst"] - 1.0) < 1e-12


def test_state_report_carries_the_promotion_yield():
    """"2 pole(s)" reads like a small system; "2/144" reads like a threshold.

    On the CNT bed 142 of 144 candidates are refused on ``eps_nep`` alone, and
    that is why the sector moves the answer only in the fourth digit -- not
    any property of its quadrature. The header has to say so, or the next
    reader draws the same wrong conclusion from the same log.
    """
    from quatrex.phonon.pole_keldysh import PoleCluster
    from quatrex.phonon.pole_sector import PoleSectorState

    st = PoleSectorState()
    st.iteration = 1
    st.clusters = [PoleCluster(z=np.array([9.0 - 0.1j]),
                               u=np.ones((2, 1), dtype=complex),
                               v=np.ones((2, 1), dtype=complex))]
    st.coherence = [1.0]
    st.rejected = ([(3.0 - 0.2j, "eps_nep=5.7e-03 above tolerance")] * 4
                   + [(4.0 - 0.2j, "weight below weight_min")])
    head = st.report().splitlines()[0]
    assert "1/6 pole(s) promoted" in head, head
    assert "eps_nep x4" in head and "weight below weight_min x1" in head, head


# --- memory: the sector must be able to carry more than a handful of poles -- #

def test_chunked_sector_kernels_match_the_unchunked_result_exactly():
    """Chunking is a memory transform, not a numerical one.

    The contraction that had to be broken up is ``take(c_sr, cols, axis=2)``,
    shaped ``(n_omega, Np, nnz)``. Two of those per Keldysh component per call
    is invisible at the 2 poles the CNT bed promotes and 290 GB at a few
    dozen -- so the route could never be asked to carry the pole count the
    physics needs. Splitting the pattern must not move a single bit.
    """
    from quatrex.phonon.pole_congruence import (
        sector_cell_average, sector_grid_sample,
    )

    cl, w, gk, sig = _bed()
    co = _coeffs(cl, w, gk, sig)
    rows, cols = np.meshgrid(np.arange(N_DOF), np.arange(N_DOF), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    h = np.full(w.size, float(w[1] - w[0]))

    # a budget big enough for one shot, and one that forces many chunks
    one = 1 << 40
    many = 1 << 10
    for fn, args in ((sector_grid_sample, ()), (sector_cell_average, (h,))):
        whole = np.asarray(fn(cl, w, co, rows, cols, *args, chunk_bytes=one))
        split = np.asarray(fn(cl, w, co, rows, cols, *args, chunk_bytes=many))
        np.testing.assert_array_equal(whole, split)


def test_pattern_chunk_bounds_the_working_set_as_poles_are_added():
    """The peak must scale with the budget, not with Np * nnz.

    This is the number that decides whether a promotion-yield experiment
    measures physics or memory.
    """
    from quatrex.phonon.pole_congruence import _pattern_chunk

    n_w, nnz, budget = 201, 700_000, 1 << 28
    prev = None
    for n_p in (2, 8, 32, 128):
        step = _pattern_chunk(n_w, n_p, nnz, budget)
        peak = 16 * n_w * n_p * step
        assert peak <= budget, f"Np={n_p}: peak {peak / 1e9:.2f} GB"
        assert step >= 1
        if prev is not None:
            assert step <= prev, "chunk must shrink as poles are added"
        prev = step
    # ... and a small problem is still done in one shot, so nothing that
    # already worked pays a Python-loop tax.
    assert _pattern_chunk(n_w, 2, 500, budget) == 500


# --- what the additive route is and is not required to satisfy ------------- #

def test_an_additive_remainder_may_be_indefinite_while_the_total_is_fine():
    """``G_R = G - G_S`` is a DIFFERENCE, so its sign is unconstrained.

    The congruence route's leg is built as a positive cell-averaged
    congruence, so ring-leg positivity is a real gate there. The analytic
    route's leg is an additive remainder, and requiring it to be PSD is a
    category error -- the same one ``bubble_positivity.md`` records when it
    says the gate is on the total and never on a sector.

    What the additive route must satisfy is the sector sum and the positivity
    of the TOTAL, and both are tested elsewhere. This pins the negative
    statement so it is not re-litigated from a scary-looking leg number.
    """
    rng = np.random.default_rng(2)
    n = 6

    def psd(seed):
        a = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
        return a @ np.conj(a.T)

    g = psd(0)                       # -i G, physical
    g_s = psd(1)                     # -i G_S, also physical on its own
    g_r = g - g_s                    # the remainder the ring is handed

    lam = lambda m: float(np.linalg.eigvalsh(0.5 * (m + np.conj(m.T))).min())
    assert lam(g) > -1e-10 and lam(g_s) > -1e-10
    assert lam(g_r) < -1e-6, (
        "the bed must actually exhibit an indefinite remainder, or the test "
        "asserts nothing")
    # ... and the two still sum to the physical object, which is the only
    # thing the decomposition promises.
    np.testing.assert_allclose(g_s + g_r, g, atol=1e-12)


# --------------------------------------------------------------------------- #
# The congruence reconstruction itself, off-grid.
#
# The pole sector stores one remainder per cell and treats it as piecewise
# constant. WHICH remainder it stores decides whether the reconstruction stays
# physical between grid points -- see pole_scba_divergence.md Sec. 7, where
# freezing the KELDYSH remainder is measured at eps_PSD = -5.4e-02 while every
# cell centre is positive.
# --------------------------------------------------------------------------- #

from quatrex.phonon.pole_keldysh import (          # noqa: E402
    hybrid_keldysh_congruence, pole_keldysh, pole_retarded,
)


def _psd_bed(gamma: float, n_dof: int = 3, seed: int = 1):
    """One narrow pole plus wide background; ``G^<`` PSD by construction."""
    rng = np.random.default_rng(seed)

    def vec():
        return rng.normal(size=(n_dof,)) + 1j * rng.normal(size=(n_dof,))

    z_n = np.array([4.0 - 1j * gamma])
    u_n = vec().reshape(n_dof, 1)
    cl = PoleCluster(z=z_n, u=u_n, v=u_n)
    z_bg = np.array([2.5 - 0.9j, 7.5 - 1.2j])
    u_bg = np.stack([vec(), vec()], axis=1)
    a = rng.normal(size=(n_dof, n_dof)) + 1j * rng.normal(size=(n_dof, n_dof))
    # The PSD object is -i Sigma^<, hence the explicit i.
    sigma = 1j * (a @ np.conj(a).T)

    def g_ret(w):
        w = np.atleast_1d(np.asarray(w, float))
        d = 1.0 / (w.reshape(-1, 1) - z_bg.reshape(1, -1))
        return (pole_retarded(w, cl)
                + np.einsum("ia,wa,ja->wij", u_bg, d, np.conj(u_bg)))

    def g_les(w):
        gr = g_ret(w)
        return gr @ sigma @ np.conj(gr).swapaxes(-2, -1)

    return cl, g_ret, g_les, sigma


def _herm_eigs(mats):
    """Eigenvalues of the Hermitian part of ``-i M``."""
    h = -1j * np.asarray(mats)
    return np.linalg.eigvalsh(0.5 * (h + np.conj(h).swapaxes(-2, -1)))


def test_congruence_matches_the_direct_split_at_cell_centres():
    """Both reconstructions are exact where the remainder is defined."""
    cl, g_ret, g_les, sigma = _psd_bed(gamma=0.01)
    centres = np.array([3.75, 4.0, 4.25])
    r_ret = g_ret(centres) - pole_retarded(centres, cl)
    src = np.repeat(sigma[None], centres.size, axis=0)
    got = hybrid_keldysh_congruence(centres, cl, r_ret, src,
                                    np.arange(centres.size))
    exact = g_les(centres)
    err = np.abs(got - exact).max() / np.abs(exact).max()
    assert err < 1e-10, f"congruence must be exact at centres: {err:.3e}"


@pytest.mark.parametrize("gamma", [0.1, 0.02, 0.005])
def test_congruence_stays_psd_off_grid_where_direct_subtraction_fails(gamma):
    """The point of the construction: positivity is structural off-grid.

    ``-i G^< = G^R (-i Sigma^<) (G^R)^H`` holds for any ``G^R``, so the
    reconstruction cannot go indefinite however coarse the cell. Freezing the
    Keldysh remainder instead has no such guarantee, and must be shown failing
    here or the test proves nothing.
    """
    h = 0.25
    cl, g_ret, g_les, sigma = _psd_bed(gamma=gamma)
    centres = np.array([4.0])
    x = np.polynomial.legendre.leggauss(24)[0] * 0.5
    w = centres[0] + h * x
    idx = np.zeros(w.size, dtype=int)

    r_ret = g_ret(centres) - pole_retarded(centres, cl)
    src = np.repeat(sigma[None], 1, axis=0)
    cong = hybrid_keldysh_congruence(w, cl, r_ret, src, idx)

    r_les = g_les(centres) - pole_keldysh(centres, cl,
                                          np.conj(cl.v).T @ sigma @ cl.v)
    direct = pole_keldysh(w, cl, np.conj(cl.v).T @ sigma @ cl.v) + r_les[0]

    # One global scale for both, per pole_audit.psd_residual's convention.
    scale = float(max(_herm_eigs(cong).max(), _herm_eigs(direct).max()))
    psd_cong = _herm_eigs(cong).min() / scale
    psd_direct = _herm_eigs(direct).min() / scale
    assert psd_cong > -1e-12, f"congruence lost positivity: {psd_cong:.3e}"
    assert psd_direct < -1e-6, (
        "the direct split must FAIL here, else the bed is not exercising "
        f"the failure mode: {psd_direct:.3e}")


# --- exact represented weight of an unresolved line ------------------------- #

def _w_exact(r, x):
    """Infinite trapezoidal sum of a unit-weight Lorentzian, review Eq. (1)."""
    a = 2 * np.pi / r
    return np.sinh(a) / (np.cosh(a) - np.cos(2 * np.pi * x))


@pytest.mark.parametrize("r", [0.5, 1.0, 1.35, 2.0, 3.0, 5.0, 20.0, 100.0])
@pytest.mark.parametrize("x", [0.0, 0.1, 0.25, 0.5])
def test_exact_trapezoidal_line_weight(r, x):
    r"""What the ring's ``dw``-weighted sum actually carries of a narrow line.

    Summing point samples of a unit-weight Lorentzian over a uniform grid is a
    theta-function, not the nearest-node term:

    .. math::
        W_\infty(r, x) = \frac{\sinh(2\pi/r)}
                              {\cosh(2\pi/r) - \cos(2\pi x)},
        \qquad r = h/\gamma,\ x = \text{offset in cells}.

    An earlier note quoted ``r/(pi(1 + r^2 x^2))`` for this. That is the
    NEAREST-NODE weight and it is wrong for the total by up to 2.5x -- at
    ``r = 100, x = 0.5`` it gives 0.0127 against 0.0314. The distinction
    matters because the whole argument for the sector is how much line weight
    the grid misplaces, and the two formulas disagree about it.
    """
    gamma, w_max = 0.05, 4000.0
    h = r * gamma
    n = int(2 * w_max / h) // 2 * 2 + 1
    wk = (np.arange(n) - n // 2) * h
    lor = 2 * gamma / ((wk - x * h) ** 2 + gamma ** 2)   # int dw/2pi = 1
    got = h * lor.sum() / (2 * np.pi)
    assert abs(got - _w_exact(r, x)) < 5e-4 * max(_w_exact(r, x), 1.0)


def test_line_weight_gate_inverts_the_tolerance_exactly():
    r"""``E_leg^max(r) = coth(pi/r) - 1``, and its inverse is the gate.

    The worst overestimate is at ``x = 0`` and the worst underestimate at
    ``x = 1/2``; the overestimate is the stricter side, so a worst-case
    tolerance ``eps`` needs ``h/gamma < 2 pi / log(1 + 2/eps)``. That is an
    exact statement about represented weight, where ``samples_per_halfwidth``
    was a constant chosen by hand.
    """
    for r in (0.5, 1.0, 2.0, 5.0, 20.0):
        assert abs(_w_exact(r, 0.0) - 1.0 / np.tanh(np.pi / r)) < 1e-12
        assert abs(_w_exact(r, 0.5) - np.tanh(np.pi / r)) < 1e-12
        # the overestimate is the side that binds
        assert _w_exact(r, 0.0) - 1.0 >= 1.0 - _w_exact(r, 0.5) - 1e-15

    for eps in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50):
        r_eps = 2 * np.pi / np.log(1 + 2 / eps)
        assert abs(2 / (np.exp(2 * np.pi / r_eps) - 1) - eps) < 1e-12
        # just inside the threshold the worst case is within tolerance,
        # just outside it is not -- so the gate is tight, not conservative
        assert _w_exact(r_eps * 0.999, 0.0) - 1.0 < eps
        assert _w_exact(r_eps * 1.001, 0.0) - 1.0 > eps
