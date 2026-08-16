# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The subcell covariance: what the cell-averaged ring leaves behind.

The claim under test is an identity, not an approximation:

    int_{I_k} du/2pi B[G_k(u), G_l(Omega-u)]
        = (h/2pi) B(Gbar_k, Gbar_l)  +  Delta I_kl

with the cross terms vanishing exactly because both fluctuations have zero cell
mean and the reflection maps one cell onto the other. So every test here is
against a direct quadrature of the same integrand, at roundoff, not against a
tolerance chosen to pass.
"""
import numpy as np
import pytest

from quatrex.phonon.pole_covariance import (
    cell_resolvent_mean,
    cell_variance,
    centred_gram,
    covariance_kernel,
)

H = 0.5


def _leg(zeta, res, centre, h=H):
    """The local model as a callable, and its exact cell mean."""
    d = cell_resolvent_mean(np.asarray(zeta), centre, h)

    def g(u):
        u = np.atleast_1d(np.asarray(u, dtype=complex))
        return np.einsum("p,pij->...ij", np.ones(1), np.zeros((1,) + res.shape[1:])) \
            if False else np.einsum("wp,pij->wij",
                                    1.0 / (u[:, None] - np.asarray(zeta)[None, :]),
                                    np.asarray(res))

    mean = np.einsum("p,pij->ij", d, np.asarray(res))
    return g, mean


def _quad(f, a, b, n=4001):
    """Composite Simpson on a smooth-enough integrand; refined until stable."""
    x = np.linspace(a, b, n)
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    w *= (b - a) / (3.0 * (n - 1))
    vals = f(x)
    return np.einsum("w,w...->...", w, vals)


def _bed(seed=0, n_dof=2, p_k=2, p_l=2, gamma=0.01):
    rng = np.random.default_rng(seed)

    def cx(*s):
        return rng.normal(size=s) + 1j * rng.normal(size=s)

    ck, cl = 0.0, 1.5
    zk = ck + rng.uniform(-0.3, 0.3, p_k) * H - 1j * gamma * rng.uniform(1, 3, p_k)
    zl = cl + rng.uniform(-0.3, 0.3, p_l) * H - 1j * gamma * rng.uniform(1, 3, p_l)
    return ck, cl, zk, zl, cx(p_k, n_dof, n_dof), cx(p_l, n_dof, n_dof)


# --------------------------------------------------------------------------- #

def test_centred_basis_has_zero_cell_mean():
    """Everything downstream is the statement that the cross terms vanish."""
    _, _, zk, _, _, _ = _bed()
    d = cell_resolvent_mean(zk, 0.0, H)
    for p, z in enumerate(zk):
        phi = lambda u, z=z, dp=d[p]: 1.0 / (u - z) - dp   # noqa: E731
        m = _quad(phi, -H / 2, H / 2) / H
        assert abs(m) < 1e-12, f"pole {p}: cell mean {m:.3e}"


def test_centred_gram_is_psd_and_matches_quadrature():
    """``H`` is a Gram matrix, so PSD is a property, not a hope."""
    _, _, zk, _, _, _ = _bed()
    d = cell_resolvent_mean(zk, 0.0, H)
    got = np.asarray(centred_gram(zk, 0.0, H))
    for p in range(zk.size):
        for q in range(zk.size):
            f = lambda u, p=p, q=q: ((1.0 / (u - zk[p]) - d[p])
                                     * np.conj(1.0 / (u - zk[q]) - d[q]))
            ref = _quad(f, -H / 2, H / 2) / H
            assert abs(got[p, q] - ref) < 1e-9 * max(abs(ref), 1.0)
    ev = np.linalg.eigvalsh(0.5 * (got + np.conj(got.T)))
    assert ev.min() > -1e-10 * max(abs(ev).max(), 1.0), ev


def test_variance_is_the_norm_the_bound_uses():
    _, _, zk, _, rk, _ = _bed()
    got = cell_variance(rk, zk, 0.0, H)
    d = cell_resolvent_mean(zk, 0.0, H)

    def f(u):
        dg = np.einsum("wp,pij->wij",
                       1.0 / (u[:, None] - zk[None, :]) - d[None, :], rk)
        return np.sum(np.abs(dg) ** 2, axis=(1, 2))

    ref = _quad(f, -H / 2, H / 2) / H
    assert abs(got - ref) < 1e-8 * ref, f"{got:.6e} vs {ref:.6e}"


def test_finite_cell_kernel_is_stable_at_the_combination_frequency():
    r"""The small-denominator regime, which cannot actually be reached."""
    mp = pytest.importorskip("mpmath")
    from quatrex.phonon.pole_bubble import pair_convolution

    mp.mp.dps = 60
    a, b = -H / 2, H / 2
    for g in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
        gam = g * H
        p, q, om = -1j * gam, 3.0 - 1j * gam, 3.0
        got = pair_convolution(np.array([p]), np.array([q]),
                               np.array([om]), window=(a, b))[0, 0]
        P, Q, W = mp.mpc(p), mp.mpc(q), mp.mpc(om)
        ref = complex(mp.quad(lambda u: 1 / ((u - P) * (W - u - Q)),
                              [mp.mpf(a), mp.mpf(0), mp.mpf(b)]) / (2 * mp.pi))
        assert abs(got - ref) < 1e-13 * abs(ref), (
            f"gamma/h={g}: {got} vs {ref}")


def test_rank_one_residues_factorise_the_vertex_contraction():
    r"""``B(R_p, R_q)`` is one projected pair vertex on each side.

    That is why the pole algebra is quadratic in the flattened residue count.
    A quartic implementation is carrying a redundant modal index, not doing
    more work for a reason.
    """
    rng = np.random.default_rng(5)
    n = 4

    def cx(*s):
        return rng.normal(size=s) + 1j * rng.normal(size=s)

    x_p, y_p, x_q, y_q = cx(n), cx(n), cx(n), cx(n)
    r_p = np.outer(x_p, np.conj(y_p))
    r_q = np.outer(x_q, np.conj(y_q))
    phi = cx(n, n, n)

    direct = np.einsum("mab,ad,bc,ncd->mn", phi, r_p, r_q, phi)
    left = np.einsum("mab,a,b->m", phi, x_p, x_q)
    right = np.einsum("ncd,c,d->n", phi, np.conj(y_q), np.conj(y_p))
    assert np.abs(direct - np.outer(left, right)).max() < 1e-10 * np.abs(direct).max()


# --- the spectrum-level assembly ------------------------------------------- #

def _pattern(n_dof=3):
    r, c = np.meshgrid(np.arange(n_dof), np.arange(n_dof), indexing="ij")
    return r.ravel(), c.ravel()


def _phi(n_dof=3, seed=9):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(n_dof, n_dof, n_dof))
    return {(0, 0, 0): b + np.swapaxes(b, 1, 2)}, np.array([n_dof])


def test_spectrum_correction_is_empty_without_active_cells():
    """Gate 0: no active cell, no correction, bit-exactly."""
    from quatrex.phonon.pole_covariance import spectrum_correction

    phi, sizes = _phi()
    rows, cols = _pattern()
    freqs = np.linspace(0.0, 4.0, 9)
    corr, rep = spectrum_correction(freqs, [], phi, sizes, rows, cols, 0.5)
    assert np.abs(np.asarray(corr)).max() == 0.0
    assert rep == {"pairs": 0, "applied": 0, "out_of_range": 0}


def test_spectrum_correction_bins_each_pair_at_the_sum_frequency():
    """One active pair must land on ``omega_k + omega_l`` and nowhere else,
    carrying exactly the kernel the verified pair routine gives."""
    from quatrex.phonon.pole_bridge import analytic_prefactor, modal_vertex_blocks
    from quatrex.phonon.pole_covariance import (
        covariance_kernel, spectrum_correction,
    )

    n_dof = 3
    phi, sizes = _phi(n_dof)
    rows, cols = _pattern(n_dof)
    h = 0.5
    freqs = np.arange(9) * h                      # 0 .. 4
    rng = np.random.default_rng(2)

    def cx(*s):
        return rng.normal(size=s) + 1j * rng.normal(size=s)

    zk = np.array([1.0 - 0.01j, -1.0 - 0.01j])
    zl = np.array([1.5 - 0.02j, -1.5 - 0.02j])
    cells = [(1.0, zk, cx(n_dof, 2), cx(n_dof, 2)),
             (1.5, zl, cx(n_dof, 2), cx(n_dof, 2))]
    corr, rep = spectrum_correction(freqs, cells, phi, sizes, rows, cols, h)
    corr = np.asarray(corr)

    # 4 ordered pairs -> outputs 2.0, 2.5, 2.5, 3.0 == indices 4, 5, 5, 6
    assert rep["applied"] == 4 and rep["out_of_range"] == 0
    live = {i for i in range(freqs.size) if np.abs(corr[i]).max() > 0}
    assert live == {4, 5, 6}, live

    # and index 4 is exactly the (k, k) pair through the mixed vertex
    ck, zc, pk, qk = cells[0]
    kern = np.asarray(covariance_kernel(zc, zc, ck, ck, h, np.array([freqs[4]])))[0]
    vl = np.asarray(modal_vertex_blocks(phi, sizes, pk, conjugate=False, v=pk))
    vr = np.asarray(modal_vertex_blocks(phi, sizes, qk, conjugate=False, v=qk))
    want = analytic_prefactor() * np.einsum(
        "pq,kpq,kqp->k", kern, vl[rows], vr[cols])
    assert np.abs(corr[4] - want).max() < 1e-12 * np.abs(want).max()


def test_negative_cells_produce_the_difference_channel():
    """A cell at ``-omega_l`` must correct ``omega_k - omega_l``.

    The ring integrates the whole axis, so dropping the negative cells would
    silently lose every ``Omega_a - Omega_b`` process -- half the convolution,
    and the half that lands mid-band rather than at the sum frequency.
    """
    from quatrex.phonon.pole_covariance import spectrum_correction

    n_dof = 3
    phi, sizes = _phi(n_dof)
    rows, cols = _pattern(n_dof)
    h = 0.5
    freqs = np.arange(9) * h
    rng = np.random.default_rng(4)

    def cx(*s):
        return rng.normal(size=s) + 1j * rng.normal(size=s)

    z = np.array([2.5 - 0.01j, -2.5 - 0.01j])
    zn = np.array([-1.0 - 0.01j, 1.0 - 0.01j])
    cells = [(2.5, z, cx(n_dof, 2), cx(n_dof, 2)),      # +2.5
             (-1.0, zn, cx(n_dof, 2), cx(n_dof, 2))]    # -1.0
    corr, rep = spectrum_correction(freqs, cells, phi, sizes, rows, cols, h)
    corr = np.asarray(corr)
    live = {i for i in range(freqs.size) if np.abs(corr[i]).max() > 0}
    # 2.5+2.5 = 5.0 is off the grid (max 4.0) -> counted, not silently dropped;
    # 2.5-1.0 = 1.5 (index 3) twice, and -2.0 is off the grid below zero.
    assert 3 in live, live
    assert rep["out_of_range"] == 2, rep
    assert rep["applied"] == 2


def test_the_mixed_pairing_drives_s_through_zero_and_is_still_exact():
    r"""The case the same-half-plane argument does not cover."""
    mp = pytest.importorskip("mpmath")
    from quatrex.phonon.pole_bubble import pair_convolution

    mp.mp.dps = 50
    a, b = -H / 2, H / 2
    gam = 0.02
    p, q = -1j * gam, +1j * gam            # z and conj(z), pole in this cell
    assert abs(np.imag(p + q)) < 1e-15, "the pairing must give a real sum"

    for ds in (1e-1, 1e-2, 1e-4, 1e-8, 0.0):
        om = float(np.real(p + q)) + ds
        got = pair_convolution(np.array([p]), np.array([q]),
                               np.array([om]), window=(a, b))[0, 0]
        P, Q, W = mp.mpc(p), mp.mpc(q), mp.mpc(om)
        ref = complex(mp.quad(lambda u: 1 / ((u - P) * (W - u - Q)),
                              [mp.mpf(a), mp.mpf(0), mp.mpf(b)]) / (2 * mp.pi))
        assert abs(got - ref) < 1e-10 * abs(ref), f"s={ds}: {got} vs {ref}"
    # and s = 0 is a finite number, not the zero the old branch returned
    at_zero = pair_convolution(np.array([p]), np.array([q]),
                               np.array([float(np.real(p + q))]),
                               window=(a, b))[0, 0]
    assert abs(at_zero) > 1.0


def test_centred_gram_handles_a_conjugate_pair_in_the_family():
    """``zeta_p == conj(zeta_q)`` is the rule, not an exceptional input.

    Refusing it -- as an earlier version did -- rejects every real pole set the
    sector can produce, because the flattened family is built as
    ``[z, conj(z)]``.
    """
    z = np.array([2.0 - 0.02j, -2.0 - 0.02j])
    zeta = np.concatenate([z, np.conj(z)])
    gap = zeta[:, None] - np.conj(zeta)[None, :]
    assert np.abs(gap).min() == 0.0, "the bed must contain the degeneracy"

    got = np.asarray(centred_gram(zeta, 0.0, H))
    assert np.isfinite(got).all()
    d = cell_resolvent_mean(zeta, 0.0, H)
    for p_i in range(zeta.size):
        for q_i in range(zeta.size):
            f = lambda u, i=p_i, j=q_i: ((1.0 / (u - zeta[i]) - d[i])
                                         * np.conj(1.0 / (u - zeta[j]) - d[j]))
            ref = _quad(f, -H / 2, H / 2, n=20001) / H
            assert abs(got[p_i, q_i] - ref) < 1e-7 * max(abs(ref), 1.0)
    ev = np.linalg.eigvalsh(0.5 * (got + np.conj(got.T)))
    assert ev.min() > -1e-9 * max(abs(ev).max(), 1.0), ev


def test_spectrum_correction_chunking_is_exact_and_bounds_the_working_set():
    """``vl[rows]`` is ``(nnz, P, P)``; unchunked that is what took the sector
    kernels to a 290 GB allocation. Chunking must change nothing."""
    from quatrex.phonon.pole_covariance import spectrum_correction

    n_dof = 4
    rng = np.random.default_rng(7)

    def cx(*s):
        return rng.normal(size=s) + 1j * rng.normal(size=s)

    r, c = np.meshgrid(np.arange(n_dof), np.arange(n_dof), indexing="ij")
    rows, cols = r.ravel(), c.ravel()
    phi = {(0, 0, 0): cx(n_dof, n_dof, n_dof)}
    sizes = np.array([n_dof])
    h = 0.5
    freqs = np.arange(9) * h
    cells = [(1.0, np.array([1.0 - 0.01j, -1.0 - 0.01j]),
              cx(n_dof, 2), cx(n_dof, 2)),
             (1.5, np.array([1.5 - 0.02j, -1.5 - 0.02j]),
              cx(n_dof, 2), cx(n_dof, 2))]

    full, _ = spectrum_correction(freqs, cells, phi, sizes, rows, cols, h)
    tiny, _ = spectrum_correction(freqs, cells, phi, sizes, rows, cols, h,
                                  chunk_bytes=1)          # one row at a time
    full, tiny = np.asarray(full), np.asarray(tiny)
    assert np.abs(full - tiny).max() <= 1e-13 * max(np.abs(full).max(), 1.0)
    assert np.abs(full).max() > 0.0


def test_cells_from_clusters_of_different_size_pair_correctly():
    """Two clusters, two pole counts -- the first real device call's crash."""
    from quatrex.phonon.pole_bridge import modal_vertex_blocks
    from quatrex.phonon.pole_covariance import spectrum_correction

    n_dof = 4
    rng = np.random.default_rng(13)

    def cx(*s):
        return rng.normal(size=s) + 1j * rng.normal(size=s)

    r, c = np.meshgrid(np.arange(n_dof), np.arange(n_dof), indexing="ij")
    rows, cols = r.ravel(), c.ravel()
    phi = {(0, 0, 0): cx(n_dof, n_dof, n_dof)}
    sizes = np.array([n_dof])
    h, freqs = 0.5, np.arange(9) * 0.5

    # 2 flattened modes against 6 -- clusters of different size
    small = (1.0, np.array([1.0 - 0.01j, -1.0 - 0.01j]),
             cx(n_dof, 2), cx(n_dof, 2))
    big_z = np.array([1.5 - 0.02j, 1.6 - 0.02j, 1.4 - 0.03j])
    big = (1.5, np.concatenate([big_z, np.conj(big_z)]),
           cx(n_dof, 6), cx(n_dof, 6))

    vb = np.asarray(modal_vertex_blocks(phi, sizes, small[2], conjugate=False,
                                        v=big[2]))
    assert vb.shape == (n_dof, 2, 6), vb.shape

    corr, rep = spectrum_correction(freqs, [small, big], phi, sizes,
                                    rows, cols, h)
    corr = np.asarray(corr)
    assert rep["applied"] == 4 and np.isfinite(corr).all()
    assert np.abs(corr).max() > 0.0
