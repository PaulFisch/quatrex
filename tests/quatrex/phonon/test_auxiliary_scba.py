"""Passive rational SCBA closure and local augmented-Dyson tests."""

import numpy as np
import pytest
from scipy.integrate import quad_vec

from quatrex.phonon.experimental.auxiliary_scba import (
    GlobalAuxiliaryWoodbury,
    LocalAuxiliaryRGF,
    LocalAuxiliaryChannel,
    PassiveClusterState,
    RationalKeldyshChannel,
    assemble_augmented_dense,
    lyapunov_gramian,
    passive_bubble_channel,
    physical_schur_complement,
)


@pytest.fixture(autouse=True, scope="module")
def _configure_comm():
    """DSDB/RGF objects need the single-rank communicator configured."""
    from qttools import xp
    from qttools.comm import comm

    backend = "device_mpi" if xp.__name__ == "numpy" else "host_mpi"
    cfg = {k: backend for k in ("all_to_all", "all_gather", "all_reduce",
                                "bcast")}
    comm.configure(block_comm_size=1, block_comm_config=cfg,
                   stack_comm_config=cfg, override=True)


def _rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


def _cluster(seed=0, n_dof=3):
    rng = np.random.default_rng(seed)
    z = np.array([0.92 - 0.17j, 1.31 - 0.11j])
    u = (rng.normal(size=(n_dof, 2))
         + 1j * rng.normal(size=(n_dof, 2)))
    a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q = a @ a.conj().T
    return PassiveClusterState(z, u, 0.3 * q, 0.72 * q)


def test_lyapunov_gramian_matches_infinite_line_quadrature():
    c = _cluster()
    got = lyapunov_gramian(c.poles, c.q_lesser)

    def f(w):
        d = 1.0 / (w - c.poles)
        return (d[:, None] * c.q_lesser * d.conj()[None, :]).reshape(-1) \
            / (2.0 * np.pi)

    points = sorted(c.poles.real)
    edges = [-np.inf] + points + [np.inf]
    want = sum(
        (quad_vec(f, lo, hi, epsabs=2e-11, epsrel=2e-11, limit=500)[0]
         for lo, hi in zip(edges[:-1], edges[1:])),
        np.zeros(c.rank * c.rank, dtype=complex),
    ).reshape(c.rank, c.rank)
    assert _rel(got, want) < 2e-10
    assert np.linalg.eigvalsh(got).min() > -1e-12


def test_cluster_bubble_is_exact_passive_kronecker_sum():
    c = _cluster(seed=3)
    rng = np.random.default_rng(4)
    # A real vertex keeps the independent carrier oracle uncluttered by the
    # left/right conjugation convention; complex vertices are covered by the
    # algebraic spectral-identity test below.
    phi = rng.normal(size=(2, c.n_dof, c.n_dof))
    out = passive_bubble_channel(phi, c, carrier_prefactor=0.37)

    for omega in (1.65, 2.1, 2.73):
        def f(w):
            a = c.carrier(w)
            b = c.carrier(omega - w)
            v = 0.37 * np.einsum(
                "mij,ik,jl,nkl->mn", phi, a, b, phi, optimize=True)
            return v.reshape(-1) / (2.0 * np.pi)

        centres = sorted(c.poles.real.tolist()
                         + (omega - c.poles.real).tolist())
        edges = [-np.inf] + centres + [np.inf]
        want = sum(
            (quad_vec(f, lo, hi, epsabs=3e-11, epsrel=3e-11,
                      limit=600)[0]
             for lo, hi in zip(edges[:-1], edges[1:])),
            np.zeros(4, dtype=complex),
        ).reshape(2, 2)
        assert _rel(out.carrier(omega), want) < 2e-10

    assert out.rank == c.rank ** 2
    assert np.linalg.eigvalsh(out.q_lesser).min() > -1e-11
    assert np.linalg.eigvalsh(out.q_greater).min() > -1e-11


def test_retarded_part_comes_from_same_keldysh_spectral_difference():
    c = _cluster(seed=5)
    rng = np.random.default_rng(6)
    phi = (rng.normal(size=(3, c.n_dof, c.n_dof))
           + 1j * rng.normal(size=(3, c.n_dof, c.n_dof)))
    out = passive_bubble_channel(phi, c, carrier_prefactor=0.5)
    omega = np.linspace(-1.0, 4.5, 73)
    assert out.spectral_identity_error(omega) < 3e-13
    assert _rel(out.keldysh(omega).conj().transpose(0, 2, 1),
                -out.keldysh(omega)) < 3e-13


def test_local_augmented_dyson_equals_physical_schur_and_keldysh():
    rng = np.random.default_rng(8)
    sizes = np.array([2, 2, 2])
    n = int(sizes.sum())
    # Block-tridiagonal physical operator, away from singularity.
    a = np.zeros((n, n), complex)
    for i in range(3):
        sl = slice(2 * i, 2 * i + 2)
        x = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        a[sl, sl] = 4.0 * np.eye(2) + 0.08 * x
        if i < 2:
            t = 0.13 * (rng.normal(size=(2, 2))
                        + 1j * rng.normal(size=(2, 2)))
            a[sl, slice(2 * i + 2, 2 * i + 4)] = t
            a[slice(2 * i + 2, 2 * i + 4), sl] = t.conj().T

    # The state is owned by the middle block and may couple to all three: the
    # augmented matrix is still BTD, while its Schur complement can contain a
    # physical distance-two shell.
    base = _cluster(seed=9, n_dof=n)
    channel = RationalKeldyshChannel(
        base.poles, base.coupling, base.q_lesser, base.q_greater)
    local = LocalAuxiliaryChannel(channel, owner=1, block_sizes=sizes)
    smooth_l = 1j * np.diag(np.linspace(0.08, 0.13, n))
    smooth_g = 1j * np.diag(np.linspace(0.14, 0.21, n))
    omega = 2.18
    aa, al, ag, physical = assemble_augmented_dense(
        a, smooth_l, smooth_g, local, omega)
    gr_aug = np.linalg.inv(aa)
    gr = gr_aug[np.ix_(physical, physical)]
    want_gr = np.linalg.inv(physical_schur_complement(a, channel, omega))
    assert _rel(gr, want_gr) < 3e-14

    gl_aug = gr_aug @ al @ gr_aug.conj().T
    gg_aug = gr_aug @ ag @ gr_aug.conj().T
    eff_l = smooth_l + channel.keldysh(omega)
    eff_g = smooth_g + channel.keldysh(omega, greater=True)
    assert _rel(gl_aug[np.ix_(physical, physical)],
                want_gr @ eff_l @ want_gr.conj().T) < 5e-14
    assert _rel(gg_aug[np.ix_(physical, physical)],
                want_gr @ eff_g @ want_gr.conj().T) < 5e-14

    # No augmented block outside the tridiagonal remains populated.
    ao = np.concatenate(([0], np.cumsum(local.augmented_block_sizes)))
    for i in range(3):
        for j in range(3):
            if abs(i - j) > 1:
                blk = aa[ao[i]:ao[i + 1], ao[j]:ao[j + 1]]
                assert np.linalg.norm(blk) == 0.0


def test_local_channel_refuses_a_hidden_long_range_coupling():
    c = _cluster(seed=10, n_dof=8)
    ch = RationalKeldyshChannel(
        c.poles, c.coupling, c.q_lesser, c.q_greater)
    # Owner 0 may touch cells 0/1, but the random channel also touches cell 3.
    with pytest.raises(ValueError, match="SSS spatial extension"):
        LocalAuxiliaryChannel(ch, owner=0, block_sizes=np.array([2, 2, 2, 2]))


def test_dsdb_local_auxiliary_rgf_matches_dense_augmented_oracle():
    from qttools import sparse
    from qttools.datastructures import DSDBCOO

    rng = np.random.default_rng(22)
    sizes = np.array([2, 2, 2])
    n = int(sizes.sum())
    a = np.zeros((n, n), complex)
    for i in range(3):
        si = slice(2 * i, 2 * i + 2)
        a[si, si] = (3.7 + 0.1j) * np.eye(2) + 0.03 * (
            rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        if i < 2:
            sj = slice(2 * i + 2, 2 * i + 4)
            a[si, sj] = 0.08 * (rng.normal(size=(2, 2))
                                + 1j * rng.normal(size=(2, 2)))
            a[sj, si] = a[si, sj].conj().T
    sl = 1j * np.diag(np.linspace(0.05, 0.09, n))
    sg = 1j * np.diag(np.linspace(0.11, 0.16, n))

    base = _cluster(seed=23, n_dof=n)
    ch = RationalKeldyshChannel(
        base.poles, base.coupling, base.q_lesser, base.q_greater)
    local = LocalAuxiliaryChannel(ch, owner=1, block_sizes=sizes)
    freqs = np.array([1.83, 2.27])

    aa = DSDBCOO.from_sparray(
        sparse.coo_matrix(a), sizes, global_stack_shape=(freqs.size,))
    ll = DSDBCOO.from_sparray(
        sparse.coo_matrix(sl), sizes, global_stack_shape=(freqs.size,))
    gg = DSDBCOO.from_sparray(
        sparse.coo_matrix(sg), sizes, global_stack_shape=(freqs.size,))
    # N=3, band=2 is the full physical pattern and makes this an unmasked
    # parity test of every selected block.
    proto = DSDBCOO.from_sparray(
        sparse.coo_matrix(np.ones((n, n))), sizes,
        global_stack_shape=(freqs.size,))
    ol = DSDBCOO.empty_like(proto); og = DSDBCOO.empty_like(proto)
    ore = DSDBCOO.empty_like(proto)
    for m in (ol, og, ore):
        m.allocate_data(); m.data[:] = 0.0

    solver = LocalAuxiliaryRGF(
        local, freqs, max_batch_size=1, n_offdiagonals=2)
    solver.selected_solve(aa, ll, gg, (ol, og, ore), return_retarded=True)
    got_l, got_g, got_r = (np.asarray(x.to_dense()) for x in (ol, og, ore))
    for iw, w in enumerate(freqs):
        aug_a, aug_l, aug_g, phys = assemble_augmented_dense(
            a, sl, sg, local, w)
        gr = np.linalg.inv(aug_a)
        # The production RGF returns retarded diagonal blocks only; the
        # n_offdiagonals extension applies to the Keldysh pair.
        for i in range(3):
            si = slice(2 * i, 2 * i + 2)
            assert _rel(got_r[iw, si, si],
                        gr[np.ix_(phys[si], phys[si])]) < 5e-13
        assert _rel(got_l[iw], (gr @ aug_l @ gr.conj().T)[np.ix_(phys, phys)]) \
            < 8e-13
        assert _rel(got_g[iw], (gr @ aug_g @ gr.conj().T)[np.ix_(phys, phys)]) \
            < 8e-13


def test_global_woodbury_matches_dense_augmented_oracle():
    """A propagating auxiliary mode may touch every transport cell."""
    from qttools import sparse
    from qttools.datastructures import DSDBCOO

    rng = np.random.default_rng(31)
    sizes = np.array([2, 2, 2])
    n = int(sizes.sum())
    a = np.zeros((n, n), complex)
    for i in range(3):
        si = slice(2 * i, 2 * i + 2)
        a[si, si] = (4.1 + 0.12j) * np.eye(2) + 0.04 * (
            rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        if i < 2:
            sj = slice(2 * i + 2, 2 * i + 4)
            a[si, sj] = 0.07 * (rng.normal(size=(2, 2))
                                + 1j * rng.normal(size=(2, 2)))
            a[sj, si] = a[si, sj].conj().T
    sl = 1j * np.diag(np.linspace(0.04, 0.08, n))
    sg = 1j * np.diag(np.linspace(0.09, 0.15, n))
    base = _cluster(seed=32, n_dof=n)
    ch = RationalKeldyshChannel(
        base.poles, base.coupling, base.q_lesser, base.q_greater)
    # Confirm this really is global: owner 0's local adapter must reject it.
    with pytest.raises(ValueError, match="SSS spatial extension"):
        LocalAuxiliaryChannel(ch, owner=0, block_sizes=sizes)

    freqs = np.array([1.71, 2.36])
    aa = DSDBCOO.from_sparray(
        sparse.coo_matrix(a), sizes, global_stack_shape=(freqs.size,))
    ll = DSDBCOO.from_sparray(
        sparse.coo_matrix(sl), sizes, global_stack_shape=(freqs.size,))
    gg = DSDBCOO.from_sparray(
        sparse.coo_matrix(sg), sizes, global_stack_shape=(freqs.size,))
    proto = DSDBCOO.from_sparray(
        sparse.coo_matrix(np.ones((n, n))), sizes,
        global_stack_shape=(freqs.size,))
    ol = DSDBCOO.empty_like(proto); og = DSDBCOO.empty_like(proto)
    ore = DSDBCOO.empty_like(proto)
    for m in (ol, og, ore):
        m.allocate_data(); m.data[:] = 0.0

    GlobalAuxiliaryWoodbury(
        ch, freqs, max_batch_size=1, n_offdiagonals=2,
    ).selected_solve(aa, ll, gg, (ol, og, ore), return_retarded=True)
    got_l, got_g, got_r = (np.asarray(x.to_dense()) for x in (ol, og, ore))
    local = LocalAuxiliaryChannel(ch, owner=1, block_sizes=sizes)
    for iw, w in enumerate(freqs):
        aug_a, aug_l, aug_g, phys = assemble_augmented_dense(
            a, sl, sg, local, w)
        gr = np.linalg.inv(aug_a)
        want_l = (gr @ aug_l @ gr.conj().T)[np.ix_(phys, phys)]
        want_g = (gr @ aug_g @ gr.conj().T)[np.ix_(phys, phys)]
        for i in range(3):
            si = slice(2 * i, 2 * i + 2)
            assert _rel(got_r[iw, si, si],
                        gr[np.ix_(phys[si], phys[si])]) < 8e-13
        assert _rel(got_l[iw], want_l) < 9e-13
        assert _rel(got_g[iw], want_g) < 9e-13
