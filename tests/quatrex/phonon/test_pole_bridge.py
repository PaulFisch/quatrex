# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Pattern-projected bridge kernels against their dense equivalents.

The bridge exists because the dense intermediates are unaffordable: at
production size (2001 frequencies, 300 degrees of freedom) a single
``(n_omega, n_dof, n_dof)`` buffer is 2.9 GB. Every kernel here therefore
contracts straight onto the stored sparsity pattern, and each one is pinned
against the dense form it replaces.
"""
import numpy as np
import pytest

from quatrex.phonon.experimental.pole.pole_bridge import (
    add_contact_source,
    analytic_prefactor,
    modal_vertex_blocks,
    pole_keldysh_sparse,
    project_source_sparse,
    ss_self_energy_sparse,
)
from quatrex.phonon.experimental.pole.pole_bubble import modal_convolution, modal_vertex
from quatrex.phonon.experimental.pole.pole_keldysh import PoleCluster, pole_keldysh, project_source
from quatrex.phonon.units import HBAR_SI, bubble_prefactor_thz


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _pattern(sizes):
    off = np.concatenate(([0], np.cumsum(sizes)))
    rows, cols = [], []
    for i in range(len(sizes)):
        for j in range(max(0, i - 1), min(len(sizes), i + 2)):
            for a in range(off[i], off[i + 1]):
                for b in range(off[j], off[j + 1]):
                    rows.append(a)
                    cols.append(b)
    return np.array(rows), np.array(cols), off


def _cluster(n_dof, npp=2, seed=0):
    rng = np.random.default_rng(seed)
    z = np.array([7.0 - 0.05j, 11.0 - 0.08j])[:npp]
    u = rng.normal(size=(n_dof, npp)) + 1j * rng.normal(size=(n_dof, npp))
    v = rng.normal(size=(n_dof, npp)) + 1j * rng.normal(size=(n_dof, npp))
    return PoleCluster(z=z, u=u, v=v)


def test_sparse_source_projection_matches_the_dense_form():
    sizes = np.array([3, 3, 2])
    rows, cols, _ = _pattern(sizes)
    n = int(sizes.sum())
    cl = _cluster(n)
    rng = np.random.default_rng(1)
    ne = 7
    vals = rng.normal(size=(ne, rows.size)) + 1j * rng.normal(size=(ne, rows.size))

    dense = np.zeros((ne, n, n), dtype=complex)
    dense[:, rows, cols] = vals

    got = _h(project_source_sparse(vals, rows, cols, _h(cl.v)))
    ref = _h(project_source(dense, _h(cl.v)))
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-12


def test_contact_source_matches_a_dense_corner():
    sizes = np.array([3, 3])
    n = int(sizes.sum())
    cl = _cluster(n)
    rng = np.random.default_rng(2)
    ne, b = 5, 3
    corner = rng.normal(size=(ne, b, b)) + 1j * rng.normal(size=(ne, b, b))

    zero = np.zeros((ne, cl.n_poles, cl.n_poles), dtype=complex)
    got = _h(add_contact_source(zero, corner, _h(cl.v), offset=0))

    dense = np.zeros((ne, n, n), dtype=complex)
    dense[:, :b, :b] = corner
    ref = _h(project_source(dense, _h(cl.v)))
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-12


def test_sparse_pole_keldysh_matches_the_dense_form():
    sizes = np.array([3, 3, 2])
    rows, cols, _ = _pattern(sizes)
    n = int(sizes.sum())
    cl = _cluster(n)
    rng = np.random.default_rng(3)
    omega = np.linspace(5.0, 13.0, 11)
    a = rng.normal(size=(len(omega), cl.n_poles, cl.n_poles)) \
        + 1j * rng.normal(size=(len(omega), cl.n_poles, cl.n_poles))
    src = a @ a.conj().swapaxes(-2, -1)

    got = _h(pole_keldysh_sparse(omega, cl, src, rows, cols))
    ref = _h(pole_keldysh(omega, cl, src))[:, rows, cols]
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-12


def test_block_sparse_modal_vertex_matches_the_dense_projection():
    sizes = np.array([2, 2, 2])
    n = int(sizes.sum())
    off = np.concatenate(([0], np.cumsum(sizes)))
    rng = np.random.default_rng(4)
    phi_blocks, dense = {}, np.zeros((n, n, n))
    for i in range(3):
        for k1 in range(max(0, i - 1), min(3, i + 2)):
            for k2 in range(max(0, i - 1), min(3, i + 2)):
                b = rng.normal(size=(2, 2, 2))
                phi_blocks[(i, k1, k2)] = b
                dense[off[i]:off[i + 1], off[k1]:off[k1 + 1],
                      off[k2]:off[k2 + 1]] = b
    cl = _cluster(n)

    got = _h(modal_vertex_blocks(phi_blocks, sizes, cl.u, conjugate=False))
    ref = _h(modal_vertex(dense, _h(cl.u)))
    assert np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30) < 1e-12

    # The right-hand factor is conjugated: that is what makes the contraction a
    # congruence and carries the positivity statement.
    conj = _h(modal_vertex_blocks(phi_blocks, sizes, cl.u, conjugate=True))
    assert np.abs(conj - np.conj(ref)).max() / np.abs(ref).max() < 1e-12


def test_sparse_ss_self_energy_matches_an_explicit_contraction():
    sizes = np.array([2, 2])
    rows, cols, off = _pattern(sizes)
    n = int(sizes.sum())
    cl = _cluster(n)
    rng = np.random.default_rng(5)
    phi_blocks = {}
    for i in range(2):
        for k1 in range(2):
            for k2 in range(2):
                phi_blocks[(i, k1, k2)] = rng.normal(size=(2, 2, 2))
    vl = modal_vertex_blocks(phi_blocks, sizes, cl.u, conjugate=False)
    vr = modal_vertex_blocks(phi_blocks, sizes, cl.u, conjugate=True)
    src = np.eye(cl.n_poles, dtype=complex)
    omega = np.array([6.0, 18.0])

    got = _h(ss_self_energy_sparse(omega, cl, src, src, vl, vr, rows, cols))
    c = _h(modal_convolution(omega, cl, src, src))
    ref = analytic_prefactor() * np.einsum(
        "kAB,kGD,wADBG->wk", _h(vl)[rows], _h(vr)[cols], c
    )
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-12


def test_analytic_prefactor_drops_the_grid_spacing():
    """The analytic channel evaluates the integral, so it must not carry dw.

    The production prefactor is ``0.5j*hbar*dw/(2*pi)`` -- the ``dw/(2*pi)``
    converts a discrete sum into ``int dw'/(2*pi)``. The closed form already IS
    that integral, so including dw would make a grid-free term scale with a grid.
    """
    assert analytic_prefactor() == 0.5j * HBAR_SI
    dw = 0.075
    assert np.isclose(
        bubble_prefactor_thz(dw) * (2.0 * np.pi / dw), analytic_prefactor()
    )
