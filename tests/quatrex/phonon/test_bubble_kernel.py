# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Bit-identity tests for the shared 3-phonon bubble kernel.

``phonon.solver.bubble.bubble_dense`` is the single FFT convolution
used by both the dense reference solver
(``phonon/phonon_inputs/anharmonic.py``) and the production block-sparse
solver (``quatrex.phonon.sse_phonon_phonon``). These tests pin its
output against the original inline implementations on random tensors
across the slicing/DC-handling combinations actually used in the
codebase.
"""

from __future__ import annotations

import numpy as np
import pytest

from quatrex.phonon.bubble import bubble_dense


def _ref_dense_anharmonic(Phi, G_lesser, G_greater, dw_thz, hbar_si):
    """Original inline body of ``_compute_phph_self_energy_finite``
    (anharmonic.py @ git rev before solver-unification). Returns
    ``(sig_l, sig_g)``."""
    n_freq, nd, _ = G_lesser.shape
    n_fft = 2 * n_freq - 1
    mid = n_freq // 2
    freq_sl = slice(mid, mid + n_freq)
    nd2 = nd * nd
    prefactor = 0.5j * hbar_si * dw_thz / (2 * np.pi)

    G_l_clean = G_lesser.copy()
    G_l_clean[mid] = 0.0
    G_g_clean = G_greater.copy()
    G_g_clean[mid] = 0.0

    PL_flat = Phi.reshape(nd2, nd)
    PR_flat = Phi.reshape(nd, nd2)

    sig_l = np.zeros((n_freq, nd, nd), dtype=complex)
    sig_g = np.zeros_like(sig_l)
    for G_full, sig_out in ((G_l_clean, sig_l), (G_g_clean, sig_g)):
        G_pad = np.zeros((n_fft, nd, nd), dtype=complex)
        G_pad[:n_freq] = G_full
        G_fft = np.fft.fft(G_pad, axis=0)
        A = PL_flat[None] @ G_fft
        A = A.reshape(n_fft, nd, nd, nd).transpose(0, 1, 3, 2)
        B = A @ G_fft[:, None, :, :]
        S = B.reshape(n_fft * nd, nd2) @ PR_flat.T
        sig_out[:] = prefactor * np.fft.ifft(
            S.reshape(n_fft, nd, nd), axis=0)[freq_sl]
    return sig_l, sig_g


def _ref_sparse_bubble_block(
    phi_left, phi_right, G_inner_a, G_inner_b, n_fft, prefactor
):
    """Original inline body of ``SigmaPhononPhonon._bubble_block`` (git
    rev before solver-unification). Returns one ``(ne, nI, nJ)`` block."""
    ne = G_inner_a.shape[0]
    bI, bK1, bK2 = phi_left.shape
    bJ = phi_right.shape[0]

    Ga_pad = np.zeros((n_fft, bK1, bK1), dtype=complex)
    Ga_pad[:ne] = G_inner_a
    Gb_pad = np.zeros((n_fft, bK2, bK2), dtype=complex)
    Gb_pad[:ne] = G_inner_b

    Ga_fft = np.fft.fft(Ga_pad, axis=0)
    Gb_fft = np.fft.fft(Gb_pad, axis=0)

    A = np.einsum("ace,wed->wacd", phi_left, Gb_fft)
    B = np.einsum("wacd,wcb->wabd", A, Ga_fft)
    S_hat = np.einsum("wabd,Jdb->waJ", B, phi_right)
    return prefactor * np.fft.ifft(S_hat, axis=0)[:ne]


@pytest.mark.parametrize("nd", [2, 3, 4])
@pytest.mark.parametrize("n_freq", [5, 7, 9])
def test_bubble_matches_dense_reference(nd, n_freq):
    """Dense convention: symmetric grid, DC zeroed, centered slice."""
    rng = np.random.default_rng(seed=42 + nd * 100 + n_freq)
    Phi = rng.standard_normal((nd, nd, nd)) + 1j * rng.standard_normal((nd, nd, nd))
    G_l = rng.standard_normal((n_freq, nd, nd)) + 1j * rng.standard_normal((n_freq, nd, nd))
    G_g = rng.standard_normal((n_freq, nd, nd)) + 1j * rng.standard_normal((n_freq, nd, nd))
    dw_thz = 0.17
    hbar_si = 1.0545718e-34

    sig_l_ref, sig_g_ref = _ref_dense_anharmonic(Phi, G_l, G_g, dw_thz, hbar_si)

    n_fft = 2 * n_freq - 1
    mid = n_freq // 2
    prefactor = 0.5j * hbar_si * dw_thz / (2 * np.pi)
    sig_l = bubble_dense(
        phi_left=Phi, phi_right=Phi, G_a=G_l, G_b=G_l,
        n_fft=n_fft, prefactor=prefactor,
        out_slice=slice(mid, mid + n_freq), zero_freq_idx=mid,
    )
    sig_g = bubble_dense(
        phi_left=Phi, phi_right=Phi, G_a=G_g, G_b=G_g,
        n_fft=n_fft, prefactor=prefactor,
        out_slice=slice(mid, mid + n_freq), zero_freq_idx=mid,
    )
    np.testing.assert_allclose(sig_l, sig_l_ref, atol=1e-12)
    np.testing.assert_allclose(sig_g, sig_g_ref, atol=1e-12)


@pytest.mark.parametrize("bK1, bK2", [(2, 2), (3, 4), (4, 3)])
@pytest.mark.parametrize("ne", [4, 8])
def test_bubble_matches_sparse_block_reference(bK1, bK2, ne):
    """Sparse convention: positive-only grid, no DC handling, [:ne] slice."""
    rng = np.random.default_rng(seed=7 + bK1 * 17 + bK2 * 3 + ne)
    nI, nJ = 2, 3
    phi_left = (
        rng.standard_normal((nI, bK1, bK2))
        + 1j * rng.standard_normal((nI, bK1, bK2))
    )
    phi_right = (
        rng.standard_normal((nJ, bK2, bK1))
        + 1j * rng.standard_normal((nJ, bK2, bK1))
    )
    G_a = (
        rng.standard_normal((ne, bK1, bK1))
        + 1j * rng.standard_normal((ne, bK1, bK1))
    )
    G_b = (
        rng.standard_normal((ne, bK2, bK2))
        + 1j * rng.standard_normal((ne, bK2, bK2))
    )
    n_fft = 2 * ne - 1
    prefactor = 0.5j * 1.0545718e-34 * 0.1 / (2 * np.pi)

    ref = _ref_sparse_bubble_block(
        phi_left, phi_right, G_a, G_b, n_fft, prefactor
    )
    got = bubble_dense(
        phi_left=phi_left, phi_right=phi_right,
        G_a=G_a, G_b=G_b, n_fft=n_fft, prefactor=prefactor,
    )
    np.testing.assert_allclose(got, ref, atol=1e-12)


@pytest.mark.parametrize("bK1,bK1p,bK2,bK2p", [
    # Off-diagonal G blocks: G(K, K') has bK != bK' on the trailing axis.
    # Mirrors what compute_sse_with_cutoffs passes when diag_G_in_se=False:
    # the inner G dict spans every (K, K') with |K-K'| <= 1.
    (9, 6, 6, 9),   # both G off-diagonal, asymmetric inner blocks
    (4, 6, 4, 4),   # only G_a off-diagonal
    (3, 3, 5, 7),   # only G_b off-diagonal
])
def test_bubble_handles_rectangular_offdiagonal_G(bK1, bK1p, bK2, bK2p):
    """Off-diagonal G(K, K') blocks have ``bK != bK'`` on the trailing
    axis. The kernel must size its pad from G_a/G_b directly — sizing
    from phi_left silently truncates the trailing axis."""
    rng = np.random.default_rng(seed=bK1 * 100 + bK2 + bK1p + bK2p)
    nI, nJ, ne = 2, 3, 5
    # phi_left has the "outgoing" K1, K2 shape; phi_right ties them back
    # via the K1', K2' inner indices on the J vertex.
    phi_left = (
        rng.standard_normal((nI, bK1, bK2))
        + 1j * rng.standard_normal((nI, bK1, bK2))
    )
    phi_right = (
        rng.standard_normal((nJ, bK2p, bK1p))
        + 1j * rng.standard_normal((nJ, bK2p, bK1p))
    )
    G_a = (
        rng.standard_normal((ne, bK1, bK1p))
        + 1j * rng.standard_normal((ne, bK1, bK1p))
    )
    G_b = (
        rng.standard_normal((ne, bK2, bK2p))
        + 1j * rng.standard_normal((ne, bK2, bK2p))
    )
    n_fft = 2 * ne - 1
    prefactor = 0.5j * 1.0545718e-34 * 0.1 / (2 * np.pi)

    # Direct ground truth: inline the einsum chain with the right shapes.
    Ga_pad = np.zeros((n_fft, bK1, bK1p), dtype=complex)
    Ga_pad[:ne] = G_a
    Gb_pad = np.zeros((n_fft, bK2, bK2p), dtype=complex)
    Gb_pad[:ne] = G_b
    Ga_fft = np.fft.fft(Ga_pad, axis=0)
    Gb_fft = np.fft.fft(Gb_pad, axis=0)
    A = np.einsum("ace,wed->wacd", phi_left, Gb_fft)
    B = np.einsum("wacd,wcb->wabd", A, Ga_fft)
    S_hat = np.einsum("wabd,Jdb->waJ", B, phi_right)
    expected = prefactor * np.fft.ifft(S_hat, axis=0)[:ne]

    got = bubble_dense(
        phi_left=phi_left, phi_right=phi_right,
        G_a=G_a, G_b=G_b, n_fft=n_fft, prefactor=prefactor,
    )
    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert got.shape == (ne, nI, nJ)


def test_bubble_input_not_mutated_when_zero_freq_set():
    """``zero_freq_idx`` must not modify the caller's arrays."""
    rng = np.random.default_rng(seed=1)
    nd, n_freq = 3, 7
    Phi = rng.standard_normal((nd, nd, nd)) + 0j
    G = rng.standard_normal((n_freq, nd, nd)) + 1j * rng.standard_normal((n_freq, nd, nd))
    G_snapshot = G.copy()
    bubble_dense(
        phi_left=Phi, phi_right=Phi, G_a=G, G_b=G,
        n_fft=2 * n_freq - 1, prefactor=1.0,
        out_slice=slice(0, n_freq), zero_freq_idx=n_freq // 2,
    )
    np.testing.assert_array_equal(G, G_snapshot)


# --------------------------------------------------------------------------
# Task batching: the same contraction with the task axis carried as a batch.
#
# Not a CPU optimisation -- measured 2.5x at d=1, 0.68-0.90x at d=4-6, where
# numpy is already at its FLOP/memory bound. It exists because a GPU cannot
# use the per-task form: on a GH200 the per-task path runs at 0.09-1.3x of the
# CPU while the batched one runs at 13x/5.5x/72x/111x for d=1/2/4/6. These
# tests pin it to the per-task path BIT-for-bit, because the whole argument
# for enabling it rests on it being the same arithmetic.
# --------------------------------------------------------------------------

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "phonon"))


@pytest.mark.parametrize("n_dof", [1, 2, 3])
@pytest.mark.parametrize("max_bytes", [None, 1 << 13])
def test_the_task_batched_bubble_is_bit_identical_to_the_per_task_call(
        n_dof, max_bytes):
    from solver.bubble import (bubble_dense_from_fft,
                               bubble_dense_from_fft_task_batched)

    rng = np.random.default_rng(17)
    n_task, n_fft, ne = 6, 45, 23
    d = n_dof

    def mk(*s):
        return rng.normal(size=s) + 1j * rng.normal(size=s)

    pl, pr = mk(n_task, d, d, d), mk(n_task, d, d, d)
    ga, gb = mk(n_task, n_fft, d, d), mk(n_task, n_fft, d, d)
    sl, pre = slice(11, 11 + ne), 0.5 - 0.25j

    ref = np.stack([
        bubble_dense_from_fft(phi_left=pl[t], phi_right=pr[t], G_a_fft=ga[t],
                              G_b_fft=gb[t], ne=ne, prefactor=pre,
                              out_slice=sl)
        for t in range(n_task)])
    got = bubble_dense_from_fft_task_batched(
        phi_left=pl, phi_right=pr, G_a_fft=ga, G_b_fft=gb, ne=ne,
        prefactor=pre, out_slice=sl, max_bytes=max_bytes)

    assert got.shape == ref.shape
    assert np.array_equal(got, ref), np.abs(got - ref).max()


def _random_phph_inputs(rng, n_slabs, n_dof, n_freq):
    """A small Gamma-only multi-slab problem: nearest-neighbour G and a local
    vertex, which is enough to exercise every task-key collision."""
    def mk(*s):
        return rng.normal(size=s) + 1j * rng.normal(size=s)

    gl, gg = {}, {}
    for i in range(n_slabs):
        for j in range(n_slabs):
            if abs(i - j) <= 1:
                gl[(i, j)] = mk(n_freq, n_dof, n_dof)
                gg[(i, j)] = mk(n_freq, n_dof, n_dof)
    phi = {(i, k, kp): mk(n_dof, n_dof, n_dof)
           for i in range(n_slabs)
           for k in range(n_slabs) for kp in range(n_slabs)
           if abs(i - k) <= 1 and abs(i - kp) <= 1}
    return gl, gg, phi


@pytest.mark.parametrize("n_dof", [1, 2])
def test_the_batched_self_energy_equals_the_per_task_one_bit_for_bit(n_dof):
    """The switch is opt-in precisely because every stored result came from
    the per-task path; that is only defensible if the two agree exactly."""
    from solver.se_finite import compute_phph_self_energy_finite_multi_slab

    rng = np.random.default_rng(23)
    n_slabs, n_freq = 4, 33
    gl, gg, phi = _random_phph_inputs(rng, n_slabs, n_dof, n_freq)
    freqs = np.linspace(0.0, 4.0, n_freq)
    kw = dict(sigma_cutoff=None, g_cutoff=None, dc_handling="interpolate",
              n_threads=1)

    prev = os.environ.get("QX_PHPH_BATCHED")
    try:
        os.environ.pop("QX_PHPH_BATCHED", None)
        ref_l, ref_g = compute_phph_self_energy_finite_multi_slab(
            gl, gg, phi, n_slabs, freqs, 0.125, **kw)
        os.environ["QX_PHPH_BATCHED"] = "1"
        got_l, got_g = compute_phph_self_energy_finite_multi_slab(
            gl, gg, phi, n_slabs, freqs, 0.125, **kw)
    finally:
        os.environ.pop("QX_PHPH_BATCHED", None)
        if prev is not None:
            os.environ["QX_PHPH_BATCHED"] = prev

    assert set(got_l) == set(ref_l) and set(got_g) == set(ref_g)
    for key in ref_l:
        assert np.array_equal(got_l[key], ref_l[key]), key
        assert np.array_equal(got_g[key], ref_g[key]), key


def test_an_unknown_array_module_is_refused_rather_than_silently_ignored():
    """A typo in QX_PHPH_XP must not fall back to numpy and quietly produce a
    CPU run the caller believes was on a GPU."""
    from solver.se_finite import compute_phph_self_energy_finite_multi_slab

    rng = np.random.default_rng(5)
    gl, gg, phi = _random_phph_inputs(rng, 3, 1, 17)
    prev = os.environ.get("QX_PHPH_XP")
    try:
        os.environ["QX_PHPH_XP"] = "cuppy"
        with pytest.raises(ValueError, match="expected numpy or cupy"):
            compute_phph_self_energy_finite_multi_slab(
                gl, gg, phi, 3, np.linspace(0.0, 2.0, 17), 0.125,
                sigma_cutoff=None, g_cutoff=None, n_threads=1)
    finally:
        os.environ.pop("QX_PHPH_XP", None)
        if prev is not None:
            os.environ["QX_PHPH_XP"] = prev
