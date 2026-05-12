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
