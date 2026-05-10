"""Bit-for-bit compatibility helpers for cross-checking against legacy code.

The production :mod:`finite_analysis` path uses the symmetric-ω convention
that matches :func:`phonon_inputs.anharmonic._compute_phph_self_energy_finite`
(see :func:`finite_analysis.sse_cutoffs._bubble_block_standalone`). The
older quatrex :func:`SigmaPhononPhonon._bubble_block` uses the opposite
convention: G is treated as a one-sided positive-frequency array with no
ω=0 zeroing and the IFFT output is sliced ``[:n_freq]``. This module
exposes that legacy bubble for the test fixture in
``test_phi_roundtrip.py`` and any future quatrex regression checks.
"""

from __future__ import annotations

import numpy as np


def bubble_block_legacy_quatrex(
    phi_left: np.ndarray, phi_right: np.ndarray,
    G_inner_a: np.ndarray, G_inner_b: np.ndarray,
    n_fft: int, prefactor: complex,
) -> np.ndarray:
    """Reproduce :func:`quatrex.phonon.sse_phonon_phonon._bubble_block`
    bit-for-bit on the same inputs."""
    ne = G_inner_a.shape[0]
    bI, bK1, bK2 = phi_left.shape
    bJ, bK2_prime, bK1_prime = phi_right.shape
    assert G_inner_a.shape == (ne, bK1, bK1_prime)
    assert G_inner_b.shape == (ne, bK2, bK2_prime)

    Ga_pad = np.zeros((n_fft, bK1, bK1_prime), dtype=complex)
    Gb_pad = np.zeros((n_fft, bK2, bK2_prime), dtype=complex)
    Ga_pad[:ne] = G_inner_a
    Gb_pad[:ne] = G_inner_b

    Ga_fft = np.fft.fft(Ga_pad, axis=0)
    Gb_fft = np.fft.fft(Gb_pad, axis=0)

    A = np.einsum("ace,wed->wacd", phi_left, Gb_fft)
    B = np.einsum("wacd,wcb->wabd", A, Ga_fft)
    S_hat = np.einsum("wabd,Jdb->waJ", B, phi_right)
    return prefactor * np.fft.ifft(S_hat, axis=0)[:ne]
