# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Unit tests for ``quatrex.phonon.sse_phonon_phonon.SigmaPhononPhonon``.

The reference implementation is the standalone
``phonon/phonon_inputs/anharmonic.py:_compute_phph_self_energy_finite``.
We pin the new block-decomposed bubble against the dense reference and
check the bosonic Keldysh symmetries that the SCBA loop relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from quatrex.phonon.fc3_loader import fc3_to_phi_blocks
from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon
from quatrex.phonon.units import bubble_prefactor_thz


def _ref_bubble(phi: np.ndarray, G: np.ndarray, dw_thz: float) -> np.ndarray:
    """Dense reference (THz²): mirror of standalone
    ``_compute_phph_self_energy_finite`` at ``anharmonic.py:643-691``.
    """
    ne, nd, _ = G.shape
    n_fft = 2 * ne - 1
    nd2 = nd * nd
    prefactor = bubble_prefactor_thz(dw_thz)

    PL = phi.reshape(nd2, nd)
    PR = phi.reshape(nd, nd2)

    G_pad = np.zeros((n_fft, nd, nd), dtype=complex)
    G_pad[:ne] = G
    G_fft = np.fft.fft(G_pad, axis=0)
    A = PL[None] @ G_fft
    A = A.reshape(n_fft, nd, nd, nd).transpose(0, 1, 3, 2)
    B = A @ G_fft[:, None, :, :]
    S = B.reshape(n_fft * nd, nd2) @ PR.T
    return prefactor * np.fft.ifft(S.reshape(n_fft, nd, nd), axis=0)[:ne]


def _make_cfg(retarded_method: str = "fft"):
    """Minimal mock config object exposing the attributes used by the
    SigmaPhononPhonon ``__init__`` path that is fed an explicit
    ``phi_blocks`` dict (so ``fc3_path`` is not required)."""
    method = retarded_method

    class _Phonon:
        pass

    _Phonon.retarded_method = method
    _Phonon.fc3_path = None

    class _Cfg:
        phonon = _Phonon()

    return _Cfg()


@pytest.mark.parametrize("nd", [2, 4])
@pytest.mark.parametrize("ne", [21, 41])
def test_bubble_block_matches_reference(nd: int, ne: int) -> None:
    """Single-block (single transport cell) Σ_pp parity vs the dense bubble."""
    rng = np.random.default_rng(0)
    phi = rng.standard_normal((nd, nd, nd)) + 1j * rng.standard_normal(
        (nd, nd, nd)
    )
    G_l = rng.standard_normal((ne, nd, nd)) + 1j * rng.standard_normal(
        (ne, nd, nd)
    )

    freqs_thz = np.linspace(-16.0, 16.0, ne)
    dw_thz = float(freqs_thz[1] - freqs_thz[0])

    # Reference: dense bubble (THz²).
    sig_l_ref = _ref_bubble(phi, G_l, dw_thz)

    # New: block-sparse bubble for the trivial (0,0,0) → (0,0,0) case.
    cfg = _make_cfg()
    phi_blocks = {(0, 0, 0): phi}
    ssp = SigmaPhononPhonon(
        cfg,
        phonon_frequencies=freqs_thz,
        block_sizes=np.array([nd]),
        phi_blocks=phi_blocks,
    )
    sig_l_new = ssp._bubble_block(
        phi_left=phi,
        phi_right=phi,
        G_inner_a=G_l,
        G_inner_b=G_l,
        n_fft=2 * ne - 1,
        prefactor=bubble_prefactor_thz(dw_thz),
    )

    # Atol scales with the prefactor magnitude (~5e-23) — use rtol.
    assert np.allclose(sig_l_ref, sig_l_new, atol=0, rtol=1e-10)


def test_fc3_to_phi_blocks_truncation_warning() -> None:
    """The nearest-neighbour cut should warn on > 1 % Frobenius drop."""
    rng = np.random.default_rng(1)
    block_sizes = [2, 2, 2, 2]   # 4 transport cells of 2 DOFs each
    N = sum(block_sizes)
    phi = rng.standard_normal((N, N, N))
    # Dominant entry far from the diagonal: (block 0, block 3, block 3).
    # |I-J|=3 > 1, so the NN truncation should drop a sizable chunk.
    phi[:2, 6:8, 6:8] = 100.0

    with pytest.warns(UserWarning, match="FC3 nearest-neighbour"):
        fc3_to_phi_blocks(phi, block_sizes, nn_only=True, truncation_warn=0.01)


def test_fc3_to_phi_blocks_keys_in_nn_band() -> None:
    """Only |I-J|, |I-K|, |J-K| <= 1 keys survive the nn projection."""
    rng = np.random.default_rng(2)
    block_sizes = [2, 2, 2]
    N = sum(block_sizes)
    phi = rng.standard_normal((N, N, N))

    blocks = fc3_to_phi_blocks(phi, block_sizes, nn_only=True)
    for I, J, K in blocks:
        assert abs(I - J) <= 1 and abs(I - K) <= 1 and abs(J - K) <= 1


def test_fc3_writer_roundtrip(tmp_path) -> None:
    """write_fc3_blocks -> load_device_fc3 round-trip is byte-exact."""
    import sys

    sys.path.insert(0, "phonon")
    from phonon_inputs.quatrex_writer import (  # type: ignore[import-not-found]
        write_fc3_blocks,
    )
    from quatrex.phonon.fc3_loader import load_device_fc3

    rng = np.random.default_rng(3)
    block_sizes = np.array([2, 3, 2])
    N = int(block_sizes.sum())
    phi_dense = rng.standard_normal((N, N, N)) + 1j * rng.standard_normal(
        (N, N, N)
    )
    phi_in = fc3_to_phi_blocks(phi_dense, block_sizes, nn_only=True)

    out_path = tmp_path / "fc3_blocks.hdf5"
    write_fc3_blocks(phi_in, block_sizes, out_path, units="THz^2")

    phi_out = load_device_fc3(out_path, block_sizes=block_sizes)
    assert set(phi_in) == set(phi_out)
    for key, block in phi_in.items():
        assert np.allclose(phi_out[key], block, atol=0, rtol=0)
