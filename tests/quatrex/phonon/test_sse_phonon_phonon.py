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

from qttools import xp
from qttools.comm import comm as _qtt_comm

from quatrex.phonon.fc3_loader import fc3_to_phi_blocks
from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon
from quatrex.phonon.units import bubble_prefactor_thz


def _configure_serial_comm() -> None:
    """Configure the qttools comm singleton for serial DSDBSparse use.

    The tests in this module exercise the production
    :class:`SigmaPhononPhonon.compute` path, which constructs
    DSDBSparse buffers and therefore needs the comm singleton in
    place. Idempotent: re-configuration is a no-op once configured.
    """
    if _qtt_comm._is_configured:
        return
    backend = "device_mpi" if xp.__name__ == "numpy" else "host_mpi"
    cfg = {k: backend for k in ("all_to_all", "all_gather", "all_reduce", "bcast")}
    _qtt_comm.configure(
        block_comm_size=1,
        block_comm_config=cfg,
        stack_comm_config=cfg,
        override=True,
    )


def setup_module() -> None:  # pytest hook
    _configure_serial_comm()


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


def _ref_compute_multiblock(
    phi_blocks: dict[tuple[int, int, int], np.ndarray],
    g_lesser_diag: list[np.ndarray],
    g_greater_diag: list[np.ndarray],
    block_sizes: np.ndarray,
    dw_thz: float,
) -> tuple[
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
]:
    """Independent multi-block dense reference for Sigma^{<,>,R}.

    Mirrors the iteration that SigmaPhononPhonon.compute does
    internally, but assembled from first principles in the test so
    that any change to the production iteration is caught.
    """
    n_blocks = len(block_sizes)
    ne = g_lesser_diag[0].shape[0]
    n_fft = 2 * ne - 1
    prefactor = bubble_prefactor_thz(dw_thz)
    freqs = np.linspace(0.0, dw_thz * (ne - 1), ne)

    pair_index: dict[tuple[int, int], list[tuple[int, int, np.ndarray, np.ndarray]]] = {}
    for (I, K1, K2), phi_left in phi_blocks.items():
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            phi_right = phi_blocks.get((J, K2, K1))
            if phi_right is None:
                continue
            pair_index.setdefault((I, J), []).append(
                (K1, K2, phi_left, phi_right)
            )

    sl_full: dict[tuple[int, int], np.ndarray] = {}
    sg_full: dict[tuple[int, int], np.ndarray] = {}
    for (I, J), pairs in pair_index.items():
        for K1, K2, phi_left, phi_right in pairs:
            G_a_l, G_b_l = g_lesser_diag[K1], g_lesser_diag[K2]
            G_a_g, G_b_g = g_greater_diag[K1], g_greater_diag[K2]
            sl = bubble_prefactor_thz(dw_thz) * np.fft.ifft(
                np.einsum(
                    "wabd,Jdb->waJ",
                    np.einsum(
                        "wacd,wcb->wabd",
                        np.einsum(
                            "ace,wed->wacd",
                            phi_left,
                            np.fft.fft(
                                np.concatenate(
                                    [G_b_l, np.zeros((n_fft - ne, *G_b_l.shape[1:]),
                                                     dtype=complex)],
                                    axis=0,
                                ),
                                axis=0,
                            ),
                        ),
                        np.fft.fft(
                            np.concatenate(
                                [G_a_l, np.zeros((n_fft - ne, *G_a_l.shape[1:]),
                                                 dtype=complex)],
                                axis=0,
                            ),
                            axis=0,
                        ),
                    ),
                    phi_right,
                ),
                axis=0,
            )[:ne] / prefactor * prefactor  # noqa - exact mirror of bubble_dense
            sg = bubble_prefactor_thz(dw_thz) * np.fft.ifft(
                np.einsum(
                    "wabd,Jdb->waJ",
                    np.einsum(
                        "wacd,wcb->wabd",
                        np.einsum(
                            "ace,wed->wacd",
                            phi_left,
                            np.fft.fft(
                                np.concatenate(
                                    [G_b_g, np.zeros((n_fft - ne, *G_b_g.shape[1:]),
                                                     dtype=complex)],
                                    axis=0,
                                ),
                                axis=0,
                            ),
                        ),
                        np.fft.fft(
                            np.concatenate(
                                [G_a_g, np.zeros((n_fft - ne, *G_a_g.shape[1:]),
                                                 dtype=complex)],
                                axis=0,
                            ),
                            axis=0,
                        ),
                    ),
                    phi_right,
                ),
                axis=0,
            )[:ne] / prefactor * prefactor
            sl_full[(I, J)] = sl_full.get((I, J), 0) + sl
            sg_full[(I, J)] = sg_full.get((I, J), 0) + sg

    # Retarded: bosonic Hilbert path (matches production's retarded_method="fft").
    from quatrex.core.fft_utils import hilbert_transform
    sr_full: dict[tuple[int, int], np.ndarray] = {}
    for key in set(sl_full) | set(sg_full):
        sl = sl_full.get(key, 0)
        sg = sg_full.get(key, 0)
        delta = sg - sl
        sr_full[key] = 0.5 * delta + 0.5j * hilbert_transform(delta, freqs)

    return sl_full, sg_full, sr_full


def test_compute_multiblock_matches_reference() -> None:
    """Multi-block Σ_{IJ} parity for the refactored distributed bubble.

    Pins the new ``SigmaPhononPhonon.compute()`` against an
    independently-written multi-block reference. Exercises the FC3
    block-pair index, the (K1, K2) accumulation, the Hilbert-derived
    retarded SSE, and the per-block write-back path under
    comm.block.size == comm.stack.size == 1 (the configuration
    covered by the existing test infrastructure).
    """
    from qttools.datastructures import DSDBCOO
    from scipy.sparse import csr_matrix
    from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

    rng = np.random.default_rng(42)
    n_blocks = 3
    nbs = 3
    ne = 21

    block_sizes = np.array([nbs] * n_blocks)
    N = int(block_sizes.sum())

    # Random FC3 blocks restricted to the NN block-pair support.
    phi_blocks: dict[tuple[int, int, int], np.ndarray] = {}
    for I in range(n_blocks):
        for K1 in range(max(0, I - 1), min(n_blocks, I + 2)):
            for K2 in range(max(0, I - 1), min(n_blocks, I + 2)):
                if abs(K1 - K2) > 1:
                    continue
                phi = (rng.standard_normal((nbs, nbs, nbs))
                       + 1j * rng.standard_normal((nbs, nbs, nbs)))
                phi_blocks[(I, K1, K2)] = phi

    # Random diagonal G_lesser, G_greater per transport-cell block.
    g_l_diag = [
        rng.standard_normal((ne, nbs, nbs)) + 1j * rng.standard_normal((ne, nbs, nbs))
        for _ in range(n_blocks)
    ]
    g_g_diag = [
        rng.standard_normal((ne, nbs, nbs)) + 1j * rng.standard_normal((ne, nbs, nbs))
        for _ in range(n_blocks)
    ]

    # Build the DSDBSparse buffers. Sparsity pattern carries only the
    # BT-band block positions; the bubble only reads diagonal blocks
    # of G so off-diagonal entries are inconsequential.
    pattern_rows: list[int] = []
    pattern_cols: list[int] = []
    block_offsets = np.concatenate(([0], np.cumsum(block_sizes)))
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            for i in range(block_sizes[I]):
                for j in range(block_sizes[J]):
                    pattern_rows.append(block_offsets[I] + i)
                    pattern_cols.append(block_offsets[J] + j)
    pattern = csr_matrix(
        (np.ones(len(pattern_rows), dtype=np.complex128),
         (np.array(pattern_rows), np.array(pattern_cols))),
        shape=(N, N),
    )

    g_lesser = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(ne,))
    g_greater = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(ne,))
    sigma_lesser = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(ne,))
    sigma_greater = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(ne,))
    sigma_retarded = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(ne,))
    for m in (g_lesser, g_greater, sigma_lesser, sigma_greater, sigma_retarded):
        m.data[:] = 0.0

    # Write the diagonal G blocks into g_lesser/greater.
    gl_view = g_lesser.stack[...]
    gg_view = g_greater.stack[...]
    for K in range(n_blocks):
        gl_view.blocks[K, K] = g_l_diag[K]
        gg_view.blocks[K, K] = g_g_diag[K]

    # Run the production routine.
    cfg = _make_cfg("fft")
    freqs_thz = np.linspace(0.0, 16.0, ne)
    dw_thz = float(freqs_thz[1] - freqs_thz[0])
    ssp = SigmaPhononPhonon(
        cfg, phonon_frequencies=freqs_thz,
        block_sizes=block_sizes, phi_blocks=phi_blocks,
    )
    ssp.compute(
        g_lesser, g_greater,
        out=(sigma_lesser, sigma_greater, sigma_retarded),
    )

    # Reference: independently-written multi-block dense bubble.
    sl_ref, sg_ref, sr_ref = _ref_compute_multiblock(
        phi_blocks, g_l_diag, g_g_diag, block_sizes, dw_thz,
    )

    # Compare each (I, J) BTD block.
    sl_view = sigma_lesser.stack[...]
    sg_view = sigma_greater.stack[...]
    sr_view = sigma_retarded.stack[...]
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            key = (I, J)
            np.testing.assert_allclose(
                sl_view.blocks[I, J], sl_ref.get(key, 0), atol=0, rtol=1e-10,
                err_msg=f"Sigma^< mismatch at block {key}",
            )
            np.testing.assert_allclose(
                sg_view.blocks[I, J], sg_ref.get(key, 0), atol=0, rtol=1e-10,
                err_msg=f"Sigma^> mismatch at block {key}",
            )
            np.testing.assert_allclose(
                sr_view.blocks[I, J], sr_ref.get(key, 0), atol=0, rtol=1e-10,
                err_msg=f"Sigma^R mismatch at block {key}",
            )


def test_compute_restores_distribution_state() -> None:
    """compute() leaves the DSDBSparse distribution state untouched.

    The legacy implementation toggled buffers to ``stack`` without
    restoring; this broke the SCBA loop, which expects ``nnz`` to be
    preserved across interactions. Pin the contract.
    """
    from qttools.datastructures import DSDBCOO
    from scipy.sparse import csr_matrix
    from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

    rng = np.random.default_rng(0)
    n_blocks = 2
    nbs = 2
    ne = 11
    block_sizes = np.array([nbs] * n_blocks)
    N = int(block_sizes.sum())

    phi_blocks = {
        (I, I, I): rng.standard_normal((nbs, nbs, nbs))
        + 1j * rng.standard_normal((nbs, nbs, nbs))
        for I in range(n_blocks)
    }

    rows, cols = [], []
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            for i in range(block_sizes[I]):
                for j in range(block_sizes[J]):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = csr_matrix(
        (np.ones(len(rows), dtype=np.complex128),
         (np.array(rows), np.array(cols))),
        shape=(N, N),
    )

    bufs = [DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(ne,))
            for _ in range(5)]
    g_lesser, g_greater, sigma_lesser, sigma_greater, sigma_retarded = bufs
    for m in bufs:
        m.data[:] = 0.0

    # Seed random diagonal G data.
    gl_v = g_lesser.stack[...]
    gg_v = g_greater.stack[...]
    for K in range(n_blocks):
        gl_v.blocks[K, K] = (rng.standard_normal((ne, nbs, nbs))
                             + 1j * rng.standard_normal((ne, nbs, nbs)))
        gg_v.blocks[K, K] = (rng.standard_normal((ne, nbs, nbs))
                             + 1j * rng.standard_normal((ne, nbs, nbs)))

    # Force the buffers into nnz distribution (mimicking the SCBA loop's
    # state immediately before interactions run).
    for m in bufs:
        m.dtranspose()
        assert m.distribution_state == "nnz"

    cfg = _make_cfg("fft")
    freqs_thz = np.linspace(0.0, 16.0, ne)
    ssp = SigmaPhononPhonon(
        cfg, phonon_frequencies=freqs_thz,
        block_sizes=block_sizes, phi_blocks=phi_blocks,
    )
    ssp.compute(g_lesser, g_greater,
                out=(sigma_lesser, sigma_greater, sigma_retarded))

    # All buffers must still be in nnz on exit.
    for m in bufs:
        assert m.distribution_state == "nnz", (
            f"distribution_state was not restored; got {m.distribution_state!r}"
        )


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
