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
    # Deliberately tiny, so the tests exercise the tau-chunked path: the kernel
    # must give the same answer for any chunking.
    _Phonon.sse_tau_chunk_bytes = 4096

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
    assert np.allclose(sig_l_ref, sig_l_new, atol=1e-40, rtol=1e-10)


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


def _ref_fft(G, n_fft):
    pad = np.concatenate(
        [G, np.zeros((n_fft - G.shape[0], *G.shape[1:]), dtype=complex)], axis=0)
    return np.fft.fft(pad, axis=0)


def _ref_rev(X):
    """DFT index-reversal X[(-l) mod N] for the (non-conjugating) correlation."""
    return np.concatenate([X[:1], X[:0:-1]], axis=0)


def _ref_bubble_F(phi_left, phi_right, Fa, Fb, ne, prefactor):
    """One ring quad from pre-FFT'd legs Fa, Fb."""
    S = np.einsum(
        "wabd,Jdb->waJ",
        np.einsum("wacd,wcb->wabd",
                  np.einsum("ace,wed->wacd", phi_left, Fb), Fa),
        phi_right,
    )
    return prefactor * np.fft.ifft(S, axis=0)[:ne]


def _ref_quad(phi_left, phi_right, G_a, G_b, dw_thz):
    """One-sided (decay-only) bubble for one ring quad -- the convolution
    of two equal-Keldysh G legs. Kept for the single-block test."""
    ne = G_a.shape[0]
    return _ref_bubble_F(phi_left, phi_right, _ref_fft(G_a, 2 * ne - 1),
                         _ref_fft(G_b, 2 * ne - 1), ne, bubble_prefactor_thz(dw_thz))


def _ref_quad_lesser(phi_left, phi_right, Gla, Gga_rev, Glb, Ggb_rev, dw_thz):
    """Full Σ^< for one ring quad: decay + the two absorption (negative-ω')
    correlations folded in via the EXACT bosonic continuation
    G^<_ij(q,-ω) = G^>_ji(-q,ω) -- what the production computes since the
    cross-block ji-transpose fold. ``Gga_rev``/``Ggb_rev`` must therefore be
    the already exchange-transformed partners (block-swapped, ji-transposed,
    q-negated for coupled-q); the helper only applies the ω-reversal.
    Σ^< = ring(g^<_a,g^<_b)+ring(g^<_a,rev g^>_b)+ring(rev g^>_a,g^<_b)."""
    ne = Gla.shape[0]; n_fft = 2 * ne - 1; pre = bubble_prefactor_thz(dw_thz)
    Fla, Fga = _ref_fft(Gla, n_fft), _ref_fft(Gga_rev, n_fft)
    Flb, Fgb = _ref_fft(Glb, n_fft), _ref_fft(Ggb_rev, n_fft)
    return (_ref_bubble_F(phi_left, phi_right, Fla, Flb, ne, pre)
            + _ref_bubble_F(phi_left, phi_right, Fla, _ref_rev(Fgb), ne, pre)
            + _ref_bubble_F(phi_left, phi_right, _ref_rev(Fga), Flb, ne, pre))


def _ref_compute_multiblock(
    phi_blocks: dict[tuple[int, int, int], np.ndarray],
    gl_band: dict[tuple[int, int], np.ndarray],
    gg_band: dict[tuple[int, int], np.ndarray],
    block_sizes: np.ndarray,
    dw_thz: float,
    g_band: int = 1,
) -> tuple[
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
]:
    """Independent multi-block dense reference for Sigma^{<,>,R}.

    Full off-diagonal ring (NOT diagonal-G): each quad uses
    ``G_a = G_{K1,K1'}`` and ``G_b = G_{K2,K2'}`` from the band dict.
    Assembled from first principles so any change to the production
    iteration is caught.
    """
    n_blocks = len(block_sizes)
    ne = next(iter(gl_band.values())).shape[0]
    freqs = np.linspace(0.0, dw_thz * (ne - 1), ne)

    # The grid starts at omega=0, so the production zeros the omega=0 (DC)
    # sample of G before the bubble (see SigmaPhononPhonon._compute_fft_first);
    # mirror that here so the reference matches the corrected behaviour.
    gl_band = {k: v.copy() for k, v in gl_band.items()}
    gg_band = {k: v.copy() for k, v in gg_band.items()}
    for v in gl_band.values():
        v[0] = 0.0
    for v in gg_band.values():
        v[0] = 0.0

    pair_index: dict[
        tuple[int, int],
        list[tuple[int, int, int, int, np.ndarray, np.ndarray]],
    ] = {}
    for (I, K1, K2), phi_left in phi_blocks.items():
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            for K1p in range(max(0, K1 - g_band), min(n_blocks, K1 + g_band + 1)):
                for K2p in range(max(0, K2 - g_band), min(n_blocks, K2 + g_band + 1)):
                    phi_right = phi_blocks.get((J, K2p, K1p))
                    if phi_right is None:
                        continue
                    pair_index.setdefault((I, J), []).append(
                        (K1, K2, K1p, K2p, phi_left, phi_right)
                    )

    sl_full: dict[tuple[int, int], np.ndarray] = {}
    sg_full: dict[tuple[int, int], np.ndarray] = {}
    for (I, J), pairs in pair_index.items():
        for K1, K2, K1p, K2p, phi_left, phi_right in pairs:
            la, lb = (K1, K1p), (K2, K2p)
            if la not in gl_band or lb not in gl_band:
                continue
            # Exact bosonic fold: the absorption leg of link (K, K') is the
            # ji-TRANSPOSE of the swapped block, G^<_{KK'}(-ω) = G^>_{K'K}(ω)^T.
            laT, lbT = (K1p, K1), (K2p, K2)
            sl = _ref_quad_lesser(
                phi_left, phi_right,
                gl_band[la], gg_band[laT].swapaxes(-1, -2),
                gl_band[lb], gg_band[lbT].swapaxes(-1, -2), dw_thz)
            # Σ^> is the same folded form with G^< and G^> swapped.
            sg = _ref_quad_lesser(
                phi_left, phi_right,
                gg_band[la], gl_band[laT].swapaxes(-1, -2),
                gg_band[lb], gl_band[lbT].swapaxes(-1, -2), dw_thz)
            sl_full[(I, J)] = sl_full.get((I, J), 0) + sl
            sg_full[(I, J)] = sg_full.get((I, J), 0) + sg

    # Production always zeroes the omega=0 bin of Sigma^≷ (a nonzero
    # Sigma^≷(0) hits the singular acoustic G^R(0) -> DC spike).
    for v in sl_full.values():
        v[0] = 0.0
    for v in sg_full.values():
        v[0] = 0.0

    # Retarded: bosonic Hilbert path (matches production's retarded_method="fft").
    # Built from the RAW (textbook-signed) values -- production keeps the
    # damping-correct sign here.
    from quatrex.core.fft_utils import hilbert_transform
    sr_full: dict[tuple[int, int], np.ndarray] = {}
    for key in set(sl_full) | set(sg_full):
        sl = sl_full.get(key, 0)
        sg = sg_full.get(key, 0)
        delta = sg - sl
        # The production SSE adds only the dispersive (Hilbert) part of
        # Σ^R; the SCBA loop adds the anti-Hermitian ½(Σ^>−Σ^<) itself.
        # The hard-mask path post-masks the Hilbert at the omega=0 bin.
        sr = 0.5j * hilbert_transform(delta, freqs)
        sr[0] = 0.0
        sr_full[key] = sr

    # SIGN CONVENTION (2026-06-12 production fix): the bubble is quadratic
    # in G, so fed with the solver's occupation-positive G^≷ it returns
    # textbook-signed sigma^≷ -- production negates sigma^≷ at the output
    # (Sigma^R keeps the raw sign). Mirror it.
    sl_full = {k: -v for k, v in sl_full.items()}
    sg_full = {k: -v for k, v in sg_full.items()}

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

    # Random G_lesser, G_greater on the full block-tridiagonal BAND
    # (|K-K'| <= 1) — the off-diagonal blocks the upstream RGF actually
    # produces and that the full ring contracts (NOT diagonal-only).
    gl_band: dict[tuple[int, int], np.ndarray] = {}
    gg_band: dict[tuple[int, int], np.ndarray] = {}
    for K in range(n_blocks):
        for Kp in range(max(0, K - 1), min(n_blocks, K + 2)):
            gl_band[(K, Kp)] = (rng.standard_normal((ne, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nbs, nbs)))
            gg_band[(K, Kp)] = (rng.standard_normal((ne, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nbs, nbs)))

    # Build the DSDBSparse buffers. Sparsity pattern carries the full
    # BT-band block positions; the band off-diagonal G entries are now
    # read by the contraction.
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

    # Write the band G blocks (diagonal + first off-diagonal) into
    # g_lesser/greater.
    gl_view = g_lesser.stack[...]
    gg_view = g_greater.stack[...]
    for (K, Kp) in gl_band:
        gl_view.blocks[K, Kp] = gl_band[(K, Kp)]
        gg_view.blocks[K, Kp] = gg_band[(K, Kp)]

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

    # Reference: independently-written multi-block dense bubble (full
    # off-diagonal ring).
    sl_ref, sg_ref, sr_ref = _ref_compute_multiblock(
        phi_blocks, gl_band, gg_band, block_sizes, dw_thz,
    )

    # Compare each (I, J) BTD block.
    sl_view = sigma_lesser.stack[...]
    sg_view = sigma_greater.stack[...]
    sr_view = sigma_retarded.stack[...]
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            key = (I, J)
            np.testing.assert_allclose(
                sl_view.blocks[I, J], sl_ref.get(key, 0), atol=1e-40, rtol=1e-9,
                err_msg=f"Sigma^< mismatch at block {key}",
            )
            np.testing.assert_allclose(
                sg_view.blocks[I, J], sg_ref.get(key, 0), atol=1e-40, rtol=1e-10,
                err_msg=f"Sigma^> mismatch at block {key}",
            )
            np.testing.assert_allclose(
                sr_view.blocks[I, J], sr_ref.get(key, 0), atol=1e-40, rtol=1e-10,
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


def _ref_quad_scaled(phi_left, phi_right, G_a, G_b, dw_thz, scale):
    """``_ref_quad`` with an extra scalar (the coupled-q 1/N_q factor)."""
    return scale * _ref_quad(phi_left, phi_right, G_a, G_b, dw_thz)


def test_compute_coupled_q_matches_reference() -> None:
    """Transverse-q (k>1) coupled-momentum SSE parity vs an independent
    einsum oracle.

    Pins the production ``SigmaPhononPhonon.compute()`` k>1 path (the
    Phi-tilde(q,q') momentum convolution) against a hand-written
    reference: for each external q_ext, Sigma(q_ext) = (1/N_q) sum_{q'}
    ring[ Phi(q',q2), Phi(q2,q'), G(q')_{K1K1'}, G(q2)_{K2K2'} ] with
    q2 = q_ext - q'. Random per-(iq1,iq2) vertices + a q_diff_map
    exercise the q-indexing, the ring contraction and the 1/N_q prefactor
    exactly; the (0,0) vertex defines the block-pair index.
    """
    from qttools.datastructures import DSDBCOO
    from scipy.sparse import csr_matrix
    from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

    rng = np.random.default_rng(7)
    # nq=3 so the q -> -q negation of the exact bosonic fold is actually
    # exercised (at nq=2 the negation is the identity).
    n_blocks, nbs, ne, nq = 3, 3, 13, 3
    block_sizes = np.array([nbs] * n_blocks)
    N = int(block_sizes.sum())
    dw_thz = 16.0 / (ne - 1)
    q_diff_map = np.array([[(a - b) % nq for b in range(nq)] for a in range(nq)])

    keys = [
        (I, K1, K2)
        for I in range(n_blocks)
        for K1 in range(max(0, I - 1), min(n_blocks, I + 2))
        for K2 in range(max(0, I - 1), min(n_blocks, I + 2))
        if abs(K1 - K2) <= 1
    ]

    def _phi():
        return {
            k: rng.standard_normal((nbs, nbs, nbs))
            + 1j * rng.standard_normal((nbs, nbs, nbs))
            for k in keys
        }

    qvertices = {(a, b): _phi() for a in range(nq) for b in range(nq)}

    gl_band, gg_band = {}, {}
    for K in range(n_blocks):
        for Kp in range(max(0, K - 1), min(n_blocks, K + 2)):
            gl_band[(K, Kp)] = rng.standard_normal(
                (ne, nq, nbs, nbs)
            ) + 1j * rng.standard_normal((ne, nq, nbs, nbs))
            gg_band[(K, Kp)] = rng.standard_normal(
                (ne, nq, nbs, nbs)
            ) + 1j * rng.standard_normal((ne, nq, nbs, nbs))

    rows, cols = [], []
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            for i in range(block_sizes[I]):
                for j in range(block_sizes[J]):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = csr_matrix(
        (np.ones(len(rows), np.complex128), (np.array(rows), np.array(cols))),
        shape=(N, N),
    )
    mk = lambda: DSDBCOO.from_sparray(
        pattern, block_sizes, global_stack_shape=(ne, nq)
    )
    g_l, g_g, s_l, s_g, s_r = mk(), mk(), mk(), mk(), mk()
    for m in (g_l, g_g, s_l, s_g, s_r):
        m.data[:] = 0.0
    glv, ggv = g_l.stack[...], g_g.stack[...]
    for (K, Kp) in gl_band:
        glv.blocks[K, Kp] = gl_band[(K, Kp)]
        ggv.blocks[K, Kp] = gg_band[(K, Kp)]

    cfg = _make_cfg("half")
    ssp = SigmaPhononPhonon(
        cfg,
        phonon_frequencies=np.linspace(0.0, 16.0, ne),
        block_sizes=block_sizes,
        qfold=(qvertices, q_diff_map, nq),
    )
    ssp.compute(g_l, g_g, out=(s_l, s_g, s_r))

    pair_index = {}
    for (I, K1, K2) in qvertices[(0, 0)]:
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            for K1p in range(max(0, K1 - 1), min(n_blocks, K1 + 2)):
                for K2p in range(max(0, K2 - 1), min(n_blocks, K2 + 2)):
                    if (J, K2p, K1p) in qvertices[(0, 0)]:
                        pair_index.setdefault((I, J), []).append(
                            (K1, K2, K1p, K2p)
                        )

    # Grid starts at omega=0 -> production zeros the omega=0 (DC) sample of
    # G before the bubble; mirror it in the reference (g_l/g_g were built
    # from the un-zeroed bands, so the production's DC-zeroing is exercised).
    gl_band = {k: v.copy() for k, v in gl_band.items()}
    gg_band = {k: v.copy() for k, v in gg_band.items()}
    for v in gl_band.values():
        v[0] = 0.0
    for v in gg_band.values():
        v[0] = 0.0

    ref_l = {ij: np.zeros((ne, nq, nbs, nbs), complex) for ij in pair_index}
    ref_g = {ij: np.zeros((ne, nq, nbs, nbs), complex) for ij in pair_index}
    for (I, J), quads in pair_index.items():
        for iq_ext in range(nq):
            for iqp in range(nq):
                iq2 = int(q_diff_map[iq_ext, iqp])
                phiL, phiR = qvertices[(iqp, iq2)], qvertices[(iq2, iqp)]
                for (K1, K2, K1p, K2p) in quads:
                    pl, pr = phiL.get((I, K1, K2)), phiR.get((J, K2p, K1p))
                    if pl is None or pr is None:
                        continue
                    # The LEFT vertex is CONJUGATED (the audited coupled-q
                    # pairing Phi(q',q_ext-q')^* · GG · Phi(q_ext-q',q');
                    # supercell-verified, see the production comment).
                    pl = np.conj(pl)
                    la, lb = (K1, K1p), (K2, K2p)
                    # Exact bosonic fold: the absorption leg is the ji-transposed
                    # swapped block at NEGATED q, G^<_{KK'}(q,-ω) = G^>_{K'K}(-q,ω)^T;
                    # the 1/N_q is the coupled-q mesh average.
                    laT, lbT = (K1p, K1), (K2p, K2)
                    iqp_n, iq2_n = (-iqp) % nq, (-iq2) % nq
                    ref_l[(I, J)][:, iq_ext] += (1.0 / nq) * _ref_quad_lesser(
                        pl, pr,
                        gl_band[la][:, iqp],
                        gg_band[laT][:, iqp_n].swapaxes(-1, -2),
                        gl_band[lb][:, iq2],
                        gg_band[lbT][:, iq2_n].swapaxes(-1, -2), dw_thz,
                    )
                    ref_g[(I, J)][:, iq_ext] += (1.0 / nq) * _ref_quad_lesser(
                        pl, pr,
                        gg_band[la][:, iqp],
                        gl_band[laT][:, iqp_n].swapaxes(-1, -2),
                        gg_band[lb][:, iq2],
                        gl_band[lbT][:, iq2_n].swapaxes(-1, -2), dw_thz,
                    )

    # Production always zeroes the omega=0 bin of Sigma^≷ and negates
    # sigma^≷ at the output (occupation-positive convention, 2026-06-12 fix).
    ref_l = {k: -v for k, v in ref_l.items()}
    ref_g = {k: -v for k, v in ref_g.items()}
    for v in ref_l.values():
        v[0] = 0.0
    for v in ref_g.values():
        v[0] = 0.0

    slv, sgv = s_l.stack[...], s_g.stack[...]
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            np.testing.assert_allclose(
                slv.blocks[I, J], ref_l.get((I, J), 0), atol=1e-40, rtol=1e-9,
                err_msg=f"coupled-q Sigma^< mismatch at {(I, J)}",
            )
            np.testing.assert_allclose(
                sgv.blocks[I, J], ref_g.get((I, J), 0), atol=1e-40, rtol=1e-9,
                err_msg=f"coupled-q Sigma^> mismatch at {(I, J)}",
            )


@pytest.mark.parametrize("kernel", ["gram", "reconstruct"])
@pytest.mark.parametrize("ansatz", ["INDSCAL", "CP"])
def test_compute_coupled_q_factored_matches_dense(ansatz, kernel) -> None:
    """Factored coupled-q kernel == dense path fed the SAME vertex.

    Random per-leg factors with the physical TRS structure u(-q) = u(q)*
    (Gamma blocks real); the dense q-folded dict is built FROM the factors
    (``VertexFactors.reconstruct_block``), so the dense path (``qfold=``
    injection, itself pinned against the hand-written reference above) and
    the factored path (``vfactors=`` injection) compute the same bubble.
    INDSCAL exercises the shared-leg Gram cache, CP the role-keyed two-Gram
    branch; kernel="reconstruct" covers the rank-local dense-reconstruction
    mode. G is random complex (non-TRS) -- kills any accidental g = g^T
    assumption in the Gram pairing.
    """
    from qttools.datastructures import DSDBCOO
    from scipy.sparse import csr_matrix
    from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon
    from quatrex.phonon.vertex_factors import VertexFactors

    rng = np.random.default_rng(11)
    n_blocks, nbs, ne, nq, R = 3, 3, 13, 3, 5
    block_sizes = np.array([nbs] * n_blocks)
    N = int(block_sizes.sum())
    offsets = np.array([-1, 0, 1], dtype=np.int64)
    q_diff_map = np.array([[(a - b) % nq for b in range(nq)] for a in range(nq)])

    def _tr_pair_factors():
        """(n_off, nq, nbs, R) with u(-q) = u(q)^*."""
        u = np.empty((len(offsets), nq, nbs, R), dtype=np.complex128)
        u[:, 0] = rng.standard_normal((len(offsets), nbs, R))
        for q in range(1, nq // 2 + 1):
            blk = (rng.standard_normal((len(offsets), nbs, R))
                   + 1j * rng.standard_normal((len(offsets), nbs, R)))
            u[:, q] = blk
            u[:, (nq - q) % nq] = np.conj(blk)
        return u

    D = rng.standard_normal((nbs, R))
    lambdas = np.sort(np.abs(rng.standard_normal(R)))[::-1] + 0.1
    UB = _tr_pair_factors()
    UC = UB if ansatz == "INDSCAL" else _tr_pair_factors()
    vf = VertexFactors(
        D=D, lambdas=lambdas, offsets=offsets, UB=UB, UC=UC,
        q_diff_map=q_diff_map, nk_shape=(nq,), ansatz=ansatz, meta={},
    )

    # Dense q-folded dict FROM the factors.
    qvertices = {}
    for iq1 in range(nq):
        for iq2 in range(nq):
            blocks = {}
            for I in range(n_blocks):
                for d1 in offsets:
                    K1 = I + int(d1)
                    if not 0 <= K1 < n_blocks:
                        continue
                    for d2 in offsets:
                        K2 = I + int(d2)
                        if not 0 <= K2 < n_blocks:
                            continue
                        blocks[(I, K1, K2)] = vf.reconstruct_block(
                            iq1, iq2, int(d1), int(d2))
            qvertices[(iq1, iq2)] = blocks

    # Shared random (non-TRS) G bands.
    gl_band, gg_band = {}, {}
    for K in range(n_blocks):
        for Kp in range(max(0, K - 1), min(n_blocks, K + 2)):
            gl_band[(K, Kp)] = (rng.standard_normal((ne, nq, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nq, nbs, nbs)))
            gg_band[(K, Kp)] = (rng.standard_normal((ne, nq, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nq, nbs, nbs)))

    rows, cols = [], []
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            for i in range(block_sizes[I]):
                for j in range(block_sizes[J]):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = csr_matrix(
        (np.ones(len(rows), np.complex128), (np.array(rows), np.array(cols))),
        shape=(N, N),
    )

    def _run(**ssp_kwargs):
        mk = lambda: DSDBCOO.from_sparray(
            pattern, block_sizes, global_stack_shape=(ne, nq))
        g_l, g_g, s_l, s_g, s_r = mk(), mk(), mk(), mk(), mk()
        for m in (g_l, g_g, s_l, s_g, s_r):
            m.data[:] = 0.0
        glv, ggv = g_l.stack[...], g_g.stack[...]
        for (K, Kp) in gl_band:
            glv.blocks[K, Kp] = gl_band[(K, Kp)]
            ggv.blocks[K, Kp] = gg_band[(K, Kp)]
        cfg = _make_cfg("half")
        cfg.phonon.decomposed_kernel = kernel
        ssp = SigmaPhononPhonon(
            cfg,
            phonon_frequencies=np.linspace(0.0, 16.0, ne),
            block_sizes=block_sizes,
            **ssp_kwargs,
        )
        ssp.compute(g_l, g_g, out=(s_l, s_g, s_r))
        return s_l, s_g

    sl_d, sg_d = _run(qfold=(qvertices, q_diff_map, nq))
    sl_f, sg_f = _run(vfactors=vf)

    dv_l, dv_g = sl_d.stack[...], sg_d.stack[...]
    fv_l, fv_g = sl_f.stack[...], sg_f.stack[...]
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            np.testing.assert_allclose(
                fv_l.blocks[I, J], dv_l.blocks[I, J], atol=1e-45, rtol=1e-9,
                err_msg=f"factored Sigma^< mismatch at {(I, J)} [{ansatz}]",
            )
            np.testing.assert_allclose(
                fv_g.blocks[I, J], dv_g.blocks[I, J], atol=1e-45, rtol=1e-9,
                err_msg=f"factored Sigma^> mismatch at {(I, J)} [{ansatz}]",
            )


def test_q_convolution_matches_explicit_q_diff_map_sum() -> None:
    """The FFT circular convolution == the explicit q_diff_map double sum.

    The factored kernel evaluates sum_{q'} Pa[q'] o Pb[q_ext - q'] by FFT. That
    is only legitimate because ``build_q_diff_map`` is the circulant difference
    map on the Gamma-centered mesh, so the sum is a genuine circular
    convolution.
    """
    from quatrex.phonon.bubble_factored import _convolve_q

    nkx, nky = 4, 3
    nq = nkx * nky
    rng = np.random.default_rng(3)

    def cx(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    pa, pb = cx(nq, 5, 6, 6), cx(nq, 5, 6, 6)
    q_diff_map = np.array(
        [
            [
                ((a // nky - b // nky) % nkx) * nky + (a % nky - b % nky) % nky
                for b in range(nq)
            ]
            for a in range(nq)
        ]
    )

    expected = np.stack(
        [np.einsum("qwrs,qwrs->wrs", pa, pb[q_diff_map[q]]) for q in range(nq)]
    )

    np.testing.assert_allclose(
        _convolve_q(pa, pb, (nkx, nky), np), expected, rtol=1e-12
    )


@pytest.mark.parametrize("ansatz", ["INDSCAL", "CP"])
def test_compute_gamma_factored_matches_dense(ansatz) -> None:
    """Factored kernel == dense ring at Gamma (nq == 1).

    At the zone centre the momentum convolution is the identity, so what is
    exercised here is the Gram collapse alone -- the regime of every
    transversely-finite device (nanowires, CNTs), which the factored kernel
    could not reach before.
    """
    from qttools.datastructures import DSDBCOO
    from scipy.sparse import csr_matrix

    from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon
    from quatrex.phonon.vertex_factors import VertexFactors

    rng = np.random.default_rng(23)
    n_blocks, nbs, ne, R = 3, 3, 13, 5
    block_sizes = np.array([nbs] * n_blocks)
    N = int(block_sizes.sum())
    offsets = np.array([-1, 0, 1], dtype=np.int64)

    # Gamma-only: a single momentum, so the factors are real.
    D = rng.standard_normal((nbs, R))
    lambdas = np.sort(np.abs(rng.standard_normal(R)))[::-1] + 0.1
    UB = rng.standard_normal((len(offsets), 1, nbs, R)).astype(np.complex128)
    UC = UB if ansatz == "INDSCAL" else rng.standard_normal(
        (len(offsets), 1, nbs, R)
    ).astype(np.complex128)
    vf = VertexFactors(
        D=D, lambdas=lambdas, offsets=offsets, UB=UB, UC=UC,
        q_diff_map=np.zeros((1, 1), dtype=int), nk_shape=(1,), ansatz=ansatz,
        meta={},
    )

    # The dense Gamma vertex, built FROM the same factors.
    phi_blocks = {}
    for I in range(n_blocks):
        for d1 in offsets:
            if not 0 <= I + int(d1) < n_blocks:
                continue
            for d2 in offsets:
                if not 0 <= I + int(d2) < n_blocks:
                    continue
                phi_blocks[(I, I + int(d1), I + int(d2))] = vf.reconstruct_block(
                    0, 0, int(d1), int(d2)
                )

    gl_band, gg_band = {}, {}
    for K in range(n_blocks):
        for Kp in range(max(0, K - 1), min(n_blocks, K + 2)):
            gl_band[(K, Kp)] = (rng.standard_normal((ne, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nbs, nbs)))
            gg_band[(K, Kp)] = (rng.standard_normal((ne, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nbs, nbs)))

    rows, cols = [], []
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            for i in range(block_sizes[I]):
                for j in range(block_sizes[J]):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = csr_matrix(
        (np.ones(len(rows), np.complex128), (np.array(rows), np.array(cols))),
        shape=(N, N),
    )

    def _run(**ssp_kwargs):
        mk = lambda: DSDBCOO.from_sparray(
            pattern, block_sizes, global_stack_shape=(ne,))
        g_l, g_g, s_l, s_g, s_r = mk(), mk(), mk(), mk(), mk()
        for m in (g_l, g_g, s_l, s_g, s_r):
            m.data[:] = 0.0
        glv, ggv = g_l.stack[...], g_g.stack[...]
        for (K, Kp) in gl_band:
            glv.blocks[K, Kp] = gl_band[(K, Kp)]
            ggv.blocks[K, Kp] = gg_band[(K, Kp)]
        cfg = _make_cfg("half")
        cfg.phonon.decomposed_kernel = "gram"
        ssp = SigmaPhononPhonon(
            cfg,
            phonon_frequencies=np.linspace(0.0, 16.0, ne),
            block_sizes=block_sizes,
            **ssp_kwargs,
        )
        ssp.compute(g_l, g_g, out=(s_l, s_g, s_r))
        return s_l, s_g

    sl_d, sg_d = _run(phi_blocks=phi_blocks)
    sl_f, sg_f = _run(vfactors=vf)

    dv_l, dv_g = sl_d.stack[...], sg_d.stack[...]
    fv_l, fv_g = sl_f.stack[...], sg_f.stack[...]
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            np.testing.assert_allclose(
                fv_l.blocks[I, J], dv_l.blocks[I, J], atol=1e-45, rtol=1e-9,
                err_msg=f"Gamma factored Sigma^< mismatch at {(I, J)}",
            )
            np.testing.assert_allclose(
                fv_g.blocks[I, J], dv_g.blocks[I, J], atol=1e-45, rtol=1e-9,
                err_msg=f"Gamma factored Sigma^> mismatch at {(I, J)}",
            )
