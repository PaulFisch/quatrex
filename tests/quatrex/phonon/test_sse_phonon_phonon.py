# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Unit tests for ``quatrex.phonon.sse_phonon_phonon.SigmaPhononPhonon``.

The reference implementation is the standalone
``phonon/phonon_inputs/anharmonic.py:_compute_phph_self_energy_finite``.
We pin the new block-decomposed bubble against the dense reference and
check the bosonic Keldysh symmetries that the SCBA loop relies on.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from qttools import xp
from qttools.comm import comm as _qtt_comm
from qttools.utils.gpu_utils import get_host

from quatrex.phonon.fc3_loader import fc3_to_phi_blocks
from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon
from quatrex.phonon.units import bubble_prefactor_thz


def _dev_pattern(p):
    """Host scipy pattern -> backend-native sparse (no-op under numpy)."""
    from qttools import sparse
    return sparse.csr_matrix(p) if xp.__name__ == "cupy" else p


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


def _make_cfg(retarded_method: str = "fft", g_band: int = 1):
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
    _Phonon.sse_g_band = g_band

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
    assert np.allclose(sig_l_ref, np.asarray(get_host(sig_l_new)),
                       atol=1e-40, rtol=1e-10)


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
    taper_w: list[float] | None = None,
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
            if taper_w is not None:
                # Bartlett-tapered band: the two inner G-link weights and
                # the Sigma output-block weight collapse to one scalar per
                # quad (mirrors SigmaPhononPhonon._quad_weight).
                wq = (taper_w[abs(K1 - K1p)] * taper_w[abs(K2 - K2p)]
                      * taper_w[abs(I - J)])
                sl = sl * wq
                sg = sg * wq
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
        sr = 0.5j * np.asarray(
            get_host(hilbert_transform(xp.asarray(delta), xp.asarray(freqs))))
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
    pattern = _dev_pattern(csr_matrix(
        (np.ones(len(pattern_rows), dtype=np.complex128),
         (np.array(pattern_rows), np.array(pattern_cols))),
        shape=(N, N),
    ))

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
        gl_view.blocks[K, Kp] = xp.asarray(gl_band[(K, Kp)])
        gg_view.blocks[K, Kp] = xp.asarray(gg_band[(K, Kp)])

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
                np.asarray(get_host(sl_view.blocks[I, J])), sl_ref.get(key, 0),
                atol=1e-40, rtol=1e-9,
                err_msg=f"Sigma^< mismatch at block {key}",
            )
            np.testing.assert_allclose(
                np.asarray(get_host(sg_view.blocks[I, J])), sg_ref.get(key, 0),
                atol=1e-40, rtol=1e-10,
                err_msg=f"Sigma^> mismatch at block {key}",
            )
            np.testing.assert_allclose(
                np.asarray(get_host(sr_view.blocks[I, J])), sr_ref.get(key, 0),
                atol=1e-40, rtol=1e-10,
                err_msg=f"Sigma^R mismatch at block {key}",
            )


def test_compute_restores_distribution_state() -> None:
    """compute() leaves the DSDBSparse distribution state untouched.

    The legacy implementation toggled buffers to ``stack`` without
    restoring; this broke the SCBA loop, which expects ``nnz`` to be
    preserved across interactions. Pin the contract.
    """
    g_band = 1  # these fixtures are band-1
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
        for J in range(max(0, I - g_band), min(n_blocks, I + g_band + 1)):
            for i in range(block_sizes[I]):
                for j in range(block_sizes[J]):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = _dev_pattern(csr_matrix(
        (np.ones(len(rows), dtype=np.complex128),
         (np.array(rows), np.array(cols))),
        shape=(N, N),
    ))

    bufs = [DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(ne,))
            for _ in range(5)]
    g_lesser, g_greater, sigma_lesser, sigma_greater, sigma_retarded = bufs
    for m in bufs:
        m.data[:] = 0.0

    # Seed random diagonal G data.
    gl_v = g_lesser.stack[...]
    gg_v = g_greater.stack[...]
    for K in range(n_blocks):
        gl_v.blocks[K, K] = xp.asarray(
            rng.standard_normal((ne, nbs, nbs))
            + 1j * rng.standard_normal((ne, nbs, nbs)))
        gg_v.blocks[K, K] = xp.asarray(
            rng.standard_normal((ne, nbs, nbs))
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


def _taper_fixture(n_blocks: int, nbs: int, ne: int, seed: int = 7):
    """NN phi blocks, Hermitian-PSD-per-omega band G^{<,>}, and the
    DSDBSparse buffers for a taper/causality test device.

    The G inputs are the BT band of FULL Hermitian PSD matrices P(w)
    (what the RGF hands the bubble): the causality theorem is about
    M (.) P with the Bartlett M supported on that band, so PSD-ness of
    the underlying full matrix is the correct premise.
    """
    from qttools.datastructures import DSDBCOO
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(seed)
    block_sizes = np.array([nbs] * n_blocks)
    N = int(block_sizes.sum())

    # REAL vertex blocks, symmetric under exchange of the two contracted
    # legs (incl. their block indices): Phi(I,K2,K1)_{i,b,a} =
    # Phi(I,K1,K2)_{i,a,b}. This is the (S_3-subgroup) property of the
    # physical fc3 that makes the bubble a congruence -- the PSD/causality
    # statement is only a theorem for such vertices.
    phi_blocks: dict[tuple[int, int, int], np.ndarray] = {}
    for I in range(n_blocks):
        for K1 in range(max(0, I - 1), min(n_blocks, I + 2)):
            for K2 in range(K1, min(n_blocks, I + 2)):
                if abs(K1 - K2) > 1:
                    continue
                A = rng.standard_normal((nbs, nbs, nbs))
                if K1 == K2:
                    A = 0.5 * (A + A.transpose(0, 2, 1))
                    phi_blocks[(I, K1, K2)] = A
                else:
                    phi_blocks[(I, K1, K2)] = A
                    phi_blocks[(I, K2, K1)] = A.transpose(0, 2, 1)

    def _psd_stack() -> np.ndarray:
        # RANK-1 (+ tiny ridge) spectral matrices: maximal inter-cell
        # coherence, the flat-band regime. This is where the boxcar band
        # truncation actually goes indefinite (banded truncation of a
        # rank-1 projector, cf. the indefinite tridiagonal-ones mask);
        # full-rank random PSD stacks average the defect away and the
        # one-shot masked bubble stays accidentally PSD.
        out = np.empty((ne, N, N), dtype=complex)
        for k in range(ne):
            c = rng.standard_normal((N, 1)) + 1j * rng.standard_normal((N, 1))
            out[k] = c @ c.conj().T + 1e-6 * np.eye(N)
        return out

    P_l, P_g = _psd_stack(), _psd_stack()

    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    gl_band: dict[tuple[int, int], np.ndarray] = {}
    gg_band: dict[tuple[int, int], np.ndarray] = {}
    for K in range(n_blocks):
        for Kp in range(max(0, K - 1), min(n_blocks, K + 2)):
            sK = slice(offs[K], offs[K + 1])
            sKp = slice(offs[Kp], offs[Kp + 1])
            gl_band[(K, Kp)] = P_l[:, sK, sKp].copy()
            gg_band[(K, Kp)] = P_g[:, sK, sKp].copy()

    rows, cols = [], []
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            for i in range(nbs):
                for j in range(nbs):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = _dev_pattern(csr_matrix(
        (np.ones(len(rows), dtype=np.complex128),
         (np.array(rows), np.array(cols))), shape=(N, N)))

    def make_buffers():
        bufs = [DSDBCOO.from_sparray(pattern, block_sizes,
                                     global_stack_shape=(ne,))
                for _ in range(5)]
        for m in bufs:
            m.data[:] = 0.0
        gl, gg = bufs[0], bufs[1]
        gl_v, gg_v = gl.stack[...], gg.stack[...]
        for (K, Kp) in gl_band:
            gl_v.blocks[K, Kp] = xp.asarray(gl_band[(K, Kp)])
            gg_v.blocks[K, Kp] = xp.asarray(gg_band[(K, Kp)])
        return bufs

    return phi_blocks, gl_band, gg_band, block_sizes, offs, make_buffers


def _run_sigma(phi_blocks, block_sizes, ne, make_buffers, taper: str | None):
    """Run the production bubble; return the (I,J)->block Sigma^{<,>} dicts.

    ``taper=None`` leaves the config attribute ABSENT (pre-option configs);
    a string sets ``sse_g_band_taper`` explicitly.
    """
    cfg = _make_cfg("fft")
    if taper is not None:
        cfg.phonon.sse_g_band_taper = taper
    freqs_thz = np.linspace(0.0, 16.0, ne)
    gl, gg, sl, sg, sr = make_buffers()
    ssp = SigmaPhononPhonon(
        cfg, phonon_frequencies=freqs_thz,
        block_sizes=block_sizes, phi_blocks=phi_blocks)
    ssp.compute(gl, gg, out=(sl, sg, sr))
    n_blocks = len(block_sizes)
    sl_v, sg_v = sl.stack[...], sg.stack[...]
    out_l, out_g = {}, {}
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            out_l[(I, J)] = np.asarray(get_host(sl_v.blocks[I, J])).copy()
            out_g[(I, J)] = np.asarray(get_host(sg_v.blocks[I, J])).copy()
    return out_l, out_g


def _assemble(blocks: dict, offs: np.ndarray, N: int, ne: int) -> np.ndarray:
    """Banded (I,J) block dict -> full (ne, N, N) matrices (zeros outside)."""
    out = np.zeros((ne, N, N), dtype=complex)
    for (I, J), v in blocks.items():
        out[:, offs[I]:offs[I + 1], offs[J]:offs[J + 1]] = v
    return out


def test_taper_none_is_bit_identical() -> None:
    """sse_g_band_taper='none' (and absent) reproduce the legacy output
    bit-for-bit -- the option must not perturb existing runs."""
    phi_blocks, _, _, block_sizes, _, make_buffers = _taper_fixture(4, 2, 15)
    # "none" set explicitly vs attribute entirely absent (pre-option config).
    l_none, g_none = _run_sigma(
        phi_blocks, block_sizes, 15, make_buffers, taper="none")
    l_absent, g_absent = _run_sigma(
        phi_blocks, block_sizes, 15, make_buffers, taper=None)
    for key in l_none:
        assert np.array_equal(l_none[key], l_absent[key])
        assert np.array_equal(g_none[key], g_absent[key])


def test_taper_matches_tapered_reference() -> None:
    """Production bartlett == independently-tapered dense reference."""
    phi_blocks, gl_band, gg_band, block_sizes, _, make_buffers = (
        _taper_fixture(4, 2, 15))
    ne = 15
    dw = 16.0 / (ne - 1)
    out_l, out_g = _run_sigma(
        phi_blocks, block_sizes, ne, make_buffers, taper="bartlett")
    ref_l, ref_g, _ = _ref_compute_multiblock(
        phi_blocks, gl_band, gg_band, block_sizes, dw,
        g_band=1, taper_w=[1.0, 0.5])
    for key in out_l:
        np.testing.assert_allclose(
            out_l[key], ref_l.get(key, 0), atol=1e-40, rtol=1e-9,
            err_msg=f"tapered Sigma^< mismatch at block {key}")
        np.testing.assert_allclose(
            out_g[key], ref_g.get(key, 0), atol=1e-40, rtol=1e-9,
            err_msg=f"tapered Sigma^> mismatch at block {key}")


def test_taper_restores_causality_psd() -> None:
    """THE point of the taper: with Hermitian-PSD G^{<,>} inputs, the
    boxcar band-1 Sigma^{<,>} is INDEFINITE on a >=4-block device
    (non-causal gain -- the documented sse_g_band=1 disease), while the
    Bartlett-tapered band-1 Sigma^{<,>} is PSD at every omega (Schur
    product theorem with the PSD taper matrix)."""
    ne = 15
    n_blocks, nbs = 5, 2
    phi_blocks, _, _, block_sizes, offs, make_buffers = (
        _taper_fixture(n_blocks, nbs, ne))
    N = int(block_sizes.sum())

    # Orientation calibration on a single block (no masking -> exact,
    # causal by construction): find the phase z in {1,-1,1j,-1j} that
    # makes z * Sigma(omega) Hermitian PSD. This decouples the test from
    # the production sign convention.
    phi1, _, _, bs1, offs1, mk1 = _taper_fixture(1, 4, ne)
    sl1, _ = _run_sigma(phi1, bs1, ne, mk1, taper="none")
    S1 = _assemble(sl1, offs1, int(bs1.sum()), ne)
    z_found = None
    for z in (1.0, -1.0, 1j, -1j):
        M = z * S1
        herm = np.linalg.norm(M - M.conj().swapaxes(-1, -2))
        if herm > 1e-8 * np.linalg.norm(M):
            continue
        if np.linalg.eigvalsh(M).min() >= -1e-10 * np.abs(M).max():
            z_found = z
            break
    assert z_found is not None, "no PSD orientation on the 1-block device"

    def min_eig(blocks):
        S = z_found * _assemble(blocks, offs, N, ne)
        # Hermitian part (the anti-Hermitian remainder is zero for exact
        # channels; the masked kernel's defect shows up here too).
        S = 0.5 * (S + S.conj().swapaxes(-1, -2))
        return np.linalg.eigvalsh(S).min(), np.abs(S).max()

    box_l, box_g = _run_sigma(
        phi_blocks, block_sizes, ne, make_buffers, taper="none")
    tap_l, tap_g = _run_sigma(
        phi_blocks, block_sizes, ne, make_buffers, taper="bartlett")

    lam_box, scale_box = min_eig(box_l)
    assert lam_box < -1e-4 * scale_box, (
        "expected the boxcar band-1 Sigma^< to be indefinite (non-causal "
        f"gain); min eig {lam_box:.3e} vs scale {scale_box:.3e}")

    for tag, blocks in (("<", tap_l), (">", tap_g)):
        lam, scale = min_eig(blocks)
        assert lam >= -1e-9 * scale, (
            f"tapered Sigma^{tag} lost PSD-ness: min eig {lam:.3e} vs "
            f"scale {scale:.3e}")


def _psd_orientation(blocks, offs, N, ne):
    """Phase z in {1,-1,i,-i} that makes Sigma hermitian PSD.

    Decouples the assertions from the production sign convention exactly
    as test_taper_restores_causality_psd does (the fixture feeds PSD
    matrices as G directly, not as -iG, so the convention differs by i).
    """
    S = _assemble(blocks, offs, N, ne)
    for z in (1.0, -1.0, 1j, -1j):
        M = z * S
        if np.linalg.norm(M - M.conj().swapaxes(-1, -2)) > 1e-8 * np.linalg.norm(M):
            continue
        if np.linalg.eigvalsh(M).min() >= -1e-10 * np.abs(M).max():
            return z
    return None


def _worst_neg(blocks, offs, N, ne, z):
    S = z * _assemble(blocks, offs, N, ne)
    S = 0.5 * (S + S.conj().swapaxes(-1, -2))
    return np.linalg.eigvalsh(S).min(), np.abs(S).max()


def test_bubble_psd_gamma_complete_band() -> None:
    """Theorem 1 at nq=1: PSD legs + a real, leg-exchange-symmetric vertex
    give a PSD Sigma^{<,>} when NO band mask is active.

    Two blocks is the largest device whose hard-wired output band
    |I-J| <= 1 (sse_phonon_phonon.py:410) is complete, so this isolates
    the congruence M (A x B) M^dagger from the masking of
    test_taper_restores_causality_psd. See phonon/docs/bubble_positivity.md.
    """
    ne, n_blocks, nbs = 15, 2, 3
    phi_blocks, _, _, block_sizes, offs, make_buffers = (
        _taper_fixture(n_blocks, nbs, ne))
    N = int(block_sizes.sum())
    out_l, out_g = _run_sigma(phi_blocks, block_sizes, ne, make_buffers,
                              taper="none")
    z = _psd_orientation(out_l, offs, N, ne)
    assert z is not None, "no PSD orientation found for the unmasked bubble"
    for tag, blocks in (("<", out_l), (">", out_g)):
        lam, scale = _worst_neg(blocks, offs, N, ne, z)
        assert lam >= -1e-9 * scale, (
            f"unmasked Sigma^{tag} is not PSD: min eig {lam:.3e} "
            f"vs scale {scale:.3e}")


def test_bubble_psd_broken_by_asymmetric_vertex() -> None:
    """The leg-exchange symmetry is load-bearing, not decorative.

    Same complete-band device as above, but with Phi(I,K2,K1) no longer
    the (0,2,1)-transpose of Phi(I,K1,K2): the ring stops being a
    congruence and Sigma goes indefinite.
    """
    ne, n_blocks, nbs = 15, 2, 3
    phi_blocks, _, _, block_sizes, offs, make_buffers = (
        _taper_fixture(n_blocks, nbs, ne))
    N = int(block_sizes.sum())
    good_l, _ = _run_sigma(phi_blocks, block_sizes, ne, make_buffers,
                           taper="none")
    z = _psd_orientation(good_l, offs, N, ne)
    assert z is not None

    rng = np.random.default_rng(11)
    bad = dict(phi_blocks)
    for (I, K1, K2) in list(bad):
        if K1 != K2:                       # break only the exchange pair
            bad[(I, K1, K2)] = rng.standard_normal((nbs, nbs, nbs))
    bad_l, _ = _run_sigma(bad, block_sizes, ne, make_buffers, taper="none")
    lam, scale = _worst_neg(bad_l, offs, N, ne, z)
    assert lam < -1e-6 * scale, (
        "an asymmetric vertex should destroy PSD-ness of the bubble; "
        f"got min eig {lam:.3e} vs scale {scale:.3e}")


def test_bubble_psd_coupled_q_complete_band() -> None:
    """Theorem 1 at nq>1 -- the case with no coverage before.

    At coupled q the left vertex factor is CONJUGATED
    (sse_phonon_phonon.py:1846-1848), so the congruence needs only the
    q-carrying leg exchange

        Phi(q2,q1)[(J,Kb,Ka)][a,d,b] = Phi(q1,q2)[(J,Ka,Kb)][a,b,d],

    and NOT a real vertex. The fixture enforces exactly that and nothing
    else, so a PSD result is evidence for the conjugation being right.
    """
    from qttools.datastructures import DSDBCOO
    from scipy.sparse import csr_matrix

    from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

    rng = np.random.default_rng(3)
    n_blocks, nbs, ne, nq = 2, 3, 13, 3
    block_sizes = np.array([nbs] * n_blocks)
    N = int(block_sizes.sum())
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    q_diff_map = np.array([[(a - b) % nq for b in range(nq)]
                           for a in range(nq)])
    keys = [(I, K1, K2) for I in range(n_blocks)
            for K1 in range(n_blocks) for K2 in range(n_blocks)]

    # Vertices satisfying (1') BY CONSTRUCTION: fix the (q1,q2,Ka,Kb)
    # entry freely, then define its (q2,q1,Kb,Ka) partner as the
    # (0,2,1)-transpose. Complex on purpose -- reality is not required.
    qv: dict[tuple[int, int], dict] = {(a, b): {} for a in range(nq)
                                       for b in range(nq)}
    for a in range(nq):
        for b in range(nq):
            for (I, K1, K2) in keys:
                if (I, K1, K2) in qv[(a, b)]:
                    continue
                T = (rng.standard_normal((nbs, nbs, nbs))
                     + 1j * rng.standard_normal((nbs, nbs, nbs)))
                if (a, K1) == (b, K2):
                    T = 0.5 * (T + T.transpose(0, 2, 1))
                qv[(a, b)][(I, K1, K2)] = T
                qv[(b, a)][(I, K2, K1)] = T.transpose(0, 2, 1)
    for a in range(nq):
        for b in range(nq):
            for (I, K1, K2) in keys:
                assert np.allclose(qv[(b, a)][(I, K2, K1)],
                                   qv[(a, b)][(I, K1, K2)].transpose(0, 2, 1))

    # PSD legs per (omega, q); the fold partner G^>(-q)^T is then the
    # transpose of a PSD matrix, hence PSD.
    def psd_stack():
        out = np.empty((ne, nq, N, N), complex)
        for k in range(ne):
            for iq in range(nq):
                c = (rng.standard_normal((N, 1))
                     + 1j * rng.standard_normal((N, 1)))
                out[k, iq] = c @ c.conj().T + 1e-6 * np.eye(N)
        return out

    P_l, P_g = psd_stack(), psd_stack()
    rows, cols = [], []
    for I in range(n_blocks):
        for J in range(n_blocks):
            for i in range(nbs):
                for j in range(nbs):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = _dev_pattern(csr_matrix(
        (np.ones(len(rows), np.complex128), (np.array(rows), np.array(cols))),
        shape=(N, N)))
    mk = lambda: DSDBCOO.from_sparray(pattern, block_sizes,
                                      global_stack_shape=(ne, nq))
    g_l, g_g, s_l, s_g, s_r = mk(), mk(), mk(), mk(), mk()
    for m in (g_l, g_g, s_l, s_g, s_r):
        m.data[:] = 0.0
    glv, ggv = g_l.stack[...], g_g.stack[...]
    for K in range(n_blocks):
        for Kp in range(n_blocks):
            sK = slice(offs[K], offs[K + 1])
            sKp = slice(offs[Kp], offs[Kp + 1])
            glv.blocks[K, Kp] = xp.asarray(P_l[:, :, sK, sKp])
            ggv.blocks[K, Kp] = xp.asarray(P_g[:, :, sK, sKp])

    ssp = SigmaPhononPhonon(
        _make_cfg("half", g_band=1),
        phonon_frequencies=np.linspace(0.0, 16.0, ne),
        block_sizes=block_sizes, qfold=(qv, q_diff_map, nq))
    ssp.compute(g_l, g_g, out=(s_l, s_g, s_r))

    slv, sgv = s_l.stack[...], s_g.stack[...]
    for tag, view in (("<", slv), (">", sgv)):
        S = np.zeros((ne, nq, N, N), complex)
        for I in range(n_blocks):
            for J in range(n_blocks):
                S[:, :, offs[I]:offs[I + 1], offs[J]:offs[J + 1]] = (
                    np.asarray(get_host(view.blocks[I, J])))
        z = None
        for cand in (1.0, -1.0, 1j, -1j):
            M = cand * S
            if np.linalg.norm(M - M.conj().swapaxes(-1, -2)) > 1e-8 * np.linalg.norm(M):
                continue
            if np.linalg.eigvalsh(M).min() >= -1e-10 * np.abs(M).max():
                z = cand
                break
        assert z is not None, (
            f"coupled-q Sigma^{tag} has no PSD orientation -- the ring is "
            "not behaving as a congruence at nq>1")
        M = 0.5 * (z * S + (z * S).conj().swapaxes(-1, -2))
        lam = np.linalg.eigvalsh(M).min()
        assert lam >= -1e-9 * np.abs(M).max(), (
            f"coupled-q Sigma^{tag} lost PSD-ness: min eig {lam:.3e}")


def test_taper_above_band_one_warns() -> None:
    """Combining the taper with g_band > 1 must not look safe."""
    _, _, _, block_sizes, _, _ = _taper_fixture(5, 2, 9)
    phi_blocks = _taper_fixture(5, 2, 9)[0]
    cfg = _make_cfg("fft", g_band=2)
    cfg.phonon.sse_g_band_taper = "bartlett"
    with pytest.warns(UserWarning, match="only restores.*g_band = 1"):
        SigmaPhononPhonon(
            cfg, phonon_frequencies=np.linspace(0.0, 16.0, 9),
            block_sizes=block_sizes, phi_blocks=phi_blocks)


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


@pytest.mark.parametrize("g_band,n_blocks_par", [(1, 3), (2, 3), (3, 4)])
def test_compute_coupled_q_matches_reference(g_band, n_blocks_par) -> None:
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
    n_blocks, nbs, ne, nq = n_blocks_par, 3, 13, 3
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
        for Kp in range(max(0, K - g_band), min(n_blocks, K + g_band + 1)):
            gl_band[(K, Kp)] = rng.standard_normal(
                (ne, nq, nbs, nbs)
            ) + 1j * rng.standard_normal((ne, nq, nbs, nbs))
            gg_band[(K, Kp)] = rng.standard_normal(
                (ne, nq, nbs, nbs)
            ) + 1j * rng.standard_normal((ne, nq, nbs, nbs))

    rows, cols = [], []
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    for I in range(n_blocks):
        for J in range(max(0, I - g_band), min(n_blocks, I + g_band + 1)):
            for i in range(block_sizes[I]):
                for j in range(block_sizes[J]):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = _dev_pattern(csr_matrix(
        (np.ones(len(rows), np.complex128), (np.array(rows), np.array(cols))),
        shape=(N, N),
    ))
    mk = lambda: DSDBCOO.from_sparray(
        pattern, block_sizes, global_stack_shape=(ne, nq)
    )
    g_l, g_g, s_l, s_g, s_r = mk(), mk(), mk(), mk(), mk()
    for m in (g_l, g_g, s_l, s_g, s_r):
        m.data[:] = 0.0
    glv, ggv = g_l.stack[...], g_g.stack[...]
    for (K, Kp) in gl_band:
        glv.blocks[K, Kp] = xp.asarray(gl_band[(K, Kp)])
        ggv.blocks[K, Kp] = xp.asarray(gg_band[(K, Kp)])

    cfg = _make_cfg("half", g_band=g_band)
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
            for K1p in range(max(0, K1 - g_band),
                             min(n_blocks, K1 + g_band + 1)):
                for K2p in range(max(0, K2 - g_band),
                                 min(n_blocks, K2 + g_band + 1)):
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
                np.asarray(get_host(slv.blocks[I, J])), ref_l.get((I, J), 0),
                atol=1e-40, rtol=1e-9,
                err_msg=f"coupled-q Sigma^< mismatch at {(I, J)}",
            )
            np.testing.assert_allclose(
                np.asarray(get_host(sgv.blocks[I, J])), ref_g.get((I, J), 0),
                atol=1e-40, rtol=1e-9,
                err_msg=f"coupled-q Sigma^> mismatch at {(I, J)}",
            )


@pytest.mark.parametrize("dense_batched", [True, False])
def test_coupled_q_greater_from_lesser_matches_six_ring(
    dense_batched, capsys
) -> None:
    """Coupled-q (nq > 1) Sigma^> reconstruction A/B through the full
    ``compute()`` entry: ``sse_greater_from_lesser`` (4-ring bosonic tau
    fold + external q negation) == the legacy 6-ring path, on Sigma^< AND
    Sigma^>.

    The fold identity Sigma^>_IJ(q,tau) = Sigma^<_JI(-q,-tau)^T is
    STRUCTURAL inside one compute() call (the reversed absorption legs are
    built in-process from the very same gl/gg arrays), so the G bands are
    arbitrary independent random -- no gl/gg relation is enforced. The one
    INPUT condition is the vertex reality Phi(-q1,-q2) = conj(Phi(q1,q2))
    (real real-space FC3; production qfold vertices satisfy it), enforced
    here by symmetrising the random q-folded dict (the self-paired (0,0)
    entry is made real). nq=3 so the q -> -q gather is actually exercised.
    """
    from qttools.datastructures import DSDBCOO
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(11)
    n_blocks, nbs, ne, nq, g_band = 3, 3, 13, 3, 1
    block_sizes = np.array([nbs] * n_blocks)
    N = int(block_sizes.sum())
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

    # Vertex reality of a real real-space FC3: Phi(-q1,-q2) = conj(Phi(q1,q2)),
    # with the self-paired entries ((-a)%nq, (-b)%nq) == (a, b) real.
    qvertices: dict = {}
    for a in range(nq):
        for b in range(nq):
            na, nb = (-a) % nq, (-b) % nq
            if (na, nb) in qvertices:
                qvertices[(a, b)] = {
                    k: np.conj(v) for k, v in qvertices[(na, nb)].items()
                }
            elif (na, nb) == (a, b):
                qvertices[(a, b)] = {
                    k: v.real.astype(complex) for k, v in _phi().items()
                }
            else:
                qvertices[(a, b)] = _phi()

    gl_band, gg_band = {}, {}
    for K in range(n_blocks):
        for Kp in range(max(0, K - g_band), min(n_blocks, K + g_band + 1)):
            gl_band[(K, Kp)] = rng.standard_normal(
                (ne, nq, nbs, nbs)
            ) + 1j * rng.standard_normal((ne, nq, nbs, nbs))
            gg_band[(K, Kp)] = rng.standard_normal(
                (ne, nq, nbs, nbs)
            ) + 1j * rng.standard_normal((ne, nq, nbs, nbs))

    rows, cols = [], []
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    for I in range(n_blocks):
        for J in range(max(0, I - g_band), min(n_blocks, I + g_band + 1)):
            for i in range(block_sizes[I]):
                for j in range(block_sizes[J]):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = _dev_pattern(csr_matrix(
        (np.ones(len(rows), np.complex128), (np.array(rows), np.array(cols))),
        shape=(N, N),
    ))

    def _run(g_from_l: bool, verify: int = 0):
        mk = lambda: DSDBCOO.from_sparray(
            pattern, block_sizes, global_stack_shape=(ne, nq)
        )
        g_l, g_g, s_l, s_g, s_r = mk(), mk(), mk(), mk(), mk()
        for m in (g_l, g_g, s_l, s_g, s_r):
            m.data[:] = 0.0
        glv, ggv = g_l.stack[...], g_g.stack[...]
        for (K, Kp) in gl_band:
            glv.blocks[K, Kp] = xp.asarray(gl_band[(K, Kp)])
            ggv.blocks[K, Kp] = xp.asarray(gg_band[(K, Kp)])
        cfg = _make_cfg("half", g_band=g_band)
        cfg.phonon.sse_greater_from_lesser = g_from_l
        cfg.phonon.sse_dense_q_batched = dense_batched
        cfg.phonon.sse_fold_verify_iterations = verify
        ssp = SigmaPhononPhonon(
            cfg,
            phonon_frequencies=np.linspace(0.0, 16.0, ne),
            block_sizes=block_sizes,
            qfold=(qvertices, q_diff_map, nq),
        )
        ssp.compute(g_l, g_g, out=(s_l, s_g, s_r))
        return (
            np.asarray(get_host(s_l.data)),
            np.asarray(get_host(s_g.data)),
        )

    sl6, sg6 = _run(False)
    sl4, sg4 = _run(True)
    np.testing.assert_allclose(
        sl4, sl6, rtol=1e-12, atol=1e-12 * np.abs(sl6).max(),
        err_msg="coupled-q greater_from_lesser Sigma^< mismatch",
    )
    np.testing.assert_allclose(
        sg4, sg6, rtol=1e-12, atol=1e-12 * np.abs(sg6).max(),
        err_msg="coupled-q greater_from_lesser Sigma^> mismatch",
    )

    # In-process verify gate at nq > 1: the first call runs the legacy
    # 6-ring path and checks the fold identity (tau reversal + JI/orbital
    # transpose + q negation) in place.
    capsys.readouterr()
    slv, sgv = _run(True, verify=1)
    out = capsys.readouterr().out
    assert "fold-verify" in out
    assert "OK" in out and "MISMATCH" not in out
    # The verify iteration ships the legacy result.
    np.testing.assert_allclose(
        sgv, sg6, rtol=1e-12, atol=1e-12 * np.abs(sg6).max(),
        err_msg="verify-gate iteration did not ship the legacy Sigma^>",
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
    g_band = 1  # these fixtures are band-1
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
        for Kp in range(max(0, K - g_band), min(n_blocks, K + g_band + 1)):
            gl_band[(K, Kp)] = (rng.standard_normal((ne, nq, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nq, nbs, nbs)))
            gg_band[(K, Kp)] = (rng.standard_normal((ne, nq, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nq, nbs, nbs)))

    rows, cols = [], []
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    for I in range(n_blocks):
        for J in range(max(0, I - g_band), min(n_blocks, I + g_band + 1)):
            for i in range(block_sizes[I]):
                for j in range(block_sizes[J]):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = _dev_pattern(csr_matrix(
        (np.ones(len(rows), np.complex128), (np.array(rows), np.array(cols))),
        shape=(N, N),
    ))

    def _run(**ssp_kwargs):
        mk = lambda: DSDBCOO.from_sparray(
            pattern, block_sizes, global_stack_shape=(ne, nq))
        g_l, g_g, s_l, s_g, s_r = mk(), mk(), mk(), mk(), mk()
        for m in (g_l, g_g, s_l, s_g, s_r):
            m.data[:] = 0.0
        glv, ggv = g_l.stack[...], g_g.stack[...]
        for (K, Kp) in gl_band:
            glv.blocks[K, Kp] = xp.asarray(gl_band[(K, Kp)])
            ggv.blocks[K, Kp] = xp.asarray(gg_band[(K, Kp)])
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
                np.asarray(get_host(fv_l.blocks[I, J])),
                np.asarray(get_host(dv_l.blocks[I, J])), atol=1e-45, rtol=1e-9,
                err_msg=f"factored Sigma^< mismatch at {(I, J)} [{ansatz}]",
            )
            np.testing.assert_allclose(
                np.asarray(get_host(fv_g.blocks[I, J])),
                np.asarray(get_host(dv_g.blocks[I, J])), atol=1e-45, rtol=1e-9,
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
        np.asarray(get_host(
            _convolve_q(xp.asarray(pa), xp.asarray(pb), (nkx, nky), xp))),
        expected, rtol=1e-12,
    )


@pytest.mark.parametrize("ansatz", ["INDSCAL", "CP"])
def test_compute_gamma_factored_matches_dense(ansatz) -> None:
    """Factored kernel == dense ring at Gamma (nq == 1).

    At the zone centre the momentum convolution is the identity, so what is
    exercised here is the Gram collapse alone -- the regime of every
    transversely-finite device (nanowires, CNTs), which the factored kernel
    could not reach before.
    """
    g_band = 1  # these fixtures are band-1
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
        for Kp in range(max(0, K - g_band), min(n_blocks, K + g_band + 1)):
            gl_band[(K, Kp)] = (rng.standard_normal((ne, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nbs, nbs)))
            gg_band[(K, Kp)] = (rng.standard_normal((ne, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nbs, nbs)))

    rows, cols = [], []
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    for I in range(n_blocks):
        for J in range(max(0, I - g_band), min(n_blocks, I + g_band + 1)):
            for i in range(block_sizes[I]):
                for j in range(block_sizes[J]):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = _dev_pattern(csr_matrix(
        (np.ones(len(rows), np.complex128), (np.array(rows), np.array(cols))),
        shape=(N, N),
    ))

    def _run(**ssp_kwargs):
        mk = lambda: DSDBCOO.from_sparray(
            pattern, block_sizes, global_stack_shape=(ne,))
        g_l, g_g, s_l, s_g, s_r = mk(), mk(), mk(), mk(), mk()
        for m in (g_l, g_g, s_l, s_g, s_r):
            m.data[:] = 0.0
        glv, ggv = g_l.stack[...], g_g.stack[...]
        for (K, Kp) in gl_band:
            glv.blocks[K, Kp] = xp.asarray(gl_band[(K, Kp)])
            ggv.blocks[K, Kp] = xp.asarray(gg_band[(K, Kp)])
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
                np.asarray(get_host(fv_l.blocks[I, J])),
                np.asarray(get_host(dv_l.blocks[I, J])), atol=1e-45, rtol=1e-9,
                err_msg=f"Gamma factored Sigma^< mismatch at {(I, J)}",
            )
            np.testing.assert_allclose(
                np.asarray(get_host(fv_g.blocks[I, J])),
                np.asarray(get_host(dv_g.blocks[I, J])), atol=1e-45, rtol=1e-9,
                err_msg=f"Gamma factored Sigma^> mismatch at {(I, J)}",
            )


def test_cross_slab_scale_matches_manual_ablation() -> None:
    """``sse_cross_slab_scale`` == manually scaling the non-uniform slab
    triples of the injected q-folded vertex; 1.0 is bit-identical to the
    unscaled run. Coupled-q path (nq=3), g_band=1."""
    from qttools.datastructures import DSDBCOO
    from scipy.sparse import csr_matrix
    from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

    rng = np.random.default_rng(11)
    n_blocks, nbs, ne, nq = 3, 3, 13, 3
    block_sizes = np.array([nbs] * n_blocks)
    N = int(block_sizes.sum())
    q_diff_map = np.array(
        [[(a - b) % nq for b in range(nq)] for a in range(nq)])
    keys = [
        (I, K1, K2)
        for I in range(n_blocks)
        for K1 in range(max(0, I - 1), min(n_blocks, I + 2))
        for K2 in range(max(0, I - 1), min(n_blocks, I + 2))
        if abs(K1 - K2) <= 1
    ]
    qvertices = {
        (a, b): {
            k: rng.standard_normal((nbs, nbs, nbs))
            + 1j * rng.standard_normal((nbs, nbs, nbs))
            for k in keys
        }
        for a in range(nq)
        for b in range(nq)
    }
    gl_band = {
        (K, K): rng.standard_normal((ne, nq, nbs, nbs))
        + 1j * rng.standard_normal((ne, nq, nbs, nbs))
        for K in range(n_blocks)
    }
    gg_band = {
        (K, K): rng.standard_normal((ne, nq, nbs, nbs))
        + 1j * rng.standard_normal((ne, nq, nbs, nbs))
        for K in range(n_blocks)
    }

    rows, cols = [], []
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            for i in range(block_sizes[I]):
                for j in range(block_sizes[J]):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = _dev_pattern(csr_matrix(
        (np.ones(len(rows), np.complex128),
         (np.array(rows), np.array(cols))),
        shape=(N, N),
    ))

    def _run(qverts, xscale):
        mk = lambda: DSDBCOO.from_sparray(
            pattern, block_sizes, global_stack_shape=(ne, nq))
        g_l, g_g, s_l, s_g, s_r = mk(), mk(), mk(), mk(), mk()
        for m in (g_l, g_g, s_l, s_g, s_r):
            m.data[:] = 0.0
        glv, ggv = g_l.stack[...], g_g.stack[...]
        for (K, Kp) in gl_band:
            glv.blocks[K, Kp] = xp.asarray(gl_band[(K, Kp)])
            ggv.blocks[K, Kp] = xp.asarray(gg_band[(K, Kp)])
        cfg = _make_cfg("half", g_band=1)
        cfg.phonon.sse_cross_slab_scale = xscale
        ssp = SigmaPhononPhonon(
            cfg,
            phonon_frequencies=np.linspace(0.0, 16.0, ne),
            block_sizes=block_sizes,
            qfold=(
                {qp: {k: v.copy() for k, v in d.items()}
                 for qp, d in qverts.items()},
                q_diff_map, nq),
        )
        ssp.compute(g_l, g_g, out=(s_l, s_g, s_r))
        return s_l, s_g

    def _scaled(qverts, s):
        return {
            qp: {
                k: (v if k[0] == k[1] == k[2] else s * v)
                for k, v in d.items()
            }
            for qp, d in qverts.items()
        }

    sl_a, sg_a = _run(qvertices, 0.5)
    sl_b, sg_b = _run(_scaled(qvertices, 0.5), 1.0)
    np.testing.assert_allclose(
        np.asarray(get_host(sl_a.data)), np.asarray(get_host(sl_b.data)),
        atol=1e-40, rtol=1e-12,
        err_msg="cross-slab scale 0.5 != manually scaled vertex")
    np.testing.assert_allclose(
        np.asarray(get_host(sg_a.data)), np.asarray(get_host(sg_b.data)),
        atol=1e-40, rtol=1e-12)

    # scale = 1.0 takes the legacy path (the rebuild is skipped outright);
    # two independent runs agree to reduction-order jitter.
    sl_c, _ = _run(qvertices, 1.0)
    sl_d, _ = _run(qvertices, 1.0)
    np.testing.assert_allclose(
        np.asarray(get_host(sl_c.data)), np.asarray(get_host(sl_d.data)),
        atol=1e-40, rtol=1e-12,
        err_msg="scale 1.0 must match the default path")
