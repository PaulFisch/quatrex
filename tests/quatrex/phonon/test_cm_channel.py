# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Unit tests for the CM-channel subtraction
(``quatrex.phonon.cm_channel`` + the ``sse_cm_subtraction`` seam in
``SigmaPhononPhonon``). Physics and measurements:
``phonon/docs/ir_residue_derivation.md``.
"""
from __future__ import annotations

import numpy as np
import pytest

from qttools import xp
from qttools.comm import comm as _qtt_comm
from qttools.utils.gpu_utils import get_host

from quatrex.phonon.cm_channel import (
    cm_sigma_pair, lead_velocity_matrices, translation_null_modes)
from quatrex.phonon.ir_subtraction import bose

K_SPRING = 6.0
M1, M2 = 1.0, 2.4


def _configure_serial_comm() -> None:
    if _qtt_comm._is_configured:
        return
    backend = "device_mpi" if xp.__name__ == "numpy" else "host_mpi"
    cfg = {k: backend for k in ("all_to_all", "all_gather", "all_reduce",
                                "bcast")}
    _qtt_comm.configure(
        block_comm_size=1,
        block_comm_config=cfg,
        stack_comm_config=cfg,
        override=True,
    )


def setup_module() -> None:  # pytest hook
    _configure_serial_comm()


def _chain_blocks():
    """Mass-weighted diatomic 1D chain (1 DOF per atom)."""
    s11, s22 = 2 * K_SPRING / M1, 2 * K_SPRING / M2
    s12 = -K_SPRING / np.sqrt(M1 * M2)
    d00 = np.array([[s11, s12], [s12, s22]], complex)
    d01 = np.array([[0.0, 0.0], [s12, 0.0]], complex)
    return d00, d01, d01.conj().T


def test_null_modes_are_mass_weighted_translation() -> None:
    d00, d01, d10 = _chain_blocks()
    t, eigs = translation_null_modes(d00, d01, d10)
    assert t.shape == (1, 2)
    assert abs(eigs[0]) < 1e-12
    expect = np.array([np.sqrt(M1), np.sqrt(M2)])
    expect /= np.linalg.norm(expect)
    assert np.allclose(np.abs(t[0]), expect, atol=1e-12)


def test_lead_velocities_symmetric_and_positive() -> None:
    d00, d01, d10 = _chain_blocks()
    t, _ = translation_null_modes(d00, d01, d10)
    v_l, v_r, _ = lead_velocity_matrices(d00, d01, d10, t)
    # symmetric chain: left and right leads identical
    assert np.allclose(v_l, v_r, rtol=1e-6)
    assert np.linalg.eigvalsh(v_l).min() > 0.0


def test_cm_pair_fold_kms_laurent() -> None:
    d00, d01, d10 = _chain_blocks()
    t, _ = translation_null_modes(d00, d01, d10)
    v_l, v_r, _ = lead_velocity_matrices(d00, d01, d10, t)
    t_l, t_r = 305.0, 295.0
    for w in (0.05, 0.3, 1.7):
        m_l, m_g = cm_sigma_pair(w, v_l, v_r, t_l, t_r)
        m_l_neg, m_g_neg = cm_sigma_pair(-w, v_l, v_r, t_l, t_r)
        # bosonic fold: S^<(-w) = S^>(w)^T (and vice versa)
        assert np.allclose(m_l_neg, m_g.T, rtol=1e-12)
        assert np.allclose(m_g_neg, m_l.T, rtol=1e-12)
    # KMS at equal temperatures: S^> n = S^< (n + 1) pointwise
    for w in (0.05, 0.3, 1.7):
        m_l, m_g = cm_sigma_pair(w, v_l, v_r, 300.0, 300.0)
        n = bose(w, 300.0)
        assert np.allclose(m_g * n, m_l * (n + 1.0), rtol=1e-12)
    # Laurent limit: w^2 (-i S^<) -> C2 = V^-1 (sum 2 c_a V_a) V^-1
    from quatrex.phonon.ir_subtraction import bose_pole_coeff
    v_t = v_l + v_r
    vinv = np.linalg.inv(v_t)
    c2 = vinv @ (2 * bose_pole_coeff(t_l) * v_l
                 + 2 * bose_pole_coeff(t_r) * v_r) @ vinv
    w = 1e-5
    m_l, _ = cm_sigma_pair(w, v_l, v_r, t_l, t_r)
    assert np.allclose((w * w) * (-1j * m_l).real, c2, rtol=1e-3)


def test_sse_seam_identity() -> None:
    """Flag-ON compute == flag-OFF compute on manually subtracted legs.

    The decisive seam test: it pins the nnz mapping, the Gamma-index
    application, and the pre-mask placement without re-deriving any
    bubble reference.
    """
    from qttools.datastructures import DSDBCOO
    from scipy.sparse import csr_matrix
    from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

    rng = np.random.default_rng(7)
    n_blocks, nbs, ne = 3, 3, 21
    block_sizes = np.array([nbs] * n_blocks)
    N = int(block_sizes.sum())
    freqs_thz = np.linspace(0.0, 16.0, ne)

    phi_blocks = {}
    for I in range(n_blocks):
        for K1 in range(max(0, I - 1), min(n_blocks, I + 2)):
            for K2 in range(max(0, I - 1), min(n_blocks, I + 2)):
                if abs(K1 - K2) > 1:
                    continue
                phi_blocks[(I, K1, K2)] = (
                    rng.standard_normal((nbs, nbs, nbs))
                    + 1j * rng.standard_normal((nbs, nbs, nbs)))

    # synthetic rank-2 channel on the device
    t_dev = np.linalg.qr(rng.standard_normal((N, 2)))[0].T
    v_l = np.array([[0.3, 0.05], [0.05, 0.2]])
    v_r = np.array([[0.25, 0.0], [0.0, 0.35]])
    t_left, t_right = 305.0, 295.0

    def s_dense(w):
        if abs(w) < 1e-12:
            return np.zeros((N, N), complex), np.zeros((N, N), complex)
        m_l, m_g = cm_sigma_pair(float(w), v_l, v_r, t_left, t_right)
        return t_dev.T @ m_l @ t_dev, t_dev.T @ m_g @ t_dev

    pattern_rows, pattern_cols = [], []
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            for i in range(nbs):
                for j in range(nbs):
                    pattern_rows.append(offs[I] + i)
                    pattern_cols.append(offs[J] + j)
    pattern = csr_matrix(
        (np.ones(len(pattern_rows), dtype=np.complex128),
         (np.array(pattern_rows), np.array(pattern_cols))), shape=(N, N))
    if xp.__name__ == "cupy":
        from qttools import sparse as qsp
        pattern = qsp.csr_matrix(pattern)

    def bufs():
        out = [DSDBCOO.from_sparray(pattern, block_sizes,
                                    global_stack_shape=(ne,))
               for _ in range(5)]
        for m in out:
            m.data[:] = 0.0
        return out

    gl_band, gg_band = {}, {}
    for K in range(n_blocks):
        for Kp in range(max(0, K - 1), min(n_blocks, K + 2)):
            gl_band[(K, Kp)] = (rng.standard_normal((ne, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nbs, nbs)))
            gg_band[(K, Kp)] = (rng.standard_normal((ne, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nbs, nbs)))

    def run(flag_on: bool, presubtract: bool):
        g_l, g_g, s_l, s_g, s_r = bufs()
        glv, ggv = g_l.stack[...], g_g.stack[...]
        for (K, Kp) in gl_band:
            a = gl_band[(K, Kp)].copy()
            b = gg_band[(K, Kp)].copy()
            if presubtract:
                for i, w in enumerate(freqs_thz):
                    sd_l, sd_g = s_dense(w)
                    a[i] -= sd_l[offs[K]:offs[K + 1], offs[Kp]:offs[Kp + 1]]
                    b[i] -= sd_g[offs[K]:offs[K + 1], offs[Kp]:offs[Kp + 1]]
            glv.blocks[K, Kp] = xp.asarray(a)
            ggv.blocks[K, Kp] = xp.asarray(b)

        class _Phonon:
            retarded_method = "fft"
            fc3_path = None
            sse_tau_chunk_bytes = 4096
            sse_g_band = 1
            sse_cm_subtraction = flag_on

        class _Cfg:
            phonon = _Phonon()

        ssp = SigmaPhononPhonon(
            _Cfg(), phonon_frequencies=freqs_thz,
            block_sizes=block_sizes, phi_blocks=phi_blocks)
        if flag_on:
            ssp.set_cm_channel(t_dev, v_l, v_r, t_left, t_right)
        ssp.compute(g_l, g_g, out=(s_l, s_g, s_r))
        return (np.asarray(get_host(s_l.data)),
                np.asarray(get_host(s_g.data)),
                np.asarray(get_host(s_r.data)))

    a_l, a_g, a_r = run(flag_on=True, presubtract=False)
    b_l, b_g, b_r = run(flag_on=False, presubtract=True)
    for a, b in ((a_l, b_l), (a_g, b_g), (a_r, b_r)):
        assert np.allclose(a, b, rtol=1e-10, atol=1e-30)

    # and the flag must actually change the answer
    c_l, *_ = run(flag_on=False, presubtract=False)
    rel = np.linalg.norm(a_l - c_l) / np.linalg.norm(c_l)
    assert rel > 1e-6


def test_flag_on_without_channel_raises() -> None:
    from qttools.datastructures import DSDBCOO
    from scipy.sparse import csr_matrix
    from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

    ne, nbs = 11, 2
    freqs = np.linspace(0.0, 8.0, ne)
    pattern = csr_matrix(np.ones((nbs, nbs), dtype=np.complex128))
    if xp.__name__ == "cupy":
        from qttools import sparse as qsp
        pattern = qsp.csr_matrix(pattern)
    bs = np.array([nbs])
    ms = [DSDBCOO.from_sparray(pattern, bs, global_stack_shape=(ne,))
          for _ in range(5)]
    for m in ms:
        m.data[:] = 0.0

    class _Phonon:
        retarded_method = "fft"
        fc3_path = None
        sse_tau_chunk_bytes = 4096
        sse_g_band = 1
        sse_cm_subtraction = True

    class _Cfg:
        phonon = _Phonon()

    rng = np.random.default_rng(3)
    ssp = SigmaPhononPhonon(
        _Cfg(), phonon_frequencies=freqs, block_sizes=bs,
        phi_blocks={(0, 0, 0): rng.standard_normal((nbs, nbs, nbs))})
    with pytest.raises(RuntimeError, match="no CM channel"):
        ssp.compute(ms[0], ms[1], out=(ms[2], ms[3], ms[4]))
