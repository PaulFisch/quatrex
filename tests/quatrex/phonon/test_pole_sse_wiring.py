# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""The pole sector's call sites inside the production bubble.

Two things have to be true of any optional numerics feature in this solver, and
they pull in opposite directions: switched off it must be BIT-identical to the
code that existed before, and switched on it must demonstrably act. Both are
pinned here against the real ``SigmaPhononPhonon.compute`` path rather than a
mock, because the risk lives in where the hooks sit relative to the DC mask and
the Kramers-Kronig transform.
"""
import numpy as np
import pytest

from qttools import xp
from qttools.comm import comm as _qtt_comm
from qttools.utils.gpu_utils import get_host

from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon


def _configure_serial_comm() -> None:
    if _qtt_comm._is_configured:
        return
    backend = "device_mpi" if xp.__name__ == "numpy" else "host_mpi"
    cfg = {k: backend for k in ("all_to_all", "all_gather", "all_reduce", "bcast")}
    _qtt_comm.configure(block_comm_size=1, block_comm_config=cfg,
                        stack_comm_config=cfg, override=True)


def setup_module() -> None:
    _configure_serial_comm()


class _Pole:
    """Stand-in for PoleSectorConfig carrying only what the SSE reads."""

    def __init__(self, enabled):
        self.enabled = enabled


def _make_cfg(pole_enabled=None):
    class _Phonon:
        pass

    _Phonon.retarded_method = "fft"
    _Phonon.fc3_path = None
    _Phonon.sse_tau_chunk_bytes = 4096
    _Phonon.sse_g_band = 1
    if pole_enabled is not None:
        _Phonon.pole_sector = _Pole(pole_enabled)

    class _Cfg:
        phonon = _Phonon()

    return _Cfg()


def _fixture(n_blocks=3, nbs=2, ne=13, seed=5):
    from scipy.sparse import csr_matrix

    from qttools.datastructures import DSDBCOO

    rng = np.random.default_rng(seed)
    block_sizes = np.array([nbs] * n_blocks)
    N = int(block_sizes.sum())
    offs = np.concatenate(([0], np.cumsum(block_sizes)))

    phi_blocks = {}
    for I in range(n_blocks):
        for K1 in range(max(0, I - 1), min(n_blocks, I + 2)):
            for K2 in range(max(0, I - 1), min(n_blocks, I + 2)):
                if abs(K1 - K2) > 1:
                    continue
                b = rng.normal(size=(nbs, nbs, nbs))
                phi_blocks[(I, K1, K2)] = b
    # Leg-exchange symmetry, the S3 property the congruence needs.
    for (I, K1, K2), b in list(phi_blocks.items()):
        mirror = phi_blocks.get((I, K2, K1))
        if mirror is not None:
            sym = 0.5 * (b + mirror.transpose(0, 2, 1))
            phi_blocks[(I, K1, K2)] = sym
            phi_blocks[(I, K2, K1)] = sym.transpose(0, 2, 1)

    pattern = np.zeros((N, N))
    for i in range(n_blocks):
        for j in range(max(0, i - 1), min(n_blocks, i + 2)):
            pattern[offs[i]:offs[i + 1], offs[j]:offs[j + 1]] = 1.0
    pattern = csr_matrix(pattern)

    def make():
        return tuple(
            DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(ne,))
            for _ in range(5)
        )

    freqs = np.linspace(0.0, 16.0, ne)
    return phi_blocks, block_sizes, freqs, make


def _fill_legs(gl, gg, seed=11):
    rng = np.random.default_rng(seed)
    for buf in (gl, gg):
        buf.data[:] = xp.asarray(
            1j * np.abs(rng.normal(size=buf.data.shape))
        )


def _run(pole_enabled, channel=None, analytic=None, seed=11):
    phi, bs, freqs, make = _fixture()
    gl, gg, sl, sg, sr = make()
    _fill_legs(gl, gg, seed)
    # The driver zeroes the self-energy buffers before the bubble writes into
    # them (_stash_sigma); without that the bubble's contribution sits on top of
    # whatever the allocator left behind and the comparisons are diluted.
    for buf in (sl, sg, sr):
        buf.data[:] = 0.0
    ssp = SigmaPhononPhonon(_make_cfg(pole_enabled), phonon_frequencies=freqs,
                            block_sizes=bs, phi_blocks=phi)
    if channel is not None:
        ssp.set_pole_channel(*channel)
    if analytic is not None:
        ssp.set_pole_self_energy(*analytic)
    ssp.compute(gl, gg, out=(sl, sg, sr))
    return ssp, tuple(np.asarray(get_host(b.data)).copy() for b in (sl, sg, sr))


# --------------------------------------------------------------------------- #

def test_disabled_sector_is_bit_identical_to_the_absent_option():
    """A config without the option at all, and one with it off, must agree exactly."""
    _, absent = _run(pole_enabled=None)
    _, off = _run(pole_enabled=False)
    for a, b in zip(absent, off):
        np.testing.assert_array_equal(a, b)


def test_enabled_but_uninjected_sector_is_bit_identical():
    """Enabling the switch without injecting anything must change nothing.

    The hooks are guarded on the channel being present, so an iteration in which
    no pole survived screening has to fall through to the legacy path exactly.
    """
    _, off = _run(pole_enabled=False)
    _, on = _run(pole_enabled=True)
    for a, b in zip(off, on):
        np.testing.assert_array_equal(a, b)


def test_leg_subtraction_changes_the_bubble():
    """Injecting a leg channel must actually remove weight from the bubble."""
    phi, bs, freqs, make = _fixture()
    gl, gg, *_ = make()
    _fill_legs(gl, gg)
    chan = (0.25 * np.asarray(get_host(gl.data)).copy(),
            0.25 * np.asarray(get_host(gg.data)).copy())

    _, base = _run(pole_enabled=True)
    _, sub = _run(pole_enabled=True, channel=(xp.asarray(chan[0]),
                                              xp.asarray(chan[1])))
    diff = np.abs(sub[0] - base[0]).max() / np.abs(base[0]).max()
    assert diff > 1e-3, f"leg subtraction had no effect ({diff:.3e})"


def test_analytic_term_is_added_to_all_three_outputs():
    phi, bs, freqs, make = _fixture()
    gl, gg, sl, sg, sr = make()
    shape = sl.data.shape
    add = (xp.asarray(np.full(shape, 0.5 + 0.25j)),
           xp.asarray(np.full(shape, -0.5 + 0.125j)),
           xp.asarray(np.full(shape, 0.75 - 0.5j)))

    _, base = _run(pole_enabled=True)
    _, withss = _run(pole_enabled=True, analytic=add)

    # All three are masked at omega = 0, exactly as the numerical half is.
    mask = np.zeros(shape[0], dtype=bool)
    mask[np.abs(np.asarray(get_host(freqs))) < 1e-6] = True
    assert mask.any(), "the DC bin is not on this grid; the mask is untested"

    def expect(a, sign):
        out = sign * np.asarray(get_host(a)).copy()
        out[mask] = 0.0
        return out

    # Sigma^{<,>} carry the solver's sign flip; Sigma^R is added as supplied.
    np.testing.assert_allclose(withss[0] - base[0], expect(add[0], -1), atol=1e-12)
    np.testing.assert_allclose(withss[1] - base[1], expect(add[1], -1), atol=1e-12)
    np.testing.assert_allclose(withss[2] - base[2], expect(add[2], +1), atol=1e-12)


def test_injected_channel_does_not_survive_the_iteration():
    """A channel built from one iterate must never be reused against the next."""
    phi, bs, freqs, make = _fixture()
    gl, gg, sl, sg, sr = make()
    _fill_legs(gl, gg)
    chan = (xp.asarray(0.25 * np.asarray(get_host(gl.data)).copy()),
            xp.asarray(0.25 * np.asarray(get_host(gg.data)).copy()))
    ssp, _ = _run(pole_enabled=True, channel=chan)
    assert ssp._pole_injection.channel is None
    assert ssp._pole_injection.self_energy is None


def test_mixed_channel_is_bit_identical_when_absent():
    """The fourth hook must also fall through exactly when nothing is injected."""
    _, off = _run(pole_enabled=False)
    _, on = _run(pole_enabled=True)
    for a, b in zip(off, on):
        np.testing.assert_array_equal(a, b)


def test_mixed_channel_reaches_the_retarded_self_energy():
    """Unlike Sigma_SS, the mixed terms must be SEEN by the Hilbert transform."""
    phi, bs, freqs, make = _fixture()
    gl, gg, sl, sg, sr = make()
    shape = sl.data.shape
    add = (xp.asarray(np.full(shape, 0.3 + 0.2j)),
           xp.asarray(np.full(shape, -0.3 + 0.1j)))

    def run_with_mixed():
        gl2, gg2, sl2, sg2, sr2 = make()
        _fill_legs(gl2, gg2)
        for buf in (sl2, sg2, sr2):
            buf.data[:] = 0.0
        ssp = SigmaPhononPhonon(_make_cfg(True), phonon_frequencies=freqs,
                                block_sizes=bs, phi_blocks=phi)
        ssp.set_pole_mixed(*add)
        ssp.compute(gl2, gg2, out=(sl2, sg2, sr2))
        return tuple(np.asarray(get_host(b.data)).copy() for b in (sl2, sg2, sr2))

    base = _run(pole_enabled=True)[1]
    withmx = run_with_mixed()
    assert np.abs(withmx[0] - base[0]).max() > 1e-12, "lesser unchanged"
    assert np.abs(withmx[2] - base[2]).max() > 1e-12, (
        "Sigma^R unchanged: the mixed terms were injected after delta was formed"
    )


def test_bubble_correction_reaches_the_retarded_self_energy():
    """It must be SEEN by the Hilbert transform, like the mixed sectors."""
    phi, bs, freqs, make = _fixture()
    gl, gg, sl, sg, sr = make()
    shape = sl.data.shape
    add = (xp.asarray(np.full(shape, 0.2 - 0.1j)),
           xp.asarray(np.full(shape, -0.2 + 0.4j)))

    def run_with_correction():
        gl2, gg2, sl2, sg2, sr2 = make()
        _fill_legs(gl2, gg2)
        for buf in (sl2, sg2, sr2):
            buf.data[:] = 0.0
        ssp = SigmaPhononPhonon(_make_cfg(True), phonon_frequencies=freqs,
                                block_sizes=bs, phi_blocks=phi)
        ssp.set_bubble_correction(*add)
        ssp.compute(gl2, gg2, out=(sl2, sg2, sr2))
        return tuple(np.asarray(get_host(b.data)).copy() for b in (sl2, sg2, sr2))

    base = _run(pole_enabled=True)[1]
    got = run_with_correction()
    assert np.abs(got[0] - base[0]).max() > 1e-12, "lesser unchanged"
    assert np.abs(got[2] - base[2]).max() > 1e-12, (
        "Sigma^R unchanged: the correction was injected after delta was formed")


def test_bubble_correction_falls_through_exactly_when_absent():
    """Nothing injected must be bit-identical, or the default is not free."""
    _, off = _run(pole_enabled=False)
    _, on = _run(pole_enabled=True)
    for a, b in zip(off, on):
        np.testing.assert_array_equal(a, b)


def test_bubble_correction_does_not_survive_the_iteration():
    """It is built from THIS iterate's poles and must never outlive them."""
    phi, bs, freqs, make = _fixture()
    gl, gg, sl, sg, sr = make()
    _fill_legs(gl, gg)
    shape = sl.data.shape
    ssp = SigmaPhononPhonon(_make_cfg(True), phonon_frequencies=freqs,
                            block_sizes=bs, phi_blocks=phi)
    ssp.set_bubble_correction(xp.asarray(np.full(shape, 0.1 + 0j)),
                              xp.asarray(np.full(shape, 0.1 + 0j)))
    ssp.compute(gl, gg, out=(sl, sg, sr))
    assert ssp._pole_injection.covariance is None
