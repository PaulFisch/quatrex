# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""The four correctness gates of the pole-subtracted sector.

Gate 1 (sector sum) is the load-bearing one: it asserts that splitting
``G = G_S + G_R`` and evaluating four bubble sectors reassembles the
undecomposed answer. Everything the hybrid claims rests on that identity being
exact rather than approximately true, so it is tested against a bubble
reference written independently of the production kernels, and with a negative
control that shows a dropped sector is actually visible.
"""
import numpy as np
import pytest

from quatrex.phonon.pole_audit import (
    keldysh_identity,
    psd_residual,
    sector_sum_residual,
    transpose_index,
)


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _pattern(sizes):
    """Block-tridiagonal pattern, structurally symmetric by construction."""
    off = np.concatenate(([0], np.cumsum(sizes)))
    rows, cols = [], []
    for i in range(len(sizes)):
        for j in range(max(0, i - 1), min(len(sizes), i + 2)):
            for a in range(off[i], off[i + 1]):
                for b in range(off[j], off[j + 1]):
                    rows.append(a)
                    cols.append(b)
    return np.array(rows), np.array(cols), off


# --------------------------------------------------------------------------
# Gate 1: the sector sum
# --------------------------------------------------------------------------

def _bubble(a, b, phi_l, phi_r, rows, cols, pref=1.0):
    """Reference ring on the stored pattern, written from the definition.

    ``Sigma_{mu mu'}(w) = pref * sum_{ab,cd} PhiL[mu,a,b] PhiR[mu',c,d]
    (A_{ac} * B_{bd})(w)`` with ``*`` the discrete convolution over the grid.
    Deliberately independent of ``pole_bridge``: it is the thing the sector
    decomposition is being checked against.
    """
    ne = a.shape[0]
    # conv[w, a, c, b, d] = sum_w' A[w', a, c] B[w-w', b, d]
    fa = np.fft.fft(a, axis=0)
    fb = np.fft.fft(b, axis=0)
    conv = np.fft.ifft(
        np.einsum("wac,wbd->wacbd", fa, fb), axis=0
    )
    out = np.empty((ne, rows.size), dtype=complex)
    for k, (mu, mup) in enumerate(zip(rows, cols)):
        out[:, k] = pref * np.einsum(
            "ab,cd,wacbd->w", phi_l[mu], np.conj(phi_r[mup]), conv
        )
    return out


def _bed(seed=0, ne=16, n_dof=4):
    rng = np.random.default_rng(seed)
    sizes = np.array([2, 2])
    rows, cols, _ = _pattern(sizes)
    phi_l = rng.normal(size=(n_dof, n_dof, n_dof))
    phi_r = rng.normal(size=(n_dof, n_dof, n_dof))
    g_s = (rng.normal(size=(ne, n_dof, n_dof))
           + 1j * rng.normal(size=(ne, n_dof, n_dof)))
    g_r = (rng.normal(size=(ne, n_dof, n_dof))
           + 1j * rng.normal(size=(ne, n_dof, n_dof)))
    return sizes, rows, cols, phi_l, phi_r, g_s, g_r


def test_four_sectors_reassemble_the_undecomposed_bubble():
    """B(G_S+G_R, G_S+G_R) == SS + SR + RS + RR, exactly.

    The bubble is bilinear, so this is an algebraic identity -- but it is
    exactly the identity that a dropped or double-counted sector violates, and
    the one that makes ``sectors="rr"``/``"rr_ss"`` staging settings rather
    than physics.
    """
    sizes, rows, cols, phi_l, phi_r, g_s, g_r = _bed()
    total = _bubble(g_s + g_r, g_s + g_r, phi_l, phi_r, rows, cols)
    sectors = {
        "ss": _bubble(g_s, g_s, phi_l, phi_r, rows, cols),
        "sr": _bubble(g_s, g_r, phi_l, phi_r, rows, cols),
        "rs": _bubble(g_r, g_s, phi_l, phi_r, rows, cols),
        "rr": _bubble(g_r, g_r, phi_l, phi_r, rows, cols),
    }
    rep = sector_sum_residual(total, sectors)
    assert rep["residual"] < 1e-12
    # Not vacuous: every sector carries real weight.
    for name in ("ss", "sr", "rs", "rr"):
        assert rep[f"weight_{name}"] > 0.05


@pytest.mark.parametrize("dropped", ["ss", "sr", "rs", "rr"])
def test_dropping_any_sector_is_visible(dropped):
    """Negative control: the gate must fail when a term is missing.

    Without this, a sector-sum test that passes proves nothing -- it could be
    comparing two copies of the same sum.
    """
    sizes, rows, cols, phi_l, phi_r, g_s, g_r = _bed()
    total = _bubble(g_s + g_r, g_s + g_r, phi_l, phi_r, rows, cols)
    sectors = {
        "ss": _bubble(g_s, g_s, phi_l, phi_r, rows, cols),
        "sr": _bubble(g_s, g_r, phi_l, phi_r, rows, cols),
        "rs": _bubble(g_r, g_s, phi_l, phi_r, rows, cols),
        "rr": _bubble(g_r, g_r, phi_l, phi_r, rows, cols),
    }
    del sectors[dropped]
    assert sector_sum_residual(total, sectors)["residual"] > 0.05


def test_sr_and_rs_are_not_each_others_transpose():
    """The mixed sectors are independent objects.

    Recorded error: ``Sigma_SR + Sigma_RS`` was once formed as ``out + out``.
    RS attaches the modal leg to the OTHER index pair, so it is neither a
    transpose nor a doubling of SR.
    """
    sizes, rows, cols, phi_l, phi_r, g_s, g_r = _bed(seed=3)
    sr = _bubble(g_s, g_r, phi_l, phi_r, rows, cols)
    rs = _bubble(g_r, g_s, phi_l, phi_r, rows, cols)
    t = transpose_index(rows, cols)
    assert np.abs(sr - rs).max() / np.abs(sr).max() > 0.1
    assert np.abs(sr - np.conj(rs[:, t])).max() / np.abs(sr).max() > 0.1


# --------------------------------------------------------------------------
# Gate 2: the Keldysh identity
# --------------------------------------------------------------------------

def test_transpose_index_is_an_involution_on_a_symmetric_pattern():
    rows, cols, _ = _pattern(np.array([3, 2, 3]))
    t = transpose_index(rows, cols)
    assert np.array_equal(rows[t], cols)
    assert np.array_equal(cols[t], rows)
    assert np.array_equal(t[t], np.arange(rows.size))


def test_transpose_index_refuses_an_asymmetric_pattern():
    rows = np.array([0, 0, 1])
    cols = np.array([0, 1, 1])          # (1,0) missing
    with pytest.raises(ValueError, match="structurally symmetric"):
        transpose_index(rows, cols)


def _sigma_triple(seed=0, ne=8, sizes=np.array([3, 3])):
    """A Sigma triple assembled exactly the way the production path does."""
    rows, cols, _ = _pattern(sizes)
    t = transpose_index(rows, cols)
    rng = np.random.default_rng(seed)
    n = rows.size
    # Sigma^{<,>} in the solver's occupation-positive convention: i*(PSD), so
    # each is i times a Hermitian matrix, i.e. anti-Hermitian on the pattern.
    def _anti():
        a = rng.normal(size=(ne, n)) + 1j * rng.normal(size=(ne, n))
        return a - np.conj(a[:, t])
    sl, sg = _anti(), _anti()
    # Hermitian Kramers-Kronig part, as "fft" produces.
    h = rng.normal(size=(ne, n)) + 1j * rng.normal(size=(ne, n))
    h = h + np.conj(h[:, t])
    sr = h + 0.5 * (sl - sg)
    return sr, sl, sg, rows, cols


def test_keldysh_identity_is_at_roundoff_for_a_correct_assembly():
    """eps_KI is purely algebraic and must sit at machine precision."""
    sr, sl, sg, rows, cols = _sigma_triple()
    rep = keldysh_identity(sr, sl, sg, rows, cols)
    assert rep["eps_ki"] < 1e-13
    assert rep["eps_delta_skew"] < 1e-13
    assert rep["eps_kk_hermitian"] < 1e-13


def test_keldysh_identity_catches_a_double_counted_retarded_half():
    """The recorded silent-wrong-answer: injecting more than the KK half.

    ``core/scba.py`` already adds ``0.5*(Sigma^< - Sigma^>)`` globally, so an
    injected analytic ``Sigma^R`` carrying the same half again breaks the
    identity. The error is one of MAGNITUDE, not of symmetry: the doubled
    skew part is still perfectly anti-Hermitian, so ``eps_delta_skew`` stays
    at roundoff and only ``eps_ki`` (and the recovered KK part, which absorbs
    the surplus) can see it. That asymmetry is the point of reporting three
    numbers.
    """
    sr, sl, sg, rows, cols = _sigma_triple()
    doubled = sr + 0.5 * (sl - sg)
    rep = keldysh_identity(doubled, sl, sg, rows, cols)
    assert rep["eps_ki"] > 0.5
    assert rep["eps_kk_hermitian"] > 0.5
    assert rep["eps_delta_skew"] < 1e-13


def test_keldysh_identity_catches_a_non_hermitian_retarded_part():
    sr, sl, sg, rows, cols = _sigma_triple(seed=1)
    rng = np.random.default_rng(9)
    bad = sr + 0.3 * np.abs(sr).max() * rng.normal(size=sr.shape)
    rep = keldysh_identity(bad, sl, sg, rows, cols)
    assert rep["eps_kk_hermitian"] > 1e-3
    assert rep["eps_ki"] > 1e-3


def test_keldysh_identity_catches_a_non_congruent_lesser():
    """An analytic ``Sigma^<`` that is not a congruence stops being
    anti-Hermitian, which only ``eps_delta_skew`` sees."""
    sr, sl, sg, rows, cols = _sigma_triple(seed=5)
    rng = np.random.default_rng(11)
    bad_l = sl + 0.3 * np.abs(sl).max() * rng.normal(size=sl.shape)
    rep = keldysh_identity(sr, bad_l, sg, rows, cols)
    assert rep["eps_delta_skew"] > 1e-3


def test_keldysh_identity_is_empty_safe():
    sr, sl, sg, rows, cols = _sigma_triple()
    z = np.zeros_like(sl)
    rep = keldysh_identity(np.zeros_like(sr), z, z, rows, cols)
    assert rep["eps_ki"] == 0.0


# --------------------------------------------------------------------------
# Gate 3: positivity
# --------------------------------------------------------------------------

def _congruence_lesser(seed=0, ne=6, sizes=np.array([3, 3])):
    """``G^< = -i M P M^dagger`` with P PSD: PSD by construction."""
    rows, cols, off = _pattern(sizes)
    rng = np.random.default_rng(seed)
    n = int(sizes.sum())
    m = rng.normal(size=(ne, n, n)) + 1j * rng.normal(size=(ne, n, n))
    a = rng.normal(size=(ne, n, n)) + 1j * rng.normal(size=(ne, n, n))
    p = a @ np.conj(np.swapaxes(a, -1, -2))                 # PSD
    dense = 1j * (m @ p @ np.conj(np.swapaxes(m, -1, -2)))  # -i G^< = M P M^H
    return dense[:, rows, cols], rows, cols, sizes, dense


def test_a_congruence_is_psd():
    vals, rows, cols, sizes, _ = _congruence_lesser()
    rep = psd_residual(vals, rows, cols, sizes, sign=-1.0)
    assert rep["worst"] > -1e-12


def test_both_keldysh_components_use_the_same_sign():
    """``sign=-1`` for lesser AND greater, in this solver's convention.

    ``sigma^{<,>} = +i n(+1) Gamma``, so ``-i sigma^<`` and ``-i sigma^>`` are
    both positive semidefinite. The textbook convention has ``+i G^> >= 0``,
    and using it here reports a uniformly negative spectrum (worst exactly
    -1.000) on data that is perfectly fine -- which is how this surfaced.
    """
    vals, rows, cols, sizes, _ = _congruence_lesser(seed=7)
    assert psd_residual(vals, rows, cols, sizes, sign=-1.0)["worst"] > -1e-12
    flipped = psd_residual(vals, rows, cols, sizes, sign=+1.0)
    assert flipped["worst"] < -0.99, (
        "the wrong sign must look obviously wrong, not marginally wrong")


def _full_pattern(sizes):
    """Every block, not just the tridiagonal ones -- a band mask needs
    something outside the band to remove."""
    n = int(sizes.sum())
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return rows.ravel(), cols.ravel()


def test_a_band_mask_breaks_positivity():
    """``bubble_positivity.md`` Thm 3: a boxcar band mask is indefinite.

    This is why the modal tail is carried as a congruence rather than a
    truncation, so the gate must be able to see the failure. Three structural
    facts set the test up.

    Zeroing every off-diagonal block leaves a BLOCK-DIAGONAL matrix whose
    blocks are principal submatrices of a PSD matrix, so that truncation is
    always PSD and proves nothing. The mask has to keep a band and drop what
    lies beyond it, which needs at least three blocks.

    A random PSD matrix often survives the mask by luck -- the theorem says
    positivity is not PRESERVED, not that it always fails. The counterexample
    is uniform long-range correlation (``P = v v^T + eps I`` with ``v``
    constant), which is exactly the regime the sector exists for: a device
    whose ``G`` has no block-distance decay.

    And the probe window must EXCEED the band width, not merely match it: to
    notice that block ``(0, 2)`` was removed, the window has to contain blocks
    0, 1 and 2 at once.
    """
    sizes = np.array([2, 2, 2])
    rows, cols = _full_pattern(sizes)
    n, ne = int(sizes.sum()), 3
    psd = np.ones((n, n)) + 1e-3 * np.eye(n)
    vals = np.broadcast_to(1j * psd, (ne, n, n))[:, rows, cols].copy()

    for w in (1, 2, 3):
        rep = psd_residual(vals, rows, cols, sizes, sign=-1.0, window=w)
        assert rep["worst"] > -1e-12, f"the unmasked congruence is PSD (w={w})"

    off = np.concatenate(([0], np.cumsum(sizes)))
    blk_r = np.searchsorted(off, rows, side="right") - 1
    blk_c = np.searchsorted(off, cols, side="right") - 1
    masked = vals.copy()
    masked[:, np.abs(blk_r - blk_c) > 1] = 0.0        # keep |I-J| <= 1

    seen = psd_residual(masked, rows, cols, sizes, sign=-1.0, window=3)
    assert seen["worst"] < -0.1, "window > band must see Thm 3's indefiniteness"

    for w in (1, 2):
        blind = psd_residual(masked, rows, cols, sizes, sign=-1.0, window=w)
        assert blind["worst"] > -1e-12, (
            f"window {w} cannot straddle a dropped block of a band-1 mask")


def test_window_one_is_blind_by_construction():
    """Stated as its own fact, because it is the gate's main limitation."""
    vals, rows, cols, sizes, _ = _congruence_lesser(seed=6)
    for w in (1, 2):
        rep = psd_residual(vals, rows, cols, sizes, sign=-1.0, window=w)
        assert rep["worst"] > -1e-12       # a true congruence passes at any w


def test_psd_normalisation_is_global_not_per_frequency():
    """A numerically empty frequency must not read as a violation.

    Recorded trap: per-omega normalisation once made a ballistic control
    "fail" purely on the empty tails of the window.
    """
    vals, rows, cols, sizes, _ = _congruence_lesser(seed=4)
    vals = vals.copy()
    vals[-1] *= 1e-18                       # an empty tail bin
    rep = psd_residual(vals, rows, cols, sizes, sign=-1.0)
    assert rep["worst"] > -1e-12


def test_psd_is_empty_safe():
    vals, rows, cols, sizes, _ = _congruence_lesser()
    rep = psd_residual(np.zeros_like(vals), rows, cols, sizes)
    assert rep["worst"] == 0.0 and rep["scale"] == 0.0


# --------------------------------------------------------------------------
# Gate 3, wired: the solver's production positivity check
# --------------------------------------------------------------------------

class _StubBuffer:
    def __init__(self, data, rows, cols):
        self.data, self.rows, self.cols = data, rows, cols


class _StubSolver:
    """Just enough of PhononSolver to exercise ``_check_positivity``."""

    from quatrex.phonon.solver import PhononSolver
    _check_positivity = PhononSolver._check_positivity

    def __init__(self, cfg, out, block_sizes, n_freq):
        import types
        self.config = types.SimpleNamespace(
            phonon=types.SimpleNamespace(pole_sector=cfg))
        self.block_sizes = block_sizes
        self.local_frequencies = np.zeros(n_freq)
        self.psd_report = {}
        self._psd_tol = 1e-10
        self._psd_sigma = None
        self._out = out


def _psd_bed(sizes=np.array([2, 2]), ne=3, seed=0):
    rows, cols, _ = _pattern(sizes)
    rng = np.random.default_rng(seed)
    n = int(sizes.sum())
    a = rng.normal(size=(ne, n, n)) + 1j * rng.normal(size=(ne, n, n))
    psd = a @ np.conj(np.swapaxes(a, -1, -2))
    # BOTH components are +i*(PSD) in this solver's convention, so both pass
    # the gate at sign = -1. Building greater as -i*PSD would encode the
    # textbook convention and hide the sign the gate actually uses.
    gl = (1j * psd)[:, rows, cols]
    gg = (1j * psd)[:, rows, cols]
    return (_StubBuffer(gl, rows, cols), _StubBuffer(gg, rows, cols)), sizes, ne


def test_positivity_gate_is_a_no_op_when_disabled():
    """Off by default, and off means it costs nothing and records nothing."""
    from quatrex.core.config import PoleSectorConfig

    out, sizes, ne = _psd_bed()
    s = _StubSolver(PoleSectorConfig(), out, sizes, ne)
    s._check_positivity(out)
    assert s.psd_report == {}

    s2 = _StubSolver(None, out, sizes, ne)
    s2._check_positivity(out)
    assert s2.psd_report == {}


def test_positivity_gate_reports_both_buffers_when_enabled():
    from quatrex.core.config import PoleSectorConfig

    out, sizes, ne = _psd_bed()
    cfg = PoleSectorConfig(enabled=True, psd_check=True)
    s = _StubSolver(cfg, out, sizes, ne)
    s._check_positivity(out)
    assert set(s.psd_report) == {"g_lesser", "g_greater"}   # Sigma off here
    # A congruence passes on BOTH sign conventions.
    for name in ("g_lesser", "g_greater"):
        assert s.psd_report[name]["worst"] > -1e-12


def test_positivity_gate_sees_an_indefinite_total():
    from quatrex.core.config import PoleSectorConfig

    out, sizes, ne = _psd_bed(seed=3)
    bad = out[0].data.copy()
    bad[:, 0] -= 50.0j * np.abs(bad).max()      # wreck one diagonal entry
    out = (_StubBuffer(bad, out[0].rows, out[0].cols), out[1])
    cfg = PoleSectorConfig(enabled=True, psd_check=True)
    s = _StubSolver(cfg, out, sizes, ne)
    s._check_positivity(out)
    assert s.psd_report["g_lesser"]["worst"] < -1e-6
