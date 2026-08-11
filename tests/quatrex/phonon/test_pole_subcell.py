# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The reconstructed hybrid function is physical only AT the grid points.

The sectors do not act on ``G``. They act on ``G~_h(w) = P(w) + R_k`` with
``R_k = G(w_k) - P(w_k)``: the analytic pole sum plus a frozen remainder. That
equals ``G`` exactly at each cell centre and nowhere else.

``R_k`` is a DIFFERENCE of PSD objects, hence generically indefinite. At the
centre ``P(w_k)`` cancels it; a little way off, ``P`` has decayed and the
frozen indefinite remainder dominates. Since ``SR``, ``RS`` and ``RR`` all
integrate over whole cells, the bubble of that function acquires gain --
anti-damping -- even though every stored sample is physical.

This module pins the mechanism, and pins that the congruence reconstruction
does not have it.
"""
import numpy as np
import pytest


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


N = 2
OMEGA, GAMMA = 9.0, 0.02


def _device():
    """A 2x2 device with one narrow resonance. ``G^< = G^R Sigma^< G^A`` is PSD
    by congruence, so every departure below is the RECONSTRUCTION's fault."""
    d = np.diag([OMEGA ** 2, (OMEGA + 6.0) ** 2])
    v = np.array([[1.0, 0.3], [0.3, 1.0]])
    d = v @ d @ v.T
    gam = np.array([[1.0, 0.2], [0.2, 0.8]]) * 2 * OMEGA * GAMMA

    def g_r(w):
        return np.linalg.inv((w ** 2) * np.eye(N) - d + 1j * gam)

    def sigma_l(w):
        return 1j * gam * (1.0 + 0.05 * w)          # i * PSD

    def g_l(w):
        g = g_r(w)
        return g @ sigma_l(w) @ g.conj().T

    ev, evec = np.linalg.eig(d)
    k = int(np.argmin(abs(np.sqrt(ev) - OMEGA)))
    return d, gam, g_r, sigma_l, g_l, evec[:, k:k + 1]


def _lam(m):
    herm = -1j * m
    herm = 0.5 * (herm + herm.conj().T)
    ev = np.linalg.eigvalsh(herm)
    return ev.min() / max(abs(ev).max(), 1e-300)


def test_true_lesser_is_psd_everywhere():
    """Guard the guard: the bed itself must be physical at every frequency."""
    _, _, _, _, g_l, _ = _device()
    for w in np.linspace(OMEGA - 0.5, OMEGA + 0.5, 41):
        assert _lam(g_l(w)) > 0, f"the bed must be PSD at w={w}"


def test_frozen_remainder_is_indefinite():
    """``R_k = G(w_k) - P(w_k)`` is a difference of PSD objects."""
    _, _, g_r, sigma_l, g_l, u = _device()
    z = OMEGA - 1j * GAMMA
    s_pair = u.conj().T @ sigma_l(OMEGA) @ u

    def pole(w):
        dr = u / (w - z)
        return dr @ s_pair @ dr.conj().T

    r_k = g_l(OMEGA) - pole(OMEGA)
    assert _lam(r_k) < -0.5, "the remainder is not positive semidefinite"


def test_reconstruction_is_physical_only_at_the_centre():
    """The headline: PSD at the centre, non-PSD a few percent of a cell away.

    Numbers are the measured ones, so a regression that merely softens the
    failure is still caught.
    """
    _, _, g_r, sigma_l, g_l, u = _device()
    z = OMEGA - 1j * GAMMA
    s_pair = u.conj().T @ sigma_l(OMEGA) @ u

    def pole(w):
        dr = u / (w - z)
        return dr @ s_pair @ dr.conj().T

    h = 0.25
    r_k = g_l(OMEGA) - pole(OMEGA)

    # At the centre the reconstruction IS G, to roundoff.
    assert np.abs((pole(OMEGA) + r_k) - g_l(OMEGA)).max() < 1e-12
    assert _lam(pole(OMEGA) + r_k) > 1e-2

    # Five percent of a cell away it has already collapsed.
    for frac in (0.05, 0.1, 0.5):
        w = OMEGA + frac * h
        assert _lam(pole(w) + r_k) < -0.5, (
            f"reconstruction must fail at offset {frac} of a cell")
        assert _lam(g_l(w)) > 1e-2, "while the true G stays healthy"


def test_congruence_reconstruction_stays_positive():
    """The fix: build ``G^R`` piecewise and form ``G^{<,>}`` by congruence.

    ``-i G^R S G^A = G^R (-i S) G^A`` is a congruence of a PSD matrix, so the
    sign survives at EVERY frequency. No approximation to ``G^R`` can break it,
    which is what makes this structural rather than incidental.
    """
    _, _, g_r, sigma_l, _, u = _device()
    z = OMEGA - 1j * GAMMA
    h = 0.25

    def pole_r(w):
        return (u @ u.conj().T) / (w - z) / (2 * OMEGA)

    r_k_ret = g_r(OMEGA) - pole_r(OMEGA)             # frozen RETARDED remainder
    source = sigma_l(OMEGA)                          # frozen source, as stored

    for frac in (0.0, 0.05, 0.1, 0.5):
        w = OMEGA + frac * h
        g_ret = pole_r(w) + r_k_ret
        assert _lam(g_ret @ source @ g_ret.conj().T) > 0.0, (
            f"the congruence must stay PSD at offset {frac}")


def test_subcell_metric_reports_the_failure():
    """``subcell_positivity`` must see what the bed above demonstrates."""
    from quatrex.phonon.pole_audit import subcell_positivity

    _, _, _, sigma_l, g_l, u = _device()
    z = OMEGA - 1j * GAMMA
    s_pair = u.conj().T @ sigma_l(OMEGA) @ u
    rows, cols = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    sizes = np.array([N])

    h = 0.25
    freqs = np.arange(0.0, 20.0, h)
    k = int(np.argmin(np.abs(freqs - OMEGA)))

    def pole_at(w):
        w = np.atleast_1d(_h(w))
        dr = u[None] / (w[:, None, None] - z)
        return np.einsum("wia,ab,wjb->wij", dr, s_pair,
                         np.conj(dr))[:, rows, cols]

    g_full = np.stack([g_l(x)[rows, cols] for x in freqs])
    g_pole = pole_at(freqs)

    rep = subcell_positivity(g_full, g_pole, pole_at, freqs, rows, cols,
                             sizes, centres=np.array([k]), window=1)
    assert rep["worst"] < -0.5, f"must see the failure, got {rep['worst']:.3e}"
    assert rep["worst_centre"] == k
    assert rep["at_centres"] > -1e-9, (
        "and must confirm the centres themselves are fine, so the failure is "
        "localised to the sub-cell reconstruction")


def test_report_subcell_runs_on_a_production_shaped_state():
    """Exercise the solver hook itself, not just the metric.

    The first device run of this diagnostic died with ``NameError: state`` --
    a parameter dropped during a refactor. Nothing local caught it because
    ``psd_check`` is off by default, so the whole function body was dead in
    every test. This calls it with the flag ON and a state shaped like the
    production one.
    """
    import types

    from quatrex.core.config import PoleSectorConfig
    from quatrex.phonon.pole_keldysh import PoleCluster
    from quatrex.phonon.pole_sector import PoleSectorState
    from quatrex.phonon.solver import PhononSolver

    rows, cols = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    freqs = np.linspace(0.0, 20.0, 81)

    class _Buf:
        def __init__(self, data):
            self.data, self.rows, self.cols = data, rows, cols

    rng = np.random.default_rng(0)
    a = rng.normal(size=(freqs.size, rows.size)) + 1j * rng.normal(
        size=(freqs.size, rows.size))

    st = PoleSectorState()
    u = rng.normal(size=(N, 1)) + 1j * rng.normal(size=(N, 1))
    st.clusters = [PoleCluster(z=np.array([OMEGA - 1j * GAMMA]), u=u, v=u)]
    st.legs = list(st.clusters)
    # (n_omega, Np, Np), as project_source_sparse produces in production.
    st.source_lesser = [np.full((freqs.size, 1, 1), 1.0 + 0.5j)]
    st.source_greater = [np.full((freqs.size, 1, 1), 1.0 - 0.5j)]
    st.g_pp_lesser = np.zeros((freqs.size, rows.size), dtype=complex)
    st.g_pp_greater = np.zeros((freqs.size, rows.size), dtype=complex)

    solver = object.__new__(PhononSolver)
    solver.config = types.SimpleNamespace(
        phonon=types.SimpleNamespace(
            pole_sector=PoleSectorConfig(enabled=True, psd_check=True)))
    solver.pole_state = st
    solver.local_frequencies = freqs
    solver.block_sizes = np.array([N])
    solver.psd_report = {}
    solver._psd_sigma_lesser = a          # Sigma^< on the pattern
    # out = (g_lesser, g_greater, g_retarded); the congruence route needs G^R.
    # leg="keldysh": the superseded reconstruction and its in-situ comparison
    solver.config.phonon.pole_sector.leg = "keldysh"
    solver._report_subcell((_Buf(a), _Buf(a), _Buf(a)))
    assert "subcell" in solver.psd_report
    assert "subcell_congruence" in solver.psd_report, (
        "the in-situ comparison against the congruence must also run")
    for key in ("worst", "worst_centre", "at_centres"):
        assert key in solver.psd_report["subcell"]

    # leg="congruence": the superseded metric is not reported, because it
    # would be measuring a function nothing consumes. What is reported is the
    # positivity of the leg the ring actually convolves.
    solver.psd_report = {}
    solver.config.phonon.pole_sector.leg = "congruence"
    solver._report_subcell((_Buf(a), _Buf(a), _Buf(a)))
    assert "ring_leg" in solver.psd_report
    assert "subcell" not in solver.psd_report
    # g_pp is zero here, so the leg IS G^{<,>} and the metric is finite
    assert np.isfinite(solver.psd_report["ring_leg"]["worst"])


# --------------------------------------------------------------------------
# The retarded split: G~^R = G^R_k + [P^R(w) - P^R(w_k)],  G~^< = G~^R S G~^A.
#
# Written that way the correction VANISHES at the cell centre, so the leg is
# the untouched grid sample there and the pole only supplies the sub-cell
# structure the grid cannot hold. Expanding the congruence gives the four
# sectors, and the point of the exercise is that they are not an approximation
# of it -- they ARE it.
# --------------------------------------------------------------------------


def _exact_pole():
    """Pole, residue and linewidth of the bed's resonance, in ``w`` (not w^2).

    ``g_r = inv(w^2 - (d - i gam))`` has its pole where ``w^2`` is an
    eigenvalue; near it ``w^2 - lam = 2 w0 (w - w0)``, so the residue is the
    spectral projector over ``2 w0``.
    """
    d, gam, _, _, _, _ = _device()
    lam, vec = np.linalg.eig(d - 1j * gam)
    k = int(np.argmin(np.abs(np.sqrt(lam).real - OMEGA)))
    w0 = np.sqrt(lam[k])
    w0 = w0 if w0.imag < 0 else -w0
    left = np.linalg.inv(vec).conj().T
    return w0, np.outer(vec[:, k], left[:, k].conj()) / (2.0 * w0), -w0.imag


def _cell(scale=1.0, width=20.0):
    """A cell holding the resonance well off its centre, plus the pieces the
    reconstruction needs. ``scale`` corrupts the residue deliberately."""
    _, _, g_r, sigma_l, g_l, _ = _device()
    w0, res, gw = _exact_pole()
    res = res * scale
    h = width * gw
    wk = w0.real + 0.31 * h
    s = sigma_l(wk)

    def pole(w):
        return res / (w - w0)

    def cong(w):
        g = g_r(wk) + (pole(w) - pole(wk))
        return g @ s @ g.conj().T

    def old(w):
        pl = lambda x: pole(x) @ s @ pole(x).conj().T  # noqa: E731
        return pl(w) + (g_l(wk) - pl(wk))

    return w0, res, h, wk, s, g_r, g_l, pole, cong, old


def test_four_sectors_are_the_congruence():
    """The sector split is an identity, not a model.

    ``G~^R S G~^A`` with ``G~^R = G^R_k + U dD V^H`` expands into

        RR = G^R_k S G^A_k              (the untouched ring at the grid point)
        SR = U dD [V^H S G^A_k]
        RS = [G^R_k S V] dD^H U^H
        SS = U dD [V^H S V] dD^H U^H

    and the four sum to it exactly. ``dD = D(w) - D(w_k)`` vanishes at the
    centre, so ``RR`` alone is the leg there and the rest is pure sub-cell
    structure.
    """
    w0, res, h, wk, s, g_r, _, _, cong, _ = _cell()
    gk = g_r(wk)
    for f in (0.0, 0.05, 0.2, -0.31, 0.5):
        w = wk + f * h
        dd = 1.0 / (w - w0) - 1.0 / (wk - w0)
        rr = gk @ s @ gk.conj().T
        sr = (res * dd) @ (s @ gk.conj().T)
        rs = (gk @ s) @ (res * dd).conj().T
        ss = (res * dd) @ s @ (res * dd).conj().T
        assert np.abs(cong(w) - (rr + sr + rs + ss)).max() < 1e-14
        if f == 0.0:
            assert np.abs(sr).max() == np.abs(ss).max() == 0.0


def test_congruence_survives_a_wrong_residue_and_the_old_form_does_not():
    """Why the redesign is necessary rather than merely tidier.

    On the device the residue is a Newton solution of a truncated cluster
    against a fitted source; it is never exact. The old reconstruction needs
    it accurate to better than ~20 percent JUST TO KEEP THE SIGN, and nothing
    enforced that. The congruence has no such requirement: it is a congruence
    of a PSD source, so ``-i G~^< >= 0`` for ANY pole model, right or wrong.
    """
    offs = np.linspace(-0.5, 0.5, 41)
    worst = {}
    for tag, scale in (("exact", 1.0), ("over", 1.2), ("double", 2.0),
                       ("phase", 1.0 + 0.4j)):
        _, _, h, wk, _, _, _, _, cong, old = _cell(scale=scale)
        worst[tag] = (min(_lam(old(wk + f * h)) for f in offs),
                      min(_lam(cong(wk + f * h)) for f in offs))

    # the congruence is positive in every case, corrupted model or not
    for tag, (_, c) in worst.items():
        assert c > 0.0, f"congruence went negative for {tag}: {c:.3e}"

    # ... and a 20% residue error is already enough to invert the old form
    assert worst["exact"][0] > 0.0
    for tag in ("over", "double", "phase"):
        assert worst[tag][0] < -0.5, f"{tag}: {worst[tag][0]:.3e}"


def test_reconstruction_recovers_the_cell_average():
    """What the bubble actually consumes is the cell INTEGRAL, and the raw
    grid sample of an under-resolved pole is wrong by order one."""
    _, _, h, wk, _, _, g_l, _, cong, _ = _cell(width=20.0)
    x, wt = np.polynomial.legendre.leggauss(400)
    xs, wts = wk + 0.5 * h * x, 0.5 * wt

    def avg(fn):
        return sum(wi * fn(xx) for xx, wi in zip(xs, wts))

    exact = avg(g_l)
    n = np.abs(exact).max()
    assert np.abs(avg(cong) - exact).max() / n < 1e-2     # measured 4.1e-3
    assert np.abs(g_l(wk) - exact).max() / n > 0.5        # measured 8.2e-1
