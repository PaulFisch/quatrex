# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Doc Phase 1: extract poles of G^R from a GRID-SAMPLED scattering self-energy.

This is where the three Phase-0/1 modules meet: :mod:`quatrex.phonon.experimental.pole.pole_kernel`
continues ``Sigma_s^R`` off the real axis from its grid samples,
:mod:`quatrex.phonon.experimental.pole.btd_linalg` factorises ``M(z)``, and
:mod:`quatrex.phonon.experimental.pole.pole_nevp` solves ``M(z) r = 0``.

The mechanism these tests are meant to expose:

    ``Sigma^R`` is SMOOTH -- it is a convolution, so its structure is set by the
    two-phonon joint spectrum, not by any single linewidth. The sharp structure
    lives in ``G = [z^2 - D - Sigma^R]^{-1}``, which is a *derived* object. So a
    grid that is far too coarse to resolve the resonance is still perfectly
    adequate to SAMPLE the self-energy the pole solve needs.

That is the whole reason pole subtraction can pay: the information the method
needs is not the information the grid is failing to carry.

Bed: ``D`` block-tridiagonal real symmetric; ``Delta = Sigma^> - Sigma^< = -i a(w) I``
with ``a >= 0`` a smooth, compactly supported damping profile (so ``Gamma = i Delta``
is positive semidefinite, the production sign convention).
"""
import numpy as np
import pytest

from quatrex.core.fft_utils import hilbert_transform
from quatrex.phonon.experimental.pole.btd_linalg import BTDFactorization
from quatrex.phonon.experimental.pole.pole_kernel import sigma_retarded_at_z
from quatrex.phonon.experimental.pole.pole_nevp import bordered_newton

TINY = 1e-30
FMAX = 30.0
DAMP, W_C = 0.012, 11.0


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _dynamical(sizes=(3, 3, 3), seed=0):
    """Real symmetric BTD ``D`` with a well-separated mode ladder."""
    rng = np.random.default_rng(seed)
    total = sum(sizes)
    ladder = 20.0 * 1.55 ** np.arange(total)
    d_ii, d_ij, k = [], [], 0
    for n in sizes:
        m = 0.3 * rng.normal(size=(n, n))
        d_ii.append(m + m.T + np.diag(ladder[k:k + n]))
        k += n
    for i in range(len(sizes) - 1):
        d_ij.append(0.3 * rng.normal(size=(sizes[i], sizes[i + 1])))
    return d_ii, d_ij, [b.T for b in d_ij]


def _damping(freqs):
    """``a(w) >= 0``: odd-in-w linear opening with a smooth roll-off."""
    return DAMP * freqs * np.exp(-((freqs / W_C) ** 2))


def _delta(freqs):
    """``Delta = -i a(w) I``, so ``Gamma = i Delta = a(w) I >= 0``."""
    return -1j * _damping(freqs)


def _bed(nf, sizes=(3, 3, 3), seed=0):
    freqs = np.linspace(0.0, FMAX, nf)
    d = _dynamical(sizes, seed)
    n_dof = sum(sizes)
    delta = np.einsum("w,ij->wij", _delta(freqs), np.eye(n_dof))
    return freqs, d, delta


def _operator(d, freqs, delta):
    """``M(z) = z^2 I - D - Sigma_s^{R,II}(z)`` and its exact derivative."""
    d_ii, d_ij, d_ji = d
    sizes = [b.shape[-1] for b in d_ii]
    off = np.concatenate(([0], np.cumsum(sizes)))

    def _sig(z, order):
        return _h(
            sigma_retarded_at_z(
                delta, freqs, np.array([z]), sheet="II", order=order,
                delta_order=3, delta_window=4,
            )
        )[0]

    def m_blocks(z):
        sig = _sig(z, 0)
        a_ii = [
            z * z * np.eye(sizes[i]) - d_ii[i] - sig[off[i]:off[i + 1], off[i]:off[i + 1]]
            for i in range(len(sizes))
        ]
        return a_ii, [-b + 0j for b in d_ij], [-b + 0j for b in d_ji]

    def dm_blocks(z):
        dsig = _sig(z, 1)
        a_ii = [
            2.0 * z * np.eye(sizes[i]) - dsig[off[i]:off[i + 1], off[i]:off[i + 1]]
            for i in range(len(sizes))
        ]
        return (a_ii,
                [np.zeros_like(b) + 0j for b in d_ij],
                [np.zeros_like(b) + 0j for b in d_ji])

    return m_blocks, dm_blocks


def _direct_gr(d, freqs, delta, omega):
    """``G^R(omega) = M(omega)^{-1}`` from the production real-axis Sigma^R."""
    d_ii, d_ij, d_ji = d
    sizes = [b.shape[-1] for b in d_ii]
    off = np.concatenate(([0], np.cumsum(sizes)))
    n = int(off[-1])
    # Production Sigma^R, evaluated on the SAME grid, then sampled at omega.
    sig_grid = 0.5 * delta + 0.5j * hilbert_transform(delta, freqs)
    sig = np.empty((len(omega), n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            sig[:, i, j] = np.interp(omega, freqs, _h(sig_grid)[:, i, j].real) \
                + 1j * np.interp(omega, freqs, _h(sig_grid)[:, i, j].imag)
    out = np.empty((len(omega), n, n), dtype=complex)
    for k, w in enumerate(omega):
        a_ii = [
            w * w * np.eye(sizes[i]) - d_ii[i] - sig[k, off[i]:off[i + 1], off[i]:off[i + 1]]
            for i in range(len(sizes))
        ]
        fac = BTDFactorization.factorize(a_ii, [-b + 0j for b in d_ij],
                                         [-b + 0j for b in d_ji])
        out[k] = _h(fac.solve(np.eye(n, dtype=complex)))
    return out


def _find_pole(nf, k_mode, seed=0):
    freqs, d, delta = _bed(nf, seed=seed)
    m_blocks, dm_blocks = _operator(d, freqs, delta)
    lam = np.linalg.eigvalsh(_dense_d(d))
    guess = complex(np.sqrt(lam[k_mode]), -0.01)
    return bordered_newton(m_blocks, dm_blocks, guess, max_iter=40, tol=1e-8), \
        freqs, d, delta


def _dense_d(d):
    d_ii, d_ij, d_ji = d
    sizes = [b.shape[-1] for b in d_ii]
    off = np.concatenate(([0], np.cumsum(sizes)))
    out = np.zeros((off[-1], off[-1]))
    for i in range(len(sizes)):
        out[off[i]:off[i + 1], off[i]:off[i + 1]] = d_ii[i]
        if i + 1 < len(sizes):
            out[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = d_ij[i]
            out[off[i + 1]:off[i + 2], off[i]:off[i + 1]] = d_ji[i]
    return out


# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("k_mode", [1, 4])
def test_pole_is_found_and_physical(k_mode):
    """A converged pole sits in the lower half plane with a positive linewidth."""
    sol, freqs, _, _ = _find_pole(401, k_mode)
    assert sol.converged, f"eps_nep = {sol.eps_nep:.3e}"
    assert sol.z.imag < 0.0, "a retarded pole must have negative imaginary part"
    gamma = -sol.z.imag
    h = float(freqs[1] - freqs[0])
    # The regime the pole sector exists for: narrower than the grid can see.
    assert gamma < 0.5 * h, f"gamma/h = {gamma / h:.3f} is not sub-grid"


def test_pole_is_stable_under_grid_refinement():
    """The pole depends on the grid only through the SAMPLING of a smooth Sigma^R.

    This is the mechanism claim: refining the grid by 4x must move the pole by
    far less than the linewidth, even though no grid here resolves the line.
    """
    zs, hs = [], []
    for nf in (201, 401, 801):
        sol, freqs, _, _ = _find_pole(nf, 4)
        assert sol.converged
        zs.append(sol.z)
        hs.append(float(freqs[1] - freqs[0]))
    gamma = -zs[-1].imag
    for z, h in zip(zs, hs):
        assert gamma < 0.5 * h, f"gamma/h = {gamma / h:.3f}: line is grid-resolved"
    drift = abs(zs[-1] - zs[0])
    assert drift < 0.05 * gamma, (
        f"pole moved {drift:.3e} across a 4x refinement, vs gamma {gamma:.3e}"
    )


def test_pole_subtraction_flattens_the_retarded_green_function():
    """Removing R/(w - z) must leave a background that is orders of magnitude
    flatter."""
    sol, freqs, d, delta = _find_pole(401, 4)
    assert sol.converged
    omega0, gamma = sol.z.real, -sol.z.imag

    # A dense audit window around the resonance -- the grid never sees this.
    audit = np.linspace(omega0 - 30 * gamma, omega0 + 30 * gamma, 241)
    g_direct = _direct_gr(d, freqs, delta, audit)

    res = _h(sol.residue())
    # The pole and its bosonic partner at -conj(z).
    g_pole = (
        res[None] / (audit[:, None, None] - sol.z)
        + np.conj(res)[None] / (audit[:, None, None] + np.conj(sol.z))
    )
    g_bg = g_direct - g_pole

    def swing(g):
        m = np.abs(g).max(axis=(1, 2))
        return m.max() / np.median(m)

    # The decisive measure is the AMPLITUDE the subtraction removes: the pole
    # term carries essentially all of |G^R| in its own window.
    amp_direct = float(np.abs(g_direct).max())
    amp_bg = float(np.abs(g_bg).max())
    assert amp_direct > 10.0, f"the bed has no sharp feature to remove ({amp_direct:.3g})"
    assert amp_bg < 0.01 * amp_direct, (
        f"background peak {amp_bg:.3g} vs direct {amp_direct:.3g}: the "
        "subtraction did not remove the resonance"
    )
    # ... and the remainder is genuinely flatter across the window, which is
    # what lets a coarse grid represent it.
    s_direct, s_bg = swing(g_direct), swing(g_bg)
    assert s_bg < 0.25 * s_direct, (
        f"background swing {s_bg:.2f} vs direct {s_direct:.2f}"
    )


def test_residue_reproduces_the_peak_height():
    """At the peak, G is dominated by its pole term; the residue must carry it."""
    sol, freqs, d, delta = _find_pole(401, 4)
    omega0, gamma = sol.z.real, -sol.z.imag
    g_direct = _direct_gr(d, freqs, delta, np.array([omega0]))[0]
    res = _h(sol.residue())
    g_pole = res / (omega0 - sol.z) + np.conj(res) / (omega0 + np.conj(sol.z))
    rel = np.abs(g_pole - g_direct).max() / np.abs(g_direct).max()
    assert rel < 0.05, f"peak reproduced to {rel:.3e}"


def test_registration_lottery_moves_the_grid_answer_but_not_the_pole():
    """The motivating asymmetry, stated as a test."""
    sampled, poles, hs = [], [], []
    for nf in (401, 403, 405, 407, 409):
        sol, freqs, d, delta = _find_pole(nf, 4)
        assert sol.converged
        poles.append(sol.z)
        hs.append(float(freqs[1] - freqs[0]))
        # What the grid itself can report: |G| at its nearest node to the peak.
        node = freqs[int(np.argmin(np.abs(freqs - sol.z.real)))]
        sampled.append(float(np.abs(_direct_gr(d, freqs, delta, np.array([node]))).max()))

    gamma = -np.mean([z.imag for z in poles])
    assert gamma < 0.1 * min(hs), "bed is not in the sub-grid regime"

    grid_swing = max(sampled) / min(sampled)
    pole_spread = max(abs(z - poles[0]) for z in poles)
    assert grid_swing > 3.0, (
        f"the grid answer barely moved ({grid_swing:.2f}); no lottery to show"
    )
    assert pole_spread < 0.05 * gamma, (
        f"pole moved {pole_spread:.3e} across the offsets, vs gamma {gamma:.3e}"
    )


def test_pole_term_accounts_for_the_true_peak():
    """At the resonance the pole term carries essentially all of |G^R|."""
    sol, freqs, d, delta = _find_pole(401, 4)
    omega0 = sol.z.real
    at_peak = float(np.abs(_direct_gr(d, freqs, delta, np.array([omega0]))).max())
    res = _h(sol.residue())
    g_pole = float(np.abs(res / (omega0 - sol.z)).max())
    assert g_pole > 0.9 * at_peak, (
        f"pole term {g_pole:.3g} does not account for the true peak {at_peak:.3g}"
    )
