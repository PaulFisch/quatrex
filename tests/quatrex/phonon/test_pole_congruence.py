"""Congruence reconstruction of the off-grid Keldysh function.

The pole sector stores one remainder per cell and treats it as piecewise
constant. Which remainder it stores decides whether the reconstruction stays
physical between grid points -- see `phonon/docs/pole_scba_divergence.md` Sec.
7, where freezing the KELDYSH remainder is measured at ``eps_PSD = -5.4e-02``
while the cell centres are all positive.
"""

import numpy as np
import pytest

from quatrex.phonon.pole_keldysh import (
    PoleCluster,
    hybrid_keldysh_congruence,
    pole_keldysh,
    pole_retarded,
)


def _bed(gamma: float, n_dof: int = 3, seed: int = 1):
    """One narrow pole plus wide background; ``G^<`` PSD by construction."""
    rng = np.random.default_rng(seed)

    def vec():
        return rng.normal(size=(n_dof,)) + 1j * rng.normal(size=(n_dof,))

    z_n = np.array([4.0 - 1j * gamma])
    u_n = vec().reshape(n_dof, 1)
    cl = PoleCluster(z=z_n, u=u_n, v=u_n)
    z_bg = np.array([2.5 - 0.9j, 7.5 - 1.2j])
    u_bg = np.stack([vec(), vec()], axis=1)
    a = rng.normal(size=(n_dof, n_dof)) + 1j * rng.normal(size=(n_dof, n_dof))
    # The PSD object is -i Sigma^<, hence the explicit i.
    sigma = 1j * (a @ np.conj(a).T)

    def g_ret(w):
        w = np.atleast_1d(np.asarray(w, float))
        d = 1.0 / (w.reshape(-1, 1) - z_bg.reshape(1, -1))
        return (pole_retarded(w, cl)
                + np.einsum("ia,wa,ja->wij", u_bg, d, np.conj(u_bg)))

    def g_les(w):
        gr = g_ret(w)
        return gr @ sigma @ np.conj(gr).swapaxes(-2, -1)

    return cl, g_ret, g_les, sigma


def _herm_eigs(mats):
    """Eigenvalues of the Hermitian part of ``-i M``."""
    h = -1j * np.asarray(mats)
    return np.linalg.eigvalsh(0.5 * (h + np.conj(h).swapaxes(-2, -1)))


def test_congruence_matches_the_direct_split_at_cell_centres():
    """Both reconstructions are exact where the remainder is defined."""
    cl, g_ret, g_les, sigma = _bed(gamma=0.01)
    centres = np.array([3.75, 4.0, 4.25])
    r_ret = g_ret(centres) - pole_retarded(centres, cl)
    src = np.repeat(sigma[None], centres.size, axis=0)
    got = hybrid_keldysh_congruence(centres, cl, r_ret, src,
                                    np.arange(centres.size))
    exact = g_les(centres)
    err = np.abs(got - exact).max() / np.abs(exact).max()
    assert err < 1e-10, f"congruence must be exact at centres: {err:.3e}"


@pytest.mark.parametrize("gamma", [0.1, 0.02, 0.005])
def test_congruence_stays_psd_off_grid_where_direct_subtraction_fails(gamma):
    """The point of the construction: positivity is structural off-grid.

    ``-i G^< = G^R (-i Sigma^<) (G^R)^H`` holds for any ``G^R``, so the
    reconstruction cannot go indefinite however coarse the cell. Freezing the
    Keldysh remainder instead has no such guarantee, and must be shown failing
    here or the test proves nothing.
    """
    h = 0.25
    cl, g_ret, g_les, sigma = _bed(gamma=gamma)
    centres = np.array([4.0])
    x = np.polynomial.legendre.leggauss(24)[0] * 0.5
    w = centres[0] + h * x
    idx = np.zeros(w.size, dtype=int)

    r_ret = g_ret(centres) - pole_retarded(centres, cl)
    src = np.repeat(sigma[None], 1, axis=0)
    cong = hybrid_keldysh_congruence(w, cl, r_ret, src, idx)

    r_les = g_les(centres) - pole_keldysh(centres, cl,
                                          np.conj(cl.v).T @ sigma @ cl.v)
    direct = pole_keldysh(w, cl, np.conj(cl.v).T @ sigma @ cl.v) + r_les[0]

    # One global scale for both, per pole_audit.psd_residual's convention.
    scale = float(max(_herm_eigs(cong).max(), _herm_eigs(direct).max()))
    psd_cong = _herm_eigs(cong).min() / scale
    psd_direct = _herm_eigs(direct).min() / scale
    assert psd_cong > -1e-12, f"congruence lost positivity: {psd_cong:.3e}"
    assert psd_direct < -1e-6, (
        "the direct split must FAIL here, else the bed is not exercising "
        f"the failure mode: {psd_direct:.3e}")
