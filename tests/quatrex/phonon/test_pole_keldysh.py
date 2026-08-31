# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Doc Sec. 11-16 and Sec. 53 Experiment B: the pole-cluster Keldysh matrix.

Bed: ``M(z) = z^2 I + i g z I - D`` with ``D`` real symmetric. Then
``Sigma^R = -i g z``, so ``Gamma = i(Sigma^R - Sigma^A) = 2 g omega`` -- the
physical linear-opening form of a harmonic lead -- and the contact source is
``Sigma^< = i n(omega) Gamma(omega)``, which satisfies the solver's
occupation-positive convention ``-i Sigma^< >= 0``.

Every pole quantity is known in closed form on this bed (see
``test_pole_nevp.py``): for each eigenpair ``(lam, v)`` of ``D``,
``z = (-i g + sqrt(4 lam - g^2))/2``, ``r = v``, ``l = v/(2 Omega)``.
"""
import numpy as np
import pytest

from quatrex.phonon.experimental.pole.pole_keldysh import (
    PoleCluster,
    coherence_metric,
    occupation_matrix,
    pole_keldysh,
    pole_retarded,
    project_source,
    source_poly_eval,
    source_poly_fit,
)

G_DAMP = 0.25
TEMP = 300.0
HBAR_EVS, KB_EV, THZ = 6.582119569e-16, 8.617333262e-5, 2.0 * np.pi * 1e12


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _bose(w):
    x = HBAR_EVS * THZ * np.abs(w) / (KB_EV * TEMP)
    return 1.0 / np.expm1(np.clip(x, 1e-12, None))


def _cluster_from_d(d_dense, g=G_DAMP, pick=None):
    """Exact poles / vectors of ``z^2 + i g z - lam`` for every mode of ``D``."""
    lam, vec = np.linalg.eigh(d_dense)
    z = (-1j * g + np.sqrt(np.asarray(4 * lam - g * g, dtype=complex))) / 2.0
    if pick is not None:
        z, vec = z[pick], vec[:, pick]
    omega = z.real
    u = vec.astype(complex)
    l = u / (2.0 * omega)          # normalised so l^H M'(z) r = 1
    return PoleCluster(z=z, u=u, v=l)


def _sigma_lesser(omega, n_dof, g=G_DAMP, scatter=0.6, seed=99):
    """Total Keldysh source ``Sigma^< = i n(w) [Gamma_contact +
    Gamma_scatter]``."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_dof, n_dof))
    w_psd = a @ a.T
    w_psd = w_psd / np.linalg.norm(w_psd, 2)
    gam_c = 2.0 * g * np.abs(omega)
    gam_s = scatter * np.abs(omega) * np.exp(-((omega / 40.0) ** 2))
    gam = (gam_c[:, None, None] * np.eye(n_dof)[None]
           + gam_s[:, None, None] * w_psd[None])
    return 1j * _bose(omega)[:, None, None] * gam


def _direct_keldysh(omega, d_dense, sig_less, g=G_DAMP):
    """``G^< = G^R Sigma^< G^A`` computed densely."""
    n = d_dense.shape[0]
    out = np.empty((len(omega), n, n), dtype=complex)
    for k, w in enumerate(omega):
        gr = np.linalg.inv((w * w + 1j * g * w) * np.eye(n) - d_dense)
        out[k] = gr @ sig_less[k] @ gr.conj().T
    return out


def _two_mode_d(delta, base=64.0, n=2):
    """``D`` with two eigenvalues split by a controlled fraction ``delta``."""
    lam = np.array([base, base * (1.0 + delta)])
    q = np.linalg.qr(np.array([[1.0, 0.3], [-0.2, 1.0]]))[0]
    return q @ np.diag(lam) @ q.T


# --------------------------------------------------------------------------- #
# The pole sector reproduces the exact Keldysh function in its own window.
# --------------------------------------------------------------------------- #

def test_pole_keldysh_dominates_the_direct_function_at_resonance():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(5, 5))
    d = m + m.T + np.diag(20.0 * 1.6 ** np.arange(5))
    cl = _cluster_from_d(d)

    k = 3
    w0, gam = float(_h(cl.omega)[k]), float(_h(cl.gamma)[k])
    audit = np.linspace(w0 - 8 * gam, w0 + 8 * gam, 161)

    sig = _sigma_lesser(audit, 5)
    direct = _direct_keldysh(audit, d, sig)
    s = _h(project_source(sig, _h(cl.v)))
    g_pp = _h(pole_keldysh(audit, cl, s))

    rel = np.abs(g_pp - direct).max() / np.abs(direct).max()
    assert rel < 0.05, f"pole sector misses the resonance by {rel:.3e}"


def test_pole_retarded_matches_the_direct_resolvent_at_resonance():
    rng = np.random.default_rng(1)
    m = rng.normal(size=(4, 4))
    d = m + m.T + np.diag(20.0 * 1.7 ** np.arange(4))
    cl = _cluster_from_d(d)
    k = 2
    w0, gam = float(_h(cl.omega)[k]), float(_h(cl.gamma)[k])
    audit = np.linspace(w0 - 5 * gam, w0 + 5 * gam, 81)

    pr = _h(pole_retarded(audit, cl))
    direct = np.array([
        np.linalg.inv((w * w + 1j * G_DAMP * w) * np.eye(4) - d) for w in audit
    ])
    rel = np.abs(pr - direct).max() / np.abs(direct).max()
    assert rel < 0.02, f"P^R misses G^R by {rel:.3e}"


# --------------------------------------------------------------------------- #
# Positivity: the congruence is what protects it.
# --------------------------------------------------------------------------- #

def test_pole_keldysh_inherits_source_positivity():
    """``-i Sigma^< >= 0`` implies ``-i G_PP^< >= 0`` -- exactly, by congruence."""
    rng = np.random.default_rng(2)
    m = rng.normal(size=(4, 4))
    d = m + m.T + np.diag(25.0 * 1.5 ** np.arange(4))
    cl = _cluster_from_d(d)
    audit = np.linspace(4.0, 12.0, 97)

    sig = _sigma_lesser(audit, 4)
    assert np.linalg.eigvalsh(-1j * sig).min() > -1e-12, "bed source is not PSD"

    s = _h(project_source(sig, _h(cl.v)))
    g_pp = _h(pole_keldysh(audit, cl, s))
    herm = -1j * g_pp
    herm = 0.5 * (herm + herm.conj().swapaxes(-2, -1))
    worst = float(np.linalg.eigvalsh(herm).min())
    scale = float(np.abs(herm).max())
    assert worst > -1e-10 * scale, f"worst eigenvalue {worst:.3e} vs scale {scale:.3e}"


# --------------------------------------------------------------------------- #
# Experiment B: where scalar occupations fail and the cluster matrix does not.
# --------------------------------------------------------------------------- #

def _scalar_vs_matrix(delta):
    d = _two_mode_d(delta)
    cl = _cluster_from_d(d)
    # Centre the window on ONE pole: the question is whether its partner's
    # coherence still matters there, not what happens midway between them.
    w0 = float(_h(cl.omega)[0])
    gam = float(_h(cl.gamma)[0])
    audit = np.linspace(w0 - 10 * gam, w0 + 10 * gam, 201)

    sig = _sigma_lesser(audit, 2)
    s_full = _h(project_source(sig, _h(cl.v)))
    s_diag = s_full * np.eye(2)[None]        # independent scalar occupations

    g_full = _h(pole_keldysh(audit, cl, s_full))
    g_diag = _h(pole_keldysh(audit, cl, s_diag))
    eps = np.abs(g_full - g_diag).max() / np.abs(g_full).max()
    return float(_h(cl.isolation())[0]), eps


def test_scalar_occupations_fail_for_overlapping_resonances():
    """The doc's central Keldysh claim, as a monotone sweep in the isolation ratio."""
    rows = [_scalar_vs_matrix(dl) for dl in (2e-4, 6e-3, 2e-1, 2.0)]
    etas = [r[0] for r in rows]
    errs = [r[1] for r in rows]

    assert etas == sorted(etas), f"isolation not monotone in the sweep: {etas}"
    assert errs[0] > 0.1, (
        f"at eta = {etas[0]:.2f} the scalar ansatz should be badly wrong, "
        f"got {errs[0]:.3e}"
    )
    assert errs[-1] < 1e-2, (
        f"at eta = {etas[-1]:.2f} the scalar ansatz should be fine, "
        f"got {errs[-1]:.3e}"
    )
    assert errs[0] > 30 * errs[-1], f"no separation across the sweep: {errs}"


def test_coherence_metric_tracks_the_scalar_error():
    """eps_coh is what licenses the scalar reduction, so it must move with it."""
    for delta, expect_coherent in ((2e-4, True), (2.0, False)):
        d = _two_mode_d(delta)
        cl = _cluster_from_d(d)
        w0, gam = float(_h(cl.omega)[0]), float(_h(cl.gamma)[0])
        audit = np.linspace(w0 - 10 * gam, w0 + 10 * gam, 201)
        s = _h(project_source(_sigma_lesser(audit, 2), _h(cl.v)))
        n = occupation_matrix(audit, cl, s)
        eps = coherence_metric(n)
        if expect_coherent:
            assert eps > 0.1, f"overlapping cluster reported eps_coh = {eps:.3e}"
        else:
            assert eps < 0.1, f"separated cluster reported eps_coh = {eps:.3e}"


# --------------------------------------------------------------------------- #
# Bookkeeping.
# --------------------------------------------------------------------------- #

def test_isolation_matches_the_definition():
    d = _two_mode_d(1e-2)
    cl = _cluster_from_d(d)
    z, g = _h(cl.z), _h(cl.gamma)
    want = abs(z[0] - z[1]) / (g[0] + g[1])
    assert np.allclose(_h(cl.isolation()), want)


def test_residues_match_the_outer_product():
    d = _two_mode_d(1e-2)
    cl = _cluster_from_d(d)
    res = _h(cl.residues())
    for a in range(cl.n_poles):
        want = np.outer(_h(cl.u)[:, a], np.conj(_h(cl.v)[:, a]))
        assert np.allclose(res[a], want)


def test_source_poly_model_round_trips():
    rng = np.random.default_rng(4)
    w = np.linspace(9.0, 11.0, 41)
    base = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    s = np.einsum("w,ij->wij", 1.0 + 0.1 * (w - 10.0) ** 2, base)
    coeff, resid = source_poly_fit(w, s, centre=10.0, scale=0.05, order=2)
    assert resid < 1e-12, f"quadratic source not fitted exactly: {resid:.3e}"
    back = _h(source_poly_eval(w, coeff, centre=10.0, scale=0.05))
    assert np.abs(back - s).max() / np.abs(s).max() < 1e-10


def test_source_poly_residual_flags_an_unfittable_source():
    """The fit residual is the gate that demotes a cluster instead of guessing."""
    w = np.linspace(9.0, 11.0, 81)
    sharp = 1.0 / ((w - 10.0) ** 2 + 1e-4)
    s = np.einsum("w,ij->wij", sharp, np.eye(2, dtype=complex))
    _, resid = source_poly_fit(w, s, centre=10.0, scale=0.05, order=2)
    assert resid > 0.1, f"a near-singular source was reported as fittable ({resid:.3e})"


def test_cluster_rejects_an_upper_half_plane_pole():
    with pytest.raises(ValueError, match="lower half plane"):
        PoleCluster(z=np.array([1.0 + 0.1j]), u=np.ones((3, 1)), v=np.ones((3, 1)))


def test_cluster_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        PoleCluster(z=np.array([1.0 - 0.1j]), u=np.ones((3, 1)), v=np.ones((4, 1)))
