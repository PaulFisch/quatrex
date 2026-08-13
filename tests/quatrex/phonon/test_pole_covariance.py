# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The subcell covariance: what the cell-averaged ring leaves behind.

The claim under test is an identity, not an approximation:

    int_{I_k} du/2pi B[G_k(u), G_l(Omega-u)]
        = (h/2pi) B(Gbar_k, Gbar_l)  +  Delta I_kl

with the cross terms vanishing exactly because both fluctuations have zero cell
mean and the reflection maps one cell onto the other. So every test here is
against a direct quadrature of the same integrand, at roundoff, not against a
tolerance chosen to pass.
"""
import numpy as np
import pytest

from quatrex.phonon.pole_covariance import (
    cell_resolvent_mean,
    cell_variance,
    centred_gram,
    covariance_kernel,
    pair_covariance,
)

H = 0.5


def _leg(zeta, res, centre, h=H):
    """The local model as a callable, and its exact cell mean."""
    d = cell_resolvent_mean(np.asarray(zeta), centre, h)

    def g(u):
        u = np.atleast_1d(np.asarray(u, dtype=complex))
        return np.einsum("p,pij->...ij", np.ones(1), np.zeros((1,) + res.shape[1:])) \
            if False else np.einsum("wp,pij->wij",
                                    1.0 / (u[:, None] - np.asarray(zeta)[None, :]),
                                    np.asarray(res))

    mean = np.einsum("p,pij->ij", d, np.asarray(res))
    return g, mean


def _quad(f, a, b, n=4001):
    """Composite Simpson on a smooth-enough integrand; refined until stable."""
    x = np.linspace(a, b, n)
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    w *= (b - a) / (3.0 * (n - 1))
    vals = f(x)
    return np.einsum("w,w...->...", w, vals)


def _bed(seed=0, n_dof=2, p_k=2, p_l=2, gamma=0.01):
    rng = np.random.default_rng(seed)

    def cx(*s):
        return rng.normal(size=s) + 1j * rng.normal(size=s)

    ck, cl = 0.0, 1.5
    zk = ck + rng.uniform(-0.3, 0.3, p_k) * H - 1j * gamma * rng.uniform(1, 3, p_k)
    zl = cl + rng.uniform(-0.3, 0.3, p_l) * H - 1j * gamma * rng.uniform(1, 3, p_l)
    return ck, cl, zk, zl, cx(p_k, n_dof, n_dof), cx(p_l, n_dof, n_dof)


# --------------------------------------------------------------------------- #

def test_centred_basis_has_zero_cell_mean():
    """Everything downstream is the statement that the cross terms vanish."""
    _, _, zk, _, _, _ = _bed()
    d = cell_resolvent_mean(zk, 0.0, H)
    for p, z in enumerate(zk):
        phi = lambda u, z=z, dp=d[p]: 1.0 / (u - z) - dp   # noqa: E731
        m = _quad(phi, -H / 2, H / 2) / H
        assert abs(m) < 1e-12, f"pole {p}: cell mean {m:.3e}"


def test_centred_gram_is_psd_and_matches_quadrature():
    """``H`` is a Gram matrix, so PSD is a property, not a hope."""
    _, _, zk, _, _, _ = _bed()
    d = cell_resolvent_mean(zk, 0.0, H)
    got = np.asarray(centred_gram(zk, 0.0, H))
    for p in range(zk.size):
        for q in range(zk.size):
            f = lambda u, p=p, q=q: ((1.0 / (u - zk[p]) - d[p])
                                     * np.conj(1.0 / (u - zk[q]) - d[q]))
            ref = _quad(f, -H / 2, H / 2) / H
            assert abs(got[p, q] - ref) < 1e-9 * max(abs(ref), 1.0)
    ev = np.linalg.eigvalsh(0.5 * (got + np.conj(got.T)))
    assert ev.min() > -1e-10 * max(abs(ev).max(), 1.0), ev


def test_variance_is_the_norm_the_bound_uses():
    _, _, zk, _, rk, _ = _bed()
    got = cell_variance(rk, zk, 0.0, H)
    d = cell_resolvent_mean(zk, 0.0, H)

    def f(u):
        dg = np.einsum("wp,pij->wij",
                       1.0 / (u[:, None] - zk[None, :]) - d[None, :], rk)
        return np.sum(np.abs(dg) ** 2, axis=(1, 2))

    ref = _quad(f, -H / 2, H / 2) / H
    assert abs(got - ref) < 1e-8 * ref, f"{got:.6e} vs {ref:.6e}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_covariance_identity_holds_to_roundoff(seed):
    r"""The whole method in one assertion.

        exact = ring box term + Delta I

    If the cross terms did not cancel, or the kernel's box subtraction were
    wrong, this is where it shows -- and it shows as a finite discrepancy, not
    a small one, because the mean x mean term is the dominant piece.
    """
    ck, cl, zk, zl, rk, rl = _bed(seed)
    gk, mk = _leg(zk, rk, ck)
    gl, ml = _leg(zl, rl, cl)
    omega = np.array([ck + cl])                    # the matched cell pair

    def integrand(u):
        return np.einsum("wij,wjl->wil", gk(u), gl(omega[0] - u))

    exact = _quad(integrand, ck - H / 2, ck + H / 2) / (2 * np.pi)
    box = (H / (2 * np.pi)) * (mk @ ml)
    delta = np.asarray(pair_covariance(rk, rl, zk, zl, ck, cl, H, omega))[0]

    err = np.abs(exact - (box + delta)).max() / np.abs(exact).max()
    assert err < 1e-9, (
        f"identity broken: {err:.3e}\n exact {exact}\n box+delta {box + delta}")
    # ... and the correction must not be negligible, or the test is vacuous
    assert np.abs(delta).max() > 1e-3 * np.abs(box).max()


def test_the_ring_alone_is_wrong_by_the_amount_the_correction_supplies():
    """The box term is what the FFT ring computes; measure its error directly."""
    ck, cl, zk, zl, rk, rl = _bed(0, gamma=0.002)
    gk, mk = _leg(zk, rk, ck)
    gl, ml = _leg(zl, rl, cl)
    omega = np.array([ck + cl])

    exact = _quad(lambda u: np.einsum("wij,wjl->wil", gk(u), gl(omega[0] - u)),
                  ck - H / 2, ck + H / 2) / (2 * np.pi)
    box = (H / (2 * np.pi)) * (mk @ ml)
    delta = np.asarray(pair_covariance(rk, rl, zk, zl, ck, cl, H, omega))[0]

    ring_err = np.abs(exact - box).max() / np.abs(exact).max()
    corr_err = np.abs(exact - box - delta).max() / np.abs(exact).max()
    assert ring_err > 0.05, f"bed does not exercise the error: {ring_err:.3e}"
    assert corr_err < 1e-9
    assert corr_err < 1e-6 * ring_err


@pytest.mark.parametrize("gamma", [1e-1, 1e-2, 1e-3, 1e-4])
def test_the_correction_stays_exact_as_the_line_narrows(gamma):
    """The requirement a sharp-resonance method has to meet.

    The rectangle rule diverges as ``1/gamma``; this must not. A method whose
    own error grows with the thing it exists to treat is no method.
    """
    ck, cl, zk, zl, rk, rl = _bed(3, gamma=gamma)
    gk, _ = _leg(zk, rk, ck)
    gl, _ = _leg(zl, rl, cl)
    omega = np.array([ck + cl])
    n = 200001 if gamma < 1e-3 else 20001
    exact = _quad(lambda u: np.einsum("wij,wjl->wil", gk(u), gl(omega[0] - u)),
                  ck - H / 2, ck + H / 2, n=n) / (2 * np.pi)
    d_k = cell_resolvent_mean(zk, ck, H)
    d_l = cell_resolvent_mean(zl, cl, H)
    box = (H / (2 * np.pi)) * (np.einsum("p,pij->ij", d_k, rk)
                               @ np.einsum("q,qij->ij", d_l, rl))
    delta = np.asarray(pair_covariance(rk, rl, zk, zl, ck, cl, H, omega))[0]
    err = np.abs(exact - box - delta).max() / np.abs(exact).max()
    assert err < 1e-6, f"gamma={gamma}: {err:.3e}"


def test_finite_cell_kernel_is_stable_at_the_combination_frequency():
    r"""The small-denominator regime, which cannot actually be reached.

    ``s = Omega - zeta_p - zeta_q`` has ``Im s = gamma_p + gamma_q > 0``, so
    ``|s|`` is bounded below by the combined linewidth however the output
    frequency is chosen. The logarithmic form is therefore never evaluated at
    ``s -> 0`` for physical poles, and a series fallback would be dead code
    guarding a regime that does not occur. Checked here against mpmath at 60
    digits, output placed exactly AT the combination frequency, which is the
    closest approach available.
    """
    mp = pytest.importorskip("mpmath")
    from quatrex.phonon.pole_bubble import pair_convolution

    mp.mp.dps = 60
    a, b = -H / 2, H / 2
    for g in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
        gam = g * H
        p, q, om = -1j * gam, 3.0 - 1j * gam, 3.0
        got = pair_convolution(np.array([p]), np.array([q]),
                               np.array([om]), window=(a, b))[0, 0]
        P, Q, W = mp.mpc(p), mp.mpc(q), mp.mpc(om)
        ref = complex(mp.quad(lambda u: 1 / ((u - P) * (W - u - Q)),
                              [mp.mpf(a), mp.mpf(0), mp.mpf(b)]) / (2 * mp.pi))
        assert abs(got - ref) < 1e-13 * abs(ref), (
            f"gamma/h={g}: {got} vs {ref}")


def test_rank_one_residues_factorise_the_vertex_contraction():
    r"""``B(R_p, R_q)`` is one projected pair vertex on each side.

    That is why the pole algebra is quadratic in the flattened residue count.
    A quartic implementation is carrying a redundant modal index, not doing
    more work for a reason.
    """
    rng = np.random.default_rng(5)
    n = 4

    def cx(*s):
        return rng.normal(size=s) + 1j * rng.normal(size=s)

    x_p, y_p, x_q, y_q = cx(n), cx(n), cx(n), cx(n)
    r_p = np.outer(x_p, np.conj(y_p))
    r_q = np.outer(x_q, np.conj(y_q))
    phi = cx(n, n, n)

    direct = np.einsum("mab,ad,bc,ncd->mn", phi, r_p, r_q, phi)
    left = np.einsum("mab,a,b->m", phi, x_p, x_q)
    right = np.einsum("ncd,c,d->n", phi, np.conj(y_q), np.conj(y_p))
    assert np.abs(direct - np.outer(left, right)).max() < 1e-10 * np.abs(direct).max()
