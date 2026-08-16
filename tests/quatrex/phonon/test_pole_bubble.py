# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Doc Phase 4: the analytic pole-pole bubble against high-accuracy quadrature.

The acceptance test the design note states for this phase is a comparison
against a deliberately over-resolved convolution. Here the reference is adaptive
quadrature (``scipy.integrate.quad`` with the resonances passed as break
points), which is both sharper and cheaper than an over-resolved FFT for a
rational integrand.

The last test is the one that matters for the method's justification: at a grid
spacing typical of production, the discrete convolution of two narrow modes is
not merely imprecise, it is wrong by orders of magnitude and in a direction that
depends on where the bins happen to fall.
"""
import numpy as np
import pytest
from scipy.integrate import quad

from quatrex.phonon.pole_bubble import (
    bosonic_closure,
    leg_partial_fractions,
    modal_convolution,
    modal_vertex,
    pair_convolution,
    retarded_from_pole_sum,
    ss_self_energy,
)
from quatrex.phonon.pole_keldysh import PoleCluster


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _time_domain_pair(p, q, w, t_max=600.0):
    """Independent reference for J(p, q; w), via the convolution theorem."""
    lower_p, lower_q = p.imag < 0, q.imag < 0
    if lower_p != lower_q:
        return 0.0 + 0.0j                       # disjoint support in t
    s = 1j * (w - p - q)
    lo, hi = (0.0, t_max) if lower_p else (-t_max, 0.0)
    re = quad(lambda t: (-np.exp(s * t)).real, lo, hi, limit=400)[0]
    im = quad(lambda t: (-np.exp(s * t)).imag, lo, hi, limit=400)[0]
    return re + 1j * im


def _pole_transform(p, t):
    """Time transform of a simple pole: one-sided, on the side set by Im p."""
    if p.imag < 0:
        return np.where(t > 0, -1j * np.exp(-1j * p * t), 0.0)
    return np.where(t < 0, 1j * np.exp(-1j * p * t), 0.0)


def _time_domain_modal(poles1, coeffs1, poles2, coeffs2, w, t_max=600.0):
    """Reference for a modal convolution, built from the partial fractions."""
    def leg(poles, coeffs, t):
        return sum(c * _pole_transform(pp, t) for pp, c in zip(poles, coeffs))

    def integrand(t, part):
        v = leg(poles1, coeffs1, t) * leg(poles2, coeffs2, t) * np.exp(1j * w * t)
        return v.real if part == "re" else v.imag

    out = 0.0 + 0.0j
    for lo, hi in ((0.0, t_max), (-t_max, 0.0)):
        out += (quad(integrand, lo, hi, args=("re",), limit=2000)[0]
                + 1j * quad(integrand, lo, hi, args=("im",), limit=2000)[0])
    return out


def _cluster(np_poles=2, n_dof=3, seed=0):
    rng = np.random.default_rng(seed)
    omega = np.array([7.0, 11.0])[:np_poles]
    gamma = np.array([0.05, 0.08])[:np_poles]
    z = omega - 1j * gamma
    u = rng.normal(size=(n_dof, np_poles)) + 1j * rng.normal(size=(n_dof, np_poles))
    u /= np.linalg.norm(u, axis=0, keepdims=True)
    v = rng.normal(size=(n_dof, np_poles)) + 1j * rng.normal(size=(n_dof, np_poles))
    return PoleCluster(z=z, u=u, v=v)


def _psd_source(np_poles, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(np_poles, np_poles)) + 1j * rng.normal(size=(np_poles, np_poles))
    return a @ a.conj().T


# --------------------------------------------------------------------------- #
# 1. The elementary convolution and its three cases.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "p, q",
    [
        (7.0 - 0.05j, 11.0 - 0.08j),   # both retarded  -> -i/(w-p-q)
        (7.0 + 0.05j, 11.0 + 0.08j),   # both advanced  -> +i/(w-p-q)
        (7.0 - 0.05j, 11.0 + 0.08j),   # mixed          -> 0
        (7.0 + 0.05j, 11.0 - 0.08j),   # mixed          -> 0
    ],
)
def test_pair_convolution_matches_quadrature(p, q):
    w = np.array([2.0, 9.0, 18.0, 25.0])
    got = _h(pair_convolution(np.array(p), np.array(q), w)).reshape(-1)
    ref = np.array([_time_domain_pair(p, q, wi) for wi in w])
    scale = max(np.abs(ref).max(), 1e-12)
    assert np.abs(got - ref).max() / scale < 1e-7, f"{got} vs {ref}"


def test_mixed_half_planes_convolve_to_zero():
    """Half the pole pairings drop out -- that is what keeps the sum short."""
    w = np.linspace(0.0, 30.0, 17)
    got = _h(pair_convolution(np.array(7.0 - 0.05j), np.array(11.0 + 0.08j), w))
    assert np.abs(got).max() == 0.0


def test_retarded_pair_gives_a_pole_at_the_sum():
    """Doc Eq. (30)/(31): the three-phonon Omega_a + Omega_b structure."""
    p, q = 7.0 - 0.05j, 11.0 - 0.08j
    w = np.linspace(16.0, 20.0, 4001)
    v = np.abs(_h(pair_convolution(np.array(p), np.array(q), w))).reshape(-1)
    peak = w[int(np.argmax(v))]
    assert abs(peak - 18.0) < 2e-3, f"peak at {peak}, expected Omega_1 + Omega_2 = 18"


# --------------------------------------------------------------------------- #
# 2. The four-index modal convolution.
# --------------------------------------------------------------------------- #

def test_modal_convolution_assembly_matches_an_explicit_pole_sum():
    """Index assembly of Eq. (117) against an explicit loop over pole
    pairings."""
    cl = _cluster(2)
    sa, sb = _psd_source(2, 1), _psd_source(2, 2)
    pa, ca = leg_partial_fractions(cl, sa)
    pb, cb = leg_partial_fractions(cl, sb)
    pa, ca, pb, cb = _h(pa), _h(ca), _h(pb), _h(cb)

    w = np.array([5.0, 18.0, 22.0])
    got = _h(modal_convolution(w, cl, sa, sb))

    npp = cl.n_poles
    ref = np.zeros((len(w), npp, npp, npp, npp), dtype=complex)
    for a in range(npp):
        for dd in range(npp):
            for b in range(npp):
                for g in range(npp):
                    acc = np.zeros(len(w), dtype=complex)
                    for j in range(2):
                        for k in range(2):
                            acc += ca[a, dd, j] * cb[b, g, k] * _h(
                                pair_convolution(np.array(pa[a, dd, j]),
                                                 np.array(pb[b, g, k]), w)
                            ).reshape(-1)
                    ref[:, a, dd, b, g] = acc
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-13


def test_modal_convolution_matches_the_convolution_theorem():
    """Coarser end-to-end cross-check against numerical time-domain
    quadrature."""
    cl = _cluster(2)
    sa, sb = _psd_source(2, 1), _psd_source(2, 2)
    pa, ca = _h(leg_partial_fractions(cl, sa)[0]), _h(leg_partial_fractions(cl, sa)[1])
    pb, cb = _h(leg_partial_fractions(cl, sb)[0]), _h(leg_partial_fractions(cl, sb)[1])

    w = np.array([5.0, 18.0])
    got = _h(modal_convolution(w, cl, sa, sb))
    for (a, dd, b, g) in [(0, 0, 0, 0), (0, 1, 1, 0), (1, 1, 1, 1)]:
        for iw, wi in enumerate(w):
            ref = _time_domain_modal(pa[a, dd], ca[a, dd], pb[b, g], cb[b, g], wi)
            err = abs(got[iw, a, dd, b, g] - ref) / max(abs(ref), 1e-12)
            assert err < 1e-5, f"C[{a},{dd},{b},{g}](w={wi}) off by {err:.3e}"


def test_partial_fractions_reproduce_the_leg():
    cl = _cluster(2)
    s = _psd_source(2, 3)
    poles, coeffs = leg_partial_fractions(cl, s)
    w = np.linspace(4.0, 14.0, 51)
    z = _h(cl.z)
    for a in range(2):
        for b in range(2):
            direct = s[a, b] / ((w - z[a]) * (w - np.conj(z[b])))
            split = sum(
                _h(coeffs)[a, b, j] / (w - _h(poles)[a, b, j]) for j in range(2)
            )
            assert np.abs(split - direct).max() / np.abs(direct).max() < 1e-10


def test_degenerate_leg_raises():
    """A pole coinciding with a partner's conjugate needs a higher-order part."""
    z = np.array([7.0 - 0.05j, 7.0 - 0.05j])
    cl = PoleCluster(z=z, u=np.eye(2, dtype=complex), v=np.eye(2, dtype=complex))
    # z_alpha == conj(z_beta) requires a real pole; force it.
    cl.z = np.array([7.0 - 1e-320j, 7.0 - 1e-320j])
    with pytest.raises(ValueError, match="simple-pole"):
        leg_partial_fractions(cl, np.eye(2, dtype=complex))


# --------------------------------------------------------------------------- #
# 3. The vertex contraction and the retarded partner.
# --------------------------------------------------------------------------- #

def test_ss_self_energy_matches_an_explicit_contraction():
    rng = np.random.default_rng(5)
    n_dof, npp = 3, 2
    cl = _cluster(npp, n_dof)
    sa, sb = _psd_source(npp, 1), _psd_source(npp, 2)
    phi = rng.normal(size=(n_dof,) * 3)
    phi = (phi + phi.transpose(0, 2, 1)) / 2.0          # leg-exchange symmetric
    w = np.array([6.0, 18.0])
    pref = 0.5j

    got = _h(ss_self_energy(w, cl, sa, sb, phi, phi, pref))
    c = _h(modal_convolution(w, cl, sa, sb))
    u = _h(cl.u)
    vl = np.einsum("ace,cA,eB->aAB", phi, u, u)
    vr = np.einsum("Jdb,dG,bD->JGD", phi, np.conj(u), np.conj(u))
    ref = pref * np.einsum("aAB,JGD,wADBG->waJ", vl, vr, c)
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-12


def test_retarded_from_pole_sum_reproduces_the_lorentzian_partner():
    """Keeping only the LHP poles must give i/(w - Omega + i gamma)."""
    from quatrex.phonon.pole_kernel import lorentz_retarded

    Omega, gamma = 8.0, 0.35
    # L = 2g/((w-O)^2+g^2) in partial fractions: -i/(w-p) + i/(w-q),
    # p = O + i g (upper), q = O - i g (lower).
    poles = np.array([Omega + 1j * gamma, Omega - 1j * gamma])
    coeffs = np.array([-1j, +1j])
    w = np.linspace(0.0, 20.0, 401)
    got = _h(retarded_from_pole_sum(w, poles, coeffs)).sum(axis=1)
    ref = _h(lorentz_retarded(w, complex(Omega, -gamma)))
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-12


def test_retarded_from_pole_sum_is_causal():
    """No upper-half-plane pole may survive into Sigma^R."""
    poles = np.array([5.0 + 0.2j, 9.0 - 0.3j])
    coeffs = np.array([1.0 + 0j, 2.0 + 0j])
    w = np.linspace(0.0, 20.0, 51)
    got = _h(retarded_from_pole_sum(w, poles, coeffs))
    assert np.abs(got[:, 0]).max() == 0.0, "an advanced pole leaked into Sigma^R"
    assert np.abs(got[:, 1]).max() > 0.0


def test_bosonic_closure_adds_the_partners():
    cl = _cluster(2)
    closed = bosonic_closure(cl)
    assert closed.n_poles == 2 * cl.n_poles
    z, zc = _h(cl.z), _h(closed.z)
    assert np.allclose(zc[:2], z)
    assert np.allclose(zc[2:], -np.conj(z))
    # Partners stay in the lower half plane, as a retarded pole set must.
    assert np.all(zc.imag < 0)


def test_modal_vertex_matches_the_definition():
    rng = np.random.default_rng(7)
    phi = rng.normal(size=(3, 3, 3))
    u = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    got = _h(modal_vertex(phi, u))
    ref = np.einsum("mab,aA,bB->mAB", phi, u, u)
    assert np.abs(got - ref).max() < 1e-12


# --------------------------------------------------------------------------- #
# 4. Why this exists at all.
# --------------------------------------------------------------------------- #

def test_discrete_convolution_is_catastrophic_where_the_analytic_form_is_exact():
    """The grid-cost claim, on the object the pole sector replaces."""
    p = q = 8.0 - 0.004j
    w_peak = 16.0
    exact = complex(_h(pair_convolution(np.array(p), np.array(q),
                                        np.array([w_peak]))).reshape(-1)[0])

    worst = 0.0
    for nf in (401, 801, 1601):
        grid = np.linspace(0.0, 32.0, nf)
        dw = grid[1] - grid[0]
        f = 1.0 / (grid - p)
        conv = np.convolve(f, f) * dw / (2.0 * np.pi)
        axis = np.linspace(2 * grid[0], 2 * grid[-1], 2 * nf - 1)
        got = conv[int(np.argmin(np.abs(axis - w_peak)))]
        worst = max(worst, abs(got - exact) / abs(exact))

    assert worst > 1.0, (
        f"the discrete convolution was accurate to {worst:.3e} -- pick a "
        "narrower mode, the premise of the sector is not being exercised"
    )


# --------------------------------------------------------------------------- #
# 5. The retarded partner, in closed form.
# --------------------------------------------------------------------------- #

def test_retarded_only_equals_the_lower_half_plane_pole_sum():
    """The claim, tested EXACTLY -- no quadrature, no truncation."""
    cl = _cluster(2, n_dof=2, seed=3)
    npp = cl.n_poles
    rng = np.random.default_rng(21)
    a = rng.normal(size=(npp, npp)) + 1j * rng.normal(size=(npp, npp))
    src = a @ a.conj().T

    pa, ca = leg_partial_fractions(cl, src)
    pa, ca = _h(pa), _h(ca)
    w = np.linspace(2.0, 60.0, 401)

    got = _h(modal_convolution(w, cl, src, src, retarded_only=True))
    ref = np.zeros_like(got)
    for al in range(npp):
        for dl in range(npp):
            for be in range(npp):
                for ga in range(npp):
                    poles, coeffs = [], []
                    for j in range(2):
                        for k in range(2):
                            pj, qk = pa[al, dl, j], pa[be, ga, k]
                            if pj.imag < 0 and qk.imag < 0:
                                sign = -1j
                            elif pj.imag > 0 and qk.imag > 0:
                                sign = 1j
                            else:
                                continue
                            poles.append(pj + qk)
                            coeffs.append(ca[al, dl, j] * ca[be, ga, k] * sign)
                    if not poles:
                        continue
                    contrib = _h(retarded_from_pole_sum(
                        w, np.array(poles), np.array(coeffs)))
                    ref[:, al, dl, be, ga] = contrib.sum(axis=1)

    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-13


def test_retarded_only_is_causal():
    """Every surviving pole must sit in the lower half plane."""
    cl = bosonic_closure(_cluster(2, n_dof=2, seed=4))
    s = np.eye(cl.n_poles, dtype=complex)
    pa, _ = leg_partial_fractions(cl, s)
    pa = _h(pa)
    # The (0, 0) pairing is z_alpha + z_beta, both retarded.
    combined = pa[..., 0][:, :, None, None] + pa[..., 0][None, None, :, :]
    assert (combined.imag < 0).all(), "a retained pairing left the lower half plane"


def test_cell_average_is_exact_where_point_sampling_is_not():
    """The analytic sector must enter the grid solver's own representation."""
    h = 0.25
    w = np.arange(0.0, 40.0, h)
    # The pair convolution has COMBINED width 2*gamma, so it is better
    # resolved than either pole; the thresholds below are measured, not
    # guessed. At 2*gamma/h = 0.40 point sampling is 16.5 % wrong, at 0.16 it
    # is 108 %, at 0.04 it is 653 %.
    for gamma, min_sample_err in ((0.05, 0.1), (0.02, 1.0), (0.005, 5.0)):
        p = np.array([4.0 - 1j * gamma])
        q = np.array([5.0 - 1j * gamma])
        pq = complex(p[0] + q[0])
        exact = -1j * (np.log(w[-1] + h / 2 - pq) - np.log(w[0] - h / 2 - pq))

        sample = _h(pair_convolution(p, q, w))[:, 0]
        cell = _h(pair_convolution(p, q, w, cell=h))[:, 0]

        cell_err = abs((cell * h).sum() - exact) / abs(exact)
        samp_err = abs((sample * h).sum() - exact) / abs(exact)
        assert cell_err < 1e-12, f"cell average must be exact: {cell_err:.3e}"
        assert samp_err > min_sample_err, (
            f"point sampling must be visibly wrong at gamma/h={gamma/h:.2f}: "
            f"{samp_err:.3e}")


def test_cell_average_reduces_to_the_point_sample_when_resolved():
    """At ``gamma >> h`` the two agree, so enabling it changes nothing where
    the grid was already adequate."""
    h = 0.05
    w = np.arange(0.0, 40.0, h)
    p, q = np.array([4.0 - 2.0j]), np.array([5.0 - 2.0j])
    sample = _h(pair_convolution(p, q, w))[:, 0]
    cell = _h(pair_convolution(p, q, w, cell=h))[:, 0]
    assert np.abs(cell - sample).max() / np.abs(sample).max() < 1e-3
