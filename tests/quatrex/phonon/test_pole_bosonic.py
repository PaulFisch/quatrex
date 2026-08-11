# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The bosonic half of the mixed sector.

The mixed convolution runs over the whole frequency axis, but the solver only
ever holds ``G`` for ``omega >= 0``. The negative half is fixed by
``R(-w) = R(w)^*`` -- it is not missing information -- but it does have to be
SUPPLIED, because the cell kernel integrates exactly the cells it is given.

Omitting it was the defect that broke the Phi-derivable energy balance on the
first device-scale run of ``sectors="rr_ss_sr"`` (bubble balance 2.6e-02
against 3e-08 for the baseline, and a heat profile that went negative at the
left contact).
"""
import numpy as np
import pytest

from quatrex.phonon.pole_bubble import bosonic_closure
from quatrex.phonon.pole_keldysh import PoleCluster
from quatrex.phonon.pole_audit import transpose_index
from quatrex.phonon.pole_mixed import bosonic_extend, mixed_convolution_batched


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _pattern(sizes):
    off = np.concatenate(([0], np.cumsum(sizes)))
    rows, cols = [], []
    for i in range(len(sizes)):
        for j in range(max(0, i - 1), min(len(sizes), i + 2)):
            for a in range(off[i], off[i + 1]):
                for b in range(off[j], off[j + 1]):
                    rows.append(a)
                    cols.append(b)
    return np.array(rows), np.array(cols), off


def _keldysh_pair(u, centre=11.0, gamma=0.4, kt=25.0):
    """A physically valid ``(G^<, G^>)`` pair at equilibrium.

    ``G^< = i n_B(w) A(w)``, ``G^> = i (n_B(w)+1) A(w)`` with ``A`` ODD in
    ``w``. This pair satisfies the true bosonic relation
    ``G^<(-w) = G^>(w)``; it does NOT satisfy ``G^<(-w) = conj(G^<(w))``.

    Building the bed from the physics rather than from the relation under test
    is the point: an earlier version of this module used a conjugate-symmetric
    function, which made the wrong mirror look right.
    """
    n_b = 1.0 / np.expm1(u / kt)
    a = (gamma / ((u - centre) ** 2 + gamma ** 2)
         - gamma / ((u + centre) ** 2 + gamma ** 2))
    return 1j * n_b * a, 1j * (n_b + 1.0) * a


def test_the_bed_satisfies_the_physical_relation_and_not_the_conjugate_one():
    """Guard the guard, and pin the negative control.

    Everything below rests on the bed being right, and the failure mode this
    module exists to catch is a bed that satisfies the WRONG relation.
    """
    u = np.linspace(0.5, 30.0, 121)
    gl_pos, gg_pos = _keldysh_pair(u)
    gl_neg, _ = _keldysh_pair(-u)

    assert np.abs(gl_neg - gg_pos).max() < 1e-14, "G^<(-w) == G^>(w)"
    wrong = np.abs(gl_neg - np.conj(gl_pos)).max()
    assert wrong / np.abs(gl_pos).max() > 1.0, (
        "the conjugate relation must fail by a LARGE margin, not marginally: "
        f"got {wrong:.3e}")


def test_bosonic_extend_uses_the_partner_component():
    """The negative axis must come from ``G^>``, not from ``conj(G^<)``."""
    w = np.linspace(0.0, 30.0, 301)
    gl, gg = _keldysh_pair(w)
    full, w_full = bosonic_extend(gl[:, None], gg[:, None], w)
    full, w_full = _h(full)[:, 0], _h(w_full)

    assert w_full.shape == (2 * w.size - 1,)          # zero bin once
    # n_B diverges at w = 0 (a real Bose pole), so compare away from it.
    sel = np.abs(w_full) > 0.5
    exact_l, _ = _keldysh_pair(w_full[sel])
    scale = np.abs(exact_l).max()
    err = np.abs(full[sel] - exact_l).max() / scale
    assert err < 1e-12, f"extended lesser vs exact G^<(w) on the full axis: {err:.3e}"

    # Negative control: the old same-component conjugate mirror.
    old_style, _ = bosonic_extend(gl[:, None], np.conj(gl)[:, None], w)
    bad = np.abs(_h(old_style)[sel, 0] - exact_l).max() / scale
    assert bad > 0.5, f"the conjugate mirror must be visibly wrong: {bad:.3e}"


def test_bosonic_extend_transposes_the_partner_on_the_pattern():
    """``G^<_ij(-w) = G^>_ji(w)``: the index swap is part of the relation."""
    from quatrex.phonon.pole_audit import transpose_index

    sizes = np.array([2, 2])
    rows, cols, _ = _pattern(sizes)
    t = transpose_index(rows, cols)
    w = np.linspace(0.0, 10.0, 51)
    rng = np.random.default_rng(0)
    gl = rng.normal(size=(w.size, rows.size)) + 1j * rng.normal(size=(w.size, rows.size))
    gg = rng.normal(size=(w.size, rows.size)) + 1j * rng.normal(size=(w.size, rows.size))

    full, _ = bosonic_extend(gl, gg, w, transpose_index=t)
    got = _h(full)
    # the negative half is the partner, reversed in w AND index-transposed
    assert np.allclose(got[:w.size - 1], _h(gg)[:0:-1][:, t])
    assert np.allclose(got[w.size - 1:], _h(gl))


def test_bosonic_extend_refuses_a_grid_not_anchored_at_zero():
    """The mirror is defined about omega = 0; guessing an offset would put the
    negative half in the wrong place and still return a plausible array."""
    freqs = np.linspace(1.5, 10.0, 18)
    r = np.zeros((18, 2), dtype=complex)
    with pytest.raises(NotImplementedError, match="zero-anchored"):
        bosonic_extend(r, r, freqs)


def test_bosonic_extend_refuses_a_mismatched_partner():
    freqs = np.linspace(0.0, 10.0, 11)
    with pytest.raises(ValueError, match="share the grid and pattern"):
        bosonic_extend(np.zeros((11, 4), dtype=complex),
                       np.zeros((11, 3), dtype=complex), freqs)


def test_pole_set_is_closed_before_any_leg_is_built():
    """``bubble_clusters`` was implemented, documented as mandatory, and never
    called -- ``G_PP`` was built from half a pole set."""
    rng = np.random.default_rng(1)
    z = np.array([8.0 - 0.3j, 11.0 - 0.4j])
    u = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    v = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    closed = bosonic_closure(PoleCluster(z=z, u=u, v=v))
    assert closed.n_poles == 4
    got = _h(closed.z)
    assert np.allclose(got[:2], z)
    assert np.allclose(got[2:], -np.conj(z))


def test_state_separates_the_leg_set_from_the_solved_set():
    """The tracker matches the solved set; the bubble consumes ``legs``.

    They are the same list today (see below), but keeping them distinct is what
    makes enabling the closure a one-line change once the mirrored source
    exists, and stops a closed set doubling the reported pole count.
    """
    from quatrex.phonon.pole_sector import PoleSectorState

    st = PoleSectorState()
    assert hasattr(st, "legs") and st.legs == []
    assert st.clusters == []


def test_closure_is_deferred_because_the_frozen_source_cannot_serve_both_branches():
    """A measured decision, recorded so it is not silently re-attempted.

    ``bubble_clusters()`` closes the pole set under ``z -> -z^*``, which puts
    partners at NEGATIVE real frequency. The frozen source is evaluated at a
    single index, the positive centre, so after closure one source is applied
    to poles at both ``+Omega`` and ``-Omega`` -- and the partner's source is
    the bosonic mirror of the original's, not the same matrix. Wiring the
    closure in without that made ``rr_ss`` worse by 3400x on the production
    bed (bubble balance 5.4e-08 -> 1.8e-04).
    """
    rng = np.random.default_rng(2)
    z = np.array([8.0 - 0.3j, 11.0 - 0.4j])
    u = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    v = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    closed = bosonic_closure(PoleCluster(z=z, u=u, v=v))
    got = _h(closed.z)
    # The partners sit at negative real frequency: that is exactly why one
    # frozen source evaluated at the positive centre cannot serve them.
    assert np.all(got[2:].real < 0.0)
    assert np.all(got.imag < 0.0), "partners must stay retarded"


def test_sr_equals_rs_on_a_leg_symmetric_vertex():
    """On a real vertex the two mixed sectors are the SAME object.

    hiphive's ``symmetrize=True`` gives ``Phi[i,a,b] == Phi[i,b,a]``, and under
    that symmetry ``Sigma_SR`` and ``Sigma_RS`` coincide exactly. Tests that
    assert they differ are measuring the asymmetry of a random test vertex, not
    physics -- worth pinning, because that asymmetry was once read as evidence
    of a bug.
    """
    from quatrex.phonon.pole_bridge import (
        _mixed_one_sector_blocked, mixed_vertex_block_dict)

    sizes = np.array([3, 3, 3])
    off = np.concatenate(([0], np.cumsum(sizes)))
    nb, n_dof = len(sizes), int(sizes.sum())
    rows, cols = [], []
    for i in range(nb):
        for j in range(max(0, i - 1), min(nb, i + 2)):
            for a in range(off[i], off[i + 1]):
                for b in range(off[j], off[j + 1]):
                    rows.append(a)
                    cols.append(b)
    rows, cols = np.array(rows), np.array(cols)

    rng = np.random.default_rng(0)
    raw = {(i, k1, k2): rng.normal(size=(sizes[i], sizes[k1], sizes[k2]))
           for i in range(nb)
           for k1 in range(max(0, i - 1), min(nb, i + 2))
           for k2 in range(max(0, i - 1), min(nb, i + 2))}
    phi = {k: 0.5 * (v + np.swapaxes(raw[(k[0], k[2], k[1])], 1, 2))
           for k, v in raw.items()}
    assert max(np.abs(phi[(i, a, b)] - np.swapaxes(phi[(i, b, a)], 1, 2)).max()
               for (i, a, b) in phi) == 0.0

    npp = 2
    u = rng.normal(size=(n_dof, npp)) + 1j * rng.normal(size=(n_dof, npp))
    v = rng.normal(size=(n_dof, npp)) + 1j * rng.normal(size=(n_dof, npp))
    cl = PoleCluster(z=np.array([8.0 - 0.3j, 11.0 - 0.4j]), u=u, v=v)
    freqs = np.linspace(0.0, 30.0, 128)
    omega = np.linspace(5.0, 15.0, 5)
    t = transpose_index(rows, cols)
    a = (rng.normal(size=(freqs.size, rows.size))
         + 1j * rng.normal(size=(freqs.size, rows.size)))
    g_reg = a - np.conj(a[:, t])
    b = (rng.normal(size=(freqs.size, rows.size))
         + 1j * rng.normal(size=(freqs.size, rows.size)))
    g_partner = b - np.conj(b[:, t])          # a distinct Keldysh partner
    m = rng.normal(size=(npp, npp)) + 1j * rng.normal(size=(npp, npp))
    src = 1j * (m @ m.conj().T)

    vd, kw = mixed_vertex_block_dict, dict(
        freqs=freqs, rows=rows, cols=cols, block_sizes=sizes,
        g_partner=g_partner)
    sr = _h(_mixed_one_sector_blocked(
        omega, cl, src, g_reg,
        bl=vd(phi, sizes, cl.u, leg=0, conjugate=False),
        br=vd(phi, sizes, cl.u, leg=1, conjugate=True), **kw))
    rs = _h(_mixed_one_sector_blocked(
        omega, cl, src, g_reg,
        bl=vd(phi, sizes, cl.u, leg=1, conjugate=False),
        br=vd(phi, sizes, cl.u, leg=0, conjugate=True), **kw))
    assert np.abs(sr - rs).max() / np.abs(sr).max() < 1e-13


def test_pair_source_is_exact_on_the_diagonal_and_averaged_off_it():
    """One source per pole PAIR, at that pair's own frequency.

    For ``alpha == beta`` the two poles share a real part, so the pair
    frequency IS ``Re z_alpha`` and no accuracy is lost relative to a per-pole
    evaluation. Only the cross terms are averaged, and their coefficients are
    suppressed by a large ``gap``.
    """
    from quatrex.phonon.pole_bridge import source_at_poles

    w = np.linspace(0.0, 20.0, 401)
    z = np.array([8.0 - 0.3j, 11.0 - 0.4j])
    cl = PoleCluster(z=z, u=np.zeros((4, 2), complex),
                     v=np.zeros((4, 2), complex))
    base = np.array([[1.0, 2.0], [3.0, 4.0]])
    src = np.einsum("w,ab->wab", w, base) + 0j

    got = _h(source_at_poles(src, w, cl))
    # diagonal: evaluated at 8 and 11 exactly
    assert np.isclose(got[0, 0], 8.0 * base[0, 0])
    assert np.isclose(got[1, 1], 11.0 * base[1, 1])
    # off diagonal: the midpoint 9.5, shared by (0,1) and (1,0)
    assert np.isclose(got[0, 1], 9.5 * base[0, 1])
    assert np.isclose(got[1, 0], 9.5 * base[1, 0])


def test_gpp_decays_like_one_over_omega_squared():
    """The asymptotics gate: ``c_a + c_b`` must vanish.

    Writing the leg as ``c_a/(w-z_a) + c_b/(w-conj(z_b))``, the large-``w``
    behaviour is ``(c_a+c_b)/w`` while the congruence it models decays as
    ``1/w^2``. Using a DIFFERENT source at each pole leaves the sum nonzero
    and gives ``G_PP`` a spurious ``1/w`` tail -- measured 17x too large at
    ``w = 1e2`` and 18364x at ``1e5``, which is what made ``rr_ss`` regress
    from 5.4e-08 to 4.9e-05.
    """
    from quatrex.phonon.pole_bridge import source_at_poles

    w = np.linspace(0.0, 40.0, 4001)
    z = np.array([9.0 - 0.5j, 14.0 - 0.7j])
    cl = PoleCluster(z=z, u=np.zeros((4, 2), complex),
                     v=np.zeros((4, 2), complex))
    src = np.einsum("w,ab->wab", 1.0 + 0.03 * w,
                    np.array([[1.0, 2.0], [3.0, 4.0]])) + 0j
    s_pair = _h(source_at_poles(src, w, cl))

    za, zb = z[:, None], np.conj(z)[None, :]
    gap = za - zb
    ca, cb = s_pair / gap, -s_pair / gap
    assert np.abs(ca + cb).max() == 0.0, "the 1/w coefficient must vanish"

    wl = np.logspace(2, 6, 5)
    for a in range(2):
        for b in range(2):
            pf = ca[a, b] / (wl - z[a]) + cb[a, b] / (wl - np.conj(z[b]))
            cong = s_pair[a, b] / ((wl - z[a]) * (wl - np.conj(z[b])))
            assert np.allclose(np.abs(pf / cong), 1.0, rtol=1e-10), \
                f"pair ({a},{b}) must track the congruence out to w = 1e6"

    # Negative control: distinct sources reintroduce the tail.
    bad_a, bad_b = s_pair, s_pair * 1.4
    r = (bad_a - bad_b) / gap
    assert np.abs(r).max() > 0.1
    pf_bad = bad_a[0, 1] / (wl - z[0]) - bad_b[0, 1] / (wl - np.conj(z[1]))
    cong = s_pair[0, 1] / ((wl - z[0]) * (wl - np.conj(z[1])))
    assert np.abs(pf_bad / cong)[-1] > 100.0, \
        "distinct sources must show a visibly wrong tail"


def test_negative_frequency_poles_use_the_bosonic_mirror():
    """A partner at -Omega takes S(-w) = conj(S(w)), not an extrapolation."""
    from quatrex.phonon.pole_bridge import source_at_poles

    w = np.linspace(0.0, 20.0, 401)
    z = np.array([8.0 - 0.3j, -8.0 - 0.3j])          # a closed pair
    cl = PoleCluster(z=z, u=np.zeros((4, 2), complex),
                     v=np.zeros((4, 2), complex))
    src = np.einsum("w,ab->wab", w, np.ones((2, 2))) * (1 + 2j)
    got = _h(source_at_poles(src, w, cl))
    assert np.allclose(got[1, 1], np.conj(got[0, 0])), \
        "the -Omega partner must carry the conjugated source"


def test_gpp_partial_fraction_form_matches_what_the_sectors_represent():
    """The leg subtracted from the ring must BE the leg the sectors put back.

    ``G_reg = G - G_PP`` is exact for any ``G_PP``, so the sector sum
    ``B(G,G) = SS + SR + RS + RR`` holds iff both sides use literally the same
    object. The resolved form ``U D^R S(w) D^A U^dag`` and the partial-fraction
    form agree only when ``S`` is constant in frequency; otherwise they are
    different functions and the decomposition is broken.
    """
    from quatrex.phonon.pole_bridge import pole_keldysh_pf_sparse
    from quatrex.phonon.pole_bubble import leg_partial_fractions

    rng = np.random.default_rng(0)
    n, ne = 4, 128
    w = np.linspace(0.0, 40.0, ne)
    z = np.array([9.0 - 0.8j, 14.0 - 1.1j])
    u = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
    cl = PoleCluster(z=z, u=u, v=u)
    s_a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    r, c = r.ravel(), c.ravel()

    got = _h(pole_keldysh_pf_sparse(w, cl, s_a, r, c))
    p, coef = leg_partial_fractions(cl, s_a)
    f = np.zeros((ne, 2, 2), dtype=complex)
    for a in range(2):
        for b in range(2):
            f[:, a, b] = (_h(coef)[a, b, 0] / (w - _h(p)[a, b, 0])
                          + _h(coef)[a, b, 1] / (w - _h(p)[a, b, 1]))
    ref = np.einsum("ia,wab,jb->wij", _h(u), f, np.conj(_h(u)))[:, r, c]
    assert np.abs(got - ref).max() == 0.0, "must be the SAME object, not close"


def test_resolved_and_partial_fraction_legs_differ_when_the_source_varies():
    """Negative control: the two forms are NOT interchangeable.

    Measured 7e-3 apart on a source with a 2%/THz slope, which is the size of
    inconsistency that broke the spatial energy balance.
    """
    from quatrex.phonon.pole_keldysh import pole_keldysh

    rng = np.random.default_rng(0)
    n, ne = 4, 512
    w = np.linspace(0.0, 40.0, ne)
    z = np.array([9.0 - 0.8j, 14.0 - 1.1j])
    u = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
    cl = PoleCluster(z=z, u=u, v=u)
    m = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    s0 = 1j * (m @ m.conj().T)

    za, zb = z[:, None], np.conj(z)[None, :]
    gap = za - zb

    def _pf(sa, sb):
        f = (sa / gap)[None] / (w[:, None, None] - za[None]) \
            - (sb / gap)[None] / (w[:, None, None] - zb[None])
        return np.einsum("ia,wab,jb->wij", _h(u), f, np.conj(_h(u)))

    def _pick(src, centres):
        return np.array([[src[int(np.argmin(abs(w - centres[a].real))), a, b]
                          for b in range(2)] for a in range(2)])

    for label, src, tol in (
        ("constant", np.broadcast_to(s0, (ne, 2, 2)).copy(), 1e-12),
        ("varying", np.einsum("w,ab->wab", 1.0 + 0.02 * w, s0), None),
    ):
        resolved = _h(pole_keldysh(w, cl, src))
        pf = _pf(_pick(src, z), _pick(src, np.conj(z)))
        rel = np.abs(resolved - pf).max() / np.abs(resolved).max()
        if tol is not None:
            assert rel < tol, f"{label}: {rel:.2e}"
        else:
            assert rel > 1e-3, f"{label} should differ, got {rel:.2e}"


def test_source_variation_is_a_measured_residual_not_an_asymptotic_claim():
    """``source_fit_tol`` gates on THIS, not on ``O((|Im z|/h)^(p+1))``.

    The asymptotic estimate omits the analyticity radius and higher
    derivatives of the source, the pole's offset from the fit centre, the
    window width and the conditioning of the fit. A source with structure of
    its own is invisible to it and obvious here.
    """
    from quatrex.phonon.pole_bridge import source_variation

    w = np.linspace(0.0, 40.0, 4001)
    z = np.array([9.0 - 0.5j, 14.0 - 0.7j])
    cl = PoleCluster(z=z, u=np.zeros((4, 2), complex),
                     v=np.zeros((4, 2), complex))
    base = np.array([[1.0, 2.0], [3.0, 4.0]])

    flat = np.broadcast_to(base + 0j, (w.size, 2, 2)).copy()
    assert source_variation(flat, w, cl) == 0.0

    gentle = np.einsum("w,ab->wab", 1.0 + 0.03 * w, base) + 0j
    assert source_variation(gentle, w, cl) < 1e-2

    # A source with a resonance of its own inside the pole window: exactly the
    # case an analytic model has no business carrying.
    sharp = np.einsum("w,ab->wab", 1.0 / ((w - 9.0) ** 2 + 0.05 ** 2), base) + 0j
    assert source_variation(sharp, w, cl) > 5e-2

    # And it is scale free.
    assert np.isclose(source_variation(1e6 * sharp, w, cl),
                      source_variation(sharp, w, cl))


def test_state_records_the_source_fit_per_cluster():
    from quatrex.phonon.pole_sector import PoleSectorState

    assert PoleSectorState().source_fit == []
