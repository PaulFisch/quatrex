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


def _bosonic_background(u, centre=11.0, gamma=0.4):
    """A background that genuinely satisfies ``a(-w) = conj(a(w))``.

    A single Lorentzian does NOT satisfy it; the conjugate-symmetric pair does.
    Getting this wrong makes the test measure the bed rather than the kernel.
    """
    return 1.0 / (u - centre + 1j * gamma) - 1.0 / (u + centre + 1j * gamma)


def test_background_bed_really_is_bosonic():
    """Guard the guard: the whole test rests on this identity holding."""
    u = np.linspace(0.5, 40.0, 97)
    assert np.allclose(_bosonic_background(-u), np.conj(_bosonic_background(u)))


def test_one_sided_convolution_is_badly_wrong():
    """The negative half is not a correction -- it is ~a quarter of the answer.

    This is the regression: before the fix the production path integrated only
    ``[0, w_max]``.
    """
    pole = 8.0 - 0.3j
    omega = np.array([9.0, 12.5, 15.0])
    wide = np.linspace(-200.0, 200.0, 200001)
    ref = _h(mixed_convolution_batched(
        omega, pole, 1.0 + 0j, _bosonic_background(wide)[:, None], wide))[:, 0]

    half = np.linspace(0.0, 60.0, 20001)
    one_sided = _h(mixed_convolution_batched(
        omega, pole, 1.0 + 0j, _bosonic_background(half)[:, None], half))[:, 0]
    err = np.abs(one_sided - ref).max() / np.abs(ref).max()
    assert err > 0.1, f"one-sided error should be large, got {err:.2e}"

    r_full, w_full = bosonic_extend(_bosonic_background(half)[:, None], half)
    fixed = _h(mixed_convolution_batched(
        omega, pole, 1.0 + 0j, r_full, w_full))[:, 0]
    assert np.abs(fixed - ref).max() / np.abs(ref).max() < 1e-3


def test_bosonic_extend_shapes_and_symmetry():
    freqs = np.linspace(0.0, 10.0, 11)
    rng = np.random.default_rng(0)
    r = rng.normal(size=(11, 4)) + 1j * rng.normal(size=(11, 4))
    r_full, w_full = bosonic_extend(r, freqs)
    assert w_full.shape == (21,)                      # zero bin exactly once
    assert r_full.shape == (21, 4)
    assert np.allclose(_h(w_full), np.concatenate([-freqs[:0:-1], freqs]))
    # a(-w) = conj(a(w)) on the constructed half
    assert np.allclose(_h(r_full)[:10], np.conj(_h(r)[:0:-1]))
    assert np.allclose(_h(r_full)[10:], _h(r))


def test_bosonic_extend_refuses_a_grid_not_anchored_at_zero():
    """The mirror is defined about omega = 0; guessing an offset would put the
    negative half in the wrong place and still return a plausible array."""
    freqs = np.linspace(1.5, 10.0, 18)
    r = np.zeros((18, 2), dtype=complex)
    with pytest.raises(NotImplementedError, match="zero-anchored"):
        bosonic_extend(r, freqs)


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
    m = rng.normal(size=(npp, npp)) + 1j * rng.normal(size=(npp, npp))
    src = 1j * (m @ m.conj().T)

    vd, kw = mixed_vertex_block_dict, dict(
        freqs=freqs, rows=rows, cols=cols, block_sizes=sizes)
    sr = _h(_mixed_one_sector_blocked(
        omega, cl, src, g_reg,
        bl=vd(phi, sizes, cl.u, leg=0, conjugate=False),
        br=vd(phi, sizes, cl.u, leg=1, conjugate=True), **kw))
    rs = _h(_mixed_one_sector_blocked(
        omega, cl, src, g_reg,
        bl=vd(phi, sizes, cl.u, leg=1, conjugate=False),
        br=vd(phi, sizes, cl.u, leg=0, conjugate=True), **kw))
    assert np.abs(sr - rs).max() / np.abs(sr).max() < 1e-13


def test_per_pole_sources_beat_a_frozen_one_on_a_multi_pole_cluster():
    """The exact residues use S at EACH pole, not one value for the cluster.

    With a cluster holding poles at 8 and 11 THz and a source that varies with
    frequency, the frozen value (taken at the first pole) misrepresents the
    second. The per-pole evaluation is exact for a source linear in omega.
    """
    from quatrex.phonon.pole_bridge import source_at_poles

    w = np.linspace(0.0, 20.0, 401)
    z = np.array([8.0 - 0.3j, 11.0 - 0.4j])
    cl = PoleCluster(z=z, u=np.zeros((4, 2), complex),
                     v=np.zeros((4, 2), complex))
    base = np.array([[1.0, 2.0], [3.0, 4.0]])
    src = np.einsum("w,ab->wab", w, base) + 0j

    s_a, s_b = source_at_poles(src, w, cl)
    # Row pole picks Re z_a; column pole picks Re z_b.
    assert np.allclose(_h(s_a), np.array([[8.0, 16.0], [33.0, 44.0]]))
    assert np.allclose(_h(s_b), np.array([[8.0, 22.0], [24.0, 44.0]]))

    # The frozen value would use omega = 8 everywhere, which is wrong by 27%
    # on the second pole -- and the sector carries that error into the bubble.
    frozen = src[int(np.argmin(np.abs(w - 8.0)))]
    assert abs(_h(s_a)[1, 1] - frozen[1, 1]) / abs(_h(s_a)[1, 1]) > 0.2


def test_negative_frequency_poles_use_the_bosonic_mirror():
    """A partner at -Omega takes S(-w) = conj(S(w)), not an extrapolation."""
    from quatrex.phonon.pole_bridge import source_at_poles

    w = np.linspace(0.0, 20.0, 401)
    z = np.array([8.0 - 0.3j, -8.0 - 0.3j])          # a closed pair
    cl = PoleCluster(z=z, u=np.zeros((4, 2), complex),
                     v=np.zeros((4, 2), complex))
    src = np.einsum("w,ab->wab", w, np.ones((2, 2))) * (1 + 2j)
    s_a, _ = source_at_poles(src, w, cl)
    got = _h(s_a)
    assert np.allclose(got[1], np.conj(got[0])), \
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
    s_b = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    r, c = r.ravel(), c.ravel()

    got = _h(pole_keldysh_pf_sparse(w, cl, s_a, s_b, r, c))
    p, coef = leg_partial_fractions(cl, s_a, source_b=s_b)
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
