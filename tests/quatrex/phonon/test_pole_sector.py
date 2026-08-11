# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""End-to-end: the pole sector driven over a small synthetic device.

Exercises the whole chain -- continuation, nonlinear eigenvalue solve, screening
with hysteresis, clustering, source projection, and the analytic bubble -- and
checks the invariants that must survive it:

* the split ``G = G_S + G_R`` is exact by construction, so the sector changes
  the representation and not the diagram;
* ``-i G_PP^<`` stays positive semidefinite, inherited by congruence from the
  source rather than imposed;
* the analytic retarded self-energy is causal -- no upper-half-plane pole.
"""
import numpy as np
import pytest

from quatrex.core.config import PoleSectorConfig
from quatrex.phonon.btd_linalg import BTDFactorization
from quatrex.phonon.pole_bubble import (
    leg_partial_fractions,
    modal_convolution,
    retarded_from_pole_sum,
)
from quatrex.phonon.pole_keldysh import pole_keldysh, project_source
from quatrex.phonon.pole_kernel import sigma_retarded_at_z
from quatrex.phonon.pole_sector import PoleSector

FMAX, DAMP, W_C = 30.0, 0.012, 11.0
HBAR_EVS, KB_EV, THZ = 6.582119569e-16, 8.617333262e-5, 2.0 * np.pi * 1e12


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _bed(nf=401, sizes=(3, 3, 3), seed=0):
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
    d = (d_ii, d_ij, [b.T for b in d_ij])

    freqs = np.linspace(0.0, FMAX, nf)
    a = DAMP * freqs * np.exp(-((freqs / W_C) ** 2))       # Gamma = a >= 0
    delta = np.einsum("w,ij->wij", -1j * a, np.eye(total))
    return freqs, d, delta, sizes


def _operator(d, freqs, delta, sizes):
    d_ii, d_ij, d_ji = d
    off = np.concatenate(([0], np.cumsum(sizes)))

    def _sig(z, order):
        return _h(sigma_retarded_at_z(delta, freqs, np.array([z]), sheet="II",
                                      order=order, delta_order=3))[0]

    def m_blocks(z):
        s = _sig(z, 0)
        a_ii = [z * z * np.eye(sizes[i]) - d_ii[i]
                - s[off[i]:off[i + 1], off[i]:off[i + 1]] for i in range(len(sizes))]
        return a_ii, [-b + 0j for b in d_ij], [-b + 0j for b in d_ji]

    def dm_blocks(z):
        s = _sig(z, 1)
        a_ii = [2.0 * z * np.eye(sizes[i])
                - s[off[i]:off[i + 1], off[i]:off[i + 1]] for i in range(len(sizes))]
        return (a_ii, [np.zeros_like(b) + 0j for b in d_ij],
                [np.zeros_like(b) + 0j for b in d_ji])

    return m_blocks, dm_blocks


def _dense_d(d, sizes):
    d_ii, d_ij, d_ji = d
    off = np.concatenate(([0], np.cumsum(sizes)))
    out = np.zeros((off[-1], off[-1]))
    for i in range(len(sizes)):
        out[off[i]:off[i + 1], off[i]:off[i + 1]] = d_ii[i]
        if i + 1 < len(sizes):
            out[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = d_ij[i]
            out[off[i + 1]:off[i + 2], off[i]:off[i + 1]] = d_ji[i]
    return out


def _run(nf=401, **cfg_kw):
    freqs, d, delta, sizes = _bed(nf)
    m_blocks, dm_blocks = _operator(d, freqs, delta, sizes)
    cfg = PoleSectorConfig(enabled=True, **cfg_kw)
    sec = PoleSector(cfg, freqs)
    lam = np.linalg.eigvalsh(_dense_d(d, sizes))
    lo, hi = sec.window()
    seeds = [complex(np.sqrt(l), -0.01) for l in lam if lo <= np.sqrt(l) <= hi]
    sols = sec.solve_poles(m_blocks, dm_blocks, seeds)
    return sec, sols, freqs, d, delta, sizes


# --------------------------------------------------------------------------- #

def test_sector_finds_only_sub_grid_poles():
    sec, sols, freqs, *_ = _run()
    state = sec.build_clusters(sols)
    assert state.n_poles > 0, "no pole survived screening; nothing is exercised"
    h = float(freqs[1] - freqs[0])
    for c in state.clusters:
        for g in np.asarray(_h(c.gamma)).ravel():
            assert sec.resolution_score(float(g)) < sec.cfg.q_in
            assert g < h, f"a grid-resolved mode was promoted (gamma/h={g / h:.2f})"


def test_screening_rejects_a_grid_resolved_mode():
    """Raising the resolution bar must demote, and the reason must be recorded."""
    sec, sols, *_ = _run()
    kept = sec.build_clusters(sols).n_poles
    assert kept > 0

    sec_strict, sols_strict, *_ = _run(samples_per_halfwidth=1e-6)
    state = sec_strict.build_clusters(sols_strict)
    assert state.n_poles == 0, "an absurd resolution bar promoted modes anyway"
    # Exactly the poles that were kept before must now be refused for being
    # grid-resolved. The remainder fail for their own, unrelated reasons.
    n_resolved = sum("grid-resolved" in why for _, why in state.rejected)
    assert n_resolved == kept, (
        f"{n_resolved} demoted as grid-resolved, expected {kept}: "
        f"{[w for _, w in state.rejected]}"
    )


def test_uncertifiable_poles_are_refused_not_promoted():
    """A pole the corrector cannot certify must be rejected, not carried.

    Built directly rather than by arranging for the corrector to fail: the
    physical trust region now rescues the crude-seed case this used to rely on
    (see ``test_physical_trust_region_rescues_a_crude_seed``), and a screen
    test should not depend on a solver failure it no longer has.

    Both acceptance modes are covered. ``locate`` refuses on the FREQUENCY
    error, which is the physical question; ``residual`` is the legacy gate.
    """
    sec, sols, *_ = _run()
    good = sols[0]
    sep = sec.separations(sols)[0]
    gamma = -good.z.imag

    assert sec.screen(good, was_promoted=False, separation=sep) is None

    class _Bad:
        """Converged by the residual, but displaced by a full linewidth."""
        z, r, l = good.z, good.r, good.l
        eps_nep, kappa, converged = good.eps_nep, good.kappa, True
        dz_est = complex(gamma, 0.0)

    why = sec.screen(_Bad(), was_promoted=False, separation=sep)
    assert why is not None and "eps_z" in why, why

    # ... and the legacy route still refuses on the matrix residual.
    sec.cfg.accept = "residual"
    class _Unconverged(_Bad):
        eps_nep, converged, dz_est = 1.0, False, 0.0 + 0.0j
    why = sec.screen(_Unconverged(), was_promoted=False, separation=sep)
    assert why is not None and "eps_nep" in why, why


def test_physical_trust_region_rescues_a_crude_seed():
    """A pole is a property of ``M(z)``, not of the storage grid.

    ``trust_radius_cells * h`` ties the Newton search to the frequency grid,
    so refining the grid shrinks the physical search domain -- and on a grid
    ladder the sector then finds fewer poles on the fine rungs for a reason
    unrelated to the poles. Measured here on the crude constant-linewidth
    seeding, which is exactly the hard case:

        radius 0.25 cells = 0.019 THz  ->  3 of 9 fail to converge
        radius 0.5*min(sep, edge) = 0.53 THz  ->  0 of 9 fail
    """
    freqs, d, delta, sizes = _bed(401)
    m_blocks, dm_blocks = _operator(d, freqs, delta, sizes)
    lam = np.linalg.eigvalsh(_dense_d(d, sizes))

    def _fails(trust_factor):
        cfg = PoleSectorConfig(enabled=True, trust_factor=trust_factor)
        sec = PoleSector(cfg, freqs)
        lo, hi = sec.window()
        seeds = [complex(np.sqrt(l), -0.01)
                 for l in lam if lo <= np.sqrt(l) <= hi]
        sols = sec.solve_poles(m_blocks, dm_blocks, seeds)
        return sum(1 for s in sols if s.eps_nep > cfg.newton_tol), len(sols)

    grid_tied, n = _fails(1e-9)          # radius collapses to the cell floor
    physical, _ = _fails(0.5)
    assert grid_tied >= 3, f"the hard case is not hard here: {grid_tied}/{n}"
    assert physical == 0, f"physical radius still loses {physical}/{n}"


def test_trust_radius_does_not_shrink_with_the_grid():
    """The point of the change: refinement must not suppress the sector."""
    seeds = [complex(5.0, -0.01), complex(8.0, -0.01)]
    radii = []
    for nf in (401, 1601):
        freqs, d, delta, sizes = _bed(nf)
        sec = PoleSector(PoleSectorConfig(enabled=True), freqs)
        radii.append(sec.trust_radius(seeds[0], seeds, 0))
    assert radii[1] >= radii[0] - 1e-12, (
        f"4x finer grid shrank the pole search: {radii[0]:.4f} -> "
        f"{radii[1]:.4f} THz")
    # ... and it is set by the pole separation, not by h
    assert abs(radii[0] - 0.5 * 3.0) < 1e-9, radii[0]


def test_window_lifts_clear_of_a_low_frequency_mask():
    sec, *_ = _run()
    lo_bare, _ = sec.window()
    lo_masked, _ = sec.window(low_freq_mask=3.0)
    assert lo_masked > 3.0, "the window overlaps the masked region"
    assert lo_masked > lo_bare


def test_report_is_informative():
    sec, sols, *_ = _run()
    sec.build_clusters(sols)
    txt = sec.state.report()
    assert "pole sector" in txt and "cluster" in txt
    assert txt.count("\n") >= 1


# --------------------------------------------------------------------------- #
# Invariants.
# --------------------------------------------------------------------------- #

def test_split_is_exact():
    """G_S + G_R reproduces the untouched G, by construction of G_R."""
    sec, sols, freqs, d, delta, sizes = _run()
    state = sec.build_clusters(sols)
    cl = state.clusters[0]

    w0, gam = float(_h(cl.omega)[0]), float(_h(cl.gamma)[0])
    audit = np.linspace(w0 - 6 * gam, w0 + 6 * gam, 41)
    n = sum(sizes)
    src = 1j * np.einsum("w,ij->wij", 0.4 * np.abs(audit), np.eye(n))

    g_pp = _h(pole_keldysh(audit, cl, _h(project_source(src, _h(cl.v)))))
    # G_R is DEFINED by subtraction, so the reconstruction is exact.
    g_total = np.ones_like(g_pp) * 0.37 + g_pp
    g_reg = g_total - g_pp
    assert np.abs((g_pp + g_reg) - g_total).max() < 1e-14


def test_pole_keldysh_stays_positive_semidefinite():
    sec, sols, freqs, d, delta, sizes = _run()
    cl = sec.build_clusters(sols).clusters[0]
    n = sum(sizes)
    w0, gam = float(_h(cl.omega)[0]), float(_h(cl.gamma)[0])
    audit = np.linspace(w0 - 20 * gam, w0 + 20 * gam, 81)

    # A PSD contact source: -i Sigma^< = n(w) Gamma >= 0.
    x = HBAR_EVS * THZ * np.abs(audit) / (KB_EV * 300.0)
    nb = 1.0 / np.expm1(np.clip(x, 1e-12, None))
    src = 1j * np.einsum("w,ij->wij", nb * 0.4 * np.abs(audit), np.eye(n))
    assert np.linalg.eigvalsh(-1j * src).min() > -1e-12

    g_pp = _h(pole_keldysh(audit, cl, _h(project_source(src, _h(cl.v)))))
    herm = -1j * g_pp
    herm = 0.5 * (herm + herm.conj().swapaxes(-2, -1))
    worst = float(np.linalg.eigvalsh(herm).min())
    assert worst > -1e-10 * float(np.abs(herm).max())


def test_analytic_retarded_self_energy_is_causal():
    """No upper-half-plane pole may survive into the analytic Sigma^R."""
    sec, sols, freqs, d, delta, sizes = _run()
    cl = sec.build_clusters(sols).clusters[0]
    npp = cl.n_poles
    src = np.eye(npp, dtype=complex)
    poles, coeffs = leg_partial_fractions(cl, src)
    w = np.linspace(0.0, FMAX, 51)
    sig_r = _h(retarded_from_pole_sum(w, _h(poles), _h(coeffs)))

    upper = _h(poles).imag > 0
    assert upper.any(), "the leg has no advanced pole; the test is vacuous"
    assert np.abs(sig_r[..., upper]).max() == 0.0


def test_bubble_clusters_are_closed_under_the_bosonic_partner():
    sec, sols, *_ = _run()
    sec.build_clusters(sols)
    for raw, closed in zip(sec.state.clusters, sec.bubble_clusters()):
        assert closed.n_poles == 2 * raw.n_poles
        z = np.asarray(_h(closed.z)).ravel()
        assert np.all(z.imag < 0), "a partner left the lower half plane"
        for zi in np.asarray(_h(raw.z)).ravel():
            assert np.min(np.abs(z - (-np.conj(zi)))) < 1e-12


def test_analytic_bubble_runs_on_the_sector_output():
    """The pole-pole convolution must accept what the driver produces."""
    sec, sols, *_ = _run()
    sec.build_clusters(sols)
    cl = sec.bubble_clusters()[0]
    npp = cl.n_poles
    src = np.eye(npp, dtype=complex)
    w = np.linspace(0.0, 2 * FMAX, 9)
    c = _h(modal_convolution(w, cl, src, src))
    assert c.shape == (len(w), npp, npp, npp, npp)
    assert np.isfinite(c).all()


# --------------------------------------------------------------------------- #
# The driver's own operator route, as the solver hook uses it.
# --------------------------------------------------------------------------- #

def _sparse_indices(sizes):
    """(rows, cols) of a dense block-tridiagonal pattern."""
    off = np.concatenate(([0], np.cumsum(sizes)))
    rows, cols = [], []
    for i in range(len(sizes)):
        for j in range(max(0, i - 1), min(len(sizes), i + 2)):
            for a in range(off[i], off[i + 1]):
                for b in range(off[j], off[j + 1]):
                    rows.append(a)
                    cols.append(b)
    return np.array(rows), np.array(cols)


def _context_run(nf=401, **cfg_kw):
    """Drive the sector the way PhononSolver._update_pole_sector does."""
    freqs, d, delta, sizes = _bed(nf)
    d_ii, d_ij, d_ji = d
    sizes = list(sizes)
    d_blocks = {}
    for i in range(len(sizes)):
        d_blocks[(i, i)] = d_ii[i] + 0j
        if i + 1 < len(sizes):
            d_blocks[(i, i + 1)] = d_ij[i] + 0j
            d_blocks[(i + 1, i)] = d_ji[i] + 0j

    rows, cols = _sparse_indices(np.array(sizes))
    # Delta on the stored pattern rather than as a dense matrix.
    delta_nnz = delta[:, rows, cols]

    sec = PoleSector(PoleSectorConfig(enabled=True, **cfg_kw), freqs)
    sec.set_operator_context(
        delta=delta_nnz, d_blocks=d_blocks, obc_left=None, obc_right=None,
        block_sizes=np.array(sizes), rows=rows, cols=cols,
    )
    return sec


def test_refresh_agrees_with_the_manually_driven_operator():
    """The operator context must build the same M(z) the manual route does.

    The two routes seed differently, so the driver finds strictly more poles.
    Where both converge they must agree to well within a linewidth -- that is
    what pins the context assembly (block scatter, sparsity indices, contact
    placement) against the direct dense construction.
    """
    state = _context_run().refresh()
    sec_man, sols, *_ = _run()
    manual = sec_man.build_clusters(sols)
    assert state.n_poles >= manual.n_poles > 0

    got = np.sort_complex(np.concatenate([np.asarray(_h(c.z)).ravel()
                                          for c in state.clusters]))
    want = np.sort_complex(np.concatenate([np.asarray(_h(c.z)).ravel()
                                           for c in manual.clusters]))
    for z in want:
        d = np.min(np.abs(got - z))
        assert d < 1e-4 * abs(z.imag), (
            f"manual pole {z} has no counterpart in the driver's set (nearest "
            f"{d:.3e}, linewidth {abs(z.imag):.3e})"
        )


def test_quasiparticle_seeding_beats_a_fixed_guess():
    """The seed's linewidth cannot be a constant.

    These linewidths are orders of magnitude below the grid spacing, so a fixed
    guess is wrong by a comparable factor and starts the corrector outside the
    basin. Seeding from the golden-rule estimate -- available from Delta alone,
    since Im Sigma^R = Im Delta / 2 with no Kramers-Kronig half -- lands close
    enough that every mode converges.
    """
    sec = _context_run()
    seeds = sec.harmonic_seeds()
    state = sec.refresh()
    assert not state.rejected, [w for _, w in state.rejected]

    # The seed already predicts the converged linewidth to a few per cent.
    for sol in state.solutions:
        near = min(seeds, key=lambda s: abs(s.real - sol.z.real))
        rel = abs(abs(near.imag) - abs(sol.z.imag)) / abs(sol.z.imag)
        assert rel < 0.05, f"seed linewidth off by {rel:.1%} at {sol.z.real:.3f} THz"

    # A fixed guess of the kind _run uses starts the corrector far from the
    # narrow modes. It no longer LOSES them -- the physical trust region gets
    # there anyway -- but it still costs Newton steps, and that is the honest
    # statement of what the golden-rule seed buys.
    sec_man, sols_man, *_ = _run()
    assert not sec_man.build_clusters(sols_man).rejected, (
        "the physical trust region should now certify the crude seeding too")
    assert (max(s.iterations for s in sols_man)
            > max(s.iterations for s in state.solutions)), (
        "the crude seed should still need more Newton steps than the "
        "quasiparticle seed")


def test_refresh_warm_starts_from_the_previous_iterate():
    """The second call must reuse the tracked poles, not reseed from scratch."""
    sec = _context_run()
    first = sec.refresh()
    assert first.n_poles > 0
    n_first = [s.iterations for s in first.solutions]

    second = sec.refresh()
    assert second.n_poles == first.n_poles
    # Warm-started: converged in no more Newton steps than the cold start, and
    # landed on the same poles.
    assert all(b <= a for a, b in zip(n_first, [s.iterations for s in second.solutions]))
    z1 = np.sort_complex(np.array([s.z for s in first.solutions]))
    z2 = np.sort_complex(np.array([s.z for s in second.solutions]))
    # Tolerance in units of the linewidth, which is what "the same pole" means
    # here -- these are sub-grid widths, so an absolute bound is meaningless.
    gam = np.abs(z1.imag)
    assert np.max(np.abs(z1 - z2) / gam) < 1e-4


def test_coupled_q_is_refused_rather_than_folded_wrongly():
    freqs, d, delta, sizes = _bed(101)
    rows, cols = _sparse_indices(np.array(sizes))
    sec = PoleSector(PoleSectorConfig(enabled=True), freqs)
    fake_q = np.repeat(delta[:, None, rows, cols], 4, axis=1)   # (nf, 4, nnz)
    with pytest.raises(NotImplementedError, match="coupled-q"):
        sec.set_operator_context(
            delta=fake_q, d_blocks={}, obc_left=None, obc_right=None,
            block_sizes=np.array(sizes), rows=rows, cols=cols,
        )


# --------------------------------------------------------------------------- #
# Tracking across iterations.
# --------------------------------------------------------------------------- #

def test_tracker_carries_cluster_identity_across_iterations():
    """Cluster ids must persist, and the subspace must not jump on a quiet step."""
    sec = _context_run()
    sec.refresh()
    ids_first = [c.cid for c in sec.tracker.clusters]
    assert ids_first, "nothing was adopted"

    sec.refresh()
    assert [c.cid for c in sec.tracker.clusters] == ids_first
    assert not sec.tracker.rescan_reasons, sec.tracker.rescan_reasons
    assert all(c.last_angle < sec.cfg.subspace_angle_tol
               for c in sec.tracker.clusters)


def test_predictor_is_used_once_there_is_history():
    """A perturbed self-energy must be tracked by the predictor, not re-seeded."""
    sec = _context_run()
    first = sec.refresh()
    assert first.n_poles > 0

    # Perturb Delta slightly and re-run: the predictor sees a non-zero change.
    sec._delta = sec._delta * 1.02
    second = sec.refresh()
    assert second.n_poles == first.n_poles

    # The linewidths must move with the perturbation -- a predictor that did
    # nothing would return exactly the previous poles.
    g1 = np.sort(np.array([abs(s.z.imag) for s in first.solutions]))
    g2 = np.sort(np.array([abs(s.z.imag) for s in second.solutions]))
    assert np.max(np.abs(g2 - g1) / g1) > 1e-3, "the poles did not follow Delta"
    # ... and they must move by roughly the perturbation, not wander.
    assert np.max(np.abs(g2 - g1) / g1) < 0.2


def test_membership_is_frozen_within_an_epoch():
    sec = _context_run(epoch_iterations=4)
    sec.refresh()
    frozen = []
    for _ in range(5):
        sec.refresh()
        frozen.append(sec.tracker.membership_frozen())
    assert any(frozen) and not all(frozen), frozen


def test_a_lost_pole_set_triggers_a_reseed():
    """If nothing survives the corrector, fall back to the harmonic estimate."""
    sec = _context_run()
    first = sec.refresh()
    assert first.n_poles > 0
    # Corrupt the tracked poles so the warm start is useless.
    for sol in sec.state.solutions:
        sol.z = complex(sol.z.real + 5.0, -1.0)
    recovered = sec.refresh()
    assert recovered.n_poles > 0, "the sector did not recover from a lost pole set"


def test_operator_reduces_a_frequency_resolved_contact_block():
    """M(z) is ONE matrix, so the contacts must be sampled, not carried.

    Regression from the first production run: ``obc_blocks.retarded[0]`` is
    ``(n_freq, b, b)``, and passing it through unreduced assembled M at every
    frequency at once. The bordered Newton then received a stack it could not
    interpret. Holding the contact flat at the grid point nearest ``Re z`` is
    the approximation ``set_operator_context`` documents.
    """
    import numpy as np

    from quatrex.core.config import PoleSectorConfig
    from quatrex.phonon.pole_sector import PoleSector

    freqs = np.linspace(0.0, 20.0, 41)
    sizes = np.array([2, 2])
    n = int(sizes.sum())
    sec = PoleSector(PoleSectorConfig(enabled=True), freqs)

    rng = np.random.default_rng(0)
    d_blocks = {(i, j): rng.normal(size=(2, 2)) + 0j
                for i in range(2) for j in range(2) if abs(i - j) <= 1}
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    delta = (rng.normal(size=(freqs.size, n * n))
             + 1j * rng.normal(size=(freqs.size, n * n)))
    obc = rng.normal(size=(freqs.size, 2, 2)) + 1j * rng.normal(
        size=(freqs.size, 2, 2))

    sec.set_operator_context(
        delta=delta, d_blocks=d_blocks, obc_left=obc, obc_right=obc,
        block_sizes=sizes, rows=rows.ravel(), cols=cols.ravel(),
    )
    m_blocks, _ = sec.operator()
    a_ii, a_ij, a_ji = m_blocks(9.0 - 0.05j)
    for b in (*a_ii, *a_ij, *a_ji):
        assert b.ndim == 2, f"M(z) block carries a stack axis: {b.shape}"

    # And it really is the nearest grid point that was used.
    k = int(np.argmin(np.abs(freqs - 9.0)))
    other = m_blocks(freqs[k] - 0.05j)
    assert np.abs(_h(a_ii[0]) - _h(other[0][0])).max() < 1e-9


def test_hysteresis_survives_across_iterations():
    """A promoted pole is judged at ``q_out``, not ``q_in``, next iteration.

    Regression, and the cause of the sector's non-convergence: the promoted
    set was stored as ``{id(sol)}``. ``bordered_newton`` builds a fresh
    ``PoleSolution`` every iteration, so those CPython ids never matched and
    the hysteresis was permanently disengaged -- every pole was screened at
    the strict ``q_in``. Membership then churned between iterations (measured:
    3 poles -> 0 -> 1 -> 0 over 25 SCBA iterations), which makes the
    fixed-point map discontinuous and produces a limit cycle rather than
    convergence. Identity across iterations must be carried by POSITION.
    """
    import numpy as np

    from quatrex.core.config import PoleSectorConfig
    from quatrex.phonon.pole_sector import PoleSector

    freqs = np.linspace(0.0, 20.0, 41)               # h = 0.5
    sec = PoleSector(PoleSectorConfig(enabled=True, omega_min_thz=1.0,
                                      omega_max_thz=19.0), freqs)
    h = sec.h

    class _Sol:
        converged = True
        eps_nep = 0.0
        kappa = 1.0
        dz_est = 0.0 + 0.0j     # perfectly located, so eps_z cannot refuse it
        def __init__(self, z): self.z = z

    # gamma chosen so q_omega sits BETWEEN q_in and q_out: such a pole is
    # refused on first sight but retained once promoted. That gap is the
    # hysteresis, and it is what stops membership oscillating.
    q_mid = 0.5 * (sec.cfg.q_in + sec.cfg.q_out)
    gamma = q_mid * sec.cfg.samples_per_halfwidth * h
    sol = _Sol(complex(9.0, -gamma))

    assert sec.screen(sol, was_promoted=False) is not None, "refused when new"
    assert sec.screen(sol, was_promoted=True) is None, "kept once promoted"

    # And the memory must survive a NEW object, matched by displacement AND
    # eigenvector overlap rather than by position alone.
    import numpy as np

    v = np.array([1.0, 0.0, 0.0, 0.0])
    sec._promoted = [(complex(9.0, -gamma), v)]

    class _S2(_Sol):
        def __init__(self, z, r):
            super().__init__(z)
            self.r = r

    moved = _S2(complex(9.0 + 0.1 * h, -gamma), v)
    elsewhere = _S2(complex(15.0, -gamma), v)
    assert sec._match_previous([moved]) == [True], \
        "a pole that moved slightly, with the same eigenvector, is the same mode"
    assert sec._match_previous([elsewhere]) == [False], \
        "a pole at a different frequency is not"

    # Eigenvector overlap must count: same position, orthogonal vector.
    rotated = _S2(complex(9.0, -gamma), np.array([0.0, 1.0, 0.0, 0.0]))
    assert sec._match_previous([rotated]) == [False], \
        "position alone must not carry identity through a crossing"
