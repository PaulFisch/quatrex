# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The batched pole solve must return the pole set the scalar one did.

The pole sector used to correct one candidate at a time, at ~4,800 Python calls
per candidate and 187 s per SCBA iteration on Si against the bubble's 7 s
(``phonon/docs/pole_solve_batching.md``). Batching it is a performance change
that must not be a physics change, and "must not" needs gates rather than
inspection, because the failure it risks is silent: a slightly different pole
set still looks like a pole set.

The scalar path is kept as the reference these test against, not as a
production option. Three things are separated deliberately:

* the block LAYOUT, against the scatter it replaces;
* the batched OPERATOR, against the same operator evaluated one probe at a
  time -- this is where the pinned fit anchor can go wrong;
* the batched NEWTON, against the per-candidate iteration, driven through the
  same scalar operator so that nothing else varies.

Bit-identity is NOT the gate, and cannot be: the batched reductions replace
``xp.vdot`` with a summation whose order differs, and Python's
``complex.__pow__`` differs from numpy's in the last ulp at exponent two. What
is gated is that the poles land in the same place to far below a linewidth,
and that every DECISION -- converged, iterations, the refusal reason -- is
identical.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# The synthetic device and the manual scalar operator live next door; they are
# the reference this module tests against, so they are shared rather than
# copied.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quatrex.core.config import PoleSectorConfig
from quatrex.phonon.pole_nevp import bordered_newton, bordered_newton_batch
from quatrex.phonon.pole_probe import BlockLayout, nnz_to_blocks
from quatrex.phonon.pole_sector import PoleSector

from test_pole_sector import (
    _bed, _dense_d, _h, _operator, _sparse_indices,
)


def _driver_bed(nf=401):
    """The sector driven through its own operator context, as the solver does."""
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
    sec = PoleSector(PoleSectorConfig(enabled=True), freqs)
    sec.set_operator_context(
        delta=delta[:, rows, cols], d_blocks=d_blocks, obc_left=None,
        obc_right=None, block_sizes=np.array(sizes), rows=rows, cols=cols,
    )
    return sec, freqs


# --------------------------------------------------------------------------- #
# The layout
# --------------------------------------------------------------------------- #

def test_block_layout_reproduces_the_scatter_it_replaces():
    """Including the two cases that make it more than a reindexing."""
    rng = np.random.default_rng(3)
    sizes = np.array([3, 2, 4])
    off = np.concatenate(([0], np.cumsum(sizes)))
    n = int(off[-1])
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    # a (block 0, block 2) entry exists in this pattern -> out of band
    assert ((rows < off[1]) & (cols >= off[2])).any(), "the test is vacuous"
    rows = np.append(rows, rows[5])
    cols = np.append(cols, cols[5])
    vals = (rng.standard_normal((4, rows.size))
            + 1j * rng.standard_normal((4, rows.size)))

    ref = nnz_to_blocks(vals, rows, cols, sizes, band=1)
    lay = BlockLayout(rows, cols, sizes, band=1)
    got = lay.block_dict(lay.gather(vals))

    assert set(got) == set(ref)
    for key in ref:
        assert np.array_equal(_h(got[key]), _h(ref[key])), key


def test_block_layout_blocks_are_views_of_one_buffer():
    """The point of the layout: materialising the blocks costs no kernel."""
    sizes = np.array([3, 2, 4])
    n = int(sizes.sum())
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    lay = BlockLayout(rows.ravel(), cols.ravel(), sizes, band=1)

    flat = np.zeros((5, lay.total), dtype=complex)
    a_ii, a_ij, a_ji = lay.blocks(flat)
    a_ii[1][...] = 7.0
    where = next(sl for ij, sl, _, _ in lay.slices if ij == (1, 1))
    assert (_h(flat)[:, where] == 7.0).all(), (
        "the block was a copy, not a view of the buffer")
    assert _h(flat).sum() == 7.0 * (sizes[1] ** 2) * 5, (
        "writing one block reached outside its own slice")

    # and the diagonal index really addresses the operator's diagonal
    d = np.zeros((5, lay.total), dtype=complex)
    d[..., _h(lay.diag)] = 1.0
    for i, blk in enumerate(lay.blocks(d)[0]):
        eye = np.broadcast_to(np.eye(int(sizes[i])), (5, sizes[i], sizes[i]))
        assert np.array_equal(_h(blk), eye)


# --------------------------------------------------------------------------- #
# The operator
# --------------------------------------------------------------------------- #

def test_batched_operator_equals_the_operator_probed_one_at_a_time():
    """Each candidate must see the operator its OWN anchor defines."""
    sec, freqs = _driver_bed()
    h = float(freqs[1] - freqs[0])

    rng = np.random.default_rng(7)
    ordinary = rng.uniform(2.0, 25.0, 6)
    # round(w_c/h) flips exactly at a half-integer number of cells
    boundary = h * np.array([16.5, 16.5 + 1e-9, 16.5 - 1e-9, 40.5, 4.5])
    z = np.concatenate([ordinary, boundary]) - 1j * rng.uniform(1e-6, 5e-2, 11)

    m_blocks, dm_blocks = sec.operator()
    sec._set_fit_anchors(z.real)
    got = (m_blocks(z), dm_blocks(z))

    # one probe at a time, each carrying only its own anchor
    solo = []
    for k in range(z.size):
        sec._set_fit_anchors(z[k:k + 1].real)
        solo.append((m_blocks(z[k:k + 1]), dm_blocks(z[k:k + 1])))
    sec._set_fit_anchors(None)

    for which, tag in enumerate(("M", "M'")):
        for part in range(3):
            for b in range(len(got[which][part])):
                batch = _h(got[which][part][b])
                one = np.concatenate([_h(s[which][part][b]) for s in solo])
                assert batch.shape == one.shape, (tag, part, b)
                rel = np.abs(batch - one) / np.maximum(np.abs(one), 1e-300)
                assert rel.max() < 1e-12, f"{tag} block {part},{b}: {rel.max():.2e}"


def test_a_shared_anchor_would_have_been_visible():
    """The previous test is only meaningful if the anchor changes the answer.

    Guards against the gate passing because the anchor is ignored.
    """
    sec, freqs = _driver_bed()
    h = float(freqs[1] - freqs[0])
    z = np.array([h * 16.5 - 0.01j])

    m_blocks, _ = sec.operator()
    sec._set_fit_anchors(z.real)
    own = _h(m_blocks(z)[0][0]).copy()
    sec._set_fit_anchors(z.real + 2.0 * h)          # a neighbouring stencil
    other = _h(m_blocks(z)[0][0])
    sec._set_fit_anchors(None)

    assert np.abs(own - other).max() > 0.0, (
        "the anchor does not reach the assembled operator, so the equivalence "
        "test above proves nothing")


# --------------------------------------------------------------------------- #
# The Newton iteration
# --------------------------------------------------------------------------- #

def _manual_bed(nf=401):
    freqs, d, delta, sizes = _bed(nf)
    m_blocks, dm_blocks = _operator(d, freqs, delta, sizes)
    sec = PoleSector(PoleSectorConfig(enabled=True), freqs)
    lam = np.linalg.eigvalsh(_dense_d(d, sizes))
    lo, hi = sec.window()
    seeds = [complex(np.sqrt(l), -0.01) for l in lam if lo <= np.sqrt(l) <= hi]
    return sec, m_blocks, dm_blocks, seeds


def test_batched_newton_matches_the_per_candidate_iteration():
    """Same operator both ways, so only the batching of the iteration varies."""
    sec, m_blocks, dm_blocks, seeds = _manual_bed()
    assert len(seeds) > 1, "a batch of one does not test batching"

    together = sec.solve_poles(m_blocks, dm_blocks, seeds)
    apart = [
        bordered_newton(m_blocks, dm_blocks, z0,
                        tol=sec.cfg.newton_tol,
                        max_iter=sec.cfg.newton_max_iterations,
                        trust_radius=sec.trust_radius(z0, seeds, k))
        for k, z0 in enumerate(seeds)
    ]

    assert [s.iterations for s in together] == [s.iterations for s in apart]
    assert [s.converged for s in together] == [s.converged for s in apart]
    for a, b in zip(together, apart):
        gamma = abs(b.z.imag)
        assert abs(a.z - b.z) < 1e-9 * gamma, (
            f"{a.z} vs {b.z}: moved by {abs(a.z - b.z) / gamma:.2e} linewidths")
        # kappa is a genuine O(1) quantity, unlike the residuals
        assert abs(a.kappa - b.kappa) <= 1e-10 * abs(b.kappa)


def test_the_acceptance_decision_is_unchanged_by_batching():
    """What the sector DOES with the poles, not just where they are."""
    sec, m_blocks, dm_blocks, seeds = _manual_bed()
    together = sec.solve_poles(m_blocks, dm_blocks, seeds)
    apart = [
        bordered_newton(m_blocks, dm_blocks, z0,
                        tol=sec.cfg.newton_tol,
                        max_iter=sec.cfg.newton_max_iterations,
                        trust_radius=sec.trust_radius(z0, seeds, k))
        for k, z0 in enumerate(seeds)
    ]
    sep_a, sep_b = sec.separations(together), sec.separations(apart)
    for k in range(len(seeds)):
        why_a = sec.screen(together[k], False, sep_a[k])
        why_b = sec.screen(apart[k], False, sep_b[k])
        assert (why_a is None) == (why_b is None), (why_a, why_b)
        if why_a is not None:
            assert why_a.split("=")[0] == why_b.split("=")[0]
        # and the margin to the threshold is enormous, which is WHY the
        # relative wander of eps_z is harmless
        assert sec.locate_error(together[k], sep_a[k]) < 1e-6 * sec.cfg.locate_tol


def test_a_converged_candidate_stops_advancing():
    """The active mask must freeze a candidate, not merely stop counting it.

    All candidates run to ``max_iter``; a finished one has its update masked to
    zero. If the mask leaked, extra steps would keep moving ``z`` -- so the
    same solve with a larger budget must return the same poles.
    """
    sec, m_blocks, dm_blocks, seeds = _manual_bed()
    short = sec.solve_poles(m_blocks, dm_blocks, seeds)
    sec.cfg.newton_max_iterations = 3 * sec.cfg.newton_max_iterations
    long = sec.solve_poles(m_blocks, dm_blocks, seeds)
    for a, b in zip(short, long):
        if a.iterations < 8:                 # it left the loop on its own
            assert a.z == b.z, f"{a.z} moved to {b.z} on a longer budget"
            assert a.iterations == b.iterations


def test_iterations_reports_where_the_candidate_stopped():
    """Not ``max_iter`` for everyone, which a masked loop reports by default."""
    sec, m_blocks, dm_blocks, seeds = _manual_bed()
    sols = sec.solve_poles(m_blocks, dm_blocks, seeds)
    it = [s.iterations for s in sols]
    assert all(1 <= i <= sec.cfg.newton_max_iterations for i in it), it
    assert len(set(it)) > 1, (
        f"every candidate reports {it[0]} steps; the per-candidate count was "
        "lost and the batch is reporting its own length")


def test_a_candidate_is_unaffected_by_who_it_is_batched_with():
    """A pole is a property of its operator, not of its neighbours in the solve.

    This is what the shared power-iteration start vector in ``btd_norm2`` buys:
    without it every candidate draws a different vector, so ``||M||`` -- and
    through it ``eps_nep`` and ``kappa`` -- depends on the batch size.
    """
    sec, m_blocks, dm_blocks, seeds = _manual_bed()
    full = sec.solve_poles(m_blocks, dm_blocks, seeds)
    alone = sec.solve_poles(m_blocks, dm_blocks, seeds[:1])
    a, b = alone[0], full[0]
    assert a.z == b.z
    assert a.kappa == b.kappa
    assert a.eps_nep == b.eps_nep


def test_trust_radii_agrees_with_the_per_seed_form():
    """Two implementations of one formula; they must not drift apart."""
    sec, _, _, seeds = _manual_bed()
    got = np.asarray(_h(sec.trust_radii(seeds)))
    want = np.array([sec.trust_radius(z0, seeds, k)
                     for k, z0 in enumerate(seeds)])
    assert np.array_equal(got, want)

    # a single seed has no competing scale at all -> unbounded, not the floor
    one = [seeds[0]]
    assert np.isinf(float(_h(sec.trust_radii(one))[0]))
    assert np.isinf(sec.trust_radius(seeds[0], one, 0))


def test_solve_poles_handles_an_empty_seed_set():
    sec, m_blocks, dm_blocks, _ = _manual_bed()
    assert sec.solve_poles(m_blocks, dm_blocks, []) == []


# --------------------------------------------------------------------------- #
# The cache that Delta must not outlive
# --------------------------------------------------------------------------- #

def test_changing_delta_invalidates_everything_derived_from_it():
    """Delta is rebuilt every SCBA iteration precisely so it cannot go stale."""
    sec, _ = _driver_bed()
    z = np.array([9.0 - 0.01j])
    before = _h(sec.continue_sigma(z, order=0)).copy()

    sec._delta = sec._delta * 2.0
    after = _h(sec.continue_sigma(z, order=0))
    assert np.abs(after - 2.0 * before).max() < 1e-9 * np.abs(before).max()

    # ... and the flat layout path must agree with the sparse-pattern path,
    # which is the same contraction reached two different ways.
    blocks = nnz_to_blocks(sec.continue_sigma(z, order=0), sec._rows, sec._cols,
                           sec._block_sizes, band=1)
    got = sec._layout.block_dict(sec._continue_flat(z, 0))
    for key, b in blocks.items():
        assert np.abs(_h(got[key]) - _h(b)).max() < 1e-12, key


# --------------------------------------------------------------------------- #
# Batching across q
# --------------------------------------------------------------------------- #

def _q_sectors(nq=4, nf=201, seed0=0):
    """``nq`` INDEPENDENT pole problems sharing a grid and a block layout."""
    from quatrex.phonon.pole_probe import BlockLayout

    sizes = (3, 3, 3)
    rows, cols = _sparse_indices(np.array(sizes))
    layout = BlockLayout(rows, cols, np.array(sizes), band=1)
    sectors = []
    for q in range(nq):
        freqs, d, delta, _ = _bed(nf, sizes=sizes, seed=seed0 + q)
        d_ii, d_ij, d_ji = d
        d_blocks = {}
        for i in range(len(sizes)):
            d_blocks[(i, i)] = d_ii[i] + 0j
            if i + 1 < len(sizes):
                d_blocks[(i, i + 1)] = d_ij[i] + 0j
                d_blocks[(i + 1, i)] = d_ji[i] + 0j
        sec = PoleSector(PoleSectorConfig(enabled=True), freqs)
        sec.set_operator_context(
            delta=delta[:, rows, cols] * (1.0 + 0.3 * q), d_blocks=d_blocks,
            obc_left=None, obc_right=None, block_sizes=np.array(sizes),
            rows=rows, cols=cols, layout=layout,
        )
        sectors.append(sec)
    return sectors


def _pole_set(state):
    return np.sort_complex(np.array([complex(s.z) for s in state.solutions]))


def test_q_batching_returns_the_per_q_answer():
    """Solving four q together must give what solving them one at a time does.

    The q are independent problems; sharing a corrector is a statement about
    kernel launches, not about the physics, and this is what says so.
    """
    from quatrex.phonon.pole_sector import refresh_many

    apart = [sec.refresh() for sec in _q_sectors()]
    together = refresh_many(_q_sectors())

    assert len(apart) == len(together)
    for k, (a, b) in enumerate(zip(apart, together)):
        assert a.n_poles == b.n_poles, f"q {k}: {a.n_poles} vs {b.n_poles}"
        za, zb = _pole_set(a), _pole_set(b)
        assert za.size == zb.size
        if za.size:
            gamma = np.abs(za.imag)
            assert (np.abs(za - zb) / gamma).max() < 1e-9, f"q {k}"
        assert ([s.iterations for s in a.solutions]
                == [s.iterations for s in b.solutions]), f"q {k}"
        assert ([w for _, w in a.rejected] == [w for _, w in b.rejected]), f"q {k}"


def test_q_batching_survives_unequal_candidate_counts():
    """The batch is a rectangle; the q are not."""
    from quatrex.phonon.pole_sector import refresh_many

    sectors = _q_sectors(nq=3)
    # Force different windows so the q disagree on how many candidates exist.
    for k, sec in enumerate(sectors):
        sec.cfg.omega_max_thz = 30.0 - 8.0 * k
    counts = [len(sec.harmonic_seeds()) for sec in sectors]
    assert len(set(counts)) > 1, f"the q all have {counts[0]} candidates"

    apart = [sec.refresh() for sec in sectors]
    sectors2 = _q_sectors(nq=3)
    for k, sec in enumerate(sectors2):
        sec.cfg.omega_max_thz = 30.0 - 8.0 * k
    together = refresh_many(sectors2)

    for k, (a, b) in enumerate(zip(apart, together)):
        assert a.n_poles == b.n_poles, f"q {k}: {a.n_poles} vs {b.n_poles}"
        za, zb = _pole_set(a), _pole_set(b)
        if za.size:
            assert (np.abs(za - zb) / np.abs(za.imag)).max() < 1e-9, f"q {k}"


def test_q_batch_size_does_not_change_the_answer():
    """q_batch is a memory knob. Chunking must be invisible in the result."""
    from quatrex.phonon.pole_sector import refresh_many

    ref = refresh_many(_q_sectors(nq=6))
    for chunk in (1, 2, 4, 6):
        secs = _q_sectors(nq=6)
        got = []
        for lo in range(0, len(secs), chunk):
            got.extend(refresh_many(secs[lo:lo + chunk]))
        for k, (a, b) in enumerate(zip(ref, got)):
            assert a.n_poles == b.n_poles, (chunk, k)
            za, zb = _pole_set(a), _pole_set(b)
            if za.size:
                assert (np.abs(za - zb) / np.abs(za.imag)).max() < 1e-9, (chunk, k)


def test_q_batch_refuses_sectors_that_do_not_share_a_layout():
    """A silently mismatched layout would scatter one q's Sigma into another's."""
    from quatrex.phonon.pole_sector import PoleQBatch

    sectors = _q_sectors(nq=2)
    sectors[1]._set_layout()           # a fresh, equal-but-distinct layout
    with pytest.raises(ValueError, match="do not share a block layout"):
        PoleQBatch(sectors, 3)


def test_each_q_keeps_its_own_tracker_and_promoted_set():
    """Only the SOLVE is shared. Identity is per q and must stay that way."""
    from quatrex.phonon.pole_sector import refresh_many

    sectors = _q_sectors(nq=3)
    refresh_many(sectors)
    promoted = [{complex(z) for z, _ in sec._promoted} for sec in sectors]
    assert all(p for p in promoted), "a q ended up with no promoted poles"
    assert promoted[0] != promoted[1], (
        "two unrelated q share a promoted set; the tracker was pooled")
    assert len({id(sec.tracker) for sec in sectors}) == len(sectors)


def test_q_batching_samples_each_q_contact_at_its_own_anchor():
    """The contacts are per q AND per candidate, and both indices must land."""
    from quatrex.phonon.pole_sector import PoleQBatch, refresh_many

    nq = 3
    sectors = _q_sectors(nq=nq)
    rng = np.random.default_rng(11)
    n_freq = int(sectors[0].freqs.shape[0])
    obc = [0.4 * (rng.normal(size=(n_freq, 3, 3))
                  + 1j * rng.normal(size=(n_freq, 3, 3))) for _ in range(nq)]
    for sec, o in zip(sectors, obc):
        sec._obc = (o, None)

    z = np.array([[7.0 - 0.01j, 13.0 - 0.02j],
                  [4.0 - 0.03j, 19.0 - 0.01j],
                  [9.0 - 0.02j, 22.0 - 0.05j]])
    batch = PoleQBatch(sectors, 2)
    batch.set_anchors(z.real)
    got, none_right = batch._obc_at()
    assert none_right is None
    assert got.shape == (nq * 2, 3, 3)

    for q in range(nq):
        for p in range(2):
            k = int(np.argmin(np.abs(np.asarray(_h(sectors[q].freqs))
                                     - z[q, p].real)))
            assert np.array_equal(_h(got[q * 2 + p]), _h(obc[q])[k]), (q, p)

    # end to end: contacts on, batched q still equals per-q
    apart = [sec.refresh() for sec in sectors]
    sectors2 = _q_sectors(nq=nq)
    for sec, o in zip(sectors2, obc):
        sec._obc = (o, None)
    together = refresh_many(sectors2)
    for k, (a, b) in enumerate(zip(apart, together)):
        assert a.n_poles == b.n_poles, f"q {k}: {a.n_poles} vs {b.n_poles}"
        za, zb = _pole_set(a), _pole_set(b)
        if za.size:
            assert (np.abs(za - zb) / np.abs(za.imag)).max() < 1e-9, f"q {k}"
    assert any(a.n_poles for a in apart), "contacts killed every pole; vacuous"
