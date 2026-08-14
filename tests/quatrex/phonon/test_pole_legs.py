# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The leg construction, batched over every q and every cluster.

``_build_pole_keldysh`` states the leg one cluster at a time and is what the
physics tests read. Production drove it through a Python loop of
``n_q * n_clusters`` iterations over routines that themselves looped over pole
columns and pole pairs: 6.85 million Python calls and 33 s per SCBA iteration
on Si, against a bubble of 7.4 s. :mod:`~quatrex.phonon.pole_legs` does the
same arithmetic in one pass.

Two things are pinned here and they are different questions.

*Does it compute the same leg?* Against the per-cluster path, on beds with
ragged cluster counts, ragged pole counts and non-uniform blocks -- the shapes
where a padded batch can quietly mix one q's poles into another's.

*Is the work actually bounded?* The point of the exercise is that the number of
Python calls stops depending on the size of the problem. That is a property no
correctness test can see, so it is asserted directly: grow the q count, the
cluster count and the pole count, and the call count must not follow.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from qttools import xp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quatrex.phonon.pole_keldysh import PoleCluster
from quatrex.phonon.pole_probe import BlockLayout
from quatrex.phonon.solver import PhononSolver


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _pattern(sizes):
    off = np.concatenate(([0], np.cumsum(sizes)))
    rows, cols = [], []
    for i in range(len(sizes)):
        for j in range(max(0, i - 1), min(len(sizes), i + 2)):
            rr, cc = np.meshgrid(np.arange(off[i], off[i + 1]),
                                 np.arange(off[j], off[j + 1]), indexing="ij")
            rows.append(rr.ravel())
            cols.append(cc.ravel())
    return np.concatenate(rows), np.concatenate(cols), off


class _Buf:
    def __init__(self, data, rows, cols):
        self.data, self.rows, self.cols = data, rows, cols


class _Sector:
    """Just enough of PoleSector for the leg builder: the closed cluster set."""

    def __init__(self, legs):
        self._legs = legs

    def bubble_clusters(self):
        return self._legs


def _bed(sizes=(3, 3, 3), nk=(2, 2), poles=((2, 1), (3,), (1, 2, 1), (2,)),
         n_freq=13, seed=0, contacts=True):
    """A q-resolved solver stub with real arrays and ragged clusters."""
    sizes = np.array(sizes)
    rows, cols, off = _pattern(sizes)
    n_dof, nnz = int(sizes.sum()), rows.size
    n_q = int(np.prod(nk)) if nk else 1
    assert len(poles) == n_q
    rng = np.random.default_rng(seed)

    def arr(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    tail = nk if nk else ()
    solver = object.__new__(PhononSolver)
    solver.block_sizes = sizes
    solver.local_frequencies = np.linspace(0.0, 12.0, n_freq)
    solver.local_frequency_weights = np.full(n_freq, 12.0 / (n_freq - 1))
    solver._pole_cfg = types.SimpleNamespace(leg="congruence",
                                             source_fit_tol=1e9)
    solver.config = types.SimpleNamespace(
        phonon=types.SimpleNamespace(sse_low_freq_mask_thz=0.0))
    solver._pole_layout = BlockLayout(rows, cols, sizes, band=1)

    b0, bN = int(sizes[0]), int(sizes[-1])
    if contacts:
        solver.obc_blocks = types.SimpleNamespace(
            lesser=[arr(n_freq, *tail, b0, b0), arr(n_freq, *tail, bN, bN)],
            greater=[arr(n_freq, *tail, b0, b0), arr(n_freq, *tail, bN, bN)])
    else:
        solver.obc_blocks = types.SimpleNamespace(lesser=[None, None],
                                                 greater=[None, None])

    sse_l = _Buf(arr(n_freq, *tail, nnz), rows, cols)
    sse_g = _Buf(arr(n_freq, *tail, nnz), rows, cols)
    g_r = _Buf(arr(n_freq, *tail, nnz), rows, cols)
    g_l = _Buf(arr(n_freq, *tail, nnz), rows, cols)

    states, sectors = [], []
    for q in range(n_q):
        legs = []
        for npp in poles[q]:
            z = rng.uniform(2.0, 9.0, npp) - 1j * rng.uniform(1e-3, 3e-2, npp)
            z = np.concatenate([z, -np.conj(z)])          # bosonic closure
            legs.append(PoleCluster(
                z=xp.asarray(z), u=xp.asarray(arr(n_dof, z.size)),
                v=xp.asarray(arr(n_dof, z.size)), label=f"q{q}c{len(legs)}"))
        st = types.SimpleNamespace(
            clusters=[object()] * len(legs), legs=None, source_fit=[],
            source_lesser=[], source_greater=[], c_lesser=[], c_greater=[],
            pf_lesser=[], pf_greater=[], residue_sum=[], mixed_fit=[],
            g_pp_lesser=None, g_pp_greater=None)
        states.append(st)
        sectors.append(_Sector(legs))
    return solver, sse_l, sse_g, g_r, g_l, states, sectors, n_q


def _reference(solver, sse_l, sse_g, g_r, g_l, states, sectors, nk):
    """The per-cluster path, one q at a time -- the code being replaced."""
    out_l, out_g = [], []
    for q, (st, sec) in enumerate(zip(states, sectors)):
        idx = tuple(int(i) for i in np.unravel_index(q, nk)) if nk else ()
        solver.pole_state = st
        solver._build_pole_keldysh(sse_l, sse_g, g_retarded=g_r, g_lesser=g_l,
                                   q_idx=idx, sector=sec)
        out_l.append(st.g_pp_lesser)
        out_g.append(st.g_pp_greater)
    return out_l, out_g


def _rel(got, want):
    got, want = np.asarray(_h(got)), np.asarray(_h(want))
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-300))


@pytest.mark.parametrize(
    "sizes, nk, poles, contacts",
    [
        ((3, 3, 3), (2, 2), ((2, 1), (3,), (1, 2, 1), (2,)), True),
        ((3, 3, 3), (2, 2), ((2, 1), (3,), (1, 2, 1), (2,)), False),
        ((3, 2, 4), (3,), ((2,), (1, 3), (2, 2)), True),
        ((5,), (2,), ((3,), (1, 2)), True),
        ((2, 2, 2, 2), (2,), ((1, 1, 2), (3,)), True),
        ((4, 4), (), ((2, 3),), True),
    ],
)
def test_batched_legs_equal_the_per_cluster_path(sizes, nk, poles, contacts):
    """Same leg, whatever the q count, cluster count and pole count."""
    bed = _bed(sizes=sizes, nk=nk, poles=poles, contacts=contacts)
    solver, sse_l, sse_g, g_r, g_l, states, sectors, n_q = bed
    ref_l, ref_g = _reference(solver, sse_l, sse_g, g_r, g_l, states, sectors,
                              nk)
    ref_src = [list(st.source_lesser) for st in states]
    ref_fit = [list(st.source_fit) for st in states]
    ref_c = [list(st.c_lesser) for st in states]

    bed2 = _bed(sizes=sizes, nk=nk, poles=poles, contacts=contacts)
    solver2, sse_l2, sse_g2, g_r2, g_l2, states2, sectors2, _ = bed2
    got_l, got_g = solver2._build_pole_legs(
        sse_l2, sse_g2, g_r2, states2, sectors2, np.arange(n_q))

    for q in range(n_q):
        assert _rel(got_l[q].reshape(ref_l[q].shape), ref_l[q]) < 1e-11, q
        assert _rel(got_g[q].reshape(ref_g[q].shape), ref_g[q]) < 1e-11, q
        assert len(states2[q].source_lesser) == len(ref_src[q])
        for m in range(len(ref_src[q])):
            assert _rel(states2[q].source_lesser[m], ref_src[q][m]) < 1e-11
            assert abs(states2[q].source_fit[m] - ref_fit[q][m]) \
                <= 1e-10 * max(abs(ref_fit[q][m]), 1e-300)
            # (c_sr, c_rs, c_ss) per leg, each trimmed on the axes that carry
            # poles and left alone on the one that carries degrees of freedom
            got_c = states2[q].c_lesser[m]
            want_c = ref_c[q][m]
            assert len(got_c) == 3
            for i in range(3):
                assert np.asarray(_h(got_c[i])).shape \
                    == np.asarray(_h(want_c[i])).shape, (q, m, i)
                assert _rel(got_c[i], want_c[i]) < 1e-11, (q, m, i)


def test_each_q_gets_its_own_self_energy_and_its_own_contacts():
    """A padded batch is exactly where one q's data can leak into another's.

    Perturbing ONE q's self-energy, and separately one q's contact block, must
    move that q's leg and no other. Equality of the whole stack cannot see a
    transposed q index if every q happens to be built from the same random
    draw, so the perturbation is the test.
    """
    kw = dict(sizes=(3, 3, 3), nk=(2, 2), poles=((2, 1), (3,), (1, 2, 1), (2,)))
    base = _bed(**kw)
    got0 = base[0]._build_pole_legs(*base[1:4], base[5], base[6],
                                    np.arange(base[7]))[0]
    for target in (1, 3):
        for what in ("sigma", "contact"):
            bed = _bed(**kw)
            solver, sse_l, sse_g, g_r, g_l, states, sectors, n_q = bed
            idx = np.unravel_index(target, kw["nk"])
            if what == "sigma":
                sse_l.data[(slice(None),) + idx] *= 1.5
            else:
                solver.obc_blocks.lesser[0][(slice(None),) + idx] *= 1.5
            got = solver._build_pole_legs(sse_l, sse_g, g_r, states, sectors,
                                          np.arange(n_q))[0]
            moved = [q for q in range(n_q)
                     if _rel(got[q], got0[q]) > 1e-12]
            assert moved == [target], (
                f"perturbing {what} at q={target} moved q={moved}")


def test_a_q_with_no_poles_is_a_no_op():
    """An empty q must neither crash the batch nor disturb its neighbours."""
    kw = dict(sizes=(3, 3, 3), nk=(2,), poles=((2,), (1,)))
    bed = _bed(**kw)
    solver, sse_l, sse_g, g_r, g_l, states, sectors, n_q = bed
    full = solver._build_pole_legs(sse_l, sse_g, g_r, states, sectors,
                                   np.arange(n_q))[0]

    bed = _bed(**kw)
    solver, sse_l, sse_g, g_r, g_l, states, sectors, n_q = bed
    states[1].clusters = []
    sectors[1] = _Sector([])
    part = solver._build_pole_legs(sse_l, sse_g, g_r, states, sectors,
                                   np.arange(n_q))[0]
    assert _rel(part[0], full[0]) < 1e-13, "an empty q disturbed a live one"
    assert float(np.abs(_h(part[1])).max()) == 0.0


def test_no_poles_anywhere_leaves_the_channel_unset():
    bed = _bed(sizes=(3, 3, 3), nk=(2,), poles=((1,), (1,)))
    solver, sse_l, sse_g, g_r, g_l, states, sectors, n_q = bed
    for st in states:
        st.clusters = []
    sectors = [_Sector([]) for _ in sectors]
    assert solver._build_pole_legs(sse_l, sse_g, g_r, states, sectors,
                                   np.arange(n_q)) == (None, None)


# --------------------------------------------------------------------------- #
# The invariant the batching exists for
# --------------------------------------------------------------------------- #

def _profile(**kw):
    import cProfile
    import pstats

    bed = _bed(**kw)
    solver, sse_l, sse_g, g_r, g_l, states, sectors, n_q = bed
    pr = cProfile.Profile()
    pr.enable()
    solver._build_pole_legs(sse_l, sse_g, g_r, states, sectors,
                            np.arange(n_q))
    pr.disable()
    st = pstats.Stats(pr)
    per_name = {}
    for (_, _, name), (_, nc, *_rest) in st.stats.items():
        per_name[name] = per_name.get(name, 0) + nc
    return st.total_calls, per_name


# Everything that actually dispatches work: the leg contraction, the band
# apply that carries the block-tridiagonal operator, the band gathers, and the
# local fit behind the source diagnostic.
_DISPATCHERS = ("congruence_legs", "source_fit", "apply_band", "band",
                "band_neighbours", "to_blocks", "from_blocks", "unband",
                "band_transpose", "_sectors", "_pole_factors")


def test_dispatched_work_does_not_grow_with_the_problem():
    """The invariant the batching exists for, in the unit that costs.

    Every axis the leg used to loop over -- q, clusters, poles, frequencies --
    is varied by a large factor, and the number of times a contraction is
    ISSUED must not move at all. Not smaller: identical. A loop that has only
    been shortened is a loop that comes back on the next device.
    """
    _, small = _profile(sizes=(3, 3, 3), nk=(2,), poles=((1,), (1,)))
    base = {k: small.get(k, 0) for k in _DISPATCHERS}
    assert sum(base.values()) > 0, "the probe names no longer match the code"

    cases = {
        "16x the q": dict(sizes=(3, 3, 3), nk=(4, 4),
                          poles=tuple(((1,),) * 16)),
        "4x the clusters": dict(sizes=(3, 3, 3), nk=(2,),
                                poles=((1, 1, 1, 1), (1, 1, 1, 1))),
        "8x the poles": dict(sizes=(3, 3, 3), nk=(2,), poles=((8,), (8,))),
        "4x the frequencies": dict(sizes=(3, 3, 3), nk=(2,),
                                   poles=((1,), (1,)), n_freq=52),
        "4x the blocks": dict(sizes=(3,) * 12, nk=(2,), poles=((1,), (1,))),
        "ragged clusters and poles": dict(
            sizes=(3, 3, 3), nk=(4,),
            poles=((1, 2, 3), (4,), (2,), (1, 1))),
    }
    for name, kw in cases.items():
        _, got = _profile(**kw)
        assert {k: got.get(k, 0) for k in _DISPATCHERS} == base, (
            f"{name}: the leg issues a different number of contractions "
            f"({ {k: got.get(k, 0) for k in _DISPATCHERS} } against {base})")


def test_only_per_q_bookkeeping_scales_and_only_barely():
    """What is left, and why it is left.

    The driver keeps one PoleSector per q -- identity is per q, which is
    physics, not an implementation choice -- so reading each q's cluster set
    and attaching lazy views to its state is inherently once per q. Those calls
    do no arithmetic and issue no array operation. The bound is here so that
    real work cannot creep back in under cover of bookkeeping.
    """
    small, _ = _profile(sizes=(3, 3, 3), nk=(2,), poles=((1,), (1,)))
    big, _ = _profile(sizes=(3, 3, 3), nk=(4, 4), poles=tuple(((1,),) * 16))
    per_q = (big - small) / 14
    assert per_q <= 12, (
        f"{per_q:.1f} Python calls per extra q; the leg is doing more than "
        "bookkeeping once per q")


def test_registration_report_measures_the_sub_cell_offset():
    """Where the poles sit inside their cells, over the whole promoted set.

    This is the control parameter of the bubble's registration error: an
    exactly cell-averaged leg still places a line's whole weight at the cell
    CENTRE, so the combination frequency is displaced by up to a full cell.
    The report used to be recomputed per q from a Python walk over the poles;
    it is one reduction over the whole set now, and it has to give the same
    number.
    """
    import re

    solver, sse_l, sse_g, g_r, g_l, states, sectors, n_q = _bed(
        sizes=(3, 3, 3), nk=(2,), poles=((1,), (1,)))
    w = solver.local_frequencies
    h = float(solver.local_frequency_weights[0])

    # Two poles: one exactly on a grid point, one a known fraction off it.
    on_node = float(w[4])
    off_node = float(w[6]) + 0.3 * h
    for q, x in enumerate((on_node, off_node)):
        z = np.array([x - 0.01j, -x - 0.01j])
        legs = [PoleCluster(z=xp.asarray(z),
                            u=xp.asarray(np.ones((9, 2), dtype=complex)),
                            v=xp.asarray(np.ones((9, 2), dtype=complex)),
                            label=f"q{q}c0")]
        states[q].legs = legs

    leg = xp.zeros((n_q, w.size, sse_l.rows.size), dtype=complex)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solver._report_pole_registration(states, g_l, leg, np.arange(n_q))
    text = buf.getvalue()
    assert "pole registration" in text, text
    got = float(re.search(r"offset ([0-9.]+) cells", text).group(1))
    assert abs(got - 0.3) < 1e-6, (
        f"worst sub-cell offset {got}, expected the 0.3-cell pole to dominate")

    # A set with every pole on a node reports zero, not the previous value.
    for q in range(n_q):
        z = np.array([float(w[4]) - 0.01j, -float(w[4]) - 0.01j])
        states[q].legs = [PoleCluster(
            z=xp.asarray(z), u=xp.asarray(np.ones((9, 2), dtype=complex)),
            v=xp.asarray(np.ones((9, 2), dtype=complex)), label="c")]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solver._report_pole_registration(states, g_l, leg, np.arange(n_q))
    assert float(re.search(r"offset ([0-9.]+) cells",
                           buf.getvalue()).group(1)) < 1e-12


def test_registration_report_is_silent_without_poles():
    solver, sse_l, sse_g, g_r, g_l, states, sectors, n_q = _bed(
        sizes=(3, 3, 3), nk=(2,), poles=((1,), (1,)))
    for st in states:
        st.legs = []
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solver._report_pole_registration(states, g_l, None, np.arange(n_q))
        solver._report_pole_registration(states, g_l,
                                         xp.zeros((n_q, 13, 45), dtype=complex),
                                         np.arange(n_q))
    assert buf.getvalue() == "", buf.getvalue()


@pytest.mark.parametrize("budget", [1 << 40, 1 << 14, 1])
def test_chunking_the_q_axis_does_not_change_the_leg(budget):
    """The memory cut is a memory cut. It must not be visible in the answer."""
    kw = dict(sizes=(3, 3, 3), nk=(2, 2),
              poles=((2, 1), (3,), (1, 2, 1), (2,)))
    bed = _bed(**kw)
    solver, sse_l, sse_g, g_r, g_l, states, sectors, n_q = bed
    solver._POLE_LEG_BUDGET = 1 << 40
    ref_l, ref_g = solver._build_pole_legs(sse_l, sse_g, g_r, states, sectors,
                                           np.arange(n_q))
    ref_src = [list(st.source_lesser) for st in states]

    bed = _bed(**kw)
    solver, sse_l, sse_g, g_r, g_l, states, sectors, n_q = bed
    solver._POLE_LEG_BUDGET = budget
    got_l, got_g = solver._build_pole_legs(sse_l, sse_g, g_r, states, sectors,
                                           np.arange(n_q))
    assert got_l.shape == ref_l.shape
    assert _rel(got_l, ref_l) < 1e-13
    assert _rel(got_g, ref_g) < 1e-13
    for q in range(n_q):
        for m in range(len(ref_src[q])):
            assert _rel(states[q].source_lesser[m], ref_src[q][m]) < 1e-13


def test_chunking_survives_a_q_with_no_poles():
    """A chunk boundary must not fall foul of an empty q, or of a chunk that
    is entirely empty -- that path returns None and has to be padded back."""
    kw = dict(sizes=(3, 3, 3), nk=(4,), poles=((2,), (1,), (1,), (2,)))
    bed = _bed(**kw)
    solver, sse_l, sse_g, g_r, g_l, states, sectors, n_q = bed
    solver._POLE_LEG_BUDGET = 1 << 40
    for q in (1, 2):
        states[q].clusters = []
        sectors[q] = _Sector([])
    ref_l, _ = solver._build_pole_legs(sse_l, sse_g, g_r, states, sectors,
                                       np.arange(n_q))

    bed = _bed(**kw)
    solver, sse_l, sse_g, g_r, g_l, states, sectors, n_q = bed
    solver._POLE_LEG_BUDGET = 1          # one q per chunk
    for q in (1, 2):
        states[q].clusters = []
        sectors[q] = _Sector([])
    got_l, _ = solver._build_pole_legs(sse_l, sse_g, g_r, states, sectors,
                                       np.arange(n_q))
    assert got_l.shape == ref_l.shape
    assert _rel(got_l, ref_l) < 1e-13
    assert float(np.abs(_h(got_l[1])).max()) == 0.0
