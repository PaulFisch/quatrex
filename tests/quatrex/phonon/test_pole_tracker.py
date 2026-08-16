# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""Doc Sec. 7-8 and Sec. 53 Experiment C: continuation of the pole set.

The claim under test is structural, not numerical: through an avoided crossing
the individual eigenvectors rotate arbitrarily while the invariant SUBSPACE
stays smooth, so a tracker that follows labels loses them and a tracker that
follows subspaces does not.
"""
import numpy as np
import pytest

from quatrex.phonon.pole_tracker import (
    PoleTracker,
    cluster_poles,
    match_poles,
    principal_angles,
    subspace_basis,
)

G_DAMP = 0.25


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _poles_of(d):
    """Exact poles / vectors of ``z^2 + i g z - lam`` for a symmetric ``D``."""
    lam, vec = np.linalg.eigh(d)
    z = (-1j * G_DAMP + np.sqrt(np.asarray(4 * lam - G_DAMP**2, dtype=complex))) / 2.0
    return z, vec.astype(complex)


def _crossing_d(s, gap=0.05, coupling=0.02, centre=64.0):
    """4x4 ``D`` whose first two modes undergo an avoided crossing at ``s = 0``.

    The spectator block is far away, and a small coupling links the two blocks so
    the crossing subspace is not trivially constant.
    """
    d = np.zeros((4, 4))
    d[:2, :2] = [[centre - s, gap], [gap, centre + s]]
    d[2:, 2:] = np.diag([160.0, 210.0])
    d[0, 2] = d[2, 0] = coupling
    d[1, 3] = d[3, 1] = coupling
    return d


# --------------------------------------------------------------------------- #
# Predictor (doc Eq. 43).
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Experiment C: the crossing.
# --------------------------------------------------------------------------- #

def test_frequency_sorting_swaps_character_through_the_crossing():
    """The concrete failure: the mode labelled 'lowest' is a different mode after."""
    z_lo, vec_lo = _poles_of(_crossing_d(-0.6, gap=1e-5))
    z_hi, vec_hi = _poles_of(_crossing_d(+0.6, gap=1e-5))
    lo = vec_lo[:, np.argsort(z_lo.real)[0]]
    hi = vec_hi[:, np.argsort(z_hi.real)[0]]
    assert abs(np.vdot(lo, hi)) < 0.3, (
        "the lowest-frequency label kept its character; no crossing occurred"
    )


def test_overlap_matching_recovers_the_identity_across_a_crossing():
    """Matching by vector overlap re-pairs what frequency order scrambled."""
    z_a, vec_a = _poles_of(_crossing_d(-0.02, gap=1e-5))
    z_b, vec_b = _poles_of(_crossing_d(+0.02, gap=1e-5))
    assign = match_poles(z_a, vec_a, z_b, vec_b, w_z=0.0, w_u=1.0)
    for i, j in enumerate(assign):
        ov = abs(np.vdot(vec_a[:, i], vec_b[:, j]))
        assert ov > 0.7, f"mode {i} matched to {j} with overlap {ov:.3f}"


# --------------------------------------------------------------------------- #
# Clustering and the state machine.
# --------------------------------------------------------------------------- #

def test_cluster_poles_groups_only_overlapping_modes():
    z = np.array([8.0 - 0.1j, 8.15 - 0.1j, 12.0 - 0.05j])
    groups = cluster_poles(z, factor=3.0)
    assert groups == [[0, 1], [2]], groups
    # Tightening the radius separates the pair.
    assert cluster_poles(z, factor=0.5) == [[0], [1], [2]]


def test_principal_angles_are_zero_for_the_same_span():
    rng = np.random.default_rng(3)
    v = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    q1 = subspace_basis(v)
    q2 = subspace_basis(v @ np.array([[1.0, 2.0], [-1.0, 0.5]]))  # same span
    assert float(np.max(_h(principal_angles(q1, q2)))) < 1e-10


def test_tracker_flags_a_cluster_size_change():
    rng = np.random.default_rng(4)
    n = 4
    r = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
    tr = PoleTracker(cluster_factor=3.0, rescan_iterations=1000)

    wide = np.array([8.0 - 0.1j, 8.05 - 0.1j, 12.0 - 0.05j])   # -> [[0,1],[2]]
    tr.adopt(wide, r, r)
    assert [c.size for c in tr.clusters] == [2, 1]

    split = np.array([8.0 - 0.001j, 8.05 - 0.001j, 12.0 - 0.05j])  # -> singletons
    tr.update(split, r, r)
    assert tr.rescan_reasons, "a cardinality change must request a rescan"
    assert any("count" in m or "size" in m for m in tr.rescan_reasons)


def test_tracker_is_quiet_when_poles_drift_slightly():
    rng = np.random.default_rng(5)
    r = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    tr = PoleTracker(rescan_iterations=1000)
    z = np.array([8.0 - 0.1j, 12.0 - 0.05j])
    tr.adopt(z, r, r)
    tr.update(z + np.array([1e-4, -1e-4]), r, r)
    assert not tr.rescan_reasons, tr.rescan_reasons
    assert [c.cid for c in tr.clusters] == [0, 1], "cluster identity was not kept"
    assert all(c.age == 1 for c in tr.clusters)


def test_tracker_requests_a_rescan_on_schedule_and_when_empty():
    tr = PoleTracker(rescan_iterations=3)
    assert tr.needs_rescan(), "an empty tracker must rescan"
    rng = np.random.default_rng(6)
    r = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    z = np.array([8.0 - 0.1j, 12.0 - 0.05j])
    tr.adopt(z, r, r)
    seen = [(tr.update(z, r, r), tr.needs_rescan())[1] for _ in range(4)]
    assert seen == [False, False, True, False], seen


def test_membership_freezes_within_an_epoch():
    """Sector membership may only change at an epoch boundary."""
    tr = PoleTracker(epoch_iterations=5, rescan_iterations=1000)
    rng = np.random.default_rng(7)
    r = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    z = np.array([8.0 - 0.1j, 12.0 - 0.05j])
    tr.adopt(z, r, r)
    frozen = []
    for _ in range(6):
        tr.update(z, r, r)
        frozen.append(tr.membership_frozen())
    assert frozen == [True, True, True, True, False, True], frozen
