# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The block-structured mixed contraction against the pattern-level form.

``_mixed_one_sector`` contracts at the pattern level and is O(nnz_out *
nnz_in): at device scale (nnz ~ 1e5) the single intermediate it forms has 1e10
entries, which is why it carries a hard refusal above 4096. The blocked form
computes the SAME object as a sum of block triple products, so the
pattern-level routine is the exact reference and the agreement must be at
roundoff -- not at some tolerance that would hide a bookkeeping error.
"""
import numpy as np
import pytest

from quatrex.phonon.experimental.pole.pole_bridge import (
    blocks_from_pattern,
    mixed_self_energy_blocked,
    mixed_self_energy_sparse,
    mixed_vertex_block_dict,
    mixed_vertex_blocks,
    pattern_from_blocks,
)
from quatrex.phonon.experimental.pole.pole_keldysh import PoleCluster


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


def _bed(sizes, npp=2, gamma=2.0, seed=0):
    rng = np.random.default_rng(seed)
    nb = len(sizes)
    off = np.concatenate(([0], np.cumsum(sizes)))
    n_dof = int(sizes.sum())
    phi = {}
    for i in range(nb):
        for k1 in range(max(0, i - 1), min(nb, i + 2)):
            for k2 in range(max(0, i - 1), min(nb, i + 2)):
                phi[(i, k1, k2)] = rng.normal(
                    size=(sizes[i], sizes[k1], sizes[k2]))
    z = (np.linspace(8.0, 12.0, npp) - 1j * gamma)
    u = rng.normal(size=(n_dof, npp)) + 1j * rng.normal(size=(n_dof, npp))
    v = rng.normal(size=(n_dof, npp)) + 1j * rng.normal(size=(n_dof, npp))
    a = rng.normal(size=(npp, npp)) + 1j * rng.normal(size=(npp, npp))
    return phi, PoleCluster(z=z, u=u, v=v), a @ a.conj().T, off


# --------------------------------------------------------------------------
# The block <-> pattern round trip
# --------------------------------------------------------------------------

def test_pattern_block_round_trip_is_exact():
    sizes = np.array([2, 3, 2])
    rows, cols, _ = _pattern(sizes)
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(5, rows.size)) + 1j * rng.normal(size=(5, rows.size))
    blocks = blocks_from_pattern(vals, rows, cols, sizes)
    back = pattern_from_blocks(blocks, rows, cols, sizes, 5)
    assert np.abs(_h(back) - vals).max() == 0.0


def test_pattern_from_blocks_drops_pairs_outside_the_output_pin():
    """Absent block pairs contribute zero -- that IS the |I-J| <= 1 pin."""
    sizes = np.array([2, 2, 2])
    rows, cols, _ = _pattern(sizes)
    rng = np.random.default_rng(1)
    vals = rng.normal(size=(3, rows.size)) + 1j * rng.normal(size=(3, rows.size))
    blocks = blocks_from_pattern(vals, rows, cols, sizes)
    del blocks[(0, 1)]
    back = _h(pattern_from_blocks(blocks, rows, cols, sizes, 3))
    off = np.concatenate(([0], np.cumsum(sizes)))
    br = np.searchsorted(off, rows, side="right") - 1
    bc = np.searchsorted(off, cols, side="right") - 1
    gone = (br == 0) & (bc == 1)
    assert np.abs(back[:, gone]).max() == 0.0
    assert np.abs(back[:, ~gone] - vals[:, ~gone]).max() == 0.0


# --------------------------------------------------------------------------
# The blocked vertex
# --------------------------------------------------------------------------

@pytest.mark.parametrize("leg", [0, 1])
@pytest.mark.parametrize("conjugate", [False, True])
def test_blocked_vertex_matches_the_dense_vertex(leg, conjugate):
    sizes = np.array([2, 3, 2])
    phi, cl, _, off = _bed(sizes)
    dense = _h(mixed_vertex_blocks(
        phi, sizes, cl.u, leg=leg, conjugate=conjugate))
    blocked = mixed_vertex_block_dict(
        phi, sizes, cl.u, leg=leg, conjugate=conjugate)

    rebuilt = np.zeros_like(dense)
    for (i, j), blk in blocked.items():
        rebuilt[off[i]:off[i + 1], off[j]:off[j + 1]] = _h(blk)
    assert np.abs(rebuilt - dense).max() < 1e-13 * max(np.abs(dense).max(), 1.0)


# --------------------------------------------------------------------------
# The contraction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sizes", [np.array([2, 2]), np.array([2, 3, 2])])
def test_blocked_contraction_matches_the_pattern_level_reference(sizes):
    """The load-bearing test: same object, device-scale cost."""
    phi, cl, src, _ = _bed(sizes, seed=2)
    rows, cols, _ = _pattern(sizes)
    freqs = np.linspace(0.0, 20.0, 96)   # zero-anchored, as the solver grid is
    omega = np.linspace(6.0, 16.0, 9)
    rng = np.random.default_rng(3)
    g_reg = (rng.normal(size=(freqs.size, rows.size))
             + 1j * rng.normal(size=(freqs.size, rows.size)))
    # The Keldysh partner supplies the negative frequency axis; these tests
    # exercise the contraction, so a distinct array is enough.
    g_partner = (rng.normal(size=(freqs.size, rows.size))
                 + 1j * rng.normal(size=(freqs.size, rows.size)))

    ref = _h(mixed_self_energy_sparse(
        omega, cl, src, g_reg, g_partner, freqs, phi, sizes, rows, cols))
    got = _h(mixed_self_energy_blocked(
        omega, cl, src, g_reg, g_partner, freqs, phi, sizes, rows, cols))
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-12


def test_blocked_contraction_has_no_nnz_guard():
    """The whole point: the pattern-level form refuses where this runs.

    The guard is what currently keeps ``sectors="rr_ss_sr"`` off real devices,
    so a test that the blocked route clears it is a test of the feature, not
    of an implementation detail.
    """
    sizes = np.array([2, 2])
    phi, cl, src, _ = _bed(sizes, seed=4)
    rows, cols, _ = _pattern(sizes)
    freqs = np.linspace(0.0, 20.0, 64)   # zero-anchored, as the solver grid is
    omega = np.linspace(6.0, 16.0, 5)
    rng = np.random.default_rng(5)
    g_reg = (rng.normal(size=(freqs.size, rows.size))
             + 1j * rng.normal(size=(freqs.size, rows.size)))
    # The Keldysh partner supplies the negative frequency axis; these tests
    # exercise the contraction, so a distinct array is enough.
    g_partner = (rng.normal(size=(freqs.size, rows.size))
                 + 1j * rng.normal(size=(freqs.size, rows.size)))

    with pytest.raises(NotImplementedError, match="exceeds the"):
        mixed_self_energy_sparse(
            omega, cl, src, g_reg, g_partner, freqs, phi, sizes, rows, cols,
            max_nnz=4)
    # Same arguments, no guard, and still the right answer.
    ref = _h(mixed_self_energy_sparse(
        omega, cl, src, g_reg, g_partner, freqs, phi, sizes, rows, cols))
    got = _h(mixed_self_energy_blocked(
        omega, cl, src, g_reg, g_partner, freqs, phi, sizes, rows, cols))
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-12


def test_blocked_sr_and_rs_remain_distinct():
    """SR and RS on a NON-leg-symmetric vertex."""
    from quatrex.phonon.experimental.pole.pole_bridge import _mixed_one_sector_blocked

    sizes = np.array([2, 2])
    phi, cl, src, _ = _bed(sizes, seed=6)
    rows, cols, _ = _pattern(sizes)
    freqs = np.linspace(0.0, 20.0, 64)   # zero-anchored, as the solver grid is
    omega = np.linspace(6.0, 16.0, 5)
    rng = np.random.default_rng(7)
    g_reg = (rng.normal(size=(freqs.size, rows.size))
             + 1j * rng.normal(size=(freqs.size, rows.size)))
    # The Keldysh partner supplies the negative frequency axis; these tests
    # exercise the contraction, so a distinct array is enough.
    g_partner = (rng.normal(size=(freqs.size, rows.size))
                 + 1j * rng.normal(size=(freqs.size, rows.size)))
    vd = mixed_vertex_block_dict
    kw = dict(freqs=freqs, rows=rows, cols=cols, block_sizes=sizes,
              g_partner=g_partner)
    sr = _h(_mixed_one_sector_blocked(
        omega, cl, src, g_reg,
        bl=vd(phi, sizes, cl.u, leg=0, conjugate=False),
        br=vd(phi, sizes, cl.u, leg=1, conjugate=True), **kw))
    rs = _h(_mixed_one_sector_blocked(
        omega, cl, src, g_reg,
        bl=vd(phi, sizes, cl.u, leg=1, conjugate=False),
        br=vd(phi, sizes, cl.u, leg=0, conjugate=True), **kw))
    assert np.abs(sr - rs).max() / np.abs(sr).max() > 0.1
