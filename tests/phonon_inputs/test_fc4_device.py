# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Unit tests for the FC4 -> device-tensor adapter (loop self-energy).

``solver.fc4_device.build_device_fc4_tensor`` folds a compact-reference sparse
FC4 onto a finite multi-slab device (leg-1 anchored at a slab-0 reference atom,
legs 2-4 placed by minimum image), mass-weighting each leg by ``1/sqrt(m)``.
These tests pin the folding, placement, window and mass-weighting against
hand-computed expectations -- the FC4 analogue of the FC3 device-block tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PHONON = Path(__file__).resolve().parents[2] / "phonon"
if str(_PHONON) not in sys.path:
    sys.path.insert(0, str(_PHONON))

from solver.fc4_device import (  # noqa: E402
    build_compact_reference_fc4_from_dense,
    build_device_fc4_tensor,
)
from solver.static_se import sigma_loop  # noqa: E402


def _chain_mapping(n_super_z, mass=2.0):
    """1-atom-per-cell chain: n_super_z slabs, 3 cart, uniform mass."""
    prim_indices = np.zeros(n_super_z, dtype=int)        # all the same atom
    slab_indices = np.arange(n_super_z, dtype=int)
    masses_super = np.full(n_super_z, mass, dtype=float)
    return prim_indices, slab_indices, masses_super


def test_onsite_fc4_placed_on_every_slab():
    """An on-site cluster (all legs the slab-0 atom) appears on every slab."""
    n_super_z, n_slabs, m = 3, 3, 2.0
    prim, slab, masses = _chain_mapping(n_super_z, m)
    rng = np.random.default_rng(1)
    T = rng.standard_normal((3, 3, 3, 3))
    fc4 = {(0, 0, 0, 0): T}

    Phi4 = build_device_fc4_tensor(fc4, prim, slab, masses, n_atoms=1,
                                   n_slabs=n_slabs)
    expect = T / m ** 2                                   # mass-weight 1/sqrt(m)^4
    for I in range(n_slabs):
        s = slice(3 * I, 3 * I + 3)
        np.testing.assert_allclose(Phi4[s, s, s, s], expect, atol=1e-12)
    # total weight = 3 on-site blocks only
    assert np.count_nonzero(np.abs(Phi4) > 1e-12) == 3 * np.count_nonzero(
        np.abs(expect) > 1e-12)


def test_nearest_neighbour_and_minimum_image_placement():
    """Leg-2 at slab +1 lands at (I, I+1, I, I); a slab-2 atom folds to -1."""
    n_super_z, n_slabs, m = 3, 3, 2.0       # half_window = 1
    prim, slab, masses = _chain_mapping(n_super_z, m)
    rng = np.random.default_rng(2)
    Tp = rng.standard_normal((3, 3, 3, 3))   # leg-2 forward neighbour
    Tm = rng.standard_normal((3, 3, 3, 3))   # leg-2 at slab 2 == -1 (min image)
    fc4 = {(0, 1, 0, 0): Tp, (0, 2, 0, 0): Tm}

    Phi4 = build_device_fc4_tensor(fc4, prim, slab, masses, n_atoms=1,
                                   n_slabs=n_slabs)
    mw = 1.0 / m ** 2

    def blk(a, b, c, d):
        return Phi4[3 * a:3 * a + 3, 3 * b:3 * b + 3,
                    3 * c:3 * c + 3, 3 * d:3 * d + 3]

    # forward neighbour Tp: (I, I+1, I, I) for I = 0, 1 (I=2 needs slab 3 -> drop)
    np.testing.assert_allclose(blk(0, 1, 0, 0), Tp * mw, atol=1e-12)
    np.testing.assert_allclose(blk(1, 2, 1, 1), Tp * mw, atol=1e-12)
    # min image: slab-2 atom -> offset -1, so (I, I-1, I, I) for I = 1, 2
    np.testing.assert_allclose(blk(1, 0, 1, 1), Tm * mw, atol=1e-12)
    np.testing.assert_allclose(blk(2, 1, 2, 2), Tm * mw, atol=1e-12)
    # nothing placed on-site (no on-site cluster was supplied)
    np.testing.assert_allclose(blk(0, 0, 0, 0), 0.0, atol=1e-12)
    np.testing.assert_allclose(blk(2, 2, 2, 2), 0.0, atol=1e-12)
    # exactly 4 device blocks populated (Tp at I=0,1 + Tm at I=1,2)
    expected_nnz = (2 * np.count_nonzero(np.abs(Tp) > 1e-12)
                    + 2 * np.count_nonzero(np.abs(Tm) > 1e-12))
    assert np.count_nonzero(np.abs(Phi4) > 1e-12) == expected_nnz


def test_vertex_cutoff_drops_far_quadruples():
    """vertex_cutoff bounds the maximum pairwise slab spread retained."""
    n_super_z, n_slabs, m = 5, 5, 2.0
    prim, slab, masses = _chain_mapping(n_super_z, m)
    rng = np.random.default_rng(3)
    T = rng.standard_normal((3, 3, 3, 3))
    fc4 = {(0, 2, 0, 0): T}                  # spread = 2

    keep = build_device_fc4_tensor(fc4, prim, slab, masses, 1, n_slabs,
                                   vertex_cutoff=2)
    drop = build_device_fc4_tensor(fc4, prim, slab, masses, 1, n_slabs,
                                   vertex_cutoff=1)
    assert np.max(np.abs(keep)) > 0
    assert np.max(np.abs(drop)) == 0.0


def test_loop_self_energy_symmetric_from_symmetric_fc4():
    """Sigma_L from a symmetric device FC4 + symmetric <uu> is real symmetric."""
    n_super_z, n_slabs, m = 3, 3, 2.0
    prim, slab, masses = _chain_mapping(n_super_z, m)
    rng = np.random.default_rng(4)
    # symmetric on-site quartic on the slab-0 atom
    raw = rng.standard_normal((3, 3, 3, 3))
    T = sum(np.transpose(raw, p) for p in _perms4()) / 24.0
    fc4 = {(0, 0, 0, 0): T}
    Phi4 = build_device_fc4_tensor(fc4, prim, slab, masses, 1, n_slabs)

    N_D = 3 * n_slabs
    u = rng.standard_normal((N_D, N_D))
    uu = u + u.T                              # symmetric <uu>
    sig = sigma_loop(Phi4, uu)
    assert np.allclose(sig, sig.T)
    assert np.max(np.abs(sig.imag)) < 1e-12


def _perms4():
    from itertools import permutations
    return list(permutations(range(4)))


def test_loader_reads_compact_fc4_hdf5(tmp_path):
    """`_load_device_fc4_mass_weighted` reads the hdf5 and folds identically
    to a direct `build_device_fc4_tensor` call."""
    import h5py

    from solver.dense import _load_device_fc4_mass_weighted

    n_super_z, n_slabs, m = 3, 3, 2.0
    prim, slab, masses = _chain_mapping(n_super_z, m)
    rng = np.random.default_rng(5)
    fc4 = {(0, 0, 0, 0): rng.standard_normal((3, 3, 3, 3)),
           (0, 1, 0, 0): rng.standard_normal((3, 3, 3, 3))}
    direct = build_device_fc4_tensor(fc4, prim, slab, masses, 1, n_slabs)

    path = tmp_path / "fc4.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("fc4_atoms",
                         data=np.array(list(fc4), dtype=np.int64))
        f.create_dataset("fc4_values",
                         data=np.array(list(fc4.values()), dtype=np.float64))

    loaded = _load_device_fc4_mass_weighted(
        str(path), prim, slab, n_atoms=1, n_slabs=n_slabs, masses_super=masses)
    np.testing.assert_allclose(loaded, direct, atol=1e-12)


def test_compact_reference_from_dense_slices_reference_atoms():
    """The dense->compact-reference export keeps exactly the nonzero
    leg-1=reference slices, dropping sub-tolerance blocks."""
    n_super = 4
    rng = np.random.default_rng(7)
    dense = np.zeros((n_super,) * 4 + (3, 3, 3, 3))
    # two nonzero quadruples anchored at reference atom 0, one at atom 1
    dense[0, 0, 0, 0] = rng.standard_normal((3, 3, 3, 3))
    dense[0, 1, 0, 2] = rng.standard_normal((3, 3, 3, 3))
    dense[1, 1, 1, 1] = rng.standard_normal((3, 3, 3, 3))
    dense[0, 3, 3, 3] = 1e-12                      # below tol -> dropped
    ref_sc_atoms = [0, 1]                          # two primitive atoms

    compact = build_compact_reference_fc4_from_dense(dense, ref_sc_atoms,
                                                     tol=1e-8)
    assert set(compact) == {(0, 0, 0, 0), (0, 1, 0, 2), (1, 1, 1, 1)}
    np.testing.assert_allclose(compact[(0, 1, 0, 2)], dense[0, 1, 0, 2])
    np.testing.assert_allclose(compact[(1, 1, 1, 1)], dense[1, 1, 1, 1])
