# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Tests for ``phonon/studies/engine/reblock_device.py``.

The tool re-partitions a film device so that C transport cells sit in
one BTD block, which widens the retained Sigma range without touching
any vertex weight (see ``phonon/docs/bubble_positivity.md``). Two things
have to hold for that to be a legitimate experiment rather than a new
approximation:

1. the re-partition must leave the dense device operator and the dense
   vertex bit-identical -- it is a change of blocking, nothing else;
2. the fc3 writer inlined in the tool (to avoid the phonopy import in
   ``phonon_inputs.quatrex_writer``, which is absent from the cluster
   venv) must produce exactly the production schema.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "_reblock", ROOT / "phonon/studies/engine/reblock_device.py")


def _mod():
    m = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(m)
    return m


def test_inlined_fc3_writer_matches_production(tmp_path) -> None:
    """The inlined writer and quatrex_writer.write_fc3_blocks must be
    read back identically by the production loader."""
    phonopy = pytest.importorskip(
        "phonopy", reason="quatrex_writer imports phonopy at module level")
    assert phonopy is not None
    sys.path.insert(0, str(ROOT / "phonon"))
    from phonon_inputs.quatrex_writer import write_fc3_blocks

    from quatrex.phonon.fc3_loader import load_device_fc3

    rng = np.random.default_rng(0)
    nb, nd = 3, 4
    blocks = {(I, I, I): rng.standard_normal((nd, nd, nd)) + 0j
              for I in range(nb)}
    blocks[(0, 0, 1)] = rng.standard_normal((nd, nd, nd)) + 0j
    sizes = np.array([nd] * nb)

    a, b = tmp_path / "a.hdf5", tmp_path / "b.hdf5"
    write_fc3_blocks(blocks, sizes, a, units="THz^2")
    _mod()._write_fc3_blocks(blocks, sizes, b)

    la = load_device_fc3(a, block_sizes=sizes, truncation_warn=1e9)
    lb = load_device_fc3(b, block_sizes=sizes, truncation_warn=1e9)
    assert set(la) == set(lb)
    for k in la:
        np.testing.assert_array_equal(la[k], lb[k])


@pytest.mark.parametrize("c", [1, 2, 3])
def test_reblocking_preserves_the_dense_operators(c: int) -> None:
    """Merging c primitive slabs into one block changes only the block
    boundaries: the dense 6-cell FC2 and vertex are unchanged."""
    m = _mod()
    nd, n_cells = 3, 6
    rng = np.random.default_rng(5)
    d0 = rng.standard_normal((nd, nd)) + 0j
    d0 = d0 + d0.conj().T
    d1 = rng.standard_normal((nd, nd)) + 0j
    mats = {(0, 0): {0: d0, 1: d1, -1: d1.conj().T}}

    # FC2: assemble the 6-cell operator from primitive blocks, then from
    # the c-cell superblocks, and compare.
    ref = m._dense_fc2(mats, (0, 0), n_cells, nd)
    ndn, nb = c * nd, n_cells // c
    sup = {s: m._superblock(mats[(0, 0)], c, nd, s) for s in (-1, 0, 1)}
    got = np.zeros_like(ref)
    for i in range(nb):
        for j in range(nb):
            if (j - i) in sup:
                got[i * ndn:(i + 1) * ndn, j * ndn:(j + 1) * ndn] = sup[j - i]
    np.testing.assert_allclose(got, ref, atol=1e-13, rtol=0)

    # vertex: an intra-slab dict replicated to 6 slabs, then merged.
    phi = rng.standard_normal((nd, nd, nd)) + 0j
    prim = {(I, I, I): phi for I in range(n_cells)}
    merged = m._merge(prim, n_cells, c, nd)
    np.testing.assert_array_equal(
        m._dense_vertex(merged, nb, ndn),
        m._dense_vertex(prim, n_cells, nd))


def test_replication_gate_rejects_inequivalent_slabs() -> None:
    """Replication along transport is only exact when the per-slab
    vertex blocks are translationally equivalent; the tool must refuse
    otherwise rather than silently produce a wrong device."""
    m = _mod()
    rng = np.random.default_rng(1)
    nd = 3
    good = {(I, I, I): np.ones((nd, nd, nd)) for I in range(3)}
    m._assert_slab_replicas(good, 3, "ok")          # must not raise

    bad = dict(good)
    bad[(1, 1, 1)] = rng.standard_normal((nd, nd, nd))
    with pytest.raises(SystemExit, match="NOT translationally equivalent"):
        m._assert_slab_replicas(bad, 3, "bad")
