# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Unit tests for ``quatrex.phonon.vertex_factors`` (npz round-trip,
rank truncation, reconstruction) and the config-level mutual exclusion of
the dense / decomposed coupled-q vertex sources."""

from __future__ import annotations

import numpy as np
import pytest

from quatrex.phonon.vertex_factors import (
    FORMAT_VERSION,
    VertexFactors,
    load_decomposed,
    save_decomposed,
)


def _mk_vf(rng, n_dof=6, R=7, nq=4, n_off=3, ansatz="INDSCAL"):
    UB = (rng.standard_normal((n_off, nq, n_dof, R))
          + 1j * rng.standard_normal((n_off, nq, n_dof, R)))
    UC = UB if ansatz == "INDSCAL" else (
        rng.standard_normal((n_off, nq, n_dof, R))
        + 1j * rng.standard_normal((n_off, nq, n_dof, R)))
    return VertexFactors(
        D=rng.standard_normal((n_dof, R)),
        lambdas=np.sort(np.abs(rng.standard_normal(R)))[::-1],
        offsets=np.array([-1, 0, 1], dtype=np.int64),
        UB=UB,
        UC=UC,
        q_diff_map=np.array([[(a - b) % nq for b in range(nq)]
                             for a in range(nq)]),
        nk_shape=(2, 2),
        ansatz=ansatz,
        meta={"rel_err": 0.01, "source": "test"},
    )


def test_npz_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(3)
    vf = _mk_vf(rng)
    path = tmp_path / "decomposed_vertices.npz"
    save_decomposed(path, vf)
    back = load_decomposed(path)
    np.testing.assert_array_equal(back.D, vf.D)
    np.testing.assert_array_equal(back.lambdas, vf.lambdas)
    np.testing.assert_array_equal(back.offsets, vf.offsets)
    np.testing.assert_array_equal(back.UB, vf.UB)
    np.testing.assert_array_equal(back.UC, vf.UC)
    np.testing.assert_array_equal(back.q_diff_map, vf.q_diff_map)
    assert back.nk_shape == vf.nk_shape
    assert back.ansatz == vf.ansatz
    assert back.meta == vf.meta
    assert back.rank == vf.rank and back.n_kpts == vf.n_kpts


def test_load_with_rank_truncation(tmp_path) -> None:
    rng = np.random.default_rng(4)
    vf = _mk_vf(rng, R=7)
    path = tmp_path / "vf.npz"
    save_decomposed(path, vf)
    tr = load_decomposed(path, rank=3)
    assert tr.rank == 3
    np.testing.assert_array_equal(tr.D, vf.D[:, :3])
    np.testing.assert_array_equal(tr.lambdas, vf.lambdas[:3])
    np.testing.assert_array_equal(tr.UB, vf.UB[..., :3])
    assert tr.meta["truncated_to"] == 3
    # rank 0 / >= stored rank: no-op
    assert load_decomposed(path, rank=0).rank == 7
    assert load_decomposed(path, rank=99).rank == 7


def test_reconstruct_block_matches_einsum() -> None:
    rng = np.random.default_rng(5)
    vf = _mk_vf(rng, ansatz="CP")
    pos = vf.offset_index()
    blk = vf.reconstruct_block(1, 2, -1, 1)
    ref = np.einsum("r,ar,br,cr->abc", vf.lambdas, vf.D,
                    vf.UB[pos[-1], 1], vf.UC[pos[1], 2])
    np.testing.assert_allclose(blk, ref, rtol=1e-13)


def test_format_version_gate(tmp_path) -> None:
    rng = np.random.default_rng(6)
    vf = _mk_vf(rng)
    path = tmp_path / "vf.npz"
    save_decomposed(path, vf)
    data = dict(np.load(path, allow_pickle=True))
    data["format_version"] = np.int64(FORMAT_VERSION + 1)
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="format"):
        load_decomposed(path)


def test_config_vertex_source_exclusivity(tmp_path) -> None:
    """qfold_path and decomposed_vertices_path are mutually exclusive;
    sse_vertex_rank requires decomposed_vertices_path."""
    from quatrex.core.config import PhononConfig

    qf = tmp_path / "qfold_vertices.npz"
    dv = tmp_path / "decomposed_vertices.npz"
    qf.touch()
    dv.touch()
    base = dict(model="negf", fc3_path=str(tmp_path / "fc3.hdf5"))

    PhononConfig(**base, qfold_path=str(qf))
    PhononConfig(**base, decomposed_vertices_path=str(dv), sse_vertex_rank=8)

    with pytest.raises(ValueError, match="mutually"):
        PhononConfig(**base, qfold_path=str(qf),
                     decomposed_vertices_path=str(dv))
    with pytest.raises(ValueError, match="requires"):
        PhononConfig(**base, sse_vertex_rank=8)
    with pytest.raises(ValueError, match=">= 0"):
        PhononConfig(**base, decomposed_vertices_path=str(dv),
                     sse_vertex_rank=-1)
