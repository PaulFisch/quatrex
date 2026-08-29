# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Primitive-microblock layout and generated-support tests."""

from __future__ import annotations

import numpy as np
import pytest

from qttools import xp

from quatrex.phonon.microblocks import (
    MicroblockLayout,
    build_micro_pair_index,
    grouped_band_indices,
    micro_link_views,
)


def _local_vertices(n: int, d: int, span: int = 1):
    return {
        (i, j, k): np.ones((d, d, d))
        for i in range(n)
        for j in range(max(0, i - span), min(n, i + span + 1))
        for k in range(max(0, i - span), min(n, i + span + 1))
    }


def test_layout_maps_unequal_terminal_group_without_copy() -> None:
    layout = MicroblockLayout.from_block_sizes([6, 6, 3], 3)
    assert layout.cells_per_block == (2, 2, 1)
    assert layout.n_microblocks == 5
    assert layout.locate(4) == (2, slice(0, 3))

    grouped = {
        (0, 1): np.arange(4 * 6 * 6).reshape(4, 6, 6),
        (1, 2): np.arange(4 * 6 * 3).reshape(4, 6, 3),
    }
    views = micro_link_views(grouped, layout, {(1, 2), (3, 4)})
    assert views[(1, 2)].shape == (4, 3, 3)
    assert views[(3, 4)].shape == (4, 3, 3)
    assert np.shares_memory(views[(1, 2)], grouped[(0, 1)])


def test_generated_support_is_two_p_plus_b_and_is_not_pinned() -> None:
    # Interior finite-device vertices attain the exact support law.
    layout = MicroblockLayout.from_block_sizes([5, 5], 1)
    index, p, sigma = build_micro_pair_index(
        _local_vertices(10, 1, span=1), layout, g_band=3)
    assert p == 1
    assert sigma == 5
    primitive_pairs = {pair for pairs in index.values() for pair in pairs}
    assert (1, 6) in primitive_pairs
    assert (6, 1) in primitive_pairs


def test_incomplete_grouping_fails_instead_of_dropping_far_sigma() -> None:
    layout = MicroblockLayout.from_block_sizes([2, 2, 2, 2, 2], 1)
    with pytest.raises(ValueError, match="non-adjacent Dyson blocks"):
        build_micro_pair_index(
            _local_vertices(10, 1, span=1), layout, g_band=3)


def test_layout_rejects_partial_microblock() -> None:
    with pytest.raises(ValueError, match="integer number"):
        MicroblockLayout.from_block_sizes([12, 15], 6)


def test_grouped_band_indices_are_complete_and_arrow_owned() -> None:
    sizes = np.array([2, 3, 1])
    rows, cols = grouped_band_indices(
        sizes, band=1, start_block=1, end_block=2)
    got = set(zip(rows.tolist(), cols.tolist()))
    # min(I,J)==1 owns (1,1), (1,2) and (2,1), but not (0,1).
    want = set()
    offsets = np.array([0, 2, 5, 6])
    for i, j in ((1, 1), (1, 2), (2, 1)):
        want |= {
            (r, c)
            for r in range(offsets[i], offsets[i + 1])
            for c in range(offsets[j], offsets[j + 1])
        }
    assert got == want


def test_factored_gram_decomposes_noncartesian_support_without_cross_terms() -> None:
    """Coupled support equals the sum of its admissible singleton quads."""
    from quatrex.phonon.bubble_factored import contract_tau_q_factored

    quads = {(0, 0): [(0, 0, 0, 0), (1, 1, 1, 1)]}
    g = {
        (0, 0): xp.asarray([[[[1.0 + 0.2j]]], [[[0.7 - 0.1j]]]]),
        (1, 1): xp.asarray([[[[0.3 + 0.4j]]], [[[1.1 + 0.2j]]]]),
    }
    gs = {name: g for name in ("l", "g", "lr", "gr")}
    Dt = xp.asarray([[0.8, 1.3]])
    UB = xp.asarray([[[[1.0, 0.4]]], [[[0.2, 1.1]]]])
    UC = xp.asarray([[[[0.7, 0.3]]], [[[1.2, 0.5]]]])
    args = (np.array([1, 1]), (), 0, 1, 1, gs, Dt, UB, UC,
            {0: 0, 1: 1}, 0, 2, xp, True, complex)
    got = contract_tau_q_factored(quads, *args)[(0, 0)]
    ref_parts = [
        contract_tau_q_factored({(0, 0): [quad]}, *args)[(0, 0)]
        for quad in quads[(0, 0)]
    ]
    for component in (0, 1):
        ref = ref_parts[0][component] + ref_parts[1][component]
        assert xp.allclose(got[component], ref, rtol=1e-13, atol=1e-13)
