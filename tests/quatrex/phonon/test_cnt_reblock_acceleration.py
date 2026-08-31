"""Reference tests for the private CNT reblocking acceleration audit."""

import numpy as np

from studies import _cnt_reblock_acceleration as C


def test_two_cell_grouping_has_the_recorded_primitive_support():
    sigma = C.distance_coverage(C.grouped_mask(16, 2, 1))
    green = C.distance_coverage(C.grouped_mask(16, 2, 3))
    assert [sigma[d]["kept"] for d in range(4)] == [16, 15, 14, 7]
    assert set(sigma) == {0, 1, 2, 3}
    assert [green[d]["kept"] for d in range(8)] == [
        16, 15, 14, 13, 12, 11, 10, 5]
    assert set(green) == set(range(8))


def test_cnt_vertex_enumeration_reproduces_both_archived_quad_counts():
    model = C.support_and_cost_model()
    assert model["primitive_vertices"] == 106
    assert model["merged_vertices"] == 50
    assert model["baseline"]["pairs"] == 46
    assert model["baseline"]["quads"] == 2104
    assert model["merged"]["g3"]["slow_rank_pairs"] == 12
    assert model["merged"]["g3"]["slow_rank_quads"] == 513
    assert model["microblocked_exact"]["slow_rank_quads"] == 2193


def test_ring_flop_model_and_auxiliary_break_even_are_consistent():
    model = C.support_and_cost_model()
    assert np.isclose(model["baseline"]["gflop_6ring"], 41219.550314496)
    assert np.isclose(model["merged"]["g3"]["gflop_6ring"],
                      319621.303959552)
    assert np.isclose(model["merged"]["g3"]["gflop_4ring"] * 1.5,
                      model["merged"]["g3"]["gflop_6ring"])
    assert 10.5 < model["auxiliary_break_even_rank"] < 10.6
    ratio = model["microblocked_exact"]["atom_sparse_ideal_mac_ratio"]
    # The final dense contraction is one third of the original arithmetic;
    # the two sparse vertex actions add only their measured triplet fills.
    assert 1 / 3 < ratio < 0.4
    categories = model["microblocked_exact"]["atom_sparse_quad_categories"]
    assert sum(categories.values()) == model["microblocked_exact"]["slow_rank_quads"]
    assert all(value > 0 for value in categories.values())
    layouts = model["microblocked_exact"]["atom_sparse_layout_categories"]
    assert sum(layouts.values()) == model["microblocked_exact"]["slow_rank_quads"]


def test_arrow_pair_order_matches_saved_snapshot_shapes():
    assert len(C._rank_block_pairs(0)) == 48
    assert len(C._rank_block_pairs(1)) == 16
    assert C._rank_block_pairs(2) == C._rank_block_pairs(0)
    assert C._rank_block_pairs(3) == C._rank_block_pairs(1)


def test_far_extraction_and_frobenius_rank():
    rng = np.random.default_rng(8)
    block = rng.normal(size=(4, 72, 72)) + 1j * rng.normal(size=(4, 72, 72))
    far = C._far_correction(block)
    assert np.all(far[:, 36:, :36] == 0.0)
    assert np.all(far[:, :36, :36] == block[:, :36, :36])
    u = rng.normal(size=(5, 72, 3)) + 1j*rng.normal(size=(5, 72, 3))
    v = rng.normal(size=(5, 72, 3)) + 1j*rng.normal(size=(5, 72, 3))
    planted = u @ v.conj().transpose(0, 2, 1)
    ranks = C._frobenius_ranks(planted, tolerances=(1e-10,))
    assert ranks["1e-10"]["min"] == 3
    assert ranks["1e-10"]["max"] == 3
