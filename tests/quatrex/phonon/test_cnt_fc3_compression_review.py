from __future__ import annotations

import numpy as np

from studies._cnt_fc3_compression_review import (
    _s2_hosvd,
    atom_triplet_support,
    block_tucker_sweep,
    relative_asr_errors,
    relative_s2_error,
    representative_offset_tensor,
)


def test_representative_offset_tensor_keeps_exact_support_zeros():
    b = 3
    sizes = np.array([b, b, b, b])
    blocks = {}
    for i in range(4):
        blocks[(i, i, i)] = np.full((b, b, b), 2.0)
        if i + 1 < 4:
            blocks[(i, i, i + 1)] = np.full((b, b, b), 3.0)
            blocks[(i, i + 1, i)] = np.full((b, b, b), 3.0)

    tensor, external, support = representative_offset_tensor(blocks, sizes)

    assert external in (1, 2)
    assert support == [(0, 0), (0, 1), (1, 0)]
    assert tensor.shape == (b, 3 * b, 3 * b)
    assert np.all(tensor[:, b:2 * b, b:2 * b] == 2.0)
    assert np.all(tensor[:, b:2 * b, 2 * b:] == 3.0)
    assert np.all(tensor[:, :b, :] == 0.0)


def test_atom_triplet_support_counts_cartesian_blocks_not_entries():
    block = np.zeros((6, 6, 6))
    block[:3, 3:, :3] = 1.0
    audit = atom_triplet_support(block)
    assert audit["active"] == 1
    assert audit["total"] == 8
    assert audit["squared_norm_kept"] == 1.0


def test_symmetry_and_asr_diagnostics_are_normalized():
    # Two contracted atoms with opposite contributions obey the uniform-mass
    # sum rule on each contracted leg.
    base = np.arange(27.0).reshape(3, 3, 3)
    tensor = np.zeros((3, 6, 6))
    tensor[:, :3, :3] = base
    tensor[:, 3:, :3] = -base
    tensor[:, :3, 3:] = -base
    tensor[:, 3:, 3:] = base
    assert relative_asr_errors(tensor) == (0.0, 0.0)
    assert relative_s2_error(tensor) > 0.0


def test_local_tucker_is_exact_at_full_rank_and_keeps_s2():
    rng = np.random.default_rng(24)
    block = rng.normal(size=(6, 6, 6))
    block = 0.5 * (block + block.transpose(0, 2, 1))
    reconstructed, _ = _s2_hosvd(block, rank=6)
    assert np.linalg.norm(reconstructed - block) / np.linalg.norm(block) < 1e-12
    assert relative_s2_error(reconstructed) < 1e-13


def test_support_block_tucker_keeps_pair_symmetry_at_full_rank():
    rng = np.random.default_rng(25)
    b = 6
    tensor = np.zeros((b, 3 * b, 3 * b))
    onsite = rng.normal(size=(b, b, b))
    onsite = 0.5 * (onsite + onsite.transpose(0, 2, 1))
    cross = rng.normal(size=(b, b, b))
    tensor[:, b:2 * b, b:2 * b] = onsite
    tensor[:, b:2 * b, 2 * b:] = cross
    tensor[:, 2 * b:, b:2 * b] = cross.transpose(0, 2, 1)
    row = block_tucker_sweep(tensor, b, ranks=(6,))[0]
    assert row["relative_frobenius_error"] < 1e-12
    assert row["s2_error"] < 1e-13
