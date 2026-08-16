# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Assembling M(z) in the solver's own data layout.

Covers the two things that can only go wrong at the boundary between the pole
sector and the production buffers: the SIGN of Delta as the solver stores it,
and the scatter from the sparse value vector into dense blocks. Both are silent
failures -- a sign error moves every pole to the wrong half plane, and a bad
scatter produces a plausible operator with the wrong matrix elements.
"""
import numpy as np
import pytest
import scipy.sparse as sps

from qttools import xp
from qttools.comm import comm as _qtt_comm
from qttools.datastructures import DSDBCOO

from quatrex.phonon.pole_kernel import sigma_retarded_at_z
from quatrex.phonon.pole_probe import (
    delta_from_sigma,
    nnz_to_blocks,
)

TINY = 1e-30


def _configure_serial_comm() -> None:
    if _qtt_comm._is_configured:
        return
    backend = "device_mpi" if xp.__name__ == "numpy" else "host_mpi"
    cfg = {k: backend for k in ("all_to_all", "all_gather", "all_reduce", "bcast")}
    _qtt_comm.configure(block_comm_size=1, block_comm_config=cfg,
                        stack_comm_config=cfg, override=True)


def setup_module() -> None:
    _configure_serial_comm()


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _btd_pattern(block_sizes):
    n = int(np.sum(block_sizes))
    off = np.concatenate(([0], np.cumsum(block_sizes)))
    m = np.zeros((n, n))
    for i in range(len(block_sizes)):
        for j in range(max(0, i - 1), min(len(block_sizes), i + 2)):
            m[off[i]:off[i + 1], off[j]:off[j + 1]] = 1.0
    return sps.csr_matrix(m)


# --------------------------------------------------------------------------- #
# The sign convention.
# --------------------------------------------------------------------------- #

def test_delta_sign_matches_the_stored_convention():
    """Delta = Sigma^> - Sigma^< (raw) == sigma_lesser - sigma_greater
    (stored)."""
    rng = np.random.default_rng(0)
    sl_raw = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    sg_raw = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    delta_raw = sg_raw - sl_raw                    # what the SSE transforms

    stored_l, stored_g = -sl_raw, -sg_raw          # what the solver keeps
    assert np.allclose(_h(delta_from_sigma(stored_l, stored_g)), delta_raw)


# --------------------------------------------------------------------------- #
# The batched contraction.
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# The scatter into blocks, against a real DSDBCOO pattern.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("block_sizes", [np.array([3, 3, 3]), np.array([2, 4, 3])])
def test_nnz_to_blocks_matches_the_datastructure(block_sizes):
    """The scatter must reproduce what DSDBSparse's own block view returns."""
    pattern = _btd_pattern(block_sizes)
    ne = 4
    mat = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(ne,))
    rng = np.random.default_rng(2)
    mat.data[:] = xp.asarray(
        rng.normal(size=mat.data.shape) + 1j * rng.normal(size=mat.data.shape)
    )

    got = nnz_to_blocks(mat.data, mat.rows, mat.cols, block_sizes, band=1)
    for i in range(len(block_sizes)):
        for j in range(max(0, i - 1), min(len(block_sizes), i + 2)):
            ref = _h(mat.blocks[i, j])
            assert np.abs(_h(got[(i, j)]) - ref).max() < 1e-14, f"block ({i},{j})"


def test_nnz_to_blocks_survives_the_distribution_transpose():
    """Contract in "nnz", then read blocks in "stack" -- the intended round trip."""
    block_sizes = np.array([3, 3, 3])
    pattern = _btd_pattern(block_sizes)
    ne = 8
    mat = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(ne,))
    rng = np.random.default_rng(3)
    ref_data = rng.normal(size=mat.data.shape) + 1j * rng.normal(size=mat.data.shape)
    mat.data[:] = xp.asarray(ref_data)
    blocks_before = {(i, j): _h(mat.blocks[i, j])
                     for i in range(3) for j in range(max(0, i - 1), min(3, i + 2))}

    mat.dtranspose()                 # stack -> nnz
    assert mat.distribution_state == "nnz"
    mat.dtranspose()                 # nnz -> stack
    assert mat.distribution_state == "stack"

    got = nnz_to_blocks(mat.data, mat.rows, mat.cols, block_sizes, band=1)
    for key, ref in blocks_before.items():
        assert np.abs(_h(got[key]) - ref).max() < 1e-14, f"block {key}"


# --------------------------------------------------------------------------- #
# Assembling the operator.
# --------------------------------------------------------------------------- #

