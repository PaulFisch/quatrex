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
    ProbePlan,
    assemble_m_blocks,
    delta_from_sigma,
    nnz_to_blocks,
    probe_sigma_retarded,
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
    """Delta = Sigma^> - Sigma^< (raw) == sigma_lesser - sigma_greater (stored).

    The bubble negates its output before writing, because the solver stores
    occupation-positive quantities while the Keldysh feedback expects the
    textbook sign. So the raw difference the retarded reconstruction is built
    from is the NEGATIVE of the stored one, and Delta comes back with the
    lesser/greater roles swapped.
    """
    rng = np.random.default_rng(0)
    sl_raw = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    sg_raw = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    delta_raw = sg_raw - sl_raw                    # what the SSE transforms

    stored_l, stored_g = -sl_raw, -sg_raw          # what the solver keeps
    assert np.allclose(_h(delta_from_sigma(stored_l, stored_g)), delta_raw)


# --------------------------------------------------------------------------- #
# The batched contraction.
# --------------------------------------------------------------------------- #

def test_batched_probe_matches_single_evaluations():
    rng = np.random.default_rng(1)
    freqs = np.linspace(0.0, 20.0, 201)
    delta = -1j * (0.02 * freqs * np.exp(-((freqs / 9.0) ** 2)))[:, None] * np.ones(4)
    z = np.array([7.0 - 0.01j, 11.0 - 0.05j, 13.0 - 0.2j])

    got = probe_sigma_retarded(delta, freqs, z, orders=(0, 1))
    for order in (0, 1):
        ref = _h(sigma_retarded_at_z(delta, freqs, z, sheet="II", order=order))
        assert np.abs(_h(got[order]) - ref).max() / np.abs(ref).max() < 1e-12


def test_probe_plan_batches_and_labels():
    freqs = np.linspace(0.0, 20.0, 101)
    delta = -1j * (0.02 * freqs)[:, None] * np.ones(3)
    plan = ProbePlan(orders=(0, 1))
    i0 = plan.add(7.0 - 0.01j, tag="cluster-a")
    i1 = plan.add(11.0 - 0.05j, tag="cluster-b")
    assert (i0, i1, len(plan)) == (0, 1, 2)
    out = plan.evaluate(delta, freqs)
    assert set(out) == {0, 1}
    assert _h(out[0]).shape == (2, 3)
    assert plan.tags == ["cluster-a", "cluster-b"]


def test_empty_probe_plan_is_a_no_op():
    freqs = np.linspace(0.0, 20.0, 51)
    delta = np.zeros((51, 3), dtype=complex)
    out = ProbePlan(orders=(0,)).evaluate(delta, freqs)
    assert _h(out[0]).shape == (0, 3)


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

def test_assemble_m_blocks_matches_a_dense_operator():
    block_sizes = np.array([3, 3, 2])
    n = int(block_sizes.sum())
    off = np.concatenate(([0], np.cumsum(block_sizes)))
    rng = np.random.default_rng(4)
    dense_d = rng.normal(size=(n, n))
    dense_d = dense_d + dense_d.T
    d_blocks = {
        (i, j): dense_d[off[i]:off[i + 1], off[j]:off[j + 1]] + 0j
        for i in range(3) for j in range(max(0, i - 1), min(3, i + 2))
    }
    obc_l = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    obc_r = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))

    z = 7.0 - 0.05j
    a_ii, a_ij, a_ji = assemble_m_blocks(
        z, d_blocks, obc_left=obc_l, obc_right=obc_r, block_sizes=block_sizes
    )

    # Build the reference from the block-tridiagonal blocks only: the operator
    # has no corner, and a full dense D would carry one.
    ref = z * z * np.eye(n) + 0j
    for (i, j), b in d_blocks.items():
        ref[off[i]:off[i + 1], off[j]:off[j + 1]] -= b
    ref[off[0]:off[1], off[0]:off[1]] -= obc_l
    ref[off[2]:off[3], off[2]:off[3]] -= obc_r

    from quatrex.phonon.btd_linalg import BTDFactorization
    got = _h(BTDFactorization.factorize(a_ii, a_ij, a_ji).to_dense())
    assert np.abs(got - ref).max() < 1e-12


def test_assemble_m_blocks_subtracts_the_scattering_self_energy():
    block_sizes = np.array([2, 2])
    d_blocks = {(0, 0): np.eye(2) + 0j, (1, 1): np.eye(2) + 0j,
                (0, 1): np.zeros((2, 2)) + 0j, (1, 0): np.zeros((2, 2)) + 0j}
    sig = {k: 0.5 * np.ones_like(v) for k, v in d_blocks.items()}
    z = 3.0 - 0.1j
    bare = assemble_m_blocks(z, d_blocks, block_sizes=block_sizes)[0][0]
    with_s = assemble_m_blocks(z, d_blocks, sig, block_sizes=block_sizes)[0][0]
    assert np.allclose(_h(bare) - _h(with_s), 0.5 * np.ones((2, 2)))
