"""Splitting the transverse-momentum axis across ``comm.q`` (``q_distributed``).

The axis after the stack axis used to be replicated on every rank, so a 25- or
81-point mesh multiplied every buffer everywhere and ``q_comm_size > 1`` made
per-rank memory WORSE -- it shrank the stack section and left ``nq`` whole.

The split is OPT-IN because ``q_comm_size > 1`` already meant something else:
distribute the external-q loop while keeping the data whole, which the phonon
SSE still relies on.

Run with:  mpirun -np 2 pytest --with-mpi tests/qttools/datastructures/test_dsdbsparse_qsplit.py
"""

import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD as global_comm
from qttools.comm import comm
from qttools.datastructures import DSDBCOO
from qttools.utils.gpu_utils import get_host
from scipy.sparse import csr_matrix

_BACKEND = {
    "all_to_all": "device_mpi", "all_gather": "device_mpi",
    "all_reduce": "device_mpi", "bcast": "device_mpi",
    "send_recv": "device_mpi",
}

NE, NQ = 8, 5           # nq deliberately not divisible by 2
BLOCK_SIZES = np.array([2, 2, 2])


def _configure(q_comm_size: int) -> None:
    comm.configure(
        block_comm_size=1,
        block_comm_config=_BACKEND,
        stack_comm_config=_BACKEND,
        q_comm_size=q_comm_size,
        q_comm_config=_BACKEND,
        override=True,
    )


def _make():
    n = int(BLOCK_SIZES.sum())
    dense = np.ones((n, n), dtype=np.complex128)
    return DSDBCOO.from_sparray(
        csr_matrix(dense), BLOCK_SIZES, global_stack_shape=(NE, NQ),
        q_distributed=True,
    )


@pytest.mark.mpi(min_size=2)
def test_q_axis_is_sectioned_not_replicated() -> None:
    _configure(q_comm_size=global_comm.size)
    a = _make()
    a.allocate_data()

    bounds = [r * NQ // comm.q.size for r in range(comm.q.size + 1)]
    lo, hi = bounds[comm.q.rank], bounds[comm.q.rank + 1]

    assert a.local_q_shape == (hi - lo,)
    assert a.local_q_offset == lo
    # The stored middle axis is the SECTION, which is the whole point.
    assert a.data.shape[1] == hi - lo
    assert a.data.shape[1] < NQ  # actually smaller than the replicated case
    # And the sections tile the axis exactly, with no padding.
    sizes = np.empty(comm.q.size, dtype=int)
    comm.q.all_gather(np.asarray([hi - lo], dtype=int), sizes)
    assert int(sizes.sum()) == NQ


@pytest.mark.mpi(min_size=2)
def test_q_index_helpers_round_trip_and_refuse_off_rank() -> None:
    _configure(q_comm_size=global_comm.size)
    a = _make()
    a.allocate_data()

    for iq in range(NQ):
        owner = a.q_owner(iq)
        assert 0 <= owner < comm.q.size
        if owner == comm.q.rank:
            assert a.local_q_index(iq) == iq - a.local_q_offset
        else:
            # Refusing beats wrapping: a silently wrong q index reads a
            # different momentum and shows up as physics, not as a crash.
            with pytest.raises(IndexError, match="not stored"):
                a.local_q_index(iq)

    # every global index has exactly one owner
    owners = [a.q_owner(iq) for iq in range(NQ)]
    counts = np.empty(comm.q.size, dtype=int)
    comm.q.all_gather(
        np.asarray([sum(o == comm.q.rank for o in owners)], dtype=int), counts)
    assert int(counts.sum()) == NQ


@pytest.mark.mpi(min_size=2)
def test_dtranspose_round_trips_with_a_split_q_axis() -> None:
    """The q axis is a pure batch axis, so the all-to-all must ignore it."""
    _configure(q_comm_size=global_comm.size)
    a = _make()
    a.allocate_data()
    rng = np.random.default_rng(3 + comm.q.rank)
    a.data[:] = (rng.standard_normal(a.data.shape)
                 + 1j * rng.standard_normal(a.data.shape))
    before = np.asarray(get_host(a.data)).copy()

    a.dtranspose()
    assert a.distribution_state == "nnz"
    a.dtranspose()
    assert a.distribution_state == "stack"

    np.testing.assert_allclose(
        np.asarray(get_host(a.data)), before, rtol=0, atol=0,
        err_msg="dtranspose did not round-trip with a split q axis",
    )


@pytest.mark.mpi(min_size=2)
def test_multiple_transverse_axes_are_refused() -> None:
    """Which axis to split would be a guess; the vertex fixes the flattening."""
    _configure(q_comm_size=global_comm.size)
    n = int(BLOCK_SIZES.sum())
    with pytest.raises(ValueError, match="exactly one transverse axis"):
        DSDBCOO.from_sparray(
            csr_matrix(np.ones((n, n), dtype=np.complex128)),
            BLOCK_SIZES, global_stack_shape=(NE, 2, 3), q_distributed=True,
        )
