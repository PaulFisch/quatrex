"""The contact OBC needs two blocks on the last ``comm.block`` rank.

Found by the first device run of ``nq > 1`` with ``block_comm_size = 2``
(daint 4419787): a 2-block MoS2 film split over two block ranks left every
odd rank with a single block, and ``_compute_obc`` addressed the block
before it -- local index ``-1``. The run died inside ``DSDBSparse`` with
``IndexError: Negative block indices are not supported``, four frames away
from the actual precondition.

Rank 0 is deliberately NOT part of the requirement: its ``local_block_sizes``
run to the end of the device and it holds the nnz for the trailing rows, so
``blocks[1, 0]`` reads correct values even when it owns one block. That was
verified directly against a dense reference before this guard was written.
"""

import numpy as np
import pytest
from qttools.utils.mpi_utils import get_section_sizes

from quatrex.phonon.solver import validate_obc_block_sections


@pytest.mark.parametrize("block_comm_size", [1])
@pytest.mark.parametrize("num_blocks", [1, 2, 3, 8])
def test_serial_is_never_rejected(num_blocks: int, block_comm_size: int) -> None:
    """A single block rank owns everything, so nothing can be split off."""
    validate_obc_block_sections(num_blocks, block_comm_size)


@pytest.mark.parametrize(
    "num_blocks, block_comm_size",
    [
        (2, 2),   # [1, 1] -- the daint 4419787 geometry
        (3, 2),   # [2, 1]
        (2, 3),   # [1, 1, 0] -- a rank with no blocks at all
        (5, 4),   # [2, 1, 1, 1]
        (7, 4),   # [2, 2, 2, 1]
    ],
)
def test_short_last_section_is_rejected(
    num_blocks: int, block_comm_size: int
) -> None:
    """The last rank must be able to reach its own predecessor block."""
    with pytest.raises(ValueError, match="right contact OBC"):
        validate_obc_block_sections(num_blocks, block_comm_size)


@pytest.mark.parametrize(
    "num_blocks, block_comm_size",
    [(4, 2), (5, 2), (6, 2), (6, 3), (8, 4), (16, 4)],
)
def test_sufficient_sections_pass(
    num_blocks: int, block_comm_size: int
) -> None:
    validate_obc_block_sections(num_blocks, block_comm_size)


@pytest.mark.parametrize("block_comm_size", [2, 3, 4])
def test_guard_matches_the_local_index_arithmetic(block_comm_size: int) -> None:
    """The guard passes exactly when ``m = num_local_blocks - 2 >= 0``.

    ``_compute_obc`` reads ``blocks[m, n]`` with ``n = num_local_blocks - 1``
    and ``m = n - 1`` on the last rank; anything else would let a runnable
    configuration be refused, or a broken one through.
    """
    for num_blocks in range(1, 20):
        sections, __ = get_section_sizes(num_blocks, block_comm_size)
        reachable = int(sections[-1]) - 2 >= 0
        try:
            validate_obc_block_sections(num_blocks, block_comm_size)
            passed = True
        except ValueError:
            passed = False
        assert passed == reachable, (
            f"num_blocks={num_blocks} sections={list(map(int, sections))}"
        )


def test_error_names_the_two_ways_out() -> None:
    with pytest.raises(ValueError) as exc:
        validate_obc_block_sections(2, 2)
    message = str(exc.value)
    assert "longer device" in message
    assert "block_comm_size" in message
    assert "[1, 1]" in message


def test_num_blocks_may_be_numpy_integer() -> None:
    """``num_blocks`` comes off a DSDBSparse, where it can be np.int64."""
    validate_obc_block_sections(np.int64(6), 2)
    with pytest.raises(ValueError):
        validate_obc_block_sections(np.int64(2), 2)
