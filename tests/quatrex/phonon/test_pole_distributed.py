# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
r"""The pole continuation on a split frequency axis.

``Sigma_s^{R,II}(z)`` sums over the WHOLE frequency axis, which ``comm.stack``
splits. Both of its terms -- ``F(z)`` and the second-sheet ``Delta_an(z)`` --
are linear in ``Delta`` with coefficients that depend only on the grid, and
neither reindexes frequency, so each rank can contract the columns it owns and
one sum-reduction completes the answer.

That is cheaper than the transposition the design note originally called for:
``dtranspose`` moves the whole buffer, while this moves only the ``(P, nnz)``
result. It also avoids distributed root finding entirely -- every rank ends up
with the same operator and solves the same poles.

Run the multi-rank cases with::

    $CONDA_PREFIX/bin/mpirun -n 4 $CONDA_PREFIX/bin/python -m pytest \
        tests/quatrex/phonon/test_pole_distributed.py --with-mpi

Use the launcher from the SAME prefix as mpi4py. The env links MPICH while
``/usr/bin/mpirun`` is the system OpenMPI, and mixing them does not fail --
it silently starts N independent one-rank jobs, so every MPI test reports
"only 1 MPI processes specified, skipping" and the suite still says it
passed.
"""
import numpy as np
import pytest

from quatrex.core.config import PoleSectorConfig
from quatrex.phonon.experimental.pole.pole_sector import PoleSector


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _bed(n_freq=64, nnz=12, seed=0):
    freqs = np.linspace(0.0, 20.0, n_freq)
    rng = np.random.default_rng(seed)
    delta = (rng.normal(size=(n_freq, nnz))
             + 1j * rng.normal(size=(n_freq, nnz)))
    return freqs, delta


def _sector(freqs, delta, **kw):
    sec = PoleSector(PoleSectorConfig(enabled=True), freqs, **kw)
    sec._delta = delta
    return sec


def test_serial_path_is_unchanged_by_the_distributed_plumbing():
    """No ``global_freqs``/``reduce`` must reproduce the undistributed answer
    exactly, not merely closely -- the reducer is the identity when serial."""
    from quatrex.phonon.experimental.pole.pole_kernel import sigma_retarded_at_z

    freqs, delta = _bed()
    sec = _sector(freqs, delta)
    z = np.array([5.3 - 0.02j, 12.7 - 0.05j])
    for order in (0, 1, 2):
        got = _h(sec.continue_sigma(z, order=order))
        ref = _h(sigma_retarded_at_z(
            delta, freqs, z, sheet="II", order=order,
            delta_order=sec.cfg.delta_fit_order,
            delta_window=sec.cfg.delta_fit_window_cells))
        assert np.abs(got - ref).max() < 1e-12 * max(np.abs(ref).max(), 1.0)


@pytest.mark.parametrize("n_parts", [2, 3, 4])
def test_split_axis_plus_reduction_equals_the_whole_axis(n_parts):
    """Emulated split: contract each contiguous slice, sum the partials.

    This is the algebra the MPI path relies on, tested without MPI so it runs
    in the ordinary suite. The MPI test below then checks the plumbing.
    """
    freqs, delta = _bed(n_freq=64, seed=1)
    z = np.array([4.1 - 0.03j, 9.9 - 0.02j, 16.2 - 0.04j])
    whole = _h(_sector(freqs, delta).continue_sigma(z, order=0))

    bounds = np.linspace(0, freqs.size, n_parts + 1).astype(int)
    total = np.zeros_like(whole)
    for i in range(n_parts):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        part = _sector(freqs[lo:hi], delta[lo:hi],
                       global_freqs=freqs, freq_offset=lo)
        total = total + _h(part.continue_sigma(z, order=0))
    assert np.abs(total - whole).max() < 1e-12 * max(np.abs(whole).max(), 1.0)


def test_a_rank_local_grid_alone_gives_the_wrong_answer():
    """Negative control: without the global grid and the reduction, a slice is
    not merely less accurate -- it is a different operator."""
    freqs, delta = _bed(n_freq=64, seed=2)
    z = np.array([9.9 - 0.02j])
    whole = _h(_sector(freqs, delta).continue_sigma(z, order=0))
    half = _h(_sector(freqs[:32], delta[:32]).continue_sigma(z, order=0))
    assert np.abs(half - whole).max() / np.abs(whole).max() > 0.1


@pytest.mark.mpi(min_size=2)
def test_mpi_split_matches_the_serial_answer():
    """The real plumbing: comm.stack all_gather of the grid + all_reduce."""
    from mpi4py.MPI import COMM_WORLD as mpi

    freqs, delta = _bed(n_freq=64, seed=3)
    z = np.array([4.1 - 0.03j, 9.9 - 0.02j])
    ref = _h(_sector(freqs, delta).continue_sigma(z, order=0))

    sizes = [freqs.size // mpi.size] * mpi.size
    for i in range(freqs.size % mpi.size):
        sizes[i] += 1
    off = int(np.sum(sizes[:mpi.rank]))
    lo, hi = off, off + sizes[mpi.rank]

    def _reduce(arr):
        out = np.empty_like(np.ascontiguousarray(arr))
        mpi.Allreduce(np.ascontiguousarray(arr), out)
        return out

    part = _sector(freqs[lo:hi], delta[lo:hi], global_freqs=freqs,
                   freq_offset=lo, reduce=_reduce)
    got = _h(part.continue_sigma(z, order=0))
    assert np.abs(got - ref).max() < 1e-12 * max(np.abs(ref).max(), 1.0)


@pytest.mark.mpi
def test_the_production_frequency_context_matches_the_hand_split():
    r"""Exercise ``PhononSolver._pole_frequency_context`` itself."""
    from mpi4py.MPI import COMM_WORLD as mpi

    from qttools.comm import comm
    from quatrex.phonon.solver import PhononSolver

    if mpi.size < 2:
        pytest.skip("needs >= 2 MPI ranks")

    # block_comm_size 1 => the whole world is one stack communicator, which is
    # the layout every phonon run has used (bcs=1, qcs=1).
    comm.configure(block_comm_size=1, block_comm_config={},
                   stack_comm_config={}, override=True)

    n_freq = 65                              # deliberately not divisible
    freqs = np.linspace(0.0, 20.0, n_freq)
    sizes = [n_freq // mpi.size] * mpi.size
    for i in range(n_freq % mpi.size):
        sizes[i] += 1
    off = int(np.sum(sizes[:mpi.rank]))
    local = freqs[off:off + sizes[mpi.rank]]

    ctx = PhononSolver._pole_frequency_context(None, local)

    assert ctx, "the context must be non-empty on a split axis"
    assert ctx["freq_offset"] == off
    got = np.asarray(ctx["global_freqs"], dtype=float)
    assert got.shape == freqs.shape, (
        f"gathered {got.shape} against {freqs.shape}; padding was not trimmed")
    assert np.abs(got - freqs).max() < 1e-12, "the gathered grid is wrong"

    # And the reducer must be a genuine sum over ranks.
    one = np.ones(3, dtype=complex) * (mpi.rank + 1)
    red = np.asarray(_h(ctx["reduce"](one)), dtype=complex)
    expect = sum(r + 1 for r in range(mpi.size))
    assert np.abs(red - expect).max() < 1e-12
