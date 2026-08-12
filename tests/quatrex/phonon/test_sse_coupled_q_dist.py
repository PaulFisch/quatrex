"""Transverse q together with block-parallel transport.

``nq > 1`` and ``comm.block.size > 1`` used to be refused outright. The reason
was concrete rather than deep: ``_exchange_band_halo`` sized its buffers as
``(local_tau, b_K, b_Kp)``, with no transverse axis, so the halo would have
posted buffers ``nq`` times too small. The bosonic fold was already fine --
it works on the nnz axis with ``data.shape[:-1]`` carried through untouched.

This is what lets a device be re-blocked to C transport cells per BTD block
(``phonon/studies/engine/reblock_device.py``), which widens the retained
interaction range to +-C cells at the same block band, AND still be spread
over ``comm.block`` -- which is the combination the MoS2 film needs.

Run with:  mpirun -np 2 pytest --with-mpi tests/quatrex/phonon/test_sse_coupled_q_dist.py

The comparison is between two rank layouts of the SAME two ranks, so both
sides run the production code and neither is a hand-written reference:

* ``(block=2, q=1)`` -- transport split, every rank holds all energies and a
  slice of the blocks;
* ``(block=1, q=2)`` -- transport whole on each rank, the external-q loop
  split over ``comm.q`` and summed back.

Both have ``comm.stack.size == 1``, so the energy axis is identical and the
owned blocks are directly comparable.
"""

import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD as global_comm
from qttools import xp
from qttools.comm import comm
from qttools.datastructures import DSDBCOO
from qttools.utils.gpu_utils import get_host
from scipy.sparse import csr_matrix

from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

from .test_sse_phonon_phonon import _dev_pattern, _make_cfg

# NumPy arrays go through the device_mpi (buffer) path; host_mpi is the
# GPU-staging backend and is rejected outside CuPy.
_BACKEND = {
    "all_to_all": "device_mpi", "all_gather": "device_mpi",
    "all_reduce": "device_mpi", "bcast": "device_mpi",
    "send_recv": "device_mpi",
}

N_BLOCKS, NBS, NE, NQ, G_BAND = 4, 3, 13, 3, 1


def _bed():
    """One random coupled-q bed: vertices, G bands and the device pattern.

    The vertex reality ``Phi(-q1,-q2) = conj(Phi(q1,q2))`` of a real
    real-space FC3 is imposed, because the bosonic fold assumes it and the
    q -> -q gather is exercised at ``nq = 3``.
    """
    rng = np.random.default_rng(11)
    block_sizes = np.array([NBS] * N_BLOCKS)
    n = int(block_sizes.sum())
    q_diff = np.array([[(a - b) % NQ for b in range(NQ)] for a in range(NQ)])
    keys = [
        (I, K1, K2)
        for I in range(N_BLOCKS)
        for K1 in range(max(0, I - 1), min(N_BLOCKS, I + 2))
        for K2 in range(max(0, I - 1), min(N_BLOCKS, I + 2))
        if abs(K1 - K2) <= 1
    ]

    def _phi():
        return {k: rng.standard_normal((NBS, NBS, NBS))
                + 1j * rng.standard_normal((NBS, NBS, NBS)) for k in keys}

    qv: dict = {}
    for a in range(NQ):
        for b in range(NQ):
            na, nb = (-a) % NQ, (-b) % NQ
            if (na, nb) in qv:
                qv[(a, b)] = {k: np.conj(v) for k, v in qv[(na, nb)].items()}
            elif (na, nb) == (a, b):
                qv[(a, b)] = {k: v.real.astype(complex)
                              for k, v in _phi().items()}
            else:
                qv[(a, b)] = _phi()

    gl, gg = {}, {}
    for K in range(N_BLOCKS):
        for Kp in range(max(0, K - G_BAND), min(N_BLOCKS, K + G_BAND + 1)):
            gl[(K, Kp)] = (rng.standard_normal((NE, NQ, NBS, NBS))
                           + 1j * rng.standard_normal((NE, NQ, NBS, NBS)))
            gg[(K, Kp)] = (rng.standard_normal((NE, NQ, NBS, NBS))
                           + 1j * rng.standard_normal((NE, NQ, NBS, NBS)))

    rows, cols = [], []
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    for I in range(N_BLOCKS):
        for J in range(max(0, I - G_BAND), min(N_BLOCKS, I + G_BAND + 1)):
            for i in range(block_sizes[I]):
                for j in range(block_sizes[J]):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = _dev_pattern(csr_matrix(
        (np.ones(len(rows), np.complex128), (np.array(rows), np.array(cols))),
        shape=(n, n)))
    return block_sizes, qv, q_diff, gl, gg, pattern


def _run(block_comm_size: int, q_comm_size: int,
         q_distributed: bool = False):
    """Compute Sigma under one rank layout; returns the (I, J) blocks."""
    comm.configure(
        block_comm_size=block_comm_size,
        block_comm_config=_BACKEND,
        stack_comm_config=_BACKEND,
        q_comm_size=q_comm_size,
        q_comm_config=_BACKEND,
        override=True,
    )
    block_sizes, qv, q_diff, gl, gg, pattern = _bed()

    def mk():
        return DSDBCOO.from_sparray(pattern, block_sizes,
                                    global_stack_shape=(NE, NQ),
                                    q_distributed=q_distributed)

    g_l, g_g, s_l, s_g, s_r = mk(), mk(), mk(), mk(), mk()
    for m in (g_l, g_g, s_l, s_g, s_r):
        m.data[:] = 0.0
    glv, ggv = g_l.stack[...], g_g.stack[...]
    start = int(g_l.block_section_offsets[comm.block.rank])
    end = int(g_l.block_section_offsets[comm.block.rank + 1])
    # The block view is LOCAL-indexed under a block split, and the SSE reads a
    # link exactly when ``start <= min(K, Kp) < end`` -- mirror that here so
    # both layouts are fed the identical G and the halo has something to
    # fetch on the other side of every boundary.
    for (K, Kp) in gl:
        if start <= min(K, Kp) < end:
            qs = g_l.local_q_slice
            glv.blocks[K - start, Kp - start] = xp.asarray(gl[(K, Kp)][:, qs])
            ggv.blocks[K - start, Kp - start] = xp.asarray(gg[(K, Kp)][:, qs])

    cfg = _make_cfg("half", g_band=G_BAND)
    ssp = SigmaPhononPhonon(
        cfg, phonon_frequencies=np.linspace(0.0, 16.0, NE),
        block_sizes=block_sizes, qfold=(qv, q_diff, NQ),
    )
    ssp.compute(g_l, g_g, out=(s_l, s_g, s_r))
    slv, sgv = s_l.stack[...], s_g.stack[...]
    out = {}
    for I in range(start, end):
        for J in range(max(0, I - 1), min(N_BLOCKS, I + 2)):
            if not start <= min(I, J) < end:
                continue
            out[(I, J)] = (
                np.asarray(get_host(slv.blocks[I - start, J - start])).copy(),
                np.asarray(get_host(sgv.blocks[I - start, J - start])).copy(),
            )
    return out


@pytest.mark.mpi(min_size=2)
def test_transverse_q_survives_block_parallel_transport() -> None:
    """Splitting transport over comm.block must not change Sigma."""
    assert global_comm.size >= 2
    dist = _run(block_comm_size=global_comm.size, q_comm_size=1)
    whole = _run(block_comm_size=1, q_comm_size=global_comm.size)

    shared = sorted(set(dist) & set(whole))
    assert shared, "no Sigma blocks in common between the two layouts"
    for key in shared:
        for name, a, b in (("Sigma^<", dist[key][0], whole[key][0]),
                           ("Sigma^>", dist[key][1], whole[key][1])):
            scale = max(float(np.abs(b).max()), 1e-300)
            np.testing.assert_allclose(
                a, b, rtol=1e-10, atol=1e-10 * scale,
                err_msg=f"{name} block {key} differs between the block-"
                        f"parallel and block-whole layouts",
            )


@pytest.mark.skip(
    reason="the q rotation needs a reduce-scatter of Sigma over comm.q: it "
           "accumulates at every external momentum, but the Sigma buffer is "
           "sectioned, so the full-nq accumulator overflows it")
@pytest.mark.mpi(min_size=2)
def test_internal_q_rotation_reproduces_the_replicated_result() -> None:
    """A q-SECTIONED G must give the same Sigma as a replicated one.

    This is the whole point of the rotation: the bubble is a convolution over
    q, so a rank holding only its own internal slice cannot form even one
    external momentum by itself. Leg B is passed around ``comm.q`` until every
    ordered slice pair has been visited exactly once.
    """
    whole = _run(block_comm_size=1, q_comm_size=global_comm.size)
    split = _run(block_comm_size=1, q_comm_size=global_comm.size,
                 q_distributed=True)

    shared = sorted(set(whole) & set(split))
    assert shared
    for key in shared:
        for name, a, b in (("Sigma^<", split[key][0], whole[key][0]),
                           ("Sigma^>", split[key][1], whole[key][1])):
            scale = max(float(np.abs(b).max()), 1e-300)
            np.testing.assert_allclose(
                a, b, rtol=1e-10, atol=1e-10 * scale,
                err_msg=f"{name} block {key}: the q rotation does not "
                        f"reproduce the replicated result",
            )
