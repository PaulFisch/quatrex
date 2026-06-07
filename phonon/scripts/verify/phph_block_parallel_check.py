"""Bit-identity of the distributed 3-phonon SSE under a comm.block x comm.stack
factorisation.

Runs ``SigmaPhononPhonon.compute`` with the transport blocks distributed over
``comm.block`` (the 1-D spatial-halo path) and the energies over ``comm.stack``,
and checks every owned output block against the serial dense oracle. The halo is
genuinely exercised because each block-rank owns a contiguous sub-range and the
boundary outputs need band blocks from the neighbour's arrow.

Run, e.g.::

    BLK=2 mpirun -np 4 python phonon/scripts/verify/phph_block_parallel_check.py
    BLK=2 mpirun -np 2 python phonon/scripts/verify/phph_block_parallel_check.py
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as world
from scipy.sparse import csr_matrix

from qttools.comm import comm as ranks
from qttools.datastructures import DSDBCOO

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")

BLK = int(os.environ.get("BLK", "2"))
_cfg = {k: "device_mpi" for k in ("all_to_all", "all_gather", "all_reduce", "bcast")}
ranks.configure(block_comm_size=BLK, block_comm_config=_cfg,
                stack_comm_config=_cfg, override=True)

_spec = importlib.util.spec_from_file_location(
    "_t", str(_REPO / "tests/quatrex/phonon/test_sse_phonon_phonon.py"))
_t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t)
from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon  # noqa: E402


def main(n_blocks: int = 4, nbs: int = 3, ne: int = 21) -> int:
    rng = np.random.default_rng(42)
    block_sizes = np.array([nbs] * n_blocks)
    N = int(block_sizes.sum())

    phi_blocks = {}
    for I in range(n_blocks):
        for K1 in range(max(0, I - 1), min(n_blocks, I + 2)):
            for K2 in range(max(0, I - 1), min(n_blocks, I + 2)):
                if abs(K1 - K2) > 1:
                    continue
                phi_blocks[(I, K1, K2)] = (rng.standard_normal((nbs, nbs, nbs))
                                           + 1j * rng.standard_normal((nbs, nbs, nbs)))
    gl_band, gg_band = {}, {}
    for K in range(n_blocks):
        for Kp in range(max(0, K - 1), min(n_blocks, K + 2)):
            gl_band[(K, Kp)] = (rng.standard_normal((ne, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nbs, nbs)))
            gg_band[(K, Kp)] = (rng.standard_normal((ne, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nbs, nbs)))

    rows, cols = [], []
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            for i in range(nbs):
                for j in range(nbs):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = csr_matrix((np.ones(len(rows), dtype=np.complex128),
                          (np.array(rows), np.array(cols))), shape=(N, N))
    bufs = [DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(ne,))
            for _ in range(5)]
    g_l, g_g, s_l, s_g, s_r = bufs
    for m in bufs:
        m.data[:] = 0.0

    bstart = int(g_l.block_section_offsets[ranks.block.rank])
    bend = int(g_l.block_section_offsets[ranks.block.rank + 1])
    e_lo = int(np.sum(g_l.stack_section_sizes[: ranks.stack.rank]))
    e_hi = e_lo + int(g_l.stack_section_sizes[ranks.stack.rank])
    glv, ggv = g_l.stack[...], g_g.stack[...]
    for (K, Kp) in gl_band:  # each band block lives on the rank owning min(K,Kp)
        if bstart <= min(K, Kp) < bend:
            glv.blocks[K - bstart, Kp - bstart] = gl_band[(K, Kp)][e_lo:e_hi]
            ggv.blocks[K - bstart, Kp - bstart] = gg_band[(K, Kp)][e_lo:e_hi]

    freqs = np.linspace(0.0, 16.0, ne)
    dw = float(freqs[1] - freqs[0])
    ssp = SigmaPhononPhonon(_t._make_cfg("fft"), phonon_frequencies=freqs[e_lo:e_hi],
                            block_sizes=block_sizes, phi_blocks=phi_blocks)
    ssp.compute(g_l, g_g, out=(s_l, s_g, s_r))

    sl_ref, sg_ref, sr_ref = _t._ref_compute_multiblock(
        phi_blocks, gl_band, gg_band, block_sizes, dw)
    slv, sgv, srv = s_l.stack[...], s_g.stack[...], s_r.stack[...]
    maxerr = 0.0
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            if not (bstart <= min(I, J) < bend):
                continue
            for v, ref in ((slv, sl_ref), (sgv, sg_ref), (srv, sr_ref)):
                got = np.asarray(v.blocks[I - bstart, J - bstart])
                exp = ref.get((I, J), 0)[e_lo:e_hi]
                maxerr = max(maxerr, float(
                    np.max(np.abs(got - exp)) / (np.max(np.abs(exp)) + 1e-300)))
    allerr = world.allreduce(maxerr, op=MPI.MAX)
    if world.rank == 0:
        print(f"[np={world.size} block={ranks.block.size} stack={ranks.stack.size}] "
              f"max rel err vs serial oracle = {allerr:.2e}")
        print("PASS" if allerr < 1e-12 else "FAIL")
    return 0 if allerr < 1e-12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
