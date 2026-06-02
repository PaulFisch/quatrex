"""Distributed scaling of the PRODUCTION quatrex phph self-energy.

Times src/quatrex/phonon/sse_phonon_phonon.SigmaPhononPhonon.compute on a
synthetic block-tridiagonal phonon problem, distributed across MPI ranks.
The communicator is a 2D grid (block x stack); QPHPH_BCS sets block_comm_size:
  - QPHPH_BCS=1            : pure energy(stack) parallelism
  - QPHPH_BCS=<np>         : pure block parallelism
This exposes the key fact: the 3-phonon bubble is an energy CONVOLUTION, so the
stack-distributed implementation all-gathers the full omega axis and each rank
recomputes the full-omega FFT (replicated) -- energy parallelism does not divide
the bubble compute, whereas the (I,J) block loop does.

Run:  QPHPH_BCS=1 mpirun -np N python phph_dist_scaling.py
"""
import os
import sys
import time

for p in ("/usr/scratch/mont-fort11/pfischill/quatrex/src",
          "/usr/scratch/mont-fort11/pfischill/quatrex"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from mpi4py.MPI import COMM_WORLD as world
from scipy.sparse import csr_matrix
from qttools.comm import comm
from qttools.datastructures import DSDBCOO
from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

BCS = int(os.environ.get("QPHPH_BCS", "1"))
NBLK = int(os.environ.get("QPHPH_NBLK", "8"))
BS = int(os.environ.get("QPHPH_BS", "24"))
NE = int(os.environ.get("QPHPH_NE", "256"))     # global energy points
REP = int(os.environ.get("QPHPH_REP", "3"))

_cfg = {k: "device_mpi" for k in ("all_to_all", "all_gather", "all_reduce", "bcast")}
comm.configure(block_comm_size=BCS, block_comm_config=_cfg,
               stack_comm_config=_cfg, override=True)

block_sizes = np.full(NBLK, BS)
N = NBLK * BS
# block-tridiagonal sparsity pattern
rows, cols = [], []
off = np.concatenate(([0], np.cumsum(block_sizes)))
for I in range(NBLK):
    for J in range(max(0, I - 1), min(NBLK, I + 2)):
        for i in range(BS):
            for j in range(BS):
                rows.append(off[I] + i); cols.append(off[J] + j)
pattern = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N)).astype(complex)

# nearest-neighbour FC3 blocks (same on all ranks)
rng = np.random.default_rng(0)
phi_blocks = {}
for I in range(NBLK):
    for K1 in range(max(0, I - 1), min(NBLK, I + 2)):
        for K2 in range(max(0, I - 1), min(NBLK, I + 2)):
            if abs(K1 - K2) > 1:
                continue
            phi_blocks[(I, K1, K2)] = (rng.standard_normal((BS, BS, BS))
                                       + 1j * rng.standard_normal((BS, BS, BS)))

gl = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(NE,))
gg = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(NE,))
sl = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(NE,))
sg = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(NE,))
sr = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(NE,))
for m in (gl, gg, sl, sg, sr):
    m.data[:] = 0.0

# local energy slice + local block range for this rank
ne_local = int(gl.stack_section_sizes[comm.stack.rank])
e_lo = int(np.sum(gl.stack_section_sizes[: comm.stack.rank]))
full_freqs = np.linspace(0.0, 16.0, NE)
freqs_local = full_freqs[e_lo:e_lo + ne_local]
n_local_blocks = len(gl.local_block_sizes)

glv = gl.stack[...]; ggv = gg.stack[...]
for Kl in range(n_local_blocks):
    glv.blocks[Kl, Kl] = (rng.standard_normal((ne_local, BS, BS))
                          + 1j * rng.standard_normal((ne_local, BS, BS)))
    ggv.blocks[Kl, Kl] = (rng.standard_normal((ne_local, BS, BS))
                          + 1j * rng.standard_normal((ne_local, BS, BS)))

ssp = SigmaPhononPhonon(type("C", (), {"phonon": type("P", (), {
    "retarded_method": "fft", "fc3_path": None})()})(),
    phonon_frequencies=freqs_local, block_sizes=block_sizes, phi_blocks=phi_blocks)

# warm-up + timed reps
ssp.compute(gl, gg, out=(sl, sg, sr))
times = []
for _ in range(REP):
    for m in (sl, sg, sr):
        m.data[:] = 0.0
    world.Barrier(); t0 = time.time()
    ssp.compute(gl, gg, out=(sl, sg, sr))
    world.Barrier(); times.append(time.time() - t0)
dt = float(np.median(times))
# global Sigma norm (correctness fingerprint, should be ~constant across configs)
local_norm = float(np.linalg.norm(sl.data))
norm2 = world.allreduce(local_norm ** 2)
if world.rank == 0:
    print(f"RANKS={world.size} BCS={BCS} stack={world.size//BCS} "
          f"NBLK={NBLK} BS={BS} NE={NE}  phph wall={dt:.3f}s  "
          f"|Sigma<|={norm2**0.5:.4e}", flush=True)
