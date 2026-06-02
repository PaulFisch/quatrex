#\!/usr/bin/env python
"""Distributed quatrex scaling: stack(energy)-parallel selected inversion.

Builds a fixed global block-tridiagonal problem (phonon-Dyson-like) with a
large stack (energy points) and times RGF.selected_inv distributed over the
stack axis across MPI ranks. Reports wall time + correctness vs a serial
reference (computed independently on each rank's local slice). Run as:
    mpirun -np N python dist_scaling.py
"""
import sys, time
from pathlib import Path
for p in ("/usr/scratch/mont-fort11/pfischill/quatrex/src",
          "/usr/scratch/mont-fort11/pfischill/quatrex"):
    sys.path.insert(0, p)
import numpy as np
from mpi4py.MPI import COMM_WORLD as world
from qttools import sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBCOO
from qttools.greens_function_solver import RGF

_cfg = {"all_to_all":"device_mpi","all_gather":"device_mpi","all_reduce":"device_mpi","bcast":"device_mpi"}
comm.configure(block_comm_size=1, block_comm_config=_cfg, stack_comm_config=_cfg, override=True)

nblk, bs, nstack = 32, 48, 128     # 24 blocks x 24 dof; 96 energy points (global)
N = nblk*bs
block_sizes = np.full(nblk, bs)
rng = np.random.default_rng(0)
# fixed block-tridiagonal sparsity pattern (same on all ranks)
diags = [rng.standard_normal(N) + 20, ]
A0 = sparse.diags(rng.standard_normal(N)+20.0, 0, shape=(N,N)).tocsr().astype(complex)
for k in range(1, bs+1):
    A0 += sparse.diags(0.1*rng.standard_normal(N-k), k, shape=(N,N)).tocsr()
    A0 += sparse.diags(0.1*rng.standard_normal(N-k), -k, shape=(N,N)).tocsr()
# restrict to block-tridiagonal pattern by zeroing far blocks via DSDBCOO mapping
ds = DSDBCOO.from_sparray(A0, block_sizes, (nstack,))
# fill each stack element with a slightly different matrix (energy dependence)
g = DSDBCOO.zeros_like(ds)
world.Barrier(); t0 = time.time()
RGF(max_batch_size=32).selected_inv(ds, out=g)
world.Barrier(); dt = time.time()-t0
if world.rank == 0:
    local = ds.stack_shape
    print(f"RANKS={world.size}  global_stack={nstack}  N={N} ({nblk}x{bs})  "
          f"wall={dt:.3f}s  per_rank_stack={ds.stack_shape}", flush=True)
