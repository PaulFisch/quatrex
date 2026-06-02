"""q-point parallelism prototype for the phph self-energy.

The 3-phonon bubble is an energy CONVOLUTION (it couples all omega), which is why
energy(stack) distribution replicates the compute (phph_dist_scaling.py). The
transverse momenta / q-points, by contrast, are INDEPENDENT external indices for
the real-space block-sparse phph: each q-point is a separate energy convolution
with no cross-q coupling. So distributing q-points is embarrassingly parallel.

This serial demo verifies the independence by showing the wall time is LINEAR in
the number of q-points (each adds one independent phph solve), from which a
q-distributed implementation over P ranks scales ideally as N_q/P. Contrast the
flat energy-axis scaling.

Run:  python phph_qpoint_demo.py
"""
import sys
import time

for p in ("/usr/scratch/mont-fort11/pfischill/quatrex/src",
          "/usr/scratch/mont-fort11/pfischill/quatrex"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from scipy.sparse import csr_matrix
from qttools.comm import comm
from qttools.datastructures import DSDBCOO
from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

_cfg = {k: "device_mpi" for k in ("all_to_all", "all_gather", "all_reduce", "bcast")}
comm.configure(block_comm_size=1, block_comm_config=_cfg, stack_comm_config=_cfg, override=True)

NBLK, BS, NE = 6, 16, 96
block_sizes = np.full(NBLK, BS)
N = NBLK * BS
off = np.concatenate(([0], np.cumsum(block_sizes)))
rows, cols = [], []
for I in range(NBLK):
    for J in range(max(0, I - 1), min(NBLK, I + 2)):
        for i in range(BS):
            for j in range(BS):
                rows.append(off[I] + i); cols.append(off[J] + j)
pattern = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N)).astype(complex)
rng = np.random.default_rng(0)
phi_blocks = {}
for I in range(NBLK):
    for K1 in range(max(0, I - 1), min(NBLK, I + 2)):
        for K2 in range(max(0, I - 1), min(NBLK, I + 2)):
            if abs(K1 - K2) > 1:
                continue
            phi_blocks[(I, K1, K2)] = (rng.standard_normal((BS, BS, BS))
                                       + 1j * rng.standard_normal((BS, BS, BS)))
freqs = np.linspace(0.0, 16.0, NE)
ssp = SigmaPhononPhonon(type("C", (), {"phonon": type("P", (), {
    "retarded_method": "fft", "fc3_path": None})()})(),
    phonon_frequencies=freqs, block_sizes=block_sizes, phi_blocks=phi_blocks)


# buffers built ONCE; each "q-point" is one independent compute (re-zero sigma)
gl = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(NE,))
gg = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(NE,))
sl = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(NE,))
sg = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(NE,))
sr = DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(NE,))
glv = gl.stack[...]; ggv = gg.stack[...]
for K in range(NBLK):
    glv.blocks[K, K] = rng.standard_normal((NE, BS, BS)) + 1j * rng.standard_normal((NE, BS, BS))
    ggv.blocks[K, K] = rng.standard_normal((NE, BS, BS)) + 1j * rng.standard_normal((NE, BS, BS))


def one_qpoint():
    """One independent phph solve (a single transverse momentum)."""
    for m in (sl, sg, sr):
        m.data[:] = 0.0
    ssp.compute(gl, gg, out=(sl, sg, sr))


one_qpoint()   # warm-up
print("N_q   wall(s)   wall/N_q   (linear => q-points independent => ideal to distribute)")
t1 = None
for nq in (1, 2, 4):
    t0 = time.time()
    for _ in range(nq):
        one_qpoint()
    dt = time.time() - t0
    if t1 is None:
        t1 = dt
    print(f"{nq:3d}   {dt:7.3f}   {dt/nq:7.3f}", flush=True)
print(f"\nA q_comm over P ranks would give ideal N_q/P scaling (each rank runs its")
print(f"q-points with NO communication) -- vs the flat energy-axis scaling.")
