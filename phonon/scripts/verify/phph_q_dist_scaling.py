"""Distributed q-point scaling of the COUPLED-q 3-phonon self-energy.

The q-resolved self-energy
    Sigma(q_ext, w) = (1/N_q) sum_{q'} Phi(q') [G(q') * G(q_ext - q')](w) Phi(q_ext - q')
couples the q-points by momentum conservation q_ext = q' + (q_ext - q') (build_q_diff_map)
AND convolves in energy. It is NOT embarrassingly parallel (correcting F22): every external
q needs ALL internal q' Green's functions. Distributing the EXTERNAL q over MPI ranks divides
the explicit external-q loop -> compute ~1/P, at the cost of making the internal-q G available
on every rank (here replicated; in production an all-gather over the q-communicator: the
"data exchange between CPUs" of Guo-Bescond-Zhang 2020). This is the scalable axis for the
periodic anharmonic problem -- contrast the FLAT energy axis (F22), which replicates the
convolution. This is the algorithm the production q_comm port implements.

The kernel under test is the validated se_q._se_worker_iq (the same code path
transmission_q's SCBA uses). We synthesize modest Phi/G tensors so the benchmark isolates
the external-q parallel division (the physics size is irrelevant to scaling).

Run:  for P in 1 2 4 8; do mpirun --bind-to none -np $P python phph_q_dist_scaling.py; done
Env:  QPHPH_NKX (q-mesh side, default 6), QPHPH_ND (n_dof, default 24), QPHPH_NE (default 61).
"""
import os
import sys
import time
import warnings
from pathlib import Path

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

import numpy as np
from mpi4py.MPI import COMM_WORLD as world
from phonon.phonon_inputs.separable import build_q_diff_map
from phonon.solver import se_q
from phonon_inputs.constants import HBAR_SI

NKX = int(os.environ.get("QPHPH_NKX", "6")); NKY = NKX
ND = int(os.environ.get("QPHPH_ND", "24"))
NE = int(os.environ.get("QPHPH_NE", "61"))
nk = NKX * NKY
n_fft = 2 * NE - 1
mid = NE // 2
dw = 0.3
prefactor = 0.5j * HBAR_SI * dw / (2 * np.pi) / nk
q_diff_map = build_q_diff_map(NKX, NKY)

# synthetic, identical on every rank (a fixed seed) -> replicated internal-q G.
rng = np.random.default_rng(1234)
Phi_all = (rng.standard_normal((nk, nk, ND, ND, ND))
           + 1j * rng.standard_normal((nk, nk, ND, ND, ND))) * 1e-2
# energy-domain FFT of the (zero-padded) per-q G, as the kernel consumes it
def _gfft():
    g = (rng.standard_normal((nk, NE, ND, ND)) + 1j * rng.standard_normal((nk, NE, ND, ND)))
    g[:, mid] = 0.0
    pad = np.zeros((nk, n_fft, ND, ND), dtype=complex); pad[:, :NE] = g
    return np.fft.fft(pad, axis=1)
GL_fft = _gfft(); GG_fft = _gfft()
qp_batch = max(1, min(nk, (32 * 1024 * 1024) // max(n_fft * ND * ND * 16, 1)))

# distribute EXTERNAL q over ranks (round-robin); internal q' replicated (the all-gather)
my_q = list(range(world.rank, nk, world.size))
common = (GL_fft, GG_fft, Phi_all, q_diff_map, NE, ND, nk, n_fft,
          mid, mid + NE, prefactor, qp_batch)

if my_q:
    se_q._se_worker_iq((my_q, *common))           # warm-up / page-in
times = []
for _ in range(2):
    world.Barrier(); t0 = time.time()
    if my_q:
        _iq, sl, sg = se_q._se_worker_iq((my_q, *common))
    world.Barrier(); times.append(time.time() - t0)
dt = float(min(times))
nrm = float(np.linalg.norm(sl)) if my_q else 0.0
tot = world.allreduce(nrm ** 2) ** 0.5
if world.rank == 0:
    per_rank = -(-nk // world.size)
    print(f"RANKS={world.size:2d}  q-mesh={NKX}x{NKY}={nk}  ext-q/rank<={per_rank}  "
          f"n_dof={ND} NE={NE}  ext-q self-energy wall={dt:.3f}s  |Sigma|={tot:.4e}",
          flush=True)
