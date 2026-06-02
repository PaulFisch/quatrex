"""Production q-communicator: distributed coupled-q 3-phonon self-energy == serial oracle.

Exercises the new third communicator axis `comm.q` (qttools.comm.QuatrexCommunicator,
q_comm_size) end-to-end on the q-resolved phonon-phonon self-energy:

  1. Each rank OWNS a contiguous slice of the transverse q-mesh and holds only its own
     internal-q Green's functions G(q', w).
  2. `comm.q.all_gather_v(axis=0)` reconstructs the full internal-q G on every rank --
     this is the "data exchange between CPUs" of Guo-Bescond-Zhang 2020, required because
     momentum conservation q_ext = q' + (q_ext - q') couples every external q to ALL internal q'.
  3. Each rank computes the self-energy ONLY for its external-q slice (se_q._se_worker_iq).
  4. `comm.q.all_gather_v` collects the external-q self-energy.

Asserts the gathered, distributed result equals the serial
`compute_phph_self_energy_q_dense` (the validated oracle) to ~1e-10, and that the
all-gathered internal G reproduces the full G. With q_comm_size=1 (np=1) it is a plain
serial check; with q_comm_size=np it is the distributed path.

Run:  for P in 1 2 4; do mpirun --bind-to none -np $P python phph_q_comm_validate.py; done
"""
import sys
import warnings
from pathlib import Path

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
for p in (_REPO / "src", _REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

import numpy as np
from mpi4py.MPI import COMM_WORLD as world
from qttools.comm import comm
from phonon.finite_analysis.loader import load_system
from phonon.phonon_inputs.separable import (
    build_supercell_mapping, build_realspace_fc3_matrices,
    build_gathering_matrix, build_q_diff_map,
)
from phonon.solver.dense import load_fc3_raw
from phonon.solver import se_q
from phonon.solver.se_q import compute_phph_self_energy_q_dense
from phonon_inputs.constants import HBAR_SI

# --- common inputs (identical on every rank) ---
b = load_system(_REPO / "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml",
                validate=False, transport_axis=2)
ph = b.phonon
prim_indices, cell_frac, slab_indices, ref_sc = build_supercell_mapping(ph, "z")
nat = len(ph.primitive.masses); nd = 3 * nat
M_stacked = build_realspace_fc3_matrices(
    load_fc3_raw(str(Path(b.meta["fc3_path"]).expanduser().resolve())),
    nat, ph.supercell.masses, ref_sc)
nkx = nky = 2
qpts = [(i / nkx, j / nky) for i in range(nkx) for j in range(nky)]
T_all = [build_gathering_matrix(prim_indices, cell_frac, q, nat, "z") for q in qpts]
q_diff_map = build_q_diff_map(nkx, nky)
nk = nkx * nky
NE = 31
freqs = np.linspace(0.01, 18.0, NE); dw = float(freqs[1] - freqs[0])

rng = np.random.default_rng(7)
Gl = rng.standard_normal((nk, NE, nd, nd)) + 1j * rng.standard_normal((nk, NE, nd, nd))
Gg = rng.standard_normal((nk, NE, nd, nd)) + 1j * rng.standard_normal((nk, NE, nd, nd))

# --- serial oracle (full mesh) ---
sl_ref, sg_ref = compute_phph_self_energy_q_dense(
    Gl, Gg, M_stacked, T_all, q_diff_map, nat, nk, freqs, dw, n_workers=1)

# --- configure the production q-communicator: all ranks on the q axis ---
cfg = {k: "device_mpi" for k in ("all_to_all", "all_gather", "all_reduce", "bcast")}
comm.configure(block_comm_size=1, block_comm_config=cfg, stack_comm_config=cfg,
               override=True, q_comm_size=world.size, q_comm_config=cfg)
P = comm.q.size; r = comm.q.rank
assert P == world.size, (P, world.size)

# contiguous ownership of the q-mesh across comm.q ranks
counts = [nk // P + (1 if i < nk % P else 0) for i in range(P)]
starts = [sum(counts[:i]) for i in range(P)]
lo, n_loc = starts[r], counts[r]
my_q = list(range(lo, lo + n_loc))

# 1+2: each rank holds its internal-q G slice; all-gather reconstructs the full G
Gl_full = comm.q.all_gather_v(Gl[lo:lo + n_loc].copy(), axis=0)
Gg_full = comm.q.all_gather_v(Gg[lo:lo + n_loc].copy(), axis=0)
gather_err = max(np.max(np.abs(Gl_full - Gl)), np.max(np.abs(Gg_full - Gg)))

# 3: build kernel setup from the gathered G and compute ONLY the local external-q SE
n_fft = 2 * NE - 1; mid = NE // 2
prefactor = 0.5j * HBAR_SI * dw / (2 * np.pi) / nk
Gp = np.zeros((nk, n_fft, nd, nd), dtype=complex)
g = Gl_full.copy(); g[:, mid] = 0; Gp[:, :NE] = g; GL_fft = np.fft.fft(Gp, axis=1)
Gp[:] = 0; g = Gg_full.copy(); g[:, mid] = 0; Gp[:, :NE] = g; GG_fft = np.fft.fft(Gp, axis=1)
dim_t = M_stacked.shape[1]; Mb = M_stacked.reshape(nd, dim_t, dim_t)
Ta = np.array(T_all); TM = np.einsum('qci,aij->qacj', Ta, Mb)
Phi_all = np.einsum('qacj,rjd->qracd', TM, Ta.conj().transpose(0, 2, 1).copy())
qp_batch = max(1, min(nk, (16 * 1024 * 1024) // max(n_fft * nd**3 * 16, 1)))
common = (GL_fft, GG_fft, Phi_all, q_diff_map, NE, nd, nk, n_fft,
          mid, mid + NE, prefactor, qp_batch)
if n_loc:
    _iq, sl_loc, sg_loc = se_q._se_worker_iq((my_q, *common))
else:
    sl_loc = np.zeros((0, NE, nd, nd), dtype=complex); sg_loc = sl_loc.copy()

# 4: gather the external-q self-energy back over comm.q
sl_dist = comm.q.all_gather_v(sl_loc, axis=0)
sg_dist = comm.q.all_gather_v(sg_loc, axis=0)

se_err = max(np.max(np.abs(sl_dist - sl_ref)) / (np.max(np.abs(sl_ref)) + 1e-30),
             np.max(np.abs(sg_dist - sg_ref)) / (np.max(np.abs(sg_ref)) + 1e-30))
if world.rank == 0:
    ok = gather_err < 1e-12 and se_err < 1e-9
    print(f"q_comm: P={P} (block=1,stack={comm.stack.size},q={P})  q-mesh={nkx}x{nky}={nk}  "
          f"own/rank={counts}  all-gather-G err={gather_err:.1e}  "
          f"distributed-SE vs oracle rel-err={se_err:.2e}  -> {'PASS' if ok else 'FAIL'}",
          flush=True)
