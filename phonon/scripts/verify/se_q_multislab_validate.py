"""Validate the full off-diagonal q-resolved multi-slab self-energy.

At a 1x1 transverse mesh the new coupled-q multi-slab kernel
(se_q.compute_phph_self_energy_q_dense_multi_slab) must reproduce the Gamma-only
off-diagonal template (se_finite.compute_phph_self_energy_finite_multi_slab) block
for block, on identical G. This anchors the off-diagonal slab structure + q-fold.
"""
import sys
import warnings
from pathlib import Path

_W = Path("/usr/scratch/mont-fort11/pfischill/quatrex/phonon")
for p in (_W.parent, _W):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

import numpy as np
import h5py
from phonon.phonon_inputs.separable import (
    build_supercell_mapping, build_realspace_fc3_matrices)
from phonon.solver.fc3_device import build_device_fc3_blocks
from phonon.solver.se_finite import compute_phph_self_energy_finite_multi_slab
from phonon.solver.se_q import compute_phph_self_energy_q_dense_multi_slab
from phonon.scripts.verify.si_film_kappa import load_bulk_si

ph, fc3_path = load_bulk_si()
nat = len(ph.primitive.masses); nd = 3 * nat
tdir = "x"
prim_indices, cell_frac, slab_indices, ref_sc = build_supercell_mapping(ph, tdir)
with h5py.File(fc3_path, "r") as f:
    fc3 = f["fc3"][:]
M_stacked = build_realspace_fc3_matrices(fc3, nat, ph.supercell.masses, ref_sc)

n_slabs = 3
NE = 31
freqs = np.linspace(-16.0, 16.0, NE); freqs -= freqs[NE // 2]
dw = float(freqs[1] - freqs[0])
DC = "interpolate"

# synthetic device G blocks (K,K') within nearest-slab range
rng = np.random.default_rng(3)
gl_blocks, gg_blocks = {}, {}
gl_q, gg_q = {}, {}
for K in range(n_slabs):
    for Kp in range(n_slabs):
        if abs(K - Kp) > 1:
            continue
        a = rng.standard_normal((NE, nd, nd)) + 1j * rng.standard_normal((NE, nd, nd))
        b = rng.standard_normal((NE, nd, nd)) + 1j * rng.standard_normal((NE, nd, nd))
        gl_blocks[(K, Kp)] = a; gg_blocks[(K, Kp)] = b
        gl_q[(K, Kp)] = a[None]; gg_q[(K, Kp)] = b[None]   # n_kpts=1

# Gamma-only reference
phi_dev = build_device_fc3_blocks(M_stacked, prim_indices, slab_indices, nat, n_slabs)
sl_ref, sg_ref = compute_phph_self_energy_finite_multi_slab(
    gl_blocks, gg_blocks, phi_dev, n_slabs, freqs, dw, dc_handling=DC)

# new q-multi-slab at 1x1
sl_q, sg_q = compute_phph_self_energy_q_dense_multi_slab(
    gl_q, gg_q, M_stacked, prim_indices, cell_frac, slab_indices,
    nat, n_slabs, 1, [(0.0, 0.0)], np.array([[0]]), freqs, dw, tdir, dc_handling=DC)

keys = sorted(set(sl_ref) | set(sl_q))
maxerr = 0.0
n_off = 0
for (I, J) in keys:
    rl = sl_ref.get((I, J)); ql = sl_q.get((I, J))
    if rl is None or ql is None:
        print(f"  block {(I,J)} present in only one: ref={rl is not None} q={ql is not None}")
        continue
    e = np.max(np.abs(ql[0] - rl)) / (np.max(np.abs(rl)) + 1e-30)
    maxerr = max(maxerr, e)
    if I != J:
        n_off += 1
print(f"n_slabs={n_slabs}: {len(keys)} sigma blocks ({n_off} off-diagonal), "
      f"max rel err (q-multislab @1x1 vs Gamma-multislab) = {maxerr:.2e}")
print("[PASS] full off-diagonal q self-energy reduces to Gamma template"
      if maxerr < 1e-9 and n_off > 0 else "[CHECK] mismatch or no off-diagonal blocks")
