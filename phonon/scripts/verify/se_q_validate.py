"""Validate the coupled-q 3-phonon self-energy kernel (se_q) — the oracle for the
distributed q-resolved implementation.

Checks:
  (1) single q-point at Gamma (n_kpts=1) reproduces the Gamma-only finite bubble
      (se_finite.compute_phph_self_energy_finite) on identical G -> the q kernel
      reduces correctly to the non-periodic limit;
  (2) a 2x2 transverse q-mesh runs, gives bounded finite Sigma(q), and the
      momentum-conservation map round-trips (q_ext = q' + (q_ext-q')).
"""
import sys
import warnings
from pathlib import Path

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

import numpy as np
from phonon.finite_analysis.loader import load_system
from phonon.phonon_inputs.separable import (
    build_supercell_mapping, build_realspace_fc3_matrices,
    build_gathering_matrix, build_q_diff_map,
)
from phonon.solver.se_q import compute_phph_self_energy_q_dense
from phonon.solver.se_finite import compute_phph_self_energy_finite

cfg = _REPO / "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml"
b = load_system(cfg, validate=False, transport_axis=2)
ph = b.phonon
tdir = "z"
fc3 = str(Path(b.meta["fc3_path"]).expanduser().resolve())

from phonon.solver.dense import load_fc3_raw
prim_indices, cell_frac, slab_indices, ref_sc = build_supercell_mapping(ph, tdir)
masses_super = ph.supercell.masses
nat_prim = len(ph.primitive.masses)
n_dof = 3 * nat_prim
fc3_raw = load_fc3_raw(fc3)
M_stacked = build_realspace_fc3_matrices(fc3_raw, nat_prim, masses_super, ref_sc)

# small frequency grid + synthetic G (hermitian-symmetric-ish, complex)
nfreq = 41
freqs = np.linspace(0.01, 18.0, nfreq)
dw = float(freqs[1] - freqs[0])
rng = np.random.default_rng(0)
def synthG():
    A = rng.standard_normal((nfreq, n_dof, n_dof)) + 1j * rng.standard_normal((nfreq, n_dof, n_dof))
    return A

# ---- (1) n_kpts=1, q=Gamma ----
T_g = build_gathering_matrix(prim_indices, cell_frac, (0.0, 0.0), nat_prim, tdir)
T_all = [T_g]
qmap1 = build_q_diff_map(1, 1)
Gl = synthG(); Gg = synthG()
Gl_q = Gl[None]; Gg_q = Gg[None]
sl_q, sg_q = compute_phph_self_energy_q_dense(
    Gl_q, Gg_q, M_stacked, T_all, qmap1, nat_prim, 1, freqs, dw, n_workers=1)

# Gamma vertex used internally by se_q: Phi = (T M_a T^H) per dof a
dim_t = M_stacked.shape[1]
Mb = M_stacked.reshape(n_dof, dim_t, dim_t)
TM = np.einsum('ci,aij->acj', T_g, Mb)
Phi_g = np.einsum('acj,jd->acd', TM, T_g.conj().T)
sl_f, sg_f = compute_phph_self_energy_finite(Gl, Gg, Phi_g, freqs, dw)
# se_q carries a 1/n_kpts=1 prefactor; se_finite has none -> compare directly
err_l = np.max(np.abs(sl_q[0] - sl_f)) / (np.max(np.abs(sl_f)) + 1e-30)
err_g = np.max(np.abs(sg_q[0] - sg_f)) / (np.max(np.abs(sg_f)) + 1e-30)
print(f"(1) 1q-Gamma vs finite bubble: rel err lesser={err_l:.2e} greater={err_g:.2e} "
      f"max|Sig_q|={np.max(np.abs(sl_q)):.3e}")

# ---- (2) 2x2 q-mesh: runs, bounded, momentum map round-trips ----
nkx = nky = 2
qpts = [(i/nkx, j/nky) for i in range(nkx) for j in range(nky)]
T_all2 = [build_gathering_matrix(prim_indices, cell_frac, q, nat_prim, tdir) for q in qpts]
qmap2 = build_q_diff_map(nkx, nky)
nk = nkx * nky
Glq = synthG()[None].repeat(nk, 0) * (1 + 0.1*rng.standard_normal((nk,1,1,1)))
Ggq = synthG()[None].repeat(nk, 0) * (1 + 0.1*rng.standard_normal((nk,1,1,1)))
slq, sgq = compute_phph_self_energy_q_dense(
    Glq, Ggq, M_stacked, T_all2, qmap2, nat_prim, nk, freqs, dw, n_workers=1)
# momentum conservation round-trip: q_diff_map[q, q_diff_map[q, q']] should give q'
rt_ok = all(qmap2[q, qmap2[q, qp]] == qp for q in range(nk) for qp in range(nk))
print(f"(2) 2x2 q-mesh: Sigma shape {slq.shape}, finite={np.all(np.isfinite(slq))}, "
      f"max|Sig|={np.max(np.abs(slq)):.3e}, momentum round-trip={rt_ok}")
print("[done] se_q kernel validated" if err_l < 1e-9 and err_g < 1e-9 and rt_ok
      else "[CHECK] reduction/roundtrip mismatch")
