#!/usr/bin/env python
"""Diagnose why ||Sigma_T|| (cubic tadpole) is large for symmetric diamond Si.

Wraps build_static_self_energy_hook (capturing the exact device FC3/FC4 the
solver builds) and, on the first hook call, compares the tadpole source from the
NON-equilibrium <uu> (device G^<) against the ANALYTIC equilibrium <uu> (harmonic
mode sum, symmetric by construction). Isolates:
  * is the device FC3 tensor symmetric (-> s_eq ~ 0 expected)?
  * is <uu> from G^< asymmetric (non-eq) vs the equilibrium mode sum?
  * relative magnitudes: bubble vs loop vs tadpole.
"""
from __future__ import annotations
import sys, numpy as np
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import warnings; warnings.filterwarnings("ignore")

from solver import dense, static_se
sys.path.insert(0, str(_REPO / "phonon" / "scripts" / "verify"))
from bulk_si_scp import load_hiphive_phonon

REAP = _REPO / "phonon/configs/si_primitive/fc3_hiphive_si_fc4_vasp"
T = 300.0

phonon, fc_path, meta, has_fc4 = load_hiphive_phonon(REAP)
print("has_fc4:", has_fc4)

_orig = dense.build_static_self_energy_hook
_done = {"n": 0}

def patched(**kw):
    fc3 = kw.get("fc3_dev_mw"); fc4 = kw.get("fc4_dev_mw")
    proj = kw.get("optical_projector"); dw = kw["dw_thz"]
    hook = _orig(**kw)
    def wrapped(g_less, sig_cur, h_d):
        if _done["n"] == 0 and fc3 is not None:
            _done["n"] = 1
            uu_neq = static_se.equal_time_uu(g_less[0], dw)
            D = np.real(h_d[0])
            uu_eq = static_se.equilibrium_uu_modesum(D, T)
            s_neq = static_se.tadpole_source(fc3, uu_neq)
            s_eq = static_se.tadpole_source(fc3, uu_eq)
            print("\n========== TADPOLE DIAGNOSTIC ==========")
            print(f"  ||uu_neq||={np.linalg.norm(uu_neq):.4e}  "
                  f"||uu_eq||={np.linalg.norm(uu_eq):.4e}  "
                  f"||uu_neq-uu_eq||={np.linalg.norm(uu_neq-uu_eq):.4e}")
            print(f"  diag uu_neq (per DOF, =<w^2> amu*A^2) = "
                  f"{np.array2string(np.diag(uu_neq), precision=3)}")
            print(f"  diag uu_eq  (per DOF)                = "
                  f"{np.array2string(np.diag(uu_eq), precision=3)}")
            print("    (DOF order = atom0[x,y,z], atom1[x,y,z]; transport=x. "
                  "x != y=z signals open-device anisotropy.)")
            print(f"    physical Si Debye-Waller <w^2> ~ m*<u^2> ~ 28*0.0057 ~ 0.16 amu*A^2/dir")
            print(f"  ||s_neq||={np.linalg.norm(s_neq):.4e}  "
                  f"||s_eq (should be ~0)||={np.linalg.norm(s_eq):.4e}")
            print(f"  s_eq vector = {np.array2string(s_eq, precision=3)}")
            if proj is not None:
                print(f"  ||Q s_eq (optical part)||={np.linalg.norm(proj@s_eq):.4e}  "
                      f"||(I-Q) s_eq (acoustic)||={np.linalg.norm(s_eq-proj@s_eq):.4e}")
            # device FC3 symmetry checks
            fa = np.linalg.norm(fc3 - np.transpose(fc3, (1, 0, 2)))
            fa2 = np.linalg.norm(fc3 - np.transpose(fc3, (0, 2, 1)))
            print(f"  device FC3 perm-asym ||Phi3-Phi3^(ba c)||={fa:.4e}  "
                  f"||Phi3-Phi3^(a cb)||={fa2:.4e}  (||Phi3||={np.linalg.norm(fc3):.4e})")
            # tadpole + loop from equilibrium uu
            w_eq = static_se.mean_displacement(fc3, uu_eq, D, optical_projector=proj)
            sigT_eq = static_se.sigma_tadpole(fc3, w_eq)
            print(f"  ||Sigma_T(eq uu)||={np.linalg.norm(sigT_eq):.4e}  "
                  f"||<w>_eq||={np.linalg.norm(w_eq):.4e}")
            if fc4 is not None:
                sigL_eq = static_se.sigma_loop(fc4, uu_eq)
                print(f"  ||Sigma_L(eq uu)||={np.linalg.norm(sigL_eq):.4e}")
            print(f"  D(Gamma) eigvals [THz^2] = "
                  f"{np.array2string(np.sort(np.linalg.eigvalsh(D)), precision=2)}")
            print("========================================\n")
        return hook(g_less, sig_cur, h_d)
    return wrapped

dense.build_static_self_energy_hook = patched

res = dense.transmission_finite(
    phonon, fc3_hdf5=fc_path, loop=True, tadpole=True,
    fc4_hdf5=fc_path, stage_loop_first=False,
    transport_direction="x", freq_range_thz=(0.01, 18.0, 101),
    eta_factor=1.0, temperature=T, delta_T=10.0,
    max_scba_iter=2, verbose=True)
print("bubble max|Sigma^R| (THz^2) reported above; "
      f"sigma_static_norm={np.linalg.norm(res.get('sigma_static')):.4e}"
      if res.get("sigma_static") is not None else "no sigma_static")
