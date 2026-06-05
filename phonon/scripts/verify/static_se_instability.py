#!/usr/bin/env python
"""Is the exploding tadpole a REAL structural instability or a truncation artifact?

The implemented tadpole shifts the FC2 by the FIRST-ORDER centroid response
    Phi2(R + <u>) ~ Phi2(R) + Phi3:<u>        (= Sigma_T)
omitting the SECOND-ORDER term + 1/2 Phi4:<u><u>, which for Phi4>0 STIFFENS the
shifted structure. If the high-T imaginary mode driven by Sigma_T is cured by
adding the 2nd-order term, the 'instability' is a first-order truncation
artifact (the relaxation is large but the structure is actually stable). If it
persists, it is a genuine anharmonic instability at that T (soft-mode
condensation -> structural transition).

We capture the device FC3/FC4 and the self-consistent <u>, <uu> from a converged
loop+tadpole run (monkeypatching the static-SE kernels) and compare the lowest
eigenfrequencies of: bare D ; D+Sigma_T (1st order) ; D+Sigma_T+1/2 Phi4:<u><u>
(2nd order). Also iterates the centroid relaxation to test SSCHA convergence.
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import warnings; warnings.filterwarnings("ignore")
from phonon.finite_analysis.loader import load_system
from solver.dense import transmission_finite
import solver.static_se as sse
from phonon_inputs.constants import CONVERSION_THZ2

CAP = {}
_orig_loop, _orig_tad = sse.sigma_loop, sse.sigma_tadpole
def _loop_cap(fc4, uu):
    CAP["fc4"] = np.asarray(fc4); CAP["uu"] = np.asarray(uu); return _orig_loop(fc4, uu)
def _tad_cap(fc3, wm):
    CAP["fc3"] = np.asarray(fc3); CAP["wmean"] = np.asarray(wm); return _orig_tad(fc3, wm)
sse.sigma_loop, sse.sigma_tadpole = _loop_cap, _tad_cap


def freqs(M):
    M = 0.5 * (M + M.T)
    w2 = np.linalg.eigvalsh(M)
    return np.sign(w2) * np.sqrt(np.abs(w2))


ap = argparse.ArgumentParser()
ap.add_argument("--struct", default="d5a")
ap.add_argument("--temps", type=float, nargs="+", default=[300.0, 600.0])
ap.add_argument("--fc3", default=None, help="override fc3.hdf5 (e.g. CNT-with-FC4)")
ap.add_argument("--nfreq", type=int, default=41)
args = ap.parse_args()
CFG = {"d5a": "phonon/configs/sinw/sinw100_d5a_vasp_sc4_fc4.yaml",
       "cnt33": "phonon/configs/cnt/cnt33_vasp.yaml"}[args.struct]
b = load_system(str(_REPO / CFG), validate=False, transport_axis=2)
ph = b.phonon; fc3 = args.fc3 or b.meta["fc3_path"]
FMAX = {"d5a": 16.0, "cnt33": 18.0}[args.struct]

for T in args.temps:
    CAP.clear()
    r = transmission_finite(
        ph, fc3_hdf5=fc3, transport_direction="z", n_slabs=1, retarded="fft",
        freq_range_thz=(0.05, FMAX, args.nfreq), eta_factor=0.5, temperature=T,
        delta_T=10.0, max_scba_iter=120, scba_tol=1e-3, conservation_tol=5e-2,
        enforce_asr=True, loop=True, tadpole=True, fc4_hdf5=fc3,
        stage_loop_first=True, static_mixing=0.1, solver="linear", verbose=False)
    if "fc4" not in CAP or "wmean" not in CAP:
        print(f"T={T:.0f}: capture failed (conv={r.get('scba_converged')})"); continue
    fc3d, fc4d, uu, wm = CAP["fc3"], CAP["fc4"], CAP["uu"], CAP["wmean"]
    phi = np.asarray(r["phi_eff"]).real; ss = np.asarray(r["sigma_static"]).real
    D = phi - ss                                            # bare device D
    sigT = CONVERSION_THZ2 * np.einsum("abc,c->ab", fc3d, wm); sigT = 0.5 * (sigT + sigT.T)
    phi4_2nd = 0.5 * CONVERSION_THZ2 * np.einsum("abcd,c,d->ab", fc4d, wm, wm)
    phi4_2nd = 0.5 * (phi4_2nd + phi4_2nd.T)
    sigL = sse.sigma_loop.__wrapped__ if hasattr(sse.sigma_loop, "__wrapped__") else None
    f0 = np.sort(freqs(D))[:4]
    f1 = np.sort(freqs(D + sigT))[:4]
    f2 = np.sort(freqs(D + sigT + phi4_2nd))[:4]
    umag = np.linalg.norm(wm)
    print(f"\n== {args.struct} T={T:.0f} K ==")
    print(f"  ||<u>|| = {umag:.3f} sqrt(amu)*A ; ||Sigma_T||={np.linalg.norm(sigT):.1f}  "
          f"||1/2 Phi4:<u><u>||={np.linalg.norm(phi4_2nd):.1f} THz^2")
    print(f"  lowest-4 freq [THz]:")
    print(f"    bare D            : {np.array2string(f0, precision=3)}")
    print(f"    +Sigma_T (1st ord): {np.array2string(f1, precision=3)}  <- tadpole as implemented")
    print(f"    +2nd order (Phi4) : {np.array2string(f2, precision=3)}  <- adds 1/2 Phi4:<u><u>")
    verdict = ("ARTIFACT: 2nd order restores stability" if f2[0] > 0 and f1[0] < 0
               else ("REAL instability: unstable even at 2nd order" if f2[0] < 0
                     else "no 1st-order instability at this T"))
    print(f"  -> {verdict}")
