"""Post-hoc local observables from a production run NPZ.

Consumes the engine snapshot keys written by ``phonon/studies/engine/run.py``:
``gr_diag_imag`` / ``gl_diag_imag`` (final-iterate per-DOF -Im G^R and
Im G^<), ``bubble_balance_spectrum`` and ``current_spectrum``. Produces the
thesis observables that need no solver rerun:

  * per-DOF / per-slab local DOS (eq:dos),
  * non-equilibrium occupation n_i(omega) (eq:neq_occupation),
  * local effective temperature T_eff(i) (eq:Teff_local),
  * the per-omega energy sum rule D(omega) (eq:sumrule).

NOTE on signs: the production solver stores occupation-positive
G^{<,>} (-i G^< <= 0 is the textbook convention; here Im G^< >= 0 for
omega > 0), so ``gl_diag_imag`` IS the occupation numerator and
``gr_diag_imag`` the spectral weight -- no extra flip.

Usage:
    python phonon/postproc/local_observables.py <run.npz> [--n-dof N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PHONON = Path(__file__).resolve().parents[1]
if str(_PHONON) not in sys.path:
    sys.path.insert(0, str(_PHONON))

from solver.observables import bose, effective_temperature  # noqa: E402


def load(npz_path):
    d = np.load(npz_path)
    if "gr_diag_imag" not in d.files:
        raise SystemExit(
            f"{npz_path} lacks gr_diag_imag (rerun with QX_SAVE_DIAG_G=1)")
    freqs = np.abs(np.asarray(d["energies"]).real)
    gr = np.asarray(d["gr_diag_imag"])   # (ne, *nk, N_D): -Im G^R
    gl = np.asarray(d["gl_diag_imag"])   # (ne, *nk, N_D): +Im G^< (occ.-pos.)
    # Average any transverse-q axes.
    while gr.ndim > 2:
        gr = gr.mean(axis=1)
        gl = gl.mean(axis=1)
    return d, freqs, gr, gl


def compute(npz_path, n_dof=None):
    d, freqs, gr, gl = load(npz_path)
    out = {"freqs_thz": freqs}
    # Local DOS per DOF: rho_mu(omega) = (2 omega / pi) * (-Im G^R_mumu).
    out["ldos"] = (2.0 * freqs[:, None] / np.pi) * gr
    # Occupation: n = Im G^< / (2 * -Im G^R)  (A = -2 Im G^R = i(G^>-G^<)).
    A = 2.0 * gr
    out["occupation"] = gl / np.where(np.abs(A) > 1e-30, A, 1e-30)
    if n_dof:
        n_slabs = gr.shape[1] // int(n_dof)
        iGl = gl.reshape(gl.shape[0], n_slabs, int(n_dof)).sum(axis=-1)
        A_s = A.reshape(A.shape[0], n_slabs, int(n_dof)).sum(axis=-1)
        out["T_eff"] = effective_temperature(freqs, iGl, A_s)
    if "bubble_balance_spectrum" in d.files and "current_spectrum" in d.files:
        # D(omega) = J_L - J_R - (P_out - P_in), hbar*omega-weighted like
        # iter_heat: current_spectrum carries the number current.
        spec = np.asarray(d["current_spectrum"])
        while spec.ndim > 2:
            spec = spec.sum(axis=1)
        p_in, p_out = np.asarray(d["bubble_balance_spectrum"])
        out["D_omega"] = (freqs * spec[:, 0] - freqs * spec[:, -1]
                          - (p_out.real - p_in.real))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--n-dof", type=int, default=None,
                    help="DOF per slab for slab-resolved T_eff")
    ap.add_argument("--out", default=None, help="save results NPZ here")
    args = ap.parse_args()
    res = compute(args.npz, n_dof=args.n_dof)
    print(f"{args.npz}:")
    print(f"  ldos {res['ldos'].shape}, occupation {res['occupation'].shape}")
    if "T_eff" in res:
        print(f"  T_eff per slab: {np.round(res['T_eff'], 2)} K")
    if "D_omega" in res:
        print(f"  max |D(omega)|: {np.max(np.abs(res['D_omega'])):.3e}")
    if args.out:
        np.savez(args.out, **res)
        print(f"  saved {args.out}")


if __name__ == "__main__":
    main()
