"""Validate the continuum residue derivation of the near-DC channel
(phonon/docs/ir_residue_derivation.md) against the dense film bed.

(phonon/docs/ir_residue_derivation.md) against the dense film bed.
Run:  QTX_ARRAY_MODULE=numpy OMP_NUM_THREADS=1         python phonon/studies/_ir_residue_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.studies._ir_killtest import (  # noqa: E402
    ND, NSLAB, NU, OUT, contact_sigmas, device_g, load_gamma_blocks,
    translations)
from quatrex.phonon.ir_subtraction import bose_pole_coeff  # noqa: E402


def tbasis(x, t54):
    """Project a (54,54) matrix onto the 3-dim translation basis."""
    return t54 @ x @ t54.T


def main():
    d00, d01, d10 = load_gamma_blocks()
    t18 = translations()
    rec = np.load(NU / "run_ballistic.npz")
    eta = float(rec["eta"])
    t_l, t_r = float(rec["t_left"]), float(rec["t_right"])

    n = NSLAB * ND
    t54 = np.zeros((3, n))
    for b in range(3):
        t54[b] = np.tile(t18[b], NSLAB)
    t54 /= np.linalg.norm(t54, axis=1, keepdims=True)

    # ---- (i) static screened stiffness annihilates translations ----
    w_small = 1e-4
    sig_l, sig_r = contact_sigmas(np.array([w_small]), eta, d00, d01, d10)
    dev = np.zeros((n, n), complex)
    for i in range(NSLAB):
        dev[i * ND:(i + 1) * ND, i * ND:(i + 1) * ND] = d00
        if i + 1 < NSLAB:
            dev[i * ND:(i + 1) * ND, (i + 1) * ND:(i + 2) * ND] = d01
            dev[(i + 1) * ND:(i + 2) * ND, i * ND:(i + 1) * ND] = d10
    K = dev.copy()
    K[:ND, :ND] += 0.5 * (sig_l[0] + sig_l[0].conj().T)
    K[-ND:, -ND:] += 0.5 * (sig_r[0] + sig_r[0].conj().T)
    kt = max(np.linalg.norm(K @ t54[b]) for b in range(3))
    print(f"(i)  static screened stiffness: max_b |K t| / |K| = "
          f"{kt / np.linalg.norm(K):.3e}  (fc2's own ASR quality ~5e-6)")

    # ---- (ii) linear opening of the lead broadening ----
    # V(w) = Gamma(w)/2w has O(w^2) physical corrections but also NEVP
    # noise at very small w; fit V(w) = V0 + a w^2 on a small ladder and
    # use the extrapolated V0.
    print("\n(ii) Gamma_alpha(w) / (2w)  ->  V_alpha  (translation-basis "
          "eigenvalues, THz):")
    ladder = np.array([2e-3, 4e-3, 8e-3, 1.6e-2])
    V_T = {"L": [], "R": []}
    for w in ladder:
        sig_l, sig_r = contact_sigmas(np.array([w]), eta, d00, d01, d10)
        for name, sig, sl in (("L", sig_l[0], slice(0, ND)),
                              ("R", sig_r[0], slice(n - ND, n))):
            gam = np.zeros((n, n), complex)
            gam[sl, sl] = 1j * (sig - sig.conj().T)
            V_T[name].append(tbasis(gam.real, t54) / (2 * w))
    V0 = {}
    for name in ("L", "R"):
        stack = np.array(V_T[name])
        coef = np.polynomial.polynomial.polyfit(
            ladder ** 2, stack.reshape(len(ladder), -1), 1)
        V0[name] = 0.5 * (coef[0].reshape(3, 3) + coef[0].reshape(3, 3).T)
        ev = np.linalg.eigvalsh(V0[name])
        resid = np.linalg.norm(stack[0] - V0[name]) / np.linalg.norm(V0[name])
        print(f"     V_{name},T eigs (w->0 extrapolated): "
              f"{np.round(ev, 5).tolist()}  (finite-w resid at "
              f"w={ladder[0]:g}: {resid:.1e})")

    # ---- (iii) the C2 residue: predicted vs measured ----
    vl, vr = V0["L"], V0["R"]
    v_tot = vl + vr
    c_l, c_r = bose_pole_coeff(t_l), bose_pole_coeff(t_r)
    B = 2 * c_l * vl + 2 * c_r * vr
    vinv = np.linalg.inv(v_tot)
    c2_pred = vinv @ B @ vinv

    w0 = 1e-3
    gr0, gl0, _ = device_g(w0, eta, d00, d01, d10, t_l, t_r)
    c2_meas = tbasis((w0 * w0 * (-1j * gl0)).real, t54)
    grr = tbasis((w0 * gr0), t54)

    print(f"\n(iii) C2 residue (translation basis, THz units), "
          f"c_L={c_l:.4f} c_R={c_r:.4f}:")
    print("     predicted V^-1 (2cL VL + 2cR VR) V^-1 eigs: "
          f"{np.round(np.linalg.eigvalsh(0.5 * (c2_pred + c2_pred.T)), 3).tolist()}")
    print("     measured  w^2(-iG^<)|_T eigs:              "
          f"{np.round(np.linalg.eigvalsh(0.5 * (c2_meas + c2_meas.T)), 3).tolist()}")
    rel = np.linalg.norm(c2_pred - c2_meas) / np.linalg.norm(c2_meas)
    print(f"     matrix relative error |pred - meas|/|meas| = {rel:.3e}")

    # retarded residue: G^R|_T -> -i V_T^-1 / w  (Gamma = 2wV >= 0 forces
    # Im Sigma^R = -wV, hence [w^2 + i w V_T]^{-1} ~ -i V^{-1}/w).
    print("\n     w G^R|_T vs -i V_T^-1:")
    ivinv = -1j * vinv
    rel_r = np.linalg.norm(grr - ivinv) / np.linalg.norm(ivinv)
    print(f"     |w G^R|_T - (-i V^-1)| / |V^-1| = {rel_r:.3e}")
    print(f"     eig(V_T) = {np.round(np.linalg.eigvalsh(v_tot), 4).tolist()} THz")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "residue_check.json").write_text(json.dumps({
        "K_t_rel": kt / np.linalg.norm(K),
        "V_T_eigs": np.linalg.eigvalsh(v_tot).tolist(),
        "c2_pred_eigs": np.linalg.eigvalsh(0.5 * (c2_pred + c2_pred.T)).tolist(),
        "c2_meas_eigs": np.linalg.eigvalsh(0.5 * (c2_meas + c2_meas.T)).tolist(),
        "c2_rel_err": rel, "gr_residue_rel_err": rel_r}, indent=1))
    print(f"\nwrote {OUT / 'residue_check.json'}")


if __name__ == "__main__":
    main()
