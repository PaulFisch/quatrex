"""Grid-convergence demonstration of the surgical rank-3 subtraction
(phonon/docs/ir_residue_derivation.md, Sec. 3) on the dense film bed.

(phonon/docs/ir_residue_derivation.md, Sec. 3) on the dense film bed.
Run:  QTX_ARRAY_MODULE=numpy OMP_NUM_THREADS=4         python phonon/studies/_ir_subtraction_demo.py
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
    ND, NSLAB, NU, OUT, contact_sigmas, device_g_batch, load_gamma_blocks,
    load_vertex, ring, translations)
from quatrex.phonon.ir_subtraction import bose_pole_coeff  # noqa: E402

W_MAX = 12.0            # axis half-width (THz); same for every rung
DWS = [0.1, 0.05, 0.025, 0.0125]   # nested ladder
W_OUTS = [0.5, 1.0, 3.0]           # output bins (THz), on every rung


def cm_channel(d00, d01, d10, eta, t54):
    """(V_L,T, V_R,T) 3x3 in the translation basis, from the lead model."""
    ladder = np.array([2e-3, 4e-3, 8e-3, 1.6e-2])
    n = NSLAB * ND
    V = {"L": [], "R": []}
    for w in ladder:
        sig_l, sig_r = contact_sigmas(np.array([w]), eta, d00, d01, d10)
        for name, sig, sl in (("L", sig_l[0], slice(0, ND)),
                              ("R", sig_r[0], slice(n - ND, n))):
            gam = np.zeros((n, n), complex)
            gam[sl, sl] = 1j * (sig - sig.conj().T)
            V[name].append(t54 @ gam.real @ t54.T / (2 * w))
    V0 = {}
    for name in ("L", "R"):
        coef = np.polynomial.polynomial.polyfit(
            ladder ** 2, np.array(V[name]).reshape(len(ladder), -1), 1)
        V0[name] = 0.5 * (coef[0].reshape(3, 3) + coef[0].reshape(3, 3).T)
    return V0["L"], V0["R"]


def s_less(w, vl, vr, t54, t_l, t_r):
    """Exact CM-channel S^<(w) (production convention), 0 at w=0.

    S^R = [w^2 + i w V_T]^{-1} on the translation subspace; lesser
    drive with the full per-lead Bose factors. Fold- and KMS-exact;
    lesser tail ~ 1/w^4 beyond the damping scale (no mid-band
    over-subtraction, unlike the strict-Laurent form)."""
    if abs(w) < 1e-12:
        return np.zeros((t54.shape[1], t54.shape[1]), complex)
    from quatrex.phonon.ir_subtraction import bose
    v_t = vl + vr
    sr = np.linalg.inv((w * w) * np.eye(3) + 1j * w * v_t)
    drive = 2.0 * w * (bose(w, t_l) * vl + bose(w, t_r) * vr)
    s3 = 1j * sr @ drive @ sr.conj().T
    return t54.T @ s3 @ t54


def main():
    d00, d01, d10 = load_gamma_blocks()
    t18 = translations()
    n = NSLAB * ND
    t54 = np.zeros((3, n))
    for b in range(3):
        t54[b] = np.tile(t18[b], NSLAB)
    t54 /= np.linalg.norm(t54, axis=1, keepdims=True)

    rec = np.load(NU / "run_ballistic.npz")
    eta = float(rec["eta"])
    t_l, t_r = float(rec["t_left"]), float(rec["t_right"])
    phi = load_vertex()

    vl, vr = cm_channel(d00, d01, d10, eta, t54)
    print(f"channel: V_LT eigs {np.round(np.linalg.eigvalsh(vl), 4).tolist()}"
          f"  V_RT eigs {np.round(np.linalg.eigvalsh(vr), 4).tolist()}")

    # dense G on the finest positive grid once; coarser rungs subsample
    dw_min = DWS[-1]
    ws_pos = np.arange(1, int(round(W_MAX / dw_min)) + 1) * dw_min
    cache = OUT / f"dense_g_cache_{len(ws_pos)}_{W_MAX:g}.npz"
    if cache.exists():
        cc = np.load(cache)
        gl_pos, gg_pos = cc["gl"], cc["gg"]
        print(f"loaded dense G cache {cache.name}")
    else:
        print(f"solving dense G on {len(ws_pos)} positive bins "
              f"(dw={dw_min}, W={W_MAX}) ...")
        _, gl_pos, gg_pos = device_g_batch(ws_pos, eta, d00, d01, d10,
                                           t_l, t_r)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, gl=gl_pos, gg=gg_pos)

    results = {}
    for dw in DWS:
        step = int(round(dw / dw_min))
        sel = np.arange(step - 1, len(ws_pos), step)
        wpos = ws_pos[sel]
        npos = len(wpos)
        # full symmetric axis: index k in [-npos .. npos], w = k*dw
        # legs via the fold: G^<(-w) = G^>(w)^T
        gl_full = {}
        for i, w in enumerate(wpos):
            gl_full[i + 1] = gl_pos[sel[i]]
            gl_full[-(i + 1)] = gg_pos[sel[i]].T
        gl_full[0] = np.zeros((n, n), complex)

        def leg(k, sub):
            g = gl_full[k]
            if not sub or k == 0:
                return g
            return g - s_less(k * dw, vl, vr, t54, t_l, t_r)

        row = {}
        for w_out in W_OUTS:
            m = int(round(w_out / dw))
            for sub in (False, True):
                acc = np.zeros((n, n), complex)
                for k in range(-npos, npos + 1):
                    k2 = m - k
                    if -npos <= k2 <= npos:
                        acc += ring(phi, leg(k, sub), leg(k2, sub))
                row[("sub" if sub else "bare", w_out)] = \
                    np.linalg.norm(acc) * dw
        results[dw] = row
        print(f"dw={dw:7.4f}  " + "  ".join(
            f"w={w}: bare {row[('bare', w)]:.4e} sub {row[('sub', w)]:.4e}"
            for w in W_OUTS))

    print("\nconvergence (value at dw normalised to value at dw=0.1):")
    for w_out in W_OUTS:
        b = [results[dw][("bare", w_out)] / results[0.1][("bare", w_out)]
             for dw in DWS]
        s = [results[dw][("sub", w_out)] / results[0.1][("sub", w_out)]
             for dw in DWS]
        print(f"  w_out={w_out}: bare {np.round(b, 3).tolist()}  "
              f"sub {np.round(s, 3).tolist()}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subtraction_demo.json").write_text(json.dumps(
        {str(dw): {f"{k[0]}_{k[1]}": v for k, v in row.items()}
         for dw, row in results.items()}, indent=1))
    print(f"wrote {OUT / 'subtraction_demo.json'}")


if __name__ == "__main__":
    main()
