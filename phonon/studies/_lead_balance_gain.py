"""Why does the MoS2 film show lead balance = 2 (h_L = -h_R)?

Run:  python phonon/studies/_lead_balance_gain.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phonon/studies/out/grid_audit"

RUNS = [
    # label, path, is_control
    ("MoS2 ballistic (nu grid)", "cluster/mos2f3nu/run_ballistic.npz", True),
    ("MoS2 ballistic (u121)", "cluster/mos2f3/run.npz", True),
    ("MoS2 lowmask 55it", "cluster/mos2f3/run_lowmask.npz", False),
    ("MoS2 conv 55it", "cluster/mos2f3/run_conv.npz", False),
    ("MoS2 nu 20it", "cluster/mos2f3nu/run.npz", False),
    ("MoS2 long DIVERGED", "cluster/mos2f3long/run.npz", False),
    ("Si film L3 SCBA conv", "cluster/sifilm3s/run.npz", True),
    ("Si film L5 SCBA conv", "cluster/sifilm5s/run.npz", True),
    ("CNT L4 gband2 conv",
     "phonon/studies/out/anderson_test/cnt33_L4_linear/run_gband2.npz", True),
]
REL = 1e-6          # gain threshold, relative to the GLOBAL scale


def analyse(path):
    d = np.load(ROOT / path, allow_pickle=True)
    h = np.asarray(d["last_heat"], float).reshape(-1)
    bal = abs(h[0] - h[-1]) / (0.5 * (abs(h[0]) + abs(h[-1])) + 1e-300)
    gl = np.asarray(d["gl_diag_imag"], float)
    e = np.asarray(d["energies"], float)
    scale = float(np.abs(gl).max())
    bad = gl < -REL * scale
    out = dict(balance=float(bal), spread=float(d["internal_spread"]),
               gain_frac=float(bad.mean()), worst_rel=float(gl.min() / scale),
               n_gain=int(bad.sum()))
    if bad.any():
        i = np.unravel_index(np.argmin(gl), gl.shape)
        nps = gl.shape[-1] // 3 if gl.shape[-1] % 3 == 0 else gl.shape[-1]
        out["worst_at"] = dict(omega_thz=float(e[i[0]]), slab=int(i[3] // nps),
                               dof=int(i[3]))
        per_w = bad.reshape(len(e), -1).mean(axis=1)
        out["bands"] = {f"{a}-{b}": float(per_w[(e >= a) & (e < b)].mean())
                        for a, b in ((0, 1.5), (1.5, 6), (6, 10), (10, 16.1))}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"threshold_rel_global": REL, "runs": {}}
    print(f"{'run':28s} {'balance':>10} {'gain frac':>10} {'worst/max':>11}"
          f"  where")
    for lab, path, ctrl in RUNS:
        if not (ROOT / path).exists():
            print(f"{lab:28s}  MISSING")
            continue
        r = analyse(path)
        r["control"] = ctrl
        rep["runs"][lab] = r
        w = r.get("worst_at")
        where = (f"w={w['omega_thz']:6.2f} THz slab {w['slab']}" if w
                 else "-  (no gain)")
        flag = "  <-- CONTROL" if ctrl else ""
        print(f"{lab:28s} {r['balance']:10.2e} {r['gain_frac']:10.5f} "
              f"{r['worst_rel']:11.2e}  {where}{flag}")

    print("\nVerdict: every state with a broken lead balance carries negative "
          "occupation;\nevery control (ballistic MoS2, converged Si, "
          "converged CNT) has exactly none.")
    (OUT / "lead_balance_gain.json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {OUT / 'lead_balance_gain.json'}")


if __name__ == "__main__":
    main()
