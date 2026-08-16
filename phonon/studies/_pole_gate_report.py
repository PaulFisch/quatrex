#!/usr/bin/env python
"""Evaluate the pole-sector correctness gates on a saved production run.

Run:
    python phonon/studies/_pole_gate_report.py cluster/pgate/sig_fullf.npz         --pattern cluster/pgate/run_fullf.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def _pattern_from(npz, nnz):
    """Rows/cols of the stored pattern, from the snapshot or reconstructed."""
    for rk, ck in (("rows", "cols"), ("sigma_rows", "sigma_cols")):
        if rk in npz and ck in npz:
            return np.asarray(npz[rk]), np.asarray(npz[ck])
    return None, None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("sigma", help="QX_SAVE_SIGMA .npz")
    p.add_argument("--pattern", help="npz carrying rows/cols, if separate")
    p.add_argument("--block-sizes", type=int, nargs="*", default=None)
    p.add_argument("--assume-kk-only", action="store_true",
                   help="treat sigma_retarded as the KK half and rebuild the "
                        "full retarded before testing")
    a = p.parse_args()

    from quatrex.phonon.pole_audit import keldysh_identity, psd_residual

    z = np.load(a.sigma, allow_pickle=True)
    sl = z["sigma_lesser"].reshape(z["sigma_lesser"].shape[0], -1)
    sg = z["sigma_greater"].reshape(z["sigma_greater"].shape[0], -1)
    sr = z["sigma_retarded"].reshape(z["sigma_retarded"].shape[0], -1)
    if a.assume_kk_only:
        sr = sr + 0.5 * (sl - sg)

    rows = cols = None
    if a.pattern:
        rows, cols = _pattern_from(np.load(a.pattern, allow_pickle=True), sl.shape[1])
    if rows is None:
        rows, cols = _pattern_from(z, sl.shape[1])
    if rows is None:
        print("no pattern (rows/cols) in the snapshot; cannot form Sigma^A on "
              "the pattern. Re-run with --pattern pointing at a file that has "
              "them, or extend the saver to record them.")
        return 2

    rep = keldysh_identity(sr, sl, sg, rows, cols)
    print(f"file            {a.sigma}")
    print(f"shape           {sl.shape[0]} frequencies x {sl.shape[1]} nnz")
    print(f"eps_ki          {rep['eps_ki']:.3e}   (identity residual)")
    print(f"eps_delta_skew  {rep['eps_delta_skew']:.3e}   "
          f"(Sigma^< - Sigma^> anti-Hermitian?)")
    print(f"eps_kk_hermitian{rep['eps_kk_hermitian']:.3e}   (KK part Hermitian?)")

    if a.block_sizes:
        bs = np.array(a.block_sizes, dtype=int)
        for name, vals, sign in (("sigma_lesser", sl, -1.0),
                                 ("sigma_greater", sg, +1.0)):
            r = psd_residual(vals, rows, cols, bs, sign=sign)
            print(f"psd {name:14s} worst={r['worst']:.3e} "
                  f"(scale {r['scale']:.3e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
