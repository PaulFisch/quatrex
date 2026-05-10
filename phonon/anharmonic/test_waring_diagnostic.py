"""Diagnose the Waring plateau: verify the lift is S3-symmetric and sweep ranks.

Usage: python test_waring_diagnostic.py [num_restarts]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs import fc3_compression as fc3c


def main():
    n_restarts = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print("Loading FC3 from phono3py (primitive cell) ...")
    import os
    os.chdir(work_dir)
    phonon, _ = load_primitive_cell(Path("."))
    with h5py.File("reaps/si_primitive_work/fc3.hdf5", "r") as f:
        fc3_raw = np.asarray(f["fc3"])
    print(f"  FC3 raw shape: {fc3_raw.shape}")

    target = fc3c.build_fc3_target(fc3_raw, phonon)
    print(
        f"  n_dof={target.n_dof}  dim_sc={target.dim_sc}  "
        f"|T|_F={target.target_norm:.4e}  |T_lifted|_F={np.linalg.norm(target.T_lifted):.4e}"
    )

    # Check S3 symmetry of the lift
    L = target.T_lifted
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    print("\nS3 symmetry check on T_lifted:")
    for p in perms:
        err = np.linalg.norm(L - np.transpose(L, p)) / np.linalg.norm(L)
        print(f"  perm {p}: rel diff = {err:.3e}")

    # Check that slice(T_lifted) == T
    print("\nSlice consistency:")
    T_back = fc3c._slice_to_ndof(target.T_lifted, target.p2s_map)
    err = np.linalg.norm(target.T - T_back) / target.target_norm
    print(f"  |T - slice(T_lifted)| / |T| = {err:.3e}")

    # Sweep Waring ranks with varying restarts
    print(f"\nWaring sweep with n_restarts={n_restarts}:")
    ranks = [4, 8, 16, 24, 36, 48]
    results = {}
    for R in ranks:
        t0 = time.time()
        res = fc3c.fit_waring(
            target,
            rank=R,
            n_restarts=n_restarts,
            n_power_repeats=10,
            n_power_iters=200,
            lbfgs_iters=500,
            seed=0,
            verbose=False,
        )
        dt = time.time() - t0
        per_restart = res.info["restart_errs"]
        print(
            f"  R={R:3d}  sliced={res.rel_err:.4e}  full_lift={res.info['full_lift_rel_err']:.4e}  "
            f"best_init={res.info['best_init']}  "
            f"min/max restart={min(per_restart):.3e}/{max(per_restart):.3e}  "
            f"t={dt:.1f}s"
        )
        results[R] = res

    # Monotonicity check
    print("\nMonotonicity in sliced error:")
    prev = None
    for R in ranks:
        r = results[R].rel_err
        flag = ""
        if prev is not None and r > prev - 1e-4:
            flag = "  (!! NOT MONOTONIC)"
        print(f"  R={R:3d}: {r:.4e}{flag}")
        prev = r


if __name__ == "__main__":
    main()
