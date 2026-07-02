"""Bulk-Si FC3 compressibility measurement -- the factored-SSE go/no-go.

There is no in-repo measurement of how well BULK silicon tensor-decomposes
(the recollection traces to Luo et al. 2025, a cited literature result).
This driver produces it, on the two bulk-Si FC3 sets in the tree:

  big   : phonon/reaps/si_big_hiphive     (5^3 supercell, 4th-NN cutoff --
          the FILM's FC3; the production target). Lift skipped (91 GB).
  small : phonon/reaps/si_primitive_work  (2^3 supercell, 5.5 A cutoff) --
          cheap cross-check where the S3 lift (and hence Waring) is feasible.

Outputs (phonon/scripts/out/bulk_si_compressibility/):
  msvd_spectrum_{tag}.csv   full singular spectrum of the (mu j)|k
                            matricisation (Eckart-Young lower envelope)
  rank_sweep.csv            method, tag, rank, n_params, rel_err, asr legs,
                            fit seconds

Go/no-go per the plan: proceed to the factored film campaign iff some
R* <= ~64 reaches rel_err <= 2% on the big tensor (transport confirmation
follows in the film R-sweep).

Run:  OMP_NUM_THREADS=8 python phonon/scripts/verify/bulk_si_compressibility.py
      [--ranks 4 8 16 32 64 128 256] [--quick]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h5py  # noqa: E402
from phonon_inputs import fc3_compression as fcc  # noqa: E402

OUT = _REPO / "phonon/scripts/out/bulk_si_compressibility"


def load_bulk(subdir: str):
    from phono3py import load as phono3py_load
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    d = _REPO / "phonon" / subdir
    ph3 = phono3py_load(phono3py_yaml=str(d / "phono3py.yaml"),
                        produce_fc=False, log_level=0)
    cell = PhonopyAtoms(symbols=ph3.unitcell.symbols, cell=ph3.unitcell.cell,
                        scaled_positions=ph3.unitcell.scaled_positions)
    phonon = Phonopy(cell, supercell_matrix=ph3.supercell_matrix,
                     primitive_matrix=np.eye(3))
    with h5py.File(d / "fc3.hdf5", "r") as f:
        fc3 = f["fc3"][:]
    return phonon, fc3


def msvd_spectrum(target: fcc.FC3Target, tag: str):
    M = target.T.reshape(target.n_dof * target.dim_sc, target.dim_sc)
    s = np.linalg.svd(M, compute_uv=False)
    tail = np.sqrt(np.maximum(0.0, np.cumsum(s[::-1] ** 2)[::-1]))
    rel_tail = tail / (np.linalg.norm(s) or 1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"msvd_spectrum_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "sigma", "rel_err_at_rank"])
        for r in range(len(s)):
            w.writerow([r, f"{s[r]:.8e}",
                        f"{rel_tail[r]:.8e}"])
    for pct, name in [(0.10, "10%"), (0.02, "2%"), (0.01, "1%")]:
        idx = int(np.searchsorted(-rel_tail, -pct))
        print(f"  [{tag}] mSVD rank for {name} rel err: {idx}")
    return s


def run_sweep(target: fcc.FC3Target, tag: str, methods: dict, writer):
    for method, ranks in methods.items():
        for R in ranks:
            t0 = time.time()
            try:
                if method == "mSVD":
                    res = fcc.fit_msvd(target, rank=R)
                elif method == "INDSCAL":
                    res = fcc.fit_indscal(target, rank=R, n_restarts=8, seed=0)
                elif method == "CP":
                    res = fcc.fit_cp(target, rank=R, n_restarts=8, seed=0)
                elif method == "Waring":
                    res = fcc.fit_waring(target, rank=R, n_restarts=3, seed=0)
                else:
                    continue
                fcc.annotate_result(res, target)
            except Exception as exc:
                print(f"  [{tag}] {method} R={R} FAILED: {exc}", flush=True)
                continue
            asr = res.info.get("asr", {})
            row = [method, tag, R, res.n_params, f"{res.rel_err:.6e}",
                   f"{asr.get('leg_j', np.nan):.3e}",
                   f"{asr.get('leg_k', np.nan):.3e}",
                   f"{time.time() - t0:.1f}"]
            writer.writerow(row)
            print(f"  [{tag}] {method:8s} R={R:4d} rel_err={res.rel_err:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+",
                    default=[4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--quick", action="store_true",
                    help="small tensor + reduced ranks only (smoke)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fcsv = open(OUT / "rank_sweep.csv", "w", newline="")
    writer = csv.writer(fcsv)
    writer.writerow(["method", "tag", "rank", "n_params", "rel_err",
                     "asr_leg_j", "asr_leg_k", "fit_s"])

    # small 2^3 tensor: lift feasible -> all methods incl. Waring
    phonon_s, fc3_s = load_bulk("reaps/si_primitive_work")
    t_small = fcc.build_fc3_target(fc3_s, phonon_s, build_lift=True)
    print(f"[small] target ({t_small.n_dof},{t_small.dim_sc},{t_small.dim_sc}) "
          f"norm={t_small.target_norm:.3e} s2={t_small.s2_residual:.1e}")
    msvd_spectrum(t_small, "small")
    ranks_s = [r for r in args.ranks if r <= t_small.dim_sc]
    run_sweep(t_small, "small",
              {"mSVD": ranks_s, "INDSCAL": ranks_s, "CP": ranks_s,
               "Waring": [r for r in (16, 64) if r in ranks_s or True]},
              writer)
    fcsv.flush()

    if args.quick:
        fcsv.close()
        print("quick mode: done (small only)")
        return

    # big 5^3 tensor: the film's FC3 -- the production target (no lift)
    phonon_b, fc3_b = load_bulk("reaps/si_big_hiphive")
    t_big = fcc.build_fc3_target(fc3_b, phonon_b, build_lift=False)
    print(f"[big] target ({t_big.n_dof},{t_big.dim_sc},{t_big.dim_sc}) "
          f"norm={t_big.target_norm:.3e} s2={t_big.s2_residual:.1e}")
    msvd_spectrum(t_big, "big")
    run_sweep(t_big, "big",
              {"mSVD": args.ranks, "INDSCAL": args.ranks, "CP": args.ranks},
              writer)
    fcsv.close()
    print(f"\nwrote {OUT / 'rank_sweep.csv'}")


if __name__ == "__main__":
    main()
