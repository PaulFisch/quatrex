"""Correctness A/B for the SSE fast-path options (WP3/WP4/WP5).

Usage:
    python phonon/studies/_verify_sse_opts.py         --base phonon/studies/out/anderson_test/cnt33_linear/quatrex_config.toml         --iters 4 [--ring 8] [--workdir .../sse_opt_verify]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

VARIANTS = {
    "legacy": [],
    "gfl": ["sse_greater_from_lesser = true",
            "sse_fold_verify_iterations = 2"],
    "herm": ["sse_hermitian_pairs = true"],
    "both": ["sse_greater_from_lesser = true",
             "sse_hermitian_pairs = true"],
}
RTOL = {"gfl": 1e-10, "herm": 1e-8, "both": 1e-8}


def make_config(base: Path, workdir: Path, tag: str, lines: list[str]) -> Path:
    text = base.read_text()
    vdir = workdir / tag
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "out").mkdir(exist_ok=True)
    base_dir = str(base.parent)
    text = text.replace(base_dir, str(vdir))
    # inputs stay in the base dir: point the symlinked files back
    for f in ("dynamical_matrix.mat", "fc3_blocks.hdf5", "structure.xyz",
              "phonon_energies.npy"):
        if not (vdir / f).exists():
            (vdir / f).symlink_to(base.parent / f)
    if lines:
        text = text.replace("[phonon]", "[phonon]\n" + "\n".join(lines), 1)
    cfg = vdir / "quatrex_config.toml"
    cfg.write_text(text)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--ring", default="8")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    base = Path(args.base).resolve()
    workdir = Path(args.workdir or base.parent.parent / "sse_opt_verify")
    engine = Path(__file__).resolve().parents[1] / "studies/engine/run.py"

    results, logs = {}, {}
    for tag, lines in VARIANTS.items():
        cfg = make_config(base, workdir, tag, lines)
        npz = cfg.parent / "run.npz"
        env = os.environ.copy()
        env.update(
            QX_CONFIG=str(cfg), QX_MAXIT=str(args.iters), QX_MINIT="1",
            QX_NPZ=str(npz), QX_SAVE_DIAG_G="0",
            QUATREX_PHPH_RING_THREADS=args.ring,
            OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        )
        print(f"--- {tag}: running {args.iters} iterations", flush=True)
        proc = subprocess.run([sys.executable, str(engine)], env=env,
                              capture_output=True, text=True)
        logs[tag] = proc.stdout + proc.stderr
        (cfg.parent / "run.log").write_text(logs[tag])
        if not npz.exists():
            print(logs[tag][-3000:])
            print(f"    {tag} FAILED to produce {npz}")
            return 1
        d = np.load(npz)
        results[tag] = (np.asarray(d["iter_heat"]),
                        np.asarray(d["iter_sigma_max"]))
        for ln in logs[tag].splitlines():
            if "fold-verify" in ln or "PhPh SSE ring:" in ln:
                print("   ", ln)
        bal = [ln for ln in logs[tag].splitlines()
               if "Bubble energy balance" in ln]
        if bal:
            print("   ", bal[-1])

    ref_h, ref_s = results["legacy"]
    ok = True
    print(f"\n{'variant':>8} {'max rel dJ':>12} {'max rel dSigma':>15}  verdict")
    for tag in ("gfl", "herm", "both"):
        h, s = results[tag]
        dh = float(np.max(np.abs(h - ref_h) / np.maximum(np.abs(ref_h), 1e-300)))
        ds = float(np.max(np.abs(s - ref_s) / np.maximum(np.abs(ref_s), 1e-300)))
        good = dh < RTOL[tag] and ds < RTOL[tag]
        ok &= good
        print(f"{tag:>8} {dh:12.3e} {ds:15.3e}  "
              f"{'PASS' if good else f'FAIL (tol {RTOL[tag]:.0e})'}")
    gate = re.findall(r"fold-verify.*", logs["gfl"])
    if gate and not all("OK" in g for g in gate):
        print("fold-verify gate reported MISMATCH")
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
