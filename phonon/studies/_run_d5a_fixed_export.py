"""d5a re-baseline on the FIXED FC2 export (Phase 0 of the twist-mode plan).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "phonon/studies/out/d5a_fixed_export"
INPUTS = OUT / "inputs_L2"

NCELLS = 2
NFREQ = 361
FMAX = 132.0
NRANKS = 64
MAX_ITER = 300
GEOM = ("dynamical_matrix.mat", "fc3_blocks.hdf5", "phonon_energies.npy",
        "structure.xyz")

BI = REPO / "phonon/studies/engine/build_inputs.py"
WC = REPO / "phonon/studies/engine/write_config.py"
RUN = REPO / "phonon/studies/engine/run.py"

# Expected corrected Gamma spectrum (from the reap FC2, folded export):
# three exact zeros, the 0.0075 THz twist, optical onset 2.7475 THz.
GAMMA_TWIST = 0.00754
GAMMA_OPT = 2.7475


def build_and_validate_inputs() -> None:
    if not (INPUTS / "dynamical_matrix.mat").exists():
        subprocess.run(
            [sys.executable, str(BI), "--system", "sinw_d5a",
             "-L", str(NCELLS), "--nfreq", str(NFREQ), "--fmax", str(FMAX),
             "--out", str(INPUTS)],
            check=True)
    import scipy.io as sio
    M = sio.loadmat(str(INPUTS / "dynamical_matrix.mat"))
    keys = [k for k in M if k.startswith("[")]
    D = sum(M[k] for k in keys)
    D = 0.5 * (D + D.conj().T)
    w2 = np.linalg.eigvalsh(D)
    w = np.sign(w2) * np.sqrt(np.abs(w2))
    print(f"[check] emitted Gamma lowest-6: "
          f"{np.array2string(w[:6], precision=5)}", flush=True)
    if w.min() < -1e-4:
        sys.exit(f"[fatal] imaginary Gamma modes in the fixed export: {w[:4]}")
    if abs(w[3] - GAMMA_TWIST) > 5e-3 or abs(w[4] - GAMMA_OPT) > 5e-2:
        sys.exit("[fatal] emitted spectrum does not match the reap FC2 "
                 f"(twist {w[3]:.5f} vs {GAMMA_TWIST}, optical {w[4]:.4f} "
                 f"vs {GAMMA_OPT}).")
    if np.abs(w[:3]).max() > 1e-4:
        sys.exit(f"[fatal] translations not exact: {w[:3]}")
    print("[check] fixed export validated (ASR exact, twist "
          f"{w[3]*1e3:.2f} mTHz, optical {w[4]:.3f} THz).", flush=True)


def hygiene() -> None:
    r = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f",
                        "d5a_fixed_export.*run.py"],
                       capture_output=True, text=True)
    for p in [p for p in r.stdout.split() if p]:
        subprocess.run(["kill", "-9", p], capture_output=True)


def run_rung(tag: str) -> None:
    d = OUT / tag
    npz = d / "run.npz"
    if npz.exists():
        print(f"[skip ] {tag}: run.npz exists", flush=True)
        return
    d.mkdir(parents=True, exist_ok=True)
    for f in GEOM:
        dst = d / f
        if not dst.exists():
            dst.symlink_to(INPUTS / f)
    subprocess.run(
        [sys.executable, str(WC), "--system", "sinw_d5a", "--work", str(d),
         "-L", str(NCELLS), "--eta", "0", "--nfreq", str(NFREQ),
         "--fmax", str(FMAX), "--retarded", "fft", "--mix", "0.1",
         "--max-iter", str(MAX_ITER)],
        check=True)
    hygiene()
    env = dict(os.environ,
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               QX_SCATCONTACTS="0", QX_BBCHECK="1",
               QX_CONFIG=str(d / "quatrex_config.toml"),
               QX_NPZ=str(npz))
    t0 = time.time()
    print(f"[run  ] {tag} (eta=0, raw interaction, {NRANKS} ranks)",
          flush=True)
    with open(OUT / f"{tag}.log", "w") as log:
        rc = subprocess.run(
            ["mpirun", "--bind-to", "core", "--map-by", "numa",
             "-np", str(NRANKS), sys.executable, str(RUN)],
            env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"[done ] {tag}: rc={rc} npz={'yes' if npz.exists() else 'MISSING'} "
          f"wall={(time.time() - t0) / 60:.1f} min", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    build_and_validate_inputs()
    run_rung("bare")
    print("[done ] d5a fixed-export re-baseline complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
