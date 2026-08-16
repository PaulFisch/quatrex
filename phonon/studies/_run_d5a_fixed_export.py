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
NFREQ = 181
FMAX = 66.0
AUX_DW = FMAX / (NFREQ - 1)
AUX_FMAX = 132.0                      # >= 2*omega_max (H modes) -- complete KK
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


def run_rung(tag: str, lowmask: float) -> None:
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
         "--max-iter", str(MAX_ITER),
         "--aux-dw", str(AUX_DW), "--aux-fmax", str(AUX_FMAX)],
        check=True)
    hygiene()
    env = dict(os.environ,
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               QX_SCATCONTACTS="0", QX_BBCHECK="1",
               QX_CONFIG=str(d / "quatrex_config.toml"),
               QX_NPZ=str(npz))
    if lowmask > 0.0:
        env["QX_SSE_LOWMASK"] = str(lowmask)
    t0 = time.time()
    print(f"[run  ] {tag} (eta=0, lowmask={lowmask}, {NRANKS} ranks)",
          flush=True)
    with open(OUT / f"{tag}.log", "w") as log:
        rc = subprocess.run(
            ["mpirun", "--bind-to", "core", "--map-by", "numa",
             "-np", str(NRANKS), sys.executable, str(RUN)],
            env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"[done ] {tag}: rc={rc} npz={'yes' if npz.exists() else 'MISSING'} "
          f"wall={(time.time() - t0) / 60:.1f} min", flush=True)


OLD_FC2 = (REPO / "phonon/scripts/out/prod/sinw_d5a/work/T100/"
           "dynamical_matrix.mat")


def prep_oldfc2_inputs() -> bool:
    """Attribution control: the CORRUPTED historical FC2 with otherwise
    identical settings (grid, aux/KK support, FC3), so the corrected-
    baseline shift can be attributed to the FC2 fix alone. The old .mat
    may be pre-placed in inputs_oldfc2/ (e.g. scp'd from the laptop
    mirror) or found at the historical prod path."""
    d = OUT / "inputs_oldfc2"
    d.mkdir(parents=True, exist_ok=True)
    mat = d / "dynamical_matrix.mat"
    if not mat.exists():
        if not OLD_FC2.exists():
            return False
        mat.symlink_to(OLD_FC2)
    for f in GEOM:
        if f == "dynamical_matrix.mat":
            continue
        dst = d / f
        if not dst.exists():
            dst.symlink_to(INPUTS / f)
    return True


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    build_and_validate_inputs()
    run_rung("bare", 0.0)
    run_rung("ircut", 1.5)
    # Attribution control on the historical corrupted FC2.
    if prep_oldfc2_inputs():
        global INPUTS
        saved = INPUTS
        INPUTS = OUT / "inputs_oldfc2"
        try:
            run_rung("ircut_oldfc2", 1.5)
        finally:
            INPUTS = saved
    else:
        print(f"[warn ] old FC2 not found ({OLD_FC2} or pre-placed "
              "inputs_oldfc2/dynamical_matrix.mat); skipping the "
              "attribution control.", flush=True)
    print("[done ] d5a fixed-export re-baseline complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
