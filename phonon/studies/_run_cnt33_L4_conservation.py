"""Why does lead balance fail with dressed contacts? Conservation-identity
diagnostic on CNT (3,3) L4 (cheaper than L8, same mechanism).

    nohup python phonon/studies/_run_cnt33_L4_conservation.py >         phonon/studies/out/cnt33_L4_conservation/cons.log 2>&1 &
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "phonon/studies/out/anderson_test/cnt33_L4_linear"
OUT = REPO / "phonon/studies/out/cnt33_L4_conservation"
FMAX = 55.0
NFREQ = 361
AUX_DW = FMAX / (NFREQ - 1)
AUX_FMAX = 88.0
NRANKS = 64
MAX_ITER = 120
GEOM = ("dynamical_matrix.mat", "fc3_blocks.hdf5", "structure.xyz")

WC = REPO / "phonon/studies/engine/write_config.py"
RUN = REPO / "phonon/studies/engine/run.py"

# (tag, obc_scattering)
RUNGS = [("bare", False), ("dressed", True)]


def prep(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for f in GEOM:
        dst = d / f
        if not dst.exists():
            dst.symlink_to(SRC / f)
    subprocess.run(
        [sys.executable, str(WC), "--system", "cnt33", "--work", str(d),
         "-L", "4", "--eta", "0", "--nfreq", str(NFREQ), "--fmax", str(FMAX),
         "--retarded", "fft", "--mix", "0.2", "--max-iter", str(MAX_ITER),
         "--aux-dw", str(AUX_DW), "--aux-fmax", str(AUX_FMAX)],
        check=True)


def hygiene() -> None:
    r = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f",
                        "cnt33_L4_conservation.*run.py"],
                       capture_output=True, text=True)
    for p in [p for p in r.stdout.split() if p]:
        subprocess.run(["kill", "-9", p], capture_output=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, obc in RUNGS:
        d = OUT / tag
        npz = d / "run.npz"
        if npz.exists():
            print(f"[skip ] {tag}: run.npz exists", flush=True)
            continue
        prep(d)
        hygiene()
        env = dict(os.environ,
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                   QX_GBAND="2", QX_BBCHECK="1",
                   QX_SCATCONTACTS=("1" if obc else "0"),
                   QX_CONFIG=str(d / "quatrex_config.toml"),
                   QX_NPZ=str(npz))
        t0 = time.time()
        print(f"[run  ] {tag} (obc_scattering={obc}, bubble_balance_check, "
              f"{NRANKS} ranks, g_band=2)", flush=True)
        with open(OUT / f"{tag}.log", "w") as log:
            rc = subprocess.run(
                ["mpirun", "--bind-to", "core", "--map-by", "core",
                 "-np", str(NRANKS), sys.executable, str(RUN)],
                env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"[done ] {tag}: rc={rc} npz={'yes' if npz.exists() else 'MISSING'} "
              f"wall={(time.time() - t0) / 60:.1f} min", flush=True)
    print("[done ] cnt33 L4 conservation diagnostic complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
