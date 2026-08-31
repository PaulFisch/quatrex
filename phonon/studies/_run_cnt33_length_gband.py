"""Does a wider inner-G self-energy band (g_band) cure the eta = 0 divergence
of the longer CNT (3,3) chains?

L8 (cnt33_L8_linear/run_gband2.npz: diverged at iter 63) and L10 (all
Idempotent: a rung whose run.npz exists is skipped, so relaunching after the
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AT = REPO / "phonon/studies/out/anderson_test"
OUT = REPO / "phonon/studies/out/cnt33_gband_length"

# Per-length source dir holding the geom set (all four files present on the
# cluster scratch: dynamical_matrix.mat, fc3_blocks.hdf5, phonon_energies.npy,
# structure.xyz).
SRC = {
    8: AT / "cnt33_L8_inputs",
    10: AT / "cnt33_L10_linear",
}
GEOM = ("dynamical_matrix.mat", "fc3_blocks.hdf5", "phonon_energies.npy",
        "structure.xyz")

FMAX = 88.0
NFREQ = 577
NRANKS = 64
MAX_ITER = 600

WC = REPO / "phonon/studies/engine/write_config.py"
RUN = REPO / "phonon/studies/engine/run.py"


def prep(d: Path, L: int) -> None:
    src = SRC[L]
    if not src.is_dir():
        sys.exit(f"[fatal] geom source {src} missing -- build it with "
                 f"_tile_device_inputs.py or point SRC[{L}] elsewhere.")
    d.mkdir(parents=True, exist_ok=True)
    for f in GEOM:
        dst = d / f
        if not dst.exists():
            if not (src / f).exists():
                sys.exit(f"[fatal] {src / f} missing for L{L}")
            dst.symlink_to(src / f)
    subprocess.run(
        [sys.executable, str(WC), "--system", "cnt33", "--work", str(d),
         "-L", str(L), "--eta", "0", "--nfreq", str(NFREQ),
         "--fmax", str(FMAX), "--retarded", "fft", "--mix", "0.2",
         "--max-iter", str(MAX_ITER)],
        check=True)


def hygiene() -> None:
    r = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f",
                        "cnt33_gband_length.*run.py"],
                       capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if p]
    for p in pids:
        subprocess.run(["kill", "-9", p], capture_output=True)
    if pids:
        print(f"[hygiene] killed {len(pids)} leftover ranks", flush=True)
        time.sleep(3)


def run_rung(L: int, g: int) -> None:
    tag = f"L{L}_g{g}"
    d = OUT / tag
    npz = d / "run.npz"
    if npz.exists():
        print(f"[skip ] {tag}: run.npz exists", flush=True)
        return
    prep(d, L)
    hygiene()
    env = dict(os.environ,
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               QX_GBAND=str(g),
               QX_SCATCONTACTS="0",              # bare harmonic reservoirs
               QX_BBCHECK="1",                    # log the bubble balance
               QX_CONFIG=str(d / "quatrex_config.toml"),
               QX_NPZ=str(npz))
    t0 = time.time()
    print(f"[run  ] {tag} (eta=0, bare contacts, fmax=88, {NRANKS} ranks)",
          flush=True)
    with open(OUT / f"{tag}.log", "w") as log:
        rc = subprocess.run(
            ["mpirun", "--bind-to", "core", "--map-by", "core",
             "-np", str(NRANKS), sys.executable, str(RUN)],
            env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"[done ] {tag}: rc={rc} "
          f"npz={'yes' if npz.exists() else 'MISSING'} "
          f"wall={(time.time() - t0) / 60:.1f} min", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lengths", type=int, nargs="+", default=[8, 10],
                    help="device cell counts to sweep (default 8 10)")
    ap.add_argument("--gbands", type=int, nargs="+", default=[1, 2],
                    help="sse_g_band values to sweep (default 1 2; add 3 once "
                         "the k=3 RGF recursion lands)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for L in args.lengths:
        for g in args.gbands:
            run_rung(L, g)
    print(f"[done ] cnt33 g_band-length sweep complete "
          f"(lengths={args.lengths}, gbands={args.gbands}).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
