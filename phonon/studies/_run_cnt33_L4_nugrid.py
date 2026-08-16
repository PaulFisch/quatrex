"""CNT (3,3) L4 uniform-vs-non-uniform grid A/B -- a cheap, known-to-converge
cross-check of the dual-grid SSE (companion to _run_d5a_nugrid.py).

    nohup python phonon/studies/_run_cnt33_L4_nugrid.py >         phonon/studies/out/cnt33_L4_nugrid/nugrid.log 2>&1 &
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "phonon/studies/out/anderson_test/cnt33_L4_linear"
DOS_NPZ = SRC / "run_gband2.npz"          # the converged baseline (DOS source)
OUT = REPO / "phonon/studies/out/cnt33_L4_nugrid"
FMAX = 55.0
NFREQ_UNI = 361
AUX_DW = FMAX / (NFREQ_UNI - 1)           # 0.1528 THz -- nu's aux == uni's grid
NRANKS = 64
MAX_ITER = 450
GBAND = 2
GEOM = ("dynamical_matrix.mat", "fc3_blocks.hdf5", "structure.xyz")

WC = REPO / "phonon/studies/engine/write_config.py"
MG = REPO / "phonon/studies/make_grid.py"
RUN = REPO / "phonon/studies/engine/run.py"


def _prep_geom(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for f in GEOM:
        dst = d / f
        if not dst.exists():
            dst.symlink_to(SRC / f)


def prep_uni(d: Path) -> None:
    _prep_geom(d)
    subprocess.run(
        [sys.executable, str(WC), "--system", "cnt33", "--work", str(d),
         "-L", "4", "--eta", "0", "--nfreq", str(NFREQ_UNI),
         "--fmax", str(FMAX), "--retarded", "fft", "--mix", "0.2",
         "--max-iter", str(MAX_ITER)],
        check=True)


def prep_nu(d: Path) -> None:
    _prep_geom(d)
    # Non-uniform primary grid from the converged device DOS peaks.
    # DOS-peak comb + 0.35 THz background. NOTE: this UNDER-resolves the
    # acoustic band -- kept as the diagnostic rung; nu2 adds the fix.
    subprocess.run(
        [sys.executable, str(MG), "--npz", str(DOS_NPZ),
         "--fmax", str(FMAX), "--width-thz", "0.3", "--pts-per-line", "8",
         "--max-spacing", "0.35", "--peak-prominence", "0.01",
         "--out", str(d / "phonon_energies.npy")],
        check=True)


def prep_nu2(d: Path) -> None:
    _prep_geom(d)
    # Corrected grid: same comb + background, but the acoustic /
    # low-frequency propagating window (< 9 THz) is held at the uni
    # reference spacing (0.153 THz). The acoustic continuum is smooth and
    # low-DOS -- the comb misses it -- but carries ~23% of the heat, so
    # coarse sampling there (nu's 0.34 THz) shifted the transport by ~5%.
    subprocess.run(
        [sys.executable, str(MG), "--npz", str(DOS_NPZ),
         "--fmax", str(FMAX), "--width-thz", "0.3", "--pts-per-line", "8",
         "--max-spacing", "0.35", "--peak-prominence", "0.01",
         "--lowfreq-fmax", "9", "--lowfreq-spacing", str(AUX_DW),
         "--out", str(d / "phonon_energies.npy")],
        check=True)
    g = np.load(d / "phonon_energies.npy")
    print(f"[grid ] nu2: {g.size} pts on [0, {FMAX}] THz, spacing "
          f"[{np.diff(g).min():.4f}, {np.diff(g).max():.4f}] THz "
          f"(acoustic floor {AUX_DW:.4f} < 9 THz)", flush=True)
    subprocess.run(
        [sys.executable, str(WC), "--system", "cnt33", "--work", str(d),
         "-L", "4", "--eta", "0", "--nfreq", str(NFREQ_UNI),
         "--fmax", str(FMAX), "--retarded", "fft", "--mix", "0.2",
         "--max-iter", str(MAX_ITER), "--freq-grid", "file",
         "--aux-dw", str(AUX_DW), "--aux-fmax", str(FMAX)],
        check=True)
    g = np.load(d / "phonon_energies.npy")
    print(f"[grid ] nu: {g.size} pts on [0, {FMAX}] THz, spacing "
          f"[{np.diff(g).min():.4f}, {np.diff(g).max():.4f}] THz "
          f"(uni reference: {NFREQ_UNI})", flush=True)
    subprocess.run(
        [sys.executable, str(WC), "--system", "cnt33", "--work", str(d),
         "-L", "4", "--eta", "0", "--nfreq", str(NFREQ_UNI),
         "--fmax", str(FMAX), "--retarded", "fft", "--mix", "0.2",
         "--max-iter", str(MAX_ITER), "--freq-grid", "file",
         "--aux-dw", str(AUX_DW), "--aux-fmax", str(FMAX)],
        check=True)


RUNGS = [("uni", prep_uni), ("nu", prep_nu), ("nu2", prep_nu2)]

ENV = dict(os.environ,
           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
           QX_GBAND=str(GBAND))


def hygiene() -> None:
    r = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f",
                        "cnt33_L4_nugrid.*run.py"],
                       capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if p]
    for p in pids:
        subprocess.run(["kill", "-9", p], capture_output=True)
    if pids:
        print(f"[hygiene] killed {len(pids)} leftover ranks", flush=True)
        time.sleep(3)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, prep in RUNGS:
        d = OUT / tag
        npz = d / "run.npz"
        if npz.exists():
            print(f"[skip ] {tag}: run.npz exists", flush=True)
            continue
        prep(d)
        hygiene()
        env = dict(ENV, QX_CONFIG=str(d / "quatrex_config.toml"),
                   QX_NPZ=str(npz))
        t0 = time.time()
        print(f"[run  ] {tag} ({NRANKS} ranks, g_band={GBAND}, "
              f"max_iter={MAX_ITER})", flush=True)
        with open(OUT / f"{tag}.log", "w") as log:
            rc = subprocess.run(
                ["mpirun", "--bind-to", "core", "--map-by", "core",
                 "-np", str(NRANKS), sys.executable, str(RUN)],
                env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        ok = npz.exists()
        print(f"[done ] {tag}: rc={rc} npz={'yes' if ok else 'MISSING'} "
              f"wall={dt / 60:.1f} min", flush=True)
    print("[done ] cnt33 L4 nugrid A/B complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
