"""CNT (3,3) L4 uniform-vs-non-uniform grid A/B -- a cheap, known-to-converge
cross-check of the dual-grid SSE (companion to _run_d5a_nugrid.py).

CNT (3,3) L4 converges at eta = 0 (the production baseline reached a genuine
fixed point, best lead balance ~4e-7), so this is a VALIDATION, not a
stability probe: uniform and non-uniform must land on the same conductance
and spectra. The A/B is designed to isolate the grid alone:

  uni : uniform primary = auxiliary grid, nfreq = 361 on [0, 55] THz
        (dw = 0.1528 THz; well-resolved reference).
  nu  : NON-UNIFORM primary grid, ~150 points from the peaks of the
        converged run's device DOS (make_grid --npz, the a-posteriori
        refinement mode -- CNT is dispersive, so the grid follows the real
        spectral structure, not just zone-centre modes), feeding the
        IDENTICAL auxiliary bubble grid (aux_dw = 0.1528, aux_fmax = 55).

Because the auxiliary grid is byte-identical to uni's grid, the 3-phonon
bubble, its fold and the Kramers-Kronig transform are computed the same in
both runs; the ONLY difference is that nu solves the Dyson equation (and
stores G/Sigma) on ~150 non-uniform points instead of 361 uniform ones. If
the conductance and spectra agree, the dual-grid machinery is validated on a
converging device at ~2.4x fewer Dyson solves.

eta = 0 throughout (NO artificial broadening -- see CLAUDE.md). sse_g_band = 2
(required for L >= 3; L4 = 4 transport cells).

Run (background, cluster):
    cd <repo>
    nohup python phonon/studies/_run_cnt33_L4_nugrid.py > \
        phonon/studies/out/cnt33_L4_nugrid/nugrid.log 2>&1 &
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
    # Conservative validation grid: background floor 0.35 THz (finer than
    # 2x the uni reference spacing) so it is provably at least as resolved
    # as uni everywhere the DOS has weight, refined further at the peaks.
    subprocess.run(
        [sys.executable, str(MG), "--npz", str(DOS_NPZ),
         "--fmax", str(FMAX), "--width-thz", "0.3", "--pts-per-line", "8",
         "--max-spacing", "0.35", "--peak-prominence", "0.01",
         "--out", str(d / "phonon_energies.npy")],
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


RUNGS = [("uni", prep_uni), ("nu", prep_nu)]

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
