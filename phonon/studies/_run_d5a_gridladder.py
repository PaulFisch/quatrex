"""d5a L2 eta=0 BARE-SSE grid-resolution ladder + grid-band alignment scan.

Hypothesis (flat bands): d5a's flat branches are sharp Lorentzian resonances
whose self-consistent linewidth Gamma must be RESOLVED by the frequency grid
(d_omega < Gamma) -- the eta=0 non-convergence may be a resolution problem,
not a physics one. Two experiments, all rungs BARE (ir_taper_cells = 0; the
harmonic support machinery -- band_support_margin, smooth window -- stays, it
is a G-independent support mask shown interior-continuity-innocent):

  (1) RESOLUTION ladder: nfreq in {181, 361, 721, 1441} at fmax = 66 THz
      (d_omega 0.367 -> 0.046 THz). Does the residual floor / limit cycle
      shrink as d_omega crosses below the flat-band Gamma_anh
      (phonon/scripts/verify/d5a_gamma_anh.npz predicts the threshold)?
  (2) ALIGNMENT scan: nfreq in {185, 189, 193} -- ~constant resolution, but
      the bins shift RELATIVE to the flat bands (equivalent to shifting the
      bands; the grid must stay zero-based -- an emin offset would break the
      bosonic reflection fold). If the marginal bins are grid-band HITS, the
      convergence behaviour changes qualitatively between micro-rungs.

Every rung runs with QX_DIAG_SPECTRAL=1 (per-iteration full-omega G/Sigma
spectra -> per-bin iteration variance localises the limit-cycling bins) and
saves the standard npz (+ slab_absorption). Sequential rungs, 64 MPI ranks
(omega-axis stack split; 63-DOF blocks -> ring pool not useful), single-thread
BLAS per phonon/CLAUDE.md.

Run (background):
    cd <repo>
    nohup python phonon/studies/_run_d5a_gridladder.py > \
        phonon/studies/out/d5a_gridladder/ladder.log 2>&1 &

Figure: phonon/scripts/figures/d5a_grid_ladder.py.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "phonon/studies/out/conv1e10/work/sinw_d5a_L2_eta0_diag"
OUT = REPO / "phonon/studies/out/d5a_gridladder"
FMAX = 66.0
NRANKS = 64
MAX_ITER = 150
GEOM = ("dynamical_matrix.mat", "fc3_blocks.hdf5", "structure.xyz")

# resolution ladder first (181 calibrates), then the cheap alignment trio,
# then the expensive fine rungs.
RUNGS = [181, 185, 189, 193, 361, 721, 1441]

ENV = dict(os.environ,
           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
           QX_DIAG_SPECTRAL="1")


def prep(nf: int) -> Path:
    d = OUT / f"nf{nf}"
    d.mkdir(parents=True, exist_ok=True)
    for f in GEOM:
        dst = d / f
        if not dst.exists():
            dst.symlink_to(SRC / f)
    np.save(d / "phonon_energies.npy", np.linspace(0.0, FMAX, nf))
    cfg = (SRC / "quatrex_config.toml").read_text()
    cfg = cfg.replace(str(SRC), str(d))
    cfg = re.sub(r"(?m)^energy_window_num = \d+",
                 f"energy_window_num = {nf}", cfg)
    cfg = re.sub(r"(?m)^max_iterations = \d+",
                 f"max_iterations = {MAX_ITER}", cfg)
    cfg = re.sub(r"(?m)^ir_taper_cells = [0-9.eE+-]+",
                 "ir_taper_cells = 0.0", cfg)
    # Migrate pre-namespace mixer keys (the diag config predates the
    # [scba.experimental_mixer] move; the current schema forbids extras).
    moved, keep = [], []
    for ln in cfg.splitlines():
        key = ln.split("=")[0].strip()
        if key in ("rre_cycle", "broyden_warmup_iters", "broyden_ridge",
                   "broyden_trust", "rpm_max_subspace") or \
                key.startswith("jfnk_"):
            moved.append(ln)
        else:
            keep.append(ln)
    cfg = "\n".join(keep)
    if moved:
        block = "\n[scba.experimental_mixer]\n" + "\n".join(moved) + "\n"
        cfg = (cfg.replace("[electron]", block + "\n[electron]", 1)
               if "[electron]" in cfg else cfg + block)
    (d / "quatrex_config.toml").write_text(cfg + "\n")
    return d


def hygiene() -> None:
    r = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f",
                        "d5a_gridladder.*run.py"],
                       capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if p]
    for p in pids:
        subprocess.run(["kill", "-9", p], capture_output=True)
    if pids:
        print(f"[hygiene] killed {len(pids)} leftover ranks", flush=True)
        time.sleep(3)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for nf in RUNGS:
        d = prep(nf)
        npz = d / "run.npz"
        if npz.exists():
            print(f"[skip ] nf{nf}: run.npz exists", flush=True)
            continue
        hygiene()
        env = dict(ENV, QX_CONFIG=str(d / "quatrex_config.toml"),
                   QX_NPZ=str(npz))
        t0 = time.time()
        print(f"[run  ] nf{nf} (d_omega={FMAX / (nf - 1):.4f} THz, "
              f"{NRANKS} ranks, max_iter={MAX_ITER})", flush=True)
        with open(OUT / f"nf{nf}.log", "w") as log:
            rc = subprocess.run(
                ["mpirun", "--bind-to", "core", "--map-by", "core",
                 "-np", str(NRANKS), sys.executable,
                 str(REPO / "phonon/studies/engine/run.py")],
                env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        ok = npz.exists()
        print(f"[done ] nf{nf}: rc={rc} npz={'yes' if ok else 'MISSING'} "
              f"wall={dt / 60:.1f} min", flush=True)
        if not ok:
            # A diverged rung is a RESULT here (the residual trace lives in
            # the log); keep climbing -- finer grids are the hypothesis.
            print(f"[warn ] nf{nf} produced no snapshot (diverged?) -- "
                  "continuing with the next rung", flush=True)
    print("[done ] d5a grid ladder complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
