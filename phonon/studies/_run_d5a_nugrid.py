"""d5a L2 eta=0 NON-UNIFORM-grid experiment (dual-grid SSE).

    nohup python phonon/studies/_run_d5a_nugrid.py >         phonon/studies/out/d5a_nugrid/nugrid.log 2>&1 &
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "phonon/studies/out/conv1e10/work/sinw_d5a_L2_eta0_diag"
OUT = REPO / "phonon/studies/out/d5a_nugrid"
FMAX = 66.0
AUX_DW = FMAX / 1440.0          # the 1441-rung resolution
NRANKS = 64
MAX_ITER = 150
GEOM = ("dynamical_matrix.mat", "fc3_blocks.hdf5", "structure.xyz")

sys.path.insert(0, str(REPO / "phonon/studies"))
from make_grid import _modes_from_dyn, build_grid  # noqa: E402

ENV = dict(os.environ,
           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
           QX_DIAG_SPECTRAL="1")

# (tag, nfreq_or_None, aux_dw, aux_fmax); None nfreq = non-uniform grid.
RUNGS = [
    ("uni361", 361, 0.0, 0.0),
    ("nu", None, AUX_DW, 66.0),
    ("nu_kk", None, AUX_DW, 132.0),
]


def _nu_grid(d: Path) -> np.ndarray:
    modes = _modes_from_dyn(d / "dynamical_matrix.mat")
    return build_grid(modes, FMAX, width=0.15, pts_per_line=10,
                      max_spacing=FMAX / 360.0, min_spacing=AUX_DW)


def prep(tag: str, nf: int | None, aux_dw: float, aux_fmax: float) -> Path:
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    for f in GEOM:
        dst = d / f
        if not dst.exists():
            dst.symlink_to(SRC / f)
    if nf is None:
        grid = _nu_grid(d)
        nf = int(grid.size)
        sp = np.diff(grid)
        print(f"[grid ] {tag}: {nf} pts, spacing "
              f"[{sp.min():.4f}, {sp.max():.4f}] THz "
              f"(uniform at min spacing: {int(FMAX / sp.min()) + 1})",
              flush=True)
    else:
        grid = np.linspace(0.0, FMAX, nf)
    np.save(d / "phonon_energies.npy", grid)
    cfg = (SRC / "quatrex_config.toml").read_text()
    cfg = cfg.replace(str(SRC), str(d))
    cfg = re.sub(r"(?m)^energy_window_num = \d+",
                 f"energy_window_num = {nf}", cfg)
    cfg = re.sub(r"(?m)^max_iterations = \d+",
                 f"max_iterations = {MAX_ITER}", cfg)
    # keys REMOVED from the schema since the diag config was written
    # (smooth window 2026-07-06; masks/taps/caps folded into
    # sse_low_freq_mask_thz; fermi_level dropped for phonon runs) --
    # delete them, the current defaults ARE the fully-raw recipe.
    for key in ("sse_smooth_window", "support_taper_cells",
                "ir_taper_cells", "band_limit_sse",
                "band_support_margin_thz", "sse_freeze_occupation",
                "sse_cutoff_zero_g", "sse_low_freq_cutoff_thz",
                "spectral_sharp_cap", "fermi_level"):
        cfg = re.sub(rf"(?m)^{key} = .*\n", "", cfg)
    # strip any stale grid keys, then set this rung's
    cfg = re.sub(r"(?m)^(frequency_grid|sse_aux_grid_dw_thz|"
                 r"sse_aux_grid_fmax_thz) = .*\n", "", cfg)
    gridkeys = ""
    if aux_dw > 0.0:
        gridkeys = (f'frequency_grid = "file"\n'
                    f"sse_aux_grid_dw_thz = {aux_dw}\n"
                    f"sse_aux_grid_fmax_thz = {aux_fmax}\n")
    cfg = cfg.replace("[phonon.solver]", gridkeys + "[phonon.solver]", 1)
    # Migrate pre-namespace mixer keys (cf. _run_d5a_gridladder.py).
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
                        "d5a_nugrid.*run.py"],
                       capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if p]
    for p in pids:
        subprocess.run(["kill", "-9", p], capture_output=True)
    if pids:
        print(f"[hygiene] killed {len(pids)} leftover ranks", flush=True)
        time.sleep(3)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, nf, aux_dw, aux_fmax in RUNGS:
        d = prep(tag, nf, aux_dw, aux_fmax)
        npz = d / "run.npz"
        if npz.exists():
            print(f"[skip ] {tag}: run.npz exists", flush=True)
            continue
        hygiene()
        env = dict(ENV, QX_CONFIG=str(d / "quatrex_config.toml"),
                   QX_NPZ=str(npz))
        t0 = time.time()
        print(f"[run  ] {tag} ({NRANKS} ranks, max_iter={MAX_ITER})",
              flush=True)
        with open(OUT / f"{tag}.log", "w") as log:
            rc = subprocess.run(
                ["mpirun", "--bind-to", "core", "--map-by", "core",
                 "-np", str(NRANKS), sys.executable,
                 str(REPO / "phonon/studies/engine/run.py")],
                env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        ok = npz.exists()
        print(f"[done ] {tag}: rc={rc} npz={'yes' if ok else 'MISSING'} "
              f"wall={dt / 60:.1f} min", flush=True)
        if not ok:
            print(f"[warn ] {tag} produced no snapshot (diverged?) -- "
                  "continuing", flush=True)
    print("[done ] d5a nugrid experiment complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
