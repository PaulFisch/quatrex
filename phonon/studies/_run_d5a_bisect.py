"""d5a eta=0 divergence bisection: surgical frequency ablation (nf181).

    `phonon/docs/d5a_eta0_bisection.md` document stored data.
Run:  nohup python phonon/studies/_run_d5a_bisect.py >           phonon/studies/out/d5a_bisect/bisect.log 2>&1 &
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
TEMPLATE = REPO / "phonon/studies/out/d5a_gridladder/nf181"  # raw rung (built)
OUT = REPO / "phonon/studies/out/d5a_bisect"
FMAX, NF, NRANKS, MAX_ITER = 66.0, 181, 64, 150
GEOM = ("dynamical_matrix.mat", "fc3_blocks.hdf5", "structure.xyz",
        "phonon_energies.npy")

# tag -> (sse_low_freq_mask_thz, zero_bands)
VARIANTS = {
    "ircut":      (1.5, []),
    "gapzero":    (0.0, [[28.0, 45.0]]),
    "flatzero":   (0.0, [[16.5, 28.0]]),
    "ir_gap":     (1.5, [[28.0, 45.0]]),
    "ir_flat":    (1.5, [[16.5, 28.0]]),
    "ir_gap_flat": (1.5, [[16.5, 45.0]]),
}

# Keys deleted from the schema since this driver was written; an archived TOML
# that still sets them is rejected by `extra="forbid"` before the run starts.
# The union of MISSING.md item 2 and the list _run_d5a_nugrid.py already
# strips (commit b5744af3), which is where this same drift was fixed before.
DEAD_KEYS = ("sse_low_freq_cutoff_thz", "sse_zero_bands_thz",
             "sse_cutoff_zero_g", "sse_smooth_window", "support_taper_cells",
             "ir_taper_cells", "sse_ir_subtraction", "band_limit_sse",
             "spectral_support_tol", "band_support_margin_thz",
             "sse_freeze_occupation", "spectral_sharp_cap", "fermi_level")

ENV = dict(os.environ,
           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
           QX_DIAG_SPECTRAL="1")


def prep(tag: str, cutoff: float, bands: list) -> Path:
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    for f in GEOM:
        dst = d / f
        if not dst.exists():
            dst.symlink_to(TEMPLATE / f)
    if bands:
        raise RuntimeError(
            f"variant {tag!r} needs sse_zero_bands_thz, which was removed from "
            "the schema (MISSING.md item 2) with no replacement. Reintroduce "
            "that diagnostic knob before rerunning the band-ablation arms."
        )
    cfg = (TEMPLATE / "quatrex_config.toml").read_text()
    cfg = cfg.replace(str(TEMPLATE), str(d))
    for key in DEAD_KEYS:
        cfg = re.sub(rf"(?m)^{key} = .*\n", "", cfg)
    # Set the live IR knob. `sse_low_freq_mask_thz` is the exact semantic
    # successor of the removed `sse_low_freq_cutoff_thz`.
    cfg = re.sub(r"(?m)^sse_low_freq_mask_thz = .*\n", "", cfg)
    cfg = cfg.replace("[phonon.solver]",
                      f"sse_low_freq_mask_thz = {cutoff}\n[phonon.solver]", 1)
    if f"sse_low_freq_mask_thz = {cutoff}" not in cfg:
        # The failure mode this whole patch exists to prevent: an edit that
        # silently matches nothing and leaves the ablation unapplied.
        raise RuntimeError(
            f"failed to set sse_low_freq_mask_thz in {tag!r}: the template has "
            "no [phonon.solver] section to anchor against."
        )
    (d / "quatrex_config.toml").write_text(cfg)
    return d


def localize(tag: str) -> None:
    """Divergence-localization table from the run npz (if any)."""
    npz = OUT / tag / "run.npz"
    if not npz.exists():
        print(f"[{tag}] no npz -- localization from log only", flush=True)
        return
    z = np.load(npz, allow_pickle=True)
    if "iter_sigL_w" not in z.files:
        return
    s = np.asarray(z["iter_sigL_w"], float)
    if s.shape[0] < 12:
        return
    w = np.linspace(0, FMAX, s.shape[1])
    growth = s[-1] / np.maximum(s[8], 1e-300)
    top = np.argsort(growth)[-6:][::-1]
    print(f"[{tag}] top growth bins (it8 -> end):", flush=True)
    for j in top:
        print(f"    w={w[j]:7.3f} THz  x{growth[j]:.1e}  "
              f"|Sig|end={s[-1, j]:.2e}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, (cutoff, bands) in VARIANTS.items():
        if bands:
            print(f"[skip ] {tag}: needs sse_zero_bands_thz, removed from the "
                  "schema (MISSING.md item 2)", flush=True)
            continue
        d = prep(tag, cutoff, bands)
        npz = d / "run.npz"
        if npz.exists():
            print(f"[skip ] {tag}", flush=True)
            continue
        env = dict(ENV, QX_CONFIG=str(d / "quatrex_config.toml"),
                   QX_NPZ=str(npz))
        t0 = time.time()
        print(f"[run  ] {tag} (low_freq_mask={cutoff}, bands={bands})", flush=True)
        with open(OUT / f"{tag}.log", "w") as log:
            rc = subprocess.run(
                ["mpirun", "--bind-to", "core", "--map-by", "core",
                 "-np", str(NRANKS), sys.executable,
                 str(REPO / "phonon/studies/engine/run.py")],
                env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        txt = (OUT / f"{tag}.log").read_text(errors="ignore")
        n_it = txt.count("rel Sigma^R residual")
        aborted = "ABORTED" in txt
        conv = "SCBA converged" in txt
        print(f"[done ] {tag}: rc={rc} its={n_it} "
              f"{'CONVERGED' if conv else 'ABORTED' if aborted else 'budget'}"
              f" wall={(time.time() - t0) / 60:.1f} min", flush=True)
        localize(tag)
    print("[done ] bisection tree complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
