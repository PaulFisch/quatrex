"""Phase-1 gate: does gapping the d5a twist mode alone converge eta=0?

Emulates the oxide embedding with transverse pinning springs on the 12
H-shell atoms of the CORRECTED d5a L2 inputs (the fixed n=2-folded
export, validated by _run_d5a_fixed_export), with the translational ASR
re-imposed (LA/TA stay gapless; only the rotational/twist stiffening
survives -- phonon_inputs.embedded_extract). k_pin is bisected so the
Gamma twist gap hits {0.2, 0.5, 1.0} THz.

Rungs run BARE at eta=0 (no IR floor, no low-freq mask, standard linear
mixing): if a gapped twist converges the SCBA, the embedding hypothesis
is validated and the needed gap calibrated -> commit the oxide DFT
campaign (Phase 2). If not, STOP and reassess. Divergence is a RESULT.

Idempotent; needs the d5a_fixed_export inputs (cluster):
    python phonon/scripts/tortin.py launch --name d5agate -- \
        python phonon/studies/_run_d5a_springgate.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "phonon"))

OUT = REPO / "phonon/studies/out/d5a_springgate"
INPUTS = REPO / "phonon/studies/out/d5a_fixed_export/inputs_L2"

NCELLS = 2
NFREQ = 181
FMAX = 66.0
AUX_DW = FMAX / (NFREQ - 1)
AUX_FMAX = 132.0
NRANKS = 64
MAX_ITER = 300
LINKED = ("fc3_blocks.hdf5", "phonon_energies.npy", "structure.xyz")

WC = REPO / "phonon/studies/engine/write_config.py"
RUN = REPO / "phonon/studies/engine/run.py"

TARGETS = (0.2, 0.5, 1.0)  # THz twist gaps


def make_pinned_inputs(target: float) -> Path:
    """Pinned+ASR dynamical matrix for the given twist gap; returns dir."""
    import scipy.io as sio
    from phonon_inputs.embedded_extract import (
        gamma_spectrum, pin_for_twist_gap, read_structure_xyz)

    d = OUT / f"inputs_gap{target:g}"
    mat = d / "dynamical_matrix.mat"
    if mat.exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    M = sio.loadmat(str(INPUTS / "dynamical_matrix.mat"))
    blocks = {}
    for k in M:
        if k.startswith("["):
            blocks[tuple(int(x) for x in k.strip("[]").split(","))] = M[k]
    syms, pos, masses = read_structure_xyz(INPUTS / "structure.xyz")
    surface = np.array([i for i, s in enumerate(syms) if s == "H"])
    fixed, k_pin, gap = pin_for_twist_gap(
        blocks, pos, masses, surface, target_gap_thz=target)
    w = gamma_spectrum(fixed)
    print(f"[pin  ] target {target} THz: k_pin={k_pin:.4f} THz^2*amu, "
          f"achieved gap {gap:.4f} THz; Gamma lowest-5 "
          f"{np.array2string(w[:5], precision=4)}", flush=True)
    if np.abs(w[:3]).max() > 1e-4 or w.min() < -1e-4:
        sys.exit(f"[fatal] pinned matrix invalid: {w[:5]}")
    sio.savemat(str(mat), {f"[{a}, {b}, {c}]": v
                           for (a, b, c), v in fixed.items()})
    for f in LINKED:
        (d / f).symlink_to(INPUTS / f)
    return d


def hygiene() -> None:
    r = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f",
                        "d5a_springgate.*run.py"],
                       capture_output=True, text=True)
    for p in [p for p in r.stdout.split() if p]:
        subprocess.run(["kill", "-9", p], capture_output=True)


def run_rung(target: float) -> None:
    tag = f"gap{target:g}"
    d = OUT / tag
    npz = d / "run.npz"
    if npz.exists():
        print(f"[skip ] {tag}: run.npz exists", flush=True)
        return
    src = make_pinned_inputs(target)
    d.mkdir(parents=True, exist_ok=True)
    for f in ("dynamical_matrix.mat",) + LINKED:
        dst = d / f
        if not dst.exists():
            dst.symlink_to(src / f)
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
    t0 = time.time()
    print(f"[run  ] {tag} (eta=0 BARE, {NRANKS} ranks)", flush=True)
    with open(OUT / f"{tag}.log", "w") as log:
        rc = subprocess.run(
            ["mpirun", "--bind-to", "core", "--map-by", "numa",
             "-np", str(NRANKS), sys.executable, str(RUN)],
            env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"[done ] {tag}: rc={rc} npz={'yes' if npz.exists() else 'MISSING'} "
          f"wall={(time.time() - t0) / 60:.1f} min", flush=True)


def main() -> int:
    if not (INPUTS / "dynamical_matrix.mat").exists():
        sys.exit("[fatal] run _run_d5a_fixed_export.py first (needs the "
                 "corrected inputs_L2).")
    OUT.mkdir(parents=True, exist_ok=True)
    for t in TARGETS:
        run_rung(t)
    print("[done ] d5a spring-pinning gate complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
