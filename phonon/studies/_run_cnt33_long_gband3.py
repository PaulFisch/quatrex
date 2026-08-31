"""Distributed banded RGF on long CNT (3,3) chains: parity, scaling, and the
L16/L24/L32 eta=0 physics push.

Phases (all idempotent -- a rung with run.npz / bench.json is skipped):
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AT = REPO / "phonon/studies/out/anderson_test"
SRC_L8 = AT / "cnt33_L8_inputs"
OUT = REPO / "phonon/studies/out/cnt33_long_gband3"

FMAX = 55.0
NFREQ = 361
AUX_DW = FMAX / (NFREQ - 1)
AUX_FMAX = 88.0
MAX_ITER = 600
GEOM = ("dynamical_matrix.mat", "fc3_blocks.hdf5", "phonon_energies.npy",
        "structure.xyz")

WC = REPO / "phonon/studies/engine/write_config.py"
RUN = REPO / "phonon/studies/engine/run.py"
TILE = REPO / "phonon/studies/_tile_device_inputs.py"


def tiled_inputs(L: int) -> Path:
    if L == 8:
        return SRC_L8
    d = AT / f"cnt33_L{L}_inputs"
    if not (d / "fc3_blocks.hdf5").exists():
        subprocess.run(
            [sys.executable, str(TILE), "--src", str(SRC_L8),
             "--cells", str(L), "--out", str(d)],
            check=True)
    return d


def prep(d: Path, L: int, max_iter: int) -> None:
    src = tiled_inputs(L)
    d.mkdir(parents=True, exist_ok=True)
    for f in GEOM:
        dst = d / f
        if not dst.exists():
            dst.symlink_to(src / f)
    subprocess.run(
        [sys.executable, str(WC), "--system", "cnt33", "--work", str(d),
         "-L", str(L), "--eta", "0", "--nfreq", str(NFREQ),
         "--fmax", str(FMAX), "--retarded", "fft", "--mix", "0.2",
         "--max-iter", str(max_iter),
         "--aux-dw", str(AUX_DW), "--aux-fmax", str(AUX_FMAX)],
        check=True)


def hygiene() -> None:
    r = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f",
                        "cnt33_long_gband3.*run.py"],
                       capture_output=True, text=True)
    for p in [p for p in r.stdout.split() if p]:
        subprocess.run(["kill", "-9", p], capture_output=True)


def run_rung(tag: str, L: int, g: int, bcs: int, nranks: int,
             max_iter: int, bind: str = "core") -> float:
    """Returns wall seconds (0.0 if skipped)."""
    d = OUT / tag
    npz = d / "run.npz"
    if npz.exists():
        print(f"[skip ] {tag}: run.npz exists", flush=True)
        return 0.0
    prep(d, L, max_iter)
    hygiene()
    env = dict(os.environ,
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               QX_GBAND=str(g), QX_BCS=str(bcs),
               QX_SCATCONTACTS="0", QX_BBCHECK="1",
               QX_CONFIG=str(d / "quatrex_config.toml"),
               QX_NPZ=str(npz))
    bind_args = (["--bind-to", "core", "--map-by", "core"]
                 if bind == "core" else
                 ["--bind-to", "core", "--map-by", "numa"])
    t0 = time.time()
    print(f"[run  ] {tag} (L={L} g={g} bcs={bcs} np={nranks} bind={bind} "
          f"max_iter={max_iter})", flush=True)
    with open(OUT / f"{tag}.log", "w") as log:
        rc = subprocess.run(
            ["mpirun", *bind_args, "-np", str(nranks),
             sys.executable, str(RUN)],
            env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    print(f"[done ] {tag}: rc={rc} npz={'yes' if npz.exists() else 'MISSING'} "
          f"wall={wall / 60:.1f} min", flush=True)
    return wall


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- A. parity: bcs=1 vs bcs=2, identical 25-iteration L8 g3 runs.
    print("===== PHASE A: distributed parity (L8 g3, 25 it) =====",
          flush=True)
    run_rung("parity_bcs1", 8, 3, 1, 64, 25)
    run_rung("parity_bcs2", 8, 3, 2, 64, 25)
    try:
        import numpy as np
        d1 = np.load(OUT / "parity_bcs1/run.npz")
        d2 = np.load(OUT / "parity_bcs2/run.npz")
        # lead_current and last_heat are all-reduced over the stack, so
        # they are comparable across different (stack, block) layouts;
        # the reduced-system path reorders flops -> ~1e-8, not bitwise.
        lc1, lc2 = float(d1["lead_current"]), float(d2["lead_current"])
        rel = abs(lc1 - lc2) / max(abs(lc1), 1e-300)
        # The distributed RGF NaNs the INTERIOR interfaces by design
        # (only lead currents are valid) -- compare the two leads only.
        h1 = np.asarray(d1["last_heat"])[[0, -1]]
        h2 = np.asarray(d2["last_heat"])[[0, -1]]
        lh = (np.max(np.abs(h1 - h2))
              / max(np.max(np.abs(h1)), 1e-300))
        verdict = "PASS" if (rel < 1e-6 and lh < 1e-6) else "FAIL"
        print(f"[parity] lead_current bcs1={lc1:.10g} bcs2={lc2:.10g} "
              f"rel={rel:.2e}; last_heat rel-max={lh:.2e} -> {verdict}",
              flush=True)
        if verdict == "FAIL":
            print("[parity] ABORTING before burning cluster time on "
                  "long devices.", flush=True)
            return 1
    except Exception as e:  # noqa: BLE001 -- report, do not crash the chain
        print(f"[parity] comparison failed to run: {e!r}", flush=True)
        return 1

    # ---- B. bench grid on L16 g3 (4 iterations per point).
    print("===== PHASE B: scaling bench (L16 g3, 4 it) =====", flush=True)
    bench_file = OUT / "bench.json"
    bench = json.loads(bench_file.read_text()) if bench_file.exists() else {}
    grid = [
        # (bcs, nranks, bind)
        (1, 64, "core"), (1, 64, "numa"), (1, 128, "core"),
        (2, 64, "core"), (2, 128, "core"),
        (4, 64, "core"), (4, 128, "core"),
    ]
    for bcs, nranks, bind in grid:
        key = f"bcs{bcs}_np{nranks}_{bind}"
        if key in bench:
            print(f"[skip ] bench {key}", flush=True)
            continue
        wall = run_rung(f"bench_{key}", 16, 3, bcs, nranks, 4, bind=bind)
        if wall > 0.0:
            bench[key] = wall / 4.0  # crude s/iteration incl. setup share
            bench_file.write_text(json.dumps(bench, indent=1))
    if bench:
        best = min(bench, key=bench.get)
        print(f"[bench] s/it: " + ", ".join(
            f"{k}={v:.1f}" for k, v in sorted(bench.items(),
                                              key=lambda kv: kv[1]))
            + f"  -> best {best}", flush=True)
        b_bcs = int(best.split("_")[0][3:])
        b_np = int(best.split("_")[1][2:])
        b_bind = best.split("_")[2]
    else:
        b_bcs, b_np, b_bind = 2, 64, "core"

    # ---- C. physics pushes, longest last.
    print("===== PHASE C: physics (L16 -> L24 -> L32) =====", flush=True)
    for L in (16, 24, 32):
        run_rung(f"L{L}_g3", L, 3, b_bcs, b_np, MAX_ITER, bind=b_bind)
    print("===== ALL PHASES COMPLETE =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
