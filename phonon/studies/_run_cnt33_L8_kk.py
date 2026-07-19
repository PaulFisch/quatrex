"""Longer CNT (3,3) L8 at eta = 0, g_band = 2: does a SUPPORT-COMPLETE
Kramers-Kronig integral + contact-dressing stabilise the run that
diverged before?

CNT (3,3) L8 with g_band = 2 DIVERGED at iteration 63 on the old code
(cnt33_L8_linear/run_gband2.npz: converged=False, diverged=True). That run
used fmax = 55 THz, but omega_max = 43.73 THz so 2*omega_max = 87.5 THz > 55:
the 3-phonon bubble's two-phonon weight on (55, 87.5] was TRUNCATED, and with
it the Kramers-Kronig integral for Re Sigma^R -- exactly the truncation that
drove the d5a eta = 0 runs unstable and that the support-complete KK
(nu_kk rung) removed. Here the dual grid supplies that support cheaply: the
primary (Dyson) grid stops at 55, the auxiliary bubble grid reaches
aux_fmax = 88 >= 2*omega_max, so Re Sigma^R is reconstructed from the full
convolution support at no extra Dyson solves.

Two levers, 2x2:
  KK support : truncated (aux off, fmax = 55) vs complete (aux_fmax = 88).
  contacts   : bare harmonic reservoirs (obc_scattering_contacts = False)
               vs dressed with the device boundary Sigma^R (= True, the
               GW-style self-consistent contact).

Rungs (all eta = 0, g_band = 2, uniform primary nf = 361 on [0, 55] so the
grid resolution is not a confound):
  trunc_bare : fmax 55, aux off,      obc_scattering = False  (reproduces
               the original divergence on current code -- the control).
  kk_bare    : aux_fmax 88,           obc_scattering = False.
  kk_dressed : aux_fmax 88,           obc_scattering = True.

(trunc + dressed is omitted: dressing cannot cure a truncated Re Sigma^R.)

Run (background, cluster):
    cd <repo>
    nohup python phonon/studies/_run_cnt33_L8_kk.py > \
        phonon/studies/out/cnt33_L8_kk/kk.log 2>&1 &
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "phonon/studies/out/anderson_test/cnt33_L8_inputs"
OUT = REPO / "phonon/studies/out/cnt33_L8_kk"
FMAX = 55.0
NFREQ = 361
AUX_DW = FMAX / (NFREQ - 1)               # 0.1528 THz
AUX_FMAX = 88.0                           # >= 2*omega_max (43.73) -- complete KK
NRANKS = 64
MAX_ITER = 600
NCELLS = 8
GBAND = 2
GEOM = ("dynamical_matrix.mat", "fc3_blocks.hdf5", "structure.xyz")

WC = REPO / "phonon/studies/engine/write_config.py"
RUN = REPO / "phonon/studies/engine/run.py"

# (tag, aux_on, obc_scattering)
RUNGS = [
    ("trunc_bare", False, False),
    ("kk_bare", True, False),
    ("kk_dressed", True, True),
]


def prep(d: Path, aux_on: bool) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for f in GEOM:
        dst = d / f
        if not dst.exists():
            dst.symlink_to(SRC / f)
    cmd = [sys.executable, str(WC), "--system", "cnt33", "--work", str(d),
           "-L", str(NCELLS), "--eta", "0", "--nfreq", str(NFREQ),
           "--fmax", str(FMAX), "--retarded", "fft", "--mix", "0.2",
           "--max-iter", str(MAX_ITER)]
    if aux_on:
        cmd += ["--aux-dw", str(AUX_DW), "--aux-fmax", str(AUX_FMAX)]
    subprocess.run(cmd, check=True)


def hygiene() -> None:
    r = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f",
                        "cnt33_L8_kk.*run.py"],
                       capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if p]
    for p in pids:
        subprocess.run(["kill", "-9", p], capture_output=True)
    if pids:
        print(f"[hygiene] killed {len(pids)} leftover ranks", flush=True)
        time.sleep(3)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, aux_on, obc in RUNGS:
        d = OUT / tag
        npz = d / "run.npz"
        if npz.exists():
            print(f"[skip ] {tag}: run.npz exists", flush=True)
            continue
        prep(d, aux_on)
        hygiene()
        env = dict(os.environ,
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                   QX_GBAND=str(GBAND),
                   QX_SCATCONTACTS=("1" if obc else "0"),
                   QX_CONFIG=str(d / "quatrex_config.toml"),
                   QX_NPZ=str(npz))
        t0 = time.time()
        print(f"[run  ] {tag} (aux_fmax={'88' if aux_on else 'off'}, "
              f"obc_scattering={obc}, {NRANKS} ranks, g_band={GBAND})",
              flush=True)
        with open(OUT / f"{tag}.log", "w") as log:
            rc = subprocess.run(
                ["mpirun", "--bind-to", "core", "--map-by", "core",
                 "-np", str(NRANKS), sys.executable, str(RUN)],
                env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        ok = npz.exists()
        print(f"[done ] {tag}: rc={rc} npz={'yes' if ok else 'MISSING'} "
              f"wall={dt / 60:.1f} min", flush=True)
    print("[done ] cnt33 L8 KK-support x contacts experiment complete.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
