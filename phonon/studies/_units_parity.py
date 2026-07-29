"""Units-bridge parity: production engine (ballistic) vs exact mode count.

Same device (cnt33 L4, dynamical_matrix.mat as consumed by the engine):

  A. reference -- Landauer G from the lead Bloch mode count M(nu)
     (assumption-free for a pristine wire; no NEGF, no surface GF),
     integrated with the dense-stack convention
     J = sum(hbar omega_rad (n_L - n_R) M) * dnu * 1e12, G = J/(A dT).
  B. engine -- QX_BALLISTIC=1 single-rank run, then
     phonon.postproc.units.run_npz_conductance on run.npz.

Validated 2026-07-29: engine T_eff = spectrum/(n_L - n_R) reproduces
M(nu) on 361/361 bins exactly; G ratio 1.00000 (797.0 MW/m^2/K).

The retired Sancho-Rubio reference (dense_reference below) is kept as
a secondary check only: its eta_factor lead seed smears the stepped
transmission and reads ~10% LOW on cnt33 (T peak 10.5 vs the true
11-channel step at 40 THz). The same caveat applies to the stored
ballistic.py d5a/d11a curves at ETA_FACTOR = 0.3.

Run locally:  python phonon/studies/_units_parity.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "phonon")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SRC = ROOT / "phonon/studies/out/anderson_test/cnt33_L4_inputs"
WORK = ROOT / "phonon/studies/out/units_parity/cnt33_L4"
GEOM = ("dynamical_matrix.mat", "fc3_blocks.hdf5", "phonon_energies.npy",
        "structure.xyz")
L = 4
NFREQ, FMAX = 361, 55.0
T0, DT = 300.0, 10.0


def load_mat_blocks():
    from scipy.io import loadmat

    d = loadmat(str(SRC / "dynamical_matrix.mat"))
    keys = [k for k in d if k.startswith("[")]
    h00 = d["[0, 0, 0]"]
    h01 = d["[0, 0, 1]"]
    print(f"[mat  ] keys: {sorted(keys)}; block {h00.shape}", flush=True)
    return np.asarray(h00, dtype=complex), np.asarray(h01, dtype=complex)


def lattice_from_xyz():
    with open(SRC / "structure.xyz") as f:
        f.readline()
        header = f.readline()
    lat = header.split('Lattice="')[1].split('"')[0].split()
    return np.array([float(x) for x in lat]).reshape(3, 3)


def dense_reference() -> float:
    from phonon.phonon_inputs.constants import (
        CONVERSION_THZ2, HBAR_SI, THZ_TO_RAD)
    from phonon.solver.grids import bose_full_axis, build_frequency_grid
    from phonon.solver.leads import (
        ballistic_transmission_z2, build_device_hamiltonian)

    h00, h01 = load_mat_blocks()
    # .mat blocks are (rad/s)^2 (writer docstring); solver works in THz^2.
    scale = np.abs(h00).max()
    if scale > 1e10:  # (rad/s)^2 magnitudes ~ 1e27
        h00 = h00 * CONVERSION_THZ2
        h01 = h01 * CONVERSION_THZ2
        print(f"[mat  ] converted (rad/s)^2 -> THz^2 "
              f"(max |H00| now {np.abs(h00).max():.1f})", flush=True)

    n_dof = h00.shape[0]
    H_D = build_device_hamiltonian(h00, h01, L)
    N_D = L * n_dof
    H_LD = np.zeros((n_dof, N_D), dtype=complex)
    H_LD[:, :n_dof] = h01
    H_DR = np.zeros((N_D, n_dof), dtype=complex)
    H_DR[-n_dof:, :] = h01

    freqs_thz, dw_thz, eta_w, z2_arr, pos_mask, mid = build_frequency_grid(
        (0.01, FMAX, NFREQ), eta_factor=0.3)
    trans = np.array([
        ballistic_transmission_z2(z2, H_D, h00, h01, H_LD, H_DR)
        for z2 in z2_arr])
    omega_rad = freqs_thz * THZ_TO_RAD
    n_L = bose_full_axis(freqs_thz, T0 + DT / 2)
    n_R = bose_full_axis(freqs_thz, T0 - DT / 2)
    spec = HBAR_SI * omega_rad * (n_L - n_R) * trans
    J = float(np.sum(spec[pos_mask]) * dw_thz * 1e12)
    lat = lattice_from_xyz()
    A_c = float(np.linalg.norm(np.cross(lat[0], lat[1])) * 1e-20)
    G = J / (A_c * DT)
    print(f"[dense] T(omega) peak {trans.max():.2f}; J = {J:.4e} W; "
          f"A_c = {A_c:.3e} m^2; G = {G / 1e6:.3f} MW/m^2/K", flush=True)
    return G


def engine_ballistic() -> float:
    WORK.mkdir(parents=True, exist_ok=True)
    for f in GEOM:
        dst = WORK / f
        if not dst.exists():
            dst.symlink_to(SRC / f)
    subprocess.run(
        [sys.executable, str(ROOT / "phonon/studies/engine/write_config.py"),
         "--system", "cnt33", "--work", str(WORK), "-L", str(L),
         "--eta", "0", "--temperature", str(T0), "--dt", str(DT),
         "--nfreq", str(NFREQ), "--fmax", str(FMAX), "--retarded", "fft",
         "--mix", "0.2", "--max-iter", "3"],
        check=True)
    npz = WORK / "run.npz"
    env = dict(os.environ,
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               QX_BALLISTIC="1",
               QX_CONFIG=str(WORK / "quatrex_config.toml"),
               QX_NPZ=str(npz))
    t0 = time.time()
    with open(WORK / "engine.log", "w") as log:
        rc = subprocess.run(
            [sys.executable, str(ROOT / "phonon/studies/engine/run.py")],
            env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"[engin] rc={rc.returncode} in {time.time() - t0:.0f} s "
          f"(log: {WORK / 'engine.log'})", flush=True)
    if rc.returncode != 0:
        raise RuntimeError("engine ballistic run failed")

    from phonon.postproc.units import cross_section_area_m2, \
        run_npz_conductance
    lat = lattice_from_xyz()
    res = run_npz_conductance(npz, area_m2=cross_section_area_m2(lat, "z"))
    print(f"[engin] J_lead = {res['J_lead_watts']:.4e} W; "
          f"G = {res['G_Wm2K'] / 1e6:.3f} MW/m^2/K", flush=True)
    return float(res["G_Wm2K"])


def bloch_mode_reference(npz_path) -> float:
    """Exact ballistic reference: Landauer G from the lead Bloch mode
    count M(nu) (no NEGF, no surface GF -- assumption-free for a pristine
    wire), integrated with the identical unit convention. The engine's
    T_eff = spectrum/(n_L - n_R) matched M(nu) on 361/361 bins (2026-07-29).
    """
    import numpy as np
    from phonon.phonon_inputs.constants import HBAR_SI, THZ_TO_RAD
    from phonon.solver.grids import bose_full_axis

    h00, h01 = load_mat_blocks()      # nu^2 THz^2 as written by the exporter
    nk = 4001
    ks = np.linspace(0, np.pi, nk)
    bands = np.zeros((nk, h00.shape[0]))
    for i, k in enumerate(ks):
        Hk = h00 + h01 * np.exp(1j * k) + h01.conj().T * np.exp(-1j * k)
        bands[i] = np.sqrt(np.clip(np.linalg.eigvalsh(Hk), 0, None))

    d = np.load(npz_path)
    en = np.asarray(d["energies"], float)
    M = np.zeros(len(en))
    for n in range(bands.shape[1]):
        s = np.signbit(bands[:, n][None, :] - en[:, None])
        M += (s[:, 1:] != s[:, :-1]).sum(axis=1)
    nL = bose_full_axis(en, T0 + DT / 2)
    nR = bose_full_axis(en, T0 - DT / 2)
    w = np.asarray(d["frequency_cell_widths"], float)
    J = HBAR_SI * THZ_TO_RAD * 1e12 * np.sum(w * np.abs(en) * (nL - nR) * M)
    lat = lattice_from_xyz()
    A_c = float(np.linalg.norm(np.cross(lat[0], lat[1])) * 1e-20)
    G = J / (A_c * DT)
    print(f"[bloch] omega_max {bands.max():.2f} THz; "
          f"G_ref = {G / 1e6:.2f} MW/m^2/K", flush=True)
    return G

def main() -> int:
    G_engine = engine_ballistic()
    G_ref = bloch_mode_reference(WORK / "run.npz")
    ratio = G_engine / G_ref
    print(f"[parity] G_engine / G_ref(mode count) = {ratio:.5f}", flush=True)
    ok = 0.99 < ratio < 1.01
    print(f"[{'PASS' if ok else 'FAIL'}] units bridge "
          f"{'validated' if ok else 'INCONSISTENT -- do not quote kappa'}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
