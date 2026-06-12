"""Direct Caroli ballistic conductance/transmission curves (d5a, d11a, cnt33).

Pure harmonic NEGF: ``T(omega) = Tr(Gamma_L G^R Gamma_R G^A)`` via
Sancho-Rubio surface Green's functions, with NO third-order force constants
and NO bubble (unlike the SCBA transport sweep, whose ``--ballistic-only``
still assembles the FC3 vertex). The ballistic transmission is temperature
independent, so it is computed once per (wire, length) and integrated against
every temperature weight for free. Reuses the exact conductance formula of
``transmission_finite`` (phonon/solver/dense.py).

CLI::

    python -m phonon.studies ballistic run  [--wires ...] [--lengths ...]
    python -m phonon.studies ballistic plot [--npz PATH]
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# The solver/phonon_inputs packages use flat intra-repo imports
# (``from phonon_inputs...``, ``from solver...``), so the phonon/ directory
# must be importable in addition to the repo root.
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "phonon")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phonon.studies import style

WIRES = {
    "d5a": "phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml",
    "d11a": "phonon/configs/sinw/sinw100_d11a_vasp_sc4.yaml",
    "cnt33": "phonon/configs/cnt/cnt33_vasp.yaml",
}
LENGTHS = [1, 2, 4]
TEMPS = [200, 300, 400, 500, 600]
DELTA_T = 10.0
# fmax covers 2*omega_max (~138 THz) so the integral support is complete.
FREQ_RANGE = (0.01, 140.0, 701)
ETA_FACTOR = 0.3

LABELS = {"d5a": "d5a SiNW (63 DOF/cell)",
          "d11a": "d11a SiNW (135 DOF/cell)",
          "cnt33": "(3,3) CNT (36 DOF/cell)"}

OUT = Path(__file__).resolve().parent / "out"
DEFAULT_NPZ = OUT / "ballistic_curves.npz"
FIG_NAME = "ballistic_curves"


def run(argv) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phonon.studies ballistic run",
        description="Caroli ballistic transmission/conductance curves.")
    parser.add_argument("--wires", nargs="+", default=list(WIRES),
                        choices=list(WIRES),
                        help=f"wires to compute (default: {list(WIRES)})")
    parser.add_argument("--lengths", nargs="+", type=int, default=LENGTHS,
                        help=f"device lengths in cells (default {LENGTHS})")
    parser.add_argument("--temps", nargs="+", type=float, default=TEMPS,
                        help=f"temperatures in K (default {TEMPS})")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ,
                        help=f"results snapshot path (default {DEFAULT_NPZ})")
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")

    from phonon.finite_analysis.loader import load_system
    from phonon.phonon_inputs.constants import (
        CONVERSION_THZ2, HBAR_SI, THZ_TO_RAD)
    from phonon.phonon_inputs.convention import get_btd_blocks
    from phonon.solver.grids import bose_full_axis, build_frequency_grid
    from phonon.solver.leads import (
        build_device_hamiltonian, ballistic_transmission_z2)

    rows = []
    curves = {}          # (wire, n_slabs) -> T(omega)
    freqs_out = None
    for wire in args.wires:
        cfg = WIRES[wire]
        b = load_system(ROOT / cfg, validate=False, transport_axis=2)
        ph = b.phonon
        H_00, H_01 = get_btd_blocks(
            ph, (0.0, 0.0), transport_direction="z",
            conversion_factor=CONVERSION_THZ2)
        n_dof = H_00.shape[0]
        lattice = ph.primitive.cell
        a1, a2 = lattice[0], lattice[1]  # perp to transport (z)
        A_c = np.linalg.norm(np.cross(a1, a2)) * 1e-20

        freqs_thz, dw_thz, eta_w, z2_arr, pos_mask, mid = build_frequency_grid(
            FREQ_RANGE, eta_factor=ETA_FACTOR)
        omega_rad = freqs_thz * THZ_TO_RAD
        freqs_out = freqs_thz

        for n_slabs in args.lengths:
            t0 = time.time()
            H_D = build_device_hamiltonian(H_00, H_01, n_slabs)
            N_D = n_slabs * n_dof
            H_LD = np.zeros((n_dof, N_D), dtype=complex)
            H_LD[:, :n_dof] = H_01
            H_DR = np.zeros((N_D, n_dof), dtype=complex)
            H_DR[-n_dof:, :] = H_01
            trans = np.zeros(len(freqs_thz))
            for iw, z2 in enumerate(z2_arr):
                trans[iw] = ballistic_transmission_z2(
                    z2, H_D, H_00, H_01, H_LD, H_DR)
            wall = time.time() - t0
            curves[(wire, n_slabs)] = trans
            for T in args.temps:
                n_L = bose_full_axis(freqs_thz, T + DELTA_T / 2)
                n_R = bose_full_axis(freqs_thz, T - DELTA_T / 2)
                spec = HBAR_SI * omega_rad * (n_L - n_R) * trans
                J = np.sum(spec[pos_mask]) * dw_thz * 1e12
                G = J / (A_c * DELTA_T)
                rows.append(dict(
                    wire=wire, n_slabs=n_slabs, T=T, delta_T=DELTA_T,
                    G_ball=G, maxT=float(trans.max()), n_dof=n_dof))
            g300 = next((r["G_ball"] for r in rows
                         if r["wire"] == wire and r["n_slabs"] == n_slabs
                         and r["T"] == 300), None)
            g300_s = f"G_ball(300)={g300:.3e}, " if g300 is not None else ""
            print(f"[{wire}] L={n_slabs}: maxT={trans.max():.3f}, "
                  f"{g300_s}{wall:.1f}s", flush=True)

    args.npz.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "freqs_thz": freqs_out,
        "wire": np.array([r["wire"] for r in rows]),
        "n_slabs": np.array([r["n_slabs"] for r in rows]),
        "T": np.array([r["T"] for r in rows]),
        "delta_T": np.array([r["delta_T"] for r in rows]),
        "G_ball": np.array([r["G_ball"] for r in rows]),
        "maxT": np.array([r["maxT"] for r in rows]),
        "n_dof": np.array([r["n_dof"] for r in rows]),
    }
    for (wire, n_slabs), trans in curves.items():
        payload[f"trans_{wire}_L{n_slabs}"] = trans
    np.savez(args.npz, **payload)
    print(f"[done] wrote {args.npz} ({len(rows)} rows)", flush=True)
    _figure(args.npz)
    return 0


def plot(argv) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phonon.studies ballistic plot",
        description="Re-draw the ballistic curves figure from the npz.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ,
                        help=f"results snapshot (default {DEFAULT_NPZ})")
    args = parser.parse_args(argv)
    _figure(args.npz)
    return 0


def _figure(npz_path: Path) -> None:
    """Transmission T(omega) at L=1 + G_ball vs T (L=1) + G_ball vs L (300 K)."""
    data = np.load(npz_path, allow_pickle=True)
    wire = data["wire"].astype(str)
    n_slabs, T = data["n_slabs"], data["T"]
    G_ball = data["G_ball"]
    wires = [w for w in WIRES if w in set(wire)]

    fig, axes = style.figure(ncols=3, width=4.2, height=3.4)

    ax = axes[0]
    for w in wires:
        key = f"trans_{w}_L1"
        if key in data.files:
            ax.plot(data["freqs_thz"], data[key], lw=1.0,
                    label=LABELS.get(w, w))
    ax.set_xlabel("frequency (THz)")
    ax.set_ylabel(r"ballistic transmission $T(\omega)$")
    ax.set_title("transmission (L=1)", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[1]
    for w in wires:
        sel = (wire == w) & (n_slabs == 1)
        order = np.argsort(T[sel])
        ax.plot(T[sel][order], G_ball[sel][order] / 1e6, marker="o",
                label=LABELS.get(w, w))
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"$G_\mathrm{ball}\ (\mathrm{MW\,m^{-2}\,K^{-1}})$")
    ax.set_title("conductance vs T (L=1)", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[2]
    for w in wires:
        sel = (wire == w) & (T == 300)
        order = np.argsort(n_slabs[sel])
        ax.plot(n_slabs[sel][order], G_ball[sel][order] / 1e6, marker="o",
                label=LABELS.get(w, w))
    ax.set_xlabel("device length (transport cells)")
    ax.set_ylabel(r"$G_\mathrm{ball}\ (\mathrm{MW\,m^{-2}\,K^{-1}})$")
    ax.set_title("conductance vs length (300 K)", fontsize=10)
    ax.legend(fontsize=7)

    style.save(fig, FIG_NAME)
