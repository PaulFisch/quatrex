"""Compare finite-displacement and DFPT force constants in anharmonic NEGF.

Runs the same transport calculation twice:
  1. finite-displacement FC2+FC3 from phono3py
  2. DFPT FC2+FC3 from the DFPT pipeline

Then overlays ballistic and anharmonic observables.

Assumes:
  - finite-displacement reference is available under fc3_prim/
  - DFPT reference is available under dfpt/fc3.hdf5
"""

import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent  # input_calc/
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell, load_primitive_cell_dfpt
from phonon_inputs.anharmonic import anharmonic_transmission
from phonon_inputs.constants import HBAR_SI


def _run_case(label, phonon, fc3_data):
    print("\n" + "=" * 72)
    print(f"Running case: {label}")
    print("=" * 72)

    t0 = time.time()
    result = anharmonic_transmission(
        phonon,
        fc3_data,
        q_mesh_transverse=(10, 10),
        freq_range_thz=(0.0, 15.0, 51),
        transport_direction="x",
        eta_factor=0.5,
        temperature=300.0,
        max_scba_iter=100,
        scba_tol=0.005,
        mixing=0.3,
        fc3_mode="full",
        verbose=True,
    )
    t1 = time.time()

    print(f"{label} completed in {t1 - t0:.1f} s")
    print(f"  Ballistic max T:    {result['transmission_ballistic'].max():.4f}")
    print(f"  G_ballistic:        {result['thermal_conductance_ballistic']/1e6:.2f} MW/(m^2 K)")
    print(f"  G_anharmonic:       {result['thermal_conductance_anharmonic']/1e6:.2f} MW/(m^2 K)")
    print(f"  Heat conserv.:      {result['heat_flow_conservation']:.2e}")

    return result


def _safe_ratio(num, den, rel_cut=1e-2, abs_eps=1e-30):
    out = np.full_like(num, np.nan, dtype=float)
    scale = np.max(np.abs(den))
    mask = np.abs(den) > max(abs_eps, rel_cut * scale)
    out[mask] = num[mask] / den[mask]
    return out


if __name__ == "__main__":
    print("=" * 72)
    print("Loading finite-displacement reference...")
    phonon_fd, fc3_fd = load_primitive_cell(work_dir)

    print("\n" + "=" * 72)
    print("Loading DFPT reference...")
    phonon_dfpt, fc3_dfpt = load_primitive_cell_dfpt(work_dir)

    # Run both transport calculations
    result_fd = _run_case("finite-displacement", phonon_fd, fc3_fd)
    result_dfpt = _run_case("DFPT", phonon_dfpt, fc3_dfpt)

    freqs_fd = result_fd["freqs_thz"]
    freqs_dfpt = result_dfpt["freqs_thz"]

    if not np.allclose(freqs_fd, freqs_dfpt):
        raise ValueError("Frequency grids differ between FD and DFPT runs.")

    freqs = freqs_fd

    # ------------------------------------------------------------------
    # Summary table in stdout
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"{'Case':20s} {'G_ball [MW/m^2K]':>18s} {'G_anh [MW/m^2K]':>18s} {'Reduction [%]':>14s}")
    for label, res in [("finite-displacement", result_fd), ("DFPT", result_dfpt)]:
        g_ball = res["thermal_conductance_ballistic"] / 1e6
        g_anh = res["thermal_conductance_anharmonic"] / 1e6
        reduction = 100 * (1 - g_anh / g_ball) if g_ball > 0 else np.nan
        print(f"{label:20s} {g_ball:18.4f} {g_anh:18.4f} {reduction:14.2f}")


    # ------------------------------------------------------------------
    # Derived effective anharmonic transmission
    # ------------------------------------------------------------------
    def _effective_transmission(result, rel_cut=1e-2, abs_eps=1e-30):
        T_ball = result["transmission_ballistic"]
        J_ball = result["spectral_heat_current_ballistic"]
        J_anh = result["spectral_heat_current"]

        T_eff = np.full_like(T_ball, np.nan, dtype=float)

        scale = np.max(np.abs(J_ball))
        mask = np.abs(J_ball) > max(abs_eps, rel_cut * scale)

        T_eff[mask] = T_ball[mask] * (J_anh[mask] / J_ball[mask])
        return T_eff

    T_eff_fd = _effective_transmission(result_fd)
    T_eff_dfpt = _effective_transmission(result_dfpt)

    # ------------------------------------------------------------------
    # Plot 1: transmission
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs, result_fd["transmission_ballistic"], "-", lw=1.8,
            label="FD ballistic")
    ax.plot(freqs, T_eff_fd, "--", lw=1.8,
            label="FD anharmonic (effective)")
    ax.plot(freqs, result_dfpt["transmission_ballistic"], "-", lw=1.8,
            label="DFPT ballistic")
    ax.plot(freqs, T_eff_dfpt, "--", lw=1.8,
            label="DFPT anharmonic (effective)")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Transmission")
    ax.set_title("Ballistic and effective anharmonic transmission")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 16)
    fig.tight_layout()
    fig.savefig(script_dir / "compare_transport_transmission.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: spectral heat current
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs, result_fd["spectral_heat_current_ballistic"], "-", lw=1.8,
            label="FD ballistic")
    ax.plot(freqs, result_fd["spectral_heat_current"], "--", lw=1.8,
            label="FD anharmonic")
    ax.plot(freqs, result_dfpt["spectral_heat_current_ballistic"], "-", lw=1.8,
            label="DFPT ballistic")
    ax.plot(freqs, result_dfpt["spectral_heat_current"], "--", lw=1.8,
            label="DFPT anharmonic")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Spectral heat current (W)")
    ax.set_title("Spectral heat current: finite-displacement vs DFPT")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 16)
    fig.tight_layout()
    fig.savefig(script_dir / "compare_transport_spectral_current.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 3: anharmonic / ballistic ratio
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ratio_fd = _safe_ratio(
        result_fd["spectral_heat_current"],
        result_fd["spectral_heat_current_ballistic"],
    )
    ratio_dfpt = _safe_ratio(
        result_dfpt["spectral_heat_current"],
        result_dfpt["spectral_heat_current_ballistic"],
    )
    ax.plot(freqs, ratio_fd, lw=1.8, label="FD: J_anh / J_ball")
    ax.plot(freqs, ratio_dfpt, lw=1.8, label="DFPT: J_anh / J_ball")
    ax.axhline(1.0, ls="--", lw=1.0, alpha=0.6)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Anharmonic / ballistic")
    ax.set_title("Relative anharmonic suppression")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 16)
    fig.tight_layout()
    fig.savefig(script_dir / "compare_transport_ratio.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 4: integrated conductance
    # ------------------------------------------------------------------
    labels = ["FD", "DFPT"]
    g_ball = np.array([
        result_fd["thermal_conductance_ballistic"],
        result_dfpt["thermal_conductance_ballistic"],
    ]) / 1e6
    g_anh = np.array([
        result_fd["thermal_conductance_anharmonic"],
        result_dfpt["thermal_conductance_anharmonic"],
    ]) / 1e6

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, g_ball, width, label="Ballistic")
    ax.bar(x + width / 2, g_anh, width, label="Anharmonic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Thermal conductance (MW / m$^2$ K)")
    ax.set_title("Integrated conductance comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(script_dir / "compare_transport_conductance.png", dpi=160)
    plt.close(fig)