"""Test multi-slab anharmonic phonon-phonon NEGF.

Validates the N-slab extension and runs a thickness sweep to
reproduce the thickness-dependent thermal conductance from
Guo et al., PRB 102, 195412 (2020), Table III / Figure 9.

Supports checkpoint/restart: each completed slab count is saved
immediately.  On restart the script loads existing results and
skips already-completed cases.

Usage:
    python test_anharmonic_multislab.py            # run / resume
    python test_anharmonic_multislab.py --plot      # replot from saved data
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

work_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(work_dir.parents[1] / "src"))

CHECKPOINT_FILE = work_dir / "anharmonic_multislab_checkpoint.json"
DATA_FILE = work_dir / "anharmonic_multislab_data.npz"
PLOT_FILE = work_dir / "anharmonic_multislab.png"

# Guo Table III values (5 uc Si film, 300K)
GUO_TABLE3 = {
    (4, 5): {"G_ball": 1090.59, "G_anh": 928.05},
    (6, 5): {"G_ball": 1040.03, "G_anh": 890.97},
    (8, 5): {"G_ball": 1053.53, "G_anh": 872.05},
}

SLAB_COUNTS = [1, 3, 5, 10]

# SCBA parameters — conservative mixing for stability at large n_slabs;
# adaptive damping in the SCBA loop reduces further if needed.
COMMON = dict(
    q_mesh_transverse=(4, 4),
    freq_range_thz=(0.5, 15.0, 51),
    transport_direction="z",
    eta_factor=0.5,
    temperature=300.0,
    max_scba_iter=40,
    scba_tol=0.005,
    mixing=0.15,
    fc3_mode="full",
)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(results: dict, a_conv: float):
    """Write completed slab results to both JSON (checkpoint) and npz (data)."""
    # JSON checkpoint: lightweight, for restart logic
    ckpt = {}
    for ns, r in results.items():
        ckpt[str(ns)] = {
            "G_ball": float(r["thermal_conductance_ballistic"]),
            "G_anh": float(r["thermal_conductance_anharmonic"]),
            "n_scba_iterations": int(r["n_scba_iterations"]),
            "max_J_ball": float(np.max(r["spectral_heat_current_ballistic"])),
            "max_J_anh": float(np.max(r["spectral_heat_current"])),
        }
    CHECKPOINT_FILE.write_text(json.dumps(ckpt, indent=2))

    # Full npz data: everything needed for replotting
    save_dict = {
        "slab_counts": np.array(sorted(results.keys())),
        "a_conv": a_conv,
        "freqs_thz": next(iter(results.values()))["freqs_thz"],
    }
    for ns, r in results.items():
        save_dict[f"J_ball_{ns}"] = r["spectral_heat_current_ballistic"]
        save_dict[f"J_anh_{ns}"] = r["spectral_heat_current"]
        save_dict[f"G_ball_{ns}"] = float(r["thermal_conductance_ballistic"])
        save_dict[f"G_anh_{ns}"] = float(r["thermal_conductance_anharmonic"])
        save_dict[f"n_scba_iter_{ns}"] = int(r["n_scba_iterations"])
        if "convergence_history" in r:
            save_dict[f"conv_hist_{ns}"] = np.array(r["convergence_history"])
    np.savez(DATA_FILE, **save_dict)


def _load_checkpoint() -> dict:
    """Load checkpoint and return dict of {n_slabs: result_dict} for
    completed cases.  Returns {} if no checkpoint exists."""
    if not DATA_FILE.exists():
        return {}
    if not CHECKPOINT_FILE.exists():
        return {}

    try:
        ckpt = json.loads(CHECKPOINT_FILE.read_text())
        data = np.load(DATA_FILE, allow_pickle=False)
        results = {}
        for ns_str in ckpt:
            ns = int(ns_str)
            key_ball = f"J_ball_{ns}"
            key_anh = f"J_anh_{ns}"
            if key_ball not in data or key_anh not in data:
                continue
            results[ns] = {
                "freqs_thz": data["freqs_thz"],
                "spectral_heat_current_ballistic": data[key_ball],
                "spectral_heat_current": data[key_anh],
                "thermal_conductance_ballistic": float(data[f"G_ball_{ns}"]),
                "thermal_conductance_anharmonic": float(data[f"G_anh_{ns}"]),
                "n_scba_iterations": int(data[f"n_scba_iter_{ns}"]),
                "convergence_history": (
                    list(data[f"conv_hist_{ns}"])
                    if f"conv_hist_{ns}" in data else []
                ),
            }
        return results
    except Exception as e:
        print(f"  Warning: could not load checkpoint ({e}), starting fresh")
        return {}


# ---------------------------------------------------------------------------
# Plotting (works from saved data alone)
# ---------------------------------------------------------------------------

def make_plot(results: dict, a_conv: float):
    """Generate the 3-panel multislab comparison figure."""
    completed = sorted(results.keys())
    if not completed:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    freqs = results[completed[0]]["freqs_thz"]
    colors = {1: "C0", 3: "C1", 5: "C2", 10: "C3", 15: "C4", 20: "C5"}

    # (a) Spectral heat current
    ax = axes[0]
    ax.plot(freqs, results[completed[0]]["spectral_heat_current_ballistic"],
            "k--", lw=1, alpha=0.5, label="Ballistic")
    for ns in completed:
        ax.plot(freqs, results[ns]["spectral_heat_current"],
                color=colors.get(ns, "C6"), lw=1.5, label=f"{ns} slab(s)")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Spectral heat current (W)")
    ax.set_title("(a) Si spectral heat current at 300K (4x4 q-mesh)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 16)

    # (b) Thermal conductance reduction vs thickness
    ax = axes[1]
    thicknesses_nm = [ns * a_conv / 10.0 for ns in completed]
    reductions = []
    for ns in completed:
        G_b = results[ns]["thermal_conductance_ballistic"]
        G_a = results[ns]["thermal_conductance_anharmonic"]
        reductions.append((1 - G_a / G_b) * 100 if G_b > 0 else 0)
    ax.plot(thicknesses_nm, reductions, "bo-", ms=8, lw=1.5,
            label="This work (4x4)")
    guo = GUO_TABLE3[(4, 5)]
    red_guo = (1 - guo["G_anh"] / guo["G_ball"]) * 100
    ax.plot([5 * a_conv / 10.0], [red_guo], "rs", ms=10, label="Guo (4x4)")
    ax.set_xlabel("Device thickness (nm)")
    ax.set_ylabel("Thermal conductance reduction (%)")
    ax.set_title("(b) Scattering vs thickness")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Effective thermal conductivity vs thickness
    ax = axes[2]
    kappas = []
    for ns in completed:
        G_a = results[ns]["thermal_conductance_anharmonic"]
        kappas.append(G_a * ns * a_conv * 1e-10)
    ax.plot(thicknesses_nm, kappas, "bo-", ms=8, lw=1.5,
            label="This work (4x4)")
    kappa_guo = guo["G_anh"] * 1e6 * 5 * a_conv * 1e-10
    ax.plot([5 * a_conv / 10.0], [kappa_guo], "rs", ms=10, label="Guo (4x4)")
    ax.axhline(y=148, color="gray", ls="--", alpha=0.5, label="Bulk Si (exp)")
    ax.set_xlabel("Device thickness (nm)")
    ax.set_ylabel("Effective kappa (W/m K)")
    ax.set_title("(c) Effective thermal conductivity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)
    plt.close("all")
    print(f"Saved {PLOT_FILE.name}")


def print_summary(results: dict, a_conv: float):
    """Print the summary table."""
    completed = sorted(results.keys())
    if not completed:
        return

    print(f"\n{'n_slabs':>8} {'L [nm]':>8} {'G_ball':>10} {'G_anh':>10} "
          f"{'Red. [%]':>9} {'kappa [W/mK]':>13} {'Iters':>6}")
    print("-" * 68)
    for ns in completed:
        r = results[ns]
        G_b = r["thermal_conductance_ballistic"]
        G_a = r["thermal_conductance_anharmonic"]
        red = (1 - G_a / G_b) * 100 if G_b > 0 else 0
        L_nm = ns * a_conv / 10.0
        kappa = G_a * ns * a_conv * 1e-10
        print(f"{ns:>8d} {L_nm:>8.2f} {G_b/1e6:>10.1f} {G_a/1e6:>10.1f} "
              f"{red:>9.1f} {kappa:>13.2f} {r['n_scba_iterations']:>6d}")

    guo = GUO_TABLE3[(4, 5)]
    red_guo = (1 - guo["G_anh"] / guo["G_ball"]) * 100
    kappa_guo = guo["G_anh"] * 1e6 * 5 * a_conv * 1e-10
    print(f"\nGuo et al. Table III (5 uc, 4x4 q-mesh):")
    print(f"  G_ball = {guo['G_ball']:.2f}, G_anh = {guo['G_anh']:.2f} MW/(m^2 K)")
    print(f"  Reduction: {red_guo:.1f}%, kappa = {kappa_guo:.2f} W/(m K)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true",
                        help="Only replot from saved data, no computation")
    args = parser.parse_args()

    # Plot-only mode
    if args.plot:
        if not DATA_FILE.exists():
            print(f"No data file found at {DATA_FILE}")
            sys.exit(1)
        data = np.load(DATA_FILE, allow_pickle=False)
        a_conv = float(data["a_conv"])
        results = {}
        for ns in data["slab_counts"]:
            ns = int(ns)
            results[ns] = {
                "freqs_thz": data["freqs_thz"],
                "spectral_heat_current_ballistic": data[f"J_ball_{ns}"],
                "spectral_heat_current": data[f"J_anh_{ns}"],
                "thermal_conductance_ballistic": float(data[f"G_ball_{ns}"]),
                "thermal_conductance_anharmonic": float(data[f"G_anh_{ns}"]),
                "n_scba_iterations": int(data[f"n_scba_iter_{ns}"]),
            }
        print_summary(results, a_conv)
        make_plot(results, a_conv)
        sys.exit(0)

    # --- Computation mode ---
    from phonon_inputs.structure import load_phonopy_calculation
    from phonon_inputs.force_constants import load_fc3_thirdorder
    from phonon_inputs.anharmonic import anharmonic_transmission
    from test_anharmonic import remap_fc3_to_conventional

    print("=" * 60)
    print("Loading Si phonopy calculation...")
    phonon_si = load_phonopy_calculation(
        phonopy_yaml=work_dir / "scf_disp" / "phonopy_disp.yaml",
        force_sets_filename=work_dir / "scf_disp" / "FORCE_SETS",
        calculator="qe",
    )
    a_conv = phonon_si.unitcell.cell[0, 0]

    print("Loading and remapping FC3...")
    fc3_data = load_fc3_thirdorder(work_dir / "fc3_si" / "FORCE_CONSTANTS_3RD")
    fcc_cell = np.array([
        [0.0, a_conv / 2, a_conv / 2],
        [a_conv / 2, 0.0, a_conv / 2],
        [a_conv / 2, a_conv / 2, 0.0],
    ])
    fcc_frac = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    fc3_conv = remap_fc3_to_conventional(
        fc3_data, fcc_cell, fcc_frac,
        phonon_si.unitcell.cell, phonon_si.unitcell.scaled_positions,
    )

    # Load checkpoint — skip already-completed slab counts
    results = _load_checkpoint()
    if results:
        done = sorted(results.keys())
        print(f"\nResuming from checkpoint: n_slabs={done} already completed")
    else:
        print("\nNo checkpoint found, starting fresh")

    # -----------------------------------------------------------------------
    # Ballistic sanity check (quick, always re-run)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Ballistic sanity check (1 vs 3 vs 5 slabs)")
    print("=" * 60)
    for ns in [1, 3, 5]:
        ballistic_params = {**COMMON, "max_scba_iter": 0}
        res = anharmonic_transmission(
            phonon_si, fc3_conv, **ballistic_params,
            n_slabs=ns, verbose=False,
        )
        print(f"  n_slabs={ns}: max T_ballistic = "
              f"{res['transmission_ballistic'].max():.4f}, "
              f"G_ballistic = "
              f"{res['thermal_conductance_ballistic']/1e6:.2f} MW/(m^2 K)")

    # -----------------------------------------------------------------------
    # Thickness sweep with SCBA — skip completed, save after each
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Thickness sweep: n_slabs = {SLAB_COUNTS}")
    print("=" * 60)

    for ns in SLAB_COUNTS:
        if ns in results:
            G_b = results[ns]["thermal_conductance_ballistic"]
            G_a = results[ns]["thermal_conductance_anharmonic"]
            red = (1 - G_a / G_b) * 100 if G_b > 0 else 0
            print(f"\n--- n_slabs = {ns}: LOADED from checkpoint "
                  f"(G_anh = {G_a/1e6:.2f} MW/(m^2 K), "
                  f"red = {red:.1f}%) ---")
            continue

        print(f"\n--- n_slabs = {ns} (device = {ns * a_conv:.1f} A) ---")
        t0 = time.time()
        res = anharmonic_transmission(
            phonon_si, fc3_conv, **COMMON,
            n_slabs=ns, verbose=True,
        )
        t1 = time.time()
        results[ns] = res

        G_b = res["thermal_conductance_ballistic"]
        G_a = res["thermal_conductance_anharmonic"]
        red = (1 - G_a / G_b) * 100 if G_b > 0 else 0
        L = ns * a_conv * 1e-10
        kappa = G_a * L

        print(f"  Time: {t1-t0:.1f} s")
        print(f"  G_ball = {G_b/1e6:.2f}, G_anh = {G_a/1e6:.2f} MW/(m^2 K)")
        print(f"  Reduction: {red:.1f}%")
        print(f"  kappa_eff = G * L = {kappa:.2f} W/(m K)")
        print(f"  SCBA iters: {res['n_scba_iterations']}")

        # Save checkpoint after each completed slab count
        _save_checkpoint(results, a_conv)
        print(f"  >> Checkpoint saved ({len(results)}/{len(SLAB_COUNTS)} done)")

    # -----------------------------------------------------------------------
    # Summary and plot
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print_summary(results, a_conv)
    make_plot(results, a_conv)
