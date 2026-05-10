"""Test multi-slab anharmonic phonon-phonon NEGF.

Validates the N-slab extension and runs a thickness sweep to
reproduce the thickness-dependent thermal conductance from
Guo et al., PRB 102, 195412 (2020), Table III / Figure 9.

Uses the 2-atom FCC primitive cell with FC3 from phono3py + symfc.
Transport along a1 = [011]. Requires fc3_prim/fc3.hdf5.

Supports checkpoint/restart: each completed slab count is saved
immediately.  On restart the script loads existing results and
skips already-completed cases.

Usage:
    python test_anharmonic_multislab.py              # run / resume
    python test_anharmonic_multislab.py --plot        # replot from saved data
    python test_anharmonic_multislab.py --hilbert     # use Hilbert-transform Sigma^R
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

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent  # phonon/
sys.path.insert(0, str(work_dir))

CHECKPOINT_FILE = script_dir / "anharmonic_multislab_checkpoint.json"
DATA_FILE = script_dir / "anharmonic_multislab_data.npz"
PLOT_FILE = script_dir / "anharmonic_multislab.png"

# Guo Table III values (5 uc Si film, 300K) — for reference
GUO_TABLE3 = {
    (4, 5): {"G_ball": 1090.59, "G_anh": 928.05},
    (6, 5): {"G_ball": 1040.03, "G_anh": 890.97},
    (8, 5): {"G_ball": 1053.53, "G_anh": 872.05},
}

SLAB_COUNTS = [1, 3, 5, 10]

COMMON = dict(
    q_mesh_transverse=(4, 4),
    freq_range_thz=(0.0, 15.0, 51),  # (_, fmax, nfreq_pos); symmetric grid 0..fmax
    transport_direction="x",
    eta_factor=0.5,
    temperature=300.0,
    max_scba_iter=40,
    scba_tol=0.005,
    mixing=0.15,
    scattering_contacts=False,
    anderson_mixing = True,
    anderson_depth = 5,
)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(results: dict, a1_len: float):
    """Write completed slab results to both JSON (checkpoint) and npz (data)."""
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

    save_dict = {
        "slab_counts": np.array(sorted(results.keys())),
        "a1_len": a1_len,
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
    """Load checkpoint and return dict of {n_slabs: result_dict}."""
    if not DATA_FILE.exists() or not CHECKPOINT_FILE.exists():
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
# Plotting
# ---------------------------------------------------------------------------

def make_plot(results: dict, a1_len: float):
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
    thicknesses_nm = [ns * a1_len / 10.0 for ns in completed]
    reductions = []
    for ns in completed:
        G_b = results[ns]["thermal_conductance_ballistic"]
        G_a = results[ns]["thermal_conductance_anharmonic"]
        reductions.append((1 - G_a / G_b) * 100 if G_b > 0 else 0)
    ax.plot(thicknesses_nm, reductions, "bo-", ms=8, lw=1.5,
            label="This work (4x4)")
    # Guo reference point (approximate, from conventional cell)
    guo = GUO_TABLE3[(4, 5)]
    red_guo = (1 - guo["G_anh"] / guo["G_ball"]) * 100
    # 5 uc of conventional cell ≈ 5 * 5.466 Å ≈ 2.73 nm
    ax.plot([5 * 5.466 / 10.0], [red_guo], "rs", ms=10, label="Guo (4x4, conv)")
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
        kappas.append(G_a * ns * a1_len * 1e-10)
    ax.plot(thicknesses_nm, kappas, "bo-", ms=8, lw=1.5,
            label="This work (4x4)")
    kappa_guo = guo["G_anh"] * 1e6 * 5 * 5.466 * 1e-10
    ax.plot([5 * 5.466 / 10.0], [kappa_guo], "rs", ms=10, label="Guo (4x4, conv)")
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


def print_summary(results: dict, a1_len: float):
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
        L_nm = ns * a1_len / 10.0
        kappa = G_a * ns * a1_len * 1e-10
        print(f"{ns:>8d} {L_nm:>8.2f} {G_b/1e6:>10.1f} {G_a/1e6:>10.1f} "
              f"{red:>9.1f} {kappa:>13.2f} {r['n_scba_iterations']:>6d}")

    guo = GUO_TABLE3[(4, 5)]
    red_guo = (1 - guo["G_anh"] / guo["G_ball"]) * 100
    kappa_guo = guo["G_anh"] * 1e6 * 5 * 5.466 * 1e-10
    print(f"\nGuo et al. Table III (5 uc conventional, 4x4 q-mesh):")
    print(f"  G_ball = {guo['G_ball']:.2f}, G_anh = {guo['G_anh']:.2f} MW/(m^2 K)")
    print(f"  Reduction: {red_guo:.1f}%, kappa = {kappa_guo:.2f} W/(m K)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true",
                        help="Only replot from saved data, no computation")
    parser.add_argument("--hilbert", action="store_true",
                        help="Use Hilbert-transform retarded self-energy")
    args = parser.parse_args()

    if args.hilbert:
        COMMON["hilbert_retarded"] = True
        CHECKPOINT_FILE = script_dir / "anharmonic_multislab_checkpoint_hilbert.json"
        DATA_FILE = script_dir / "anharmonic_multislab_data_hilbert.npz"
        PLOT_FILE = script_dir / "anharmonic_multislab_hilbert.png"
        print("*** Hilbert-transform retarded self-energy enabled ***")

    # Plot-only mode
    if args.plot:
        if not DATA_FILE.exists():
            print(f"No data file found at {DATA_FILE}")
            sys.exit(1)
        data = np.load(DATA_FILE, allow_pickle=False)
        a1_len = float(data["a1_len"])
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
        print_summary(results, a1_len)
        make_plot(results, a1_len)
        sys.exit(0)

    # --- Computation mode ---
    from run_anharmonic import load_primitive_cell
    from phonon_inputs.anharmonic import anharmonic_transmission_q

    print("=" * 60)
    print("Loading Si primitive cell (phono3py)...")
    phonon, _ = load_primitive_cell(work_dir)
    fc3_hdf5 = str(work_dir / "reaps" / "si_primitive_work" / "fc3.hdf5")

    a1_len = np.linalg.norm(phonon.primitive.cell[0])
    print(f"  |a1| = {a1_len:.4f} A")

    # Load checkpoint
    results = _load_checkpoint()
    if results:
        done = sorted(results.keys())
        print(f"\nResuming from checkpoint: n_slabs={done} already completed")
    else:
        print("\nNo checkpoint found, starting fresh")

    # ------------------------------------------------------------------
    # Ballistic sanity check
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Ballistic sanity check (1 vs 3 vs 5 slabs)")
    print("=" * 60)
    for ns in [1, 3, 5]:
        ballistic_params = {**COMMON, "max_scba_iter": 0}
        res = anharmonic_transmission_q(
            phonon, fc3_hdf5, **ballistic_params,
            n_slabs=ns, verbose=False,
        )
        print(f"  n_slabs={ns}: max T_ballistic = "
              f"{res['transmission_ballistic'].max():.4f}, "
              f"G_ballistic = "
              f"{res['thermal_conductance_ballistic']/1e6:.2f} MW/(m^2 K)")

    # ------------------------------------------------------------------
    # Thickness sweep with SCBA
    # ------------------------------------------------------------------
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

        print(f"\n--- n_slabs = {ns} (device = {ns * a1_len:.1f} A) ---")
        t0 = time.time()
        res = anharmonic_transmission_q(
            phonon, fc3_hdf5, **COMMON,
            n_slabs=ns, verbose=True,
        )
        t1 = time.time()
        results[ns] = res

        G_b = res["thermal_conductance_ballistic"]
        G_a = res["thermal_conductance_anharmonic"]
        red = (1 - G_a / G_b) * 100 if G_b > 0 else 0
        L = ns * a1_len * 1e-10
        kappa = G_a * L

        print(f"  Time: {t1-t0:.1f} s")
        print(f"  G_ball = {G_b/1e6:.2f}, G_anh = {G_a/1e6:.2f} MW/(m^2 K)")
        print(f"  Reduction: {red:.1f}%")
        print(f"  kappa_eff = G * L = {kappa:.2f} W/(m K)")
        print(f"  SCBA iters: {res['n_scba_iterations']}")

        _save_checkpoint(results, a1_len)
        print(f"  >> Checkpoint saved ({len(results)}/{len(SLAB_COUNTS)} done)")

    # ------------------------------------------------------------------
    # Summary and plot
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print_summary(results, a1_len)
    make_plot(results, a1_len)
