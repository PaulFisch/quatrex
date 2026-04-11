"""Compare dense, separable, and PCP anharmonic transport.

Runs all methods, saves results via io.py, generates multi-panel comparison plot.

Usage:
    python compare_methods.py [--load]

    --load: skip computation, load saved results from results/ directory
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

from run_anharmonic import load_primitive_cell
from phonon_inputs.pcp import pcp_anharmonic_transmission
from phonon_inputs.separable import (
    decompose_fc3_supercell,
    reconstruction_error,
    separable_anharmonic_transmission,
)
from phonon_inputs.anharmonic import anharmonic_transmission_q
from phonon_inputs.io import save_transport_results, load_transport_results


RESULTS_DIR = script_dir / "results"


def run_all_methods(phonon, fc3_path, common):
    """Run all methods and save results."""
    results = {}
    timings = {}

    # Dense
    print("\n=== Dense ===")
    t0 = time.time()
    r = anharmonic_transmission_q(phonon, str(fc3_path), **common)
    timings["dense"] = time.time() - t0
    results["dense"] = r
    save_transport_results(r, RESULTS_DIR / "dense", {"method": "dense", "time_s": timings["dense"]})
    print(f"  Time: {timings['dense']:.1f}s, G_anh: {r['thermal_conductance_anharmonic']:.2e}")

    # Separable full rank
    print("\n=== Separable (full rank) ===")
    t0 = time.time()
    r = separable_anharmonic_transmission(phonon, str(fc3_path), rank=None, **common)
    timings["sep_full"] = time.time() - t0
    results["sep_full"] = r
    save_transport_results(r, RESULTS_DIR / "sep_full", {"method": "separable", "rank": "full", "time_s": timings["sep_full"]})
    print(f"  Time: {timings['sep_full']:.1f}s, G_anh: {r['thermal_conductance_anharmonic']:.2e}")

    # Separable R=6
    print("\n=== Separable R=6 ===")
    t0 = time.time()
    r = separable_anharmonic_transmission(phonon, str(fc3_path), rank=6, **common)
    timings["sep_r6"] = time.time() - t0
    results["sep_r6"] = r
    save_transport_results(r, RESULTS_DIR / "sep_r6", {"method": "separable", "rank": 6, "time_s": timings["sep_r6"]})
    print(f"  Time: {timings['sep_r6']:.1f}s, G_anh: {r['thermal_conductance_anharmonic']:.2e}")

    # PCP ranks
    for N_c in [8, 24]:
        key = f"pcp_{N_c}"
        print(f"\n=== PCP N_c={N_c} ===")
        t0 = time.time()
        r = pcp_anharmonic_transmission(phonon, str(fc3_path), pcp_rank=N_c, **common)
        timings[key] = time.time() - t0
        results[key] = r
        save_transport_results(r, RESULTS_DIR / key, {"method": "pcp", "N_c": N_c, "time_s": timings[key]})
        print(f"  Time: {timings[key]:.1f}s, G_anh: {r['thermal_conductance_anharmonic']:.2e}")

    return results, timings


def load_all_results():
    """Load previously saved results."""
    results = {}
    for d in RESULTS_DIR.iterdir():
        if d.is_dir() and (d / "transport.npz").exists():
            results[d.name] = load_transport_results(d)
    return results


def plot_comparison(results, out_dir):
    """Generate multi-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    styles = {
        "dense":    ("Dense (exact)",      "tab:red",    "-",  2.0),
        "sep_full": ("Separable (full)",   "tab:green",  "--", 1.5),
        "sep_r6":   ("Separable R=6",      "tab:orange", "-.", 1.5),
        "pcp_24":   ("PCP $N_c$=24",       "tab:blue",   "-",  1.5),
        "pcp_8":    ("PCP $N_c$=8",        "tab:cyan",   ":",  1.5),
    }

    # --- Panel (0,0): Spectral heat current ---
    ax = axes[0, 0]
    for key in ["dense", "sep_full", "sep_r6", "pcp_24", "pcp_8"]:
        if key not in results:
            continue
        r = results[key]
        label, color, ls, lw = styles[key]
        freqs = r["freqs_thz"]
        ax.plot(freqs, r["spectral_heat_current"], ls, color=color, lw=lw, label=label)

    # Ballistic (from dense, as reference)
    if "dense" in results:
        ax.plot(results["dense"]["freqs_thz"],
                results["dense"]["spectral_heat_current_ballistic"],
                ":", color="gray", alpha=0.5, lw=1, label="Ballistic")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Spectral heat current (W/THz)")
    ax.set_title("(a) Spectral heat current")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel (0,1): Self-energy trace (Im Sigma^R) ---
    ax = axes[0, 1]
    for key in ["dense", "sep_full", "sep_r6", "pcp_24", "pcp_8"]:
        if key not in results or "self_energy_retarded" not in results[key]:
            continue
        r = results[key]
        label, color, ls, lw = styles[key]
        freqs = r["freqs_thz"]
        # Average over q-points, take trace
        Sigma_R = r["self_energy_retarded"]
        if Sigma_R.ndim == 4:
            # (n_kpts, n_freq, n_dof, n_dof) — q-resolved
            trace_per_q = -np.imag(np.trace(Sigma_R, axis1=-2, axis2=-1))
            trace_avg = np.mean(trace_per_q, axis=0)
            n_dof = Sigma_R.shape[-1]
        elif Sigma_R.ndim == 3:
            # (n_freq, n_dof, n_dof) — slab-averaged
            trace_avg = -np.imag(np.trace(Sigma_R, axis1=-2, axis2=-1))
            n_dof = Sigma_R.shape[-1]
        else:
            continue
        ax.plot(freqs, trace_avg / n_dof, ls, color=color, lw=lw, label=label)

    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(r"$-\mathrm{Tr}[\mathrm{Im}\,\Sigma^R] / n_\mathrm{dof}$ (THz$^2$)")
    ax.set_title(r"(b) Self-energy broadening")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel (1,0): Thermal conductance summary ---
    ax = axes[1, 0]
    methods = []
    g_vals = []
    colors = []
    for key in ["dense", "sep_full", "sep_r6", "pcp_8", "pcp_24"]:
        if key not in results:
            continue
        r = results[key]
        methods.append(styles[key][0])
        g_vals.append(r["thermal_conductance_anharmonic"])
        colors.append(styles[key][1])

    if methods:
        bars = ax.barh(range(len(methods)), [g / 1e6 for g in g_vals], color=colors, alpha=0.8)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=9)
        ax.set_xlabel(r"$G_\mathrm{anh}$ (MW/m$^2$K)")
        ax.set_title("(c) Thermal conductance")
        # Add value labels
        for bar, g in zip(bars, g_vals):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                    f"{g/1e6:.0f}", va='center', fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')

    # --- Panel (1,1): Difference from dense ---
    ax = axes[1, 1]
    if "dense" in results:
        g_ref = results["dense"]["thermal_conductance_anharmonic"]
        freqs_ref = results["dense"]["freqs_thz"]
        j_ref = results["dense"]["spectral_heat_current"]

        for key in ["sep_full", "sep_r6", "pcp_24", "pcp_8"]:
            if key not in results:
                continue
            r = results[key]
            label, color, ls, lw = styles[key]
            freqs = r["freqs_thz"]
            j_diff = r["spectral_heat_current"] - j_ref
            ax.plot(freqs, j_diff, ls, color=color, lw=lw, label=label)

        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel(r"$\Delta J(\omega)$ vs dense (W/THz)")
        ax.set_title("(d) Spectral heat current difference from dense")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "method_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close(fig)


def print_summary(results):
    """Print summary table."""
    print(f"\n{'Method':<22} {'G_ball (MW)':>12} {'G_anh (MW)':>12} {'G_anh/G_ball':>14} {'Conserv':>10}")
    print("-" * 72)
    for key in ["dense", "sep_full", "sep_r6", "pcp_8", "pcp_24"]:
        if key not in results:
            continue
        r = results[key]
        g_ball = r["thermal_conductance_ballistic"]
        g_anh = r["thermal_conductance_anharmonic"]
        cons = r.get("heat_flow_conservation", float('nan'))
        label = key
        print(f"  {label:<20} {g_ball/1e6:>12.1f} {g_anh/1e6:>12.1f} {g_anh/g_ball:>14.4f} {cons:>10.2e}")


def main():
    load_only = "--load" in sys.argv

    if load_only:
        print("Loading saved results...")
        results = load_all_results()
        if not results:
            print("No saved results found. Run without --load first.")
            return
    else:
        phonon, _ = load_primitive_cell(work_dir)
        fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

        common = dict(
            q_mesh_transverse=(4, 4),
            freq_range_thz=(1.0, 14.0, 101),
            max_scba_iter=10,
            scba_tol=0.005,
            mixing=0.3,
            n_slabs=1,
            verbose=True,
        )

        results, timings = run_all_methods(phonon, fc3_path, common)

        print("\n\nTimings:")
        for k, t in timings.items():
            print(f"  {k}: {t:.1f}s")

    print_summary(results)
    plot_comparison(results, script_dir)


if __name__ == "__main__":
    main()
