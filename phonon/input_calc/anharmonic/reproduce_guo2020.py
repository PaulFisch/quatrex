"""Reproduce thickness-dependent thermal conductivity from Guo et al. (2020).

Reference: Yangyu Guo et al., Phys. Rev. B 102, 195412 (2020)
           "Quantum mechanical modeling of anharmonic phonon-phonon
            scattering in nanostructures"

Reproduces Fig. 8(b): cross-plane thermal conductivity of a silicon thin
film vs. thickness at 300 K, comparing ballistic and anharmonic NEGF.

Our setup uses the FCC primitive cell (2 atoms) as one slab. The
conventional cell (8 atoms, a = 5.40 A) corresponds to 2 primitive slabs.
Transport direction: x (along [100]).
Slab thickness: d = n_slabs * a_x, where a_x = a/2 for FCC along [100].

The paper's key data (1st nearest-neighbor FC3, Table IV and Fig. 8b):
  d = 20 uc (10.8 nm): kappa_NEGF = 7.45 W/mK
  d = 24 uc (13.0 nm): kappa_NEGF = 8.40 W/mK
"""

import argparse
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
from phonon_inputs.separable import separable_anharmonic_transmission

OUT_DIR = script_dir / "figures"
CACHE_DIR = script_dir / "guo2020_cache"


def get_slab_thickness(phonon, n_slabs, transport_direction="x"):
    """Compute device thickness in meters.

    For FCC primitive cell with transport along x, each slab
    has thickness a/2 where a is the conventional lattice constant.
    """
    lattice = phonon.primitive.cell
    tidx = "xyz".index(transport_direction)
    # The slab repeat distance along transport is the lattice parameter
    # projected onto the transport direction.  For FCC, this equals a/2.
    # We take the maximum x-projection among the lattice vectors
    # (since one of them defines the stacking).
    a_transport = max(abs(lattice[i, tidx]) for i in range(3))
    d_ang = n_slabs * a_transport  # Angstrom
    return d_ang * 1e-10  # meters


def run_thickness_sweep(phonon, fc3_path, slab_counts, rank=12,
                        q_mesh=(4, 4), nfreq=101, temperature=300.0,
                        mixing=None):
    """Run ballistic and anharmonic transport for each thickness."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for ns in slab_counts:
        cache_file = CACHE_DIR / f"slabs{ns}_R{rank}_q{q_mesh[0]}.npz"
        if cache_file.exists():
            print(f"  [cached] n_slabs={ns}")
            data = np.load(cache_file, allow_pickle=True)
            results[ns] = {k: (v.item() if v.ndim == 0 else v)
                           for k, v in data.items()}
            continue

        # Adaptive mixing: smaller for larger devices to aid convergence
        if mixing is not None:
            mix = mixing
        elif ns <= 6:
            mix = 0.3
        elif ns <= 10:
            mix = 0.2
        else:
            mix = 0.1

        print(f"\n=== n_slabs = {ns}, R = {rank}, mixing = {mix} ===")
        t0 = time.time()
        r = separable_anharmonic_transmission(
            phonon, str(fc3_path),
            q_mesh_transverse=q_mesh,
            freq_range_thz=(0.5, 15.5, nfreq),
            transport_direction="x",
            eta_factor=0.5,
            temperature=temperature,
            delta_T=10.0,
            max_scba_iter=40,
            scba_tol=0.005,
            mixing=mix,
            n_slabs=ns,
            rank=rank,
            verbose=True,
        )
        elapsed = time.time() - t0
        print(f"  Wall time: {elapsed:.1f} s")

        # Save cache (excluding large self-energy arrays)
        save_data = {}
        for k, v in r.items():
            if k == "self_energy_retarded":
                continue  # too large for thickness sweeps
            if isinstance(v, np.ndarray):
                save_data[k] = v
            elif isinstance(v, (int, float)):
                save_data[k] = np.array(v)
            elif isinstance(v, list):
                save_data[k] = np.array(v)
        save_data["wall_time"] = np.array(elapsed)
        np.savez(cache_file, **save_data)

        results[ns] = r
        results[ns]["wall_time"] = elapsed

    return results


def plot_thickness_conductivity(phonon, results, temperature=300.0):
    """Plot thermal conductivity vs thickness, comparing with Guo et al."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    slab_counts = sorted(results.keys())
    thicknesses_m = [get_slab_thickness(phonon, ns) for ns in slab_counts]
    thicknesses_nm = [d * 1e9 for d in thicknesses_m]

    G_ball = [float(results[ns]["thermal_conductance_ballistic"])
              for ns in slab_counts]
    G_anh = [float(results[ns]["thermal_conductance_anharmonic"])
             for ns in slab_counts]

    # kappa = G * d
    kappa_ball = [G_ball[i] * thicknesses_m[i] for i in range(len(slab_counts))]
    kappa_anh = [G_anh[i] * thicknesses_m[i] for i in range(len(slab_counts))]

    # Print summary table
    print(f"\n{'n_slabs':>8} {'d (nm)':>8} {'G_ball':>12} {'G_anh':>12} "
          f"{'kappa_ball':>12} {'kappa_anh':>12} {'Reduction':>10}")
    print(f"{'':>8} {'':>8} {'(MW/m2K)':>12} {'(MW/m2K)':>12} "
          f"{'(W/mK)':>12} {'(W/mK)':>12} {'(%)':>10}")
    print("-" * 88)
    for i, ns in enumerate(slab_counts):
        red = (1 - G_anh[i] / G_ball[i]) * 100 if G_ball[i] > 0 else 0
        print(f"{ns:>8} {thicknesses_nm[i]:>8.2f} {G_ball[i]/1e6:>12.1f} "
              f"{G_anh[i]/1e6:>12.1f} {kappa_ball[i]:>12.4f} "
              f"{kappa_anh[i]:>12.4f} {red:>10.1f}")

    # --- Reference data from Guo et al. (2020), Fig. 8(b) ---
    # Digitized from the figure (1st NN FC3, T=300K)
    # Ballistic NEGF+DFT (circles in Fig 8b)
    guo_d_ball_nm = [1.08, 2.16, 3.24, 5.40, 10.80]
    guo_kappa_ball = [3.0, 6.3, 9.5, 13.0, 13.5]  # approximate from Fig 8b

    # Anharmonic NEGF+DFT (triangles in Fig 8b)
    guo_d_anh_nm = [1.08, 2.16, 3.24, 5.40, 10.80, 13.0]
    guo_kappa_anh = [1.6, 3.0, 4.2, 5.1, 7.45, 8.40]

    # Ballistic BTE (dashed line in Fig 8b) - linear: kappa = G_ball * d
    # G_ballistic from Table III: ~1074 MW/m2K for 4x4 mesh
    # BTE line is approximately kappa = 10.74 * d(nm) (W/mK)
    bte_d = np.linspace(0.5, max(max(thicknesses_nm), 14), 50)
    bte_kappa = 10.74 * bte_d  # linear in d for ballistic

    # Monte Carlo (diamonds in Fig 8b)
    guo_d_mc_nm = [1.08, 2.16, 3.24, 5.40]
    guo_kappa_mc = [1.8, 3.2, 4.5, 5.8]

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: thermal conductivity vs thickness (log-log like Fig 8b)
    ax1.loglog(thicknesses_nm, kappa_ball, "o-", color="#1f77b4", ms=8, lw=2,
               label="Present ballistic NEGF", zorder=5)
    ax1.loglog(thicknesses_nm, kappa_anh, "^-", color="#d62728", ms=8, lw=2,
               label="Present anharmonic NEGF", zorder=5)

    # Reference data
    ax1.loglog(guo_d_ball_nm, guo_kappa_ball, "o", color="#1f77b4",
               ms=6, mfc="none", lw=1.5, label="Guo (2020) ballistic NEGF")
    ax1.loglog(guo_d_anh_nm, guo_kappa_anh, "^", color="#d62728",
               ms=6, mfc="none", lw=1.5, label="Guo (2020) anharmonic NEGF")
    ax1.loglog(guo_d_mc_nm, guo_kappa_mc, "D", color="#2ca02c",
               ms=6, mfc="none", lw=1.5, label="Guo (2020) MC+DFT")

    ax1.set_xlabel("Thickness (nm)", fontsize=12)
    ax1.set_ylabel("Thermal conductivity (W/m K)", fontsize=12)
    ax1.set_title(f"(a) Si thin film, T = {temperature:.0f} K")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_xlim(0.3, 20)
    ax1.set_ylim(0.3, 30)

    # Right panel: thermal conductance vs thickness
    ax2.plot(thicknesses_nm, [g / 1e6 for g in G_ball], "o-",
             color="#1f77b4", ms=8, lw=2, label="Ballistic")
    ax2.plot(thicknesses_nm, [g / 1e6 for g in G_anh], "^-",
             color="#d62728", ms=8, lw=2, label="Anharmonic (SCBA)")

    ax2.set_xlabel("Thickness (nm)", fontsize=12)
    ax2.set_ylabel("Thermal conductance (MW/m$^2$ K)", fontsize=12)
    ax2.set_title(f"(b) Conductance vs thickness, T = {temperature:.0f} K")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "thickness_conductivity.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "thickness_conductivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR / 'thickness_conductivity.pdf'}")

    # --- Spectral heat current comparison across thicknesses ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

    colors_slab = plt.cm.viridis(np.linspace(0.2, 0.9, len(slab_counts)))
    for i, ns in enumerate(slab_counts):
        r = results[ns]
        freqs = r["freqs_thz"]
        d_nm = thicknesses_nm[i]

        ax3.plot(freqs, r["spectral_heat_current_ballistic"],
                 ls="--", color=colors_slab[i], lw=1, alpha=0.5)
        ax3.plot(freqs, r["spectral_heat_current"],
                 ls="-", color=colors_slab[i], lw=1.5,
                 label=f"d={d_nm:.1f} nm ({ns} slabs)")

    ax3.set_xlabel("Frequency (THz)", fontsize=12)
    ax3.set_ylabel("Spectral heat current (W/THz)", fontsize=12)
    ax3.set_title("(a) Spectral current (solid=anh, dashed=ball)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Anharmonic reduction ratio: 1 - G_anh/G_ball
    d_arr = np.array(thicknesses_nm)
    reduction = np.array([(1 - G_anh[i] / G_ball[i]) * 100
                          for i in range(len(slab_counts))])
    ax4.plot(d_arr, reduction, "s-", color="#d62728", ms=8, lw=2)
    ax4.set_xlabel("Thickness (nm)", fontsize=12)
    ax4.set_ylabel("Conductance reduction (%)", fontsize=12)
    ax4.set_title("(b) Anharmonic reduction vs thickness")
    ax4.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "thickness_spectral.pdf", bbox_inches="tight")
    fig2.savefig(OUT_DIR / "thickness_spectral.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {OUT_DIR / 'thickness_spectral.pdf'}")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Guo et al. (2020) thickness-dependent kappa")
    parser.add_argument("--rank", type=int, default=12,
                        help="SVD rank (default: 12)")
    parser.add_argument("--nk", type=int, default=4,
                        help="Transverse q-mesh (nk x nk, default: 4)")
    parser.add_argument("--nfreq", type=int, default=101,
                        help="Frequency points (default: 101)")
    parser.add_argument("--temperature", type=float, default=300.0,
                        help="Temperature in K (default: 300)")
    parser.add_argument("--mixing", type=float, default=None,
                        help="SCBA mixing (default: adaptive)")
    parser.add_argument("--slabs", type=int, nargs="+",
                        default=[2, 4, 6, 10],
                        help="Slab counts (default: 2 4 6 10)")
    parser.add_argument("--load", action="store_true",
                        help="Only load cached results")
    args = parser.parse_args()

    phonon, _ = load_primitive_cell(work_dir)

    # Print geometry info
    lattice = phonon.primitive.cell
    print(f"Primitive cell lattice vectors:")
    for i in range(3):
        print(f"  a{i+1} = {lattice[i]}")
    for ns in args.slabs:
        d = get_slab_thickness(phonon, ns)
        print(f"  n_slabs={ns}: d = {d*1e10:.2f} A = {d*1e9:.3f} nm")

    if args.load:
        # Load only
        results = {}
        for ns in args.slabs:
            cache_file = CACHE_DIR / f"slabs{ns}_R{args.rank}_q{args.nk}.npz"
            if cache_file.exists():
                data = np.load(cache_file, allow_pickle=True)
                results[ns] = {k: (v.item() if v.ndim == 0 else v)
                               for k, v in data.items()}
        if not results:
            print("No cached results found. Run without --load first.")
            return
    else:
        fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"
        results = run_thickness_sweep(
            phonon, fc3_path, args.slabs,
            rank=args.rank, q_mesh=(args.nk, args.nk),
            nfreq=args.nfreq, temperature=args.temperature,
            mixing=args.mixing,
        )

    plot_thickness_conductivity(phonon, results, args.temperature)


if __name__ == "__main__":
    main()
