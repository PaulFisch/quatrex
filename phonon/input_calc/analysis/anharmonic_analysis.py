"""Comprehensive analysis of anharmonic phonon transport in Si.

Compares FD and DFPT FC3 sources using separable FC3 decomposition:
  1. Self-energy physics: Im/Re Sigma from separable SCBA
  2. Transport & SVD convergence: spectral heat current, rank sweep
  3. Temperature dependence

Usage:
    python analysis/anharmonic_analysis.py [--nk 4] [--nfreq-fine 101]
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

from anharmonic.run_anharmonic import load_primitive_cell, load_primitive_cell_dfpt
from phonon_inputs.separable import separable_anharmonic_transmission

# -- Style ---------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "savefig.dpi": 300,
    "figure.dpi": 100,
})


# ========================================================================
# Computation helpers
# ========================================================================

def _common_kwargs(nfreq, max_iter, temperature=300.0):
    return dict(
        freq_range_thz=(0.0, 14.0, nfreq),
        transport_direction="x",
        eta_factor=0.5,
        temperature=temperature,
        delta_T=10.0,
        max_scba_iter=max_iter,
        scba_tol=0.005,
        mixing=0.3,
        n_slabs=1,
    )


def compute_self_energy_data(phonon_fd, fc3_hdf5_fd,
                              phonon_dfpt, fc3_hdf5_dfpt, nk, nfreq):
    """Run separable full-rank SCBA for both FD and DFPT."""
    print("\n--- Computing self-energy data ---")
    kw = _common_kwargs(nfreq, max_iter=10)
    results = {}

    for label, phonon, fc3_h5 in [
        ("FD",   phonon_fd,   fc3_hdf5_fd),
        ("DFPT", phonon_dfpt, fc3_hdf5_dfpt),
    ]:
        t0 = time.time()
        res = separable_anharmonic_transmission(
            phonon, str(fc3_h5),
            q_mesh_transverse=(nk, nk),
            rank=None, svd_tol=1e-12,
            verbose=True, **kw,
        )
        print(f"  {label}: {time.time() - t0:.1f}s")
        results[label] = res

    return results


def compute_transport_data(phonon_fd, fc3_hdf5_fd,
                            phonon_dfpt, fc3_hdf5_dfpt, nk, nfreq_fine):
    """Run separable full and R=6 for both sources, plus rank sweep."""
    print("\n--- Computing transport / SVD data ---")
    kw_fine = _common_kwargs(nfreq_fine, max_iter=10)
    results = {}

    for label, phonon, fc3_h5 in [
        ("FD",   phonon_fd,   fc3_hdf5_fd),
        ("DFPT", phonon_dfpt, fc3_hdf5_dfpt),
    ]:
        t0 = time.time()
        res_full = separable_anharmonic_transmission(
            phonon, str(fc3_h5),
            q_mesh_transverse=(nk, nk),
            rank=None, svd_tol=1e-12,
            verbose=True, **kw_fine,
        )
        print(f"  {label} sep full: {time.time() - t0:.1f}s")

        t0 = time.time()
        res_r6 = separable_anharmonic_transmission(
            phonon, str(fc3_h5),
            q_mesh_transverse=(nk, nk),
            rank=6,
            verbose=True, **kw_fine,
        )
        print(f"  {label} sep R=6: {time.time() - t0:.1f}s")

        # Rank sweep (coarser grid, fewer iterations)
        kw_coarse = _common_kwargs(31, max_iter=3)
        ranks = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]
        rank_results = []
        t0 = time.time()
        for r in ranks:
            res = separable_anharmonic_transmission(
                phonon, str(fc3_h5),
                q_mesh_transverse=(nk, nk),
                rank=r,
                verbose=False, **kw_coarse,
            )
            rank_results.append(res)
            print(f"    {label} R={r:2d}: G_anh = "
                  f"{res['thermal_conductance_anharmonic']/1e6:.1f} MW/m²K")
        print(f"  {label} rank sweep: {time.time() - t0:.1f}s")

        results[label] = {
            "full": res_full,
            "r6": res_r6,
            "ranks": ranks,
            "rank_results": rank_results,
        }

    return results


def compute_temperature_data(phonon_fd, fc3_hdf5_fd,
                              phonon_dfpt, fc3_hdf5_dfpt, nk):
    """Temperature sweep for both FD and DFPT using separable full rank."""
    print("\n--- Computing temperature data ---")
    temps = [100, 200, 300, 400, 500]
    results = {}

    for label, phonon, fc3_h5 in [
        ("FD",   phonon_fd,   fc3_hdf5_fd),
        ("DFPT", phonon_dfpt, fc3_hdf5_dfpt),
    ]:
        temp_results = []
        t0 = time.time()
        for T in temps:
            kw = _common_kwargs(51, max_iter=10, temperature=float(T))
            res = separable_anharmonic_transmission(
                phonon, str(fc3_h5),
                q_mesh_transverse=(nk, nk),
                rank=None, svd_tol=1e-12,
                verbose=False, **kw,
            )
            temp_results.append(res)
            G = res["thermal_conductance_anharmonic"]
            Gb = res["thermal_conductance_ballistic"]
            print(f"    {label} T={T}K: G_ball={Gb/1e6:.1f}, "
                  f"G_anh={G/1e6:.1f} MW/m²K")
        print(f"  {label} temp sweep: {time.time() - t0:.1f}s")
        results[label] = temp_results

    return temps, results


# ========================================================================
# Plotting
# ========================================================================

def plot_self_energy(se_results, out_dir):
    """Figure 1: Self-energy physics (2x3 grid)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    result_fd = se_results["FD"]
    result_dfpt = se_results["DFPT"]
    freqs_fd = result_fd["freqs_thz"]
    freqs_dfpt = result_dfpt["freqs_thz"]
    Sigma_R_fd = result_fd["self_energy_retarded"]
    Sigma_R_dfpt = result_dfpt["self_energy_retarded"]
    n_dof = Sigma_R_fd.shape[-1]

    Sig_fd = Sigma_R_fd[0]  # (nfreq, n_dof, n_dof)
    Sig_dfpt = Sigma_R_dfpt[0]

    # (a) -Im[Sigma^R] trace average
    ax = axes[0, 0]
    trace_fd = -np.trace(Sig_fd, axis1=1, axis2=2).imag / n_dof
    trace_dfpt = -np.trace(Sig_dfpt, axis1=1, axis2=2).imag / n_dof
    ax.plot(freqs_fd, trace_fd, "b-", lw=2.0, label="FD")
    ax.plot(freqs_dfpt, trace_dfpt, "r--", lw=2.0, label="DFPT")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(r"$-\mathrm{Im}\,\mathrm{Tr}\,\Sigma^R / n$ (THz$^2$)")
    ax.set_title("(a) Phonon damping rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Re[Sigma^R] trace average
    ax = axes[0, 1]
    trace_re_fd = np.trace(Sig_fd, axis1=1, axis2=2).real / n_dof
    trace_re_dfpt = np.trace(Sig_dfpt, axis1=1, axis2=2).real / n_dof
    ax.plot(freqs_fd, trace_re_fd, "b-", lw=2.0, label="FD")
    ax.plot(freqs_dfpt, trace_re_dfpt, "r--", lw=2.0, label="DFPT")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(r"$\mathrm{Re}\,\mathrm{Tr}\,\Sigma^R / n$ (THz$^2$)")
    ax.set_title("(b) Frequency renormalization")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) -Im[Sigma^R] per DOF (FD)
    ax = axes[0, 2]
    for i in range(n_dof):
        ax.plot(freqs_fd, -Sig_fd[:, i, i].imag, lw=1.0, alpha=0.7,
                label=f"DOF {i}")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(r"$-\mathrm{Im}\,\Sigma^R_{ii}$ (THz$^2$)")
    ax.set_title("(c) FD damping per DOF")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # (d) Spectral heat current: FD vs DFPT
    ax = axes[1, 0]
    for label, res, color, ls in [
        ("FD", result_fd, "b", "-"),
        ("DFPT", result_dfpt, "r", "--"),
    ]:
        f = res["freqs_thz"]
        ax.plot(f, res["spectral_heat_current_ballistic"],
                color=color, ls=ls, lw=0.8, alpha=0.4)
        G = res["thermal_conductance_anharmonic"] / 1e6
        ax.plot(f, res["spectral_heat_current"],
                color=color, ls=ls, lw=1.8,
                label=f"{label} ({G:.0f} MW/m²K)")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Spectral heat current (W)")
    ax.set_title("(d) Heat current: FD vs DFPT")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (e) Eigenvalues of Im[Sigma^R(w)] — FD
    ax = axes[1, 1]
    eig_curves = np.zeros((len(freqs_fd), n_dof))
    for iw in range(len(freqs_fd)):
        sig_im = Sig_fd[iw].imag
        sig_sym = 0.5 * (sig_im + sig_im.T)
        eigvals = np.sort(np.linalg.eigvalsh(sig_sym))[::-1]
        eig_curves[iw] = eigvals
    for i in range(n_dof):
        ax.plot(freqs_fd, -eig_curves[:, i], lw=1.0, label=f"eig {i}")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(r"$-\lambda_i[\mathrm{Im}\,\Sigma^R]$ (THz$^2$)")
    ax.set_title("(e) FD self-energy eigenvalues")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # (f) max|Sigma^R| comparison
    ax = axes[1, 2]
    for label, res, color in [("FD", result_fd, "b"), ("DFPT", result_dfpt, "r")]:
        sig = res["self_energy_retarded"]
        sig_max_w = np.array([np.max(np.abs(sig[0, iw]))
                              for iw in range(len(res["freqs_thz"]))])
        ax.plot(res["freqs_thz"], sig_max_w, color=color, lw=1.5, label=label)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(r"$\max_{ij}|\Sigma^R_{ij}|$ (THz$^2$)")
    ax.set_title(r"(f) Self-energy magnitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / "fig1_self_energy.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_transport(se_results, tp_results, out_dir):
    """Figure 2: Transport & SVD convergence (2x2)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Spectral heat current: full vs R=6
    ax = axes[0, 0]
    for src, color_f, color_r, ls in [
        ("FD", "tab:blue", "tab:cyan", "-"),
        ("DFPT", "tab:red", "salmon", "--"),
    ]:
        res_f = tp_results[src]["full"]
        res_r = tp_results[src]["r6"]
        f = res_f["freqs_thz"]
        Gf = res_f["thermal_conductance_anharmonic"] / 1e6
        Gr = res_r["thermal_conductance_anharmonic"] / 1e6
        ax.plot(f, res_f["spectral_heat_current"],
                color=color_f, ls=ls, lw=1.8,
                label=f"{src} full ({Gf:.0f})")
        ax.plot(res_r["freqs_thz"], res_r["spectral_heat_current"],
                color=color_r, ls=ls, lw=1.2,
                label=f"{src} R=6 ({Gr:.0f})")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Spectral heat current (W)")
    ax.set_title("(a) Spectral heat current (MW/m²K)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (b) J_anh / J_ball ratio
    ax = axes[0, 1]
    for src, color, ls in [("FD", "tab:blue", "-"), ("DFPT", "tab:red", "--")]:
        res = se_results[src]
        f = res["freqs_thz"]
        J_ball = res["spectral_heat_current_ballistic"]
        J_anh = res["spectral_heat_current"]
        mask = np.abs(J_ball) > np.max(np.abs(J_ball)) * 0.02
        ratio = np.ones_like(f)
        ratio[mask] = J_anh[mask] / J_ball[mask]
        ax.plot(f[mask], ratio[mask], color=color, ls=ls, lw=1.5, label=src)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(r"$J_\mathrm{anh} / J_\mathrm{ball}$")
    ax.set_title("(b) Scattering suppression ratio")
    ax.set_ylim(0, 1.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) G_anh vs SVD rank
    ax = axes[1, 0]
    for src, color in [("FD", "tab:blue"), ("DFPT", "tab:red")]:
        tp = tp_results[src]
        ranks = tp["ranks"]
        G_ranks = [r["thermal_conductance_anharmonic"] / 1e6
                    for r in tp["rank_results"]]
        G_full = se_results[src]["thermal_conductance_anharmonic"] / 1e6
        ax.plot(ranks, G_ranks, "o-", color=color, ms=4, lw=1.5,
                label=f"{src} sep")
        ax.axhline(G_full, color=color, ls="--", lw=1.0, alpha=0.6)
    ax.set_xlabel("SVD rank R")
    ax.set_ylabel(r"$G_\mathrm{anh}$ (MW/m$^2$K)")
    ax.set_title("(c) Conductance vs SVD rank")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Bar chart summary
    ax = axes[1, 1]
    labels_bar = ["FD", "DFPT"]
    x = np.arange(len(labels_bar))
    width = 0.25
    for i, (key, color, name) in enumerate([
        ("full", "tab:blue", "Full rank"),
        ("r6", "tab:green", "R=6"),
    ]):
        vals = [tp_results[src][key]["thermal_conductance_anharmonic"] / 1e6
                for src in labels_bar]
        ax.bar(x + (i - 0.5) * width, vals, width, color=color, label=name)
    for j, src in enumerate(labels_bar):
        Gb = tp_results[src]["full"]["thermal_conductance_ballistic"] / 1e6
        ax.plot([j - width, j + width], [Gb, Gb],
                "k--", lw=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar)
    ax.set_ylabel("G (MW / m² K)")
    ax.set_title("(d) Thermal conductance comparison")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = out_dir / "fig2_transport.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_temperature(temps, temp_results, out_dir):
    """Figure 3: Temperature dependence (2x2)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) G(T) ballistic and anharmonic
    ax = axes[0, 0]
    for src, color in [("FD", "tab:blue"), ("DFPT", "tab:red")]:
        G_balls = [r["thermal_conductance_ballistic"] / 1e6
                    for r in temp_results[src]]
        G_anhs = [r["thermal_conductance_anharmonic"] / 1e6
                   for r in temp_results[src]]
        ax.plot(temps, G_balls, "o--", color=color, lw=1.0, ms=5, alpha=0.5,
                label=f"{src} ball.")
        ax.plot(temps, G_anhs, "s-", color=color, lw=1.5, ms=5,
                label=f"{src} anh.")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$G$ (MW/m$^2$K)")
    ax.set_title("(a) Thermal conductance vs temperature")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Reduction factor
    ax = axes[0, 1]
    for src, color in [("FD", "tab:blue"), ("DFPT", "tab:red")]:
        G_balls = [r["thermal_conductance_ballistic"]
                    for r in temp_results[src]]
        G_anhs = [r["thermal_conductance_anharmonic"]
                   for r in temp_results[src]]
        reduction = [1 - Ga / Gb if Gb > 0 else 0
                     for Ga, Gb in zip(G_anhs, G_balls)]
        ax.plot(temps, reduction, "o-", color=color, lw=1.5, ms=5, label=src)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$1 - G_\mathrm{anh}/G_\mathrm{ball}$")
    ax.set_title("(b) Anharmonic reduction factor")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Conservation error vs temperature
    ax = axes[1, 0]
    for src, color in [("FD", "tab:blue"), ("DFPT", "tab:red")]:
        cons = [r["heat_flow_conservation"] for r in temp_results[src]]
        ax.semilogy(temps, cons, "o-", color=color, lw=1.5, ms=5, label=src)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Heat flow conservation error")
    ax.set_title("(c) Conservation error vs T")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) max|Sigma^R| vs temperature
    ax = axes[1, 1]
    for src, color in [("FD", "tab:blue"), ("DFPT", "tab:red")]:
        sig_maxes = [np.max(np.abs(r["self_energy_retarded"]))
                     for r in temp_results[src]]
        ax.plot(temps, sig_maxes, "o-", color=color, lw=1.5, ms=5, label=src)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\max|\Sigma^R|$ (THz$^2$)")
    ax.set_title(r"(d) Self-energy magnitude vs T")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / "fig3_temperature.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ========================================================================
# Main
# ========================================================================

def main():
    parser = argparse.ArgumentParser(description="Anharmonic transport analysis")
    parser.add_argument("--nk", type=int, default=4, help="q-mesh size (nk x nk)")
    parser.add_argument("--nfreq-fine", type=int, default=101,
                        help="Frequency points for Fig 1/2")
    parser.add_argument("--out", type=str, default=str(script_dir),
                        help="Output directory for plots")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Anharmonic phonon transport analysis: Si (FD + DFPT)")
    print("=" * 60)

    phonon_fd, _ = load_primitive_cell(work_dir)
    fc3_hdf5_fd = work_dir / "fc3_prim" / "fc3.hdf5"

    phonon_dfpt, _ = load_primitive_cell_dfpt(work_dir)
    fc3_hdf5_dfpt = work_dir / "dfpt" / "fc3.hdf5"

    # --- Figure 1: Self-energy physics ---
    se_results = compute_self_energy_data(
        phonon_fd, fc3_hdf5_fd,
        phonon_dfpt, fc3_hdf5_dfpt,
        args.nk, args.nfreq_fine,
    )
    plot_self_energy(se_results, out_dir)

    # --- Figure 2: Transport & SVD convergence ---
    tp_results = compute_transport_data(
        phonon_fd, fc3_hdf5_fd,
        phonon_dfpt, fc3_hdf5_dfpt,
        args.nk, args.nfreq_fine,
    )
    plot_transport(se_results, tp_results, out_dir)

    # --- Figure 3: Temperature ---
    temps, temp_results = compute_temperature_data(
        phonon_fd, fc3_hdf5_fd,
        phonon_dfpt, fc3_hdf5_dfpt,
        args.nk,
    )
    plot_temperature(temps, temp_results, out_dir)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for src in ["FD", "DFPT"]:
        G_ball = se_results[src]["thermal_conductance_ballistic"] / 1e6
        G_anh = se_results[src]["thermal_conductance_anharmonic"] / 1e6
        G_r6 = tp_results[src]["r6"]["thermal_conductance_anharmonic"] / 1e6
        cons = se_results[src]["heat_flow_conservation"]
        print(f"  {src}:")
        print(f"    G_ballistic:        {G_ball:.1f} MW/m²K")
        print(f"    G_anh (full rank):  {G_anh:.1f} MW/m²K")
        print(f"    G_anh (R=6):        {G_r6:.1f} MW/m²K")
        print(f"    Conservation error:  {cons:.4e}")

    print(f"\nAll figures saved to {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
