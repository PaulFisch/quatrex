"""Approximation quality analysis: dense vs SVD vs PCP.

Runs SCBA transport for multiple SVD ranks and PCP cell counts on the
Si 2x2x2 test case, then generates quality plots.

Usage:
    python plot_quality.py           # run all computations (slow, ~10 min)
    python plot_quality.py --load    # load cached results, regenerate plots
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
from phonon_inputs.separable import (
    decompose_fc3_supercell,
    reconstruction_error,
    separable_anharmonic_transmission,
)
from phonon_inputs.anharmonic import anharmonic_transmission_q
from phonon_inputs.pcp import pcp_anharmonic_transmission

OUT_DIR = script_dir / "figures"
CACHE_DIR = script_dir / "quality_cache"

# Shared SCBA parameters
COMMON = dict(
    q_mesh_transverse=(4, 4),
    freq_range_thz=(1.0, 14.0, 101),
    max_scba_iter=10,
    scba_tol=0.005,
    mixing=0.3,
    n_slabs=1,
    verbose=True,
)


# --------------------------------------------------------------------------
# Data collection
# --------------------------------------------------------------------------

def save_result(result, name):
    """Save transport result dict as npz."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            data[k] = v
        elif isinstance(v, (int, float)):
            data[k] = np.array(v)
        elif isinstance(v, list):
            try:
                data[k] = np.array(v)
            except (ValueError, TypeError):
                pass
        elif isinstance(v, dict):
            # Save scalar entries from sub-dicts (e.g., pcp_info)
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    data[f"_info_{kk}"] = np.array(vv)
                elif isinstance(vv, np.ndarray) and vv.size < 1000:
                    data[f"_info_{kk}"] = vv
    np.savez(CACHE_DIR / f"{name}.npz", **data)


def load_result(name):
    """Load transport result dict from npz."""
    path = CACHE_DIR / f"{name}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    result = {}
    for k in data.files:
        v = data[k]
        result[k] = v.item() if v.ndim == 0 else v
    return result


def run_dense(phonon, fc3_path):
    """Run dense reference."""
    cached = load_result("dense")
    if cached is not None:
        print("  [cached] dense")
        return cached
    print("\n=== Dense ===")
    t0 = time.time()
    r = anharmonic_transmission_q(phonon, str(fc3_path), **COMMON)
    r["wall_time"] = time.time() - t0
    save_result(r, "dense")
    return r


def run_separable(phonon, fc3_path, R):
    """Run separable at given rank."""
    name = f"sep_R{R}"
    cached = load_result(name)
    if cached is not None:
        print(f"  [cached] {name}")
        return cached
    print(f"\n=== Separable R={R} ===")
    t0 = time.time()
    r = separable_anharmonic_transmission(phonon, str(fc3_path), rank=R, **COMMON)
    r["wall_time"] = time.time() - t0
    save_result(r, name)
    return r


def run_separable_asr(phonon, fc3_path, R):
    """Run separable at given rank with ASR enforcement."""
    name = f"sep_asr_R{R}"
    cached = load_result(name)
    if cached is not None:
        print(f"  [cached] {name}")
        return cached
    print(f"\n=== Separable+ASR R={R} ===")
    t0 = time.time()
    r = separable_anharmonic_transmission(
        phonon, str(fc3_path), rank=R, enforce_asr=True, **COMMON,
    )
    r["wall_time"] = time.time() - t0
    save_result(r, name)
    return r


def run_pcp(phonon, fc3_path, Nc):
    """Run PCP at given rank."""
    name = f"pcp_Nc{Nc}"
    cached = load_result(name)
    if cached is not None:
        print(f"  [cached] {name}")
        return cached
    print(f"\n=== PCP Nc={Nc} ===")
    t0 = time.time()
    r = pcp_anharmonic_transmission(phonon, str(fc3_path), pcp_rank=Nc, **COMMON)
    r["wall_time"] = time.time() - t0
    save_result(r, name)
    return r


def compute_fc3_errors(phonon, fc3_path, R_values):
    """Compute FC3 reconstruction errors for SVD and SVD+ASR at various ranks."""
    import h5py
    from phonon_inputs.separable import build_supercell_mapping

    cached = load_result("fc3_errors")
    if cached is not None:
        print("  [cached] fc3_errors")
        return cached

    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(phonon)
    nat_prim = len(phonon.primitive.masses)
    masses_super = phonon.supercell.masses

    # SVD errors at different ranks
    svd_errors = []
    svd_svals = None
    for R in R_values:
        F_list, H, svals = decompose_fc3_supercell(
            fc3_raw, nat_prim,
            masses_super, prim_indices, ref_sc_atoms,
            rank=R,
        )
        err = reconstruction_error(
            fc3_raw, nat_prim,
            masses_super, ref_sc_atoms,
            F_list, H,
        )
        svd_errors.append(err)
        if svd_svals is None:
            # Get full singular values for spectrum plot
            F_full, H_full, svals_full = decompose_fc3_supercell(
                fc3_raw, nat_prim,
                masses_super, prim_indices, ref_sc_atoms,
                rank=None, tol=1e-15,
            )
            svd_svals = svals_full
        print(f"  SVD R={R}: rel_err = {err:.4e}")

    # SVD+ASR errors at same ranks
    svd_asr_errors = []
    for R in R_values:
        F_list, H, svals = decompose_fc3_supercell(
            fc3_raw, nat_prim,
            masses_super, prim_indices, ref_sc_atoms,
            rank=R, enforce_asr=True,
        )
        err = reconstruction_error(
            fc3_raw, nat_prim,
            masses_super, ref_sc_atoms,
            F_list, H,
        )
        svd_asr_errors.append(err)
        print(f"  SVD+ASR R={R}: rel_err = {err:.4e}")

    result = {
        "R_values": np.array(R_values),
        "svd_errors": np.array(svd_errors),
        "svd_asr_errors": np.array(svd_asr_errors),
        "svd_svals": svd_svals,
    }
    save_result(result, "fc3_errors")
    return result


def collect_all(load_only=False):
    """Run or load all configurations."""
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"

    R_values = [1, 2, 3, 4, 6, 8, 12, 18, 24]

    results = {}

    # FC3 reconstruction errors
    print("\n--- FC3 reconstruction errors ---")
    results["fc3_errors"] = compute_fc3_errors(phonon, fc3_path, R_values)

    # Dense reference
    results["dense"] = run_dense(phonon, fc3_path)

    # Separable sweep
    for R in [2, 4, 6, 12, 24]:
        results[f"sep_R{R}"] = run_separable(phonon, fc3_path, R)

    # PCP sweep
    for Nc in [2, 4, 8, 12, 24]:
        results[f"pcp_Nc{Nc}"] = run_pcp(phonon, fc3_path, Nc)

    return results


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------

COLORS_SEP = {2: "#77aa44", 4: "#449944", 6: "#2ca02c", 12: "#227722", 24: "#115511"}
COLOR_DENSE = "#d62728"
COLOR_PCP = "#1f77b4"


def _get_G(r):
    """Extract G_anh in MW/m^2K."""
    g = r.get("thermal_conductance_anharmonic", None)
    if g is None:
        return np.nan
    return float(g) / 1e6


def _get_conservation(r):
    """Extract conservation error."""
    return float(r.get("heat_flow_conservation", np.nan))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_fc3_error_vs_rank(results):
    """Fig Q1: FC3 reconstruction error vs rank (SVD and SVD+ASR), plus singular value spectrum."""
    fc3 = results["fc3_errors"]
    R_vals = fc3["R_values"]
    svd_err = fc3["svd_errors"]
    svals = fc3["svd_svals"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: singular value spectrum
    ax1.semilogy(np.arange(1, len(svals) + 1), svals / svals[0], "k.-", ms=4)
    ax1.set_xlabel("Singular value index", fontsize=12)
    ax1.set_ylabel("Normalized singular value $\\sigma_r / \\sigma_1$", fontsize=12)
    ax1.set_title("(a) FC3 singular value spectrum")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(svals))

    # Right: reconstruction error
    ax2.semilogy(R_vals, svd_err, "s-", color="#2ca02c", lw=2, ms=7,
                 label="SVD (Frobenius)")

    # PCP FC3 fitting errors from cached results
    pcp_ncs = []
    pcp_errs = []
    for Nc in [2, 4, 8, 12, 24]:
        key = f"pcp_Nc{Nc}"
        if key in results:
            rel_err = results[key].get("_info_rel_err", None)
            if rel_err is not None:
                pcp_ncs.append(Nc)
                pcp_errs.append(float(rel_err))
    if pcp_ncs:
        ax2.semilogy(pcp_ncs, pcp_errs, "^--", color=COLOR_PCP, lw=2, ms=7,
                     label="PCP (fitting)")

    ax2.set_xlabel("Rank $R$ / $N_c$", fontsize=12)
    ax2.set_ylabel("Relative FC3 error $\\|\\tilde\\Phi - \\Phi\\|_F / \\|\\Phi\\|_F$",
                    fontsize=12)
    ax2.set_title("(b) FC3 reconstruction error vs rank")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fc3_error_vs_rank.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fc3_error_vs_rank.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fc3_error_vs_rank.pdf")


def fig_transport_vs_rank(results):
    """Fig Q2: G_anh and transport error vs rank for both methods."""
    G_dense = _get_G(results["dense"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Collect separable data, sorted by rank numerically
    sep_data = []
    for key, val in results.items():
        if key.startswith("sep_R") and not key.startswith("sep_asr"):
            R = int(key.split("R")[1])
            sep_data.append((R, _get_G(val)))
    sep_data.sort(key=lambda x: x[0])
    sep_ranks = [d[0] for d in sep_data]
    sep_G = [d[1] for d in sep_data]

    # Collect PCP data, sorted by Nc numerically
    pcp_data = []
    for key, val in results.items():
        if key.startswith("pcp_Nc"):
            Nc = int(key.split("Nc")[1])
            pcp_data.append((Nc, _get_G(val)))
    pcp_data.sort(key=lambda x: x[0])
    pcp_ranks = [d[0] for d in pcp_data]
    pcp_G = [d[1] for d in pcp_data]

    # Left: absolute G_anh
    ax1.axhline(G_dense, color=COLOR_DENSE, lw=2, ls="-", label=f"Dense ({G_dense:.0f})")
    ax1.plot(sep_ranks, sep_G, "s-", color="#2ca02c", lw=1.5, ms=7, label="SVD")
    if pcp_ranks:
        ax1.plot(pcp_ranks, pcp_G, "o--", color=COLOR_PCP, lw=1.5, ms=7, label="PCP")

    ax1.set_xlabel("Rank $R$ / $N_c$", fontsize=12)
    ax1.set_ylabel("$G_\\mathrm{anh}$ (MW/m$^2$K)", fontsize=12)
    ax1.set_title("(a) Thermal conductance vs rank")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: relative transport error
    sep_err = [(g - G_dense) / G_dense * 100 for g in sep_G]
    pcp_err = [(g - G_dense) / G_dense * 100 for g in pcp_G]

    ax2.plot(sep_ranks, sep_err, "s-", color="#2ca02c", lw=1.5, ms=7, label="SVD")
    if pcp_ranks:
        ax2.plot(pcp_ranks, pcp_err, "o--", color=COLOR_PCP, lw=1.5, ms=7, label="PCP")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")

    # Shade ±5% band
    ax2.axhspan(-5, 5, alpha=0.1, color="green")
    x_max = max(sep_ranks[-1], pcp_ranks[-1]) if pcp_ranks else sep_ranks[-1]
    ax2.text(x_max * 0.98, 4.5, "±5%", ha="right",
             fontsize=9, color="green", alpha=0.7)

    ax2.set_xlabel("Rank $R$ / $N_c$", fontsize=12)
    ax2.set_ylabel("Transport error vs dense (%)", fontsize=12)
    ax2.set_title("(b) Transport error vs rank")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "transport_vs_rank.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "transport_vs_rank.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> transport_vs_rank.pdf")


def fig_spectral_current(results):
    """Fig Q3: Spectral heat current comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    r_dense = results["dense"]
    freqs = r_dense["freqs_thz"]

    # Left: absolute spectral current
    ax1.plot(freqs, r_dense["spectral_heat_current_ballistic"],
             ":", color="gray", lw=1, alpha=0.6, label="Ballistic")
    ax1.plot(freqs, r_dense["spectral_heat_current"],
             "-", color=COLOR_DENSE, lw=2, label="Dense")

    for key, label, color, ls in [
        ("sep_R2", "SVD $R=2$", "#77aa44", "-"),
        ("sep_R6", "SVD $R=6$", "#2ca02c", "-"),
        ("sep_R12", "SVD $R=12$", "#227722", "--"),
        ("sep_R24", "SVD $R=24$", "#115511", "--"),
        ("pcp_Nc2", "PCP $N_c=2$", "#aec7e8", "-."),
        ("pcp_Nc8", "PCP $N_c=8$", "#1f77b4", "-."),
        ("pcp_Nc24", "PCP $N_c=24$", "#0b4d8a", "-."),
    ]:
        if key in results:
            r = results[key]
            ax1.plot(r["freqs_thz"], r["spectral_heat_current"],
                     ls, color=color, lw=1.5, label=label)

    ax1.set_xlabel("Frequency (THz)", fontsize=12)
    ax1.set_ylabel("Spectral heat current (W/THz)", fontsize=12)
    ax1.set_title("(a) Spectral heat current")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(freqs[0], freqs[-1])

    # Right: difference from dense
    j_ref = r_dense["spectral_heat_current"]
    for key, label, color, ls in [
        ("sep_R2", "SVD $R=2$", "#77aa44", "-"),
        ("sep_R6", "SVD $R=6$", "#2ca02c", "-"),
        ("sep_R12", "SVD $R=12$", "#227722", "--"),
        ("sep_R24", "SVD $R=24$", "#115511", "--"),
        ("pcp_Nc2", "PCP $N_c=2$", "#aec7e8", "-."),
        ("pcp_Nc8", "PCP $N_c=8$", "#1f77b4", "-."),
        ("pcp_Nc24", "PCP $N_c=24$", "#0b4d8a", "-."),
    ]:
        if key in results:
            r = results[key]
            diff = r["spectral_heat_current"] - j_ref
            ax2.plot(r["freqs_thz"], diff, ls, color=color, lw=1.5, label=label)

    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_xlabel("Frequency (THz)", fontsize=12)
    ax2.set_ylabel("$\\Delta J(\\omega)$ vs dense (W/THz)", fontsize=12)
    ax2.set_title("(b) Spectral current difference from dense")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(freqs[0], freqs[-1])

    fig.tight_layout()
    fig.savefig(OUT_DIR / "spectral_current.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "spectral_current.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> spectral_current.pdf")


def fig_self_energy_comparison(results):
    """Fig Q4: Self-energy broadening comparison (Im Sigma^R trace)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    plot_configs = [
        ("dense", "Dense", COLOR_DENSE, "-", 2.0),
        ("sep_R6", "SVD $R=6$", "#2ca02c", "-", 1.5),
        ("sep_R12", "SVD $R=12$", "#227722", "--", 1.5),
        ("sep_R24", "SVD $R=24$", "#115511", "--", 1.5),
        ("pcp_Nc8", "PCP $N_c=8$", COLOR_PCP, "-.", 1.5),
        ("pcp_Nc24", "PCP $N_c=24$", "#0b4d8a", "-.", 1.5),
    ]

    traces = {}
    for key, label, color, ls, lw in plot_configs:
        if key not in results:
            continue
        r = results[key]
        Sigma_R = r.get("self_energy_retarded", None)
        if Sigma_R is None:
            continue
        freqs = r["freqs_thz"]

        # Sigma_R may be (n_slabs, n_kpts, n_freq, n_dof, n_dof) or
        # (n_kpts, n_freq, n_dof, n_dof)
        if Sigma_R.ndim == 5:
            # (n_slabs, n_kpts, n_freq, n_dof, n_dof) — average over slabs and kpts
            trace = -np.imag(np.trace(Sigma_R, axis1=-2, axis2=-1))
            trace_avg = np.mean(trace, axis=(0, 1))
            n_dof = Sigma_R.shape[-1]
        elif Sigma_R.ndim == 4:
            trace = -np.imag(np.trace(Sigma_R, axis1=-2, axis2=-1))
            trace_avg = np.mean(trace, axis=0)
            n_dof = Sigma_R.shape[-1]
        elif Sigma_R.ndim == 3:
            trace_avg = -np.imag(np.trace(Sigma_R, axis1=-2, axis2=-1))
            n_dof = Sigma_R.shape[-1]
        else:
            continue

        broadening = trace_avg / n_dof
        traces[key] = broadening
        ax1.plot(freqs, broadening, ls, color=color, lw=lw, label=label)

    ax1.set_xlabel("Frequency (THz)", fontsize=12)
    ax1.set_ylabel(r"$-\mathrm{Tr}[\mathrm{Im}\,\Sigma^R] / n_\mathrm{dof}$ (THz$^2$)",
                    fontsize=12)
    ax1.set_title("(a) Self-energy broadening")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: relative self-energy error (Frobenius norm per frequency)
    if "dense" in traces:
        Sigma_ref = results["dense"].get("self_energy_retarded", None)
        if Sigma_ref is not None:
            freqs = results["dense"]["freqs_thz"]
            # Compute Frobenius norm per frequency of reference
            if Sigma_ref.ndim == 5:
                Sigma_ref_avg = np.mean(Sigma_ref, axis=(0, 1))  # (n_freq, n_dof, n_dof)
            elif Sigma_ref.ndim == 4:
                Sigma_ref_avg = np.mean(Sigma_ref, axis=0)
            else:
                Sigma_ref_avg = Sigma_ref

            ref_norm = np.sqrt(np.sum(np.abs(Sigma_ref_avg)**2, axis=(-2, -1)))
            ref_norm = np.maximum(ref_norm, 1e-30)

            for key, label, color, ls, lw in plot_configs[1:]:
                if key not in results:
                    continue
                Sigma = results[key].get("self_energy_retarded", None)
                if Sigma is None:
                    continue
                if Sigma.ndim == 5:
                    Sigma_avg = np.mean(Sigma, axis=(0, 1))
                elif Sigma.ndim == 4:
                    Sigma_avg = np.mean(Sigma, axis=0)
                else:
                    Sigma_avg = Sigma

                diff_norm = np.sqrt(np.sum(np.abs(Sigma_avg - Sigma_ref_avg)**2,
                                           axis=(-2, -1)))
                rel_err = diff_norm / ref_norm
                ax2.plot(freqs, rel_err * 100, ls, color=color, lw=lw, label=label)

            ax2.set_xlabel("Frequency (THz)", fontsize=12)
            ax2.set_ylabel(r"$\|\Sigma - \Sigma_\mathrm{dense}\|_F / \|\Sigma_\mathrm{dense}\|_F$ (%)",
                           fontsize=12)
            ax2.set_title("(b) Relative self-energy error vs frequency")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "self_energy_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "self_energy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> self_energy_comparison.pdf")


def fig_conservation_and_timing(results):
    """Fig Q5: Conservation error and wall time comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    methods = []
    labels = []
    conserv = []
    g_anh = []
    colors = []
    times = []

    order = (["dense"]
             + [f"sep_R{R}" for R in [2, 4, 6, 12, 24]]
             + [f"pcp_Nc{Nc}" for Nc in [2, 4, 8, 12, 24]])

    label_map = {
        "dense": "Dense",
        "sep_R2": "SVD R=2", "sep_R4": "SVD R=4", "sep_R6": "SVD R=6",
        "sep_R12": "SVD R=12", "sep_R24": "SVD R=24",
        "pcp_Nc2": "PCP Nc=2", "pcp_Nc4": "PCP Nc=4", "pcp_Nc8": "PCP Nc=8",
        "pcp_Nc12": "PCP Nc=12", "pcp_Nc24": "PCP Nc=24",
    }

    for key in order:
        if key not in results:
            continue
        r = results[key]
        methods.append(key)
        labels.append(label_map.get(key, key))
        conserv.append(_get_conservation(r) * 100)  # as percentage
        g_anh.append(_get_G(r))
        times.append(float(r.get("wall_time", np.nan)))
        if key.startswith("pcp"):
            colors.append(COLOR_PCP)
        elif key.startswith("sep"):
            colors.append("#2ca02c")
        else:
            colors.append(COLOR_DENSE)

    # Left: conservation error
    x = np.arange(len(methods))
    bars = ax1.bar(x, conserv, color=colors, alpha=0.8, edgecolor="gray", lw=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Heat flow conservation error (%)", fontsize=12)
    ax1.set_title("(a) Conservation error")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, conserv):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    # Right: G_anh vs wall time (Pareto plot)
    G_dense = g_anh[0] if methods[0] == "dense" else None
    for i, key in enumerate(methods):
        marker = "o" if key == "dense" else ("^" if key.startswith("pcp") else "s")
        ax2.scatter(times[i], g_anh[i], c=colors[i], s=80, marker=marker,
                    edgecolors="gray", lw=0.5, zorder=5)
        ax2.annotate(labels[i], (times[i], g_anh[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

    if G_dense is not None:
        ax2.axhline(G_dense, color=COLOR_DENSE, lw=0.8, ls="--", alpha=0.5)
        ax2.axhspan(G_dense * 0.95, G_dense * 1.05, alpha=0.08, color="green")

    ax2.set_xlabel("Wall time (s)", fontsize=12)
    ax2.set_ylabel("$G_\\mathrm{anh}$ (MW/m$^2$K)", fontsize=12)
    ax2.set_title("(b) Accuracy vs cost (Pareto front)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "conservation_and_timing.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "conservation_and_timing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> conservation_and_timing.pdf")


def fig_scba_convergence(results):
    """Fig Q6: SCBA convergence (J vs iteration)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_configs = [
        ("dense", "Dense", COLOR_DENSE, "-", 2.0),
        ("sep_R6", "SVD $R=6$", "#2ca02c", "-", 1.5),
        ("sep_R12", "SVD $R=12$", "#227722", "--", 1.5),
        ("sep_R24", "SVD $R=24$", "#115511", "--", 1.5),
        ("pcp_Nc8", "PCP $N_c=8$", COLOR_PCP, "-.", 1.5),
        ("pcp_Nc24", "PCP $N_c=24$", "#0b4d8a", "-.", 1.5),
    ]

    for key, label, color, ls, lw in plot_configs:
        if key not in results:
            continue
        r = results[key]
        conv = r.get("convergence_history", None)
        if conv is None or len(conv) == 0:
            continue

        conv = np.asarray(conv)
        iters = np.arange(2, 2 + len(conv))  # starts at iter 2

        ax1.semilogy(iters, conv, f"{ls}", color=color, lw=lw, label=label,
                     marker=".", ms=4)

    ax1.axhline(COMMON["scba_tol"], color="gray", lw=0.8, ls="--", alpha=0.5)
    ax1.text(1.5, COMMON["scba_tol"] * 1.3, f"tol={COMMON['scba_tol']}", fontsize=8,
             color="gray")
    ax1.set_xlabel("SCBA iteration", fontsize=12)
    ax1.set_ylabel("Relative change in $J$", fontsize=12)
    ax1.set_title("(a) SCBA convergence")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, which="both")

    # Right: number of iterations to convergence
    methods_conv = []
    labels_conv = []
    n_iters = []
    colors_conv = []

    for key in (["dense"]
                + [f"sep_R{R}" for R in [2, 4, 6, 12, 24]]
                + [f"pcp_Nc{Nc}" for Nc in [2, 4, 8, 12, 24]]):
        if key not in results:
            continue
        r = results[key]
        n_it = r.get("n_scba_iterations", np.nan)
        methods_conv.append(key)
        n_iters.append(float(n_it))
        if key.startswith("pcp"):
            labels_conv.append(f"PCP Nc={key.split('Nc')[1]}")
            colors_conv.append(COLOR_PCP)
        elif key.startswith("sep"):
            labels_conv.append(f"SVD R={key.split('R')[1]}")
            colors_conv.append("#2ca02c")
        else:
            labels_conv.append("Dense")
            colors_conv.append(COLOR_DENSE)

    x = np.arange(len(methods_conv))
    ax2.bar(x, n_iters, color=colors_conv, alpha=0.8, edgecolor="gray", lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_conv, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("SCBA iterations", fontsize=12)
    ax2.set_title("(b) Iterations to convergence")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "scba_convergence.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "scba_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> scba_convergence.pdf")


def fig_error_decomposition(results):
    """Fig Q7: Decompose transport error into FC3 error and structural (projection) error."""
    fc3 = results["fc3_errors"]

    G_dense = _get_G(results["dense"])

    fig, ax = plt.subplots(figsize=(7, 5))

    # SVD: FC3 error vs transport error
    sep_fc3_err = []
    sep_trans_err = []
    sep_ranks = []
    for R in [2, 4, 6, 12, 24]:
        key = f"sep_R{R}"
        if key not in results:
            continue
        # Find matching FC3 error
        idx = np.where(fc3["R_values"] == R)[0]
        if len(idx) == 0:
            continue
        fc3_err = fc3["svd_errors"][idx[0]]
        trans_err = abs(_get_G(results[key]) - G_dense) / G_dense * 100
        sep_fc3_err.append(fc3_err * 100)
        sep_trans_err.append(trans_err)
        sep_ranks.append(R)

    ax.scatter(sep_fc3_err, sep_trans_err, c="#2ca02c", s=100, marker="s",
               edgecolors="gray", lw=0.5, zorder=5, label="Separable (SVD)")
    for i, R in enumerate(sep_ranks):
        ax.annotate(f"$R={R}$", (sep_fc3_err[i], sep_trans_err[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9,
                    color="#2ca02c")

    # PCP: FC3 error vs transport error (from _info_rel_err)
    pcp_fc3_err = []
    pcp_trans_err = []
    pcp_ncs = []
    for Nc in [2, 4, 8, 12, 24]:
        key = f"pcp_Nc{Nc}"
        if key not in results:
            continue
        r = results[key]
        rel_err = r.get("_info_rel_err", None)
        if rel_err is None:
            continue
        fc3_err = float(rel_err) * 100
        trans_err = abs(_get_G(r) - G_dense) / G_dense * 100
        pcp_fc3_err.append(fc3_err)
        pcp_trans_err.append(trans_err)
        pcp_ncs.append(Nc)

    if pcp_fc3_err:
        ax.scatter(pcp_fc3_err, pcp_trans_err, c=COLOR_PCP, s=100, marker="^",
                   edgecolors="gray", lw=0.5, zorder=5, label="PCP")
        for i, Nc in enumerate(pcp_ncs):
            ax.annotate(f"$N_c={Nc}$", (pcp_fc3_err[i], pcp_trans_err[i]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9,
                        color=COLOR_PCP)

    # Reference lines: y=x (error = FC3 error) and y=2x (quadratic amplification)
    all_fc3 = sep_fc3_err + pcp_fc3_err
    all_trans = sep_trans_err + pcp_trans_err
    max_err = max(max(all_fc3, default=30), max(all_trans, default=30)) * 1.2
    xx = np.linspace(0, max_err, 100)
    ax.plot(xx, xx, "k--", alpha=0.2, lw=1, label="$\\epsilon_G = \\delta_\\Phi$")
    ax.plot(xx, 2 * xx, "k:", alpha=0.2, lw=1, label="$\\epsilon_G = 2\\delta_\\Phi$")

    ax.set_xlabel("FC3 reconstruction error $\\delta_\\Phi$ (%)", fontsize=12)
    ax.set_ylabel("Transport error $|G - G_\\mathrm{dense}|/G_\\mathrm{dense}$ (%)",
                   fontsize=12)
    ax.set_title("FC3 error vs transport error")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "error_decomposition.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "error_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> error_decomposition.pdf")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = collect_all(load_only="--load" in sys.argv)

    print("\nGenerating quality figures...")
    fig_fc3_error_vs_rank(results)
    fig_transport_vs_rank(results)
    fig_spectral_current(results)
    fig_self_energy_comparison(results)
    fig_conservation_and_timing(results)
    fig_scba_convergence(results)
    fig_error_decomposition(results)

    # Print summary table
    G_dense = _get_G(results["dense"])
    print(f"\n{'Method':<18} {'G_anh':>8} {'Err%':>7} {'Conserv%':>9} {'Time(s)':>8}")
    print("-" * 54)
    for key in (["dense"]
                + [f"sep_R{R}" for R in [2, 4, 6, 12, 24]]
                + [f"pcp_Nc{Nc}" for Nc in [2, 4, 8, 12, 24]]):
        if key not in results:
            continue
        r = results[key]
        g = _get_G(r)
        err = (g - G_dense) / G_dense * 100
        cons = _get_conservation(r) * 100
        t = float(r.get("wall_time", np.nan))
        print(f"  {key:<16} {g:>8.1f} {err:>+7.1f} {cons:>9.2f} {t:>8.1f}")

    print(f"\nAll quality figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
