"""FC3 approximation quality: all four methods (SVD, PSCP, SCP3, FSCP).

Uses anharmonic_transmission_q from phonon_inputs with M_stacked_override
to run the same SCBA code path for all methods.

Usage:
    python plot_quality.py           # run all computations (slow)
    python plot_quality.py --load    # only regenerate plots from cache
"""

import sys
import time
from pathlib import Path

import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent  # input_calc/
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs.separable import (
    build_supercell_mapping,
    build_realspace_fc3_matrices,
)
from phonon_inputs.anharmonic import anharmonic_transmission_q
from phonon_inputs.pcp import fit_supercell_cp

from compare_fc3_approximations import (
    svd_approximation, svd_n_params,
    pscp_decomposition, pscp_reconstruct, pscp_n_params,
    scp3_reconstruct_M_stacked, scp3_n_params,
    fit_fscp, fscp_reconstruct_M_stacked, fscp_n_params,
)

FIG_DIR = script_dir / "figures"
CACHE_DIR = script_dir / "quality_cache"

TRANSPORT_KW = dict(
    q_mesh_transverse=(8, 8),
    freq_range_thz=(0.1, 18.0, 141),
    transport_direction="x",
    eta_factor=0.5,
    temperature=300.0,
    delta_T=10.0,
    max_scba_iter=100,
    scba_tol=0.005,
    mixing=0.3,
    n_slabs=1,
    verbose=True,
)

METHODS = ["SVD", "PSCP", "SCP3", "FSCP"]
COLORS = {"SVD": "#2ca02c", "PSCP": "#ff7f0e", "SCP3": "#1f77b4", "FSCP": "#9467bd"}
MARKERS = {"SVD": "o", "PSCP": "s", "SCP3": "^", "FSCP": "D"}
COLOR_DENSE = "#d62728"

# Ranks to sweep for each method
RANK_SWEEP = {
    "SVD": [4, 8, 12, 16, 24],
    "PSCP": [6, 12, 24, 36, 48],
    "SCP3": [4, 8, 16, 24],
    "FSCP": [8, 16, 24, 48],
}


# =========================================================================
# Caching
# =========================================================================

def save_result(result, name):
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
    np.savez(CACHE_DIR / f"{name}.npz", **data)


def load_result(name):
    path = CACHE_DIR / f"{name}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    result = {}
    for k in data.files:
        v = data[k]
        result[k] = v.item() if v.ndim == 0 else v
    return result


# =========================================================================
# M_stacked construction per method
# =========================================================================

def build_M_approx(method, rank, M_ref, n_dof, dim_sc,
                   d_pscp_full, v_pscp_full, fc3_raw, phonon):
    if method == "SVD":
        return svd_approximation(M_ref, rank)
    elif method == "PSCP":
        return pscp_reconstruct(d_pscp_full[:rank], v_pscp_full[:rank], n_dof, dim_sc)
    elif method == "SCP3":
        u_modes, lambdas, info = fit_supercell_cp(
            fc3_raw, phonon, N_c=rank, max_iter=2000, verbose=False)
        return scp3_reconstruct_M_stacked(
            u_modes, lambdas, phonon, info["target_norm"], n_dof, dim_sc)
    elif method == "FSCP":
        v_modes, lambdas, info = fit_fscp(
            fc3_raw, phonon, R=rank, max_iter=2000, verbose=False)
        return fscp_reconstruct_M_stacked(
            v_modes, lambdas, phonon, info["target_norm"], n_dof, dim_sc)


def n_params_for(method, rank, n_dof, dim_sc):
    if method == "SVD":
        return svd_n_params(rank, n_dof, dim_sc)
    elif method == "PSCP":
        return pscp_n_params(rank, n_dof, dim_sc)
    elif method == "SCP3":
        return scp3_n_params(rank, dim_sc)
    elif method == "FSCP":
        return fscp_n_params(rank, dim_sc)


# =========================================================================
# Data collection
# =========================================================================

def collect_all(load_only=False):
    phonon, _ = load_primitive_cell(work_dir)
    fc3_path = work_dir / "fc3_prim" / "fc3.hdf5"
    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    nat_prim = len(phonon.primitive.masses)
    n_super = len(phonon.supercell.masses)
    n_dof = 3 * nat_prim
    dim_sc = 3 * n_super
    masses_super = phonon.supercell.masses

    _, _, _, ref_sc_atoms = build_supercell_mapping(phonon)
    M_ref = build_realspace_fc3_matrices(fc3_raw, nat_prim, masses_super, ref_sc_atoms)
    M_norm = np.linalg.norm(M_ref, "fro")

    d_pscp_full, v_pscp_full, _ = pscp_decomposition(M_ref, n_dof, dim_sc)

    results = {}

    # --- Dense reference ---
    cached = load_result("dense")
    if cached is not None:
        print("[cached] dense")
        results["dense"] = cached
    elif not load_only:
        print("Running dense reference...")
        t0 = time.time()
        res = anharmonic_transmission_q(phonon, str(fc3_path), **TRANSPORT_KW)
        res["wall_time"] = time.time() - t0
        res["frob_err"] = 0.0
        res["n_params"] = n_dof * dim_sc ** 2
        save_result(res, "dense")
        results["dense"] = res
        print(f"  G_anh = {res['thermal_conductance_anharmonic']/1e6:.2f} MW/(m^2 K)")

    # --- Each method at various ranks ---
    for method in METHODS:
        for rank in RANK_SWEEP[method]:
            key = f"{method}_R{rank}"
            cached = load_result(key)
            if cached is not None:
                print(f"[cached] {key}")
                results[key] = cached
                continue
            if load_only:
                continue

            print(f"\nRunning {method} R={rank}...")
            t0_fit = time.time()
            M_approx = build_M_approx(
                method, rank, M_ref, n_dof, dim_sc,
                d_pscp_full, v_pscp_full, fc3_raw, phonon)
            frob_err = np.linalg.norm(M_ref - M_approx, "fro") / M_norm
            t_fit = time.time() - t0_fit

            t0 = time.time()
            res = anharmonic_transmission_q(
                phonon, M_stacked_override=M_approx, **TRANSPORT_KW)
            res["wall_time"] = time.time() - t0
            res["fit_time"] = t_fit
            res["frob_err"] = frob_err
            res["n_params"] = n_params_for(method, rank, n_dof, dim_sc)
            save_result(res, key)
            results[key] = res

            G = res["thermal_conductance_anharmonic"]
            print(f"  {key}: frob={frob_err:.4e}, "
                  f"G_anh={G/1e6:.2f} MW/(m^2K) ({res['wall_time']:.0f}s)")

    return results


# =========================================================================
# Helpers
# =========================================================================

def _get(r, key, default=np.nan):
    v = r.get(key, default)
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return v.item()
    return float(v) if np.isscalar(v) else v


def _method_data(results, method):
    pairs = []
    for key, val in results.items():
        if key.startswith(f"{method}_R"):
            rank = int(key.split("R")[-1])
            pairs.append((rank, val))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}.pdf")


# =========================================================================
# Figures
# =========================================================================

def fig_ganh_absolute(results):
    """Absolute G_anh vs rank + relative transport error."""
    G_dense = _get(results["dense"], "thermal_conductance_anharmonic") / 1e6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.axhline(G_dense, color=COLOR_DENSE, lw=2, ls="-",
                label=f"Dense ({G_dense:.0f})")
    ax1.axhspan(G_dense * 0.95, G_dense * 1.05, alpha=0.08, color="green")

    for method in METHODS:
        data = _method_data(results, method)
        if not data:
            continue
        ranks = [d[0] for d in data]
        G_vals = [_get(d[1], "thermal_conductance_anharmonic") / 1e6 for d in data]
        ax1.plot(ranks, G_vals, f"-{MARKERS[method]}", color=COLORS[method],
                 lw=1.5, ms=7, label=method)

    ax1.set_xlabel("Rank $R$", fontsize=12)
    ax1.set_ylabel(r"$G_{\mathrm{anh}}$ (MW/m$^2$K)", fontsize=12)
    ax1.set_title(r"(a) Thermal conductance vs rank")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    for method in METHODS:
        data = _method_data(results, method)
        if not data:
            continue
        ranks = [d[0] for d in data]
        errs = [abs(_get(d[1], "thermal_conductance_anharmonic") / 1e6 - G_dense)
                / G_dense * 100 for d in data]
        ax2.semilogy(ranks, errs, f"-{MARKERS[method]}", color=COLORS[method],
                     lw=1.5, ms=7, label=method)

    ax2.set_xlabel("Rank $R$", fontsize=12)
    ax2.set_ylabel(r"$|G - G_{\mathrm{dense}}| / G_{\mathrm{dense}}$ (%)", fontsize=12)
    ax2.set_title("(b) Relative transport error vs rank")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    _save(fig, "ganh_vs_rank")


def fig_error_vs_params(results):
    """Frobenius error and transport error vs number of parameters."""
    G_dense = _get(results["dense"], "thermal_conductance_anharmonic")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for method in METHODS:
        data = _method_data(results, method)
        if not data:
            continue
        n_params = [_get(d[1], "n_params") for d in data]
        frob = [_get(d[1], "frob_err") for d in data]
        G_err = [abs(_get(d[1], "thermal_conductance_anharmonic") - G_dense)
                 / abs(G_dense) for d in data]
        ax1.semilogy(n_params, frob, f"-{MARKERS[method]}", color=COLORS[method],
                     lw=1.5, ms=7, label=method)
        ax2.semilogy(n_params, G_err, f"-{MARKERS[method]}", color=COLORS[method],
                     lw=1.5, ms=7, label=method)

    ax1.set_xlabel("Number of parameters", fontsize=12)
    ax1.set_ylabel(r"$\|\Phi - \tilde\Phi\|_F / \|\Phi\|_F$", fontsize=12)
    ax1.set_title("(a) Frobenius error vs parameters")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, which="both")

    ax2.set_xlabel("Number of parameters", fontsize=12)
    ax2.set_ylabel(r"$|G - G_{\mathrm{dense}}| / G_{\mathrm{dense}}$", fontsize=12)
    ax2.set_title("(b) Transport error vs parameters")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    _save(fig, "error_vs_params")


def fig_frob_vs_transport(results):
    """Frobenius error vs transport error correlation."""
    G_dense = _get(results["dense"], "thermal_conductance_anharmonic")
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for method in METHODS:
        data = _method_data(results, method)
        if not data:
            continue
        frob = [_get(d[1], "frob_err") for d in data]
        G_err = [abs(_get(d[1], "thermal_conductance_anharmonic") - G_dense)
                 / abs(G_dense) for d in data]
        ranks = [d[0] for d in data]
        ax.loglog(frob, G_err, MARKERS[method], color=COLORS[method],
                  ms=9, label=method)
        for i, R in enumerate(ranks):
            ax.annotate(f"$R={R}$", (frob[i], G_err[i]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7,
                        color=COLORS[method])

    xx = np.logspace(-4, 0, 50)
    ax.plot(xx, xx, "k--", alpha=0.3, lw=1, label="1:1")
    ax.set_xlabel(r"Frobenius error $\|\Phi - \tilde\Phi\|_F / \|\Phi\|_F$", fontsize=12)
    ax.set_ylabel(r"Transport error $|G - G_{\mathrm{dense}}| / G_{\mathrm{dense}}$",
                  fontsize=12)
    ax.set_title("Frobenius vs transport error")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    _save(fig, "frob_vs_transport_error")


def fig_conservation(results):
    """Conservation error bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    labels, conserv, bar_colors = [], [], []

    if "dense" in results:
        labels.append("Dense")
        conserv.append(_get(results["dense"], "heat_flow_conservation") * 100)
        bar_colors.append(COLOR_DENSE)

    for method in METHODS:
        for rank, res in _method_data(results, method):
            labels.append(f"{method} R={rank}")
            conserv.append(_get(res, "heat_flow_conservation") * 100)
            bar_colors.append(COLORS[method])

    x = np.arange(len(labels))
    bars = ax.bar(x, conserv, color=bar_colors, alpha=0.8, edgecolor="gray", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Conservation error (%)", fontsize=12)
    ax.set_title("Heat flow conservation error")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, conserv):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=6)
    fig.tight_layout()
    _save(fig, "conservation_error")


def fig_scba_convergence(results):
    """SCBA convergence history."""
    fig, ax = plt.subplots(figsize=(7, 5))

    if "dense" in results:
        conv = results["dense"].get("convergence_history", None)
        if conv is not None and len(conv) > 0:
            conv = np.asarray(conv)
            ax.semilogy(np.arange(2, 2 + len(conv)), conv, "-", color=COLOR_DENSE,
                        lw=2, label="Dense", marker=".", ms=4)

    repr_ranks = {"SVD": 8, "PSCP": 24, "SCP3": 8, "FSCP": 16}
    for method in METHODS:
        key = f"{method}_R{repr_ranks[method]}"
        if key not in results:
            continue
        conv = results[key].get("convergence_history", None)
        if conv is None or len(conv) == 0:
            continue
        conv = np.asarray(conv)
        ax.semilogy(np.arange(2, 2 + len(conv)), conv, f"-{MARKERS[method]}",
                    color=COLORS[method], lw=1.5, ms=5,
                    label=f"{method} $R={repr_ranks[method]}$")

    ax.axhline(TRANSPORT_KW["scba_tol"], color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("SCBA iteration", fontsize=12)
    ax.set_ylabel("Relative change in $J$", fontsize=12)
    ax.set_title("SCBA convergence")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    _save(fig, "scba_convergence")


def fig_spectral_current(results):
    """Spectral heat current: absolute and difference from dense."""
    if "dense" not in results:
        return

    r_dense = results["dense"]
    freqs = r_dense["freqs_thz"]
    J_ref = r_dense["spectral_heat_current"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(freqs, r_dense["spectral_heat_current_ballistic"],
             ":", color="gray", lw=1, alpha=0.6, label="Ballistic")
    ax1.plot(freqs, J_ref, "-", color=COLOR_DENSE, lw=2, label="Dense")

    repr_configs = [
        ("SVD", 8, "-"), ("SVD", 24, "--"),
        ("PSCP", 24, "-"),
        ("SCP3", 8, "-"), ("SCP3", 24, "--"),
        ("FSCP", 16, "-"), ("FSCP", 48, "--"),
    ]
    for method, rank, ls in repr_configs:
        key = f"{method}_R{rank}"
        if key not in results:
            continue
        r = results[key]
        ax1.plot(r["freqs_thz"], r["spectral_heat_current"],
                 ls, color=COLORS[method], lw=1.2,
                 label=f"{method} $R={rank}$", alpha=0.85)

    ax1.set_xlabel("Frequency (THz)", fontsize=12)
    ax1.set_ylabel("Spectral heat current (W/THz)", fontsize=12)
    ax1.set_title("(a) Spectral heat current")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(freqs[0], freqs[-1])

    for method, rank, ls in repr_configs:
        key = f"{method}_R{rank}"
        if key not in results:
            continue
        r = results[key]
        ax2.plot(r["freqs_thz"], r["spectral_heat_current"] - J_ref,
                 ls, color=COLORS[method], lw=1.2,
                 label=f"{method} $R={rank}$", alpha=0.85)

    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_xlabel("Frequency (THz)", fontsize=12)
    ax2.set_ylabel(r"$\Delta J(\omega)$ vs dense (W/THz)", fontsize=12)
    ax2.set_title("(b) Spectral current difference")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(freqs[0], freqs[-1])

    fig.tight_layout()
    _save(fig, "spectral_current")


def fig_spectral_current_lr(results):
    """Left vs right contact spectral currents."""
    if "dense" not in results:
        return

    r_dense = results["dense"]
    freqs = r_dense["freqs_thz"]
    J_L_dense = r_dense.get("spectral_heat_current_L", None)
    J_R_dense = r_dense.get("spectral_heat_current_R", None)
    if J_L_dense is None or J_R_dense is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(freqs, J_L_dense, "-", color=COLOR_DENSE, lw=1.5, label="Dense $J_L$")
    ax1.plot(freqs, J_R_dense, "--", color=COLOR_DENSE, lw=1.5, label="Dense $J_R$")

    for method, rank in [("SCP3", 8), ("SVD", 8)]:
        key = f"{method}_R{rank}"
        if key not in results:
            continue
        r = results[key]
        J_L = r.get("spectral_heat_current_L", None)
        J_R = r.get("spectral_heat_current_R", None)
        if J_L is not None:
            ax1.plot(r["freqs_thz"], J_L, "-", color=COLORS[method], lw=1.2,
                     label=f"{method} R={rank} $J_L$", alpha=0.8)
            ax1.plot(r["freqs_thz"], J_R, "--", color=COLORS[method], lw=1.2,
                     label=f"{method} R={rank} $J_R$", alpha=0.8)

    ax1.set_xlabel("Frequency (THz)", fontsize=12)
    ax1.set_ylabel("Spectral heat current (W/THz)", fontsize=12)
    ax1.set_title("(a) Left vs right contact")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(freqs[0], freqs[-1])

    ax2.plot(freqs, J_L_dense - J_R_dense, "-", color=COLOR_DENSE, lw=1.5, label="Dense")
    for method, rank in [("SVD", 8), ("SCP3", 8), ("FSCP", 16)]:
        key = f"{method}_R{rank}"
        if key not in results:
            continue
        r = results[key]
        J_L = r.get("spectral_heat_current_L", None)
        J_R = r.get("spectral_heat_current_R", None)
        if J_L is not None:
            ax2.plot(r["freqs_thz"], J_L - J_R, f"-{MARKERS[method]}",
                     color=COLORS[method], lw=1.2, ms=3,
                     label=f"{method} $R={rank}$", alpha=0.8)

    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_xlabel("Frequency (THz)", fontsize=12)
    ax2.set_ylabel(r"$J_L(\omega) - J_R(\omega)$ (W/THz)", fontsize=12)
    ax2.set_title("(b) Spectral conservation")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(freqs[0], freqs[-1])

    fig.tight_layout()
    _save(fig, "spectral_current_lr")


# =========================================================================
# Summary
# =========================================================================

def print_summary(results):
    G_dense = _get(results["dense"], "thermal_conductance_anharmonic")

    print(f"\n{'Method':<10} {'R':>4} {'Params':>8} {'Frob%':>8} "
          f"{'G [MW/m2K]':>12} {'G_err%':>8} {'Conserv%':>9} {'Time':>6}")
    print("-" * 72)

    c = _get(results["dense"], "heat_flow_conservation") * 100
    t = _get(results["dense"], "wall_time")
    print(f"{'Dense':<10} {'--':>4} {'--':>8} {'--':>8} "
          f"{G_dense/1e6:>12.2f} {'0.00':>8} {c:>9.2f} {t:>6.0f}")
    print("-" * 72)

    for method in METHODS:
        for rank, res in _method_data(results, method):
            G = _get(res, "thermal_conductance_anharmonic")
            frob = _get(res, "frob_err") * 100
            G_err = abs(G - G_dense) / abs(G_dense) * 100
            c = _get(res, "heat_flow_conservation") * 100
            t = _get(res, "wall_time")
            n_p = int(_get(res, "n_params"))
            print(f"{method:<10} {rank:>4} {n_p:>8} {frob:>8.2f} "
                  f"{G/1e6:>12.2f} {G_err:>8.2f} {c:>9.2f} {t:>6.0f}")
        print("-" * 72)


# =========================================================================
# Main
# =========================================================================

def main():
    results = collect_all(load_only="--load" in sys.argv)

    if "dense" not in results:
        print("ERROR: no dense reference. Run without --load first.")
        return

    print("\nGenerating figures...")
    fig_ganh_absolute(results)
    fig_error_vs_params(results)
    fig_frob_vs_transport(results)
    fig_conservation(results)
    fig_scba_convergence(results)
    fig_spectral_current(results)
    fig_spectral_current_lr(results)

    print_summary(results)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
