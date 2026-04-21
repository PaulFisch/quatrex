"""FC3 approximation quality: transport error for the five canonical ansatze.

Drives the new :mod:`phonon_inputs.fc3_compression` module (mSVD, HOSVD, CP,
INDSCAL, Waring) and feeds each reconstruction back into
:func:`phonon_inputs.anharmonic.anharmonic_transmission_q` via
``M_stacked_override``, so every rank is evaluated under the same SCBA code
path as the dense reference.

Usage:
    python plot_quality.py                    # run all computations
    python plot_quality.py --load             # only regenerate plots from cache
    python plot_quality.py --hilbert          # Hilbert-transform retarded SE
    python plot_quality.py --hilbert --load   # plots from Hilbert cache
    python plot_quality.py --fc3-subdir fc3_prim      # switch FC3 dataset
"""

import argparse
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
from phonon_inputs.anharmonic import anharmonic_transmission_q
from phonon_inputs import fc3_compression as fc3c


# =========================================================================
# Configuration
# =========================================================================

# Methods and styles — naming follows the thesis (fc3_compression subsections).
METHODS = ["mSVD", "HOSVD", "CP", "INDSCAL", "Waring"]
COLORS = {
    "mSVD":    "#2ca02c",
    "HOSVD":   "#8c564b",
    "CP":      "#ff7f0e",
    "INDSCAL": "#1f77b4",
    "Waring":  "#9467bd",
}
MARKERS = {
    "mSVD":    "o",
    "HOSVD":   "v",
    "CP":      "s",
    "INDSCAL": "^",
    "Waring":  "D",
}
COLOR_DENSE = "#d62728"

# Rank sweep per method. HOSVD ranks are (R1, R2) tuples.
RANK_SWEEP = {
    "mSVD":    [4, 8, 12, 16, 24, 36],
    "HOSVD":   [(3, 4), (3, 8), (6, 8), (6, 16), (6, 24), (6, 36)],
    "CP":      [8, 16, 24, 36, 48],
    "INDSCAL": [8, 16, 24, 36, 48],
    "Waring":  [8, 16, 24, 36, 48, 64],
}

# Fit-time hyperparameters. Mirrors the defaults in compare_fc3_approximations.py
# so Frobenius errors reported here match the offline sweep.
FIT_KWARGS = {
    "mSVD":    {},
    "HOSVD":   {"refine": True, "hooi_iters": 12},
    "CP":      {"n_restarts": 10, "max_iter": 800, "lbfgs_iters": 500},
    "INDSCAL": {"n_restarts": 8,  "max_iter": 800, "lbfgs_iters": 600},
    "Waring":  {"n_restarts": 10, "n_power_repeats": 30,
                "n_power_iters": 300, "lbfgs_iters": 600},
}


TRANSPORT_KW = dict(
    q_mesh_transverse=(4, 4),
    freq_range_thz=(0.0, 18.0, 141),
    transport_direction="x",
    eta_factor=0.5,
    temperature=300.0,
    delta_T=10.0,
    anderson_mixing=True,
    anderson_depth=5,
    max_scba_iter=50,
    scba_tol=1e-10,
    mixing=0.2,
    n_slabs=1,
    verbose=True,
)


FIG_DIR = script_dir / "figures"
CACHE_DIR_BASE = script_dir / "quality_cache_new"
CACHE_DIR = CACHE_DIR_BASE  # overridden in main() when --hilbert


# =========================================================================
# Caching
# =========================================================================

def _rank_tag(rank) -> str:
    if isinstance(rank, tuple):
        return "_".join(str(r) for r in rank)
    return str(rank)


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
# Fit + reconstruct per method
# =========================================================================

def fit_and_reconstruct(method, rank, target):
    """Run the fc3_compression fitter and return (M_stacked, rel_err, fit_time, n_params)."""
    fitter = fc3c.FITTERS[method]
    kw = dict(FIT_KWARGS.get(method, {}))
    t0 = time.time()
    if method == "HOSVD":
        R1, R2 = rank
        res = fitter(target, R1=R1, R2=R2, **kw)
    else:
        res = fitter(target, rank=rank, **kw)
    fit_time = time.time() - t0

    T_approx = fc3c.reconstruct(method, res, target)  # (n_dof, dim_sc, dim_sc)
    M_stacked = T_approx.reshape(target.n_dof * target.dim_sc, target.dim_sc)
    return M_stacked, float(res.rel_err), fit_time, int(res.n_params)


# =========================================================================
# Data collection
# =========================================================================

def collect_all(fc3_subdir: str, load_only: bool = False):
    phonon, _ = load_primitive_cell(work_dir, fc3_subdir=fc3_subdir)
    fc3_path = work_dir / fc3_subdir / "fc3.hdf5"
    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    target = fc3c.build_fc3_target(fc3_raw, phonon)
    print(f"FC3 target: n_dof={target.n_dof}, dim_sc={target.dim_sc}, "
          f"||T||_F={target.target_norm:.4e}")

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
        res["n_params"] = target.n_dof * target.dim_sc ** 2
        save_result(res, "dense")
        results["dense"] = res
        print(f"  G_anh = {res['thermal_conductance_anharmonic']/1e6:.2f} "
              f"MW/(m^2 K)")

    # --- Per-method rank sweeps ---
    for method in METHODS:
        for rank in RANK_SWEEP[method]:
            key = f"{method}_R{_rank_tag(rank)}"
            cached = load_result(key)
            if cached is not None:
                print(f"[cached] {key}")
                results[key] = cached
                continue
            if load_only:
                continue

            print(f"\nRunning {method} R={rank}...")
            M_approx, frob_err, fit_time, n_params = fit_and_reconstruct(
                method, rank, target)

            t0 = time.time()
            res = anharmonic_transmission_q(
                phonon, M_stacked_override=M_approx, **TRANSPORT_KW)
            res["wall_time"] = time.time() - t0
            res["fit_time"] = fit_time
            res["frob_err"] = frob_err
            res["n_params"] = n_params
            save_result(res, key)
            results[key] = res

            G = res["thermal_conductance_anharmonic"]
            print(f"  {key}: frob={frob_err:.4e}, params={n_params}, "
                  f"G_anh={G/1e6:.2f} MW/(m^2K) "
                  f"({fit_time:.1f}s fit + {res['wall_time']:.0f}s transport)")

    return results


# =========================================================================
# Helpers
# =========================================================================

def _get(r, key, default=np.nan):
    v = r.get(key, default)
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return v.item()
    return float(v) if np.isscalar(v) else v


def _rank_to_scalar(rank):
    if isinstance(rank, tuple):
        return int(max(rank))
    return int(rank)


def _parse_rank(s: str):
    """Inverse of _rank_tag: 'R8' or 'R3_8'."""
    parts = s.split("_")
    if len(parts) == 1:
        return int(parts[0])
    return tuple(int(p) for p in parts)


def _method_data(results, method):
    prefix = f"{method}_R"
    pairs = []
    for key, val in results.items():
        if key.startswith(prefix):
            rank = _parse_rank(key[len(prefix):])
            pairs.append((rank, val))
    pairs.sort(key=lambda x: _rank_to_scalar(x[0]))
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
    G_dense = _get(results["dense"], "thermal_conductance_anharmonic") / 1e6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.axhline(G_dense, color=COLOR_DENSE, lw=2, ls="-",
                label=f"Dense ({G_dense:.0f})")
    ax1.axhspan(G_dense * 0.95, G_dense * 1.05, alpha=0.08, color="green")

    for method in METHODS:
        data = _method_data(results, method)
        if not data:
            continue
        ranks = [_rank_to_scalar(d[0]) for d in data]
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
        ranks = [_rank_to_scalar(d[0]) for d in data]
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
        ax1.loglog(n_params, frob, f"-{MARKERS[method]}", color=COLORS[method],
                   lw=1.5, ms=7, label=method)
        ax2.loglog(n_params, G_err, f"-{MARKERS[method]}", color=COLORS[method],
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
    G_dense = _get(results["dense"], "thermal_conductance_anharmonic")
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for method in METHODS:
        data = _method_data(results, method)
        if not data:
            continue
        frob = [_get(d[1], "frob_err") for d in data]
        G_err = [abs(_get(d[1], "thermal_conductance_anharmonic") - G_dense)
                 / abs(G_dense) for d in data]
        ax.loglog(frob, G_err, MARKERS[method], color=COLORS[method],
                  ms=9, label=method)
        for i, rank in enumerate(data):
            R = _rank_to_scalar(rank[0])
            ax.annotate(f"$R={R}$", (frob[i], G_err[i]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7,
                        color=COLORS[method])

    xx = np.logspace(-4, 0, 50)
    ax.plot(xx, xx, "k--", alpha=0.3, lw=1, label="1:1")
    ax.plot(xx, xx ** 2, "k:", alpha=0.3, lw=1, label=r"$\varepsilon^2$")
    ax.set_xlabel(r"Frobenius error $\|\Phi - \tilde\Phi\|_F / \|\Phi\|_F$", fontsize=12)
    ax.set_ylabel(r"Transport error $|G - G_{\mathrm{dense}}| / G_{\mathrm{dense}}$",
                  fontsize=12)
    ax.set_title("Frobenius vs transport error")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    _save(fig, "frob_vs_transport_error")


def fig_conservation(results):
    fig, ax = plt.subplots(figsize=(10, 5))

    labels, conserv, bar_colors = [], [], []

    if "dense" in results:
        labels.append("Dense")
        conserv.append(_get(results["dense"], "heat_flow_conservation") * 100)
        bar_colors.append(COLOR_DENSE)

    for method in METHODS:
        for rank, res in _method_data(results, method):
            labels.append(f"{method} R={_rank_tag(rank)}")
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
    fig, ax = plt.subplots(figsize=(7, 5))

    if "dense" in results:
        conv = results["dense"].get("convergence_history", None)
        if conv is not None and len(conv) > 0:
            conv = np.asarray(conv)
            ax.semilogy(np.arange(2, 2 + len(conv)), conv, "-", color=COLOR_DENSE,
                        lw=2, label="Dense", marker=".", ms=4)

    repr_ranks = {
        "mSVD":    16,
        "HOSVD":   (6, 16),
        "CP":      24,
        "INDSCAL": 24,
        "Waring":  24,
    }
    for method in METHODS:
        key = f"{method}_R{_rank_tag(repr_ranks[method])}"
        if key not in results:
            continue
        conv = results[key].get("convergence_history", None)
        if conv is None or len(conv) == 0:
            continue
        conv = np.asarray(conv)
        ax.semilogy(np.arange(2, 2 + len(conv)), conv, f"-{MARKERS[method]}",
                    color=COLORS[method], lw=1.5, ms=5,
                    label=f"{method} $R={_rank_tag(repr_ranks[method])}$")

    ax.axhline(TRANSPORT_KW["scba_tol"], color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("SCBA iteration", fontsize=12)
    ax.set_ylabel("Relative change in $J$", fontsize=12)
    ax.set_title("SCBA convergence")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    _save(fig, "scba_convergence")


def fig_spectral_current(results):
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
        ("mSVD",    16,      "-"),
        ("mSVD",    24,      "--"),
        ("HOSVD",   (6, 16), "-"),
        ("CP",      24,      "-"),
        ("INDSCAL", 24,      "-"),
        ("INDSCAL", 48,      "--"),
        ("Waring",  24,      "-"),
        ("Waring",  48,      "--"),
    ]
    for method, rank, ls in repr_configs:
        key = f"{method}_R{_rank_tag(rank)}"
        if key not in results:
            continue
        r = results[key]
        ax1.plot(r["freqs_thz"], r["spectral_heat_current"],
                 ls, color=COLORS[method], lw=1.2,
                 label=f"{method} $R={_rank_tag(rank)}$", alpha=0.85)

    ax1.set_xlabel("Frequency (THz)", fontsize=12)
    ax1.set_ylabel("Spectral heat current (W/THz)", fontsize=12)
    ax1.set_title("(a) Spectral heat current")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(freqs[0], freqs[-1])

    for method, rank, ls in repr_configs:
        key = f"{method}_R{_rank_tag(rank)}"
        if key not in results:
            continue
        r = results[key]
        ax2.plot(r["freqs_thz"], r["spectral_heat_current"] - J_ref,
                 ls, color=COLORS[method], lw=1.2,
                 label=f"{method} $R={_rank_tag(rank)}$", alpha=0.85)

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

    for method, rank in [("mSVD", 16), ("INDSCAL", 24), ("Waring", 24)]:
        key = f"{method}_R{_rank_tag(rank)}"
        if key not in results:
            continue
        r = results[key]
        J_L = r.get("spectral_heat_current_L", None)
        J_R = r.get("spectral_heat_current_R", None)
        if J_L is not None:
            ax1.plot(r["freqs_thz"], J_L, "-", color=COLORS[method], lw=1.2,
                     label=f"{method} R={_rank_tag(rank)} $J_L$", alpha=0.8)
            ax1.plot(r["freqs_thz"], J_R, "--", color=COLORS[method], lw=1.2,
                     label=f"{method} R={_rank_tag(rank)} $J_R$", alpha=0.8)

    ax1.set_xlabel("Frequency (THz)", fontsize=12)
    ax1.set_ylabel("Spectral heat current (W/THz)", fontsize=12)
    ax1.set_title("(a) Left vs right contact")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(freqs[0], freqs[-1])

    ax2.plot(freqs, J_L_dense - J_R_dense, "-", color=COLOR_DENSE, lw=1.5,
             label="Dense")
    for method, rank in [("mSVD", 16), ("INDSCAL", 24), ("Waring", 24)]:
        key = f"{method}_R{_rank_tag(rank)}"
        if key not in results:
            continue
        r = results[key]
        J_L = r.get("spectral_heat_current_L", None)
        J_R = r.get("spectral_heat_current_R", None)
        if J_L is not None:
            ax2.plot(r["freqs_thz"], J_L - J_R, f"-{MARKERS[method]}",
                     color=COLORS[method], lw=1.2, ms=3,
                     label=f"{method} $R={_rank_tag(rank)}$", alpha=0.8)

    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_xlabel("Frequency (THz)", fontsize=12)
    ax2.set_ylabel(r"$J_L(\omega) - J_R(\omega)$ (W/THz)", fontsize=12)
    ax2.set_title("(b) Spectral conservation")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(freqs[0], freqs[-1])

    fig.tight_layout()
    _save(fig, "spectral_current_lr")


def fig_hilbert_comparison():
    """Compare instantaneous vs Hilbert-transform retarded self-energy."""
    cache_plain = CACHE_DIR_BASE
    cache_hilb = CACHE_DIR_BASE.parent / (CACHE_DIR_BASE.name + "_hilbert")

    def _load_from(cache, name):
        p = cache / f"{name}.npz"
        if not p.exists():
            return None
        data = np.load(p, allow_pickle=True)
        return {k: (data[k].item() if data[k].ndim == 0 else data[k])
                for k in data.files}

    r_plain = _load_from(cache_plain, "dense")
    r_hilb = _load_from(cache_hilb, "dense")
    if r_plain is None or r_hilb is None:
        return

    freqs = r_plain["freqs_thz"]
    J_plain = r_plain["spectral_heat_current"]
    J_hilb = r_hilb["spectral_heat_current"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(freqs, r_plain["spectral_heat_current_ballistic"],
             ":", color="gray", lw=1, alpha=0.6, label="Ballistic")
    ax1.plot(freqs, J_plain, "-", color=COLOR_DENSE, lw=2,
             label=r"Dense (instant. $\Sigma^R$)")
    ax1.plot(freqs, J_hilb, "--", color="#1f77b4", lw=2,
             label=r"Dense (Hilbert $\Sigma^R$)")

    for method, rank, ls in [("INDSCAL", 24, "-"), ("mSVD", 16, "-")]:
        key = f"{method}_R{_rank_tag(rank)}"
        rp = _load_from(cache_plain, key)
        rh = _load_from(cache_hilb, key)
        if rp is not None:
            ax1.plot(rp["freqs_thz"], rp["spectral_heat_current"],
                     ls, color=COLORS[method], lw=1, alpha=0.6,
                     label=f"{method} R={_rank_tag(rank)} (inst.)")
        if rh is not None:
            ax1.plot(rh["freqs_thz"], rh["spectral_heat_current"],
                     "--", color=COLORS[method], lw=1, alpha=0.6,
                     label=f"{method} R={_rank_tag(rank)} (Hilb.)")

    ax1.set_xlabel("Frequency (THz)", fontsize=12)
    ax1.set_ylabel("Spectral heat current (W/THz)", fontsize=12)
    ax1.set_title(r"(a) Instantaneous vs Hilbert $\Sigma^R$")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(freqs[0], freqs[-1])

    ax2.plot(freqs, J_hilb - J_plain, "-", color="#1f77b4", lw=2, label="Dense")
    for method, rank in [("INDSCAL", 24), ("mSVD", 16)]:
        key = f"{method}_R{_rank_tag(rank)}"
        rp = _load_from(cache_plain, key)
        rh = _load_from(cache_hilb, key)
        if rp is not None and rh is not None:
            ax2.plot(rp["freqs_thz"],
                     rh["spectral_heat_current"] - rp["spectral_heat_current"],
                     f"-{MARKERS[method]}", color=COLORS[method], lw=1, ms=3,
                     label=f"{method} R={_rank_tag(rank)}")

    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_xlabel("Frequency (THz)", fontsize=12)
    ax2.set_ylabel(r"$\Delta J(\omega)$ (Hilbert $-$ instant.) (W/THz)",
                   fontsize=12)
    ax2.set_title(r"(b) Effect of Hilbert transform on $\Sigma^R$")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(freqs[0], freqs[-1])

    fig.tight_layout()
    _save(fig, "hilbert_comparison")


# =========================================================================
# Summary
# =========================================================================

def print_summary(results):
    G_dense = _get(results["dense"], "thermal_conductance_anharmonic")

    print(f"\n{'Method':<10} {'R':>8} {'Params':>8} {'Frob%':>8} "
          f"{'G [MW/m2K]':>12} {'G_err%':>8} {'Conserv%':>9} {'Time':>6}")
    print("-" * 78)

    c = _get(results["dense"], "heat_flow_conservation") * 100
    t = _get(results["dense"], "wall_time")
    print(f"{'Dense':<10} {'--':>8} {'--':>8} {'--':>8} "
          f"{G_dense/1e6:>12.2f} {'0.00':>8} {c:>9.2f} {t:>6.0f}")
    print("-" * 78)

    for method in METHODS:
        for rank, res in _method_data(results, method):
            G = _get(res, "thermal_conductance_anharmonic")
            frob = _get(res, "frob_err") * 100
            G_err = abs(G - G_dense) / abs(G_dense) * 100
            c = _get(res, "heat_flow_conservation") * 100
            t = _get(res, "wall_time")
            n_p = int(_get(res, "n_params"))
            print(f"{method:<10} {_rank_tag(rank):>8} {n_p:>8} {frob:>8.2f} "
                  f"{G/1e6:>12.2f} {G_err:>8.2f} {c:>9.2f} {t:>6.0f}")
        print("-" * 78)


# =========================================================================
# Main
# =========================================================================

def main():
    global CACHE_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", action="store_true",
                        help="regenerate plots from cache only")
    parser.add_argument("--hilbert", action="store_true",
                        help="use Hilbert-transform retarded self-energy")
    parser.add_argument("--fc3-subdir", default="fc3_prim_vasp",
                        help="FC3 dataset subdirectory under input_calc/ "
                             "(default: fc3_prim_vasp; use fc3_prim for 2x2x2)")
    args = parser.parse_args()

    if args.hilbert:
        CACHE_DIR = CACHE_DIR_BASE.parent / (CACHE_DIR_BASE.name + "_hilbert")
        TRANSPORT_KW["hilbert_retarded"] = True
        print("*** Hilbert-transform retarded self-energy enabled ***")

    print(f"FC3 dataset: {args.fc3_subdir}")
    print(f"Cache dir:   {CACHE_DIR}")

    results = collect_all(fc3_subdir=args.fc3_subdir, load_only=args.load)

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
    fig_hilbert_comparison()

    print_summary(results)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
