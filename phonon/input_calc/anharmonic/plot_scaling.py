"""Scaling benchmarks: dense vs separable vs PCP for large primitive cells.

Measures per-(q,q') pair wall times with synthetic random data across a
range of n_dof values.  Generates separate PDF figures for inclusion in
the LaTeX analysis.

Usage:
    python plot_scaling.py [--rerun]   # default: load cached data if available
    python plot_scaling.py --rerun     # force re-measurement

Memory notes:
  Dense kernel needs O(n_fft * n_dof^4 * 16) bytes per pair.
  n_dof=24 => ~1.2 GB.  We cap dense at n_dof=24.
  Separable/PCP scale much more gently and go up to n_dof=48.
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker

S3_PERMS = list(itertools.permutations(range(3)))
OUT_DIR = Path(__file__).resolve().parent / "figures"
DATA_FILE = Path(__file__).resolve().parent / "scaling_data.json"

# --------------------------------------------------------------------------
# Profiling kernels (synthetic random data, same structure as real kernels)
# --------------------------------------------------------------------------

def profile_dense(n_dof, n_fft, n_freq, n_rep=5):
    """Dense kernel: outer product + IFFT + two einsums."""
    rng = np.random.default_rng(0)
    GL_fft_qp = rng.standard_normal((n_fft, n_dof, n_dof)).astype(complex)
    GL_fft_qd = rng.standard_normal((n_fft, n_dof, n_dof)).astype(complex)
    Phi_L = rng.standard_normal((n_dof, n_dof, n_dof)).astype(complex)
    Phi_R = rng.standard_normal((n_dof, n_dof, n_dof)).astype(complex)
    Sigma = np.zeros((n_freq, n_dof, n_dof), dtype=complex)
    freq_sl = slice(0, n_freq)

    # Warmup
    product = GL_fft_qp[:, :, None, :, None] * GL_fft_qd[:, None, :, None, :]
    K = np.fft.ifft(product, axis=0)[freq_sl]
    temp = np.einsum('acd,wcdfe->wafe', Phi_L, K)
    Sigma += np.einsum('wafe,bef->wab', temp, Phi_R)

    t0 = time.perf_counter()
    for _ in range(n_rep):
        product = GL_fft_qp[:, :, None, :, None] * GL_fft_qd[:, None, :, None, :]
        K = np.fft.ifft(product, axis=0)[freq_sl]
        temp = np.einsum('acd,wcdfe->wafe', Phi_L, K)
        Sigma += np.einsum('wafe,bef->wab', temp, Phi_R)
    return (time.perf_counter() - t0) / n_rep


def profile_separable(n_dof, n_fft, n_freq, R, n_rep=10):
    """Separable kernel: two einsums + contraction + IFFT."""
    rng = np.random.default_rng(0)
    F_qp = rng.standard_normal((R, n_dof, n_dof)).astype(complex)
    F_diff_conj = rng.standard_normal((R, n_dof, n_dof)).astype(complex)
    UL_hat = rng.standard_normal((n_fft, n_dof, R)).astype(complex)
    VL_hat = rng.standard_normal((n_fft, n_dof, R)).astype(complex)
    Sigma = np.zeros((n_freq, n_dof, n_dof), dtype=complex)
    freq_sl = slice(0, n_freq)

    # Warmup
    M1 = np.einsum('rac,wdr->wacd', F_qp, UL_hat)
    M2 = np.einsum('wcs,sbd->wcbd', VL_hat, F_diff_conj)
    conv_hat = np.einsum('wacd,wcbd->wab', M1, M2)
    Sigma += np.fft.ifft(conv_hat, axis=0)[freq_sl]

    t0 = time.perf_counter()
    for _ in range(n_rep):
        M1 = np.einsum('rac,wdr->wacd', F_qp, UL_hat)
        M2 = np.einsum('wcs,sbd->wcbd', VL_hat, F_diff_conj)
        conv_hat = np.einsum('wacd,wcbd->wab', M1, M2)
        Sigma += np.fft.ifft(conv_hat, axis=0)[freq_sl]
    return (time.perf_counter() - t0) / n_rep


def profile_pcp(n_dof, n_fft, n_freq, N_c, n_rep=5):
    """PCP kernel: 36 products + IFFT + matmul accumulation."""
    rng = np.random.default_rng(0)
    gLp = rng.standard_normal((N_c, N_c, 3, 3, n_fft)).astype(complex)
    gLd = rng.standard_normal((N_c, N_c, 3, 3, n_fft)).astype(complex)
    f_modes = rng.standard_normal((3, N_c, n_dof)).astype(complex)
    lam_outer = rng.standard_normal((N_c, N_c))
    freq_sl = slice(0, n_freq)
    Sigma = np.zeros((n_freq, n_dof, n_dof), dtype=complex)

    # Warmup
    sum_prod = np.zeros((N_c, N_c, 3, 3, n_fft), dtype=complex)
    for s1, s2, s3 in S3_PERMS:
        for s1p, s2p, s3p in S3_PERMS:
            sum_prod[:, :, s1, s1p, :] += gLp[:, :, s2, s2p, :] * gLd[:, :, s3, s3p, :]
    conv_all = np.fft.ifft(sum_prod, axis=-1)[:, :, :, :, freq_sl]
    for s1 in range(3):
        for s1p in range(3):
            CL = (lam_outer[:, :, None] * conv_all[:, :, s1, s1p, :]).transpose(2, 0, 1)
            TL = CL @ np.conj(f_modes[s1p])
            Sigma += np.einsum('xa,wxy->way', f_modes[s1], TL)

    t0 = time.perf_counter()
    for _ in range(n_rep):
        sum_prod = np.zeros((N_c, N_c, 3, 3, n_fft), dtype=complex)
        for s1, s2, s3 in S3_PERMS:
            for s1p, s2p, s3p in S3_PERMS:
                sum_prod[:, :, s1, s1p, :] += gLp[:, :, s2, s2p, :] * gLd[:, :, s3, s3p, :]
        conv_all = np.fft.ifft(sum_prod, axis=-1)[:, :, :, :, freq_sl]
        for s1 in range(3):
            for s1p in range(3):
                CL = (lam_outer[:, :, None] * conv_all[:, :, s1, s1p, :]).transpose(2, 0, 1)
                TL = CL @ np.conj(f_modes[s1p])
                Sigma += np.einsum('xa,wxy->way', f_modes[s1], TL)
    return (time.perf_counter() - t0) / n_rep


def profile_pcp_parts(n_dof, n_fft, n_freq, N_c, n_rep=5):
    """Breakdown: products, IFFT, accumulation."""
    rng = np.random.default_rng(0)
    gLp = rng.standard_normal((N_c, N_c, 3, 3, n_fft)).astype(complex)
    gLd = rng.standard_normal((N_c, N_c, 3, 3, n_fft)).astype(complex)
    f_modes = rng.standard_normal((3, N_c, n_dof)).astype(complex)
    lam_outer = rng.standard_normal((N_c, N_c))
    freq_sl = slice(0, n_freq)

    # Products
    t0 = time.perf_counter()
    for _ in range(n_rep):
        sum_prod = np.zeros((N_c, N_c, 3, 3, n_fft), dtype=complex)
        for s1, s2, s3 in S3_PERMS:
            for s1p, s2p, s3p in S3_PERMS:
                sum_prod[:, :, s1, s1p, :] += gLp[:, :, s2, s2p, :] * gLd[:, :, s3, s3p, :]
    t_products = (time.perf_counter() - t0) / n_rep

    # IFFT
    t0 = time.perf_counter()
    for _ in range(n_rep):
        conv_all = np.fft.ifft(sum_prod, axis=-1)[:, :, :, :, freq_sl]
    t_ifft = (time.perf_counter() - t0) / n_rep

    # Accumulation (matmul)
    Sigma = np.zeros((n_freq, n_dof, n_dof), dtype=complex)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        for s1 in range(3):
            for s1p in range(3):
                CL = (lam_outer[:, :, None] * conv_all[:, :, s1, s1p, :]).transpose(2, 0, 1)
                TL = CL @ np.conj(f_modes[s1p])
                Sigma += np.einsum('xa,wxy->way', f_modes[s1], TL)
    t_accum = (time.perf_counter() - t0) / n_rep

    return t_products, t_ifft, t_accum


# --------------------------------------------------------------------------
# Memory estimates (analytical)
# --------------------------------------------------------------------------

def memory_dense_per_pair(n_dof, n_fft):
    """Peak memory for dense kernel per (q,q') pair in bytes."""
    # K tensor: n_fft * n_dof^4 * 16 (complex128)
    # Plus Phi_L, Phi_R: 2 * n_dof^3 * 16
    # Plus temp einsum: n_freq * n_dof^4 * 16 (similar to K)
    return n_fft * n_dof**4 * 16 + 2 * n_dof**3 * 16


def memory_separable_per_pair(n_dof, n_fft, R):
    """Peak memory for separable kernel."""
    # M1: n_fft * n_dof^2 * R * 16
    # M2: n_fft * n_dof^2 * R * 16
    # conv_hat: n_fft * n_dof^2 * 16
    # F arrays: 2 * R * n_dof^2 * 16
    return 2 * n_fft * n_dof**2 * R * 16 + n_fft * n_dof**2 * 16


def memory_pcp_per_pair(n_dof, n_fft, N_c):
    """Peak memory for PCP kernel (sum_prod + conv_all)."""
    # sum_prod: N_c^2 * 9 * n_fft * 16
    # conv_all: same
    # f_modes: 3 * N_c * n_dof * 16
    return 2 * N_c**2 * 9 * n_fft * 16 + 3 * N_c * n_dof * 16


def memory_pcp_precompute(n_dof, n_fft, N_c, n_kpts):
    """Memory for precomputed g arrays (global, not per-pair)."""
    # gL + gG: 2 * N_c^2 * 9 * n_kpts * n_fft * 16
    return 2 * N_c**2 * 9 * n_kpts * n_fft * 16


# --------------------------------------------------------------------------
# Theoretical operation counts
# --------------------------------------------------------------------------

def ops_dense(n_dof, n_fft, n_freq):
    """FLOPs for dense kernel per (q,q') pair."""
    outer = n_fft * n_dof**4          # outer product
    ifft = n_fft * n_dof**4 * np.log2(n_fft)  # IFFT
    e1 = n_freq * n_dof**5            # einsum 1
    e2 = n_freq * n_dof**4            # einsum 2
    return outer + ifft + e1 + e2


def ops_separable(n_dof, n_fft, n_freq, R):
    """FLOPs for separable kernel per pair."""
    m1 = n_fft * n_dof**2 * R * n_dof     # einsum M1
    m2 = n_fft * n_dof**2 * R * n_dof     # einsum M2
    contraction = n_fft * n_dof**2 * n_dof * R  # einsum contraction (approx)
    ifft = n_fft * n_dof**2 * np.log2(n_fft)
    return m1 + m2 + contraction + ifft


def ops_pcp(n_dof, n_fft, n_freq, N_c):
    """FLOPs for PCP kernel per pair."""
    products = 36 * N_c**2 * n_fft
    ifft = 9 * N_c**2 * n_fft * np.log2(n_fft)
    accum = 9 * N_c**2 * n_freq * n_dof   # matmul accumulation
    return products + ifft + accum


# --------------------------------------------------------------------------
# Data collection
# --------------------------------------------------------------------------

def collect_data():
    """Run all benchmarks and return results dict."""
    n_freq = 101
    n_low = 8
    n_fft = 2 * (n_low + n_freq)

    # n_dof values: 6 (Si 2-atom), 12 (4-atom), 18 (6-atom), 24 (8-atom),
    # 30 (10-atom), 36 (12-atom), 48 (16-atom)
    ndofs_all = [6, 12, 18, 24, 30, 36, 48]
    ndofs_dense = [m for m in ndofs_all if m <= 24]  # memory limit

    R_values = [6, 12, 24, 48]
    Nc_values = [2, 4, 8, 12, 24]

    data = {
        "n_freq": n_freq,
        "n_fft": n_fft,
        "ndofs_all": ndofs_all,
        "ndofs_dense": ndofs_dense,
        "R_values": R_values,
        "Nc_values": Nc_values,
        "dense": {},
        "separable": {},
        "pcp": {},
        "pcp_parts": {},
    }

    # Dense
    print("=== Dense ===")
    for m in ndofs_dense:
        n_rep = max(2, 20 // max(1, (m // 6) ** 2))
        print(f"  n_dof={m} (n_rep={n_rep})...", end=" ", flush=True)
        t = profile_dense(m, n_fft, n_freq, n_rep=n_rep)
        data["dense"][str(m)] = t
        print(f"{t*1000:.2f} ms")

    # Separable
    print("\n=== Separable ===")
    for m in ndofs_all:
        data["separable"][str(m)] = {}
        for R in R_values:
            n_rep = max(2, 20 // max(1, (m // 12)))
            print(f"  n_dof={m}, R={R} (n_rep={n_rep})...", end=" ", flush=True)
            t = profile_separable(m, n_fft, n_freq, R, n_rep=n_rep)
            data["separable"][str(m)][str(R)] = t
            print(f"{t*1000:.2f} ms")

    # PCP
    print("\n=== PCP ===")
    for m in ndofs_all:
        data["pcp"][str(m)] = {}
        for Nc in Nc_values:
            n_rep = max(2, 10 // max(1, (Nc // 8)))
            print(f"  n_dof={m}, N_c={Nc} (n_rep={n_rep})...", end=" ", flush=True)
            t = profile_pcp(m, n_fft, n_freq, Nc, n_rep=n_rep)
            data["pcp"][str(m)][str(Nc)] = t
            print(f"{t*1000:.2f} ms")

    # PCP breakdown (selected points)
    print("\n=== PCP breakdown ===")
    for m in [6, 12, 24, 48]:
        data["pcp_parts"][str(m)] = {}
        for Nc in [4, 8, 24]:
            n_rep = max(2, 10 // max(1, (Nc // 8)))
            print(f"  n_dof={m}, N_c={Nc}...", end=" ", flush=True)
            tp, ti, ta = profile_pcp_parts(m, n_fft, n_freq, Nc, n_rep=n_rep)
            data["pcp_parts"][str(m)][str(Nc)] = [tp, ti, ta]
            print(f"prod={tp*1000:.2f} ifft={ti*1000:.2f} accum={ta*1000:.2f} ms")

    return data


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

# Consistent style
COLORS = {
    "dense": "#d62728",
    "sep": "#2ca02c",
    "pcp": "#1f77b4",
}
SEP_MARKERS = {6: "s", 12: "D", 24: "^", 48: "v"}
PCP_MARKERS = {2: "o", 4: "s", 8: "D", 12: "^", 24: "v"}


def fig_perpair_cost(data):
    """Fig 1: Per-pair wall time vs n_dof for all methods."""
    n_fft = data["n_fft"]
    n_freq = data["n_freq"]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Dense (measured + extrapolated)
    ndofs_d = [int(m) for m in data["ndofs_dense"]]
    times_d = [data["dense"][str(m)] * 1000 for m in ndofs_d]
    ax.plot(ndofs_d, times_d, "o-", color=COLORS["dense"], lw=2, ms=7,
            label="Dense", zorder=5)

    # Dense theoretical extrapolation
    ndofs_ext = data["ndofs_all"]
    # Fit: t = c * ops_dense(m) => c from last measured point
    c_dense = times_d[-1] / ops_dense(ndofs_d[-1], n_fft, n_freq)
    t_dense_theory = [c_dense * ops_dense(m, n_fft, n_freq) for m in ndofs_ext]
    ax.plot(ndofs_ext, t_dense_theory, "--", color=COLORS["dense"], alpha=0.4,
            lw=1.5, label="Dense (extrap.)")

    # Separable: show R=6 and R=24
    for R, ls in [(6, "-"), (24, "--")]:
        ndofs = [int(m) for m in data["ndofs_all"]]
        times = [data["separable"][str(m)][str(R)] * 1000 for m in ndofs]
        ax.plot(ndofs, times, f"{SEP_MARKERS[R]}{ls}", color=COLORS["sep"],
                lw=1.5, ms=6, label=f"Separable $R={R}$")

    # PCP: show N_c=4, 8, 24
    for Nc, ls in [(4, "-"), (8, "--"), (24, ":")]:
        ndofs = [int(m) for m in data["ndofs_all"]]
        times = [data["pcp"][str(m)][str(Nc)] * 1000 for m in ndofs]
        ax.plot(ndofs, times, f"{PCP_MARKERS[Nc]}{ls}", color=COLORS["pcp"],
                lw=1.5, ms=6, label=f"PCP $N_c={Nc}$")

    ax.set_xlabel(r"$n_\mathrm{dof} = 3 \times n_\mathrm{atoms}$", fontsize=12)
    ax.set_ylabel("Wall time per (q,q') pair (ms)", fontsize=12)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(data["ndofs_all"])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title("Per-pair self-energy cost vs primitive cell size")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "perpair_cost_vs_ndof.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "perpair_cost_vs_ndof.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> perpair_cost_vs_ndof.pdf")


def fig_speedup_over_dense(data):
    """Fig 2: Speedup of PCP and separable over dense."""
    n_fft = data["n_fft"]
    n_freq = data["n_freq"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ndofs_d = [int(m) for m in data["ndofs_dense"]]

    # --- Left: speedup for measured n_dof ---
    # Separable
    for R in [6, 12, 24]:
        speedups = []
        for m in ndofs_d:
            t_d = data["dense"][str(m)]
            t_s = data["separable"][str(m)][str(R)]
            speedups.append(t_d / t_s)
        ax1.plot(ndofs_d, speedups, f"{SEP_MARKERS[R]}-", color=COLORS["sep"],
                 lw=1.5, ms=6, label=f"Sep $R={R}$")

    # PCP
    for Nc in [4, 8, 24]:
        speedups = []
        for m in ndofs_d:
            t_d = data["dense"][str(m)]
            t_p = data["pcp"][str(m)][str(Nc)]
            speedups.append(t_d / t_p)
        ax1.plot(ndofs_d, speedups, f"{PCP_MARKERS[Nc]}--", color=COLORS["pcp"],
                 lw=1.5, ms=6, label=f"PCP $N_c={Nc}$")

    ax1.axhline(1, color="gray", lw=0.8, ls="--")
    ax1.set_xlabel(r"$n_\mathrm{dof}$", fontsize=12)
    ax1.set_ylabel("Speedup over dense", fontsize=12)
    ax1.set_yscale("log")
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_title("(a) Measured speedup")

    # --- Right: theoretical speedup for large n_dof ---
    ndofs_theory = np.arange(6, 97, 3)

    for R in [6, 12, 24]:
        s = [ops_dense(m, n_fft, n_freq) / ops_separable(m, n_fft, n_freq, R)
             for m in ndofs_theory]
        ax2.plot(ndofs_theory, s, "-", color=COLORS["sep"], alpha=0.5 + 0.15 * (R / 8),
                 lw=1.5, label=f"Sep $R={R}$")

    for Nc in [4, 8, 24]:
        s = [ops_dense(m, n_fft, n_freq) / ops_pcp(m, n_fft, n_freq, Nc)
             for m in ndofs_theory]
        ax2.plot(ndofs_theory, s, "--", color=COLORS["pcp"], alpha=0.5 + 0.15 * (Nc / 8),
                 lw=1.5, label=f"PCP $N_c={Nc}$")

    ax2.axhline(1, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel(r"$n_\mathrm{dof}$", fontsize=12)
    ax2.set_ylabel("Theoretical speedup over dense", fontsize=12)
    ax2.set_yscale("log")
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_title("(b) Theoretical (operation count ratio)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "speedup_over_dense.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "speedup_over_dense.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> speedup_over_dense.pdf")


def fig_crossover_heatmap(data):
    """Fig 3: Heatmap — fastest method for each (n_dof, rank) combination."""
    n_fft = data["n_fft"]
    n_freq = data["n_freq"]

    ndofs = np.arange(6, 97, 3)
    ranks = np.arange(2, 49, 1)

    # For each (n_dof, rank): compute ops ratio PCP/dense and sep/dense
    # Use rank for both N_c (PCP) and R (separable)
    speedup_pcp = np.zeros((len(ndofs), len(ranks)))
    speedup_sep = np.zeros((len(ndofs), len(ranks)))

    for i, m in enumerate(ndofs):
        od = ops_dense(m, n_fft, n_freq)
        for j, r in enumerate(ranks):
            speedup_pcp[i, j] = od / ops_pcp(m, n_fft, n_freq, int(r))
            speedup_sep[i, j] = od / ops_separable(m, n_fft, n_freq, int(r))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # PCP speedup
    im1 = ax1.pcolormesh(ranks, ndofs, speedup_pcp, norm=LogNorm(vmin=0.1, vmax=1000),
                          cmap="RdYlGn", shading="auto")
    cs1 = ax1.contour(ranks, ndofs, speedup_pcp, levels=[1], colors="black", linewidths=2)
    ax1.clabel(cs1, fmt="1×")
    ax1.set_xlabel(r"PCP rank $N_c$", fontsize=12)
    ax1.set_ylabel(r"$n_\mathrm{dof}$", fontsize=12)
    ax1.set_title("(a) PCP speedup over dense")
    fig.colorbar(im1, ax=ax1, label="Speedup (op count ratio)")

    # Separable speedup
    im2 = ax2.pcolormesh(ranks, ndofs, speedup_sep, norm=LogNorm(vmin=0.1, vmax=1000),
                          cmap="RdYlGn", shading="auto")
    cs2 = ax2.contour(ranks, ndofs, speedup_sep, levels=[1], colors="black", linewidths=2)
    ax2.clabel(cs2, fmt="1×")
    ax2.set_xlabel(r"SVD rank $R$", fontsize=12)
    ax2.set_ylabel(r"$n_\mathrm{dof}$", fontsize=12)
    ax2.set_title("(b) Separable speedup over dense")
    fig.colorbar(im2, ax=ax2, label="Speedup (op count ratio)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "crossover_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "crossover_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> crossover_heatmap.pdf")


def fig_pcp_breakdown(data):
    """Fig 4: PCP cost breakdown (stacked bars) for different n_dof and N_c."""
    parts_data = data["pcp_parts"]

    ndofs = sorted(int(m) for m in parts_data.keys())
    Nc_vals = sorted(int(n) for n in next(iter(parts_data.values())).keys())

    fig, axes = plt.subplots(1, len(Nc_vals), figsize=(4 * len(Nc_vals), 5),
                              sharey=True)
    if len(Nc_vals) == 1:
        axes = [axes]

    bar_width = 0.6
    colors = ["#ff9999", "#66b3ff", "#99ff99"]
    labels = ["Perm. products", "Batch IFFT", "Matmul accum."]

    for ax_idx, Nc in enumerate(Nc_vals):
        ax = axes[ax_idx]
        prods = [parts_data[str(m)][str(Nc)][0] * 1000 for m in ndofs]
        iffts = [parts_data[str(m)][str(Nc)][1] * 1000 for m in ndofs]
        accum = [parts_data[str(m)][str(Nc)][2] * 1000 for m in ndofs]

        x = np.arange(len(ndofs))
        ax.bar(x, prods, bar_width, color=colors[0], label=labels[0])
        ax.bar(x, iffts, bar_width, bottom=prods, color=colors[1], label=labels[1])
        bottoms = [p + i for p, i in zip(prods, iffts)]
        ax.bar(x, accum, bar_width, bottom=bottoms, color=colors[2], label=labels[2])

        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in ndofs])
        ax.set_xlabel(r"$n_\mathrm{dof}$")
        ax.set_title(f"$N_c = {Nc}$")
        if ax_idx == 0:
            ax.set_ylabel("Wall time per pair (ms)")
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("PCP kernel cost breakdown by phase", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pcp_breakdown.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "pcp_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> pcp_breakdown.pdf")


def fig_memory_scaling(data):
    """Fig 5: Memory footprint vs n_dof."""
    n_fft = data["n_fft"]
    n_kpts = 16  # 4x4 mesh

    ndofs = np.array([6, 12, 18, 24, 30, 36, 48, 60, 72, 96])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: per-pair peak memory ---
    mem_dense = [memory_dense_per_pair(m, n_fft) / 1e6 for m in ndofs]
    ax1.plot(ndofs, mem_dense, "o-", color=COLORS["dense"], lw=2, label="Dense")

    for R in [6, 24]:
        mem = [memory_separable_per_pair(m, n_fft, R) / 1e6 for m in ndofs]
        ax1.plot(ndofs, mem, f"{SEP_MARKERS[R]}-", color=COLORS["sep"],
                 lw=1.5, label=f"Sep $R={R}$")

    for Nc in [4, 8, 24]:
        mem = [memory_pcp_per_pair(m, n_fft, Nc) / 1e6 for m in ndofs]
        ax1.plot(ndofs, mem, f"{PCP_MARKERS[Nc]}--", color=COLORS["pcp"],
                 lw=1.5, label=f"PCP $N_c={Nc}$")

    ax1.set_xlabel(r"$n_\mathrm{dof}$", fontsize=12)
    ax1.set_ylabel("Peak memory per pair (MB)", fontsize=12)
    ax1.set_yscale("log")
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_title("(a) Per-pair peak memory")

    # --- Right: PCP precompute memory (global g arrays) ---
    for Nc in [4, 8, 24]:
        mem = [memory_pcp_precompute(m, n_fft, Nc, n_kpts) / 1e9 for m in ndofs]
        ax2.plot(ndofs, mem, f"{PCP_MARKERS[Nc]}--", color=COLORS["pcp"],
                 lw=1.5, label=f"PCP $N_c={Nc}$")

    # Separable precompute: V/U arrays = 4 * n_kpts * n_fft * n_dof * R * 16
    for R in [6, 24]:
        mem = [4 * n_kpts * n_fft * m * R * 16 / 1e9 for m in ndofs]
        ax2.plot(ndofs, mem, f"{SEP_MARKERS[R]}-", color=COLORS["sep"],
                 lw=1.5, label=f"Sep $R={R}$")

    # Dense precompute: G_fft = 2 * n_kpts * n_fft * n_dof^2 * 16
    mem_d = [2 * n_kpts * n_fft * m**2 * 16 / 1e9 for m in ndofs]
    ax2.plot(ndofs, mem_d, "o-", color=COLORS["dense"], lw=2, label="Dense (G_fft)")

    ax2.set_xlabel(r"$n_\mathrm{dof}$", fontsize=12)
    ax2.set_ylabel("Precomputed array memory (GB)", fontsize=12)
    ax2.set_yscale("log")
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_title(f"(b) Precomputed arrays ($N_q = {n_kpts}$)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "memory_scaling.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "memory_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> memory_scaling.pdf")


def fig_pcp_scaling_nc(data):
    """Fig 6: PCP per-pair cost vs N_c for various n_dof."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ndof_colors = {6: "#1f77b4", 12: "#ff7f0e", 24: "#2ca02c", 48: "#d62728"}

    for m_str, nc_dict in sorted(data["pcp"].items(), key=lambda x: int(x[0])):
        m = int(m_str)
        if m not in ndof_colors:
            continue
        Ncs = sorted(int(n) for n in nc_dict.keys())
        times = [nc_dict[str(n)] * 1000 for n in Ncs]
        ax.plot(Ncs, times, "o-", color=ndof_colors[m], lw=1.5, ms=6,
                label=f"$n_\\mathrm{{dof}} = {m}$")

    # Theoretical N_c^2 scaling reference
    Ncs_ref = np.array([2, 4, 8, 12, 24])
    ref_line = 0.1 * (Ncs_ref / 2) ** 2
    ax.plot(Ncs_ref, ref_line, "k--", alpha=0.3, lw=1, label=r"$\propto N_c^2$")

    ax.set_xlabel(r"PCP rank $N_c$", fontsize=12)
    ax.set_ylabel("Wall time per pair (ms)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(r"PCP per-pair cost scaling with $N_c$")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "pcp_scaling_nc.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "pcp_scaling_nc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> pcp_scaling_nc.pdf")


def fig_sep_scaling_R(data):
    """Fig 7: Separable per-pair cost vs R for various n_dof."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ndof_colors = {6: "#1f77b4", 12: "#ff7f0e", 24: "#2ca02c", 48: "#d62728"}

    for m_str, r_dict in sorted(data["separable"].items(), key=lambda x: int(x[0])):
        m = int(m_str)
        if m not in ndof_colors:
            continue
        Rs = sorted(int(r) for r in r_dict.keys())
        times = [r_dict[str(r)] * 1000 for r in Rs]
        ax.plot(Rs, times, "s-", color=ndof_colors[m], lw=1.5, ms=6,
                label=f"$n_\\mathrm{{dof}} = {m}$")

    ax.set_xlabel(r"SVD rank $R$", fontsize=12)
    ax.set_ylabel("Wall time per pair (ms)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(r"Separable per-pair cost scaling with $R$")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sep_scaling_R.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "sep_scaling_R.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> sep_scaling_R.pdf")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rerun = "--rerun" in sys.argv

    if DATA_FILE.exists() and not rerun:
        print(f"Loading cached data from {DATA_FILE}")
        with open(DATA_FILE) as f:
            data = json.load(f)
    else:
        print("Running benchmarks (this will take a few minutes)...\n")
        data = collect_data()
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved benchmark data to {DATA_FILE}")

    print("\nGenerating figures...")
    fig_perpair_cost(data)
    fig_speedup_over_dense(data)
    fig_crossover_heatmap(data)
    fig_pcp_breakdown(data)
    fig_memory_scaling(data)
    fig_pcp_scaling_nc(data)
    fig_sep_scaling_R(data)

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
