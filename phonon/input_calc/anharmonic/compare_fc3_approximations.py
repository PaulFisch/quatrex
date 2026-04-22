"""Compare FC3 compression ansatze and report error vs parameter count.

Thin driver around :mod:`phonon_inputs.fc3_compression`.  Fits every requested
(method, rank) pair to the mass-weighted FC3 read from the selected dataset
(``fc3_prim``, ``fc3_prim_vasp``, ...) and produces diagnostic plots:

  1. error vs number of parameters
  2. error vs rank
  3. mSVD singular-value spectrum
  4. HOSVD mode-1 and mode-(2,3) spectra

Methods (naming follows the thesis text):
  mSVD     — truncated matricization SVD
  HOSVD    — S2-symmetric Tucker (HOSVD + HOOI refinement)
  CP       — unconstrained canonical polyadic (tensorly ALS + L-BFGS)
  INDSCAL  — CP with internal-leg symmetry
  Waring   — symmetric CP on the S3-lifted tensor
  PCP      — permanent CP of Luo et al. 2025

Variants: each fit is run twice, once unconstrained and once with
``enforce_asr=True`` (null-space reparametrisation of the supercell-axis
factor).  ASR-enforced variants use the cache suffix ``_ASR`` and appear on
plots with a dashed linestyle at 75% alpha.

Performance notes
-----------------
The Waring defaults have been relaxed from the original ``(30, 300, 600)`` to
``(10, 200, 400)``; inner power iterations dominated wall time without
measurable gain on the L-BFGS-refined output.  For ranks above 32 we also
initialise CP / INDSCAL / Waring from the previous rank result plus a
residual-leading rank-1 column (progressive warm-starts), which removes the
O(n_restarts) cold-start cost at high rank — the standard "folding" trick
from the tensor-decomposition literature.

Each fit is cached to
``anharmonic/quality_cache/<fc3_subdir>/fc3_compression_*.npz``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent  # input_calc/
sys.path.insert(0, str(work_dir))

from run_anharmonic import load_primitive_cell
from phonon_inputs import fc3_compression as fc3c


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


DEFAULT_RANKS = {
    "mSVD":    [1, 2, 4, 8, 12, 16, 24, 36, 48],
    "HOSVD":   [(3, 4), (3, 8), (6, 8), (6, 16), (6, 24), (6, 36)],
    "CP":      [2, 4, 8, 16, 24, 36, 48],
    "INDSCAL": [2, 4, 8, 16, 24, 36, 48],
    "Waring":  [4, 8, 16, 24, 36, 48, 64, 96],
    "PCP":     [2, 4, 8, 16, 24],
}


METHOD_STYLE = {
    "mSVD":    {"color": "C0", "marker": "o"},
    "HOSVD":   {"color": "C4", "marker": "v"},
    "CP":      {"color": "C2", "marker": "^"},
    "INDSCAL": {"color": "C1", "marker": "s"},
    "Waring":  {"color": "C3", "marker": "D"},
    "PCP":     {"color": "C5", "marker": "X"},
}


# VARIANTS: unconstrained and ASR-enforced fits share the same rank grid and
# plot on the same axes.  The cache suffix keeps the two result files
# separate so re-running with different flags does not clobber prior work.
VARIANTS = [
    {"tag": "plain", "suffix": "",     "ls": "-",  "label_suffix": "",
     "alpha": 1.0, "fit_kw": {}},
    {"tag": "asr",   "suffix": "_ASR", "ls": "--", "label_suffix": " (ASR)",
     "alpha": 0.7,  "fit_kw": {"enforce_asr": True}},
]


def _rank_to_scalar(rank) -> int:
    """Collapse a (R1, R2) HOSVD rank to a scalar for plotting on the rank axis."""
    if isinstance(rank, tuple):
        return int(max(rank))
    return int(rank)


def _extend_ranks_for_large_dim(dim_sc: int):
    """For the larger fc3_prim_vasp supercell we extend the rank sweep so the
    error curves flatten visibly.

    Rationale: mode-(2,3) spectral analysis on fc3_prim_vasp (dim_sc=162)
    needs ~118 singular values for 99.99% of Frobenius mass vs ~36 on
    fc3_prim (dim_sc=48).  The Waring (symmetric-CP) rank scales roughly with
    this mode-(2,3) effective rank, so the grid must extend to 128-256 for
    fair comparison.  Keeps defaults unchanged for dim_sc<=60.
    """
    if dim_sc <= 60:
        return DEFAULT_RANKS
    extra_scalar = [r for r in (64, 96, 128, 192, 256) if r < dim_sc * 2]
    extra_hosvd = [(6, r) for r in (48, 64, 96, 128) if r < dim_sc]
    ranks = {k: list(v) for k, v in DEFAULT_RANKS.items()}
    for m in ("mSVD", "CP", "INDSCAL", "Waring"):
        ranks[m] = sorted(set(ranks[m] + extra_scalar))
    # mSVD is bounded by min(n_dof*dim_sc, dim_sc) = dim_sc, so cap there.
    ranks["mSVD"] = [r for r in ranks["mSVD"] if r <= dim_sc]
    ranks["HOSVD"] = sorted(set(ranks["HOSVD"] + extra_hosvd))
    return ranks


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(method: str, rank, variant_suffix: str = "") -> str:
    if isinstance(rank, tuple):
        base = f"{method}{variant_suffix}_R{'_'.join(str(r) for r in rank)}"
    else:
        base = f"{method}{variant_suffix}_R{rank}"
    return base


def _save_cache(cache_dir: Path, method: str, res: fc3c.CompressionResult,
                variant_suffix: str = "") -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_dir / f"fc3_compression_{_cache_key(method, res.rank, variant_suffix)}.npz",
        name=res.name,
        rank=np.array(res.rank, dtype=object),
        n_params=res.n_params,
        rel_err=res.rel_err,
        fit_time_s=res.fit_time_s,
        **{f"factor_{k}": v for k, v in res.factors.items()},
    )


def _load_cache(cache_dir: Path, method: str, rank, variant_suffix: str = ""):
    path = cache_dir / f"fc3_compression_{_cache_key(method, rank, variant_suffix)}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    factors = {
        k[len("factor_"):]: data[k] for k in data.files if k.startswith("factor_")
    }
    rk = data["rank"].item()
    return fc3c.CompressionResult(
        name=str(data["name"]),
        rank=rk,
        n_params=int(data["n_params"]),
        rel_err=float(data["rel_err"]),
        fit_time_s=float(data["fit_time_s"]),
        factors=factors,
        info={},
    )


# ---------------------------------------------------------------------------
# Progressive warm-start helpers (affordable high-quality init)
# ---------------------------------------------------------------------------
#
# For CP / INDSCAL / Waring the dominant cost at large rank is the random-init
# restart loop.  A single warm start from the rank R-1 factors plus one
# residual-leading rank-1 column reaches the same minimum in ~1 restart's
# worth of L-BFGS.  This is the standard "folding" trick from Bro 1998 /
# Acar-Kolda-Dunlavy 2011 / Phan-Tichavsky-Cichocki for large-rank CP.


def _residual_leading_rank1(T_res: np.ndarray, enforce_asr: bool,
                            n_super: int | None):
    """Leading rank-1 approximation of the residual tensor T_res via a
    few power iterations on the mode-1 unfolding.  Cheap (a handful of
    matvecs) and more informative than a random init."""
    n_dof, d, _ = T_res.shape
    # Unfold mode-1 and grab leading singular vector cheaply.
    M = T_res.reshape(n_dof, d * d)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    a = U[:, 0] * np.sqrt(S[0])
    bc = (Vt[0] * np.sqrt(S[0])).reshape(d, d)
    # Symmetrise bc into an outer product b * c^T.
    Ub, Sb, Vtb = np.linalg.svd(bc, full_matrices=False)
    b = Ub[:, 0] * np.sqrt(Sb[0])
    c = Vtb[0] * np.sqrt(Sb[0])
    if enforce_asr and n_super is not None:
        b = fc3c.asr_project_factor(b[:, None], n_super, axis=0)[:, 0]
        c = fc3c.asr_project_factor(c[:, None], n_super, axis=0)[:, 0]
    return a, b, c


def _progressive_cp_init(prev_res: fc3c.CompressionResult | None,
                         target: fc3c.FC3Target, new_rank: int,
                         enforce_asr: bool):
    """Return (A0, B0, C0, lam0) initialised from ``prev_res`` + residual
    leading rank-1 column.  Accepts either a CP-shaped parent (factors with
    A/B/C/lambdas) or an INDSCAL-shaped parent (D/V -> A=D, B=C=V, lam=1).
    Returns None if no usable parent is available."""
    if prev_res is None:
        return None
    f = prev_res.factors
    if "A" in f and "B" in f and "C" in f:
        A = f["A"]; B = f["B"]; C = f["C"]
        lam = f.get("lambdas", np.ones(A.shape[1]))
    elif "D" in f and "V" in f:
        A = f["D"]; B = f["V"]; C = f["V"]
        lam = np.ones(A.shape[1])
    else:
        return None
    prev_rank = A.shape[1]
    if prev_rank >= new_rank:
        return A[:, :new_rank], B[:, :new_rank], C[:, :new_rank], lam[:new_rank]
    T_recon = np.einsum("r,mr,jr,kr->mjk", lam, A, B, C, optimize=True)
    T_res = target.T - T_recon
    new_cols_a, new_cols_b, new_cols_c = [], [], []
    for _ in range(new_rank - prev_rank):
        a, b, c = _residual_leading_rank1(T_res, enforce_asr, target.n_super)
        new_cols_a.append(a); new_cols_b.append(b); new_cols_c.append(c)
        T_res = T_res - np.einsum("m,j,k->mjk", a, b, c)
    A0 = np.concatenate([A, np.stack(new_cols_a, axis=1)], axis=1)
    B0 = np.concatenate([B, np.stack(new_cols_b, axis=1)], axis=1)
    C0 = np.concatenate([C, np.stack(new_cols_c, axis=1)], axis=1)
    lam0 = np.concatenate([lam, np.ones(new_rank - prev_rank)])
    return A0, B0, C0, lam0


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


_ERR_FLOOR = 1e-6


def _plot_points(results):
    """Return (x_params, x_rank, y_err) with machine-precision points removed."""
    out = []
    for r in sorted(results, key=lambda x: x.n_params):
        if r.rel_err < _ERR_FLOOR:
            continue
        out.append((r.n_params, _rank_to_scalar(r.rank), r.rel_err))
    return out


def plot_error_vs_params(results_by_method_variant, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    all_errs: list[float] = []
    for (method, variant_tag), results in results_by_method_variant.items():
        if not results:
            continue
        pts = _plot_points(results)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[2] for p in pts]
        all_errs.extend(ys)
        style = METHOD_STYLE[method]
        var = next(v for v in VARIANTS if v["tag"] == variant_tag)
        ax.loglog(
            xs, ys,
            marker=style["marker"], linestyle=var["ls"],
            color=style["color"], alpha=var["alpha"],
            label=f"{method}{var['label_suffix']}",
            markersize=6,
        )
    if all_errs:
        lo = 10 ** np.floor(np.log10(min(all_errs)) - 0.3)
        ax.set_ylim(bottom=max(lo, _ERR_FLOOR), top=1.2)
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel(r"Relative Frobenius error")
    ax.set_title("FC3 compression: error vs parameter count")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_error_vs_rank(results_by_method_variant, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    all_errs: list[float] = []
    for (method, variant_tag), results in results_by_method_variant.items():
        if not results:
            continue
        pts = sorted(_plot_points(results), key=lambda p: p[1])
        if not pts:
            continue
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        all_errs.extend(ys)
        style = METHOD_STYLE[method]
        var = next(v for v in VARIANTS if v["tag"] == variant_tag)
        ax.semilogy(
            xs, ys,
            marker=style["marker"], linestyle=var["ls"],
            color=style["color"], alpha=var["alpha"],
            label=f"{method}{var['label_suffix']}",
            markersize=6,
        )
    if all_errs:
        lo = 10 ** np.floor(np.log10(min(all_errs)) - 0.3)
        ax.set_ylim(bottom=max(lo, _ERR_FLOOR), top=1.2)
    ax.set_xlabel("Rank R")
    ax.set_ylabel(r"Relative Frobenius error")
    ax.set_title("FC3 compression: error vs rank")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_msvd_spectrum(target, out_path: Path) -> None:
    M = target.T.reshape(target.n_dof * target.dim_sc, target.dim_sc)
    S = np.linalg.svd(M, compute_uv=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(np.arange(1, len(S) + 1), S / S[0], "o-", markersize=4)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel(r"$\sigma_r / \sigma_1$")
    ax.set_title("mSVD spectrum of FC3 $(\\mu, j)|k$ matricization")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_hosvd_spectra(target, out_path: Path) -> None:
    T = target.T
    n_dof, dim_sc, _ = T.shape
    S1 = np.linalg.svd(T.reshape(n_dof, -1), compute_uv=False)
    M23 = np.concatenate(
        [T.transpose(1, 0, 2).reshape(dim_sc, -1),
         T.transpose(2, 0, 1).reshape(dim_sc, -1)],
        axis=1,
    )
    S2 = np.linalg.svd(M23, compute_uv=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.semilogy(np.arange(1, len(S1) + 1), S1 / S1[0], "o-", markersize=4)
    ax1.set_xlabel("Index"); ax1.set_ylabel(r"$\sigma_r / \sigma_1$")
    ax1.set_title("HOSVD mode-1 spectrum (external leg)")
    ax1.grid(True, alpha=0.3)
    ax2.semilogy(np.arange(1, len(S2) + 1), S2 / S2[0], "o-", markersize=4, color="C4")
    ax2.set_xlabel("Index"); ax2.set_ylabel(r"$\sigma_r / \sigma_1$")
    ax2.set_title("HOSVD mode-(2,3) spectrum (internal legs, S2-symmetrised)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(results_by_method_variant) -> None:
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Method':<18} {'Rank':>10} {'Params':>10} {'Rel Error':>14} {'Time':>8}")
    print("-" * 72)
    for (method, variant_tag), results in results_by_method_variant.items():
        if not results:
            continue
        var = next(v for v in VARIANTS if v["tag"] == variant_tag)
        label = f"{method}{var['label_suffix']}"
        for r in sorted(results, key=lambda x: x.n_params):
            rank_str = (
                f"({r.rank[0]},{r.rank[1]})" if isinstance(r.rank, tuple) else str(r.rank)
            )
            print(
                f"{label:<18} {rank_str:>10} {r.n_params:>10} "
                f"{r.rel_err:>14.4e} {r.fit_time_s:>7.1f}s"
            )
        print("-" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_RANKS),
        help="Subset of methods to run (default: all)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[v["tag"] for v in VARIANTS],
        choices=[v["tag"] for v in VARIANTS],
        help="Variant tags to run (default: plain asr)",
    )
    parser.add_argument(
        "--fc3-subdir",
        type=str,
        default="fc3_prim_vasp",
        help="FC3 dataset subdirectory under input_calc/ (e.g. fc3_prim, "
             "fc3_prim_vasp).  Selects both the primitive cell and the "
             "fc3.hdf5 file.  Default: fc3_prim",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save plots (default: anharmonic/figures/<fc3_subdir>)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Fit cache directory (default: anharmonic/quality_cache/<fc3_subdir>)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable cache lookup and write"
    )
    parser.add_argument(
        "--cp-restarts", type=int, default=10,
        help="Number of CP random restarts (default: 10)",
    )
    parser.add_argument(
        "--indscal-restarts", type=int, default=8,
        help="Number of INDSCAL random restarts (default: 8)",
    )
    parser.add_argument(
        "--waring-restarts", type=int, default=5,
        help="Number of Waring random restarts (default: 5)",
    )
    parser.add_argument(
        "--pcp-max-iter", type=int, default=2000,
        help="PCP total optimisation iterations per rank (default: 2000)",
    )
    parser.add_argument(
        "--max-time-per-fit", type=float, default=900.0,
        help="Wall-clock budget per individual fit in seconds "
             "(Waring; other methods honour it where supported). Default 900s.",
    )
    parser.add_argument(
        "--no-warm-start", action="store_true",
        help="Disable progressive rank warm-starts for CP/INDSCAL.",
    )
    args = parser.parse_args(argv)

    # Resolve per-dataset paths.
    if args.output_dir is None:
        args.output_dir = script_dir / "figures" / args.fc3_subdir
    if args.cache_dir is None:
        args.cache_dir = script_dir / "quality_cache" / args.fc3_subdir

    phonon, _ = load_primitive_cell(work_dir, fc3_subdir=args.fc3_subdir)
    fc3_path = work_dir / args.fc3_subdir / "fc3.hdf5"
    with h5py.File(fc3_path, "r") as f:
        fc3_raw = np.array(f["fc3"])

    target = fc3c.build_fc3_target(fc3_raw, phonon)
    print(
        f"FC3 target [{args.fc3_subdir}]: n_dof={target.n_dof}, "
        f"dim_sc={target.dim_sc}, ||T||_F={target.target_norm:.4e}"
    )
    print(f"  Full tensor entries: {target.n_dof * target.dim_sc**2}")

    # Spectra plots are dataset-dependent, so route into per-dataset output dir.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_msvd_spectrum(target, args.output_dir / "fc3_msvd_spectrum.pdf")
    plot_hosvd_spectra(target, args.output_dir / "fc3_hosvd_spectra.pdf")
    print(f"Saved spectra to {args.output_dir}")

    ranks_by_method = _extend_ranks_for_large_dim(target.dim_sc)

    # Power-iter init is slow (~60-200s per rank at dim_sc=162) and modern
    # symmetric-CP practice (Kolda 2015 "Numerical optimization for symmetric
    # tensor decomposition") drops it entirely — CP-ALS-then-symmetrise
    # followed by L-BFGS reaches the same minimum faster.  We retain it only
    # on the small dataset where it's cheap.
    large_dim = target.dim_sc > 60
    waring_power_init = not large_dim

    def _base_kwargs(method: str) -> dict:
        if method == "CP":
            return {"n_restarts": args.cp_restarts, "max_iter": 800,
                    "lbfgs_iters": 500}
        if method == "INDSCAL":
            return {"n_restarts": args.indscal_restarts, "max_iter": 800,
                    "lbfgs_iters": 600}
        if method == "Waring":
            return {"n_restarts": args.waring_restarts,
                    "n_power_repeats": 10, "n_power_iters": 200,
                    "power_init": waring_power_init,
                    "cp_init": True,
                    "lbfgs_iters": 500 if large_dim else 400,
                    "max_time_s": args.max_time_per_fit,
                    "early_stop_rel_err": 1e-8}
        if method == "HOSVD":
            return {"refine": True, "hooi_iters": 12}
        if method == "PCP":
            return {"phonon": phonon, "fc3_raw": fc3_raw,
                    "max_iter": args.pcp_max_iter, "verbose": False}
        return {}

    active_variants = [v for v in VARIANTS if v["tag"] in args.variants]

    results_by_method_variant: dict[tuple[str, str], list[fc3c.CompressionResult]] = {}
    t_start = time.time()
    for method in args.methods:
        if method not in DEFAULT_RANKS:
            print(f"Unknown method {method}, skipping.")
            continue
        ranks = ranks_by_method[method]

        for variant in active_variants:
            vsuffix = variant["suffix"]
            key = (method, variant["tag"])
            results_by_method_variant[key] = []

            # Track the most recent fit to feed as warm-start init for the
            # next rank in the sweep.  Supported for CP / INDSCAL / Waring.
            prev_res: fc3c.CompressionResult | None = None
            warm_start_ok = (
                not args.no_warm_start
                and method in ("CP", "INDSCAL", "Waring")
            )

            for rank in ranks:
                # PCP does not support enforce_asr; skip ASR variant gracefully.
                if method == "PCP" and variant["tag"] == "asr":
                    continue

                cached = None
                if not args.no_cache:
                    cached = _load_cache(args.cache_dir, method, rank, vsuffix)
                if cached is not None:
                    print(f"[{method}{vsuffix}] rank={rank}: loaded from cache "
                          f"(params={cached.n_params}, err={cached.rel_err:.4e})")
                    results_by_method_variant[key].append(cached)
                    prev_res = cached
                    continue

                print(f"[{method}{vsuffix}] rank={rank} ...", flush=True)
                try:
                    fitter = fc3c.FITTERS[method]
                    kw = dict(_base_kwargs(method))
                    kw.update(variant["fit_kw"])

                    # --- progressive warm-start (CP/INDSCAL/Waring) ---
                    init_override = None
                    if (warm_start_ok and prev_res is not None
                            and _rank_to_scalar(rank)
                                > _rank_to_scalar(prev_res.rank)):
                        init_override = _progressive_cp_init(
                            prev_res, target, _rank_to_scalar(rank),
                            enforce_asr=variant["fit_kw"].get("enforce_asr", False),
                        )
                        # Reduce the restart loop once a warm init is available
                        # — cold restarts are expensive and rarely beat a
                        # well-initialised L-BFGS at high rank.
                        if method == "INDSCAL":
                            kw["n_restarts"] = min(kw.get("n_restarts", 8), 2)
                        elif method == "CP":
                            kw["n_restarts"] = min(kw.get("n_restarts", 10), 2)
                        elif method == "Waring":
                            kw["n_restarts"] = min(kw.get("n_restarts", 5), 1)

                    if method == "HOSVD":
                        res = fitter(target, R1=rank[0], R2=rank[1], **kw)
                    elif init_override is not None and method == "CP":
                        res = _fit_cp_with_init(
                            fitter, target, rank, init_override, kw)
                    elif init_override is not None and method == "INDSCAL":
                        res = _fit_indscal_with_init(
                            fitter, target, rank, init_override, kw)
                    elif init_override is not None and method == "Waring":
                        res = _fit_waring_with_init(
                            fitter, target, rank, init_override, kw)
                    else:
                        res = fitter(target, rank=rank, **kw)
                except Exception as exc:
                    print(f"  FAILED: {type(exc).__name__}: {exc}")
                    continue

                elapsed = time.time() - t_start
                print(
                    f"    params={res.n_params}, rel_err={res.rel_err:.4e}, "
                    f"t_fit={res.fit_time_s:.1f}s  (total elapsed {elapsed:.0f}s)"
                )
                results_by_method_variant[key].append(res)
                prev_res = res
                if not args.no_cache:
                    _save_cache(args.cache_dir, method, res, vsuffix)

    _print_summary(results_by_method_variant)

    # --- Plots ---
    plot_error_vs_params(
        results_by_method_variant,
        args.output_dir / "fc3_error_vs_params.pdf",
    )
    plot_error_vs_rank(
        results_by_method_variant,
        args.output_dir / "fc3_error_vs_rank.pdf",
    )
    print(f"\nSaved plots to {args.output_dir}")

    # --- JSON dump of numeric results (no factor arrays) ---
    summary: dict = {}
    for (method, variant_tag), results in results_by_method_variant.items():
        v = next(vv for vv in VARIANTS if vv["tag"] == variant_tag)
        label = f"{method}{v['suffix']}"
        summary[label] = [
            {
                "rank": r.rank if not isinstance(r.rank, tuple) else list(r.rank),
                "n_params": r.n_params,
                "rel_err": r.rel_err,
                "fit_time_s": r.fit_time_s,
            }
            for r in results
        ]
    summary["_meta"] = {
        "fc3_subdir": args.fc3_subdir,
        "n_dof": target.n_dof,
        "dim_sc": target.dim_sc,
        "target_norm": target.target_norm,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.output_dir / "fc3_compression_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved summary JSON to {args.output_dir / 'fc3_compression_summary.json'}")


# ---------------------------------------------------------------------------
# Warm-start bootstraps for CP / INDSCAL
# ---------------------------------------------------------------------------


def _fit_cp_with_init(fitter, target, rank, init_factors, kw):
    """CP fit that skips the ALS restart loop when a warm-start init is
    available: build a CompressionResult-shaped object by running the L-BFGS
    refine directly from the warm init, then fall back to the normal fitter
    if that fails to reach the usual quality."""
    from phonon_inputs.fc3_compression import _cp_lbfgs_refine, CompressionResult, n_params_cp, _cp_error
    import time
    A0, B0, C0, lam0 = init_factors
    t0 = time.time()
    try:
        A, B, C, lam, err = _cp_lbfgs_refine(
            target.T, A0, B0, C0, lam0,
            n_iter=kw.get("lbfgs_iters", 500),
            target_norm=target.target_norm,
            enforce_asr=kw.get("enforce_asr", False),
            n_super=target.n_super,
        )
        order = np.argsort(-np.abs(lam))
        A, B, C, lam = A[:, order], B[:, order], C[:, order], lam[order]
        warm = CompressionResult(
            name="CP",
            rank=rank,
            n_params=n_params_cp(rank, target.n_dof, target.dim_sc),
            rel_err=err,
            fit_time_s=time.time() - t0,
            factors={"A": A, "B": B, "C": C, "lambdas": lam},
            info={"warm_start": True,
                  "enforce_asr": kw.get("enforce_asr", False)},
        )
    except Exception:
        warm = None

    # Always also run the cold fit and keep whichever is better, so warm-start
    # never regresses quality.
    cold = fitter(target, rank=rank, **kw)
    if warm is None or cold.rel_err < warm.rel_err:
        return cold
    return warm


def _fit_indscal_with_init(fitter, target, rank, init_factors, kw):
    """Warm-start INDSCAL from a CP init.  INDSCAL has factors (D, V) with
    internal-leg symmetry V = B = C, so we take D := A0 and V := (B0 + C0)/2,
    fold any lam0 weights into D, then L-BFGS.  Falls back to the cold fit if
    the warm result is worse."""
    from phonon_inputs.fc3_compression import (
        _indscal_lbfgs, CompressionResult, n_params_indscal,
    )
    import time
    A0, B0, C0, lam0 = init_factors
    # Absorb the CP lambdas into D so INDSCAL's signed amplitude lives there.
    D0 = A0 * lam0[None, :]
    V0 = 0.5 * (B0 + C0)
    # Symmetrise the target for INDSCAL's internal constraint.
    T_sym = 0.5 * (target.T + target.T.transpose(0, 2, 1))
    t0 = time.time()
    try:
        D, V, err = _indscal_lbfgs(
            T_sym, D0, V0,
            n_iter=kw.get("lbfgs_iters", 600),
            target_norm=target.target_norm,
            enforce_asr=kw.get("enforce_asr", False),
            n_super=target.n_super,
        )
        # Final error measured on the raw target (not T_sym) for consistency.
        T_approx = np.einsum("mr,jr,kr->mjk", D, V, V, optimize=True)
        err = float(np.linalg.norm(target.T - T_approx) /
                    (target.target_norm or 1.0))
        warm = CompressionResult(
            name="INDSCAL",
            rank=rank,
            n_params=n_params_indscal(rank, target.n_dof, target.dim_sc),
            rel_err=err,
            fit_time_s=time.time() - t0,
            factors={"D": D, "V": V},
            info={"warm_start": True,
                  "enforce_asr": kw.get("enforce_asr", False)},
        )
    except Exception:
        warm = None

    cold = fitter(target, rank=rank, **kw)
    if warm is None or cold.rel_err < warm.rel_err:
        return cold
    return warm


def _fit_waring_with_init(fitter, target, rank, init_factors, kw):
    """Warm-start Waring (symmetric CP) from a CP-style init.

    The Waring ansatz has a single factor V of shape (dim_sc, R) with
    T ~ sum_r lam_r V[:, r]^{otimes 3}.  We take V0 := (B0 + C0 + A0_lifted)/3
    where A0_lifted maps A (n_dof=3*n_prim rows) onto the dim_sc axis via
    p2s_map, and feed into :func:`_waring_lbfgs_primitive`.  The cold fit
    always runs as a quality fallback."""
    from phonon_inputs.fc3_compression import (
        _waring_lbfgs_primitive, CompressionResult, n_params_waring,
    )
    import time
    A0, B0, C0, lam0 = init_factors
    # Lift A0 back onto the dim_sc axis via the inverse of prim_idx.
    # For the rows not covered by prim_idx, zero-fill — the dim_sc-axis
    # factor dominates for the symmetric part, so this is safe.
    prim_idx = np.array(
        [3 * int(target.p2s_map[i]) + a
         for i in range(target.nat_prim) for a in range(3)],
        dtype=np.int64,
    )
    A_lift = np.zeros((target.dim_sc, A0.shape[1]))
    A_lift[prim_idx] = A0
    V0 = (A_lift + B0 + C0) / 3.0
    t0 = time.time()
    try:
        V, lam, raw_err = _waring_lbfgs_primitive(
            target.T, V0, lam0, prim_idx,
            n_iter=kw.get("lbfgs_iters", 500),
            norm=target.target_norm,
            enforce_asr=kw.get("enforce_asr", False),
            n_super=target.n_super,
        )
        err = raw_err / (target.target_norm or 1.0)
        warm = CompressionResult(
            name="Waring",
            rank=rank,
            n_params=n_params_waring(rank, target.dim_sc),
            rel_err=err,
            fit_time_s=time.time() - t0,
            factors={"V": V, "lambdas": lam},
            info={"warm_start": True,
                  "enforce_asr": kw.get("enforce_asr", False)},
        )
    except Exception:
        warm = None

    cold = fitter(target, rank=rank, **kw)
    if warm is None or cold.rel_err < warm.rel_err:
        return cold
    return warm


if __name__ == "__main__":
    main()
