"""Compare FC3 compression ansatze and report error vs parameter count.

Thin driver around :mod:`phonon_inputs.fc3_compression`.  Fits every requested
(method, rank) pair to the mass-weighted FC3 from ``fc3_prim/fc3.hdf5`` and
produces diagnostic plots:

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

Each fit is cached to ``anharmonic/quality_cache/fc3_compression_*.npz``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
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


def _rank_to_scalar(rank) -> int:
    """Collapse a (R1, R2) HOSVD rank to a scalar for plotting on the rank axis."""
    if isinstance(rank, tuple):
        return int(max(rank))
    return int(rank)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(method: str, rank) -> str:
    if isinstance(rank, tuple):
        return f"{method}_R{'_'.join(str(r) for r in rank)}"
    return f"{method}_R{rank}"


def _save_cache(cache_dir: Path, method: str, res: fc3c.CompressionResult) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_dir / f"fc3_compression_{_cache_key(method, res.rank)}.npz",
        name=res.name,
        rank=np.array(res.rank, dtype=object),
        n_params=res.n_params,
        rel_err=res.rel_err,
        fit_time_s=res.fit_time_s,
        **{f"factor_{k}": v for k, v in res.factors.items()},
    )


def _load_cache(cache_dir: Path, method: str, rank):
    path = cache_dir / f"fc3_compression_{_cache_key(method, rank)}.npz"
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
# Plots
# ---------------------------------------------------------------------------


# Errors below this floor are treated as machine-precision noise and hidden
# from the convergence plots so the y-axis stays informative.
_ERR_FLOOR = 1e-6


def _plot_points(results):
    """Return (x_params, x_rank, y_err) with machine-precision points removed."""
    out = []
    for r in sorted(results, key=lambda x: x.n_params):
        if r.rel_err < _ERR_FLOOR:
            continue
        out.append((r.n_params, _rank_to_scalar(r.rank), r.rel_err))
    return out


def plot_error_vs_params(results_by_method, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    all_errs: list[float] = []
    for method, results in results_by_method.items():
        if not results:
            continue
        pts = _plot_points(results)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[2] for p in pts]
        all_errs.extend(ys)
        ax.loglog(
            xs, ys,
            f"-{METHOD_STYLE[method]['marker']}",
            color=METHOD_STYLE[method]["color"],
            label=method,
            markersize=6,
        )
    if all_errs:
        lo = 10 ** np.floor(np.log10(min(all_errs)) - 0.3)
        ax.set_ylim(bottom=max(lo, _ERR_FLOOR), top=1.2)
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel(r"Relative Frobenius error")
    ax.set_title("FC3 compression: error vs parameter count")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_error_vs_rank(results_by_method, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    all_errs: list[float] = []
    for method, results in results_by_method.items():
        if not results:
            continue
        pts = sorted(_plot_points(results), key=lambda p: p[1])
        if not pts:
            continue
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        all_errs.extend(ys)
        ax.semilogy(
            xs, ys,
            f"-{METHOD_STYLE[method]['marker']}",
            color=METHOD_STYLE[method]["color"],
            label=method,
            markersize=6,
        )
    if all_errs:
        lo = 10 ** np.floor(np.log10(min(all_errs)) - 0.3)
        ax.set_ylim(bottom=max(lo, _ERR_FLOOR), top=1.2)
    ax.set_xlabel("Rank R")
    ax.set_ylabel(r"Relative Frobenius error")
    ax.set_title("FC3 compression: error vs rank")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
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


def _print_summary(results_by_method) -> None:
    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"{'Method':<10} {'Rank':>10} {'Params':>10} {'Rel Error':>14} {'Time':>8}")
    print("-" * 64)
    for method, results in results_by_method.items():
        for r in sorted(results, key=lambda x: x.n_params):
            rank_str = (
                f"({r.rank[0]},{r.rank[1]})" if isinstance(r.rank, tuple) else str(r.rank)
            )
            print(
                f"{method:<10} {rank_str:>10} {r.n_params:>10} "
                f"{r.rel_err:>14.4e} {r.fit_time_s:>7.1f}s"
            )
        print("-" * 64)


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
        "--output-dir",
        type=Path,
        default=script_dir / "figures",
        help="Where to save plots (default: anharmonic/figures)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=script_dir / "quality_cache",
        help="Fit cache directory (default: anharmonic/quality_cache)",
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
        "--waring-restarts", type=int, default=10,
        help="Number of Waring random restarts (default: 10)",
    )
    parser.add_argument(
        "--pcp-max-iter", type=int, default=2000,
        help="PCP total optimisation iterations per rank (default: 2000)",
    )
    parser.add_argument(
        "--fc3-hdf5",
        type=Path,
        default=work_dir / "fc3_prim" / "fc3.hdf5",
        help="Path to FC3 HDF5 (default: input_calc/fc3_prim/fc3.hdf5)",
    )
    args = parser.parse_args(argv)

    phonon, _ = load_primitive_cell(work_dir)
    with h5py.File(args.fc3_hdf5, "r") as f:
        fc3_raw = np.array(f["fc3"])

    target = fc3c.build_fc3_target(fc3_raw, phonon)
    print(
        f"FC3 target: n_dof={target.n_dof}, dim_sc={target.dim_sc}, "
        f"||T||_F={target.target_norm:.4e}"
    )
    print(f"  Full tensor entries: {target.n_dof * target.dim_sc**2}")

    # --- Pre-compute spectra plots (independent of fits) ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_msvd_spectrum(target, args.output_dir / "fc3_msvd_spectrum.pdf")
    plot_hosvd_spectra(target, args.output_dir / "fc3_hosvd_spectra.pdf")
    print(f"Saved spectra to {args.output_dir}")

    # --- Per-method fits ---
    extra_kwargs = {
        "CP":      {"n_restarts": args.cp_restarts, "max_iter": 800, "lbfgs_iters": 500},
        "INDSCAL": {"n_restarts": args.indscal_restarts, "max_iter": 800, "lbfgs_iters": 600},
        "Waring":  {"n_restarts": args.waring_restarts, "n_power_repeats": 30,
                    "n_power_iters": 300, "lbfgs_iters": 600},
        "HOSVD":   {"refine": True, "hooi_iters": 12},
        "PCP":     {"phonon": phonon, "fc3_raw": fc3_raw, "max_iter": args.pcp_max_iter, "verbose": False},
    }

    results_by_method: dict[str, list[fc3c.CompressionResult]] = {}
    for method in args.methods:
        if method not in DEFAULT_RANKS:
            print(f"Unknown method {method}, skipping.")
            continue
        ranks = DEFAULT_RANKS[method]
        results_by_method[method] = []
        for rank in ranks:
            # Cache lookup
            cached = None
            if not args.no_cache:
                cached = _load_cache(args.cache_dir, method, rank)
            if cached is not None:
                print(f"[{method}] rank={rank}: loaded from cache "
                      f"(params={cached.n_params}, err={cached.rel_err:.4e})")
                results_by_method[method].append(cached)
                continue

            print(f"[{method}] rank={rank} ...", flush=True)
            try:
                fitter = fc3c.FITTERS[method]
                kw = extra_kwargs.get(method, {})
                if method == "HOSVD":
                    res = fitter(target, R1=rank[0], R2=rank[1], **kw)
                else:
                    res = fitter(target, rank=rank, **kw)
            except Exception as exc:
                print(f"  FAILED: {exc}")
                continue

            print(
                f"    params={res.n_params}, rel_err={res.rel_err:.4e}, "
                f"t={res.fit_time_s:.1f}s"
            )
            results_by_method[method].append(res)
            if not args.no_cache:
                _save_cache(args.cache_dir, method, res)

    _print_summary(results_by_method)

    # --- Plots ---
    plot_error_vs_params(results_by_method, args.output_dir / "fc3_error_vs_params.pdf")
    plot_error_vs_rank(results_by_method, args.output_dir / "fc3_error_vs_rank.pdf")
    print(f"\nSaved plots to {args.output_dir}")

    # --- JSON dump of numeric results (no factor arrays) ---
    summary = {
        method: [
            {
                "rank": r.rank if not isinstance(r.rank, tuple) else list(r.rank),
                "n_params": r.n_params,
                "rel_err": r.rel_err,
                "fit_time_s": r.fit_time_s,
            }
            for r in results_by_method[method]
        ]
        for method in results_by_method
    }
    summary["_meta"] = {
        "n_dof": target.n_dof,
        "dim_sc": target.dim_sc,
        "target_norm": target.target_norm,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.output_dir / "fc3_compression_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved summary JSON to {args.output_dir / 'fc3_compression_summary.json'}")


if __name__ == "__main__":
    main()
