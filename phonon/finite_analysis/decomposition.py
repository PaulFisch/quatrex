"""FC3 tensor-decomposition rank sweep.

Drives the existing fitters in :mod:`phonon_inputs.fc3_compression`
(``mSVD, HOSVD, CP, INDSCAL, Waring, PCP``) over a rank ladder and
reports both Frobenius error and *thresholded nonzero count of the
reconstruction* — the latter answers the user's question of whether
each decomposition genuinely reduces nonzero footprint compared to a
sparse-thresholded dense FC3 baseline.

The reconstruction nnz at relative threshold ε is computed by counting
entries of ``T_approx`` whose magnitude exceeds ``ε * max|T|``. The
baseline ``nnz_sparse_dense(ε)`` is computed the same way on the dense
FC3 itself; both numbers are reported in the CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phonon_inputs.fc3_compression import (
    CompressionResult,
    FITTERS,
    reconstruct,
)

from .loader import SystemBundle


# --------------------------------------------------------------------------- #
# Rank planning                                                               #
# --------------------------------------------------------------------------- #


def _hosvd_ranks(scalar_ranks: Iterable[int]) -> list[tuple[int, int]]:
    """HOSVD wants (R1, R2). Mirror the user's R as ``(R, R)``."""
    return [(int(r), int(r)) for r in scalar_ranks]


def default_ranks_per_method(
    scalar_ranks: Iterable[int], methods: Iterable[str] | None = None,
) -> dict[str, list]:
    methods = list(methods) if methods is not None else list(FITTERS.keys())
    out: dict[str, list] = {}
    for m in methods:
        if m == "HOSVD":
            out[m] = _hosvd_ranks(scalar_ranks)
        else:
            out[m] = [int(r) for r in scalar_ranks]
    return out


# --------------------------------------------------------------------------- #
# Sweep                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class DecompRow:
    method: str
    rank: int | tuple
    n_params: int
    frob_err: float
    fit_time_s: float
    nnz_at_1em3: int
    nnz_at_1em4: int


def fit_decomposition(
    bundle: SystemBundle,
    *,
    scalar_ranks: Iterable[int] = (2, 4, 8, 16, 32),
    methods: Iterable[str] | None = None,
    extra_kwargs: dict | None = None,
    include_pcp: bool = False,
    verbose: bool = False,
) -> list[DecompRow]:
    """Fit every (method, rank) combination and tabulate (params, error, nnz).

    PCP requires ``phonon`` and ``fc3_raw`` as kwargs, is the slowest /
    least reliable fitter, and routinely produces reconstructions that are
    *denser* than the input FC3 (rank-2 in this codebase emits 50+× the
    dense nnz). It is therefore opt-in via ``include_pcp=True``.
    """
    target = bundle.fc3_target
    methods_list = list(methods) if methods is not None else list(FITTERS.keys())
    if not include_pcp and "PCP" in methods_list:
        methods_list.remove("PCP")

    ranks_per_method = default_ranks_per_method(scalar_ranks, methods_list)

    extra = dict(extra_kwargs or {})
    if "PCP" in methods_list:
        extra.setdefault("PCP", {})
        extra["PCP"].setdefault("phonon", bundle.phonon)
        extra["PCP"].setdefault("fc3_raw", bundle.fc3_raw)

    fc3_max = float(np.max(np.abs(target.T))) or 1.0

    rows: list[DecompRow] = []
    for method, ranks in ranks_per_method.items():
        fitter = FITTERS[method]
        kwargs = extra.get(method, {})
        # Diagnostic sweep: report the RAW approximation error of each
        # (method, rank) -- no ASR projection (the production default since
        # 2026-07 would fold the target's own ASR violation into frob_err
        # and break the full-rank-exactness bound). Conserving production
        # fits go through fc3_compression.fit_production instead.
        if method != "PCP":
            kwargs = {"enforce_asr": False, **kwargs}
        for rank in ranks:
            if verbose:
                print(f"[decomp] {method} rank={rank} ...", flush=True)
            try:
                if method == "HOSVD":
                    R1, R2 = rank
                    res: CompressionResult = fitter(target, R1=R1, R2=R2, **kwargs)
                else:
                    res = fitter(target, rank=rank, **kwargs)
            except Exception as exc:  # noqa: BLE001 — we want to report and move on
                if verbose:
                    print(f"  {method} rank={rank} failed: {exc}")
                continue

            T_approx = reconstruct(res, target, phonon=bundle.phonon if method == "PCP" else None)
            nnz_1em3 = int(np.count_nonzero(np.abs(T_approx) > 1e-3 * fc3_max))
            nnz_1em4 = int(np.count_nonzero(np.abs(T_approx) > 1e-4 * fc3_max))

            rows.append(
                DecompRow(
                    method=method,
                    rank=rank,
                    n_params=res.n_params,
                    frob_err=res.rel_err,
                    fit_time_s=res.fit_time_s,
                    nnz_at_1em3=nnz_1em3,
                    nnz_at_1em4=nnz_1em4,
                )
            )
    return rows


def svd_per_atom_slice(bundle: SystemBundle) -> list[dict]:
    """Per-(primitive atom, Cartesian μ) SVD spectra of FC3 slices.

    For each ``(i_prim, μ)`` reshape ``FC3[s_i, :, :, μ, :, :]`` to a square
    matrix and run an SVD; report the rank required to capture 99 / 99.9 /
    99.99 % of the spectral energy plus the count of non-trivial singular
    values. Promoted from the legacy ``analysis/fc3_separability.py``.

    Returns a list of dicts with keys
    ``{i_prim, mu, singular_values, rank_99, rank_999, rank_9999, n_nonzero}``.
    """
    fc3 = bundle.fc3_raw
    nat_prim = bundle.phonon.primitive.masses.shape[0]
    p2s_map = bundle.phonon.primitive.p2s_map.astype(int)
    is_compact = fc3.shape[0] == nat_prim
    n_cols = fc3.shape[1]
    dim = n_cols * 3

    results: list[dict] = []
    for i_prim in range(nat_prim):
        s_i = i_prim if is_compact else int(p2s_map[i_prim])
        for mu in range(3):
            block = fc3[s_i, :, :, mu, :, :]
            mat = block.transpose(0, 2, 1, 3).reshape(dim, dim)
            sv = np.linalg.svd(mat, compute_uv=False)
            total = float(np.sum(sv ** 2))
            if total < 1e-30:
                results.append(dict(
                    i_prim=i_prim, mu=mu, singular_values=sv,
                    rank_99=0, rank_999=0, rank_9999=0, n_nonzero=0,
                ))
                continue
            cum = np.cumsum(sv ** 2) / total
            results.append(dict(
                i_prim=i_prim, mu=mu, singular_values=sv,
                rank_99=int(np.searchsorted(cum, 0.99)) + 1,
                rank_999=int(np.searchsorted(cum, 0.999)) + 1,
                rank_9999=int(np.searchsorted(cum, 0.9999)) + 1,
                n_nonzero=int(np.sum(sv > 1e-10 * sv[0])),
            ))
    return results


def baseline_dense_nnz(bundle: SystemBundle) -> dict[str, int]:
    """nnz of the dense mass-weighted FC3 itself at the same thresholds."""
    T = bundle.fc3_target.T
    fc3_max = float(np.max(np.abs(T))) or 1.0
    return {
        "nnz_at_1em3": int(np.count_nonzero(np.abs(T) > 1e-3 * fc3_max)),
        "nnz_at_1em4": int(np.count_nonzero(np.abs(T) > 1e-4 * fc3_max)),
        "size": int(T.size),
    }


def write_decomp_csv(rows: Iterable[DecompRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("method,rank,n_params,frob_err,fit_time_s,nnz_at_1em3,nnz_at_1em4\n")
        for r in rows:
            rank_repr = (
                f"{r.rank[0]}-{r.rank[1]}" if isinstance(r.rank, tuple) else str(r.rank)
            )
            f.write(
                f"{r.method},{rank_repr},{r.n_params},{r.frob_err:.6e},"
                f"{r.fit_time_s:.3f},{r.nnz_at_1em3},{r.nnz_at_1em4}\n"
            )


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #


def plot_frob_vs_params(
    rows: Iterable[DecompRow], out_path: Path, *, system_name: str = ""
) -> None:
    rows = list(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    methods = sorted({r.method for r in rows})
    for m in methods:
        m_rows = [r for r in rows if r.method == m]
        m_rows.sort(key=lambda r: r.n_params)
        ax.loglog(
            [r.n_params for r in m_rows],
            [r.frob_err for r in m_rows],
            "o-", label=m, lw=1.5, ms=5,
        )
    ax.set_xlabel("number of parameters")
    ax.set_ylabel(r"relative Frobenius error $\|T - \tilde T\|_F / \|T\|_F$")
    ax.set_title(f"FC3 decomposition: error vs. parameter count — {system_name}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)


def plot_nnz_vs_eps(
    rows: Iterable[DecompRow],
    baseline: dict[str, int],
    out_path: Path,
    *,
    system_name: str = "",
    failed_frob_threshold: float = 0.99,
) -> None:
    """Reconstructed-tensor nnz at ε = 1e-3 vs sparse-dense baseline.

    Rows whose Frobenius error is above ``failed_frob_threshold`` (default
    0.99 → fit explains < 1% of the tensor) are dropped from the plot: an
    nnz=0 reconstruction collapses to the bottom of a semilogy axis and
    produces a misleading vertical jump when the next rank suddenly fits
    something non-trivial. Skipped (method, rank) cells are listed in a
    footnote so the reader can see which fits diverged.
    """
    rows = list(rows)
    valid_rows = [r for r in rows if r.frob_err <= failed_frob_threshold]
    skipped = [r for r in rows if r.frob_err > failed_frob_threshold]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    methods = sorted({r.method for r in valid_rows})
    y_cap = 2 * baseline["nnz_at_1em3"]
    off_scale: list[DecompRow] = []
    for m in methods:
        m_rows = [r for r in valid_rows if r.method == m]
        m_rows.sort(key=lambda r: r.n_params)
        ax.semilogy(
            [r.n_params for r in m_rows],
            [r.nnz_at_1em3 for r in m_rows],
            "o-", label=m, lw=1.5, ms=5,
        )
        off_scale.extend(r for r in m_rows if r.nnz_at_1em3 > y_cap)
    ax.axhline(
        baseline["nnz_at_1em3"], color="k", ls="--", lw=1.0,
        label=f"dense FC3 nnz @ε=1e-3 ({baseline['nnz_at_1em3']:,})",
    )
    ax.set_xlabel("number of parameters")
    ax.set_ylabel("nonzeros in reconstruction at ε=1e-3")
    ax.set_xscale("log")
    ax.set_ylim(top=y_cap)
    ax.set_title(f"FC3 decomposition: reconstruction nnz vs. parameter count — {system_name}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower right")
    footnotes: list[str] = []
    if skipped:
        footnotes.append(
            f"not converged (Frobenius err > {failed_frob_threshold}): "
            + ", ".join(f"{r.method} r={_format_rank(r.rank)}" for r in skipped)
        )
    if off_scale:
        footnotes.append(
            "off-scale (nnz ≫ dense FC3 baseline): "
            + ", ".join(
                f"{r.method} r={_format_rank(r.rank)} → {r.nnz_at_1em3:,}"
                for r in off_scale
            )
        )
    if footnotes:
        ax.text(
            0.02, -0.18, "\n".join(footnotes),
            transform=ax.transAxes, fontsize=7, color="grey",
            ha="left", va="top",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(Path(out_path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _format_rank(rank) -> str:
    if isinstance(rank, tuple):
        return f"{rank[0]}-{rank[1]}"
    return str(rank)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def run_decomposition(
    bundle: SystemBundle,
    out_dir: Path,
    *,
    scalar_ranks: Iterable[int] = (2, 4, 8, 16, 32),
    methods: Iterable[str] | None = None,
    include_pcp: bool = False,
    verbose: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = fit_decomposition(
        bundle, scalar_ranks=scalar_ranks, methods=methods,
        include_pcp=include_pcp, verbose=verbose,
    )
    baseline = baseline_dense_nnz(bundle)
    write_decomp_csv(rows, out_dir / "decomp_rank_sweep.csv")
    plot_frob_vs_params(rows, out_dir / "decomp_frob_vs_params.png", system_name=bundle.name)
    plot_nnz_vs_eps(rows, baseline, out_dir / "decomp_nnz_vs_eps.png", system_name=bundle.name)
    return {
        "units": {"frob_err": "dimensionless (relative)", "nnz_at_eps": "count"},
        "n_rows": len(rows),
        "methods": sorted({r.method for r in rows}),
        "ranks": sorted({r.rank if isinstance(r.rank, int) else r.rank[0] for r in rows}),
        "baseline": baseline,
        "best_per_method": {
            m: min(
                (r.frob_err for r in rows if r.method == m),
                default=float("nan"),
            )
            for m in {r.method for r in rows}
        },
    }
