"""Cross-system overlay plots for the results section.

After per-system analyses have populated
``document/src/fig/finite_analysis/<system>/`` (sparsity / decomposition /
physical / cutoffs subdirs), this script reads the CSV outputs and produces
three cross-system overlays:

  * ``cross_fc3_decay.pdf`` — mean ‖Φ_3‖ vs triplet diameter, all systems
    on one log-y panel.
  * ``cross_decomposition_ranks.pdf`` — Frobenius reconstruction error vs
    parameter count, six methods × N systems.
  * ``cross_summary_table.pdf`` — one-page text-rendered table of
    headline numbers.

Usage::

    python -m finite_analysis.cross_system_plots \
        --systems sinw100,si_chain,cnt33_finite \
        --base-dir document/src/fig/finite_analysis \
        --out-dir document/src/fig/finite_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_decomp_csv(path: Path) -> list[dict]:
    """Read decomposition/decomp_rank_sweep.csv into a list of dicts."""
    rows: list[dict] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "method": r["method"],
                "rank": r["rank"],
                "n_params": int(r["n_params"]),
                "frob_err": float(r["frob_err"]),
            })
    return rows


def _read_summary_json(path: Path) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# FC3 decay overlay                                                           #
# --------------------------------------------------------------------------- #


def plot_fc3_decay_envelope(
    systems: list[str], base_dir: Path, out_path: Path,
) -> None:
    """One log-y panel with the per-system FC3 1D decay (sparsity output)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    cmap = plt.colormaps["tab10"]
    for idx, sys in enumerate(systems):
        # The 1D decay PNG is the artifact, but we want the underlying numbers.
        # The sparsity driver doesn't currently emit a CSV for the 1D decay;
        # fall back to re-importing the system if available, else skip.
        # For a truly cross-system plot we expect the per-system summary.json
        # to carry pre-computed (centers, mean) — extend run_sparsity later.
        sj = base_dir / sys / "summary.json"
        if not sj.exists():
            continue
        s = _read_summary_json(sj)
        sp = s.get("sparsity", {})
        # If the sparsity summary exposed (centers, mean), plot it; otherwise
        # placeholder annotation.
        centers = sp.get("decay_centers")
        means = sp.get("decay_means")
        if centers and means:
            ax.semilogy(centers, means, "o-", label=sys,
                        color=cmap(idx % 10), lw=1.3, ms=4)
        else:
            ax.text(0.05, 0.95 - 0.05 * idx, f"{sys}: no decay data",
                    transform=ax.transAxes, color=cmap(idx % 10))
    ax.set_xlabel(r"triplet diameter $\max(d_{ij}, d_{ik}, d_{jk})$  [Å]")
    ax.set_ylabel(r"mean $\|\Phi_{3,ijk}\|_F$  [eV/Å³]")
    ax.set_title("FC3 magnitude vs triplet diameter — system overlay")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Decomposition-rank overlay                                                  #
# --------------------------------------------------------------------------- #


def plot_decomposition_ranks(
    systems: list[str], base_dir: Path, out_path: Path,
) -> None:
    """Frobenius error vs parameter count, one subplot per system."""
    n = len(systems)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, sys in zip(axes, systems):
        path = base_dir / sys / "decomposition" / "decomp_rank_sweep.csv"
        if not path.exists():
            ax.text(0.5, 0.5, f"no decomposition CSV\n{path}",
                    transform=ax.transAxes, ha="center", va="center")
            continue
        rows = _read_decomp_csv(path)
        methods = sorted({r["method"] for r in rows})
        for m in methods:
            mr = sorted([r for r in rows if r["method"] == m],
                        key=lambda r: r["n_params"])
            ax.loglog(
                [r["n_params"] for r in mr],
                [r["frob_err"] for r in mr],
                "o-", label=m, lw=1.3, ms=4,
            )
        ax.set_xlabel("number of parameters")
        ax.set_ylabel(r"$\|T - \tilde T\|_F / \|T\|_F$")
        ax.set_title(sys)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Summary table                                                               #
# --------------------------------------------------------------------------- #


def plot_summary_table(
    systems: list[str], base_dir: Path, out_path: Path,
) -> None:
    """Render a small headline-numbers table as a one-page PDF."""
    rows: list[list[str]] = []
    header = ["system", "n_atoms", "FC3 max [eV/Å³]",
              "ASR rel residual", "imag modes"]
    rows.append(header)
    for sys in systems:
        sj = base_dir / sys / "summary.json"
        if not sj.exists():
            rows.append([sys, "-", "-", "-", "-"])
            continue
        s = _read_summary_json(sj)
        n_atoms = s.get("system", {}).get("n_super", "-")
        fc3_max = s.get("sparsity", {}).get("fc3_max", "-")
        asr_dict = s.get("physical", {}).get("fc3_asr_legs", {})
        asr_worst = max(
            asr_dict.get("leg_j_rel", 0.0),
            asr_dict.get("leg_k_rel", 0.0),
        ) if asr_dict else "-"
        n_imag = s.get("physical", {}).get("dispersion", {}).get("n_imaginary", "-")
        rows.append([
            sys, str(n_atoms),
            f"{fc3_max:.3e}" if isinstance(fc3_max, (int, float)) else str(fc3_max),
            f"{asr_worst:.3f}" if isinstance(asr_worst, float) else str(asr_worst),
            str(n_imag),
        ])

    fig, ax = plt.subplots(figsize=(8.0, 1.0 + 0.4 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows, loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("Usage::")[0])
    p.add_argument("--systems", required=True,
                   help="Comma-separated list of system subdirs to overlay")
    p.add_argument("--base-dir", type=Path, required=True,
                   help="Parent directory containing per-system analysis outputs")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to write cross_*.{png,pdf}")
    args = p.parse_args(argv)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_fc3_decay_envelope(systems, args.base_dir,
                             args.out_dir / "cross_fc3_decay.png")
    plot_decomposition_ranks(systems, args.base_dir,
                              args.out_dir / "cross_decomposition_ranks.png")
    plot_summary_table(systems, args.base_dir,
                        args.out_dir / "cross_summary_table.png")
    print(f"Wrote 3 cross-system overlays under {args.out_dir}")
    return 0


if __name__ == "__main__":
    import sys as _sys
    raise SystemExit(main(_sys.argv[1:]))
