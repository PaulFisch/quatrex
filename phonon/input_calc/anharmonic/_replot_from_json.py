"""Regenerate FC3 compression plots from the saved summary JSON.

Avoids re-running the expensive fits when only the plotting code changed.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_fc3_approximations import (
    plot_error_vs_params,
    plot_error_vs_rank,
)


@dataclass
class _Res:
    name: str
    rank: object
    n_params: int
    rel_err: float
    fit_time_s: float


def main():
    out_dir = Path(__file__).resolve().parent / "figures_fixed"
    with open(out_dir / "fc3_compression_summary.json") as fh:
        summary = json.load(fh)

    results_by_method = {}
    for method, rows in summary.items():
        if method == "_meta":
            continue
        acc = []
        for row in rows:
            rk = tuple(row["rank"]) if isinstance(row["rank"], list) else row["rank"]
            acc.append(_Res(
                name=method,
                rank=rk,
                n_params=row["n_params"],
                rel_err=row["rel_err"],
                fit_time_s=row["fit_time_s"],
            ))
        results_by_method[method] = acc

    plot_error_vs_params(results_by_method, out_dir / "fc3_error_vs_params.pdf")
    plot_error_vs_rank(results_by_method, out_dir / "fc3_error_vs_rank.pdf")
    print(f"Regenerated plots in {out_dir}")


if __name__ == "__main__":
    main()
