#!/usr/bin/env python3
"""Plot the causal L5 Si frequency and transverse-mesh convergence table.

Only completed rows with a numerical conductance are drawn.  Filled symbols
have passed every release gate; open symbols are retained as diagnostics so a
non-converged trajectory cannot silently become a material data point.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "phonon"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phonon.studies import style


TABLE = ROOT / "phonon/scripts/data/si_discretization_convergence.csv"
FIGDIR = ROOT / "document/fig/transport_sweeps"


def _completed_rows() -> list[dict]:
    rows = []
    with TABLE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["conductance_mw_m2k"]:
                continue
            parsed = dict(row)
            for key in ("q_mesh", "n_frequency"):
                parsed[key] = int(row[key])
            parsed["conductance_mw_m2k"] = float(
                row["conductance_mw_m2k"])
            parsed["released"] = row["released"].lower() == "true"
            rows.append(parsed)
    return rows


def _series(ax, rows: list[dict], xkey: str, xlabel: str) -> None:
    rows = sorted(rows, key=lambda row: row[xkey])
    if not rows:
        ax.text(0.5, 0.5, "awaiting converged runs", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        return
    x = [row[xkey] for row in rows]
    y = [row["conductance_mw_m2k"] for row in rows]
    ax.plot(x, y, color=style.C_ANHARMONIC, lw=1.2)
    for row in rows:
        face = style.C_ANHARMONIC if row["released"] else "none"
        ax.plot(row[xkey], row["conductance_mw_m2k"], "o",
                color=style.C_ANHARMONIC, markerfacecolor=face)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$G$ (MW m$^{-2}$ K$^{-1}$)")


def main() -> None:
    rows = _completed_rows()
    fig, (ax_f, ax_q) = style.doc_figure(ncols=2, aspect=0.40)
    _series(
        ax_f,
        [row for row in rows if row["q_mesh"] == 5],
        "n_frequency", "primary frequency points",
    )
    _series(
        ax_q,
        [row for row in rows if row["n_frequency"] == 161],
        "q_mesh", r"transverse mesh $N_q$",
    )
    style.panel_labels((ax_f, ax_q))
    style.save(fig, "si_discretization_convergence", directory=FIGDIR)
    for row in rows:
        state = "released" if row["released"] else "diagnostic"
        print(f"{row['case_id']}: {row['conductance_mw_m2k']:.6g} "
              f"MW m^-2 K^-1 ({state})")


if __name__ == "__main__":
    main()
