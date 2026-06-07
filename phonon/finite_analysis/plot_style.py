"""Shared publication figure style for the report.

One place that fixes the matplotlib look of every results figure so the PDFs
embedded in ``document/`` are visually consistent. Figures are exported
*single-panel* and *without an in-image title* -- the caption lives in LaTeX,
and multi-panel layouts are assembled there with ``subfigure``.

Usage::

    from finite_analysis.plot_style import set_publication_style, FIG_SINGLE, PALETTE, finalize
    set_publication_style()
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.plot(...)
    finalize(fig, "/path/to/document/fig/transport_sweeps/foo.pdf")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Single source of truth for the figure tree.
REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
FIG_DIR = REPO / "document/fig/transport_sweeps"

# Figure-size presets (inches). Single-column LaTeX figures.
FIG_SINGLE = (5.4, 4.0)   # one panel, ~0.6-0.7\linewidth in the report
FIG_WIDE = (7.0, 4.2)     # a wider single panel when the x-axis is busy
FIG_HALF = (4.0, 3.4)     # a half-width panel destined for a 2-up subfigure

# Paul-Tol bright qualitative palette (colour-blind safe), matching the
# existing finite_analysis.transport_quality convention.
PALETTE: dict[str, str] = {
    "blue": "#4477AA",
    "green": "#228833",
    "yellow": "#CCBB44",
    "cyan": "#66CCEE",
    "red": "#AA3377",
    "grey": "#BBBBBB",
    "purple": "#AA3377",
}
# Stable per-method colours for the decomposition figures.
METHOD_COLORS: dict[str, str] = {
    "dense": "#000000",
    "mSVD": "#AA3377",
    "HOSVD": "#4477AA",
    "CP": "#CCBB44",
    "INDSCAL": "#228833",
    "Waring": "#66CCEE",
}
# Stable per-structure colours/markers shared across figures.
STRUCT_STYLE: dict[str, dict[str, Any]] = {
    "d5a": dict(color="#4477AA", marker="o"),
    "d11a": dict(color="#AA3377", marker="s"),
    "cnt33": dict(color="#228833", marker="^"),
    "cnt80": dict(color="#CCBB44", marker="D"),
}

_RC: dict[str, Any] = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.6,
    "lines.markersize": 5.5,
    "legend.frameon": False,
}


def set_publication_style() -> None:
    """Apply the shared rcParams. Call once before building figures."""
    matplotlib.rcParams.update(_RC)


def finalize(fig: "plt.Figure", path: str | Path) -> Path:
    """Strip any in-image title and write a single PDF (caption lives in LaTeX).

    Writes to ``path`` verbatim if absolute; otherwise under ``FIG_DIR``.
    """
    # No in-figure titles: the LaTeX \caption is the only caption.
    if fig._suptitle is not None:  # type: ignore[attr-defined]
        fig.suptitle("")
    for ax in fig.get_axes():
        ax.set_title("")
    out = Path(path)
    if not out.is_absolute():
        out = FIG_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out
