"""The single plotting style for all phonon studies.

Usage::

    from phonon.studies import style
    fig, axes = style.figure(ncols=2)
    ...
    style.save(fig, "cnt33_transmission")   # -> out/fig/<name>.png + .pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

#: Default figure output directory (phonon/studies/out/fig).
FIG_DIR = Path(__file__).resolve().parent / "out" / "fig"

#: One uniform rcParams set: compact panels, readable fonts, light grid.
RC = {
    "figure.dpi": 110,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "axes.prop_cycle": plt.cycler(
        color=["#0173b2", "#de8f05", "#029e73", "#d55e00",
               "#cc78bc", "#ca9161", "#fbafe4", "#949494"]
    ),
    "figure.constrained_layout.use": True,
}


def figure(ncols: int = 1, nrows: int = 1, width: float = 4.2,
           height: float = 3.2, **kwargs):
    """A styled figure with ``nrows x ncols`` panels (width/height per panel).

    Returns ``(fig, axes)`` with ``axes`` always squeezed the matplotlib way.
    """
    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(width * ncols, height * nrows), **kwargs
        )
    # rc_context does not stick to the created artists' future children,
    # so re-apply the style params on the figure-level context manager.
    plt.rcParams.update(RC)
    return fig, axes


def save(fig, name: str, directory: Path | str | None = None) -> Path:
    """Save ``fig`` as png+pdf under ``FIG_DIR`` (or ``directory``)."""
    directory = Path(directory) if directory is not None else FIG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{ext}")
    print(f"wrote {directory / name}.png (+.pdf)", flush=True)
    return directory / f"{name}.png"
