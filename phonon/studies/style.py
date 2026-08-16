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

#: The thesis figure directory. Generators for the report write here, so
#: `make_all.py` can check every referenced figure was regenerated.
DOC_FIG = Path(__file__).resolve().parents[2] / "document" / "fig" / "transport_sweeps"

#: The report's true \textwidth: 418.26 pt from document/report.log, at
#: 72.27 pt/in. A figure drawn at this width and included with
#: `width=\linewidth` is reproduced 1:1, so the font sizes below are the
#: font sizes on the page. Anything drawn wider is scaled DOWN by LaTeX and
#: its labels shrink with it -- which is how the existing corpus ended up
#: with 9.5 pt labels printing at 6.3 pt.
TEXTWIDTH_IN = 418.26 / 72.27      # 5.788 in

#: Fixed colour meanings, used across the results chapter.
C_BALLISTIC = "#0173b2"   # harmonic / ballistic reference   (cycle C0)
C_ANHARMONIC = "#d55e00"  # the anharmonic result            (cycle C3)
C_THIRD = "#029e73"       # a third series when one is needed (cycle C2)
C_REFERENCE = "0.35"      # external reference: literature, experiment, dense

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
    # Slot order is VALIDATED, not chosen by eye. Run
    #   node <dataviz>/scripts/validate_palette.js \
    #        "#0173b2,#cc78bc,#029e73,#d55e00" --mode light --pairs all
    # -> all four checks PASS (worst all-pairs CVD dE 8.4 protan,
    #    normal-vision floor 18.5, lightness and chroma in band).
    #
    # This is the seaborn colorblind palette with slots 2 and 5 swapped. The
    # shipped order put #de8f05 (amber) at slot 2 and #d55e00 (vermillion) at
    # slot 4, and that PAIR FAILS: dE 11.4 under NORMAL vision, below the 15
    # floor -- full-colour readers cannot separate them either, and the amber
    # also sits at 2.55:1 against the page. Any figure drawing four unlabelled
    # series hit it; phph_physics_si had both in one panel.
    #
    # Slots 5-8 are the unvalidated tail and are kept only so an existing
    # script does not crash on a 5th series. Adding a 5th HUE does not pass
    # here at all-pairs -- four is the ceiling for this palette. A 5th series
    # must carry secondary encoding (marker shape plus a direct label), which
    # is what the decomposition-family figures do.
    "axes.prop_cycle": plt.cycler(
        color=["#0173b2", "#cc78bc", "#029e73", "#d55e00",
               "#de8f05", "#ca9161", "#949494", "#fbafe4"]
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


def doc_figure(ncols: int = 1, nrows: int = 1, frac: float = 1.0,
               aspect: float = 0.62, **kwargs):
    """A figure sized for the report, drawn 1:1 with how it will be printed.

    ``frac`` is the fraction of ``\\textwidth`` the ``\\includegraphics`` will
    ask for, so ``frac=1.0`` pairs with ``width=\\linewidth`` and ``frac=0.48``
    with a two-up ``0.48\\linewidth``. The figure is then built at exactly
    that many inches and LaTeX scales it by one, which is the whole point:
    the 9.5 pt of ``RC`` is 9.5 pt on the page.

    ``aspect`` is height/width of the WHOLE figure, so a two-panel figure
    keeps the page width and splits it rather than doubling it.
    """
    width = TEXTWIDTH_IN * frac
    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(width, width * aspect), **kwargs
        )
    plt.rcParams.update(RC)
    return fig, axes


def panel_labels(axes, labels: str = "abcdefgh", loc=(0.012, 0.985)):
    """Stamp ``(a)``, ``(b)``, ... on the panels, in reading order.

    Kept out of the axes titles on purpose: the caption carries the panel
    descriptions, so the figure only needs the key back to it.
    """
    import numpy as np

    flat = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    for ax, letter in zip(flat, labels):
        ax.text(loc[0], loc[1], f"({letter})", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold")
    return axes


def save(fig, name: str, directory: Path | str | None = None) -> Path:
    """Save ``fig`` as png+pdf under ``FIG_DIR`` (or ``directory``)."""
    directory = Path(directory) if directory is not None else FIG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{ext}")
    print(f"wrote {directory / name}.png (+.pdf)", flush=True)
    return directory / f"{name}.png"
