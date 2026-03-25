#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot phonon transmission spectrum (omega vs transmission)."
    )
    parser.add_argument(
        "--spectrum",
        type=Path,
        required=True,
        help="NPZ output from phonon_transport_sancho_rubio.py containing omega and transmission.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional metadata JSON path. Defaults to <spectrum_stem>_metadata.json if present.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <spectrum_stem>.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom plot title.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib and numpy are required for plotting.") from exc

    spectrum_path = args.spectrum.expanduser().resolve()
    if not spectrum_path.exists():
        raise RuntimeError(f"Spectrum file not found: {spectrum_path}")

    data = np.load(spectrum_path)
    if "omega" not in data or "transmission" not in data:
        raise RuntimeError("Spectrum NPZ must contain arrays: omega and transmission.")

    omega = np.asarray(data["omega"], dtype=float)
    transmission = np.asarray(data["transmission"], dtype=float)
    if omega.shape != transmission.shape:
        raise RuntimeError(
            f"omega/transmission shape mismatch: {omega.shape} vs {transmission.shape}"
        )

    metadata_path = args.metadata
    metadata = None
    if metadata_path is None:
        candidate = spectrum_path.with_name(f"{spectrum_path.stem}_metadata.json")
        if candidate.exists():
            metadata_path = candidate
    if metadata_path is not None:
        metadata_path = metadata_path.expanduser().resolve()
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    output_path = args.output
    if output_path is None:
        output_path = spectrum_path.with_suffix(".png")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title = args.title
    if title is None:
        title = "Phonon transmission spectrum (Sancho-Rubio OBC)"
        if metadata is not None and "device_cells" in metadata:
            title += f"\nDevice cells = {metadata['device_cells']}"

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(omega, transmission, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("Angular frequency ω")
    ax.set_ylabel("Transmission T(ω)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if transmission.size > 0:
        y_max = max(float(transmission.max()) * 1.05, 1e-6)
        ax.set_ylim(0.0, y_max)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Spectrum source: {spectrum_path}")
    if metadata_path is not None and metadata_path.exists():
        print(f"Metadata source: {metadata_path}")
    print(f"Saved transmission plot: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
