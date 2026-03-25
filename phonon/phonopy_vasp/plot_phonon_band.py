#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

MATERIAL_CONFIG = {
    "si": {
        "default_harmonic_dir": PROJECT_ROOT / "phonon-data/si/harmonic",
        "default_output": PROJECT_ROOT / "phonon-data/si/harmonic/si_phonon_band.png",
        "title": "Bulk Si phonon band structure",
        "path_vertices": [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.625, 0.25, 0.625],            
            [0.375, 0.375, 0.75],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.25, 0.75],            
            [0.5, 0.0, 0.5],
        ],
        "labels": ["Γ", "X", "U", "K", "Γ", "L", "W", "X"],
    },
    "mos2": {
        "default_harmonic_dir": PROJECT_ROOT / "phonon-data/mos2/harmonic",
        "default_output": PROJECT_ROOT / "phonon-data/mos2/harmonic/mos2_phonon_band.png",
        "title": "Monolayer MoS2 phonon band structure",
        "path_vertices": [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        "labels": ["Γ", "M", "K", "Γ"],
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot phonon band structure from harmonic phonopy outputs."
    )
    parser.add_argument(
        "--material",
        choices=sorted(MATERIAL_CONFIG.keys()),
        default="si",
        help="Material preset for high-symmetry path and default I/O locations.",
    )
    parser.add_argument(
        "--harmonic-dir",
        type=Path,
        default=None,
        help="Directory containing POSCAR, phonopy_disp.yaml and FORCE_SETS or FORCE_CONSTANTS.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path.",
    )
    parser.add_argument(
        "--points-per-segment",
        type=int,
        default=101,
        help="Number of q-points per high-symmetry segment.",
    )
    parser.add_argument(
        "--symmetrize-force-constants",
        action="store_true",
        help="Symmetrize force constants before band calculation (helps reduce numerical acoustic drift).",
    )
    return parser


def get_phonon_object(harmonic_dir: Path):
    try:
        import phonopy
    except ImportError as exc:
        raise RuntimeError("phonopy is not installed in the current Python environment.") from exc

    yaml_path = harmonic_dir / "phonopy_disp.yaml"
    if not yaml_path.exists():
        raise RuntimeError(f"Missing {yaml_path}. Run prepare/collect first.")

    force_constants = harmonic_dir / "FORCE_CONSTANTS"
    force_sets = harmonic_dir / "FORCE_SETS"

    if force_constants.exists():
        return phonopy.load(
            phonopy_yaml=str(yaml_path),
            force_constants_filename=str(force_constants),
        )

    if force_sets.exists():
        return phonopy.load(
            phonopy_yaml=str(yaml_path),
            force_sets_filename=str(force_sets),
        )

    raise RuntimeError(
        "Missing harmonic force data. Need FORCE_CONSTANTS or FORCE_SETS in "
        f"{harmonic_dir}."
    )


def plot_band(
    phonon,
    output_path: Path,
    points_per_segment: int,
    path_vertices: list[list[float]],
    labels: list[str],
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is not installed in the current Python environment.") from exc

    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

    bands, connections = get_band_qpoints_and_path_connections(
        [
            [path_vertices[i], path_vertices[i + 1]]
            for i in range(len(path_vertices) - 1)
        ],
        npoints=points_per_segment,
    )

    phonon.run_band_structure(bands, path_connections=connections, labels=labels)
    band_dict = phonon.get_band_structure_dict()

    distances = band_dict["distances"]
    frequencies = band_dict["frequencies"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    for segment_dist, segment_freq in zip(distances, frequencies):
        branch_count = segment_freq.shape[1]
        for branch in range(branch_count):
            ax.plot(segment_dist, segment_freq[:, branch], color="tab:blue", linewidth=1.0)

    x_ticks = [distances[0][0]]
    for segment_dist in distances:
        x_ticks.append(segment_dist[-1])

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels)

    for tick in x_ticks:
        ax.axvline(tick, color="gray", linewidth=0.6, alpha=0.5)

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Frequency (THz)")
    ax.set_title(title)
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run(
    material: str,
    harmonic_dir: Path,
    output: Path,
    points_per_segment: int,
    symmetrize_force_constants: bool,
) -> Path:
    if points_per_segment < 2:
        raise RuntimeError("--points-per-segment must be at least 2")

    config = MATERIAL_CONFIG[material]
    phonon = get_phonon_object(harmonic_dir)
    if symmetrize_force_constants:
        phonon.symmetrize_force_constants()
    plot_band(
        phonon=phonon,
        output_path=output,
        points_per_segment=points_per_segment,
        path_vertices=config["path_vertices"],
        labels=config["labels"],
        title=config["title"],
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = MATERIAL_CONFIG[args.material]
    harmonic_dir = (args.harmonic_dir or config["default_harmonic_dir"]).resolve()
    output = (args.output or config["default_output"]).resolve()

    saved = run(
        material=args.material,
        harmonic_dir=harmonic_dir,
        output=output,
        points_per_segment=args.points_per_segment,
        symmetrize_force_constants=args.symmetrize_force_constants,
    )
    print(f"Saved phonon band structure plot: {saved}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
