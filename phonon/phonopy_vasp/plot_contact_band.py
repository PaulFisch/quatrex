#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot contact/lead band structure from nearest-neighbor blocks K00, K01, K10."
        )
    )
    parser.add_argument("--k00", type=Path, required=True, help="Path to K00 block (.npy).")
    parser.add_argument("--k01", type=Path, required=True, help="Path to K01 block (.npy).")
    parser.add_argument(
        "--k10",
        type=Path,
        default=None,
        help="Path to K10 block (.npy). Defaults to K01^T.",
    )
    parser.add_argument(
        "--k-min",
        type=float,
        default=-0.5,
        help="Minimum reduced wavevector k (in units of reciprocal lattice vector).",
    )
    parser.add_argument(
        "--k-max",
        type=float,
        default=0.5,
        help="Maximum reduced wavevector k.",
    )
    parser.add_argument(
        "--k-num",
        type=int,
        default=301,
        help="Number of k points.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("contact_band_structure.png"),
        help="Output PNG file path.",
    )
    parser.add_argument(
        "--save-data",
        type=Path,
        default=None,
        help="Optional NPZ output path for k-grid and omega branches.",
    )
    parser.add_argument(
        "--phonopy-yaml",
        type=Path,
        default=None,
        help=(
            "Optional phonopy YAML path (phonopy_disp.yaml) to load primitive masses. "
            "Required together with --device-metadata for mass normalization."
        ),
    )
    parser.add_argument(
        "--device-metadata",
        type=Path,
        default=None,
        help=(
            "Device metadata JSON from extract_harmonic_fc.py containing transport_atom_order. "
            "Required together with --phonopy-yaml for mass normalization."
        ),
    )
    parser.add_argument(
        "--frequency-factor",
        type=float,
        default=None,
        help=(
            "Optional multiplier applied to omega values after diagonalization "
            "(e.g. phonopy frequency unit conversion factor for THz)."
        ),
    )
    return parser


def load_matrix(path: Path):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required.") from exc

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise RuntimeError(f"Matrix file not found: {resolved}")
    matrix = np.load(resolved)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError(f"Matrix must be square: {resolved}, got {matrix.shape}")
    return matrix, resolved


def build_mass_vector(phonopy_yaml: Path, device_metadata: Path, expected_dof: int):
    import json

    try:
        import phonopy
    except ImportError as exc:
        raise RuntimeError("phonopy is required for mass normalization.") from exc

    yaml_path = phonopy_yaml.expanduser().resolve()
    metadata_path = device_metadata.expanduser().resolve()
    if not yaml_path.exists():
        raise RuntimeError(f"Phonopy yaml not found: {yaml_path}")
    if not metadata_path.exists():
        raise RuntimeError(f"Device metadata not found: {metadata_path}")

    phonon = phonopy.load(phonopy_yaml=str(yaml_path))
    primitive = phonon.primitive
    primitive_masses = [float(primitive.masses[i]) for i in range(len(primitive))]

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    order = metadata.get("transport_atom_order")
    if not isinstance(order, list):
        raise RuntimeError("device metadata missing transport_atom_order.")

    masses = []
    for atom in order:
        primitive_index = int(atom["primitive_atom_index"]) - 1
        masses.extend([primitive_masses[primitive_index]] * 3)

    if len(masses) != expected_dof:
        raise RuntimeError(
            f"Mass vector length mismatch: got {len(masses)}, expected {expected_dof}."
        )

    factor = getattr(phonon, "unit_conversion_factor", None)
    return masses, (float(factor) if factor is not None else None)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib and numpy are required.") from exc

    K00, k00_path = load_matrix(args.k00)
    K01, k01_path = load_matrix(args.k01)
    if args.k10 is None:
        K10 = K01.T.copy()
        k10_path = None
    else:
        K10, k10_path = load_matrix(args.k10)

    if K00.shape != K01.shape or K00.shape != K10.shape:
        raise RuntimeError(
            f"Shape mismatch: K00={K00.shape}, K01={K01.shape}, K10={K10.shape}"
        )

    mass_vector = None
    inferred_frequency_factor = None
    if (args.phonopy_yaml is None) != (args.device_metadata is None):
        raise RuntimeError("Use --phonopy-yaml and --device-metadata together for mass normalization.")
    if args.phonopy_yaml is not None and args.device_metadata is not None:
        mass_vector, inferred_frequency_factor = build_mass_vector(
            phonopy_yaml=args.phonopy_yaml,
            device_metadata=args.device_metadata,
            expected_dof=K00.shape[0],
        )

    if args.k_num < 2:
        raise RuntimeError("--k-num must be at least 2")
    if args.k_max <= args.k_min:
        raise RuntimeError("--k-max must be greater than --k-min")

    k_grid = np.linspace(args.k_min, args.k_max, args.k_num)
    n_mode = K00.shape[0]
    omega = np.zeros((args.k_num, n_mode), dtype=float)

    for idx, k_value in enumerate(k_grid):
        phase = np.exp(2j * np.pi * k_value)
        Dk = K00.astype(complex) + K01.astype(complex) * phase + K10.astype(complex) * phase.conjugate()
        Dk = 0.5 * (Dk + Dk.conj().T)

        if mass_vector is not None:
            inv_sqrt_m = np.diag(1.0 / np.sqrt(np.asarray(mass_vector, dtype=float)))
            Dk = inv_sqrt_m @ Dk @ inv_sqrt_m

        eigvals = np.linalg.eigvalsh(Dk)
        eigvals = np.maximum(eigvals.real, 0.0)
        omega[idx, :] = np.sqrt(eigvals)

    frequency_factor = args.frequency_factor
    if frequency_factor is None:
        frequency_factor = inferred_frequency_factor
    if frequency_factor is not None:
        omega = omega * float(frequency_factor)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for band in range(n_mode):
        ax.plot(k_grid, omega[:, band], color="tab:blue", linewidth=0.8)

    ax.set_xlabel("Reduced wavevector k")
    if frequency_factor is None:
        ax.set_ylabel("Contact mode frequency ω")
    else:
        ax.set_ylabel("Contact mode frequency (converted units)")
    ax.set_title("Contact band structure from K00/K01/K10")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    if args.save_data is not None:
        save_data_path = args.save_data.expanduser().resolve()
        save_data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_data_path, k=k_grid, omega=omega)
        print(f"Saved band data: {save_data_path}")

    print(f"K00: {k00_path}")
    print(f"K01: {k01_path}")
    if k10_path is None:
        print("K10: K01.T")
    else:
        print(f"K10: {k10_path}")
    if mass_vector is not None:
        print("Mass normalization: enabled")
    if frequency_factor is not None:
        print(f"Frequency factor: {frequency_factor}")
    print(f"Saved contact band plot: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
