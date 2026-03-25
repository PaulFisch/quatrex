#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_square_matrix(path: Path):
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


def compute_contact_bands(K00, K01, K10, k_min: float, k_max: float, k_num: int):
    import numpy as np

    if k_num < 2:
        raise RuntimeError("--k-num must be at least 2")
    if k_max <= k_min:
        raise RuntimeError("--k-max must be greater than --k-min")

    k_grid = np.linspace(k_min, k_max, k_num)
    n_mode = K00.shape[0]
    omega = np.zeros((k_num, n_mode), dtype=float)

    for idx, k_value in enumerate(k_grid):
        phase = np.exp(2j * np.pi * k_value)
        Dk = K00.astype(complex) + K01.astype(complex) * phase + K10.astype(complex) * phase.conjugate()
        Dk = 0.5 * (Dk + Dk.conj().T)
        eigvals = np.linalg.eigvalsh(Dk)
        eigvals = np.maximum(eigvals.real, 0.0)
        omega[idx, :] = np.sqrt(eigvals)

    return k_grid, omega


def build_mass_vector(phonopy_yaml: Path, device_metadata: Path, expected_dof: int):
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
    return masses


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a combined figure: transmission spectrum + contact band structure."
    )
    parser.add_argument(
        "--spectrum",
        type=Path,
        required=True,
        help="NPZ file with omega and transmission arrays.",
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
        "--metadata",
        type=Path,
        default=None,
        help="Optional transport metadata json for subtitle context.",
    )
    parser.add_argument("--k-min", type=float, default=-0.5, help="Min reduced k for contact band.")
    parser.add_argument("--k-max", type=float, default=0.5, help="Max reduced k for contact band.")
    parser.add_argument("--k-num", type=int, default=301, help="Number of k points.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("transport_contact_combined.png"),
        help="Output PNG file path.",
    )
    parser.add_argument(
        "--mark-omega",
        type=float,
        action="append",
        default=None,
        help=(
            "Repeatable frequency markers. Draws vertical guides in transmission panel "
            "and horizontal guides in contact-band panel."
        ),
    )
    parser.add_argument(
        "--phonopy-yaml",
        type=Path,
        default=None,
        help=(
            "Optional phonopy YAML path for mass-normalizing contact bands. "
            "Use together with --device-metadata."
        ),
    )
    parser.add_argument(
        "--device-metadata",
        type=Path,
        default=None,
        help=(
            "Device metadata JSON containing transport_atom_order. "
            "Use together with --phonopy-yaml."
        ),
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib and numpy are required.") from exc

    spectrum_path = args.spectrum.expanduser().resolve()
    if not spectrum_path.exists():
        raise RuntimeError(f"Spectrum file not found: {spectrum_path}")

    spectrum = np.load(spectrum_path)
    if "omega" not in spectrum or "transmission" not in spectrum:
        raise RuntimeError("Spectrum NPZ must contain omega and transmission arrays.")

    omega_transport = np.asarray(spectrum["omega"], dtype=float)
    transmission = np.asarray(spectrum["transmission"], dtype=float)
    if omega_transport.shape != transmission.shape:
        raise RuntimeError("omega/transmission shape mismatch in spectrum file.")

    K00, k00_path = load_square_matrix(args.k00)
    K01, k01_path = load_square_matrix(args.k01)
    if args.k10 is None:
        K10 = K01.T.copy()
        k10_path = None
    else:
        K10, k10_path = load_square_matrix(args.k10)

    if K00.shape != K01.shape or K00.shape != K10.shape:
        raise RuntimeError(
            f"K block shape mismatch: K00={K00.shape}, K01={K01.shape}, K10={K10.shape}"
        )

    mass_vector = None
    if (args.phonopy_yaml is None) != (args.device_metadata is None):
        raise RuntimeError("Use --phonopy-yaml and --device-metadata together for mass normalization.")
    if args.phonopy_yaml is not None and args.device_metadata is not None:
        mass_vector = build_mass_vector(
            phonopy_yaml=args.phonopy_yaml,
            device_metadata=args.device_metadata,
            expected_dof=K00.shape[0],
        )

    k_grid, omega_contact = compute_contact_bands(
        K00=K00,
        K01=K01,
        K10=K10,
        k_min=args.k_min,
        k_max=args.k_max,
        k_num=args.k_num,
    )
    if mass_vector is not None:
        inv_sqrt_m = np.diag(1.0 / np.sqrt(np.asarray(mass_vector, dtype=float)))
        for idx, k_value in enumerate(k_grid):
            phase = np.exp(2j * np.pi * k_value)
            Dk = K00.astype(complex) + K01.astype(complex) * phase + K10.astype(complex) * phase.conjugate()
            Dk = 0.5 * (Dk + Dk.conj().T)
            Dk = inv_sqrt_m @ Dk @ inv_sqrt_m
            eigvals = np.linalg.eigvalsh(Dk)
            eigvals = np.maximum(eigvals.real, 0.0)
            omega_contact[idx, :] = np.sqrt(eigvals)

    subtitle = None
    metadata_path = args.metadata
    if metadata_path is None:
        candidate = spectrum_path.with_name(f"{spectrum_path.stem}_metadata.json")
        if candidate.exists():
            metadata_path = candidate

    if metadata_path is not None:
        metadata_path = metadata_path.expanduser().resolve()
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if "device_cells" in metadata:
                subtitle = f"Device cells = {metadata['device_cells']}"

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_t, ax_b) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    ax_t.plot(omega_transport, transmission, color="tab:blue", linewidth=1.5)
    ax_t.set_xlabel("Angular frequency ω")
    ax_t.set_ylabel("Transmission T(ω)")
    ax_t.set_title("Transmission spectrum")
    ax_t.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    if transmission.size > 0:
        y_max = max(float(transmission.max()) * 1.05, 1e-6)
        ax_t.set_ylim(0.0, y_max)

    marked_omega = [float(value) for value in (args.mark_omega or [])]
    for omega_marker in marked_omega:
        ax_t.axvline(omega_marker, color="tab:red", linestyle="--", linewidth=0.9, alpha=0.75)

    for branch in range(omega_contact.shape[1]):
        ax_b.plot(k_grid, omega_contact[:, branch], color="tab:orange", linewidth=0.8)
    ax_b.set_xlabel("Reduced wavevector k")
    ax_b.set_ylabel("Contact mode frequency ω")
    ax_b.set_title("Contact band structure")
    ax_b.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    for omega_marker in marked_omega:
        ax_b.axhline(omega_marker, color="tab:red", linestyle="--", linewidth=0.9, alpha=0.75)

    suptitle = "Phonon transport + contact bands"
    if subtitle is not None:
        suptitle += f"\n{subtitle}"
    fig.suptitle(suptitle)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Spectrum: {spectrum_path}")
    print(f"K00: {k00_path}")
    print(f"K01: {k01_path}")
    print(f"K10: {k10_path if k10_path is not None else 'K01.T'}")
    if mass_vector is not None:
        print("Mass normalization for contact bands: enabled")
    if marked_omega:
        print(f"Marked frequencies: {marked_omega}")
    print(f"Saved combined plot: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
