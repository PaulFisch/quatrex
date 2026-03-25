#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_matrix(path: Path):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is not installed in the current Python environment.") from exc

    if not path.exists():
        raise RuntimeError(f"Matrix file not found: {path}")
    matrix = np.load(path)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError(f"Matrix must be square 2D: {path}, got shape {matrix.shape}")
    return matrix


def build_device_matrix_from_blocks(K00, K01, K10, num_cells: int):
    import numpy as np

    if num_cells < 1:
        raise RuntimeError("--device-cells must be >= 1 when building from K blocks.")

    block_size = K00.shape[0]
    total_size = block_size * num_cells
    dtype = np.result_type(K00, K01, K10, np.complex128)
    device = np.zeros((total_size, total_size), dtype=dtype)

    for cell in range(num_cells):
        row = cell * block_size
        device[row : row + block_size, row : row + block_size] = K00

        if cell + 1 < num_cells:
            col = (cell + 1) * block_size
            device[row : row + block_size, col : col + block_size] = K01
            device[col : col + block_size, row : row + block_size] = K10

    return device


def sancho_rubio_surface_gf(z, h0, alpha, beta, tol: float, max_iter: int):
    import numpy as np

    dim = h0.shape[0]
    identity = np.eye(dim, dtype=complex)

    es = h0.astype(complex).copy()
    e = h0.astype(complex).copy()
    a = alpha.astype(complex).copy()
    b = beta.astype(complex).copy()

    for _ in range(max_iter):
        g = np.linalg.inv(z * identity - e)

        a_g_b = a @ g @ b
        b_g_a = b @ g @ a

        es_new = es + a_g_b
        e_new = e + a_g_b + b_g_a
        a_new = a @ g @ a
        b_new = b @ g @ b

        coupling_norm = max(float(np.linalg.norm(a_new)), float(np.linalg.norm(b_new)))
        es = es_new
        e = e_new
        a = a_new
        b = b_new

        if coupling_norm < tol:
            break
    else:
        raise RuntimeError(
            "Sancho-Rubio did not converge within max iterations. "
            "Increase --max-iter or --eta, or relax --tol."
        )

    return np.linalg.inv(z * identity - es)


def compute_transmission_spectrum(
    KC,
    K00,
    K01,
    K10,
    omega_min: float,
    omega_max: float,
    omega_num: int,
    eta: float,
    sancho_tol: float,
    sancho_max_iter: int,
):
    import numpy as np

    if omega_num < 2:
        raise RuntimeError("--omega-num must be at least 2.")

    dof_block = K00.shape[0]
    if KC.shape[0] < 2 * dof_block:
        raise RuntimeError("Device matrix is too small for two-contact transport calculation.")

    omega_grid = np.linspace(omega_min, omega_max, omega_num)
    valid_omega = []
    valid_transmission = []
    skipped_frequencies = []

    identity_full = np.eye(KC.shape[0], dtype=complex)

    for idx, omega in enumerate(omega_grid):
        z = (omega + 1j * eta) ** 2
        try:
            g_left = sancho_rubio_surface_gf(
                z=z,
                h0=K00,
                alpha=K10,
                beta=K01,
                tol=sancho_tol,
                max_iter=sancho_max_iter,
            )
            g_right = sancho_rubio_surface_gf(
                z=z,
                h0=K00,
                alpha=K01,
                beta=K10,
                tol=sancho_tol,
                max_iter=sancho_max_iter,
            )

            sigma_left = K10 @ g_left @ K01
            sigma_right = K01 @ g_right @ K10

            gamma_left = 1j * (sigma_left - sigma_left.conj().T)
            gamma_right = 1j * (sigma_right - sigma_right.conj().T)

            sigma_embedded = np.zeros_like(KC, dtype=complex)
            sigma_embedded[:dof_block, :dof_block] = sigma_left
            sigma_embedded[-dof_block:, -dof_block:] = sigma_right

            G = np.linalg.inv(z * identity_full - KC.astype(complex) - sigma_embedded)

            G_1N = G[:dof_block, -dof_block:]
            T = np.trace(gamma_left @ G_1N @ gamma_right @ G_1N.conj().T)

            valid_omega.append(float(omega))
            valid_transmission.append(max(0.0, float(T.real)))
        except Exception as exc:
            skipped_frequencies.append(
                {
                    "index": int(idx),
                    "omega": float(omega),
                    "reason": str(exc),
                }
            )

    if not valid_omega:
        raise RuntimeError(
            "Transport failed for all frequency points. "
            "Increase --eta or --max-iter, or relax --tol."
        )

    return np.asarray(valid_omega, dtype=float), np.asarray(valid_transmission, dtype=float), skipped_frequencies


def save_spectrum(output_path: Path, omega, transmission, metadata: dict) -> None:
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, omega=omega, transmission=transmission)

    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phonon quantum transport using OBC self-energies from the Sancho-Rubio algorithm."
        )
    )
    parser.add_argument(
        "--k00",
        type=Path,
        required=True,
        help="Path to K00 block matrix (.npy).",
    )
    parser.add_argument(
        "--k01",
        type=Path,
        required=True,
        help="Path to K01 block matrix (.npy).",
    )
    parser.add_argument(
        "--k10",
        type=Path,
        default=None,
        help="Path to K10 block matrix (.npy). Defaults to K01^T.",
    )
    parser.add_argument(
        "--device-matrix",
        type=Path,
        default=None,
        help="Optional full device FC matrix (.npy). If omitted, built from K00/K01/K10 and --device-cells.",
    )
    parser.add_argument(
        "--device-cells",
        type=int,
        default=None,
        help="Number of transport cells when --device-matrix is not provided.",
    )
    parser.add_argument(
        "--omega-min",
        type=float,
        default=0.0,
        help="Minimum angular-frequency value (same unit as sqrt(eigenvalues of K)).",
    )
    parser.add_argument(
        "--omega-max",
        type=float,
        default=20.0,
        help="Maximum angular-frequency value.",
    )
    parser.add_argument(
        "--omega-num",
        type=int,
        default=400,
        help="Number of frequency grid points.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1e-4,
        help="Small positive broadening added as (omega + i*eta)^2.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-12,
        help="Convergence tolerance for Sancho-Rubio decimation.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum Sancho-Rubio iterations per frequency.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("transport_sancho_rubio_spectrum.npz"),
        help="Output NPZ path containing omega and transmission arrays.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    K00 = load_matrix(args.k00.expanduser().resolve())
    K01 = load_matrix(args.k01.expanduser().resolve())

    if args.k10 is None:
        K10 = K01.T.copy()
    else:
        K10 = load_matrix(args.k10.expanduser().resolve())

    if K00.shape != K01.shape or K00.shape != K10.shape:
        raise RuntimeError(
            f"K block shape mismatch: K00={K00.shape}, K01={K01.shape}, K10={K10.shape}"
        )

    if args.device_matrix is not None:
        KC = load_matrix(args.device_matrix.expanduser().resolve())
        if KC.shape[0] % K00.shape[0] != 0:
            raise RuntimeError("Device matrix size is not divisible by K block size.")
        device_cells = KC.shape[0] // K00.shape[0]
    else:
        if args.device_cells is None:
            raise RuntimeError("Provide --device-matrix or --device-cells.")
        KC = build_device_matrix_from_blocks(K00=K00, K01=K01, K10=K10, num_cells=args.device_cells)
        device_cells = args.device_cells

    omega, transmission, skipped_frequencies = compute_transmission_spectrum(
        KC=KC,
        K00=K00,
        K01=K01,
        K10=K10,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        omega_num=args.omega_num,
        eta=args.eta,
        sancho_tol=args.tol,
        sancho_max_iter=args.max_iter,
    )

    output_path = args.output.expanduser().resolve()
    metadata = {
        "k00": str(args.k00.expanduser().resolve()),
        "k01": str(args.k01.expanduser().resolve()),
        "k10": str(args.k10.expanduser().resolve()) if args.k10 is not None else "k01.T",
        "device_matrix": str(args.device_matrix.expanduser().resolve()) if args.device_matrix is not None else None,
        "device_cells": int(device_cells),
        "omega_min": float(args.omega_min),
        "omega_max": float(args.omega_max),
        "omega_num": int(args.omega_num),
        "eta": float(args.eta),
        "sancho_tol": float(args.tol),
        "sancho_max_iter": int(args.max_iter),
        "computed_points": int(len(omega)),
        "skipped_points": int(len(skipped_frequencies)),
        "skipped_frequencies": skipped_frequencies,
    }
    save_spectrum(output_path=output_path, omega=omega, transmission=transmission, metadata=metadata)

    print(f"Computed transmission points: {len(omega)}")
    if skipped_frequencies:
        print(f"Skipped non-converged points: {len(skipped_frequencies)}")
    print(f"Device cells: {device_cells}")
    print(f"Max transmission: {float(transmission.max()):.6f}")
    print(f"Saved spectrum: {output_path}")
    print(f"Saved metadata: {output_path.with_name(f'{output_path.stem}_metadata.json')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
