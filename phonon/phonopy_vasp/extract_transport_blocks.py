#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract transport blocks (K00, K01, K10) from a dense device force-constant matrix."
        )
    )
    parser.add_argument(
        "--device-matrix",
        type=Path,
        required=True,
        help="Path to dense device FC matrix (.npy).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Optional metadata JSON path from extract_harmonic_fc.py. "
            "Defaults to <device-matrix-stem>_metadata.json"
        ),
    )
    parser.add_argument(
        "--dof-per-cell",
        type=int,
        default=None,
        help="Degrees of freedom per transport cell. Overrides metadata if given.",
    )
    parser.add_argument(
        "--reference-cell",
        type=int,
        default=None,
        help=(
            "0-based transport-cell index used to extract K00 and right coupling K01. "
            "Default: middle cell when possible, else 0."
        ),
    )
    parser.add_argument(
        "--max-neighbor",
        type=int,
        default=1,
        help=(
            "Maximum longitudinal neighbor distance d to extract K0d/Kd0 from the device matrix. "
            "d=1 corresponds to K01/K10."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Prefix of output files. Defaults to "
            "<device-matrix-dir>/transport_blocks_<device-matrix-stem>"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("npy", "txt"),
        default="npy",
        help="Output format for saved blocks.",
    )
    parser.add_argument(
        "--export-transverse-hoppings",
        action="store_true",
        help=(
            "Export transverse-resolved hopping blocks from K00/K01/K10 using metadata "
            "atom ordering."
        ),
    )
    parser.add_argument(
        "--q-point",
        type=float,
        nargs=2,
        action="append",
        metavar=("Q1", "Q2"),
        help=(
            "Reduced transverse q-point components (can be passed multiple times). "
            "Computes q-resolved K00(q), K01(q), K10(q) from transverse hoppings."
        ),
    )
    parser.add_argument(
        "--q-grid",
        type=int,
        nargs=2,
        metavar=("NQ1", "NQ2"),
        default=None,
        help=(
            "Generate a uniform transverse q-grid and compute q-resolved blocks for all points. "
            "Grid points are (i/NQ1, j/NQ2) for i=0..NQ1-1, j=0..NQ2-1."
        ),
    )
    parser.add_argument(
        "--gamma-centered",
        action="store_true",
        help=(
            "Use Gamma-centered reduced coordinates for --q-grid: "
            "((i+0.5)/NQ1-0.5, (j+0.5)/NQ2-0.5)."
        ),
    )
    return parser


def load_metadata(metadata_path: Path) -> dict | None:
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_dof_per_cell(args, metadata: dict | None) -> int:
    if args.dof_per_cell is not None:
        if args.dof_per_cell <= 0:
            raise RuntimeError("--dof-per-cell must be positive.")
        return args.dof_per_cell

    if metadata is not None and "degrees_of_freedom_per_transport_cell" in metadata:
        value = int(metadata["degrees_of_freedom_per_transport_cell"])
        if value <= 0:
            raise RuntimeError("Invalid degrees_of_freedom_per_transport_cell in metadata.")
        return value

    raise RuntimeError(
        "Cannot determine DoF per transport cell. Provide --dof-per-cell or valid metadata."
    )


def choose_reference_cell(num_cells: int, requested: int | None) -> int:
    if requested is not None:
        if requested < 0 or requested >= num_cells:
            raise RuntimeError(
                f"--reference-cell out of range: {requested} (valid: 0..{num_cells - 1})"
            )
        return requested

    if num_cells >= 3:
        return num_cells // 2
    return 0


def save_block(block, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        import numpy as np

        np.save(path, block)
        return

    with path.open("w", encoding="utf-8") as out:
        out.write(f"# shape: {block.shape}\n")
        for row in block:
            out.write(" ".join(f"{value: .16e}" for value in row) + "\n")


def center_shift(value: int, period: int) -> int:
    if period <= 1:
        return 0
    return int(((value + period // 2) % period) - period // 2)


def build_transverse_hoppings(metadata: dict, K00, couplings_by_delta: dict[int, tuple], dof_per_cell: int):
    import numpy as np

    transport_axis = metadata.get("transport_axis")
    if transport_axis not in {"a", "b", "c"}:
        raise RuntimeError("Metadata missing valid 'transport_axis' (a, b, or c).")

    repeats = metadata.get("supercell_repeats")
    if not isinstance(repeats, list) or len(repeats) != 3:
        raise RuntimeError("Metadata missing valid 'supercell_repeats'.")
    repeats = [int(value) for value in repeats]

    atom_order = metadata.get("transport_atom_order")
    if not isinstance(atom_order, list) or len(atom_order) * 3 != dof_per_cell:
        raise RuntimeError(
            "Metadata 'transport_atom_order' is missing or incompatible with dof-per-cell."
        )

    axis_index = {"a": 0, "b": 1, "c": 2}[transport_axis]
    transverse_dirs = [index for index in (0, 1, 2) if index != axis_index]
    transverse_periods = [repeats[index] for index in transverse_dirs]

    grouped: dict[tuple[int, ...], list[tuple[tuple[int, int], int]]] = {}
    for atom_index, entry in enumerate(atom_order):
        offset_raw = entry.get("cell_offset")
        primitive_raw = entry.get("primitive_atom_index")
        if not isinstance(offset_raw, list) or len(offset_raw) != 3:
            raise RuntimeError("Invalid 'cell_offset' entry in metadata atom ordering.")
        offset = [int(value) for value in offset_raw]
        primitive_index = int(primitive_raw)

        trans_key = tuple(offset[index] for index in transverse_dirs)
        basis_key = (offset[axis_index], primitive_index)
        grouped.setdefault(trans_key, []).append((basis_key, atom_index))

    if not grouped:
        raise RuntimeError("No transverse grouping information found in metadata atom ordering.")

    zero_key = (0, 0)
    reference_transverse_key = zero_key if zero_key in grouped else sorted(grouped.keys())[0]

    reference_basis = sorted(grouped[reference_transverse_key], key=lambda item: item[0])
    basis_order = [item[0] for item in reference_basis]
    basis_size = len(basis_order)
    if basis_size == 0:
        raise RuntimeError("Empty basis in reference transverse cell.")

    indices_by_transverse: dict[tuple[int, ...], list[int]] = {}
    for trans_key, entries in grouped.items():
        index_map = {basis_key: atom_index for basis_key, atom_index in entries}
        missing = [basis_key for basis_key in basis_order if basis_key not in index_map]
        if missing:
            raise RuntimeError(
                f"Transverse cell {trans_key} is missing basis entries: {missing[:4]}"
            )

        ordered_atom_indices = [index_map[basis_key] for basis_key in basis_order]
        dof_indices: list[int] = []
        for atom_index in ordered_atom_indices:
            dof_indices.extend([3 * atom_index, 3 * atom_index + 1, 3 * atom_index + 2])
        indices_by_transverse[trans_key] = dof_indices

    reference_dof_indices = indices_by_transverse[reference_transverse_key]
    dof_per_transverse_basis = len(reference_dof_indices)

    dt_values: list[tuple[int, int]] = []
    K00_blocks: list[Any] = []
    deltas = sorted(couplings_by_delta.keys())
    K0d_blocks: dict[int, list[Any]] = {delta: [] for delta in deltas}
    Kd0_blocks: dict[int, list[Any]] = {delta: [] for delta in deltas}

    for trans_key in sorted(indices_by_transverse.keys()):
        dt = tuple(
            center_shift(trans_key[i] - reference_transverse_key[i], transverse_periods[i])
            for i in range(2)
        )
        target_dof_indices = indices_by_transverse[trans_key]

        K00_blocks.append(K00[np.ix_(reference_dof_indices, target_dof_indices)])
        for delta in deltas:
            K0d, Kd0 = couplings_by_delta[delta]
            K0d_blocks[delta].append(K0d[np.ix_(reference_dof_indices, target_dof_indices)])
            Kd0_blocks[delta].append(Kd0[np.ix_(reference_dof_indices, target_dof_indices)])
        dt_values.append((int(dt[0]), int(dt[1])))

    dt_array = np.asarray(dt_values, dtype=int)
    K00_array = np.asarray(K00_blocks)
    K0d_array = np.asarray([np.asarray(K0d_blocks[delta]) for delta in deltas])
    Kd0_array = np.asarray([np.asarray(Kd0_blocks[delta]) for delta in deltas])

    info = {
        "transport_axis": transport_axis,
        "axis_index": int(axis_index),
        "transverse_dirs": transverse_dirs,
        "transverse_periods": [int(value) for value in transverse_periods],
        "reference_transverse_key": [
            int(reference_transverse_key[0]),
            int(reference_transverse_key[1]),
        ],
        "num_transverse_cells": int(len(indices_by_transverse)),
        "basis_atoms_per_transverse_cell": int(basis_size),
        "dof_per_transverse_basis": int(dof_per_transverse_basis),
        "longitudinal_deltas": [int(delta) for delta in deltas],
    }
    return dt_array, K00_array, K0d_array, Kd0_array, np.asarray(deltas, dtype=int), info


def q_label(q1: float, q2: float) -> str:
    def part(value: float) -> str:
        text = f"{value:+.6f}".replace("+", "p").replace("-", "m")
        return text.replace(".", "d")

    return f"q_{part(q1)}_{part(q2)}"


def build_q_resolved_blocks(dt_array, K00_blocks, K0d_blocks, Kd0_blocks, deltas, q_point):
    import numpy as np

    q1, q2 = float(q_point[0]), float(q_point[1])
    phases = np.exp(2j * np.pi * (q1 * dt_array[:, 0] + q2 * dt_array[:, 1]))

    K00_q = np.tensordot(phases, K00_blocks, axes=(0, 0))
    K0d_q = np.tensordot(phases, K0d_blocks, axes=(0, 1))
    Kd0_q = np.tensordot(phases, Kd0_blocks, axes=(0, 1))

    first = np.where(deltas == 1)[0]
    if len(first) == 0:
        raise RuntimeError("Missing delta=1 coupling in transverse hopping decomposition.")
    first_index = int(first[0])
    K01_q = K0d_q[first_index]
    K10_q = Kd0_q[first_index]
    return K00_q, K01_q, K10_q


def build_q_points(args) -> list[tuple[float, float]]:
    q_points: list[tuple[float, float]] = []

    for q_point in args.q_point or []:
        q_points.append((float(q_point[0]), float(q_point[1])))

    if args.q_grid is not None:
        nq1 = int(args.q_grid[0])
        nq2 = int(args.q_grid[1])
        if nq1 <= 0 or nq2 <= 0:
            raise RuntimeError("--q-grid values must be positive integers.")

        for i in range(nq1):
            for j in range(nq2):
                if args.gamma_centered:
                    q1 = (i + 0.5) / nq1 - 0.5
                    q2 = (j + 0.5) / nq2 - 0.5
                else:
                    q1 = i / nq1
                    q2 = j / nq2
                q_points.append((q1, q2))

    deduped: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for q1, q2 in q_points:
        key = (round(q1, 12), round(q2, 12))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((q1, q2))

    return deduped


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is not installed in the current Python environment.") from exc

    device_matrix_path = args.device_matrix.expanduser().resolve()
    if not device_matrix_path.exists():
        raise RuntimeError(f"Device matrix file not found: {device_matrix_path}")

    metadata_path = args.metadata
    if metadata_path is None:
        metadata_path = device_matrix_path.with_name(f"{device_matrix_path.stem}_metadata.json")
    metadata_path = metadata_path.expanduser().resolve()
    metadata = load_metadata(metadata_path)

    matrix = np.load(device_matrix_path)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError(f"Device matrix must be square 2D. Got shape {matrix.shape}")

    dof_per_cell = resolve_dof_per_cell(args, metadata)
    if matrix.shape[0] % dof_per_cell != 0:
        raise RuntimeError(
            f"Matrix dimension {matrix.shape[0]} is not divisible by dof-per-cell {dof_per_cell}."
        )

    num_cells = matrix.shape[0] // dof_per_cell
    if num_cells < 2:
        raise RuntimeError(
            "Need at least two transport cells in the device matrix to extract K01/K10."
        )

    reference_cell = choose_reference_cell(num_cells, args.reference_cell)
    if reference_cell + 1 >= num_cells:
        raise RuntimeError(
            f"Reference cell {reference_cell} has no right neighbor in a {num_cells}-cell matrix."
        )

    r0 = reference_cell * dof_per_cell
    r1 = (reference_cell + 1) * dof_per_cell

    K00 = matrix[r0:r0 + dof_per_cell, r0:r0 + dof_per_cell]
    if args.max_neighbor < 1:
        raise RuntimeError("--max-neighbor must be >= 1.")

    couplings_by_delta: dict[int, tuple[Any, Any]] = {}
    for delta in range(1, int(args.max_neighbor) + 1):
        right_cell = reference_cell + delta
        if right_cell >= num_cells:
            break
        c0 = right_cell * dof_per_cell
        K0d = matrix[r0:r0 + dof_per_cell, c0:c0 + dof_per_cell]
        Kd0 = matrix[c0:c0 + dof_per_cell, r0:r0 + dof_per_cell]
        couplings_by_delta[delta] = (K0d, Kd0)

    if 1 not in couplings_by_delta:
        raise RuntimeError(
            f"Reference cell {reference_cell} has no delta=1 neighbor in a {num_cells}-cell matrix."
        )

    K01, K10 = couplings_by_delta[1]

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = device_matrix_path.parent / f"transport_blocks_{device_matrix_path.stem}"
    output_prefix = output_prefix.expanduser().resolve()

    ext = ".npy" if args.format == "npy" else ".txt"
    k00_path = output_prefix.with_name(f"{output_prefix.name}_K00{ext}")
    k01_path = output_prefix.with_name(f"{output_prefix.name}_K01{ext}")
    k10_path = output_prefix.with_name(f"{output_prefix.name}_K10{ext}")

    save_block(K00, k00_path, args.format)
    save_block(K01, k01_path, args.format)
    save_block(K10, k10_path, args.format)

    summary = {
        "device_matrix": str(device_matrix_path),
        "metadata": str(metadata_path) if metadata is not None else None,
        "dof_per_cell": int(dof_per_cell),
        "num_cells": int(num_cells),
        "reference_cell": int(reference_cell),
        "K00": str(k00_path),
        "K01": str(k01_path),
        "K10": str(k10_path),
        "max_neighbor": int(args.max_neighbor),
        "available_longitudinal_deltas": [int(delta) for delta in sorted(couplings_by_delta.keys())],
    }

    summary["K00_max_antisymmetry"] = float(np.abs(K00 - K00.T).max())
    summary["K10_minus_K01T_max"] = float(np.abs(K10 - K01.T).max())

    q_points = build_q_points(args)

    if args.export_transverse_hoppings or q_points:
        if metadata is None:
            raise RuntimeError(
                "Metadata is required to resolve transverse hopping blocks and q-points."
            )

        dt_array, K00_blocks, K0d_blocks, Kd0_blocks, deltas, transverse_info = build_transverse_hoppings(
            metadata=metadata,
            K00=K00,
            couplings_by_delta=couplings_by_delta,
            dof_per_cell=dof_per_cell,
        )

        transverse_npz_path = output_prefix.with_name(
            f"{output_prefix.name}_transverse_hoppings.npz"
        )
        np.savez(
            transverse_npz_path,
            dt=dt_array,
            K00=K00_blocks,
            deltas=deltas,
            K0d=K0d_blocks,
            Kd0=Kd0_blocks,
            K01=K0d_blocks[int((deltas == 1).nonzero()[0][0])],
            K10=Kd0_blocks[int((deltas == 1).nonzero()[0][0])],
        )
        summary["transverse_hoppings"] = str(transverse_npz_path)
        summary["transverse_hopping_info"] = transverse_info

        q_results = []
        for q_point in q_points:
            K00_q, K01_q, K10_q = build_q_resolved_blocks(
                dt_array=dt_array,
                K00_blocks=K00_blocks,
                K0d_blocks=K0d_blocks,
                Kd0_blocks=Kd0_blocks,
                deltas=deltas,
                q_point=q_point,
            )
            label = q_label(float(q_point[0]), float(q_point[1]))
            q_k00_path = output_prefix.with_name(f"{output_prefix.name}_{label}_K00.npy")
            q_k01_path = output_prefix.with_name(f"{output_prefix.name}_{label}_K01.npy")
            q_k10_path = output_prefix.with_name(f"{output_prefix.name}_{label}_K10.npy")
            np.save(q_k00_path, K00_q)
            np.save(q_k01_path, K01_q)
            np.save(q_k10_path, K10_q)
            q_results.append(
                {
                    "q": [float(q_point[0]), float(q_point[1])],
                    "K00": str(q_k00_path),
                    "K01": str(q_k01_path),
                    "K10": str(q_k10_path),
                    "K00_max_antihermitian": float(np.abs(K00_q - K00_q.conj().T).max()),
                    "K10_minus_K01H_max": float(np.abs(K10_q - K01_q.conj().T).max()),
                }
            )

        if q_results:
            summary["q_resolved_blocks"] = q_results
            if args.q_grid is not None:
                summary["q_grid"] = [int(args.q_grid[0]), int(args.q_grid[1])]
                summary["gamma_centered"] = bool(args.gamma_centered)

    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Loaded device matrix: {device_matrix_path}")
    if metadata is not None:
        print(f"Loaded metadata: {metadata_path}")
    else:
        print("Metadata not found; used --dof-per-cell for block size.")
    print(f"Matrix shape: {matrix.shape}")
    print(f"DoF per cell: {dof_per_cell}")
    print(f"Transport cells: {num_cells}")
    print(f"Reference cell: {reference_cell}")
    print(f"Saved K00: {k00_path}")
    print(f"Saved K01: {k01_path}")
    print(f"Saved K10: {k10_path}")
    if "transverse_hoppings" in summary:
        print(f"Saved transverse hoppings: {summary['transverse_hoppings']}")
    if "q_resolved_blocks" in summary:
        for item in summary["q_resolved_blocks"]:
            print(
                "Saved q-resolved blocks at q="
                f"({item['q'][0]:.6f}, {item['q'][1]:.6f}): "
                f"{item['K00']}, {item['K01']}, {item['K10']}"
            )
    print(f"Saved summary: {summary_path}")

    print(f"K00 max |K00-K00^T|: {summary['K00_max_antisymmetry']:.6e}")
    print(f"max |K10-K01^T|: {summary['K10_minus_K01T_max']:.6e}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
