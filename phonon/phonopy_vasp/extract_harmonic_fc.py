#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract harmonic force-constant tensor from phonopy outputs."
    )
    parser.add_argument(
        "--harmonic-dir",
        type=Path,
        default=PROJECT_ROOT / "phonon-data/si/harmonic",
        help="Directory containing phonopy output files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output tensor path. Defaults to <harmonic-dir>/harmonic_fc_tensor.npy",
    )
    parser.add_argument(
        "--format",
        choices=("npy", "txt"),
        default="npy",
        help="Output format for extracted tensor.",
    )
    parser.add_argument(
        "--prefer",
        choices=("auto", "force-constants", "force-sets"),
        default="auto",
        help="Preferred source for FC reconstruction.",
    )
    parser.add_argument(
        "--device-axis",
        choices=("a", "b", "c"),
        default=None,
        help=(
            "Repeat the phonopy supercell along the chosen lattice axis and "
            "assemble a longer device FC matrix for transport."
        ),
    )
    parser.add_argument(
        "--device-cells",
        type=int,
        default=None,
        help="Number of repeated transport cells in the device FC matrix.",
    )
    parser.add_argument(
        "--device-output",
        type=Path,
        default=None,
        help=(
            "Output path for the device FC matrix. Defaults to "
            "<harmonic-dir>/device_fc_matrix_<axis>_<cells>.<device-format>"
        ),
    )
    parser.add_argument(
        "--device-format",
        choices=("npy", "txt"),
        default="npy",
        help="Output format for the assembled device FC matrix.",
    )
    parser.add_argument(
        "--device-no-asr",
        action="store_true",
        help=(
            "Disable ASR projection during device FC construction. By default, "
            "ASR is imposed on the transport-cell q=0 block sum."
        ),
    )
    return parser


def resolve_input_files(harmonic_dir: Path, prefer: str) -> tuple[Path, str, Path]:
    yaml_candidates = [harmonic_dir / "phonopy_disp.yaml", harmonic_dir / "phonopy.yaml"]
    yaml_path = next((path for path in yaml_candidates if path.exists()), None)
    if yaml_path is None:
        raise RuntimeError(
            f"Missing phonopy YAML in {harmonic_dir}. Expected one of: "
            f"{', '.join(str(path.name) for path in yaml_candidates)}"
        )

    force_constants = harmonic_dir / "FORCE_CONSTANTS"
    force_sets = harmonic_dir / "FORCE_SETS"

    source_order: list[tuple[str, Path]]
    if prefer == "force-constants":
        source_order = [("force-constants", force_constants), ("force-sets", force_sets)]
    elif prefer == "force-sets":
        source_order = [("force-sets", force_sets), ("force-constants", force_constants)]
    else:
        source_order = [("force-constants", force_constants), ("force-sets", force_sets)]

    source_name, source_path = next(((name, path) for name, path in source_order if path.exists()), (None, None))
    if source_name is None or source_path is None:
        raise RuntimeError(
            f"Missing force data in {harmonic_dir}. Need FORCE_CONSTANTS or FORCE_SETS."
        )

    return yaml_path, source_name, source_path


def load_force_constants(harmonic_dir: Path, prefer: str):
    try:
        import phonopy
    except ImportError as exc:
        raise RuntimeError("phonopy is not installed in the current Python environment.") from exc

    yaml_path, source_name, source_path = resolve_input_files(harmonic_dir, prefer)

    if source_name == "force-constants":
        phonon = phonopy.load(
            phonopy_yaml=str(yaml_path),
            force_constants_filename=str(source_path),
        )
    else:
        phonon = phonopy.load(
            phonopy_yaml=str(yaml_path),
            force_sets_filename=str(source_path),
        )
        if phonon.force_constants is None:
            phonon.produce_force_constants()

    force_constants = phonon.force_constants
    if force_constants is None:
        raise RuntimeError("Failed to build force constants from phonopy inputs.")

    return phonon, force_constants, source_name, yaml_path, source_path


def save_tensor_npy(tensor, output_path: Path) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is not installed in the current Python environment.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, tensor)


def save_tensor_txt(tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_i = tensor.shape[0]
    n_j = tensor.shape[1]

    with output_path.open("w", encoding="utf-8") as out:
        out.write(f"# shape: {tensor.shape}\n")
        out.write("# blocks: i j then 3x3 matrix\n")
        for i in range(n_i):
            for j in range(n_j):
                out.write(f"{i + 1:4d} {j + 1:4d}\n")
                block = tensor[i, j]
                for row in block:
                    out.write(" ".join(f"{value: .16e}" for value in row) + "\n")


def save_matrix_npy(matrix, output_path: Path) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is not installed in the current Python environment.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix)


def save_matrix_txt(matrix, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        out.write(f"# shape: {matrix.shape}\n")
        out.write("# dense device FC matrix in eV/Angstrom^2\n")
        for row in matrix:
            out.write(" ".join(f"{value: .16e}" for value in row) + "\n")


def save_device_metadata(metadata: dict, output_path: Path) -> Path:
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def require_diagonal_supercell_matrix(phonon):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is not installed in the current Python environment.") from exc

    supercell_matrix = np.array(phonon.supercell_matrix, dtype=int)
    diagonal_matrix = np.diag(np.diag(supercell_matrix))
    if not np.array_equal(supercell_matrix, diagonal_matrix):
        raise RuntimeError(
            "Device FC construction currently requires a diagonal phonopy supercell_matrix."
        )

    repeats = np.diag(supercell_matrix).astype(int)
    if (repeats <= 0).any():
        raise RuntimeError(f"Invalid supercell repeats: {repeats.tolist()}")
    return repeats


def to_compact_force_constants(phonon, force_constants):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is not installed in the current Python environment.") from exc

    primitive = phonon.primitive
    supercell = phonon.supercell
    num_prim = len(primitive)
    num_super = len(supercell)
    shape = force_constants.shape

    if len(shape) != 4 or shape[2:] != (3, 3):
        raise RuntimeError(f"Unexpected force-constant tensor shape: {shape}")
    if shape[0] == num_prim and shape[1] == num_super:
        return force_constants
    if shape[0] == num_super and shape[1] == num_super:
        row_map = np.asarray(primitive.p2s_map, dtype=int)
        return force_constants[row_map]

    raise RuntimeError(
        "Unable to convert force constants to phonopy compact form from shape "
        f"{shape}."
    )


def center_translation(translation: int, period: int) -> int:
    if period <= 1:
        return 0
    return int(((translation + period // 2) % period) - period // 2)


def transpose_interaction_block(block):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is not installed in the current Python environment.") from exc

    return np.transpose(block, (1, 0, 3, 2))


def build_primitive_interaction_blocks(phonon, force_constants):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is not installed in the current Python environment.") from exc

    primitive = phonon.primitive
    compact_fc = to_compact_force_constants(phonon, force_constants)
    repeats = require_diagonal_supercell_matrix(phonon)
    num_prim = len(primitive)

    # Phonopy stores shortest vectors in primitive fractional coordinates with
    # multiplicity/address tables used when assembling D(q).
    shortest_vectors, multiplicity = primitive.get_smallest_vectors()
    primitive_positions = np.asarray(primitive.scaled_positions, dtype=float)
    s2p_map = np.asarray(primitive.s2p_map, dtype=int)
    p2p_map = primitive.p2p_map

    super_indices_by_primitive: list[list[int]] = [[] for _ in range(num_prim)]
    for super_j, primitive_anchor in enumerate(s2p_map):
        primitive_j = int(p2p_map[primitive_anchor])
        super_indices_by_primitive[primitive_j].append(super_j)

    tol = 1e-7
    stored: dict[tuple[int, int, int], Any] = {}
    for primitive_i in range(num_prim):
        for primitive_j in range(num_prim):
            tau_delta = primitive_positions[primitive_j] - primitive_positions[primitive_i]
            for super_j in super_indices_by_primitive[primitive_j]:
                mult = int(multiplicity[super_j, primitive_i, 0])
                adrs = int(multiplicity[super_j, primitive_i, 1])
                if mult <= 0:
                    continue

                block = compact_fc[primitive_i, super_j] / float(mult)
                vectors = shortest_vectors[adrs : adrs + mult]
                for vector in vectors:
                    translation_float = vector - tau_delta
                    translation_rounded = np.rint(translation_float).astype(int)
                    if np.max(np.abs(translation_float - translation_rounded)) > tol:
                        raise RuntimeError(
                            "Failed to recover integer primitive translation from "
                            "phonopy shortest-vector mapping."
                        )
                    key = tuple(int(value) for value in translation_rounded.tolist())
                    interaction = stored.setdefault(
                        key,
                        np.zeros((num_prim, num_prim, 3, 3), dtype=float),
                    )
                    interaction[primitive_i, primitive_j] += block

    # Enforce Phi_ij(R) = Phi_ji(-R)^T for numerical stability.
    completed: dict[tuple[int, int, int], Any] = {}
    processed: set[tuple[int, int, int]] = set()
    for translation, block in stored.items():
        if translation in processed:
            continue
        negative_translation = tuple(-value for value in translation)
        partner = stored.get(negative_translation)
        if partner is None:
            symmetrized = 0.5 * (block + transpose_interaction_block(block))
            completed[translation] = symmetrized
            completed[negative_translation] = transpose_interaction_block(symmetrized)
        else:
            averaged = 0.5 * (block + transpose_interaction_block(partner))
            completed[translation] = averaged
            completed[negative_translation] = transpose_interaction_block(averaged)
            processed.add(negative_translation)
        processed.add(translation)

    return completed, repeats


def construct_device_fc_matrix(
    phonon,
    force_constants,
    axis: str,
    num_cells: int,
    impose_asr: bool = True,
):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is not installed in the current Python environment.") from exc

    if num_cells < 1:
        raise RuntimeError("--device-cells must be a positive integer.")

    axis_index = {"a": 0, "b": 1, "c": 2}[axis]
    interaction_blocks_raw, repeats = build_primitive_interaction_blocks(phonon, force_constants)
    primitive = phonon.primitive
    primitive_symbols = list(primitive.symbols)
    num_prim = len(primitive)

    transport_offsets = list(
        itertools.product(*(range(int(repeat)) for repeat in repeats.tolist()))
    )
    transport_atoms: list[tuple[Any, int]] = []
    transport_atom_order = []
    for offset in transport_offsets:
        offset_array = np.asarray(offset, dtype=int)
        for primitive_index in range(num_prim):
            transport_atoms.append((offset_array, primitive_index))
            transport_atom_order.append(
                {
                    "cell_offset": [int(value) for value in offset],
                    "primitive_atom_index": primitive_index + 1,
                    "symbol": primitive_symbols[primitive_index],
                }
            )

    transport_atoms_per_cell = len(transport_atoms)
    dof_per_transport_cell = transport_atoms_per_cell * 3
    segment_shift = np.zeros(3, dtype=int)
    segment_shift[axis_index] = int(repeats[axis_index])

    # Fold raw primitive-lattice interactions into transverse periodic images.
    interaction_blocks: dict[tuple[int, int, int], Any] = {}
    for translation, block in interaction_blocks_raw.items():
        folded = []
        for direction, value in enumerate(translation):
            if direction == axis_index:
                folded.append(int(value))
            else:
                folded.append(center_translation(int(value), int(repeats[direction])))
        key = tuple(folded)
        if key in interaction_blocks:
            interaction_blocks[key] = interaction_blocks[key] + block
        else:
            interaction_blocks[key] = block.copy()

    block_by_delta: dict[int, Any] = {}
    for delta in range(-(num_cells - 1), num_cells):
        block_matrix = np.zeros((dof_per_transport_cell, dof_per_transport_cell), dtype=float)
        has_nonzero = False

        for atom_i, (offset_i, primitive_i) in enumerate(transport_atoms):
            row = atom_i * 3
            for atom_j, (offset_j, primitive_j) in enumerate(transport_atoms):
                col = atom_j * 3
                raw_translation = offset_j + delta * segment_shift - offset_i
                translation = tuple(int(value) for value in raw_translation.tolist())

                interaction = interaction_blocks.get(translation)
                if interaction is None:
                    continue

                pair_block = interaction[primitive_i, primitive_j]
                block_matrix[row : row + 3, col : col + 3] = pair_block
                if not has_nonzero and abs(pair_block).max() > 0.0:
                    has_nonzero = True

        if has_nonzero:
            block_by_delta[delta] = block_matrix

    # Impose translational ASR on D(q=0)=sum_delta B_delta by correcting onsite block.
    # This keeps inter-cell couplings unchanged while restoring acoustic zero modes.
    asr_before = None
    asr_after = None
    if impose_asr and 0 in block_by_delta:
        num_dof = dof_per_transport_cell
        if num_dof % 3 != 0:
            raise RuntimeError("Transport-cell DoF is not divisible by 3 for ASR projection.")

        n_atoms = num_dof // 3
        translation = np.zeros((num_dof, 3), dtype=float)
        for cart in range(3):
            translation[cart::3, cart] = 1.0
        u = translation / np.sqrt(float(n_atoms))

        d0 = np.zeros((num_dof, num_dof), dtype=float)
        for _, block in block_by_delta.items():
            d0 = d0 + block

        row_sum_before = d0.sum(axis=1)
        asr_before = {
            "max_abs_row_sum": float(np.max(np.abs(row_sum_before))),
            "rms_row_sum": float(np.sqrt(np.mean(row_sum_before**2))),
            "norm_right_residual": float(np.linalg.norm(d0 @ u)),
            "norm_left_residual": float(np.linalg.norm(u.T @ d0)),
        }

        projector = np.eye(num_dof, dtype=float) - (u @ u.T)
        d0_asr = projector @ d0 @ projector
        d0_asr = 0.5 * (d0_asr + d0_asr.T)
        block_by_delta[0] = block_by_delta[0] + (d0_asr - d0)

        d0_new = np.zeros((num_dof, num_dof), dtype=float)
        for _, block in block_by_delta.items():
            d0_new = d0_new + block
        row_sum_after = d0_new.sum(axis=1)
        asr_after = {
            "max_abs_row_sum": float(np.max(np.abs(row_sum_after))),
            "rms_row_sum": float(np.sqrt(np.mean(row_sum_after**2))),
            "norm_right_residual": float(np.linalg.norm(d0_new @ u)),
            "norm_left_residual": float(np.linalg.norm(u.T @ d0_new)),
        }

    device_matrix = np.zeros(
        (num_cells * dof_per_transport_cell, num_cells * dof_per_transport_cell),
        dtype=float,
    )
    for cell_i in range(num_cells):
        row = cell_i * dof_per_transport_cell
        for delta, block_matrix in block_by_delta.items():
            cell_j = cell_i + delta
            if cell_j < 0 or cell_j >= num_cells:
                continue

            col = cell_j * dof_per_transport_cell
            device_matrix[
                row : row + dof_per_transport_cell,
                col : col + dof_per_transport_cell,
            ] = block_matrix

    device_matrix = 0.5 * (device_matrix + device_matrix.T)
    metadata = {
        "transport_axis": axis,
        "device_cells": int(num_cells),
        "transverse_periodic_wrapping": True,
        "supercell_repeats": [int(value) for value in repeats.tolist()],
        "primitive_atoms_per_cell": int(num_prim),
        "transport_atoms_per_cell": int(transport_atoms_per_cell),
        "degrees_of_freedom_per_transport_cell": int(dof_per_transport_cell),
        "device_matrix_shape": [int(value) for value in device_matrix.shape],
        "asr_applied": bool(impose_asr and asr_before is not None),
        "asr_before": asr_before,
        "asr_after": asr_after,
        "nonzero_segment_couplings": [
            {
                "delta": int(delta),
                "frobenius_norm": float((block_matrix**2).sum() ** 0.5),
            }
            for delta, block_matrix in sorted(block_by_delta.items())
        ],
        "transport_atom_order": transport_atom_order,
    }
    return device_matrix, metadata


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    harmonic_dir = args.harmonic_dir.expanduser().resolve()
    if not harmonic_dir.exists():
        raise RuntimeError(f"Harmonic directory not found: {harmonic_dir}")

    output = args.output
    if output is None:
        suffix = ".npy" if args.format == "npy" else ".txt"
        output = harmonic_dir / f"harmonic_fc_tensor{suffix}"
    output = output.expanduser().resolve()

    phonon, tensor, source_name, yaml_path, source_path = load_force_constants(
        harmonic_dir=harmonic_dir,
        prefer=args.prefer,
    )

    if args.format == "npy":
        save_tensor_npy(tensor, output)
    else:
        save_tensor_txt(tensor, output)

    if (args.device_axis is None) != (args.device_cells is None):
        raise RuntimeError("Use --device-axis and --device-cells together.")

    device_output = None
    metadata_output = None
    if args.device_axis is not None and args.device_cells is not None:
        device_output = args.device_output
        if device_output is None:
            device_suffix = ".npy" if args.device_format == "npy" else ".txt"
            device_output = harmonic_dir / (
                f"device_fc_matrix_{args.device_axis}_{args.device_cells}{device_suffix}"
            )
        device_output = device_output.expanduser().resolve()

        device_matrix, metadata = construct_device_fc_matrix(
            phonon=phonon,
            force_constants=tensor,
            axis=args.device_axis,
            num_cells=args.device_cells,
            impose_asr=not args.device_no_asr,
        )
        if args.device_format == "npy":
            save_matrix_npy(device_matrix, device_output)
        else:
            save_matrix_txt(device_matrix, device_output)
        metadata_output = save_device_metadata(metadata, device_output)

    print(f"YAML source: {yaml_path}")
    print(f"Force-data source ({source_name}): {source_path}")
    print(f"Extracted tensor shape: {tensor.shape}")
    print(f"Saved harmonic FC tensor: {output}")
    if device_output is not None:
        print(
            "Constructed device FC matrix by repeating the phonopy supercell "
            f"{args.device_cells} times along axis {args.device_axis}."
        )
        print(f"Saved device FC matrix: {device_output}")
        if metadata_output is not None:
            print(f"Saved device metadata: {metadata_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
