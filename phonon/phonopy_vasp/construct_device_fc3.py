#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from extract_fc3_tensor import parse_fc3_file

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_axis(axis: str) -> int:
    mapping = {"a": 0, "b": 1, "c": 2}
    if axis not in mapping:
        raise RuntimeError(f"Invalid axis: {axis}")
    return mapping[axis]


def center_translation(value: int, period: int) -> int:
    if period <= 1:
        return 0
    return int(((value + period // 2) % period) - period // 2)


def require_diagonal_supercell_matrix(supercell_matrix):
    import numpy as np

    matrix = np.array(supercell_matrix, dtype=int)
    diagonal_matrix = np.diag(np.diag(matrix))
    if not np.array_equal(matrix, diagonal_matrix):
        raise RuntimeError("FC3 device construction currently requires diagonal supercell_matrix.")

    repeats = np.diag(matrix).astype(int)
    if (repeats <= 0).any():
        raise RuntimeError(f"Invalid supercell repeats: {repeats.tolist()}")
    return repeats


def resolve_harmonic_context(harmonic_dir: Path):
    try:
        import phonopy
    except ImportError as exc:
        raise RuntimeError("phonopy is not installed in the current Python environment.") from exc

    yaml_candidates = [harmonic_dir / "phonopy_disp.yaml", harmonic_dir / "phonopy.yaml"]
    yaml_path = next((path for path in yaml_candidates if path.exists()), None)
    if yaml_path is None:
        raise RuntimeError(
            f"Missing phonopy YAML in {harmonic_dir}. Expected one of: "
            f"{', '.join(path.name for path in yaml_candidates)}"
        )

    phonon = phonopy.load(phonopy_yaml=str(yaml_path))
    primitive = phonon.primitive
    repeats = require_diagonal_supercell_matrix(phonon.supercell_matrix)

    return {
        "phonon": phonon,
        "primitive": primitive,
        "repeats": repeats,
        "yaml_path": yaml_path,
    }


def convert_fc3_block_translations(rj, rk, primitive_cell):
    import numpy as np

    inv_primitive_cell = np.linalg.inv(primitive_cell)
    t_j = np.rint(np.array(rj, dtype=float) @ inv_primitive_cell).astype(int)
    t_k = np.rint(np.array(rk, dtype=float) @ inv_primitive_cell).astype(int)
    return t_j, t_k


def decompose_transport_coordinate(value: int, period: int) -> tuple[int, int]:
    quotient = value // period
    remainder = value % period
    return int(quotient), int(remainder)


def make_transport_atom_index(repeats, num_prim: int):
    import itertools

    local_atoms: list[tuple[tuple[int, int, int], int, str]] = []
    for offset in itertools.product(*(range(int(rep)) for rep in repeats.tolist())):
        for primitive_index in range(num_prim):
            key = (int(offset[0]), int(offset[1]), int(offset[2]), primitive_index)
            local_atoms.append(((int(offset[0]), int(offset[1]), int(offset[2])), primitive_index, str(key)))

    local_atom_to_index: dict[tuple[int, int, int, int], int] = {}
    for local_index, (offset, primitive_index, _) in enumerate(local_atoms):
        local_atom_to_index[(offset[0], offset[1], offset[2], primitive_index)] = local_index

    return local_atoms, local_atom_to_index


def build_device_fc3_entries(
    fc3_data: dict,
    primitive,
    repeats,
    axis_index: int,
    num_device_cells: int,
):
    import numpy as np

    num_prim = len(primitive)
    local_atoms, local_atom_to_index = make_transport_atom_index(repeats=repeats, num_prim=num_prim)
    local_atoms_per_cell = len(local_atoms)

    tensors = fc3_data["tensors"]
    atoms = fc3_data["atoms"]
    rj = fc3_data["rj"]
    rk = fc3_data["rk"]

    primitive_cell = np.asarray(primitive.cell, dtype=float)

    blocks_by_i: dict[int, list[tuple[np.ndarray, np.ndarray, int, int, np.ndarray]]] = {}
    for block_index in range(fc3_data["parsed_blocks"]):
        atom_i = int(atoms[block_index][0]) - 1
        atom_j = int(atoms[block_index][1]) - 1
        atom_k = int(atoms[block_index][2]) - 1
        if atom_i < 0 or atom_i >= num_prim:
            raise RuntimeError(f"FC3 atom i index out of range at block {block_index + 1}.")
        if atom_j < 0 or atom_j >= num_prim:
            raise RuntimeError(f"FC3 atom j index out of range at block {block_index + 1}.")
        if atom_k < 0 or atom_k >= num_prim:
            raise RuntimeError(f"FC3 atom k index out of range at block {block_index + 1}.")

        translation_j, translation_k = convert_fc3_block_translations(
            rj=rj[block_index],
            rk=rk[block_index],
            primitive_cell=primitive_cell,
        )

        blocks_by_i.setdefault(atom_i, []).append(
            (
                translation_j,
                translation_k,
                atom_j,
                atom_k,
                np.asarray(tensors[block_index], dtype=float),
            )
        )

    entries: dict[tuple[int, int, int], np.ndarray] = {}

    axis_period = int(repeats[axis_index])
    transverse_dirs = [d for d in (0, 1, 2) if d != axis_index]

    for device_cell in range(num_device_cells):
        for local_i, (offset_i, atom_i, _) in enumerate(local_atoms):
            available_blocks = blocks_by_i.get(atom_i, [])
            if not available_blocks:
                continue

            for translation_j, translation_k, atom_j, atom_k, tensor in available_blocks:
                offset_j_raw = [offset_i[d] + int(translation_j[d]) for d in range(3)]
                offset_k_raw = [offset_i[d] + int(translation_k[d]) for d in range(3)]

                delta_cell_j, local_axis_j = decompose_transport_coordinate(
                    value=offset_j_raw[axis_index],
                    period=axis_period,
                )
                delta_cell_k, local_axis_k = decompose_transport_coordinate(
                    value=offset_k_raw[axis_index],
                    period=axis_period,
                )

                device_cell_j = device_cell + delta_cell_j
                device_cell_k = device_cell + delta_cell_k
                if device_cell_j < 0 or device_cell_j >= num_device_cells:
                    continue
                if device_cell_k < 0 or device_cell_k >= num_device_cells:
                    continue

                local_offset_j = [0, 0, 0]
                local_offset_k = [0, 0, 0]
                local_offset_j[axis_index] = local_axis_j
                local_offset_k[axis_index] = local_axis_k

                for direction in transverse_dirs:
                    period = int(repeats[direction])
                    wrapped_j = offset_j_raw[direction] % period
                    wrapped_k = offset_k_raw[direction] % period
                    local_offset_j[direction] = int(wrapped_j)
                    local_offset_k[direction] = int(wrapped_k)

                key_j = (local_offset_j[0], local_offset_j[1], local_offset_j[2], atom_j)
                key_k = (local_offset_k[0], local_offset_k[1], local_offset_k[2], atom_k)
                if key_j not in local_atom_to_index or key_k not in local_atom_to_index:
                    continue

                local_j = local_atom_to_index[key_j]
                local_k = local_atom_to_index[key_k]

                global_i = device_cell * local_atoms_per_cell + local_i
                global_j = device_cell_j * local_atoms_per_cell + local_j
                global_k = device_cell_k * local_atoms_per_cell + local_k

                key = (global_i, global_j, global_k)
                if key in entries:
                    entries[key] = entries[key] + tensor
                else:
                    entries[key] = tensor.copy()

    sorted_keys = sorted(entries.keys())
    global_i = np.array([key[0] for key in sorted_keys], dtype=int)
    global_j = np.array([key[1] for key in sorted_keys], dtype=int)
    global_k = np.array([key[2] for key in sorted_keys], dtype=int)
    tensor_values = np.stack([entries[key] for key in sorted_keys], axis=0)

    metadata = {
        "num_device_cells": int(num_device_cells),
        "transport_axis_index": int(axis_index),
        "supercell_repeats": [int(value) for value in repeats.tolist()],
        "primitive_atoms": int(num_prim),
        "transport_atoms_per_cell": int(local_atoms_per_cell),
        "device_atoms": int(num_device_cells * local_atoms_per_cell),
        "num_fc3_entries": int(len(sorted_keys)),
        "transverse_periodic_wrapping": True,
        "transport_boundaries": "open",
    }

    return global_i, global_j, global_k, tensor_values, metadata


def save_npz(output_path: Path, i, j, k, tensors, metadata: dict) -> None:
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        i=i,
        j=j,
        k=k,
        tensors=tensors,
    )

    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_txt(output_path: Path, i, j, k, tensors, metadata: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        out.write("# device FC3 sparse entries\n")
        out.write(f"# metadata: {json.dumps(metadata)}\n")
        out.write("# columns: i j k then 27 tensor values (alpha,beta,gamma; gamma fastest)\n")
        for idx in range(len(i)):
            head = f"{int(i[idx])} {int(j[idx])} {int(k[idx])}"
            values = " ".join(f"{value:.16e}" for value in tensors[idx].reshape(-1))
            out.write(head + " " + values + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct device-level sparse FC3 entries from FORCE_CONSTANTS_3RD."
    )
    parser.add_argument(
        "--fc3-file",
        type=Path,
        default=PROJECT_ROOT / "phonon-data/si/anharmonic/FORCE_CONSTANTS_3RD",
        help="Input FORCE_CONSTANTS_3RD path.",
    )
    parser.add_argument(
        "--harmonic-dir",
        type=Path,
        default=PROJECT_ROOT / "phonon-data/si/harmonic",
        help="Directory containing phonopy_disp.yaml or phonopy.yaml.",
    )
    parser.add_argument(
        "--device-axis",
        choices=("a", "b", "c"),
        default="a",
        help="Transport axis for device construction.",
    )
    parser.add_argument(
        "--device-cells",
        type=int,
        default=8,
        help="Number of transport cells in the device.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to <fc3_dir>/device_fc3_<axis>_<cells>.<format>",
    )
    parser.add_argument(
        "--format",
        choices=("npz", "txt"),
        default="npz",
        help="Output format for sparse device FC3 entries.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    fc3_path = args.fc3_file.expanduser().resolve()
    if not fc3_path.exists():
        raise RuntimeError(f"FC3 file not found: {fc3_path}")

    harmonic_dir = args.harmonic_dir.expanduser().resolve()
    if not harmonic_dir.exists():
        raise RuntimeError(f"Harmonic directory not found: {harmonic_dir}")

    if args.device_cells < 1:
        raise RuntimeError("--device-cells must be a positive integer.")

    output_path = args.output
    if output_path is None:
        suffix = ".npz" if args.format == "npz" else ".txt"
        output_path = fc3_path.parent / f"device_fc3_{args.device_axis}_{args.device_cells}{suffix}"
    output_path = output_path.expanduser().resolve()

    fc3_data = parse_fc3_file(fc3_path)
    harmonic_context = resolve_harmonic_context(harmonic_dir)
    axis_index = parse_axis(args.device_axis)

    i, j, k, tensors, metadata = build_device_fc3_entries(
        fc3_data=fc3_data,
        primitive=harmonic_context["primitive"],
        repeats=harmonic_context["repeats"],
        axis_index=axis_index,
        num_device_cells=args.device_cells,
    )

    metadata["fc3_file"] = str(fc3_path)
    metadata["harmonic_yaml"] = str(harmonic_context["yaml_path"])
    metadata["declared_fc3_blocks"] = int(fc3_data["declared_blocks"])
    metadata["parsed_fc3_blocks"] = int(fc3_data["parsed_blocks"])

    if args.format == "npz":
        save_npz(output_path=output_path, i=i, j=j, k=k, tensors=tensors, metadata=metadata)
    else:
        save_txt(output_path=output_path, i=i, j=j, k=k, tensors=tensors, metadata=metadata)

    print(f"FC3 source: {fc3_path}")
    print(f"Harmonic YAML: {harmonic_context['yaml_path']}")
    print(f"Device axis: {args.device_axis}")
    print(f"Device cells: {args.device_cells}")
    print(f"Sparse FC3 entries: {len(i)}")
    print(f"Saved device FC3: {output_path}")
    if args.format == "npz":
        print(f"Saved metadata: {output_path.with_name(f'{output_path.stem}_metadata.json')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
