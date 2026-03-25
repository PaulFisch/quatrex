#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def center_shift(value: int, period: int) -> int:
    if period <= 1:
        return 0
    return int(((value + period // 2) % period) - period // 2)


def load_inputs(device_fc3_path: Path, metadata_path: Path | None):
    import numpy as np

    if not device_fc3_path.exists():
        raise RuntimeError(f"Device FC3 file not found: {device_fc3_path}")

    if metadata_path is None:
        metadata_path = device_fc3_path.with_name(f"{device_fc3_path.stem}_metadata.json")
    metadata_path = metadata_path.expanduser().resolve()
    if not metadata_path.exists():
        raise RuntimeError(f"Metadata file not found: {metadata_path}")

    data = np.load(device_fc3_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return data, metadata, metadata_path


def build_local_atom_tables(repeats, primitive_atoms: int):
    import itertools

    local_atom_data: list[tuple[tuple[int, int, int], int]] = []
    for offset in itertools.product(*(range(int(rep)) for rep in repeats)):
        for primitive_index in range(primitive_atoms):
            local_atom_data.append(((int(offset[0]), int(offset[1]), int(offset[2])), primitive_index))

    return local_atom_data


def atom_descriptor(global_atom: int, atoms_per_cell: int, local_atom_data):
    device_cell = int(global_atom // atoms_per_cell)
    local_atom = int(global_atom % atoms_per_cell)
    offset, primitive_index = local_atom_data[local_atom]
    return device_cell, offset, primitive_index


def build_basis_map(local_atom_data, axis_index: int):
    basis_map: dict[tuple[int, int], int] = {}
    for _, (offset, primitive_index) in enumerate(local_atom_data):
        key = (int(offset[axis_index]), int(primitive_index))
        if key not in basis_map:
            basis_map[key] = len(basis_map)
    return basis_map


def parse_q_pairs(args):
    qj = [(float(value[0]), float(value[1])) for value in (args.qj_point or [])]
    qk = [(float(value[0]), float(value[1])) for value in (args.qk_point or [])]

    if args.qj_grid is not None:
        nqj1, nqj2 = int(args.qj_grid[0]), int(args.qj_grid[1])
        if nqj1 <= 0 or nqj2 <= 0:
            raise RuntimeError("--qj-grid values must be positive integers.")
        for i in range(nqj1):
            for j in range(nqj2):
                if args.gamma_centered:
                    q1 = (i + 0.5) / nqj1 - 0.5
                    q2 = (j + 0.5) / nqj2 - 0.5
                else:
                    q1 = i / nqj1
                    q2 = j / nqj2
                qj.append((float(q1), float(q2)))

    if args.qk_grid is not None:
        nqk1, nqk2 = int(args.qk_grid[0]), int(args.qk_grid[1])
        if nqk1 <= 0 or nqk2 <= 0:
            raise RuntimeError("--qk-grid values must be positive integers.")
        for i in range(nqk1):
            for j in range(nqk2):
                if args.gamma_centered:
                    q1 = (i + 0.5) / nqk1 - 0.5
                    q2 = (j + 0.5) / nqk2 - 0.5
                else:
                    q1 = i / nqk1
                    q2 = j / nqk2
                qk.append((float(q1), float(q2)))

    def dedupe(points):
        seen = set()
        unique = []
        for p1, p2 in points:
            key = (round(p1, 12), round(p2, 12))
            if key in seen:
                continue
            seen.add(key)
            unique.append((p1, p2))
        return unique

    qj = dedupe(qj)
    qk = dedupe(qk)

    if not qj and not qk:
        return []
    if qj and not qk:
        return [((q1, q2), (q1, q2)) for q1, q2 in qj]
    if qk and not qj:
        raise RuntimeError("Use --qj-point or --qj-grid when --qk-point/--qk-grid is provided.")

    if args.qj_grid is not None and args.qk_grid is not None:
        return [(qj_point, qk_point) for qj_point in qj for qk_point in qk]

    if len(qj) != len(qk):
        raise RuntimeError(
            "When both qj and qk lists are explicit, they must have the same length. "
            "Use both --qj-grid and --qk-grid to generate a Cartesian product."
        )
    return list(zip(qj, qk))


def label_q(q):
    def one(value: float) -> str:
        return f"{value:+.6f}".replace("+", "p").replace("-", "m").replace(".", "d")

    return f"{one(q[0])}_{one(q[1])}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract reference-cell FC3 transverse vertices and optionally compute q-resolved "
            "weighted sparse FC3 entries."
        )
    )
    parser.add_argument(
        "--device-fc3",
        type=Path,
        required=True,
        help="Path to device FC3 sparse file (.npz) created by construct_device_fc3.py.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Metadata JSON path. Defaults to <device-fc3-stem>_metadata.json.",
    )
    parser.add_argument(
        "--reference-cell",
        type=int,
        default=None,
        help="0-based reference transport cell. Defaults to middle cell if possible.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix. Defaults to <device-fc3-dir>/fc3_vertices_<device-fc3-stem>",
    )
    parser.add_argument(
        "--qj-point",
        type=float,
        nargs=2,
        action="append",
        metavar=("QJ1", "QJ2"),
        help="Transverse q_j point in reduced coordinates. Repeatable.",
    )
    parser.add_argument(
        "--qk-point",
        type=float,
        nargs=2,
        action="append",
        metavar=("QK1", "QK2"),
        help="Transverse q_k point in reduced coordinates. Repeatable (paired with --qj-point).",
    )
    parser.add_argument(
        "--qj-grid",
        type=int,
        nargs=2,
        metavar=("NQJ1", "NQJ2"),
        default=None,
        help=(
            "Uniform grid for q_j points in reduced coordinates. "
            "Points are (i/NQJ1, j/NQJ2), or Gamma-centered with --gamma-centered."
        ),
    )
    parser.add_argument(
        "--qk-grid",
        type=int,
        nargs=2,
        metavar=("NQK1", "NQK2"),
        default=None,
        help=(
            "Uniform grid for q_k points. If omitted while q_j grid/points are provided, "
            "q_k defaults to q_j."
        ),
    )
    parser.add_argument(
        "--gamma-centered",
        action="store_true",
        help="Use Gamma-centered coordinates ((i+0.5)/N - 0.5) for q-grid generation.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is not installed in the current Python environment.") from exc

    device_fc3_path = args.device_fc3.expanduser().resolve()
    metadata_path = args.metadata
    if metadata_path is not None:
        metadata_path = metadata_path.expanduser().resolve()

    data, metadata, metadata_path = load_inputs(device_fc3_path=device_fc3_path, metadata_path=metadata_path)

    i = np.asarray(data["i"], dtype=int)
    j = np.asarray(data["j"], dtype=int)
    k = np.asarray(data["k"], dtype=int)
    tensors = np.asarray(data["tensors"], dtype=float)

    if i.shape != j.shape or i.shape != k.shape or i.shape[0] != tensors.shape[0]:
        raise RuntimeError("Inconsistent FC3 sparse arrays in device file.")

    repeats = [int(value) for value in metadata["supercell_repeats"]]
    axis_index = int(metadata["transport_axis_index"])
    primitive_atoms = int(metadata["primitive_atoms"])
    atoms_per_cell = int(metadata["transport_atoms_per_cell"])
    num_cells = int(metadata["num_device_cells"])

    if num_cells < 1:
        raise RuntimeError("Invalid num_device_cells in metadata.")

    reference_cell = args.reference_cell
    if reference_cell is None:
        reference_cell = num_cells // 2 if num_cells >= 3 else 0
    if reference_cell < 0 or reference_cell >= num_cells:
        raise RuntimeError(f"--reference-cell out of range: {reference_cell} (valid: 0..{num_cells - 1})")

    transverse_dirs = [direction for direction in (0, 1, 2) if direction != axis_index]
    transverse_periods = [repeats[direction] for direction in transverse_dirs]

    local_atom_data = build_local_atom_tables(repeats=repeats, primitive_atoms=primitive_atoms)
    if len(local_atom_data) != atoms_per_cell:
        raise RuntimeError("Metadata atoms-per-cell is inconsistent with repeats and primitive_atoms.")

    basis_map = build_basis_map(local_atom_data=local_atom_data, axis_index=axis_index)

    reference_transverse = (0, 0)

    out_i_basis: list[int] = []
    out_j_basis: list[int] = []
    out_k_basis: list[int] = []
    out_dc_j: list[int] = []
    out_dc_k: list[int] = []
    out_dt_j: list[tuple[int, int]] = []
    out_dt_k: list[tuple[int, int]] = []
    out_tensors: list[np.ndarray] = []

    for idx in range(i.shape[0]):
        cell_i, offset_i, prim_i = atom_descriptor(int(i[idx]), atoms_per_cell, local_atom_data)
        if cell_i != reference_cell:
            continue

        trans_i = (int(offset_i[transverse_dirs[0]]), int(offset_i[transverse_dirs[1]]))
        if trans_i != reference_transverse:
            continue

        cell_j, offset_j, prim_j = atom_descriptor(int(j[idx]), atoms_per_cell, local_atom_data)
        cell_k, offset_k, prim_k = atom_descriptor(int(k[idx]), atoms_per_cell, local_atom_data)

        basis_i = basis_map[(int(offset_i[axis_index]), int(prim_i))]
        basis_j = basis_map[(int(offset_j[axis_index]), int(prim_j))]
        basis_k = basis_map[(int(offset_k[axis_index]), int(prim_k))]

        trans_j = (int(offset_j[transverse_dirs[0]]), int(offset_j[transverse_dirs[1]]))
        trans_k = (int(offset_k[transverse_dirs[0]]), int(offset_k[transverse_dirs[1]]))

        dt_j = (
            center_shift(trans_j[0] - reference_transverse[0], transverse_periods[0]),
            center_shift(trans_j[1] - reference_transverse[1], transverse_periods[1]),
        )
        dt_k = (
            center_shift(trans_k[0] - reference_transverse[0], transverse_periods[0]),
            center_shift(trans_k[1] - reference_transverse[1], transverse_periods[1]),
        )

        out_i_basis.append(int(basis_i))
        out_j_basis.append(int(basis_j))
        out_k_basis.append(int(basis_k))
        out_dc_j.append(int(cell_j - reference_cell))
        out_dc_k.append(int(cell_k - reference_cell))
        out_dt_j.append((int(dt_j[0]), int(dt_j[1])))
        out_dt_k.append((int(dt_k[0]), int(dt_k[1])))
        out_tensors.append(np.asarray(tensors[idx], dtype=float))

    if not out_tensors:
        raise RuntimeError("No FC3 entries selected for reference cell/transverse sector.")

    arr_i_basis = np.asarray(out_i_basis, dtype=int)
    arr_j_basis = np.asarray(out_j_basis, dtype=int)
    arr_k_basis = np.asarray(out_k_basis, dtype=int)
    arr_dc_j = np.asarray(out_dc_j, dtype=int)
    arr_dc_k = np.asarray(out_dc_k, dtype=int)
    arr_dt_j = np.asarray(out_dt_j, dtype=int)
    arr_dt_k = np.asarray(out_dt_k, dtype=int)
    arr_tensors = np.asarray(out_tensors, dtype=float)

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = device_fc3_path.parent / f"fc3_vertices_{device_fc3_path.stem}"
    output_prefix = output_prefix.expanduser().resolve()

    base_npz = output_prefix.with_name(f"{output_prefix.name}_refcell{reference_cell}.npz")
    np.savez_compressed(
        base_npz,
        i_basis=arr_i_basis,
        j_basis=arr_j_basis,
        k_basis=arr_k_basis,
        delta_cell_j=arr_dc_j,
        delta_cell_k=arr_dc_k,
        dt_j=arr_dt_j,
        dt_k=arr_dt_k,
        tensors=arr_tensors,
    )

    summary = {
        "device_fc3": str(device_fc3_path),
        "metadata": str(metadata_path),
        "reference_cell": int(reference_cell),
        "transport_axis_index": int(axis_index),
        "transverse_dirs": [int(transverse_dirs[0]), int(transverse_dirs[1])],
        "transverse_periods": [int(transverse_periods[0]), int(transverse_periods[1])],
        "basis_size": int(len(basis_map)),
        "selected_entries": int(arr_tensors.shape[0]),
        "real_space_vertices": str(base_npz),
    }

    q_pairs = parse_q_pairs(args)
    q_outputs = []
    for (qj1, qj2), (qk1, qk2) in q_pairs:
        phase = np.exp(
            2j
            * np.pi
            * (
                qj1 * arr_dt_j[:, 0]
                + qj2 * arr_dt_j[:, 1]
                + qk1 * arr_dt_k[:, 0]
                + qk2 * arr_dt_k[:, 1]
            )
        )
        tensors_q = phase[:, None, None, None] * arr_tensors

        q_label = f"qj_{label_q((qj1, qj2))}_qk_{label_q((qk1, qk2))}"
        q_npz = output_prefix.with_name(f"{output_prefix.name}_refcell{reference_cell}_{q_label}.npz")
        np.savez_compressed(
            q_npz,
            i_basis=arr_i_basis,
            j_basis=arr_j_basis,
            k_basis=arr_k_basis,
            delta_cell_j=arr_dc_j,
            delta_cell_k=arr_dc_k,
            tensors=tensors_q,
        )
        q_outputs.append(
            {
                "qj": [float(qj1), float(qj2)],
                "qk": [float(qk1), float(qk2)],
                "output": str(q_npz),
            }
        )

    if q_outputs:
        summary["q_resolved_vertices"] = q_outputs
        if args.qj_grid is not None:
            summary["qj_grid"] = [int(args.qj_grid[0]), int(args.qj_grid[1])]
        if args.qk_grid is not None:
            summary["qk_grid"] = [int(args.qk_grid[0]), int(args.qk_grid[1])]
        if args.qj_grid is not None or args.qk_grid is not None:
            summary["gamma_centered"] = bool(args.gamma_centered)

    summary_path = output_prefix.with_name(f"{output_prefix.name}_refcell{reference_cell}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Device FC3: {device_fc3_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Reference cell: {reference_cell}")
    print(f"Selected entries: {arr_tensors.shape[0]}")
    print(f"Saved real-space vertices: {base_npz}")
    if q_outputs:
        for item in q_outputs:
            print(
                f"Saved q-resolved vertices at qj={item['qj']} qk={item['qk']}: {item['output']}"
            )
    print(f"Saved summary: {summary_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
