#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_vec3_float(text: str) -> tuple[float, float, float]:
    fields = text.split()
    if len(fields) != 3:
        raise ValueError(f"Expected 3 floats, got: {text!r}")
    return float(fields[0]), float(fields[1]), float(fields[2])


def parse_vec3_int(text: str) -> tuple[int, int, int]:
    fields = text.split()
    if len(fields) != 3:
        raise ValueError(f"Expected 3 integers, got: {text!r}")
    return int(fields[0]), int(fields[1]), int(fields[2])


def parse_fc3_file(path: Path):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to extract FC3 tensors.") from exc

    raw_lines = path.read_text().splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]
    if not lines:
        raise RuntimeError(f"Empty FC3 file: {path}")

    declared_blocks = int(lines[0])
    cursor = 1

    block_ids: list[int] = []
    atoms: list[tuple[int, int, int]] = []
    rj: list[tuple[float, float, float]] = []
    rk: list[tuple[float, float, float]] = []
    tensor_blocks: list["np.ndarray"] = []

    while cursor < len(lines):
        block_id = int(lines[cursor])
        cursor += 1

        block_rj = parse_vec3_float(lines[cursor])
        cursor += 1

        block_rk = parse_vec3_float(lines[cursor])
        cursor += 1

        block_atoms = parse_vec3_int(lines[cursor])
        cursor += 1

        tensor = np.zeros((3, 3, 3), dtype=float)
        for alpha in range(3):
            for beta in range(3):
                for gamma in range(3):
                    parts = lines[cursor].split()
                    if len(parts) != 4:
                        raise RuntimeError(f"Malformed tensor line at {cursor + 1}: {lines[cursor]!r}")
                    tensor[alpha, beta, gamma] = float(parts[3])
                    cursor += 1

        block_ids.append(block_id)
        atoms.append(block_atoms)
        rj.append(block_rj)
        rk.append(block_rk)
        tensor_blocks.append(tensor)

    tensors = np.stack(tensor_blocks, axis=0) if tensor_blocks else np.zeros((0, 3, 3, 3), dtype=float)

    return {
        "declared_blocks": declared_blocks,
        "parsed_blocks": len(block_ids),
        "block_ids": np.array(block_ids, dtype=int),
        "atoms": np.array(atoms, dtype=int),
        "rj": np.array(rj, dtype=float),
        "rk": np.array(rk, dtype=float),
        "tensors": tensors,
    }


def save_npz(data: dict, output_path: Path) -> None:
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        declared_blocks=data["declared_blocks"],
        parsed_blocks=data["parsed_blocks"],
        block_ids=data["block_ids"],
        atoms=data["atoms"],
        rj=data["rj"],
        rk=data["rk"],
        tensors=data["tensors"],
    )


def save_txt(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        out.write(f"# declared_blocks: {data['declared_blocks']}\n")
        out.write(f"# parsed_blocks:   {data['parsed_blocks']}\n")
        out.write("# columns: block_id i j k rj_x rj_y rj_z rk_x rk_y rk_z then 27 tensor values (alpha,beta,gamma fastest in gamma)\n")

        for idx in range(data["parsed_blocks"]):
            block_id = int(data["block_ids"][idx])
            atom_i, atom_j, atom_k = data["atoms"][idx]
            rj_x, rj_y, rj_z = data["rj"][idx]
            rk_x, rk_y, rk_z = data["rk"][idx]
            values = data["tensors"][idx].reshape(-1)

            header = (
                f"{block_id:d} {atom_i:d} {atom_j:d} {atom_k:d} "
                f"{rj_x:.10f} {rj_y:.10f} {rj_z:.10f} "
                f"{rk_x:.10f} {rk_y:.10f} {rk_z:.10f}"
            )
            tensor_text = " ".join(f"{value:.16e}" for value in values)
            out.write(header + " " + tensor_text + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract 3rd-order force-constant tensors from a FORCE_CONSTANTS_3RD file. "
            "Output stores one 3x3x3 tensor per block plus metadata (atoms and lattice vectors)."
        )
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("phonon-data/si/anharmonic/FORCE_CONSTANTS_3RD"),
        help="Input FORCE_CONSTANTS_3RD file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. Defaults to <input_dir>/fc3_tensor_data.npz",
    )
    parser.add_argument(
        "--format",
        choices=("npz", "txt"),
        default="npz",
        help="Output format.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = args.file.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"FC3 file not found: {input_path}")

    output_path = args.output
    if output_path is None:
        suffix = ".npz" if args.format == "npz" else ".txt"
        output_path = input_path.parent / f"fc3_tensor_data{suffix}"
    output_path = output_path.expanduser().resolve()

    data = parse_fc3_file(input_path)

    if args.format == "npz":
        save_npz(data, output_path)
    else:
        save_txt(data, output_path)

    print(f"Input file:      {input_path}")
    print(f"Declared blocks: {data['declared_blocks']}")
    print(f"Parsed blocks:   {data['parsed_blocks']}")
    print(f"Tensor shape:    {data['tensors'].shape}  # (n_blocks, 3, 3, 3)")
    print(f"Saved output:    {output_path}")

    if data["declared_blocks"] != data["parsed_blocks"]:
        print("Warning: declared block count does not match parsed block count.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
