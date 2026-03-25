#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class FC3Block:
    block_id: int
    rj: tuple[float, float, float]
    rk: tuple[float, float, float]
    atoms: tuple[int, int, int]
    values: list[float]

    @property
    def fro_norm(self) -> float:
        return math.sqrt(sum(value * value for value in self.values))

    @property
    def max_abs(self) -> float:
        return max(abs(value) for value in self.values)

    @property
    def near_zero_count(self) -> int:
        return sum(1 for value in self.values if abs(value) < 1e-12)


def parse_vec3(text: str) -> tuple[float, float, float]:
    fields = text.split()
    if len(fields) != 3:
        raise ValueError(f"Expected 3-vector, got: {text!r}")
    return float(fields[0]), float(fields[1]), float(fields[2])


def parse_atoms(text: str) -> tuple[int, int, int]:
    fields = text.split()
    if len(fields) != 3:
        raise ValueError(f"Expected 3 atom indices, got: {text!r}")
    return int(fields[0]), int(fields[1]), int(fields[2])


def parse_fc3(path: Path) -> tuple[int, list[FC3Block]]:
    raw_lines = path.read_text().splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]
    if not lines:
        raise ValueError("Empty FC3 file")

    declared_blocks = int(lines[0])
    cursor = 1
    blocks: list[FC3Block] = []

    while cursor < len(lines):
        block_id = int(lines[cursor])
        cursor += 1

        rj = parse_vec3(lines[cursor])
        cursor += 1

        rk = parse_vec3(lines[cursor])
        cursor += 1

        atoms = parse_atoms(lines[cursor])
        cursor += 1

        values: list[float] = []
        for _ in range(27):
            parts = lines[cursor].split()
            if len(parts) != 4:
                raise ValueError(f"Malformed tensor line: {lines[cursor]!r}")
            values.append(float(parts[3]))
            cursor += 1

        blocks.append(FC3Block(block_id=block_id, rj=rj, rk=rk, atoms=atoms, values=values))

    return declared_blocks, blocks


def vec_norm(vec: tuple[float, float, float]) -> float:
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


def vec_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def min_max_avg(values: Iterable[float]) -> tuple[float, float, float]:
    data = list(values)
    if not data:
        return math.nan, math.nan, math.nan
    return min(data), max(data), sum(data) / len(data)


def format_stats(name: str, values: list[float]) -> str:
    vmin, vmax, vavg = min_max_avg(values)
    return f"{name:12s} min={vmin:10.5f}  max={vmax:10.5f}  avg={vavg:10.5f}"


def top_components(block: FC3Block, top_n: int) -> list[tuple[int, int, int, float]]:
    entries: list[tuple[int, int, int, float]] = []
    idx = 0
    for alpha in range(1, 4):
        for beta in range(1, 4):
            for gamma in range(1, 4):
                entries.append((alpha, beta, gamma, block.values[idx]))
                idx += 1
    entries.sort(key=lambda item: abs(item[3]), reverse=True)
    return entries[:top_n]


def print_summary(path: Path, declared_blocks: int, blocks: list[FC3Block], top_n: int, zero_tol: float) -> None:
    print(f"File: {path}")
    print(f"Declared blocks: {declared_blocks}")
    print(f"Parsed blocks:   {len(blocks)}")
    if declared_blocks != len(blocks):
        print("WARNING: Declared and parsed block counts do not match")

    if not blocks:
        return

    rj_norms = [vec_norm(block.rj) for block in blocks]
    rk_norms = [vec_norm(block.rk) for block in blocks]
    jk_norms = [vec_norm(vec_sub(block.rj, block.rk)) for block in blocks]

    all_values = [value for block in blocks for value in block.values]
    abs_values = [abs(value) for value in all_values]
    zero_count = sum(1 for value in all_values if abs(value) <= zero_tol)

    print()
    print("Distance stats (Angstrom):")
    print(format_stats("|Rj|", rj_norms))
    print(format_stats("|Rk|", rk_norms))
    print(format_stats("|Rj-Rk|", jk_norms))

    print()
    print("Tensor value stats (eV/Angstrom^3):")
    print(format_stats("value", all_values))
    print(format_stats("|value|", abs_values))
    print(f"near-zero (|x| <= {zero_tol:g}): {zero_count}/{len(all_values)}")

    strongest_block = max(blocks, key=lambda block: block.max_abs)
    strongest_value = max(strongest_block.values, key=lambda value: abs(value))

    print()
    print("Global strongest component:")
    print(
        f"block={strongest_block.block_id}, atoms={strongest_block.atoms}, "
        f"max|Phi|={abs(strongest_value):.6f}, value={strongest_value:.6f}"
    )

    ranked = sorted(blocks, key=lambda block: block.fro_norm, reverse=True)
    print()
    print(f"Top {min(top_n, len(ranked))} blocks by Frobenius norm:")
    for block in ranked[:top_n]:
        print(
            f"block={block.block_id:4d} atoms={block.atoms} "
            f"|Rj|={vec_norm(block.rj):7.3f} |Rk|={vec_norm(block.rk):7.3f} "
            f"||Phi||_F={block.fro_norm:10.4f} max|Phi|={block.max_abs:10.4f}"
        )

    print()
    print("Largest components in strongest block:")
    for alpha, beta, gamma, value in top_components(strongest_block, top_n=min(6, top_n)):
        print(f"({alpha},{beta},{gamma}) {value: .6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze thirdorder FORCE_CONSTANTS_3RD file.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("phonon-data/si/anharmonic/FORCE_CONSTANTS_3RD"),
        help="Path to FORCE_CONSTANTS_3RD file.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top blocks to print.",
    )
    parser.add_argument(
        "--zero-tol",
        type=float,
        default=1e-12,
        help="Absolute threshold used for near-zero counting.",
    )
    args = parser.parse_args()

    path = args.file.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    declared_blocks, blocks = parse_fc3(path)
    print_summary(path=path, declared_blocks=declared_blocks, blocks=blocks, top_n=max(1, args.top), zero_tol=args.zero_tol)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
