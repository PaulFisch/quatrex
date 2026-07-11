"""Tile an L-cell device input set from a stored per-length device dir.

The cnt33_L{2,3,4} device inputs are exact translates of a bulk offset set
with pure edge truncation (verified bit-for-bit across the stored lengths:
interior blocks of L4 are translates of each other, edges and dynamical
matrices bit-equal across L). This script materializes that structure for
ANY device length:

  * fc3_blocks.hdf5 : bulk offset blocks Phi(d1, d2) extracted from one
    interior cell of the source, emitted as (I, I+d1, I+d2) for every cell
    with in-range indices (edge truncation = index validity, exactly the
    generator's behaviour);
  * dynamical_matrix.mat, phonon_energies.npy, structure.xyz : copied
    unchanged (unit-cell quantities, identical across lengths).

Usage:
    python phonon/studies/_tile_device_inputs.py \
        --src <geom/cnt33_L4> --cells 10 --out <geom-like dir>
    python phonon/studies/_tile_device_inputs.py --src <...L4> \
        --selfcheck <...L2> <...L3> <...L4>     # bit-equality gate
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


def load_blocks(path: Path) -> dict[tuple[int, int, int], np.ndarray]:
    with h5py.File(path / "fc3_blocks.hdf5") as f:
        return {tuple(int(x) for x in k.split("_")): f["fc3_blocks"][k][:]
                for k in f["fc3_blocks"]}


def bulk_offsets(blocks: dict) -> dict[tuple[int, int], np.ndarray]:
    """Offset set Phi(d1, d2) from an interior cell of the source device."""
    n_src = max(k[0] for k in blocks) + 1
    interior = [i for i in range(n_src)
                if 1 <= i <= n_src - 2]
    if not interior:
        sys.exit("source device has no interior cell (need L >= 3, or L == 2 "
                 "plus the offsets it can supply)")
    i0 = interior[0]
    offs = {(k1 - i0, k2 - i0): v for (i, k1, k2), v in blocks.items()
            if i == i0}
    # Bit-exactness gate on the source itself: every stored block must be
    # the translate of an offset block (edges = pure truncation).
    for (i, k1, k2), v in blocks.items():
        ref = offs.get((k1 - i, k2 - i))
        if ref is None or not np.array_equal(v, ref):
            sys.exit(f"source violates translation invariance at block "
                     f"({i},{k1},{k2}) -- tiling would not be exact.")
    return offs


def tile(offs: dict, n_cells: int) -> dict[tuple[int, int, int], np.ndarray]:
    out = {}
    for i in range(n_cells):
        for (d1, d2), v in offs.items():
            k1, k2 = i + d1, i + d2
            if 0 <= k1 < n_cells and 0 <= k2 < n_cells:
                out[(i, k1, k2)] = v
    return out


def write_device(src: Path, out: Path, n_cells: int, offs: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    blocks = tile(offs, n_cells)
    with h5py.File(src / "fc3_blocks.hdf5") as fs:
        units = dict(fs["meta"].attrs).get("units", "THz^2")
        b = int(np.asarray(fs["meta"]["block_sizes"])[0])
    with h5py.File(out / "fc3_blocks.hdf5", "w") as f:
        g = f.create_group("fc3_blocks")
        for (i, k1, k2), v in sorted(blocks.items()):
            g.create_dataset(f"{i}_{k1}_{k2}", data=v)
        m = f.create_group("meta")
        m.attrs["units"] = units
        m.create_dataset("block_sizes", data=np.full(n_cells, b, dtype=np.int64))
        m.create_dataset("keys", data=np.array(sorted(blocks), dtype=np.int64))
    for name in ("dynamical_matrix.mat", "phonon_energies.npy",
                 "structure.xyz"):
        shutil.copy(src / name, out / name)
    print(f"wrote {out}: {len(blocks)} fc3 blocks for {n_cells} cells")


def selfcheck(src: Path, refs: list[Path]) -> int:
    offs = bulk_offsets(load_blocks(src))
    ok = True
    for ref in refs:
        stored = load_blocks(ref)
        n = max(k[0] for k in stored) + 1
        made = tile(offs, n)
        same = (set(made) == set(stored)
                and all(np.array_equal(made[k], stored[k]) for k in stored))
        print(f"{ref.name}: L={n} blocks={len(stored)} "
              f"{'BIT-EQUAL' if same else 'MISMATCH'}")
        ok &= same
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path,
                    help="source device dir (needs an interior cell)")
    ap.add_argument("--cells", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selfcheck", nargs="+", type=Path,
                    help="stored device dirs to regenerate + compare")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck(args.src, args.selfcheck)
    if not (args.cells and args.out):
        ap.error("--cells and --out required (unless --selfcheck)")
    offs = bulk_offsets(load_blocks(args.src))
    write_device(args.src, args.out, args.cells, offs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
