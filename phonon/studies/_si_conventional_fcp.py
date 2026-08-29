#!/usr/bin/env python3
"""Materialise a conventional cubic Si force-constant model from hiPhive.

The fitted hiPhive force-constant potential is tied to the infinite crystal,
not to one particular choice of unit cell.  This utility changes the basis
from the two-atom fcc primitive cell to the eight-atom conventional cubic
cell and evaluates the same potential on a sufficiently large supercell.
It therefore isolates the [100] film geometry from DFT and fit differences.

The full production input is obtained with, for example,

    python phonon/studies/_si_conventional_fcp.py \
        --fcp reaps/si_big_hiphive/fcp.fcp \
        --output reaps/si_big_hiphive_conventional \
        --repeat 3 --orders 2,3 --validate-folding

For the published 7/5 Angstrom FC2/FC3 cutoffs, a 3 by 3 by 3 conventional
supercell has an 8.2 Angstrom half-width and is large enough for both orders.
Use ``--orders 2`` for a cheap harmonic-only materialisation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


CONVENTIONAL_TRANSFORM = np.array(
    [[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=int
)


def silicon_primitive_cell(lattice_constant: float):
    """Return the two-atom fcc primitive cell used by the Si film inputs."""
    from ase import Atoms

    half = 0.5 * lattice_constant
    return Atoms(
        symbols=["Si", "Si"],
        cell=[[0, half, half], [half, 0, half], [half, half, 0]],
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        pbc=True,
    )


def conventional_cell(primitive):
    """Return the eight-atom cubic cell equivalent to an fcc primitive."""
    from ase.build import make_supercell

    cell = make_supercell(primitive, CONVENTIONAL_TRANSFORM, wrap=True)
    lengths = np.asarray(cell.cell.lengths())
    angles = np.asarray(cell.cell.angles())
    if len(primitive) != 2 or len(cell) != 8:
        raise ValueError(
            "the conventional Si transform requires a two-atom primitive "
            f"and must produce eight atoms, got {len(primitive)} and {len(cell)}"
        )
    if not np.allclose(lengths, lengths[0], rtol=0.0, atol=1.0e-8):
        raise ValueError(f"transformed cell is not cubic: lengths {lengths}")
    if not np.allclose(angles, 90.0, rtol=0.0, atol=1.0e-8):
        raise ValueError(f"transformed cell is not orthogonal: angles {angles}")
    return cell


def _phonopy_supercell(unit, repeat: int):
    """Build an ASE supercell in exactly the atom order used by phonopy."""
    from ase import Atoms
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    if repeat < 1:
        raise ValueError("repeat must be positive")
    ph_unit = PhonopyAtoms(
        symbols=unit.get_chemical_symbols(),
        cell=unit.cell.array,
        scaled_positions=unit.get_scaled_positions(wrap=True),
    )
    phonon = Phonopy(
        ph_unit,
        supercell_matrix=np.diag([repeat] * 3),
        primitive_matrix=np.eye(3),
    )
    ph_supercell = phonon.supercell
    supercell = Atoms(
        symbols=ph_supercell.symbols,
        cell=ph_supercell.cell,
        scaled_positions=ph_supercell.scaled_positions,
        pbc=True,
    )
    return phonon, supercell


def folded_primitive_q_points(primitive, conventional) -> np.ndarray:
    """Primitive reciprocal points folded onto conventional-cell Gamma."""
    primitive_reciprocal = primitive.cell.reciprocal().array
    conventional_reciprocal = conventional.cell.reciprocal().array
    expected = abs(round(np.linalg.det(CONVENTIONAL_TRANSFORM)))
    representatives: list[np.ndarray] = []
    for index in np.ndindex(*(expected,) * 3):
        frac = (
            np.asarray(index)
            @ conventional_reciprocal
            @ np.linalg.inv(primitive_reciprocal)
        )
        frac -= np.floor(frac + 1.0e-10)
        if not any(np.allclose(frac, old, atol=1.0e-9) for old in representatives):
            representatives.append(frac)
        if len(representatives) == expected:
            break
    if len(representatives) != expected:
        raise RuntimeError(
            f"found {len(representatives)} reciprocal cosets, expected {expected}"
        )
    return np.asarray(representatives)


def validate_gamma_folding(
    fcp,
    conventional,
    primitive_repeat: int = 3,
    conventional_repeat: int = 2,
) -> dict[str, object]:
    """Compare conventional Gamma with its four folded primitive spectra."""
    primitive = fcp.primitive_structure
    primitive_phonon, primitive_supercell = _phonopy_supercell(
        primitive, primitive_repeat
    )
    conventional_phonon, conventional_supercell = _phonopy_supercell(
        conventional, conventional_repeat
    )
    primitive_phonon.force_constants = fcp.get_force_constants(
        primitive_supercell
    ).get_fc_array(order=2)
    conventional_phonon.force_constants = fcp.get_force_constants(
        conventional_supercell
    ).get_fc_array(order=2)
    q_points = folded_primitive_q_points(primitive, conventional)
    primitive_frequencies = np.concatenate(
        [primitive_phonon.get_frequencies(q) for q in q_points]
    )
    conventional_frequencies = conventional_phonon.get_frequencies([0, 0, 0])
    difference = np.sort(primitive_frequencies) - np.sort(conventional_frequencies)
    return {
        "primitive_repeat": primitive_repeat,
        "conventional_repeat": conventional_repeat,
        "folded_primitive_q_points": q_points.tolist(),
        "maximum_frequency_difference_thz": float(np.max(np.abs(difference))),
        "rms_frequency_difference_thz": float(np.sqrt(np.mean(difference**2))),
    }


def _atoms_record(atoms) -> dict[str, object]:
    return {
        "symbols": atoms.get_chemical_symbols(),
        "cell": atoms.cell.array.tolist(),
        "scaled_positions": atoms.get_scaled_positions(wrap=True).tolist(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def primitive_atom_labels(atoms, primitive) -> tuple[np.ndarray, np.ndarray]:
    """Map atoms onto integer primitive translations and basis indices."""
    coordinates = atoms.positions @ np.linalg.inv(primitive.cell.array)
    basis_positions = primitive.get_scaled_positions(wrap=True)
    cells = np.empty((len(atoms), 3), dtype=int)
    basis = np.empty(len(atoms), dtype=int)
    for atom, coordinate in enumerate(coordinates):
        matches: list[tuple[int, np.ndarray]] = []
        for basis_index, basis_position in enumerate(basis_positions):
            translation = np.rint(coordinate - basis_position).astype(int)
            residual = coordinate - basis_position - translation
            if np.max(np.abs(residual)) < 1.0e-7:
                matches.append((basis_index, translation))
        if len(matches) != 1:
            raise ValueError(
                f"atom {atom} has {len(matches)} primitive labels at {coordinate}"
            )
        basis[atom], cells[atom] = matches[0]
    return cells, basis


def primitive_supercell_lookup(
    supercell, primitive, repeat: int
) -> tuple[dict[tuple[int, int, int, int], int], np.ndarray]:
    """Return periodic primitive-cell labels and home indices."""
    cells, basis = primitive_atom_labels(supercell, primitive)
    lookup: dict[tuple[int, int, int, int], int] = {}
    for atom, (cell, basis_index) in enumerate(zip(cells, basis, strict=True)):
        periodic = np.mod(cell, repeat)
        key = (*map(int, periodic), int(basis_index))
        if key in lookup:
            raise ValueError(f"duplicate primitive label {key}")
        lookup[key] = atom
    expected = repeat**3 * len(primitive)
    if len(lookup) != expected:
        raise ValueError(f"found {len(lookup)} labels, expected {expected}")
    home = np.array([lookup[(0, 0, 0, index)] for index in range(len(primitive))])
    return lookup, home


def dense_remap_indices(
    primitive, source_repeat: int, target_repeat: int, target_supercell
) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
    """Source-index groups for periodic folding into the target cell."""
    _, source_supercell = _phonopy_supercell(primitive, source_repeat)
    _, home = primitive_supercell_lookup(
        source_supercell, primitive, source_repeat
    )
    source_cells, source_basis = primitive_atom_labels(source_supercell, primitive)
    target_cells, target_basis = primitive_atom_labels(target_supercell, primitive)
    source_matrix = source_repeat * np.eye(3, dtype=int)
    target_matrix = target_repeat * CONVENTIONAL_TRANSFORM
    determinant = abs(round(np.linalg.det(target_matrix)))
    adjugate = np.rint(
        round(np.linalg.det(target_matrix)) * np.linalg.inv(target_matrix)
    ).astype(int)

    def quotient_key(vector: np.ndarray) -> tuple[int, int, int]:
        return tuple(map(int, np.mod(vector @ adjugate, determinant)))

    basis_positions = primitive.get_scaled_positions(wrap=True)
    shifts = np.asarray(list(np.ndindex(3, 3, 3)), dtype=int) - 1
    grouped: list[dict[tuple[int, int, int, int], list[int]]] = [
        {} for _ in range(len(primitive))
    ]
    for first_basis in range(len(primitive)):
        for atom, (cell, second_basis) in enumerate(
            zip(source_cells, source_basis, strict=True)
        ):
            candidates = cell + shifts @ source_matrix
            cartesian = (
                candidates
                + basis_positions[int(second_basis)]
                - basis_positions[first_basis]
            ) @ primitive.cell.array
            canonical = candidates[np.argmin(np.sum(cartesian**2, axis=1))]
            key = (*quotient_key(canonical), int(second_basis))
            grouped[first_basis].setdefault(key, []).append(atom)

    relative_groups: list[list[np.ndarray]] = []
    for first in range(len(target_supercell)):
        row: list[np.ndarray] = []
        for second in range(len(target_supercell)):
            cell = target_cells[second] - target_cells[first]
            key = (*quotient_key(cell), int(target_basis[second]))
            indices = grouped[int(target_basis[first])].get(key, [])
            row.append(np.asarray(indices, dtype=int))
        relative_groups.append(row)
    return home, target_basis, relative_groups


def _hdf5_compression_kwargs(compression: str | None) -> dict[str, object]:
    return {} if compression is None else {"compression": compression, "shuffle": True}


def materialise_dense(
    source_path: Path,
    output: Path,
    source_repeat: int,
    target_repeat: int,
    lattice_constant: float,
    orders: Iterable[int],
    compression: str | None = "gzip",
    validate_folding: bool = False,
    force: bool = False,
) -> dict[str, object]:
    """Relabel full primitive FC arrays into a conventional supercell.

    The source FC3 is read only for the two primitive home atoms. Target
    reference-atom slices are assembled and written one at a time, keeping
    memory independent of the dense target tensor size.
    """
    import h5py

    requested_orders = tuple(sorted(set(int(order) for order in orders)))
    if not requested_orders or any(order not in (2, 3) for order in requested_orders):
        raise ValueError("orders must be a non-empty subset of {2, 3}")
    source_path = source_path.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hdf5_name = "fc3.hdf5" if 3 in requested_orders else "fc2.hdf5"
    hdf5_path = output / hdf5_name
    meta_path = output / "hiphive_meta.json"
    if not force and (hdf5_path.exists() or meta_path.exists()):
        raise FileExistsError(
            f"{hdf5_path} or {meta_path} already exists; pass --force to replace"
        )

    primitive = silicon_primitive_cell(lattice_constant)
    conventional = conventional_cell(primitive)
    _, target_supercell = _phonopy_supercell(conventional, target_repeat)
    home, target_basis, relative_groups = dense_remap_indices(
        primitive, source_repeat, target_repeat, target_supercell
    )
    target_atoms = len(target_supercell)
    expected_source_atoms = len(primitive) * source_repeat**3
    diagnostics: dict[str, object] = {}
    temporary = hdf5_path.with_suffix(hdf5_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    kwargs = _hdf5_compression_kwargs(compression)

    with h5py.File(source_path, "r") as source, h5py.File(temporary, "w") as target:
        for order in requested_orders:
            name = f"fc{order}"
            if name not in source:
                raise KeyError(f"{source_path} has no {name} dataset")
            expected_source_shape = (expected_source_atoms,) * order + (3,) * order
            if source[name].shape != expected_source_shape:
                raise ValueError(
                    f"source {name} shape {source[name].shape}, "
                    f"expected {expected_source_shape}"
                )
            target_shape = (target_atoms,) * order + (3,) * order
            chunks = (1,) + (min(32, target_atoms),) * (order - 1) + (3,) * order
            dataset = target.create_dataset(
                name,
                shape=target_shape,
                dtype=source[name].dtype,
                chunks=chunks,
                **kwargs,
            )
            maximum = 0.0
            asr_maximum = 0.0
            home_slices = {
                basis_index: source[name][home[basis_index]]
                for basis_index in range(len(primitive))
            }
            for first in range(target_atoms):
                source_slice = home_slices[int(target_basis[first])]
                if order == 2:
                    output_slice = np.stack(
                        [
                            np.sum(source_slice[indices], axis=0)
                            for indices in relative_groups[first]
                        ]
                    )
                else:
                    output_slice = np.empty(
                        (target_atoms, target_atoms, 3, 3, 3),
                        dtype=source_slice.dtype,
                    )
                    for second, second_indices in enumerate(
                        relative_groups[first]
                    ):
                        for third, third_indices in enumerate(
                            relative_groups[first]
                        ):
                            output_slice[second, third] = np.sum(
                                source_slice[
                                    second_indices[:, np.newaxis],
                                    third_indices[np.newaxis, :],
                                ],
                                axis=(0, 1),
                            )
                dataset[first] = output_slice
                maximum = max(maximum, float(np.max(np.abs(output_slice))))
                asr_maximum = max(
                    asr_maximum,
                    float(np.max(np.abs(np.sum(output_slice, axis=order - 2)))),
                )
            diagnostics[f"fc{order}_maximum_ev_a{order}"] = maximum
            diagnostics[f"fc{order}_asr_maximum_ev_a{order}"] = asr_maximum
    temporary.replace(hdf5_path)

    meta: dict[str, object] = {
        "supercell": [target_repeat, target_repeat, target_repeat],
        "primitive": _atoms_record(conventional),
        "supercell_atoms": _atoms_record(target_supercell),
        "materialisation": {
            "source_force_constants": str(source_path),
            "source_force_constants_sha256": _sha256(source_path),
            "source_supercell": [source_repeat] * 3,
            "source_primitive": _atoms_record(primitive),
            "primitive_to_conventional_transform": CONVENTIONAL_TRANSFORM.tolist(),
            "orders": list(requested_orders),
            "hdf5_file": hdf5_name,
            "compression": compression,
            "diagnostics": diagnostics,
        },
    }
    if validate_folding:
        with h5py.File(source_path, "r") as source, h5py.File(hdf5_path, "r") as target:
            if "fc2" not in source or "fc2" not in target:
                raise ValueError("folding validation requires order 2")
            source_phonon, _ = _phonopy_supercell(primitive, source_repeat)
            target_phonon, _ = _phonopy_supercell(conventional, target_repeat)
            source_phonon.force_constants = source["fc2"][...]
            target_phonon.force_constants = target["fc2"][...]
            q_points = folded_primitive_q_points(primitive, conventional)
            primitive_frequencies = np.concatenate(
                [source_phonon.get_frequencies(q) for q in q_points]
            )
            conventional_frequencies = target_phonon.get_frequencies([0, 0, 0])
            difference = np.sort(primitive_frequencies) - np.sort(
                conventional_frequencies
            )
            meta["materialisation"]["gamma_folding_validation"] = {
                "folded_primitive_q_points": q_points.tolist(),
                "maximum_frequency_difference_thz": float(
                    np.max(np.abs(difference))
                ),
                "rms_frequency_difference_thz": float(
                    np.sqrt(np.mean(difference**2))
                ),
            }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def materialise(
    fcp_path: Path,
    output: Path,
    repeat: int,
    orders: Iterable[int],
    compression: str | None = "gzip",
    validate_folding: bool = False,
    force: bool = False,
) -> dict[str, object]:
    """Evaluate ``fcp_path`` on a conventional supercell and write HDF5."""
    import h5py
    from hiphive import ForceConstantPotential

    requested_orders = tuple(sorted(set(int(order) for order in orders)))
    if not requested_orders or any(order not in (2, 3) for order in requested_orders):
        raise ValueError("orders must be a non-empty subset of {2, 3}")
    fcp_path = fcp_path.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hdf5_name = "fc3.hdf5" if 3 in requested_orders else "fc2.hdf5"
    hdf5_path = output / hdf5_name
    meta_path = output / "hiphive_meta.json"
    if not force and (hdf5_path.exists() or meta_path.exists()):
        raise FileExistsError(
            f"{hdf5_path} or {meta_path} already exists; pass --force to replace"
        )

    fcp = ForceConstantPotential.read(str(fcp_path))
    conventional = conventional_cell(fcp.primitive_structure)
    _, supercell = _phonopy_supercell(conventional, repeat)
    force_constants = fcp.get_force_constants(supercell)

    temporary = hdf5_path.with_suffix(hdf5_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    diagnostics: dict[str, object] = {}
    with h5py.File(temporary, "w") as handle:
        for order in requested_orders:
            array = force_constants.get_fc_array(order=order)
            expected = (len(supercell),) * order + (3,) * order
            if array.shape != expected:
                raise RuntimeError(
                    f"unexpected FC{order} shape {array.shape}, expected {expected}"
                )
            diagnostics[f"fc{order}_maximum_ev_a{order}"] = float(
                np.max(np.abs(array))
            )
            diagnostics[f"fc{order}_asr_maximum_ev_a{order}"] = float(
                np.max(np.abs(np.sum(array, axis=order - 1)))
            )
            handle.create_dataset(
                f"fc{order}", data=array, compression=compression, shuffle=True
            )
            del array
    temporary.replace(hdf5_path)

    meta: dict[str, object] = {
        "supercell": [repeat, repeat, repeat],
        "primitive": _atoms_record(conventional),
        "supercell_atoms": _atoms_record(supercell),
        "materialisation": {
            "source_fcp": str(fcp_path),
            "source_fcp_sha256": _sha256(fcp_path),
            "source_primitive": _atoms_record(fcp.primitive_structure),
            "primitive_to_conventional_transform": CONVENTIONAL_TRANSFORM.tolist(),
            "orders": list(requested_orders),
            "hdf5_file": hdf5_name,
            "compression": compression,
            "diagnostics": diagnostics,
        },
    }
    if validate_folding:
        meta["materialisation"]["gamma_folding_validation"] = (
            validate_gamma_folding(fcp, conventional)
        )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _parse_orders(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("orders must look like 2 or 2,3") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fcp", type=Path)
    source.add_argument("--fc-hdf5", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--source-repeat", type=int, default=5)
    parser.add_argument("--lattice-constant", type=float, default=5.468)
    parser.add_argument("--orders", type=_parse_orders, default=(2, 3))
    parser.add_argument(
        "--compression", choices=("gzip", "lzf", "none"), default="gzip"
    )
    parser.add_argument("--validate-folding", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    compression = None if args.compression == "none" else args.compression
    if args.fcp is not None:
        result = materialise(
            args.fcp,
            args.output,
            args.repeat,
            args.orders,
            compression=compression,
            validate_folding=args.validate_folding,
            force=args.force,
        )
    else:
        result = materialise_dense(
            args.fc_hdf5,
            args.output,
            args.source_repeat,
            args.repeat,
            args.lattice_constant,
            args.orders,
            compression=compression,
            validate_folding=args.validate_folding,
            force=args.force,
        )
    print(json.dumps(result["materialisation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
