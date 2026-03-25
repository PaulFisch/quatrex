"""Crystal structure loading and Phonopy object creation."""

from pathlib import Path

import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from .config import PhononInputConfig, StructureConfig


def load_structure(config: StructureConfig) -> PhonopyAtoms:
    """Load a crystal structure from any supported format.

    Parameters
    ----------
    config : StructureConfig
        Structure specification (source format + path or inline data).

    Returns
    -------
    PhonopyAtoms
    """
    source = config.source.lower()

    if source == "inline":
        if config.symbols is None or config.lattice is None:
            raise ValueError("Inline source requires symbols and lattice.")
        return PhonopyAtoms(
            symbols=config.symbols,
            cell=np.array(config.lattice),
            scaled_positions=np.array(config.scaled_positions),
        )

    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    if source == "phonopy_yaml":
        from phonopy.interface.phonopy_yaml import PhonopyYaml

        pyaml = PhonopyYaml()
        pyaml.read(path)
        return pyaml.unitcell

    if source in ("cif", "poscar", "vasp"):
        import ase.io

        atoms = ase.io.read(str(path))
        return structure_from_ase(atoms)

    if source == "qe_input":
        from phonopy.interface.qe import read_qe_input

        cell, _ = read_qe_input(path)
        return cell

    raise ValueError(f"Unknown structure source: {source}")


def structure_from_ase(atoms) -> PhonopyAtoms:
    """Convert ASE Atoms to PhonopyAtoms."""
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )


def structure_to_ase(cell: PhonopyAtoms):
    """Convert PhonopyAtoms to ASE Atoms."""
    import ase

    return ase.Atoms(
        symbols=cell.symbols,
        cell=cell.cell,
        scaled_positions=cell.scaled_positions,
        pbc=True,
    )


def create_phonopy(
    cell: PhonopyAtoms,
    supercell_matrix: np.ndarray,
    primitive_matrix: np.ndarray | str = "auto",
    displacement_distance: float = 0.01,
) -> Phonopy:
    """Create a Phonopy object with displacements generated.

    Parameters
    ----------
    cell : PhonopyAtoms
        Unit cell.
    supercell_matrix : array_like, shape (3,3) or (3,)
        Supercell matrix (diagonal or full).
    primitive_matrix : array_like or "auto"
        Primitive cell matrix. Use np.eye(3) to keep the input cell as-is.
    displacement_distance : float
        Displacement amplitude in Angstrom.

    Returns
    -------
    Phonopy
    """
    sc = np.array(supercell_matrix)
    if sc.ndim == 1:
        sc = np.diag(sc)

    if isinstance(primitive_matrix, str) and primitive_matrix == "auto":
        pm = "auto"
    else:
        pm = np.array(primitive_matrix)

    phonon = Phonopy(cell, supercell_matrix=sc, primitive_matrix=pm)
    phonon.generate_displacements(distance=displacement_distance)
    return phonon


def create_phonopy_from_config(config: PhononInputConfig) -> Phonopy:
    """Create a Phonopy object from the full configuration."""
    cell = load_structure(config.structure)
    return create_phonopy(
        cell,
        supercell_matrix=np.array(config.supercell.matrix),
        primitive_matrix=np.eye(3),
        displacement_distance=config.supercell.displacement_distance,
    )
