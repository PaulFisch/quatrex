"""Crystal structure loading and Phonopy object creation."""

from pathlib import Path

import numpy as np
from phonopy import Phonopy, load as phonopy_load
from phonopy.structure.atoms import PhonopyAtoms

from .config import PhononInputConfig, StructureConfig

# Unit conversion: phonopy QE internal (Ry/bohr²) -> eV/Å²
BOHR_TO_ANG = 0.52917721067
RY_TO_EV = 13.605693009
RY_BOHR2_TO_EV_ANG2 = RY_TO_EV / BOHR_TO_ANG**2  # ≈ 48.587


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


def load_phonopy_calculation(
    phonopy_yaml: str | Path,
    force_sets_filename: str | Path | None = None,
    force_constants_filename: str | Path | None = None,
    primitive_matrix: np.ndarray | str = np.eye(3),
    calculator: str = "qe",
) -> Phonopy:
    """Load an existing phonopy calculation and normalize to eV/Angstrom units.

    Phonopy's QE interface stores lengths in bohr and force constants in
    Ry/bohr². This function converts everything to Angstrom / eV/Å² so
    that downstream code can use the standard CONVERSION factor.

    Parameters
    ----------
    phonopy_yaml : Path
        Path to phonopy_disp.yaml or phonopy.yaml.
    force_sets_filename : Path, optional
        Path to FORCE_SETS file.
    force_constants_filename : Path, optional
        Path to FORCE_CONSTANTS file.
    primitive_matrix : array or "auto"
        Primitive matrix for the output Phonopy object. Default np.eye(3)
        gives the conventional cell (same as unit cell).
    calculator : str
        Calculator used for the original DFT. "qe" applies bohr/Ry
        conversion; "vasp" assumes eV/Å already.

    Returns
    -------
    Phonopy
        Phonopy object with FC in eV/Å², lattice in Å, ready for use
        with CONVERSION factor.
    """
    ph_native = phonopy_load(
        phonopy_yaml=str(phonopy_yaml),
        force_sets_filename=str(force_sets_filename) if force_sets_filename else None,
        force_constants_filename=(
            str(force_constants_filename) if force_constants_filename else None
        ),
    )
    ph_native.produce_force_constants()

    if calculator.lower() == "qe":
        cell_native = ph_native.unitcell.cell
        cell_ang = cell_native * BOHR_TO_ANG
        fc_scale = RY_BOHR2_TO_EV_ANG2
    else:
        cell_ang = ph_native.unitcell.cell
        fc_scale = 1.0

    unitcell = PhonopyAtoms(
        symbols=ph_native.unitcell.symbols,
        cell=cell_ang,
        scaled_positions=ph_native.unitcell.scaled_positions,
    )

    sc_matrix = ph_native.supercell_matrix
    if isinstance(primitive_matrix, str):
        pm = primitive_matrix
    else:
        pm = np.array(primitive_matrix)

    ph = Phonopy(unitcell, supercell_matrix=sc_matrix, primitive_matrix=pm)
    ph.force_constants = ph_native.force_constants * fc_scale

    return ph


def clone_with_masses(
    phonon: Phonopy,
    symbols: list[str] | None = None,
    masses: list[float] | None = None,
) -> Phonopy:
    """Create a Phonopy object with different masses but same force constants.

    For mass-mismatch interface models: use FC from one material (e.g. Si)
    but substitute atomic masses (e.g. Ge) so the dynamical matrix
    D = Phi / sqrt(m_i m_j) reflects the mass difference.

    Parameters
    ----------
    phonon : Phonopy
        Source Phonopy object with force constants set.
    symbols : list of str, optional
        New chemical symbols for the unit cell atoms. If given, masses
        are taken from the periodic table for these symbols.
    masses : list of float, optional
        Explicit masses in amu. Overrides symbols if both are given.

    Returns
    -------
    Phonopy
        New Phonopy object with same lattice, positions, supercell, and FC,
        but different masses.
    """
    if phonon.force_constants is None:
        raise ValueError("Source phonopy must have force_constants set.")

    old_cell = phonon.unitcell

    new_cell = PhonopyAtoms(
        symbols=symbols if symbols is not None else old_cell.symbols,
        cell=old_cell.cell,
        scaled_positions=old_cell.scaled_positions,
    )
    if masses is not None:
        new_cell.masses = masses

    ph = Phonopy(
        new_cell,
        supercell_matrix=phonon.supercell_matrix,
        primitive_matrix=phonon.primitive_matrix,
    )
    ph.force_constants = phonon.force_constants
    return ph
