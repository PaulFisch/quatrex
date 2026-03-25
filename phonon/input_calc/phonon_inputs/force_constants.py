"""Force constant extraction from phonopy.

Handles second-order (harmonic) force constants via phonopy, with stubs
for third-order force constants from thirdorder.py or phono3py.
"""

from pathlib import Path

import numpy as np
from phonopy import Phonopy


def produce_force_constants(
    phonon: Phonopy,
    forces: list[np.ndarray] | None = None,
    force_constants_file: Path | None = None,
    force_sets_file: Path | None = None,
) -> np.ndarray:
    """Produce second-order force constants.

    Priority: explicit forces > FORCE_CONSTANTS file > FORCE_SETS file.

    Parameters
    ----------
    phonon : Phonopy
        Phonopy object with displacements generated.
    forces : list of ndarray, optional
        Forces for each displacement (eV/Angstrom).
    force_constants_file : Path, optional
        Path to a phonopy FORCE_CONSTANTS file.
    force_sets_file : Path, optional
        Path to a phonopy FORCE_SETS file.

    Returns
    -------
    fc : ndarray
        Force constants array (phonopy internal shape).
    """
    if forces is not None:
        phonon.forces = forces
        phonon.produce_force_constants()
    elif force_constants_file is not None:
        from phonopy.file_IO import parse_FORCE_CONSTANTS

        fc = parse_FORCE_CONSTANTS(str(force_constants_file))
        phonon.force_constants = fc
    elif force_sets_file is not None:
        from phonopy.file_IO import parse_FORCE_SETS

        force_sets = parse_FORCE_SETS(str(force_sets_file))
        phonon.dataset = force_sets
        phonon.produce_force_constants()
    else:
        raise ValueError(
            "Provide one of: forces, force_constants_file, or force_sets_file."
        )

    return phonon.force_constants


def load_forces_phonopy_qe(
    n_atoms_supercell: int, output_files: list[str | Path]
) -> list[np.ndarray]:
    """Load forces using phonopy's QE parser.

    QE outputs forces in Ry/Bohr; phonopy expects eV/Angstrom.
    This function applies the conversion factor (1 Ry/Bohr = 25.711 eV/A).

    Parameters
    ----------
    n_atoms_supercell : int
        Number of atoms in the supercell.
    output_files : list of str or Path
        QE output files, one per displacement.

    Returns
    -------
    forces : list of ndarray, in eV/Angstrom
    """
    import phonopy.interface.qe

    RY_BOHR_TO_EV_A = 25.71104309541616  # 13.605693009 eV/Ry / 0.52917721067 A/Bohr
    raw = phonopy.interface.qe.parse_set_of_forces(
        n_atoms_supercell, [str(f) for f in output_files]
    )
    return [f * RY_BOHR_TO_EV_A for f in raw]


# ---- Third-order force constants (stubs) ----


def load_fc3_thirdorder(path: Path | str) -> dict:
    """Load third-order force constants from thirdorder.py output.

    Reads the FORCE_CONSTANTS_3RD file format produced by thirdorder.py.
    The file lists blocks of Phi_3(0 kappa, R' kappa', R'' kappa'')
    with lattice vectors R', R'' and atom indices.

    Parameters
    ----------
    path : Path
        Path to FORCE_CONSTANTS_3RD file.

    Returns
    -------
    fc3_data : dict
        Keys:
        - "n_blocks": int, number of FC3 triplets
        - "blocks": list of dicts, each with:
            - "cell_j": (3,) int, lattice vector R' in units of lattice vectors
            - "cell_k": (3,) int, lattice vector R''
            - "atom_i": int (0-based)
            - "atom_j": int (0-based)
            - "atom_k": int (0-based)
            - "tensor": (3, 3, 3) float, in eV/Angstrom^3
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FC3 file not found: {path}")

    blocks = []
    with open(path) as f:
        n_blocks = int(f.readline().strip())
        for _ in range(n_blocks):
            # Cell indices for atoms j and k
            cell_j = np.array([int(x) for x in f.readline().split()])
            cell_k = np.array([int(x) for x in f.readline().split()])
            # Atom indices (1-based in file -> 0-based)
            atom_i, atom_j, atom_k = [
                int(x) - 1 for x in f.readline().split()
            ]
            # 3x3x3 tensor
            tensor = np.zeros((3, 3, 3))
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        parts = f.readline().split()
                        # File format: alpha beta gamma value
                        tensor[int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1] = float(
                            parts[3]
                        )
            blocks.append(
                {
                    "cell_j": cell_j,
                    "cell_k": cell_k,
                    "atom_i": atom_i,
                    "atom_j": atom_j,
                    "atom_k": atom_k,
                    "tensor": tensor,
                }
            )

    return {"n_blocks": n_blocks, "blocks": blocks}


def load_fc3_phono3py(yaml_path: Path | str):
    """Load FC3 from a phono3py calculation.

    Parameters
    ----------
    yaml_path : Path
        Path to phono3py YAML or fc3.hdf5 file.

    Returns
    -------
    Phono3py object with fc3 loaded.
    """
    raise NotImplementedError(
        "phono3py FC3 loading is not yet implemented. "
        "Use load_fc3_thirdorder() for FORCE_CONSTANTS_3RD files."
    )
