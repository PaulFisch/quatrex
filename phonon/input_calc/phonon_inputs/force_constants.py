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
            # Blank line + block number
            f.readline()  # blank
            f.readline()  # block index (1-based, not used)
            # Lattice vectors for cells j and k (in nm, float)
            cell_j = np.array([float(x) for x in f.readline().split()])
            cell_k = np.array([float(x) for x in f.readline().split()])
            # Atom indices (1-based in file -> 0-based)
            atom_i, atom_j, atom_k = [
                int(x) - 1 for x in f.readline().split()
            ]
            # 3x3x3 tensor
            tensor = np.zeros((3, 3, 3))
            for _ in range(27):
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


def _dense_fc3_to_blocks(fc3: np.ndarray, primitive, supercell) -> dict:
    """Convert dense FC3 to sparse block format.

    Parameters
    ----------
    fc3 : ndarray
        Dense FC3 array, either compact:
            (n_prim, n_super, n_super, 3, 3, 3)
        or full:
            (n_super, n_super, n_super, 3, 3, 3)
    primitive : Primitive
        phono3py primitive object.
    supercell : Supercell
        phono3py supercell object.

    Returns
    -------
    fc3_data : dict
        Sparse block format matching ``load_fc3_thirdorder()`` output.
    """
    p2s = primitive.p2s_map
    s2u = supercell.s2u_map
    n_super = len(supercell.masses)
    n_prim = len(primitive.masses)

    # Map each supercell atom to:
    #   - primitive atom index
    #   - lattice vector relative to the corresponding unit-cell atom
    p2s_inv = {int(s_idx): p_idx for p_idx, s_idx in enumerate(p2s)}
    sc_to_prim = np.array([p2s_inv[int(s2u[j])] for j in range(n_super)], dtype=int)
    cell_vectors_cart = np.array(
        [supercell.positions[j] - supercell.positions[int(s2u[j])] for j in range(n_super)]
    )

    is_compact = fc3.shape[0] == n_prim
    blocks = []
    threshold = 1e-15  # eV/Ang^3

    # For compact format, i_idx is already the primitive atom index.
    # For full format, only iterate over p2s_map atoms (reference cell)
    # to avoid 8x overcounting and incorrect cell vector offsets.
    if is_compact:
        i_indices = range(n_prim)
    else:
        i_indices = [int(s) for s in p2s]

    for i_idx in i_indices:
        atom_i = i_idx if is_compact else p2s_inv[int(s2u[i_idx])]
        for j in range(n_super):
            for k in range(n_super):
                tensor = fc3[i_idx, j, k]
                if np.max(np.abs(tensor)) < threshold:
                    continue

                blocks.append(
                    {
                        "cell_j": cell_vectors_cart[j].copy(),
                        "cell_k": cell_vectors_cart[k].copy(),
                        "atom_i": int(atom_i),
                        "atom_j": int(sc_to_prim[j]),
                        "atom_k": int(sc_to_prim[k]),
                        "tensor": tensor.copy(),
                    }
                )

    return {"n_blocks": len(blocks), "blocks": blocks}


def load_fc3_dfpt_hdf5(
        dfpt_hdf5: Path | str,
        phono3py_yaml: Path | str,
        log_level: int = 0,
) -> dict:
    """Load DFPT FC2+FC3 from the HDF5 produced by your DFPT reap step.

    Assumes:
      - ``dfpt_hdf5`` contains datasets ``fc2`` and ``fc3``
      - ``fc3`` is already in phono3py compact convention
      - ``phono3py_yaml`` describes the same primitive/supercell mapping

    Parameters
    ----------
    dfpt_hdf5 : Path or str
        Path to DFPT ``fc3.hdf5`` written by your reap step.
    phono3py_yaml : Path or str
        Reference phono3py_disp.yaml used to reconstruct structure/mapping.
    log_level : int
        phono3py log level.

    Returns
    -------
    payload : dict
        Keys:
        - ``fc2`` : ndarray
        - ``fc3`` : ndarray
        - ``fc3_data`` : sparse block dict
        - ``ph3`` : loaded phono3py object
    """
    import h5py
    from phono3py import load as phono3py_load

    dfpt_hdf5 = Path(dfpt_hdf5)
    if not dfpt_hdf5.exists():
        raise FileNotFoundError(f"DFPT HDF5 not found: {dfpt_hdf5}")

    with h5py.File(dfpt_hdf5, "r") as f:
        fc2 = f["fc2"][:]
        fc3 = f["fc3"][:]

    ph3 = phono3py_load(
        phono3py_yaml=str(phono3py_yaml),
        produce_fc=False,
        log_level=log_level,
    )

    fc3_data = _dense_fc3_to_blocks(fc3, ph3.primitive, ph3.supercell)

    return {
        "fc2": fc2,
        "fc3": fc3,
        "fc3_data": fc3_data,
        "ph3": ph3,
    }


def _phono3py_fc3_to_blocks(ph3) -> dict:
    """Convert phono3py dense FC3 to sparse block format."""
    if ph3.fc3 is None:
        raise RuntimeError("ph3.fc3 is not set.")
    return _dense_fc3_to_blocks(ph3.fc3, ph3.primitive, ph3.supercell)


def load_fc3_phono3py(
        phono3py_yaml: Path | str | None = None,
        fc3_hdf5: Path | str | None = None,
        unitcell=None,
        supercell_matrix: np.ndarray | None = None,
        forces: list[np.ndarray] | None = None,
        calculator: str | None = None,
        fc_calculator: str | None = None,
        log_level: int = 0,
) -> dict:
    """Load or compute FC3 from phono3py.

    Supports three workflows:
    1. Load from phono3py YAML (with forces already set or in directory)
    2. Load from pre-computed fc3.hdf5
    3. Provide unitcell + supercell_matrix + forces to compute FC3

    Use ``fc_calculator="symfc"`` for efficient FC3 production via symfc.

    Parameters
    ----------
    phono3py_yaml : Path, optional
        Path to phono3py_disp.yaml file.
    fc3_hdf5 : Path, optional
        Path to pre-computed fc3.hdf5 file.
    unitcell : PhonopyAtoms, optional
        Unit cell for creating Phono3py from scratch.
    supercell_matrix : ndarray, optional
        Supercell matrix (3x3 or diagonal).
    forces : list of ndarray, optional
        Forces for each displacement in eV/Angstrom.
    calculator : str, optional
        Calculator name (e.g., "qe").
    fc_calculator : str, optional
        FC calculator backend (e.g., "symfc" for symmetry-adapted).
    log_level : int
        Phono3py log level (0=silent).

    Returns
    -------
    fc3_data : dict
        Same format as ``load_fc3_thirdorder()``:
        - "n_blocks": int
        - "blocks": list of dicts with cell_j, cell_k, atom_i/j/k, tensor
        FC3 tensors are in eV/Angstrom^3.
    """
    from phono3py import Phono3py

    ph3 = None

    if fc3_hdf5 is not None:
        # Load pre-computed FC3 from HDF5
        import h5py

        fc3_hdf5 = Path(fc3_hdf5)
        with h5py.File(fc3_hdf5, "r") as f:
            fc3_dense = f["fc3"][:]

        if phono3py_yaml is not None:
            from phono3py import load as phono3py_load
            ph3 = phono3py_load(
                phono3py_yaml=str(phono3py_yaml),
                produce_fc=False,
                log_level=log_level,
            )
        elif unitcell is not None and supercell_matrix is not None:
            ph3 = Phono3py(unitcell, supercell_matrix=supercell_matrix)
        else:
            raise ValueError(
                "When loading fc3.hdf5, also provide phono3py_yaml or "
                "unitcell+supercell_matrix for structure info."
            )
        ph3.fc3 = fc3_dense

    elif phono3py_yaml is not None:
        from phono3py import load as phono3py_load

        yaml_path = Path(phono3py_yaml)
        fc3_path = yaml_path.parent / "fc3.hdf5"

        if fc3_path.exists():
            ph3 = phono3py_load(
                phono3py_yaml=str(yaml_path),
                fc3_filename=str(fc3_path),
                produce_fc=False,
                log_level=log_level,
            )
        else:
            kwargs = {}
            if forces is not None:
                kwargs["forces_fc3_filename"] = None
            if fc_calculator:
                kwargs["fc_calculator"] = fc_calculator
            ph3 = phono3py_load(
                phono3py_yaml=str(yaml_path),
                calculator=calculator,
                produce_fc=True,
                log_level=log_level,
                **kwargs,
            )
            if forces is not None:
                ph3.forces = forces
                ph3.produce_fc3(fc_calculator=fc_calculator)

    elif unitcell is not None and supercell_matrix is not None:
        ph3 = Phono3py(unitcell, supercell_matrix=np.asarray(supercell_matrix))
        if forces is not None:
            ph3.forces = forces
            ph3.produce_fc3(fc_calculator=fc_calculator)
        else:
            raise ValueError("Provide forces or use phono3py_yaml/fc3_hdf5.")
    else:
        raise ValueError(
            "Provide one of: phono3py_yaml, fc3_hdf5, or unitcell+supercell_matrix."
        )

    if ph3.fc3 is None:
        raise RuntimeError("FC3 not produced. Check inputs and forces.")

    return _phono3py_fc3_to_blocks(ph3)


def _phono3py_fc3_to_blocks(ph3) -> dict:
    """Convert phono3py dense FC3 to sparse block format."""
    if ph3.fc3 is None:
        raise RuntimeError("ph3.fc3 is not set.")
    return _dense_fc3_to_blocks(ph3.fc3, ph3.primitive, ph3.supercell)
