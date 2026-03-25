"""Write quatrex NEGF input files (dynamical_matrix.mat, structure.xyz, config.toml)."""

from pathlib import Path

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from scipy.io import savemat

from .config import QuatrexOutputConfig


def write_dynamical_matrix_mat(
    blocks: dict[tuple[int, int, int], np.ndarray],
    output_path: Path,
) -> None:
    """Write dynamical_matrix.mat with keys '[nx, ny, nz]'.

    Parameters
    ----------
    blocks : dict[(nx, ny, nz), ndarray]
        Real-space blocks in (rad/s)^2.
    output_path : Path
        Output .mat file path.
    """
    mat_dict = {}
    for (nx, ny, nz), D in blocks.items():
        mat_dict[f"[{nx}, {ny}, {nz}]"] = D
    savemat(str(output_path), mat_dict)


def write_structure_xyz(
    cell: PhonopyAtoms,
    output_path: Path,
) -> None:
    """Write extended XYZ file with lattice in header.

    Parameters
    ----------
    cell : PhonopyAtoms
        Unit cell structure.
    output_path : Path
        Output .xyz file path.
    """
    lv = cell.cell
    symbols = cell.symbols
    positions = cell.positions  # Cartesian, Angstrom

    with open(output_path, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(
            f'Lattice="{lv[0,0]} {lv[0,1]} {lv[0,2]} '
            f'{lv[1,0]} {lv[1,1]} {lv[1,2]} '
            f'{lv[2,0]} {lv[2,1]} {lv[2,2]}" '
            f'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        )
        for sym, pos in zip(symbols, positions):
            f.write(f"{sym}  {pos[0]:.8f}  {pos[1]:.8f}  {pos[2]:.8f}\n")


def write_quatrex_config_toml(
    cell: PhonopyAtoms,
    config: QuatrexOutputConfig,
    transport_direction: str,
    output_path: Path,
) -> None:
    """Write quatrex_config.toml.

    Parameters
    ----------
    cell : PhonopyAtoms
        Unit cell (to derive species and num_orbitals_per_atom).
    config : QuatrexOutputConfig
        Output configuration.
    transport_direction : str
        "x", "y", or "z".
    output_path : Path
        Output .toml file path.
    """
    unique_species = list(dict.fromkeys(cell.symbols))

    # num_orbitals_per_atom: always 3 DOFs per atom for phonons
    orbitals_section = "\n".join(
        f"{sp} = 3" for sp in unique_species
    )

    # kpoint_grid: transport direction component must be 1
    kg = list(config.kpoint_grid)
    tidx = "xyz".index(transport_direction)
    kg[tidx] = 1

    ks = list(config.kpoint_shift)
    nc = list(config.neighbor_cell_cutoff)

    text = f"""simulation_dir = "."
input_dir = "."

formalism = "negf"
simulation_type = "phonon"

[device]
transport_direction = '{transport_direction}'
construct_from_unit_cell = true
num_transport_cells = {config.num_transport_cells}
neighbor_cell_cutoff = [{nc[0]}, {nc[1]}, {nc[2]}]
kpoint_grid = [{kg[0]}, {kg[1]}, {kg[2]}]
kpoint_shift = [{ks[0]}, {ks[1]}, {ks[2]}]

[device.num_orbitals_per_atom]
{orbitals_section}

[scba]
max_iterations = 1
phonon = true

[electron]
energy_window_min = -1.0
energy_window_max = 1.0
energy_window_num = 10
fermi_level = 0.0
conduction_band_edge = 0.5
valence_band_edge = -0.5
left_fermi_level = 0.0
right_fermi_level = 0.0

[phonon]
eta = {config.eta}
eta_obc = 0.0
left_temperature = {config.left_temperature}
right_temperature = {config.right_temperature}
model = "negf"
phonon_energy = 0.063
deformation_potential = 1.0

[phonon.solver]
compute_current = true

[phonon.obc]
algorithm = "sancho-rubio"
block_sections = 1
"""
    with open(output_path, "w") as f:
        f.write(text)


def write_all(
    cell: PhonopyAtoms,
    blocks: dict[tuple[int, int, int], np.ndarray],
    config: QuatrexOutputConfig,
    transport_direction: str = "z",
) -> Path:
    """Write all quatrex input files.

    Parameters
    ----------
    cell : PhonopyAtoms
        Unit cell.
    blocks : dict
        Real-space dynamical matrix blocks.
    config : QuatrexOutputConfig
        Output settings.
    transport_direction : str
        Transport direction.

    Returns
    -------
    output_dir : Path
    """
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_dynamical_matrix_mat(blocks, out / "dynamical_matrix.mat")
    write_structure_xyz(cell, out / "structure.xyz")
    write_quatrex_config_toml(cell, config, transport_direction, out / "quatrex_config.toml")

    return out
