"""Quantum ESPRESSO input/output interface."""

import subprocess
from pathlib import Path

import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from .config import QEConfig

# Ry/bohr -> eV/Angstrom
RY_BOHR_TO_EV_ANG = 25.71104309541616


def write_qe_scf_input(
    filepath: Path,
    supercell: PhonopyAtoms,
    qe_config: QEConfig,
    calculation: str = "scf",
) -> None:
    """Write a pw.x input file for a (displaced) supercell.

    Parameters
    ----------
    filepath : Path
        Output file path.
    supercell : PhonopyAtoms
        Supercell structure (possibly with displacement applied).
    qe_config : QEConfig
        QE calculation parameters.
    calculation : str
        QE calculation type ("scf", "relax", etc.).
    """
    filepath = Path(filepath)
    symbols = supercell.symbols
    unique_species = list(dict.fromkeys(symbols))
    cell = supercell.cell
    positions = supercell.scaled_positions
    natoms = len(symbols)

    # Get masses from phonopy atoms (amu)
    masses = {}
    for sym, mass in zip(symbols, supercell.masses):
        masses[sym] = mass

    ecutrho = qe_config.ecutwfc * qe_config.ecutrho_factor
    kpts = qe_config.kpoints_scf

    # Use absolute paths so QE works regardless of CWD
    pseudo_dir = Path(qe_config.pseudo_dir).absolute()
    outdir = (filepath.parent / "results").absolute()

    with open(filepath, "w") as f:
        f.write("&CONTROL\n")
        f.write(f"   calculation      = '{calculation}'\n")
        f.write("   restart_mode     = 'from_scratch'\n")
        f.write("   prefix           = 'phonon_fc'\n")
        f.write(f"   pseudo_dir       = '{pseudo_dir}'\n")
        f.write(f"   outdir           = '{outdir}'\n")
        f.write("   tprnfor          = .true.\n")
        f.write("/\n")

        f.write("&SYSTEM\n")
        f.write("   ibrav            = 0\n")
        f.write(f"   nat              = {natoms}\n")
        f.write(f"   ntyp             = {len(unique_species)}\n")
        f.write(f"   ecutwfc          = {qe_config.ecutwfc}\n")
        f.write(f"   ecutrho          = {ecutrho}\n")
        f.write("   occupations      = 'smearing'\n")
        f.write(f"   smearing         = '{qe_config.smearing}'\n")
        f.write(f"   degauss          = {qe_config.degauss}\n")
        f.write("/\n")

        f.write("&ELECTRONS\n")
        f.write(f"   conv_thr         = {qe_config.conv_thr}\n")
        f.write("/\n")

        if calculation == "relax":
            f.write("&IONS\n")
            f.write("   ion_dynamics     = 'bfgs'\n")
            f.write("/\n")
            f.write("&CELL\n")
            f.write("/\n")

        f.write("ATOMIC_SPECIES\n")
        for sp in unique_species:
            pseudo = qe_config.pseudopotentials.get(sp, f"{sp}.UPF")
            f.write(f"  {sp}  {masses[sp]}  {pseudo}\n")

        f.write("ATOMIC_POSITIONS crystal\n")
        for sym, pos in zip(symbols, positions):
            f.write(f"  {sym}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}\n")

        f.write("CELL_PARAMETERS angstrom\n")
        for v in cell:
            f.write(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}\n")

        f.write("K_POINTS automatic\n")
        f.write(f"  {kpts[0]} {kpts[1]} {kpts[2]}  0 0 0\n")


def read_qe_forces(filepath: Path, natoms: int) -> np.ndarray:
    """Parse forces from a pw.x output file.

    Parameters
    ----------
    filepath : Path
        QE output file.
    natoms : int
        Expected number of atoms.

    Returns
    -------
    forces : ndarray, shape (natoms, 3)
        Forces in eV/Angstrom.
    """
    with open(filepath) as f:
        text = f.read()

    if "JOB DONE" not in text:
        raise RuntimeError(f"QE did not converge: {filepath}")

    forces = []
    for line in text.split("Forces acting on atoms")[-1].split("\n"):
        if "force =" in line:
            vals = line.split("force =")[1].split()
            forces.append([float(v) for v in vals[:3]])
        if len(forces) == natoms:
            break

    if len(forces) != natoms:
        raise ValueError(f"Expected {natoms} forces, got {len(forces)} in {filepath}")

    return np.array(forces) * RY_BOHR_TO_EV_ANG


def run_qe(input_file: Path, output_file: Path, pw_command: str = "pw.x") -> None:
    """Run a QE pw.x calculation.

    Parameters
    ----------
    input_file : Path
        QE input file.
    output_file : Path
        QE output file (stdout).
    pw_command : str
        Command to invoke pw.x (e.g., "mpirun -np 4 pw.x").
    """
    cmd = f"{pw_command} -in {input_file}"
    with open(output_file, "w") as fout:
        subprocess.run(cmd, shell=True, stdout=fout, stderr=subprocess.STDOUT)


def run_qe_displacements(
    phonon: Phonopy,
    work_dir: Path,
    qe_config: QEConfig,
    skip_existing: bool = True,
) -> list[np.ndarray]:
    """Run QE SCF for each phonopy displacement.

    Parameters
    ----------
    phonon : Phonopy
        Phonopy object with displacements generated.
    work_dir : Path
        Working directory for QE files.
    qe_config : QEConfig
        QE parameters.
    skip_existing : bool
        Skip if output file already exists and contains "JOB DONE".

    Returns
    -------
    forces : list of ndarray
        Forces for each displacement, in eV/Angstrom.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir = work_dir / "results"
    results_dir.mkdir(exist_ok=True)

    supercells = phonon.supercells_with_displacements
    natoms = len(phonon.supercell.positions)
    forces = []

    for i, scell in enumerate(supercells):
        idx = i + 1
        inp = work_dir / f"disp-{idx:03d}.in"
        out = work_dir / f"disp-{idx:03d}.out"

        if skip_existing and out.exists():
            try:
                f = read_qe_forces(out, natoms)
                forces.append(f)
                continue
            except (RuntimeError, ValueError):
                pass  # Re-run if output is incomplete

        # Create PhonopyAtoms for this displaced supercell
        displaced = PhonopyAtoms(
            symbols=phonon.supercell.symbols,
            cell=scell.cell,
            scaled_positions=scell.scaled_positions,
        )
        write_qe_scf_input(inp, displaced, qe_config)
        run_qe(inp, out, qe_config.pw_command)
        forces.append(read_qe_forces(out, natoms))

    return forces


def load_existing_forces(
    phonon: Phonopy,
    output_dir: Path,
    filename_pattern: str = "disp-{:03d}.out",
) -> list[np.ndarray]:
    """Load forces from existing QE output files.

    Also supports phonopy's QE parser if available via
    ``phonopy.interface.qe.parse_set_of_forces``.

    Parameters
    ----------
    phonon : Phonopy
        Phonopy object (for natoms).
    output_dir : Path
        Directory containing QE output files.
    filename_pattern : str
        Pattern for output filenames (uses 1-based index).

    Returns
    -------
    forces : list of ndarray
    """
    output_dir = Path(output_dir)
    natoms = len(phonon.supercell.positions)
    n_disp = len(phonon.displacements)
    forces = []

    for i in range(1, n_disp + 1):
        out = output_dir / filename_pattern.format(i)
        if not out.exists():
            # Try phonopy's naming convention
            out = output_dir / f"supercell-{i:03d}.out"
        forces.append(read_qe_forces(out, natoms))

    return forces
