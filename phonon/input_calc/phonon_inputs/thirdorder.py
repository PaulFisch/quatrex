"""Third-order force constant generation via thirdorder.py + QE.

Wraps the thirdorder_espresso.py sow/reap workflow for any crystal structure.

Workflow:
    1. sow:  Write unitcell.in + template, run thirdorder sow -> DISP files
    2. run:  Execute QE pw.x for each displacement
    3. reap: Pipe output filenames to thirdorder reap -> FORCE_CONSTANTS_3RD
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms

from .config import QEConfig


def write_unitcell_qe(cell: PhonopyAtoms, path: Path, qe_config: QEConfig) -> None:
    """Write a QE-format unitcell.in for thirdorder.

    Parameters
    ----------
    cell : PhonopyAtoms
        Unit cell (primitive).
    path : Path
        Output file path.
    qe_config : QEConfig
        QE parameters (pseudo_dir, ecutwfc, etc.).
    """
    symbols = cell.symbols
    unique_species = list(dict.fromkeys(symbols))
    lattice = cell.cell
    positions = cell.scaled_positions
    natoms = len(symbols)

    masses = {}
    for sym, mass in zip(symbols, cell.masses):
        masses[sym] = mass

    ecutrho = qe_config.ecutwfc * qe_config.ecutrho_factor

    with open(path, "w") as f:
        f.write("&CONTROL\n")
        f.write("   calculation      = 'scf'\n")
        f.write(f"   pseudo_dir       = '{qe_config.pseudo_dir}'\n")
        f.write("   outdir           = './results'\n")
        f.write("   tprnfor          = .true.\n")
        f.write("/\n")

        f.write("&SYSTEM\n")
        f.write("   ibrav            = 0\n")
        f.write(f"   nat              = {natoms}\n")
        f.write(f"   ntyp             = {len(unique_species)}\n")
        f.write(f"   ecutwfc          = {qe_config.ecutwfc}\n")
        f.write(f"   ecutrho          = {ecutrho}\n")
        f.write("/\n")

        f.write("&ELECTRONS\n")
        f.write(f"   conv_thr         = {qe_config.conv_thr}\n")
        f.write("/\n")

        f.write("ATOMIC_SPECIES\n")
        for sp in unique_species:
            pseudo = qe_config.pseudopotentials.get(sp, f"{sp}.UPF")
            f.write(f"  {sp}  {masses[sp]}  {pseudo}\n")

        f.write("ATOMIC_POSITIONS crystal\n")
        for sym, pos in zip(symbols, positions):
            f.write(f"  {sym}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}\n")

        f.write("CELL_PARAMETERS angstrom\n")
        for v in lattice:
            f.write(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}\n")

        kpts = qe_config.kpoints_relax  # unitcell uses finer k-mesh
        f.write("K_POINTS automatic\n")
        f.write(f"  {kpts[0]} {kpts[1]} {kpts[2]}  0 0 0\n")


def write_supercell_template(
    path: Path,
    qe_config: QEConfig,
    unique_species: list[str],
    masses: dict[str, float],
) -> None:
    """Write the supercell template with ##NATOMS##, ##COORDINATES##, ##CELL## placeholders.

    Parameters
    ----------
    path : Path
        Output file path.
    qe_config : QEConfig
        QE parameters.
    unique_species : list of str
        Unique atomic species.
    masses : dict
        Masses keyed by symbol.
    """
    ecutrho = qe_config.ecutwfc * qe_config.ecutrho_factor

    with open(path, "w") as f:
        f.write("&CONTROL\n")
        f.write("   calculation      = 'scf'\n")
        f.write("   restart_mode     = 'from_scratch'\n")
        f.write("   prefix           = 'fc3_disp'\n")
        f.write(f"   pseudo_dir       = '{qe_config.pseudo_dir}'\n")
        f.write("   outdir           = './results'\n")
        f.write("   tprnfor          = .true.\n")
        f.write("/\n")

        f.write("&SYSTEM\n")
        f.write("   ibrav            = 0\n")
        f.write("   nat              = ##NATOMS##\n")
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

        f.write("ATOMIC_SPECIES\n")
        for sp in unique_species:
            pseudo = qe_config.pseudopotentials.get(sp, f"{sp}.UPF")
            f.write(f"  {sp}  {masses[sp]}  {pseudo}\n")

        f.write("##COORDINATES##\n")
        f.write("##CELL##\n")

        kpts = qe_config.kpoints_scf  # supercell uses coarser k-mesh
        f.write("K_POINTS automatic\n")
        f.write(f"  {kpts[0]} {kpts[1]} {kpts[2]}  0 0 0\n")


def sow(
    cell: PhonopyAtoms,
    work_dir: Path,
    qe_config: QEConfig,
    supercell: tuple[int, int, int] = (2, 2, 2),
    cutoff: str = "-3",
    thirdorder_cmd: str = "thirdorder_espresso.py",
) -> int:
    """Run thirdorder sow to generate displaced supercells.

    Parameters
    ----------
    cell : PhonopyAtoms
        Primitive unit cell.
    work_dir : Path
        Working directory (will be created).
    qe_config : QEConfig
        QE parameters.
    supercell : tuple of int
        Supercell dimensions (na, nb, nc).
    cutoff : str
        Cutoff for thirdorder ("-3" = 3rd neighbor, or distance in nm).
    thirdorder_cmd : str
        Command for thirdorder_espresso.py.

    Returns
    -------
    n_disp : int
        Number of displacement files generated.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write unitcell.in
    unitcell_path = work_dir / "unitcell.in"
    write_unitcell_qe(cell, unitcell_path, qe_config)

    # Write supercell_template.in
    template_path = work_dir / "supercell_template.in"
    symbols = cell.symbols
    unique_species = list(dict.fromkeys(symbols))
    masses = {sym: mass for sym, mass in zip(symbols, cell.masses)}
    write_supercell_template(template_path, qe_config, unique_species, masses)

    # Run thirdorder sow
    na, nb, nc = supercell
    result = subprocess.run(
        [thirdorder_cmd, "unitcell.in", "sow",
         str(na), str(nb), str(nc), cutoff, "supercell_template.in"],
        capture_output=True, text=True,
        cwd=str(work_dir),
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"thirdorder sow failed:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError("thirdorder sow failed")

    # Count generated DISP files
    disp_files = sorted(work_dir.glob("DISP.supercell_template.in.*"))
    n_disp = len(disp_files)
    print(f"Generated {n_disp} displacement files in {work_dir}")
    return n_disp


def run_displacements(
    work_dir: Path,
    n_disp: int,
    pw_command: str = "pw.x",
    timeout: int = 3600,
) -> None:
    """Run QE pw.x for each displacement.

    Parameters
    ----------
    work_dir : Path
        Directory containing DISP.supercell_template.in.* files.
    n_disp : int
        Number of displacements.
    pw_command : str
        Command to invoke pw.x.
    timeout : int
        Timeout per job in seconds.
    """
    work_dir = Path(work_dir)
    results_dir = work_dir / "results"
    results_dir.mkdir(exist_ok=True)

    for i in range(1, n_disp + 1):
        width = len(str(n_disp))
        inp_file = work_dir / f"DISP.supercell_template.in.{i:0{width}d}"
        out_file = work_dir / f"DISP.supercell_template.out.{i:0{width}d}"

        if out_file.exists():
            with open(out_file) as f:
                if "JOB DONE" in f.read():
                    print(f"  Skipping disp {i}/{n_disp} (done)")
                    continue

        print(f"  Running disp {i}/{n_disp}...")
        result = subprocess.run(
            [pw_command, "-in", str(inp_file)],
            capture_output=True, text=True,
            cwd=str(work_dir),
            timeout=timeout,
        )
        with open(out_file, "w") as f:
            f.write(result.stdout)

        if "JOB DONE" not in result.stdout:
            print(f"  ERROR: disp {i} did not converge!")
            if result.stderr:
                print(f"  stderr: {result.stderr[-300:]}")
            raise RuntimeError(f"QE displacement {i} failed")

        for line in result.stdout.split("\n"):
            if "WALL" in line and "PWSCF" in line:
                print(f"  Done: {line.strip()}")
                break

    print(f"\nAll {n_disp} displacements completed.")


def reap(
    work_dir: Path,
    n_disp: int,
    supercell: tuple[int, int, int] = (2, 2, 2),
    cutoff: str = "-3",
    thirdorder_cmd: str = "thirdorder_espresso.py",
) -> Path:
    """Run thirdorder reap to produce FORCE_CONSTANTS_3RD.

    Parameters
    ----------
    work_dir : Path
        Directory containing unitcell.in and output files.
    n_disp : int
        Number of displacements.
    supercell : tuple of int
        Supercell dimensions (na, nb, nc).
    cutoff : str
        Cutoff matching the sow call.
    thirdorder_cmd : str
        Command for thirdorder_espresso.py.

    Returns
    -------
    fc3_path : Path
        Path to the generated FORCE_CONSTANTS_3RD file.
    """
    work_dir = Path(work_dir)
    width = len(str(n_disp))

    file_list = []
    for i in range(1, n_disp + 1):
        out_file = work_dir / f"DISP.supercell_template.out.{i:0{width}d}"
        if not out_file.exists():
            raise FileNotFoundError(f"Missing output: {out_file}")
        file_list.append(str(out_file))

    stdin_text = "\n".join(file_list) + "\n"
    na, nb, nc = supercell

    print(f"Reaping {n_disp} displacement outputs...")
    result = subprocess.run(
        [thirdorder_cmd, "unitcell.in", "reap",
         str(na), str(nb), str(nc), cutoff],
        input=stdin_text, text=True,
        capture_output=True,
        cwd=str(work_dir),
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Reap failed:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError("thirdorder reap failed")

    fc3_path = work_dir / "FORCE_CONSTANTS_3RD"
    if fc3_path.exists():
        print(f"FORCE_CONSTANTS_3RD written ({fc3_path.stat().st_size} bytes)")
    else:
        raise RuntimeError("FORCE_CONSTANTS_3RD not created")

    return fc3_path


def generate_fc3(
    cell: PhonopyAtoms,
    work_dir: Path,
    qe_config: QEConfig,
    supercell: tuple[int, int, int] = (2, 2, 2),
    cutoff: str = "-3",
    thirdorder_cmd: str = "thirdorder_espresso.py",
    skip_existing: bool = True,
) -> Path:
    """Full FC3 pipeline: sow + run + reap.

    Parameters
    ----------
    cell : PhonopyAtoms
        Primitive unit cell.
    work_dir : Path
        Working directory.
    qe_config : QEConfig
        QE parameters.
    supercell : tuple of int
        Supercell dimensions.
    cutoff : str
        Neighbor cutoff for thirdorder.
    thirdorder_cmd : str
        Command for thirdorder_espresso.py.
    skip_existing : bool
        Skip QE jobs that already completed.

    Returns
    -------
    fc3_path : Path
        Path to FORCE_CONSTANTS_3RD.
    """
    n_disp = sow(cell, work_dir, qe_config, supercell, cutoff, thirdorder_cmd)
    run_displacements(work_dir, n_disp, qe_config.pw_command)
    return reap(work_dir, n_disp, supercell, cutoff, thirdorder_cmd)
