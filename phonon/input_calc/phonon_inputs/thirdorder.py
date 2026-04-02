"""Force constant generation via phono3py + symfc + QE.

Replaces the previous thirdorder_espresso.py workflow with phono3py for
displacement generation and symfc for efficient FC3 (and optionally FC2)
production.

Workflow:
    1. sow:  Create phono3py object, generate displacements, write QE inputs
    2. run:  Execute QE pw.x for each displaced supercell
    3. reap: Read forces from QE outputs, produce FC3 via symfc
    4. save: Write fc3.hdf5 and phono3py_disp.yaml

Usage via CLI:
    python -m phonon_inputs fc3-sow  --config config.yaml
    python -m phonon_inputs fc3-run  --config config.yaml
    python -m phonon_inputs fc3-reap --config config.yaml
    python -m phonon_inputs fc3-all  --config config.yaml
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms

import shlex
from .config import QEConfig


def _write_qe_input(
    path: Path,
    cell: np.ndarray,
    symbols: list[str],
    positions_cart: np.ndarray,
    qe_config: QEConfig,
    prefix: str = "fc3_disp",
) -> None:
    """Write a QE pw.x SCF input file.

    Parameters
    ----------
    path : Path
        Output file path.
    cell : (3, 3) array
        Lattice vectors in Angstrom (rows).
    symbols : list of str
        Atomic symbols.
    positions_cart : (N, 3) array
        Cartesian positions in Angstrom.
    qe_config : QEConfig
        QE parameters.
    prefix : str
        QE prefix for output files.
    """
    unique_species = list(dict.fromkeys(symbols))
    natoms = len(symbols)
    ecutrho = qe_config.ecutwfc * qe_config.ecutrho_factor

    # Collect masses from PhonopyAtoms standard values
    from phonopy.structure.atoms import atom_data
    masses = {}
    for sym in unique_species:
        for entry in atom_data:
            if entry and entry[1] == sym:
                masses[sym] = entry[3]
                break

    with open(path, "w") as f:
        f.write("&CONTROL\n")
        f.write("   calculation      = 'scf'\n")
        f.write("   restart_mode     = 'from_scratch'\n")
        f.write(f"   prefix           = '{prefix}'\n")
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
            f.write(f"  {sp}  {masses.get(sp, 0.0):.4f}  {pseudo}\n")

        f.write("ATOMIC_POSITIONS angstrom\n")
        for sym, pos in zip(symbols, positions_cart):
            f.write(f"  {sym}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}\n")

        f.write("CELL_PARAMETERS angstrom\n")
        for v in cell:
            f.write(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}\n")

        kpts = qe_config.kpoints_scf
        f.write("K_POINTS automatic\n")
        f.write(f"  {kpts[0]} {kpts[1]} {kpts[2]}  0 0 0\n")


def sow(
    cell: PhonopyAtoms,
    work_dir: Path,
    qe_config: QEConfig,
    supercell: tuple[int, int, int] = (2, 2, 2),
    cutoff_pair_distance: float | None = None,
    distance: float = 0.03,
    is_plusminus: bool = True,
) -> int:
    """Generate phono3py displaced supercells and write QE inputs.

    Parameters
    ----------
    cell : PhonopyAtoms
        Unit cell.
    work_dir : Path
        Working directory (created if needed).
    qe_config : QEConfig
        QE parameters.
    supercell : tuple of int
        Supercell dimensions.
    cutoff_pair_distance : float, optional
        Cutoff distance for FC3 atom pairs in Angstrom.
        None = no cutoff (all pairs). Reduces displacement count.
    distance : float
        Displacement amplitude in Angstrom (default 0.03).
    is_plusminus : bool
        Use +/- displacements for better accuracy.

    Returns
    -------
    n_disp : int
        Number of displacement files generated.
    """
    from phono3py import Phono3py

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    sc_matrix = np.diag(supercell)
    ph3 = Phono3py(cell, supercell_matrix=sc_matrix)
    ph3.generate_displacements(
        distance=distance,
        cutoff_pair_distance=cutoff_pair_distance,
        is_plusminus=is_plusminus,
    )

    # Save phono3py state for reaping later
    ph3.save(work_dir / "phono3py_disp.yaml")

    supercells = ph3.supercells_with_displacements
    n_disp = len(supercells)
    sc_cell = ph3.supercell.cell

    print(f"Generated {n_disp} displacements (phono3py)")
    print(f"  Unit cell: {len(cell.symbols)} atoms")
    print(f"  Supercell: {supercell}, {len(ph3.supercell.symbols)} atoms")
    if cutoff_pair_distance is not None:
        print(f"  Pair cutoff: {cutoff_pair_distance:.1f} A")

    # Resolve pseudo_dir relative to work_dir
    pseudo_dir_abs = Path(qe_config.pseudo_dir).resolve()
    try:
        pseudo_dir_rel = pseudo_dir_abs.relative_to(work_dir.resolve())
    except ValueError:
        pseudo_dir_rel = pseudo_dir_abs
    # Create a modified config with resolved pseudo path
    from dataclasses import replace
    qe_local = replace(qe_config, pseudo_dir=str(pseudo_dir_rel))

    # Write QE inputs
    (work_dir / "results").mkdir(exist_ok=True)

    for i, sc in enumerate(supercells):
        if sc is None:
            continue
        inp_path = work_dir / f"disp-{i+1:05d}.in"
        _write_qe_input(
            inp_path, sc_cell, list(sc.symbols), sc.positions, qe_local,
            prefix=f"fc3_{i+1:05d}",
        )

    print(f"Wrote {n_disp} QE input files in {work_dir}")
    return n_disp


def run_displacements(
    work_dir: Path,
    pw_command: str = "pw.x",
    timeout: int = 3600,
    n_parallel: int = 1,
) -> None:
    """Run QE pw.x for each displacement.

    Parameters
    ----------
    work_dir : Path
        Directory containing disp-*.in files.
    pw_command : str
        Command to invoke pw.x.
    timeout : int
        Timeout per job in seconds.
    n_parallel : int
        Number of parallel QE jobs (future use, currently sequential).
    """
    work_dir = Path(work_dir)
    inp_files = sorted(work_dir.glob("disp-*.in"))
    n_disp = len(inp_files)

    if n_disp == 0:
        raise FileNotFoundError("No displacement input files found. Run sow first.")

    print(f"Running {n_disp} QE displacements...")

    for i, inp_file in enumerate(inp_files):
        out_file = inp_file.with_suffix(".out")

        if out_file.exists():
            with open(out_file) as f:
                if "JOB DONE" in f.read():
                    print(f"  [{i + 1}/{n_disp}] Skipping (done)")
                    continue

        print(f"  [{i + 1}/{n_disp}] Running {inp_file.name}...")

        cmd = shlex.split(pw_command) + ["-in", inp_file.name]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(work_dir),
            timeout=timeout,
        )

        with open(out_file, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\n===== STDERR =====\n")
                f.write(result.stderr)

        if result.returncode != 0 or "JOB DONE" not in result.stdout:
            print(f"  ERROR: {inp_file.name} failed")
            print(f"  Command: {' '.join(cmd)}")
            if result.stderr:
                print(f"  stderr: {result.stderr[-500:]}")
            raise RuntimeError(f"QE displacement {inp_file.name} failed")
    print(f"\nAll {n_disp} displacements completed.")


def _parse_qe_forces(output_path: Path, n_atoms: int) -> np.ndarray:
    """Parse forces from QE output file.

    QE outputs forces in Ry/bohr. Converts to eV/Angstrom.

    Parameters
    ----------
    output_path : Path
        QE output file.
    n_atoms : int
        Expected number of atoms.

    Returns
    -------
    forces : (n_atoms, 3) array in eV/Angstrom.
    """
    RY_BOHR_TO_EV_ANG = 25.71104309541616

    forces = []
    reading = False
    with open(output_path) as f:
        for line in f:
            if "Forces acting on atoms" in line:
                forces = []
                reading = True
                continue
            if reading and "force =" in line:
                parts = line.split("force =")[1].split()
                forces.append([float(x) for x in parts[:3]])
            if reading and len(forces) == n_atoms:
                reading = False

    if len(forces) != n_atoms:
        raise ValueError(
            f"Expected {n_atoms} force lines in {output_path}, got {len(forces)}"
        )

    return np.array(forces) * RY_BOHR_TO_EV_ANG


def reap(
    work_dir: Path,
    fc_calculator: str = "symfc",
    symmetrize: bool = True,
) -> Path:
    """Read QE forces and produce FC3 (and FC2) via phono3py.

    Parameters
    ----------
    work_dir : Path
        Directory containing phono3py_disp.yaml and disp-*.out files.
    fc_calculator : str
        FC calculator backend: "symfc" (recommended) or None (ALM/default).
    symmetrize : bool
        Apply symmetrization to force constants.

    Returns
    -------
    fc3_path : Path
        Path to the saved fc3.hdf5 file.
    """
    from phono3py import load as phono3py_load

    work_dir = Path(work_dir)
    yaml_path = work_dir / "phono3py_disp.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"phono3py_disp.yaml not found in {work_dir}")

    # Load the phono3py object (without producing FC yet)
    ph3 = phono3py_load(
        phono3py_yaml=str(yaml_path),
        produce_fc=False,
        log_level=0,
    )

    n_super = len(ph3.supercell.symbols)
    n_disp = len(ph3.supercells_with_displacements)

    # Read forces from QE outputs
    print(f"Reading forces from {n_disp} QE output files...")
    force_sets = []
    for i in range(n_disp):
        out_file = work_dir / f"disp-{i+1:05d}.out"
        if not out_file.exists():
            raise FileNotFoundError(f"Missing: {out_file}")
        forces = _parse_qe_forces(out_file, n_super)
        force_sets.append(forces)

    ph3.forces = np.array(force_sets)
    print(f"  Loaded forces: shape {ph3.forces.shape}")

    # Produce FC3 (and FC2)
    print(f"Producing FC3 via {fc_calculator or 'default'}...")
    ph3.produce_fc3(
        symmetrize_fc3r=symmetrize,
        fc_calculator=fc_calculator,
    )

    # Also produce FC2 for harmonic calculations
    print("Producing FC2...")
    ph3.produce_fc2(
        symmetrize_fc2=symmetrize,
        fc_calculator=fc_calculator,
    )

    fc3 = ph3.fc3
    fc2 = ph3.fc2
    print(f"  FC3 shape: {fc3.shape}, max: {np.max(np.abs(fc3)):.4e} eV/A^3")
    print(f"  FC2 shape: {fc2.shape}, max: {np.max(np.abs(fc2)):.4e} eV/A^2")

    # Save
    fc3_path = work_dir / "fc3.hdf5"
    import h5py
    with h5py.File(fc3_path, "w") as f:
        f.create_dataset("fc3", data=fc3, compression="gzip")
        f.create_dataset("fc2", data=fc2, compression="gzip")
    print(f"  Saved: {fc3_path} ({fc3_path.stat().st_size / 1e6:.1f} MB)")

    # Also save phono3py yaml with forces
    ph3.save(work_dir / "phono3py_params.yaml")

    return fc3_path


def generate_fc3(
    cell: PhonopyAtoms,
    work_dir: Path,
    qe_config: QEConfig,
    supercell: tuple[int, int, int] = (2, 2, 2),
    cutoff_pair_distance: float | None = None,
    distance: float = 0.03,
    fc_calculator: str = "symfc",
    skip_existing: bool = True,
) -> Path:
    """Full FC3 pipeline: sow + run + reap.

    Parameters
    ----------
    cell : PhonopyAtoms
        Unit cell.
    work_dir : Path
        Working directory.
    qe_config : QEConfig
        QE parameters.
    supercell : tuple of int
        Supercell dimensions.
    cutoff_pair_distance : float, optional
        Cutoff for FC3 atom pairs (Angstrom).
    distance : float
        Displacement amplitude.
    fc_calculator : str
        "symfc" for symmetry-adapted FC fitting.
    skip_existing : bool
        Skip completed QE jobs.

    Returns
    -------
    fc3_path : Path
        Path to fc3.hdf5.
    """
    sow(cell, work_dir, qe_config, supercell, cutoff_pair_distance, distance)
    run_displacements(work_dir, qe_config.pw_command)
    return reap(work_dir, fc_calculator=fc_calculator)
