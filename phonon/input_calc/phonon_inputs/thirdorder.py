"""Force constant generation via phono3py + symfc + DFT (QE or VASP).

Replaces the previous thirdorder_espresso.py workflow with phono3py for
displacement generation and symfc for efficient FC3 (and optionally FC2)
production.

Supported DFT calculators:
  - "qe":   Quantum ESPRESSO pw.x
  - "vasp": VASP (vasp_std / vasp_gam / ...)

Workflow:
    1. sow:  Create phono3py object, generate displacements, write DFT inputs
    2. run:  Execute DFT for each displaced supercell
    3. reap: Read forces from DFT outputs, produce FC3 via symfc
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
from .config import QEConfig, VASPConfig


# ========================================================================
# QE input / output
# ========================================================================


def _write_qe_input(
    path: Path,
    cell: np.ndarray,
    symbols: list[str],
    positions_cart: np.ndarray,
    qe_config: QEConfig,
    prefix: str = "fc3_disp",
) -> None:
    """Write a QE pw.x SCF input file."""
    unique_species = list(dict.fromkeys(symbols))
    natoms = len(symbols)
    ecutrho = qe_config.ecutwfc * qe_config.ecutrho_factor

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


def _parse_qe_forces(output_path: Path, n_atoms: int) -> np.ndarray:
    """Parse forces from QE output file.

    QE outputs forces in Ry/bohr. Converts to eV/Angstrom.
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


def _is_qe_done(out_file: Path) -> bool:
    """Check if a QE output file indicates a completed job."""
    if not out_file.exists():
        return False
    with open(out_file) as f:
        return "JOB DONE" in f.read()


# ========================================================================
# VASP input / output
# ========================================================================


def _write_vasp_inputs(
    disp_dir: Path,
    cell: np.ndarray,
    symbols: list[str],
    positions_cart: np.ndarray,
    vasp_config: VASPConfig,
) -> None:
    """Write VASP input files (POSCAR, INCAR, KPOINTS, POTCAR) into disp_dir.

    Parameters
    ----------
    disp_dir : Path
        Directory for this displacement (created if needed).
    cell : (3, 3) array
        Lattice vectors in Angstrom (rows).
    symbols : list of str
        Atomic symbols per atom.
    positions_cart : (N, 3) array
        Cartesian positions in Angstrom.
    vasp_config : VASPConfig
        VASP parameters.
    """
    disp_dir.mkdir(parents=True, exist_ok=True)

    # --- POSCAR ---
    # Group atoms by species, preserving order of first appearance
    unique_species = list(dict.fromkeys(symbols))
    species_counts = [symbols.count(sp) for sp in unique_species]
    # Reorder atoms: group by species
    sorted_indices = []
    for sp in unique_species:
        sorted_indices.extend(i for i, s in enumerate(symbols) if s == sp)
    sorted_positions = positions_cart[sorted_indices]

    # Convert to fractional coordinates and wrap into [0, 1)
    inv_cell = np.linalg.inv(cell)  # cell rows are lattice vectors
    sorted_frac = sorted_positions @ inv_cell.T
    sorted_frac -= np.floor(sorted_frac)  # wrap into [0, 1)

    with open(disp_dir / "POSCAR", "w") as f:
        f.write("phono3py displacement\n")
        f.write("1.0\n")
        for v in cell:
            f.write(f"  {v[0]:20.14f}  {v[1]:20.14f}  {v[2]:20.14f}\n")
        f.write("  " + "  ".join(unique_species) + "\n")
        f.write("  " + "  ".join(str(c) for c in species_counts) + "\n")
        f.write("Direct\n")
        for pos in sorted_frac:
            f.write(f"  {pos[0]:20.14f}  {pos[1]:20.14f}  {pos[2]:20.14f}\n")

    # Save the atom reordering so we can unsort the forces later
    np.savetxt(disp_dir / ".atom_order", sorted_indices, fmt="%d")

    # --- INCAR ---
    # Determine number of k-points for KPAR sanity check
    n_kpts = vasp_config.kpoints_scf[0] * vasp_config.kpoints_scf[1] * vasp_config.kpoints_scf[2]

    with open(disp_dir / "INCAR", "w") as f:
        f.write(f"PREC = {vasp_config.prec}\n")
        f.write(f"ENCUT = {vasp_config.encut}\n")
        f.write(f"EDIFF = {vasp_config.ediff}\n")
        f.write(f"ISMEAR = {vasp_config.ismear}\n")
        f.write(f"SIGMA = {vasp_config.sigma}\n")
        f.write(f"LREAL = {vasp_config.lreal}\n")
        f.write(f"LWAVE = .{'TRUE' if vasp_config.lwave else 'FALSE'}.\n")
        f.write(f"LCHARG = .{'TRUE' if vasp_config.lcharg else 'FALSE'}.\n")
        f.write("IBRION = -1\n")
        f.write("NSW = 0\n")
        f.write("ISYM = 0\n")
        # KPAR must not exceed the number of k-points
        if vasp_config.kpar is not None:
            kpar = min(vasp_config.kpar, n_kpts)
            f.write(f"KPAR = {kpar}\n")
        if vasp_config.ncore is not None:
            f.write(f"NCORE = {vasp_config.ncore}\n")

    # --- KPOINTS ---
    kpts = vasp_config.kpoints_scf
    with open(disp_dir / "KPOINTS", "w") as f:
        f.write("Automatic mesh\n")
        f.write("0\n")
        f.write("Gamma\n")
        f.write(f"  {kpts[0]} {kpts[1]} {kpts[2]}\n")
        f.write("  0 0 0\n")

    # --- POTCAR ---
    potcar_dir = Path(vasp_config.potcar_dir).resolve()
    potcar_path = disp_dir / "POTCAR"
    with open(potcar_path, "w") as potcar_out:
        for sp in unique_species:
            # Look up the POTCAR subdirectory for this species
            subdir = vasp_config.potcar_map.get(sp, sp)
            pp_path = potcar_dir / subdir / "POTCAR"
            if not pp_path.exists():
                # Try without subdirectory (flat layout)
                pp_path = potcar_dir / f"POTCAR.{sp}"
            if not pp_path.exists():
                raise FileNotFoundError(
                    f"POTCAR for {sp} not found. Tried:\n"
                    f"  {potcar_dir / subdir / 'POTCAR'}\n"
                    f"  {potcar_dir / f'POTCAR.{sp}'}\n"
                    f"Set vasp.potcar_dir and vasp.potcar_map in config."
                )
            with open(pp_path) as pp_in:
                potcar_out.write(pp_in.read())


def _parse_vasp_forces(disp_dir: Path, n_atoms: int) -> np.ndarray:
    """Parse forces from VASP vasprun.xml, restoring the original atom order.

    Returns forces in eV/Angstrom (VASP native units).
    """
    import xml.etree.ElementTree as ET

    vasprun = disp_dir / "vasprun.xml"
    if not vasprun.exists():
        raise FileNotFoundError(f"vasprun.xml not found in {disp_dir}")

    tree = ET.parse(vasprun)
    root = tree.getroot()

    # Find the last <varray name="forces"> block
    forces_elem = None
    for va in root.iter("varray"):
        if va.get("name") == "forces":
            forces_elem = va

    if forces_elem is None:
        raise ValueError(f"No forces found in {vasprun}")

    forces_sorted = []
    for v in forces_elem.findall("v"):
        forces_sorted.append([float(x) for x in v.text.split()])

    forces_sorted = np.array(forces_sorted)
    if len(forces_sorted) != n_atoms:
        raise ValueError(
            f"Expected {n_atoms} atoms in {vasprun}, got {len(forces_sorted)}"
        )

    # Unsort: POSCAR was written with atoms grouped by species
    order_file = disp_dir / ".atom_order"
    if order_file.exists():
        sorted_indices = np.loadtxt(order_file, dtype=int)
        forces = np.empty_like(forces_sorted)
        forces[sorted_indices] = forces_sorted
    else:
        forces = forces_sorted

    return forces


def _is_vasp_done(disp_dir: Path) -> bool:
    """Check if VASP completed successfully in disp_dir."""
    outcar = disp_dir / "OUTCAR"
    if not outcar.exists():
        return False
    # Check for convergence marker in OUTCAR (read tail)
    with open(outcar, "rb") as f:
        try:
            f.seek(-2000, 2)
        except OSError:
            f.seek(0)
        tail = f.read().decode(errors="replace")
    return "Total CPU time used" in tail


# ========================================================================
# Calculator-agnostic workflow
# ========================================================================


def sow(
    cell: PhonopyAtoms,
    work_dir: Path,
    dft_config: QEConfig | VASPConfig,
    supercell: tuple[int, int, int] = (2, 2, 2),
    cutoff_pair_distance: float | None = None,
    distance: float = 0.03,
    is_plusminus: bool = True,
    calculator: str = "qe",
) -> int:
    """Generate phono3py displaced supercells and write DFT inputs.

    Parameters
    ----------
    cell : PhonopyAtoms
        Unit cell.
    work_dir : Path
        Working directory (created if needed).
    dft_config : QEConfig or VASPConfig
        DFT parameters.
    supercell : tuple of int
        Supercell dimensions.
    cutoff_pair_distance : float, optional
        Cutoff distance for FC3 atom pairs in Angstrom.
    distance : float
        Displacement amplitude in Angstrom.
    is_plusminus : bool
        Use +/- displacements for better accuracy.
    calculator : str
        "qe" or "vasp".

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

    ph3.save(work_dir / "phono3py_disp.yaml")

    supercells = ph3.supercells_with_displacements
    n_disp = len(supercells)
    sc_cell = ph3.supercell.cell

    print(f"Generated {n_disp} displacements (phono3py)")
    print(f"  Unit cell: {len(cell.symbols)} atoms")
    print(f"  Supercell: {supercell}, {len(ph3.supercell.symbols)} atoms")
    if cutoff_pair_distance is not None:
        print(f"  Pair cutoff: {cutoff_pair_distance:.1f} A")
    print(f"  Calculator: {calculator}")

    if calculator == "qe":
        assert isinstance(dft_config, QEConfig)
        # Resolve pseudo_dir relative to work_dir
        pseudo_dir_abs = Path(dft_config.pseudo_dir).resolve()
        try:
            pseudo_dir_rel = pseudo_dir_abs.relative_to(work_dir.resolve())
        except ValueError:
            pseudo_dir_rel = pseudo_dir_abs
        from dataclasses import replace
        qe_local = replace(dft_config, pseudo_dir=str(pseudo_dir_rel))

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

    elif calculator == "vasp":
        assert isinstance(dft_config, VASPConfig)
        for i, sc in enumerate(supercells):
            if sc is None:
                continue
            disp_dir = work_dir / f"disp-{i+1:05d}"
            _write_vasp_inputs(
                disp_dir, sc_cell, list(sc.symbols), sc.positions, dft_config,
            )
        print(f"Wrote {n_disp} VASP displacement directories in {work_dir}")

    else:
        raise ValueError(f"Unknown calculator: {calculator!r}. Use 'qe' or 'vasp'.")

    return n_disp


def _needs_shell(command: str) -> bool:
    """Check if command needs shell=True (env vars, pipes, redirects)."""
    return any(tok in command for tok in ("=", "|", ">", "<", ";", "&&"))


def _run_and_tee(cmd, log_path: Path, cwd: str, timeout: int,
                 use_shell: bool) -> int:
    """Run a command, streaming stdout/stderr to both terminal and log file.

    Returns the process exit code.
    """
    import select
    import time

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            shell=use_shell,
        )
        t0 = time.monotonic()
        fds = {proc.stdout.fileno(): ("stdout", proc.stdout, sys.stdout),
               proc.stderr.fileno(): ("stderr", proc.stderr, sys.stderr)}
        while fds:
            elapsed = time.monotonic() - t0
            if elapsed > timeout:
                proc.kill()
                raise subprocess.TimeoutExpired(cmd, timeout)
            remaining = max(0.1, timeout - elapsed)
            ready, _, _ = select.select(list(fds.keys()), [], [],
                                        min(remaining, 1.0))
            for fd in ready:
                label, pipe, tty = fds[fd]
                data = pipe.read1(8192) if hasattr(pipe, "read1") else pipe.read(8192)
                if not data:
                    fds.pop(fd)
                    continue
                text = data.decode(errors="replace")
                log_f.write(text)
                log_f.flush()
                tty.write(text)
                tty.flush()
        proc.wait()
    return proc.returncode


def run_displacements(
    work_dir: Path,
    dft_command: str = "pw.x",
    timeout: int = 3600,
    calculator: str = "qe",
) -> None:
    """Run DFT for each displacement.

    Output is streamed live to the terminal and saved to a log file.

    Parameters
    ----------
    work_dir : Path
        Directory containing displacement inputs.
    dft_command : str
        Command to invoke.  Supports shell syntax including inline
        environment variables, e.g.
        ``"LD_LIBRARY_PATH=/opt/lib /path/to/vasp_std"``.
    timeout : int
        Timeout per job in seconds.
    calculator : str
        "qe" or "vasp".
    """
    work_dir = Path(work_dir)
    use_shell = _needs_shell(dft_command)

    if calculator == "qe":
        inp_files = sorted(work_dir.glob("disp-*.in"))
        n_disp = len(inp_files)
        if n_disp == 0:
            raise FileNotFoundError("No QE input files found. Run sow first.")

        print(f"Running {n_disp} QE displacements...")

        for i, inp_file in enumerate(inp_files):
            out_file = inp_file.with_suffix(".out")
            if _is_qe_done(out_file):
                print(f"  [{i + 1}/{n_disp}] Skipping (done)")
                continue

            print(f"  [{i + 1}/{n_disp}] Running {inp_file.name}...")
            if use_shell:
                cmd = f"{dft_command} -in {inp_file.name}"
            else:
                cmd = shlex.split(dft_command) + ["-in", inp_file.name]

            rc = _run_and_tee(cmd, out_file, str(work_dir), timeout, use_shell)

            if rc != 0 or not _is_qe_done(out_file):
                print(f"  ERROR: {inp_file.name} failed (exit {rc})")
                raise RuntimeError(f"QE displacement {inp_file.name} failed")

    elif calculator == "vasp":
        disp_dirs = sorted(work_dir.glob("disp-*/"))
        disp_dirs = [d for d in disp_dirs if d.is_dir()]
        n_disp = len(disp_dirs)
        if n_disp == 0:
            raise FileNotFoundError("No VASP displacement dirs found. Run sow first.")

        print(f"Running {n_disp} VASP displacements...")

        for i, disp_dir in enumerate(disp_dirs):
            if _is_vasp_done(disp_dir):
                print(f"  [{i + 1}/{n_disp}] Skipping {disp_dir.name} (done)")
                continue

            print(f"  [{i + 1}/{n_disp}] Running {disp_dir.name}...")
            if use_shell:
                cmd = dft_command
            else:
                cmd = shlex.split(dft_command)

            log_file = disp_dir / "vasp.log"
            rc = _run_and_tee(cmd, log_file, str(disp_dir), timeout, use_shell)

            if rc != 0 or not _is_vasp_done(disp_dir):
                print(f"  ERROR: {disp_dir.name} failed (exit {rc})")
                raise RuntimeError(f"VASP displacement {disp_dir.name} failed")
    else:
        raise ValueError(f"Unknown calculator: {calculator!r}")

    print(f"\nAll {n_disp} displacements completed.")


def reap(
    work_dir: Path,
    fc_calculator: str = "symfc",
    symmetrize: bool = True,
    calculator: str = "qe",
) -> Path:
    """Read DFT forces and produce FC3 (and FC2) via phono3py.

    Parameters
    ----------
    work_dir : Path
        Directory containing phono3py_disp.yaml and DFT outputs.
    fc_calculator : str
        FC calculator backend: "symfc" (recommended) or None (ALM/default).
    symmetrize : bool
        Apply symmetrization to force constants.
    calculator : str
        "qe" or "vasp".

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

    ph3 = phono3py_load(
        phono3py_yaml=str(yaml_path),
        produce_fc=False,
        log_level=0,
    )

    n_super = len(ph3.supercell.symbols)
    n_disp = len(ph3.supercells_with_displacements)

    print(f"Reading forces from {n_disp} {calculator.upper()} outputs...")
    force_sets = []

    if calculator == "qe":
        for i in range(n_disp):
            out_file = work_dir / f"disp-{i+1:05d}.out"
            if not out_file.exists():
                raise FileNotFoundError(f"Missing: {out_file}")
            forces = _parse_qe_forces(out_file, n_super)
            force_sets.append(forces)

    elif calculator == "vasp":
        for i in range(n_disp):
            disp_dir = work_dir / f"disp-{i+1:05d}"
            if not disp_dir.exists():
                raise FileNotFoundError(f"Missing: {disp_dir}")
            forces = _parse_vasp_forces(disp_dir, n_super)
            force_sets.append(forces)
    else:
        raise ValueError(f"Unknown calculator: {calculator!r}")

    ph3.forces = np.array(force_sets)
    print(f"  Loaded forces: shape {ph3.forces.shape}")

    print(f"Producing FC3 via {fc_calculator or 'default'}...")
    ph3.produce_fc3(
        symmetrize_fc3r=symmetrize,
        fc_calculator=fc_calculator,
    )

    print("Producing FC2...")
    ph3.produce_fc2(
        symmetrize_fc2=symmetrize,
        fc_calculator=fc_calculator,
    )

    fc3 = ph3.fc3
    fc2 = ph3.fc2
    print(f"  FC3 shape: {fc3.shape}, max: {np.max(np.abs(fc3)):.4e} eV/A^3")
    print(f"  FC2 shape: {fc2.shape}, max: {np.max(np.abs(fc2)):.4e} eV/A^2")

    fc3_path = work_dir / "fc3.hdf5"
    import h5py
    with h5py.File(fc3_path, "w") as f:
        f.create_dataset("fc3", data=fc3, compression="gzip")
        f.create_dataset("fc2", data=fc2, compression="gzip")
    print(f"  Saved: {fc3_path} ({fc3_path.stat().st_size / 1e6:.1f} MB)")

    ph3.save(work_dir / "phono3py_params.yaml")

    return fc3_path


def generate_fc3(
    cell: PhonopyAtoms,
    work_dir: Path,
    dft_config: QEConfig | VASPConfig,
    supercell: tuple[int, int, int] = (2, 2, 2),
    cutoff_pair_distance: float | None = None,
    distance: float = 0.03,
    fc_calculator: str = "symfc",
    skip_existing: bool = True,
    calculator: str = "qe",
) -> Path:
    """Full FC3 pipeline: sow + run + reap.

    Parameters
    ----------
    cell : PhonopyAtoms
        Unit cell.
    work_dir : Path
        Working directory.
    dft_config : QEConfig or VASPConfig
        DFT parameters.
    supercell : tuple of int
        Supercell dimensions.
    cutoff_pair_distance : float, optional
        Cutoff for FC3 atom pairs (Angstrom).
    distance : float
        Displacement amplitude.
    fc_calculator : str
        "symfc" for symmetry-adapted FC fitting.
    skip_existing : bool
        Skip completed DFT jobs.
    calculator : str
        "qe" or "vasp".

    Returns
    -------
    fc3_path : Path
        Path to fc3.hdf5.
    """
    dft_command = (dft_config.pw_command if calculator == "qe"
                   else dft_config.vasp_command)
    sow(cell, work_dir, dft_config, supercell, cutoff_pair_distance, distance,
        calculator=calculator)
    run_displacements(work_dir, dft_command, calculator=calculator)
    return reap(work_dir, fc_calculator=fc_calculator, calculator=calculator)
