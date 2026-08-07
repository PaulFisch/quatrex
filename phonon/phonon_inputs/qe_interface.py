"""DFT input/output interface (QE and VASP)."""

import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from .config import QEConfig, VASPConfig
from .thirdorder import _run_and_tee, _needs_shell

# Ry/bohr -> eV/Angstrom
RY_BOHR_TO_EV_ANG = 25.71104309541616

BOHR_TO_ANG = 0.52917721067


def write_qe_scf_input(
    filepath: Path,
    supercell: PhonopyAtoms,
    qe_config: QEConfig,
    calculation: str = "scf",
    forc_conv_thr: float | None = None,
    press_conv_thr: float | None = None,
    prefix: str = "phonon_fc",
    relative_paths: bool = False,
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
        QE calculation type ("scf", "relax", "vc-relax").
    forc_conv_thr : float, optional
        Force convergence threshold for relax/vc-relax (Ry/bohr).
    press_conv_thr : float, optional
        Pressure convergence threshold for vc-relax (kbar).
    """
    filepath = Path(filepath)
    symbols = supercell.symbols
    unique_species = list(dict.fromkeys(symbols))
    cell = supercell.cell
    positions = supercell.scaled_positions
    natoms = len(symbols)

    masses = {}
    for sym, mass in zip(symbols, supercell.masses):
        masses[sym] = mass

    ecutrho = qe_config.ecutwfc * qe_config.ecutrho_factor
    is_relax = calculation in ("relax", "vc-relax")
    kpts = qe_config.kpoints_relax if is_relax else qe_config.kpoints_scf

    if relative_paths:
        pseudo_dir = qe_config.pseudo_dir
        outdir = "./results"
    else:
        pseudo_dir = Path(qe_config.pseudo_dir).absolute()
        outdir = (filepath.parent / "results").absolute()

    with open(filepath, "w") as f:
        f.write("&CONTROL\n")
        f.write(f"   calculation      = '{calculation}'\n")
        f.write("   restart_mode     = 'from_scratch'\n")
        f.write(f"   prefix           = '{prefix}'\n")
        f.write(f"   pseudo_dir       = '{pseudo_dir}'\n")
        f.write(f"   outdir           = '{outdir}'\n")
        f.write("   tprnfor          = .true.\n")
        if calculation == "vc-relax":
            f.write("   tstress          = .true.\n")
        if forc_conv_thr is not None:
            f.write(f"   forc_conv_thr    = {forc_conv_thr}\n")
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

        if is_relax:
            f.write("&IONS\n")
            f.write("   ion_dynamics     = 'bfgs'\n")
            f.write("/\n")
        if calculation == "vc-relax":
            f.write("&CELL\n")
            if press_conv_thr is not None:
                f.write(f"   press_conv_thr   = {press_conv_thr}\n")
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


# ---------------------------------------------------------------------------
# Structural relaxation
# ---------------------------------------------------------------------------


def parse_qe_relax_output(filepath: Path) -> PhonopyAtoms:
    """Parse relaxed structure from a QE relax/vc-relax output.

    For vc-relax, reads from the "Begin final coordinates" block.
    For relax (fixed cell), reads final ATOMIC_POSITIONS and the
    input CELL_PARAMETERS.

    Parameters
    ----------
    filepath : Path
        QE output file.

    Returns
    -------
    cell : PhonopyAtoms
        Relaxed structure.
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        text = f.read()

    if "JOB DONE" not in text:
        raise RuntimeError(f"QE did not finish successfully: {filepath}")

    # --- Try vc-relax: "Begin final coordinates" block ---
    m_final = re.search(
        r"Begin final coordinates(.*?)End final coordinates",
        text, re.DOTALL,
    )

    if m_final:
        block = m_final.group(1)
    else:
        # Fixed-cell relax: use the full output text
        block = text

    # Parse CELL_PARAMETERS (last occurrence in block)
    cell_matches = list(re.finditer(
        r"CELL_PARAMETERS\s*\(?\s*(\w+)\s*\)?\s*\n"
        r"((?:\s*[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s*\n){3})",
        block,
    ))
    if not cell_matches:
        raise ValueError(f"Could not parse CELL_PARAMETERS in {filepath}")

    m_cell = cell_matches[-1]
    cell_unit = m_cell.group(1).lower()
    cell_lines = m_cell.group(2).strip().split("\n")
    cell = np.array([[float(x) for x in line.split()] for line in cell_lines])
    if cell_unit == "bohr":
        cell *= BOHR_TO_ANG

    # Parse ATOMIC_POSITIONS (last occurrence in block)
    pos_matches = list(re.finditer(
        r"ATOMIC_POSITIONS\s*\(?\s*(\w+)\s*\)?\s*\n"
        r"((?:\s*\w+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s*\n)+)",
        block,
    ))
    if not pos_matches:
        raise ValueError(f"Could not parse ATOMIC_POSITIONS in {filepath}")

    m_pos = pos_matches[-1]
    pos_unit = m_pos.group(1).lower()
    symbols = []
    positions = []
    for line in m_pos.group(2).strip().split("\n"):
        parts = line.split()
        symbols.append(parts[0])
        positions.append([float(x) for x in parts[1:4]])
    positions = np.array(positions)

    if pos_unit == "crystal":
        return PhonopyAtoms(
            symbols=symbols, cell=cell, scaled_positions=positions,
        )
    if pos_unit == "bohr":
        positions *= BOHR_TO_ANG
    # angstrom or converted bohr -> Cartesian positions
    inv_cell = np.linalg.inv(cell)
    scaled = positions @ inv_cell
    return PhonopyAtoms(symbols=symbols, cell=cell, scaled_positions=scaled)


def run_qe_relax(
    cell: PhonopyAtoms,
    work_dir: Path,
    qe_config: QEConfig,
    calculation: str = "vc-relax",
    forc_conv_thr: float = 1e-4,
    press_conv_thr: float = 0.5,
    skip_existing: bool = True,
) -> PhonopyAtoms:
    """Run a QE structural relaxation and return the relaxed structure.

    Parameters
    ----------
    cell : PhonopyAtoms
        Initial structure.
    work_dir : Path
        Working directory for relax files.
    qe_config : QEConfig
        QE parameters. Uses ``kpoints_relax`` for the k-mesh.
    calculation : str
        "relax" (ions only) or "vc-relax" (ions + cell).
    forc_conv_thr : float
        Force convergence threshold (Ry/bohr).
    press_conv_thr : float
        Pressure convergence threshold (kbar), for vc-relax.
    skip_existing : bool
        If True and output already contains "JOB DONE", skip re-running.

    Returns
    -------
    relaxed : PhonopyAtoms
        Relaxed structure.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "results").mkdir(exist_ok=True)

    inp = work_dir / "relax.in"
    out = work_dir / "relax.out"

    if skip_existing and out.exists():
        with open(out) as f:
            if "JOB DONE" in f.read():
                print(f"  Relax already done, loading {out}")
                return parse_qe_relax_output(out)

    write_qe_scf_input(
        inp, cell, qe_config, calculation=calculation,
        forc_conv_thr=forc_conv_thr,
        press_conv_thr=press_conv_thr if calculation == "vc-relax" else None,
    )
    print(f"  Running {calculation}...")
    run_qe(inp, out, qe_config.pw_command)

    relaxed = parse_qe_relax_output(out)
    a_new = np.linalg.norm(relaxed.cell, axis=1)
    print(f"  Relaxed lattice vectors: |a|={a_new[0]:.4f}, "
          f"|b|={a_new[1]:.4f}, |c|={a_new[2]:.4f} A")
    return relaxed


# ---------------------------------------------------------------------------
# VASP relaxation
# ---------------------------------------------------------------------------


def _write_vasp_relax_inputs(
    work_dir: Path,
    cell: PhonopyAtoms,
    vasp_config: VASPConfig,
    calculation: str = "vc-relax",
    forc_conv_thr: float = 1e-4,
) -> None:
    """Write VASP input files for a structural relaxation.

    Parameters
    ----------
    work_dir : Path
        Working directory (created if needed).
    cell : PhonopyAtoms
        Initial structure.
    vasp_config : VASPConfig
        VASP parameters.
    calculation : str
        "relax" (ions only, ISIF=2) or "vc-relax" (ions+cell, ISIF=3).
    forc_conv_thr : float
        Force convergence in eV/Angstrom (mapped to negative EDIFFG).
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    symbols = list(cell.symbols)
    lattice = cell.cell
    positions_frac = cell.scaled_positions

    unique_species = list(dict.fromkeys(symbols))
    species_counts = [symbols.count(sp) for sp in unique_species]
    sorted_indices = _species_sort_indices(symbols)
    sorted_positions = positions_frac[sorted_indices]

    # --- POSCAR ---
    with open(work_dir / "POSCAR", "w") as f:
        f.write("relaxation\n")
        f.write("1.0\n")
        for v in lattice:
            f.write(f"  {v[0]:20.14f}  {v[1]:20.14f}  {v[2]:20.14f}\n")
        f.write("  " + "  ".join(unique_species) + "\n")
        f.write("  " + "  ".join(str(c) for c in species_counts) + "\n")
        f.write("Direct\n")
        for pos in sorted_positions:
            f.write(f"  {pos[0]:20.14f}  {pos[1]:20.14f}  {pos[2]:20.14f}\n")

    # --- INCAR ---
    isif = 3 if calculation == "vc-relax" else 2
    with open(work_dir / "INCAR", "w") as f:
        f.write(f"PREC = {vasp_config.prec}\n")
        f.write(f"ENCUT = {vasp_config.encut}\n")
        f.write(f"EDIFF = {vasp_config.ediff}\n")
        f.write(f"EDIFFG = {-abs(forc_conv_thr)}\n")
        f.write(f"ISMEAR = {vasp_config.ismear}\n")
        f.write(f"SIGMA = {vasp_config.sigma}\n")
        f.write(f"LREAL = .FALSE.\n")  # always reciprocal for small relax cells
        f.write(f"LWAVE = .FALSE.\n")
        f.write(f"LCHARG = .FALSE.\n")
        f.write("IBRION = 2\n")      # CG relaxation
        f.write(f"NSW = {vasp_config.nsw}\n")  # max ionic steps
        f.write(f"ISIF = {isif}\n")   # 2=ions, 3=ions+cell
        f.write("ISYM = 2\n")         # symmetry on for relaxation
        if vasp_config.ncore is not None:
            f.write(f"NCORE = {vasp_config.ncore}\n")
        if vasp_config.kpar is not None:
            f.write(f"KPAR = {vasp_config.kpar}\n")
        for key, val in getattr(vasp_config, "incar_extra", {}).items():
            f.write(f"{key} = {val}\n")

    # --- KPOINTS (use denser mesh for relaxation) ---
    kpts = vasp_config.kpoints_scf
    with open(work_dir / "KPOINTS", "w") as f:
        f.write("Automatic mesh\n")
        f.write("0\n")
        f.write("Gamma\n")
        f.write(f"  {kpts[0]} {kpts[1]} {kpts[2]}\n")
        f.write("  0 0 0\n")

    # --- POTCAR ---
    from .thirdorder import _write_vasp_inputs
    # Reuse the POTCAR assembly logic
    potcar_dir = Path(vasp_config.potcar_dir).resolve()
    potcar_path = work_dir / "POTCAR"
    with open(potcar_path, "w") as potcar_out:
        for sp in unique_species:
            subdir = vasp_config.potcar_map.get(sp, sp)
            pp_path = potcar_dir / subdir / "POTCAR"
            if not pp_path.exists():
                pp_path = potcar_dir / f"POTCAR.{sp}"
            if not pp_path.exists():
                raise FileNotFoundError(
                    f"POTCAR for {sp} not found at {potcar_dir / subdir / 'POTCAR'}"
                )
            with open(pp_path) as pp_in:
                potcar_out.write(pp_in.read())


def _species_sort_indices(symbols: list) -> list:
    """POSCAR species-grouping permutation: original index of each atom
    in write order (species blocks in first-appearance order)."""
    unique_species = list(dict.fromkeys(symbols))
    idx: list = []
    for sp in unique_species:
        idx.extend(i for i, s in enumerate(symbols) if s == sp)
    return idx


def restore_original_order(
    parsed: PhonopyAtoms, original_cell: PhonopyAtoms
) -> PhonopyAtoms:
    """Invert the POSCAR species sorting of a parsed CONTCAR structure.

    The POSCAR writer groups atoms by species, so CONTCAR atom order
    differs from the original cell's whenever species interleave (e.g.
    wire Si/H followed by shell Si/O: CONTCAR's first block is ALL Si,
    wire and shell mixed). Any positional diagnostic against the
    original cell must run in the original order. No-op (with a
    warning) if the symbol multisets do not match.
    """
    import warnings

    symbols = list(original_cell.symbols)
    if sorted(symbols) != sorted(list(parsed.symbols)):
        warnings.warn(
            "restore_original_order: symbol sets differ between parsed "
            "CONTCAR and original cell; returning CONTCAR order.",
            stacklevel=2)
        return parsed
    idx = _species_sort_indices(symbols)
    scaled = np.empty_like(parsed.scaled_positions)
    scaled[idx] = parsed.scaled_positions
    return PhonopyAtoms(symbols=symbols, cell=parsed.cell,
                        scaled_positions=scaled)


def _relax_converged(work_dir: Path) -> bool:
    """True if the OUTCAR records IONIC convergence (EDIFFG met).

    ``_is_vasp_done`` only certifies a graceful VASP exit -- a relax
    that runs out of NSW ionic steps exits gracefully too (measured:
    oxrelax4 'completed' at step 300 with dE still -4e-3 eV/step).
    """
    outcar = Path(work_dir) / "OUTCAR"
    if not outcar.exists():
        return False
    with open(outcar, "rb") as f:
        try:
            f.seek(-500_000, 2)
        except OSError:
            f.seek(0)
        tail = f.read().decode(errors="replace")
    return "reached required accuracy" in tail


def parse_vasp_relax_output(work_dir: Path) -> PhonopyAtoms:
    """Parse relaxed structure from VASP CONTCAR.

    Parameters
    ----------
    work_dir : Path
        Directory containing VASP output files.

    Returns
    -------
    cell : PhonopyAtoms
        Relaxed structure.
    """
    contcar = work_dir / "CONTCAR"
    if not contcar.exists():
        raise FileNotFoundError(f"CONTCAR not found in {work_dir}")

    with open(contcar) as f:
        lines = f.readlines()

    scale = float(lines[1].strip())
    lattice = np.array([
        [float(x) for x in lines[i].split()] for i in range(2, 5)
    ]) * scale

    species_line = lines[5].split()
    counts_line = [int(x) for x in lines[6].split()]

    symbols = []
    for sp, count in zip(species_line, counts_line):
        symbols.extend([sp] * count)

    coord_type = lines[7].strip()[0].lower()
    natoms = sum(counts_line)
    positions = np.array([
        [float(x) for x in lines[8 + i].split()[:3]] for i in range(natoms)
    ])

    if coord_type == "d":  # Direct/fractional
        return PhonopyAtoms(symbols=symbols, cell=lattice,
                            scaled_positions=positions)
    else:  # Cartesian
        positions *= scale
        inv_cell = np.linalg.inv(lattice)
        scaled = positions @ inv_cell
        return PhonopyAtoms(symbols=symbols, cell=lattice,
                            scaled_positions=scaled)


def _maybe_restart_cell(
    work_dir: Path, cell: PhonopyAtoms, restart_from_contcar: bool
) -> PhonopyAtoms:
    """Seed a relaxation leg from the work dir's CONTCAR if requested.

    Long relaxations run in timeout-bounded legs; without this, every
    leg rewrites POSCAR from the ORIGINAL cell and re-pays the whole
    descent (measured on oxrelax4: leg 4 restarted at F = -997.17 eV,
    the unrelaxed energy, and spent its 48 h regaining -1033.9).
    Only consulted for INCOMPLETE runs -- the completed-run skip in
    :func:`run_vasp_relax` short-circuits before this.
    """
    if not restart_from_contcar:
        return cell
    contcar = Path(work_dir) / "CONTCAR"
    if contcar.exists() and contcar.stat().st_size > 0:
        seeded = restore_original_order(
            parse_vasp_relax_output(Path(work_dir)), cell)
        print(f"  Restarting from {contcar} "
              f"({len(seeded.symbols)} atoms, prior unconverged leg)")
        return seeded
    return cell


def run_vasp_relax(
    cell: PhonopyAtoms,
    work_dir: Path,
    vasp_config: VASPConfig,
    calculation: str = "vc-relax",
    forc_conv_thr: float = 0.01,
    press_conv_thr: float = 0.5,
    skip_existing: bool = True,
    timeout: int = 7200,
    restart_from_contcar: bool = False,
) -> PhonopyAtoms:
    """Run a VASP structural relaxation and return the relaxed structure.

    Parameters
    ----------
    cell : PhonopyAtoms
        Initial structure.
    work_dir : Path
        Working directory.
    vasp_config : VASPConfig
        VASP parameters.
    calculation : str
        "relax" or "vc-relax".
    forc_conv_thr : float
        Force convergence in eV/Angstrom (EDIFFG = -forc_conv_thr).
    press_conv_thr : float
        Not directly used by VASP (stress converges via EDIFFG + ISIF).
    skip_existing : bool
        If True and CONTCAR exists from a completed run, skip re-running.
    restart_from_contcar : bool
        If True and the work dir holds a CONTCAR from an INCOMPLETE
        prior leg (timeout-bounded relaxations), seed POSCAR from it
        instead of the passed ``cell``. Default False (legacy: every
        leg restarts from the original structure).

    Returns
    -------
    relaxed : PhonopyAtoms
        Relaxed structure.
    """
    from .thirdorder import _is_vasp_done

    work_dir = Path(work_dir)

    if skip_existing and _is_vasp_done(work_dir) and _relax_converged(work_dir):
        contcar = work_dir / "CONTCAR"
        if contcar.exists() and contcar.stat().st_size > 0:
            print(f"  Relax already done, loading {contcar}")
            relaxed = restore_original_order(
                parse_vasp_relax_output(work_dir), cell)
            a_new = np.linalg.norm(relaxed.cell, axis=1)
            print(f"  Relaxed: {len(relaxed.symbols)} atoms")
            for i, v in enumerate(relaxed.cell):
                print(f"    a{i+1} = [{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}] "
                      f"(|a{i+1}| = {np.linalg.norm(v):.4f} A)")
            return relaxed

    cell = _maybe_restart_cell(work_dir, cell, restart_from_contcar)
    _write_vasp_relax_inputs(work_dir, cell, vasp_config, calculation,
                             forc_conv_thr)

    print(f"  Running VASP {calculation}...")
    use_shell = _needs_shell(vasp_config.vasp_command)
    cmd = vasp_config.vasp_command if use_shell else vasp_config.vasp_command.split()
    log_file = work_dir / "vasp_relax.log"
    import sys
    rc = _run_and_tee(cmd, log_file, str(work_dir), timeout=timeout,
                      use_shell=use_shell)

    if rc != 0 or not _is_vasp_done(work_dir):
        raise RuntimeError(f"VASP relaxation failed (exit {rc}). "
                           f"Check {log_file}")

    relaxed = restore_original_order(parse_vasp_relax_output(work_dir), cell)
    a_new = np.linalg.norm(relaxed.cell, axis=1)
    print(f"  Relaxed lattice vectors: |a|={a_new[0]:.4f}, "
          f"|b|={a_new[1]:.4f}, |c|={a_new[2]:.4f} A")
    return relaxed
