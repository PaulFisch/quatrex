"""Force constant generation via phono3py + symfc + DFT (QE or VASP).

Replaces the previous thirdorder_espresso.py workflow with phono3py for
displacement generation and symfc for efficient FC3 (and optionally FC2)
production.

Supported DFT calculators:
  - "qe":   Quantum ESPRESSO pw.x
  - "vasp": VASP (vasp_std / vasp_gam / ...)

Workflow:
    1. sow:  Create phono3py object, generate displacements, write DFT inputs
             including a reference (undisplaced) supercell calculation
    2. run:  Execute reference DFT first if needed, then each displaced supercell
             while reusing previous charge density / wavefunctions
    3. reap: Read forces from DFT outputs, produce FC3 via symfc
    4. save: Write fc3.hdf5 and phono3py_disp.yaml

Behavior added in this version:
    - Completed calculations are skipped automatically
    - A reference supercell calculation is created and run first if needed
    - QE can reuse previous pot/wfc through shared outdir/prefix
    - VASP reuses previous WAVECAR/CHGCAR only when valid restart files exist
    - For VASP, existing displacement folders and existing input files are not
      overwritten during sow()
"""

import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms

from .config import QEConfig, VASPConfig


# ========================================================================
# Common paths / naming
# ========================================================================

QE_RESTART_PREFIX = "fc3_seed"
QE_RESULTS_DIRNAME = "results"
REFERENCE_QE_INPUT = "reference.in"
REFERENCE_QE_OUTPUT = "reference.out"
REFERENCE_VASP_DIRNAME = "reference"


# ========================================================================
# Small helpers
# ========================================================================


def _write_text_if_missing(path: Path, text: str) -> None:
    """Write text file only if it does not already exist."""
    if not path.exists():
        path.write_text(text)


def _copy_file_if_missing(src: Path, dst: Path) -> None:
    """Copy src to dst only if dst does not already exist."""
    if not dst.exists():
        shutil.copy2(src, dst)


def _copy_if_exists(src: Path, dst: Path) -> None:
    """Copy a file if it exists."""
    if src.exists():
        shutil.copy2(src, dst)


# ========================================================================
# QE input / output
# ========================================================================


def _write_qe_input(
    path: Path,
    cell: np.ndarray,
    symbols: list[str],
    positions_cart: np.ndarray,
    qe_config: QEConfig,
    prefix: str = QE_RESTART_PREFIX,
    use_restart_data: bool = False,
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

    lines = []
    lines.append("&CONTROL\n")
    lines.append("   calculation      = 'scf'\n")
    lines.append("   restart_mode     = 'from_scratch'\n")
    lines.append(f"   prefix           = '{prefix}'\n")
    lines.append(f"   pseudo_dir       = '{qe_config.pseudo_dir}'\n")
    lines.append(f"   outdir           = './{QE_RESULTS_DIRNAME}'\n")
    lines.append("   tprnfor          = .true.\n")
    lines.append("/\n")

    lines.append("&SYSTEM\n")
    lines.append("   ibrav            = 0\n")
    lines.append(f"   nat              = {natoms}\n")
    lines.append(f"   ntyp             = {len(unique_species)}\n")
    lines.append(f"   ecutwfc          = {qe_config.ecutwfc}\n")
    lines.append(f"   ecutrho          = {ecutrho}\n")
    lines.append("   occupations      = 'smearing'\n")
    lines.append(f"   smearing         = '{qe_config.smearing}'\n")
    lines.append(f"   degauss          = {qe_config.degauss}\n")
    lines.append("/\n")

    lines.append("&ELECTRONS\n")
    lines.append(f"   conv_thr         = {qe_config.conv_thr}\n")
    if use_restart_data:
        lines.append("   startingpot      = 'file'\n")
        lines.append("   startingwfc      = 'file'\n")
    lines.append("/\n")

    lines.append("ATOMIC_SPECIES\n")
    for sp in unique_species:
        pseudo = qe_config.pseudopotentials.get(sp, f"{sp}.UPF")
        lines.append(f"  {sp}  {masses.get(sp, 0.0):.4f}  {pseudo}\n")

    lines.append("ATOMIC_POSITIONS angstrom\n")
    for sym, pos in zip(symbols, positions_cart):
        lines.append(f"  {sym}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}\n")

    lines.append("CELL_PARAMETERS angstrom\n")
    for v in cell:
        lines.append(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}\n")

    kpts = qe_config.kpoints_scf
    lines.append("K_POINTS automatic\n")
    lines.append(f"  {kpts[0]} {kpts[1]} {kpts[2]}  0 0 0\n")

    path.write_text("".join(lines))


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


def _qe_restart_data_available(work_dir: Path) -> bool:
    """Return True if QE restart data seem to exist in the shared results dir."""
    results_dir = work_dir / QE_RESULTS_DIRNAME
    if not results_dir.exists():
        return False
    prefix_dir = results_dir / f"{QE_RESTART_PREFIX}.save"
    return prefix_dir.exists()


def _rewrite_qe_restart_flags(inp_path: Path, use_restart_data: bool) -> None:
    """Rewrite QE input file so startingpot/startingwfc match seed availability."""
    text = inp_path.read_text()

    text = re.sub(r"(?m)^\s*startingpot\s*=.*\n?", "", text)
    text = re.sub(r"(?m)^\s*startingwfc\s*=.*\n?", "", text)

    if use_restart_data:
        marker = "&ELECTRONS\n"
        replacement = (
            "&ELECTRONS\n"
            "   startingpot      = 'file'\n"
            "   startingwfc      = 'file'\n"
        )
        if marker not in text:
            raise ValueError(f"Could not find &ELECTRONS in {inp_path}")
        text = text.replace(marker, replacement, 1)

    inp_path.write_text(text)


# ========================================================================
# VASP input / output
# ========================================================================


def _write_vasp_inputs(
    disp_dir: Path,
    cell: np.ndarray,
    symbols: list[str],
    positions_cart: np.ndarray,
    vasp_config: VASPConfig,
    use_restart_data: bool = False,
    overwrite: bool = False,
) -> None:
    """Write VASP input files into disp_dir.

    If overwrite=False, existing files are preserved.
    """
    disp_dir.mkdir(parents=True, exist_ok=True)

    # --- POSCAR content ---
    unique_species = list(dict.fromkeys(symbols))
    species_counts = [symbols.count(sp) for sp in unique_species]

    sorted_indices = []
    for sp in unique_species:
        sorted_indices.extend(i for i, s in enumerate(symbols) if s == sp)
    sorted_positions = positions_cart[sorted_indices]

    inv_cell = np.linalg.inv(cell)
    sorted_frac = sorted_positions @ inv_cell.T
    sorted_frac %= 1.0
    sorted_frac[sorted_frac > 1.0 - 1e-10] = 0.0

    poscar_lines = []
    poscar_lines.append("phono3py displacement\n")
    poscar_lines.append("1.0\n")
    for v in cell:
        poscar_lines.append(f"  {v[0]:20.14f}  {v[1]:20.14f}  {v[2]:20.14f}\n")
    poscar_lines.append("  " + "  ".join(unique_species) + "\n")
    poscar_lines.append("  " + "  ".join(str(c) for c in species_counts) + "\n")
    poscar_lines.append("Direct\n")
    for pos in sorted_frac:
        poscar_lines.append(f"  {pos[0]:20.14f}  {pos[1]:20.14f}  {pos[2]:20.14f}\n")
    poscar_text = "".join(poscar_lines)

    atom_order_text = "\n".join(str(i) for i in sorted_indices) + "\n"

    # --- INCAR content ---
    n_kpts = (
        vasp_config.kpoints_scf[0]
        * vasp_config.kpoints_scf[1]
        * vasp_config.kpoints_scf[2]
    )

    incar_lines = []
    incar_lines.append(f"PREC = {vasp_config.prec}\n")
    incar_lines.append(f"ENCUT = {vasp_config.encut}\n")
    incar_lines.append(f"EDIFF = {vasp_config.ediff}\n")
    incar_lines.append(f"ISMEAR = {vasp_config.ismear}\n")
    incar_lines.append(f"SIGMA = {vasp_config.sigma}\n")
    incar_lines.append(f"LREAL = {vasp_config.lreal}\n")
    incar_lines.append(f"LWAVE = .{'TRUE' if vasp_config.lwave else 'FALSE'}.\n")
    incar_lines.append(f"LCHARG = .{'TRUE' if vasp_config.lcharg else 'FALSE'}.\n")
    incar_lines.append("IBRION = -1\n")
    incar_lines.append("NSW = 0\n")
    incar_lines.append("ISYM = 0\n")
    incar_lines.append(f"ISTART = {1 if use_restart_data else 0}\n")
    incar_lines.append(f"ICHARG = {1 if use_restart_data else 2}\n")
    if vasp_config.kpar is not None:
        kpar = min(vasp_config.kpar, n_kpts)
        incar_lines.append(f"KPAR = {kpar}\n")
    if vasp_config.ncore is not None:
        incar_lines.append(f"NCORE = {vasp_config.ncore}\n")
    incar_text = "".join(incar_lines)

    # --- KPOINTS content ---
    kpts = vasp_config.kpoints_scf
    kpoints_text = (
        "Automatic mesh\n"
        "0\n"
        "Gamma\n"
        f"  {kpts[0]} {kpts[1]} {kpts[2]}\n"
        "  0 0 0\n"
    )

    # --- write only if missing unless overwrite=True ---
    if overwrite:
        (disp_dir / "POSCAR").write_text(poscar_text)
        (disp_dir / ".atom_order").write_text(atom_order_text)
        (disp_dir / "INCAR").write_text(incar_text)
        (disp_dir / "KPOINTS").write_text(kpoints_text)
    else:
        _write_text_if_missing(disp_dir / "POSCAR", poscar_text)
        _write_text_if_missing(disp_dir / ".atom_order", atom_order_text)
        _write_text_if_missing(disp_dir / "INCAR", incar_text)
        _write_text_if_missing(disp_dir / "KPOINTS", kpoints_text)

    # --- POTCAR ---
    potcar_dir = Path(vasp_config.potcar_dir).resolve()
    potcar_path = disp_dir / "POTCAR"
    if not potcar_path.exists() or overwrite:
        with open(potcar_path, "w") as potcar_out:
            for sp in unique_species:
                subdir = vasp_config.potcar_map.get(sp, sp)
                pp_path = potcar_dir / subdir / "POTCAR"
                if not pp_path.exists():
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
    """Parse forces from VASP vasprun.xml, restoring the original atom order."""
    import xml.etree.ElementTree as ET

    vasprun = disp_dir / "vasprun.xml"
    if not vasprun.exists():
        raise FileNotFoundError(f"vasprun.xml not found in {disp_dir}")

    tree = ET.parse(vasprun)
    root = tree.getroot()

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
    vasprun = disp_dir / "vasprun.xml"

    if not outcar.exists() or not vasprun.exists():
        return False

    with open(outcar, "rb") as f:
        try:
            f.seek(-4000, 2)
        except OSError:
            f.seek(0)
        tail = f.read().decode(errors="replace")

    return "General timing and accounting informations for this job" in tail


def _vasp_restart_files_valid(seed_dir: Path) -> tuple[bool, bool]:
    """Return (has_valid_wavecar, has_valid_chgcar) for a seed directory."""
    wavecar = seed_dir / "WAVECAR"
    chgcar = seed_dir / "CHGCAR"

    has_wavecar = wavecar.exists() and wavecar.stat().st_size > 0
    has_chgcar = chgcar.exists() and chgcar.stat().st_size > 0

    return has_wavecar, has_chgcar


def _prepare_vasp_restart(target_dir: Path, seed_dir: Path | None) -> None:
    """Seed a VASP run from a previous completed calculation if valid."""
    for fname in ("WAVECAR", "CHGCAR"):
        target = target_dir / fname
        if target.exists():
            target.unlink()

    has_wavecar = False
    has_chgcar = False

    if seed_dir is not None:
        src_wavecar = seed_dir / "WAVECAR"
        src_chgcar = seed_dir / "CHGCAR"

        if src_wavecar.exists() and src_wavecar.stat().st_size > 0:
            shutil.copy2(src_wavecar, target_dir / "WAVECAR")
            has_wavecar = True

        if src_chgcar.exists() and src_chgcar.stat().st_size > 0:
            shutil.copy2(src_chgcar, target_dir / "CHGCAR")
            has_chgcar = True

    incar = target_dir / "INCAR"
    text = incar.read_text()

    text = re.sub(r"(?m)^\s*ISTART\s*=.*\n?", "", text)
    text = re.sub(r"(?m)^\s*ICHARG\s*=.*\n?", "", text)

    istart = 1 if has_wavecar else 0
    icharg = 1 if has_chgcar else 2

    text = text.rstrip() + f"\nISTART = {istart}\nICHARG = {icharg}\n"
    incar.write_text(text)


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

    Also writes a reference (undisplaced) supercell calculation that is used
    as the first seed for subsequent displacement calculations.

    For VASP, existing displacement directories and existing input files are
    preserved and not overwritten.
    """
    from phono3py import Phono3py
    from dataclasses import replace

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
    sc_symbols = list(ph3.supercell.symbols)
    sc_positions = ph3.supercell.positions

    print(f"Generated {n_disp} displacements (phono3py)")
    print(f"  Unit cell: {len(cell.symbols)} atoms")
    print(f"  Supercell: {supercell}, {len(ph3.supercell.symbols)} atoms")
    if cutoff_pair_distance is not None:
        print(f"  Pair cutoff: {cutoff_pair_distance:.1f} A")
    print(f"  Calculator: {calculator}")

    if calculator == "qe":
        assert isinstance(dft_config, QEConfig)

        pseudo_dir_abs = Path(dft_config.pseudo_dir).resolve()
        try:
            pseudo_dir_rel = pseudo_dir_abs.relative_to(work_dir.resolve())
        except ValueError:
            pseudo_dir_rel = pseudo_dir_abs

        qe_local = replace(dft_config, pseudo_dir=str(pseudo_dir_rel))
        (work_dir / QE_RESULTS_DIRNAME).mkdir(exist_ok=True)

        _write_qe_input(
            work_dir / REFERENCE_QE_INPUT,
            sc_cell,
            sc_symbols,
            sc_positions,
            qe_local,
            prefix=QE_RESTART_PREFIX,
            use_restart_data=False,
        )

        for i, sc in enumerate(supercells):
            if sc is None:
                continue
            inp_path = work_dir / f"disp-{i+1:05d}.in"
            _write_qe_input(
                inp_path,
                sc_cell,
                list(sc.symbols),
                sc.positions,
                qe_local,
                prefix=QE_RESTART_PREFIX,
                use_restart_data=False,
            )

        print(f"Wrote reference and {n_disp} QE input files in {work_dir}")

    elif calculator == "vasp":
        assert isinstance(dft_config, VASPConfig)

        ref_dir = work_dir / REFERENCE_VASP_DIRNAME
        if ref_dir.exists():
            print(f"  Preserving existing VASP reference dir: {ref_dir.name}")
        else:
            print(f"  Creating VASP reference dir: {ref_dir.name}")
        _write_vasp_inputs(
            ref_dir,
            sc_cell,
            sc_symbols,
            sc_positions,
            dft_config,
            use_restart_data=False,
            overwrite=False,
        )

        for i, sc in enumerate(supercells):
            if sc is None:
                continue

            disp_dir = work_dir / f"disp-{i+1:05d}"
            if disp_dir.exists():
                print(f"  Preserving existing VASP dir: {disp_dir.name}")
            else:
                print(f"  Creating VASP dir: {disp_dir.name}")

            _write_vasp_inputs(
                disp_dir,
                sc_cell,
                list(sc.symbols),
                sc.positions,
                dft_config,
                use_restart_data=False,
                overwrite=False,
            )

        print(f"Prepared reference and {n_disp} VASP displacement directories in {work_dir}")

    else:
        raise ValueError(f"Unknown calculator: {calculator!r}. Use 'qe' or 'vasp'.")

    return n_disp


def _needs_shell(command: str) -> bool:
    """Check if command needs shell=True (env vars, pipes, redirects)."""
    return any(tok in command for tok in ("=", "|", ">", "<", ";", "&&"))


def _run_and_tee(cmd, log_path: Path, cwd: str, timeout: int, use_shell: bool) -> int:
    """Run a command, streaming stdout/stderr to both terminal and log file."""
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
        fds = {
            proc.stdout.fileno(): ("stdout", proc.stdout, sys.stdout),
            proc.stderr.fileno(): ("stderr", proc.stderr, sys.stderr),
        }
        while fds:
            elapsed = time.monotonic() - t0
            if elapsed > timeout:
                proc.kill()
                raise subprocess.TimeoutExpired(cmd, timeout)
            remaining = max(0.1, timeout - elapsed)
            ready, _, _ = select.select(list(fds.keys()), [], [], min(remaining, 1.0))
            for fd in ready:
                _, pipe, tty = fds[fd]
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


def _latest_completed_vasp_seed(work_dir: Path, before_index: int | None = None) -> Path | None:
    """Return the latest completed VASP directory with usable restart files."""
    if before_index is not None:
        for idx in range(before_index - 1, 0, -1):
            disp_dir = work_dir / f"disp-{idx:05d}"
            if _is_vasp_done(disp_dir):
                has_wav, has_chg = _vasp_restart_files_valid(disp_dir)
                if has_wav or has_chg:
                    return disp_dir

    ref_dir = work_dir / REFERENCE_VASP_DIRNAME
    if _is_vasp_done(ref_dir):
        has_wav, has_chg = _vasp_restart_files_valid(ref_dir)
        if has_wav or has_chg:
            return ref_dir

    return None


def run_displacements(
    work_dir: Path,
    dft_command: str = "pw.x",
    timeout: int = 3600,
    calculator: str = "qe",
) -> None:
    """Run DFT for each displacement.

    Behavior:
      - Skip jobs that are already complete
      - If no prior calculation exists, run the reference supercell first
      - Reuse previous electronic data for subsequent jobs
    """
    work_dir = Path(work_dir)
    use_shell = _needs_shell(dft_command)

    if calculator == "qe":
        ref_inp = work_dir / REFERENCE_QE_INPUT
        ref_out = work_dir / REFERENCE_QE_OUTPUT
        inp_files = sorted(work_dir.glob("disp-*.in"))
        n_disp = len(inp_files)

        if not ref_inp.exists():
            raise FileNotFoundError("No QE reference input found. Run sow first.")
        if n_disp == 0:
            raise FileNotFoundError("No QE input files found. Run sow first.")

        print(f"Running reference + {n_disp} QE displacements...")

        if _is_qe_done(ref_out):
            print("  [ref] Skipping reference (done)")
        elif not _qe_restart_data_available(work_dir):
            print("  [ref] Running reference supercell...")
            _rewrite_qe_restart_flags(ref_inp, use_restart_data=False)
            if use_shell:
                cmd = f"{dft_command} -in {ref_inp.name}"
            else:
                cmd = shlex.split(dft_command) + ["-in", ref_inp.name]
            rc = _run_and_tee(cmd, ref_out, str(work_dir), timeout, use_shell)
            if rc != 0 or not _is_qe_done(ref_out):
                print(f"  ERROR: {ref_inp.name} failed (exit {rc})")
                raise RuntimeError("QE reference calculation failed")
        else:
            print("  [ref] Restart data already present; reference run not needed")

        for i, inp_file in enumerate(inp_files, start=1):
            out_file = inp_file.with_suffix(".out")
            if _is_qe_done(out_file):
                print(f"  [{i}/{n_disp}] Skipping {inp_file.name} (done)")
                continue

            use_restart = _qe_restart_data_available(work_dir)
            _rewrite_qe_restart_flags(inp_file, use_restart_data=use_restart)

            print(
                f"  [{i}/{n_disp}] Running {inp_file.name} "
                f"({'restart' if use_restart else 'fresh'})..."
            )

            if use_shell:
                cmd = f"{dft_command} -in {inp_file.name}"
            else:
                cmd = shlex.split(dft_command) + ["-in", inp_file.name]

            rc = _run_and_tee(cmd, out_file, str(work_dir), timeout, use_shell)

            if rc != 0 or not _is_qe_done(out_file):
                print(f"  ERROR: {inp_file.name} failed (exit {rc})")
                raise RuntimeError(f"QE displacement {inp_file.name} failed")

    elif calculator == "vasp":
        ref_dir = work_dir / REFERENCE_VASP_DIRNAME
        disp_dirs = sorted(d for d in work_dir.glob("disp-*/") if d.is_dir())
        n_disp = len(disp_dirs)

        if not ref_dir.exists():
            raise FileNotFoundError("No VASP reference dir found. Run sow first.")
        if n_disp == 0:
            raise FileNotFoundError("No VASP displacement dirs found. Run sow first.")

        print(f"Running reference + {n_disp} VASP displacements...")

        if _is_vasp_done(ref_dir):
            print("  [ref] Skipping reference (done)")
        else:
            print("  [ref] Running reference supercell...")
            _prepare_vasp_restart(ref_dir, seed_dir=None)

            cmd = dft_command if use_shell else shlex.split(dft_command)
            log_file = ref_dir / "vasp.log"
            rc = _run_and_tee(cmd, log_file, str(ref_dir), timeout, use_shell)

            if rc != 0 or not _is_vasp_done(ref_dir):
                print(f"  ERROR: {ref_dir.name} failed (exit {rc})")
                raise RuntimeError("VASP reference calculation failed")

        for i, disp_dir in enumerate(disp_dirs, start=1):
            if _is_vasp_done(disp_dir):
                print(f"  [{i}/{n_disp}] Skipping {disp_dir.name} (done)")
                continue

            seed_dir = _latest_completed_vasp_seed(work_dir, before_index=i)
            _prepare_vasp_restart(disp_dir, seed_dir=seed_dir)

            has_wavecar = (disp_dir / "WAVECAR").exists() and (disp_dir / "WAVECAR").stat().st_size > 0
            has_chgcar = (disp_dir / "CHGCAR").exists() and (disp_dir / "CHGCAR").stat().st_size > 0

            print(
                f"  [{i}/{n_disp}] Running {disp_dir.name} "
                f"(seed: {seed_dir.name if seed_dir else 'none'}, "
                f"WAVECAR={'yes' if has_wavecar else 'no'}, "
                f"CHGCAR={'yes' if has_chgcar else 'no'})..."
            )

            cmd = dft_command if use_shell else shlex.split(dft_command)
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
    """Read DFT forces and produce FC3 (and FC2) via phono3py."""
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

    skip_existing is kept for API compatibility; completed jobs are always skipped.
    """
    dft_command = (
        dft_config.pw_command if calculator == "qe" else dft_config.vasp_command
    )

    sow(
        cell,
        work_dir,
        dft_config,
        supercell,
        cutoff_pair_distance,
        distance,
        calculator=calculator,
    )
    run_displacements(work_dir, dft_command, calculator=calculator)
    return reap(work_dir, fc_calculator=fc_calculator, calculator=calculator)