"""Third-order force constant generation via hiphive + randomized rattled supercells.

Hiphive fits a force-constant potential up to user-specified order from a
small set of rattled supercell calculations. Each calculation contains
forces from many simultaneously displaced atoms, so the total number of
DFT runs is typically much smaller than for a phono3py finite-displacement
sweep over symmetry-inequivalent triplets, especially for low-symmetry
systems.

Workflow (mirrors thirdorder.py):
    1. sow:  Build the supercell, generate N rattled supercells, write DFT
             inputs (QE pw.x or VASP), plus a reference undisplaced run.
    2. run:  Execute DFT for the reference and each rattled structure.
             Reuses thirdorder.run_displacements (calculator-agnostic).
    3. reap: Parse forces, fit the cluster expansion, extract FC2 + FC3
             on the supercell, and write fc3.hdf5 (with fc2 + fc3 datasets,
             same layout as thirdorder.reap so downstream consumers do
             not need to branch).

The reaped fc3.hdf5 file mirrors thirdorder's output: one HDF5 file with
two datasets ``fc3`` (n_super, n_super, n_super, 3, 3, 3) and ``fc2``
(n_super, n_super, 3, 3), both in eV / A^n.

DFT inputs reuse the helpers in thirdorder.py so QE restart-data sharing,
VASP restart seeding, INCAR refresh, and run-skipping behave identically
across the two FC3 backends.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms

from . import thirdorder as _to
from .config import HiphiveConfig, QEConfig, VASPConfig
from .structure import structure_from_ase, structure_to_ase

REFERENCE_DIRNAME = _to.REFERENCE_VASP_DIRNAME  # "reference" (VASP)
META_FILENAME = "hiphive_meta.json"


# ========================================================================
# Supercell + rattled-structure generation
# ========================================================================


def _build_supercell(primitive: PhonopyAtoms, multipliers: tuple[int, int, int]) -> PhonopyAtoms:
    """Expand a PhonopyAtoms unit cell into a diagonal supercell.

    Delegates to phonopy.Phonopy so the atom ordering matches phono3py /
    symfc exactly. This means the FC arrays produced by the hiphive
    pipeline can be indexed and compared against those from the
    finite-displacement pipeline atom-for-atom.
    """
    from phonopy import Phonopy

    sc_matrix = np.diag(list(multipliers)).astype(int)
    ph = Phonopy(primitive, supercell_matrix=sc_matrix, primitive_matrix=np.eye(3))
    sc = ph.supercell
    return PhonopyAtoms(
        symbols=list(sc.symbols),
        cell=np.asarray(sc.cell),
        scaled_positions=np.asarray(sc.scaled_positions),
    )


def _generate_rattled(
    atoms_ideal,
    n_structures: int,
    method: str,
    rattle_std: float,
    d_min: float,
    n_iter: int,
    seed: int,
):
    """Generate rattled ASE Atoms via hiphive.

    Returns a list of ase.Atoms with displacements applied.
    """
    if method == "mc":
        from hiphive.structure_generation import generate_mc_rattled_structures

        return generate_mc_rattled_structures(
            atoms_ideal,
            n_structures,
            rattle_std=rattle_std,
            d_min=d_min,
            seed=seed,
            n_iter=n_iter,
        )
    if method == "normal":
        from hiphive.structure_generation import generate_rattled_structures

        return generate_rattled_structures(
            atoms_ideal, n_structures, rattle_std=rattle_std, seed=seed,
        )
    raise ValueError(
        f"Unknown rattle_method: {method!r}. Use 'mc' or 'normal'."
    )


# ========================================================================
# Calculator-agnostic workflow
# ========================================================================


def sow(
    cell: PhonopyAtoms,
    work_dir: Path,
    dft_config: QEConfig | VASPConfig,
    hh_config: HiphiveConfig,
) -> int:
    """Generate rattled supercells and write DFT inputs.

    Also writes a reference (undisplaced) supercell calculation, used as
    the first electronic seed for subsequent runs (see thirdorder.py for
    how QE/VASP restart data is reused).

    Returns the number of rattled structures generated (excluding the
    reference).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    multipliers = tuple(hh_config.supercell)
    if len(multipliers) != 3:
        raise ValueError(f"hiphive.supercell must have 3 entries, got {multipliers}")

    sc = _build_supercell(cell, multipliers)
    sc_cell = np.asarray(sc.cell)
    sc_symbols = list(sc.symbols)
    sc_positions = np.asarray(sc.positions)

    atoms_ideal = structure_to_ase(sc)

    rattled = _generate_rattled(
        atoms_ideal,
        n_structures=hh_config.n_structures,
        method=hh_config.rattle_method,
        rattle_std=hh_config.rattle_std,
        d_min=hh_config.rattle_d_min,
        n_iter=hh_config.rattle_n_iter,
        seed=hh_config.rattle_seed,
    )
    n_disp = len(rattled)

    # Persist metadata so reap can reconstruct atoms_ideal and ClusterSpace
    # without re-running structure generation (which would require seeding
    # again and is sensitive to hiphive version differences).
    meta = {
        "supercell": list(multipliers),
        "n_structures": n_disp,
        "rattle_method": hh_config.rattle_method,
        "rattle_std": hh_config.rattle_std,
        "rattle_d_min": hh_config.rattle_d_min,
        "rattle_n_iter": hh_config.rattle_n_iter,
        "rattle_seed": hh_config.rattle_seed,
        "cutoffs": list(hh_config.cutoffs),
        "fit_method": hh_config.fit_method,
        "fit_kwargs": dict(hh_config.fit_kwargs),
        "calculator": hh_config.calculator,
        "primitive": {
            "symbols": list(cell.symbols),
            "cell": np.asarray(cell.cell).tolist(),
            "scaled_positions": np.asarray(cell.scaled_positions).tolist(),
        },
        "supercell_atoms": {
            "symbols": sc_symbols,
            "cell": sc_cell.tolist(),
            "scaled_positions": np.asarray(sc.scaled_positions).tolist(),
        },
    }
    (work_dir / META_FILENAME).write_text(json.dumps(meta, indent=2))

    print(f"Generated {n_disp} rattled structures (hiphive, method={hh_config.rattle_method})")
    print(f"  Unit cell: {len(cell.symbols)} atoms")
    print(f"  Supercell: {multipliers}, {len(sc_symbols)} atoms")
    print(f"  rattle_std: {hh_config.rattle_std} A, d_min: {hh_config.rattle_d_min} A")
    print(f"  Cutoffs: {hh_config.cutoffs} A")
    print(f"  Calculator: {hh_config.calculator}")

    calculator = hh_config.calculator

    if calculator == "qe":
        if not isinstance(dft_config, QEConfig):
            raise TypeError("hiphive.calculator='qe' but qe config not provided")

        pseudo_dir_abs = Path(dft_config.pseudo_dir).resolve()
        try:
            pseudo_dir_rel = pseudo_dir_abs.relative_to(work_dir.resolve())
        except ValueError:
            pseudo_dir_rel = pseudo_dir_abs

        qe_local = replace(dft_config, pseudo_dir=str(pseudo_dir_rel))
        (work_dir / _to.QE_RESULTS_DIRNAME).mkdir(exist_ok=True)

        _to._write_qe_input(
            work_dir / _to.REFERENCE_QE_INPUT,
            sc_cell,
            sc_symbols,
            sc_positions,
            qe_local,
            prefix=_to.QE_RESTART_PREFIX,
            use_restart_data=False,
        )

        for i, rat_atoms in enumerate(rattled, start=1):
            inp_path = work_dir / f"disp-{i:05d}.in"
            _to._write_qe_input(
                inp_path,
                sc_cell,
                list(rat_atoms.get_chemical_symbols()),
                np.asarray(rat_atoms.positions),
                qe_local,
                prefix=_to.QE_RESTART_PREFIX,
                use_restart_data=False,
            )

        print(f"Wrote reference and {n_disp} QE input files in {work_dir}")

    elif calculator == "vasp":
        if not isinstance(dft_config, VASPConfig):
            raise TypeError("hiphive.calculator='vasp' but vasp config not provided")

        ref_dir = work_dir / REFERENCE_DIRNAME
        if ref_dir.exists():
            print(f"  Preserving existing VASP reference dir: {ref_dir.name}")
        else:
            print(f"  Creating VASP reference dir: {ref_dir.name}")
        _to._write_vasp_inputs(
            ref_dir, sc_cell, sc_symbols, sc_positions,
            dft_config, use_restart_data=False, overwrite=False,
        )

        for i, rat_atoms in enumerate(rattled, start=1):
            disp_dir = work_dir / f"disp-{i:05d}"
            if disp_dir.exists():
                print(f"  Preserving existing VASP dir: {disp_dir.name}")
            else:
                print(f"  Creating VASP dir: {disp_dir.name}")

            _to._write_vasp_inputs(
                disp_dir,
                sc_cell,
                list(rat_atoms.get_chemical_symbols()),
                np.asarray(rat_atoms.positions),
                dft_config,
                use_restart_data=False,
                overwrite=False,
            )

        print(f"Prepared reference and {n_disp} VASP rattled directories in {work_dir}")

    else:
        raise ValueError(f"Unknown calculator: {calculator!r}. Use 'qe' or 'vasp'.")

    return n_disp


def run_displacements(
    work_dir: Path,
    dft_command: str,
    timeout: int = 3600,
    calculator: str = "qe",
    dft_config: QEConfig | VASPConfig | None = None,
) -> None:
    """Run DFT for each rattled structure (delegates to thirdorder)."""
    _to.run_displacements(
        work_dir, dft_command, timeout=timeout,
        calculator=calculator, dft_config=dft_config,
    )


# ========================================================================
# Reap: parse forces, fit FCP, write fc2/fc3
# ========================================================================


def _load_meta(work_dir: Path) -> dict:
    meta_path = work_dir / META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(
            f"hiphive metadata not found at {meta_path}. Run 'hiphive-sow' first."
        )
    return json.loads(meta_path.read_text())


def _atoms_from_meta(entry: dict):
    """Reconstruct an ASE Atoms object from a metadata dict entry."""
    import ase

    return ase.Atoms(
        symbols=entry["symbols"],
        cell=np.asarray(entry["cell"]),
        scaled_positions=np.asarray(entry["scaled_positions"]),
        pbc=True,
    )


def reap(
    work_dir: Path,
    hh_config: HiphiveConfig | None = None,
) -> Path:
    """Read DFT forces, fit a cluster expansion, write fc3.hdf5.

    The output file mirrors thirdorder.reap: one HDF5 with datasets
    ``fc3`` (n_super, n_super, n_super, 3, 3, 3) in eV/A^3 and ``fc2``
    (n_super, n_super, 3, 3) in eV/A^2, both expressed on the supercell.

    Hiphive fits force constants in the chosen unit-system that matches
    the input forces. The DFT helpers in thirdorder.py emit forces in
    eV/A for both QE (after Ry/bohr conversion) and VASP, so the fitted
    FCs come out in eV/A^n consistently with thirdorder.
    """
    import ase
    import h5py
    from hiphive import (
        ClusterSpace,
        ForceConstantPotential,
        StructureContainer,
    )
    from hiphive.utilities import prepare_structures

    work_dir = Path(work_dir)
    meta = _load_meta(work_dir)

    if hh_config is not None:
        # Allow re-fitting with different cutoffs/methods without re-running DFT.
        cutoffs = list(hh_config.cutoffs)
        fit_method = hh_config.fit_method
        fit_kwargs = dict(hh_config.fit_kwargs)
        calculator = hh_config.calculator
    else:
        cutoffs = meta["cutoffs"]
        fit_method = meta["fit_method"]
        fit_kwargs = meta.get("fit_kwargs", {})
        calculator = meta["calculator"]

    n_disp = meta["n_structures"]
    atoms_ideal = _atoms_from_meta(meta["supercell_atoms"])
    primitive = _atoms_from_meta(meta["primitive"])
    n_super = len(atoms_ideal)

    print(f"Reading forces from {n_disp} {calculator.upper()} outputs...")

    rattled = []
    for i in range(1, n_disp + 1):
        if calculator == "qe":
            out_file = work_dir / f"disp-{i:05d}.out"
            if not out_file.exists():
                raise FileNotFoundError(f"Missing: {out_file}")
            forces = _to._parse_qe_forces(out_file, n_super)
            inp_file = work_dir / f"disp-{i:05d}.in"
            positions = _read_positions_from_qe_input(inp_file, n_super)
        elif calculator == "vasp":
            disp_dir = work_dir / f"disp-{i:05d}"
            if not disp_dir.exists():
                raise FileNotFoundError(f"Missing: {disp_dir}")
            forces = _to._parse_vasp_forces(disp_dir, n_super)
            positions = _read_positions_from_vasp_poscar(disp_dir / "POSCAR", n_super)
        else:
            raise ValueError(f"Unknown calculator: {calculator!r}")

        rat = ase.Atoms(
            symbols=list(atoms_ideal.get_chemical_symbols()),
            cell=atoms_ideal.cell,
            positions=positions,
            pbc=True,
        )
        rat.arrays["forces"] = forces
        rattled.append(rat)

    print(f"  Loaded {len(rattled)} rattled structures, {n_super} atoms each")

    print(f"Building ClusterSpace with cutoffs {cutoffs} A...")
    cs = ClusterSpace(primitive, cutoffs)
    print(f"  {cs}")

    structures = prepare_structures(rattled, atoms_ideal)
    sc = StructureContainer(cs)
    for s in structures:
        sc.add_structure(s)
    print(f"  {sc}")

    print(f"Fitting force-constant potential ({fit_method})...")
    from trainstation import Optimizer

    opt = Optimizer(sc.get_fit_data(), fit_method=fit_method, **fit_kwargs)
    opt.train()
    rmse_train = float(opt.rmse_train)
    rmse_test = float(opt.rmse_test) if opt.rmse_test is not None else float("nan")
    print(
        f"  parameters: {opt.n_parameters}, "
        f"RMSE train: {rmse_train:.4e}, "
        f"RMSE test: {rmse_test:.4e}"
    )

    fcp = ForceConstantPotential(cs, opt.parameters)
    fcs = fcp.get_force_constants(atoms_ideal)

    fc2 = fcs.get_fc_array(order=2)
    fc3 = fcs.get_fc_array(order=3)

    print(f"  FC2 shape: {fc2.shape}, max: {np.max(np.abs(fc2)):.4e} eV/A^2")
    print(f"  FC3 shape: {fc3.shape}, max: {np.max(np.abs(fc3)):.4e} eV/A^3")

    fc3_path = work_dir / "fc3.hdf5"
    with h5py.File(fc3_path, "w") as f:
        f.create_dataset("fc3", data=fc3, compression="gzip")
        f.create_dataset("fc2", data=fc2, compression="gzip")
    print(f"  Saved: {fc3_path} ({fc3_path.stat().st_size / 1e6:.1f} MB)")

    # Persist the fit summary alongside the FC file for later inspection.
    summary = {
        "fit_method": fit_method,
        "fit_kwargs": fit_kwargs,
        "cutoffs": cutoffs,
        "n_structures": n_disp,
        "n_parameters": int(len(opt.parameters)),
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "calculator": calculator,
    }
    (work_dir / "hiphive_fit.json").write_text(json.dumps(summary, indent=2))

    fcp_path = work_dir / "fcp.fcp"
    fcp.write(str(fcp_path))
    print(f"  Saved FCP: {fcp_path}")

    return fc3_path


# ========================================================================
# Helpers to recover supercell positions from displacement DFT inputs
# ========================================================================


def _read_positions_from_qe_input(inp_path: Path, n_atoms: int) -> np.ndarray:
    """Parse cartesian ATOMIC_POSITIONS (in Angstrom) from a QE input file."""
    text = inp_path.read_text().splitlines()
    out = []
    in_block = False
    for line in text:
        stripped = line.strip()
        if stripped.upper().startswith("ATOMIC_POSITIONS"):
            in_block = True
            continue
        if in_block:
            if not stripped or stripped.startswith("#"):
                continue
            if any(stripped.upper().startswith(card) for card in (
                "CELL_PARAMETERS", "K_POINTS", "OCCUPATIONS",
                "CONSTRAINTS", "ATOMIC_FORCES", "HUBBARD",
            )):
                break
            parts = stripped.split()
            if len(parts) < 4:
                continue
            try:
                xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
            except ValueError:
                continue
            out.append(xyz)
            if len(out) == n_atoms:
                break

    if len(out) != n_atoms:
        raise ValueError(
            f"Expected {n_atoms} atomic positions in {inp_path}, got {len(out)}"
        )
    return np.array(out)


def _read_positions_from_vasp_poscar(poscar: Path, n_atoms: int) -> np.ndarray:
    """Parse cartesian positions (Angstrom) from a VASP POSCAR.

    Restores the original (un-sorted) atom order using the .atom_order
    file that thirdorder._write_vasp_inputs wrote alongside POSCAR.
    """
    lines = poscar.read_text().splitlines()
    scale = float(lines[1].strip())
    cell = np.array(
        [[float(x) for x in lines[2 + i].split()] for i in range(3)],
    ) * scale

    counts_idx = 6
    counts = [int(x) for x in lines[counts_idx].split()]
    if sum(counts) != n_atoms:
        raise ValueError(f"POSCAR atom count mismatch: {sum(counts)} != {n_atoms}")

    pos_kind_line = lines[7].strip().lower()
    if pos_kind_line.startswith("selective"):
        offset = 9
        kind = lines[8].strip().lower()
    else:
        offset = 8
        kind = pos_kind_line

    sorted_positions = np.array(
        [[float(x) for x in lines[offset + i].split()[:3]] for i in range(n_atoms)],
    )

    if kind.startswith(("d", "f")):
        sorted_cart = sorted_positions @ cell
    else:
        sorted_cart = sorted_positions * scale

    order_file = poscar.parent / ".atom_order"
    if order_file.exists():
        sorted_indices = np.loadtxt(order_file, dtype=int)
        positions = np.empty_like(sorted_cart)
        positions[sorted_indices] = sorted_cart
    else:
        positions = sorted_cart

    return positions


# ========================================================================
# One-shot driver
# ========================================================================


def generate_fc3(
    cell: PhonopyAtoms,
    work_dir: Path,
    dft_config: QEConfig | VASPConfig,
    hh_config: HiphiveConfig,
) -> Path:
    """Full hiphive FC3 pipeline: sow + run + reap."""
    dft_command = (
        dft_config.pw_command
        if hh_config.calculator == "qe"
        else dft_config.vasp_command
    )

    sow(cell, work_dir, dft_config, hh_config)
    run_displacements(
        Path(work_dir),
        dft_command,
        timeout=hh_config.pw_timeout,
        calculator=hh_config.calculator,
        dft_config=dft_config,
    )
    return reap(Path(work_dir), hh_config=hh_config)
