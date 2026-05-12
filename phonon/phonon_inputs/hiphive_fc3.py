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
BOOTSTRAP_DIRNAME = "bootstrap"
FC2_SEED_FILENAME = "fc2_seed.npy"


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
    """Generate mc- or normal-rattled ASE Atoms via hiphive.

    For phonon-mode rattling see :func:`_generate_phonon_rattled`, which
    needs a pre-fit FC2 and is dispatched at the :func:`sow` level.
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
        f"Unknown rattle_method: {method!r}. Use 'mc' or 'normal' "
        "(phonon-rattling has its own dispatch path)."
    )


def _seed_from_existing_fc3(
    work_dir: Path, fc3_path: Path, atoms_ideal,
) -> None:
    """Extract FC2 from an existing ``fc3.hdf5`` and write
    ``work_dir/fc2_seed.npy`` in the ``(N, N, 3, 3)`` format hiphive expects.

    Used by the phonon-rattle workflow when
    ``HiphiveConfig.phonon_rattle_seed_fc3`` is set — bypasses the bootstrap
    DFT batch entirely and seeds the phonon-rattle pool from a converged
    FC2 (mc-rattle reap, phono3py reap, or any other source with the same
    primitive cell + supercell layout).
    """
    import h5py
    fc3_path = Path(fc3_path)
    if not fc3_path.exists():
        raise FileNotFoundError(
            f"phonon_rattle_seed_fc3 points at a missing file: {fc3_path}"
        )
    with h5py.File(fc3_path, "r") as f:
        if "fc2" not in f:
            raise KeyError(
                f"{fc3_path} has no 'fc2' dataset. Re-reap the upstream "
                "calculation with `fc3-hiphive-reap` (or thirdorder-reap) "
                "so it writes FC2 alongside FC3."
            )
        fc2 = np.asarray(f["fc2"][:], dtype=np.float64)
    n_atoms = len(atoms_ideal)
    if fc2.shape[:2] != (n_atoms, n_atoms):
        raise ValueError(
            f"phonon_rattle_seed_fc3 FC2 has shape {fc2.shape}; expected "
            f"({n_atoms}, {n_atoms}, 3, 3). Source supercell must match the "
            "current rattle supercell exactly."
        )
    seed_path = work_dir / FC2_SEED_FILENAME
    work_dir.mkdir(parents=True, exist_ok=True)
    np.save(seed_path, fc2)
    summary = {
        "stage": "seed_from_existing",
        "source": str(fc3_path),
        "fc2_shape": list(fc2.shape),
        "fc2_max": float(np.max(np.abs(fc2))),
    }
    (work_dir / "hiphive_bootstrap.json").write_text(json.dumps(summary, indent=2))
    print(
        f"phonon-rattle: seeded FC2 from {fc3_path} "
        f"(max |FC2| = {summary['fc2_max']:.3e}); wrote {seed_path}."
    )


def _generate_phonon_rattled(
    atoms_ideal,
    fc2_seed: np.ndarray,
    n_structures: int,
    temperature_k: float,
    qm_statistics: bool,
    imag_freq_factor: float,
    max_imag_modes: int = 6,
    atoms_for_min_image=None,
):
    """Generate phonon-rattled ASE Atoms from a seed FC2.

    Uses :func:`hiphive.structure_generation.generate_phonon_rattled_structures`,
    which samples displacements

        R_a = R_a^0 + Σ_s X_{as} √[ℏ(0.5 + n_BE(T, |ω_s|))/(m_a ω_s)] · ...

    Imaginary modes get ``w² → imag_freq_factor × |w²|``. ``imag_freq_factor=1.0``
    (the hiphive default) flips them to the same magnitude with positive sign.
    **``imag_freq_factor=0.0`` is unsafe**: it zeroes the frequency, then
    hiphive divides by zero in ``np.sqrt(... / ω_s)`` and the resulting
    displacements are NaN. This wrapper rejects that combination upfront
    and prints the imaginary-mode count + max-magnitude diagnostic so
    the user knows whether the seed FC2 is healthy enough to phonon-rattle.
    """
    if imag_freq_factor == 0.0:
        raise ValueError(
            "phonon_rattle_imag_freq_factor=0.0 produces NaN displacements "
            "(hiphive zeroes the frequency and then divides by it). Use 1.0 "
            "(the hiphive default — flips imaginary modes to positive same-"
            "magnitude frequencies) or a small positive value (< 1) to damp them. "
            "If the seed FC2 has many imaginary modes the right fix is to "
            "increase phonon_rattle_bootstrap_n or switch to rattle_method=mc."
        )

    # Diagnose the seed FC2 before sampling. Hiphive accepts both
    # (3N, 3N) and (N, N, 3, 3); reshape to (3N, 3N) for diagonalisation
    # and mass-weight via the atoms object.
    import numpy as _np
    fc2 = _np.asarray(fc2_seed)
    if fc2.ndim == 4:
        n = fc2.shape[0]
        D = fc2.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    else:
        D = fc2.copy()
    masses = _np.asarray(atoms_ideal.get_masses())
    minv = 1.0 / _np.sqrt(_np.repeat(masses, 3))
    Dm = D * minv[:, None] * minv[None, :]
    Dm = 0.5 * (Dm + Dm.T)
    eig = _np.linalg.eigvalsh(Dm)
    n_imag = int(_np.sum(eig < -1e-8))
    n_zero = int(_np.sum(_np.abs(eig) <= 1e-8))
    if n_imag > 0:
        max_imag = float(_np.sqrt(-eig.min()))
        print(
            f"  seed FC2 has {n_imag} imaginary mode(s) (max |ω| ≈ "
            f"{max_imag:.3f} √eV/(Å²·amu), plus {n_zero} zero/acoustic)."
        )
        if n_imag > max_imag_modes:
            raise RuntimeError(
                f"Seed FC2 has {n_imag} imaginary modes (cap = "
                f"{max_imag_modes} via phonon_rattle_max_seed_imag). "
                "Phonon-rattle's per-mode amplitude scales as "
                "1/√|ω|, so a handful of small-|ω| imaginary modes "
                "blow up displacements and produce hiphive's 'Duplicates "
                "in permutation' or NaN POSCARs.\n\n"
                "Recommended fix: complete an mc-rattle reap on the same "
                "supercell and point ``phonon_rattle_seed_fc3`` at its "
                "fc3.hdf5. The phonon-rattle pool is then a finite-T "
                "refinement on a converged harmonic model — the workflow "
                "used by Eriksson et al. 2019 (Adv. Theory Simul. 2, "
                "1800184) and Carrete et al. 2017.\n\n"
                "Override with phonon_rattle_max_seed_imag: <large int> "
                "if you understand the risk."
            )

    from hiphive.structure_generation import generate_phonon_rattled_structures

    rattled = generate_phonon_rattled_structures(
        atoms_ideal,
        fc2_seed,
        n_structures=n_structures,
        temperature=temperature_k,
        QM_statistics=qm_statistics,
        imag_freq_factor=imag_freq_factor,
    )

    # Hiphive's ``find_permutation`` matches each rattled atom to the
    # nearest ideal-supercell atom in *Cartesian* coordinates. If
    # any atom has wandered across more than half a lattice vector
    # the matching can collide ("Duplicates in permutation"). Wrap
    # each rattled position back to its starting min-image so the
    # matching is unambiguous even at finite T. This is a no-op for
    # well-conditioned seeds where displacements ≪ NN spacing.
    if atoms_for_min_image is not None:
        ref_pos = _np.asarray(atoms_for_min_image.positions)
        cell = _np.asarray(atoms_for_min_image.cell)
        cell_inv = _np.linalg.inv(cell)
        max_disp = 0.0
        for r in rattled:
            pos = _np.asarray(r.positions)
            delta = pos - ref_pos
            frac = delta @ cell_inv
            frac -= _np.round(frac)
            wrapped_pos = ref_pos + frac @ cell
            r.set_positions(wrapped_pos)
            max_disp = max(max_disp, float(_np.max(_np.linalg.norm(
                wrapped_pos - ref_pos, axis=1,
            ))))
        print(
            f"  phonon-rattle: max atomic displacement after min-image "
            f"fold = {max_disp:.3f} Å."
        )

    # Hard catch: if any structure has NaN positions the cluster job
    # will silently produce garbage. Fail loud here.
    for i, r in enumerate(rattled, start=1):
        pos = _np.asarray(r.positions)
        if not _np.isfinite(pos).all():
            n_bad = int((~_np.isfinite(pos)).sum())
            raise RuntimeError(
                f"phonon-rattle structure #{i} has {n_bad} non-finite "
                f"position entries. This always indicates an unsafe "
                f"imag_freq_factor or a degenerate seed FC2 (zero-mode "
                f"divided by zero). Refit the seed (more bootstrap_n, "
                f"set phonon_rattle_seed_fc3 to a converged fc3.hdf5, "
                f"or switch to rattle_method=mc) and try again."
            )
    return rattled


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

    ``rattle_method='phonon'`` routes through a two-stage workflow:

      1. **Bootstrap.** If ``work_dir/fc2_seed.npy`` is missing, write a
         small mc-rattle pool into ``work_dir/bootstrap/`` (see
         :data:`HiphiveConfig.phonon_rattle_bootstrap_n`). The caller
         then runs DFT in the bootstrap dir and invokes
         :func:`bootstrap_reap` to produce the seed.
      2. **Main.** With the seed present, this function generates
         :data:`HiphiveConfig.n_structures` phonon-rattled structures
         via :func:`generate_phonon_rattled_structures`.

    Returns the number of rattled structures generated in the active
    stage (excluding the reference).
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

    # ---- Phonon-rattle dispatch (two-stage) --------------------------------
    if hh_config.rattle_method == "phonon":
        seed_path = work_dir / FC2_SEED_FILENAME
        # Preferred path: load FC2 from an existing converged fc3.hdf5
        # (e.g. a completed mc-rattle reap on the same primitive). This
        # mirrors how the literature uses phonon-rattle — as a finite-T
        # refinement on top of a converged harmonic model, NOT as a
        # from-scratch sampler.
        if not seed_path.exists() and hh_config.phonon_rattle_seed_fc3:
            _seed_from_existing_fc3(
                work_dir,
                Path(hh_config.phonon_rattle_seed_fc3).expanduser(),
                atoms_ideal,
            )
        if not seed_path.exists():
            # Fallback: emit a bootstrap mc-rattle pool. Works well for
            # high-symmetry bulk crystals; on low-symmetry quasi-1-D
            # systems (SiNW) the bootstrap FC2 is usually saturated
            # with imaginary modes — bootstrap_reap will refuse to
            # write the seed in that case, and you should pre-build
            # an mc-rattle reap and point phonon_rattle_seed_fc3 at
            # it instead.
            boot_dir = work_dir / BOOTSTRAP_DIRNAME
            print(
                f"hiphive phonon-rattle: no seed FC2 found at {seed_path}, "
                f"no phonon_rattle_seed_fc3 set. Writing bootstrap "
                f"mc-rattle pool to {boot_dir}/ "
                f"(n={hh_config.phonon_rattle_bootstrap_n})."
            )
            boot_cfg = replace(
                hh_config,
                rattle_method="mc",
                n_structures=hh_config.phonon_rattle_bootstrap_n,
                rattle_seed=hh_config.phonon_rattle_bootstrap_seed,
                work_dir=str(boot_dir),
            )
            return sow(cell, boot_dir, dft_config, boot_cfg)
        # Stage 2: phonon-rattled main pool from the seed FC2.
        fc2_seed = np.load(seed_path)
        print(
            f"hiphive phonon-rattle: loaded seed FC2 from {seed_path} "
            f"(shape {fc2_seed.shape}); generating {hh_config.n_structures} "
            f"phonon-rattled structures at T={hh_config.phonon_rattle_temperature_k} K."
        )
        rattled = _generate_phonon_rattled(
            atoms_ideal,
            fc2_seed=fc2_seed,
            n_structures=hh_config.n_structures,
            temperature_k=hh_config.phonon_rattle_temperature_k,
            qm_statistics=hh_config.phonon_rattle_qm,
            imag_freq_factor=hh_config.phonon_rattle_imag_freq_factor,
            max_imag_modes=hh_config.phonon_rattle_max_seed_imag,
            atoms_for_min_image=atoms_ideal,
        )
    else:
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


def bootstrap_reap(
    work_dir: Path,
    hh_config: HiphiveConfig,
) -> Path:
    """Fit an FC2-only force-constant model on the bootstrap pool.

    Stage 2 of the phonon-rattle workflow expects ``work_dir/fc2_seed.npy``
    to exist; this function produces it from ``work_dir/bootstrap/``:

      1. Read forces from each ``disp-XXXXX`` calculation in the
         bootstrap directory (created by :func:`sow` when
         ``rattle_method == "phonon"`` and the seed is missing).
      2. Build a hiphive :class:`ClusterSpace` with only the FC2
         cutoff (``cutoffs[:1]``) and fit by least-squares.
      3. Extract the supercell FC2 in the ``(N, N, 3, 3)`` phonopy
         convention and save it to ``work_dir/fc2_seed.npy``.

    The next :func:`sow` call detects the seed and switches to the
    phonon-rattled main stage.
    """
    import ase
    from hiphive import (
        ClusterSpace, ForceConstantPotential, StructureContainer,
    )
    from hiphive.utilities import prepare_structures
    from trainstation import Optimizer

    work_dir = Path(work_dir)
    boot_dir = work_dir / BOOTSTRAP_DIRNAME
    if not boot_dir.exists():
        raise FileNotFoundError(
            f"Bootstrap directory missing: {boot_dir}. Run "
            f"`fc3-hiphive-sow` first with rattle_method='phonon' to "
            "create the bootstrap mc-rattle pool."
        )
    meta = _load_meta(boot_dir)
    n_disp = meta["n_structures"]
    atoms_ideal = _atoms_from_meta(meta["supercell_atoms"])
    primitive = _atoms_from_meta(meta["primitive"])
    n_super = len(atoms_ideal)
    calculator = meta["calculator"]

    print(f"Bootstrap reap: reading {n_disp} {calculator.upper()} outputs "
          f"from {boot_dir}...")

    rattled = []
    for i in range(1, n_disp + 1):
        if calculator == "qe":
            out_file = boot_dir / f"disp-{i:05d}.out"
            forces = _to._parse_qe_forces(out_file, n_super)
            inp_file = boot_dir / f"disp-{i:05d}.in"
            positions = _read_positions_from_qe_input(inp_file, n_super)
        elif calculator == "vasp":
            disp_dir = boot_dir / f"disp-{i:05d}"
            forces = _to._parse_vasp_forces(disp_dir, n_super)
            positions = _read_positions_from_vasp_poscar(
                disp_dir / "POSCAR", n_super,
            )
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

    fc2_cutoff = list(hh_config.cutoffs)[:1]
    print(f"  Fitting FC2-only model with cutoffs={fc2_cutoff} A...")
    cs = ClusterSpace(primitive, fc2_cutoff)
    structures = prepare_structures(rattled, atoms_ideal)
    sc = StructureContainer(cs)
    for s in structures:
        sc.add_structure(s)

    # Use ARDR (or ridge as fallback) instead of bare least-squares.
    # Bootstrap pools are always small (typically 4–8 structures), so
    # least-squares fits marginally-determined long-wavelength modes by
    # noise and the resulting FC2 has many imaginary modes. ARDR's
    # automatic relevance pruning produces a much cleaner FC2 from the
    # same pool. The post-fit rotational sum rule (Huang + Born–Huang)
    # then closes the spurious ZA-flexural gap.
    bootstrap_fit_method = "ardr"
    try:
        opt = Optimizer(
            sc.get_fit_data(), fit_method=bootstrap_fit_method,
        )
        opt.train()
    except Exception as exc:  # noqa: BLE001
        print(f"  ardr failed ({exc!r}); falling back to ridge.")
        bootstrap_fit_method = "ridge"
        opt = Optimizer(
            sc.get_fit_data(), fit_method=bootstrap_fit_method,
        )
        opt.train()
    print(
        f"  fit_method={bootstrap_fit_method}, "
        f"parameters: {opt.n_parameters}, "
        f"RMSE train: {opt.rmse_train:.4e}, "
        f"RMSE test: "
        f"{opt.rmse_test if opt.rmse_test is not None else float('nan'):.4e}"
    )
    params = opt.parameters

    # Apply Huang + Born-Huang rotational projection if requested.
    rot_mode = getattr(hh_config, "rotational_sum_rule", "off")
    rot_before = float("nan")
    rot_after = float("nan")
    if rot_mode == "post_fit":
        from .hiphive_convergence import (
            _apply_rotational_sum_rules, _rotational_residual_pair,
        )
        rot_before, _ = _rotational_residual_pair(cs, params, mode="off")
        params = _apply_rotational_sum_rules(cs, params)
        rot_after, _ = _rotational_residual_pair(cs, params, mode="off")
        print(
            f"  rotational sum rules projected: "
            f"residual {rot_before:.3e} -> {rot_after:.3e}"
        )

    fcp = ForceConstantPotential(cs, params)
    fcs = fcp.get_force_constants(atoms_ideal)
    fc2 = fcs.get_fc_array(order=2)

    # Diagnose FC2 PSD-ness BEFORE writing the seed. The downstream
    # phonon-rattle amplitude diverges as 1/√|ω| for small |ω|, so a
    # seed with even a handful of imaginary modes produces "Duplicates
    # in permutation" or NaN POSCARs. Refuse to write a bad seed.
    masses = np.asarray(atoms_ideal.get_masses())
    minv = 1.0 / np.sqrt(np.repeat(masses, 3))
    n = n_super
    D = fc2.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    Dm = D * minv[:, None] * minv[None, :]
    Dm = 0.5 * (Dm + Dm.T)
    eig = np.linalg.eigvalsh(Dm)
    n_imag = int(np.sum(eig < -1e-8))
    max_seed_imag = getattr(hh_config, "phonon_rattle_max_seed_imag", 6)

    summary = {
        "stage": "bootstrap",
        "n_structures": n_disp,
        "cutoffs": fc2_cutoff,
        "fit_method": bootstrap_fit_method,
        "rotational_sum_rule": rot_mode,
        "rotational_residual_before": rot_before,
        "rotational_residual_after": rot_after,
        "rmse_train": float(opt.rmse_train),
        "rmse_test": (
            float(opt.rmse_test) if opt.rmse_test is not None else None
        ),
        "n_parameters": int(len(opt.parameters)),
        "calculator": calculator,
        "fc2_max": float(np.max(np.abs(fc2))),
        "n_imag_modes_in_seed": n_imag,
        "max_seed_imag_allowed": max_seed_imag,
    }
    (work_dir / "hiphive_bootstrap.json").write_text(json.dumps(summary, indent=2))

    if n_imag > max_seed_imag:
        bad_path = work_dir / (FC2_SEED_FILENAME + ".rejected")
        np.save(bad_path, fc2)
        raise RuntimeError(
            f"Bootstrap seed FC2 has {n_imag} imaginary mode(s) "
            f"(cap = {max_seed_imag} via phonon_rattle_max_seed_imag). "
            f"Refusing to write {FC2_SEED_FILENAME} — phonon-rattle "
            f"would produce 'Duplicates in permutation' or NaN POSCARs.\n\n"
            f"Saved the bad fit to {bad_path.name} for diagnosis. "
            f"hiphive_bootstrap.json carries the full numerics.\n\n"
            "Recommended next step: run an mc-rattle reap on the same "
            "supercell to convergence, then set\n"
            "    hiphive.phonon_rattle_seed_fc3: "
            "<path-to-mc-rattle-fc3.hdf5>\n"
            "and re-run; the phonon-rattle workflow then refines the "
            "converged FC2 at finite T (the workflow used in "
            "Eriksson et al. 2019 / Carrete et al. 2017)."
        )

    # Phonopy convention for generate_phonon_rattled_structures.
    seed_path = work_dir / FC2_SEED_FILENAME
    np.save(seed_path, fc2)
    print(
        f"  Saved seed FC2: {seed_path} "
        f"(max |FC2| = {summary['fc2_max']:.3e}, n_imag = {n_imag})"
    )
    return seed_path


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

    # Rotational sum-rule policy: "off" | "post_fit" | "constrained".
    # See HIPHIVE_FITTING_NOTES.md § 2.3. ``constrained`` is not yet
    # implemented because no current YAML uses it; an explicit error is
    # raised so the user notices the config drift.
    rotational_mode = (
        getattr(hh_config, "rotational_sum_rule", "off") if hh_config is not None
        else meta.get("rotational_sum_rule", "off")
    )
    params = opt.parameters
    rotational_residual_before = float("nan")
    rotational_residual_after = float("nan")
    if rotational_mode in ("post_fit", "constrained"):
        from .hiphive_convergence import (
            _apply_rotational_sum_rules,
            _rotational_residual_pair,
        )
        if rotational_mode == "constrained":
            raise NotImplementedError(
                "rotational_sum_rule='constrained' is not yet wired up "
                "in reap(); use 'post_fit' instead. "
                "See HIPHIVE_FITTING_NOTES.md."
            )
        rotational_residual_before, _ = _rotational_residual_pair(
            cs, params, mode="off",
        )
        params = _apply_rotational_sum_rules(cs, params)
        rotational_residual_after, _ = _rotational_residual_pair(
            cs, params, mode="off",
        )
        print(
            f"  Rotational sum rules ('Huang' + 'Born-Huang') projected: "
            f"residual {rotational_residual_before:.3e} -> "
            f"{rotational_residual_after:.3e}"
        )

    fcp = ForceConstantPotential(cs, params)
    fcs = fcp.get_force_constants(atoms_ideal)

    # Acoustic sum rule status. Hiphive 1.5 has no FC-tensor-level
    # enforce_acoustic_sum_rules method — translational ASR is built into
    # the ClusterSpace orbits at construction time (always on by default),
    # so the extracted FC tensors are ASR-respecting along axis i to within
    # rounding. We log the actual residuals via the diagnostic call
    # ForceConstants.assert_acoustic_sum_rules and persist them in the
    # summary JSON for the audit trail. Axis-j / axis-k ASR (relevant for
    # the SSE bubble) is enforced separately downstream in
    # load_quatrex_blocks(asr_project=True).
    asr_residuals: dict[int, float] = {}
    for order in (2, 3):
        try:
            fcs.assert_acoustic_sum_rules(order=order, tol=1e-3)
            asr_residuals[order] = 0.0
            print(f"  FC{order} ASR residual within tol (≤ 1e-3).")
        except AssertionError as exc:
            # The exception message carries the numeric residual; parse it
            # robustly without depending on the exact wording.
            import re
            m = re.search(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", str(exc))
            r = float(m.group(1)) if m else float("nan")
            asr_residuals[order] = r
            print(f"  FC{order} ASR residual = {r:.3e} (orbit-basis ASR was "
                  f"applied; axis-j/k residual reflects the parameter fit).")

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
        "rotational_sum_rule": rotational_mode,
        "rotational_residual_before": rotational_residual_before,
        "rotational_residual_after": rotational_residual_after,
        "fc2_asr_residual": asr_residuals.get(2, float("nan")),
        "fc3_asr_residual": asr_residuals.get(3, float("nan")),
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
    """Full hiphive FC3 pipeline.

    For mc / normal rattling this is the canonical
    sow → run → reap sequence. For ``rattle_method='phonon'`` the
    pipeline performs the longer two-stage workflow:

      1. **sow** writes the bootstrap mc-rattle pool into
         ``work_dir/bootstrap/`` (because ``fc2_seed.npy`` is missing).
      2. **run_displacements** in ``work_dir/bootstrap/``.
      3. **bootstrap_reap** fits an FC2-only model and writes
         ``work_dir/fc2_seed.npy``.
      4. **sow** detects the seed and writes the main phonon-rattled
         pool into ``work_dir/`` itself.
      5. **run_displacements** in ``work_dir/``.
      6. **reap** fits the cluster expansion to all orders, applies
         the rotational sum rule, and writes ``fc3.hdf5``.

    The bootstrap stages are skipped if ``fc2_seed.npy`` already exists.
    """
    work_dir = Path(work_dir)
    dft_command = (
        dft_config.pw_command
        if hh_config.calculator == "qe"
        else dft_config.vasp_command
    )

    # --- Stage 1 + bootstrap reap (phonon-rattle only) ------------------
    if hh_config.rattle_method == "phonon":
        seed_path = work_dir / FC2_SEED_FILENAME
        if not seed_path.exists():
            boot_dir = work_dir / BOOTSTRAP_DIRNAME
            # If sow has already emitted the bootstrap pool, don't re-sow
            # (preserves any partial DFT output the user has already run).
            need_boot_sow = not (boot_dir / META_FILENAME).exists()
            if need_boot_sow:
                print(
                    "hiphive phonon-rattle pipeline: emitting bootstrap pool."
                )
                sow(cell, work_dir, dft_config, hh_config)
            else:
                print(
                    "hiphive phonon-rattle pipeline: bootstrap pool already "
                    f"present in {boot_dir}; proceeding to DFT."
                )
            print("hiphive phonon-rattle pipeline: running bootstrap DFT.")
            run_displacements(
                boot_dir,
                dft_command,
                timeout=hh_config.pw_timeout,
                calculator=hh_config.calculator,
                dft_config=dft_config,
            )
            print("hiphive phonon-rattle pipeline: fitting bootstrap FC2.")
            bootstrap_reap(work_dir, hh_config=hh_config)
        else:
            print(
                "hiphive phonon-rattle pipeline: found existing seed FC2 "
                f"at {seed_path}; skipping bootstrap."
            )

    # --- Stage 2 (main pool) --------------------------------------------
    # Detect whether main-stage sow has already run (avoids re-emitting
    # rattled inputs over a partially-completed DFT batch).
    if not (work_dir / META_FILENAME).exists():
        sow(cell, work_dir, dft_config, hh_config)
    else:
        print(
            f"hiphive pipeline: main-stage sow already done in {work_dir}; "
            "proceeding to DFT."
        )
    run_displacements(
        work_dir,
        dft_command,
        timeout=hh_config.pw_timeout,
        calculator=hh_config.calculator,
        dft_config=dft_config,
    )
    return reap(work_dir, hh_config=hh_config)
