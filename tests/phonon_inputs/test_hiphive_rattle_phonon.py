"""Smoke test for the phonon-rattle two-stage workflow.

Exercises the new ``rattle_method="phonon"`` dispatch end-to-end on an
8-atom Si chain by mocking DFT forces with the analytic FC2 force law
``F_i = -Σ_j Φ_{ij} u_j``. Checks:

  1. The first :func:`sow` call falls back to mc-rattle bootstrap into
     ``work_dir/bootstrap/``.
  2. :func:`bootstrap_reap` produces ``fc2_seed.npy`` and the recovered
     FC2 matches the analytic input to within fit noise.
  3. A second :func:`sow` call picks up the seed and generates the main
     phonon-rattled pool with non-trivial displacements.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PHONON = Path(__file__).resolve().parents[2] / "phonon"
if str(_PHONON) not in sys.path:
    sys.path.insert(0, str(_PHONON))


def _chain_primitive(n_atoms: int = 8, spacing: float = 2.35):
    """Return the chain :class:`PhonopyAtoms` used by the smoke test."""
    from phonopy.structure.atoms import PhonopyAtoms

    c_len = n_atoms * spacing
    cell = np.diag([15.0, 15.0, c_len])
    scaled = np.array([[0.5, 0.5, (i + 0.5) / n_atoms] for i in range(n_atoms)])
    return PhonopyAtoms(
        symbols=["Si"] * n_atoms, cell=cell, scaled_positions=scaled,
    )


def _chain_fc2(n_atoms: int = 8, k: float = 6.0) -> np.ndarray:
    """Analytic NN FC2 with PBC, shape ``(N, N, 3, 3)`` in eV/Å²."""
    fc2 = np.zeros((n_atoms, n_atoms, 3, 3))
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            d = (i - j) % n_atoms
            if d == 1 or d == n_atoms - 1:
                fc2[i, j] = -k * np.eye(3)
    for i in range(n_atoms):
        fc2[i, i] = -fc2[i, :, :, :].sum(axis=0)
    return 0.5 * (fc2 + fc2.transpose(1, 0, 3, 2))


def _fake_vasp_outputs(boot_dir: Path, fc2: np.ndarray) -> None:
    """For each bootstrap displacement, write a minimal vasprun.xml whose
    forces match the harmonic force law ``F = -Φ · u`` on the displaced
    POSCAR. The POSCAR was emitted by :func:`sow` and contains the
    rattled positions.
    """
    from ase.io import read as ase_read

    # Read the reference (undisplaced) positions from the reference dir.
    ref_dir = boot_dir / "reference"
    ref_poscar = ref_dir / "POSCAR"
    ref_atoms = ase_read(str(ref_poscar), format="vasp")
    ref_pos = np.asarray(ref_atoms.positions)

    disp_dirs = sorted(p for p in boot_dir.iterdir() if p.name.startswith("disp-"))
    n_super = ref_pos.shape[0]
    for d in disp_dirs:
        atoms = ase_read(str(d / "POSCAR"), format="vasp")
        u = np.asarray(atoms.positions) - ref_pos
        forces = -np.einsum("ijab,jb->ia", fc2, u)  # eV/Å, shape (N, 3)

        # Minimal vasprun.xml that _parse_vasp_forces can read.
        # _parse_vasp_forces looks for the LAST <varray name="forces"> block.
        lines = [
            '<?xml version="1.0" encoding="ISO-8859-1"?>',
            '<modeling>',
            '  <calculation>',
            '    <varray name="forces">',
        ]
        for fx in forces:
            lines.append(
                f'      <v> {fx[0]:.10f} {fx[1]:.10f} {fx[2]:.10f} </v>'
            )
        lines += [
            '    </varray>',
            '  </calculation>',
            '</modeling>',
        ]
        (d / "vasprun.xml").write_text("\n".join(lines))


def test_phonon_rattle_two_stage_workflow(tmp_path):
    """End-to-end: bootstrap sow -> mock DFT -> bootstrap_reap -> main sow."""
    pytest.importorskip("ase.io.vasp")
    pytest.importorskip("hiphive")

    from phonon_inputs.config import HiphiveConfig, VASPConfig
    from phonon_inputs.hiphive_fc3 import (
        BOOTSTRAP_DIRNAME, FC2_SEED_FILENAME, bootstrap_reap, sow,
    )

    # Create a fake POTCAR tree the VASP writer can find.
    potcar_root = tmp_path / "potcars"
    (potcar_root / "Si").mkdir(parents=True)
    (potcar_root / "Si" / "POTCAR").write_text("# fake Si POTCAR for unit test\n")

    primitive = _chain_primitive()
    fc2_true = _chain_fc2()

    hh = HiphiveConfig(
        supercell=[1, 1, 1],
        n_structures=6,
        cutoffs=[3.0],
        fit_method="least-squares",
        rattle_method="phonon",
        rattle_std=0.05,
        rattle_d_min=1.5,
        rattle_n_iter=5,
        calculator="vasp",
        phonon_rattle_temperature_k=300.0,
        phonon_rattle_bootstrap_n=4,
        phonon_rattle_bootstrap_seed=11,
        phonon_rattle_imag_freq_factor=1.0,
        phonon_rattle_qm=True,
    )
    # Minimal VASP config; _write_vasp_inputs only reads the fields we
    # set here to emit POSCAR/INCAR/KPOINTS — the test never invokes vasp.
    vasp = VASPConfig(
        potcar_dir=str(potcar_root),
        potcar_map={"Si": "Si"},
        kpoints_scf=[1, 1, 1],
        encut=200,
        vasp_command="echo",
    )

    work_dir = tmp_path / "fc3_hiphive"

    # --- Stage 1: bootstrap sow ------------------------------------------
    n_boot = sow(primitive, work_dir, vasp, hh)
    assert n_boot == hh.phonon_rattle_bootstrap_n
    boot_dir = work_dir / BOOTSTRAP_DIRNAME
    assert boot_dir.exists(), "Bootstrap directory was not created"
    assert not (work_dir / FC2_SEED_FILENAME).exists(), (
        "Seed FC2 must not exist before bootstrap reap"
    )
    disp_dirs = sorted(p for p in boot_dir.iterdir() if p.name.startswith("disp-"))
    assert len(disp_dirs) == hh.phonon_rattle_bootstrap_n

    # --- Mock DFT: write vasprun.xml with harmonic forces ----------------
    _fake_vasp_outputs(boot_dir, fc2_true)

    # --- Stage 1b: bootstrap reap ----------------------------------------
    seed = bootstrap_reap(work_dir, hh_config=hh)
    assert seed == work_dir / FC2_SEED_FILENAME
    assert seed.exists()
    fc2_seed = np.load(seed)
    # Hiphive reshapes to (N, N, 3, 3); compare directly.
    assert fc2_seed.shape == fc2_true.shape, (
        f"seed shape {fc2_seed.shape} != true {fc2_true.shape}"
    )
    # Harmonic forces -> exact recovery within numeric noise.
    rel = np.linalg.norm(fc2_seed - fc2_true) / np.linalg.norm(fc2_true)
    assert rel < 1e-3, (
        f"Recovered FC2 is off by {rel:.3%}; should be exact on synthetic data"
    )

    # --- Stage 2: main sow picks up the seed -----------------------------
    n_main = sow(primitive, work_dir, vasp, hh)
    assert n_main == hh.n_structures, (
        f"Main sow emitted {n_main} structures, expected {hh.n_structures}"
    )

    # Main pool sits in work_dir (NOT under bootstrap/), with non-zero
    # displacements relative to the reference.
    main_dirs = sorted(p for p in work_dir.iterdir() if p.name.startswith("disp-"))
    assert len(main_dirs) == hh.n_structures, (
        f"Found {len(main_dirs)} main disp dirs, expected {hh.n_structures}"
    )

    from ase.io import read as ase_read
    ref_pos = np.asarray(
        ase_read(str(work_dir / "reference" / "POSCAR"), format="vasp").positions,
    )
    rms_disp = []
    for d in main_dirs:
        pos = np.asarray(ase_read(str(d / "POSCAR"), format="vasp").positions)
        rms_disp.append(float(np.sqrt(np.mean((pos - ref_pos) ** 2))))
    rms_disp = np.array(rms_disp)
    # Phonon-rattle at 300 K should produce displacements on the order of
    # the Debye-Waller scale; for k=6 eV/Å² it's ~0.05–0.2 Å.
    assert (rms_disp > 0.01).all(), (
        f"Phonon-rattle produced near-zero displacements: rms={rms_disp}"
    )
