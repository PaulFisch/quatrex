"""Restart seeding of timeout-bounded VASP relaxation legs
(``qe_interface._maybe_restart_cell``): an incomplete leg's CONTCAR
must seed the next leg's POSCAR; without the flag (legacy) the
original cell is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonopy.structure.atoms import PhonopyAtoms

from phonon_inputs.qe_interface import _maybe_restart_cell


def _write_contcar(work_dir: Path, frac: np.ndarray) -> None:
    lines = [
        "restart fixture",
        "1.0",
        "  5.0 0.0 0.0",
        "  0.0 5.0 0.0",
        "  0.0 0.0 5.0",
        "  Si O",
        "  1 1",
        "Direct",
    ]
    for row in frac:
        lines.append("  " + " ".join(f"{x:.12f}" for x in row))
    (work_dir / "CONTCAR").write_text("\n".join(lines) + "\n")


def _cell() -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=["Si", "O"],
        cell=np.eye(3) * 5.0,
        scaled_positions=np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
    )


def test_restart_seeds_from_contcar(tmp_path: Path) -> None:
    moved = np.array([[0.01, 0.02, 0.03], [0.30, 0.28, 0.26]])
    _write_contcar(tmp_path, moved)
    seeded = _maybe_restart_cell(tmp_path, _cell(), True)
    assert np.allclose(seeded.scaled_positions, moved, atol=1e-10)
    assert list(seeded.symbols) == ["Si", "O"]
    assert np.allclose(seeded.cell, np.eye(3) * 5.0)


def test_legacy_default_keeps_original_cell(tmp_path: Path) -> None:
    _write_contcar(tmp_path, np.array([[0.1, 0.1, 0.1], [0.4, 0.4, 0.4]]))
    orig = _cell()
    out = _maybe_restart_cell(tmp_path, orig, False)
    assert out is orig


def test_no_contcar_falls_back_to_original(tmp_path: Path) -> None:
    orig = _cell()
    out = _maybe_restart_cell(tmp_path, orig, True)
    assert out is orig
