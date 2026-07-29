"""Regression: _write_vasp_inputs must round-trip non-diagonal cells.

The cart->frac conversion used ``@ inv(cell).T``, which is only correct
for diagonal cells (every system before MoS2). On the hexagonal 2H-MoS2
[4,4,1] supercell it scrambled the geometry to 0.98 A minimum pair
distances and VASP computed forces on garbage (2026-07-29).
"""
from pathlib import Path

import numpy as np
import pytest

from phonon_inputs.config import VASPConfig


def _read_poscar(path: Path):
    lines = path.read_text().splitlines()
    scale = float(lines[1])
    cell = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)])
    cell *= scale
    counts = [int(x) for x in lines[6].split()]
    n = sum(counts)
    assert lines[7].strip().lower().startswith("d")
    frac = np.array([[float(x) for x in ln.split()[:3]]
                     for ln in lines[8:8 + n]])
    return cell, frac


@pytest.mark.parametrize("cell", [
    # hexagonal (2H-MoS2-like)
    np.array([[3.16, 0.0, 0.0],
              [-1.58, 2.7366402760, 0.0],
              [0.0, 0.0, 12.294]]),
    # monoclinic (TiS3-like, beta != 90)
    np.array([[4.97, 0.0, 0.0],
              [0.0, 3.39, 0.0],
              [-1.55, 0.0, 8.63]]),
    # diagonal control
    np.diag([5.0, 6.0, 7.0]),
])
def test_poscar_cart_frac_roundtrip(tmp_path, cell):
    from phonon_inputs.thirdorder import _write_vasp_inputs

    rng = np.random.default_rng(7)
    frac_in = rng.uniform(0.05, 0.95, size=(8, 3))
    cart_in = frac_in @ cell
    symbols = ["Mo"] * 3 + ["S"] * 5

    # fake POTCAR library so the writer completes
    potdir = tmp_path / "pot"
    for sp in ("Mo", "S"):
        (potdir / sp).mkdir(parents=True)
        (potdir / sp / "POTCAR").write_text(f"fake {sp}\n")
    cfg = VASPConfig(potcar_dir=str(potdir), potcar_map={})

    out = tmp_path / "disp"
    _write_vasp_inputs(out, cell, symbols, cart_in, cfg)

    cell_out, frac_out = _read_poscar(out / "POSCAR")
    np.testing.assert_allclose(cell_out, cell, atol=1e-10)

    # writer groups by species (Mo block then S block) -- same order here
    cart_out = frac_out @ cell_out
    # compare pairwise distances to be independent of wrapping
    def pdists(c):
        d = c[:, None, :] - c[None, :, :]
        # minimum image over neighbour cells
        imgs = np.array([[i, j, k] for i in (-1, 0, 1)
                         for j in (-1, 0, 1) for k in (-1, 0, 1)])
        dd = np.linalg.norm(
            d[None, :, :, :] + (imgs @ cell)[:, None, None, :], axis=-1)
        return dd.min(axis=0)

    np.testing.assert_allclose(pdists(cart_out), pdists(cart_in), atol=1e-8)
