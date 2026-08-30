from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


_PATH = Path(__file__).parents[3] / "phonon/studies/regrid_sigma.py"
_SPEC = importlib.util.spec_from_file_location("regrid_sigma", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
regrid_sigma = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(regrid_sigma)


def test_distributed_checkpoint_regrid_and_partition(tmp_path, monkeypatch):
    old_grid = np.array([0.0, 1.0, 2.0])
    new_grid = np.linspace(0.0, 2.0, 5)
    shape = (old_grid.size, 2, 2)
    base = np.arange(np.prod(shape), dtype=float).reshape(shape)
    state = {
        "sigma_lesser": 1j * base,
        "sigma_greater": (2.0 + 3.0j) * base,
        "sigma_retarded": (4.0 - 1.0j) * base,
    }

    source = tmp_path / "source"
    target = tmp_path / "target.npz"
    regrid_sigma._save_sigma(source, state, parts=2)
    np.savez(tmp_path / "old.npz", energies=old_grid)
    np.save(tmp_path / "new.npy", new_grid)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "regrid_sigma.py",
            "--sigma",
            str(source),
            "--input-parts",
            "2",
            "--old-grid",
            str(tmp_path / "old.npz"),
            "--new-grid",
            str(tmp_path / "new.npy"),
            "--out",
            str(target),
            "--output-parts",
            "4",
            "--scale",
            "0.5",
        ],
    )
    regrid_sigma.main()

    result = regrid_sigma._load_sigma(target, parts=4)
    for key, value in state.items():
        expected = np.empty((new_grid.size,) + value.shape[1:], complex)
        for index in np.ndindex(value.shape[1:]):
            expected[(slice(None),) + index] = 0.5 * np.interp(
                new_grid, old_grid, value[(slice(None),) + index].real
            ) + 0.5j * np.interp(
                new_grid, old_grid, value[(slice(None),) + index].imag
            )
        np.testing.assert_allclose(result[key], expected)

    assert [
        np.load(regrid_sigma._rank_path(target, rank))["sigma_lesser"].shape[0]
        for rank in range(4)
    ] == [2, 1, 1, 1]
