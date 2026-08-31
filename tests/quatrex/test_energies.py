from types import SimpleNamespace

import numpy as np

from quatrex.grid.energies import get_electron_energies


def test_energy_output_creation_is_race_safe(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    monkeypatch.setattr("os.path.exists", lambda _path: False)
    config = SimpleNamespace(
        electron=SimpleNamespace(
            energy_window_min=0.0,
            energy_window_max=1.0,
            energy_window_num=3,
            energy_window_num_per_rank=None,
        ),
        input_dir=tmp_path,
        output_dir=output_dir,
    )

    energies = get_electron_energies(config)

    np.testing.assert_allclose(energies, [0.0, 0.5, 1.0])
    assert (output_dir / "electron_energies.npy").is_file()
