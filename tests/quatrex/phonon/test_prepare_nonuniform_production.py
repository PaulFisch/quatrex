"""Small tests for matched nonuniform production preparation."""

from pathlib import Path

import numpy as np

from studies import _prepare_nonuniform_production as P


def _pilot(path: Path) -> None:
    w = np.linspace(0.0, 4.0, 129)
    current = np.exp(-((w - 1.1) / 0.05) ** 2)[:, None]
    gr = np.exp(-((w - 2.8) / 0.08) ** 2)[:, None, None]
    np.savez(path, energies=w, frequency_cell_widths=np.gradient(w),
             current_spectrum=current, gr_diag_imag=gr)


def test_selected_grid_contains_endpoints_and_beats_uniform_count(tmp_path):
    run = tmp_path / "run.npz"
    _pilot(run)
    ids = P.selected_knots(run, 1e-2)
    assert ids[0] == 0 and ids[-1] == 128
    assert np.all(np.diff(ids) > 0)
    assert ids.size < 60


def test_prepare_preserves_aux_spacing_and_regrids_q_rank_snapshot(tmp_path):
    base = tmp_path / "base"
    target = tmp_path / "target"
    base.mkdir()
    run = base / "run.npz"
    _pilot(run)
    (base / "quatrex_config.toml").write_text(
        f'simulation_dir = "{base}"\ninput_dir = "{base}"\n'
        f'output_dir = "{base}/out"\n[electron]\nenergy_window_num = 129\n'
        '[phonon]\nfrequency_grid = "window"\n'
        'sse_aux_grid_dw_thz = 0.0\nsse_aux_grid_fmax_thz = 0.0\n'
        '[phonon.solver]\n')
    shape = (129, 2, 2)
    for rank in range(2):
        values = np.arange(np.prod(shape)).reshape(shape) * (rank + 1j)
        np.savez(tmp_path / f"warm.rank{rank}.npz",
                 sigma_lesser=values, sigma_greater=2 * values,
                 sigma_retarded=3 * values)
    row = P.prepare_case(base, run, target, 1e-2, warm_parts=2,
                         warm_base=tmp_path / "warm")
    assert row["primary_points"] < row["uniform_points"]
    assert row["aux_points"] == 129
    assert row["aux_dw_thz"] == 4.0 / 128
    new = np.load(target / "phonon_energies.npy")
    snap = np.load(target / "sigma_init.rank1.npz")
    assert snap["sigma_lesser"].shape[0] == new.size
    cfg = (target / "quatrex_config.toml").read_text()
    assert 'frequency_grid = "file"' in cfg
