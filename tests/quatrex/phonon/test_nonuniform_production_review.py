"""Tests for cross-grid production result normalisation."""

from pathlib import Path

import numpy as np

from studies import _nonuniform_production_review as R


def test_uniform_legacy_current_is_scaled_but_nonuniform_is_not(tmp_path):
    uniform = tmp_path / "u.npz"
    nonuniform = tmp_path / "n.npz"
    np.savez(uniform, lead_current=8.0, uniform_frequency_grid=True,
             energies=np.linspace(0.0, 2.0, 5))
    np.savez(nonuniform, lead_current=4.0, uniform_frequency_grid=False,
             energies=np.array([0.0, 0.2, 0.8, 2.0]))
    with np.load(uniform) as u, np.load(nonuniform) as n:
        assert R.physical_current(u) == 4.0
        assert R.physical_current(n) == 4.0


def test_timing_drops_first_use_sample(tmp_path):
    path = tmp_path / "run"
    path.mkdir()
    (path / "x_quatrex_times.out").write_text(
        "    PhononSolver all : 9.0s\n"
        "    PhononSolver all : 2.0s\n"
        "    PhononSolver all : 4.0s\n"
        "  SCBA: Iteration all : 12.0s\n"
        "  SCBA: Iteration all : 6.0s\n")
    row = R._timing(path)
    assert row["solver_seconds"] == 3.0
    assert row["iteration_seconds"] == 6.0


def test_aux_point_count_from_config(tmp_path):
    (tmp_path / "quatrex_config.toml").write_text(
        "[phonon]\nsse_aux_grid_dw_thz = 0.1\n"
        "sse_aux_grid_fmax_thz = 4.0\n")
    assert R._aux_points(tmp_path, np.linspace(0.0, 2.0, 11)) == 41


def test_frequency_moment_commutes_only_on_matching_grid():
    uniform = np.linspace(0.0, 4.0, 41)
    assert R.moment_intertwining_defect(uniform, uniform) == (0.0, 0.0)
    primary = uniform[[0, 1, 2, 5, 10, 20, 40]]
    relative, maximum = R.moment_intertwining_defect(primary, uniform)
    assert relative > 1e-2
    assert maximum > 0.1


def test_incomplete_rejected_run_is_summarised_from_log(tmp_path):
    path = tmp_path / "failed"
    path.mkdir()
    np.save(path / "phonon_energies.npy", np.array([0.0, 0.4, 1.0]))
    (path / "quatrex_config.toml").write_text(
        "[phonon]\nsse_aux_grid_dw_thz = 0.1\n"
        "sse_aux_grid_fmax_thz = 1.0\n")
    (path / "slurm-1.out").write_text(
        "Phonon: rel Sigma^R residual 2.0e-3; lead balance 8.0e-2; "
        "internal spread 8.0e-2; lead current 1.2e+1\n")
    row = R.case(path)
    assert row["log_only"]
    assert not row["converged"]
    assert row["last_sigma_residual"] == 2e-3
    assert row["internal_spread"] == 8e-2


def test_committed_summary_paths_are_checkout_relative(tmp_path):
    run = tmp_path / "cluster" / "case"
    run.mkdir(parents=True)
    row = {
        "path": str(run),
        "source_log": str(run / "slurm.out"),
        "timing": {"file": str(run / "times.out"), "iteration_seconds": 1.0},
    }
    got = R._relative_paths(row, tmp_path.resolve())
    assert got["path"] == "cluster/case"
    assert got["source_log"] == "cluster/case/slurm.out"
    assert got["timing"]["file"] == "cluster/case/times.out"
