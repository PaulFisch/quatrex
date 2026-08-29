import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "_si_micro_gate", ROOT / "phonon/scripts/si_micro_gate.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

antihermiticity_defect = MODULE.antihermiticity_defect
compare = MODULE.compare
load_sigma = MODULE.load_sigma
negativity_fraction = MODULE.negativity_fraction
relative_error = MODULE.relative_error
relative_l1 = MODULE.relative_l1


def test_array_metrics():
    source = np.diag([1.0, 2.0])
    sigma = (1j * source).reshape(1, 4)

    assert antihermiticity_defect(sigma) == pytest.approx(0.0)
    assert negativity_fraction(sigma) == pytest.approx(0.0)
    assert relative_error(source, 1.01 * source) == pytest.approx(0.01)
    assert relative_l1(source, 1.01 * source) == pytest.approx(0.01)


def test_negativity_is_reported_not_projected():
    sigma = np.diag([1j, -0.25j]).reshape(1, 4)
    assert negativity_fraction(sigma) == pytest.approx(0.2)


def _write_case(directory: Path, name: str, scale: float, negative: float):
    run = directory / f"{name}_run.npz"
    sigma = directory / f"{name}_sigma.npz"
    matrix = np.diag([1j * scale, -1j * negative]).reshape(1, 4)
    np.savez(
        run,
        vertex_representation=name,
        sse_vertex_rank=8 if name == "candidate" else 0,
        converged=True,
        n_iter=12,
        lead_current=2.0 * scale,
        current_spectrum=np.array([1.0, 2.0]) * scale,
    )
    np.savez(
        sigma,
        sigma_lesser=matrix,
        sigma_greater=2.0 * matrix,
        sigma_retarded=3.0 * matrix,
    )
    return run, sigma


def test_complete_gate(tmp_path):
    ref_run, ref_sigma = _write_case(tmp_path, "reference", 1.0, 0.1)
    cand_run, cand_sigma = _write_case(tmp_path, "candidate", 1.01, 0.2)

    result = compare(ref_run, cand_run, ref_sigma, cand_sigma)

    assert result["candidate_rank"] == 8
    assert result["current_gate_applicable"] is True
    assert result["lead_current_relative_error"] == pytest.approx(0.01)
    assert result["spectral_current_l1_error"] == pytest.approx(0.01)
    assert result["sigma_lesser_relative_error"] > 0.09
    assert result["additional_sigma_lesser_negativity"] > 0.07


def test_current_gate_requires_two_converged_fixed_points(tmp_path):
    ref_run, ref_sigma = _write_case(tmp_path, "reference", 1.0, 0.1)
    cand_run, cand_sigma = _write_case(tmp_path, "candidate", 1.01, 0.2)
    with np.load(cand_run) as old:
        data = {key: old[key] for key in old.files}
    data["converged"] = False
    np.savez(cand_run, **data)

    result = compare(ref_run, cand_run, ref_sigma, cand_sigma)

    assert result["current_gate_applicable"] is False
    assert result["lead_current_relative_error"] is None
    assert result["spectral_current_l1_error"] is None


def test_load_sigma_concatenates_stack_rank_slices(tmp_path):
    for rank, start in enumerate((0.0, 2.0)):
        arrays = {
            key: np.full((2, 1, 4), start + offset, complex)
            for offset, key in enumerate(MODULE.SIGMA_KEYS)
        }
        np.savez(tmp_path / f"sigma.rank{rank}.npz", **arrays)

    got = load_sigma(tmp_path / "sigma")
    assert got["sigma_lesser"].shape == (4, 1, 4)
    np.testing.assert_array_equal(
        got["sigma_lesser"][:, 0, 0], [0.0, 0.0, 2.0, 2.0]
    )
