"""Historical Si-film ledger extraction and classification gates."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "_si_census", ROOT / "phonon/scripts/si_run_census.py")


def _module():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


def test_artifact_scalar_and_conservation_extraction(tmp_path) -> None:
    module = _module()
    path = tmp_path / "run.npz"
    np.savez(
        path, energies=np.linspace(0.0, 35.0, 281), eta=0.0,
        retarded="fft", n_iter=12, lead_current=2.5,
        last_heat=np.array([2.50, 2.49, 2.50]), internal_spread=4e-4,
        final_bubble_balance=np.array([10.0, 10.0 + 1e-11]),
        converged=True, diverged=False, ballistic=False,
        sse_microblock_dof=6, sse_microblock_g_band=3,
        block_sizes=np.array([30, 12]),
    )
    got = module._artifact_values(path)
    assert got["sse_microblock_dof"] == 6
    assert got["block_sizes"].tolist() == [30, 12]
    assert got["lead_balance"] < 1e-12
    assert got["bubble_balance"] < 2e-12


def test_known_historical_and_certified_classifications() -> None:
    module = _module()
    old = {
        "scba_iterations": 57, "frequency_max_thz": 15.0,
        "eta_thz": 0.0, "microblock_dof": 0, "group_layout": "1,1,1,1,1",
        "retarded_method": "half", "converged": True,
    }
    assert module.classify(old)[0] == "frequency-truncated"

    failed = dict(old, diverged=True)
    assert module.classify(failed)[0] == "divergent"

    certified = {
        "scba_iterations": 18, "frequency_max_thz": 35.0,
        "eta_thz": 0.0, "microblock_dof": 6, "group_layout": "5,5",
        "retarded_method": "fft", "converged": True,
        "internal_spread": 8e-4,
    }
    assert module.classify(certified)[0] == "trustworthy"


def test_log_environment_overrides_toml() -> None:
    module = _module()
    cfg = {"phonon": {"sse_microblock_g_band": 2}}
    env = {"QX_MICRO_GBAND": "5"}
    assert module._cfg_value(
        cfg, env, "phonon", "sse_microblock_g_band") == 5
