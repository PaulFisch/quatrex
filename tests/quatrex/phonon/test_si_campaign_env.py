"""Environment-boundary tests for the Si Daint campaign driver."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "_si_env_aliases", ROOT / "phonon/studies/engine/env_aliases.py")


def _module():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


def test_campaign_aliases_populate_effective_driver_names() -> None:
    module = _module()
    env = {
        "QX_RETARDED_METHOD": "fft",
        "QX_MIX_METHOD": "anderson",
        "QX_MIXING": "0.3",
        "QX_SSE_VERTEX_SCALE": "0.5",
        "QX_BUBBLE_BALANCE_CHECK": "1",
        "QX_TAU_CHUNK_BYTES": "8589934592",
    }
    module.normalise_env(env)
    assert env["QX_RETARDED"] == "fft"
    assert env["QX_MIXMETHOD"] == "anderson"
    assert env["QX_MIX"] == "0.3"
    assert env["QX_VSCALE"] == "0.5"
    assert env["QX_BBCHECK"] == "1"
    assert env["QX_TAUCHUNK"] == "8589934592"


def test_campaign_alias_conflict_is_fatal() -> None:
    module = _module()
    env = {"QX_RETARDED_METHOD": "fft", "QX_RETARDED": "half"}
    with pytest.raises(ValueError, match="conflicting environment overrides"):
        module.normalise_env(env)


def test_restartable_campaign_requires_final_and_live_checkpoints() -> None:
    module = _module()
    with pytest.raises(ValueError, match="QX_SAVE_SIGMA"):
        module.validate_restartable_env({"QX_REQUIRE_RESTARTABLE": "1"})

    env = {
        "QX_REQUIRE_RESTARTABLE": "1",
        "QX_SAVE_SIGMA": "/tmp/sigma.npz",
        "QX_SAVE_SIGMA_BEST": "/tmp/sigma_best.npz",
        "QX_SIGMA_BEST_LIVE": "1",
    }
    module.validate_restartable_env(env)


@pytest.mark.parametrize("raw, expected", [(None, 1), ("1", 1), ("5", 5)])
def test_live_best_checkpoint_stride(raw, expected) -> None:
    module = _module()
    env = {} if raw is None else {"QX_SIGMA_BEST_LIVE_STRIDE": raw}
    assert module.best_checkpoint_stride(env) == expected


@pytest.mark.parametrize("raw", ["0", "-2", "1.5", "many"])
def test_live_best_checkpoint_stride_rejects_invalid_values(raw) -> None:
    module = _module()
    with pytest.raises(ValueError, match="positive integer"):
        module.best_checkpoint_stride({"QX_SIGMA_BEST_LIVE_STRIDE": raw})


def test_affine_sigma_restart_terms_preserve_legacy_and_add_secant() -> None:
    module = _module()
    assert module.sigma_restart_terms({
        "QX_SIGMA_INIT": "b.npz", "QX_SIGMA_SCALE": "1.25",
    }) == [("b.npz", 1.25)]
    assert module.sigma_restart_terms({
        "QX_SIGMA_INIT": "b.npz", "QX_SIGMA_SCALE": "2",
        "QX_SIGMA_INIT_SECOND": "a.npz",
        "QX_SIGMA_SCALE_SECOND": "-1",
    }) == [("b.npz", 2.0), ("a.npz", -1.0)]


@pytest.mark.parametrize("env", [
    {"QX_SIGMA_INIT_SECOND": "a.npz"},
    {"QX_SIGMA_INIT": "b.npz", "QX_SIGMA_SCALE_SECOND": "-1"},
])
def test_affine_sigma_restart_rejects_incomplete_pairs(env) -> None:
    module = _module()
    with pytest.raises(ValueError, match="requires"):
        module.sigma_restart_terms(env)
