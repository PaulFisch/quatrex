"""Tests for the parameter-validation framework.

Exercise each checker on three configs (pass / warn / fail) where possible,
plus end-to-end checks against the headline YAMLs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_PHONON = Path(__file__).resolve().parents[3] / "phonon"
sys.path.insert(0, str(_PHONON))


# --------------------------------------------------------------------------- #
# Per-checker tests on minimal SimpleNamespace cfgs                           #
# --------------------------------------------------------------------------- #


def _vasp_cfg(*, encut, potcar_map=None, kpoints_scf=(1, 1, 4)):
    return SimpleNamespace(
        vasp=SimpleNamespace(
            encut=encut,
            potcar_map=potcar_map or {"Si": "Si", "H": "H"},
            kpoints_scf=list(kpoints_scf),
        ),
        structure=SimpleNamespace(
            symbols=["Si", "Si"],
            lattice=[[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
        ),
    )


def test_vasp_encut_pass():
    from finite_analysis.parameter_validation import _check_vasp_encut

    res = _check_vasp_encut(_vasp_cfg(encut=400))
    assert res is not None and res.passed and res.severity == "info"


def test_vasp_encut_warn():
    from finite_analysis.parameter_validation import _check_vasp_encut

    # 1.3 × max(Si=245, H=250) = 325; encut=300 → warn
    res = _check_vasp_encut(_vasp_cfg(encut=300))
    assert res is not None and not res.passed and res.severity == "warn"


def test_vasp_encut_error():
    from finite_analysis.parameter_validation import _check_vasp_encut

    # encut < max ENMAX → error
    res = _check_vasp_encut(_vasp_cfg(encut=200))
    assert res is not None and not res.passed and res.severity == "error"


def test_kpoints_scf_finite_cell_warns_if_kx_gt_1():
    from finite_analysis.parameter_validation import _check_kpoints_scf

    finite_cfg = SimpleNamespace(
        vasp=SimpleNamespace(kpoints_scf=[2, 2, 2]),
        qe=None,
        structure=SimpleNamespace(
            symbols=["Si"],
            lattice=[[20, 0, 0], [0, 20, 0], [0, 0, 20]],  # all vacuum
        ),
    )
    res = _check_kpoints_scf(finite_cfg)
    assert res is not None and not res.passed and res.severity == "warn"


def test_kpoints_scf_periodic_pass():
    from finite_analysis.parameter_validation import _check_kpoints_scf

    cfg = SimpleNamespace(
        vasp=SimpleNamespace(kpoints_scf=[1, 1, 4]),
        qe=None,
        structure=SimpleNamespace(
            symbols=["Si"],
            lattice=[[20, 0, 0], [0, 20, 0], [0, 0, 5.43]],
        ),
    )
    res = _check_kpoints_scf(cfg)
    assert res is not None and res.passed


def _hiphive_cfg(*, n_structures=6, supercell=(1, 1, 1), cutoffs=(5.0, 4.0),
                 symbols=None):
    return SimpleNamespace(
        relax=SimpleNamespace(fc_method="hiphive"),
        hiphive=SimpleNamespace(
            n_structures=n_structures, supercell=list(supercell),
            cutoffs=list(cutoffs),
        ),
        structure=SimpleNamespace(symbols=symbols or ["Si"]),
    )


def test_hiphive_n_structures_warn():
    from finite_analysis.parameter_validation import _check_hiphive_n_structures

    cfg = _hiphive_cfg(n_structures=4, supercell=(1, 1, 4),
                       symbols=["Si"] * 53)
    res = _check_hiphive_n_structures(cfg)
    assert res is not None and not res.passed
    assert res.severity in ("warn", "error")


def test_hiphive_cutoffs_pass():
    from finite_analysis.parameter_validation import _check_hiphive_cutoffs

    cfg = _hiphive_cfg(cutoffs=(5.5, 4.5))
    res = _check_hiphive_cutoffs(cfg)
    assert res is not None and res.passed


def test_hiphive_cutoffs_warn():
    from finite_analysis.parameter_validation import _check_hiphive_cutoffs

    cfg = _hiphive_cfg(cutoffs=(3.0, 2.0))
    res = _check_hiphive_cutoffs(cfg)
    assert res is not None and not res.passed and res.severity == "warn"


def test_displacement_distance_warn_too_large():
    from finite_analysis.parameter_validation import _check_displacement_distance

    cfg = SimpleNamespace(thirdorder=SimpleNamespace(displacement_distance=0.10))
    res = _check_displacement_distance(cfg)
    assert res is not None and not res.passed


# --------------------------------------------------------------------------- #
# End-to-end check on headline configs                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rel", [
    "configs/si_primitive/prim_vasp.yaml",
    "configs/sinw/sinw100_vasp.yaml",
    "configs/chain/si_chain.yaml",
])
def test_validate_headline_configs_no_error(rel):
    from finite_analysis.parameter_validation import (
        validate_config, max_severity,
    )
    from phonon_inputs.config import config_from_dict

    path = _PHONON / rel
    if not path.exists():
        pytest.skip(f"config not present: {path}")
    cfg = config_from_dict(yaml.safe_load(path.read_text()))
    results = validate_config(cfg)
    sev = max_severity(results)
    # Headline configs may warn but must not error.
    assert sev in ("info", "warn"), (
        f"{rel}: max severity = {sev}\n"
        + "\n".join(
            f"  [{r.severity}] {r.check.key}: {r.actual!r} -- "
            f"{r.check.recommendation}"
            for r in results if r.severity != "info"
        )
    )


def test_render_table_no_color_smoke():
    from finite_analysis.parameter_validation import (
        Check, CheckResult, render_table,
    )

    rs = [
        CheckResult(Check("a.b", "msg", "rec"), passed=True, actual=1, severity="info"),
        CheckResult(Check("c.d", "msg", "rec"), passed=False, actual=2, severity="warn"),
    ]
    out = render_table(rs, color=False)
    assert "OK" in out and "WARN" in out
