"""Regression tests for the independent Si Kramers--Kronig audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "_si_kk_audit", ROOT / "phonon/studies/_si_kk_audit.py"
)


def _module():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


def test_checkpoint_assembly_includes_q_mirror_and_dc_mask(tmp_path) -> None:
    module = _module()
    rng = np.random.default_rng(7)
    ne, nq, nnz = 25, 3, 4
    w = np.linspace(0.0, 6.0, ne)
    delta = rng.normal(size=(ne, nq, nnz))
    delta = 1j * delta * np.exp(-((w / 2.0) ** 4))[:, None, None]
    delta[0] = 0.0  # production masks the raw bubble before the KK transform
    lesser = 0.5 * delta
    greater = -0.5 * delta
    retarded = 0.5 * delta + 0.5j * module.hilbert_cell_constant(
        delta, w, (nq,)
    )
    retarded[0] = 0.0

    # Split unevenly, as a production stack communicator does.
    for rank, selection in enumerate((slice(0, 13), slice(13, None))):
        np.savez(
            tmp_path / f"sigma_best.rank{rank}.npz",
            sigma_lesser=lesser[selection],
            sigma_greater=greater[selection],
            sigma_retarded=retarded[selection],
        )
    got = module.audit_checkpoint(tmp_path, ne, 6.0, (nq,))
    assert got["assembly_max_relative"] < 1e-13
    assert got["edge_over_peak"] < 1e-10


def test_piecewise_linear_rule_improves_a_resolved_pole() -> None:
    module = _module()
    row = module.pole_error(0.125, 80.0, centre=5.0, halfwidth=0.5)
    assert row["linear_max_relative"] < 0.4 * row["constant_max_relative"]


def test_piecewise_linear_hat_kernel_has_second_order_convergence() -> None:
    module = _module()
    errors = []
    for n in (41, 81, 161):
        w = np.linspace(0.0, 1.0, n)
        delta = 1.0 - w**2
        got = module.hilbert_piecewise_linear(delta, w)
        selected = (w > 0.05) & (w < 0.95)
        logarithm = np.log((1.0 + w[selected]) / (1.0 - w[selected]))
        exact = ((1.0 - w[selected] ** 2) * logarithm
                 + 2.0 * w[selected]) / np.pi
        errors.append(float(np.abs(got[selected] - exact).max()))

    assert errors[1] < 0.3 * errors[0]
    assert errors[2] < 0.3 * errors[1]


def test_neither_rule_resolves_a_pole_much_narrower_than_a_cell() -> None:
    module = _module()
    row = module.pole_error(0.25, 80.0, centre=23.0, halfwidth=0.002)
    assert row["constant_l2_relative"] > 0.95
    assert row["linear_l2_relative"] > 0.95


def test_kms_defect_recognises_bosonic_detailed_balance() -> None:
    module = _module()
    rng = np.random.default_rng(19)
    w = np.linspace(0.0, 40.0, 161)
    greater = 1j * rng.normal(size=(w.size, 3, 5))
    factor = np.exp(
        -module.PLANCK_EV_PER_THz * w
        / (module.BOLTZMANN_EV_PER_K * 300.0)
    )[:, None, None]
    lesser = factor * greater

    exact = module.kms_defect(lesser, greater, w, 300.0)
    assert exact["global_max_relative"] < 1e-15
    assert exact["l2_relative"] < 1e-15

    wrong_temperature = module.kms_defect(lesser, greater, w, 250.0)
    assert wrong_temperature["global_max_relative"] > 1e-2
    assert wrong_temperature["l2_relative"] > 1e-2


def test_equilibrium_current_diagnostic_uses_the_frequency_measure(tmp_path) -> None:
    module = _module()
    w = np.linspace(0.0, 2.0, 5)
    widths = np.array([0.25, 0.5, 0.5, 0.5, 0.25])
    spectrum = np.zeros((5, 2, 2))
    spectrum[1, :, 0] = (1.0, -1.0)
    spectrum[2, :, 1] = (2.0, -2.0)
    last_heat = np.sum(
        spectrum * (w * widths)[:, None, None], axis=(0, 1)
    )
    path = tmp_path / "run.npz"
    np.savez(
        path, energies=w, frequency_cell_widths=widths,
        current_spectrum=spectrum, last_heat=last_heat,
        left_temperature=300.0, right_temperature=300.0,
    )

    got = module.equilibrium_current_defect(path)
    assert got["integrated_current"] == [0.0, 0.0]
    assert got["spectrum_integral_vs_saved_max_absolute"] == 0.0
    assert got["zero_current_over_absolute_spectral_budget"] == 0.0
