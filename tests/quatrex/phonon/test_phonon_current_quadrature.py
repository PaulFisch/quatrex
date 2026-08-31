from types import SimpleNamespace

import numpy as np

from qttools.utils.gpu_utils import get_host

from quatrex.core.scba import SCBA


def test_uniform_phonon_current_weights_include_cell_measure():
    solver = SimpleNamespace(
        local_frequencies=np.array([0.0, 0.25, 0.5]),
        local_frequency_weights=np.full(3, 0.25),
        uniform_frequency_grid=True,
    )
    got = np.asarray(get_host(SCBA._phonon_hw_weights(solver)))
    np.testing.assert_array_equal(got, [0.0, 0.0625, 0.125])


def test_nonuniform_phonon_current_weights_use_local_cells():
    solver = SimpleNamespace(
        local_frequencies=np.array([0.0, 0.2, 0.7]),
        local_frequency_weights=np.array([0.2, 0.35, 0.5]),
        uniform_frequency_grid=False,
    )
    got = np.asarray(get_host(SCBA._phonon_hw_weights(solver)))
    np.testing.assert_allclose(got, [0.0, 0.07, 0.35])


def test_missing_internal_currents_do_not_duplicate_lead_balance():
    balance, spread = SCBA._phonon_heat_flow_metrics([10.0, np.nan, 9.0])
    np.testing.assert_allclose(balance, 1.0 / 9.5)
    assert np.isnan(spread)


def test_complete_internal_current_spread_is_reported():
    balance, spread = SCBA._phonon_heat_flow_metrics([10.0, 8.0, 9.0])
    np.testing.assert_allclose(balance, 1.0 / 9.5)
    np.testing.assert_allclose(spread, 2.0 / 9.5)


def test_converged_checkpoint_uses_the_measured_iterate():
    solver = object.__new__(SCBA)
    current = [SimpleNamespace(data=np.array([i])) for i in range(3)]
    previous = [SimpleNamespace(data=np.array([i + 3])) for i in range(3)]
    solver.data = SimpleNamespace(
        sigma_lesser=current[0],
        sigma_greater=current[1],
        sigma_retarded_hermitian=current[2],
        sigma_lesser_prev=previous[0],
        sigma_greater_prev=previous[1],
        sigma_retarded_hermitian_prev=previous[2],
    )
    solver._converged = True
    solver._diverged = False
    assert solver._sigma_checkpoint_buffers() == tuple(previous)
    solver._converged = False
    assert solver._sigma_checkpoint_buffers() == tuple(current)


def test_max_iteration_checkpoint_uses_the_mixed_iterate():
    solver = object.__new__(SCBA)
    current = [SimpleNamespace(data=np.array([i])) for i in range(3)]
    solver.data = SimpleNamespace(
        sigma_lesser=current[0],
        sigma_greater=current[1],
        sigma_retarded_hermitian=current[2],
    )
    solver._converged = False
    solver._diverged = False
    assert solver._sigma_checkpoint_buffers() == tuple(current)
