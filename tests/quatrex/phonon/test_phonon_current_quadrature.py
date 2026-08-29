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
