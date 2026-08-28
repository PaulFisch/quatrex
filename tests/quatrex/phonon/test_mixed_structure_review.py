"""Tests for the material-blind mixed-representation selector."""

import h5py
import numpy as np

from studies import _mixed_structure_review as M


def test_linear_reconstruction_is_exact_for_linear_vector_function():
    x = np.linspace(0.0, 1.0, 17)
    values = np.column_stack((2.0 * x + 1.0, -3.0 * x + 4.0))
    got = M.linear_reconstruction(x, values, np.array([0, 16]))
    np.testing.assert_allclose(got, values, rtol=0, atol=1e-14)


def test_adaptive_grid_targets_a_local_feature():
    x = np.linspace(0.0, 1.0, 129)
    broad = np.exp(-((x - 0.25) / 0.18) ** 2)
    narrow = np.exp(-((x - 0.73) / 0.012) ** 2)
    values = np.column_stack((broad, narrow))
    weights = np.full(x.size, x[1] - x[0])
    adaptive = M.adaptive_knots(x, values, weights, (1e-2,))
    uniform = M.uniform_knots(x, values, weights, (1e-2,))
    assert adaptive["at_tolerance"]["0.01"]["points"] < uniform[
        "at_tolerance"]["0.01"]["points"]


def test_fc3_reach_density_and_full_band_reblock(tmp_path):
    path = tmp_path / "fc3.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("meta/keys", data=np.array([[0, 0, 0], [1, 2, 1]]))
        f.create_dataset("meta/block_sizes", data=np.array([2, 2, 2]))
        group = f.create_group("fc3_blocks")
        a = np.zeros((2, 2, 2))
        a.flat[:2] = 1.0
        group.create_dataset("0_0_0", data=a)
        group.create_dataset("1_2_1", data=a)
    row = M.fc3_gate(str(path))
    assert row["transport_reach"] == 1
    assert row["stored_density"] == 0.25
    # A size-3 BTD reblock reaches distance five, but misses some distance-5
    # pairs.  Covering every pair in a full band of radius five needs c=5.
    assert M.minimum_reblock_for_full_band(2) == 2
    assert M.minimum_reblock_for_full_band(5) == 5


def test_reference_gate_refuses_nonconverged_run(tmp_path):
    path = tmp_path / "run.npz"
    np.savez(path, converged=False, diverged=False, n_iter=20, eta=0.0,
             lead_current=1.0, internal_spread=0.1,
             final_heat=np.ones(4), final_bubble_balance=np.array([1.0, 1.0]))
    row = M.reference_gate(str(path))
    assert not row["certified"]
    assert "not converged" in row["reasons"][0]


def test_reference_gate_uses_saved_scalar_when_profile_is_partial(tmp_path):
    path = tmp_path / "run.npz"
    np.savez(path, converged=True, diverged=False, n_iter=12, eta=0.0,
             lead_current=2.0, internal_spread=5e-4,
             final_heat=np.array([2.0, np.nan, np.nan, 2.0]))
    row = M.reference_gate(str(path))
    assert row["certified"]
    assert row["current_profile_finite_fraction"] == 0.5


def test_frequency_pilot_drops_only_unavailable_nan_channels(tmp_path):
    path = tmp_path / "run.npz"
    w = np.linspace(0.0, 1.0, 9)
    current = np.column_stack((w, np.full_like(w, np.nan)))
    gr = np.stack((w + 1.0, 2.0 * w + 1.0), axis=-1)[:, None, :]
    np.savez(path, energies=w, frequency_cell_widths=np.gradient(w),
             current_spectrum=current, gr_diag_imag=gr)
    got_w, got_weights, channels = M._frequency_channels(str(path))
    np.testing.assert_array_equal(got_w, w)
    np.testing.assert_array_equal(got_weights, np.gradient(w))
    assert channels.shape == (w.size, 2)
    assert np.all(np.isfinite(channels))


def test_pole_gate_records_width_and_source_variation(tmp_path):
    path = tmp_path / "poles.npz"
    np.savez(path,
             poles=np.array([1.0 - 0.01j, 2.0 - 0.2j]),
             pole_offsets=np.array([0, 1, 2]),
             local_frequency_weights=np.full(5, 0.1),
             source_fit=np.array([0.05, 0.2]))
    row = M.pole_gate(str(path))
    assert row["clusters"] == 2
    np.testing.assert_allclose(row["gamma_over_h"]["min"], 0.1)
    assert row["source_variation"]["fraction_at_or_below_0.1"] == 0.5
