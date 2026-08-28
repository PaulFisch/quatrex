"""Tests for the private nonuniform-grid complexity study."""

import numpy as np

from studies import _nonuniform_grid_review as N


def test_lorentzian_bubble_oracle_matches_fine_fft():
    model = N.LorentzianMixture(
        np.array([0.7, 1.2]), np.array([0.08, 0.11]), np.array([1.0, 0.4]))
    grid = N.uniform_grid(12.0, 0.0025)
    out_w, got = N.fft_bubble(grid, model.eval(grid))
    want = model.bubble(out_w)
    live = np.abs(out_w) < 4.0
    assert N._relative_l2(got[live], want[live], out_w[1] - out_w[0]) < 2e-4


def test_energy_adjoint_preserves_pairing_but_sampling_does_not():
    primary = np.unique(np.concatenate((
        np.linspace(0.0, 4.0, 19), np.linspace(0.7, 1.3, 27))))
    auxiliary = np.linspace(0.0, 4.0, 321)
    adjoint, sample = N.energy_pairing_defect(primary, auxiliary)
    assert adjoint < 2e-14
    assert sample > 1e-3


def test_adaptive_primary_size_grows_only_logarithmically_with_inverse_width():
    sizes = []
    for width in (0.25, 0.01, 0.00025):
        model = N.LorentzianMixture(
            np.array([1.0 - width, 1.0 + width]),
            np.array([width, 1.15 * width]), np.array([1.0, 0.6]))
        sizes.append(N.adaptive_grid(model, 4.0, 0.25).size)
    # A 1000x linewidth reduction costs only about 3x more primary nodes;
    # contrast the 1000x uniform auxiliary growth checked below.
    assert max(sizes) / min(sizes) < 3.5


def test_fine_auxiliary_is_accurate_but_keeps_inverse_linewidth_cost():
    result = N.run_sweep()
    fine = result["summary"]["eight_per_hwhm"]
    assert max(v["max_bubble_relative_l2"] for v in fine.values()) < 2e-2
    assert max(v["max_peak_area_error"] for v in fine.values()) < 3e-2
    assert fine["0.001"]["max_auxiliary_points"] > 500 * fine["1.0"][
        "max_auxiliary_points"]
    assert fine["0.001"]["primary_points"] < 3.5 * fine["1.0"][
        "primary_points"]


def test_coarse_auxiliary_fails_for_subcell_lines():
    result = N.run_sweep()
    coarse = result["summary"]["background"]
    assert coarse["0.001"]["max_bubble_relative_l2"] > 0.5
    assert coarse["0.001"]["max_peak_area_error"] > 0.5


def test_nonuniform_p1_product_integration_is_exact_for_hat():
    grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    values = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    output = np.linspace(-2.0, 2.0, 17)
    got = N.p1_product_integration(grid, values, output)
    # Independent high-resolution rectangle oracle for the triangular hat.
    x = np.linspace(-2.0, 2.0, 200001)
    hat = np.maximum(1.0 - np.abs(x), 0.0)
    want = np.array([np.trapezoid(
        hat * np.maximum(1.0 - np.abs(z - x), 0.0), x) for z in output])
    assert np.allclose(got, want, rtol=2e-9, atol=2e-9)


def test_direct_nonuniform_bubble_stays_accurate_as_line_narrows():
    result = N.run_sweep()
    direct = result["direct_nonuniform_summary"]
    assert max(v["max_bubble_relative_l2"] for v in direct.values()) < 2e-2
    assert max(v["max_peak_area_error"] for v in direct.values()) < 3e-2
    assert direct["0.001"]["primary_points"] < 3.5 * direct["1.0"][
        "primary_points"]
