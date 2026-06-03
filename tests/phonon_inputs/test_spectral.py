# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Unit tests for the spectral-function / band-renormalisation postproc.

``postproc.spectral`` builds ``A(q, omega) = -1/pi Im Tr G^R`` and the decomposed
quasiparticle bands (bare -> +loop -> +tadpole). These tests pin the band
frequencies, the spectral peak positions, the loop stiffening and the bubble
broadening on synthetic dynamical matrices (no phonopy).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PHONON = Path(__file__).resolve().parents[2] / "phonon"
if str(_PHONON) not in sys.path:
    sys.path.insert(0, str(_PHONON))

from postproc.spectral import (  # noqa: E402
    band_renormalization_bundle,
    decomposition_bands,
    frequencies_from_dynamical,
    quasiparticle_bands,
    region_projected_phi_eff,
    spectral_function_qw,
)


def test_frequencies_signed_sqrt():
    dyn = np.diag([16.0, 36.0, -4.0])            # omega^2 = 16, 36, -4
    f = frequencies_from_dynamical(dyn)
    np.testing.assert_allclose(np.sort(f), [-2.0, 4.0, 6.0], atol=1e-10)


def test_spectral_peaks_at_bare_frequencies():
    """A(q, omega) peaks at the eigenfrequencies of D when Sigma = 0."""
    omegas = np.array([4.0, 6.0, 8.0])
    D = np.diag(omegas ** 2)[None]               # (1, 3, 3)
    grid = np.linspace(0.5, 10.0, 2000)
    A = spectral_function_qw(D, grid, eta_w_thz=0.03)
    assert np.all(A >= -1e-9)                     # spectral weight >= 0 for w>0
    # each bare frequency is a local maximum of A
    for w0 in omegas:
        win = np.abs(grid - w0) < 0.5
        peak = grid[win][np.argmax(A[0][win])]
        assert abs(peak - w0) < 0.05


def test_quasiparticle_loop_stiffens():
    """A positive-definite static self-energy raises every branch frequency."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 4))
    D = (a @ a.T + 5 * np.eye(4))[None]          # SPD dynamical matrix
    sig = 3.0 * np.eye(4)                         # positive loop shift
    bare = quasiparticle_bands(D)
    loop = quasiparticle_bands(D, sigma_static=sig)
    assert np.all(loop[0] > bare[0] - 1e-9)
    assert loop[0].min() > bare[0].min() + 1e-6


def test_decomposition_band_sets():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3, 3))
    D = (a @ a.T + 4 * np.eye(3))[None]
    sl = 2.0 * np.eye(3)
    st = 0.1 * (lambda m: m + m.T)(rng.standard_normal((3, 3)))
    bands = decomposition_bands(D, sigma_loop=sl, sigma_tadpole=st)
    assert set(bands) == {"bare", "loop", "loop_tadpole"}
    # loop set stiffer than bare
    assert np.all(bands["loop"][0] >= bands["bare"][0] - 1e-9)
    assert bands["bare"].shape == (1, 3)


def test_bubble_broadens_peak():
    """A finite-Im bubble self-energy widens the spectral peak (FWHM up)."""
    w0 = 6.0
    D = np.array([[w0 ** 2]])[None]              # single mode, N=1
    grid = np.linspace(3.0, 9.0, 4000)

    def fwhm(A):
        half = A.max() / 2.0
        idx = np.where(A >= half)[0]
        return grid[idx[-1]] - grid[idx[0]]

    A0 = spectral_function_qw(D, grid, eta_w_thz=0.02)[0]
    # constant negative-imaginary bubble -> extra broadening (Gamma = -2 Im Sigma)
    sig_b = np.full((1, grid.size, 1, 1), -2.0j)  # Im Sigma_B < 0
    A1 = spectral_function_qw(D, grid, eta_w_thz=0.02, sigma_b=sig_b)[0]
    assert fwhm(A1) > 1.5 * fwhm(A0)
    # peak stays near w0 (no real shift)
    assert abs(grid[np.argmax(A1)] - w0) < 0.1


def test_region_projection_returns_block_and_freqs():
    n_dof, n_slabs = 3, 4
    rng = np.random.default_rng(2)
    blocks = [rng.standard_normal((n_dof, n_dof)) for _ in range(n_slabs)]
    phi = np.zeros((n_slabs * n_dof, n_slabs * n_dof))
    for l, b in enumerate(blocks):
        s = slice(l * n_dof, (l + 1) * n_dof)
        phi[s, s] = b @ b.T + np.eye(n_dof)
    block, freqs = region_projected_phi_eff(phi, n_dof, layer=2)
    assert block.shape == (n_dof, n_dof)
    np.testing.assert_allclose(
        np.sort(freqs),
        np.sort(frequencies_from_dynamical(phi[6:9, 6:9])), atol=1e-10)


def test_bundle_assembles_bands_and_spectrum(tmp_path):
    """`band_renormalization_bundle` produces consistent decomposition + A and
    round-trips through `save_spectral`."""
    from postproc.io import save_spectral

    rng = np.random.default_rng(8)
    nq = 6
    a = rng.standard_normal((3, 3))
    base = a @ a.T + 6 * np.eye(3)
    D_q = np.stack([base + 0.3 * np.sin(t) * np.eye(3)
                    for t in np.linspace(0, np.pi, nq)])
    sl = 2.0 * np.eye(3)
    grid = np.linspace(0.5, 12.0, 400)
    bundle = band_renormalization_bundle(
        D_q, grid, eta_w_thz=0.06, sigma_loop=sl)

    assert set(bundle["bands"]) == {"bare", "loop"}
    assert bundle["A"].shape == (nq, grid.size)
    # bundle's A used the static loop shift -> peaks track the loop bands
    iq = nq // 2
    peak = grid[np.argmax(bundle["A"][iq])]
    assert np.min(np.abs(peak - bundle["bands"]["loop"][iq])) < 0.1

    path = save_spectral(tmp_path / "b.npz", **bundle)
    d = np.load(path, allow_pickle=True)
    np.testing.assert_allclose(d["A"], bundle["A"])
    np.testing.assert_allclose(d["band_loop"], bundle["bands"]["loop"])


def test_io_roundtrip(tmp_path):
    from postproc.io import save_spectral, write_reference_plot_script

    q = np.linspace(0, 1, 5)
    w = np.linspace(0, 10, 7)
    A = np.random.default_rng(3).standard_normal((5, 7))
    bands = {"bare": np.random.default_rng(4).standard_normal((5, 3))}
    path = save_spectral(tmp_path / "spec.npz", q_distance=q,
                         omega_grid_thz=w, A=A, bands=bands,
                         tick_positions=[0.0, 1.0], tick_labels=["G", "X"])
    d = np.load(path, allow_pickle=True)
    np.testing.assert_allclose(d["A"], A)
    np.testing.assert_allclose(d["band_bare"], bands["bare"])
    assert list(d["tick_labels"]) == ["G", "X"]
    script = write_reference_plot_script(tmp_path / "plot.py")
    assert Path(script).exists()
