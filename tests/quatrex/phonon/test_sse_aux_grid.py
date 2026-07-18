# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Tests for the auxiliary-uniform-grid (non-uniform primary grid) SSE path.

The bubble FFT + bosonic fold only exist on a uniform, zero-anchored
grid. With ``sse_aux_grid_dw_thz > 0`` the legs are linearly
interpolated from the (possibly non-uniform) primary grid onto an
auxiliary uniform grid, the FFT pipeline runs there, and the outputs
are sampled back. Pinned here:

* aux path == legacy path when the aux grid coincides with a uniform
  primary grid (the bridge degenerates to the identity);
* a non-uniform primary grid clustered on the spectral peaks reproduces
  the fine-uniform-reference self-energy at the primary points;
* a non-uniform grid without the aux grid refuses to run;
* the quadrature-cell-width helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from qttools import xp
from qttools.comm import comm as _qtt_comm


def _configure_serial_comm() -> None:
    if _qtt_comm._is_configured:
        return
    backend = "device_mpi" if xp.__name__ == "numpy" else "host_mpi"
    cfg = {k: backend for k in ("all_to_all", "all_gather", "all_reduce", "bcast")}
    _qtt_comm.configure(
        block_comm_size=1,
        block_comm_config=cfg,
        stack_comm_config=cfg,
        override=True,
    )


def setup_module() -> None:  # pytest hook
    _configure_serial_comm()


def _make_cfg(retarded_method: str = "fft", aux_dw: float = 0.0,
              aux_fmax: float = 0.0, aux_restrict: str = "adjoint"):
    """Minimal mock config (cf. test_sse_phonon_phonon._make_cfg)."""

    class _Phonon:
        pass

    _Phonon.retarded_method = retarded_method
    _Phonon.fc3_path = None
    _Phonon.sse_tau_chunk_bytes = 4096
    _Phonon.sse_aux_grid_dw_thz = aux_dw
    _Phonon.sse_aux_grid_fmax_thz = aux_fmax
    _Phonon.sse_aux_restrict = aux_restrict

    class _Cfg:
        phonon = _Phonon()

    return _Cfg()


def _run_production(freqs: np.ndarray, phi_blocks, block_sizes,
                    gl_band, gg_band, cfg):
    """Run SigmaPhononPhonon.compute on the given grid; return the
    (I, J) -> Sigma^{<,>,R} block dicts."""
    from qttools.datastructures import DSDBCOO
    from scipy.sparse import csr_matrix
    from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

    n_blocks = len(block_sizes)
    N = int(np.sum(block_sizes))
    ne = int(freqs.shape[0])
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    rows, cols = [], []
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            for i in range(block_sizes[I]):
                for j in range(block_sizes[J]):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = csr_matrix(
        (np.ones(len(rows), np.complex128), (np.array(rows), np.array(cols))),
        shape=(N, N),
    )
    mk = lambda: DSDBCOO.from_sparray(
        pattern, np.asarray(block_sizes), global_stack_shape=(ne,))
    g_l, g_g, s_l, s_g, s_r = mk(), mk(), mk(), mk(), mk()
    for m in (g_l, g_g, s_l, s_g, s_r):
        m.data[:] = 0.0
    glv, ggv = g_l.stack[...], g_g.stack[...]
    for (K, Kp) in gl_band:
        glv.blocks[K, Kp] = gl_band[(K, Kp)]
        ggv.blocks[K, Kp] = gg_band[(K, Kp)]

    ssp = SigmaPhononPhonon(
        cfg, phonon_frequencies=freqs,
        block_sizes=np.asarray(block_sizes), phi_blocks=phi_blocks,
    )
    ssp.compute(g_l, g_g, out=(s_l, s_g, s_r))

    out_l, out_g, out_r = {}, {}, {}
    slv, sgv, srv = s_l.stack[...], s_g.stack[...], s_r.stack[...]
    for I in range(n_blocks):
        for J in range(max(0, I - 1), min(n_blocks, I + 2)):
            out_l[(I, J)] = np.asarray(slv.blocks[I, J])
            out_g[(I, J)] = np.asarray(sgv.blocks[I, J])
            out_r[(I, J)] = np.asarray(srv.blocks[I, J])
    return out_l, out_g, out_r


def _random_bands(rng, n_blocks, nbs, ne):
    gl, gg = {}, {}
    for K in range(n_blocks):
        for Kp in range(max(0, K - 1), min(n_blocks, K + 2)):
            gl[(K, Kp)] = (rng.standard_normal((ne, nbs, nbs))
                           + 1j * rng.standard_normal((ne, nbs, nbs)))
            gg[(K, Kp)] = (rng.standard_normal((ne, nbs, nbs))
                           + 1j * rng.standard_normal((ne, nbs, nbs)))
    return gl, gg


def _random_phi(rng, n_blocks, nbs):
    phi = {}
    for I in range(n_blocks):
        for K1 in range(max(0, I - 1), min(n_blocks, I + 2)):
            for K2 in range(max(0, I - 1), min(n_blocks, I + 2)):
                if abs(K1 - K2) > 1:
                    continue
                phi[(I, K1, K2)] = (rng.standard_normal((nbs, nbs, nbs))
                                    + 1j * rng.standard_normal((nbs, nbs, nbs)))
    return phi


@pytest.mark.parametrize("restrict", ["adjoint", "sample"])
@pytest.mark.parametrize("retarded", ["half", "fft"])
def test_aux_matches_legacy_on_matching_grid(retarded: str,
                                             restrict: str) -> None:
    """aux grid == uniform primary grid: the bridge is the identity and
    the aux path must reproduce the legacy path to rounding (both
    restriction modes degenerate to the identity)."""
    rng = np.random.default_rng(5)
    n_blocks, nbs, ne = 3, 3, 21
    block_sizes = [nbs] * n_blocks
    freqs = np.linspace(0.0, 16.0, ne)
    dw = float(freqs[1] - freqs[0])
    phi = _random_phi(rng, n_blocks, nbs)
    gl, gg = _random_bands(rng, n_blocks, nbs, ne)

    ref = _run_production(freqs, phi, block_sizes, gl, gg,
                          _make_cfg(retarded))
    aux = _run_production(freqs, phi, block_sizes, gl, gg,
                          _make_cfg(retarded, aux_dw=dw,
                                    aux_restrict=restrict))

    for r, a, name in zip(ref, aux, ("Sigma^<", "Sigma^>", "Sigma^R")):
        for key in r:
            np.testing.assert_allclose(
                a[key], r[key], atol=1e-38, rtol=1e-9,
                err_msg=f"{name} aux/legacy mismatch at block {key}",
            )


def _lorentz_bands(freqs, centers, width, amps_l, amps_g):
    """Smooth Lorentzian-comb legs evaluated on an arbitrary grid: the
    (K,K') band entries are shared complex mixtures of the same comb, so
    both grids sample the SAME analytic function."""
    w = np.asarray(freqs, dtype=float)
    comb = np.stack(
        [width**2 / ((w - c) ** 2 + width**2) for c in centers], axis=0
    )  # (n_lines, ne)
    gl, gg = {}, {}
    for key, al in amps_l.items():
        gl[key] = np.einsum("sw,sij->wij", comb, al)
        gg[key] = np.einsum("sw,sij->wij", comb, amps_g[key])
    return gl, gg


@pytest.mark.parametrize("restrict", ["adjoint", "sample"])
def test_nonuniform_matches_fine_uniform_reference(restrict: str) -> None:
    """A non-uniform primary grid clustered on the spectral lines +
    auxiliary grid reproduces the fine-uniform-reference Sigma at the
    primary points (leg-interpolation error, second order in the local
    spacing). "sample" is compared pointwise; "adjoint" against the
    SAME hat-averaging applied to the reference (its pointwise values
    at unresolved combination peaks are conservation-correct averages
    by design)."""
    rng = np.random.default_rng(9)
    n_blocks, nbs = 2, 2
    block_sizes = [nbs] * n_blocks
    fmax, width = 16.0, 0.25
    centers = [2.5, 3.1, 9.7, 13.4]

    # Shared analytic band content.
    amps_l, amps_g = {}, {}
    for K in range(n_blocks):
        for Kp in range(max(0, K - 1), min(n_blocks, K + 2)):
            amps_l[(K, Kp)] = (rng.standard_normal((len(centers), nbs, nbs))
                               + 1j * rng.standard_normal((len(centers), nbs, nbs)))
            amps_g[(K, Kp)] = (rng.standard_normal((len(centers), nbs, nbs))
                               + 1j * rng.standard_normal((len(centers), nbs, nbs)))
    phi = _random_phi(rng, n_blocks, nbs)

    # Fine uniform reference (legacy path).
    dw_fine = width / 8.0
    ne_fine = int(round(fmax / dw_fine)) + 1
    freqs_fine = np.linspace(0.0, fmax, ne_fine)
    gl_f, gg_f = _lorentz_bands(freqs_fine, centers, width, amps_l, amps_g)
    ref_l, ref_g, ref_r = _run_production(
        freqs_fine, phi, block_sizes, gl_f, gg_f, _make_cfg("fft"))

    # Non-uniform primary grid: dense patches around the lines + a coarse
    # background, ~5x fewer points than the reference.
    pieces = [np.linspace(0.0, fmax, 40)]
    for c in centers:
        pieces.append(np.linspace(c - 8 * width, c + 8 * width, 45))
    freqs_nu = np.unique(np.concatenate(pieces))
    freqs_nu = freqs_nu[(freqs_nu >= 0.0) & (freqs_nu <= fmax)]
    assert freqs_nu.size < 0.45 * ne_fine
    gl_n, gg_n = _lorentz_bands(freqs_nu, centers, width, amps_l, amps_g)
    nu_l, nu_g, nu_r = _run_production(
        freqs_nu, phi, block_sizes, gl_n, gg_n,
        _make_cfg("fft", aux_dw=dw_fine, aux_fmax=fmax,
                  aux_restrict=restrict))

    if restrict == "adjoint":
        # The reference grid IS the aux grid (same dw and span): apply
        # the identical restriction operator to the reference.
        from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

        ssp = SigmaPhononPhonon(
            _make_cfg("half", aux_dw=dw_fine, aux_fmax=fmax),
            phonon_frequencies=freqs_nu,
            block_sizes=np.array([nbs]),
            phi_blocks={(0, 0, 0): np.zeros((nbs, nbs, nbs), complex)},
        )
        aux, _, r_plan = ssp._aux_grid_plan(
            ssp._full_frequencies(freqs_nu.size))
        np.testing.assert_allclose(np.asarray(aux), freqs_fine, atol=1e-9)

        def _interp_ref(block):
            out = np.asarray(
                ssp._restrict_from_aux(xp.asarray(block), r_plan))
            out[0] = 0.0  # the production out-masks the omega=0 bin
            return out
    else:
        def _interp_ref(block):  # reference sampled at the nu points
            out = np.empty((freqs_nu.size,) + block.shape[1:],
                           dtype=complex)
            for i in range(block.shape[1]):
                for j in range(block.shape[2]):
                    out[:, i, j] = (
                        np.interp(freqs_nu, freqs_fine, block[:, i, j].real)
                        + 1j * np.interp(freqs_nu, freqs_fine,
                                         block[:, i, j].imag))
            return out

    for ref, nu, name, tol in (
        (ref_l, nu_l, "Sigma^<", 0.02),
        (ref_g, nu_g, "Sigma^>", 0.02),
        (ref_r, nu_r, "Sigma^R", 0.03),
    ):
        scale = max(np.abs(np.asarray(list(ref.values()))).max(), 1e-300)
        for key in ref:
            err = np.abs(nu[key] - _interp_ref(ref[key])).max() / scale
            assert err < tol, (
                f"{name} block {key}: non-uniform vs fine-uniform "
                f"reference error {err:.3e} exceeds {tol}"
            )


def test_dc_bin_not_smeared_by_interpolation() -> None:
    """The masked primary omega=0 bin (near-singular acoustic spectral
    peak) must be excluded from the interpolation source: legacy zeroes
    it before the FFT, so a huge G(0) must not leak into the aux bins."""
    rng = np.random.default_rng(3)
    n_blocks, nbs = 2, 2
    block_sizes = [nbs] * n_blocks
    # Coarse-at-DC non-uniform grid: aux bins fall between 0 and the
    # first positive primary point.
    freqs = np.concatenate(([0.0], np.linspace(0.9, 16.0, 30)))
    phi = _random_phi(rng, n_blocks, nbs)
    gl, gg = _random_bands(rng, n_blocks, nbs, freqs.size)
    gl_hot = {k: v.copy() for k, v in gl.items()}
    gg_hot = {k: v.copy() for k, v in gg.items()}
    for v in gl_hot.values():
        v[0] = 1e6  # the DC pole
    for v in gg_hot.values():
        v[0] = 1e6

    cfg = _make_cfg("half", aux_dw=0.3)
    ref = _run_production(freqs, phi, block_sizes, gl, gg, cfg)
    hot = _run_production(freqs, phi, block_sizes, gl_hot, gg_hot,
                          _make_cfg("half", aux_dw=0.3))
    for r, h, name in zip(ref, hot, ("Sigma^<", "Sigma^>", "Sigma^R")):
        for key in r:
            np.testing.assert_allclose(
                h[key], r[key], atol=1e-38, rtol=1e-12,
                err_msg=f"{name} {key}: DC bin leaked through the bridge",
            )


def test_nonuniform_without_aux_grid_raises() -> None:
    """A non-uniform primary grid must refuse the legacy FFT path."""
    rng = np.random.default_rng(1)
    n_blocks, nbs = 2, 2
    freqs = np.sort(rng.uniform(0.0, 16.0, 15))
    freqs[0] = 0.0
    phi = _random_phi(rng, n_blocks, nbs)
    gl, gg = _random_bands(rng, n_blocks, nbs, freqs.size)
    with pytest.raises(ValueError, match="uniform frequency grid"):
        _run_production(freqs, phi, [nbs] * n_blocks, gl, gg,
                        _make_cfg("half"))


def test_aux_fmax_extends_convolution_support() -> None:
    """sse_aux_grid_fmax_thz extends the aux grid beyond the primary top
    (the [omega_max, 2*omega_max] KK support without Dyson solves)."""
    from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

    rng = np.random.default_rng(2)
    nbs, ne = 2, 31
    freqs = np.linspace(0.0, 16.0, ne)
    phi = {(0, 0, 0): rng.standard_normal((nbs, nbs, nbs)) + 0j}
    ssp = SigmaPhononPhonon(
        _make_cfg("half", aux_dw=0.5, aux_fmax=32.0,
                  aux_restrict="sample"),
        phonon_frequencies=freqs, block_sizes=np.array([nbs]),
        phi_blocks=phi,
    )
    aux, p_plan, r_plan = ssp._aux_grid_plan(ssp._full_frequencies(ne))
    assert float(aux[-1]) >= 32.0
    assert float(aux[1] - aux[0]) == 0.5
    # P: linear functions are reproduced exactly inside the primary span,
    # and zeroed beyond it (G has no support there).
    lin = (2.0 * np.asarray(ssp._full_frequencies(ne)) + 1.0)[:, None]
    interp = np.asarray(ssp._interp_axis0(xp.asarray(lin + 0j), p_plan))
    inside = np.asarray(aux) <= 16.0 + 1e-12
    np.testing.assert_allclose(
        interp[inside, 0].real, 2.0 * np.asarray(aux)[inside] + 1.0,
        rtol=1e-12)
    assert np.all(interp[~inside] == 0.0)
    # R ("sample"): sampling back an aux-grid linear function is exact.
    lin_aux = (3.0 * np.asarray(aux) - 0.5)[:, None]
    back = np.asarray(ssp._restrict_from_aux(xp.asarray(lin_aux + 0j),
                                             r_plan))
    np.testing.assert_allclose(
        back[:, 0].real, 3.0 * np.asarray(freqs) - 0.5, rtol=1e-12)


def test_adjoint_restriction_conserves_pairing() -> None:
    """R = W_prim^-1 P^T W_aux transfers the aux-grid pairing exactly:
    sum_m w_m (R S)(m) G(m) == dw_aux sum_n S(n) (P G)(n) -- the
    discrete identity that keeps the dual-grid bubble Phi-derivable."""
    from quatrex.grid.energies import frequency_cell_widths
    from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon

    rng = np.random.default_rng(4)
    nbs = 2
    freqs = np.unique(np.concatenate(
        [np.linspace(0.0, 16.0, 12), np.linspace(4.0, 6.0, 25)]))
    phi = {(0, 0, 0): rng.standard_normal((nbs, nbs, nbs)) + 0j}
    ssp = SigmaPhononPhonon(
        _make_cfg("half", aux_dw=0.11, aux_fmax=20.0),
        phonon_frequencies=freqs, block_sizes=np.array([nbs]),
        phi_blocks=phi,
    )
    aux, p_plan, r_plan = ssp._aux_grid_plan(
        ssp._full_frequencies(freqs.size))
    assert r_plan[0] == "adjoint"
    ne_aux = int(np.asarray(aux).size)
    sig = (rng.standard_normal((ne_aux, 3))
           + 1j * rng.standard_normal((ne_aux, 3)))
    g = (rng.standard_normal((freqs.size, 3))
         + 1j * rng.standard_normal((freqs.size, 3)))
    w_prim = np.asarray(frequency_cell_widths(freqs))
    dw = float(aux[1] - aux[0])
    lhs = np.sum(w_prim[:, None]
                 * np.asarray(ssp._restrict_from_aux(xp.asarray(sig),
                                                     r_plan)) * g)
    pg = np.asarray(ssp._interp_axis0(xp.asarray(g), p_plan))
    rhs = dw * np.sum(sig * pg)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


def test_frequency_cell_widths() -> None:
    from quatrex.grid.energies import frequency_cell_widths, is_uniform_grid

    uni = np.linspace(0.0, 10.0, 21)
    cw = np.asarray(frequency_cell_widths(uni))
    np.testing.assert_allclose(cw, 0.5)  # every bin, edges included
    assert is_uniform_grid(uni)

    nu = np.array([0.0, 0.1, 0.15, 0.4, 1.0])
    cw = np.asarray(frequency_cell_widths(nu))
    np.testing.assert_allclose(cw, [0.1, 0.075, 0.15, 0.425, 0.6])
    assert not is_uniform_grid(nu)
    # The cell widths tile the axis (plus one half-gap overhang per edge).
    assert cw.sum() == pytest.approx(
        (nu[-1] - nu[0]) + 0.5 * (nu[1] - nu[0]) + 0.5 * (nu[-1] - nu[-2]))
