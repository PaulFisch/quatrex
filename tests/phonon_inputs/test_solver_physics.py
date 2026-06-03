"""Physics regression tests for the dense SCBA phonon solver.

Locks in the invariants established by the solver-verification campaign
(``phonon/scripts/verify_*.py``):

  * the FFT 3-phonon bubble equals an explicit frequency convolution and
    conserves frequency (omega_out = omega_a + omega_b);
  * detailed balance ``Sigma^>(omega) = exp(hbar omega / kT) Sigma^<``
    at equilibrium -- the no-double-counting guarantee;
  * the raw bubble output is Keldysh-symmetric and anti-Hermitian;
  * the zero-padded FFT Hilbert transform is accurate (regression for
    the periodicity-error fix in ``solver/retarded.py``);
  * ``build_retarded`` "fft" and "pv" agree on a resolved grid and
    match the documented Kramers-Kronig relation;
  * ``cutoff=None`` reproduces an independent brute-force full sum;
  * the SCBA fixed-point loop runs, converges and conserves heat.

All tests run on the analytic toy systems in
``phonon.solver.toy_models`` and stay small enough for CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PHONON = Path(__file__).resolve().parents[2] / "phonon"
if str(_PHONON) not in sys.path:
    sys.path.insert(0, str(_PHONON))

from phonon_inputs.constants import HBAR_SI, KB_SI, THZ_TO_RAD  # noqa: E402
from solver.bubble import bubble_dense  # noqa: E402
from solver.grids import build_frequency_grid  # noqa: E402
from solver.retarded import build_retarded, hilbert_transform_axis  # noqa: E402
from solver.se_finite import (  # noqa: E402
    compute_phph_self_energy_finite_multi_slab,
)
from solver.toy_models import (  # noqa: E402
    diatomic_chain,
    equilibrium_lesser_greater,
    harmonic_green_retarded,
    single_oscillator,
    symmetric_cubic_vertex,
)


# ---------------------------------------------------------------------------
# Bubble: convolution + frequency conservation
# ---------------------------------------------------------------------------


def _brute_bubble(phi, g_a, g_b, prefactor, mid, ne):
    out = np.zeros((ne, phi.shape[0], phi.shape[0]), dtype=complex)
    for o in range(ne):
        n = o + mid
        for m in range(ne):
            mp = n - m
            if 0 <= mp < ne:
                out[o] += np.einsum("ace,ed,cb,Jdb->aJ",
                                    phi, g_b[mp], g_a[m], phi, optimize=True)
    return prefactor * out


def test_bubble_matches_explicit_convolution():
    rng = np.random.default_rng(0)
    ne, n_dof = 25, 3
    mid = ne // 2
    phi = symmetric_cubic_vertex(n_dof, rng)
    g_a = (rng.standard_normal((ne, n_dof, n_dof))
           + 1j * rng.standard_normal((ne, n_dof, n_dof)))
    g_b = (rng.standard_normal((ne, n_dof, n_dof))
           + 1j * rng.standard_normal((ne, n_dof, n_dof)))
    fft = bubble_dense(phi_left=phi, phi_right=phi, G_a=g_a, G_b=g_b,
                       n_fft=2 * ne - 1, prefactor=0.5j,
                       out_slice=slice(mid, mid + ne), zero_freq_idx=None)
    brute = _brute_bubble(phi, g_a, g_b, 0.5j, mid, ne)
    np.testing.assert_allclose(fft, brute, rtol=1e-10, atol=1e-12)


def test_bubble_conserves_frequency():
    """A delta probe must land the output at omega_a + omega_b."""
    ne = 31
    mid = ne // 2
    phi = np.ones((1, 1, 1), dtype=complex)
    i_a, i_b = mid + 6, mid + 4
    g_a = np.zeros((ne, 1, 1), dtype=complex)
    g_a[i_a] = 1.0
    g_b = np.zeros((ne, 1, 1), dtype=complex)
    g_b[i_b] = 1.0
    out = bubble_dense(phi_left=phi, phi_right=phi, G_a=g_a, G_b=g_b,
                       n_fft=2 * ne - 1, prefactor=1.0,
                       out_slice=slice(mid, mid + ne),
                       zero_freq_idx=None)[:, 0, 0]
    nz = np.where(np.abs(out) > 1e-12)[0]
    assert nz.tolist() == [i_a + i_b - mid]


# ---------------------------------------------------------------------------
# Detailed balance + Keldysh symmetry
# ---------------------------------------------------------------------------


def _equilibrium_sigma(toy, temperature, freq_range, dc_handling="keep",
                       eta_factor=1.0):
    freqs, dw, eta_w, z2, pos_mask, mid = build_frequency_grid(
        freq_range, eta_factor=eta_factor)
    g_ret = harmonic_green_retarded(toy.h00, z2)
    g_l, g_g = equilibrium_lesser_greater(g_ret, freqs, temperature)
    sl, sg = compute_phph_self_energy_finite_multi_slab(
        {(0, 0): g_l}, {(0, 0): g_g}, {(0, 0, 0): toy.phi.astype(complex)},
        1, freqs, dw, dc_handling=dc_handling, n_threads=1)
    return freqs, pos_mask, mid, sl[(0, 0)], sg[(0, 0)]


def test_detailed_balance_no_double_counting():
    """Sigma^>(omega) = exp(hbar omega / kT) Sigma^< at equilibrium."""
    temperature = 300.0
    freqs, pos_mask, mid, sl, sg = _equilibrium_sigma(
        single_oscillator(omega0_thz=5.0), temperature, (0.01, 20.0, 60))
    beta_hw = HBAR_SI * freqs * THZ_TO_RAD / (KB_SI * temperature)
    expected = np.exp(beta_hw[pos_mask])
    ratio = sg[pos_mask, 0, 0] / sl[pos_mask, 0, 0]
    big = np.abs(sl[pos_mask, 0, 0]) > 1e-14 * np.max(np.abs(sl))
    rel = np.abs(ratio - expected) / expected
    assert np.max(rel[big]) < 1e-6


def test_raw_bubble_keldysh_symmetric_and_anti_hermitian():
    freqs, pos_mask, mid, sl, sg = _equilibrium_sigma(
        diatomic_chain(), 250.0, (0.01, 18.0, 50))
    scale = np.max(np.abs(sl)) + np.max(np.abs(sg))
    # anti-Hermitian: Sigma^x = -Sigma^x^dagger
    ah = (np.max(np.abs(sl + sl.conj().transpose(0, 2, 1)))
          + np.max(np.abs(sg + sg.conj().transpose(0, 2, 1))))
    assert ah / scale < 1e-10
    # bosonic full-axis symmetry: Sigma^<(omega) = [Sigma^>(-omega)]^T
    sym = np.max(np.abs(sl[mid + 1:]
                        - sg[:mid][::-1].transpose(0, 2, 1)))
    assert sym / scale < 1e-10


# ---------------------------------------------------------------------------
# Retarded reconstruction (the Hilbert-padding fix)
# ---------------------------------------------------------------------------


def _lorentzian_pair(n):
    w = np.linspace(-40.0, 40.0, n)
    gamma = 2.0
    im = (gamma / (w ** 2 + gamma ** 2)).astype(complex)
    re_true = w / (w ** 2 + gamma ** 2)
    return w, im, re_true


def test_hilbert_padding_is_accurate():
    """The zero-padded FFT Hilbert matches the analytic Lorentzian pair."""
    n = 401
    interior = slice(n // 4, 3 * n // 4)
    _, im, re_true = _lorentzian_pair(n)
    h = np.real(hilbert_transform_axis(
        im[None, :, None, None], axis=1)[0, :, 0, 0])
    assert np.max(np.abs(h - re_true)[interior]) < 1e-3


def test_hilbert_unpadded_is_worse():
    """Regression: pad_factor=1 (periodic Hilbert) carries a ~1% error."""
    n = 401
    interior = slice(n // 4, 3 * n // 4)
    _, im, re_true = _lorentzian_pair(n)
    err_padded = np.max(np.abs(np.real(hilbert_transform_axis(
        im[None, :, None, None], axis=1, pad_factor=8)[0, :, 0, 0])
        - re_true)[interior])
    err_periodic = np.max(np.abs(np.real(hilbert_transform_axis(
        im[None, :, None, None], axis=1, pad_factor=1)[0, :, 0, 0])
        - re_true)[interior])
    assert err_periodic > 20 * err_padded
    assert err_periodic > 5e-3


def test_retarded_fft_matches_pv_on_resolved_grid():
    # eta/d_omega = 2: the device resonances are resolved, so the two
    # Kramers-Kronig quadratures must agree (cf. verify_bubble Part 2).
    freqs, pos_mask, mid, sl, sg = _equilibrium_sigma(
        diatomic_chain(), 300.0, (0.01, 18.0, 120), eta_factor=2.0)
    sr_fft = build_retarded(sl, sg, freqs, method="fft")
    sr_pv = build_retarded(sl, sg, freqs, method="pv")
    rel = np.max(np.abs(sr_fft - sr_pv)) / np.max(np.abs(sr_fft))
    assert rel < 1e-2


def test_retarded_half_is_anti_hermitian_part():
    freqs, pos_mask, mid, sl, sg = _equilibrium_sigma(
        diatomic_chain(), 300.0, (0.01, 18.0, 40))
    sr_half = build_retarded(sl, sg, freqs, method="half")
    np.testing.assert_allclose(sr_half, 0.5 * (sg - sl), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Cutoff controllability
# ---------------------------------------------------------------------------


def _random_multislab(n_slabs, n_dof, n_freq, seed):
    rng = np.random.default_rng(seed)

    def blk(s):
        return rng.standard_normal(s) + 1j * rng.standard_normal(s)

    phi = {(i, k, kp): blk((n_dof, n_dof, n_dof))
           for i in range(n_slabs) for k in range(n_slabs)
           for kp in range(n_slabs)}
    g_l = {(k, kp): blk((n_freq, n_dof, n_dof))
           for k in range(n_slabs) for kp in range(n_slabs)}
    g_g = {(k, kp): blk((n_freq, n_dof, n_dof))
           for k in range(n_slabs) for kp in range(n_slabs)}
    return phi, g_l, g_g


def test_cutoff_none_equals_brute_force():
    """sigma_cutoff=None, g_cutoff=None == an independent brute-force sum."""
    n_slabs, n_dof, n_freq = 3, 3, 13
    omega = np.linspace(-6.0, 6.0, n_freq)
    dw = omega[1] - omega[0]
    n_fft, mid = 2 * n_freq - 1, n_freq // 2
    freq_sl = slice(mid, mid + n_freq)
    prefactor = 0.5j * HBAR_SI * dw / (2 * np.pi)
    phi, g_l, g_g = _random_multislab(n_slabs, n_dof, n_freq, seed=1)

    sl_drv, _ = compute_phph_self_energy_finite_multi_slab(
        g_l, g_g, phi, n_slabs, omega, dw,
        sigma_cutoff=None, g_cutoff=None, dc_handling="zero", n_threads=2)

    brute: dict[tuple[int, int], np.ndarray] = {}
    for (i, k1, k2), phi_l in phi.items():
        for (j, k2p, k1p), phi_r in phi.items():
            blk = bubble_dense(
                phi_left=phi_l, phi_right=phi_r,
                G_a=g_l[(k1, k1p)], G_b=g_l[(k2, k2p)],
                n_fft=n_fft, prefactor=prefactor,
                out_slice=freq_sl, zero_freq_idx=mid, dc_handling="zero")
            brute[(i, j)] = (blk if (i, j) not in brute
                             else brute[(i, j)] + blk)
    assert set(sl_drv) == set(brute)
    for k in brute:
        np.testing.assert_allclose(sl_drv[k], brute[k], rtol=1e-9, atol=1e-12)


def test_sigma_cutoff_filters_output_only():
    """sigma_cutoff drops (I,J) pairs; retained blocks are unchanged."""
    n_slabs, n_dof, n_freq = 4, 3, 11
    omega = np.linspace(-5.0, 5.0, n_freq)
    dw = omega[1] - omega[0]
    phi, g_l, g_g = _random_multislab(n_slabs, n_dof, n_freq, seed=2)
    sl_full, _ = compute_phph_self_energy_finite_multi_slab(
        g_l, g_g, phi, n_slabs, omega, dw,
        sigma_cutoff=None, g_cutoff=None, dc_handling="zero", n_threads=1)
    sl_c, _ = compute_phph_self_energy_finite_multi_slab(
        g_l, g_g, phi, n_slabs, omega, dw,
        sigma_cutoff=1, g_cutoff=None, dc_handling="zero", n_threads=1)
    assert all(abs(i - j) <= 1 for (i, j) in sl_c)
    for k in sl_c:
        np.testing.assert_allclose(sl_c[k], sl_full[k], rtol=1e-10, atol=0)


def test_dc_handling_modes_are_distinct():
    n_slabs, n_dof, n_freq = 2, 3, 15
    omega = np.linspace(-7.0, 7.0, n_freq)
    dw = omega[1] - omega[0]
    phi, g_l, g_g = _random_multislab(n_slabs, n_dof, n_freq, seed=3)
    sig = {}
    for dc in ("zero", "interpolate", "keep"):
        sl, _ = compute_phph_self_energy_finite_multi_slab(
            g_l, g_g, phi, n_slabs, omega, dw, dc_handling=dc, n_threads=1)
        sig[dc] = sl[(0, 0)]

    def rel(a, b):
        return (np.max(np.abs(a - b))
                / (np.max(np.abs(a)) + np.max(np.abs(b)) + 1e-300))

    assert rel(sig["zero"], sig["interpolate"]) > 1e-3
    assert rel(sig["zero"], sig["keep"]) > 1e-3
    assert rel(sig["interpolate"], sig["keep"]) > 1e-3


# ---------------------------------------------------------------------------
# SCBA loop
# ---------------------------------------------------------------------------


def test_scba_loop_runs_and_converges():
    """The production SCBA fixed-point loop converges and conserves heat."""
    from solver.dense import scba_loop
    from solver.leads import build_device_hamiltonian, compute_obc_batch

    toy = diatomic_chain()
    n_slabs, n_dof = 3, toy.n_dof
    N_D = n_slabs * n_dof
    freqs, dw, eta_w, z2, pos_mask, mid = build_frequency_grid(
        (0.01, 20.0, 40), eta_factor=1.0)
    nfreq = len(freqs)
    h00 = toy.h00.astype(complex)
    h01 = toy.h01.astype(complex)
    h_d = build_device_hamiltonian(h00, h01, n_slabs)
    t_l, t_r = 310.0, 290.0
    obc = compute_obc_batch(z2, h00, h01, freqs, t_l, t_r, n_slabs=n_slabs)
    phi_dev = {(i, i, i): toy.phi.astype(complex) for i in range(n_slabs)}

    def se_kernel(g_less_dev_q, g_great_dev_q):
        sig_l = np.zeros((1, nfreq, N_D, N_D), dtype=complex)
        sig_g = np.zeros_like(sig_l)

        def gd(dense):
            return {(k, kp): dense[:, k * n_dof:(k + 1) * n_dof,
                                   kp * n_dof:(kp + 1) * n_dof]
                    for k in range(n_slabs) for kp in range(n_slabs)}

        sl_b, sg_b = compute_phph_self_energy_finite_multi_slab(
            gd(g_less_dev_q[0]), gd(g_great_dev_q[0]), phi_dev, n_slabs,
            freqs, dw, dc_handling="interpolate", n_threads=1)
        for (i, j), b in sl_b.items():
            sig_l[0, :, i * n_dof:(i + 1) * n_dof,
                  j * n_dof:(j + 1) * n_dof] = b
        for (i, j), b in sg_b.items():
            sig_g[0, :, i * n_dof:(i + 1) * n_dof,
                  j * n_dof:(j + 1) * n_dof] = b
        return sig_l, sig_g

    result = scba_loop(
        z2_arr=z2, freqs_thz=freqs, dw_thz=dw,
        omega_rad=freqs * THZ_TO_RAD, pos_mask=pos_mask,
        n_slabs=n_slabs, n_dof=n_dof, N_D=N_D,
        H_D_list=[h_d], obc_list=[obc], btd_blocks_list=[(h00, h01)],
        n_kpts=1, se_kernel=se_kernel, T_L=t_l, T_R=t_r,
        max_scba_iter=40, scba_tol=1e-3, conservation_tol=1e-2,
        mixing=0.5, anderson_mixing=False, anderson_depth=5,
        scattering_contacts=False, retarded="fft", verbose=False,
        masses_primitive=toy.masses)

    # the loop produced finite self-energies and a finite conservation
    assert np.all(np.isfinite(result["Sigma_R"]))
    assert result["conservation_err"] < 1e-2
    # Sigma^R is consistent with the stored Sigma^{<,>} (no lag bug)
    rebuilt = build_retarded(result["Sigma_l"], result["Sigma_g"],
                             freqs, method="fft")
    np.testing.assert_allclose(result["Sigma_R"], rebuilt, rtol=1e-10,
                               atol=0)


def _diatomic_device(n_slabs=3, freq_range=(0.01, 20.0, 40), eta_factor=1.0):
    """Shared toy diatomic-chain finite device for the static-SE tests."""
    from solver.leads import build_device_hamiltonian, compute_obc_batch

    toy = diatomic_chain()
    n_dof = toy.n_dof
    N_D = n_slabs * n_dof
    freqs, dw, eta_w, z2, pos_mask, mid = build_frequency_grid(
        freq_range, eta_factor=eta_factor)
    h00 = toy.h00.astype(complex)
    h01 = toy.h01.astype(complex)
    h_d = build_device_hamiltonian(h00, h01, n_slabs)
    t_l, t_r = 310.0, 290.0
    obc = compute_obc_batch(z2, h00, h01, freqs, t_l, t_r, n_slabs=n_slabs)
    return dict(toy=toy, n_slabs=n_slabs, n_dof=n_dof, N_D=N_D, freqs=freqs,
                dw=dw, z2=z2, pos_mask=pos_mask, h00=h00, h01=h01, h_d=h_d,
                t_l=t_l, t_r=t_r, obc=obc)


def _zero_bubble_kernel(nfreq, N_D):
    def se_kernel(g_less_dev_q, g_great_dev_q):
        z = np.zeros((1, nfreq, N_D, N_D), dtype=complex)
        return z, z.copy()
    return se_kernel


def _run_scba(dev, se_kernel, **kw):
    from solver.dense import scba_loop

    mixing = kw.pop("mixing", 0.5)
    return scba_loop(
        z2_arr=dev["z2"], freqs_thz=dev["freqs"], dw_thz=dev["dw"],
        omega_rad=dev["freqs"] * THZ_TO_RAD, pos_mask=dev["pos_mask"],
        n_slabs=dev["n_slabs"], n_dof=dev["n_dof"], N_D=dev["N_D"],
        H_D_list=[dev["h_d"]], obc_list=[dev["obc"]],
        btd_blocks_list=[(dev["h00"], dev["h01"])],
        n_kpts=1, se_kernel=se_kernel, T_L=dev["t_l"], T_R=dev["t_r"],
        scba_tol=1e-3, conservation_tol=1e-2, mixing=mixing,
        anderson_mixing=False, anderson_depth=5, scattering_contacts=False,
        retarded="fft", verbose=False, masses_primitive=dev["toy"].masses, **kw)


def test_static_hook_zero_fc_matches_baseline():
    """A loop/tadpole hook with zero force constants must not perturb the
    default SCBA path (Sigma_static stays 0 -> identical G/Sigma)."""
    from solver.bubble import bubble_dense  # noqa: F401 (ensure import path)
    from solver.static_se import build_static_self_energy_hook

    dev = _diatomic_device()
    nfreq = len(dev["freqs"])
    se = _zero_bubble_kernel(nfreq, dev["N_D"])

    base = _run_scba(dev, se, max_scba_iter=10)

    fc3_zero = np.zeros((dev["N_D"],) * 3)
    fc4_zero = np.zeros((dev["N_D"],) * 4)
    hook = build_static_self_energy_hook(
        dw_thz=dev["dw"], n_dof=dev["n_dof"], n_slabs=dev["n_slabs"],
        fc3_dev_mw=fc3_zero, fc4_dev_mw=fc4_zero,
        use_loop=True, use_tadpole=True)
    withhook = _run_scba(dev, se, max_scba_iter=10, static_se_hook=hook,
                         stage_loop_first=True)

    assert withhook["Sigma_static"] is not None
    assert np.max(np.abs(withhook["Sigma_static"])) < 1e-12
    np.testing.assert_allclose(withhook["Sigma_R"], base["Sigma_R"],
                               rtol=1e-10, atol=1e-12)


def test_static_loop_stiffens_phi_eff():
    """A positive on-site quartic loop raises every dynamical-matrix
    eigenvalue (frequency stiffening), with a real symmetric Sigma_static."""
    from solver.static_se import build_static_self_energy_hook

    dev = _diatomic_device()
    nfreq = len(dev["freqs"])
    se = _zero_bubble_kernel(nfreq, dev["N_D"])

    g4 = 30.0
    fc4 = np.zeros((dev["N_D"],) * 4)
    idx = np.arange(dev["N_D"])
    fc4[idx, idx, idx, idx] = g4                      # on-site positive quartic
    hook = build_static_self_energy_hook(
        dw_thz=dev["dw"], n_dof=dev["n_dof"], n_slabs=dev["n_slabs"],
        fc4_dev_mw=fc4, use_loop=True, use_tadpole=False)

    res = _run_scba(dev, se, max_scba_iter=60, static_se_hook=hook,
                    stage_loop_first=True, loop_propagator="loop_only")

    sig = res["Sigma_static"][0]
    assert np.allclose(sig, sig.conj().T)             # Hermitian
    assert np.max(np.abs(sig.imag)) < 1e-8            # real
    ev_bare = np.linalg.eigvalsh(dev["h_d"])
    ev_eff = np.linalg.eigvalsh(dev["h_d"] + sig)
    # every mode stiffens (positive-definite loop shift)
    assert np.all(ev_eff > ev_bare - 1e-9)
    assert ev_eff.min() > ev_bare.min() + 1e-6
    assert np.trace(sig.real) > 0


def _bubble_kernel(dev, n_slabs):
    """Real toy-cubic bubble se_kernel for the device (for the SCBA loop)."""
    nfreq = len(dev["freqs"])
    n_dof = dev["n_dof"]
    N_D = n_slabs * n_dof
    phi_dev = {(i, i, i): dev["toy"].phi.astype(complex)
               for i in range(n_slabs)}

    def se_kernel(gl, gg):
        sig_l = np.zeros((1, nfreq, N_D, N_D), dtype=complex)
        sig_g = np.zeros_like(sig_l)

        def gd(dense):
            return {(k, kp): dense[:, k * n_dof:(k + 1) * n_dof,
                                   kp * n_dof:(kp + 1) * n_dof]
                    for k in range(n_slabs) for kp in range(n_slabs)}

        sl_b, sg_b = compute_phph_self_energy_finite_multi_slab(
            gd(gl[0]), gd(gg[0]), phi_dev, n_slabs, dev["freqs"], dev["dw"],
            dc_handling="interpolate", n_threads=1)
        for (i, j), b in sl_b.items():
            sig_l[0, :, i * n_dof:(i + 1) * n_dof,
                  j * n_dof:(j + 1) * n_dof] = b
        for (i, j), b in sg_b.items():
            sig_g[0, :, i * n_dof:(i + 1) * n_dof,
                  j * n_dof:(j + 1) * n_dof] = b
        return sig_l, sig_g

    return se_kernel


def test_static_loop_anderson_matches_linear():
    """The loop+tadpole static self-energy is now carried in the fixed-point
    vector, so Anderson drives it jointly with the bubble: it must reach the
    SAME converged Sigma_static as linear mixing, in no more iterations."""
    from solver.static_se import build_static_self_energy_hook

    n_slabs = 2
    dev = _diatomic_device(n_slabs=n_slabs)
    se = _bubble_kernel(dev, n_slabs)
    N_D = dev["N_D"]
    fc4 = np.zeros((N_D,) * 4)
    idx = np.arange(N_D)
    fc4[idx, idx, idx, idx] = 20.0
    hook_kw = dict(dw_thz=dev["dw"], n_dof=dev["n_dof"], n_slabs=n_slabs,
                   fc4_dev_mw=fc4, use_loop=True)

    # No staging + low mixing: Sigma_static must converge in the main loop,
    # where linear mixing is slow -> a fair linear-vs-Anderson comparison.
    res_lin = _run_scba(dev, se, max_scba_iter=300, mixing=0.15,
                        static_se_hook=build_static_self_energy_hook(**hook_kw),
                        stage_loop_first=False, solver="linear")
    res_and = _run_scba(dev, se, max_scba_iter=300, mixing=0.15,
                        static_se_hook=build_static_self_energy_hook(**hook_kw),
                        stage_loop_first=False, solver="anderson")

    n_lin = len(res_lin["convergence_history"])
    n_and = len(res_and["convergence_history"])
    print(f"\n  linear: {n_lin} iters (resid {res_lin['scba_residual']:.2e}); "
          f"anderson: {n_and} iters (resid {res_and['scba_residual']:.2e})")
    assert res_lin["converged"] and res_and["converged"]
    # same fixed point (Sigma_static) regardless of accelerator
    np.testing.assert_allclose(res_and["Sigma_static"][0],
                               res_lin["Sigma_static"][0], atol=1e-3, rtol=1e-3)
    # Anderson (with the loosened static step cap) is much faster than linear:
    # it carries Sigma_static in the fixed-point vector and accelerates the
    # slow static mode (toy: ~35 linear -> a few Anderson).
    assert n_and < n_lin
    assert n_and <= max(5, n_lin // 3)


def test_spectral_bands_from_scba_static_se():
    """End-to-end: a loop Sigma_static from the SCBA renormalises the band
    structure (postproc.spectral) -- every branch stiffens at every q."""
    from postproc.spectral import decomposition_bands, spectral_function_qw
    from solver.static_se import build_static_self_energy_hook

    dev = _diatomic_device(n_slabs=1)            # one cell: N_D = n_dof = 2
    nfreq = len(dev["freqs"])
    se = _zero_bubble_kernel(nfreq, dev["N_D"])

    fc4 = np.zeros((dev["N_D"],) * 4)
    idx = np.arange(dev["N_D"])
    fc4[idx, idx, idx, idx] = 40.0               # on-site positive quartic
    hook = build_static_self_energy_hook(
        dw_thz=dev["dw"], n_dof=dev["n_dof"], n_slabs=1,
        fc4_dev_mw=fc4, use_loop=True, use_tadpole=False)
    res = _run_scba(dev, se, max_scba_iter=60, static_se_hook=hook,
                    stage_loop_first=True)
    sigma_static = res["Sigma_static"][0].real    # (2, 2) on-site loop shift

    # periodic diatomic-chain bands D(q) = h00 + h01 e^{iq} + h01^dag e^{-iq}
    qs = np.linspace(0.05, np.pi, 12)
    h00, h01 = dev["h00"], dev["h01"]
    D_q = np.stack([h00 + h01 * np.exp(1j * q) + h01.conj().T * np.exp(-1j * q)
                    for q in qs])

    bands = decomposition_bands(D_q, sigma_loop=sigma_static)
    assert set(bands) == {"bare", "loop"}
    # the loop stiffens every branch at every q-point
    assert np.all(bands["loop"] >= bands["bare"] - 1e-9)
    assert bands["loop"].max() > bands["bare"].max() + 1e-6

    # spectral function with the static shift peaks at the renormalised bands
    grid = np.linspace(0.1, 40.0, 3000)
    A = spectral_function_qw(D_q, grid, eta_w_thz=0.05,
                             sigma_static=sigma_static)
    assert np.all(A >= -1e-9)
    iq = len(qs) // 2
    peak = grid[np.argmax(A[iq])]
    assert np.min(np.abs(peak - bands["loop"][iq])) < 0.15


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
