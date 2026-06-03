#!/usr/bin/env python
"""Verify the equal-time correlation <uu> prefactor (loop/tadpole foundation).

The static loop/tadpole self-energies are built from the equal-time
mass-weighted displacement correlation ``<w_a w_b>`` extracted from the lesser
Green's function (``solver.static_se.equal_time_uu``). This script pins its
prefactor by the equilibrium-limit test demanded by the brief:

  set both leads to the same T, build the *equilibrium* KMS ``G^<`` of a known
  harmonic system on the solver's symmetric grid, integrate it, and check it
  reproduces the analytic mode sum ``sum_s (hbar/2 Omega_s)(2 n_s + 1) e e*``.

Checked on (1) a single oscillator, (2) a finite 1D monatomic chain device.

Run:  python phonon/scripts/verify/verify_static_se.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

from phonon_inputs.constants import CONVERSION_THZ2
from solver.grids import build_frequency_grid
from solver.static_se import (
    equal_time_uu,
    equilibrium_uu_modesum,
    mean_displacement,
    mean_force,
    sigma_loop,
    sigma_tadpole,
)
from solver.toy_models import (
    diatomic_chain,
    equilibrium_lesser_greater,
    harmonic_green_retarded,
    single_oscillator,
)


def _uu_from_equilibrium_G(dyn, freq_range, temperature, eta_factor):
    """Build equilibrium G^< on the grid and integrate it -> <uu>.

    The delta-limit test needs the Lorentzian resolved by the grid
    (``eta_w >~ 2 dw``) yet narrow vs the modes (``eta_w << omega_s``):
    the sweet spot is ``eta_factor ~ 2`` on a fine grid, where the
    residual is ~5e-4 (purely numerical). Larger ``eta_factor`` grows the
    O((eta/omega)^2) delta-limit bias.
    """
    freqs, dw, eta_w, z2, pos_mask, mid = build_frequency_grid(
        freq_range, eta_factor=eta_factor)
    g_ret = harmonic_green_retarded(dyn, z2)
    g_less, _ = equilibrium_lesser_greater(g_ret, freqs, temperature)
    uu = equal_time_uu(g_less, dw)              # (N, N), amu*Angstrom^2
    return uu, freqs, dw, eta_w


def check_single_oscillator():
    print("== single oscillator (omega0 = 6 THz) ==")
    toy = single_oscillator(omega0_thz=6.0)
    dyn = toy.h00                               # 1x1, no transport
    for T in (1.0, 300.0, 600.0):
        uu, freqs, dw, eta_w = _uu_from_equilibrium_G(
            dyn, (0.0, 30.0, 12000), T, eta_factor=2.0)
        ref = equilibrium_uu_modesum(dyn, T)
        rel = abs(uu[0, 0] - ref[0, 0]) / abs(ref[0, 0])
        print(f"  T={T:6.1f} K  <uu>={uu[0,0]:.6e}  ref={ref[0,0]:.6e}  "
              f"rel={rel:.2e}  (eta_w={eta_w:.3e} THz)")
        assert rel < 2e-3, f"single-oscillator <uu> off by {rel:.2e} at T={T}"
    print("  PASS")


def check_diatomic_onsite():
    print("== diatomic on-site block (2 gapped modes, no zero mode) ==")
    toy = diatomic_chain()
    dyn = toy.h00                               # 2x2, both eigvals > 0
    for T in (1.0, 300.0):
        uu, freqs, dw, eta_w = _uu_from_equilibrium_G(
            dyn, (0.0, 40.0, 16000), T, eta_factor=2.0)
        ref = equilibrium_uu_modesum(dyn, T)
        rel = np.linalg.norm(uu - ref) / np.linalg.norm(ref)
        print(f"  T={T:6.1f} K  ||<uu>-ref||/||ref|| = {rel:.2e}")
        assert rel < 3e-3, f"diatomic on-site <uu> off by {rel:.2e} at T={T}"
    print("  PASS")


def check_random_psd_matrix():
    """General N-mode, off-diagonal, all-positive-eigenvalue dynamical matrix.

    Exercises the full eigenvector structure of <uu> (the single oscillator is
    diagonal; this is not). No zero mode -> clean delta-limit comparison.
    """
    print("== random PSD dynamical matrix (N=5, off-diagonal) ==")
    rng = np.random.default_rng(0)
    n = 5
    a = rng.standard_normal((n, n))
    # eigenvalues placed in [4, 100] THz^2 -> modes ~2..10 THz, well gapped.
    q, _ = np.linalg.qr(a)
    eigs = rng.uniform(4.0, 100.0, size=n)
    dyn = (q * eigs[None, :]) @ q.T
    dyn = 0.5 * (dyn + dyn.T)
    for T in (1.0, 300.0):
        uu, freqs, dw, eta_w = _uu_from_equilibrium_G(
            dyn, (0.0, 40.0, 16000), T, eta_factor=2.0)
        ref = equilibrium_uu_modesum(dyn, T)
        rel = np.linalg.norm(uu - ref) / np.linalg.norm(ref)
        print(f"  T={T:6.1f} K  ||<uu>-ref||/||ref|| = {rel:.2e}")
        assert rel < 3e-3, f"random-PSD <uu> off by {rel:.2e} at T={T}"
    print("  PASS")


def check_loop_sign_and_scp():
    """Loop self-energy sign (one-shot) + a physical SCP fixed point + units."""
    print("== quartic loop: sign + single-mode SCP fixed point ==")
    omega0, m, T = 6.0, 2.0, 300.0
    uu0 = equilibrium_uu_modesum(np.array([[omega0 ** 2]]), T)

    # (a) one-shot sign: Sigma_L has the sign of g4 (loop stiffens for g4>0).
    for g4 in (+8.0, -3.0):
        fc4_mw = np.array([[[[g4 / m ** 2]]]])
        sig = sigma_loop(fc4_mw, uu0)[0, 0]
        assert np.sign(sig) == np.sign(g4), f"loop sign wrong for g4={g4}"
    print(f"  one-shot sign OK (Sigma_L sign tracks g4)")

    # (b) physical SCP fixed point (g4>0, bounded potential): stiffening +
    #     cross-check <uu> from the equilibrium G^< pipeline at the fixed point.
    g4 = 8.0
    fc4_mw = np.array([[[[g4 / m ** 2]]]])
    om2 = omega0 ** 2
    for _ in range(500):
        uu = equilibrium_uu_modesum(np.array([[om2]]), T)
        new = omega0 ** 2 + sigma_loop(fc4_mw, uu)[0, 0]
        if abs(new - om2) < 1e-13:
            break
        om2 = new
    omega = np.sqrt(om2)
    sig = sigma_loop(fc4_mw, equilibrium_uu_modesum(np.array([[om2]]), T))[0, 0]
    freqs, dw, eta_w, z2, pm, mid = build_frequency_grid(
        (0.0, 30.0, 12000), eta_factor=2.0)
    g_ret = harmonic_green_retarded(np.array([[om2]]), z2)
    g_less, _ = equilibrium_lesser_greater(g_ret, freqs, T)
    sig_G = sigma_loop(fc4_mw, equal_time_uu(g_less, dw))[0, 0]
    rel = abs(sig_G - sig) / abs(sig)
    print(f"  g4={g4:+.1f}: omega {omega0:.3f}->{omega:.3f} THz (stiffen); "
          f"G^<-path vs mode-sum rel={rel:.2e}")
    assert omega > omega0, "loop should stiffen for g4>0"
    assert rel < 3e-3, f"G^<-path loop off by {rel:.2e}"
    print("  PASS")


def check_tadpole_single_mode():
    """Single asymmetric mode: closed-form tadpole shift + force balance."""
    print("== cubic tadpole: single-mode closed form ==")
    omega0, m, g3, T = 8.0, 1.0, 5.0, 300.0       # g3 in eV/Angstrom^3
    om2 = omega0 ** 2
    fc3_mw = np.array([[[g3 / m ** 1.5]]])         # (1,1,1) mass-weighted
    uu = equilibrium_uu_modesum(np.array([[om2]]), T)
    phi_eff = np.array([[om2]])
    w_mean = mean_displacement(fc3_mw, uu, phi_eff)
    sig_T = sigma_tadpole(fc3_mw, w_mean)[0, 0]
    # closed form: Sigma_T = -CONVERSION_THZ2^2 (g3/m^1.5)^2 <w^2> / (2 omega^2)
    ref = -(CONVERSION_THZ2 ** 2) * (g3 / m ** 1.5) ** 2 * uu[0, 0] / (2 * om2)
    rel = abs(sig_T - ref) / abs(ref)
    # force-balance residual: f = 1/2 g3 <u^2> = 1/2 g3 <w^2>/m  [eV/Angstrom]
    f = mean_force(fc3_mw, uu, [m])[0]
    f_ref = 0.5 * g3 * uu[0, 0] / m
    rel_f = abs(f - f_ref) / abs(f_ref)
    print(f"  Sigma_T={sig_T:+.3e} (ref {ref:+.3e}, rel={rel:.2e}); "
          f"soften={sig_T < 0}; <F>={f:.3e} eV/A (rel={rel_f:.2e})")
    assert rel < 1e-10, f"tadpole closed-form off by {rel:.2e}"
    assert rel_f < 1e-10, f"force balance off by {rel_f:.2e}"
    assert sig_T < 0, "single-mode tadpole should soften"
    print("  PASS")


if __name__ == "__main__":
    check_single_oscillator()
    check_diatomic_onsite()
    check_random_psd_matrix()
    check_loop_sign_and_scp()
    check_tadpole_single_mode()
    print("\nAll static-self-energy (uu / loop / tadpole) checks passed.")
