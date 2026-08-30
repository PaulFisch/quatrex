"""Identity tests for the dense-reference observables
(``phonon/solver/observables.py``) on an analytic ballistic chain:

  * equilibrium occupation equals the Bose function (eq:neq_occupation);
  * the effective temperature recovers the bath temperature (eq:Teff_local);
  * contact bond currents equal the Meir-Wingreen lead spectra (eq:I_local
    normalization);
  * the energy sum rule D(omega) vanishes ballistically (eq:sumrule);
  * Fisher-Lee mode sum equals the Caroli transmission (eq:mode_T);
  * the LDOS integrates to one state per DOF (eq:sum_rule_zero);
  * the perfect-channel conductance reproduces pi^2 kB^2 T / (3 h) (eq:g0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PHONON = Path(__file__).resolve().parents[2] / "phonon"
if str(_PHONON) not in sys.path:
    sys.path.insert(0, str(_PHONON))

from phonon_inputs.constants import KB_SI, HBAR_SI, THZ_TO_RAD  # noqa: E402
from solver.grids import build_frequency_grid  # noqa: E402
from solver.leads import compute_obc_batch, solve_green_batch  # noqa: E402
from solver.toy_models import monatomic_chain  # noqa: E402
from solver import observables as obs  # noqa: E402

N_SLABS = 4
W_MAX = 8.0


def _ballistic_chain(T_L=300.0, T_R=300.0, nfreq_pos=160, fmax=10.0):
    toy = monatomic_chain(omega_max_thz=W_MAX)
    freqs, dw, _eta, z2, pos, _mid = build_frequency_grid(
        (0.01, fmax, nfreq_pos), eta_factor=0.01)
    n_dof = toy.n_dof
    N_D = N_SLABS * n_dof
    H_D = toy.device_dynamical_matrix(N_SLABS)
    obc = compute_obc_batch(z2, toy.h00, toy.h01, freqs, T_L, T_R,
                            n_slabs=N_SLABS)
    zero = np.zeros((freqs.size, N_D, N_D), dtype=complex)
    G_R, G_l, G_g = solve_green_batch(z2, H_D, obc, zero, zero, zero)
    return dict(freqs=freqs, dw=dw, pos=pos, n_dof=n_dof, H_D=H_D,
                obc=obc, G_R=G_R, G_l=G_l, G_g=G_g, zero=zero)


@pytest.fixture(scope="module")
def eq_chain():
    return _ballistic_chain(300.0, 300.0)


@pytest.fixture(scope="module")
def neq_chain():
    return _ballistic_chain(305.0, 295.0)


def test_equilibrium_occupation_is_bose(eq_chain):
    c = eq_chain
    n = obs.local_occupation(c["G_l"], c["G_g"], n_dof=c["n_dof"],
                             n_slabs=N_SLABS)
    n_ref = obs.bose(c["freqs"], 300.0)
    # In-band positive frequencies with real spectral weight (the ratio is
    # ill-conditioned where A ~ 0 near the band edge and above the band).
    A = np.diagonal(
        1j * (c["G_g"] - c["G_l"]), axis1=-2, axis2=-1).real
    A_slab = A.reshape(A.shape[0], N_SLABS, c["n_dof"]).sum(axis=-1)
    sel2 = ((c["freqs"][:, None] > 0.5) & (c["freqs"][:, None] < 0.9 * W_MAX)
            & (A_slab > 1e-3 * A_slab.max()))
    assert sel2.any()
    rel_err = (np.abs(n - n_ref[:, None])
               / np.maximum(n_ref, 1e-3)[:, None])[sel2]
    assert np.max(rel_err) < 5e-2


def test_effective_temperature_recovers_bath(eq_chain):
    c = eq_chain
    pos = c["pos"]
    gl = np.diagonal(c["G_l"], axis1=-2, axis2=-1)
    A = np.diagonal(
        1j * (c["G_R"] - c["G_R"].conj().swapaxes(-2, -1)),
        axis1=-2, axis2=-1).real
    iGl = (1j * gl).real
    T_eff = obs.effective_temperature(c["freqs"][pos], iGl[pos], A[pos])
    assert np.all(np.isfinite(T_eff))
    assert np.max(np.abs(T_eff - 300.0)) < 6.0  # K (grid tails)


def test_contact_bond_current_equals_meir_wingreen(neq_chain):
    c = neq_chain
    spec_L, spec_R = obs.meir_wingreen_spectra(
        c["G_l"], c["G_g"], c["obc"], c["freqs"], c["n_dof"], N_SLABS)
    bonds = obs.bond_currents(c["G_l"], c["H_D"], c["freqs"],
                              c["n_dof"], N_SLABS)
    cuts = obs.harmonic_cut_currents(
        c["G_l"], c["H_D"], c["freqs"], c["n_dof"], N_SLABS)
    np.testing.assert_allclose(cuts, bonds, rtol=1e-13, atol=1e-30)
    pos = c["pos"]
    # Ballistic: every interior bond carries the lead current. Restrict to
    # in-band bins with real current (band edges carry eta artifacts).
    sel = pos & (np.abs(spec_L) > 0.05 * np.abs(spec_L[pos]).max())
    for i in range(N_SLABS - 1):
        ratio = bonds[sel, i] / spec_L[sel]
        assert abs(np.median(ratio) - 1.0) < 2e-2, (i, np.median(ratio))
    # And the two lead spectra agree (ballistic conservation).
    assert np.max(np.abs(spec_L[sel] - spec_R[sel])
                  / np.abs(spec_L[sel])) < 5e-2


def test_harmonic_cut_current_includes_long_range_fc2_pair():
    freqs = np.array([1.0, 2.0])
    H = np.zeros((3, 3), dtype=complex)
    G_l = np.zeros((2, 3, 3), dtype=complex)
    H[0, 2] = H[2, 0] = 3.0
    G_l[:, 2, 0] = np.array([2.0, -0.5])

    cuts = obs.harmonic_cut_currents(G_l, H, freqs, 1, 3)
    expected = (-2.0 * HBAR_SI * freqs * THZ_TO_RAD
                * np.array([6.0, -1.5]))
    # The 0--2 pair crosses both cuts and therefore contributes to both.
    np.testing.assert_allclose(cuts[:, 0], expected)
    np.testing.assert_allclose(cuts[:, 1], expected)
    # The nearest-slab expression omits this deliberately non-BTD pair.
    assert np.all(obs.bond_currents(G_l, H, freqs, 1, 3) == 0.0)


def test_telescoped_current_removes_interaction_redistribution():
    # The bare channel current changes when the cubic interaction absorbs
    # energy in one slab and returns it in another, although the leads agree.
    p_int = np.array([-3.0, 0.0, 3.0])
    bare = np.array([10.0, 7.0, 7.0, 10.0])
    assert np.ptp(bare) == 3.0
    assert obs.telescoped_spread(bare, p_int) == 0.0


def test_sum_rule_D_vanishes_ballistically(neq_chain):
    c = neq_chain
    spec_L, spec_R = obs.meir_wingreen_spectra(
        c["G_l"], c["G_g"], c["obc"], c["freqs"], c["n_dof"], N_SLABS)
    J_s = obs.scattering_terminal_current(
        c["zero"], c["zero"], c["G_l"], c["G_g"], c["freqs"],
        c["n_dof"], N_SLABS)
    D = obs.sumrule_D_omega(spec_L, spec_R, J_s)
    pos = c["pos"]
    assert np.max(np.abs(D[pos])) < 1e-8 * max(np.abs(spec_L[pos]).max(),
                                               1e-300)


def test_fisher_lee_matches_caroli(neq_chain):
    c = neq_chain
    T_mode, T_tot = obs.fisher_lee_modes(
        c["G_R"], c["obc"]["Gamma_L"], c["obc"]["Gamma_R"],
        c["n_dof"], N_SLABS)
    # Caroli reference: Tr[Gamma_L G^A_{1N} Gamma_R G^R_{N1}].
    sl0 = slice(0, c["n_dof"])
    slN = slice((N_SLABS - 1) * c["n_dof"], N_SLABS * c["n_dof"])
    GR_N1 = c["G_R"][:, slN, sl0]
    caroli = np.einsum(
        "wab,wbc,wcd,wda->w",
        c["obc"]["Gamma_L"][:, sl0, sl0],
        GR_N1.conj().swapaxes(-2, -1),
        c["obc"]["Gamma_R"][:, slN, slN],
        GR_N1,
    ).real
    pos = c["pos"]
    assert np.max(np.abs(T_tot[pos] - caroli[pos])) < 1e-10 + 1e-10 * caroli[pos].max()
    # Single channel: the in-band transmission is ~1 and one mode carries it.
    band = (c["freqs"] > 0.5) & (c["freqs"] < 0.9 * W_MAX)
    assert np.all(np.abs(T_mode[band, 0] - 1.0) < 5e-2)
    # The mode sum equals the total.
    assert np.max(np.abs(T_mode.sum(axis=1) - T_tot)) < 1e-10


def test_ldos_sum_rule():
    toy = monatomic_chain(omega_max_thz=W_MAX)
    freqs, dw, _eta, z2, pos, _mid = build_frequency_grid(
        (0.01, 24.0, 1200), eta_w_thz=0.15)
    H_D = toy.device_dynamical_matrix(N_SLABS)
    obc = compute_obc_batch(z2, toy.h00, toy.h01, freqs, 300.0, 300.0,
                            n_slabs=N_SLABS)
    zero = np.zeros((freqs.size,) + H_D.shape, dtype=complex)
    G_R, _gl, _gg = solve_green_batch(z2, H_D, obc, zero, zero, zero)
    c = dict(G_R=G_R, freqs=freqs, dw=dw, pos=pos)
    rho = obs.local_dos(c["G_R"], c["freqs"])
    # One state per DOF over the positive axis (eq:sum_rule_zero; rho is
    # even in omega, so the symmetric axis carries the sum rule twice).
    total = rho[c["pos"]].sum(axis=0) * c["dw"]
    assert np.max(np.abs(total - 1.0)) < 2e-2


def test_perfect_channel_conductance_matches_g0():
    # T(omega) = 1 on a wide grid reproduces the thermal conductance
    # quantum g0 = pi^2 kB^2 T / (3 h) (eq:g0) at low temperature.
    T = 5.0  # K: band top >> kB T so the truncation error is negligible
    freqs = np.linspace(0.0, 4.0, 4001)
    G = obs.thermal_conductance(np.ones_like(freqs), freqs, T)
    g0 = np.pi**2 * KB_SI**2 * T / (3.0 * 2.0 * np.pi * HBAR_SI)
    assert abs(G - g0) < 5e-3 * g0  # grid truncation
