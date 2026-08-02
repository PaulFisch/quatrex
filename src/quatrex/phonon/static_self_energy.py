# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Self-consistent static (SCP) self-energy for the phonon SCBA.

The dynamic 3-phonon bubble (SigmaPhononPhonon) is the leading anharmonic
self-energy, but on a soft-mode structure, its G^< at
omega -> 0 injects a singularity that destabilises the SCBA. The
self-consistent-phonon (SCP) cubic tadpole

    Sigma_T = Phi3 : <u>,   <u> = -Phi_eff^{-1} ( 1/2 Phi3 : <uu> )

is a frequency-independent real self-energy that stiffens the soft
mode.
"""

from __future__ import annotations

import numpy as np

AMU_KG = 1.66053906660e-27
EV_TO_J = 1.602176634e-19
HBAR_EV = 6.582119569e-16
HBAR_SI = HBAR_EV * EV_TO_J
THZ_TO_RAD = 2 * np.pi * 1e12
EV_PER_A2_TO_SI = EV_TO_J / (1e-10) ** 2
CONVERSION = EV_PER_A2_TO_SI / AMU_KG               # ~ 9.648e27
CONVERSION_THZ2 = CONVERSION / THZ_TO_RAD**2        # ~ 244.5
CONVERSION_FC3 = CONVERSION / (1e-10 * np.sqrt(AMU_KG))
CONVERSION_FC3_THZ = CONVERSION_FC3 / THZ_TO_RAD**2.5
UU_PREFACTOR = HBAR_SI / (THZ_TO_RAD * AMU_KG * 1e-20)


def assemble_device_fc3_tensor(phi_blocks, n_blocks, n_dof):
    """Dense device FC3 tensor Phi3_dev[A, B, C] (device DOFs) from the
    {(I, K, K'): Phi[n_dof, n_dof, n_dof]} bubble blocks"""
    N_D = n_blocks * n_dof
    tensor = np.zeros((N_D, N_D, N_D), dtype=float)
    for (I, K, Kp), blk in phi_blocks.items():
        tensor[I * n_dof:(I + 1) * n_dof,
               K * n_dof:(K + 1) * n_dof,
               Kp * n_dof:(Kp + 1) * n_dof] = np.asarray(blk).real
    return tensor


def device_fc3_mass_weighted(phi_blocks, n_blocks, n_dof):
    """Device FC3 in the loop/tadpole mass-weighting"""
    return assemble_device_fc3_tensor(phi_blocks, n_blocks, n_dof) / CONVERSION_FC3_THZ


def equal_time_uu_from_sum(g_lesser_freq_sum, dw_thz):
    """Equal-time mass-weighted <w_a w_b> [amu*A^2] from the (already
    frequency-summed) sum_omega G^<(omega) device matrix.

    <uu> = UU_PREFACTOR * (dw/2pi) * sum_omega i G^<
    """
    uu = 1j * np.asarray(g_lesser_freq_sum) * (dw_thz / (2.0 * np.pi)) * UU_PREFACTOR
    uu = 0.5 * (uu + uu.conj().T)
    return uu.real


def tadpole_source(fc3_mw, uu):
    """Force source s_a = 1/2 Phi3_{acd} <u_c u_d>"""
    return 0.5 * np.einsum("acd,cd->a", fc3_mw, uu, optimize=True)


def mean_displacement(fc3_mw, uu, phi_eff, *, optical_projector=None,
                      omega2_floor_abs=0.0):
    """Static mean displacement <w> = -CONVERSION_THZ2 Phi_eff^{+} s.

    Solves Phi_eff <w> = -CONVERSION_THZ2 s on the subspace above
    ``omega2_floor_abs`` [THz^2] (config ``phonon.scp_floor_thz`` squared).
    """
    s = tadpole_source(fc3_mw, uu)
    rhs = -CONVERSION_THZ2 * s
    if optical_projector is not None:
        rhs = optical_projector @ rhs
    d = 0.5 * (np.asarray(phi_eff) + np.asarray(phi_eff).conj().T).real
    evals, evecs = np.linalg.eigh(d)
    inv = np.where(evals > float(omega2_floor_abs), 1.0 / evals, 0.0)
    w_mean = evecs @ (inv * (evecs.conj().T @ rhs))
    if optical_projector is not None:
        w_mean = optical_projector @ w_mean
    return w_mean.real


def sigma_tadpole(fc3_mw, w_mean):
    """Cubic tadpole self-energy Sigma_T = Phi3 : <u> [THz^2]"""
    sig = CONVERSION_THZ2 * np.einsum("abc,c->ab", fc3_mw, w_mean, optimize=True)
    return 0.5 * (sig + sig.conj().T).real


CONVERSION_FC4 = CONVERSION / (1e-20 * AMU_KG)
CONVERSION_FC4_THZ = CONVERSION_FC4 / THZ_TO_RAD**3


def sigma_loop_blocks(fc4_blocks, uu, n_blocks, n_dof):
    """Quartic (SCP) loop self-energy Sigma_L = 1/2 Phi4 : <uu> [THz^2].

    ``fc4_blocks``: {(I, J, K, Kp): Phi4[n_dof, n_dof, n_dof, n_dof]}
    mass-weighted quartic device blocks in eV/(A^4 amu^2) — the (I, J)
    legs are the Sigma indices, (K, Kp) are contracted against the
    equal-time mass-weighted <w w> [amu A^2] (equal_time_uu_from_sum),
    giving eV/(A^2 amu) -> CONVERSION_THZ2 -> THz^2. For a locally
    stable quartic the loop is the SCP stiffening term: it GROWS with
    <uu>, providing the restoring feedback the cubic-only bubble lacks
    on soft-mode structures.
    """
    N_D = n_blocks * n_dof
    sig = np.zeros((N_D, N_D), dtype=float)
    uu = np.asarray(uu)
    for (I, J, K, Kp), blk in fc4_blocks.items():
        uu_blk = uu[K * n_dof:(K + 1) * n_dof, Kp * n_dof:(Kp + 1) * n_dof]
        sig[I * n_dof:(I + 1) * n_dof, J * n_dof:(J + 1) * n_dof] += (
            0.5 * np.einsum("abcd,cd->ab", np.asarray(blk).real, uu_blk,
                            optimize=True))
    sig *= CONVERSION_THZ2
    return 0.5 * (sig + sig.T)


def equal_time_uu_dressed(phi_eff, temperature_k, floor_thz=1e-3):
    """SCP closed-form equal-time <w w> [amu A^2] from the dressed
    harmonic model: sum_modes (hbar/2 omega) coth(hbar omega / 2 kT)
    |e><e| over the eigenmodes of Phi_eff [THz^2]. Exact in equilibrium
    and immune to the eta=0 NEGF quadrature ill-conditioning that makes
    the raw G^< integral unusable on IR-resolved grids (modes below
    ``floor_thz`` are excluded)."""
    d = 0.5 * (np.asarray(phi_eff) + np.asarray(phi_eff).T).real
    evals, evecs = np.linalg.eigh(d)
    w = np.sqrt(np.clip(evals, 0.0, None))
    keep = w > float(floor_thz)
    x = HBAR_EV * THZ_TO_RAD * w[keep] / (2.0 * 8.617333262e-5 * temperature_k)
    amp = UU_PREFACTOR * 0.5 / w[keep] / np.tanh(x)
    V = evecs[:, keep]
    return (V * amp) @ V.T
