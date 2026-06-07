"""Self-consistent static (SCP) self-energy for the production phonon SCBA.

The dynamic 3-phonon bubble (``SigmaPhononPhonon``) is the leading anharmonic
self-energy, but on a *soft*-mode structure (e.g. a 1-D wire whose
quasi-Goldstone twist sits near zero frequency) its Bose-enhanced ``G^<`` at
``omega -> 0`` injects an IR singularity that destabilises the SCBA. The
self-consistent-phonon (SCP) **cubic tadpole**

    Sigma_T = Phi3 : <u>,   <u> = -Phi_eff^{-1} ( 1/2 Phi3 : <uu> )

is a STATIC (frequency-independent) real self-energy that *stiffens* the soft
mode (raises its frequency) -- the physically-correct finite-T renormalisation
(self-consistent phonon / SSCHA; Paulatto, Errea & Calandra, PRB 91, 054304,
2015) -- so the bubble becomes stable. Recomputed every SCBA iteration from the
current device ``G^<`` it self-limits: as the mode stiffens, ``<uu> ~ 1/omega^2``
drops, and ``||Sigma_T||`` saturates.

This module is a self-contained port of the device-level tadpole pieces of the
research ``phonon/solver/static_se.py`` (which is not importable from the
production package); the numerics and unit conventions are identical, so the
self-energy matches the dense reference. The quartic loop (which needs FC4 the
production pipeline does not carry) is deliberately omitted.
"""

from __future__ import annotations

import numpy as np

# --- physical constants / unit conversions (identical to
#     phonon_inputs/constants.py, so the FC3 written by that pipeline -- which
#     carries CONVERSION_FC3_THZ -- is consumed consistently here). ----------
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
# <w_a w_b> = UU_PREFACTOR * sum_omega i G^<(omega) dw/(2 pi)  [amu * A^2]
UU_PREFACTOR = HBAR_SI / (THZ_TO_RAD * AMU_KG * 1e-20)


def assemble_device_fc3_tensor(phi_blocks, n_blocks, n_dof):
    """Dense device FC3 tensor ``Phi3_dev[A, B, C]`` (device DOFs) from the
    ``{(I, K, K'): Phi[n_dof, n_dof, n_dof]}`` bubble blocks (carrying
    ``CONVERSION_FC3_THZ``). Returns ``(N_D, N_D, N_D)`` real."""
    N_D = n_blocks * n_dof
    tensor = np.zeros((N_D, N_D, N_D), dtype=float)
    for (I, K, Kp), blk in phi_blocks.items():
        tensor[I * n_dof:(I + 1) * n_dof,
               K * n_dof:(K + 1) * n_dof,
               Kp * n_dof:(Kp + 1) * n_dof] = np.asarray(blk).real
    return tensor


def device_fc3_mass_weighted(phi_blocks, n_blocks, n_dof):
    """Device FC3 in the loop/tadpole mass-weighting (the bubble blocks divided
    by ``CONVERSION_FC3_THZ``; the tadpole bridges to THz^2 with
    ``CONVERSION_THZ2`` instead). Returns ``(N_D, N_D, N_D)``."""
    return assemble_device_fc3_tensor(phi_blocks, n_blocks, n_dof) / CONVERSION_FC3_THZ


def equal_time_uu_from_sum(g_lesser_freq_sum, dw_thz):
    """Equal-time mass-weighted ``<w_a w_b>`` [amu*A^2] from the (already
    frequency-summed) ``sum_omega G^<(omega)`` device matrix.

    ``i G^<`` equals ``n_B(omega) A(omega)`` (real, symmetric) on the grid, so
    ``<uu> = UU_PREFACTOR * (dw/2pi) * sum_omega i G^<`` is a real Riemann
    integral. The caller supplies ``sum_omega G^<`` (so the frequency reduction
    -- incl. any MPI all-reduce over a distributed energy axis -- happens
    upstream). Returns the real, symmetrised ``(N_D, N_D)`` matrix.
    """
    uu = 1j * np.asarray(g_lesser_freq_sum) * (dw_thz / (2.0 * np.pi)) * UU_PREFACTOR
    uu = 0.5 * (uu + uu.conj().T)
    return uu.real


def tadpole_source(fc3_mw, uu):
    """Thermally-dressed force source ``s_a = 1/2 Phi3_{acd} <u_c u_d>``."""
    return 0.5 * np.einsum("acd,cd->a", fc3_mw, uu)


def mean_displacement(fc3_mw, uu, phi_eff, *, optical_projector=None,
                      omega2_floor_rel=1e-3, omega2_floor_abs=None):
    """Static mean displacement ``<w> = -CONVERSION_THZ2 Phi_eff^{+} s``.

    Solves ``Phi_eff <w> = -CONVERSION_THZ2 s`` for the finite-T relaxation of
    the internal coordinates, with a **regularised pseudo-inverse on Phi_eff's
    own spectrum**: invert only eigenvalues above
    ``max(omega2_floor_rel*max|eig|, omega2_floor_abs)``, dropping the
    rigid-body / soft / unstable modes (whose ~1/omega^2 amplification would
    otherwise blow the solve up to NaN). Robust as ``Phi_eff`` drifts across
    SCBA iterations. ``optical_projector`` (if given) additionally removes the
    symmetry-defined components. Returns ``<w>`` in ``sqrt(amu)*Angstrom``.
    """
    s = tadpole_source(fc3_mw, uu)
    rhs = -CONVERSION_THZ2 * s
    if optical_projector is not None:
        rhs = optical_projector @ rhs
    d = 0.5 * (np.asarray(phi_eff) + np.asarray(phi_eff).conj().T).real
    evals, evecs = np.linalg.eigh(d)
    scale = float(np.max(np.abs(evals))) + 1e-300
    cutoff = float(omega2_floor_rel) * scale
    if omega2_floor_abs is not None:
        cutoff = max(cutoff, float(omega2_floor_abs))
    inv = np.where(evals > cutoff, 1.0 / evals, 0.0)
    w_mean = evecs @ (inv * (evecs.conj().T @ rhs))
    if optical_projector is not None:
        w_mean = optical_projector @ w_mean
    return w_mean.real


def sigma_tadpole(fc3_mw, w_mean):
    """Cubic tadpole self-energy ``Sigma_T = Phi3 : <u>`` [THz^2], static, real,
    symmetric; added to the dynamical matrix. ~0 for a relaxed symmetric
    crystal (then ``<w> ~ 0``)."""
    sig = CONVERSION_THZ2 * np.einsum("abc,c->ab", fc3_mw, w_mean)
    return 0.5 * (sig + sig.conj().T).real
