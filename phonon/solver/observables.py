"""Dense-reference phonon observables from the fixed-point Green's functions.

Post-processing of the ``transmission(..., return_greens=True)`` output:
every function takes plain arrays in the dense solver's internal units
(frequencies in THz, matrices in THz^2, textbook signs G^< = -i n A) and
mirrors its spectral-current conventions, so contact bond currents equal
``spectral_J_L/R`` and all heat spectra integrate as ``sum * dw * 1e12``.

Implements the observables of the thesis theory chapter: local DOS
(eq:dos), non-equilibrium occupation (eq:neq_occupation), local effective
temperature (eq:Teff_local), mean-square displacement, nearest-bond and
all-crossing-pair harmonic cut currents (eq:I_local), the scattering terminal
current and per-slab absorption
(eq:Ja, eq:P_abs), the energy sum rule D(omega) (eq:sumrule), the
telescoped bond-current spread, the linear-response thermal conductance
(eq:G_thermal) and the mode-resolved Fisher-Lee transmission (eq:mode_T).
"""

from __future__ import annotations

import numpy as np

from phonon_inputs.constants import HBAR_SI, KB_EV, THZ_TO_RAD

_THZ_TO_EV = 6.582119569e-16 * THZ_TO_RAD  # hbar * (2 pi 1e12) in eV


def bose(freqs_thz, temperature):
    """Bose-Einstein occupation on a THz grid; the omega <= 0 bins are 0."""
    f = np.asarray(freqs_thz, dtype=float)
    n = np.zeros_like(f)
    pos = f > 1e-12
    x = _THZ_TO_EV * f[pos] / (KB_EV * float(temperature))
    n[pos] = 1.0 / np.expm1(x)
    return n


def local_dos(G_retarded, freqs_thz):
    """Per-DOF local density of states rho_mu(omega) = -(2 omega/pi) Im G^R_mumu.

    eq:dos with the omega^2 spectral variable of the phonon Dyson equation:
    rho(omega) = 2 omega rho(omega^2). Normalised such that
    ``sum(rho, axis=0) * dw = 1`` per DOF over the full symmetric axis
    (the eq:sum_rule_zero check).
    """
    w = np.asarray(freqs_thz, dtype=float)
    diag = np.diagonal(np.asarray(G_retarded), axis1=-2, axis2=-1)
    return -(2.0 * w[:, None] / np.pi) * diag.imag


def local_occupation(G_lesser, G_greater, n_dof=None, n_slabs=None):
    """Non-equilibrium occupation n(omega) = Tr[i G^<] / Tr[i (G^> - G^<)].

    eq:neq_occupation with textbook signs (G^< = -i n A, so i G^< = n A
    and i (G^> - G^<) = A). With ``n_dof``/``n_slabs`` the traces are
    slab-restricted and the result is (nfreq, n_slabs); otherwise per-DOF
    diagonals are used (nfreq, N_D).
    """
    gl = np.diagonal(np.asarray(G_lesser), axis1=-2, axis2=-1)
    gg = np.diagonal(np.asarray(G_greater), axis1=-2, axis2=-1)
    if n_dof is not None and n_slabs is not None:
        gl = gl.reshape(gl.shape[0], n_slabs, n_dof).sum(axis=-1)
        gg = gg.reshape(gg.shape[0], n_slabs, n_dof).sum(axis=-1)
    num = (1j * gl).real
    den = (1j * (gg - gl)).real
    return num / np.where(np.abs(den) > 1e-30, den, 1e-30)


def effective_temperature(freqs_thz, trace_iGl, trace_A, bracket=(1.0, 2000.0)):
    """Local effective temperature per site/slab (eq:Teff_local).

    Solves for T such that the Bose energy density matches the
    non-equilibrium one:

        int domega omega * A_i(omega) * n_BE(omega, T)
            = int domega omega * [i G^<]_i(omega).

    Parameters are the positive-frequency traces (nfreq, nsites):
    ``trace_iGl`` = Re[i G^<]_i and ``trace_A`` = spectral weight A_i.
    Returns (nsites,) temperatures in K; NaN where the bracket fails.
    """
    from scipy.optimize import brentq

    w = np.asarray(freqs_thz, dtype=float)
    iGl = np.asarray(trace_iGl, dtype=float)
    A = np.asarray(trace_A, dtype=float)
    target = np.sum(w[:, None] * iGl, axis=0)
    out = np.full(iGl.shape[1], np.nan)
    for i in range(iGl.shape[1]):
        def f(T, i=i):
            return float(np.sum(w * A[:, i] * bose(w, T)) - target[i])
        lo, hi = bracket
        try:
            # F(T) is monotonically increasing in T (dn_BE/dT > 0); widen
            # the bracket geometrically if it does not straddle the root.
            for _ in range(8):
                if f(lo) * f(hi) <= 0:
                    out[i] = brentq(f, lo, hi, xtol=1e-6)
                    break
                lo, hi = lo / 2.0, hi * 2.0
        except ValueError:
            pass
    return out


def mean_square_displacement(G_lesser, dw_thz, pos_mask=None):
    """Equal-time mass-weighted <u u> [amu A^2] from the frequency-summed G^<.

    Wraps :func:`phonon.solver.static_se.equal_time_uu` (textbook G^<); on
    a full symmetric axis pass the plain sum, on a positive-only grid use
    the bosonic fold first. ``pos_mask`` restricts the sum.
    """
    from .static_se import equal_time_uu

    G = np.asarray(G_lesser)
    if pos_mask is not None:
        G = G[pos_mask]
    return equal_time_uu(G, dw_thz)


def bond_currents(G_lesser, H_D, freqs_thz, n_dof, n_slabs):
    """Per-interface Hardy bond-current spectra (eq:I_local).

    j_{i,i+1}(omega) = -2 hbar omega_rad Re Tr[H_{i,i+1} G^<_{i+1,i}(omega)],

    in the same units as the dense solver's ``spectral_J_L/R`` (integrate
    with ``sum * dw * 1e12``). Returns (nfreq, n_slabs - 1).
    """
    G = np.asarray(G_lesser)
    H = np.asarray(H_D)
    w_rad = np.asarray(freqs_thz, dtype=float) * THZ_TO_RAD
    out = np.zeros((G.shape[0], n_slabs - 1))
    for i in range(n_slabs - 1):
        ri = slice(i * n_dof, (i + 1) * n_dof)
        rj = slice((i + 1) * n_dof, (i + 2) * n_dof)
        tr = np.trace(H[ri, rj] @ G[:, rj, ri], axis1=-2, axis2=-1)
        out[:, i] = -2.0 * HBAR_SI * w_rad * tr.real
    return out


def harmonic_cut_currents(G_lesser, H_D, freqs_thz, n_dof, n_slabs):
    """Harmonic energy-current spectra with every FC2 pair crossing a cut.

    For the cut after slab ``i``, this evaluates

    ``-2 hbar omega Re Tr[H[L,R] G^<[R,L]]``

    with ``L = {0, ..., i}`` and ``R = {i + 1, ..., n_slabs - 1}``.
    It reduces to :func:`bond_currents` for a block-tridiagonal FC2 matrix,
    but remains complete when force constants span more than one slab.  This
    is the blocking-independent harmonic part of a local continuity audit;
    an anharmonic calculation must additionally account for the interaction
    power/current generated by an off-diagonal scattering self-energy.
    """
    G = np.asarray(G_lesser)
    H = np.asarray(H_D)
    n_total = int(n_dof) * int(n_slabs)
    if H.shape != (n_total, n_total):
        raise ValueError(
            f"H_D has shape {H.shape}, expected {(n_total, n_total)}")
    if G.shape[-2:] != H.shape:
        raise ValueError(
            f"G_lesser has matrix shape {G.shape[-2:]}, expected {H.shape}")

    w_rad = np.asarray(freqs_thz, dtype=float) * THZ_TO_RAD
    if G.shape[0] != w_rad.size:
        raise ValueError(
            f"G_lesser has {G.shape[0]} frequencies, expected {w_rad.size}")
    out = np.zeros((G.shape[0], int(n_slabs) - 1))
    for i in range(int(n_slabs) - 1):
        split = (i + 1) * int(n_dof)
        tr = np.trace(
            H[:split, split:] @ G[:, split:, :split], axis1=-2, axis2=-1)
        out[:, i] = -2.0 * HBAR_SI * w_rad * tr.real
    return out


def meir_wingreen_spectra(G_lesser, G_greater, obc, freqs_thz, n_dof, n_slabs):
    """Lead heat-current spectra (eq:meir_wingreen), the dense convention.

    Returns ``(spec_L, spec_R)``, each (nfreq,), identical in form to the
    solver's ``spectral_J_L/R`` (both positive for left-to-right flow).
    """
    w_rad = np.asarray(freqs_thz, dtype=float) * THZ_TO_RAD
    sl0 = slice(0, n_dof)
    slN = slice((n_slabs - 1) * n_dof, n_slabs * n_dof)
    Gl, Gg = np.asarray(G_lesser), np.asarray(G_greater)
    tl = np.trace(
        obc["Sigma_L_greater"][:, sl0, sl0] @ Gl[:, sl0, sl0]
        - obc["Sigma_L_lesser"][:, sl0, sl0] @ Gg[:, sl0, sl0],
        axis1=-2, axis2=-1)
    tr = np.trace(
        obc["Sigma_R_lesser"][:, slN, slN] @ Gg[:, slN, slN]
        - obc["Sigma_R_greater"][:, slN, slN] @ Gl[:, slN, slN],
        axis1=-2, axis2=-1)
    return HBAR_SI * w_rad * tl.real, HBAR_SI * w_rad * tr.real


def slab_absorption(Sigma_l, Sigma_g, G_lesser, G_greater, freqs_thz,
                    n_dof, n_slabs):
    """Per-slab scattering absorption spectra (eq:P_abs, row-binned trace).

    P_abs(k, omega) = hbar omega_rad Re Tr_k[Sigma_s^> G^< - Sigma_s^< G^>],
    with Tr_k restricted to rows in slab k (the attribution verified to
    reconstruct the interior bond currents). Returns (nfreq, n_slabs).
    """
    w_rad = np.asarray(freqs_thz, dtype=float) * THZ_TO_RAD
    Sl, Sg = np.asarray(Sigma_l), np.asarray(Sigma_g)
    Gl, Gg = np.asarray(G_lesser), np.asarray(G_greater)
    prod = Sg @ Gl - Sl @ Gg   # (nfreq, N_D, N_D)
    diag = np.diagonal(prod, axis1=-2, axis2=-1).real
    per_slab = diag.reshape(diag.shape[0], n_slabs, n_dof).sum(axis=-1)
    return HBAR_SI * w_rad[:, None] * per_slab


def scattering_terminal_current(Sigma_l, Sigma_g, G_lesser, G_greater,
                                freqs_thz, n_dof, n_slabs):
    """Scattering terminal-current spectrum J_s(omega) (eq:Ja third terminal):
    the slab-summed absorption. Returns (nfreq,)."""
    return slab_absorption(Sigma_l, Sigma_g, G_lesser, G_greater,
                           freqs_thz, n_dof, n_slabs).sum(axis=-1)


def sumrule_D_omega(spec_L, spec_R, J_s_spec):
    """Energy sum rule D(omega) = J_L(omega) - J_R(omega) - J_s(omega)
    (eq:sumrule with the dense left-to-right sign convention: energy in
    from the left lead = energy out the right lead + energy absorbed by
    the scattering terminal). Vanishes at the Phi-derivable fixed point."""
    return np.asarray(spec_L) - np.asarray(spec_R) - np.asarray(J_s_spec)


def telescoped_spread(interface_currents, p_abs_per_slab):
    """Corrected bond-current convergence criterion: the spread of the
    telescoped currents I_{i,i+1} - sum_{j<=i} P_abs(j), which removes the
    physical interaction-channel redistribution from the raw spread."""
    heat = np.asarray(interface_currents, dtype=float)
    pa = np.asarray(p_abs_per_slab, dtype=float)
    recon = heat[0] + np.concatenate(([0.0], np.cumsum(pa)[: heat.size - 1]))
    resid = heat - recon
    return float(np.max(resid) - np.min(resid))


def thermal_conductance(T_of_omega, freqs_thz, temperature, area_m2=None):
    """Linear-response Landauer thermal conductance (eq:G_thermal):

        G = (1 / 2 pi) int domega_rad  hbar omega_rad
            (dn_BE/dT) T(omega)          [W / K]

    ``T_of_omega`` is the (effective) transmission on ``freqs_thz``
    (positive axis). Divides by ``area_m2`` when given (W / m^2 / K).
    """
    w = np.asarray(freqs_thz, dtype=float)
    T_w = np.asarray(T_of_omega, dtype=float)
    pos = w > 1e-12
    w_rad = w[pos] * THZ_TO_RAD
    x = _THZ_TO_EV * w[pos] / (KB_EV * float(temperature))
    # dn_BE/dT = x/T * e^x / (e^x - 1)^2
    with np.errstate(over="ignore"):
        ex = np.exp(np.clip(x, None, 500.0))
        dndT = (x / float(temperature)) * ex / np.expm1(x) ** 2
    dw_rad = (w[1] - w[0]) * THZ_TO_RAD
    G = np.sum(HBAR_SI * w_rad * dndT * T_w[pos]) * dw_rad / (2.0 * np.pi)
    if area_m2:
        G = G / float(area_m2)
    return float(G)


def fisher_lee_modes(G_retarded, Gamma_L, Gamma_R, n_dof, n_slabs,
                     floor=1e-12):
    """Mode-resolved Fisher-Lee transmission amplitudes (eq:mode_T).

    t(omega) = Gamma_R^{1/2} G^R_{N1} Gamma_L^{1/2} via the eigen-
    decompositions of the lead broadenings (eigenvalues clipped at 0;
    channels with weight below ``floor * max`` dropped). Returns
    ``(T_mode, T_total)``: the per-channel transmissions (nfreq, n_ch)
    zero-padded to the max channel count, and their sum, which satisfies
    ``T_total = Tr[Gamma_L G^A Gamma_R G^R]`` (Caroli) exactly.
    """
    GR = np.asarray(G_retarded)
    nfreq = GR.shape[0]
    sl0 = slice(0, n_dof)
    slN = slice((n_slabs - 1) * n_dof, n_slabs * n_dof)
    T_mode = np.zeros((nfreq, n_dof))
    T_tot = np.zeros(nfreq)
    for k in range(nfreq):
        gl = np.asarray(Gamma_L)[k, sl0, sl0]
        gr = np.asarray(Gamma_R)[k, slN, slN]
        wl, Ul = np.linalg.eigh(0.5 * (gl + gl.conj().T))
        wr, Ur = np.linalg.eigh(0.5 * (gr + gr.conj().T))
        wl = np.clip(wl.real, 0.0, None)
        wr = np.clip(wr.real, 0.0, None)
        g_N1 = GR[k, slN, sl0]
        t = (np.sqrt(wr)[:, None] * (Ur.conj().T @ g_N1 @ Ul)) * np.sqrt(wl)
        # Per-incoming-channel transmission, sorted descending.
        tc = np.sort(np.sum(np.abs(t) ** 2, axis=0))[::-1]
        T_mode[k, : tc.size] = tc[:n_dof]
        T_tot[k] = float(np.sum(np.abs(t) ** 2))
    if floor > 0:
        T_mode[T_mode < floor * max(T_tot.max(), 1e-300)] = 0.0
    return T_mode, T_tot
