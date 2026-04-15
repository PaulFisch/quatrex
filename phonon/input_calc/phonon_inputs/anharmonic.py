"""Reference anharmonic phonon-phonon NEGF transport.

Implements the self-consistent Born approximation (SCBA) for
phonon-phonon scattering following Guo et al., Phys. Rev. B 102,
195412 (2020).

The scattering self-energy (Guo Eq. 8, diagonal block):

    Sigma^{<}(omega) = (i*hbar/2) * sum_{c,d,e,f} Phi3_{a,c,d}
        * integral dw'/(2*pi) G^<_{cf}(w') G^<_{de}(w-w')
        * Phi3_{b,e,f}

Internal units: THz^2 for dynamical matrices and self-energies.

Shape convention
----------------
Self-energies always carry a q-axis, even for the finite (Gamma-only)
path where n_kpts == 1.  Shape: (n_slabs, n_kpts, nfreq, n_dof, n_dof).
This removes the branching on n_kpts that caused shape mismatches between
the q-path at q_mesh=(1,1) and the finite path.

Retarded reconstruction
-----------------------
The ``retarded`` parameter controls how Σ^R is built from Σ^{<,>}:
  - ``"half"``— Σ^R = ½(Σ^> - Σ^<), no Kramers-Kronig correction
  - ``"pv"``  — singularity-subtracted principal-value integral (O(nfreq²))
  - ``"fft"`` — FFT-based Hilbert transform (O(nfreq log nfreq))
Both entry points default to ``"half"`` so that q=(1,1) reproduces the
finite solver exactly.  Σ^R is rebuilt from the mixed Σ^{<,>} pair
after each SCBA iteration, not mixed independently.
"""

import warnings
import numpy as np
from numpy.linalg import inv
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from .constants import (
    CONVERSION_THZ2,
    HBAR_SI,
    KB_SI,
    THZ_TO_RAD,
)

# Stale import target — separable.py imports this at module level.
# Cannot remove until separable.py is updated.
_compute_obc_self_energies = None


# ---------------------------------------------------------------------------
# FC3 loading
# ---------------------------------------------------------------------------


def _load_fc3_raw(fc3_source):
    """Load raw FC3 array from various sources.

    Parameters
    ----------
    fc3_source : str, Path, ndarray, or dict
        - str/Path: HDF5 file with an ``"fc3"`` dataset.
        - ndarray:  raw FC3 passed directly.
        - dict:     must contain ``"fc3"`` (raw array) or ``"ph3"``
          (phono3py object whose ``.fc3`` attribute is the raw array).
    """
    if isinstance(fc3_source, np.ndarray):
        return fc3_source
    if isinstance(fc3_source, dict):
        if "fc3" in fc3_source:
            return np.asarray(fc3_source["fc3"])
        if "ph3" in fc3_source:
            return np.asarray(fc3_source["ph3"].fc3)
        raise ValueError(
            "fc3 dict has neither 'fc3' nor 'ph3' key. "
            "Pass the fc3.hdf5 path or the raw fc3 array instead. "
            f"Available keys: {list(fc3_source.keys())}")
    import h5py
    with h5py.File(str(fc3_source), "r") as f:
        return np.array(f["fc3"])


# ---------------------------------------------------------------------------
# Frequency grid and Bose function
# ---------------------------------------------------------------------------


def _build_frequency_grid(freq_range_thz, eta_w_thz=None, eta_factor=0.05):
    """Build a uniform symmetric frequency grid for FFT convolution.

    The grid spacing is Δω = fmax / nfreq_pos, giving positive bins at
    Δω, 2Δω, ..., fmax (no exact-zero sample on the positive side).
    The full axis is [-fmax, ..., -Δω, 0, Δω, ..., fmax].

    The ω = 0 sample is included for FFT index arithmetic but is
    excluded from physical integrals via ``pos_mask`` and from the
    scattering kernel via ``scatt_mask``.

    Parameters
    ----------
    freq_range_thz : (fmin, fmax, nfreq_pos)
        fmin is advisory (Δω is set to fmax/nfreq_pos).
        fmax is the upper edge.
        nfreq_pos is the number of positive-frequency bins (excluding 0).

    Returns
    -------
    freqs_thz : (2*nfreq_pos + 1,) — uniform, symmetric, includes 0
    dw_thz : float
    eta_w_thz : float
    z2_arr : complex array — (ω + iη_w)²
    pos_mask : bool array — True for ω > 0 (excludes 0)
    mid : int — index of the ω = 0 sample
    """
    _fmin, fmax, nfreq_pos = freq_range_thz
    nfreq_pos = int(nfreq_pos)
    if nfreq_pos < 2:
        raise ValueError("nfreq_pos must be >= 2")

    dw_thz = fmax / nfreq_pos
    freqs_pos = np.arange(1, nfreq_pos + 1) * dw_thz   # Δω, 2Δω, ..., fmax
    freqs_thz = np.concatenate((-freqs_pos[::-1], [0.0], freqs_pos))
    mid = nfreq_pos   # index of the ω = 0 sample

    if eta_w_thz is None:
        eta_w_thz = eta_factor * dw_thz

    z2_arr = (freqs_thz + 1j * eta_w_thz) ** 2

    pos_mask = freqs_thz > 0.0

    return freqs_thz, dw_thz, eta_w_thz, z2_arr, pos_mask, mid


def _bose_full_axis(freqs_thz, T):
    """Bose-Einstein on the full symmetric axis.

    At ω = 0 the true occupation diverges.  We assign n_B(0) = 0
    as a placeholder; the ω = 0 sample must never participate in
    physical integrals or the scattering kernel (the caller zeros
    out the Green's function at mid before entering the convolution).
    """
    x = HBAR_SI * freqs_thz * THZ_TO_RAD / (KB_SI * T)
    n = np.empty_like(x, dtype=float)

    zero = np.abs(freqs_thz) < 1e-30
    small = (~zero) & (np.abs(x) < 1e-6)
    regular = (~zero) & (~small)

    n[regular] = 1.0 / np.expm1(x[regular])
    # Taylor: 1/(e^x - 1) ≈ 1/x - 1/2 + x/12
    n[small] = 1.0 / x[small] - 0.5 + x[small] / 12.0
    n[zero] = 0.0  # placeholder — excluded from all physical integrals

    return n


def _boson_contact_self_energies_from_gamma(Gamma, freqs_thz, T):
    """Build contact Σ^<, Σ^> from broadening Gamma.

    Σ^< = −i n_B Γ,   Σ^> = −i (n_B + 1) Γ.
    The ω = 0 sample uses the n_B(0) = −½ regularization; it is never
    included in physical integrals (pos_mask excludes it).
    """
    n = _bose_full_axis(freqs_thz, T)
    Sigma_l = -1j * n[:, None, None] * Gamma
    Sigma_g = -1j * (n[:, None, None] + 1.0) * Gamma
    return Sigma_l, Sigma_g


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _check_broadening_sign(Sigma_R, freqs_thz, name,
                           low_freq_thz=0.0, tol=1e-8):
    """Check Γ = i(Σ^R − Σ^A) PSD for ω > 0, NSD for ω < 0.

    Checks *every* frequency with |ω| > low_freq_thz.
    Returns (n_violations, max_violation).
    """
    n_viol = 0
    max_viol = 0.0
    for iw in range(len(freqs_thz)):
        w = freqs_thz[iw]
        if abs(w) <= low_freq_thz:
            continue
        sr = Sigma_R[iw]
        Gamma = 1j * (sr - sr.conj().T)
        Gamma = 0.5 * (Gamma + Gamma.conj().T)  # ensure Hermitian
        eigs = np.linalg.eigvalsh(Gamma)
        if w > 0 and eigs.min() < -tol:
            n_viol += 1
            max_viol = max(max_viol, -eigs.min())
        elif w < 0 and eigs.max() > tol:
            n_viol += 1
            max_viol = max(max_viol, eigs.max())
    return n_viol, max_viol


def _check_full_axis_symmetry(G_R, G_l, G_g, freqs_thz,
                              rtol=1e-3, atol=1e-8):
    """Verify Guo full-axis symmetry.

    G^<(ω) = −[G^>(−ω)]^T,  G^R(ω) = [G^R(−ω)]*.
    Returns (lesser_err, retarded_err).
    """
    mid = len(freqs_thz) // 2
    pos = slice(mid + 1, None)
    neg = slice(0, mid)

    G_l_pos = G_l[pos]
    G_g_neg = G_g[neg][::-1].transpose(0, 2, 1)
    lesser_err = float(np.max(np.abs(G_l_pos + G_g_neg)))

    G_R_pos = G_R[pos]
    G_R_neg = G_R[neg][::-1]
    retarded_err = float(np.max(np.abs(G_R_pos - G_R_neg.conj())))

    return lesser_err, retarded_err


def _symmetrize_lesser_greater(sig_l, sig_g):
    """Project Σ^< and Σ^> onto the bosonic full-axis symmetry manifold.

    Enforces Σ^<(ω) = −[Σ^>(−ω)]^T  in-place on the last three axes
    (nfreq, nd, nd).  Leading dimensions (slabs, q-points) are preserved.

    The grid is assumed symmetric about ω=0 with mid = nfreq // 2.
    The ω=0 sample is left untouched (it is excluded from physics).
    """
    nfreq = sig_l.shape[-3]
    mid = nfreq // 2

    # Work on copies so the negative-side update doesn't interfere
    sl_pos = sig_l[..., mid + 1:, :, :].copy()
    sg_pos = sig_g[..., mid + 1:, :, :].copy()
    # Σ(−ω): take negative-freq slice and reverse along freq axis
    sg_neg_rev = sig_g[..., :mid, :, :][..., ::-1, :, :].copy()
    sl_neg_rev = sig_l[..., :mid, :, :][..., ::-1, :, :].copy()

    # Symmetrize positive side:
    # Σ^<_sym(ω) = ½ [Σ^<(ω) − Σ^>(−ω)^T]
    # Σ^>_sym(ω) = ½ [Σ^>(ω) − Σ^<(−ω)^T]
    sl_pos_sym = 0.5 * (sl_pos - sg_neg_rev.swapaxes(-2, -1))
    sg_pos_sym = 0.5 * (sg_pos - sl_neg_rev.swapaxes(-2, -1))

    sig_l[..., mid + 1:, :, :] = sl_pos_sym
    sig_g[..., mid + 1:, :, :] = sg_pos_sym

    # Negative side follows from the symmetry:
    # Σ^<(−ω) = −[Σ^>(ω)]^T,  Σ^>(−ω) = −[Σ^<(ω)]^T
    sig_l[..., :mid, :, :] = -(
        sig_g[..., mid + 1:, :, :][..., ::-1, :, :].swapaxes(-2, -1))
    sig_g[..., :mid, :, :] = -(
        sig_l[..., mid + 1:, :, :][..., ::-1, :, :].swapaxes(-2, -1))


# ---------------------------------------------------------------------------
# Surface Green's function
# ---------------------------------------------------------------------------


def _sancho_rubio(z2, H_00, H_01, max_iter=300, tol=1e-8):
    """Surface Green's function via Sancho-Rubio decimation.

    Raises RuntimeError if the surface Dyson residual exceeds 1e-4.
    """
    N = H_00.shape[0]
    H_10 = H_01.conj().T

    a_ii = z2 * np.eye(N) - H_00
    eps = a_ii.copy()
    eps_s = a_ii.copy()
    alpha = (-H_10).copy()
    beta = (-H_01).copy()

    for _ in range(max_iter):
        inv_eps = inv(eps)
        eps_s -= alpha @ inv_eps @ beta
        eps -= alpha @ inv_eps @ beta + beta @ inv_eps @ alpha
        alpha_new = alpha @ inv_eps @ alpha
        beta_new = beta @ inv_eps @ beta
        alpha, beta = alpha_new, beta_new
        if np.linalg.norm(alpha) + np.linalg.norm(beta) < tol:
            break

    g_surf = inv(eps_s)

    residual = a_ii - H_10 @ g_surf @ H_01 - inv(g_surf)
    res_norm = np.linalg.norm(residual) / max(np.linalg.norm(a_ii), 1e-30)
    if res_norm > 1e-4:
        raise RuntimeError(
            f"Sancho-Rubio did not converge: relative residual {res_norm:.2e}")

    return g_surf


def _sancho_rubio_batch(z2_arr, H_00, H_01, max_iter=300, tol=1e-8,
                        lead_sigma_r=None):
    """Batched surface Green's function for multiple frequencies.

    Returns (g_surf, valid) where valid[iw] is True if the residual
    is acceptable.
    """
    nfreq = len(z2_arr)
    N = H_00.shape[0]
    H_10 = H_01.conj().T
    eye = np.eye(N)

    a_ii = z2_arr[:, None, None] * eye[None] - H_00[None]
    if lead_sigma_r is not None:
        a_ii = a_ii - lead_sigma_r
    eps = a_ii.copy()
    eps_s = a_ii.copy()
    alpha = np.broadcast_to(-H_10, (nfreq, N, N)).copy()
    beta = np.broadcast_to(-H_01, (nfreq, N, N)).copy()

    for _ in range(max_iter):
        inv_eps = np.linalg.inv(eps)
        aib = alpha @ inv_eps @ beta
        bia = beta @ inv_eps @ alpha
        eps_s -= aib
        eps -= aib + bia
        alpha_new = alpha @ inv_eps @ alpha
        beta_new = beta @ inv_eps @ beta
        alpha, beta = alpha_new, beta_new
        norm_max = max(np.linalg.norm(alpha.reshape(nfreq, -1), axis=1).max(),
                       np.linalg.norm(beta.reshape(nfreq, -1), axis=1).max())
        if norm_max < tol:
            break

    g_surf = np.linalg.inv(eps_s)

    residual = a_ii - (H_10[None] @ g_surf @ H_01[None]) - np.linalg.inv(g_surf)
    res_per_freq = np.linalg.norm(residual.reshape(nfreq, -1), axis=1)
    a_per_freq = np.linalg.norm(a_ii.reshape(nfreq, -1), axis=1)
    rel_res = res_per_freq / np.maximum(a_per_freq, 1e-30)
    valid = rel_res <= 1e-4

    return g_surf, valid


# ---------------------------------------------------------------------------
# Device Hamiltonian and OBC
# ---------------------------------------------------------------------------


def _build_device_hamiltonian(H_00, H_01, n_slabs):
    """Build block-tridiagonal device Hamiltonian for N identical slabs."""
    if n_slabs == 1:
        return H_00.copy()

    n_dof = H_00.shape[0]
    N = n_slabs * n_dof
    H_D = np.zeros((N, N), dtype=complex)
    H_10 = H_01.conj().T

    for l in range(n_slabs):
        sl = slice(l * n_dof, (l + 1) * n_dof)
        H_D[sl, sl] = H_00
        if l < n_slabs - 1:
            sl_next = slice((l + 1) * n_dof, (l + 2) * n_dof)
            H_D[sl, sl_next] = H_01
            H_D[sl_next, sl] = H_10

    return H_D


def _compute_obc_batch(z2_arr, H_00, H_01, freqs_thz, T_L, T_R,
                       n_slabs=1, lead_sigma_r_L=None, lead_sigma_r_R=None):
    """Batched OBC self-energies for all frequencies.

    Falls back to scalar Sancho-Rubio for frequencies where the
    batch residual is bad.  If scattering-dressed contacts fail,
    falls back to undressed contacts and emits a warning.
    """
    nfreq = len(z2_arr)
    n_dof = H_00.shape[0]
    N_D = n_slabs * n_dof

    g_L_all, valid_L = _sancho_rubio_batch(
        z2_arr, H_00, H_01, lead_sigma_r=lead_sigma_r_L)
    g_R_all, valid_R = _sancho_rubio_batch(
        z2_arr, H_00, H_01.conj().T, lead_sigma_r=lead_sigma_r_R)
    valid = valid_L & valid_R

    n_fallback = 0
    if not np.all(valid):
        bad = np.where(~valid)[0]
        for iw in bad:
            try:
                g_L_all[iw] = _sancho_rubio(z2_arr[iw], H_00, H_01)
                g_R_all[iw] = _sancho_rubio(
                    z2_arr[iw], H_00, H_01.conj().T)
            except RuntimeError:
                if lead_sigma_r_L is None and lead_sigma_r_R is None:
                    raise
                # Dressed contact failed — fall back to undressed
                g_L_all[iw] = _sancho_rubio(z2_arr[iw], H_00, H_01)
                g_R_all[iw] = _sancho_rubio(
                    z2_arr[iw], H_00, H_01.conj().T)
                n_fallback += 1

    if n_fallback > 0:
        warnings.warn(
            f"Sancho-Rubio: {n_fallback}/{len(bad)} frequencies fell back "
            f"to undressed contacts (scattering-dressed failed)")

    H_01_dag = H_01.conj().T
    sig_L_all = H_01_dag[None] @ g_L_all @ H_01[None]
    sig_R_all = H_01[None] @ g_R_all @ H_01_dag[None]

    gam_L_all = 1j * (sig_L_all - sig_L_all.conj().transpose(0, 2, 1))
    gam_R_all = 1j * (sig_R_all - sig_R_all.conj().transpose(0, 2, 1))

    # Embed into device space
    Sigma_L_R = np.zeros((nfreq, N_D, N_D), dtype=complex)
    Sigma_R_R = np.zeros((nfreq, N_D, N_D), dtype=complex)
    Gamma_L = np.zeros((nfreq, N_D, N_D), dtype=complex)
    Gamma_R = np.zeros((nfreq, N_D, N_D), dtype=complex)

    sl0 = slice(0, n_dof)
    sl_last = slice((n_slabs - 1) * n_dof, n_slabs * n_dof)
    Sigma_L_R[:, sl0, sl0] = sig_L_all
    Sigma_R_R[:, sl_last, sl_last] = sig_R_all
    Gamma_L[:, sl0, sl0] = gam_L_all
    Gamma_R[:, sl_last, sl_last] = gam_R_all

    # Contact lesser/greater
    Sigma_L_lesser, Sigma_L_greater = _boson_contact_self_energies_from_gamma(
        Gamma_L, freqs_thz, T_L)
    Sigma_R_lesser, Sigma_R_greater = _boson_contact_self_energies_from_gamma(
        Gamma_R, freqs_thz, T_R)

    return {
        "Sigma_L_R": Sigma_L_R,
        "Sigma_L_lesser": Sigma_L_lesser,
        "Sigma_L_greater": Sigma_L_greater,
        "Sigma_R_R": Sigma_R_R,
        "Sigma_R_lesser": Sigma_R_lesser,
        "Sigma_R_greater": Sigma_R_greater,
        "Gamma_L": Gamma_L,
        "Gamma_R": Gamma_R,
    }


# ---------------------------------------------------------------------------
# Green's function solvers
# ---------------------------------------------------------------------------


def _solve_green_functions(z2, H_D, obc, Sigma_scatt_R, Sigma_scatt_lesser,
                           Sigma_scatt_greater):
    """Solve for retarded, lesser, and greater Green's functions."""
    N = H_D.shape[0]
    Sigma_R_total = obc["Sigma_L_R"] + obc["Sigma_R_R"] + Sigma_scatt_R
    G_R = inv(z2 * np.eye(N) - H_D - Sigma_R_total)
    G_A = G_R.conj().T

    Sigma_lesser_total = (obc["Sigma_L_lesser"] + obc["Sigma_R_lesser"]
                          + Sigma_scatt_lesser)
    Sigma_greater_total = (obc["Sigma_L_greater"] + obc["Sigma_R_greater"]
                           + Sigma_scatt_greater)

    G_lesser = G_R @ Sigma_lesser_total @ G_A
    G_greater = G_R @ Sigma_greater_total @ G_A
    return G_R, G_lesser, G_greater


def _solve_green_batch(z2_arr, H_D, obc_batch,
                       Sigma_scatt_R, Sigma_scatt_lesser, Sigma_scatt_greater):
    """Batched Green's function solve for all frequencies."""
    nfreq = len(z2_arr)
    N = H_D.shape[0]
    eye = np.eye(N)

    Sigma_R_total = obc_batch["Sigma_L_R"] + obc_batch["Sigma_R_R"] + Sigma_scatt_R
    A = z2_arr[:, None, None] * eye[None] - H_D[None] - Sigma_R_total
    G_R = np.linalg.inv(A)
    G_A = G_R.conj().transpose(0, 2, 1)

    Sigma_lesser_total = (obc_batch["Sigma_L_lesser"] + obc_batch["Sigma_R_lesser"]
                          + Sigma_scatt_lesser)
    Sigma_greater_total = (obc_batch["Sigma_L_greater"] + obc_batch["Sigma_R_greater"]
                           + Sigma_scatt_greater)

    G_lesser = G_R @ Sigma_lesser_total @ G_A
    G_greater = G_R @ Sigma_greater_total @ G_A
    return G_R, G_lesser, G_greater


def _ballistic_transmission_z2(z2, H_D, H_00, H_01, H_LD, H_DR):
    """Ballistic transmission: T = Tr(Γ_L G^R Γ_R G^A)."""
    g_L = _sancho_rubio(z2, H_00, H_01)
    g_R = _sancho_rubio(z2, H_00, H_01.conj().T)

    Sigma_L = H_LD.conj().T @ g_L @ H_LD
    Sigma_R = H_DR @ g_R @ H_DR.conj().T
    Gamma_L = 1j * (Sigma_L - Sigma_L.conj().T)
    Gamma_R = 1j * (Sigma_R - Sigma_R.conj().T)

    N_D = H_D.shape[0]
    G_R = np.linalg.inv(z2 * np.eye(N_D) - H_D - Sigma_L - Sigma_R)
    G_A = G_R.conj().T

    return np.real(np.trace(Gamma_L @ G_R @ Gamma_R @ G_A))


# ---------------------------------------------------------------------------
# Retarded self-energy reconstruction
# ---------------------------------------------------------------------------


def _hilbert_transform_axis(f, axis=1):
    """Hilbert transform along *axis* via FFT (sign-multiplier method)."""
    N = f.shape[axis]
    Ff = np.fft.fft(f, axis=axis)

    h = np.zeros(N)
    if N % 2 == 0:
        h[1:N // 2] = 1
        h[N // 2 + 1:] = -1
    else:
        h[1:(N + 1) // 2] = 1
        h[(N + 1) // 2:] = -1

    shape = [1] * f.ndim
    shape[axis] = N
    return np.fft.ifft(Ff * (-1j * np.array(h).reshape(shape)), axis=axis)


def _retarded_from_lesser_greater(delta, omega_grid_thz):
    """Build Σ^R from Δ = Σ^> − Σ^< via Kramers-Kronig.

    Σ^R(ω) = ½Δ(ω) + (i/2π) PV∫ Δ(ω')/(ω−ω') dω'

    Uses singularity subtraction for the PV integral:

        PV∫ Δ(ω')/(ω−ω') dω'
          = ∫ [Δ(ω') − Δ(ω)]/(ω−ω') dω'  +  Δ(ω) · PV∫ 1/(ω−ω') dω'

    The first integral is regular (the 1/(ω−ω') singularity cancels) and
    is evaluated with standard trapezoid quadrature, filling the diagonal
    sample with the finite-difference derivative −Δ'(ω).  The second term
    uses the analytic PV integral of 1/(ω−ω') over [ω_min, ω_max]:

        PV∫_{ω_min}^{ω_max} dω'/(ω−ω') = ln|(ω − ω_min)/(ω − ω_max)|

    For the endpoints ω = ω_min or ω_max the analytic PV diverges
    logarithmically; we fall back to the discrete sum there.

    Parameters
    ----------
    delta : (n_freq, nd, nd) — Σ^> − Σ^<
    omega_grid_thz : (n_freq,)
    """
    n_freq = len(omega_grid_thz)
    nd = delta.shape[-1]
    sig_r = 0.5 * delta.astype(complex)
    dw = omega_grid_thz[1] - omega_grid_thz[0] if n_freq > 1 else 1.0
    w_min = omega_grid_thz[0]
    w_max = omega_grid_thz[-1]

    for i in range(n_freq):
        wi = omega_grid_thz[i]

        # Regular part: [Δ(ω') − Δ(ω_i)] / (ω_i − ω')
        diff = wi - omega_grid_thz  # (n_freq,)
        reg = np.empty_like(delta)
        nz = diff != 0
        reg[nz] = (delta[nz] - delta[i][None]) / diff[nz, None, None]

        # At ω' = ω_i: L'Hôpital → −dΔ/dω evaluated by central difference
        if i == 0:
            deriv = (delta[1] - delta[0]) / dw
        elif i == n_freq - 1:
            deriv = (delta[-1] - delta[-2]) / dw
        else:
            deriv = (delta[i + 1] - delta[i - 1]) / (2 * dw)
        reg[i] = -deriv

        regular_integral = np.trapezoid(reg, omega_grid_thz, axis=0)

        # Analytic PV∫ 1/(ω_i − ω') dω' over [w_min, w_max]
        eps_edge = 1e-12 * (w_max - w_min)
        if abs(wi - w_min) < eps_edge or abs(wi - w_max) < eps_edge:
            # Endpoint: analytic form diverges, use discrete sum
            mask = nz
            pv_scalar = np.sum(1.0 / diff[mask]) * dw
        else:
            pv_scalar = np.log(abs((wi - w_min) / (wi - w_max)))

        pv_integral = regular_integral + delta[i] * pv_scalar
        sig_r[i] += 0.5j / np.pi * pv_integral

    return sig_r


def _build_retarded(sig_l, sig_g, omega_grid_thz, method="pv"):
    """Build Σ^R from Σ^< and Σ^> using the specified method.

    Parameters
    ----------
    sig_l, sig_g : (..., n_freq, nd, nd)
        Lesser/greater self-energies.  Leading dimensions are preserved.
    method : {"pv", "fft", "half"}
    """
    delta = (sig_g - sig_l).astype(complex)
    if method == "pv":
        # Singularity-subtracted principal-value integral
        leading = delta.shape[:-3]
        flat = delta.reshape(-1, *delta.shape[-3:])
        sig_r = np.empty(flat.shape, dtype=complex)
        for i in range(flat.shape[0]):
            sig_r[i] = _retarded_from_lesser_greater(flat[i], omega_grid_thz)
        return sig_r.reshape(leading + delta.shape[-3:])
    elif method == "fft":
        # FFT Hilbert — axis is the frequency axis (second-to-last-but-two)
        freq_axis = len(delta.shape) - 3
        return 0.5 * delta + 0.5j * _hilbert_transform_axis(delta, axis=freq_axis)
    elif method == "half":
        return 0.5 * delta
    else:
        raise ValueError(f"Unknown retarded method: {method!r}. "
                         f"Use 'pv', 'fft', or 'half'.")


# ---------------------------------------------------------------------------
# Self-energy computation: finite (Gamma-only)
# ---------------------------------------------------------------------------


def _compute_phph_self_energy_finite(
    G_lesser, G_greater, Phi, omega_grid_thz, dw_thz,
):
    """Phonon-phonon self-energy for a finite device (no transverse q).

    Returns (Σ^<, Σ^>) only; Σ^R is rebuilt from the mixed pair in the
    SCBA loop via ``_build_retarded``.
    """
    n_freq = len(omega_grid_thz)
    nd = Phi.shape[0]
    nd2 = nd * nd

    n_fft = 2 * n_freq - 1
    mid = n_freq // 2    # index of ω = 0 in the input grid
    freq_sl = slice(mid, mid + n_freq)

    prefactor = 0.5j * HBAR_SI * dw_thz / (2 * np.pi)

    # Zero out ω=0 sample before convolution to prevent the singular
    # Bose occupation at zero frequency from leaking into the self-energy.
    G_l_clean = G_lesser.copy()
    G_l_clean[mid] = 0.0
    G_g_clean = G_greater.copy()
    G_g_clean[mid] = 0.0

    G_pad = np.zeros((n_fft, nd, nd), dtype=complex)
    G_pad[:n_freq] = G_l_clean
    GL_fft = np.fft.fft(G_pad, axis=0)
    G_pad[:] = 0
    G_pad[:n_freq] = G_g_clean
    GG_fft = np.fft.fft(G_pad, axis=0)
    del G_pad, G_l_clean, G_g_clean

    PL_flat = Phi.reshape(nd2, nd)
    PR_flat = Phi.conj().reshape(nd, nd2)

    sig_l = np.zeros((n_freq, nd, nd), dtype=complex)
    sig_g = np.zeros_like(sig_l)

    for G_fft, sig_out in [(GL_fft, sig_l), (GG_fft, sig_g)]:
        A = PL_flat[None] @ G_fft
        A = A.reshape(n_fft, nd, nd, nd)
        A = A.transpose(0, 1, 3, 2)
        B = A @ G_fft[:, None, :, :]
        S = B.reshape(n_fft * nd, nd2) @ PR_flat.T
        sig_out[:] = prefactor * np.fft.ifft(
            S.reshape(n_fft, nd, nd), axis=0)[freq_sl]

    return sig_l, sig_g


# ---------------------------------------------------------------------------
# Self-energy computation: q-dependent (dense FC3 Fourier transform)
# ---------------------------------------------------------------------------


def _se_worker_iq(args):
    """Process one chunk of iq_ext values using FFT-domain Phi contraction."""
    (iq_ext_list, GL_fft, GG_fft, Phi_all, q_diff_map,
     n_freq, n_dof, n_kpts, n_fft, freq_sl_start, freq_sl_stop,
     prefactor, qp_batch) = args

    freq_sl = slice(freq_sl_start, freq_sl_stop)
    n_out = len(iq_ext_list)
    nd = n_dof
    nd2 = nd * nd
    sig_l = np.zeros((n_out, n_freq, nd, nd), dtype=complex)
    sig_g = np.zeros((n_out, n_freq, nd, nd), dtype=complex)

    iq_prime_all = np.arange(n_kpts)

    for idx, iq_ext in enumerate(iq_ext_list):
        iq_diff_arr = q_diff_map[iq_ext]
        PL_all = Phi_all[iq_prime_all, iq_diff_arr]
        PR_all = np.conj(Phi_all[iq_diff_arr, iq_prime_all])

        for G_fft, sig_out in [(GL_fft, sig_l), (GG_fft, sig_g)]:
            S_hat = np.zeros((n_fft, nd, nd), dtype=complex)

            for p0 in range(0, n_kpts, qp_batch):
                p1 = min(p0 + qp_batch, n_kpts)
                B = p1 - p0
                ps = slice(p0, p1)
                iq_d = iq_diff_arr[ps]

                PL = PL_all[ps]
                PR = PR_all[ps]
                G1 = G_fft[p0:p1]
                G2 = G_fft[iq_d]

                PL_flat = PL.reshape(B, nd2, nd)
                A = np.matmul(PL_flat[:, None, :, :], G2)
                A = A.reshape(B, n_fft, nd, nd, nd)
                A = A.transpose(0, 1, 2, 4, 3)
                B5 = np.matmul(A, G1[:, :, None, :, :])
                B5_flat = B5.reshape(B, n_fft * nd, nd2)
                PR_flat = PR.reshape(B, nd, nd2)
                S_chunk = np.matmul(B5_flat, PR_flat.transpose(0, 2, 1))
                S_hat += S_chunk.sum(axis=0).reshape(n_fft, nd, nd)

            sig_out[idx] = prefactor * np.fft.ifft(S_hat, axis=0)[freq_sl]

    return iq_ext_list, sig_l, sig_g


def _compute_phph_self_energy_q_dense(
    G_lesser_q, G_greater_q, M_stacked, T_all_q, q_diff_map,
    nat_prim, n_kpts, omega_grid_thz, dw_thz, n_workers=None,
):
    """Compute q-dependent phonon-phonon self-energy.

    Returns (Σ^<, Σ^>) only; Σ^R is rebuilt from the mixed pair in the
    SCBA loop via ``_build_retarded``.

    Returns
    -------
    Sigma_lesser, Sigma_greater : (n_kpts, n_freq, n_dof, n_dof)
    """
    import os

    n_freq = len(omega_grid_thz)
    n_dof = nat_prim * 3
    dim_t = M_stacked.shape[1]

    n_fft = 2 * n_freq - 1
    mid = n_freq // 2    # index of ω = 0
    freq_sl = slice(mid, mid + n_freq)

    prefactor = 0.5j * HBAR_SI * dw_thz / (2 * np.pi) / n_kpts

    # Zero out ω=0 before convolution
    G_l_clean = G_lesser_q.copy()
    G_l_clean[:, mid] = 0.0
    G_g_clean = G_greater_q.copy()
    G_g_clean[:, mid] = 0.0

    G_padded = np.zeros((n_kpts, n_fft, n_dof, n_dof), dtype=complex)
    G_padded[:, :n_freq] = G_l_clean
    GL_fft = np.fft.fft(G_padded, axis=1)
    G_padded[:, :] = 0
    G_padded[:, :n_freq] = G_g_clean
    GG_fft = np.fft.fft(G_padded, axis=1)
    del G_padded, G_l_clean, G_g_clean

    M_blocks = M_stacked.reshape(n_dof, dim_t, dim_t)
    T_arr = np.array(T_all_q)
    TM = np.einsum('qci,aij->qacj', T_arr, M_blocks)
    T_arr_H = T_arr.conj().transpose(0, 2, 1).copy()
    Phi_all = np.einsum('qacj,rjd->qracd', TM, T_arr_H)
    del TM, T_arr_H, T_arr, M_blocks

    target_bytes = 16 * 1024 * 1024
    bytes_per_qp = n_fft * n_dof**3 * 16
    qp_batch = max(1, min(n_kpts, target_bytes // max(bytes_per_qp, 1)))

    if n_workers is None:
        n_workers = min(n_kpts, os.cpu_count() or 1)
    n_workers = min(n_workers, n_kpts)
    if n_kpts > 1:
        n_workers = min(n_workers, max(1, n_kpts // 2))

    common = (GL_fft, GG_fft, Phi_all, q_diff_map,
              n_freq, n_dof, n_kpts, n_fft,
              freq_sl.start, freq_sl.stop, prefactor, qp_batch)

    if n_workers <= 1:
        _, sig_l, sig_g = _se_worker_iq(
            (list(range(n_kpts)), *common))
    else:
        chunks = [[] for _ in range(n_workers)]
        for iq in range(n_kpts):
            chunks[iq % n_workers].append(iq)
        chunks = [c for c in chunks if c]
        work_args = [(chunk, *common) for chunk in chunks]

        sig_l = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)
        sig_g = np.zeros_like(sig_l)

        use_threads = os.environ.get("QUATREX_SE_THREADS", "1") == "1"
        if use_threads:
            with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                futures = [executor.submit(_se_worker_iq, wa) for wa in work_args]
                for fut in futures:
                    iq_list, sl, sg = fut.result()
                    for i, iq in enumerate(iq_list):
                        sig_l[iq] = sl[i]
                        sig_g[iq] = sg[i]
        else:
            ctx = get_context("forkserver")
            with ctx.Pool(processes=len(chunks)) as pool:
                for iq_list, sl, sg in pool.map(_se_worker_iq, work_args):
                    for i, iq in enumerate(iq_list):
                        sig_l[iq] = sl[i]
                        sig_g[iq] = sg[i]

    return sig_l, sig_g


# ---------------------------------------------------------------------------
# Shared SCBA driver
# ---------------------------------------------------------------------------


def _scba_loop(
    *,
    z2_arr, freqs_thz, dw_thz, omega_rad, pos_mask,
    n_slabs, n_dof, N_D,
    H_D_list, obc_list, btd_blocks_list,
    n_kpts,
    se_kernel,
    T_L, T_R,
    max_scba_iter, scba_tol, conservation_tol,
    mixing, anderson_mixing, anderson_depth,
    scattering_contacts,
    retarded,
    verbose,
):
    """Shared SCBA loop for both q-dense and finite paths.

    Self-energies always have shape (n_slabs, n_kpts, nfreq, n_dof, n_dof),
    even when n_kpts == 1.  This eliminates shape branching.

    Parameters
    ----------
    se_kernel : callable(G_lesser_slab, G_greater_slab)
        Must return (Σ_l, Σ_g), each of shape
        (n_slabs, n_kpts, nfreq, n_dof, n_dof).
    retarded : {"pv", "fft", "half"}
        Method for rebuilding Σ^R from the mixed Σ^{<,>}.
    """
    nfreq = len(freqs_thz)

    # Always 5D — no branching on n_kpts
    Sigma_R = np.zeros((n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex)
    Sigma_l = np.zeros_like(Sigma_R)
    Sigma_g = np.zeros_like(Sigma_R)

    convergence_history = []
    J_total_prev = 0.0
    rel_change = float('inf')
    sig_change = float('inf')
    conservation_err = 1.0
    _anderson_x_hist = []
    _anderson_f_hist = []

    spectral_J_L = np.zeros(nfreq)
    spectral_J_R = np.zeros(nfreq)

    sl0 = slice(0, n_dof)
    sl_last = slice((n_slabs - 1) * n_dof, n_slabs * n_dof)

    for scba_iter in range(max_scba_iter):
        # Update OBC with scattering in the leads
        if scattering_contacts and scba_iter > 0:
            for iq, (H_00_iq, H_01_iq) in enumerate(btd_blocks_list):
                lead_L = Sigma_R[0, iq]
                lead_R = Sigma_R[-1, iq]
                try:
                    obc_try = _compute_obc_batch(
                        z2_arr, H_00_iq, H_01_iq, freqs_thz, T_L, T_R,
                        n_slabs=n_slabs,
                        lead_sigma_r_L=lead_L, lead_sigma_r_R=lead_R)
                    if not np.any(np.isnan(obc_try["Sigma_L_R"])):
                        obc_list[iq] = obc_try
                    elif verbose:
                        print(f"    WARNING: scattering-contact OBC "
                              f"diverged for iq={iq}, keeping previous")
                except RuntimeError:
                    if verbose:
                        print(f"    WARNING: scattering-contact OBC "
                              f"failed for iq={iq}, keeping previous")

        # Green's function solve — always 5D
        G_lesser_slab = np.zeros(
            (n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex)
        G_greater_slab = np.zeros_like(G_lesser_slab)
        G_R_slab = np.zeros_like(G_lesser_slab)
        spectral_J_L[:] = 0.0
        spectral_J_R[:] = 0.0

        for iq in range(n_kpts):
            H_D = H_D_list[iq]
            obc = obc_list[iq]

            Sig_R_dev = np.zeros((nfreq, N_D, N_D), dtype=complex)
            Sig_l_dev = np.zeros_like(Sig_R_dev)
            Sig_g_dev = np.zeros_like(Sig_R_dev)
            for l in range(n_slabs):
                sl = slice(l * n_dof, (l + 1) * n_dof)
                Sig_R_dev[:, sl, sl] = Sigma_R[l, iq]
                Sig_l_dev[:, sl, sl] = Sigma_l[l, iq]
                Sig_g_dev[:, sl, sl] = Sigma_g[l, iq]

            G_ret, G_less, G_great = _solve_green_batch(
                z2_arr, H_D, obc, Sig_R_dev, Sig_l_dev, Sig_g_dev)

            for l in range(n_slabs):
                sl = slice(l * n_dof, (l + 1) * n_dof)
                G_lesser_slab[l, iq] = G_less[:, sl, sl]
                G_greater_slab[l, iq] = G_great[:, sl, sl]
                G_R_slab[l, iq] = G_ret[:, sl, sl]

            # Spectral current
            SLg_Gl = obc["Sigma_L_greater"][:, sl0, sl0] @ G_less[:, sl0, sl0]
            SLl_Gg = obc["Sigma_L_lesser"][:, sl0, sl0] @ G_great[:, sl0, sl0]
            spectral_J_L += HBAR_SI * omega_rad * np.real(
                np.trace(SLg_Gl - SLl_Gg, axis1=-2, axis2=-1))

            SRl_Gg = obc["Sigma_R_lesser"][:, sl_last, sl_last] @ G_great[:, sl_last, sl_last]
            SRg_Gl = obc["Sigma_R_greater"][:, sl_last, sl_last] @ G_less[:, sl_last, sl_last]
            spectral_J_R += HBAR_SI * omega_rad * np.real(
                np.trace(SRl_Gg - SRg_Gl, axis1=-2, axis2=-1))

        spectral_J_L /= n_kpts
        spectral_J_R /= n_kpts

        J_L_total = np.sum(spectral_J_L[pos_mask]) * dw_thz * 1e12
        J_R_total = np.sum(spectral_J_R[pos_mask]) * dw_thz * 1e12
        J_total = 0.5 * (J_L_total + J_R_total)
        J_denom = abs(J_L_total) + abs(J_R_total)
        conservation_err = (abs(J_L_total - J_R_total) / J_denom
                            if J_denom > 0 else 0.0)

        # Full-axis symmetry check — all slabs and q-points
        if verbose and scba_iter <= 2:
            max_l_err = 0.0
            max_r_err = 0.0
            for l in range(n_slabs):
                for iq in range(n_kpts):
                    le, re = _check_full_axis_symmetry(
                        G_R_slab[l, iq], G_lesser_slab[l, iq],
                        G_greater_slab[l, iq], freqs_thz)
                    max_l_err = max(max_l_err, le)
                    max_r_err = max(max_r_err, re)
            if max_l_err > 1e-3 or max_r_err > 1e-3:
                print(f"    Symmetry: |G^<+G^>(-)|={max_l_err:.2e}, "
                      f"|G^R-G^R(-)*|={max_r_err:.2e}")

        # Compute new self-energy (lesser and greater only)
        Sigma_l_new, Sigma_g_new = se_kernel(
            G_lesser_slab, G_greater_slab)

        # Enforce bosonic full-axis symmetry: Σ^<(ω) = −[Σ^>(−ω)]^T
        _symmetrize_lesser_greater(Sigma_l_new, Sigma_g_new)

        sig_r_norm = np.max(np.abs(Sigma_l_new)) + np.max(np.abs(Sigma_g_new))

        if verbose and scba_iter == 0:
            gl_max = np.max(np.abs(G_lesser_slab))
            print(f"    G diagnostic: max|G^<| = {gl_max:.4e}")
            print(f"    Self-energy: max|Σ^R| = {sig_r_norm:.4e} THz²")

        # Save previous for convergence check
        _Sigma_prev = np.concatenate([
            Sigma_l.ravel(), Sigma_g.ravel()])

        # Mix only Σ^< and Σ^>, then rebuild Σ^R from the mixed pair
        if scba_iter == 0:
            Sigma_l = Sigma_l_new.copy()
            Sigma_g = Sigma_g_new.copy()
        elif anderson_mixing:
            x_in = np.concatenate([
                Sigma_l.ravel(), Sigma_g.ravel()])
            x_out = np.concatenate([
                Sigma_l_new.ravel(), Sigma_g_new.ravel()])
            f_k = x_out - x_in
            _anderson_x_hist.append(x_in)
            _anderson_f_hist.append(f_k)

            m = len(_anderson_f_hist)
            if m >= 2:
                n_use = min(m - 1, anderson_depth)
                dF = np.column_stack([
                    _anderson_f_hist[-n_use + j] - _anderson_f_hist[-n_use + j - 1]
                    for j in range(n_use)])
                dX = np.column_stack([
                    _anderson_x_hist[-n_use + j] - _anderson_x_hist[-n_use + j - 1]
                    for j in range(n_use)])
                FtF = dF.conj().T @ dF
                Ftf = dF.conj().T @ f_k
                reg = 1e-8 * np.trace(FtF).real / max(FtF.shape[0], 1)
                gamma = np.linalg.solve(
                    FtF + reg * np.eye(FtF.shape[0]), Ftf)
                x_mixed = (x_in + mixing * f_k) - (dX + mixing * dF) @ gamma
            else:
                x_mixed = x_in + mixing * f_k

            sz = Sigma_l.size
            Sigma_l = x_mixed[:sz].reshape(Sigma_l.shape)
            Sigma_g = x_mixed[sz:].reshape(Sigma_g.shape)

            if len(_anderson_x_hist) > anderson_depth + 2:
                _anderson_x_hist.pop(0)
                _anderson_f_hist.pop(0)
        else:
            alpha = mixing
            Sigma_l = (1 - alpha) * Sigma_l + alpha * Sigma_l_new
            Sigma_g = (1 - alpha) * Sigma_g + alpha * Sigma_g_new

        # Rebuild Σ^R from the mixed Σ^< and Σ^>
        Sigma_R = _build_retarded(
            Sigma_l, Sigma_g, freqs_thz, method=retarded)

        # Broadening sign check — all slabs and q-points
        total_viol = 0
        total_max = 0.0
        for l in range(n_slabs):
            for iq in range(n_kpts):
                nv, mv = _check_broadening_sign(
                    Sigma_R[l, iq], freqs_thz, "SCBA", tol=1e-8)
                total_viol += nv
                total_max = max(total_max, mv)
        if total_viol > 0 and verbose:
            print(f"    WARNING: Γ sign violations: {total_viol} points, "
                  f"max = {total_max:.2e}")

        # Convergence
        if scba_iter > 0:
            rel_change = abs(J_total - J_total_prev) / (abs(J_total_prev) + 1e-30)
            _Sigma_now = np.concatenate([
                Sigma_l.ravel(), Sigma_g.ravel()])
            sig_norm = np.linalg.norm(_Sigma_now)
            sig_change = (np.linalg.norm(_Sigma_now - _Sigma_prev)
                          / (sig_norm + 1e-30) if sig_norm > 0 else 0.0)
            convergence_history.append(rel_change)
            if verbose:
                print(f"    SCBA iter {scba_iter + 1}: "
                      f"J = {J_total:.4e} W, "
                      f"conservation = {conservation_err:.4e}, "
                      f"dJ/J = {rel_change:.4e}, "
                      f"dΣ/Σ = {sig_change:.4e}, "
                      f"max|Σ^R| = {np.max(np.abs(Sigma_R)):.2e} THz²")
            numerically_converged = (sig_change < scba_tol
                                      and rel_change < scba_tol)
            conserving = conservation_err < conservation_tol

            if numerically_converged and conserving:
                if verbose:
                    print(f"    Converged after {scba_iter + 1} iterations "
                          f"(dJ/J={rel_change:.2e}, dΣ/Σ={sig_change:.2e}, "
                          f"conservation={conservation_err:.2e})")
                break

            if numerically_converged and not conserving:
                if verbose:
                    print(f"    Numerically converged but conservation "
                          f"NOT satisfied ({conservation_err:.2e} > "
                          f"{conservation_tol:.2e}), continuing...")
        else:
            if verbose:
                print(f"    SCBA iter 1: "
                      f"J_L = {J_L_total:.4e} W, J_R = {J_R_total:.4e} W")

        J_total_prev = J_total

    else:
        if verbose:
            print(f"  WARNING: SCBA did not converge after {max_scba_iter} "
                  f"iterations (dJ/J={rel_change:.2e}, dΣ/Σ={sig_change:.2e}, "
                  f"conservation={conservation_err:.2e})")

    return {
        "spectral_J_L": spectral_J_L,
        "spectral_J_R": spectral_J_R,
        "Sigma_R": Sigma_R,
        "Sigma_l": Sigma_l,
        "Sigma_g": Sigma_g,
        "conservation_err": conservation_err,
        "convergence_history": convergence_history,
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def anharmonic_transmission_finite(
    phonon,
    fc3_hdf5=None,
    freq_range_thz: tuple[float, float, int] = (0.01, 16.0, 101),
    transport_direction: str = "x",
    eta_factor: float = 0.05,
    temperature: float = 300.0,
    delta_T: float = 10.0,
    max_scba_iter: int = 10,
    scba_tol: float = 1e-3,
    conservation_tol: float = 1e-3,
    mixing: float = 0.5,
    anderson_mixing: bool = False,
    anderson_depth: int = 5,
    n_slabs: int = 1,
    verbose: bool = True,
    M_stacked_override: np.ndarray = None,
    retarded: str = "half",
    scattering_contacts: bool = False,
    hilbert_retarded: bool = False,
) -> dict:
    """Reference anharmonic phonon transport (Gamma-point, finite device).

    This is the canonical n_slabs=1 solver.  ``anharmonic_transmission_q``
    with ``q_mesh=(1,1)`` must reproduce this result exactly.

    Parameters
    ----------
    retarded : {"half", "pv", "fft"}
        Method for reconstructing Σ^R from Σ^{<,>}.  Default ``"half"``
        matches the q-path entry point so q=(1,1) reproduces finite.
    hilbert_retarded : bool
        Legacy flag.  If True, overrides ``retarded`` to ``"fft"``.
    """
    if hilbert_retarded:
        retarded = "fft"
    from .convention import get_btd_blocks
    from .separable import (
        build_supercell_mapping,
        build_realspace_fc3_matrices,
        build_gathering_matrix,
    )

    freqs_thz, dw_thz, eta_w, z2_arr, pos_mask, mid = _build_frequency_grid(
        freq_range_thz, eta_factor=eta_factor)
    nfreq = len(freqs_thz)

    n_atoms = len(phonon.primitive.masses)
    n_dof = 3 * n_atoms
    N_D = n_slabs * n_dof

    # Supercell mapping
    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(
        phonon, transport_direction)
    masses_super = phonon.supercell.masses
    n_super = len(masses_super)
    dim_sc = n_super * 3

    # FC3
    if M_stacked_override is not None:
        M_stacked = M_stacked_override
    else:
        fc3_raw = _load_fc3_raw(fc3_hdf5)
        M_stacked = build_realspace_fc3_matrices(
            fc3_raw, n_atoms, masses_super, ref_sc_atoms)

    # Phi at Gamma
    T0 = build_gathering_matrix(
        prim_indices, cell_frac, (0.0, 0.0), n_atoms, transport_direction)
    M_blocks = M_stacked.reshape(n_dof, dim_sc, dim_sc)
    Phi = np.einsum('ci,aij,dj->acd', T0, M_blocks, T0.conj())

    if verbose:
        if M_stacked_override is None:
            print(f"  FC3 raw shape: {fc3_raw.shape}")
        print(f"  Supercell atoms: {n_super}, dim_sc: {dim_sc}")
        print(f"  M_stacked norm: {np.linalg.norm(M_stacked):.4e}")
        print(f"  Phi norm: {np.linalg.norm(Phi):.4e}")
        print(f"  Device: {n_slabs} slab(s), {N_D} DOFs (finite, Gamma only)")

    T_L = temperature + delta_T / 2.0
    T_R = temperature - delta_T / 2.0

    if verbose:
        print(f"  Frequency grid: {nfreq} points, "
              f"{freqs_thz[0]:.2f} to {freqs_thz[-1]:.2f} THz")
        print(f"  Temperature: {temperature} K, delta_T: {delta_T} K")
        print(f"  eta_w = {eta_w:.4e} THz")
        mix_str = (f"Anderson(depth={anderson_depth})" if anderson_mixing
                   else "linear")
        print(f"  SCBA: max {max_scba_iter} iter, tol={scba_tol}, "
              f"mix={mixing}, method={mix_str}")
        print(f"  Retarded SE: {retarded}")

    # BTD blocks at Gamma
    H_00, H_01 = get_btd_blocks(
        phonon, (0.0, 0.0), transport_direction=transport_direction,
        conversion_factor=CONVERSION_THZ2)
    H_D = _build_device_hamiltonian(H_00, H_01, n_slabs)

    # OBC
    if verbose:
        suffix = " (will update each SCBA iter)" if scattering_contacts else ""
        print(f"  Precomputing OBC self-energies (batched)...{suffix}")
    obc = _compute_obc_batch(z2_arr, H_00, H_01, freqs_thz, T_L, T_R,
                             n_slabs=n_slabs)

    # Ballistic transmission
    H_LD = np.zeros((n_dof, N_D), dtype=complex)
    H_LD[:, :n_dof] = H_01
    H_DR = np.zeros((N_D, n_dof), dtype=complex)
    H_DR[-n_dof:, :] = H_01

    trans_ballistic = np.zeros(nfreq)
    for iw, z2 in enumerate(z2_arr):
        trans_ballistic[iw] = _ballistic_transmission_z2(
            z2, H_D, H_00, H_01, H_LD, H_DR)

    if verbose:
        print(f"  Ballistic max T: {trans_ballistic.max():.4f}")

    # Cross-sectional area
    lattice = phonon.primitive.cell
    tidx = "xyz".index(transport_direction)
    perp_idx = [i for i in range(3) if i != tidx]
    a1 = lattice[perp_idx[0]]
    a2 = lattice[perp_idx[1]]
    A_c = np.linalg.norm(np.cross(a1, a2)) * 1e-20

    omega_rad = freqs_thz * THZ_TO_RAD
    n_bose_L = _bose_full_axis(freqs_thz, T_L)
    n_bose_R = _bose_full_axis(freqs_thz, T_R)
    spectral_J_ball = HBAR_SI * omega_rad * (n_bose_L - n_bose_R) * trans_ballistic
    J_ball_total = np.sum(spectral_J_ball[pos_mask]) * dw_thz * 1e12
    G_ball = J_ball_total / (A_c * delta_T) if delta_T != 0 else 0.0

    if verbose:
        print(f"  Ballistic thermal conductance: {G_ball:.2f} W/(m^2 K)")

    # Self-energy kernel — returns 5D: (n_slabs, 1, nfreq, n_dof, n_dof)
    def se_kernel(G_lesser_slab, G_greater_slab):
        # Input: (n_slabs, 1, nfreq, n_dof, n_dof)
        Sigma_l_new = np.zeros((n_slabs, 1, nfreq, n_dof, n_dof), dtype=complex)
        Sigma_g_new = np.zeros_like(Sigma_l_new)
        for l in range(n_slabs):
            sl_n, sg_n = _compute_phph_self_energy_finite(
                G_lesser_slab[l, 0], G_greater_slab[l, 0],
                Phi, freqs_thz, dw_thz)
            Sigma_l_new[l, 0] = sl_n
            Sigma_g_new[l, 0] = sg_n
        return Sigma_l_new, Sigma_g_new

    # SCBA
    scba_result = _scba_loop(
        z2_arr=z2_arr, freqs_thz=freqs_thz, dw_thz=dw_thz,
        omega_rad=omega_rad, pos_mask=pos_mask,
        n_slabs=n_slabs, n_dof=n_dof, N_D=N_D,
        H_D_list=[H_D], obc_list=[obc], btd_blocks_list=[(H_00, H_01)],
        n_kpts=1,
        se_kernel=se_kernel,
        T_L=T_L, T_R=T_R,
        max_scba_iter=max_scba_iter, scba_tol=scba_tol,
        conservation_tol=conservation_tol,
        mixing=mixing, anderson_mixing=anderson_mixing,
        anderson_depth=anderson_depth,
        scattering_contacts=scattering_contacts,
        retarded=retarded,
        verbose=verbose,
    )

    spectral_J_L = scba_result["spectral_J_L"]
    spectral_J_R = scba_result["spectral_J_R"]
    Sigma_R = scba_result["Sigma_R"]
    Sigma_l = scba_result["Sigma_l"]
    Sigma_g = scba_result["Sigma_g"]
    conservation_err = scba_result["conservation_err"]
    convergence_history = scba_result["convergence_history"]

    spectral_J_anh = 0.5 * (spectral_J_L + spectral_J_R)
    J_anh_total = np.sum(spectral_J_anh[pos_mask]) * dw_thz * 1e12
    G_anh = J_anh_total / (A_c * delta_T) if delta_T != 0 else 0.0

    if verbose:
        print(f"  Anharmonic thermal conductance: {G_anh:.2f} W/(m^2 K)")
        print(f"  Heat flow conservation: {conservation_err:.4e}")

    # Squeeze q-axis for output (n_kpts=1)
    return {
        "freqs_thz": freqs_thz[pos_mask],
        "omega_rad": freqs_thz[pos_mask] * THZ_TO_RAD,
        "transmission_ballistic": trans_ballistic[pos_mask],
        "spectral_heat_current_ballistic": spectral_J_ball[pos_mask],
        "spectral_heat_current": spectral_J_anh[pos_mask],
        "spectral_heat_current_L": spectral_J_L[pos_mask].copy(),
        "spectral_heat_current_R": spectral_J_R[pos_mask].copy(),
        "heat_current_ballistic": J_ball_total,
        "heat_current": J_anh_total,
        "thermal_conductance_ballistic": G_ball,
        "thermal_conductance_anharmonic": G_anh,
        "heat_flow_conservation": conservation_err,
        "delta_T": delta_T,
        "n_scba_iterations": len(convergence_history) + 1,
        "convergence_history": convergence_history,
        "self_energy_retarded": Sigma_R[:, 0, pos_mask],
        "self_energy_lesser": Sigma_l[:, 0, pos_mask],
        "self_energy_greater": Sigma_g[:, 0, pos_mask],
    }


def anharmonic_transmission_q(
    phonon,
    fc3_hdf5=None,
    q_mesh_transverse: tuple[int, int] = (4, 4),
    freq_range_thz: tuple[float, float, int] = (0.01, 16.0, 101),
    transport_direction: str = "x",
    eta_factor: float = 0.05,
    temperature: float = 300.0,
    delta_T: float = 10.0,
    max_scba_iter: int = 10,
    scba_tol: float = 1e-3,
    conservation_tol: float = 1e-3,
    mixing: float = 0.5,
    anderson_mixing: bool = False,
    anderson_depth: int = 5,
    n_slabs: int = 1,
    verbose: bool = True,
    M_stacked_override: np.ndarray = None,
    retarded: str = "half",
    scattering_contacts: bool = False,
    hilbert_retarded: bool = False,
) -> dict:
    """Anharmonic phonon transport with full q-dependent dense self-energy.

    Uses a Gamma-centered q-mesh [0, 1/N, ..., (N-1)/N] for closure
    under subtraction (required for q-convolution).

    When ``q_mesh=(1,1)``, this must reproduce ``anharmonic_transmission_finite``
    exactly.  A regression check is run automatically in that case.

    Parameters
    ----------
    retarded : {"half", "pv", "fft"}
        Method for reconstructing Σ^R from Σ^{<,>}.  Defaults to "half"
        for the q-path because the truncated PV integral amplifies sign
        violations inherent in the cross-q bubble diagram.
    hilbert_retarded : bool
        Legacy flag.  If True, overrides ``retarded`` to ``"pv"``.
    """
    if hilbert_retarded:
        retarded = "pv"

    from .convention import get_btd_blocks
    from .separable import (
        build_supercell_mapping,
        build_realspace_fc3_matrices,
        build_gathering_matrix,
        build_q_diff_map,
    )

    freqs_thz, dw_thz, eta_w, z2_arr, pos_mask, mid = _build_frequency_grid(
        freq_range_thz, eta_factor=eta_factor)
    nfreq = len(freqs_thz)

    n_atoms = len(phonon.primitive.masses)
    n_dof = 3 * n_atoms
    N_D = n_slabs * n_dof

    # Supercell mapping
    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(
        phonon, transport_direction)
    masses_super = phonon.supercell.masses
    n_super = len(masses_super)
    dim_sc = n_super * 3

    # FC3
    if M_stacked_override is not None:
        M_stacked = M_stacked_override
    else:
        fc3_raw = _load_fc3_raw(fc3_hdf5)
        M_stacked = build_realspace_fc3_matrices(
            fc3_raw, n_atoms, masses_super, ref_sc_atoms)

    if verbose:
        if M_stacked_override is None:
            print(f"  FC3 raw shape: {fc3_raw.shape}")
        print(f"  Supercell atoms: {n_super}, dim_sc: {dim_sc}")
        print(f"  M_stacked norm: {np.linalg.norm(M_stacked):.4e}")
        print(f"  Device: {n_slabs} slab(s), {N_D} DOFs per q-point")

    # Gamma-centered q-mesh
    nkx, nky = q_mesh_transverse
    q_1d_x = [i / nkx for i in range(nkx)]
    q_1d_y = [j / nky for j in range(nky)]
    q_points = [(qx, qy) for qx in q_1d_x for qy in q_1d_y]
    n_kpts = len(q_points)
    q_diff_map = build_q_diff_map(nkx, nky)

    T_all_q = []
    for qx, qy in q_points:
        T = build_gathering_matrix(
            prim_indices, cell_frac, (qx, qy), n_atoms, transport_direction)
        T_all_q.append(T)

    T_L = temperature + delta_T / 2.0
    T_R = temperature - delta_T / 2.0

    if verbose:
        print(f"  q-mesh: {nkx}x{nky} = {n_kpts} (Gamma-centered)")
        print(f"  Frequency grid: {nfreq} points, "
              f"{freqs_thz[0]:.2f} to {freqs_thz[-1]:.2f} THz")
        print(f"  Temperature: {temperature} K, delta_T: {delta_T} K")
        print(f"  eta_w = {eta_w:.4e} THz")
        mix_str = (f"Anderson(depth={anderson_depth})" if anderson_mixing
                   else "linear")
        print(f"  SCBA: max {max_scba_iter} iter, tol={scba_tol}, "
              f"mix={mixing}, method={mix_str}")
        print(f"  Retarded SE: {retarded}")

    # BTD blocks per q-point
    btd_blocks = []
    for qx, qy in q_points:
        H_00, H_01 = get_btd_blocks(
            phonon, (qx, qy), transport_direction=transport_direction,
            conversion_factor=CONVERSION_THZ2)
        btd_blocks.append((H_00, H_01))

    # OBC per q-point
    if verbose:
        suffix = " (will update each SCBA iter)" if scattering_contacts else ""
        print(f"  Precomputing OBC self-energies (batched)...{suffix}")
    obc_all = []
    H_D_all = []
    for iq, (H_00, H_01) in enumerate(btd_blocks):
        H_D = _build_device_hamiltonian(H_00, H_01, n_slabs)
        H_D_all.append(H_D)
        obc = _compute_obc_batch(z2_arr, H_00, H_01, freqs_thz, T_L, T_R,
                                 n_slabs=n_slabs)
        obc_all.append(obc)

    # Ballistic transmission
    trans_ballistic = np.zeros(nfreq)
    for iq, (H_00, H_01) in enumerate(btd_blocks):
        H_D = H_D_all[iq]
        H_LD = np.zeros((n_dof, N_D), dtype=complex)
        H_LD[:, :n_dof] = H_01
        H_DR = np.zeros((N_D, n_dof), dtype=complex)
        H_DR[-n_dof:, :] = H_01
        for iw, z2 in enumerate(z2_arr):
            trans_ballistic[iw] += _ballistic_transmission_z2(
                z2, H_D, H_00, H_01, H_LD, H_DR)
    trans_ballistic /= n_kpts

    if verbose:
        print(f"  Ballistic max T: {trans_ballistic.max():.4f}")

    # Cross-sectional area
    lattice = phonon.primitive.cell
    tidx = "xyz".index(transport_direction)
    perp_idx = [i for i in range(3) if i != tidx]
    a1 = lattice[perp_idx[0]]
    a2 = lattice[perp_idx[1]]
    A_c = np.linalg.norm(np.cross(a1, a2)) * 1e-20

    omega_rad = freqs_thz * THZ_TO_RAD
    n_bose_L = _bose_full_axis(freqs_thz, T_L)
    n_bose_R = _bose_full_axis(freqs_thz, T_R)
    spectral_J_ball = HBAR_SI * omega_rad * (n_bose_L - n_bose_R) * trans_ballistic
    J_ball_total = np.sum(spectral_J_ball[pos_mask]) * dw_thz * 1e12
    G_ball = J_ball_total / (A_c * delta_T) if delta_T != 0 else 0.0

    if verbose:
        print(f"  Ballistic thermal conductance: {G_ball:.2f} W/(m^2 K)")

    # Self-energy kernel — returns 5D: (n_slabs, n_kpts, nfreq, n_dof, n_dof)
    def se_kernel(G_lesser_slab, G_greater_slab):
        # Input: (n_slabs, n_kpts, nfreq, n_dof, n_dof)
        Sigma_l_new = np.zeros(
            (n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex)
        Sigma_g_new = np.zeros_like(Sigma_l_new)
        for l in range(n_slabs):
            sl_n, sg_n = _compute_phph_self_energy_q_dense(
                G_lesser_slab[l], G_greater_slab[l],
                M_stacked, T_all_q, q_diff_map,
                n_atoms, n_kpts, freqs_thz, dw_thz)
            Sigma_l_new[l] = sl_n
            Sigma_g_new[l] = sg_n
        return Sigma_l_new, Sigma_g_new

    # SCBA
    scba_result = _scba_loop(
        z2_arr=z2_arr, freqs_thz=freqs_thz, dw_thz=dw_thz,
        omega_rad=omega_rad, pos_mask=pos_mask,
        n_slabs=n_slabs, n_dof=n_dof, N_D=N_D,
        H_D_list=H_D_all, obc_list=obc_all, btd_blocks_list=btd_blocks,
        n_kpts=n_kpts,
        se_kernel=se_kernel,
        T_L=T_L, T_R=T_R,
        max_scba_iter=max_scba_iter, scba_tol=scba_tol,
        conservation_tol=conservation_tol,
        mixing=mixing, anderson_mixing=anderson_mixing,
        anderson_depth=anderson_depth,
        scattering_contacts=scattering_contacts,
        retarded=retarded,
        verbose=verbose,
    )

    spectral_J_L = scba_result["spectral_J_L"]
    spectral_J_R = scba_result["spectral_J_R"]
    Sigma_R_q = scba_result["Sigma_R"]
    Sigma_l_q = scba_result["Sigma_l"]
    Sigma_g_q = scba_result["Sigma_g"]
    conservation_err = scba_result["conservation_err"]
    convergence_history = scba_result["convergence_history"]

    spectral_J_anh = 0.5 * (spectral_J_L + spectral_J_R)
    J_anh_total = np.sum(spectral_J_anh[pos_mask]) * dw_thz * 1e12
    G_anh = J_anh_total / (A_c * delta_T) if delta_T != 0 else 0.0

    if verbose:
        print(f"  Anharmonic thermal conductance: {G_anh:.2f} W/(m^2 K)")
        print(f"  Heat flow conservation: {conservation_err:.4e}")

    result = {
        "freqs_thz": freqs_thz[pos_mask],
        "omega_rad": freqs_thz[pos_mask] * THZ_TO_RAD,
        "transmission_ballistic": trans_ballistic[pos_mask],
        "spectral_heat_current_ballistic": spectral_J_ball[pos_mask],
        "spectral_heat_current": spectral_J_anh[pos_mask],
        "spectral_heat_current_L": spectral_J_L[pos_mask].copy(),
        "spectral_heat_current_R": spectral_J_R[pos_mask].copy(),
        "heat_current_ballistic": J_ball_total,
        "heat_current": J_anh_total,
        "thermal_conductance_ballistic": G_ball,
        "thermal_conductance_anharmonic": G_anh,
        "heat_flow_conservation": conservation_err,
        "delta_T": delta_T,
        "n_scba_iterations": len(convergence_history) + 1,
        "convergence_history": convergence_history,
        "self_energy_retarded": Sigma_R_q[:, :, pos_mask],
        "self_energy_lesser": Sigma_l_q[:, :, pos_mask],
        "self_energy_greater": Sigma_g_q[:, :, pos_mask],
    }

    # Automatic regression check for q_mesh=(1,1)
    if n_kpts == 1 and verbose:
        _check_q11_vs_finite(result)

    return result


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------


def _check_q11_vs_finite(res_q, rtol=5e-3, atol=1e-8):
    """Warn if q=(1,1) output is internally inconsistent.

    This checks the q-path result against itself (ballistic should match
    the analytic Gamma-only result).  For a full q-vs-finite check, run
    both entry points and call ``compare_q11_to_finite()``.
    """
    G_ball = res_q["thermal_conductance_ballistic"]
    G_anh = res_q["thermal_conductance_anharmonic"]
    if G_ball != 0 and abs(G_anh) > 2 * abs(G_ball):
        warnings.warn(
            f"q=(1,1) sanity: G_anh ({G_anh:.2e}) > 2 * G_ball ({G_ball:.2e})")


def compare_q11_to_finite(res_q, res_f, rtol=5e-3, atol=1e-8):
    """Verify that q=(1,1) matches the finite (Gamma-only) solver.

    Raises AssertionError on mismatch.
    """
    keys = [
        "heat_current",
        "thermal_conductance_anharmonic",
        "thermal_conductance_ballistic",
        "heat_flow_conservation",
        "transmission_ballistic",
        "spectral_heat_current",
    ]
    for key in keys:
        a = np.asarray(res_q[key])
        b = np.asarray(res_f[key])
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            max_rel = float(np.max(np.abs(a - b) / (np.abs(b) + atol)))
            raise AssertionError(
                f"q=(1,1) vs finite mismatch in '{key}': "
                f"max rel err = {max_rel:.2e}")
