"""Reference anharmonic phonon-phonon NEGF transport.

Implements the self-consistent Born approximation (SCBA) for
phonon-phonon scattering following Guo et al., Phys. Rev. B 102,
195412 (2020).

The scattering self-energy (Guo Eq. 8, diagonal block):

    Sigma^{<}(omega) = (i*hbar/2) * sum_{c,d,e,f} Phi3_{a,c,d}
        * integral dw'/(2*pi) G^<_{cf}(w') G^<_{de}(w-w')
        * Phi3_{b,e,f}

Internal units: THz^2 for dynamical matrices and self-energies.
"""

import warnings
import numpy as np
from numpy.linalg import inv
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from .constants import (
    CONVERSION_FC3_THZ,
    CONVERSION_THZ2,
    HBAR_SI,
    KB_SI,
    THZ_TO_RAD,
)


# ---------------------------------------------------------------------------
# FC3 loading helpers
# ---------------------------------------------------------------------------


def _load_fc3_raw(fc3_hdf5):
    """Load raw FC3 array from an HDF5 path or extract from fc3_data dict.

    Parameters
    ----------
    fc3_hdf5 : str, Path, or dict
        Either a path to an HDF5 file containing an "fc3" dataset,
        or a dict returned by ``load_primitive_cell`` / ``load_fc3_phono3py``
        that contains the raw FC3 under key "fc3" or the phono3py object
        under key "ph3".
    """
    import h5py

    if isinstance(fc3_hdf5, np.ndarray):
        # Raw FC3 array passed directly
        return fc3_hdf5
    if isinstance(fc3_hdf5, dict):
        # Dict from load_fc3_dfpt_hdf5 (has "fc3" key with raw array)
        if "fc3" in fc3_hdf5:
            return np.asarray(fc3_hdf5["fc3"])
        if "ph3" in fc3_hdf5:
            return np.asarray(fc3_hdf5["ph3"].fc3)
        raise ValueError(
            "fc3_data dict has neither 'fc3' nor 'ph3' key. "
            "Pass the fc3.hdf5 path or the raw fc3 array instead. "
            f"Available keys: {list(fc3_hdf5.keys())}")
    else:
        with h5py.File(str(fc3_hdf5), "r") as f:
            return np.array(f["fc3"])


# Compat alias so separable.py can import the old name without changes.
_compute_obc_self_energies = None  # replaced by _compute_obc_batch


# ---------------------------------------------------------------------------
# Shared helpers: frequency grid, Bose function, zero-sample repair
# ---------------------------------------------------------------------------


def _build_frequency_grid(freq_range_thz, eta_w_thz=None, eta_factor=0.05):
    """Build a symmetric frequency grid for FFT convolution.

    Returns
    -------
    freqs_thz : (nfreq,)
    dw_thz : float
    eta_w_thz : float
    z2_arr : (nfreq,) complex
    pos_mask : (nfreq,) bool  — True for ω > 0 (excludes exact zero)
    mid : int  — index of the ω=0 sample
    """
    _fmin, fmax, nfreq_pos = freq_range_thz
    nfreq_pos = int(nfreq_pos)
    if nfreq_pos < 2:
        raise ValueError("nfreq_pos must be >= 2")

    freqs_pos = np.linspace(0.0, fmax, nfreq_pos)
    dw_thz = freqs_pos[1] - freqs_pos[0]

    freqs_thz = np.concatenate((-freqs_pos[:0:-1], freqs_pos))
    mid = len(freqs_thz) // 2

    if eta_w_thz is None:
        eta_w_thz = eta_factor * dw_thz

    z2_arr = (freqs_thz + 1j * eta_w_thz) ** 2

    # Exclude exact zero from physical output/integration.
    pos_mask = freqs_thz > 0.0

    return freqs_thz, dw_thz, eta_w_thz, z2_arr, pos_mask, mid


def _bose_full_axis(freqs_thz, T):
    """Bose-Einstein on the full symmetric axis with proper ω=0 handling.

    At ω=0 the occupation diverges; we use n_B(0) = -1/2 which is the
    finite part of the Laurent expansion and preserves the full-axis
    symmetry n_B(-ω) = -(1 + n_B(ω)).
    """
    x = HBAR_SI * freqs_thz * THZ_TO_RAD / (KB_SI * T)
    n = np.empty_like(x, dtype=float)

    zero = np.isclose(freqs_thz, 0.0)
    small = (~zero) & (np.abs(x) < 1e-6)
    regular = (~zero) & (~small)

    n[regular] = 1.0 / np.expm1(x[regular])
    n[small] = 1.0 / x[small] - 0.5 + x[small] / 12.0
    n[zero] = -0.5

    return n


def _repair_zero_frequency_sample(X):
    """Symmetric interpolation of the single ω=0 sample on a mirrored grid.

    X shape: (n_freq, ...) with zero sample at mid = n_freq // 2.
    """
    mid = X.shape[0] // 2
    if 0 < mid < X.shape[0] - 1:
        X[mid] = 0.5 * (X[mid - 1] + X[mid + 1])
    return X


def _boson_contact_self_energies_from_gamma(Gamma, freqs_thz, T):
    """Build contact Σ^<, Σ^> from Gamma and repair the zero sample."""
    n = _bose_full_axis(freqs_thz, T)

    Sigma_l = -1j * n[:, None, None] * Gamma
    Sigma_g = -1j * (n[:, None, None] + 1.0) * Gamma

    mid = len(freqs_thz) // 2
    if 0 < mid < len(freqs_thz) - 1:
        Sigma_l[mid] = 0.5 * (Sigma_l[mid - 1] + Sigma_l[mid + 1])
        Sigma_g[mid] = 0.5 * (Sigma_g[mid - 1] + Sigma_g[mid + 1])

    return Sigma_l, Sigma_g


# ---------------------------------------------------------------------------
# Causality / dissipation checks
# ---------------------------------------------------------------------------


def _check_antihermitian_sign(Sigma_R, freqs_thz, name, tol=1e-8):
    """Check that i(Σ^R - Σ^{R†}) is negative semidefinite for ω > 0.

    Parameters
    ----------
    Sigma_R : (nfreq, nd, nd)
    freqs_thz : (nfreq,)
    """
    # Only check positive frequencies away from zero
    for iw in range(len(freqs_thz)):
        if abs(freqs_thz[iw]) < 0.5:
            continue
        sr = Sigma_R[iw]
        anti_herm = 1j * (sr - sr.conj().T)
        anti_herm = 0.5 * (anti_herm + anti_herm.conj().T)
        worst = np.linalg.eigvalsh(anti_herm).max()
        if freqs_thz[iw] > 0 and worst > tol:
            warnings.warn(
                f"{name}: i(Σ^R-Σ^A) not NSD at ω={freqs_thz[iw]:.2f} THz, "
                f"max eig={worst:.3e}")
            return
        if freqs_thz[iw] < 0 and np.linalg.eigvalsh(anti_herm).min() < -tol:
            warnings.warn(
                f"{name}: i(Σ^R-Σ^A) wrong sign at ω={freqs_thz[iw]:.2f} THz")
            return


def _assert_full_axis_symmetry(G_R, G_l, G_g, freqs_thz,
                               rtol=1e-4, atol=1e-8):
    """Verify Guo full-axis symmetry relations.

    G^<(ω) = -[G^>(-ω)]^T
    G^R(ω) = [G^R(-ω)]*
    """
    mid = len(freqs_thz) // 2
    pos = slice(mid + 1, None)
    neg = slice(0, mid)

    G_l_pos = G_l[pos]
    G_g_neg = G_g[neg][::-1].transpose(0, 2, 1)
    if not np.allclose(G_l_pos, -G_g_neg, rtol=rtol, atol=atol):
        max_err = np.max(np.abs(G_l_pos + G_g_neg))
        warnings.warn(f"G^<(ω) != -[G^>(-ω)]^T, max err = {max_err:.2e}")

    G_R_pos = G_R[pos]
    G_R_neg = G_R[neg][::-1]
    if not np.allclose(G_R_pos, G_R_neg.conj(), rtol=rtol, atol=atol):
        max_err = np.max(np.abs(G_R_pos - G_R_neg.conj()))
        warnings.warn(f"G^R(ω) != [G^R(-ω)]*, max err = {max_err:.2e}")


# ---------------------------------------------------------------------------
# Green's function and self-energy computation
# ---------------------------------------------------------------------------


def _sancho_rubio(z2, H_00, H_01, max_iter=300, tol=1e-8):
    """Surface Green's function via Sancho-Rubio decimation.

    Parameters
    ----------
    z2 : complex
        Causal frequency squared: z² = (ω + iη_w)².

    Raises
    ------
    RuntimeError
        If the surface Dyson equation residual exceeds 1e-4.
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

    Returns
    -------
    g_surf : ndarray, shape (nfreq, N, N)
    valid : ndarray, shape (nfreq,), bool
        True if the residual is acceptable for that frequency.
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

    # Residual check per frequency
    residual = a_ii - (H_10[None] @ g_surf @ H_01[None]) - np.linalg.inv(g_surf)
    res_per_freq = np.linalg.norm(residual.reshape(nfreq, -1), axis=1)
    a_per_freq = np.linalg.norm(a_ii.reshape(nfreq, -1), axis=1)
    rel_res = res_per_freq / np.maximum(a_per_freq, 1e-30)
    valid = rel_res <= 1e-4

    return g_surf, valid


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

    Uses _bose_full_axis + zero-sample repair for contact Σ^{<,>}.
    Falls back to scalar Sancho-Rubio for any frequency where the
    batch residual is bad.
    """
    nfreq = len(z2_arr)
    n_dof = H_00.shape[0]
    N_D = n_slabs * n_dof

    g_L_all, valid_L = _sancho_rubio_batch(
        z2_arr, H_00, H_01, lead_sigma_r=lead_sigma_r_L)
    g_R_all, valid_R = _sancho_rubio_batch(
        z2_arr, H_00, H_01.conj().T, lead_sigma_r=lead_sigma_r_R)
    valid = valid_L & valid_R

    # Frequency-local fallback for bad residuals
    if not np.all(valid):
        bad = np.where(~valid)[0]
        for iw in bad:
            try:
                g_L_all[iw] = _sancho_rubio(z2_arr[iw], H_00, H_01)
                g_R_all[iw] = _sancho_rubio(
                    z2_arr[iw], H_00, H_01.conj().T)
                valid[iw] = True
            except RuntimeError:
                if lead_sigma_r_L is None and lead_sigma_r_R is None:
                    raise
                # Scattering-dressed contact failed; use undressed fallback
                g_L_all[iw] = _sancho_rubio(z2_arr[iw], H_00, H_01)
                g_R_all[iw] = _sancho_rubio(
                    z2_arr[iw], H_00, H_01.conj().T)
                valid[iw] = True

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

    # Contact lesser/greater with zero-sample repair
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
    """Ballistic transmission: T = Tr(Γ_L G^R Γ_R G^A).

    Uses the same causal z² as the SCBA solver.
    """
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
    """Hilbert transform along *axis* via FFT (sign-multiplier method).

    Kept as a fast option for large q-meshes. For n_slabs=1, prefer
    _retarded_from_lesser_greater() which uses a direct principal-value
    integral and avoids wraparound artifacts.
    """
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
    """Build Σ^R from Δ = Σ^> - Σ^< via discrete principal-value integral.

    Uses Guo Eq. (14) on the working frequency grid. Avoids the wraparound
    artifacts of the FFT Hilbert transform.

    Parameters
    ----------
    delta : (n_freq, nd, nd)
    omega_grid_thz : (n_freq,)
    """
    n_freq = len(omega_grid_thz)
    sig_r = 0.5 * delta.copy()

    for i in range(n_freq):
        diff = omega_grid_thz[i] - omega_grid_thz
        mask = np.ones(n_freq, dtype=bool)
        mask[i] = False
        pv = np.trapz(
            delta[mask] / diff[mask, None, None],
            omega_grid_thz[mask],
            axis=0,
        )
        sig_r[i] += 0.5j / np.pi * pv

    return sig_r


# ---------------------------------------------------------------------------
# Full q-dependent self-energy (dense FC3 Fourier transform)
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
    hilbert_retarded=True,
):
    """Compute q-dependent phonon-phonon self-energy.

    Returns
    -------
    Sigma_lesser, Sigma_greater, Sigma_retarded : (n_kpts, n_freq, n_dof, n_dof)
    """
    import os

    n_freq = len(omega_grid_thz)
    n_dof = nat_prim * 3
    dim_t = M_stacked.shape[1]

    n_fft = 2 * n_freq - 1
    mid = (n_freq - 1) // 2
    freq_sl = slice(mid, mid + n_freq)

    prefactor = 0.5j * HBAR_SI * dw_thz / (2 * np.pi) / n_kpts

    G_padded = np.zeros((n_kpts, n_fft, n_dof, n_dof), dtype=complex)
    G_padded[:, :n_freq] = G_lesser_q
    GL_fft = np.fft.fft(G_padded, axis=1)
    G_padded[:, :] = 0
    G_padded[:, :n_freq] = G_greater_q
    GG_fft = np.fft.fft(G_padded, axis=1)
    del G_padded

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

    # Repair zero sample before building retarded
    for iq in range(n_kpts):
        _repair_zero_frequency_sample(sig_l[iq])
        _repair_zero_frequency_sample(sig_g[iq])

    delta = sig_g - sig_l
    if hilbert_retarded:
        if hilbert_retarded == "pv":
            # Direct principal-value (expensive but accurate)
            sig_r = np.zeros_like(delta)
            for iq in range(n_kpts):
                sig_r[iq] = _retarded_from_lesser_greater(
                    delta[iq], omega_grid_thz)
        else:
            sig_r = 0.5 * delta + 0.5j * _hilbert_transform_axis(delta, axis=1)
    else:
        sig_r = 0.5 * delta

    for iq in range(n_kpts):
        _repair_zero_frequency_sample(sig_r[iq])

    return sig_l, sig_g, sig_r


def _compute_phph_self_energy_finite(
    G_lesser, G_greater, Phi, omega_grid_thz, dw_thz,
    hilbert_retarded=True,
):
    """Phonon-phonon self-energy for a finite device (no transverse q).

    For n_slabs=1, uses the direct principal-value integral by default
    instead of the FFT Hilbert transform to avoid wraparound artifacts.
    """
    n_freq = len(omega_grid_thz)
    nd = Phi.shape[0]
    nd2 = nd * nd

    n_fft = 2 * n_freq - 1
    mid = (n_freq - 1) // 2
    freq_sl = slice(mid, mid + n_freq)

    prefactor = 0.5j * HBAR_SI * dw_thz / (2 * np.pi)

    G_pad = np.zeros((n_fft, nd, nd), dtype=complex)
    G_pad[:n_freq] = G_lesser
    GL_fft = np.fft.fft(G_pad, axis=0)
    G_pad[:] = 0
    G_pad[:n_freq] = G_greater
    GG_fft = np.fft.fft(G_pad, axis=0)
    del G_pad

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

    # Repair zero sample
    _repair_zero_frequency_sample(sig_l)
    _repair_zero_frequency_sample(sig_g)

    delta = sig_g - sig_l
    if hilbert_retarded:
        # Default: use direct PV integral (no wraparound artifacts)
        sig_r = _retarded_from_lesser_greater(delta, omega_grid_thz)
    else:
        sig_r = 0.5 * delta

    _repair_zero_frequency_sample(sig_r)
    return sig_l, sig_g, sig_r


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
    verbose,
):
    """Shared SCBA loop for both q-dense and finite paths.

    Parameters
    ----------
    H_D_list : list of (N_D, N_D) arrays, length n_kpts
    obc_list : list of dicts, length n_kpts
    btd_blocks_list : list of (H_00, H_01) tuples, length n_kpts
    se_kernel : callable(G_lesser_slab, G_greater_slab) -> (Σ_l, Σ_g, Σ_R)
        Returns arrays of shape (n_slabs, nkpts_or_1, nfreq, ndof, ndof)
        for the q-path, or (n_slabs, nfreq, ndof, ndof) for the finite path.

    Returns
    -------
    result : dict
    """
    nfreq = len(freqs_thz)

    # Self-energy shape depends on path
    # For q-dense: (n_slabs, n_kpts, nfreq, n_dof, n_dof)
    # For finite:  (n_slabs, nfreq, n_dof, n_dof)
    # Detect from the first se_kernel call; start with zeros matching q-path
    if n_kpts > 1:
        Sigma_R = np.zeros((n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex)
    else:
        Sigma_R = np.zeros((n_slabs, nfreq, n_dof, n_dof), dtype=complex)
    Sigma_l = np.zeros_like(Sigma_R)
    Sigma_g = np.zeros_like(Sigma_R)

    convergence_history = []
    J_total_prev = 0.0
    rel_change = float('inf')
    sig_change = float('inf')
    conservation_err = 1.0
    best_conservation = 1.0
    best_state = None

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
                if n_kpts > 1:
                    lead_L = Sigma_R[0, iq]
                    lead_R = Sigma_R[-1, iq]
                else:
                    lead_L = Sigma_R[0]
                    lead_R = Sigma_R[-1]
                try:
                    obc_try = _compute_obc_batch(
                        z2_arr, H_00_iq, H_01_iq, freqs_thz, T_L, T_R,
                        n_slabs=n_slabs,
                        lead_sigma_r_L=lead_L, lead_sigma_r_R=lead_R)
                    if not np.any(np.isnan(obc_try["Sigma_L_R"])):
                        obc_list[iq] = obc_try
                    elif verbose:
                        print(f"    WARNING: scattering-contact OBC "
                              f"diverged for iq={iq}, keeping ballistic")
                except RuntimeError:
                    if verbose:
                        print(f"    WARNING: scattering-contact OBC "
                              f"failed for iq={iq}, keeping ballistic")

        # Green's function solve
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
                if n_kpts > 1:
                    Sig_R_dev[:, sl, sl] = Sigma_R[l, iq]
                    Sig_l_dev[:, sl, sl] = Sigma_l[l, iq]
                    Sig_g_dev[:, sl, sl] = Sigma_g[l, iq]
                else:
                    Sig_R_dev[:, sl, sl] = Sigma_R[l]
                    Sig_l_dev[:, sl, sl] = Sigma_l[l]
                    Sig_g_dev[:, sl, sl] = Sigma_g[l]

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

        # Track best-conservation state (full iterate)
        if scba_iter >= 3 and conservation_err < best_conservation:
            best_conservation = conservation_err
            best_state = {
                "spectral_J_L": spectral_J_L.copy(),
                "spectral_J_R": spectral_J_R.copy(),
                "Sigma_R": Sigma_R.copy(),
                "Sigma_l": Sigma_l.copy(),
                "Sigma_g": Sigma_g.copy(),
                "conservation_err": conservation_err,
                "iter": scba_iter + 1,
            }

        # Full-axis symmetry check on first iteration
        if scba_iter == 0:
            _assert_full_axis_symmetry(
                G_R_slab[0, 0], G_lesser_slab[0, 0], G_greater_slab[0, 0],
                freqs_thz, rtol=1e-3, atol=1e-8)

        # Compute new self-energy via the path-specific kernel
        Sigma_l_new, Sigma_g_new, Sigma_r_new = se_kernel(
            G_lesser_slab, G_greater_slab)

        sig_r_norm = np.max(np.abs(Sigma_r_new))

        if verbose and scba_iter == 0:
            gl_max = np.max(np.abs(G_lesser_slab))
            print(f"    G diagnostic: max|G^<| = {gl_max:.4e}")
            print(f"    Self-energy: max|Σ^R| = {sig_r_norm:.4e} THz²")

        # Save previous for convergence check
        _Sigma_prev = np.concatenate([
            Sigma_l.ravel(), Sigma_g.ravel(), Sigma_R.ravel()])

        # Mix self-energies
        if scba_iter == 0:
            Sigma_l = Sigma_l_new.copy()
            Sigma_g = Sigma_g_new.copy()
            Sigma_R = Sigma_r_new.copy()
        elif anderson_mixing:
            x_in = np.concatenate([
                Sigma_l.ravel(), Sigma_g.ravel(), Sigma_R.ravel()])
            x_out = np.concatenate([
                Sigma_l_new.ravel(), Sigma_g_new.ravel(), Sigma_r_new.ravel()])
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
            Sigma_g = x_mixed[sz:2*sz].reshape(Sigma_g.shape)
            Sigma_R = x_mixed[2*sz:].reshape(Sigma_R.shape)

            if len(_anderson_x_hist) > anderson_depth + 2:
                _anderson_x_hist.pop(0)
                _anderson_f_hist.pop(0)
        else:
            alpha = mixing
            Sigma_l = (1 - alpha) * Sigma_l + alpha * Sigma_l_new
            Sigma_g = (1 - alpha) * Sigma_g + alpha * Sigma_g_new
            Sigma_R = (1 - alpha) * Sigma_R + alpha * Sigma_r_new

        # Causality check on mixed self-energy
        if n_kpts > 1:
            _check_antihermitian_sign(
                Sigma_R[0, 0], freqs_thz, "SCBA self-energy")
        else:
            _check_antihermitian_sign(
                Sigma_R[0], freqs_thz, "SCBA self-energy")

        # Convergence
        if scba_iter > 0:
            rel_change = abs(J_total - J_total_prev) / (abs(J_total_prev) + 1e-30)
            _Sigma_now = np.concatenate([
                Sigma_l.ravel(), Sigma_g.ravel(), Sigma_R.ravel()])
            sig_norm = np.linalg.norm(_Sigma_now)
            sig_change = (np.linalg.norm(_Sigma_now - _Sigma_prev)
                          / (sig_norm + 1e-30) if sig_norm > 0 else 0.0)
            convergence_history.append(rel_change)
            if verbose:
                best_mark = " *" if conservation_err <= best_conservation else ""
                print(f"    SCBA iter {scba_iter + 1}: "
                      f"J = {J_total:.4e} W, "
                      f"conservation = {conservation_err:.4e}, "
                      f"dJ/J = {rel_change:.4e}, "
                      f"dΣ/Σ = {sig_change:.4e}, "
                      f"max|Σ^R| = {np.max(np.abs(Sigma_R)):.2e} THz²"
                      f"{best_mark}")
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

    # Restore best-conservation state if final state is poor
    if (best_state is not None
            and best_state["conservation_err"] < 0.5 * conservation_err):
        spectral_J_L = best_state["spectral_J_L"]
        spectral_J_R = best_state["spectral_J_R"]
        Sigma_R = best_state["Sigma_R"]
        Sigma_l = best_state["Sigma_l"]
        Sigma_g = best_state["Sigma_g"]
        conservation_err = best_state["conservation_err"]
        if verbose:
            print(f"  Using best-conservation state from iter "
                  f"{best_state['iter']} "
                  f"(conservation={conservation_err:.4e})")

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
    fc3_hdf5: str = None,
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
    hilbert_retarded: bool = True,
    scattering_contacts: bool = False,
) -> dict:
    """Reference anharmonic phonon transport (Gamma-point, finite device).

    This is the canonical n_slabs=1 solver. ``anharmonic_transmission_q``
    with ``q_mesh=(1,1)`` must reproduce this result.
    """
    import h5py
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
        if hilbert_retarded:
            print("  Retarded SE: principal-value integral")

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

    # Self-energy kernel for the finite path
    def se_kernel(G_lesser_slab, G_greater_slab):
        # G_lesser_slab: (n_slabs, 1, nfreq, n_dof, n_dof)
        Sigma_l_new = np.zeros((n_slabs, nfreq, n_dof, n_dof), dtype=complex)
        Sigma_g_new = np.zeros_like(Sigma_l_new)
        Sigma_r_new = np.zeros_like(Sigma_l_new)
        for l in range(n_slabs):
            sl_n, sg_n, sr_n = _compute_phph_self_energy_finite(
                G_lesser_slab[l, 0], G_greater_slab[l, 0],
                Phi, freqs_thz, dw_thz,
                hilbert_retarded=hilbert_retarded)
            Sigma_l_new[l] = sl_n
            Sigma_g_new[l] = sg_n
            Sigma_r_new[l] = sr_n
        return Sigma_l_new, Sigma_g_new, Sigma_r_new

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
        "self_energy_retarded": Sigma_R[:, pos_mask],
        "self_energy_lesser": Sigma_l[:, pos_mask],
        "self_energy_greater": Sigma_g[:, pos_mask],
    }


def anharmonic_transmission_q(
    phonon,
    fc3_hdf5: str = None,
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
    hilbert_retarded: bool = True,
    scattering_contacts: bool = False,
) -> dict:
    """Anharmonic phonon transport with full q-dependent dense self-energy.

    Uses a Gamma-centered q-mesh [0, 1/N, ..., (N-1)/N] for closure
    under subtraction (required for q-convolution).
    """
    import h5py
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

    # Self-energy kernel for the q-dense path
    def se_kernel(G_lesser_slab, G_greater_slab):
        # G_lesser_slab: (n_slabs, n_kpts, nfreq, n_dof, n_dof)
        Sigma_l_new = np.zeros(
            (n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex)
        Sigma_g_new = np.zeros_like(Sigma_l_new)
        Sigma_r_new = np.zeros_like(Sigma_l_new)
        for l in range(n_slabs):
            sl_n, sg_n, sr_n = _compute_phph_self_energy_q_dense(
                G_lesser_slab[l], G_greater_slab[l],
                M_stacked, T_all_q, q_diff_map,
                n_atoms, n_kpts, freqs_thz, dw_thz,
                hilbert_retarded=hilbert_retarded)
            Sigma_l_new[l] = sl_n
            Sigma_g_new[l] = sg_n
            Sigma_r_new[l] = sr_n
        return Sigma_l_new, Sigma_g_new, Sigma_r_new

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
        "self_energy_retarded": Sigma_R_q[:, :, pos_mask],
        "self_energy_lesser": Sigma_l_q[:, :, pos_mask],
        "self_energy_greater": Sigma_g_q[:, :, pos_mask],
    }


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------


def _compare_q11_to_finite(res_q, res_f, rtol=5e-3, atol=1e-8):
    """Verify that q=(1,1) matches the finite (Gamma-only) solver."""
    keys = [
        "heat_current",
        "thermal_conductance_anharmonic",
        "heat_flow_conservation",
        "transmission_ballistic",
        "spectral_heat_current",
    ]
    for key in keys:
        a = np.asarray(res_q[key])
        b = np.asarray(res_f[key])
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            raise AssertionError(
                f"q=(1,1) mismatch in {key}: "
                f"max rel err = {np.max(np.abs(a - b) / (np.abs(b) + atol)):.2e}")
