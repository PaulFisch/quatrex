"""Open-boundary leads and device Green's function for the dense solver.

Provides:

  * :func:`sancho_rubio` / :func:`sancho_rubio_batch` — surface Green's
    function via iterative decimation; the residual is checked against a
    relative tolerance and a fallback to scalar iteration is offered.
  * :func:`build_device_hamiltonian` — block-tridiagonal device H for
    ``n_slabs`` identical slabs.
  * :func:`compute_obc_batch` — assembles Σ^R_L, Σ^R_R, Σ^<_{L,R},
    Σ^>_{L,R}, Γ_L, Γ_R on a frequency batch.
  * :func:`solve_green_functions` / :func:`solve_green_batch` —
    G^R, G^<, G^> from the total self-energy.
  * :func:`ballistic_transmission_z2` — T(ω²) = Tr(Γ_L G^R Γ_R G^A).

The retarded scattering self-energy may optionally dress the leads via
``lead_sigma_r_L`` / ``lead_sigma_r_R``; if Sancho-Rubio fails for the
dressed lead the routine falls back to the undressed contact and emits
a warning.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.linalg import inv

from .grids import boson_contact_self_energies_from_gamma


def sancho_rubio(z2, H_00, H_01, max_iter=300, tol=1e-8):
    """Surface Green's function via Sancho-Rubio decimation.

    Raises ``RuntimeError`` if the surface Dyson residual exceeds 1e-4.
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
            f"Sancho-Rubio did not converge: relative residual {res_norm:.2e}"
        )
    return g_surf


def sancho_rubio_batch(z2_arr, H_00, H_01, max_iter=300, tol=1e-8,
                       lead_sigma_r=None):
    """Batched surface Green's function for multiple frequencies.

    Returns ``(g_surf, valid)`` where ``valid[iw]`` is ``True`` if the
    relative residual is acceptable.
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


def build_device_hamiltonian(H_00, H_01, n_slabs):
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


def compute_obc_batch(z2_arr, H_00, H_01, freqs_thz, T_L, T_R,
                      n_slabs=1, lead_sigma_r_L=None, lead_sigma_r_R=None):
    """Batched OBC self-energies for all frequencies.

    Falls back to scalar Sancho-Rubio for frequencies where the batch
    residual is bad. If scattering-dressed contacts fail, falls back to
    undressed contacts and emits a warning.
    """
    nfreq = len(z2_arr)
    n_dof = H_00.shape[0]
    N_D = n_slabs * n_dof

    g_L_all, valid_L = sancho_rubio_batch(
        z2_arr, H_00, H_01, lead_sigma_r=lead_sigma_r_L)
    g_R_all, valid_R = sancho_rubio_batch(
        z2_arr, H_00, H_01.conj().T, lead_sigma_r=lead_sigma_r_R)
    valid = valid_L & valid_R

    n_fallback = 0
    bad: np.ndarray = np.array([], dtype=int)
    if not np.all(valid):
        bad = np.where(~valid)[0]
        for iw in bad:
            try:
                g_L_all[iw] = sancho_rubio(z2_arr[iw], H_00, H_01)
                g_R_all[iw] = sancho_rubio(
                    z2_arr[iw], H_00, H_01.conj().T)
            except RuntimeError:
                if lead_sigma_r_L is None and lead_sigma_r_R is None:
                    raise
                g_L_all[iw] = sancho_rubio(z2_arr[iw], H_00, H_01)
                g_R_all[iw] = sancho_rubio(
                    z2_arr[iw], H_00, H_01.conj().T)
                n_fallback += 1

    if n_fallback > 0:
        warnings.warn(
            f"Sancho-Rubio: {n_fallback}/{len(bad)} frequencies fell back "
            f"to undressed contacts (scattering-dressed failed)"
        )

    H_01_dag = H_01.conj().T
    sig_L_all = H_01_dag[None] @ g_L_all @ H_01[None]
    sig_R_all = H_01[None] @ g_R_all @ H_01_dag[None]

    gam_L_all = 1j * (sig_L_all - sig_L_all.conj().transpose(0, 2, 1))
    gam_R_all = 1j * (sig_R_all - sig_R_all.conj().transpose(0, 2, 1))

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

    Sigma_L_lesser, Sigma_L_greater = boson_contact_self_energies_from_gamma(
        Gamma_L, freqs_thz, T_L)
    Sigma_R_lesser, Sigma_R_greater = boson_contact_self_energies_from_gamma(
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


def solve_green_functions(z2, H_D, obc, Sigma_scatt_R, Sigma_scatt_lesser,
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


def solve_green_batch(z2_arr, H_D, obc_batch,
                      Sigma_scatt_R, Sigma_scatt_lesser, Sigma_scatt_greater):
    """Batched Green's function solve for all frequencies."""
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


def ballistic_transmission_z2(z2, H_D, H_00, H_01, H_LD, H_DR):
    """Ballistic transmission: T = Tr(Γ_L G^R Γ_R G^A)."""
    g_L = sancho_rubio(z2, H_00, H_01)
    g_R = sancho_rubio(z2, H_00, H_01.conj().T)

    Sigma_L = H_LD.conj().T @ g_L @ H_LD
    Sigma_R = H_DR @ g_R @ H_DR.conj().T
    Gamma_L = 1j * (Sigma_L - Sigma_L.conj().T)
    Gamma_R = 1j * (Sigma_R - Sigma_R.conj().T)

    N_D = H_D.shape[0]
    G_R = np.linalg.inv(z2 * np.eye(N_D) - H_D - Sigma_L - Sigma_R)
    G_A = G_R.conj().T
    return float(np.real(np.trace(Gamma_L @ G_R @ Gamma_R @ G_A)))
