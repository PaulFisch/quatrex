"""Reference anharmonic phonon-phonon NEGF transport.

Implements the self-consistent Born approximation (SCBA) for
phonon-phonon scattering following Guo et al., Phys. Rev. B 102,
195412 (2020).

The scattering self-energy (Guo Eq. 8, diagonal block):

    Sigma^{<}(omega) = (i*hbar/2) * sum_{c,d,e,f} Phi3_{a,c,d}
        * integral dw'/(2*pi) G^<_{cf}(w') G^<_{de}(w-w')
        * Phi3_{b,e,f}

Internal units: THz^2 for dynamical matrices and self-energies.
This keeps magnitudes O(1)-O(100) for numerical stability, compared
to (rad/s)^2 which gives O(10^25) magnitudes.

This module provides a reference Python implementation for validation
of the quatrex GPU solver, matching the style of validation.py.
"""

import numpy as np
from numpy.linalg import inv
from concurrent.futures import ThreadPoolExecutor

from .constants import (
    CONVERSION_FC3_THZ,
    CONVERSION_THZ2,
    EV_TO_J,
    HBAR_EV,
    HBAR_SI,
    KB_EV,
    THZ_TO_RAD,
)


# ---------------------------------------------------------------------------
# FC3 loading and Fourier transform
# ---------------------------------------------------------------------------


def load_fc3_for_transport(
    fc3_data: dict,
    n_atoms_prim: int,
    prim_cell: np.ndarray,
) -> dict:
    """Reorganise FC3 data from thirdorder/phono3py format for transport.

    Parameters
    ----------
    fc3_data : dict
        Output of ``force_constants.load_fc3_thirdorder()`` or
        ``force_constants.load_fc3_phono3py()``.
    n_atoms_prim : int
        Number of atoms in the primitive cell.
    prim_cell : ndarray, shape (3, 3)
        Primitive lattice vectors in Angstrom (rows).

    Returns
    -------
    fc3 : dict
        Keys: (cell_j_tuple, cell_k_tuple, atom_i, atom_j, atom_k)
        Values: (3, 3, 3) ndarray in eV/A^3
    """
    fc3 = {}
    for block in fc3_data["blocks"]:
        key = (
            tuple(np.round(block["cell_j"], 8)),
            tuple(np.round(block["cell_k"], 8)),
            block["atom_i"],
            block["atom_j"],
            block["atom_k"],
        )
        fc3[key] = block["tensor"]
    return fc3


def build_fc3_matrix(
    fc3: dict,
    n_atoms: int,
    n_slabs: int,
) -> np.ndarray:
    """Build the FC3 coupling matrix for a finite 1D device.

    Parameters
    ----------
    fc3 : dict
        From ``load_fc3_for_transport()``.
    n_atoms : int
        Atoms per unit cell.
    n_slabs : int
        Number of transport cells (slabs) in the device region.

    Returns
    -------
    fc3_blocks : list of tuples
    """
    return list(fc3.items())


# ---------------------------------------------------------------------------
# Green's function and self-energy computation
# ---------------------------------------------------------------------------


def _sancho_rubio(omega_sq, H_00, H_01, eta=1e-4, max_iter=300, tol=1e-8):
    """Surface Green's function via Sancho-Rubio decimation.

    All quantities in THz^2.
    """
    N = H_00.shape[0]
    H_10 = H_01.conj().T

    a_ii = (omega_sq + 1j * eta) * np.eye(N) - H_00
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

    return inv(eps_s)


def _sancho_rubio_batch(omega_sq_arr, H_00, H_01, eta=1e-4, max_iter=300, tol=1e-8):
    """Batched surface Green's function for multiple frequencies at once.

    Parameters
    ----------
    omega_sq_arr : ndarray, shape (nfreq,)
    H_00, H_01 : ndarray, shape (N, N)

    Returns
    -------
    g_surf : ndarray, shape (nfreq, N, N), complex
    """
    nfreq = len(omega_sq_arr)
    N = H_00.shape[0]
    H_10 = H_01.conj().T
    eye = np.eye(N)

    # (nfreq, N, N) broadcast
    a_ii = (omega_sq_arr[:, None, None] + 1j * eta) * eye[None] - H_00[None]
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
        # Check convergence on the worst-case frequency
        norm_max = max(np.linalg.norm(alpha.reshape(nfreq, -1), axis=1).max(),
                       np.linalg.norm(beta.reshape(nfreq, -1), axis=1).max())
        if norm_max < tol:
            break

    return np.linalg.inv(eps_s)


def _build_device_hamiltonian(H_00, H_01, n_slabs):
    """Build block-tridiagonal device Hamiltonian for N identical slabs.

    H_D = | H_00  H_01   0    ...  0   |
          | H_10  H_00  H_01  ...  0   |
          |  0    H_10  H_00  ...  0   |
          | ...                    ... |
          |  0     0    ...  H_10 H_00 |

    where H_10 = H_01^dag.

    Parameters
    ----------
    H_00 : ndarray, shape (n_dof, n_dof)
    H_01 : ndarray, shape (n_dof, n_dof)
    n_slabs : int

    Returns
    -------
    H_D : ndarray, shape (n_slabs*n_dof, n_slabs*n_dof)
    """
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


def _compute_obc_self_energies(omega_sq, H_00, H_01, eta, n_bose_L, n_bose_R,
                                n_slabs=1):
    """Compute contact self-energies embedded in the N-slab device space.

    All dynamical matrix quantities in THz^2.

    Parameters
    ----------
    omega_sq : float
        w^2 in THz^2.
    H_00, H_01 : ndarray
        On-site and coupling blocks for the lead (THz^2).
    eta : float
        Broadening in THz^2.
    n_bose_L, n_bose_R : float
        Bose-Einstein occupation at the left/right contacts.
    n_slabs : int
        Number of slabs in the device region.

    Returns
    -------
    dict with Sigma_L_R, Sigma_R_R, Gamma_L, Gamma_R, and lesser/greater,
    all of shape (N_D, N_D) where N_D = n_slabs * n_dof.
    """
    n_dof = H_00.shape[0]
    N_D = n_slabs * n_dof

    # Compute lead surface Green's functions (n_dof x n_dof)
    g_L = _sancho_rubio(omega_sq, H_00, H_01, eta)
    g_R = _sancho_rubio(omega_sq, H_00, H_01.conj().T, eta)

    # Lead self-energies (n_dof x n_dof blocks)
    sig_L = H_01.conj().T @ g_L @ H_01
    sig_R = H_01 @ g_R @ H_01.conj().T

    gam_L = 1j * (sig_L - sig_L.conj().T)
    gam_R = 1j * (sig_R - sig_R.conj().T)

    # Embed into device space
    def _embed(block, position):
        M = np.zeros((N_D, N_D), dtype=complex)
        sl = slice(position * n_dof, (position + 1) * n_dof)
        M[sl, sl] = block
        return M

    Sigma_L_R = _embed(sig_L, 0)
    Sigma_R_R = _embed(sig_R, n_slabs - 1)
    Gamma_L = _embed(gam_L, 0)
    Gamma_R = _embed(gam_R, n_slabs - 1)

    Sigma_L_lesser = 1j * n_bose_L * Gamma_L
    Sigma_L_greater = 1j * (n_bose_L + 1) * Gamma_L
    Sigma_R_lesser = 1j * n_bose_R * Gamma_R
    Sigma_R_greater = 1j * (n_bose_R + 1) * Gamma_R

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


def _compute_obc_batch(omega_sq_arr, H_00, H_01, eta, n_bose_L, n_bose_R,
                        n_slabs=1):
    """Batched OBC self-energies for all frequencies at once.

    Returns arrays of shape (nfreq,) for scalar quantities and
    (nfreq, N_D, N_D) for matrices.
    """
    nfreq = len(omega_sq_arr)
    n_dof = H_00.shape[0]
    N_D = n_slabs * n_dof

    # Batched Sancho-Rubio: (nfreq, n_dof, n_dof)
    g_L_all = _sancho_rubio_batch(omega_sq_arr, H_00, H_01, eta)
    g_R_all = _sancho_rubio_batch(omega_sq_arr, H_00, H_01.conj().T, eta)

    H_01_dag = H_01.conj().T
    # sig_L[w] = H_01^dag @ g_L[w] @ H_01, batched
    sig_L_all = H_01_dag[None] @ g_L_all @ H_01[None]  # (nfreq, n_dof, n_dof)
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

    # lesser/greater: (nfreq, N_D, N_D)
    Sigma_L_lesser = 1j * n_bose_L[:, None, None] * Gamma_L
    Sigma_L_greater = 1j * (n_bose_L[:, None, None] + 1) * Gamma_L
    Sigma_R_lesser = 1j * n_bose_R[:, None, None] * Gamma_R
    Sigma_R_greater = 1j * (n_bose_R[:, None, None] + 1) * Gamma_R

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


def _solve_green_functions(omega_sq, H_D, obc, Sigma_scatt_R, Sigma_scatt_lesser,
                           Sigma_scatt_greater, eta):
    """Solve for retarded, lesser, and greater Green's functions.

    G^R = [(w^2 + i*eta)*I - H_D - Sigma_L^R - Sigma_R^R - Sigma_scatt^R]^{-1}
    G^{<,>} = G^R * Sigma^{<,>}_total * G^A

    All in THz^2 units.
    """
    N = H_D.shape[0]

    Sigma_R_total = obc["Sigma_L_R"] + obc["Sigma_R_R"] + Sigma_scatt_R
    G_R = inv((omega_sq + 1j * eta) * np.eye(N) - H_D - Sigma_R_total)
    G_A = G_R.conj().T

    Sigma_lesser_total = (obc["Sigma_L_lesser"] + obc["Sigma_R_lesser"]
                          + Sigma_scatt_lesser)
    Sigma_greater_total = (obc["Sigma_L_greater"] + obc["Sigma_R_greater"]
                           + Sigma_scatt_greater)

    G_lesser = G_R @ Sigma_lesser_total @ G_A
    G_greater = G_R @ Sigma_greater_total @ G_A

    return G_R, G_lesser, G_greater


def _solve_green_batch(omega_sq_arr, H_D, obc_batch,
                       Sigma_scatt_R, Sigma_scatt_lesser, Sigma_scatt_greater,
                       eta):
    """Batched Green's function solve for all frequencies.

    Parameters
    ----------
    omega_sq_arr : (nfreq,)
    H_D : (N_D, N_D)
    obc_batch : dict with (nfreq, N_D, N_D) arrays
    Sigma_scatt_R/lesser/greater : (nfreq, N_D, N_D)

    Returns
    -------
    G_R, G_lesser, G_greater : (nfreq, N_D, N_D)
    """
    nfreq = len(omega_sq_arr)
    N = H_D.shape[0]
    eye = np.eye(N)

    Sigma_R_total = obc_batch["Sigma_L_R"] + obc_batch["Sigma_R_R"] + Sigma_scatt_R
    # A[w] = (w^2 + i*eta)*I - H_D - Sigma_R_total[w]
    A = (omega_sq_arr[:, None, None] + 1j * eta) * eye[None] - H_D[None] - Sigma_R_total
    G_R = np.linalg.inv(A)  # (nfreq, N, N)
    G_A = G_R.conj().transpose(0, 2, 1)

    Sigma_lesser_total = (obc_batch["Sigma_L_lesser"] + obc_batch["Sigma_R_lesser"]
                          + Sigma_scatt_lesser)
    Sigma_greater_total = (obc_batch["Sigma_L_greater"] + obc_batch["Sigma_R_greater"]
                           + Sigma_scatt_greater)

    G_lesser = G_R @ Sigma_lesser_total @ G_A
    G_greater = G_R @ Sigma_greater_total @ G_A

    return G_R, G_lesser, G_greater


# ---------------------------------------------------------------------------
# Full q-dependent self-energy (dense FC3 Fourier transform)
# ---------------------------------------------------------------------------


def _fourier_transform_fc3_pair(M_stacked, T_q1, T_q2, nat_prim):
    """Compute FC3 Fourier transform Phi_hat(q1, q2) on the fly.

    Phi_hat(q1, q2)[a, c, d] = (T(q1) @ M_a @ T(q2)^T)[c, d]

    Parameters
    ----------
    M_stacked : ndarray, shape (n_dof * dim_t, dim_t)
    T_q1, T_q2 : ndarray, shape (n_dof, dim_t)
    nat_prim : int

    Returns
    -------
    Phi_hat : ndarray, shape (n_dof, n_dof, n_dof), complex
    """
    n_dof = nat_prim * 3
    dim_t = T_q1.shape[1]
    # Reshape M_stacked -> (n_dof, dim_t, dim_t), compute all a at once
    M_blocks = M_stacked.reshape(n_dof, dim_t, dim_t)
    # Phi_hat[a] = T_q1 @ M_blocks[a] @ T_q2^T
    # = (n_dof, dim_t) @ (n_dof, dim_t, dim_t) @ (dim_t, n_dof)
    return np.einsum('ci,aij,dj->acd', T_q1, M_blocks, T_q2.conj())


def _compute_phph_self_energy_q_dense(
    G_lesser_q: np.ndarray,
    G_greater_q: np.ndarray,
    M_stacked: np.ndarray,
    T_all_q: list[np.ndarray],
    q_diff_map: np.ndarray,
    nat_prim: int,
    n_kpts: int,
    omega_grid_thz: np.ndarray,
    dw_thz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute q-dependent self-energy via full dense FC3 Fourier transform.

    For each (q, q'), Fourier-transforms the FC3 on-the-fly at (q', q-q')
    and performs the ring contraction via FFT convolution over omega.

    Cost: O(N_q^2 * n_dof^4 * n_fft * log(n_fft))

    Parameters
    ----------
    G_lesser_q : ndarray, shape (n_kpts, n_freq, n_dof, n_dof)
    G_greater_q : ndarray, shape (n_kpts, n_freq, n_dof, n_dof)
    M_stacked : ndarray, shape (n_dof * dim_sc, dim_sc)
        Real-space FC3 matrices from build_realspace_fc3_matrices.
    T_all_q : list of ndarray, each (n_dof, dim_sc)
        Gathering matrices T(q) for each q-point.
    q_diff_map : ndarray, shape (n_kpts, n_kpts)
    nat_prim, n_kpts : int
    omega_grid_thz : ndarray, shape (n_freq,)
    dw_thz : float

    Returns
    -------
    Sigma_lesser, Sigma_greater, Sigma_retarded :
        ndarray, shape (n_kpts, n_freq, n_dof, n_dof)
    """
    n_freq = len(omega_grid_thz)
    n_dof = nat_prim * 3
    dim_t = M_stacked.shape[1]

    # Frequency padding for linear convolution
    n_low = max(0, int(np.round(omega_grid_thz[0] / dw_thz)))
    n_ext = n_low + n_freq
    n_fft = 2 * n_ext
    freq_sl = slice(n_low, n_low + n_freq)

    prefactor = 0.5j * HBAR_SI * dw_thz / (2 * np.pi) / n_kpts

    Sigma_lesser = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)
    Sigma_greater = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)

    # Pad and FFT all Green's functions: (n_kpts, n_fft, n_dof, n_dof)
    GL_padded = np.zeros((n_kpts, n_fft, n_dof, n_dof), dtype=complex)
    GG_padded = np.zeros((n_kpts, n_fft, n_dof, n_dof), dtype=complex)
    GL_padded[:, n_low:n_low + n_freq] = G_lesser_q
    GG_padded[:, n_low:n_low + n_freq] = G_greater_q
    GL_fft = np.fft.fft(GL_padded, axis=1)
    GG_fft = np.fft.fft(GG_padded, axis=1)

    # Precompute TM[iq] = T(q) @ M_blocks, shape (n_kpts, n_dof, n_dof, dim_t)
    # M_blocks: (n_dof, dim_t, dim_t)
    M_blocks = M_stacked.reshape(n_dof, dim_t, dim_t)
    T_arr = np.array(T_all_q)  # (n_kpts, n_dof, dim_t)
    # TM[iq, a, c, j] = T_arr[iq, c, i] * M_blocks[a, i, j]
    TM = np.einsum('qci,aij->qacj', T_arr, M_blocks)  # (n_kpts, n_dof, n_dof, dim_t)

    # Precompute T^T array: (n_kpts, dim_t, n_dof)
    T_arr_T = T_arr.transpose(0, 2, 1)  # (n_kpts, dim_t, n_dof)

    for iq_ext in range(n_kpts):
        for iq_prime in range(n_kpts):
            iq_diff = q_diff_map[iq_ext, iq_prime]

            # Phi_L[a,c,d] = TM[q',a,c,j] * T_arr_T[q-q',j,d]
            Phi_L = TM[iq_prime] @ T_arr_T[iq_diff]  # (n_dof, n_dof, n_dof)
            # Phi_R = conj(TM[q-q'] @ T^T[q'])
            Phi_R = np.conj(TM[iq_diff] @ T_arr_T[iq_prime])

            for G_fft, Sigma_out in [(GL_fft, Sigma_lesser),
                                     (GG_fft, Sigma_greater)]:
                # K[w,c,d,f,e] = IFFT(G_fft[q',w,c,f] * G_fft[q-q',w,d,e])
                product = (G_fft[iq_prime][:, :, None, :, None]
                           * G_fft[iq_diff][:, None, :, None, :])
                K = np.fft.ifft(product, axis=0)[freq_sl]

                # Sigma[w,a,b] += pref * Phi_L[a,c,d] K[w,c,d,f,e] Phi_R[b,e,f]
                temp = np.einsum('acd,wcdfe->wafe', Phi_L, K)
                Sigma_out[iq_ext] += prefactor * np.einsum(
                    'wafe,bef->wab', temp, Phi_R
                )

    Sigma_retarded = 0.5 * (Sigma_greater - Sigma_lesser)
    return Sigma_lesser, Sigma_greater, Sigma_retarded


def _compute_self_energy_parallel(
    G_lesser_q, G_greater_q, M_stacked, T_all_q, q_diff_map,
    nat_prim, n_kpts, omega_grid_thz, dw_thz, n_threads=128,
):
    """Parallelized self-energy: distribute iq_ext over threads.

    Same interface as _compute_phph_self_energy_q_dense but uses
    ThreadPoolExecutor to parallelize the outer q-loop.
    Threads work because the inner loop releases the GIL via NumPy.
    """
    n_freq = len(omega_grid_thz)
    n_dof = nat_prim * 3
    dim_t = M_stacked.shape[1]

    n_low = max(0, int(np.round(omega_grid_thz[0] / dw_thz)))
    n_ext = n_low + n_freq
    n_fft = 2 * n_ext
    freq_sl = slice(n_low, n_low + n_freq)
    prefactor = 0.5j * HBAR_SI * dw_thz / (2 * np.pi) / n_kpts

    # Pad + FFT Green's functions
    GL_padded = np.zeros((n_kpts, n_fft, n_dof, n_dof), dtype=complex)
    GG_padded = np.zeros((n_kpts, n_fft, n_dof, n_dof), dtype=complex)
    GL_padded[:, n_low:n_low + n_freq] = G_lesser_q
    GG_padded[:, n_low:n_low + n_freq] = G_greater_q
    GL_fft = np.fft.fft(GL_padded, axis=1)
    GG_fft = np.fft.fft(GG_padded, axis=1)

    # Precompute
    M_blocks = M_stacked.reshape(n_dof, dim_t, dim_t)
    T_arr = np.array(T_all_q)
    TM = np.einsum('qci,aij->qacj', T_arr, M_blocks)
    T_arr_T = T_arr.transpose(0, 2, 1)

    def process_iq_ext(iq_ext):
        """Compute Sigma contributions for a single external q-point."""
        sig_l = np.zeros((n_freq, n_dof, n_dof), dtype=complex)
        sig_g = np.zeros((n_freq, n_dof, n_dof), dtype=complex)

        for iq_prime in range(n_kpts):
            iq_diff = q_diff_map[iq_ext, iq_prime]

            Phi_L = TM[iq_prime] @ T_arr_T[iq_diff]
            Phi_R = np.conj(TM[iq_diff] @ T_arr_T[iq_prime])

            for G_fft, sig_out in [(GL_fft, sig_l), (GG_fft, sig_g)]:
                product = (G_fft[iq_prime][:, :, None, :, None]
                           * G_fft[iq_diff][:, None, :, None, :])
                K = np.fft.ifft(product, axis=0)[freq_sl]
                temp = np.einsum('acd,wcdfe->wafe', Phi_L, K)
                sig_out += prefactor * np.einsum('wafe,bef->wab', temp, Phi_R)

        return iq_ext, sig_l, sig_g

    Sigma_lesser = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)
    Sigma_greater = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(process_iq_ext, iq) for iq in range(n_kpts)]
        for f in futures:
            iq_ext, sig_l, sig_g = f.result()
            Sigma_lesser[iq_ext] = sig_l
            Sigma_greater[iq_ext] = sig_g

    Sigma_retarded = 0.5 * (Sigma_greater - Sigma_lesser)
    return Sigma_lesser, Sigma_greater, Sigma_retarded


def anharmonic_transmission_q(
    phonon,
    fc3_hdf5: str = None,
    q_mesh_transverse: tuple[int, int] = (4, 4),
    freq_range_thz: tuple[float, float, int] = (0.01, 16.0, 101),
    transport_direction: str = "x",
    eta_factor: float = 0.5,
    temperature: float = 300.0,
    delta_T: float = 10.0,
    max_scba_iter: int = 10,
    scba_tol: float = 0.01,
    mixing: float = 0.5,
    n_slabs: int = 1,
    verbose: bool = True,
    M_stacked_override: np.ndarray = None,
    n_threads: int = 4,
) -> dict:
    """Anharmonic phonon transport with full q-dependent dense self-energy.

    Uses a Gamma-centered q-mesh [0, 1/N, ..., (N-1)/N] for closure
    under subtraction (required for q-convolution).

    Parameters
    ----------
    phonon : Phonopy
    fc3_hdf5 : str or Path
        Path to raw fc3.hdf5 from phono3py.
    q_mesh_transverse : (int, int)
    freq_range_thz : (fmin, fmax, nfreq)
    transport_direction : str
    eta_factor, temperature, delta_T, max_scba_iter, scba_tol, mixing : float
    n_slabs : int
    verbose : bool
    M_stacked_override : ndarray, optional
        If provided, use this M_stacked instead of loading from fc3_hdf5.
    n_threads : int
        Number of threads for self-energy parallelization.

    Returns
    -------
    result : dict
    """
    import h5py
    from .convention import get_btd_blocks
    from .validation import _ballistic_transmission
    from .separable import (
        build_supercell_mapping,
        build_realspace_fc3_matrices,
        build_gathering_matrix,
        build_q_diff_map,
    )

    # --- Setup ---
    fmin, fmax, nfreq = freq_range_thz
    nfreq = int(nfreq)
    freqs_thz = np.linspace(fmin, fmax, nfreq)
    omega_sq_thz2 = freqs_thz ** 2
    dw_thz = freqs_thz[1] - freqs_thz[0]
    eta = dw_thz ** 2 * eta_factor

    n_atoms = len(phonon.primitive.masses)
    n_dof = 3 * n_atoms
    N_D = n_slabs * n_dof

    # --- Supercell mapping ---
    prim_indices, cell_frac, slab_indices, ref_sc_atoms = build_supercell_mapping(
        phonon, transport_direction
    )
    masses_super = phonon.supercell.masses
    n_super = len(masses_super)
    dim_sc = n_super * 3

    # --- Load raw FC3 and build real-space matrices ---
    if M_stacked_override is not None:
        M_stacked = M_stacked_override
    else:
        with h5py.File(fc3_hdf5, "r") as f:
            fc3_raw = np.array(f["fc3"])
        M_stacked = build_realspace_fc3_matrices(
            fc3_raw, n_atoms, masses_super, ref_sc_atoms
        )

    if verbose:
        if M_stacked_override is None:
            print(f"  FC3 raw shape: {fc3_raw.shape}")
        print(f"  Supercell atoms: {n_super}, dim_sc: {dim_sc}")
        print(f"  M_stacked norm: {np.linalg.norm(M_stacked):.4e}")
        print(f"  Device: {n_slabs} slab(s), {N_D} DOFs per q-point")

    # --- Gamma-centered q-mesh (closed under subtraction) ---
    nkx, nky = q_mesh_transverse
    q_1d_x = [i / nkx for i in range(nkx)]
    q_1d_y = [j / nky for j in range(nky)]
    q_points = [(qx, qy) for qx in q_1d_x for qy in q_1d_y]
    n_kpts = len(q_points)
    q_diff_map = build_q_diff_map(nkx, nky)

    # Build gathering matrices T(q) for each q-point
    T_all_q = []
    for qx, qy in q_points:
        T = build_gathering_matrix(
            prim_indices, cell_frac,
            (qx, qy), n_atoms, transport_direction,
        )
        T_all_q.append(T)

    # --- Bose-Einstein ---
    def bose_einstein(freq_thz_arr, T):
        hw = HBAR_EV * np.abs(freq_thz_arr) * THZ_TO_RAD
        x = hw / (KB_EV * T)
        n = np.zeros_like(x)
        valid = x > 1e-10
        n[valid] = 1.0 / (np.exp(x[valid]) - 1.0)
        return n

    T_L = temperature + delta_T / 2.0
    T_R = temperature - delta_T / 2.0
    n_bose_L = bose_einstein(freqs_thz, T_L)
    n_bose_R = bose_einstein(freqs_thz, T_R)

    if verbose:
        print(f"  q-mesh: {nkx}x{nky} = {n_kpts} (Gamma-centered)")
        print(f"  Frequency grid: {nfreq} points, {fmin:.2f} to {fmax:.2f} THz")
        print(f"  Temperature: {temperature} K, delta_T: {delta_T} K")
        print(f"  eta = {eta:.4e} THz^2")
        print(f"  SCBA: max {max_scba_iter} iter, tol={scba_tol}, mix={mixing}")
        print(f"  Threads: {n_threads}")

    # --- BTD blocks per q-point ---
    btd_blocks = []
    for qx, qy in q_points:
        H_00, H_01 = get_btd_blocks(
            phonon, (qx, qy), transport_direction=transport_direction,
            conversion_factor=CONVERSION_THZ2,
        )
        btd_blocks.append((H_00, H_01))

    # --- Precompute OBC self-energies for all q and all frequencies ---
    # This replaces per-(iq, iw) calls with batched computation
    if verbose:
        print("  Precomputing OBC self-energies (batched)...")
    obc_all = []  # list of dicts, one per q-point
    H_D_all = []
    for iq, (H_00, H_01) in enumerate(btd_blocks):
        H_D = _build_device_hamiltonian(H_00, H_01, n_slabs)
        H_D_all.append(H_D)
        obc = _compute_obc_batch(omega_sq_thz2, H_00, H_01, eta,
                                 n_bose_L, n_bose_R, n_slabs=n_slabs)
        obc_all.append(obc)

    # --- Ballistic transmission (vectorized over frequencies) ---
    trans_ballistic = np.zeros(nfreq)
    for iq, (H_00, H_01) in enumerate(btd_blocks):
        H_D = H_D_all[iq]
        H_LD = np.zeros((n_dof, N_D), dtype=complex)
        H_LD[:, :n_dof] = H_01
        H_DR = np.zeros((N_D, n_dof), dtype=complex)
        H_DR[-n_dof:, :] = H_01
        for iw, w2 in enumerate(omega_sq_thz2):
            trans_ballistic[iw] += _ballistic_transmission(
                w2, H_D, H_00, H_01, H_00, H_01, H_LD, H_DR, eta=eta
            )
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
    spectral_J_ball = (HBAR_SI * omega_rad
                       * (n_bose_L - n_bose_R) * trans_ballistic)
    J_ball_total = np.sum(spectral_J_ball) * dw_thz * 1e12
    G_ball = J_ball_total / (A_c * delta_T)

    if verbose:
        print(f"  Ballistic thermal conductance: {G_ball:.2f} W/(m^2 K)")

    # --- SCBA with q-dependent dense self-energy ---
    Sigma_R_q = np.zeros((n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex)
    Sigma_l_q = np.zeros_like(Sigma_R_q)
    Sigma_g_q = np.zeros_like(Sigma_R_q)

    convergence_history = []
    J_total_prev = 0.0
    conservation_err = 1.0

    spectral_J_L = np.zeros(nfreq)
    spectral_J_R = np.zeros(nfreq)

    sl0 = slice(0, n_dof)
    sl_last = slice((n_slabs - 1) * n_dof, n_slabs * n_dof)

    for scba_iter in range(max_scba_iter):
        G_lesser_slab_q = np.zeros(
            (n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex
        )
        G_greater_slab_q = np.zeros_like(G_lesser_slab_q)
        spectral_J_L[:] = 0.0
        spectral_J_R[:] = 0.0

        # --- Vectorized Green's function solve per q-point ---
        for iq in range(n_kpts):
            H_D = H_D_all[iq]
            obc = obc_all[iq]

            # Build scattering self-energy arrays: (nfreq, N_D, N_D)
            Sig_R_dev = np.zeros((nfreq, N_D, N_D), dtype=complex)
            Sig_l_dev = np.zeros((nfreq, N_D, N_D), dtype=complex)
            Sig_g_dev = np.zeros((nfreq, N_D, N_D), dtype=complex)
            for l in range(n_slabs):
                sl = slice(l * n_dof, (l + 1) * n_dof)
                Sig_R_dev[:, sl, sl] = Sigma_R_q[l, iq]  # (nfreq, n_dof, n_dof)
                Sig_l_dev[:, sl, sl] = Sigma_l_q[l, iq]
                Sig_g_dev[:, sl, sl] = Sigma_g_q[l, iq]

            # Batched solve: all frequencies at once
            _, G_less, G_great = _solve_green_batch(
                omega_sq_thz2, H_D, obc, Sig_R_dev, Sig_l_dev, Sig_g_dev, eta)

            # Extract slab-diagonal blocks
            for l in range(n_slabs):
                sl = slice(l * n_dof, (l + 1) * n_dof)
                G_lesser_slab_q[l, iq] = G_less[:, sl, sl]
                G_greater_slab_q[l, iq] = G_great[:, sl, sl]

            # Spectral current from contacts (vectorized over freq)
            # J_L[w] += hbar*omega * Tr(Sig_L^> G^< - Sig_L^< G^>)
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

        J_L_total = np.sum(spectral_J_L) * dw_thz * 1e12
        J_R_total = np.sum(spectral_J_R) * dw_thz * 1e12
        J_total = 0.5 * (J_L_total + J_R_total)
        J_denom = abs(J_L_total) + abs(J_R_total)
        conservation_err = (abs(J_L_total - J_R_total) / J_denom
                            if J_denom > 0 else 0.0)

        # Compute per-slab self-energy via dense q-dependent kernel
        Sigma_l_new = np.zeros_like(Sigma_l_q)
        Sigma_g_new = np.zeros_like(Sigma_g_q)
        Sigma_r_new = np.zeros_like(Sigma_R_q)

        for l in range(n_slabs):
            sl_n, sg_n, sr_n = _compute_self_energy_parallel(
                G_lesser_slab_q[l], G_greater_slab_q[l],
                M_stacked, T_all_q, q_diff_map,
                n_atoms, n_kpts, freqs_thz, dw_thz,
                n_threads=n_threads,
            )
            Sigma_l_new[l] = sl_n
            Sigma_g_new[l] = sg_n
            Sigma_r_new[l] = sr_n

        sig_r_norm = np.max(np.abs(Sigma_r_new))

        if verbose and scba_iter == 0:
            gl_max = np.max(np.abs(G_lesser_slab_q))
            h00_max = max(np.max(np.abs(H_bl)) for H_bl, _ in btd_blocks)
            print(f"    G diagnostic: max|G^<| = {gl_max:.4e}")
            print(f"    Self-energy: max|Sigma^R| = {sig_r_norm:.4e} THz^2, "
                  f"|Sigma^R|/|H_00| = {sig_r_norm / h00_max:.4e}")

        # Mix
        if scba_iter > 0:
            alpha = mixing
            Sigma_l_q = (1 - alpha) * Sigma_l_q + alpha * Sigma_l_new
            Sigma_g_q = (1 - alpha) * Sigma_g_q + alpha * Sigma_g_new
            Sigma_R_q = (1 - alpha) * Sigma_R_q + alpha * Sigma_r_new
        else:
            Sigma_l_q = Sigma_l_new.copy()
            Sigma_g_q = Sigma_g_new.copy()
            Sigma_R_q = Sigma_r_new.copy()

        # Convergence
        if scba_iter > 0:
            rel_change = abs(J_total - J_total_prev) / (abs(J_total_prev) + 1e-30)
            convergence_history.append(rel_change)
            if verbose:
                print(f"    SCBA iter {scba_iter + 1}: "
                      f"J = {J_total:.4e} W, "
                      f"conservation = {conservation_err:.4e}, "
                      f"rel. change = {rel_change:.4e}, "
                      f"max|Sigma^R| = {np.max(np.abs(Sigma_R_q)):.2e} THz^2")
            if conservation_err < scba_tol:
                if verbose:
                    print(f"    Converged after {scba_iter + 1} iterations")
                break
        else:
            if verbose:
                print(f"    SCBA iter 1: "
                      f"J_L = {J_L_total:.4e} W, J_R = {J_R_total:.4e} W")

        J_total_prev = J_total

    # Final results
    spectral_J_anh = 0.5 * (spectral_J_L + spectral_J_R)
    J_anh_total = np.sum(spectral_J_anh) * dw_thz * 1e12
    G_anh = J_anh_total / (A_c * delta_T)

    if verbose:
        print(f"  Anharmonic thermal conductance: {G_anh:.2f} W/(m^2 K)")
        print(f"  Heat flow conservation: {conservation_err:.4e}")

    return {
        "freqs_thz": freqs_thz,
        "omega_rad": freqs_thz * THZ_TO_RAD,
        "transmission_ballistic": trans_ballistic,
        "spectral_heat_current_ballistic": spectral_J_ball,
        "spectral_heat_current": spectral_J_anh,
        "spectral_heat_current_L": spectral_J_L.copy(),
        "spectral_heat_current_R": spectral_J_R.copy(),
        "heat_current_ballistic": J_ball_total,
        "heat_current": J_anh_total,
        "thermal_conductance_ballistic": G_ball,
        "thermal_conductance_anharmonic": G_anh,
        "heat_flow_conservation": conservation_err,
        "delta_T": delta_T,
        "n_scba_iterations": len(convergence_history) + 1,
        "convergence_history": convergence_history,
        "self_energy_retarded": Sigma_R_q,
        "self_energy_lesser": Sigma_l_q,
        "self_energy_greater": Sigma_g_q,
    }
