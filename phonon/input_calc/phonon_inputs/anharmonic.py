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


def _sancho_rubio_batch(omega_sq_arr, H_00, H_01, eta=1e-4, max_iter=300, tol=1e-8,
                        lead_sigma_r=None):
    """Batched surface Green's function for multiple frequencies at once.

    Parameters
    ----------
    omega_sq_arr : ndarray, shape (nfreq,)
    H_00, H_01 : ndarray, shape (N, N)
    lead_sigma_r : ndarray, shape (nfreq, N, N), optional
        Retarded scattering self-energy in the lead bulk.  Included in
        the on-site block of the semi-infinite lead, following the
        standard NEGF approach (cf. qttools / OMEN / TranSIESTA).

    Returns
    -------
    g_surf : ndarray, shape (nfreq, N, N), complex
    """
    nfreq = len(omega_sq_arr)
    N = H_00.shape[0]
    H_10 = H_01.conj().T
    eye = np.eye(N)

    # (nfreq, N, N) broadcast — system-matrix diagonal block:
    # a_ii = (w^2 + i*eta)*I - H_00 - Sigma_scatt^R
    a_ii = (omega_sq_arr[:, None, None] + 1j * eta) * eye[None] - H_00[None]
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
                        n_slabs=1, lead_sigma_r_L=None, lead_sigma_r_R=None):
    """Batched OBC self-energies for all frequencies at once.

    Parameters
    ----------
    lead_sigma_r_L, lead_sigma_r_R : ndarray (nfreq, n_dof, n_dof), optional
        Retarded scattering self-energy for the left / right lead.
        Included in the Sancho-Rubio on-site block so that the
        contacts carry the same scattering as the device boundary.

    Returns arrays of shape (nfreq,) for scalar quantities and
    (nfreq, N_D, N_D) for matrices.
    """
    nfreq = len(omega_sq_arr)
    n_dof = H_00.shape[0]
    N_D = n_slabs * n_dof

    # Batched Sancho-Rubio: (nfreq, n_dof, n_dof)
    g_L_all = _sancho_rubio_batch(omega_sq_arr, H_00, H_01, eta,
                                  lead_sigma_r=lead_sigma_r_L)
    g_R_all = _sancho_rubio_batch(omega_sq_arr, H_00, H_01.conj().T, eta,
                                  lead_sigma_r=lead_sigma_r_R)

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


def _hilbert_transform_axis(f, axis=1):
    """Hilbert transform along *axis* via FFT (sign-multiplier method).

    Computes  H[f](x) = (1/pi) P int f(x') / (x - x') dx'
    on a uniformly spaced grid, assuming periodic boundary conditions.

    Parameters
    ----------
    f : ndarray, complex
    axis : int
        Axis along which to transform (frequency axis).

    Returns
    -------
    Hf : ndarray, same shape as *f*
    """
    N = f.shape[axis]
    Ff = np.fft.fft(f, axis=axis)

    # Multiplier: -i * sign(k)
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


def _se_worker_iq(args):
    """Process one chunk of iq_ext values using FFT-domain Phi contraction.

    Key optimizations:
    1. Contracts Phi with G in FFT domain before IFFT, reducing the
       IFFT from (n_fft, n_dof^4) to (n_fft, n_dof^2).
    2. Uses np.matmul (BLAS gemm) instead of einsum for ~2.5x speedup.

    Contraction chain per (iq_ext, q'):
        A[p,W,ac,e] = PL_flat[p,ac,d] @ G2[p,W,d,e]   (matmul over d)
        B[p,W,a,e,f] = A[p,W,a,e,c] @ G1[p,W,c,f]     (matmul over c)
        S_hat[Wa,b]  += B_flat[p,Wa,ef] @ PR_flat[p,ef,b].T  (matmul over ef)
    Then Sigma = prefactor * IFFT(S_hat)[freq_sl]
    """
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
        iq_diff_arr = q_diff_map[iq_ext]  # (n_kpts,)

        # Gather Phi for all q' pairs: (nq, nd, nd, nd)
        PL_all = Phi_all[iq_prime_all, iq_diff_arr]
        PR_all = np.conj(Phi_all[iq_diff_arr, iq_prime_all])

        for G_fft, sig_out in [(GL_fft, sig_l), (GG_fft, sig_g)]:
            S_hat = np.zeros((n_fft, nd, nd), dtype=complex)

            for p0 in range(0, n_kpts, qp_batch):
                p1 = min(p0 + qp_batch, n_kpts)
                B = p1 - p0
                ps = slice(p0, p1)
                iq_d = iq_diff_arr[ps]

                PL = PL_all[ps]     # (B, nd, nd, nd)
                PR = PR_all[ps]     # (B, nd, nd, nd)
                G1 = G_fft[p0:p1]   # (B, nfft, nd, nd)
                G2 = G_fft[iq_d]    # (B, nfft, nd, nd)

                # Step 1: A[p,W,ac,e] = PL[p,ac,d] @ G2[p,W,d,e]
                # PL_flat: (B, nd*nd, nd) @ G2: (B, nfft, nd, nd)
                # broadcast: (B, 1, nd*nd, nd) @ (B, nfft, nd, nd) -> (B, nfft, nd*nd, nd)
                PL_flat = PL.reshape(B, nd2, nd)
                A = np.matmul(PL_flat[:, None, :, :], G2)  # (B, nfft, nd2, nd)
                A = A.reshape(B, n_fft, nd, nd, nd)         # (B, nfft, a, c, e)

                # Step 2: B5[p,W,a,e,f] = A[p,W,a,e,c] @ G1[p,W,c,f]
                # transpose A -> (B, nfft, a, e, c), matmul with G1 (B, nfft, 1, c, f)
                A = A.transpose(0, 1, 2, 4, 3)             # (B, nfft, a, e, c)
                B5 = np.matmul(A, G1[:, :, None, :, :])    # (B, nfft, a, e, f)

                # Step 3: S_hat[W,a,b] += sum_{p,e,f} B5[p,W,a,e,f] * PR[p,b,e,f]
                # reshape to (B, nfft*nd, nd*nd) and (B, nd, nd*nd)
                B5_flat = B5.reshape(B, n_fft * nd, nd2)
                PR_flat = PR.reshape(B, nd, nd2)
                # (B, nfft*nd, nd2) @ (B, nd2, nd) -> (B, nfft*nd, nd), sum over B
                S_chunk = np.matmul(B5_flat, PR_flat.transpose(0, 2, 1))
                S_hat += S_chunk.sum(axis=0).reshape(n_fft, nd, nd)

            sig_out[idx] = prefactor * np.fft.ifft(S_hat, axis=0)[freq_sl]

    return iq_ext_list, sig_l, sig_g


def _compute_phph_self_energy_q_dense(
    G_lesser_q, G_greater_q, M_stacked, T_all_q, q_diff_map,
    nat_prim, n_kpts, omega_grid_thz, dw_thz, n_workers=None,
    hilbert_retarded=False,
):
    """Compute q-dependent phonon-phonon self-energy.

    Uses two key optimizations:
    1. Contracts Phi with G in the FFT domain before IFFT, reducing
       the 6-index (n_fft, n_dof^4) temporary to 5-index (n_fft, n_dof^3)
       and the final IFFT to (n_fft, n_dof^2).
    2. Distributes iq_ext over OS processes (fork) to bypass GIL.
       Workers auto-tuned to min(n_kpts, available_cores).

    Parameters
    ----------
    G_lesser_q, G_greater_q : (n_kpts, n_freq, n_dof, n_dof)
    M_stacked : (n_dof * dim_sc, dim_sc)
    T_all_q : list of (n_dof, dim_sc) arrays
    q_diff_map : (n_kpts, n_kpts) int array
    nat_prim, n_kpts : int
    omega_grid_thz : (n_freq,)
    dw_thz : float
    n_workers : int or None
        Number of worker processes. None = auto (min of n_kpts, cpu_count).
    hilbert_retarded : bool
        If True, compute the retarded self-energy via the full
        Kramers-Kronig relation (Hilbert transform of Sigma^> - Sigma^<)
        instead of just the instantaneous part 0.5*(Sigma^> - Sigma^<).

    Returns
    -------
    Sigma_lesser, Sigma_greater, Sigma_retarded : (n_kpts, n_freq, n_dof, n_dof)
    """
    import os

    n_freq = len(omega_grid_thz)
    n_dof = nat_prim * 3
    dim_t = M_stacked.shape[1]

    # Linear convolution via zero-padded FFT.
    # With a symmetric grid [-fmax,...,0,...,fmax], the convolution of
    # two n_freq-length signals has length 2*n_freq-1.  The output
    # mapping back to our grid starts at index mid = (n_freq-1)//2.
    n_fft = 2 * n_freq - 1
    mid = (n_freq - 1) // 2
    freq_sl = slice(mid, mid + n_freq)

    # Prefactor: (i*hbar/2) * dw_THz/(2*pi) / n_kpts.
    # dw_thz/(2*pi) is correct (not dw_thz*1e12) because the FC3
    # conversion factor CONVERSION_FC3_THZ absorbs THZ_TO_RAD^{5/2},
    # so G is in THz^{-2} and the convolution integral measure is dw'_THz/(2*pi).
    prefactor = 0.5j * HBAR_SI * dw_thz / (2 * np.pi) / n_kpts

    # Pad + FFT all Green's functions: (n_kpts, n_fft, n_dof, n_dof)
    G_padded = np.zeros((n_kpts, n_fft, n_dof, n_dof), dtype=complex)
    G_padded[:, :n_freq] = G_lesser_q
    GL_fft = np.fft.fft(G_padded, axis=1)
    G_padded[:, :] = 0
    G_padded[:, :n_freq] = G_greater_q
    GG_fft = np.fft.fft(G_padded, axis=1)
    del G_padded

    # Precompute ALL Phi matrices: Phi_all[q1, q2, a, c, d]
    M_blocks = M_stacked.reshape(n_dof, dim_t, dim_t)
    T_arr = np.array(T_all_q)  # (n_kpts, n_dof, dim_t)
    TM = np.einsum('qci,aij->qacj', T_arr, M_blocks)
    T_arr_T = T_arr.transpose(0, 2, 1).copy()
    Phi_all = np.einsum('qacj,rjd->qracd', TM, T_arr_T)
    del TM, T_arr_T, T_arr, M_blocks

    # Batch size for q' loop: keep intermediates in L2/L3 cache
    # Each batch creates (B, n_fft, n_dof, n_dof, n_dof) ~ B*n_fft*n_dof^3*16 bytes
    target_bytes = 16 * 1024 * 1024  # 16 MB target per intermediate
    bytes_per_qp = n_fft * n_dof**3 * 16  # complex128
    qp_batch = max(1, min(n_kpts, target_bytes // max(bytes_per_qp, 1)))

    if n_workers is None:
        n_workers = min(n_kpts, os.cpu_count() or 1)
    n_workers = min(n_workers, n_kpts)

    # Ensure each worker has >= 2 iq_ext to amortize fork overhead
    if n_kpts > 1:
        n_workers = min(n_workers, max(1, n_kpts // 2))

    common = (GL_fft, GG_fft, Phi_all, q_diff_map,
              n_freq, n_dof, n_kpts, n_fft,
              freq_sl.start, freq_sl.stop, prefactor, qp_batch)

    if n_workers <= 1:
        _, sig_l, sig_g = _se_worker_iq(
            (list(range(n_kpts)), *common))
        delta = sig_g - sig_l
        if hilbert_retarded:
            sig_r = 0.5 * delta + 0.5j * _hilbert_transform_axis(delta, axis=1)
        else:
            sig_r = 0.5 * delta
        return sig_l, sig_g, sig_r

    # Round-robin iq_ext across workers
    chunks = [[] for _ in range(n_workers)]
    for iq in range(n_kpts):
        chunks[iq % n_workers].append(iq)
    chunks = [c for c in chunks if c]

    work_args = [(chunk, *common) for chunk in chunks]

    Sigma_lesser = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)
    Sigma_greater = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)

    # Two parallelism strategies:
    # - Threads: safe with any BLAS, but all share one OpenMP pool.
    #   Best with OMP_NUM_THREADS=1 and many workers.
    # - Processes (forkserver): each gets its own BLAS/OpenMP runtime.
    #   Best with fewer workers × more BLAS threads per worker.
    #   forkserver avoids fork+OpenMP deadlock (unlike plain fork).
    use_threads = os.environ.get("QUATREX_SE_THREADS", "1") == "1"

    if use_threads:
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [executor.submit(_se_worker_iq, wa) for wa in work_args]
            for fut in futures:
                iq_list, sig_l, sig_g = fut.result()
                for i, iq in enumerate(iq_list):
                    Sigma_lesser[iq] = sig_l[i]
                    Sigma_greater[iq] = sig_g[i]
    else:
        ctx = get_context("forkserver")
        with ctx.Pool(processes=len(chunks)) as pool:
            for iq_list, sig_l, sig_g in pool.map(_se_worker_iq, work_args):
                for i, iq in enumerate(iq_list):
                    Sigma_lesser[iq] = sig_l[i]
                    Sigma_greater[iq] = sig_g[i]

    delta = Sigma_greater - Sigma_lesser
    if hilbert_retarded:
        Sigma_retarded = 0.5 * delta + 0.5j * _hilbert_transform_axis(delta, axis=1)
    else:
        Sigma_retarded = 0.5 * delta
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
    anderson_mixing: bool = False,
    anderson_depth: int = 5,
    n_slabs: int = 1,
    verbose: bool = True,
    M_stacked_override: np.ndarray = None,
    hilbert_retarded: bool = False,
    scattering_contacts: bool = False,
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
    freq_range_thz : (_, fmax, nfreq_pos)
        The first element is ignored (kept for backward compatibility).
        ``nfreq_pos`` uniformly-spaced points from 0 to ``fmax`` THz
        are generated; the internal grid is the symmetric mirror
        ``[-fmax, ..., 0, ..., fmax]`` (``2*nfreq_pos - 1`` points).
        Only the positive-frequency side is returned.
    transport_direction : str
    eta_factor, temperature, delta_T, max_scba_iter, scba_tol, mixing : float
    anderson_mixing : bool
        Use Anderson acceleration instead of plain linear mixing.
        Works well when q-mesh matches supercell resolution (e.g. 4x4
        for 3x3x3 supercell). May overshoot for large q-meshes.
    anderson_depth : int
        Number of history vectors for Anderson mixing (default 5).
    n_slabs : int
    verbose : bool
    M_stacked_override : ndarray, optional
        If provided, use this M_stacked instead of loading from fc3_hdf5.
    hilbert_retarded : bool
        If True, compute Sigma^R via the full Kramers-Kronig relation
        (includes the Hilbert-transform / principal-value integral of
        Sigma^> - Sigma^<) instead of the default instantaneous
        approximation 0.5*(Sigma^> - Sigma^<).
    scattering_contacts : bool
        If True, include the device-boundary scattering self-energy in
        the lead surface Green's functions (recomputed each SCBA
        iteration).  The system-matrix on-site block passed to
        Sancho-Rubio becomes (w^2+i*eta)I - H_00 - Sigma^R_scatt,
        mirroring what production NEGF codes do for electrons.  Falls
        back to ballistic OBC for any q-point where Sancho-Rubio
        diverges.

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

    # --- Setup: symmetric frequency grid [-fmax, ..., 0, ..., fmax] ---
    # Grid must be uniform for FFT convolution. nfreq_pos points from 0 to fmax,
    # mirrored to negative side. fmin is ignored for grid construction (the full
    # symmetric grid always includes 0). Output returns ω >= 0 only.
    _fmin, fmax, nfreq_pos = freq_range_thz
    nfreq_pos = int(nfreq_pos)
    freqs_pos = np.linspace(0.0, fmax, nfreq_pos)
    freqs_thz = np.concatenate((-freqs_pos[:0:-1], freqs_pos))
    nfreq = len(freqs_thz)
    dw_thz = freqs_pos[1] - freqs_pos[0]
    omega_sq_thz2 = freqs_thz ** 2
    eta = dw_thz ** 2 * eta_factor
    pos_mask = freqs_thz >= 0.0

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

    # --- Gamma-centered q-mesh ---
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

    # --- Bose-Einstein (SI units, expm1 for numerical stability) ---
    def bose_einstein(freq_thz_arr, T):
        omega_rad_s = np.abs(freq_thz_arr) * THZ_TO_RAD
        x = HBAR_SI * omega_rad_s / (KB_SI * T)
        n = np.zeros_like(x)
        valid = x > 1e-12
        n[valid] = 1.0 / np.expm1(x[valid])
        return n

    T_L = temperature + delta_T / 2.0
    T_R = temperature - delta_T / 2.0
    n_bose_L = bose_einstein(freqs_thz, T_L)
    n_bose_R = bose_einstein(freqs_thz, T_R)

    if verbose:
        print(f"  q-mesh: {nkx}x{nky} = {n_kpts} (Gamma-centered)")
        print(f"  Frequency grid: {nfreq} points ({nfreq_pos} positive), "
              f"{freqs_thz[0]:.2f} to {freqs_thz[-1]:.2f} THz")
        print(f"  Temperature: {temperature} K, delta_T: {delta_T} K")
        print(f"  eta = {eta:.4e} THz^2")
        mix_str = (f"Anderson(depth={anderson_depth})" if anderson_mixing
                   else "linear")
        print(f"  SCBA: max {max_scba_iter} iter, tol={scba_tol}, "
              f"mix={mixing}, method={mix_str}")


    # --- BTD blocks per q-point ---
    btd_blocks = []
    for qx, qy in q_points:
        H_00, H_01 = get_btd_blocks(
            phonon, (qx, qy), transport_direction=transport_direction,
            conversion_factor=CONVERSION_THZ2,
        )
        btd_blocks.append((H_00, H_01))

    # --- Precompute OBC self-energies for all q and all frequencies ---
    if verbose:
        suffix = " (will update each SCBA iter)" if scattering_contacts else ""
        print(f"  Precomputing OBC self-energies (batched)...{suffix}")
    obc_all = []  # list of dicts, one per q-point
    H_D_all = []
    for iq, (H_00, H_01) in enumerate(btd_blocks):
        H_D = _build_device_hamiltonian(H_00, H_01, n_slabs)
        H_D_all.append(H_D)
        obc = _compute_obc_batch(omega_sq_thz2, H_00, H_01, eta,
                                 n_bose_L, n_bose_R, n_slabs=n_slabs)
        obc_all.append(obc)

    # --- Ballistic transmission ---
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
    # Integrate only over positive frequencies (physical spectrum)
    J_ball_total = np.sum(spectral_J_ball[pos_mask]) * dw_thz * 1e12
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
    best_conservation = 1.0
    best_state = None  # will hold spectral currents at min conservation

    # Anderson mixing history
    _anderson_x_hist = []
    _anderson_f_hist = []

    spectral_J_L = np.zeros(nfreq)
    spectral_J_R = np.zeros(nfreq)

    sl0 = slice(0, n_dof)
    sl_last = slice((n_slabs - 1) * n_dof, n_slabs * n_dof)

    for scba_iter in range(max_scba_iter):
        # Update OBC with scattering in the leads
        if scattering_contacts and scba_iter > 0:
            for iq, (H_00_iq, H_01_iq) in enumerate(btd_blocks):
                obc_try = _compute_obc_batch(
                    omega_sq_thz2, H_00_iq, H_01_iq, eta,
                    n_bose_L, n_bose_R, n_slabs=n_slabs,
                    lead_sigma_r_L=Sigma_R_q[0, iq],
                    lead_sigma_r_R=Sigma_R_q[-1, iq])
                if np.any(np.isnan(obc_try["Sigma_L_R"])):
                    if verbose:
                        print(f"    WARNING: scattering-contact OBC "
                              f"diverged for iq={iq}, using ballistic")
                else:
                    obc_all[iq] = obc_try

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

        # Integrate only over positive frequencies (physical spectrum)
        J_L_total = np.sum(spectral_J_L[pos_mask]) * dw_thz * 1e12
        J_R_total = np.sum(spectral_J_R[pos_mask]) * dw_thz * 1e12
        J_total = 0.5 * (J_L_total + J_R_total)
        J_denom = abs(J_L_total) + abs(J_R_total)
        conservation_err = (abs(J_L_total - J_R_total) / J_denom
                            if J_denom > 0 else 0.0)

        # Track best-conservation state (SCBA fixed point may overshoot)
        # Skip first 3 iterations — conservation is trivially good before
        # self-energy has been mixed in properly
        if scba_iter >= 200 and conservation_err < best_conservation:
            best_conservation = conservation_err
            best_state = {
                "spectral_J_L": spectral_J_L.copy(),
                "spectral_J_R": spectral_J_R.copy(),
                "conservation_err": conservation_err,
                "iter": scba_iter + 1,
            }

        # Compute per-slab self-energy via dense q-dependent kernel
        Sigma_l_new = np.zeros_like(Sigma_l_q)
        Sigma_g_new = np.zeros_like(Sigma_g_q)
        Sigma_r_new = np.zeros_like(Sigma_R_q)

        for l in range(n_slabs):
            sl_n, sg_n, sr_n = _compute_phph_self_energy_q_dense(
                G_lesser_slab_q[l], G_greater_slab_q[l],
                M_stacked, T_all_q, q_diff_map,
                n_atoms, n_kpts, freqs_thz, dw_thz,
                hilbert_retarded=hilbert_retarded,
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

        # Mix self-energies
        if scba_iter == 0:
            Sigma_l_q = Sigma_l_new.copy()
            Sigma_g_q = Sigma_g_new.copy()
            Sigma_R_q = Sigma_r_new.copy()
        elif anderson_mixing and scba_iter >= 1:
            # Anderson acceleration: quasi-Newton using residual history.
            x_in = np.concatenate([
                Sigma_l_q.ravel(), Sigma_g_q.ravel(), Sigma_R_q.ravel()])
            x_out = np.concatenate([
                Sigma_l_new.ravel(), Sigma_g_new.ravel(), Sigma_r_new.ravel()])
            f_k = x_out - x_in
            _anderson_x_hist.append(x_in)
            _anderson_f_hist.append(f_k)

            m = len(_anderson_f_hist)
            if m >= 2:
                # Use up to anderson_depth history pairs
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
                    FtF + reg * np.eye(FtF.shape[0]), Ftf).real
                x_mixed = (x_in + mixing * f_k) - (dX + mixing * dF) @ gamma
            else:
                # First iteration: plain linear mixing
                x_mixed = x_in + mixing * f_k

            sz = Sigma_l_q.size
            Sigma_l_q = x_mixed[:sz].reshape(Sigma_l_q.shape)
            Sigma_g_q = x_mixed[sz:2*sz].reshape(Sigma_g_q.shape)
            Sigma_R_q = x_mixed[2*sz:].reshape(Sigma_R_q.shape)

            # Keep history bounded
            if len(_anderson_x_hist) > anderson_depth + 2:
                _anderson_x_hist.pop(0)
                _anderson_f_hist.pop(0)
        else:
            # Plain linear mixing
            alpha = mixing
            Sigma_l_q = (1 - alpha) * Sigma_l_q + alpha * Sigma_l_new
            Sigma_g_q = (1 - alpha) * Sigma_g_q + alpha * Sigma_g_new
            Sigma_R_q = (1 - alpha) * Sigma_R_q + alpha * Sigma_r_new

        # Convergence: stop when conservation passes through minimum
        # and starts rising, or when rel_change is small enough
        if scba_iter > 0:
            rel_change = abs(J_total - J_total_prev) / (abs(J_total_prev) + 1e-30)
            convergence_history.append(rel_change)
            if verbose:
                best_mark = " *" if conservation_err <= best_conservation else ""
                print(f"    SCBA iter {scba_iter + 1}: "
                      f"J = {J_total:.4e} W, "
                      f"conservation = {conservation_err:.4e}, "
                      f"rel. change = {rel_change:.4e}, "
                      f"max|Sigma^R| = {np.max(np.abs(Sigma_R_q)):.2e} THz^2"
                      f"{best_mark}")
            # Stop when conservation starts rising above 2x its minimum
            # (the minimum has been passed, further iterations make it worse)
            if (best_conservation < 0.5
                    and conservation_err > 2 * best_conservation
                    and scba_iter >= 200):
                if verbose:
                    print(f"    Stopping: conservation rising "
                          f"({conservation_err:.2e} > 2 * best {best_conservation:.2e}). "
                          f"Using best state from iter {best_state['iter']}.")
                break
            if rel_change < scba_tol:
                if verbose:
                    print(f"    Converged after {scba_iter + 1} iterations "
                          f"(rel_change={rel_change:.2e}, "
                          f"conservation={conservation_err:.2e})")
                break
        else:
            if verbose:
                print(f"    SCBA iter 1: "
                      f"J_L = {J_L_total:.4e} W, J_R = {J_R_total:.4e} W")

        J_total_prev = J_total

    # Use best-conservation state if it's significantly better than final
    if (best_state is not None
            and best_state["conservation_err"] < 0.5 * conservation_err):
        spectral_J_L = best_state["spectral_J_L"]
        spectral_J_R = best_state["spectral_J_R"]
        conservation_err = best_state["conservation_err"]
        if verbose:
            print(f"  Using best-conservation state from iter {best_state['iter']} "
                  f"(conservation={conservation_err:.4e})")

    # Final results — integrate positive frequencies only
    spectral_J_anh = 0.5 * (spectral_J_L + spectral_J_R)
    J_anh_total = np.sum(spectral_J_anh[pos_mask]) * dw_thz * 1e12
    G_anh = J_anh_total / (A_c * delta_T)

    if verbose:
        print(f"  Anharmonic thermal conductance: {G_anh:.2f} W/(m^2 K)")
        print(f"  Heat flow conservation: {conservation_err:.4e}")

    # Return positive-frequency side only
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
# Finite-device (Gamma-point) self-energy and transport
# ---------------------------------------------------------------------------


def _compute_phph_self_energy_finite(
    G_lesser, G_greater, Phi, omega_grid_thz, dw_thz,
    hilbert_retarded=False,
):
    """Phonon-phonon self-energy for a finite device (no transverse q).

    Same bubble diagram as the q-dependent version but evaluated at the
    Gamma point only, appropriate for devices that are finite in the
    transverse directions.

    Parameters
    ----------
    G_lesser, G_greater : ndarray, shape (n_freq, n_dof, n_dof)
        Slab-diagonal lesser/greater Green's functions.
    Phi : ndarray, shape (n_dof, n_dof, n_dof)
        Mass-weighted FC3 tensor at Gamma: Phi[a, c, d].
    omega_grid_thz : ndarray, shape (n_freq,)
    dw_thz : float
    hilbert_retarded : bool

    Returns
    -------
    Sigma_lesser, Sigma_greater, Sigma_retarded : (n_freq, n_dof, n_dof)
    """
    n_freq = len(omega_grid_thz)
    nd = Phi.shape[0]
    nd2 = nd * nd

    # Linear convolution via zero-padded FFT (see q-dense version for details).
    n_fft = 2 * n_freq - 1
    mid = (n_freq - 1) // 2
    freq_sl = slice(mid, mid + n_freq)

    prefactor = 0.5j * HBAR_SI * dw_thz / (2 * np.pi)

    # Pad + FFT Green's functions: (n_fft, nd, nd)
    G_pad = np.zeros((n_fft, nd, nd), dtype=complex)
    G_pad[:n_freq] = G_lesser
    GL_fft = np.fft.fft(G_pad, axis=0)
    G_pad[:] = 0
    G_pad[:n_freq] = G_greater
    GG_fft = np.fft.fft(G_pad, axis=0)
    del G_pad

    PL_flat = Phi.reshape(nd2, nd)          # Phi[ac, d]
    PR_flat = Phi.conj().reshape(nd, nd2)   # Phi*[b, ef]

    sig_l = np.zeros((n_freq, nd, nd), dtype=complex)
    sig_g = np.zeros_like(sig_l)

    for G_fft, sig_out in [(GL_fft, sig_l), (GG_fft, sig_g)]:
        # Step 1: A[W,ac,e] = PL[ac,d] @ G[W,d,e]
        A = PL_flat[None] @ G_fft                       # (nfft, nd2, nd)
        A = A.reshape(n_fft, nd, nd, nd)                 # (nfft, a, c, e)

        # Step 2: B[W,a,e,f] = A[W,a,e,c] @ G[W,c,f]
        A = A.transpose(0, 1, 3, 2)                      # (nfft, a, e, c)
        B = A @ G_fft[:, None, :, :]                     # (nfft, a, e, f)

        # Step 3: S[W,a,b] = sum_{ef} B[W,a,ef] @ PR[ef,b]
        S = B.reshape(n_fft * nd, nd2) @ PR_flat.T       # (nfft*nd, nd)
        sig_out[:] = prefactor * np.fft.ifft(
            S.reshape(n_fft, nd, nd), axis=0)[freq_sl]

    delta = sig_g - sig_l
    if hilbert_retarded:
        sig_r = 0.5 * delta + 0.5j * _hilbert_transform_axis(delta, axis=0)
    else:
        sig_r = 0.5 * delta
    return sig_l, sig_g, sig_r


def anharmonic_transmission_finite(
    phonon,
    fc3_hdf5: str = None,
    freq_range_thz: tuple[float, float, int] = (0.01, 16.0, 101),
    transport_direction: str = "x",
    eta_factor: float = 0.5,
    temperature: float = 300.0,
    delta_T: float = 10.0,
    max_scba_iter: int = 10,
    scba_tol: float = 0.01,
    mixing: float = 0.5,
    anderson_mixing: bool = False,
    anderson_depth: int = 5,
    n_slabs: int = 1,
    verbose: bool = True,
    M_stacked_override: np.ndarray = None,
    hilbert_retarded: bool = False,
    scattering_contacts: bool = False,
) -> dict:
    """Anharmonic phonon transport for a finite device (no transverse q).

    Equivalent to ``anharmonic_transmission_q`` with a 1x1 q-mesh (Gamma
    point only), but avoids all q-space machinery for clarity and lower
    overhead.

    Parameters
    ----------
    phonon : Phonopy
    fc3_hdf5 : str or Path
        Path to raw fc3.hdf5 from phono3py.
    freq_range_thz : (_, fmax, nfreq_pos)
        See ``anharmonic_transmission_q`` for details.
    transport_direction : str
    eta_factor, temperature, delta_T, max_scba_iter, scba_tol, mixing : float
    anderson_mixing : bool
    anderson_depth : int
    n_slabs : int
    verbose : bool
    M_stacked_override : ndarray, optional
    hilbert_retarded : bool
        Use Hilbert-transform retarded self-energy.
    scattering_contacts : bool
        Include device-boundary scattering self-energy in the lead
        surface Green's functions.  Falls back to ballistic OBC if
        Sancho-Rubio diverges.

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
    )

    # --- Setup: symmetric frequency grid [-fmax, ..., 0, ..., fmax] ---
    _fmin, fmax, nfreq_pos = freq_range_thz
    nfreq_pos = int(nfreq_pos)
    freqs_pos = np.linspace(0.0, fmax, nfreq_pos)
    freqs_thz = np.concatenate((-freqs_pos[:0:-1], freqs_pos))
    nfreq = len(freqs_thz)
    dw_thz = freqs_pos[1] - freqs_pos[0]
    omega_sq_thz2 = freqs_thz ** 2
    eta = dw_thz ** 2 * eta_factor
    pos_mask = freqs_thz >= 0.0

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

    # --- Phi at Gamma: Phi[a,c,d] = T(0) @ M_a @ T(0)^T ---
    T0 = build_gathering_matrix(
        prim_indices, cell_frac,
        (0.0, 0.0), n_atoms, transport_direction,
    )
    M_blocks = M_stacked.reshape(n_dof, dim_sc, dim_sc)
    Phi = np.einsum('ci,aij,dj->acd', T0, M_blocks, T0.conj())

    if verbose:
        if M_stacked_override is None:
            print(f"  FC3 raw shape: {fc3_raw.shape}")
        print(f"  Supercell atoms: {n_super}, dim_sc: {dim_sc}")
        print(f"  M_stacked norm: {np.linalg.norm(M_stacked):.4e}")
        print(f"  Phi norm: {np.linalg.norm(Phi):.4e}")
        print(f"  Device: {n_slabs} slab(s), {N_D} DOFs (finite, Gamma only)")

    # --- Bose-Einstein (SI units, expm1 for numerical stability) ---
    def bose_einstein(freq_thz_arr, T):
        omega_rad_s = np.abs(freq_thz_arr) * THZ_TO_RAD
        x = HBAR_SI * omega_rad_s / (KB_SI * T)
        n = np.zeros_like(x)
        valid = x > 1e-12
        n[valid] = 1.0 / np.expm1(x[valid])
        return n

    T_L = temperature + delta_T / 2.0
    T_R = temperature - delta_T / 2.0
    n_bose_L = bose_einstein(freqs_thz, T_L)
    n_bose_R = bose_einstein(freqs_thz, T_R)

    if verbose:
        print(f"  Frequency grid: {nfreq} points ({nfreq_pos} positive), "
              f"{freqs_thz[0]:.2f} to {freqs_thz[-1]:.2f} THz")
        print(f"  Temperature: {temperature} K, delta_T: {delta_T} K")
        print(f"  eta = {eta:.4e} THz^2")
        mix_str = (f"Anderson(depth={anderson_depth})" if anderson_mixing
                   else "linear")
        print(f"  SCBA: max {max_scba_iter} iter, tol={scba_tol}, "
              f"mix={mixing}, method={mix_str}")
        if hilbert_retarded:
            print("  Retarded SE: Hilbert transform (Kramers-Kronig)")

    # --- BTD blocks at Gamma ---
    H_00, H_01 = get_btd_blocks(
        phonon, (0.0, 0.0), transport_direction=transport_direction,
        conversion_factor=CONVERSION_THZ2,
    )
    H_D = _build_device_hamiltonian(H_00, H_01, n_slabs)

    # --- OBC self-energies (batched over frequency) ---
    if verbose:
        suffix = " (will update each SCBA iter)" if scattering_contacts else ""
        print(f"  Precomputing OBC self-energies (batched)...{suffix}")
    obc = _compute_obc_batch(omega_sq_thz2, H_00, H_01, eta,
                             n_bose_L, n_bose_R, n_slabs=n_slabs)

    # --- Ballistic transmission ---
    H_LD = np.zeros((n_dof, N_D), dtype=complex)
    H_LD[:, :n_dof] = H_01
    H_DR = np.zeros((N_D, n_dof), dtype=complex)
    H_DR[-n_dof:, :] = H_01
    trans_ballistic = np.zeros(nfreq)
    for iw, w2 in enumerate(omega_sq_thz2):
        trans_ballistic[iw] = _ballistic_transmission(
            w2, H_D, H_00, H_01, H_00, H_01, H_LD, H_DR, eta=eta
        )

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
    # Integrate only over positive frequencies (physical spectrum)
    J_ball_total = np.sum(spectral_J_ball[pos_mask]) * dw_thz * 1e12
    G_ball = J_ball_total / (A_c * delta_T)

    if verbose:
        print(f"  Ballistic thermal conductance: {G_ball:.2f} W/(m^2 K)")

    # --- SCBA ---
    Sigma_R = np.zeros((n_slabs, nfreq, n_dof, n_dof), dtype=complex)
    Sigma_l = np.zeros_like(Sigma_R)
    Sigma_g = np.zeros_like(Sigma_R)

    convergence_history = []
    J_total_prev = 0.0
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
            obc_try = _compute_obc_batch(
                omega_sq_thz2, H_00, H_01, eta,
                n_bose_L, n_bose_R, n_slabs=n_slabs,
                lead_sigma_r_L=Sigma_R[0],
                lead_sigma_r_R=Sigma_R[-1])
            if np.any(np.isnan(obc_try["Sigma_L_R"])):
                if verbose:
                    print("    WARNING: scattering-contact OBC "
                          "diverged, using ballistic")
            else:
                obc = obc_try

        G_lesser_slab = np.zeros((n_slabs, nfreq, n_dof, n_dof), dtype=complex)
        G_greater_slab = np.zeros_like(G_lesser_slab)
        spectral_J_L[:] = 0.0
        spectral_J_R[:] = 0.0

        # --- Green's function solve ---
        Sig_R_dev = np.zeros((nfreq, N_D, N_D), dtype=complex)
        Sig_l_dev = np.zeros_like(Sig_R_dev)
        Sig_g_dev = np.zeros_like(Sig_R_dev)
        for l in range(n_slabs):
            sl = slice(l * n_dof, (l + 1) * n_dof)
            Sig_R_dev[:, sl, sl] = Sigma_R[l]
            Sig_l_dev[:, sl, sl] = Sigma_l[l]
            Sig_g_dev[:, sl, sl] = Sigma_g[l]

        _, G_less, G_great = _solve_green_batch(
            omega_sq_thz2, H_D, obc, Sig_R_dev, Sig_l_dev, Sig_g_dev, eta)

        for l in range(n_slabs):
            sl = slice(l * n_dof, (l + 1) * n_dof)
            G_lesser_slab[l] = G_less[:, sl, sl]
            G_greater_slab[l] = G_great[:, sl, sl]

        # Spectral current from contacts
        SLg_Gl = obc["Sigma_L_greater"][:, sl0, sl0] @ G_less[:, sl0, sl0]
        SLl_Gg = obc["Sigma_L_lesser"][:, sl0, sl0] @ G_great[:, sl0, sl0]
        spectral_J_L = HBAR_SI * omega_rad * np.real(
            np.trace(SLg_Gl - SLl_Gg, axis1=-2, axis2=-1))

        SRl_Gg = obc["Sigma_R_lesser"][:, sl_last, sl_last] @ G_great[:, sl_last, sl_last]
        SRg_Gl = obc["Sigma_R_greater"][:, sl_last, sl_last] @ G_less[:, sl_last, sl_last]
        spectral_J_R = HBAR_SI * omega_rad * np.real(
            np.trace(SRl_Gg - SRg_Gl, axis1=-2, axis2=-1))

        # Integrate only over positive frequencies (physical spectrum)
        J_L_total = np.sum(spectral_J_L[pos_mask]) * dw_thz * 1e12
        J_R_total = np.sum(spectral_J_R[pos_mask]) * dw_thz * 1e12
        J_total = 0.5 * (J_L_total + J_R_total)
        J_denom = abs(J_L_total) + abs(J_R_total)
        conservation_err = (abs(J_L_total - J_R_total) / J_denom
                            if J_denom > 0 else 0.0)

        if scba_iter >= 3 and conservation_err < best_conservation:
            best_conservation = conservation_err
            best_state = {
                "spectral_J_L": spectral_J_L.copy(),
                "spectral_J_R": spectral_J_R.copy(),
                "conservation_err": conservation_err,
                "iter": scba_iter + 1,
            }

        # Compute per-slab self-energy (finite device, no q)
        Sigma_l_new = np.zeros_like(Sigma_l)
        Sigma_g_new = np.zeros_like(Sigma_g)
        Sigma_r_new = np.zeros_like(Sigma_R)

        for l in range(n_slabs):
            sl_n, sg_n, sr_n = _compute_phph_self_energy_finite(
                G_lesser_slab[l], G_greater_slab[l],
                Phi, freqs_thz, dw_thz,
                hilbert_retarded=hilbert_retarded,
            )
            Sigma_l_new[l] = sl_n
            Sigma_g_new[l] = sg_n
            Sigma_r_new[l] = sr_n

        sig_r_norm = np.max(np.abs(Sigma_r_new))

        if verbose and scba_iter == 0:
            gl_max = np.max(np.abs(G_lesser_slab))
            h00_max = np.max(np.abs(H_00))
            print(f"    G diagnostic: max|G^<| = {gl_max:.4e}")
            print(f"    Self-energy: max|Sigma^R| = {sig_r_norm:.4e} THz^2, "
                  f"|Sigma^R|/|H_00| = {sig_r_norm / h00_max:.4e}")

        # Mix self-energies
        if scba_iter == 0:
            Sigma_l = Sigma_l_new.copy()
            Sigma_g = Sigma_g_new.copy()
            Sigma_R = Sigma_r_new.copy()
        elif anderson_mixing and scba_iter >= 1:
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
                    FtF + reg * np.eye(FtF.shape[0]), Ftf).real
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

        if scba_iter > 0:
            rel_change = abs(J_total - J_total_prev) / (abs(J_total_prev) + 1e-30)
            convergence_history.append(rel_change)
            if verbose:
                best_mark = " *" if conservation_err <= best_conservation else ""
                print(f"    SCBA iter {scba_iter + 1}: "
                      f"J = {J_total:.4e} W, "
                      f"conservation = {conservation_err:.4e}, "
                      f"rel. change = {rel_change:.4e}, "
                      f"max|Sigma^R| = {np.max(np.abs(Sigma_R)):.2e} THz^2"
                      f"{best_mark}")
            if (best_conservation < 0.5
                    and conservation_err > 2 * best_conservation
                    and scba_iter >= 5):
                if verbose:
                    print(f"    Stopping: conservation rising "
                          f"({conservation_err:.2e} > 2 * best "
                          f"{best_conservation:.2e}). "
                          f"Using best state from iter {best_state['iter']}.")
                break
            if rel_change < scba_tol:
                if verbose:
                    print(f"    Converged after {scba_iter + 1} iterations "
                          f"(rel_change={rel_change:.2e}, "
                          f"conservation={conservation_err:.2e})")
                break
        else:
            if verbose:
                print(f"    SCBA iter 1: "
                      f"J_L = {J_L_total:.4e} W, J_R = {J_R_total:.4e} W")

        J_total_prev = J_total

    # Use best-conservation state if significantly better than final
    if (best_state is not None
            and best_state["conservation_err"] < 0.5 * conservation_err):
        spectral_J_L = best_state["spectral_J_L"]
        spectral_J_R = best_state["spectral_J_R"]
        conservation_err = best_state["conservation_err"]
        if verbose:
            print(f"  Using best-conservation state from iter "
                  f"{best_state['iter']} "
                  f"(conservation={conservation_err:.4e})")

    # Final results — integrate positive frequencies only
    spectral_J_anh = 0.5 * (spectral_J_L + spectral_J_R)
    J_anh_total = np.sum(spectral_J_anh[pos_mask]) * dw_thz * 1e12
    G_anh = J_anh_total / (A_c * delta_T)

    if verbose:
        print(f"  Anharmonic thermal conductance: {G_anh:.2f} W/(m^2 K)")
        print(f"  Heat flow conservation: {conservation_err:.4e}")

    # Return positive-frequency side only
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
