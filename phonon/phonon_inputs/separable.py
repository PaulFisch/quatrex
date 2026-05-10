"""Separable FC3 approximation for phonon-phonon self-energy.

1. Decomposes the FC3 in supercell real space via stacked SVD.

2. Fourier-transforms the low-rank factors to q-space.

3. Computes a q-DEPENDENT self-energy via the separable bilinear
   forms with FFT (omega) and explicit sum (q-convolution).

Stacked SVD:  For each (i, mu) in the primitive cell, form the
mass-weighted matrix fc3[i, :, :, mu, :, :] using all supercell
atoms, then stack all slices:

    M_stacked = [M_{0,x}; M_{0,y}; ...; M_{nat,z}]
    M_stacked approx= U_R diag(sigma) V_R^T

This gives a SHARED right factor h_r = sigma_r * V_r (column of dim_sc)
and per-DOF left factors f_r^a = U[(a*dim_sc):(a+1)*dim_sc, r].

The self-energy:

    Sigma_{ab}(w, q) = (i*hbar / 2*N_q) sum_{q',r,s}
        P^a_{rs}(w', q')  conv_w  Q^b_{rs}(w'', q-q')

with two-stage evaluation:

    v_s(w, q) = G(w, q) @ conj(h_hat_s(q))       [shared, R matvecs]
    u_r(w, q) = G(w, q)^T @ h_hat_r(q)            [shared, R matvecs]
    P^a_{rs}(w, q) = (F_hat_r(q) @ v_s(w, q))[a]  [R^2 matvecs]
    Q^b_{rs}(w, q) = (conj(F_hat_s(q)) @ u_r)[b]  [R^2 matvecs]

Convolution over omega via FFT; sum over q' explicit (small mesh).
"""

import numpy as np
from numpy.linalg import inv

from .constants import (
    CONVERSION_FC3_THZ,
    CONVERSION_THZ2,
    HBAR_SI,
    KB_SI,
    THZ_TO_RAD,
)
from .anharmonic import (
    _build_device_hamiltonian,
    _compute_obc_self_energies,
    _solve_green_functions,
)


# ---------------------------------------------------------------------------
# Supercell mapping
# ---------------------------------------------------------------------------


def build_supercell_mapping(phonon, transport_direction="x"):
    """Map supercell atoms to (primitive atom, cell translation).

    Parameters
    ----------
    phonon : Phonopy
    transport_direction : str

    Returns
    -------
    prim_indices : ndarray, shape (n_super,)
        Which primitive atom each supercell atom corresponds to.
    cell_frac : ndarray, shape (n_super, 3)
        Cell translation in fractional coordinates of the primitive cell.
    slab_indices : ndarray, shape (n_super,)
        Integer slab index (cell_frac component along transport).
    """
    prim_cell = phonon.primitive.cell           # (3, 3), rows are lattice vecs
    prim_pos = phonon.primitive.positions        # (nat_prim, 3) Cartesian
    sc_pos = phonon.supercell.positions          # (n_super, 3) Cartesian
    nat_prim = len(prim_pos)
    n_super = len(sc_pos)
    tidx = "xyz".index(transport_direction)

    inv_cell = np.linalg.inv(prim_cell.T)

    prim_indices = np.zeros(n_super, dtype=int)
    cell_frac = np.zeros((n_super, 3))

    for s in range(n_super):
        found = False
        for a in range(nat_prim):
            diff = sc_pos[s] - prim_pos[a]
            frac = inv_cell @ diff
            R_int = np.round(frac)
            err = np.linalg.norm(diff - prim_cell.T @ R_int)
            if err < 0.1:
                prim_indices[s] = a
                cell_frac[s] = R_int
                found = True
                break
        if not found:
            raise ValueError(f"Cannot map supercell atom {s}")

    slab_indices = np.round(cell_frac[:, tidx]).astype(int)

    # ref_sc_atoms[a] = supercell atom index for primitive atom a at cell (0,0,0)
    ref_sc_atoms = np.full(nat_prim, -1, dtype=int)
    for s in range(n_super):
        if np.allclose(cell_frac[s], 0.0):
            ref_sc_atoms[prim_indices[s]] = s
    assert np.all(ref_sc_atoms >= 0), "Not all primitive atoms found at cell (0,0,0)"

    return prim_indices, cell_frac, slab_indices, ref_sc_atoms


# ---------------------------------------------------------------------------
# Real-space FC3 matrices and Fourier transform utilities
# ---------------------------------------------------------------------------


def build_realspace_fc3_matrices(fc3_raw, nat_prim, masses_super,
                                 ref_sc_atoms):
    """Build mass-weighted FC3 matrices in real space.

    For each DOF a = (i_prim, alpha), builds M_a of shape (dim_sc, dim_sc)
    from fc3_raw using all supercell atoms (no same-slab restriction).

    Parameters
    ----------
    fc3_raw : ndarray, shape (n_super, n_super, n_super, 3, 3, 3)
        or compact (nat_prim, n_super, n_super, 3, 3, 3).
    nat_prim : int
    masses_super : ndarray, shape (n_super,)
    ref_sc_atoms : ndarray, shape (nat_prim,)

    Returns
    -------
    M_stacked : ndarray, shape (n_dof * dim_sc, dim_sc)
        M_stacked[a*dim_sc:(a+1)*dim_sc, :] = M_a.
        dim_sc = n_super * 3.
    """
    is_compact = fc3_raw.shape[0] == nat_prim
    n_dof = nat_prim * 3
    n_super = fc3_raw.shape[1]
    dim_sc = n_super * 3

    m_all = np.repeat(np.sqrt(masses_super), 3)  # (dim_sc,)

    M_stacked = np.zeros((n_dof * dim_sc, dim_sc))

    for i_prim in range(nat_prim):
        s_i = ref_sc_atoms[i_prim]
        fc3_idx = i_prim if is_compact else s_i
        m_i = masses_super[s_i]
        for alpha in range(3):
            a = 3 * i_prim + alpha
            block = fc3_raw[fc3_idx, :, :, alpha, :, :]
            mat = block.transpose(0, 2, 1, 3).reshape(dim_sc, dim_sc)
            mat = mat / (np.sqrt(m_i) * m_all[:, None] * m_all[None, :])
            mat *= CONVERSION_FC3_THZ
            M_stacked[a * dim_sc:(a + 1) * dim_sc, :] = mat

    return M_stacked


def enforce_asr_fc3_matrices(M_stacked, nat_prim, prim_indices):
    """Enforce acoustic sum rule on FC3 matrices via projection.

    For each M_a (dim_sc x dim_sc), removes the component that couples
    to uniform translations by projecting out the null space of T(Gamma).

    The ASR requires sum_{l,b} Phi(0b1, lb2, l'b3) = 0, which in
    Fourier space means Phi(q=0, q') = 0.  This is equivalent to
    T(Gamma) @ M_a = 0 and M_a @ T(Gamma)^T = 0.

    Parameters
    ----------
    M_stacked : ndarray, shape (n_dof * dim_sc, dim_sc)
    nat_prim : int
    prim_indices : ndarray, shape (n_super,)

    Returns
    -------
    M_corrected : ndarray, same shape as M_stacked
    """
    n_dof = nat_prim * 3
    n_super = len(prim_indices)
    dim_sc = n_super * 3

    counts = np.zeros(nat_prim)
    for s in range(n_super):
        counts[prim_indices[s]] += 1

    P = np.zeros((dim_sc, n_dof))
    for s in range(n_super):
        kappa = prim_indices[s]
        w = 1.0 / np.sqrt(counts[kappa])
        for beta in range(3):
            P[s * 3 + beta, kappa * 3 + beta] = w

    PPt = P @ P.T  # (dim_sc, dim_sc)
    I_minus_PPt = np.eye(dim_sc) - PPt

    M_corrected = np.zeros_like(M_stacked)
    for a in range(n_dof):
        M_a = M_stacked[a * dim_sc:(a + 1) * dim_sc, :]
        M_corrected[a * dim_sc:(a + 1) * dim_sc, :] = I_minus_PPt @ M_a @ I_minus_PPt

    return M_corrected


def build_gathering_matrix(prim_indices, cell_frac, q_frac,
                           nat_prim, transport_direction):
    """Build the gathering matrix T(q) for FC3 Fourier transform.

    T[kappa*3+beta, s*3+beta] = exp(-2*pi*i * q . R_frac[s])
    when prim_indices[s] == kappa, else 0.

    Parameters
    ----------
    prim_indices : ndarray, shape (n_super,)
    cell_frac : ndarray, shape (n_super, 3)
    q_frac : tuple (qx, qy)
    nat_prim : int
    transport_direction : str

    Returns
    -------
    T : ndarray, shape (n_dof, dim_sc), complex
        dim_sc = n_super * 3.
    """
    tidx = "xyz".index(transport_direction)
    n_super = len(prim_indices)
    n_dof = nat_prim * 3
    dim_sc = n_super * 3
    perp_idx = [i for i in range(3) if i != tidx]
    q_full = np.zeros(3)
    q_full[perp_idx[0]] = q_frac[0]
    q_full[perp_idx[1]] = q_frac[1]

    phases = np.exp(-2j * np.pi * cell_frac @ q_full)

    T = np.zeros((n_dof, dim_sc), dtype=complex)
    for s in range(n_super):
        kappa = prim_indices[s]
        phase = phases[s]
        for beta in range(3):
            T[kappa * 3 + beta, s * 3 + beta] = phase

    return T


def build_q_diff_map(nkx, nky):
    """Build lookup for q-difference indices on a 2D Gamma-centered mesh.

    For q_{ix,iy} indexed as iq = ix*nky + iy, returns
    q_diff_map[iq, iq'] = index of (q_iq - q_iq') on the mesh.

    Returns
    -------
    q_diff_map : ndarray, shape (nkx*nky, nkx*nky), int
    """
    n_kpts = nkx * nky
    q_diff_map = np.zeros((n_kpts, n_kpts), dtype=int)
    for iq in range(n_kpts):
        ix, iy = divmod(iq, nky)
        for iq_prime in range(n_kpts):
            ix_p, iy_p = divmod(iq_prime, nky)
            q_diff_map[iq, iq_prime] = ((ix - ix_p) % nkx) * nky + (iy - iy_p) % nky
    return q_diff_map


# ---------------------------------------------------------------------------
# Stacked SVD decomposition
# ---------------------------------------------------------------------------


def decompose_fc3_supercell(
    fc3_raw: np.ndarray,
    nat_prim: int,
    masses_super: np.ndarray,
    prim_indices: np.ndarray,
    ref_sc_atoms: np.ndarray,
    rank: int | None = None,
    tol: float = 1e-8,
    enforce_asr: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Low-rank decomposition of supercell FC3 with shared right factor.

    For each (i_prim, mu) in the primitive cell, forms the mass-weighted
    matrix from fc3_raw[i, :, :, mu, :, :] using all supercell atoms.
    Stacks all n_dof slices and SVDs:

        M_stacked = U diag(sigma) V^T,  shape (n_dof * dim_sc, dim_sc)

    where dim_sc = n_super * 3.

    Parameters
    ----------
    fc3_raw : ndarray, shape (n_super, n_super, n_super, 3, 3, 3)
        Raw FC3 from phono3py HDF5, or compact format.
    nat_prim : int
    masses_super : ndarray, shape (n_super,)
    prim_indices : ndarray, shape (n_super,)
    ref_sc_atoms : ndarray, shape (nat_prim,)
    rank, tol : SVD truncation control.
    enforce_asr : bool
        If True, project out uniform-translation components before SVD.

    Returns
    -------
    F_list : list of ndarray, each shape (n_dof, dim_sc)
        F_r[a, c] — left factors, one per rank component.
    H : ndarray, shape (dim_sc, R)
        Shared right factor; columns are h_r vectors.
    svals : ndarray, shape (R,)
    """
    n_dof = nat_prim * 3
    n_super = fc3_raw.shape[1]
    dim_sc = n_super * 3

    M_stacked = build_realspace_fc3_matrices(
        fc3_raw, nat_prim, masses_super, ref_sc_atoms
    )

    if enforce_asr:
        M_stacked = enforce_asr_fc3_matrices(
            M_stacked, nat_prim, prim_indices
        )

    # SVD
    U, S, Vt = np.linalg.svd(M_stacked, full_matrices=False)

    if rank is None:
        cutoff = tol * S[0] if S[0] > 0 else 0.0
        R = max(1, int(np.sum(S > cutoff)))
    else:
        R = min(rank, len(S))

    U_R = U[:, :R]
    S_R = S[:R]
    Vt_R = Vt[:R, :]

    # Absorb singular values into H (shared right factor)
    H = (S_R[:, None] * Vt_R).T   # (dim_sc, R)

    F_list = [U_R[:, r].reshape(n_dof, dim_sc) for r in range(R)]

    return F_list, H, S_R


def reconstruction_error(fc3_raw, nat_prim, masses_super,
                         ref_sc_atoms, F_list, H):
    """Relative Frobenius error of the stacked reconstruction."""
    R = H.shape[1]

    M_stacked = build_realspace_fc3_matrices(
        fc3_raw, nat_prim, masses_super, ref_sc_atoms
    )

    M_approx = np.zeros_like(M_stacked)
    for r in range(R):
        M_approx += F_list[r].reshape(-1, 1) @ H[:, r:r + 1].T

    norm = np.linalg.norm(M_stacked, "fro")
    if norm == 0:
        return 0.0
    return np.linalg.norm(M_stacked - M_approx, "fro") / norm


# ---------------------------------------------------------------------------
# Fourier transform of factors
# ---------------------------------------------------------------------------


def fourier_transform_factors(
    F_list: list[np.ndarray],
    H: np.ndarray,
    prim_indices: np.ndarray,
    cell_frac: np.ndarray,
    q_perp_frac: tuple[float, float],
    nat_prim: int,
    transport_direction: str = "x",
) -> tuple[list[np.ndarray], np.ndarray]:
    """Fourier-transform real-space factors to a transverse q-point.

    f_hat_r^a(q)[kappa, beta] = sum_{R} f_r^a(s(kappa, R), beta)
                                  * exp(-2*pi*i * q . R_frac)

    Uses ALL supercell atoms (no same-slab restriction).

    Parameters
    ----------
    F_list : list of ndarray, each (n_dof, dim_sc)
    H : ndarray, shape (dim_sc, R)
    prim_indices, cell_frac : from build_supercell_mapping.
    q_perp_frac : (qx, qy) in fractional coordinates of primitive cell.
    nat_prim : int
    transport_direction : str

    Returns
    -------
    F_hat_list : list of ndarray, each (n_dof, n_dof), complex
        F_hat_r(q) — Fourier-transformed left factor matrices.
    H_hat : ndarray, shape (n_dof, R), complex
        Fourier-transformed shared right factor.
    """
    T = build_gathering_matrix(
        prim_indices, cell_frac, q_perp_frac, nat_prim, transport_direction
    )

    # F_hat_r(q) = F_r @ T^T,  shape (n_dof, n_dof)
    F_hat_list = [F_r @ T.T for F_r in F_list]

    # H_hat(q) = T @ H,  shape (n_dof, R)
    H_hat = T @ H

    return F_hat_list, H_hat


# ---------------------------------------------------------------------------
# Separable self-energy kernel with q-convolution
# ---------------------------------------------------------------------------


def _compute_phph_self_energy_separable(
    G_lesser_q: np.ndarray,
    G_greater_q: np.ndarray,
    F_hat_q: list[list[np.ndarray]],
    H_hat_q: list[np.ndarray],
    n_dof: int,
    n_kpts: int,
    omega_grid_thz: np.ndarray,
    dw_thz: float,
    q_diff_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute q-dependent self-energy via separable FC3.

    Performs FFT convolution over omega and explicit sum over q'.

    Parameters
    ----------
    G_lesser_q : ndarray, shape (n_kpts, n_freq, n_dof, n_dof)
    G_greater_q : ndarray, shape (n_kpts, n_freq, n_dof, n_dof)
    F_hat_q : list (len n_kpts) of list (len R) of ndarray (n_dof, n_dof)
        F_hat_r(q) for each q-point and rank r.
    H_hat_q : list (len n_kpts) of ndarray (n_dof, R)
        h_hat_r(q) for each q-point.
    n_dof, n_kpts : int
    omega_grid_thz : ndarray, shape (n_freq,)
    dw_thz : float
    q_diff_map : ndarray, shape (n_kpts, n_kpts), optional
        Lookup table for q-difference indices (from build_q_diff_map).

    Returns
    -------
    Sigma_lesser, Sigma_greater, Sigma_retarded :
        ndarray, shape (n_kpts, n_freq, n_dof, n_dof)
    """
    n_freq = len(omega_grid_thz)
    R = H_hat_q[0].shape[1]

    # Linear convolution via zero-padded FFT (symmetric grid).
    n_fft = 2 * n_freq - 1
    mid = (n_freq - 1) // 2
    freq_sl = slice(mid, mid + n_freq)

    prefactor = 0.5j * HBAR_SI * dw_thz / (2 * np.pi) / n_kpts

    Sigma_lesser = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)
    Sigma_greater = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)

    # Pad Green's functions: (n_kpts, n_fft, n_dof, n_dof)
    def _pad(G_q):
        out = np.zeros((n_kpts, n_fft, n_dof, n_dof), dtype=complex)
        out[:, :n_freq] = G_q
        return out

    GL = _pad(G_lesser_q)
    GG = _pad(G_greater_q)

    # Precompute Stage 1 products for ALL q-points:
    # v_s(w, q) = G(w, q) @ conj(h_hat_s(q)),  shape (n_fft, n_dof) per (s, q)
    # u_r(w, q) = G(w, q)^T @ h_hat_r(q),      shape (n_fft, n_dof) per (r, q)

    # For lesser:
    VL = np.zeros((n_kpts, n_fft, n_dof, R), dtype=complex)
    UL = np.zeros((n_kpts, n_fft, n_dof, R), dtype=complex)
    VG = np.zeros((n_kpts, n_fft, n_dof, R), dtype=complex)
    UG = np.zeros((n_kpts, n_fft, n_dof, R), dtype=complex)

    for iq in range(n_kpts):
        H_hat_conj = np.conj(H_hat_q[iq])         # (n_dof, R)
        H_hat = H_hat_q[iq]                        # (n_dof, R)
        VL[iq] = GL[iq] @ H_hat_conj              # (n_fft, n_dof, R)
        UL[iq] = GL[iq].transpose(0, 2, 1) @ H_hat
        VG[iq] = GG[iq] @ H_hat_conj
        UG[iq] = GG[iq].transpose(0, 2, 1) @ H_hat

    # Precompute FFTs of V, U arrays (eliminates R^2 per-element FFTs)
    VL_hat = np.fft.fft(VL, axis=1)  # (n_kpts, n_fft, n_dof, R)
    UL_hat = np.fft.fft(UL, axis=1)
    VG_hat = np.fft.fft(VG, axis=1)
    UG_hat = np.fft.fft(UG, axis=1)

    # Stack F_hat matrices for batch operations
    F_stack_all = np.array([
        [F_hat_q[iq][r] for r in range(R)] for iq in range(n_kpts)
    ])  # (n_kpts, R, n_dof, n_dof)

    # Main loop: vectorized over (r, s) ranks via einsum
    for iq_ext in range(n_kpts):
        for iq_prime in range(n_kpts):
            if q_diff_map is not None:
                iq_diff = q_diff_map[iq_ext, iq_prime]
            else:
                iq_diff = (iq_ext - iq_prime) % n_kpts

            F_qp = F_stack_all[iq_prime]              # (R, n_dof, n_dof)
            F_diff_conj = np.conj(F_stack_all[iq_diff])  # (R, n_dof, n_dof)

            # Factor the R^2 sum analytically:
            # sum_{r,s} P_hat_rs[w,a] * Q_hat_rs[w,b]
            #   = sum_{c,d} M1[w,a,c,d] * M2[w,c,b,d]
            # where:
            #   M1[w,a,c,d] = sum_r F_qp[r,a,c] * UL_hat[iq_diff,w,d,r]
            #   M2[w,c,b,d] = sum_s VL_hat[iq',w,c,s] * F_diff_conj[s,b,d]

            # --- Lesser ---
            M1 = np.einsum('rac,wdr->wacd', F_qp, UL_hat[iq_diff])
            M2 = np.einsum('wcs,sbd->wcbd', VL_hat[iq_prime], F_diff_conj)
            conv_hat = np.einsum('wacd,wcbd->wab', M1, M2)
            Sigma_lesser[iq_ext] += prefactor * np.fft.ifft(conv_hat, axis=0)[freq_sl]

            # --- Greater ---
            M1 = np.einsum('rac,wdr->wacd', F_qp, UG_hat[iq_diff])
            M2 = np.einsum('wcs,sbd->wcbd', VG_hat[iq_prime], F_diff_conj)
            conv_hat = np.einsum('wacd,wcbd->wab', M1, M2)
            Sigma_greater[iq_ext] += prefactor * np.fft.ifft(conv_hat, axis=0)[freq_sl]

    Sigma_retarded = 0.5 * (Sigma_greater - Sigma_lesser)
    return Sigma_lesser, Sigma_greater, Sigma_retarded


# ---------------------------------------------------------------------------
# SCBA driver
# ---------------------------------------------------------------------------


def separable_anharmonic_transmission(
    phonon,
    fc3_hdf5: str,
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
    rank: int | None = None,
    svd_tol: float = 1e-8,
    enforce_asr: bool = False,
    verbose: bool = True,
) -> dict:
    """Anharmonic phonon transmission via separable FC3 self-energy.

    Uses q-dependent self-energy with proper q-convolution, going
    beyond Approximation III.  The FC3 is decomposed in supercell
    real space and Fourier-transformed per q-point.

    Uses a Gamma-centered q-mesh [0, 1/N, ..., (N-1)/N] so that
    q - q' maps back onto the mesh (closure under subtraction).

    Parameters
    ----------
    phonon : Phonopy
    fc3_hdf5 : str or Path
        Path to fc3.hdf5 from phono3py.
    rank : int, optional
        SVD rank.  None for automatic (via svd_tol).
    svd_tol : float
        Relative singular value cutoff.
    enforce_asr : bool
        If True, project out uniform-translation components before SVD.

    Returns
    -------
    result : dict
    """
    import h5py
    from .convention import get_btd_blocks
    from .validation import _ballistic_transmission

    # --- Setup: symmetric frequency grid [-fmax, ..., 0, ..., fmax] ---
    _fmin, fmax, nfreq_pos = freq_range_thz
    nfreq_pos = int(nfreq_pos)
    freqs_pos = np.linspace(0.0, fmax, nfreq_pos)
    freqs_thz = np.concatenate((-freqs_pos[:0:-1], freqs_pos))
    nfreq = len(freqs_thz)
    dw_thz = freqs_pos[1] - freqs_pos[0]
    eta_w = dw_thz * eta_factor
    z2_arr = (freqs_thz + 1j * eta_w) ** 2
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

    # --- Load and decompose FC3 ---
    with h5py.File(fc3_hdf5, "r") as f:
        fc3_raw = np.array(f["fc3"])

    if verbose:
        print(f"  FC3 raw shape: {fc3_raw.shape}")

    F_list, H, svals = decompose_fc3_supercell(
        fc3_raw, n_atoms, masses_super, prim_indices,
        ref_sc_atoms, rank=rank, tol=svd_tol, enforce_asr=enforce_asr,
    )
    R = len(F_list)
    recon_err = reconstruction_error(
        fc3_raw, n_atoms, masses_super,
        ref_sc_atoms, F_list, H,
    )

    if verbose:
        print(f"  Supercell atoms: {n_super}, dim_sc: {dim_sc}")
        print(f"  SVD rank: {R}/{dim_sc}, reconstruction error: {recon_err:.2e}")
        print(f"  Singular values: {svals[:min(8, R)]}"
              + (" ..." if R > 8 else ""))
        print(f"  Device: {n_slabs} slab(s), {N_D} DOFs per q-point")

    # --- Gamma-centered q-mesh (closed under subtraction for q-convolution) ---
    nkx, nky = q_mesh_transverse
    q_1d_x = [i / nkx for i in range(nkx)]
    q_1d_y = [j / nky for j in range(nky)]
    q_points = [(qx, qy) for qx in q_1d_x for qy in q_1d_y]
    n_kpts = len(q_points)
    q_diff_map = build_q_diff_map(nkx, nky)

    # Fourier-transform FC3 factors for each q-point
    F_hat_q = []  # F_hat_q[iq][r] is (n_dof, n_dof)
    H_hat_q = []  # H_hat_q[iq] is (n_dof, R)
    for qx, qy in q_points:
        Fh, Hh = fourier_transform_factors(
            F_list, H, prim_indices, cell_frac,
            (qx, qy), n_atoms, transport_direction,
        )
        F_hat_q.append(Fh)
        H_hat_q.append(Hh)

    # --- Bose-Einstein (SI units, expm1 for numerical stability) ---
    def bose_einstein(freq_thz_arr, T):
        omega_rad_s = freq_thz_arr * THZ_TO_RAD  # signed
        x = HBAR_SI * omega_rad_s / (KB_SI * T)
        n = np.zeros_like(x)
        valid = np.abs(x) > 1e-12
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
        print(f"  eta_w = {eta_w:.4e} THz")
        print(f"  SCBA: max {max_scba_iter} iter, tol={scba_tol}, mix={mixing}")

    # --- BTD blocks per q-point ---
    btd_blocks = []
    for qx, qy in q_points:
        H_00, H_01 = get_btd_blocks(
            phonon, (qx, qy), transport_direction=transport_direction,
            conversion_factor=CONVERSION_THZ2,
        )
        btd_blocks.append((H_00, H_01))

    # --- Ballistic transmission ---
    trans_ballistic = np.zeros(nfreq)
    for iq, (H_00, H_01) in enumerate(btd_blocks):
        H_D = _build_device_hamiltonian(H_00, H_01, n_slabs)
        H_LD = np.zeros((n_dof, N_D), dtype=complex)
        H_LD[:, :n_dof] = H_01
        H_DR = np.zeros((N_D, n_dof), dtype=complex)
        H_DR[-n_dof:, :] = H_01
        for iw, z2 in enumerate(z2_arr):
            trans_ballistic[iw] += _ballistic_transmission(
                z2, H_D, H_00, H_01, H_00, H_01, H_LD, H_DR
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
    J_ball_total = np.sum(spectral_J_ball[pos_mask]) * dw_thz * 1e12
    G_ball = J_ball_total / (A_c * delta_T)

    if verbose:
        print(f"  Ballistic thermal conductance: {G_ball:.2f} W/(m^2 K)")

    # --- SCBA with q-dependent separable self-energy ---
    # Per-slab, per-q self-energy
    Sigma_R_q = np.zeros((n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex)
    Sigma_l_q = np.zeros_like(Sigma_R_q)
    Sigma_g_q = np.zeros_like(Sigma_R_q)

    convergence_history = []
    J_total_prev = 0.0

    spectral_J_L = np.zeros(nfreq)
    spectral_J_R = np.zeros(nfreq)

    for scba_iter in range(max_scba_iter):
        # Collect G at all (q, omega) for each slab
        G_lesser_slab_q = np.zeros(
            (n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex
        )
        G_greater_slab_q = np.zeros_like(G_lesser_slab_q)
        spectral_J_L[:] = 0.0
        spectral_J_R[:] = 0.0

        for iq, (H_00, H_01) in enumerate(btd_blocks):
            H_D = _build_device_hamiltonian(H_00, H_01, n_slabs)

            for iw, z2 in enumerate(z2_arr):
                # Embed per-slab, per-q self-energy
                Sig_R_dev = np.zeros((N_D, N_D), dtype=complex)
                Sig_l_dev = np.zeros((N_D, N_D), dtype=complex)
                Sig_g_dev = np.zeros((N_D, N_D), dtype=complex)
                for l in range(n_slabs):
                    sl = slice(l * n_dof, (l + 1) * n_dof)
                    Sig_R_dev[sl, sl] = Sigma_R_q[l, iq, iw]
                    Sig_l_dev[sl, sl] = Sigma_l_q[l, iq, iw]
                    Sig_g_dev[sl, sl] = Sigma_g_q[l, iq, iw]

                obc = _compute_obc_self_energies(
                    z2, H_00, H_01, n_bose_L[iw], n_bose_R[iw],
                    n_slabs=n_slabs,
                )
                _, G_less, G_great = _solve_green_functions(
                    z2, H_D, obc, Sig_R_dev, Sig_l_dev, Sig_g_dev,
                )

                for l in range(n_slabs):
                    sl = slice(l * n_dof, (l + 1) * n_dof)
                    G_lesser_slab_q[l, iq, iw] = G_less[sl, sl]
                    G_greater_slab_q[l, iq, iw] = G_great[sl, sl]

                # Meir-Wingreen heat current
                sl0 = slice(0, n_dof)
                spectral_J_L[iw] += HBAR_SI * omega_rad[iw] * np.real(
                    np.trace(
                        obc["Sigma_L_greater"][sl0, sl0] @ G_less[sl0, sl0]
                        - obc["Sigma_L_lesser"][sl0, sl0] @ G_great[sl0, sl0]
                    ))
                sl_last = slice((n_slabs - 1) * n_dof, n_slabs * n_dof)
                spectral_J_R[iw] += HBAR_SI * omega_rad[iw] * np.real(
                    np.trace(
                        obc["Sigma_R_lesser"][sl_last, sl_last]
                        @ G_great[sl_last, sl_last]
                        - obc["Sigma_R_greater"][sl_last, sl_last]
                        @ G_less[sl_last, sl_last]
                    ))

        spectral_J_L /= n_kpts
        spectral_J_R /= n_kpts

        J_L_total = np.sum(spectral_J_L[pos_mask]) * dw_thz * 1e12
        J_R_total = np.sum(spectral_J_R[pos_mask]) * dw_thz * 1e12
        J_total = 0.5 * (J_L_total + J_R_total)
        J_denom = abs(J_L_total) + abs(J_R_total)
        conservation_err = (abs(J_L_total - J_R_total) / J_denom
                            if J_denom > 0 else 0.0)

        # Compute per-slab self-energy via separable kernel.
        Sigma_l_new = np.zeros_like(Sigma_l_q)
        Sigma_g_new = np.zeros_like(Sigma_g_q)
        Sigma_r_new = np.zeros_like(Sigma_R_q)

        for l in range(n_slabs):
            sl_n, sg_n, sr_n = _compute_phph_self_energy_separable(
                G_lesser_slab_q[l], G_greater_slab_q[l],
                F_hat_q, H_hat_q, n_dof, n_kpts, freqs_thz, dw_thz,
                q_diff_map=q_diff_map,
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
        "svd_rank": R,
        "svd_singular_values": svals,
        "svd_reconstruction_error": recon_err,
    }
