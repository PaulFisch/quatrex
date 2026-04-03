"""Reference anharmonic phonon-phonon NEGF transport.

Implements the self-consistent Born approximation (SCBA) for
phonon-phonon scattering following Guo et al., Phys. Rev. B 102,
195412 (2020).

The scattering self-energy (Guo Eq. 8, diagonal block):

    Sigma^{<}(omega) = (i*hbar/2) * sum_{c,d,e,f} Phi3_{a,c,d}
        * integral dw'/(2*pi) G^<_{cf}(w') G^<_{de}(w-w')
        * Phi3_{b,e,f}

The self-energy uses the LOCAL (q-averaged) Green's function:
    G_local(w) = (1/N_q) * sum_q G(q, w)
This is Approximation III from Guo et al.

Internal units: THz^2 for dynamical matrices and self-energies.
This keeps magnitudes O(1)-O(100) for numerical stability, compared
to (rad/s)^2 which gives O(10^25) magnitudes.

This module provides a reference Python implementation for validation
of the quatrex GPU solver, matching the style of validation.py.
"""

import numpy as np
from numpy.linalg import inv

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


def _compute_phph_self_energy(
    G_lesser: np.ndarray,
    G_greater: np.ndarray,
    fc3_tensor: np.ndarray,
    n_dof: int,
    omega_grid_thz: np.ndarray,
    dw_thz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute phonon-phonon scattering self-energy via FFT convolution.

    Implements Guo Eq. (8) for a single slab (diagonal block only,
    Approximation III), decay channel only.

    All internal quantities in THz^2 units. The self-energy output
    is in THz^2 (consistent with the Green's functions being in 1/THz^2).

    The frequency integral (changing variables from ω to ν in THz):
        ∫ dω'/(2π) G_ω(ω') G_ω(ω-ω')
        = (1/Λ³) × ∫ dν'/(2π) G_ν(ν') G_ν(ν-ν')

    where Λ = THZ_TO_RAD. With Φ_ν = Φ_ω/Λ^{5/2}, the self-energy becomes:
        Σ_ν = (iℏ/2) × Φ_ν² × ∫ dν'/(2π) G_ν(ν') G_ν(ν-ν')

    Discretized: prefactor = (iℏ/2) × Δν/(2π), where Δν is in THz.

    Parameters
    ----------
    G_lesser : ndarray, shape (n_freq, n_dof, n_dof)
        In 1/THz^2.
    G_greater : ndarray, shape (n_freq, n_dof, n_dof)
        In 1/THz^2.
    fc3_tensor : ndarray, shape (n_dof, n_dof, n_dof)
        FC3 tensor in THz^{5/2} (mass-weighted + CONVERSION_FC3_THZ applied).
    n_dof : int
    omega_grid_thz : ndarray, shape (n_freq,)
        Frequency grid in THz.
    dw_thz : float
        Frequency spacing in THz.

    Returns
    -------
    Sigma_lesser, Sigma_greater, Sigma_retarded : ndarray, shape (n_freq, n_dof, n_dof)
        In THz^2.
    """
    n_freq = len(omega_grid_thz)
    Sigma_lesser = np.zeros((n_freq, n_dof, n_dof), dtype=complex)
    Sigma_greater = np.zeros((n_freq, n_dof, n_dof), dtype=complex)

    Phi3 = fc3_tensor

    # Pad G arrays so the frequency axis starts at w=0.
    n_low = max(0, int(np.round(omega_grid_thz[0] / dw_thz)))
    n_ext = n_low + n_freq

    G_l_pad = np.zeros((n_ext, n_dof, n_dof), dtype=complex)
    G_g_pad = np.zeros((n_ext, n_dof, n_dof), dtype=complex)
    G_l_pad[n_low:] = G_lesser
    G_g_pad[n_low:] = G_greater

    # Zero-pad for linear (non-circular) convolution
    n_fft = 2 * n_ext

    G_l_fft = np.fft.fft(G_l_pad, n=n_fft, axis=0)
    G_g_fft = np.fft.fft(G_g_pad, n=n_fft, axis=0)

    # Prefactor: (i*hbar_SI / 2) * dw_thz / (2*pi)
    # The integral is ∫ dν'/(2π) G(ν') G(ν-ν') where ν is in THz.
    # Discretizing: Σ_m × Δν/(2π). This gives self-energy in THz^2 when
    # FC3 is in THz^{5/2} and G is in 1/THz^2.
    prefactor = 0.5j * HBAR_SI * dw_thz / (2 * np.pi)

    for c in range(n_dof):
        for d in range(n_dof):
            phi_left = Phi3[:, c, d]
            if np.max(np.abs(phi_left)) < 1e-30:
                continue
            for e in range(n_dof):
                for f in range(n_dof):
                    phi_right = Phi3[:, e, f]
                    if np.max(np.abs(phi_right)) < 1e-30:
                        continue

                    conv_l = np.fft.ifft(
                        G_l_fft[:, c, f] * G_l_fft[:, d, e], n=n_fft
                    )[n_low: n_low + n_freq]
                    conv_g = np.fft.ifft(
                        G_g_fft[:, c, f] * G_g_fft[:, d, e], n=n_fft
                    )[n_low: n_low + n_freq]

                    outer = phi_left[:, None] * phi_right[None, :]
                    Sigma_lesser += (prefactor * conv_l)[:, None, None] * outer[None, :, :]
                    Sigma_greater += (prefactor * conv_g)[:, None, None] * outer[None, :, :]

    # Retarded: Sigma^R = (1/2)(Sigma^> - Sigma^<)
    Sigma_retarded = 0.5 * (Sigma_greater - Sigma_lesser)

    return Sigma_lesser, Sigma_greater, Sigma_retarded


def _assemble_fc3_dense(fc3_data, n_atoms, masses_amu):
    """Assemble mass-weighted FC3 tensor for a single slab (on-site only).

    Only includes on-site (R'=0, R''=0) triplets, corresponding to
    interactions within a single unit cell. This is Approximation (III)
    from Guo et al.

    Parameters
    ----------
    fc3_data : dict
        Output of ``force_constants.load_fc3_thirdorder()`` or
        ``force_constants.load_fc3_phono3py()``.
    n_atoms : int
        Number of atoms in the primitive cell.
    masses_amu : array-like, shape (n_atoms,)
        Atomic masses in amu.

    Returns
    -------
    Phi3 : ndarray, shape (n_dof, n_dof, n_dof)
        Mass-weighted FC3 in eV/(A^3 amu^{3/2}).
    """
    n_dof = 3 * n_atoms
    Phi3 = np.zeros((n_dof, n_dof, n_dof))
    masses = np.asarray(masses_amu)

    for block in fc3_data["blocks"]:
        cell_j = block["cell_j"]
        cell_k = block["cell_k"]
        if not (np.allclose(cell_j, 0.0) and np.allclose(cell_k, 0.0)):
            continue
        ai = block["atom_i"]
        aj = block["atom_j"]
        ak = block["atom_k"]
        mass_factor = np.sqrt(masses[ai] * masses[aj] * masses[ak])
        tensor = block["tensor"] / mass_factor  # (3, 3, 3)
        for alpha in range(3):
            for beta in range(3):
                for gamma in range(3):
                    Phi3[3 * ai + alpha, 3 * aj + beta, 3 * ak + gamma] = tensor[
                        alpha, beta, gamma
                    ]

    return Phi3


def _assemble_fc3_full(fc3_data, n_atoms, masses_amu, prim_cell,
                       transport_direction="z"):
    """Assemble mass-weighted FC3 including nearest-neighbour cells.

    Includes all FC3 entries where both R' and R'' map to the same slab
    along the transport direction (delta_l=0). This includes transverse
    nearest-neighbour interactions.

    Parameters
    ----------
    fc3_data : dict
        Output of ``force_constants.load_fc3_thirdorder()`` or
        ``force_constants.load_fc3_phono3py()``.
    n_atoms : int
        Number of atoms in the primitive cell.
    masses_amu : array-like, shape (n_atoms,)
        Atomic masses in amu.
    prim_cell : ndarray, shape (3, 3)
        Primitive lattice vectors (rows) in Angstrom.
    transport_direction : str
        "x", "y", or "z".

    Returns
    -------
    Phi3 : ndarray, shape (n_dof, n_dof, n_dof)
        Mass-weighted FC3 in eV/(A^3 amu^{3/2}).
    n_included : int
        Number of FC3 blocks included.
    n_total : int
        Total number of FC3 blocks.
    """
    n_dof = 3 * n_atoms
    Phi3 = np.zeros((n_dof, n_dof, n_dof))
    masses = np.asarray(masses_amu)
    tidx = "xyz".index(transport_direction)
    inv_cell = np.linalg.inv(prim_cell.T)

    n_included = 0
    n_total = len(fc3_data["blocks"])

    for block in fc3_data["blocks"]:
        cell_j_cart = block["cell_j"]
        cell_k_cart = block["cell_k"]

        frac_j = inv_cell @ cell_j_cart
        frac_k = inv_cell @ cell_k_cart

        dl_j = int(np.round(frac_j[tidx]))
        dl_k = int(np.round(frac_k[tidx]))

        if dl_j != 0 or dl_k != 0:
            continue

        n_included += 1
        ai = block["atom_i"]
        aj = block["atom_j"]
        ak = block["atom_k"]
        mass_factor = np.sqrt(masses[ai] * masses[aj] * masses[ak])
        tensor = block["tensor"] / mass_factor
        for alpha in range(3):
            for beta in range(3):
                for gamma in range(3):
                    Phi3[3 * ai + alpha, 3 * aj + beta, 3 * ak + gamma] += tensor[
                        alpha, beta, gamma
                    ]

    return Phi3, n_included, n_total


# ---------------------------------------------------------------------------
# SCBA loop
# ---------------------------------------------------------------------------


def anharmonic_transmission(
    phonon,
    fc3_data: dict,
    q_mesh_transverse: tuple[int, int] = (4, 4),
    freq_range_thz: tuple[float, float, int] = (0.01, 16.0, 101),
    transport_direction: str = "z",
    eta_factor: float = 0.5,
    temperature: float = 300.0,
    delta_T: float = 10.0,
    max_scba_iter: int = 10,
    scba_tol: float = 0.01,
    mixing: float = 0.5,
    fc3_mode: str = "full",
    n_slabs: int = 1,
    verbose: bool = True,
) -> dict:
    """Compute anharmonic phonon transmission via SCBA.

    The device region consists of ``n_slabs`` identical unit cells
    coupled to semi-infinite leads of the same material.  The
    self-energy is computed per slab from the q-averaged local
    Green's function (diagonal block of each slab), following
    Guo Eq. 8 with Approximation III.

    Internal units: THz^2 for all dynamical matrix quantities.

    Parameters
    ----------
    phonon : Phonopy
        Phonopy object with harmonic force constants.
    fc3_data : dict
        Third-order force constants from ``load_fc3_thirdorder()``
        or ``load_fc3_phono3py()``.
    q_mesh_transverse : (int, int)
        Transverse q-mesh.
    freq_range_thz : (fmin, fmax, nfreq)
        Frequency grid in THz.
    transport_direction : str
        Transport direction.
    eta_factor : float
        Broadening: eta = dw^2 * eta_factor (in THz^2).
    temperature : float
        Mean temperature in Kelvin.
    delta_T : float
        Temperature difference between contacts in Kelvin.
    max_scba_iter : int
        Maximum SCBA iterations.
    scba_tol : float
        Convergence tolerance for heat flow conservation.
    mixing : float
        Self-energy mixing factor (0 < alpha <= 1).
    fc3_mode : str
        "onsite": only R'=R''=0 blocks.
        "full": all blocks with delta_l_transport=0.
    n_slabs : int
        Number of unit-cell slabs in the device region.
    verbose : bool
        Print progress.

    Returns
    -------
    result : dict
        - freqs_thz: frequency grid
        - transmission_ballistic: Caroli transmission (ballistic reference)
        - spectral_heat_current_ballistic: hbar*w*(n_L - n_R)*T(w) in W per freq bin
        - spectral_heat_current: anharmonic spectral heat current (Meir-Wingreen) in W
        - heat_current_ballistic: total ballistic heat current in W
        - heat_current: total anharmonic heat current in W
        - thermal_conductance_ballistic: G = J/(A_c*delta_T) in W/(m^2 K)
        - thermal_conductance_anharmonic: G = J/(A_c*delta_T) in W/(m^2 K)
        - heat_flow_conservation: |J_L - J_R| / (|J_L| + |J_R|)
        - n_scba_iterations: number of iterations performed
        - convergence_history: list of relative changes in J
        - self_energy_retarded: per-slab self-energy (n_slabs, nfreq, n_dof, n_dof)
    """
    from .convention import get_btd_blocks
    from .validation import _ballistic_transmission

    fmin, fmax, nfreq = freq_range_thz
    nfreq = int(nfreq)
    freqs_thz = np.linspace(fmin, fmax, nfreq)
    omega_sq_thz2 = freqs_thz ** 2  # THz^2
    dw_thz = freqs_thz[1] - freqs_thz[0]  # THz
    eta = dw_thz ** 2 * eta_factor  # THz^2

    n_atoms = len(phonon.primitive.masses)
    n_dof = 3 * n_atoms
    N_D = n_slabs * n_dof
    masses_amu = phonon.primitive.masses
    prim_cell = phonon.primitive.cell

    # Assemble FC3 (mass-weighted, in eV/(A^3 amu^{3/2}))
    if fc3_mode == "onsite":
        Phi3 = _assemble_fc3_dense(fc3_data, n_atoms, masses_amu)
        if verbose:
            nz = np.count_nonzero(Phi3)
            print(f"  FC3 mode: onsite, non-zero elements: {nz}/{Phi3.size}")
    elif fc3_mode == "full":
        Phi3, n_inc, n_tot = _assemble_fc3_full(
            fc3_data, n_atoms, masses_amu, prim_cell, transport_direction
        )
        if verbose:
            nz = np.count_nonzero(Phi3)
            print(f"  FC3 mode: full, blocks: {n_inc}/{n_tot}, "
                  f"non-zero elements: {nz}/{Phi3.size}")
    else:
        raise ValueError(f"Unknown fc3_mode: {fc3_mode}")

    # Convert FC3 to THz^{5/2} units
    Phi3_converted = Phi3 * CONVERSION_FC3_THZ

    if verbose:
        print(f"  FC3 max (THz^5/2): {np.max(np.abs(Phi3_converted)):.4e}")
        print(f"  Device: {n_slabs} slab(s), {N_D} DOFs per q-point")

    # Set up q-mesh
    nkx, nky = q_mesh_transverse
    q_1d_x = [(2 * n - nkx - 1) / (2 * nkx) for n in range(1, nkx + 1)]
    q_1d_y = [(2 * n - nky - 1) / (2 * nky) for n in range(1, nky + 1)]
    q_points = [(qx, qy) for qx in q_1d_x for qy in q_1d_y]
    n_kpts = len(q_points)

    # Bose-Einstein distribution (needs hbar*omega in eV)
    def bose_einstein(freq_thz_arr, T):
        hw = HBAR_EV * np.abs(freq_thz_arr) * THZ_TO_RAD  # eV
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
        print(f"  q-mesh: {nkx}x{nky} = {n_kpts} points")
        print(f"  Frequency grid: {nfreq} points, {fmin:.2f} to {fmax:.2f} THz")
        print(f"  Temperature: {temperature} K, delta_T: {delta_T} K")
        print(f"  eta = {eta:.4e} THz^2")
        print(f"  SCBA: max {max_scba_iter} iter, tol={scba_tol}, mix={mixing}")

    # Precompute BTD blocks for all q-points (in THz^2)
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
        for iw, w2 in enumerate(omega_sq_thz2):
            trans_ballistic[iw] += _ballistic_transmission(
                w2, H_D, H_00, H_01, H_00, H_01, H_LD, H_DR, eta=eta
            )
    trans_ballistic /= n_kpts

    if verbose:
        print(f"  Ballistic max T: {trans_ballistic.max():.4f}")

    # Cross-sectional area perpendicular to transport
    lattice = phonon.primitive.cell
    tidx = "xyz".index(transport_direction)
    perp_idx = [i for i in range(3) if i != tidx]
    a1 = lattice[perp_idx[0]]
    a2 = lattice[perp_idx[1]]
    A_c = np.linalg.norm(np.cross(a1, a2)) * 1e-20  # Ang^2 -> m^2

    # Ballistic spectral heat current from Caroli transmission:
    # J(w) = hbar*omega * (n_L(w) - n_R(w)) * T(w)
    # omega in rad/s for physical current (Watts)
    omega_rad = freqs_thz * THZ_TO_RAD
    spectral_J_ball = (HBAR_SI * omega_rad
                       * (n_bose_L - n_bose_R) * trans_ballistic)
    # Integration: sum * dw_rad / (2*pi) = sum * dw_thz * 1e12
    J_ball_total = np.sum(spectral_J_ball) * dw_thz * 1e12
    G_ball = J_ball_total / (A_c * delta_T)

    if verbose:
        print(f"  Ballistic thermal conductance: {G_ball:.2f} W/(m^2 K)")

    # --- SCBA with per-slab q-averaged Green's functions ---
    Sigma_slab_R = np.zeros((n_slabs, nfreq, n_dof, n_dof), dtype=complex)
    Sigma_slab_lesser = np.zeros((n_slabs, nfreq, n_dof, n_dof), dtype=complex)
    Sigma_slab_greater = np.zeros((n_slabs, nfreq, n_dof, n_dof), dtype=complex)

    convergence_history = []
    current_mixing = mixing
    conservation_err = 1.0
    J_total_prev = 0.0

    Sigma_slab_lesser_prev = Sigma_slab_lesser.copy()
    Sigma_slab_greater_prev = Sigma_slab_greater.copy()
    Sigma_slab_R_prev = Sigma_slab_R.copy()

    # Spectral heat current at left/right boundaries
    spectral_J_L = np.zeros(nfreq)
    spectral_J_R = np.zeros(nfreq)

    for scba_iter in range(max_scba_iter):
        G_lesser_slab = np.zeros((n_slabs, nfreq, n_dof, n_dof), dtype=complex)
        G_greater_slab = np.zeros((n_slabs, nfreq, n_dof, n_dof), dtype=complex)
        spectral_J_L[:] = 0.0
        spectral_J_R[:] = 0.0

        for iq, (H_00, H_01) in enumerate(btd_blocks):
            H_D = _build_device_hamiltonian(H_00, H_01, n_slabs)

            for iw, w2 in enumerate(omega_sq_thz2):
                Sigma_R_dev = np.zeros((N_D, N_D), dtype=complex)
                Sigma_l_dev = np.zeros((N_D, N_D), dtype=complex)
                Sigma_g_dev = np.zeros((N_D, N_D), dtype=complex)
                for l in range(n_slabs):
                    sl = slice(l * n_dof, (l + 1) * n_dof)
                    Sigma_R_dev[sl, sl] = Sigma_slab_R[l, iw]
                    Sigma_l_dev[sl, sl] = Sigma_slab_lesser[l, iw]
                    Sigma_g_dev[sl, sl] = Sigma_slab_greater[l, iw]

                obc = _compute_obc_self_energies(
                    w2, H_00, H_01, eta, n_bose_L[iw], n_bose_R[iw],
                    n_slabs=n_slabs,
                )
                _, G_less, G_great = _solve_green_functions(
                    w2, H_D, obc, Sigma_R_dev, Sigma_l_dev, Sigma_g_dev, eta,
                )

                for l in range(n_slabs):
                    sl = slice(l * n_dof, (l + 1) * n_dof)
                    G_lesser_slab[l, iw] += G_less[sl, sl]
                    G_greater_slab[l, iw] += G_great[sl, sl]

                # Meir-Wingreen heat current at boundaries
                # J_L = hbar*omega * Tr[Sigma_L^> G^< - Sigma_L^< G^>]
                # omega in rad/s for physical current (W)
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

        G_lesser_slab /= n_kpts
        G_greater_slab /= n_kpts
        spectral_J_L /= n_kpts
        spectral_J_R /= n_kpts

        # Total heat current: integrate spectral current
        # sum * dw_rad / (2*pi) = sum * dw_thz * 1e12
        J_L_total = np.sum(spectral_J_L) * dw_thz * 1e12
        J_R_total = np.sum(spectral_J_R) * dw_thz * 1e12
        J_total = 0.5 * (J_L_total + J_R_total)
        J_denom = abs(J_L_total) + abs(J_R_total)
        conservation_err = (abs(J_L_total - J_R_total) / J_denom
                            if J_denom > 0 else 0.0)

        if verbose and scba_iter == 0:
            gl_max = np.max(np.abs(G_lesser_slab))
            print(f"    G diagnostic (q-avg, per-slab): max|G^<| = {gl_max:.4e}")

        # Compute per-slab phonon-phonon self-energy
        Sigma_slab_l_new = np.zeros_like(Sigma_slab_lesser)
        Sigma_slab_g_new = np.zeros_like(Sigma_slab_greater)
        Sigma_slab_r_new = np.zeros_like(Sigma_slab_R)

        for l in range(n_slabs):
            sl_new, sg_new, sr_new = _compute_phph_self_energy(
                G_lesser_slab[l], G_greater_slab[l],
                Phi3_converted, n_dof, freqs_thz, dw_thz,
            )
            Sigma_slab_l_new[l] = sl_new
            Sigma_slab_g_new[l] = sg_new
            Sigma_slab_r_new[l] = sr_new

        sig_r_new_norm = np.max(np.abs(Sigma_slab_r_new))

        if verbose and scba_iter == 0:
            h00_max = max(np.max(np.abs(H)) for H, _ in btd_blocks)
            print(f"    Self-energy: max|Sigma^R| = {sig_r_new_norm:.4e} THz^2, "
                  f"|Sigma^R|/|H_00| = {sig_r_new_norm/h00_max:.4e}")

        # Mix with adaptive damping
        if scba_iter > 0:
            alpha = current_mixing
            Sigma_slab_lesser = ((1 - alpha) * Sigma_slab_lesser
                                 + alpha * Sigma_slab_l_new)
            Sigma_slab_greater = ((1 - alpha) * Sigma_slab_greater
                                  + alpha * Sigma_slab_g_new)
            Sigma_slab_R = ((1 - alpha) * Sigma_slab_R
                            + alpha * Sigma_slab_r_new)
        else:
            Sigma_slab_lesser = Sigma_slab_l_new.copy()
            Sigma_slab_greater = Sigma_slab_g_new.copy()
            Sigma_slab_R = Sigma_slab_r_new.copy()

        Sigma_slab_lesser_prev = Sigma_slab_lesser.copy()
        Sigma_slab_greater_prev = Sigma_slab_greater.copy()
        Sigma_slab_R_prev = Sigma_slab_R.copy()

        # Check convergence
        if scba_iter > 0:
            rel_change = (abs(J_total - J_total_prev)
                          / (abs(J_total_prev) + 1e-30))
            convergence_history.append(rel_change)
            if verbose:
                sig_r_now = np.max(np.abs(Sigma_slab_R))
                print(f"    SCBA iter {scba_iter + 1}: "
                      f"J = {J_total:.4e} W, "
                      f"conservation = {conservation_err:.4e}, "
                      f"rel. change = {rel_change:.4e}, "
                      f"max|Sigma^R| = {sig_r_now:.2e} THz^2")
            if conservation_err < scba_tol:
                if verbose:
                    print(f"    Converged after {scba_iter + 1} iterations "
                          f"(heat flow conservation: {conservation_err:.2e})")
                break
        else:
            if verbose:
                print(f"    SCBA iter 1: "
                      f"J_L = {J_L_total:.4e} W, J_R = {J_R_total:.4e} W")

        J_total_prev = J_total

    # Anharmonic thermal conductance
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
        "heat_current_ballistic": J_ball_total,
        "heat_current": J_anh_total,
        "thermal_conductance_ballistic": G_ball,
        "thermal_conductance_anharmonic": G_anh,
        "heat_flow_conservation": conservation_err,
        "delta_T": delta_T,
        "n_scba_iterations": len(convergence_history) + 1,
        "convergence_history": convergence_history,
        "self_energy_retarded": Sigma_slab_R,
        "self_energy_lesser": Sigma_slab_lesser,
        "self_energy_greater": Sigma_slab_greater,
    }


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
    Phi_hat = np.zeros((n_dof, n_dof, n_dof), dtype=complex)
    T_q2_T = T_q2.T  # (dim_t, n_dof)
    for a in range(n_dof):
        M_a = M_stacked[a * dim_t:(a + 1) * dim_t, :]
        Phi_hat[a] = T_q1 @ M_a @ T_q2_T
    return Phi_hat


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
    M_stacked : ndarray, shape (n_dof * dim_t, dim_t)
        Real-space FC3 matrices from build_realspace_fc3_matrices.
    T_all_q : list of ndarray, each (n_dof, dim_t)
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

    # Pad and FFT all Green's functions
    def _pad_fft(G_q):
        out = np.zeros((n_kpts, n_fft, n_dof, n_dof), dtype=complex)
        out[:, n_low:n_low + n_freq] = G_q
        return np.fft.fft(out, axis=1)

    GL_fft = _pad_fft(G_lesser_q)
    GG_fft = _pad_fft(G_greater_q)

    # Precompute TM[iq][a] = T(q) @ M_a, shape (n_dof, dim_t) per (iq, a)
    TM = np.zeros((n_kpts, n_dof, n_dof, dim_t), dtype=complex)
    for iq in range(n_kpts):
        for a in range(n_dof):
            TM[iq, a] = T_all_q[iq] @ M_stacked[a * dim_t:(a + 1) * dim_t, :]

    for iq_ext in range(n_kpts):
        for iq_prime in range(n_kpts):
            iq_diff = q_diff_map[iq_ext, iq_prime]

            # Phi_L(q', q-q')[a] = TM[q', a] @ T(q-q')^T
            # Phi_R = conj(Phi_hat(q-q', q'))
            T_diff_T = T_all_q[iq_diff].T   # (dim_t, n_dof)
            T_prime_T = T_all_q[iq_prime].T  # (dim_t, n_dof)

            Phi_L = np.zeros((n_dof, n_dof, n_dof), dtype=complex)
            Phi_R_raw = np.zeros((n_dof, n_dof, n_dof), dtype=complex)
            for a in range(n_dof):
                Phi_L[a] = TM[iq_prime, a] @ T_diff_T
                Phi_R_raw[a] = TM[iq_diff, a] @ T_prime_T
            Phi_R = np.conj(Phi_R_raw)

            for G_fft, Sigma_out in [(GL_fft, Sigma_lesser),
                                     (GG_fft, Sigma_greater)]:
                # K[w,c,d,f,e] = IFFT(G_fft[q',w,c,f] * G_fft[q-q',w,d,e])
                product = (G_fft[iq_prime][:, :, None, :, None]
                           * G_fft[iq_diff][:, None, :, None, :])
                K = np.fft.ifft(product, axis=0)[freq_sl]

                # Sigma[w,a,b] += pref * Phi_L[a,c,d] K[w,c,d,f,e] Phi_R[b,e,f]
                # Two-step einsum for efficiency
                temp = np.einsum('acd,wcdfe->wafe', Phi_L, K)
                Sigma_out[iq_ext] += prefactor * np.einsum(
                    'wafe,bef->wab', temp, Phi_R
                )

    Sigma_retarded = 0.5 * (Sigma_greater - Sigma_lesser)
    return Sigma_lesser, Sigma_greater, Sigma_retarded


def anharmonic_transmission_q(
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
    verbose: bool = True,
) -> dict:
    """Anharmonic phonon transport with full q-dependent dense self-energy.

    Unlike the Approximation III ``anharmonic_transmission`` (which
    q-averages G before computing a single self-energy), this function
    retains the explicit q'-sum in the self-energy and Fourier-transforms
    the raw supercell FC3 on-the-fly for each (q', q-q') pair.

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
    trans_atoms = np.where(slab_indices == 0)[0]

    # --- Load raw FC3 and build real-space matrices ---
    with h5py.File(fc3_hdf5, "r") as f:
        fc3_raw = np.array(f["fc3"])

    M_stacked = build_realspace_fc3_matrices(
        fc3_raw, n_atoms, masses_super, ref_sc_atoms, trans_atoms
    )
    dim_t = len(trans_atoms) * 3

    if verbose:
        print(f"  FC3 raw shape: {fc3_raw.shape}")
        print(f"  Same-slab atoms: {len(trans_atoms)}, dim_trans: {dim_t}")
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
            trans_atoms, prim_indices, cell_frac,
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

    for scba_iter in range(max_scba_iter):
        G_lesser_slab_q = np.zeros(
            (n_slabs, n_kpts, nfreq, n_dof, n_dof), dtype=complex
        )
        G_greater_slab_q = np.zeros_like(G_lesser_slab_q)
        spectral_J_L[:] = 0.0
        spectral_J_R[:] = 0.0

        for iq, (H_00, H_01) in enumerate(btd_blocks):
            H_D = _build_device_hamiltonian(H_00, H_01, n_slabs)

            for iw, w2 in enumerate(omega_sq_thz2):
                Sig_R_dev = np.zeros((N_D, N_D), dtype=complex)
                Sig_l_dev = np.zeros((N_D, N_D), dtype=complex)
                Sig_g_dev = np.zeros((N_D, N_D), dtype=complex)
                for l in range(n_slabs):
                    sl = slice(l * n_dof, (l + 1) * n_dof)
                    Sig_R_dev[sl, sl] = Sigma_R_q[l, iq, iw]
                    Sig_l_dev[sl, sl] = Sigma_l_q[l, iq, iw]
                    Sig_g_dev[sl, sl] = Sigma_g_q[l, iq, iw]

                obc = _compute_obc_self_energies(
                    w2, H_00, H_01, eta, n_bose_L[iw], n_bose_R[iw],
                    n_slabs=n_slabs,
                )
                _, G_less, G_great = _solve_green_functions(
                    w2, H_D, obc, Sig_R_dev, Sig_l_dev, Sig_g_dev, eta,
                )

                for l in range(n_slabs):
                    sl = slice(l * n_dof, (l + 1) * n_dof)
                    G_lesser_slab_q[l, iq, iw] = G_less[sl, sl]
                    G_greater_slab_q[l, iq, iw] = G_great[sl, sl]

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
            sl_n, sg_n, sr_n = _compute_phph_self_energy_q_dense(
                G_lesser_slab_q[l], G_greater_slab_q[l],
                M_stacked, T_all_q, q_diff_map,
                n_atoms, n_kpts, freqs_thz, dw_thz,
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
