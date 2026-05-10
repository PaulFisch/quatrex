"""Dynamical matrix convention transformation and block extraction.

Phonopy uses Convention A: phase = exp(2*pi*i * q . (R + tau' - tau))
Quatrex uses Convention B:  phase = exp(2*pi*i * q . R)

The gauge transformation between them is:
    D_B(q) = P(q) D_A(q) P(q)^dagger
where P(q) = diag(exp(2*pi*i * q . tau_kappa)) repeated 3x per atom.

Convention B IDFT gives real-valued blocks H(R) = Phi(0,R) / sqrt(m*m').
"""

import numpy as np
from phonopy import Phonopy

from .constants import CONVERSION


def gauge_transform_A_to_B(
    D_A: np.ndarray,
    q: np.ndarray,
    tau_frac: np.ndarray,
) -> np.ndarray:
    """Apply gauge transform from phonopy Convention A to Convention B.

    D_B(q) = P(q) D_A(q) P(q)^dagger

    Parameters
    ----------
    D_A : ndarray, shape (n_dof, n_dof), complex
        Dynamical matrix in Convention A (from phonopy).
    q : ndarray, shape (3,)
        q-point in fractional reciprocal coordinates.
    tau_frac : ndarray, shape (n_atoms, 3)
        Basis-atom positions in fractional coordinates.

    Returns
    -------
    D_B : ndarray, shape (n_dof, n_dof), complex
        Dynamical matrix in Convention B.
    """
    phases_atom = np.exp(2j * np.pi * tau_frac @ q)  # (n_atoms,)
    P = np.repeat(phases_atom, 3)  # (n_dof,)
    return P[:, None] * D_A * P[None, :].conj()


def evaluate_on_mesh(
    phonon: Phonopy,
    q_mesh: tuple[int, int, int],
    convention: str = "B",
) -> np.ndarray:
    """Evaluate the dynamical matrix on a uniform q-mesh.

    Parameters
    ----------
    phonon : Phonopy
        Phonopy object with force constants set.
    q_mesh : (Nqx, Nqy, Nqz)
        Number of q-points along each axis.
    convention : "A" or "B"
        "B" applies the gauge transform to Convention B.

    Returns
    -------
    D_q : ndarray, shape (Nqx, Nqy, Nqz, n_dof, n_dof), complex
    """
    Nx, Ny, Nz = q_mesh
    tau_frac = phonon.primitive.scaled_positions
    n_dof = len(phonon.primitive.masses) * 3

    D_q = np.zeros((Nx, Ny, Nz, n_dof, n_dof), dtype=complex)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                q = np.array([i / Nx, j / Ny, k / Nz])
                D = phonon.get_dynamical_matrix_at_q(q)
                if convention.upper() == "B":
                    D = gauge_transform_A_to_B(D, q, tau_frac)
                D_q[i, j, k] = D

    return D_q


def idft_to_blocks(
    D_q: np.ndarray,
    conversion_factor: float = CONVERSION,
    amplitude_cutoff: float = 1e10,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Inverse DFT of D_B(q) to real-space blocks H(R).

    H(nx, ny, nz) = (CONVERSION / Nq^3) * sum_{ijk} D_B(qi, qj, qk)
                     * exp(-2*pi*i*(nx*i + ny*j + nz*k) / Nq)

    Since D_B is a lattice Fourier series of real force constants,
    the result H(R) is real. The imaginary part (numerical noise) is
    discarded.

    Parameters
    ----------
    D_q : ndarray, shape (Nx, Ny, Nz, n_dof, n_dof), complex
        Convention B dynamical matrix on a uniform mesh.
    conversion_factor : float
        Multiply result by this (default: phonopy eigenvalue -> (rad/s)^2).
    amplitude_cutoff : float
        Drop blocks with max |H| below this threshold.

    Returns
    -------
    blocks : dict[(nx, ny, nz), ndarray]
        Real-space blocks. Keys are integer lattice translations
        centered around 0, e.g., {-1, 0, +1} for a 3^3 mesh.
        Values are (n_dof, n_dof) real arrays in (rad/s)^2.
    """
    Nx, Ny, Nz = D_q.shape[:3]
    n_dof = D_q.shape[3]
    blocks = {}

    for nx in range(-(Nx // 2), Nx // 2 + 1):
        for ny in range(-(Ny // 2), Ny // 2 + 1):
            for nz in range(-(Nz // 2), Nz // 2 + 1):
                H = np.zeros((n_dof, n_dof), dtype=complex)
                for i in range(Nx):
                    for j in range(Ny):
                        for k in range(Nz):
                            phase = np.exp(
                                -2j * np.pi * (nx * i / Nx + ny * j / Ny + nz * k / Nz)
                            )
                            H += D_q[i, j, k] * phase
                H /= Nx * Ny * Nz
                H *= conversion_factor

                if np.max(np.abs(H.real)) > amplitude_cutoff:
                    blocks[(nx, ny, nz)] = H.real

    return blocks


def extract_blocks(
    phonon: Phonopy,
    q_mesh: tuple[int, int, int] = (3, 3, 3),
    conversion_factor: float = CONVERSION,
    amplitude_cutoff: float = 1e10,
) -> dict[tuple[int, int, int], np.ndarray]:
    """High-level: phonopy object -> real-space Convention B blocks.

    Combines evaluate_on_mesh + idft_to_blocks.

    Parameters
    ----------
    phonon : Phonopy
        Phonopy object with force constants set.
    q_mesh : (int, int, int)
        Mesh dimensions for IDFT.
    conversion_factor : float
        Unit conversion factor.
    amplitude_cutoff : float
        Drop negligible blocks.

    Returns
    -------
    blocks : dict[(nx, ny, nz), ndarray]
        Real-space blocks in (rad/s)^2.
    """
    D_q = evaluate_on_mesh(phonon, q_mesh, convention="B")
    return idft_to_blocks(D_q, conversion_factor, amplitude_cutoff)


def get_btd_blocks(
    phonon: Phonopy,
    q_perp_frac: tuple[float, float],
    transport_direction: str = "z",
    n_qz: int = 3,
    conversion_factor: float = CONVERSION,
) -> tuple[np.ndarray, np.ndarray]:
    """Get BTD on-site and coupling blocks at a transverse q-point.

    Evaluates D_B(q_perp, qz) for qz on a 1D mesh and performs
    the 1D IDFT to get H_00(q_perp) and H_01(q_perp).

    These are COMPLEX at non-zero q_perp (2D Fourier sum of real
    coefficients evaluated at non-zero argument).

    Parameters
    ----------
    phonon : Phonopy
        Phonopy object with force constants.
    q_perp_frac : (float, float)
        Transverse q-point in fractional coordinates.
    transport_direction : str
        "x", "y", or "z".
    n_qz : int
        Number of q-points along the transport direction.
    conversion_factor : float
        Unit conversion factor.

    Returns
    -------
    H_00 : ndarray, shape (n_dof, n_dof), complex
        On-site block (coupling within one layer).
    H_01 : ndarray, shape (n_dof, n_dof), complex
        Coupling to next layer along +transport.
    """
    tau = phonon.primitive.scaled_positions
    nd = len(phonon.primitive.masses) * 3
    tidx = "xyz".index(transport_direction)

    D_qz = np.zeros((n_qz, nd, nd), dtype=complex)
    for k in range(n_qz):
        q = [0.0, 0.0, 0.0]
        # Fill transverse components
        perp_idx = [i for i in range(3) if i != tidx]
        q[perp_idx[0]] = q_perp_frac[0]
        q[perp_idx[1]] = q_perp_frac[1]
        q[tidx] = k / n_qz
        q = np.array(q)

        D_A = phonon.get_dynamical_matrix_at_q(q)
        D_qz[k] = gauge_transform_A_to_B(D_A, q, tau)

    # 1D IDFT over transport direction
    phases = np.exp(-2j * np.pi * np.arange(n_qz) / n_qz)
    H_00 = np.mean(D_qz, axis=0) * conversion_factor
    H_01 = np.einsum("k,kij->ij", phases, D_qz) / n_qz * conversion_factor

    return H_00, H_01
