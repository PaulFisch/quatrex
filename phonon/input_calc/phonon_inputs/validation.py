"""Validation and reference calculations for phonon transport inputs.

Provides:
- Gamma-point acoustic sum rule check
- Block symmetry check: H(R)^T = H(-R)
- Band structure comparison (extracted blocks vs phonopy)
- Reference Sancho-Rubio + Caroli ballistic transmission
- Thermal conductance from the Landauer formula

Internal units: THz^2 by default (set conversion_factor=CONVERSION for legacy (rad/s)^2).
"""

import numpy as np
from numpy.linalg import eigvalsh, inv, norm
from phonopy import Phonopy

from .constants import CONVERSION_THZ2, EV_TO_J, HBAR_EV, HBAR_SI, KB_EV, THZ_TO_RAD
from .convention import gauge_transform_A_to_B, get_btd_blocks


def check_gamma_point(
        blocks: dict[tuple[int, int, int], np.ndarray],
        n_acoustic: int = 3,
        conversion_factor: float | None = None,
) -> dict:
    """Check that summing all blocks gives n_acoustic zero eigenvalues.

    Parameters
    ----------
    blocks : dict[(nx, ny, nz), ndarray]
        Real-space blocks (in THz^2 or (rad/s)^2).
    n_acoustic : int
        Expected number of acoustic (zero) modes.
    conversion_factor : float, optional
        If provided, eigenvalues are in this unit system. Default assumes THz^2.

    Returns
    -------
    result : dict
        - frequencies_thz: all eigenfrequencies in THz
        - acoustic_freqs_thz: the n_acoustic smallest |freq|
        - symmetry_error: max|D_gamma - D_gamma^T|
    """
    D_gamma = sum(blocks.values())
    eigs = eigvalsh(D_gamma)

    # Convert eigenvalues to THz
    if conversion_factor is not None:
        # Legacy: blocks in (rad/s)^2
        freqs = np.sign(eigs) * np.sqrt(np.abs(eigs)) / THZ_TO_RAD
    else:
        # Default: blocks in THz^2
        freqs = np.sign(eigs) * np.sqrt(np.abs(eigs))

    sorted_abs = np.sort(np.abs(freqs))
    return {
        "frequencies_thz": np.sort(freqs),
        "acoustic_freqs_thz": sorted_abs[:n_acoustic],
        "symmetry_error": np.max(np.abs(D_gamma - D_gamma.T)),
    }


def check_block_symmetry(
        blocks: dict[tuple[int, int, int], np.ndarray],
) -> dict:
    """Check H(R)^T = H(-R) for all block pairs.

    Returns
    -------
    result : dict
        - max_error: maximum |H(R)^T - H(-R)| over all pairs.
        - pair_errors: dict[(R, -R), error]
    """
    max_err = 0.0
    pair_errors = {}
    for R, H in blocks.items():
        mR = tuple(-x for x in R)
        if mR in blocks:
            err = np.max(np.abs(H.T - blocks[mR]))
            pair_errors[(R, mR)] = err
            max_err = max(max_err, err)
    return {"max_error": max_err, "pair_errors": pair_errors}


def compare_arrays(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    """Return simple comparison metrics for two arrays."""
    if a.shape != b.shape:
        raise ValueError(f"{name}: shape mismatch {a.shape} vs {b.shape}")

    diff = a - b
    a_norm = np.linalg.norm(a.ravel())
    b_norm = np.linalg.norm(b.ravel())
    d_norm = np.linalg.norm(diff.ravel())

    result = {
        "name": name,
        "shape": a.shape,
        "max_abs_a": float(np.max(np.abs(a))),
        "max_abs_b": float(np.max(np.abs(b))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "rms_diff": float(np.sqrt(np.mean(diff ** 2))),
        "fro_a": float(a_norm),
        "fro_b": float(b_norm),
        "fro_diff": float(d_norm),
        "rel_diff_vs_a": float(d_norm / a_norm) if a_norm > 0 else np.nan,
        "rel_diff_vs_b": float(d_norm / b_norm) if b_norm > 0 else np.nan,
    }
    return result


def print_comparison(result: dict) -> None:
    """Pretty-print comparison metrics."""
    print(f"{result['name']}:")
    print(f"  shape:         {result['shape']}")
    print(f"  max |A|:       {result['max_abs_a']:.6e}")
    print(f"  max |B|:       {result['max_abs_b']:.6e}")
    print(f"  max |A-B|:     {result['max_abs_diff']:.6e}")
    print(f"  RMS(A-B):      {result['rms_diff']:.6e}")
    print(f"  ||A||_F:       {result['fro_a']:.6e}")
    print(f"  ||B||_F:       {result['fro_b']:.6e}")
    print(f"  ||A-B||_F:     {result['fro_diff']:.6e}")
    print(f"  rel vs A:      {result['rel_diff_vs_a']:.6e}")
    print(f"  rel vs B:      {result['rel_diff_vs_b']:.6e}")


def compare_band_structure(
        phonon: Phonopy,
        blocks: dict[tuple[int, int, int], np.ndarray],
        band_paths: list[list[list[float]]],
        labels: list[str],
        npoints: int = 51,
        save_path=None,
        conversion_factor: float = CONVERSION_THZ2,
) -> dict:
    """Compare eigenfrequencies from extracted blocks vs phonopy.

    Both are evaluated in Convention B.

    Parameters
    ----------
    phonon : Phonopy
        Phonopy object with force constants.
    blocks : dict
        Extracted real-space blocks (THz^2 by default).
    band_paths : list of segments
    labels : list of str
    npoints : int
    save_path : Path, optional
    conversion_factor : float
        Conversion factor used for blocks (default: CONVERSION_THZ2).

    Returns
    -------
    result : dict
        - max_diff_thz: maximum eigenfrequency difference in THz
    """
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

    qpoints, connections = get_band_qpoints_and_path_connections(
        band_paths, npoints=npoints
    )
    tau_frac = phonon.primitive.scaled_positions

    freqs_phonopy = []
    freqs_blocks = []

    for seg_qpts in qpoints:
        for q in seg_qpts:
            # Phonopy -> Convention B -> THz^2
            D_A = phonon.get_dynamical_matrix_at_q(q)
            D_ph = gauge_transform_A_to_B(D_A, q, tau_frac) * conversion_factor
            e_ph = eigvalsh(D_ph)

            if conversion_factor == CONVERSION_THZ2:
                freqs_phonopy.append(
                    np.sign(e_ph) * np.sqrt(np.abs(e_ph))
                )
            else:
                freqs_phonopy.append(
                    np.sign(e_ph) * np.sqrt(np.abs(e_ph)) / THZ_TO_RAD
                )

            # Reconstructed from blocks
            D_rec = sum(
                H * np.exp(2j * np.pi * (nx * q[0] + ny * q[1] + nz * q[2]))
                for (nx, ny, nz), H in blocks.items()
            )
            e_bl = eigvalsh(D_rec)

            if conversion_factor == CONVERSION_THZ2:
                freqs_blocks.append(
                    np.sign(e_bl) * np.sqrt(np.abs(e_bl))
                )
            else:
                freqs_blocks.append(
                    np.sign(e_bl) * np.sqrt(np.abs(e_bl)) / THZ_TO_RAD
                )

    freqs_phonopy = np.array(freqs_phonopy)
    freqs_blocks = np.array(freqs_blocks)
    max_diff = np.max(np.abs(freqs_phonopy - freqs_blocks))

    if save_path is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(freqs_phonopy))
        for b in range(freqs_phonopy.shape[1]):
            ax.plot(
                x, freqs_phonopy[:, b], "b-", lw=1, alpha=0.6,
                label="phonopy" if b == 0 else None,
            )
            ax.plot(
                x, freqs_blocks[:, b], "r--", lw=0.8, alpha=0.6,
                label="extracted blocks" if b == 0 else None,
            )

        cum = 0
        ticks = [0]
        for seg_qpts in qpoints:
            cum += len(seg_qpts)
            ticks.append(cum - 1)
        ax.set_xticks(ticks[: len(labels)])
        ax.set_xticklabels(labels)
        for t in ticks:
            ax.axvline(x=t, color="gray", lw=0.5)

        ax.set_ylabel("Frequency (THz)")
        ax.set_title("Band structure: phonopy vs extracted blocks")
        ax.legend()
        ax.set_ylim(0, None)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close("all")

    return {"max_diff_thz": max_diff}


# ---- Reference transmission calculation ----


def _sancho_rubio(z2, H_00, H_01, max_iter=300, tol=1e-8):
    """Surface Green's function via Sancho-Rubio decimation.

    Parameters
    ----------
    z2 : complex
        Causal frequency squared: z² = (ω + iη_w)².
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
        if norm(alpha) + norm(beta) < tol:
            break

    return inv(eps_s)


def _ballistic_transmission(z2, H_D, H_L00, H_L01, H_R00, H_R01,
                            H_LD, H_DR):
    """Ballistic transmission via Caroli: T = Tr(Gamma_L G^R Gamma_R G^A).

    Parameters
    ----------
    z2 : complex
        Causal frequency squared: z² = (ω + iη_w)².
    """
    N_D = H_D.shape[0]

    g_L = _sancho_rubio(z2, H_L00, H_L01)
    g_R = _sancho_rubio(z2, H_R00, H_R01.conj().T)

    Sigma_L = H_LD.conj().T @ g_L @ H_LD
    Sigma_R = H_DR @ g_R @ H_DR.conj().T

    Gamma_L = 1j * (Sigma_L - Sigma_L.conj().T)
    Gamma_R = 1j * (Sigma_R - Sigma_R.conj().T)

    G_R = inv(z2 * np.eye(N_D) - H_D - Sigma_L - Sigma_R)
    G_A = G_R.conj().T

    T = np.real(np.trace(Gamma_L @ G_R @ Gamma_R @ G_A))
    return max(T, 0.0)


def reference_transmission(
        phonon: Phonopy,
        q_mesh_transverse: tuple[int, int],
        freq_range_thz: tuple[float, float, int] = (0.01, 16.0, 201),
        transport_direction: str = "z",
        eta_factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reference ballistic transmission via Sancho-Rubio + Caroli.

    Uses THz^2 internal units.

    Parameters
    ----------
    phonon : Phonopy
        Phonopy object with force constants.
    q_mesh_transverse : (Nkx, Nky)
        Transverse Monkhorst-Pack mesh.
    freq_range_thz : (fmin, fmax, nfreq)
        Frequency grid in THz.
    transport_direction : str
        "x", "y", or "z".
    eta_factor : float
        eta = dw^2 * eta_factor, where dw is frequency spacing in THz.

    Returns
    -------
    freqs_thz : ndarray
    transmission : ndarray
    """
    fmin, fmax, nfreq = freq_range_thz
    freqs_thz = np.linspace(fmin, fmax, int(nfreq))
    dw = freqs_thz[1] - freqs_thz[0]  # THz
    eta_w = dw * eta_factor  # THz
    z2_arr = (freqs_thz + 1j * eta_w) ** 2

    nkx, nky = q_mesh_transverse
    q_1d = [(2 * n - nkx - 1) / (2 * nkx) for n in range(1, nkx + 1)]
    q_1d_y = [(2 * n - nky - 1) / (2 * nky) for n in range(1, nky + 1)]
    q_points = [(qx, qy) for qx in q_1d for qy in q_1d_y]

    trans = np.zeros(int(nfreq))

    for qx, qy in q_points:
        H_00, H_01 = get_btd_blocks(
            phonon, (qx, qy), transport_direction=transport_direction,
            conversion_factor=CONVERSION_THZ2,
        )
        for iw, z2 in enumerate(z2_arr):
            trans[iw] += _ballistic_transmission(
                z2, H_00, H_00, H_01, H_00, H_01, H_01, H_01
            )

    trans /= len(q_points)
    return freqs_thz, trans


def interface_transmission(
        phonon_left: Phonopy,
        phonon_right: Phonopy,
        phonon_device: Phonopy,
        q_mesh_transverse: tuple[int, int],
        freq_range_thz: tuple[float, float, int] = (0.01, 16.0, 201),
        transport_direction: str = "z",
        eta_factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ballistic transmission through an interface.

    Uses THz^2 internal units.

    Parameters
    ----------
    phonon_left : Phonopy
    phonon_right : Phonopy
    phonon_device : Phonopy
    q_mesh_transverse : (Nkx, Nky)
    freq_range_thz : (fmin, fmax, nfreq)
    transport_direction : str
    eta_factor : float

    Returns
    -------
    freqs_thz : ndarray
    transmission : ndarray
    """
    fmin, fmax, nfreq = freq_range_thz
    nfreq = int(nfreq)
    freqs_thz = np.linspace(fmin, fmax, nfreq)
    dw = freqs_thz[1] - freqs_thz[0]
    eta_w = dw * eta_factor
    z2_arr = (freqs_thz + 1j * eta_w) ** 2

    nkx, nky = q_mesh_transverse
    q_1d_x = [(2 * n - nkx - 1) / (2 * nkx) for n in range(1, nkx + 1)]
    q_1d_y = [(2 * n - nky - 1) / (2 * nky) for n in range(1, nky + 1)]
    q_points = [(qx, qy) for qx in q_1d_x for qy in q_1d_y]

    trans = np.zeros(nfreq)

    for iq, (qx, qy) in enumerate(q_points):
        if (iq + 1) % 50 == 0 or iq == 0:
            print(f"  k-point {iq + 1}/{len(q_points)}")

        qp = (qx, qy)
        H_00_L, H_01_L = get_btd_blocks(
            phonon_left, qp, transport_direction=transport_direction,
            conversion_factor=CONVERSION_THZ2,
        )
        H_00_R, H_01_R = get_btd_blocks(
            phonon_right, qp, transport_direction=transport_direction,
            conversion_factor=CONVERSION_THZ2,
        )
        H_00_D, _ = get_btd_blocks(
            phonon_device, qp, transport_direction=transport_direction,
            conversion_factor=CONVERSION_THZ2,
        )

        for iw, z2 in enumerate(z2_arr):
            trans[iw] += _ballistic_transmission(
                z2,
                H_D=H_00_D,
                H_L00=H_00_L,
                H_L01=H_01_L,
                H_R00=H_00_R,
                H_R01=H_01_R,
                H_LD=H_01_L,
                H_DR=H_01_R,
            )

    trans /= len(q_points)
    return freqs_thz, trans


def thermal_conductance(
        freqs_thz: np.ndarray,
        transmission: np.ndarray,
        temperature_k: float,
        lattice_vectors: np.ndarray,
        transport_direction: str = "z",
) -> float:
    """Ballistic thermal conductance per unit area [W/(m^2 K)].

    Parameters
    ----------
    freqs_thz : ndarray
        Frequency grid in THz.
    transmission : ndarray
        Transmission function.
    temperature_k : float
        Temperature in Kelvin.
    lattice_vectors : ndarray, shape (3, 3)
        Lattice vectors in Angstrom.
    transport_direction : str

    Returns
    -------
    G : float
        Thermal conductance in W/(m^2 K).
    """
    tidx = "xyz".index(transport_direction)
    perp = [i for i in range(3) if i != tidx]
    a1 = lattice_vectors[perp[0]]
    a2 = lattice_vectors[perp[1]]
    A_c = np.linalg.norm(np.cross(a1, a2)) * 1e-20  # Angstrom^2 -> m^2

    omegas_rad = freqs_thz * THZ_TO_RAD
    hw = HBAR_EV * omegas_rad  # eV
    x = hw / (KB_EV * temperature_k)

    dfdt = np.zeros_like(x)
    valid = (x > 1e-10) & (x < 500)
    dfdt[valid] = (hw[valid] / (KB_EV * temperature_k ** 2)) * (
            np.exp(x[valid]) / (np.exp(x[valid]) - 1) ** 2
    )

    # Integration: sum * dw_thz * 1e12 (since dw_rad/(2*pi) = dw_thz * 1e12)
    dw_thz = freqs_thz[1] - freqs_thz[0]
    integrand = transmission * hw * dfdt
    return float(np.sum(integrand) * dw_thz * 1e12 * EV_TO_J / A_c)
