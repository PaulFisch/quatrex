"""Transport observables for the cutoff-sensitivity sweep.

The full quatrex observable stack (``src/quatrex/core/observables.py``) is
MPI-distributed and tied to the ``DSDBSparse`` runtime. For our finite-
structure validation we want plain-NumPy versions that consume the
:class:`SSEResult` produced by :mod:`sse_cutoffs` and produce three
scalar/spectral observables:

  * ``dos(omega)`` — total density of states from ``Im Tr G^R``.
  * ``ballistic_transmission(omega)`` — Caroli transmission with a
    synthetic broadening matrix ``Γ_lead`` placed on the first and last
    slab. This is *not* a full lead self-energy, but it gives a
    well-defined T(ω) that responds linearly to changes in the device
    self-energy.
  * ``landauer_heat_current(T, ω, T_L, T_R)`` — integrated phonon heat
    current at fixed lead temperatures.

The harness then maps changes in Σ from the cutoff sweep into changes in
T(ω) and Q, answering "what cutoffs preserve transport observables?"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phonon_inputs.constants import HBAR_SI, KB_SI, THZ_TO_RAD

from .loader import SystemBundle
from .synthetic_gf import dynamical_matrix


# --------------------------------------------------------------------------- #
# Σ^R from Σ^{<,>}                                                            #
# --------------------------------------------------------------------------- #


def sigma_retarded_half(
    sigma_lesser: np.ndarray, sigma_greater: np.ndarray,
) -> np.ndarray:
    """Σ^R(ω) ≈ ½(Σ^>(ω) − Σ^<(ω)). The 'half' approximation: imaginary
    part only, no Kramers-Kronig real-part reconstruction. Use
    :func:`sigma_retarded` (default ``method="fft"``) for absolute T(ω) /
    Q values; the half approximation is fine for comparing *changes*
    across cutoff configurations.
    """
    return 0.5 * (sigma_greater - sigma_lesser)


def sigma_retarded(
    sigma_lesser: np.ndarray, sigma_greater: np.ndarray,
    omega_grid_thz: np.ndarray,
    *,
    method: str = "fft",
) -> np.ndarray:
    """Build Σ^R(ω) from Σ^{<,>} via one of three reconstructions.

    Parameters
    ----------
    method : {"fft", "pv", "half"}
        ``"fft"`` (default): ``Σ^R = ½Δ + ½i · H[Δ]`` where ``H`` is the
        FFT Hilbert transform along ω. Captures both the imaginary
        (broadening) and real (level-shift / phonon renormalisation)
        parts of Σ^R.

        ``"pv"``: same physics via singularity-subtracted principal-value
        quadrature. More accurate at endpoints, slower (O(n_freq²)).

        ``"half"``: imaginary part only (Σ^R = ½Δ). Inadequate for
        absolute T(ω) but cheap; useful for comparing the *change* in Σ
        across cutoff configurations.

    See :func:`phonon.solver.retarded.build_retarded` (this delegates to it).
    """
    from solver.retarded import build_retarded as _build_retarded
    return _build_retarded(sigma_lesser, sigma_greater, omega_grid_thz, method=method)


def block_dict_to_dense(
    sigma_blocks: dict[tuple[int, int], np.ndarray],
    block_sizes: np.ndarray,
) -> np.ndarray:
    """Reassemble a block-tridiagonal Σ dict into a dense ``(n_freq, N, N)``."""
    block_sizes = np.asarray(block_sizes, dtype=int)
    offsets = np.concatenate(([0], np.cumsum(block_sizes)))
    n_freq = next(iter(sigma_blocks.values())).shape[0]
    N = int(offsets[-1])
    out = np.zeros((n_freq, N, N), dtype=complex)
    for (I, J), block in sigma_blocks.items():
        out[:, offsets[I]:offsets[I + 1], offsets[J]:offsets[J + 1]] = block
    return out


# --------------------------------------------------------------------------- #
# Dense Green's function with a synthetic lead broadening                     #
# --------------------------------------------------------------------------- #


def _lead_broadening(
    n_dof: int, block_sizes: np.ndarray, gamma_thz: float,
) -> np.ndarray:
    """Diagonal broadening Γ that's nonzero only on the first and last slab.

    SYNTHETIC fallback when the Sancho-Rubio lead self-energy isn't
    available. Returns ``Γ_L, Γ_R`` each of shape ``(n_dof, n_dof)``.
    """
    block_sizes = np.asarray(block_sizes, dtype=int)
    sl_L = slice(0, int(block_sizes[0]))
    sl_R = slice(n_dof - int(block_sizes[-1]), n_dof)
    Gamma_L = np.zeros((n_dof, n_dof))
    Gamma_R = np.zeros_like(Gamma_L)
    np.fill_diagonal(Gamma_L[sl_L, sl_L], gamma_thz)
    np.fill_diagonal(Gamma_R[sl_R, sl_R], gamma_thz)
    return Gamma_L, Gamma_R


def sancho_rubio_lead_self_energies(
    bundle: SystemBundle,
    omega_grid_thz: np.ndarray,
    *,
    eta_thz: float = 0.05,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the (left, right) phonon lead self-energies via Sancho-Rubio
    decimation on the bundle's first/last transport slabs.

    The "lead" model: take ``H_00 = D[slab_0, slab_0]`` (mass-weighted
    intra-slab block) and ``H_01 = D[slab_0, slab_1]`` (inter-slab
    coupling) as the period of a semi-infinite chain. The surface Green's
    function ``g_lead(z² = (ω+iη)²)`` is built by the Sancho-Rubio
    iteration; the lead self-energy on the device side is then
    ``Σ_L^R(ω) = H_01† g_lead(ω) H_01`` (left), and analogously with
    ``H_10 = H_01†`` for the right.

    Returns
    -------
    Sigma_L_R, Sigma_R_R : (n_freq, n_dof, n_dof) complex
        Σ_L^R and Σ_R^R, both nonzero only on the first/last slab block
        (zero everywhere else). Frequencies where Sancho-Rubio failed to
        converge are filled with NaN; the caller can fall back to the
        synthetic broadening for those.
    """
    from solver.leads import sancho_rubio_batch as _sancho_rubio_batch
    from .synthetic_gf import dynamical_matrix

    D = dynamical_matrix(bundle, z_sorted=True)
    block_sizes = np.asarray(bundle.block_sizes, dtype=int)
    offsets = np.concatenate(([0], np.cumsum(block_sizes)))
    n_dof = D.shape[0]
    n_freq = omega_grid_thz.size

    # Left lead: slab 0 + coupling to slab 1.
    sL = slice(offsets[0], offsets[1])
    sL_next = slice(offsets[1], offsets[2]) if block_sizes.size >= 2 else sL
    H_00_L = D[sL, sL]
    H_01_L = D[sL, sL_next]

    # Right lead: slab N-1 + coupling to slab N-2 (reverse propagation).
    sR = slice(offsets[-2], offsets[-1])
    sR_prev = slice(offsets[-3], offsets[-2]) if block_sizes.size >= 2 else sR
    H_00_R = D[sR, sR]
    H_01_R = D[sR, sR_prev]

    z2 = (omega_grid_thz + 1j * eta_thz) ** 2

    # The batch implementation can hit singular eps matrices at frequencies
    # within an eigenmode's broadening; iterate per-frequency with a fallback.
    g_L = np.empty((omega_grid_thz.size, H_00_L.shape[0], H_00_L.shape[1]), dtype=complex)
    g_R = np.empty((omega_grid_thz.size, H_00_R.shape[0], H_00_R.shape[1]), dtype=complex)
    valid_L = np.zeros(omega_grid_thz.size, dtype=bool)
    valid_R = np.zeros(omega_grid_thz.size, dtype=bool)
    try:
        g_L_batch, valid_L_batch = _sancho_rubio_batch(
            z2, H_00_L, H_01_L, max_iter=max_iter, tol=tol)
        g_L[:] = g_L_batch
        valid_L[:] = valid_L_batch
    except np.linalg.LinAlgError:
        # Per-ω fallback for singular cases.
        from solver.leads import sancho_rubio as _sancho_rubio
        for iw in range(omega_grid_thz.size):
            try:
                g_L[iw] = _sancho_rubio(z2[iw], H_00_L, H_01_L,
                                         max_iter=max_iter, tol=tol)
                valid_L[iw] = True
            except (np.linalg.LinAlgError, RuntimeError):
                g_L[iw] = np.nan
                valid_L[iw] = False
    try:
        g_R_batch, valid_R_batch = _sancho_rubio_batch(
            z2, H_00_R, H_01_R, max_iter=max_iter, tol=tol)
        g_R[:] = g_R_batch
        valid_R[:] = valid_R_batch
    except np.linalg.LinAlgError:
        from solver.leads import sancho_rubio as _sancho_rubio
        for iw in range(omega_grid_thz.size):
            try:
                g_R[iw] = _sancho_rubio(z2[iw], H_00_R, H_01_R,
                                         max_iter=max_iter, tol=tol)
                valid_R[iw] = True
            except (np.linalg.LinAlgError, RuntimeError):
                g_R[iw] = np.nan
                valid_R[iw] = False

    Sigma_L_R = np.zeros((n_freq, n_dof, n_dof), dtype=complex)
    Sigma_R_R = np.zeros((n_freq, n_dof, n_dof), dtype=complex)
    H_10_L = H_01_L.conj().T
    H_10_R = H_01_R.conj().T
    for iw in range(n_freq):
        if valid_L[iw]:
            Sigma_L_R[iw, sL, sL] = H_10_L @ g_L[iw] @ H_01_L
        else:
            Sigma_L_R[iw] = np.nan
        if valid_R[iw]:
            Sigma_R_R[iw, sR, sR] = H_10_R @ g_R[iw] @ H_01_R
        else:
            Sigma_R_R[iw] = np.nan

    return Sigma_L_R, Sigma_R_R


def _z2_grid(freqs_thz: np.ndarray, eta_thz: float) -> np.ndarray:
    """``(ω + iη)²`` along the frequency axis (matches the standalone
    convention in :mod:`phonon.solver.grids`)."""
    return (freqs_thz + 1j * eta_thz) ** 2


@dataclass
class TransportTrace:
    omega_thz: np.ndarray
    transmission: np.ndarray  # T(ω), real
    dos: np.ndarray           # A(ω), real
    heat_current_W: float


def transport_trace_from_sigma(
    bundle: SystemBundle,
    sigma_lesser: dict[tuple[int, int], np.ndarray],
    sigma_greater: dict[tuple[int, int], np.ndarray],
    omega_grid_thz: np.ndarray,
    *,
    eta_thz: float = 0.05,
    gamma_lead_thz: float = 0.5,
    T_L: float = 305.0,
    T_R: float = 295.0,
    sigma_retarded_method: str = "fft",
    lead_model: str = "sancho_rubio",
) -> TransportTrace:
    """Compute ``T(ω)``, ``A(ω)``, and the integrated heat current.

    **Lead model.** ``lead_model="sancho_rubio"`` (default) builds the true
    semi-infinite lead self-energy by Sancho-Rubio decimation on the
    first / last transport slab — i.e. the lead is treated as the same
    slab repeated periodically. ``"synthetic"`` falls back to a scalar
    wide-band Γ = γ · I on the first/last slab DOFs (only useful for
    cutoff-sweep relative diagnostics).

    Parameters
    ----------
    bundle
        System for which Σ was computed; provides FC2 and the slab decomposition.
    sigma_lesser, sigma_greater
        Block-tridiagonal Σ dicts (z-sorted DOF order).
    omega_grid_thz
        The frequency grid Σ was tabulated on (the symmetric
        ``_build_frequency_grid`` axis).
    eta_thz
        Numerical broadening for the bare ``z² - D`` inverse.
    gamma_lead_thz
        Synthetic lead broadening — used only when ``lead_model="synthetic"``.
    T_L, T_R
        Lead temperatures in K.
    sigma_retarded_method : {"fft", "pv", "half"}
        How to reconstruct Σ^R from (Σ^<, Σ^>). ``"fft"`` (default)
        captures both broadening (Im Σ^R) and level-shift (Re Σ^R via
        Hilbert). ``"half"`` drops the level-shift; OK for cutoff diffs,
        not for absolute transport.
    lead_model : {"sancho_rubio", "synthetic"}
        See above.
    """
    D = dynamical_matrix(bundle, z_sorted=True)
    n_dof = D.shape[0]
    if lead_model == "sancho_rubio":
        Sigma_L_lead_R, Sigma_R_lead_R = sancho_rubio_lead_self_energies(
            bundle, omega_grid_thz, eta_thz=eta_thz,
        )
    elif lead_model == "synthetic":
        Gamma_L, Gamma_R = _lead_broadening(n_dof, bundle.block_sizes, gamma_lead_thz)
        Sigma_L_lead_R = -0.5j * np.broadcast_to(Gamma_L, (omega_grid_thz.size, n_dof, n_dof))
        Sigma_R_lead_R = -0.5j * np.broadcast_to(Gamma_R, (omega_grid_thz.size, n_dof, n_dof))
    else:
        raise ValueError(
            f"lead_model must be 'sancho_rubio' or 'synthetic', got {lead_model!r}"
        )

    # Reassemble Σ^{<,>} into dense (n_freq, N, N) so the Hilbert/PV
    # reconstruction can act on the full frequency axis at once.
    Sigma_l_dense = block_dict_to_dense(sigma_lesser, bundle.block_sizes)
    Sigma_g_dense = block_dict_to_dense(sigma_greater, bundle.block_sizes)
    Sigma_R_dense = sigma_retarded(
        Sigma_l_dense, Sigma_g_dense, omega_grid_thz,
        method=sigma_retarded_method,
    )

    z2 = _z2_grid(omega_grid_thz, eta_thz)
    n_freq = omega_grid_thz.size
    transmission = np.zeros(n_freq)
    dos = np.zeros(n_freq)
    eye = np.eye(n_dof)

    for iw, w in enumerate(omega_grid_thz):
        if w <= 0:
            continue
        sl_iw = Sigma_L_lead_R[iw]
        sr_iw = Sigma_R_lead_R[iw]
        if np.any(np.isnan(sl_iw)) or np.any(np.isnan(sr_iw)):
            # Sancho-Rubio failed at this ω — fall back to a small η bath.
            sl_iw = np.zeros((n_dof, n_dof), dtype=complex)
            sr_iw = np.zeros((n_dof, n_dof), dtype=complex)
        # Γ_α = i (Σ_α^R − (Σ_α^R)†)
        Gamma_L_iw = 1j * (sl_iw - sl_iw.conj().T)
        Gamma_R_iw = 1j * (sr_iw - sr_iw.conj().T)
        G_R = np.linalg.inv(
            z2[iw] * eye - D - Sigma_R_dense[iw] - sl_iw - sr_iw
        )
        G_A = G_R.conj().T
        transmission[iw] = float(np.real(np.trace(
            Gamma_L_iw @ G_R @ Gamma_R_iw @ G_A
        )))
        dos[iw] = float(-1.0 / np.pi * np.imag(np.trace(G_R)))

    # Landauer heat current Q = ∫₀^∞ dω/(2π) ℏω T(ω) [n_L(ω) - n_R(ω)]
    pos = omega_grid_thz > 0.0
    omega_pos = omega_grid_thz[pos] * THZ_TO_RAD  # rad/s
    T_pos = transmission[pos]
    n_L = 1.0 / np.expm1(HBAR_SI * omega_pos / (KB_SI * T_L))
    n_R = 1.0 / np.expm1(HBAR_SI * omega_pos / (KB_SI * T_R))
    integrand = HBAR_SI * omega_pos * T_pos * (n_L - n_R)
    if omega_pos.size > 1:
        dw_grid_rad = (omega_grid_thz[1] - omega_grid_thz[0]) * THZ_TO_RAD
        dw = float(omega_pos[1] - omega_pos[0])
        # _build_frequency_grid produces uniform Δω; the positive subset has
        # the same spacing because we filter by the symmetric mask, not strided.
        assert abs(dw - dw_grid_rad) < 1e-9 * abs(dw_grid_rad), (
            f"Heat-current dω mismatch: positive-subset spacing {dw:.3e} rad/s "
            f"vs symmetric-grid spacing {dw_grid_rad:.3e} rad/s. "
            "Check _build_frequency_grid uniformity."
        )
    else:
        dw = 0.0
    heat_current = float(np.sum(integrand) * dw / (2.0 * np.pi))

    return TransportTrace(
        omega_thz=omega_grid_thz,
        transmission=transmission,
        dos=dos,
        heat_current_W=heat_current,
    )


# --------------------------------------------------------------------------- #
# Self-consistent SCBA wrapper                                                #
# --------------------------------------------------------------------------- #


def transport_trace_scba(
    bundle: SystemBundle,
    phi_blocks: dict[tuple[int, int, int], np.ndarray],
    omega_grid_thz: np.ndarray,
    *,
    n_iter: int = 3,
    mixing: float = 0.5,
    temperature_k: float = 300.0,
    eta_thz: float = 0.05,
    gamma_lead_thz: float = 0.5,
    T_L: float = 305.0,
    T_R: float = 295.0,
    sigma_retarded_method: str = "fft",
    lead_model: str = "sancho_rubio",
) -> tuple[TransportTrace, list[float]]:
    """Self-consistent Born-approximation (SCBA) transport.

    Iterates Σ ↔ G via Dyson + equilibrium-Keldysh closure with linear
    mixing. Stops after ``n_iter`` iterations or when the relative-
    Frobenius change in Σ^< drops below 1e-3. Returns the final
    :class:`TransportTrace` and the per-iteration rel-change history.

    Algorithm (per iteration):
      1. Σ^R = full reconstruction (Hilbert) of (Σ^>, Σ^<).
      2. G^R(ω) = [(ω+iη)² · I − D − Σ^R(ω)]^{-1} per ω (dense Dyson).
      3. Γ(ω) = i (Σ^R − Σ^A) — anharmonic broadening only.
      4. Equilibrium FDT closure on the device:
         G^<(ω) = i n_B(ω) (G^R Γ G^A),  G^>(ω) = -i (n_B(ω)+1) (G^R Γ G^A).
      5. Recompute Σ^{<,>}_phph from new G via the bubble.
      6. Mix with previous Σ.

    The equilibrium FDT closure is a simplification: it treats the
    device as if it sat at a uniform T (fluctuation-dissipation), not
    the true non-equilibrium two-lead steady state. For absolute
    transport between two reservoirs at T_L ≠ T_R the proper closure
    couples Γ_lead with the lead temperatures; this is what
    :func:`phonon.solver.transmission_finite` does
    (Sancho-Rubio leads + the full Keldysh system per ω). Here we focus
    on Σ self-consistency and use the simpler closure to keep the SCBA
    loop tractable; the final transport trace is then computed by
    :func:`transport_trace_from_sigma` with real leads.
    """
    from phonon_inputs.constants import HBAR_SI, KB_SI, THZ_TO_RAD

    from .sse_cutoffs import compute_sse_with_cutoffs
    from .synthetic_gf import dynamical_matrix, synthetic_gf_dense, gf_to_block_dict

    dw = float(omega_grid_thz[1] - omega_grid_thz[0])
    n_freq = omega_grid_thz.size
    D = dynamical_matrix(bundle, z_sorted=True)
    n_dof = D.shape[0]
    eye = np.eye(n_dof)
    z2 = (omega_grid_thz + 1j * eta_thz) ** 2

    # n_B(ω) on the symmetric grid — handle ω→0 carefully (replace by limit).
    x = HBAR_SI * np.abs(omega_grid_thz) * THZ_TO_RAD / (KB_SI * temperature_k)
    n_B = np.zeros_like(omega_grid_thz, dtype=float)
    safe = x > 1e-12
    n_B[safe] = 1.0 / np.expm1(x[safe])
    # Sign branch: n_B(-ω) = -1 - n_B(ω) for bosonic full-axis use.
    n_B = np.where(omega_grid_thz < 0, -1.0 - n_B, n_B)

    # Initial Σ from one bubble pass on the synthetic eigenmode G.
    G_l_dense, G_g_dense, _, _, _ = synthetic_gf_dense(
        bundle, n_freq_pos=(n_freq - 1) // 2,
        eta_thz=eta_thz, temperature_k=temperature_k,
    )
    gl_blocks = gf_to_block_dict(G_l_dense, bundle.block_sizes, nn_only=False)
    gg_blocks = gf_to_block_dict(G_g_dense, bundle.block_sizes, nn_only=False)
    res = compute_sse_with_cutoffs(
        phi_blocks, gl_blocks, gg_blocks, bundle.block_sizes, dw,
    )
    sigma_l = res.sigma_lesser
    sigma_g = res.sigma_greater

    rel_changes: list[float] = []
    for _ in range(max(n_iter, 0)):
        Sigma_l_dense = block_dict_to_dense(sigma_l, bundle.block_sizes)
        Sigma_g_dense = block_dict_to_dense(sigma_g, bundle.block_sizes)
        Sigma_R = sigma_retarded(
            Sigma_l_dense, Sigma_g_dense, omega_grid_thz,
            method=sigma_retarded_method,
        )

        # Dyson per ω.
        G_R = np.empty((n_freq, n_dof, n_dof), dtype=complex)
        for iw in range(n_freq):
            G_R[iw] = np.linalg.inv(z2[iw] * eye - D - Sigma_R[iw])
        G_A = G_R.conj().swapaxes(-1, -2)

        # Γ (anharmonic-only) and equilibrium G^{<,>}.
        Gamma = 1j * (Sigma_R - Sigma_R.conj().swapaxes(-1, -2))
        GRGGA = G_R @ Gamma @ G_A
        G_l_new = 1j * n_B[:, None, None] * GRGGA
        G_g_new = -1j * (n_B[:, None, None] + 1.0) * GRGGA

        gl_blocks = gf_to_block_dict(G_l_new, bundle.block_sizes, nn_only=False)
        gg_blocks = gf_to_block_dict(G_g_new, bundle.block_sizes, nn_only=False)
        res_new = compute_sse_with_cutoffs(
            phi_blocks, gl_blocks, gg_blocks, bundle.block_sizes, dw,
        )

        # Mix.
        new_l: dict = {}
        new_g: dict = {}
        for k in set(sigma_l) | set(res_new.sigma_lesser):
            new_l[k] = ((1.0 - mixing) * sigma_l.get(k, 0)
                        + mixing * res_new.sigma_lesser.get(k, 0))
        for k in set(sigma_g) | set(res_new.sigma_greater):
            new_g[k] = ((1.0 - mixing) * sigma_g.get(k, 0)
                        + mixing * res_new.sigma_greater.get(k, 0))

        # Convergence: relative-Frobenius change in Σ^<.
        denom = sum(np.linalg.norm(v) ** 2 for v in sigma_l.values()) ** 0.5
        diff = sum(
            np.linalg.norm(new_l[k] - sigma_l.get(k, 0)) ** 2
            for k in new_l
        ) ** 0.5
        rel_change = float(diff / max(denom, 1e-30))
        rel_changes.append(rel_change)
        sigma_l, sigma_g = new_l, new_g
        if rel_change < 1e-3:
            break

    trace = transport_trace_from_sigma(
        bundle, sigma_l, sigma_g, omega_grid_thz,
        eta_thz=eta_thz, gamma_lead_thz=gamma_lead_thz,
        T_L=T_L, T_R=T_R,
        sigma_retarded_method=sigma_retarded_method,
        lead_model=lead_model,
    )
    return trace, rel_changes


# --------------------------------------------------------------------------- #
# Plot                                                                        #
# --------------------------------------------------------------------------- #


def plot_transport_compare(
    traces: dict[str, TransportTrace], out_path: Path, system_name: str = "",
) -> None:
    """Overlay T(ω) for several cutoff configurations + bar of Q."""
    labels = list(traces.keys())
    fig, (ax_t, ax_q) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    cmap = plt.colormaps["tab10"]

    for idx, label in enumerate(labels):
        tr = traces[label]
        pos = tr.omega_thz > 0
        ax_t.semilogy(
            tr.omega_thz[pos], np.maximum(tr.transmission[pos], 1e-20),
            label=label, color=cmap(idx % 10), lw=1.2,
        )
    ax_t.set_xlabel(r"$\omega$  [THz]")
    ax_t.set_ylabel(r"$T(\omega)$")
    # The transmission curves below include the phonon-phonon SE built
    # from the cutoff-truncated bubble inserted into the Dyson equation —
    # not the pristine harmonic transmission. Naming is intentionally
    # "with SE inserted" so the absolute scale (which depends on Γ_lead
    # and the half-Hilbert Σ^R choice) is not mistaken for the
    # harmonic-Caroli reference.
    ax_t.set_title(f"Transmission with phonon-phonon SE — {system_name}")
    ax_t.grid(alpha=0.3, which="both")
    ax_t.legend(fontsize=8)

    ax_q.bar(labels, [traces[l].heat_current_W for l in labels])
    ax_q.set_ylabel(r"$Q$  [W]  ($T_L=305$\,K, $T_R=295$\,K)")
    ax_q.set_title("Integrated heat current")
    ax_q.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver hook for the cutoff sweep                                            #
# --------------------------------------------------------------------------- #


def transport_for_cutoff_sweep(
    bundle: SystemBundle,
    sweep_results: dict[str, "SSEResult"],  # type: ignore[name-defined]
    omega_grid_thz: np.ndarray,
    *,
    eta_thz: float = 0.05,
    gamma_lead_thz: float = 0.5,
    T_L: float = 305.0,
    T_R: float = 295.0,
    sigma_retarded_method: str = "fft",
) -> dict[str, TransportTrace]:
    """Run :func:`transport_trace_from_sigma` for every entry of a sweep."""
    out: dict[str, TransportTrace] = {}
    for label, res in sweep_results.items():
        out[label] = transport_trace_from_sigma(
            bundle, res.sigma_lesser, res.sigma_greater,
            omega_grid_thz,
            eta_thz=eta_thz, gamma_lead_thz=gamma_lead_thz,
            T_L=T_L, T_R=T_R,
            sigma_retarded_method=sigma_retarded_method,
        )
    return out
