"""Synthetic phonon Green's function for SSE sparsity / cutoff sweeps.

Builds a representative ``G^{<,>}(ω)`` from the harmonic eigenmodes of the
finite supercell — no SCBA, no contacts, no leads. This is the cheapest
"reasonable" Green's function the bubble can be evaluated against, suitable
for sweeping cutoff approximations against a fixed input.

Convention follows :mod:`phonon_inputs.anharmonic` (frequencies in THz on a
symmetric grid built by :func:`_build_frequency_grid`). The eigenmode form
of the bosonic Green's function on this grid is

    G^<(ω) = Σ_n  (ε_n ε_n^T) / (2 ω_n) × {
                 n_B(ω_n)         × L_η(ω − ω_n)
               + (n_B(ω_n) + 1)   × L_η(ω + ω_n)  }
    G^>(ω) = G^<(ω)  with  n_B ↔ (n_B + 1)  at the corresponding peaks,

where ``L_η(x) = (η/π) / (x² + η²)`` is the Lorentzian on ω. Only positive
``ω_n`` modes contribute; imaginary modes (FC2 instabilities) are flagged
and skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from phonon_inputs.constants import AMU_KG, CONVERSION, THZ_TO_RAD
from phonon_inputs.anharmonic import _build_frequency_grid, _bose_full_axis

from ._utils import expand_atom_perm_to_dofs
from .loader import SystemBundle


# --------------------------------------------------------------------------- #
# Eigenmode decomposition                                                     #
# --------------------------------------------------------------------------- #


def _supercell_dynamical_matrix_thz2(bundle: SystemBundle) -> np.ndarray:
    """Mass-weighted dynamical matrix D (THz²), unsorted (supercell DOF order).

    D_{i α, j β} = Φ₂_{ij,αβ} / sqrt(m_i m_j) × (CONVERSION / THZ_TO_RAD²)

    Returns a real symmetric ``(3 n_super, 3 n_super)`` matrix in THz²
    addressed by the *original* supercell atom order. Use
    :func:`dynamical_matrix` for the z-sorted convention that matches
    ``bundle.block_sizes``.
    """
    fc2 = bundle.fc2  # (n, n, 3, 3) in eV/Å²
    n = fc2.shape[0]
    masses = bundle.masses  # amu
    inv_sqrt_m = 1.0 / np.sqrt(masses)
    weight = inv_sqrt_m[:, None] * inv_sqrt_m[None, :]

    D_eVA2_per_amu = fc2 * weight[:, :, None, None]
    # eV/Å²/amu -> (rad/s)² -> (THz)² by /(2π×10¹²)²
    D_thz2 = D_eVA2_per_amu * CONVERSION / (THZ_TO_RAD ** 2)

    D = D_thz2.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    D = 0.5 * (D + D.T)
    return D


def dynamical_matrix(bundle: SystemBundle, *, z_sorted: bool = True) -> np.ndarray:
    """Mass-weighted dynamical matrix in THz².

    With ``z_sorted=True`` (default), DOFs are permuted into z-sorted
    order so contiguous slices of ``bundle.block_sizes`` address contiguous
    slabs — the convention every other module in :mod:`finite_analysis`
    operates in.
    """
    D = _supercell_dynamical_matrix_thz2(bundle)
    if not z_sorted:
        return D
    perm = expand_atom_perm_to_dofs(bundle.atom_perm)
    return D[np.ix_(perm, perm)]


@dataclass
class HarmonicModes:
    """Eigenmodes of the supercell dynamical matrix.

    ``omega_thz`` carries the *signed* sqrt of the eigenvalues, so unstable
    modes (Ω² < 0) appear as negative entries. Downstream Green's-function
    routines use only the positive non-acoustic subset.
    """

    omega_thz: np.ndarray   # (3n,) — signed √(eigvals(D)), THz
    polarisation: np.ndarray  # (3n, 3n) — column-wise normalised eigenvectors
    n_acoustic: int   # |ω| ≤ THZ_ACOUSTIC_BAND — translational/rotational zeros
    n_imaginary: int  # ω < -THZ_ACOUSTIC_BAND — FC2 instabilities (concerning)
    n_unstable: int = 0  # DEPRECATED: kept for one release; equals n_imaginary


def diagonalise(bundle: SystemBundle) -> HarmonicModes:
    """Diagonalise the supercell dynamical matrix.

    Splits the spectrum into three categories:
      * **acoustic**: |ω| ≤ THZ_ACOUSTIC_BAND — true Γ-zeros from
        translational/rotational invariance.
      * **imaginary**: ω < -THZ_ACOUSTIC_BAND — FC2 indefinite, signals
        either an unrelaxed structure or insufficient FC2 cutoff. A
        ``UserWarning`` is emitted whenever any imaginary modes are found.
      * **physical**: |ω| > THZ_ACOUSTIC_BAND with ω > 0 — used by
        :func:`synthetic_gf_dense` to build G^{<,>}.
    """
    import warnings as _warnings
    from .constants import THZ_ACOUSTIC_BAND

    D = _supercell_dynamical_matrix_thz2(bundle)
    eigvals, eigvecs = np.linalg.eigh(D)
    omega = np.sign(eigvals) * np.sqrt(np.abs(eigvals))

    n_acoustic = int(np.sum(np.abs(omega) <= THZ_ACOUSTIC_BAND))
    n_imaginary = int(np.sum(omega < -THZ_ACOUSTIC_BAND))
    if n_imaginary > 0:
        _warnings.warn(
            f"Synthetic GF: {n_imaginary} imaginary modes "
            f"(min ω = {float(omega.min()):.3f} THz). "
            "FC2 is indefinite — either the structure is unrelaxed or the "
            "FC2 cutoff is too short. These modes are excluded from G^{<,>}.",
            stacklevel=2,
        )
    return HarmonicModes(
        omega_thz=omega,
        polarisation=eigvecs,
        n_acoustic=n_acoustic,
        n_imaginary=n_imaginary,
        n_unstable=n_imaginary,
    )


# --------------------------------------------------------------------------- #
# Frequency grid and Green's function                                         #
# --------------------------------------------------------------------------- #


def make_frequency_grid(
    omega_max_thz: float,
    n_freq_pos: int,
    *,
    eta_factor: float = 0.5,
    eta_w_thz: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Wraps :func:`_build_frequency_grid` and returns ``(freqs, dw, eta_w)``."""
    freqs, dw, eta, *_ = _build_frequency_grid(
        (-omega_max_thz, omega_max_thz, n_freq_pos),
        eta_w_thz=eta_w_thz,
        eta_factor=eta_factor,
    )
    return freqs, dw, eta


def _lorentzian(x: np.ndarray, eta: float) -> np.ndarray:
    return (eta / np.pi) / (x * x + eta * eta)


def synthetic_gf_dense(
    bundle: SystemBundle,
    *,
    omega_max_thz: float | None = None,
    n_freq_pos: int = 100,
    temperature_k: float = 300.0,
    eta_thz: float | None = None,
    drop_threshold_thz: float = 0.1,
    in_z_sorted_order: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, HarmonicModes]:
    """Eigenmode Green's function ``G^{<,>}`` on the symmetric ω grid.

    **Derivation** (cross-references to ``document/src/theory.tex``):

    1. FDT (eq. 873–874): ``G^<(ω) = -2i n_B(ω) Im G^R(ω)`` and
       ``G^>(ω) = -2i (1 + n_B(ω)) Im G^R(ω)``.
    2. Eigenmode expansion of the retarded propagator (mass-weighted basis):
       ``G^R(ω) = Σ_n (ε_n ε_n^T) / [(ω + iη)² − ω_n²]``,
       so ``Im G^R(ω) = -π Σ_n (ε_n ε_n^T)/(2ω_n)·[δ(ω−ω_n) − δ(ω+ω_n)]``.
    3. Substituting (2) into (1) and using ``n_B(−ω) = −1 − n_B(ω)`` gives
       the implemented form:

       ``G^<(ω) = -2iπ Σ_n (ε_n ε_n^T)/(2ω_n) · [n_B(ω_n) L_η(ω−ω_n) +
                                                  (n_B(ω_n)+1) L_η(ω+ω_n)]``

       where ``L_η`` is the Lorentzian replacement for the δ. ``G^>`` swaps
       the (n_B, n_B+1) Bose factors. The prefactor ``-2iπ`` is locked by
       the unit test ``test_synthetic_gf_correlator_normalisation`` which
       checks ``∫ iG^<dω/(2π)`` reproduces ``Σ_n |ε|²(n_B+½)/ω_n`` (the
       standard equal-time displacement correlator) to 5 % on the analytic
       8-atom chain.

    The Bose-factor asymmetry ((n_B, n_B+1) at +ω_n flips to (n_B+1, n_B)
    at −ω_n) realises the bosonic full-axis identity
    ``G^<(ω) = [G^>(−ω)]^T`` — pinned by ``test_synthetic_gf_bose_symmetry``.

    Parameters
    ----------
    bundle : :class:`SystemBundle`
    omega_max_thz : float, optional
        Upper edge of the frequency grid; defaults to ``1.05 × max(ω_n)``.
    n_freq_pos : int
        Number of positive-frequency bins.
    temperature_k : float
        Temperature for the Bose factor.
    eta_thz : float, optional
        Lorentzian half-width. Defaults to ``2 × dω``.
    drop_threshold_thz : float
        Modes with ``|ω_n| < drop_threshold_thz`` are excluded — they would
        otherwise diverge through the ``1/(2 ω_n)`` factor (acoustic Γ-modes).
    in_z_sorted_order : bool
        If True, the returned arrays are reordered so contiguous slabs of
        ``bundle.block_sizes`` correspond to contiguous z-slabs. Set False
        if you intend to feed the result back into pure ``anharmonic.py``
        (which doesn't care about slab order).

    Returns
    -------
    G_lesser : ``(n_freq, n_dof, n_dof)`` complex
    G_greater : ``(n_freq, n_dof, n_dof)`` complex
    freqs_thz : ``(n_freq,)``
    dw_thz : float
    modes : :class:`HarmonicModes` (for inspection / logging)
    """
    modes = diagonalise(bundle)
    n_dof = bundle.n_dof

    if omega_max_thz is None:
        omega_max_thz = 1.05 * float(np.max(np.abs(modes.omega_thz)))
        if omega_max_thz < 1.0:
            omega_max_thz = 1.0

    from .constants import (
        DEFAULT_DROP_THRESHOLD_REL, DEFAULT_ETA_FACTOR_DW,
        DEFAULT_ETA_THZ_FLOOR,
    )

    freqs, dw, eta = make_frequency_grid(
        omega_max_thz, n_freq_pos, eta_w_thz=eta_thz,
    )
    if eta_thz is None:
        eta = max(DEFAULT_ETA_FACTOR_DW * dw, DEFAULT_ETA_THZ_FLOOR)

    # Adaptive zero-mode filter: the absolute drop_threshold AND a relative
    # cut at 1e-3 × max|ω_n| both apply. The relative cut catches FP
    # zero-modes when max|ω| is anomalously small.
    omega_max_abs = float(np.max(np.abs(modes.omega_thz))) if modes.omega_thz.size else 0.0
    threshold = max(drop_threshold_thz,
                    DEFAULT_DROP_THRESHOLD_REL * omega_max_abs)
    keep = np.abs(modes.omega_thz) >= threshold
    omega_n = modes.omega_thz[keep]
    eps_n = modes.polarisation[:, keep]   # (n_dof, n_modes)

    n_B = _bose_full_axis(omega_n, temperature_k)
    np_factor = (n_B / (2.0 * omega_n))     # (n_modes,)
    nm1_factor = ((n_B + 1.0) / (2.0 * omega_n))

    # Build per-(ω, mode) Lorentzian weights; sum over modes to assemble.
    G_l = np.zeros((freqs.size, n_dof, n_dof), dtype=complex)
    G_g = np.zeros_like(G_l)
    for n_idx in range(omega_n.size):
        wn = omega_n[n_idx]
        eps = eps_n[:, n_idx]
        outer = np.outer(eps, eps).astype(complex)  # (n_dof, n_dof)
        L_pos = _lorentzian(freqs - wn, eta)
        L_neg = _lorentzian(freqs + wn, eta)
        # G^<: positive peak weighted by n_B, negative peak by (n_B+1)
        # (sign convention: G^< at +ω carries occupation, at -ω it's empty)
        G_l += (
            -2j * np.pi * (
                np_factor[n_idx] * L_pos[:, None, None]
                + nm1_factor[n_idx] * L_neg[:, None, None]
            ) * outer[None]
        )
        G_g += (
            -2j * np.pi * (
                nm1_factor[n_idx] * L_pos[:, None, None]
                + np_factor[n_idx] * L_neg[:, None, None]
            ) * outer[None]
        )

    if in_z_sorted_order:
        dof_perm = expand_atom_perm_to_dofs(bundle.atom_perm)
        G_l = G_l[:, dof_perm[:, None], dof_perm[None, :]]
        G_g = G_g[:, dof_perm[:, None], dof_perm[None, :]]

    return G_l, G_g, freqs, dw, modes


# --------------------------------------------------------------------------- #
# Block view                                                                  #
# --------------------------------------------------------------------------- #


def gf_to_block_dict(
    G: np.ndarray, block_sizes: np.ndarray, *, nn_only: bool = True
) -> dict[tuple[int, int], np.ndarray]:
    """Slice a dense ``(n_freq, n_dof, n_dof)`` G into ``(I, J)`` blocks.

    Returns blocks with ``|I - J| <= 1`` if ``nn_only``, else all (I, J).
    """
    block_sizes = np.asarray(block_sizes, dtype=int)
    offsets = np.concatenate(([0], np.cumsum(block_sizes)))
    if G.shape[1] != offsets[-1]:
        raise ValueError(
            f"G has {G.shape[1]} DOFs but block_sizes sum to {offsets[-1]}"
        )

    n_blocks = block_sizes.size
    out: dict[tuple[int, int], np.ndarray] = {}
    for I in range(n_blocks):
        for J in range(n_blocks):
            if nn_only and abs(I - J) > 1:
                continue
            sI = slice(offsets[I], offsets[I + 1])
            sJ = slice(offsets[J], offsets[J + 1])
            out[(I, J)] = np.ascontiguousarray(G[:, sI, sJ])
    return out


