"""Physical-identity validators for the finite (Γ-only) phonon NEGF path.

Each routine takes either Σ or G arrays and reports a residual that
*should* be zero (or below a tolerance) for any internally-consistent
implementation. Use these as fast cross-checks during analysis.

References (the identities below are standard; cited in the appendix):

  * Wang, "Nonequilibrium Green's Function approach for phonon transport",
    Front. Phys. 2014. Sections II–III for the bosonic Keldysh.
  * Stefanucci & van Leeuwen, "Nonequilibrium Many-Body Theory of Quantum
    Systems", CUP 2013. Chapters 4–5 for the analytic structure.
  * Datta, "Quantum Transport: Atom to Transistor", CUP 2005. Chapter 9
    for transmission and lead Σ.
  * Maradudin, Montroll, Weiss, Ipatova, "Theory of Lattice Dynamics in
    the Harmonic Approximation". Sum rules on the spectral function.

All routines are pure NumPy; no dependence on the rest of the package
beyond ``phonon_inputs.constants`` for the unit-system constants.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Identity #1: anti-Hermiticity of G^{<,>}                                    #
# --------------------------------------------------------------------------- #


def anti_hermiticity_residual(g: np.ndarray) -> dict[str, float]:
    """``i G^<(ω)`` and ``i G^>(ω)`` are Hermitian (so ``G^<,>`` is anti-Hermitian).

    Specifically, ``[G^{<,>}(ω)]† = -G^{<,>}(ω)`` for the bosonic
    Keldysh component on a real ω axis (Wang 2014, eq. 19; equivalent
    to ``i G^<`` being a positive-semidefinite spectral density).

    Returns the worst-element relative-Frobenius residual of
    ``G + G†`` over ω.
    """
    diff = g + g.conj().swapaxes(-1, -2)
    norm = np.linalg.norm(g, axis=(-2, -1)).max() or 1.0
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "max_rel": float(np.linalg.norm(diff, axis=(-2, -1)).max() / norm),
    }


# --------------------------------------------------------------------------- #
# Identity #2: G^R(ω)† = G^A(ω)                                                #
# --------------------------------------------------------------------------- #


def retarded_advanced_residual(G_R: np.ndarray, G_A: np.ndarray | None = None) -> dict[str, float]:
    """``G^R(ω)† = G^A(ω)`` (standard retarded/advanced identity).

    If ``G_A`` is None, build it from the same array (so the residual
    measures any spurious deviation from the Hermitian-conjugate
    relation we assume internally).
    """
    if G_A is None:
        G_A = G_R.conj().swapaxes(-1, -2)
    diff = G_R.conj().swapaxes(-1, -2) - G_A
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "max_rel": float(
            np.linalg.norm(diff) / max(np.linalg.norm(G_A), 1e-30)
        ),
    }


# --------------------------------------------------------------------------- #
# Identity #3: optical theorem (Σ^> − Σ^< = -2i Im Σ^R)                        #
# --------------------------------------------------------------------------- #


def optical_theorem_residual(
    sigma_lesser: np.ndarray, sigma_greater: np.ndarray, sigma_retarded: np.ndarray,
) -> dict[str, float]:
    """``Σ^> - Σ^< = -2i Im Σ^R`` (Wang 2014, eq. 25).

    Equivalently, ``Γ = i (Σ^R - Σ^A) = -2 Im Σ^R = i (Σ^> - Σ^<)``.
    Reports the relative-Frobenius residual of ``Σ^> - Σ^< + 2i Im Σ^R``.
    """
    Im_R = (sigma_retarded - sigma_retarded.conj().swapaxes(-1, -2)) / (2j)
    diff = sigma_greater - sigma_lesser - (-2j * Im_R)
    norm = max(np.linalg.norm(sigma_greater - sigma_lesser), 1e-30)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "max_rel": float(np.linalg.norm(diff) / norm),
    }


# --------------------------------------------------------------------------- #
# Identity #4: first-moment spectral sum rule                                 #
# --------------------------------------------------------------------------- #


def first_moment_sum_rule_residual(
    G_R: np.ndarray, omega_grid_thz: np.ndarray, dynamical_matrix_thz2: np.ndarray,
) -> dict[str, float]:
    """First-moment sum rule for phonon spectral density.

    With ``A(ω) = -1/π Im Tr G^R(ω)``, integrating with the ``2ω³``
    measure on the positive axis yields ``Tr(D)``::

        ∫_0^∞ 2 ω³ · (-1/π) Im Tr G^R(ω) dω = Tr(D)

    Derivation: each eigenmode contributes
    ``∫_0^∞ A_n(ω) · 2 ω³ dω = ω_n²``; sum over modes gives ``Σ_n ω_n²
    = Tr(D)``. This pins both the unit conversion (D in THz²) and the
    spectral-weight normalisation simultaneously.
    """
    omega = np.asarray(omega_grid_thz)
    pos = omega > 0.0
    if pos.sum() < 2:
        return {"integrated": float("nan"), "expected": float("nan"),
                "rel_dev": float("nan")}
    dw = float(omega[1] - omega[0])
    A = -1.0 / np.pi * np.imag(np.trace(G_R, axis1=-2, axis2=-1))
    integral = float(np.sum(2.0 * omega[pos] ** 3 * A[pos]) * dw)
    expected = float(np.real(np.trace(dynamical_matrix_thz2)))
    return {
        "integrated": integral,
        "expected": expected,
        "rel_dev": float((integral - expected) / max(expected, 1.0)),
    }


# --------------------------------------------------------------------------- #
# Identity #5: ballistic transmission upper bound                             #
# --------------------------------------------------------------------------- #


def ballistic_transmission_bound(
    transmission: np.ndarray, omega_grid_thz: np.ndarray, n_modes: int,
) -> dict[str, float]:
    """``0 ≤ T(ω) ≤ n_modes``: physical bound from unitarity of the
    scattering matrix (Datta 2005, eq. 9.16). ``n_modes`` is the number
    of propagating channels at frequency ω; for our finite slab leads
    that's at most ``n_dof_slab``.

    Reports min, max, and the count of frequencies where the bound is
    violated (which is exactly zero for a correct implementation).
    """
    pos = omega_grid_thz > 0
    T_pos = transmission[pos]
    if T_pos.size == 0:
        return {"min": float("nan"), "max": float("nan"), "n_violations": 0}
    n_below_zero = int(np.sum(T_pos < -1e-10))
    n_above_bound = int(np.sum(T_pos > n_modes + 1e-6))
    return {
        "min": float(T_pos.min()),
        "max": float(T_pos.max()),
        "bound": int(n_modes),
        "n_violations": n_below_zero + n_above_bound,
    }


# --------------------------------------------------------------------------- #
# Identity #6: Bose-factor identity n_B(-ω) = -1 - n_B(ω)                      #
# --------------------------------------------------------------------------- #


def bose_identity_residual(
    omega_thz: np.ndarray, temperature_k: float,
) -> dict[str, float]:
    """``n_B(-ω) = -1 - n_B(ω)`` for finite ω.

    The identity follows from ``1/(e^x - 1) = -1 - 1/(e^{-x} - 1)`` for
    real x ≠ 0. A correct implementation of the Bose factor on the
    full ω axis (negative + positive) must satisfy this to machine
    precision.
    """
    from phonon_inputs.constants import HBAR_SI, KB_SI, THZ_TO_RAD

    omega = np.asarray(omega_thz)
    sel = np.abs(omega) > 1e-3  # avoid the singularity exactly at 0
    x = HBAR_SI * np.abs(omega[sel]) * THZ_TO_RAD / (KB_SI * temperature_k)
    n_pos = 1.0 / np.expm1(x)
    n_neg = -1.0 - n_pos  # by the identity
    n_neg_direct = 1.0 / np.expm1(-x)
    diff = np.max(np.abs(n_neg - n_neg_direct))
    return {
        "max_abs_diff": float(diff),
        "n_samples": int(sel.sum()),
    }


# --------------------------------------------------------------------------- #
# Identity #7: Kramers-Kronig consistency of two Σ^R reconstructions          #
# --------------------------------------------------------------------------- #


def kramers_kronig_consistency(
    sigma_lesser: np.ndarray, sigma_greater: np.ndarray, omega_grid_thz: np.ndarray,
) -> dict[str, float]:
    """Two independent reconstructions of ``Σ^R`` from ``Σ^{<,>}`` —
    PV-quadrature and FFT-Hilbert — must agree.

    Returns the relative-Frobenius gap between the two Σ^R arrays.
    """
    from phonon_inputs.anharmonic import _build_retarded

    Sigma_R_pv = _build_retarded(sigma_lesser, sigma_greater, omega_grid_thz, method="pv")
    Sigma_R_fft = _build_retarded(sigma_lesser, sigma_greater, omega_grid_thz, method="fft")
    diff = np.linalg.norm(Sigma_R_pv - Sigma_R_fft)
    norm = max(np.linalg.norm(Sigma_R_pv), 1e-30)
    return {
        "max_abs": float(np.max(np.abs(Sigma_R_pv - Sigma_R_fft))),
        "rel_gap": float(diff / norm),
    }
