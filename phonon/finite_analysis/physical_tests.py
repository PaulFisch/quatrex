"""Physical sanity checks on FC2/FC3 of a finite-structure system.

Reports a JSON blob covering:

  * **Hermiticity** of FC2 (the eV/Å² array, swap of (i,j) and (α,β)).
  * **Acoustic sum rule** (ASR) residuals for FC2 (∑_j Φ₂_{ij,αβ} ≈ 0)
    and for the mass-weighted FC3 target on legs 2 and 3 — using the
    existing :func:`phonon_inputs.fc3_compression.asr_residual`.
  * **Permutational symmetry** of the lifted ``T_lifted`` on (i, j, k).
  * **Dispersion-acoustic sanity**: lowest 3N modes at q=0 should be
    near-zero (translational acoustic), and a small-q pull should give
    finite real LA / TA slopes (no negative ω in the relevant low-q
    regime, modulo phonopy's ASR enforcement).

For finite (no-q) wires/chains, "small-q" is taken along the periodic
axis identified by ``bundle.transport_axis``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phonon_inputs.fc3_compression import asr_residual

from .constants import THZ_TO_CM1
from .loader import SystemBundle


# --------------------------------------------------------------------------- #
# FC2 checks                                                                  #
# --------------------------------------------------------------------------- #


def fc2_hermiticity(fc2: np.ndarray) -> dict[str, float]:
    """Φ₂_{ij,αβ} should equal Φ₂_{ji,βα}. Returns max-abs and rel-Frob."""
    swap = fc2.transpose(1, 0, 3, 2)
    diff = fc2 - swap
    norm = float(np.linalg.norm(fc2)) or 1.0
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "rel_frob": float(np.linalg.norm(diff) / norm),
    }


def fc2_asr_residual(fc2: np.ndarray) -> dict[str, float]:
    """∑_j Φ₂_{ij,αβ} ≈ 0. Returns the per-i Frobenius residual."""
    s = fc2.sum(axis=1)  # (n, 3, 3)
    return {
        "max_abs": float(np.max(np.abs(s))),
        "rel_frob": float(np.linalg.norm(s) / (np.linalg.norm(fc2) or 1.0)),
    }


# --------------------------------------------------------------------------- #
# FC3 checks                                                                  #
# --------------------------------------------------------------------------- #


def fc3_perm_symmetry(T_lifted: np.ndarray) -> dict[str, float]:
    """Max relative-Frobenius residual across all 5 non-trivial S3 perms."""
    norm = float(np.linalg.norm(T_lifted)) or 1.0
    perms = [
        ("ikj", T_lifted.transpose(0, 2, 1)),
        ("jik", T_lifted.transpose(1, 0, 2)),
        ("jki", T_lifted.transpose(1, 2, 0)),
        ("kij", T_lifted.transpose(2, 0, 1)),
        ("kji", T_lifted.transpose(2, 1, 0)),
    ]
    return {
        f"perm_{name}_rel": float(np.linalg.norm(T_lifted - P) / norm)
        for name, P in perms
    }


def fc3_asr_legs(T_lifted: np.ndarray, n_super: int) -> dict[str, float]:
    """Wraps :func:`asr_residual` on the (n_dof, dim_sc, dim_sc) layout.

    Reports ``||Σ_j Φ³_{ijk}|| / ||Φ³||`` (axis-j) and the analogous
    axis-k residual. For an S3-symmetric lifted FC3 these two are equal.

    **What this measures (and what it does NOT).**
    Hiphive's :class:`hiphive.ClusterSpace` enforces the *axis-i*
    (translational) ASR on FC3 by orbit construction. It does **not**
    enforce the axis-j and axis-k variants — those would constrain the
    same parameter set that already encodes axis-i ASR plus the cluster
    symmetries, and a generic cluster expansion's FC3 will have a
    nonzero ``leg_j_rel``.

    Empirical baselines (from `scratch/imag_audit/`):
      * Bulk Si fits — `leg_j_rel = 0.000` (perfectly symmetric, all
        species identical, cluster space is small and tightly
        constrained).
      * H-passivated SiNW fits — `leg_j_rel ≈ 0.80` consistently across
        every diameter and every fit_method we have tried. Driven by
        the reduced (1-D) periodicity, the Si/H mass imbalance, and the
        fact that the cluster space alone does not constrain axis-j/k.

    So ``leg_j_rel ≈ 0.8`` on a SiNW is **not** a fit defect — it is
    the expected per-axis violation from a hiphive cluster expansion
    on a low-symmetry quasi-1-D system. It IS a real consequence for
    downstream consumers that integrate FC3 along axis-j or axis-k
    (the SSE bubble Σ at ω→0 picks up a Drude-like weight), and
    :mod:`synthetic_gf` / :mod:`sse_cutoffs` neither project nor
    re-symmetrise FC3 upstream. Project explicitly via
    :func:`phonon_inputs.fc3_compression.asr_project_factor` (or
    :func:`load_quatrex_blocks(asr_project=True)`) before consuming
    FC3 if the bubble needs the sum rule strictly enforced.

    A ``UserWarning`` is emitted when the relative residual on either
    leg exceeds :data:`constants.ASR_REL_RESIDUAL_WARN`.
    """
    import warnings as _warnings
    from .constants import ASR_REL_RESIDUAL_WARN

    out = asr_residual(T_lifted, n_super)
    norm = max(out["norm"], 1e-30)
    rel_j = out["leg_j"] / norm
    rel_k = out["leg_k"] / norm
    out["leg_j_rel"] = float(rel_j)
    out["leg_k_rel"] = float(rel_k)
    worst = max(rel_j, rel_k)
    if worst > ASR_REL_RESIDUAL_WARN:
        _warnings.warn(
            f"FC3 ASR residual {worst:.3f} > {ASR_REL_RESIDUAL_WARN:.3f}. "
            "Bubble does not ASR-project FC3 — expect finite Drude-like "
            "weight in Σ at ω→0.",
            stacklevel=2,
        )
    return out


# --------------------------------------------------------------------------- #
# Dispersion sanity                                                           #
# --------------------------------------------------------------------------- #


def dispersion_along_axis(
    bundle: SystemBundle, n_q: int = 21
) -> tuple[np.ndarray, np.ndarray]:
    """Sample phonon frequencies on a 1D q-line through the periodic axis.

    For a finite wire with periodicity only along ``transport_axis``, this
    is the only physically meaningful direction in the BZ. Returns
    ``(q_frac, freq_thz)`` of shape ``(n_q,)`` and ``(n_q, 3*nat_prim)``.
    """
    ph = bundle.phonon
    ph.force_constants = bundle.fc2
    qs = np.zeros((n_q, 3))
    qs[:, bundle.transport_axis] = np.linspace(0.0, 0.5, n_q)
    ph.run_qpoints(qs.tolist())
    freqs = np.asarray(ph.get_qpoints_dict()["frequencies"])
    return qs[:, bundle.transport_axis], freqs


def dispersion_sanity(freqs_thz: np.ndarray, n_acoustic: int = 3) -> dict[str, float]:
    """Translate dispersion arrays into a scalar sanity report.

    Splits the spectrum into three categories using
    :data:`constants.THZ_ACOUSTIC_BAND` as the band edge:
    *acoustic* (|ω| ≤ band) — translational/rotational zeros at Γ;
    *imaginary* (ω < −band) — FC2 instabilities (concerning);
    *optical* (ω > band) — physical phonon modes.

    The legacy ``n_negative_modes`` field is kept (deprecated) and equals
    ``n_imaginary``.
    """
    from .constants import THZ_ACOUSTIC_BAND

    f0 = freqs_thz[0]
    n_acoustic_count = int(np.sum(np.abs(freqs_thz) <= THZ_ACOUSTIC_BAND))
    n_imaginary = int(np.sum(freqs_thz < -THZ_ACOUSTIC_BAND))
    n_optical = int(np.sum(freqs_thz > THZ_ACOUSTIC_BAND))
    return {
        "min_omega_overall_thz": float(freqs_thz.min()),
        "n_acoustic": n_acoustic_count,
        "n_imaginary": n_imaginary,
        "n_optical": n_optical,
        "n_negative_modes": n_imaginary,  # DEPRECATED — equals n_imaginary
        "max_acoustic_at_gamma_thz": float(np.abs(f0[:n_acoustic]).max()),
        "mean_acoustic_at_gamma_thz": float(np.abs(f0[:n_acoustic]).mean()),
        "highest_optical_thz": float(freqs_thz.max()),
    }


def detailed_balance_residual(
    sigma_lesser: np.ndarray, sigma_greater: np.ndarray,
    omega_grid_thz: np.ndarray, temperature_k: float,
) -> dict[str, float]:
    """Quantify how well Σ^{<,>} satisfy the bosonic detailed-balance
    relation ``Σ^>(ω) = exp(βℏω) Σ^<(ω)`` (cf. theory.tex eq. 873–874).

    Reports the relative-Frobenius gap of ``Σ^> − exp(βℏω) Σ^<`` over a
    finite Bose window (excludes ω=0 and modes far above ``kT/ℏ``). A
    clean bubble on a thermal G should pass at ~ 1 % relative.
    """
    from phonon_inputs.constants import HBAR_SI, KB_SI, THZ_TO_RAD

    omega = np.asarray(omega_grid_thz)
    cutoff_thz = 10.0 * KB_SI * temperature_k / (HBAR_SI * THZ_TO_RAD)
    mask = (np.abs(omega) > 0.01) & (np.abs(omega) < cutoff_thz)
    if not mask.any():
        return {"max_rel_dev": float("nan"), "n_samples": 0}
    x = HBAR_SI * omega[mask] * THZ_TO_RAD / (KB_SI * temperature_k)
    bose_ratio = np.exp(x)
    sl = sigma_lesser[mask]
    sg = sigma_greater[mask]
    expected_g = bose_ratio[:, None, None] * sl
    diff = sg - expected_g
    return {
        "max_rel_dev": float(
            np.linalg.norm(diff) / max(np.linalg.norm(expected_g), 1e-20)
        ),
        "n_samples": int(mask.sum()),
    }


def dos_sum_rule_residual(
    G_R: np.ndarray, omega_grid_thz: np.ndarray, n_dof: int,
) -> dict[str, float]:
    """Phonon spectral-function sum rule for a finite system.

    For the displacement-displacement retarded Green's function
    ``G^R(ω) = Σ_n |ε_n|² / [(ω+iη)² − ω_n²]`` (mass-weighted), the
    standard sum rule with ``A(ω) = -1/π Im Tr G^R(ω)`` is::

        ∫_0^∞ 2 ω · A(ω) dω = N_dof.

    The factor ``2ω`` is the phonon measure: each eigenmode contributes
    ``∫_0^∞ A_n(ω) dω = 1/(2 ω_n)``, so the unweighted integral has no
    universal value, while the ``2ω``-weighted one yields N_dof exactly.

    Returns the integral, the expected value (``N_dof``), and the
    relative deviation. A clean eigenmode G passes within the
    Lorentzian-broadening leakage at the band edge.
    """
    omega = np.asarray(omega_grid_thz)
    A = -1.0 / np.pi * np.imag(np.trace(G_R, axis1=-2, axis2=-1))
    pos = omega > 0.0
    if pos.sum() < 2:
        return {"n_dof": int(n_dof), "integrated_dos": float("nan"),
                "expected": float(n_dof), "rel_dev": float("nan")}
    dw = float(omega[1] - omega[0])
    integral = float(np.sum(2.0 * omega[pos] * A[pos]) * dw)
    expected = float(n_dof)
    return {
        "n_dof": int(n_dof),
        "integrated_dos": integral,
        "expected": expected,
        "rel_dev": float((integral - expected) / max(expected, 1.0)),
    }


def fc2_psd_summary(fc2: np.ndarray, masses: np.ndarray) -> dict[str, float]:
    """Eigenvalue summary of the mass-weighted dynamical matrix.

    Returns counts of negative eigenvalues, the most-negative eigenvalue,
    and the worst negative-to-positive ratio. A non-PSD FC2 invalidates
    the harmonic approximation; this is a quick sanity flag.
    """
    n = fc2.shape[0]
    inv_sqrt_m = 1.0 / np.sqrt(masses)
    weight = inv_sqrt_m[:, None] * inv_sqrt_m[None, :]
    D_block = fc2 * weight[:, :, None, None]
    D = D_block.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    D = 0.5 * (D + D.T)
    eigvals = np.linalg.eigvalsh(D)
    pos = eigvals[eigvals > 0]
    return {
        "n_neg_eigvals": int(np.sum(eigvals < -1e-12)),
        "min_eigval": float(eigvals.min()),
        "max_eigval": float(eigvals.max()),
        "max_neg_rel": float(
            -eigvals.min() / pos.max() if pos.size and eigvals.min() < 0 else 0.0
        ),
    }


def dispersion_residuals(bundle: SystemBundle, *, n_q: int = 21) -> dict[str, float]:
    """Compute the dispersion-sanity residuals without rendering a plot.

    The PDF dispersion plot used to be emitted from this routine into
    ``physical_dispersion.pdf``, but that figure was a near-perfect
    duplicate of the one produced by ``fc_quality.plot_dispersion_compare``.
    The merged ``fc_quality/`` directory keeps the visualisation; only
    the numerical residuals live here.
    """
    _, freqs = dispersion_along_axis(bundle, n_q=n_q)
    return dispersion_sanity(freqs)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class PhysicalSummary:
    fc2_hermiticity: dict
    fc2_asr: dict
    fc2_psd: dict
    fc3_perm_sym: dict
    fc3_asr_legs: dict
    dispersion: dict


def run_physical_tests(bundle: SystemBundle, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = PhysicalSummary(
        fc2_hermiticity=fc2_hermiticity(bundle.fc2),
        fc2_asr=fc2_asr_residual(bundle.fc2),
        fc2_psd=fc2_psd_summary(bundle.fc2, bundle.masses),
        fc3_perm_sym=fc3_perm_symmetry(bundle.fc3_target.T_lifted),
        fc3_asr_legs=fc3_asr_legs(bundle.fc3_target.T_lifted, bundle.n_super),
        dispersion=dispersion_residuals(bundle),
    )
    payload = asdict(summary)
    payload["units"] = {
        "dispersion.*_thz": "THz",
        "dispersion.*omega*": "THz",
        "fc2_psd.min_eigval": "eV/(Å²·amu)",
        "fc2_psd.max_eigval": "eV/(Å²·amu)",
    }
    (out_dir / "physical.json").write_text(json.dumps(payload, indent=2))
    return payload
