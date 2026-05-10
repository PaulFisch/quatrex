"""Single home for the magic numbers and naming conventions used across
``finite_analysis``.

Every default that previously lived as an argparse literal or a sprinkled
module-level constant now lives here. CLI defaults consume these.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Unit conversions                                                            #
# --------------------------------------------------------------------------- #

THZ_TO_CM1 = 33.35641
"""1 THz = 33.35641 cm⁻¹. Used wherever a dispersion is plotted in cm⁻¹."""


# --------------------------------------------------------------------------- #
# Default tolerances and broadenings                                          #
# --------------------------------------------------------------------------- #

DEFAULT_TRUNCATION_WARN = 0.01
"""Warn threshold for the NN-tridiagonal block projection of FC3:
a Frobenius drop above 1% triggers a UserWarning. Used by
:func:`quatrex.phonon.fc3_loader.fc3_to_phi_blocks` via
:func:`finite_analysis.loader.load_quatrex_blocks`."""

SSE_TRUNCATION_WARN = 0.5
"""Looser warn threshold used when projecting FC3 for SSE bubble inputs.

The SSE bubble only consumes the NN-tridiagonal sub-block of FC3 (and only
the diagonal G(K, K) blocks unless ``diag_G_in_se=False``); for systems
with longer-range FC3 the dropped weight can be tens of percent without
indicating a problem. A separate, looser threshold avoids spurious warnings
in the cutoff sweep."""

DEFAULT_DROP_THRESHOLD_THZ = 0.05
"""Acoustic-mode drop threshold in :func:`synthetic_gf.synthetic_gf_dense`.

Modes with ``|ω_n| < drop_threshold_thz`` are excluded from the eigenmode
Green's function: the ``n_B(ω_n) / (2 ω_n)`` factor would otherwise blow
up at the Γ-zero modes (``n_B ~ kT/(ℏω) → ∞``). For a finite cluster
this filters the 3 (or 4) exact translational/rotational zero modes
without dropping the lowest physical acoustic-band modes (which start
at ~ 1 cm⁻¹ ≈ 0.03 THz at most for nm-scale wires).

A finer adaptive cut is applied additionally:
``ω_min_adaptive = 1e-3 × max|ω_n|`` (relative cut). The effective
threshold is ``max(DEFAULT_DROP_THRESHOLD_THZ, ω_min_adaptive)``."""

DEFAULT_DROP_THRESHOLD_REL = 1e-3
"""Relative-magnitude drop threshold: also filter modes with
``|ω_n| < DEFAULT_DROP_THRESHOLD_REL × max|ω_n|``. Catches floating-point
zero-modes that exceed the absolute threshold."""

THZ_ACOUSTIC_BAND = 0.01
"""|ω| ≤ THZ_ACOUSTIC_BAND counts as the acoustic band when classifying
eigenmodes. Below this threshold a mode is treated as a translational/
rotational zero mode; above this in absolute value but with negative ω
it is counted as imaginary (FC2 instability)."""

ASR_REL_RESIDUAL_WARN = 0.05
"""Threshold for warning that the FC3 acoustic sum rule is violated.

Reported as the per-leg Frobenius residual divided by the FC3 norm. Above
this, downstream Σ values may carry a finite Drude-like weight at ω→0
because the bubble does not ASR-project FC3 before consuming it."""

DEFAULT_ETA_FACTOR_DW = 0.5
"""Default Lorentzian half-width for the synthetic GF expressed as a
multiple of the frequency-grid spacing dω.

Lower bound is enforced on the resulting eta to avoid sub-grid peaks:
``eta = max(DEFAULT_ETA_FACTOR_DW × dω, DEFAULT_ETA_THZ_FLOOR)``.

Use a small factor (≤ 1) when the user wants to resolve individual
modes (e.g. for ``physical_dispersion.pdf``); use a larger factor (~ 2)
to smear modes and stabilise FFT-bubble convolutions in cutoff sweeps."""

DEFAULT_ETA_THZ_FLOOR = 0.02
"""Absolute floor on the synthetic GF Lorentzian width — prevents sub-
grid peaks and the resulting aliasing when ``DEFAULT_ETA_FACTOR_DW × dω``
would be too small."""

DEFAULT_ETA_THZ_TRANSPORT = 0.05
"""Default broadening (THz) for the bare ``z² - D`` inverse in
:func:`transport_metrics.transport_trace_from_sigma`. Smaller than the
synthetic-GF eta because the transport routine resolves spectral peaks."""

DEFAULT_GAMMA_LEAD_THZ = 0.5
"""Default per-DOF synthetic lead broadening on the first and last slab."""


# --------------------------------------------------------------------------- #
# Default temperatures                                                        #
# --------------------------------------------------------------------------- #

DEFAULT_TEMPERATURE_K = 300.0
"""Default Bose-occupation temperature for the synthetic GF."""

DEFAULT_T_L_K = 305.0
"""Default left-lead temperature for the Landauer heat-current calculation."""

DEFAULT_T_R_K = 295.0
"""Default right-lead temperature."""


# --------------------------------------------------------------------------- #
# Magnitude-threshold ladder                                                  #
# --------------------------------------------------------------------------- #

EPS_LADDER = (1e-2, 1e-3, 1e-4, 1e-5)
"""Relative-magnitude thresholds at which the nnz tables are reported.

Each entry counts entries of an array whose magnitude exceeds
``eps × max|array|``. Used by the sparsity nnz table and the cutoff
magnitude-threshold sweep."""


# --------------------------------------------------------------------------- #
# Output naming                                                               #
# --------------------------------------------------------------------------- #

OUT_PREFIX: dict[str, str] = {
    "sparsity": "sparsity",
    "fc_quality": "fc_quality",
    "decomposition": "decomp",
    "physical": "physical",
    "sse_sparsity": "sse",
    "cutoffs": "cutoffs",
}
"""Per-analysis filename prefix. Every artifact a driver writes is named
``<OUT_PREFIX[analysis]>_<descriptive>.<ext>`` so the source analysis is
discoverable from the filename alone."""
