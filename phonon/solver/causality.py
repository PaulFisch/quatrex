"""Causality enforcement and spectral diagnostics for the dense
SCBA solver.

The retarded self-energy ``Sigma^R(omega)`` is causal iff the
"scattering rate" matrix ``Gamma_Sigma(omega) = i(Sigma^R - Sigma^A)``
is positive semi-definite for every ``omega > 0`` (and negative
semi-definite for ``omega < 0``). The bare Migdal bubble produces a
causal ``Sigma^R`` by construction; once discretised and stuffed into
a self-consistent loop, however, the retarded reconstruction
(Hilbert transform via zero-padded FFT) and the per-iteration
projection / symmetrisation can leak negative eigenvalues into
``Gamma_Sigma``. The loop then feeds an acausal ``G^R`` (poles in the
upper half-plane) back into the next bubble, the violations grow, and
the iterate eventually catapults to nonsense -- the failure mode
documented for d5a above ``2*omega_max`` (after the fmax aliasing
has been removed) and for nanowire devices in
\\href{https://doi.org/10.1109/iwce.2010.5677923}{Pourfath et al.\\
(2010)}.

The remedy used by causal-NEGF implementations is to project
``Gamma_Sigma`` onto the PSD cone at every iteration: take the
eigendecomposition of the hermitised ``Gamma_Sigma(omega)``, clip
negative eigenvalues to zero, and rebuild ``Sigma^R`` from the clean
``Gamma_Sigma``. This module supplies that projector and the
companion diagnostic.
"""

from __future__ import annotations

import numpy as np

from .retarded import hilbert_transform_axis


def causality_diagnostic(sigma_R, freqs_thz, low_freq_thz=0.0, tol=1e-8):
    """Worst-case PSD violation of ``Gamma_Sigma = i(Sigma^R - Sigma^A)``.

    Returns a dict with
      * ``n_violation_points`` -- number of frequencies with at least
        one eigenvalue beyond ``tol`` on the wrong side (negative for
        ``omega>0``, positive for ``omega<0``).
      * ``max_violation`` -- largest absolute violation across the
        grid (a measure of how far ``Sigma^R`` is from causal).
      * ``omega_at_max`` -- the frequency where the worst violation
        lives, useful for spotting which mode is causing trouble.
      * ``mean_violation`` -- average over the violating points.
    Only frequencies with ``|omega| > low_freq_thz`` are scored.
    """
    sigma_R = np.asarray(sigma_R)
    nfreq = sigma_R.shape[-3]
    n_viol = 0
    max_viol = 0.0
    sum_viol = 0.0
    omega_at_max = 0.0
    flat = sigma_R.reshape(-1, *sigma_R.shape[-3:])
    for iw in range(nfreq):
        w = float(freqs_thz[iw])
        if abs(w) <= low_freq_thz:
            continue
        worst_here = 0.0
        for sl in range(flat.shape[0]):
            sr = flat[sl, iw]
            gamma = 1j * (sr - sr.conj().T)
            gamma = 0.5 * (gamma + gamma.conj().T)
            eigs = np.linalg.eigvalsh(gamma)
            if w > 0:
                v = -eigs.min().real
            else:
                v = eigs.max().real
            worst_here = max(worst_here, v)
        if worst_here > tol:
            n_viol += 1
            sum_viol += worst_here
            if worst_here > max_viol:
                max_viol = worst_here
                omega_at_max = w
    mean_viol = sum_viol / max(n_viol, 1)
    return {
        "n_violation_points": int(n_viol),
        "max_violation": float(max_viol),
        "mean_violation": float(mean_viol),
        "omega_at_max": float(omega_at_max),
    }


def enforce_causality_psd(sigma_R, freqs_thz, *,
                           rebuild_real_part=True, atol=0.0):
    """Project ``Sigma^R`` onto the causal manifold by enforcing
    ``Gamma_Sigma = i(Sigma^R - Sigma^A) >= 0`` for ``omega > 0``.

    Writes the self-energy as
    ``Sigma^R = X(omega) + i Y(omega)`` with matrix-Hermitian
    ``X = (Sigma^R + Sigma^A)/2`` and ``Y = (Sigma^R - Sigma^A)/(2i)``;
    then ``Gamma_Sigma = -2 Y``. For each ``omega > 0`` we
    eigendecompose ``Y(omega)`` and clip its \\emph{positive}
    eigenvalues to zero (equivalently: clip the negative eigenvalues
    of ``Gamma_Sigma`` to zero), giving the causally-allowed
    ``Y_clean = -Gamma_Sigma_clean/2 \\le 0``. With
    ``rebuild_real_part=True`` (default), ``X`` is then rebuilt from
    ``Y_clean`` by the Kramers-Kronig relation
    ``X(omega) = -(1/pi) Hilbert[Y_clean](omega)`` -- using the
    same zero-padded FFT kernel as :func:`build_retarded`, so the
    cleaned ``Sigma^R`` is internally KK-consistent.

    Parameters
    ----------
    sigma_R : (..., nfreq, n, n) complex
    freqs_thz : (nfreq,) array
    rebuild_real_part : bool
        Recompute ``X`` from the cleaned ``Y`` (recommended).
    atol : float
        Eigenvalues within ``atol`` of zero are zeroed.
    """
    sigma_R = np.asarray(sigma_R, dtype=complex)
    out_shape = sigma_R.shape
    flat = sigma_R.reshape(-1, *sigma_R.shape[-3:])
    n_batch, nfreq, n, _ = flat.shape

    # Matrix Hermitian decomposition Sigma^R = X + i Y.
    SR_dag = flat.conj().swapaxes(-2, -1)
    Y = (flat - SR_dag) / (2j)                     # Hermitian per omega
    X_in = 0.5 * (flat + SR_dag)                   # Hermitian per omega

    Y_clean = np.empty_like(Y)
    for sl in range(n_batch):
        for iw in range(nfreq):
            w = float(freqs_thz[iw])
            M = 0.5 * (Y[sl, iw] + Y[sl, iw].conj().T)  # enforce Hermitian
            eigs, U = np.linalg.eigh(M)
            if w > 0:
                # Gamma = -2 Y must be PSD, i.e. Y must be NSD: clip
                # positive eigenvalues of Y to zero.
                eigs = np.where(eigs < -atol, eigs, 0.0)
            elif w < 0:
                eigs = np.where(eigs > atol, eigs, 0.0)
            Y_clean[sl, iw] = (U * eigs[None, :]) @ U.conj().T

    if rebuild_real_part:
        # Kramers-Kronig: X(omega) = -(1/pi) PV \int Y(omega')/(omega-omega')
        # which under the FFT Hilbert kernel of :func:`hilbert_transform_axis`
        # reads X = -H[Y].
        X_clean = -hilbert_transform_axis(Y_clean, axis=-3, pad_factor=8)
    else:
        X_clean = X_in

    sigma_R_clean = X_clean + 1j * Y_clean
    return sigma_R_clean.reshape(out_shape)


def dynamical_stability_diagnostic(H_D, sigma_R, freqs_thz):
    """Smallest eigenvalue of ``H_D + Re Sigma^R(omega=0)``.

    Negative values mean the self-consistent self-energy has driven a
    mode's ``omega^2`` past zero -- a ``G^R`` pole on the real axis
    that explodes the next Dyson solve. Reports the worst case across
    ``omega``.
    """
    sigma_R = np.asarray(sigma_R)
    flat = sigma_R.reshape(-1, *sigma_R.shape[-3:])
    worst_lambda_min = np.inf
    worst_w = 0.0
    for sl in range(flat.shape[0]):
        for iw in range(len(freqs_thz)):
            M = (H_D + flat[sl, iw].real)
            M = 0.5 * (M + M.conj().T)
            eigs = np.linalg.eigvalsh(M).real
            lam = eigs.min()
            if lam < worst_lambda_min:
                worst_lambda_min = lam
                worst_w = float(freqs_thz[iw])
    return {
        "min_eig_HD_plus_ReSigma": float(worst_lambda_min),
        "omega_at_min": worst_w,
    }
