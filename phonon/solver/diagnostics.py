"""Diagnostic helpers for the dense phonon solver.

These functions check physical invariants that the SCBA loop relies on
but does not itself enforce:

  * :func:`check_broadening_sign` — Γ = i(Σ^R − Σ^A) must be PSD for
    ω > 0, NSD for ω < 0.
  * :func:`check_full_axis_symmetry` — bosonic Keldysh symmetry
    ``G^<(ω) = [G^>(−ω)]^T`` and ``G^R(ω) = [G^R(−ω)]*``.
  * :func:`symmetrize_lesser_greater` — projection onto the
    full-axis symmetry manifold; called after each SCBA iteration.
"""

from __future__ import annotations

import numpy as np


def check_broadening_sign(Sigma_R, freqs_thz, name,
                          low_freq_thz=0.0, tol=1e-8):
    """Check Γ = i(Σ^R − Σ^A) PSD for ω > 0, NSD for ω < 0.

    Returns ``(n_violations, max_violation)`` over all frequencies with
    ``|ω| > low_freq_thz``.
    """
    n_viol = 0
    max_viol = 0.0
    for iw in range(len(freqs_thz)):
        w = freqs_thz[iw]
        if abs(w) <= low_freq_thz:
            continue
        sr = Sigma_R[iw]
        Gamma = 1j * (sr - sr.conj().T)
        Gamma = 0.5 * (Gamma + Gamma.conj().T)
        eigs = np.linalg.eigvalsh(Gamma)
        if w > 0 and eigs.min() < -tol:
            n_viol += 1
            max_viol = max(max_viol, -eigs.min())
        elif w < 0 and eigs.max() > tol:
            n_viol += 1
            max_viol = max(max_viol, eigs.max())
    return n_viol, max_viol


def check_full_axis_symmetry(G_R, G_l, G_g, freqs_thz,
                             rtol=1e-3, atol=1e-8):
    """Verify bosonic full-axis symmetry.

    ``G^<(ω) = [G^>(−ω)]^T``, ``G^R(ω) = [G^R(−ω)]*``.

    Returns ``(lesser_err, retarded_err)``.
    """
    mid = len(freqs_thz) // 2
    pos = slice(mid + 1, None)
    neg = slice(0, mid)

    G_l_pos = G_l[pos]
    G_g_neg = G_g[neg][::-1].transpose(0, 2, 1)
    lesser_err = float(np.max(np.abs(G_l_pos - G_g_neg)))

    G_R_pos = G_R[pos]
    G_R_neg = G_R[neg][::-1]
    retarded_err = float(np.max(np.abs(G_R_pos - G_R_neg.conj())))
    return lesser_err, retarded_err


def symmetrize_lesser_greater(sig_l, sig_g):
    """Project Σ^< and Σ^> onto the bosonic full-axis symmetry manifold.

    Enforces ``Σ^<(ω) = [Σ^>(−ω)]^T`` in-place on the last three axes
    ``(nfreq, nd, nd)``. Leading dimensions are preserved.

    Derivation: Σ^< = −i n_B Γ, Σ^> = −i(n_B+1)Γ, and for bosons
    ``n_B(−ω) = −(n_B(ω)+1), Γ(−ω) = −Γ(ω)^T``, giving
    ``Σ^<(ω) = [Σ^>(−ω)]^T`` (no minus sign).

    The grid is assumed symmetric about ω = 0 with ``mid = nfreq // 2``.
    The ω = 0 sample is left untouched.
    """
    nfreq = sig_l.shape[-3]
    mid = nfreq // 2

    sl_pos = sig_l[..., mid + 1:, :, :].copy()
    sg_pos = sig_g[..., mid + 1:, :, :].copy()
    sg_neg_rev = sig_g[..., :mid, :, :][..., ::-1, :, :].copy()
    sl_neg_rev = sig_l[..., :mid, :, :][..., ::-1, :, :].copy()

    sl_pos_sym = 0.5 * (sl_pos + sg_neg_rev.swapaxes(-2, -1))
    sg_pos_sym = 0.5 * (sg_pos + sl_neg_rev.swapaxes(-2, -1))

    sig_l[..., mid + 1:, :, :] = sl_pos_sym
    sig_g[..., mid + 1:, :, :] = sg_pos_sym

    sig_l[..., :mid, :, :] = (
        sg_pos_sym[..., ::-1, :, :].swapaxes(-2, -1))
    sig_g[..., :mid, :, :] = (
        sl_pos_sym[..., ::-1, :, :].swapaxes(-2, -1))
