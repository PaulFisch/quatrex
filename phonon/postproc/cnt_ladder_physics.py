"""Numerical reductions used by the CNT ladder physics figures."""

from __future__ import annotations

import numpy as np

HBAR = 1.054571817e-34
KB = 1.380649e-23


def bose(freqs_thz, temperature):
    """Bose occupation on an ordinary-frequency grid in THz."""
    f = np.asarray(freqs_thz, dtype=float)
    x = HBAR * 2.0 * np.pi * f * 1e12 / (KB * float(temperature))
    out = np.zeros_like(f)
    np.divide(1.0, np.expm1(np.clip(x, 0.0, 500.0)), out=out, where=x > 1e-12)
    return out


def lead_spectrum(current_spectrum):
    """Mean magnitude of the two contact number-current spectra."""
    current = np.real(np.asarray(current_spectrum))
    while current.ndim > 2:
        current = current.mean(axis=1)
    return 0.5 * (np.abs(current[:, 0]) + np.abs(current[:, -1]))


def effective_transmission(spectrum, freqs_thz, left_temperature,
                           right_temperature):
    """Convert a contact number-current spectrum to effective transmission."""
    dn = bose(freqs_thz, left_temperature) - bose(freqs_thz, right_temperature)
    out = np.full_like(dn, np.nan)
    np.divide(spectrum, dn, out=out, where=dn > 1e-12)
    return out


def spectral_quantiles(freqs_thz, density, weights_thz, quantiles=(0.5, 0.9)):
    """Frequencies below which selected fractions of heat current flow."""
    f = np.asarray(freqs_thz, dtype=float)
    mass = np.clip(np.asarray(density, dtype=float), 0.0, None) \
        * np.asarray(weights_thz, dtype=float)
    cumulative = np.cumsum(mass)
    if not cumulative.size or cumulative[-1] <= 0.0:
        return np.full(len(quantiles), np.nan)
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, f)


def apparent_mfp(length_nm, conductance_ratio):
    """MFP inferred pointwise from G/Gball = 1/(1 + L/lambda)."""
    length = np.asarray(length_nm, dtype=float)
    ratio = np.asarray(conductance_ratio, dtype=float)
    out = np.full(np.broadcast_shapes(length.shape, ratio.shape), np.nan)
    valid = (ratio > 0.0) & (ratio < 1.0)
    np.divide(length, 1.0 / ratio - 1.0, out=out, where=valid)
    return out


def mean_local_spectrum(freqs_thz, gr_diag_imag, gl_diag_imag):
    """Per-stored-DOF LDOS, occupation, and spectral temperature."""
    f = np.asarray(freqs_thz, dtype=float)
    gr = np.asarray(gr_diag_imag, dtype=float)
    gl = np.asarray(gl_diag_imag, dtype=float)
    while gr.ndim > 2:
        gr = gr.mean(axis=1)
        gl = gl.mean(axis=1)
    ldos = (2.0 * f[:, None] / np.pi * gr).mean(axis=1)
    denom = 2.0 * gr.sum(axis=1)
    occupation = np.divide(gl.sum(axis=1), denom, out=np.zeros_like(f),
                           where=denom > 1e-30)
    quantum = HBAR * 2.0 * np.pi * f * 1e12 / KB
    temperature = np.divide(
        quantum, np.log1p(np.divide(1.0, occupation,
                                    out=np.full_like(f, np.inf),
                                    where=occupation > 0.0)),
        out=np.full_like(f, np.nan), where=occupation > 0.0)
    return ldos, occupation, temperature


def linewidth_sector_matrix(coupling, mode_freqs, edges):
    """Receiving-sector shares of a source-mode linewidth matrix."""
    matrix = np.asarray(coupling, dtype=float)
    freqs = np.asarray(mode_freqs, dtype=float)
    edges = np.asarray(edges, dtype=float)
    nsector = edges.size - 1
    out = np.zeros((nsector, nsector))
    for receiver in range(nsector):
        rows = (freqs >= edges[receiver]) & (freqs < edges[receiver + 1])
        for source in range(nsector):
            cols = (freqs >= edges[source]) & (freqs < edges[source + 1])
            out[receiver, source] = matrix[np.ix_(rows, cols)].sum()
    total = out.sum(axis=1, keepdims=True)
    return np.divide(out, total, out=np.zeros_like(out), where=total > 0.0)


def modal_bubble_properties(freqs_thz, dynamical, sigma_b):
    """On-shell harmonic-mode shift and HWHM from a retarded bubble."""
    f = np.asarray(freqs_thz, dtype=float)
    dyn = 0.5 * (np.asarray(dynamical) + np.asarray(dynamical).conj().T)
    omega2, modes = np.linalg.eigh(dyn)
    omega = np.sqrt(np.clip(omega2.real, 0.0, None))
    projected = np.einsum("ai,wab,bi->wi", modes.conj(), sigma_b, modes,
                          optimize=True)
    onshell = np.array([
        np.interp(value, f, projected[:, mode].real)
        + 1j * np.interp(value, f, projected[:, mode].imag)
        for mode, value in enumerate(omega)
    ])
    shift = np.full_like(omega, np.nan)
    width = np.full_like(omega, np.nan)
    np.divide(onshell.real, 2.0 * omega, out=shift, where=omega > 1e-8)
    np.divide(-onshell.imag, 2.0 * omega, out=width, where=omega > 1e-8)
    return omega, shift, width
