"""Exact trigonometric q-mesh resampling for finite-support vertex factors."""

from __future__ import annotations

import numpy as np


def q_difference_map(nk: int) -> np.ndarray:
    """C-order lookup for ``(q-q') mod nk`` on a square mesh."""
    index = np.arange(nk * nk)
    x, y = np.divmod(index, nk)
    dx = (x[:, None] - x[None, :]) % nk
    dy = (y[:, None] - y[None, :]) % nk
    return (dx * nk + dy).astype(np.int64)


def trigonometric_resample(values: np.ndarray, target_nk: int,
                           support_radius: int = 2) -> tuple[np.ndarray, dict]:
    """Resample ``(offset, nq, dof, rank)`` q values on a square mesh.

    The source is interpreted as samples of a Fourier polynomial whose
    real-space support is bounded by ``support_radius`` in both transverse
    directions.  The returned diagnostics expose aliasing and sign/order
    mistakes rather than silently interpolating an under-resolved input.
    """
    values = np.asarray(values, dtype=np.complex128)
    source_nk = int(round(np.sqrt(values.shape[1])))
    if source_nk * source_nk != values.shape[1] or source_nk % 2 != 1:
        raise ValueError("source factor mesh must be an odd square")
    if target_nk % 2 != 1:
        raise ValueError("target factor mesh must be odd")
    if source_nk < 2 * support_radius + 1:
        raise ValueError(
            f"source nk={source_nk} cannot resolve support radius "
            f"{support_radius}")

    sampled = values.reshape(
        values.shape[0], source_nk, source_nk, *values.shape[2:])
    coefficients = np.fft.ifft2(sampled, axes=(1, 2))
    translations = np.rint(np.fft.fftfreq(source_nk) * source_nk).astype(int)
    outside = ((np.abs(translations)[:, None] > support_radius)
               | (np.abs(translations)[None, :] > support_radius))
    total_norm = float(np.linalg.norm(coefficients.ravel()))
    tail_norm = float(np.linalg.norm(coefficients[:, outside].ravel()))

    def evaluate(nk: int) -> np.ndarray:
        q = np.arange(nk, dtype=float) / nk
        phase = np.exp(-2j * np.pi * q[:, None] * translations[None, :])
        return np.einsum(
            "ar,orsdk,bs->oabdk", phase, coefficients, phase,
            optimize=True,
        )

    resampled = evaluate(target_nk)
    reconstructed = evaluate(source_nk)
    roundtrip = float(
        np.linalg.norm(reconstructed - sampled)
        / max(np.linalg.norm(sampled), np.finfo(float).tiny))
    return (
        np.ascontiguousarray(resampled.reshape(
            values.shape[0], target_nk * target_nk, *values.shape[2:])),
        {
            "source_nk": source_nk,
            "target_nk": int(target_nk),
            "support_radius": int(support_radius),
            "relative_fourier_tail": tail_norm / max(
                total_norm, np.finfo(float).tiny),
            "relative_source_roundtrip": roundtrip,
        },
    )
