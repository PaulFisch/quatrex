#!/usr/bin/env python3
"""Independent audit of the one-sided phonon Kramers--Kronig assembly.

The production transform is an exact convolution for a cell-wise constant
model of ``Delta = Sigma^> - Sigma^<``.  This study keeps an independent NumPy
implementation so that sign, bosonic momentum reversal, the zero-frequency
mask and finite-window effects can be checked without calling the production
kernel.  It also compares the constant-cell rule with an exact hat-function
(piecewise-linear) convolution on a uniform symmetric grid.

Examples
--------
Write the analytic pole sweep used by the Si-film campaign::

    python phonon/studies/_si_kk_audit.py --output \
        phonon/studies/out/si_kk_audit.json

Add an audit of a four-rank production checkpoint::

    python phonon/studies/_si_kk_audit.py --checkpoint \
        cluster/si-l5-b3-v0953125save-from9375-q9-w40-dw025-t1e4 \
        --ne 161 --wmax 40 --q-shape 9 9 --output audit.json

For an equal-temperature checkpoint, add ``--temperature 300`` to test the
bosonic KMS relation on the production bubble itself.  In the occupation-
positive storage convention used by Quatrex this relation is

``Sigma^<(w) = exp[-h w/(k_B T)] Sigma^>(w)`` for positive frequency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.constants import physical_constants


# Use the same CODATA constants as ``quatrex.core.constants``.  Keeping only
# ten printed digits creates an artificial O(1e-10) KMS defect at 40 THz.
PLANCK_EV_PER_THz = (
    physical_constants["reduced Planck constant in eV s"][0]
    * 2.0 * np.pi * 1e12
)
BOLTZMANN_EV_PER_K = physical_constants[
    "Boltzmann constant in eV/K"
][0]


def _fft_convolve_axis0(a: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Full convolution along axis zero, with singleton trailing axes."""
    n = a.shape[0] + kernel.size - 1
    shape = (n,) + (1,) * (a.ndim - 1)
    return np.fft.ifft(
        np.fft.fft(a, n, axis=0) * np.fft.fft(kernel, n).reshape(shape),
        axis=0,
    )


def hilbert_cell_constant(
    delta: np.ndarray,
    frequencies: np.ndarray,
    transverse_shape: tuple[int, ...] = (),
) -> np.ndarray:
    """Production-equivalent exact-cell transform, implemented in NumPy."""
    d = np.asarray(delta, dtype=complex)
    w = np.asarray(frequencies, dtype=float)
    ne = w.size
    h = float(w[1] - w[0])
    w0 = float(w[0])

    j = np.arange(-(ne - 1), ne, dtype=float)
    aj = np.abs(j)
    safe = np.where(aj > 0, aj, 1.0)
    positive = np.where(
        aj > 0,
        np.log((safe + 0.5) / (safe - 0.5)) * np.sign(j),
        0.0,
    )
    out = _fft_convolve_axis0(d, positive)[ne - 1:2 * ne - 1]

    m = np.arange(2 * ne - 1, dtype=float)
    numerator = 2.0 * w0 + m * h + 0.5 * h
    denominator = 2.0 * w0 + m * h - 0.5 * h
    mirror_kernel = np.where(
        denominator > 0,
        np.log(
            np.where(denominator > 0, numerator, 1.0)
            / np.where(denominator > 0, denominator, 1.0)
        ),
        0.0,
    )
    mirror = d
    for axis, size in enumerate(transverse_shape, start=1):
        mirror = np.take(mirror, (-np.arange(size)) % size, axis=axis)
    mirror = mirror[::-1].conj().copy()
    if abs(w0) < 0.25 * h:
        mirror[-1] = 0.0
    out += _fft_convolve_axis0(mirror, mirror_kernel)[ne - 1:2 * ne - 1]
    return out / np.pi


def hilbert_piecewise_linear(
    delta: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    r"""Exact hat-function transform of a scalar positive-frequency spectrum.

    The positive samples are first completed with
    ``Delta(-w) = Delta(w)*``.  For an interior node, the integral of its hat
    basis against ``1/(w_n-w')`` depends only on ``j=n-k`` and is

    ``(j+1) log |(j+1)/j| + (1-j) log |j/(j-1)|``.

    The spectrum is required to vanish at the outer endpoints.  This is the
    appropriate comparison for a converged KK support window and avoids the
    logarithmic endpoint value of a nonzero finite-window truncation.
    """
    d = np.asarray(delta, dtype=complex)
    w = np.asarray(frequencies, dtype=float)
    if d.ndim != 1 or d.shape[0] != w.size:
        raise ValueError("piecewise-linear audit expects one scalar spectrum")
    scale = max(float(np.abs(d).max()), np.finfo(float).tiny)
    if abs(d[-1]) > 1e-10 * scale:
        raise ValueError("piecewise-linear audit requires a decayed outer edge")

    full = np.concatenate((d[:0:-1].conj(), d))
    size = full.size
    j = np.arange(-(size - 1), size, dtype=float)
    kernel = np.zeros_like(j)
    regular = np.abs(j) > 1
    jr = j[regular]
    kernel[regular] = (
        (jr + 1.0) * np.log(np.abs((jr + 1.0) / jr))
        + (1.0 - jr) * np.log(np.abs(jr / (jr - 1.0)))
    )
    kernel[j == 1] = 2.0 * np.log(2.0)
    kernel[j == -1] = -2.0 * np.log(2.0)
    transformed = _fft_convolve_axis0(full, kernel)[size - 1:2 * size - 1]
    return transformed[size // 2:] / np.pi


def bosonic_pole_pair(
    frequencies: np.ndarray,
    centre: float,
    halfwidth: float,
) -> np.ndarray:
    """Causal bosonic pole pair whose spectral difference is the oracle."""
    w = np.asarray(frequencies, dtype=float)
    return (
        1j / (w - centre + 1j * halfwidth)
        + 1j / (w + centre + 1j * halfwidth)
    )


def pole_error(
    spacing: float,
    wmax: float,
    centre: float,
    halfwidth: float,
) -> dict[str, float]:
    """Error of sampled KK reconstruction on the transport interval."""
    w = np.arange(0.0, wmax + 0.5 * spacing, spacing)
    exact = bosonic_pole_pair(w, centre, halfwidth)
    delta = exact - exact.conj()
    constant = 0.5 * delta + 0.5j * hilbert_cell_constant(delta, w)
    constant[0] = 0.0
    selected = (w >= 0.5) & (w <= min(30.0, wmax - spacing))
    scale = float(np.abs(exact[selected]).max())
    row = {
        "centre_thz": centre,
        "halfwidth_thz": halfwidth,
        "spacing_thz": spacing,
        "wmax_thz": wmax,
        "gamma_over_h": halfwidth / spacing,
        "constant_max_relative": float(
            np.abs(constant[selected] - exact[selected]).max() / scale
        ),
        "constant_l2_relative": float(
            np.linalg.norm(constant[selected] - exact[selected])
            / np.linalg.norm(exact[selected])
        ),
    }

    # The Lorentzian never vanishes exactly.  Zeroing only the final endpoint
    # changes its contribution below 30 THz by the finite-window tail we want
    # to exclude from this interpolation-order comparison.
    linear_delta = delta.copy()
    linear_delta[-1] = 0.0
    linear = 0.5 * linear_delta + 0.5j * hilbert_piecewise_linear(
        linear_delta, w
    )
    linear[0] = 0.0
    row["linear_max_relative"] = float(
        np.abs(linear[selected] - exact[selected]).max() / scale
    )
    row["linear_l2_relative"] = float(
        np.linalg.norm(linear[selected] - exact[selected])
        / np.linalg.norm(exact[selected])
    )
    return row


def load_distributed_checkpoint(directory: Path) -> tuple[np.ndarray, ...]:
    """Concatenate deterministic stack-rank checkpoint slices."""
    def rank(path: Path) -> int:
        return int(path.stem.rsplit("rank", 1)[1])

    files = sorted(directory.glob("sigma_best.rank*.npz"), key=rank)
    if not files:
        files = sorted(directory.glob("sigma.rank*.npz"), key=rank)
    if not files:
        raise FileNotFoundError(f"no distributed Sigma checkpoint in {directory}")
    pieces = [np.load(path) for path in files]
    return tuple(
        np.concatenate([piece[key] for piece in pieces], axis=0)
        for key in ("sigma_lesser", "sigma_greater", "sigma_retarded")
    )


def audit_checkpoint(
    directory: Path,
    ne: int,
    wmax: float,
    q_shape: tuple[int, ...],
) -> dict[str, float | int | list[int]]:
    """Check a stored production state against its defining KK functional."""
    lesser, greater, retarded = load_distributed_checkpoint(directory)
    lesser, greater, retarded = lesser[:ne], greater[:ne], retarded[:ne]
    frequencies = np.linspace(0.0, wmax, ne)
    # Stored occupation-positive convention: raw textbook Delta is < - >.
    delta = lesser - greater
    hilbert = hilbert_cell_constant(delta, frequencies, q_shape)
    reconstructed = 0.5 * delta + 0.5j * hilbert
    reconstructed[0] = 0.0  # production's acoustic-bin output mask
    scale = max(float(np.abs(retarded).max()), np.finfo(float).tiny)
    peak = max(float(np.abs(delta).max()), np.finfo(float).tiny)
    return {
        "ne": ne,
        "wmax_thz": wmax,
        "q_shape": list(q_shape),
        "assembly_max_relative": float(
            np.abs(reconstructed - retarded).max() / scale
        ),
        "edge_over_peak": float(np.abs(delta[-1]).max() / peak),
        "kk_over_retarded": float(np.abs(0.5j * hilbert).max() / scale),
        "half_over_retarded": float(np.abs(0.5 * delta).max() / scale),
    }


def kms_defect(
    lesser: np.ndarray,
    greater: np.ndarray,
    frequencies: np.ndarray,
    temperature: float,
    active_fraction: float = 1e-8,
) -> dict[str, float | int]:
    """Measure the positive-frequency bosonic detailed-balance defect.

    Global and L2 normalisations remain meaningful at spectral zeros.  The
    pointwise maximum is additionally restricted to entries whose local scale
    exceeds ``active_fraction`` of the global peak, so divisions by numerical
    noise in an empty matrix element cannot dominate the result.
    """
    sl = np.asarray(lesser, dtype=complex)
    sg = np.asarray(greater, dtype=complex)
    w = np.asarray(frequencies, dtype=float)
    if sl.shape != sg.shape or sl.shape[0] != w.size:
        raise ValueError("lesser, greater and frequency dimensions disagree")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    exponent = -PLANCK_EV_PER_THz * w / (BOLTZMANN_EV_PER_K * temperature)
    factor = np.exp(exponent).reshape((w.size,) + (1,) * (sl.ndim - 1))
    rhs = factor * sg
    defect = sl - rhs
    local_scale = np.maximum(np.abs(sl), np.abs(rhs))
    peak = max(float(local_scale.max()), np.finfo(float).tiny)
    active = local_scale >= active_fraction * peak
    pointwise = np.zeros_like(local_scale, dtype=float)
    pointwise[active] = np.abs(defect[active]) / local_scale[active]
    flat_worst = int(np.argmax(np.abs(defect)))
    worst_frequency_index = int(np.unravel_index(flat_worst, defect.shape)[0])
    denominator_l2 = max(
        float(np.linalg.norm(sl.ravel())),
        float(np.linalg.norm(rhs.ravel())),
        np.finfo(float).tiny,
    )
    return {
        "temperature_K": float(temperature),
        "global_max_relative": float(np.abs(defect).max() / peak),
        "l2_relative": float(np.linalg.norm(defect.ravel()) / denominator_l2),
        "active_pointwise_max_relative": float(pointwise.max()),
        "active_fraction": float(active_fraction),
        "active_entries": int(active.sum()),
        "worst_frequency_index": worst_frequency_index,
        "worst_frequency_thz": float(w[worst_frequency_index]),
    }


def equilibrium_current_defect(
    path: Path,
    reference_current: float | None = None,
) -> dict[str, float | list[float]]:
    """Integrate the saved Meir--Wingreen spectrum of an equilibrium run."""
    run = np.load(path)
    frequencies = np.abs(np.asarray(run["energies"], dtype=float))
    widths = np.asarray(run["frequency_cell_widths"], dtype=float)
    spectrum = np.asarray(run["current_spectrum"], dtype=float)
    if spectrum.shape[0] != frequencies.size or widths.shape != frequencies.shape:
        raise ValueError("run spectrum and frequency measure disagree")
    weighted = spectrum * (frequencies * widths).reshape(
        (frequencies.size,) + (1,) * (spectrum.ndim - 1)
    )
    reduce_axes = tuple(range(spectrum.ndim - 1))
    integrated = np.sum(weighted, axis=reduce_axes)
    absolute_budget = np.sum(np.abs(weighted), axis=reduce_axes)
    scale = max(float(absolute_budget.max()), np.finfo(float).tiny)
    saved = np.asarray(run["last_heat"], dtype=float)
    result: dict[str, float | list[float]] = {
        "left_temperature_K": float(run["left_temperature"]),
        "right_temperature_K": float(run["right_temperature"]),
        "integrated_current": integrated.tolist(),
        "saved_current": saved.tolist(),
        "spectrum_integral_vs_saved_max_absolute": float(
            np.abs(integrated - saved).max()
        ),
        "absolute_spectral_budget": absolute_budget.tolist(),
        "zero_current_over_absolute_spectral_budget": float(
            np.abs(integrated).max() / scale
        ),
        "current_spectrum_max_absolute": float(np.abs(spectrum).max()),
    }
    if reference_current is not None:
        if reference_current <= 0:
            raise ValueError("reference current must be positive")
        result["transport_reference_current"] = float(reference_current)
        result["zero_current_over_transport_reference"] = float(
            np.abs(integrated).max() / reference_current
        )
    return result


def analytic_sweep() -> list[dict[str, float]]:
    """Reduced Si-like pole grid used in the durable campaign report."""
    rows = []
    for centre in (5.0, 23.0):
        for halfwidth in (0.002, 0.01, 0.05, 0.1, 0.5):
            for spacing, wmax in (
                (0.25, 40.0),
                (0.25, 80.0),
                (0.125, 80.0),
                (0.0625, 80.0),
            ):
                rows.append(pole_error(spacing, wmax, centre, halfwidth))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--run", type=Path,
        help="equal-temperature run.npz; also audit zero heat current",
    )
    parser.add_argument(
        "--current-reference", type=float,
        help="non-equilibrium current used to normalise the equilibrium zero",
    )
    parser.add_argument("--ne", type=int, default=161)
    parser.add_argument("--wmax", type=float, default=40.0)
    parser.add_argument("--q-shape", type=int, nargs="*", default=(9, 9))
    parser.add_argument(
        "--temperature", type=float,
        help="equal temperature in K; also audit the bosonic KMS relation",
    )
    parser.add_argument(
        "--skip-analytic", action="store_true",
        help="omit the analytic pole sweep from a checkpoint-only result",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result: dict[str, object] = {}
    if not args.skip_analytic:
        result["analytic_poles"] = analytic_sweep()
    if args.checkpoint:
        result["checkpoint"] = audit_checkpoint(
            args.checkpoint, args.ne, args.wmax, tuple(args.q_shape)
        )
        if args.temperature is not None:
            lesser, greater, _ = load_distributed_checkpoint(args.checkpoint)
            frequencies = np.linspace(0.0, args.wmax, args.ne)
            result["checkpoint"]["kms"] = kms_defect(
                lesser[:args.ne], greater[:args.ne], frequencies,
                args.temperature,
            )
    if args.run:
        result["equilibrium_run"] = equilibrium_current_defect(
            args.run, args.current_reference
        )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
