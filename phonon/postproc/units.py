"""Physical-unit bridge for production ``run.npz`` outputs.

The distributed engine's ``lead_current`` / ``iter_heat`` are *internal*
THz-weighted numbers: the RGF lead current is the bare Meir--Wingreen
trace ``Tr[Sigma_L^> G^< - G^> Sigma_L^<]`` (a per-THz number-current
density on the ``energies`` grid), and ``run.py:_heat`` folds in only
``|nu|`` (plus quadrature cell widths on non-uniform grids). The dense
stack (phonon/solver/observables.py) carries the full physical
convention

    J [W] = sum_nu  hbar * (2 pi nu 1e12) * mw(nu) * dnu_THz * 1e12,

i.e. spectra are integrated as ``sum * dw * 1e12`` with the ordinary-
frequency grid absorbing the 1/(2 pi). This module applies exactly that
convention to ``run.npz``:

    heat_current_watts:  per-interface J in W from ``current_spectrum``
    conductance:         G = J_lead / delta_T in W/K (+ per-area)

Cross-section conventions: per-area numbers use the transverse
primitive-cell area (dense ``_cross_section_area``). For experimental
CNT comparisons rescale explicitly to the pi*d*h shell convention
(h = 3.35 A) in the figure script -- never silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phonon_inputs.constants import HBAR_SI, THZ_TO_RAD

#: J per (internal THz^2-weighted current unit): hbar * 2pi * 1e24.
_INTERNAL_TO_W = HBAR_SI * THZ_TO_RAD * 1e12


def _quadrature_weights(freqs_thz: np.ndarray,
                        stored: np.ndarray | None) -> np.ndarray:
    """Per-bin cell widths in THz (stored weights win; else trapezoid)."""
    if stored is not None:
        return np.asarray(stored, dtype=float)
    f = np.asarray(freqs_thz, dtype=float)
    w = np.empty_like(f)
    w[1:-1] = 0.5 * (f[2:] - f[:-2])
    w[0] = 0.5 * (f[1] - f[0])
    w[-1] = 0.5 * (f[-1] - f[-2])
    return w


def heat_current_watts(
    freqs_thz: np.ndarray,
    current_spectrum: np.ndarray,
    weights_thz: np.ndarray | None = None,
) -> np.ndarray:
    """Per-interface heat current in W from the MW number-current spectrum.

    Parameters
    ----------
    freqs_thz : (ne,) positive-frequency grid (THz, DC bin dropped).
    current_spectrum : (ne, ..., n_interfaces) real MW spectrum
        (``run.npz["current_spectrum"]``; any transverse-q axes are
        averaged -- the engine stores the q-summed convention already
        normalised per transverse cell).
    weights_thz : optional per-bin quadrature widths (THz). When absent
        a trapezoidal rule on ``freqs_thz`` is used (exact for the
        uniform legacy grid; consistent to O(h^2) with the engine's
        cell-width quadrature on non-uniform grids).

    Returns
    -------
    (n_interfaces,) heat currents in Watts (positive = left-to-right).
    """
    f = np.abs(np.asarray(freqs_thz, dtype=float))
    spec = np.real(np.asarray(current_spectrum))
    w = _quadrature_weights(freqs_thz, weights_thz)
    # collapse any transverse-q axes between energy and interface
    while spec.ndim > 2:
        spec = spec.mean(axis=1)
    integrand = (w * f)[:, None] * spec
    return _INTERNAL_TO_W * integrand.sum(axis=0)


def run_npz_conductance(npz_path: str | Path,
                        area_m2: float | None = None) -> dict:
    """Physical-unit summary of a production ``run.npz``.

    Returns a dict with per-interface ``J_watts``, the lead-averaged
    ``J_lead_watts`` (0.5(|J_L|+|J_R|), matching ``lead_current``),
    ``delta_T`` (from stored lead temperatures), ``G_WK`` = J/dT and,
    when ``area_m2`` is given, ``G_Wm2K``.
    """
    d = np.load(npz_path, allow_pickle=True)
    if "current_spectrum" not in d:
        raise KeyError(f"{npz_path}: no current_spectrum -- rerun with the "
                       "spectrum output enabled (engine default) or use the "
                       "dense stack for physical units.")
    freqs = np.asarray(d["energies"], dtype=float)
    weights = (np.asarray(d["frequency_cell_widths"], dtype=float)
               if "frequency_cell_widths" in d else None)
    J = heat_current_watts(freqs, d["current_spectrum"], weights)
    out = {"J_watts": J,
           "J_lead_watts": 0.5 * (abs(J[0]) + abs(J[-1]))}
    if "t_left" in d and "t_right" in d:
        dT = float(d["t_left"]) - float(d["t_right"])
        out["delta_T"] = dT
        if dT != 0.0:
            out["G_WK"] = out["J_lead_watts"] / dT
            if area_m2:
                out["G_Wm2K"] = out["G_WK"] / float(area_m2)
    return out


def cross_section_area_m2(lattice: np.ndarray,
                          transport_direction: str = "z") -> float:
    """Transverse cell area in m^2 (dense-stack convention): |a1 x a2|
    of the two non-transport lattice vectors, lattice in Angstrom."""
    lat = np.asarray(lattice, dtype=float)
    tidx = "xyz".index(transport_direction)
    perp = [i for i in range(3) if i != tidx]
    return float(np.linalg.norm(np.cross(lat[perp[0]], lat[perp[1]])) * 1e-20)
