#\!/usr/bin/env python
"""Verify the ballistic (Caroli) transmission path is eta-independent.

Monatomic 1D chain: dynamical matrix D(q) = 2K(1 - cos q), so
omega^2 in [0, 4K], omega_max = 2*sqrt(K). The Landauer transmission is
exactly T(omega) = 1 for 0 < omega < omega_max and 0 outside -- a single
propagating channel. The ballistic thermal conductance must therefore be
INDEPENDENT of the numerical broadening eta in the eta -> 0 limit.

This isolates leads.ballistic_transmission_z2 + the dense.py conductance
assembly from FC3 / SCBA / grid auto-extension.
"""
from __future__ import annotations
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from phonon_inputs.constants import HBAR_SI, KB_SI, THZ_TO_RAD
from phonon.solver.leads import ballistic_transmission_z2
from phonon.solver.grids import bose_full_axis

K = 100.0           # THz^2  -> omega_max = 2*sqrt(K) = 20 THz
H_00 = np.array([[2 * K]], dtype=complex)
H_01 = np.array([[-K]], dtype=complex)
H_LD = H_01.copy()
H_DR = H_01.copy()
H_D = H_00.copy()   # n_slabs = 1 device
omega_max = 2 * np.sqrt(K)

fmax, npts = 30.0, 300
dw = fmax / npts
freqs = np.arange(1, npts + 1) * dw          # positive grid
T_mean, dT = 300.0, 10.0
T_L, T_R = T_mean + dT / 2, T_mean - dT / 2

# analytic single-channel conductance: G = (1/A) * sum hbar*omega*(nL-nR)*1 dw
# use A = 1 (per-channel, not per-area) so we test eta-independence cleanly
nL = bose_full_axis(freqs, T_L)
nR = bose_full_axis(freqs, T_R)
omega_rad = freqs * THZ_TO_RAD
inband = freqs < omega_max
G_analytic = np.sum((HBAR_SI * omega_rad * (nL - nR))[inband]) * dw * 1e12 / dT

print(f"omega_max = {omega_max:.3f} THz, grid dw = {dw:.4f} THz")
print(f"analytic single-channel G (per channel, /dT) = {G_analytic:.6e}\n")
print(f"{'eta_factor':>10} {'eta_w[THz]':>11} {'T(mid-band)':>12} "
      f"{'max T':>8} {'G_ball':>14} {'G/G_analytic':>13}")
for eta_factor in (9.0, 6.75, 4.5, 3.0, 1.0, 0.3, 0.05):
    eta_w = eta_factor * dw
    z2 = (freqs + 1j * eta_w) ** 2
    T = np.array([ballistic_transmission_z2(z2[i], H_D, H_00, H_01, H_LD, H_DR)
                  for i in range(npts)])
    G = np.sum(HBAR_SI * omega_rad * (nL - nR) * T) * dw * 1e12 / dT
    mid = np.argmin(np.abs(freqs - omega_max / 2))
    print(f"{eta_factor:>10.3f} {eta_w:>11.4f} {T[mid]:>12.5f} "
          f"{T.max():>8.4f} {G:>14.6e} {G / G_analytic:>13.4f}")
