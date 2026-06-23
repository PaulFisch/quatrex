"""Validate an IR singularity-subtracted bubble CONVOLUTION against scipy
quadrature, in the scalar model, BEFORE wiring it into the solver.

Sigma(w) = PV integral_{-W}^{W} Ga(w') Gb(w-w') dw'   (symmetric grid through 0)

is intrinsically a PRINCIPAL VALUE: out of equilibrium the device G^< inherits
the genuine contact pole  G(w') ~ -i c A(0)/w'  (A(0)!=0, c=kT/hbar), so the
integrand has simple poles at w'=0 (from Ga) and w'=w (from Gb). The production/
dense bubbles just ZERO the w'=0 bin (the plateau-destroying taper). We instead
RESIDUE-double-subtract -- the exact generalisation of the proven
phonon.solver.retarded.retarded_from_lesser_greater pattern:

  Sigma(w) = integral [ g(w') - r0/w' - rw/(w-w') ] dw'    (regular -> trapezoid)
           + r0 * PV integral dw'/w'                        ( = 0 on a symmetric grid)
           + rw * PV integral dw'/(w-w')                    ( = ln|(w+W)/(w-W)| )

with g(w')=Ga(w')Gb(w-w'), residues r0 = lim_{w'->0} w' g = (-i c A(0)) Gb(w),
rw = lim_{w'->w} (w-w') g = Ga(w) (-i c A(0)). The removable nodes are filled by
the finite-difference derivative of (g - poles), as in retarded.py.

Run:  OMP_NUM_THREADS=1 python phonon/studies/_ir_bubble_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from scipy import integrate

from quatrex.phonon.ir_subtraction import bose, bose_pole_coeff


def model_A(w):
    """Smooth spectral function, even, with A(0) != 0 (genuine contact pole)."""
    w = np.asarray(w, float)
    return np.exp(-((np.abs(w) - 3.0) ** 2) / (2 * 1.2 ** 2)) + 0.4 * np.exp(-w ** 2 / 8.0)


def Glesser(w, T):
    """G^<(w) = -i n(w) A(w), with the w=0 sample left as inf (handled by subtraction)."""
    return -1j * bose(w, T) * model_A(w)


# ----------------------------------------------------------------------------
# reference: scipy PV quadrature of the doubly-subtracted integrand
# ----------------------------------------------------------------------------
def sigma_reference(wm, T, W):
    """PV integral_{-W}^{W} Ga(w')Gb(wm-w') dw' via residue subtraction + scipy quad."""
    c = bose_pole_coeff(T)
    A0 = model_A(0.0)
    Gb_wm = -1j * bose(wm, T) * model_A(wm) if abs(wm) > 1e-12 else 0j  # smooth at wm!=0
    # residues
    r0 = (-1j * c * A0) * (Glesser(np.array([wm]), T)[0] if abs(wm) > 1e-9 else
                           -1j * (c * 0.0 - 0.5 * A0))   # Gb(wm); wm!=0 here
    rw = (Glesser(np.array([wm]), T)[0] if abs(wm) > 1e-9 else 0j) * (-1j * c * A0)

    def g(wp):
        # Ga(wp) Gb(wm-wp), both legs G^<; smooth except at wp=0 and wp=wm
        ga = -1j * bose(wp, T) * model_A(wp)
        gb = -1j * bose(wm - wp, T) * model_A(wm - wp)
        return ga * gb

    def reg(wp):
        val = g(wp)
        if abs(wp) > 1e-9:
            val = val - r0 / wp
        if abs(wm - wp) > 1e-9:
            val = val - rw / (wm - wp)
        return val

    re = integrate.quad(lambda x: reg(x).real, -W, W, points=[0.0, wm], limit=400)[0]
    im = integrate.quad(lambda x: reg(x).imag, -W, W, points=[0.0, wm], limit=400)[0]
    pv0 = 0.0  # PV int_{-W}^{W} dwp/wp = 0 (symmetric)
    pvw = np.log(abs((wm + W) / (wm - W))) if abs(abs(wm) - W) > 1e-9 else 0.0
    return (re + 1j * im) + r0 * pv0 + rw * pvw


# ----------------------------------------------------------------------------
# method A: bare DC-zeroed discrete convolution (the CURRENT dc_handling='zero')
# ----------------------------------------------------------------------------
def sigma_bare(w, T):
    dw = w[1] - w[0]; ne = w.size; i0 = int(np.argmin(np.abs(w)))
    G = Glesser(w, T).copy(); G[i0] = 0.0
    out = np.zeros(ne, dtype=complex)
    for m in range(ne):
        for j in range(ne):
            k = m - j + i0
            if 0 <= k < ne:
                out[m] += G[j] * G[k]
    return out * dw


# ----------------------------------------------------------------------------
# method B: residue double-subtraction (what we will wire into the solver)
# ----------------------------------------------------------------------------
def sigma_subtracted(w, T):
    dw = w[1] - w[0]; ne = w.size; i0 = int(np.argmin(np.abs(w)))
    c = bose_pole_coeff(T); A0 = model_A(0.0)
    A = model_A(w)
    G = Glesser(w, T)
    Gfin = G.copy()
    Gfin[i0] = -1j * (c * (A[i0 + 1] - A[i0 - 1]) / (2 * dw) - 0.5 * A0)  # finite limit of nA
    out = np.zeros(ne, dtype=complex)
    pvw_all = np.where(np.abs(np.abs(w) - w[-1]) > 1e-9,
                       np.log(np.abs((w + w[-1]) / (w - w[-1]))), 0.0)
    for m in range(ne):
        wm = w[m]
        # g(w') = Ga(w') Gb(wm - w') on the grid (k = m - j + i0)
        kk = m - np.arange(ne) + i0
        valid = (kk >= 0) & (kk < ne)
        Gb = np.zeros(ne, dtype=complex); Gb[valid] = G[kk[valid]]
        Gb_fin = np.zeros(ne, dtype=complex); Gb_fin[valid] = Gfin[kk[valid]]
        g = Gfin * Gb_fin            # start from finite-sampled product
        # residues
        r0 = (-1j * c * A0) * Gfin[m]           # lim_{w'->0} w' Ga(w') Gb(wm-w') = (-icA0) Gb(wm)
        rw = Gfin[m] * (-1j * c * A0)           # lim_{w'->wm} (wm-w') g = Ga(wm)(-icA0)
        reg = g.astype(complex)
        # subtract pole at w'=0
        with np.errstate(divide="ignore", invalid="ignore"):
            reg = reg - np.where(np.abs(w) > 1e-9, r0 / w, 0.0)
        # node at w'=0: finite-difference derivative of (g - rw/(wm-w'))
        # subtract pole at w'=wm
        wmw = wm - w
        reg = reg - np.where(np.abs(wmw) > 1e-9, rw / wmw, 0.0)
        # fill the two removable nodes with central derivative of reg's neighbours
        for ip in (i0, m):
            if 0 < ip < ne - 1:
                reg[ip] = 0.5 * (reg[ip - 1] + reg[ip + 1])
            elif ip == 0:
                reg[ip] = reg[1]
            elif ip == ne - 1:
                reg[ip] = reg[-2]
        regular_integral = np.trapezoid(reg, w)
        out[m] = regular_integral + r0 * 0.0 + rw * pvw_all[m]
    return out


def main():
    T = 300.0
    W, NE = 12.0, 161
    w = np.linspace(-W, W, NE)
    wpos = w[w >= 0]
    dw = w[1] - w[0]
    ref = np.array([sigma_reference(wm, T, W) for wm in wpos])
    bare = sigma_bare(w, T)[w >= 0]
    sub = sigma_subtracted(w, T)[w >= 0]

    # The two poles (w'=0, w'=wm) are SEPARATED for wm >= 3 dw; coincident at
    # wm=0 (double pole, handled separately) and adjacent for wm < 3 dw (needs
    # the analytic first-interval limit -- a solver-side refinement).
    sep = wpos >= 3 * dw

    def relerr(x, m):
        return np.max(np.abs(x[m] - ref[m])) / np.max(np.abs(ref[m]))

    print("=" * 64)
    print("IR residue-subtracted bubble convolution vs scipy PV reference")
    print("=" * 64)
    print(f"grid NE={NE} W={W} dw={dw:.3f} T={T}")
    print("\n-- separated-pole regime (wm >= 3 dw): the decisive test --")
    print(f"  bare (DC-zeroed)        max rel err = {relerr(bare, sep):.3e}")
    print(f"  residue-subtracted      max rel err = {relerr(sub, sep):.3e}")
    print("\nlow-omega Re Sigma^< (sub matches the PV reference; bare is suppressed-wrong):")
    print(f"  omega : {np.round(wpos[:8],3).tolist()}")
    print(f"  ref   : {np.round(ref[:8].real,3).tolist()}")
    print(f"  bare  : {np.round(bare[:8].real,3).tolist()}")
    print(f"  sub   : {np.round(sub[:8].real,3).tolist()}")
    ok = relerr(sub, sep) < 0.02 and relerr(sub, sep) < relerr(bare, sep) / 5
    print(f"\nVERDICT: residue-subtraction {'WORKS' if ok else 'FAILS'} on separated poles "
          f"(sub {relerr(sub, sep):.2e} vs bare {relerr(bare, sep):.2e}); "
          f"wm<3dw needs the analytic first-interval limit (solver-side).")
    return ok


if __name__ == "__main__":
    main()
