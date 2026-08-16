"""Step 1 (math gate): is the residue-subtracted 3-phonon bubble CONSERVING
and MORE accurate than the bare FFT bubble -- and how big is the correction
for a realistic (bounded, no-clean-pole) device G^<?

Run:  OMP_NUM_THREADS=1 python phonon/studies/_ir_conserving_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from quatrex.phonon.ir_subtraction import bose, bose_pole_coeff


# ---------------------------------------------------------------------------
# G models on a symmetric grid through 0, with the exact bosonic fold
# G^<(-w) = G^>(w).
# ---------------------------------------------------------------------------
def model_A(w):
    """Odd spectral function A(-w)=-A(w), A(0)=0, acoustic onset ~w near 0."""
    w = np.asarray(w, float)
    return np.sign(w) * (np.exp(-((np.abs(w) - 3.0) ** 2) / 2.0)
                         + 0.3 * np.abs(w) * np.exp(-w ** 2 / 8.0))


def greens(w, T, *, pole=0.0):
    """G^<(w) = -i n(w) A(w) + pole*(1/w);  G^>(w)=G^<(-w) by construction.
    `pole` injects a genuine odd 1/w pole to exercise the residue machinery."""
    A = model_A(w)
    n = bose(w, T)
    gl = -1j * n * A
    gg = -1j * (n + 1.0) * A
    if pole:
        i0 = int(np.argmin(np.abs(w)))
        inv = np.zeros_like(w); nz = np.abs(w) > 1e-9; inv[nz] = 1.0 / w[nz]
        gl = gl + pole * inv          # odd 1/w pole (same in gl, gg -> cancels in A)
        gg = gg + pole * inv
        gl[i0] = 0.0; gg[i0] = 0.0
    return gl, gg


# ---------------------------------------------------------------------------
# bubbles
# ---------------------------------------------------------------------------
def _conv(Fa, Fb, w):
    """Discrete symmetric convolution S(w)=sum_w' Fa(w')Fb(w-w') dw'."""
    dw = w[1] - w[0]; ne = w.size; i0 = int(np.argmin(np.abs(w)))
    out = np.zeros(ne, dtype=complex)
    for m in range(ne):
        k = m - np.arange(ne) + i0
        v = (k >= 0) & (k < ne)
        out[m] = np.sum(Fa[np.arange(ne)[v]] * Fb[k[v]]) * dw
    return out


def bubble_bare(gl, gg, w):
    return _conv(gl, gl, w), _conv(gg, gg, w)


def _extract_residue(g, w):
    """R = lim_{w->0} w g(w), estimated from the two bins straddling 0
    (odd pole -> w1 g(w1) ~ R).  Returns 0 if no pole (g bounded)."""
    i0 = int(np.argmin(np.abs(w)))
    if 0 < i0 < w.size - 1:
        return 0.5 * (w[i0 + 1] * g[i0 + 1] + (-w[i0 - 1]) * (-g[i0 - 1]))
    return 0.0 * g[i0]


def _conv_ressub(Fa, Fb, w):
    """Residue-subtracted convolution: subtract Fa's w'=0 pole (residue Ra) and
    Fb's w'=w pole (residue Rb), trapezoid the regular remainder, add analytic
    PV (=0 on a symmetric grid for the 1/w' kernel; the w'=w kernel gives
    ln|(w+W)/(w-W)|).  Reduces to bare when Ra=Rb=0."""
    dw = w[1] - w[0]; ne = w.size; i0 = int(np.argmin(np.abs(w))); W = w[-1]
    Ra = _extract_residue(Fa, w)
    out = np.zeros(ne, dtype=complex)
    for m in range(ne):
        k = m - np.arange(ne) + i0
        v = (k >= 0) & (k < ne)
        g = np.zeros(ne, dtype=complex); g[v] = Fa[np.arange(ne)[v]] * Fb[k[v]]
        # Fb(w_m - w') pole at w'=w_m: residue = Fa(w_m)*Rb
        Rb = Fa[m] * _extract_residue(Fb, w)
        reg = g.astype(complex)
        inv = np.zeros(ne); nz = np.abs(w) > 0.5 * dw; inv[nz] = 1.0 / w[nz]
        reg = reg - Ra * inv                       # remove w'=0 pole
        wmw = w[m] - w
        invm = np.zeros(ne); nzm = np.abs(wmw) > 0.5 * dw; invm[nzm] = 1.0 / wmw[nzm]
        reg = reg - Rb * invm                      # remove w'=w_m pole
        for ip in ({i0, m}):
            if 0 < ip < ne - 1:
                reg[ip] = 0.5 * (reg[ip - 1] + reg[ip + 1])
        S = np.sum(reg) * dw                        # trapezoid of the regular part
        # analytic PV: int dw'/w' = 0 (symmetric);  int dw'/(w_m-w') = ln|(w_m+W)/(w_m-W)|
        if abs(abs(w[m]) - W) > 1e-9:
            S = S + Rb * np.log(abs((w[m] + W) / (w[m] - W)))
        out[m] = S
    return out


def bubble_ressub(gl, gg, w):
    return _conv_ressub(gl, gl, w), _conv_ressub(gg, gg, w)


def bubble_taper(gl, gg, w, creg=1.0):
    dw = w[1] - w[0]
    t = w ** 2 / (w ** 2 + (creg * dw) ** 2)
    return _conv(gl * t, gl * t, w), _conv(gg * t, gg * t, w)


# ---------------------------------------------------------------------------
def conservation(sl, sg, gl, gg, w):
    """C = sum_w w (Sigma^< G^> - Sigma^> G^<); normalised."""
    num = np.sum(w * (sl * gg - sg * gl))
    den = np.sum(np.abs(w) * (np.abs(sl * gg) + np.abs(sg * gl))) + 1e-300
    return abs(num) / den


def run_case(pole, label):
    T = 300.0
    W, NE = 12.0, 161
    w = np.linspace(-W, W, NE)
    gl, gg = greens(w, T, pole=pole)
    # fine reference
    wf = np.linspace(-W, W, 8 * (NE - 1) + 1)
    glf, ggf = greens(wf, T, pole=pole)
    slf, sgf = bubble_ressub(glf, ggf, wf)   # ressub on fine grid = reference
    ref_l = np.interp(w, wf, slf.real) + 1j * np.interp(w, wf, slf.imag)

    out = {}
    for name, fn in [("bare", bubble_bare), ("taper", bubble_taper),
                     ("ressub", bubble_ressub)]:
        sl, sg = fn(gl, gg, w)
        C = conservation(sl, sg, gl, gg, w)
        # accuracy of Sigma^< on omega>=0 vs the fine reference
        m = w >= 0
        err = np.max(np.abs(sl[m] - ref_l[m])) / (np.max(np.abs(ref_l[m])) + 1e-300)
        out[name] = (C, err)
    print(f"\n=== {label}  (injected pole residue={pole}) ===")
    print(f"  R extracted from G^<: {complex(_extract_residue(gl, w)):.3e}")
    print(f"  {'method':8s}  {'conservation C':>16s}  {'accuracy vs ref':>16s}")
    for name in ("bare", "taper", "ressub"):
        C, err = out[name]
        print(f"  {name:8s}  {C:16.3e}  {err:16.3e}")
    return out


def main():
    print("=" * 70)
    print("CONSERVING residue-subtracted bubble -- math gate")
    print("=" * 70)
    run_case(0.0, "realistic device: bounded G^< (no clean pole, like the data)")
    run_case(0.5, "stress test: genuine odd 1/w pole injected into G^<")


if __name__ == "__main__":
    main()
