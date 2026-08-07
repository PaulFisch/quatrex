"""Grid EXTENT: measure the sideband weight above 2*omega_max.

Tests thesis claims #11, #12, #15, #16 of the grid audit
(phonon/docs/grid_audit.md). The thesis asserts, and flags with its own
todo at document/src/theory/50_computation.tex:106
("Measure the sideband weight above $2\\omega_{\\max}$ on a converged run
to bound the truncation error"):

  #11 in the BALLISTIC limit G^{<,>} lives inside [-w_max, w_max], so the
      first bubble is supported on [-2 w_max, 2 w_max]  (eq:conv_support)
  #12 self-consistency does NOT preserve that bound: the fixed point
      carries multi-phonon sidebands at every order, the n-phonon one at
      n-1 powers of |Phi3|^2, so the grid extent is a TRUNCATION, not an
      exact window
  #15 the grid must reach 2 w_max to retain the two-phonon shoulder,
      which carries no terminal heat but enters Re Sigma^R through
      Kramers-Kronig
  #16 a grid reaching exactly 2 w_max discards only sidebands carrying
      further powers of |Phi3|^2

The toy bed is the right instrument precisely because the cubic
amplitude g is a free knob, so the predicted |Phi3|^2-per-order scaling
of #12/#16 is directly falsifiable: fit log(sideband weight) vs log(g).

Method: converge the 2-DOF flat-band chain of _toy_grid_study on a grid
extending to EXT * w_max (default 4x) at fixed dw, then integrate the
scattering-rate density |Tr i(Sigma^> - Sigma^<)| and the spectral
weight |Tr A| over (w_max, 2 w_max] and (2 w_max, top], relative to the
whole positive axis.

Run:  QTX_ARRAY_MODULE=numpy OMP_NUM_THREADS=4 \
        python phonon/studies/_grid_sideband.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "phonon"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.studies._toy_grid_study import (  # noqa: E402
    N_SLABS, T_L, T_R, flatband_chain, run_case)
from solver.dense import _device_omega_max  # noqa: E402
from solver.grids import build_frequency_grid  # noqa: E402
from solver.leads import (  # noqa: E402
    build_device_hamiltonian, compute_obc_batch, solve_green_batch)
from solver.se_finite import (  # noqa: E402
    compute_phph_self_energy_finite_multi_slab)

OUT = ROOT / "phonon/studies/out/grid_audit"

EXT = 4.0          # grid top = EXT * omega_max
DW = 0.1467        # fixed spacing (= the E1 nf=240 rung)
# cubic amplitudes: the E1 calibration points and two more decades, so the
# |Phi3|^2 scaling of #12/#16 has a lever arm
G_VALUES = (7.4676e17, 2.3615e18, 4.0e18, 7.4676e18)


def _hwhm_interp(f, a):
    """Half-width at half-max with LINEARLY INTERPOLATED crossings.

    The grid-point reader of _toy_grid_study.measure_gamma quantises the
    width in units of dw, which on an extent ladder at fixed dw shows up
    as a spurious constant offset (measured: an identical 3.7e-3 'error'
    at 1.5x, 2x and 3x, which was pure quantisation). Interpolating the
    two half-max crossings removes it.
    """
    ipk = int(np.argmax(a))
    half = a[ipk] / 2.0
    lo = np.where(a[:ipk] < half)[0]
    hi = np.where(a[ipk:] < half)[0]
    if not lo.size or not hi.size:
        return float("nan")
    il, ir = int(lo[-1]), int(ipk + hi[0])

    def cross(i0, i1):
        y0, y1 = a[i0], a[i1]
        if y1 == y0:
            return f[i0]
        return f[i0] + (half - y0) * (f[i1] - f[i0]) / (y1 - y0)

    return 0.5 * (cross(ir, ir - 1) - cross(il, il + 1))


def _band_fracs(w, dens, w_max):
    """Fractions of |dens| on (w_max, 2 w_max] and (2 w_max, top]."""
    pos = w > 0
    ww, dd = w[pos], np.abs(dens[pos])
    tot = np.trapezoid(dd, ww)
    if tot <= 0:
        return dict(total=0.0, f_in=np.nan, f_shoulder=np.nan, f_beyond=np.nan)

    def seg(lo, hi):
        m = (ww > lo) & (ww <= hi)
        return float(np.trapezoid(dd[m], ww[m])) if m.sum() > 1 else 0.0

    return dict(
        total=float(tot),
        f_in=seg(0.0, w_max) / tot,
        f_shoulder=seg(w_max, 2 * w_max) / tot,
        f_beyond=seg(2 * w_max, ww[-1]) / tot,
        abs_beyond=seg(2 * w_max, ww[-1]),
    )


def ballistic_support(h00, h01, phi, fmax, nfreq_pos, w_max):
    """#11: one bubble on the BALLISTIC G -- support must stop at 2 w_max."""
    freqs, dw, _, z2, _, _ = build_frequency_grid(
        (0.01, fmax, nfreq_pos), eta_w_thz=1e-4)
    nfreq, n_dof = len(freqs), h00.shape[0]
    N_D = N_SLABS * n_dof
    h_d = build_device_hamiltonian(h00, h01, N_SLABS)
    obc = compute_obc_batch(z2, h00, h01, freqs, T_L, T_R, n_slabs=N_SLABS)
    zero = np.zeros((nfreq, N_D, N_D), complex)
    _, gl, gg = solve_green_batch(z2, h_d, obc, zero, zero, zero)
    phi_dev = {(i, i, i): phi.astype(complex) for i in range(N_SLABS)}

    def gd(dense):
        return {(k, kp): dense[:, k * n_dof:(k + 1) * n_dof,
                               kp * n_dof:(kp + 1) * n_dof]
                for k in range(N_SLABS) for kp in range(N_SLABS)}

    sl_b, sg_b = compute_phph_self_energy_finite_multi_slab(
        gd(gl), gd(gg), phi_dev, N_SLABS, freqs, dw,
        dc_handling="interpolate", n_threads=1)
    sl = np.zeros((nfreq, N_D, N_D), complex)
    sg = np.zeros_like(sl)
    for (i, j), b in sl_b.items():
        sl[:, i * n_dof:(i + 1) * n_dof, j * n_dof:(j + 1) * n_dof] = b
    for (i, j), b in sg_b.items():
        sg[:, i * n_dof:(i + 1) * n_dof, j * n_dof:(j + 1) * n_dof] = b
    gam = np.einsum("wii->w", 1j * (sg - sl))
    # ballistic G legs for reference
    a_ball = np.einsum("wii->w", -1j * (gg - gl))
    return freqs, gam, a_ball


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"EXT": EXT, "DW": DW, "g_values": list(G_VALUES)}

    h00, h01, phi = flatband_chain(g=G_VALUES[1])
    w_max = float(_device_omega_max(h00, h01))
    fmax = EXT * w_max
    nfreq_pos = int(round(fmax / DW))
    report.update(omega_max=w_max, fmax=fmax, nfreq_pos=nfreq_pos,
                  dw=fmax / nfreq_pos)
    print(f"device omega_max = {w_max:.4f} THz;  grid top {fmax:.2f} THz "
          f"= {EXT}x;  nfreq_pos {nfreq_pos}, dw {fmax/nfreq_pos:.5f}")
    print(f"band edges: w_max {w_max:.3f}, 2*w_max {2*w_max:.3f}")

    # ---- #11: ballistic first-Born support ----
    print("\n[#11] first Born on BALLISTIC G (support must stop at 2 w_max):")
    for g in (G_VALUES[1],):
        h00, h01, phi = flatband_chain(g=g)
        w, gam, a_ball = ballistic_support(h00, h01, phi, fmax, nfreq_pos,
                                           w_max)
        fs = _band_fracs(w, gam, w_max)
        fa = _band_fracs(w, a_ball, w_max)
        print(f"  ballistic legs A:  in-band {fa['f_in']:.6f}  "
              f"shoulder {fa['f_shoulder']:.3e}  beyond2 {fa['f_beyond']:.3e}")
        print(f"  first-Born Gamma_S: in-band {fs['f_in']:.6f}  "
              f"shoulder {fs['f_shoulder']:.6f}  beyond2 {fs['f_beyond']:.3e}"
              "   <- claim #11: beyond2 must be ~0")
        report["born_ballistic"] = {"sigma": fs, "A_legs": fa, "g": g}

    # ---- #12/#16: self-consistent sidebands vs the cubic amplitude ----
    print("\n[#12/#16] CONVERGED fixed point: weight above 2*w_max vs g")
    print(f"  {'g':>12} {'conv':>6} {'it':>4} {'shoulder':>11} "
          f"{'beyond2(sig)':>13} {'beyond2(A)':>11}")
    rows = []
    for g in G_VALUES:
        h00, h01, phi = flatband_chain(g=g)
        res = run_case(h00, h01, phi, nfreq_pos, fmax, mixing=0.2,
                       max_iter=400, tol=1e-7, return_greens=True)
        w = res["freqs"]
        sl, sg = res["Sigma_l"][0], res["Sigma_g"][0]
        gam = np.einsum("wii->w", 1j * (sg - sl))
        gr = res["G_retarded"][0]
        a = np.einsum("wii->w", -2.0 * gr.imag)
        fs = _band_fracs(w, gam, w_max)
        fa = _band_fracs(w, a, w_max)
        rows.append(dict(g=g, converged=bool(res["converged"]),
                         n_iter=int(len(res["convergence_history"])),
                         sigma=fs, A=fa))
        print(f"  {g:12.4e} {str(res['converged']):>6} "
              f"{len(res['convergence_history']):4d} "
              f"{fs['f_shoulder']:11.4e} {fs['f_beyond']:13.4e} "
              f"{fa['f_beyond']:11.4e}")
    report["sc_ladder"] = rows

    # scaling exponent: sideband weight vs g  (claim: n-phonon sideband
    # carries n-1 extra powers of |Phi3|^2, i.e. of g^2). CONVERGED rungs
    # only -- a run stopped at the iteration cap has no fixed point to
    # attribute weight to.
    conv = [r for r in rows if r["converged"]]
    gg = np.array([r["g"] for r in conv], float)
    for key, lab in (("f_beyond", "relative"), ("abs_beyond", "absolute")):
        yy = np.array([r["sigma"][key] for r in conv], float)
        m = np.isfinite(yy) & (yy > 0)
        if m.sum() >= 3:
            sl_ = np.polyfit(np.log(gg[m]), np.log(yy[m]), 1)[0]
            print(f"  scaling of {lab} beyond-2w_max weight ({m.sum()} "
                  f"converged rungs): d log W / d log g = {sl_:.3f}")
            report[f"scaling_{key}"] = float(sl_)
    print("  (claim #12/#16: one extra power of |Phi3|^2 per sideband "
          "order => relative exponent 2)")

    # ---- #13/#15: the truncation ERROR, not just the weight ----
    # Weight above the top is not the same as the damage done by cutting
    # there. Ladder the grid top at FIXED dw and compare the converged
    # fixed point against the EXT*w_max reference.
    print(f"\n[#13/#15] truncation ERROR vs grid top (fixed dw, ref = "
          f"{EXT}x w_max)")
    g_ref = G_VALUES[1]
    h00, h01, phi = flatband_chain(g=g_ref)
    ref = None
    lad = []
    print(f"  {'top/w_max':>10} {'nfreq':>6} {'conv':>6} {'Gamma_em':>10} "
          f"{'J':>12} {'dGamma/Gamma':>13} {'dJ/J':>10}")
    # dw must be IDENTICAL across rungs or the comparison is confounded by
    # pole-grid re-registration (claim #4's mechanism): a naive ladder at
    # constant TARGET dw rounds nfreq per rung, moving the actual dw by
    # ~0.4% and the measured width by ~5% -- larger than the truncation
    # effect being measured. So fix dw exactly and let the top land on the
    # nearest multiple of it.
    for mult in (EXT, 3.0, 2.0, 1.75, 1.5, 1.35, 1.2, 1.1, 1.0):
        npos = int(round(mult * w_max / DW))
        top = npos * DW
        res = run_case(h00, h01, phi, npos, top, mixing=0.2, max_iter=400,
                       tol=1e-7, return_greens=True)
        w = res["freqs"]
        gr = res["G_retarded"][0]
        a_b = -gr[:, 1, 1].imag - gr[:, 3, 3].imag
        pos = w > 0
        f, a = w[pos], a_b[pos]
        gam_em = _hwhm_interp(f, a)
        jj = float(np.real(np.sum(res["spectral_J_L"])) * res["dw"])
        if ref is None:
            ref = (gam_em, jj)
        dg = (gam_em - ref[0]) / ref[0] if np.isfinite(gam_em) else np.nan
        dj = (jj - ref[1]) / ref[1]
        lad.append(dict(mult=mult, top=top, nfreq_pos=npos,
                        converged=bool(res["converged"]),
                        gamma_em=float(gam_em), J=jj,
                        rel_dgamma=float(dg), rel_dJ=float(dj)))
        print(f"  {mult:10.2f} {npos:6d} {str(res['converged']):>6} "
              f"{gam_em:10.5f} {jj:12.5e} {dg:+13.3e} {dj:+10.3e}")
    report["extent_ladder"] = lad

    (OUT / "sideband.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT / 'sideband.json'}")


if __name__ == "__main__":
    main()
