"""Every observable, dense vertex vs decomposed (fig:res_decomp_observables).

Data:
  Data: phonon/scripts/data/decomposed_sse_spectra.npz.

Run:  python phonon/scripts/figures/decomposed_sse_observables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style
from phonon.solver.observables import bose

NPZ = ROOT / "phonon/scripts/data/decomposed_sse_spectra.npz"
FIGDIR = ROOT / "document/fig/transport_sweeps"

SHOW = [8, 32]          # ranks drawn against the dense reference
OMEGA_IR = 0.25         # THz: IR exclusion window (= 2 grid cells)
T = 300.0               # K, the mean lead temperature


def _legs(z, length):
    out = []
    if f"{length}_dense/energies" in z.files:
        out.append(("dense", "dense vertex", "0.25", "-", 2.0))
    for i, r in enumerate(SHOW):
        if f"{length}_r{r}/energies" in z.files:
            out.append((f"r{r}", rf"$R={r}$", f"C{3 if r == SHOW[-1] else 1}",
                        "--", 1.4))
    return out


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    z = np.load(NPZ)
    # prefer the length that actually has a dense reference
    length = next((L for L in ("L3", "L10") if f"{L}_dense/energies" in z.files),
                  None)
    if length is None:
        length = "L10"
        print("no dense leg in the archive -- drawing the ranks only")
    legs = _legs(z, length)
    if not legs:
        raise SystemExit("no legs to draw")

    fig, axes = style.figure(ncols=2, nrows=2, width=4.4, height=3.2)
    (a_ldos, a_occ), (a_abs, a_bub) = axes

    print(f"observables, {length}: dense vertex vs the decomposed vertex")
    for leg, lab, c, ls, lw in legs:
        tag = f"{length}_{leg}"
        f = np.abs(np.asarray(z[f"{tag}/energies"], dtype=float))
        ir = f > OMEGA_IR                      # the IR exclusion window

        gr = np.asarray(z[f"{tag}/gr_diag_imag"], dtype=float)   # (ne, N_D) = -Im G^R
        gl = np.asarray(z[f"{tag}/gl_diag_imag"], dtype=float)   #           = +Im G^<

        # The LDOS carries a 2*omega/pi Jacobian (phonon/solver/observables.py::
        # local_dos). Without it the omega=0 bin -- where -Im G^R diverges as the
        # acoustic 1/omega^2 -- is 99.97% of the total and flattens the spectrum.
        ldos = (2.0 * f / np.pi)[:, None] * gr
        ldos = ldos.sum(axis=1)
        a_ldos.plot(f[ir], ldos[ir], ls, color=c, lw=lw, label=lab)

        # n(w) = Im G^< / (2 * (-Im G^R)). Mask on the IR window, NOT on the
        # spectral weight: -Im G^R peaks at omega=0, so a "where there is
        # spectrum" threshold keeps only the DC bin and throws the band away.
        occ = np.divide(gl.sum(axis=1), 2.0 * gr.sum(axis=1),
                        out=np.full(f.shape, np.nan), where=gr.sum(axis=1) > 1e-12)
        a_occ.semilogy(f[ir], occ[ir], ls, color=c, lw=lw, label=lab)

        pabs = np.asarray(z[f"{tag}/slab_absorption"], dtype=float)[0]  # (n_slabs,)
        a_abs.plot(np.arange(1, len(pabs) + 1), pabs, ls, color=c, lw=lw,
                   marker="o", ms=3.5, label=lab)

        # P_in and P_out agree to ~1e-5 -- plotting both draws one curve twice.
        # The informative quantity is the NET rate P_out - P_in: the per-frequency
        # energy the three-phonon bubble takes out of the phonon system.
        bb = np.asarray(z[f"{tag}/bubble_balance_spectrum"], dtype=float)  # (2, ne)
        a_bub.plot(f[ir], (bb[1] - bb[0])[ir], ls, color=c, lw=lw, label=lab)

        # the LDOS sum rule: int rho dw should be N_dof (short by the part of the
        # optical band above fmax)
        dw = float(f[1] - f[0])
        print(f"  {lab:>14}: int LDOS = {ldos.sum() * dw:8.2f} of N_dof={gr.shape[1]}"
              f" ({100 * ldos.sum() * dw / gr.shape[1]:5.1f}%)   "
              f"sum P_abs {pabs.sum():9.4f}   "
              f"n(w) / n_Bose (median, in band) = "
              f"{np.nanmedian((occ / bose(f, T))[ir & (f < 0.8 * f.max())]):.3f}")

    # The Bose function is the verification gate on n(w): with no scattering the
    # ballistic occupation must lie between n_B(T_L) and n_B(T_R). Finite eta
    # pushes it BELOW Bose (the eta damping enters -Im G^R with no matching
    # fluctuation), so this overlay is how the eta=0 runs are checked.
    fb = np.abs(np.asarray(z[f"{length}_{legs[0][0]}/energies"], dtype=float))
    a_occ.semilogy(fb[fb > OMEGA_IR], bose(fb, T)[fb > OMEGA_IR], "-",
                   color="0.6", lw=2.2, zorder=0, label=rf"Bose, ${T:.0f}$ K")

    a_ldos.set_xlabel("frequency (THz)")
    a_ldos.set_ylabel(r"LDOS $\rho(\omega) = \frac{2\omega}{\pi}(-\mathrm{Im}\,G^R)$")
    a_ldos.legend(fontsize=7)

    a_occ.set_xlabel("frequency (THz)")
    a_occ.set_ylabel(r"occupation $n(\omega)$")
    a_occ.legend(fontsize=7)

    a_abs.set_xlabel("slab index (transport direction)")
    a_abs.set_ylabel(r"absorbed power $P_{\rm abs}$")
    a_abs.axhline(0.0, color="0.6", lw=0.8)
    a_abs.legend(fontsize=7)

    a_bub.set_xlabel("frequency (THz)")
    a_bub.set_ylabel(r"net bubble rate $P_{\rm out}(\omega)-P_{\rm in}(\omega)$")
    a_bub.axhline(0.0, color="0.6", lw=0.8)
    a_bub.legend(fontsize=7)

    for a in (a_ldos, a_occ, a_bub):
        a.axvspan(0, OMEGA_IR, color="0.9", zorder=0)

    style.save(fig, "decomp_observables", directory=FIGDIR)


if __name__ == "__main__":
    main()
