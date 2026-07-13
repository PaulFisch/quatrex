"""Anharmonic vs harmonic transport with the decomposed vertex (fig:res_decomp_harmonic).

  decomp_harmonic  left:  effective transmission T_eff(w) = I(w)/[n_L(w)-n_R(w)],
                          harmonic (vertex zeroed) against the three-phonon SCBA
                          at several ranks;
                   right: spectral heat current j(w) = h f I(w), same legs;
                   inset/annotation: the integrated ratio G_anh/G_ball.

READ THE RATIO BEFORE QUOTING IT. On this film at eta = 0.4 THz the anharmonic
effective transmission comes out ABOVE the harmonic one (G_anh/G_ball > 1), most
strongly at the top of the band where T_ball is nearly zero. That is not a rank
artefact -- every rank reproduces it to three digits, and the dense vertex does
too -- and it is not the decomposition's doing. It is the finite-eta / short-device
regime: the anharmonic Sigma broadens the spectral function past the harmonic band
edge and leaks current into frequencies that carry none ballistically. The repo's
own physicality audit (phonon/scripts/figures/transmission_physicality.py) tests
exactly this T_anh <= T_ball condition. The figure shows it rather than hiding it.

Data: phonon/scripts/data/decomposed_sse_spectra.npz (harmonic leg = `ball`,
      produced with QX_BALLISTIC=1, which zeroes the vertex in place and keeps
      the leads/OBC/Meir-Wingreen machinery identical).

Run:  python phonon/scripts/figures/decomposed_sse_harmonic.py
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

H_SI = 6.62607015e-34
SHOW = [8, 32]          # ranks drawn against the harmonic reference


def _spectrum(z, tag):
    """(freqs, T_eff, j[nW/THz]) for one leg of the committed archive."""
    f = np.abs(np.asarray(z[f"{tag}/energies"], dtype=float))
    spec = np.asarray(z[f"{tag}/current_spectrum"], dtype=float)   # (ne, n_int)
    lead = spec[:, 0]
    lead = np.sign(np.nansum(lead)) * lead
    if f[0] < 1e-6:
        lead = lead.copy()
        lead[0] = 0.0                                   # the DC bin carries no heat
    dn = bose(f, float(z[f"{tag}/t_left"])) - bose(f, float(z[f"{tag}/t_right"]))
    t_eff = np.where(dn > 0, lead / np.where(dn > 0, dn, 1.0), 0.0)
    j_nw = H_SI * (f * 1e12) * lead * 1e12 * 1e9        # W/THz -> nW/THz
    return f, t_eff, j_nw


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    z = np.load(NPZ)
    lengths = [L for L in ("L3", "L10") if f"{L}_ball/energies" in z.files]
    if not lengths:
        raise SystemExit("no ballistic leg in the archive -- run QX_BALLISTIC=1")

    fig, axes = style.figure(ncols=2, nrows=len(lengths), width=4.4, height=3.2,
                             squeeze=False)

    for row, L in enumerate(lengths):
        a0, a1 = axes[row]
        fb, tb, jb = _spectrum(z, f"{L}_ball")
        a0.plot(fb, tb, "-", color="C0", lw=1.8, label="harmonic")
        a1.plot(fb, jb, "-", color="C0", lw=1.8, label="harmonic")

        ok = fb > 1e-9
        print(f"\n{L}:  harmonic  sum T_eff = {np.nansum(tb[ok]):.3f}")
        for i, r in enumerate([x for x in SHOW if f"{L}_r{x}/energies" in z.files]):
            f, t, j = _spectrum(z, f"{L}_r{r}")
            c = "C3" if r == SHOW[-1] else "C1"
            a0.plot(f, t, "--" if i == 0 else "-", color=c, lw=1.4,
                    label=rf"anharmonic, $R={r}$")
            a1.plot(f, j, "--" if i == 0 else "-", color=c, lw=1.4,
                    label=rf"anharmonic, $R={r}$")
            ratio = np.nansum(t[ok]) / np.nansum(tb[ok])
            print(f"      R={r:<3}  sum T_eff = {np.nansum(t[ok]):.3f}   "
                  f"G_anh/G_ball = {ratio:.4f}")
        # the dense vertex, where we have it -- proves the ratio is not a rank effect
        if f"{L}_dense/energies" in z.files:
            f, t, j = _spectrum(z, f"{L}_dense")
            a0.plot(f, t, ":", color="0.25", lw=1.6, label="anharmonic, dense")
            a1.plot(f, j, ":", color="0.25", lw=1.6, label="anharmonic, dense")
            print(f"      dense sum T_eff = {np.nansum(t[ok]):.3f}   "
                  f"G_anh/G_ball = {np.nansum(t[ok]) / np.nansum(tb[ok]):.4f}")

        a0.set_ylabel(rf"$T_{{\rm eff}}(\omega)$  ({L})")
        a1.set_ylabel(rf"$j(\omega)$ (nW/THz)  ({L})")
        for a in (a0, a1):
            a.set_xlabel("frequency (THz)")
            a.legend(fontsize=7)

    style.save(fig, "decomp_harmonic", directory=FIGDIR)


if __name__ == "__main__":
    main()
