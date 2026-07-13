"""Every observable, dense vertex vs decomposed (fig:res_decomp_observables).

  decomp_observables  the full set of NEGF observables the SCBA produces, drawn
                      for the dense-vertex run and for the decomposed vertex at
                      two ranks:
                        LDOS(w)                 -Im G^R, q-averaged
                        occupation n(w)         Im G^< / (2 (-Im G^R))
                        per-slab absorption     P_abs(x), the 3-phonon energy sink
                        spectral bubble balance P_in(w) vs P_out(w)

The point of the figure is that the low-rank curves are not merely close in the
integrated conductance -- they lie on top of the dense ones across the whole
spectrum and across the device, including the observables nobody tunes for. The
per-slab absorption is the sharpest of these: it is a local quantity, it is what
the vertex directly controls, and it is where a bad fit would show first.

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

NPZ = ROOT / "phonon/scripts/data/decomposed_sse_spectra.npz"
FIGDIR = ROOT / "document/fig/transport_sweeps"

SHOW = [8, 32]          # ranks drawn against the dense reference


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

        gr = np.asarray(z[f"{tag}/gr_diag_imag"], dtype=float)   # (ne, N_D)
        gl = np.asarray(z[f"{tag}/gl_diag_imag"], dtype=float)
        ldos = gr.sum(axis=1)
        a_ldos.plot(f, ldos, ls, color=c, lw=lw, label=lab)

        # occupation where there is spectral weight
        live = gr.sum(axis=1) > 1e-3 * gr.sum(axis=1).max()
        occ = np.full_like(ldos, np.nan)
        occ[live] = gl.sum(axis=1)[live] / (2.0 * gr.sum(axis=1)[live])
        a_occ.semilogy(f[live], occ[live], ls, color=c, lw=lw, label=lab)

        pabs = np.asarray(z[f"{tag}/slab_absorption"], dtype=float)[0]  # (n_slabs,)
        a_abs.plot(np.arange(1, len(pabs) + 1), pabs, ls, color=c, lw=lw,
                   marker="o", ms=3.5, label=lab)

        bb = np.asarray(z[f"{tag}/bubble_balance_spectrum"], dtype=float)  # (2, ne)
        a_bub.plot(f, bb[0], ls, color=c, lw=lw, label=rf"{lab}, $P_{{\rm in}}$")
        a_bub.plot(f, bb[1], ":", color=c, lw=lw)

        print(f"  {lab:>14}: sum LDOS {ldos.sum():10.3f}   "
              f"sum P_abs {pabs.sum():10.4f}")

    a_ldos.set_xlabel("frequency (THz)")
    a_ldos.set_ylabel(r"LDOS  $-\mathrm{Im}\,G^R$ (a.u.)")
    a_ldos.legend(fontsize=7)

    a_occ.set_xlabel("frequency (THz)")
    a_occ.set_ylabel(r"occupation $n(\omega)$")
    a_occ.legend(fontsize=7)

    a_abs.set_xlabel("slab index (transport direction)")
    a_abs.set_ylabel(r"absorbed power $P_{\rm abs}$")
    a_abs.axhline(0.0, color="0.6", lw=0.8)
    a_abs.legend(fontsize=7)

    a_bub.set_xlabel("frequency (THz)")
    a_bub.set_ylabel(r"$P_{\rm in}(\omega)$ (solid), $P_{\rm out}(\omega)$ (dotted)")
    a_bub.legend(fontsize=6.5)

    style.save(fig, "decomp_observables", directory=FIGDIR)


if __name__ == "__main__":
    main()
