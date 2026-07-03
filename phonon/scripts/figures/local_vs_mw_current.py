"""Local (bond) vs Meir-Wingreen heat current: the continuity identity figure.

Two panels (cnt33 L3): measured per-interface heat current (points; indices
0/-1 are the lead Meir-Wingreen currents, interior are Hardy bond currents)
vs the energy-continuity reconstruction J_0 - cumsum(P_abs) from the
per-slab scattering absorption (line). Left: eta=1e-12 conserving fixed
point -- the reconstruction reproduces the interior profile exactly (the
interior dip is the energy carried by the three-phonon interaction channel,
which the harmonic bond current does not see). Right: eta=0.7 -- the extra
mismatch is the finite-eta ghost-reservoir absorption (the ordering
commutator), slab-resolved.

Data: phonon/studies/out/local_mw/{L3_eta0,L3_eta07}/run.npz (launch recipe
in phonon/scripts/verify/local_vs_mw_current.py).

Run:  OMP_NUM_THREADS=1 python phonon/scripts/figures/local_vs_mw_current.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style  # noqa: E402

DATA = ROOT / "phonon/studies/out/local_mw"
FIGDIR = ROOT / "document/fig/transport_sweeps"

RUNS = [("L3_eta0", r"$\eta=10^{-12}$ (conserving fixed point)"),
        ("L3_eta07", r"$\eta=0.7$ THz (ghost-reservoir regime)")]


def main() -> None:
    panels = []
    for tag, title in RUNS:
        npz = DATA / tag / "run.npz"
        if not npz.exists():
            print(f"[skip] {npz} missing -- run the recipe in "
                  "scripts/verify/local_vs_mw_current.py first")
            return
        z = np.load(npz, allow_pickle=True)
        if "slab_absorption" not in z.files:
            print(f"[skip] {npz}: no slab_absorption key")
            return
        heat = np.asarray(z["last_heat"], dtype=float).reshape(-1)
        pa = np.real(np.asarray(z["slab_absorption"]))
        panels.append((tag, title, heat, pa))

    fig, axes = style.figure(ncols=2, width=4.6, height=3.4)
    for ax, (tag, title, heat, pa) in zip(np.atleast_1d(axes), panels):
        k = np.arange(heat.size)
        recon = np.concatenate(([heat[0]], heat[0] - np.cumsum(pa)))
        ax.plot(k, heat, "o", ms=8, color="C0", label="measured $J_k$")
        ax.plot(k, recon, "-s", ms=4, lw=1.2, color="C3",
                label=r"$J_0-\sum_{j<k} P_{\rm abs}(j)$")
        ax.plot([k[0], k[-1]], [heat[0], heat[-1]], "*", ms=13, color="C0",
                mfc="none", label="lead Meir--Wingreen")
        resid = np.abs(heat[1:] - recon[1:]).max() / np.abs(heat).mean()
        ax.annotate(rf"max identity residual ${resid:.1e}$",
                    (0.5, 0.04), xycoords="axes fraction", ha="center",
                    fontsize=7.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("interface $k$ (0 = left lead)")
        ax.set_xticks(k)
        print(f"[{tag}] J = {np.round(heat, 4)}  P_abs = {np.round(pa, 4)}  "
              f"resid = {resid:.2e}")
    np.atleast_1d(axes)[0].set_ylabel(r"heat current $\sum_\omega \hbar\omega\,I_k(\omega)$")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="center")
    style.save(fig, "local_vs_mw_current", directory=FIGDIR)


if __name__ == "__main__":
    main()
