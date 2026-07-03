"""Local (bond) vs Meir-Wingreen heat current: the energy-continuity identity.

The solver's per-interface current array holds the SAME structural form
Tr[Sigma^> G^< - G^> Sigma^<] everywhere -- the first/last entries plug the
lead OBC self-energy (= the Meir-Wingreen contact current), the interior
entries plug the dynamical-matrix embedding (= the Hardy bond current). They
are related by steady-state ENERGY CONTINUITY per slab k:

    J_{k+1} = J_k - P_abs(k),
    P_abs(k) = sum_w hbar*w Tr_k[Sigma_s^> G^< - Sigma_s^< G^>]

(the block-resolved bubble balance, saved as ``slab_absorption`` since
2026-07-03), up to the finite-eta ordering-commutator absorption. The bond
current only counts energy flowing through the HARMONIC inter-slab
couplings; P_abs(k) is the energy diverted into (out of) the three-phonon
interaction channel at slab k -- the interaction itself transports energy
between slabs through the off-diagonal scattering Sigma, which the bond
current does not see but the lead Meir-Wingreen currents do. Telescoping
over the device reproduces the global bubble balance
sum_k P_abs(k) = P_out - P_in (= -J_s), machine-zero for the conserving
vertex, hence J_L = J_R while the interior dips.

This script CHECKS that identity interface-by-interface on saved runs.

Inputs (produced by phonon/studies/engine/run.py with the 2026-07-03
``slab_absorption`` snapshot key; launch recipe used for the committed data):

    cd <repo>
    BASE=phonon/scripts/out/prod/cnt33_eta0/work/L3     # geometry + config
    for tag in L3_eta0 L3_eta07; do  # eta07: eta=0.7, retarded=half, 150 it
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 QUATREX_PHPH_RING_THREADS=16 \
      QX_CONFIG=$PWD/phonon/studies/out/local_mw/$tag/quatrex_config.toml \
      QX_NPZ=$PWD/phonon/studies/out/local_mw/$tag/run.npz \
      nohup python phonon/studies/engine/run.py > phonon/studies/out/local_mw/$tag.log 2>&1 &
    done

Run:  python phonon/scripts/verify/local_vs_mw_current.py [npz ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = [ROOT / "phonon/studies/out/local_mw/L3_eta0/run.npz",
           ROOT / "phonon/studies/out/local_mw/L3_eta07/run.npz"]


def analyze(npz_path: Path) -> dict | None:
    z = np.load(npz_path, allow_pickle=True)
    if "slab_absorption" not in z.files:
        print(f"[skip] {npz_path}: no slab_absorption key (rerun with the "
              "2026-07-03 run.py)")
        return None
    heat = np.asarray(z["last_heat"], dtype=float).reshape(-1)  # n_blocks+1
    pa = np.real(np.asarray(z["slab_absorption"]))              # n_blocks
    eta = float(z.get("eta", np.nan))
    conv = bool(z.get("converged", False))

    recon = heat[0] - np.cumsum(pa)              # predicted J_1..J_n
    resid = heat[1:] - recon
    scale = abs(heat).mean()

    # Global closure: binned == unbinned bubble balance (same instant).
    closure = np.nan
    if "final_bubble_balance" in z.files:
        p_in, p_out = np.asarray(z["final_bubble_balance"])
        closure = abs(pa.sum() - np.real(p_out - p_in)) / max(
            abs(np.real(p_out - p_in)), 1e-300)

    print(f"\n== {npz_path.parent.name}  (eta={eta:g}, converged={conv})")
    print(f"   interfaces J_k      : {np.round(heat, 4)}")
    print(f"   P_abs per slab      : {np.round(pa, 4)}   "
          f"sum={pa.sum():.3e}")
    print(f"   reconstruction J_0-cumsum(P_abs): {np.round(recon, 4)}")
    print(f"   identity residual   : {np.round(resid, 6)}   "
          f"max|resid|/|J| = {abs(resid).max() / scale:.3e}")
    print(f"   binned-vs-global closure rel err: {closure:.3e}")
    return dict(heat=heat, pa=pa, recon=recon, eta=eta,
                resid_rel=abs(resid).max() / scale)


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] or DEFAULT
    worst_eta0 = 0.0
    for p in paths:
        if not p.exists():
            print(f"[skip] {p}: missing")
            continue
        r = analyze(p)
        if r and r["eta"] < 1e-6:
            worst_eta0 = max(worst_eta0, r["resid_rel"])
    if worst_eta0 and worst_eta0 > 1e-4:
        print(f"\nFAIL: eta=0 identity residual {worst_eta0:.2e} > 1e-4")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
