"""MoS2 force-constant fit audit (fig:res_mos2_fits).

Data:
  Data (committed copies of the campaign result files):
    phonon/scripts/data/mos2_refit_results_56.json  original 56-structure set
    phonon/scripts/data/mos2_refit_results_80.json  +2 VASP batches (80)
    phonon/scripts/data/mos2_scp_results.json       o3-vs-o4 CV + SCP gates
  originals in cluster/mos2_refit{,80}/refit_results.json and
  cluster/mos2_scp300v2/scp_results.json (gitignored run dirs). The
  cross_frob entries are Frobenius norms of the device FC3 blocks whose
  anchor and partner atoms lie in different slabs (the vdW-gap
  couplings); cross_frob_folds_* are their spread over k-fold refits.

Run:  python phonon/scripts/figures/mos2_fit_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.studies import style

DATA = ROOT / "phonon/scripts/data"
FIGDIR = ROOT / "document/fig/transport_sweeps"

METHODS = ["least-squares", "ridge", "bayesian-ridge", "ardr"]
LABELS = ["LS", "ridge", "bay.\nridge", "ARDR"]


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    s56 = json.loads((DATA / "mos2_refit_results_56.json").read_text())
    s80 = json.loads((DATA / "mos2_refit_results_80.json").read_text())
    scp = json.loads((DATA / "mos2_scp_results.json").read_text())

    fig, (ax_a, ax_b) = style.doc_figure(ncols=2, aspect=0.38)
    colors = style.RC["axes.prop_cycle"].by_key()["color"]

    x = np.arange(len(METHODS))
    wdt = 0.36
    for off, s, lab, c in ((-wdt / 2, s56, "56 structures", colors[0]),
                           (wdt / 2, s80, "80 structures", colors[1])):
        cf = [s[m]["cross_frob"] for m in METHODS]
        err = [s[m]["cross_frob_folds_std"] for m in METHODS]
        ax_a.bar(x + off, cf, wdt, yerr=err, capsize=2.5, color=c, label=lab)
    ax_a.set_xticks(x, LABELS)
    ax_a.set_ylabel(r"cross-slab $\|\Phi^{(3)}\|_F$ (eV\,\AA$^{-3}$amu$^{-3/2}$)")
    ax_a.legend()

    for off, s, lab, c in ((-wdt / 2, s56, "56", colors[0]),
                           (wdt / 2, s80, "80", colors[1])):
        cv = [1e3 * s[m]["rmse_cv"] for m in METHODS]
        ax_b.bar(x + off, cv, wdt, color=c)
    ax_b.axhline(1e3 * scp["rmse_cv_o4"], color=colors[2], ls="--", lw=1.2)
    ax_b.annotate("o4 (quartic) LS", (0.02, 1e3 * scp["rmse_cv_o4"]),
                  textcoords="offset points", xytext=(2, 3), fontsize=8,
                  color=colors[2])
    ax_b.set_xticks(x, LABELS)
    ax_b.set_ylabel(r"CV force RMSE (meV\,\AA$^{-1}$)")

    style.save(fig, "mos2_fit_audit", directory=FIGDIR)

    for name, s in (("56", s56), ("80", s80)):
        ls, ar = s["least-squares"], s["ardr"]
        spread = ls["cross_frob_folds_std"] / ls["cross_frob_folds_mean"]
        print(f"set {name}: LS cross_frob {ls['cross_frob']:.1f} "
              f"(fold spread {spread:.1%}), ARDR cross_frob "
              f"{ar['cross_frob']:.3g}; CV LS {ls['rmse_cv']:.4f} vs ARDR "
              f"{ar['rmse_cv']:.4f} ({ar['rmse_cv'] / ls['rmse_cv'] - 1:+.0%})")
    print(f"o3 -> o4 CV: {scp['rmse_cv_o3']:.4f} -> {scp['rmse_cv_o4']:.4f} "
          f"({scp['rmse_cv_o4'] / scp['rmse_cv_o3'] - 1:+.0%}), "
          f"o4 DOFs {scp['n_dofs_o4']}, cutoffs4 {scp['cutoffs4']}")
    for tag in ("gates_bare", "gates_scp_last", "gates_scp_tailavg"):
        g = scp[tag]["gamma_low6_THz"]
        print(f"{tag}: Gamma shear doublet {g[3]:.4f}/{g[4]:.4f} THz, "
              f"breathing {g[5]:.4f} THz")


if __name__ == "__main__":
    main()
