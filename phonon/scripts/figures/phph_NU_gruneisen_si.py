"""Normal vs Umklapp decomposition and mode Gruneisen for bulk Si (FD FC3).

Recomputes the phono3py RTA thermal conductivity with the N/U split
(is_N_U=True) and the FC3 mode Gruneisen parameters -- the two standard
measures of which scattering processes limit kappa and how strongly anharmonic
each mode is.  The (cheap, bounded-thread) phono3py recompute reads only
phonon/reaps/si_primitive_work/{phono3py.yaml, fc2.hdf5, fc3.hdf5}.

Panels:
  (a) per-mode Umklapp fraction of the 3-phonon scattering rate vs frequency,
      with a horizontal line at the kappa-weighted Umklapp fraction (and the
      unweighted scattering-rate-summed fraction printed for comparison);
  (b) mode Gruneisen distribution with the mean |gamma| line and the
      experimental bulk-Si value (~1) marked.

Run:  OMP_NUM_THREADS=8 python phonon/scripts/figures/phph_NU_gruneisen_si.py
Figure -> document/fig/transport_sweeps/phph_NU_gruneisen_si.{pdf,png}
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

import phono3py  # noqa: E402
from phono3py.phonon3.gruneisen import Gruneisen  # noqa: E402

WORK = ROOT / "phonon/reaps/si_primitive_work"
FIGDIR = ROOT / "document/fig/transport_sweeps"
MESH_NU = 13         # q-mesh for the N/U RTA recompute
MESH_GRUEN = 11      # q-mesh for the Gruneisen sampling
EXP_GRUENEISEN = 1.0  # experimental bulk-Si mode-averaged |gamma| ~ 1


def main():
    ph3 = phono3py.load(str(WORK / "phono3py.yaml"),
                        fc3_filename=str(WORK / "fc3.hdf5"),
                        fc2_filename=str(WORK / "fc2.hdf5"), log_level=0)

    # --- N/U-split RTA at 300 K ---
    ph3.mesh_numbers = [MESH_NU] * 3
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(temperatures=[300], is_N_U=True,
                                 boundary_mfp=1e6, write_kappa=False)
    tc = ph3.thermal_conductivity
    kappa = float(np.array(tc.kappa)[0, 0, 0])          # xx, W/mK
    gN, gU = (np.array(g)[0, 0] for g in tc.get_gamma_N_U())  # (nq, nb) THz
    freq = np.array(tc.frequencies)                     # (nq, nb) THz
    mk = np.array(tc.mode_kappa)[0, 0, :, :, 0]         # (nq, nb) xx
    g_tot = gN + gU
    m = g_tot > 1e-6
    u_mode = np.where(m, gU / np.where(m, g_tot, 1.0), np.nan)
    u_scat = gU[m].sum() / g_tot[m].sum()               # rate-summed
    u_kappa = np.nansum(mk[m] * u_mode[m]) / mk[m].sum()  # kappa-weighted

    print("=" * 64)
    print(f"BULK Si N/U + GRUNEISEN (phono3py, mesh {MESH_NU}^3, 300 K)")
    print("=" * 64)
    print(f"kappa(300K) = {kappa:.1f} W/mK")
    print(f"Umklapp fraction, kappa-weighted   = {u_kappa:.2f}")
    print(f"Umklapp fraction, rate-summed      = {u_scat:.2f} "
          f"(Normal = {1 - u_scat:.2f})")

    # --- mode Gruneisen from FC3 on a q-mesh ---
    gr = Gruneisen(ph3.fc2, ph3.fc3, ph3.phonon_supercell, ph3.phonon_primitive)
    gr.set_sampling_mesh([MESH_GRUEN] * 3)
    gr.run()
    graw = np.array(gr.gruneisen_parameters)            # (nq, nb, 3, 3)
    fr_g = np.ravel(np.array(gr.frequencies))
    if graw.size == fr_g.size * 9:
        gam = np.trace(graw.reshape(-1, 3, 3), axis1=1, axis2=2) / 3.0
    else:
        gam = np.ravel(graw)
    ok = np.isfinite(gam) & np.isfinite(fr_g) & (fr_g > 1e-3)
    gam, fr_g = gam[ok], fr_g[ok]
    mean_abs_g = float(np.mean(np.abs(gam)))
    print(f"mode Gruneisen: mean|g| = {mean_abs_g:.2f}, "
          f"max|g| = {np.abs(gam).max():.2f}  (exp. Si ~ {EXP_GRUENEISEN:g})")

    # --- figure ---
    fig, ax = style.doc_figure(ncols=2, aspect=0.36)

    ax[0].scatter(freq[m], u_mode[m], s=6, alpha=0.3, color="C3", lw=0)
    ax[0].axhline(u_kappa, color="0.2", lw=1.1, ls="--")
    ax[0].annotate(rf"$\kappa$-weighted: {u_kappa:.2f}",
                   (0.4, u_kappa - 0.075), fontsize=7.5, color="0.2")
    ax[0].axhline(u_scat, color="C0", lw=1.0, ls=":")
    ax[0].annotate(f"rate-summed: {u_scat:.2f}",
                   (8.5, u_scat + 0.035), fontsize=7.5, color="C0")
    ax[0].set_xlabel("frequency (THz)")
    ax[0].set_ylabel("Umklapp fraction of scattering")
    ax[0].set_ylim(-0.05, 1.05)

    ax[1].scatter(fr_g, gam, s=8, alpha=0.4, color="C0", lw=0)
    ax[1].axhline(mean_abs_g, color="0.2", lw=1.1, ls="--",
                  label=rf"mean $|\gamma|$ = {mean_abs_g:.2f}")
    ax[1].axhline(EXP_GRUENEISEN, color="C2", lw=1.1, ls=":",
                  label=r"exp. range ($|\gamma|\sim1$)")
    ax[1].set_xlabel("frequency (THz)")
    ax[1].set_ylabel("mode Grüneisen $\\gamma$")
    ax[1].legend(loc="lower right", fontsize=7.5)

    style.save(fig, "phph_NU_gruneisen_si", directory=FIGDIR)


if __name__ == "__main__":
    main()
