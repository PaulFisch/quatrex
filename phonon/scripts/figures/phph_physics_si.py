"""Mode-resolved three-phonon physics for bulk Si from the phono3py RTA result.

Reads kappa-m191919.hdf5 (FD FC3, 19^3 q-mesh, 300 K) and shows the three
observables the nanowire argument rests on:
  (a) mode lifetimes tau(omega) with the tau ~ omega^-2 envelope
      [tau = 1/(4*pi*gamma), gamma the phono3py HWHM linewidth in THz];
  (b) kappa-accumulation vs phonon mean free path, with the 50%-accumulation
      MFP, the kappa fraction below 10 nm, and the ~1 nm wire-core band
      (near-zero garbage MFPs from v_g~0 modes are dropped from the
      percentile computation, x clipped to [1e0, 1e5] nm);
  (c) the mode-kappa spectrum split acoustic (band index < 3) vs optical,
      with the acoustic share of kappa.

Reads ONLY on-disk data:
  phonon/reaps/si_primitive_work/kappa-m191919.hdf5

Run:  OMP_NUM_THREADS=1 python phonon/scripts/figures/phph_physics_si.py
Figure -> document/fig/transport_sweeps/phph_physics_si.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style  # noqa: E402

KAPPA_H5 = ROOT / "phonon/reaps/si_primitive_work/kappa-m191919.hdf5"
FIGDIR = ROOT / "document/fig/transport_sweeps"
MFP_LO, MFP_HI = 1e0, 1e5           # nm, review-mandated x clip
WIRE_CORE = (1.0, 2.0)              # nm, d5a/d11a core-diameter band


def main():
    with h5py.File(KAPPA_H5, "r") as f:
        T = np.array(f["temperature"])
        it = int(np.argmin(np.abs(T - 300)))
        freq = np.array(f["frequency"])              # (nq, nb) THz
        gamma = np.array(f["gamma"])[it]             # (nq, nb) THz (HWHM)
        vg = np.array(f["group_velocity"])           # (nq, nb, 3) THz*A
        mode_kappa = np.array(f["mode_kappa"])[it]   # (nq, nb, 6) W/mK
        weight = np.array(f["weight"])               # (nq,)
        kappa = np.array(f["kappa"])[it]             # (6,)
    nb = freq.shape[1]

    # lifetime tau [ps] = 1/(4*pi*gamma[THz]); MFP = |v_g| * tau
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = 1.0 / (4.0 * np.pi * gamma)            # ps
        vg_mag = np.linalg.norm(vg, axis=2) * 100.0  # THz*A -> m/s
        mfp_nm = vg_mag * (tau * 1e-12) * 1e9        # -> nm
    mk = mode_kappa[:, :, 0]                         # xx component, W/mK

    valid = np.isfinite(tau) & (gamma > 1e-4) & (freq > 1e-3)
    fr_v, tau_v, mk_v = freq[valid], tau[valid], mk[valid]

    # --- kappa accumulation vs MFP (drop garbage near-zero MFPs: v_g ~ 0) ---
    acc = valid & (mfp_nm >= MFP_LO)
    n_dropped = int(valid.sum() - acc.sum())
    k_dropped = mk[valid & (mfp_nm < MFP_LO)].sum() / mk[valid].sum()
    order = np.argsort(mfp_nm[acc])
    mfp_sorted = mfp_nm[acc][order]
    kcum = np.cumsum(mk[acc][order])
    kcum /= kcum[-1]
    mfp50 = mfp_sorted[np.searchsorted(kcum, 0.5)]
    mfp90 = mfp_sorted[np.searchsorted(kcum, 0.9)]
    frac10 = float(kcum[np.searchsorted(mfp_sorted, 10.0) - 1])

    # --- acoustic share of kappa ---
    ac_band = mk[:, :3].sum() / mk.sum()             # band index < 3
    ac_freq = mk[freq < 8.0].sum() / mk.sum()        # legacy f < 8 THz cut

    print("=" * 64)
    print("BULK Si 3-PHONON PHYSICS (phono3py RTA, 19^3 mesh, 300 K)")
    print("=" * 64)
    print(f"kappa(300K) = {kappa[0]:.1f} W/mK  (sum_IBZ mode_kappa / N_mesh = "
          f"{mk.sum() / weight.sum():.1f})")
    print(f"median lifetime = {np.median(tau_v):.2f} ps; max = {tau_v.max():.1f} ps")
    print(f"MFP percentiles exclude {n_dropped} modes with MFP < {MFP_LO:g} nm "
          f"(v_g ~ 0 garbage; {100 * k_dropped:.2g}% of kappa)")
    print(f"MFP @ 50% / 90% kappa accumulation: {mfp50:.0f} / {mfp90:.0f} nm")
    print(f"kappa fraction from MFP < 10 nm: {100 * frac10:.1f}%")
    print(f"acoustic share of kappa: {100 * ac_band:.1f}% (band index < 3); "
          f"{100 * ac_freq:.1f}% (f < 8 THz)")
    print("Umklapp fraction: not stored in kappa-m191919.hdf5 -- "
          "see phph_NU_gruneisen_si.py")

    fig, ax = style.figure(ncols=3, width=3.3, height=2.9)

    # (a) lifetimes + omega^-2 envelope
    ax[0].scatter(fr_v, tau_v, s=5, alpha=0.3, color="C0", lw=0)
    ff = np.linspace(2, 15.5, 50)
    ax[0].plot(ff, 2e2 * ff**-2.0, "k--", lw=0.9, alpha=0.7,
               label=r"$\propto\omega^{-2}$")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("frequency (THz)")
    ax[0].set_ylabel(r"lifetime $\tau$ (ps)")
    ax[0].legend(loc="upper right")
    ax[0].set_title(rf"median $\tau$ = {np.median(tau_v):.1f} ps", fontsize=8)

    # (b) kappa accumulation vs MFP
    ax[1].semilogx(mfp_sorted, kcum, color="C3", lw=1.6)
    ax[1].axvline(mfp50, color="0.3", lw=0.9, ls="--")
    ax[1].annotate(f"50% of $\\kappa$:\n{mfp50:.0f} nm", (mfp50 * 1.4, 0.36),
                   fontsize=7, color="0.2")
    ax[1].axvline(10.0, color="C0", lw=0.9, ls=":")
    ax[1].annotate(f"{100 * frac10:.1f}%\nbelow\n10 nm", (10 * 1.3, 0.08),
                   fontsize=7, color="C0")
    ax[1].axvspan(*WIRE_CORE, color="C1", alpha=0.35, lw=0)
    ax[1].annotate("wire core\ndiameter", (WIRE_CORE[1] * 1.2, 0.62),
                   fontsize=7, color="C1")
    ax[1].set_xlim(MFP_LO, MFP_HI)
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("phonon mean free path (nm)")
    ax[1].set_ylabel(r"cumulative $\kappa/\kappa_\mathrm{tot}$")

    # (c) mode-kappa spectrum, acoustic vs optical.  phono3py mode_kappa
    # carries the q-multiplicity (kappa = sum_IBZ / N_mesh); divide by N_mesh
    # so each dot is that mode's genuine contribution to kappa.
    mk_contrib = mk / weight.sum()
    is_ac = np.tile(np.arange(nb) < 3, (freq.shape[0], 1))
    for sel, color, lab in ((is_ac, "C0", "acoustic (bands 1–3)"),
                            (~is_ac, "C1", "optical")):
        s = valid & sel
        ax[2].scatter(freq[s], mk_contrib[s], s=5, alpha=0.35, color=color,
                      lw=0, label=lab)
    ax[2].set_xlabel("frequency (THz)")
    ax[2].set_ylabel(r"mode $\kappa$ (W m$^{-1}$K$^{-1}$)")
    ax[2].legend(loc="upper right")
    ax[2].annotate(f"acoustic: {100 * ac_band:.0f}% of $\\kappa$",
                   xy=(0.97, 0.72), xycoords="axes fraction", ha="right",
                   fontsize=8, color="C0")

    style.save(fig, "phph_physics_si", directory=FIGDIR)


if __name__ == "__main__":
    main()
