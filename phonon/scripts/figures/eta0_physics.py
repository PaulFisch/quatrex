"""Transport physics from the CONVERGENT cnt33 η=0 calculations (companion to
eta0_convergence.py). Only converged runs are used (L4_anh did NOT converge and
is excluded). Panel titles are omitted on purpose -- the LaTeX caption supplies
them; axis labels and legends are kept. All numbers are printed for the text.

  Transmission:  ballistic vs anharmonic T(omega)=I(omega)/Delta n(omega) at two
                 device lengths (L2, L3) -- the three-phonon suppression and its
                 growth with length.
  Transport:     anharmonic/ballistic conductance ratio vs temperature (L2) and
                 vs device length (L2, L3).

Sources (verified converged, eta=1e-12, retarded=fft, lead-conserving):
  phonon/scripts/out/prod/cnt33_eta0/{summary.json, L2_anh, L2_ball, L3_anh,
  L3_ball, T30..T300_anh}.npz   (181-pt / 0-55 THz grid).

Run:  OMP_NUM_THREADS=1 python phonon/scripts/figures/eta0_physics.py
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

PROD = ROOT / "phonon/scripts/out/prod/cnt33_eta0"
FIGDIR = ROOT / "document/fig/transport_sweeps"
HBAR, KB = 1.054571817e-34, 1.380649e-23
WMIN = 2.0   # exclude the sub-2 THz IR-regularised acoustic bins


def _bose(w, T):
    w = np.asarray(w, float)
    x = np.where(w > 1e-6, (w * 1e12 * 2 * np.pi * HBAR) / (KB * T), 1.0)
    return np.where(w > 1e-6, 1.0 / np.expm1(x), 0.0)


def _transmission(npz):
    """Effective transmission T(omega)=I(omega)/Delta n(omega) in units of
    conducting channels (the ballistic curve recovers the ~11 cnt33 channels);
    hot-lead interface, sign so a forward flow is positive, IR bins masked."""
    d = np.load(npz, allow_pickle=True)
    w = d["energies"]; cs = d["current_spectrum"]
    tL, tR = float(d["t_left"]), float(d["t_right"])
    dn = _bose(w, tL) - _bose(w, tR)
    I = np.sign(np.nanmean(cs[w > 5, 0])) * cs[:, 0]
    band = (w >= WMIN) & (dn > 1e-9)
    T = np.where(band, I / dn, np.nan)
    return w, T


def fig_transmission():
    """Ballistic vs anharmonic transmission at L2 and L3."""
    fig, ax = style.figure(ncols=2, width=4.4, height=3.4)
    for col, L in enumerate(("L2", "L3")):
        wa, Ta = _transmission(PROD / f"{L}_anh.npz")
        wb, Tb = _transmission(PROD / f"{L}_ball.npz")
        a = ax[col]
        a.plot(wb, Tb, "-", color="C0", lw=1.4, label="ballistic")
        a.plot(wa, Ta, "-", color="C3", lw=1.4, label="anharmonic")
        a.set_xlabel("frequency (THz)")
        a.set_ylabel(r"transmission $T(\omega)$ (channels)")
        a.set_ylim(0, None)
        a.legend(fontsize=7, loc="upper right")
        a.annotate(f"{int(L[1])} cells", (0.5, 0.92), xycoords="axes fraction",
                   fontsize=8, ha="center")
        red = np.nansum(Tb - Ta) / np.nansum(Tb) * 100
        print(f"[transmission {L}] band-integrated suppression "
              f"1 - <T_anh>/<T_ball> = {red:.1f}%  (T_ball peak={np.nanmax(Tb):.1f} channels)")
    style.save(fig, "eta0_cnt33_transmission", directory=FIGDIR)


def fig_transport():
    """Anharmonic/ballistic conductance ratio vs temperature and vs length."""
    rows = json.load(open(PROD / "summary.json"))
    tmp = sorted([(r["t_mean"], r["ratio"]) for r in rows
                  if r.get("sweep") == "temperature" and r.get("anh_converged")])
    lp = {r["tag"]: r for r in rows if r.get("sweep") == "length"
          and r.get("anh_converged")}
    Ts = [t for t, _ in tmp]; Rs = [r for _, r in tmp]
    Ls = sorted(int(k[1]) for k in lp)
    Rl = [lp[f"L{n}"]["ratio"] for n in Ls]

    fig, ax = style.figure(ncols=2, width=4.4, height=3.4)
    a = ax[0]
    a.plot(Ts, Rs, "o-", color="C3", ms=6)
    a.set_xlabel("temperature (K)")
    a.set_ylabel(r"$G_\mathrm{anh}/G_\mathrm{ball}$")
    a.set_ylim(0.5, 0.9)
    a = ax[1]
    a.plot(Ls, Rl, "s-", color="C3", ms=8)
    for n, r in zip(Ls, Rl):
        a.annotate(f"{r:.2f}", (n, r), textcoords="offset points",
                   xytext=(6, 6), fontsize=8)
    a.set_xlabel("device length (cells)")
    a.set_ylabel(r"$G_\mathrm{anh}/G_\mathrm{ball}$ (300 K)")
    a.set_xticks(Ls); a.set_ylim(0.45, 0.65)
    style.save(fig, "eta0_cnt33_transport", directory=FIGDIR)

    print("[transport] G_anh/G_ball vs T:", [(t, round(r, 3)) for t, r in tmp])
    print("  vs length (300 K, converged):",
          {f"L{n}": round(lp[f'L{n}']['ratio'], 3) for n in Ls})
    # ballistic L-independence check (Landauer baseline)
    gb = {r["tag"]: r["G_ball_W_per_m2_K"] for r in rows if r.get("sweep") == "length"}
    if gb:
        print("  G_ball vs length (should be ~flat):",
              {k: f"{gb[k]:.3e}" for k in sorted(gb)})


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("=" * 64 + "\nCONVERGENT-cnt33 TRANSPORT PHYSICS\n" + "=" * 64)
    fig_transmission()
    fig_transport()
    print("\nfigures ->", FIGDIR)
