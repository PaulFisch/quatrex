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
    """Ballistic vs anharmonic transmission at L2 and L3, against the analytic
    integer channel count N(omega) of the periodic lead (same lead for both)."""
    from transmission_physicality import channel_count

    fig, ax = style.figure(ncols=2, width=4.4, height=3.4)
    wN = np.load(PROD / "L2_ball.npz", allow_pickle=True)["energies"]
    N, _btop = channel_count(wN)
    for col, L in enumerate(("L2", "L3")):
        wa, Ta = _transmission(PROD / f"{L}_anh.npz")
        wb, Tb = _transmission(PROD / f"{L}_ball.npz")
        a = ax[col]
        a.step(wN, N, where="mid", color="0.65", lw=1.0,
               label=r"channels $N(\omega)$")
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


def fig_ir_plateau():
    """The infrared problem: the ballistic heat-current density omega*I(omega)
    plateaus as omega->0 (the Bose 1/omega cancels hbar*omega -> the quantized
    flux of N_ac acoustic channels), whereas the raw eta=0 anharmonic result
    (NO taper, NO smooth window for cnt33) is suppressed and non-monotonic at low
    omega -- the discrete grid cannot resolve the n(omega')~1/omega' pole in the
    bubble self-energy, so the soft modes are over-weighted and over-scatter."""
    db = np.load(PROD / "L2_ball.npz", allow_pickle=True)
    da = np.load(PROD / "L2_anh.npz", allow_pickle=True)
    w = db["energies"]
    dn = _bose(w, float(db["t_left"])) - _bose(w, float(db["t_right"]))

    def TI(d):
        cs = d["current_spectrum"]
        I = np.sign(np.nanmean(cs[w > 5, 0])) * cs[:, 0]
        with np.errstate(all="ignore"):
            T = np.where(dn > 1e-12, I / dn, np.nan)
        return I, T

    Ib, Tb = TI(db); Ia, Ta = TI(da)
    lowb = (w > 0.2) & (w < 1.6)
    Nac = int(round(float(np.nanmedian(Tb[lowb]))))

    fig, ax = style.figure(ncols=2, width=4.4, height=3.4)
    a = ax[0]
    a.plot(w, Tb, "-o", color="C0", ms=2.5, label="ballistic")
    a.plot(w, Ta, "-o", color="C3", ms=2.5, label=r"anharmonic ($\eta=0$)")
    a.axhline(Nac, color="C0", ls=":", lw=0.8)
    a.annotate(rf"$N_{{\rm ac}}={Nac}$ (ballistic plateau)", (4.5, Nac + 0.3),
               fontsize=7, color="C0")
    a.set_xlim(0, 15); a.set_ylim(0, 13)
    a.set_xlabel("frequency (THz)")
    a.set_ylabel(r"$T(\omega)=I/\Delta n$ (channels)")
    a.legend(fontsize=7, loc="upper left")

    a = ax[1]
    # quantised units: hbar*w*I / (kB*dT) -> the ballistic plateau is
    # EXACTLY N_ac (the quantum of thermal conductance per channel).
    dT = float(db["t_left"]) - float(db["t_right"])
    q = HBAR * (w * 1e12 * 2 * np.pi) / (2 * np.pi) / (KB * dT / (2 * np.pi))
    a.plot(w, q * Ib, "-o", color="C0", ms=3, label="ballistic")
    a.plot(w, q * Ia, "-o", color="C3", ms=3, label=r"anharmonic ($\eta=0$)")
    a.axhline(Nac, color="C0", ls=":", lw=0.8)
    a.annotate(rf"quantised plateau $N_{{\rm ac}}={Nac}$", (2.2, Nac + 0.15),
               fontsize=7, color="C0")
    a.set_xlim(0, 7); a.set_ylim(0, 6.5)
    a.set_xlabel("frequency (THz)")
    a.set_ylabel(r"$\hbar\omega\,I(\omega)\,/\,(k_B\Delta T/2\pi)$")
    a.legend(fontsize=7, loc="lower right")
    style.save(fig, "eta0_cnt33_ir_plateau", directory=FIGDIR)

    print(f"\n[ir plateau] ballistic low-w T = {float(np.nanmedian(Tb[lowb])):.3f} "
          f"(N_ac={Nac});  ballistic w*I low-w (flat?) = "
          f"{np.round((w * Ib)[1:6], 4).tolist()}")
    print(f"  anharmonic low-w T (suppressed/non-monotonic) = "
          f"{np.round(Ta[1:7], 3).tolist()}  vs ballistic {Nac}")


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("=" * 64 + "\nCONVERGENT-cnt33 TRANSPORT PHYSICS\n" + "=" * 64)
    fig_transmission()
    fig_transport()
    fig_ir_plateau()
    print("\nfigures ->", FIGDIR)
