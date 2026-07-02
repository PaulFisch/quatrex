"""Physicality audit of the eta=0 phonon transmission curve (cnt33 L2).

Checks the production MW transmission T(omega)=|I|/Delta n against ANALYTIC limits:
  (1) ballistic T_ball(omega) == integer Landauer channel count N(omega), where
      N(omega) = # bands crossing omega in the lead dispersion D(k)=H0+H1 e^{ik}+h.c.;
  (2) acoustic plateau T_ball(omega->0) -> N_ac = 4 (armchair CNT: 2 flexural + LA + twist);
  (3) unitarity bound 0 <= T_ball <= N(omega) (the unbounded I/Delta n must not overshoot);
  (4) anharmonic bound T_anh(omega) <= T_ball(omega) (scattering only removes);
  (5) quantised-conductance plateau hbar*omega*I(omega) -> N_ac * kB*dT/(2pi) as omega->0.

Reads ONLY saved data:
  cnt33 eta=0:  phonon/scripts/out/prod/cnt33_eta0/L2_{ball,anh}.npz  (current_spectrum)
  dispersion :  phonon/scripts/out/prod/geom/cnt33_L2/dynamical_matrix.mat  (H0,H1 blocks)

Run:  OMP_NUM_THREADS=1 python phonon/scripts/figures/transmission_physicality.py
Figure -> document/fig/transport_sweeps/eta0_cnt33_transmission_physicality.{pdf,png}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style

PROD = ROOT / "phonon/scripts/out/prod/cnt33_eta0"
GEOM = ROOT / "phonon/scripts/out/prod/geom/cnt33_L2"
# Supplementary audit figure -- its content is folded into the in-document
# figures (N(omega) staircase in eta0_cnt33_transmission, quantised plateau in
# eta0_cnt33_ir_plateau), so it lands in the attic, not transport_sweeps.
FIGDIR = ROOT / "document/fig/attic"
HBAR = 1.054571817e-34
KB = 1.380649e-23


def _bose(w_thz, T):
    w = np.asarray(w_thz, float)
    x = np.where(w > 1e-9, w * 1e12 * 2 * np.pi * HBAR / (KB * T), 1.0)
    return np.where(w > 1e-9, 1.0 / np.expm1(x), 0.0)


def _t_eff(npz, dn):
    cs = np.asarray(npz["current_spectrum"])
    cur = 0.5 * (np.abs(cs[:, 0]) + np.abs(cs[:, -1]))   # avg of both leads
    return cur, np.where(dn > 1e-9, cur / dn, np.nan)


def channel_count(en, nk=4000):
    """Analytic N(omega) = sum_b #crossings of omega by band b for k in (0,pi]."""
    M = sio.loadmat(str(GEOM / "dynamical_matrix.mat"))
    H0 = np.asarray(M["[0, 0, 0]"]); H1 = np.asarray(M["[0, 0, 1]"])
    Hm = np.asarray(M["[0, 0, -1]"])
    ks = np.linspace(1e-4, np.pi, nk)
    bands = np.empty((nk, H0.shape[0]))
    for i, k in enumerate(ks):
        Dk = H0 + H1 * np.exp(1j * k) + Hm * np.exp(-1j * k)
        Dk = 0.5 * (Dk + Dk.conj().T)
        w2 = np.linalg.eigvalsh(Dk)
        bands[i] = np.sign(w2) * np.sqrt(np.abs(w2))
    N = np.array([sum(np.sum(np.diff(np.sign(bands[:, b] - w)) != 0)
                      for b in range(bands.shape[1])) for w in en])
    return N, float(bands.max())


def main():
    db = np.load(PROD / "L2_ball.npz", allow_pickle=True)
    da = np.load(PROD / "L2_anh.npz", allow_pickle=True)
    en = db["energies"]
    tL, tR = float(db["t_left"]), float(db["t_right"])
    dT = tL - tR
    dn = _bose(en, tL) - _bose(en, tR)
    Ib, Tb = _t_eff(db, dn)
    Ia, Ta = _t_eff(da, dn)
    N, btop = channel_count(en)

    m = en > 0
    over_uni = int(np.sum(m & ~np.isnan(Tb) & (Tb > N + 0.5)))
    over_anh = int(np.sum(m & ~np.isnan(Ta) & ~np.isnan(Tb) & (Ta > Tb + 0.05)))
    # max relative deviation T_ball vs integer N over the supported band
    supp = m & (en < btop) & ~np.isnan(Tb) & (N > 0)
    dev = np.abs(Tb[supp] - N[supp])
    j = HBAR * (en * 1e12 * 2 * np.pi) / (2 * np.pi) * Ib   # hbar*w*I  [W/bin]
    plateau = j[1:6] / (KB * dT / (2 * np.pi))              # -> N_ac

    print("=" * 64)
    print("ETA=0 TRANSMISSION PHYSICALITY (cnt33 L2)")
    print("=" * 64)
    print(f"band-top {btop:.2f} THz, 36 branches; N_ac plateau {N[1]:d}")
    print(f"T_ball(first bin {en[1]:.2f} THz) = {Tb[1]:.4f}  (analytic N_ac=4)")
    print(f"max |T_ball - N(w)| over band = {dev.max():.4f}  (T_ball == integer N)")
    print(f"max T_ball = {np.nanmax(Tb):.3f} = max N(w) = {N[m].max():d}")
    print(f"# bins T_ball > N(w)+0.5 (unitarity breach) = {over_uni}")
    print(f"# bins T_anh > T_ball     (gain, unphysical) = {over_anh}")
    print(f"hbar*w*I plateau / (kB dT/2pi) -> {np.round(plateau,3)}  (-> N_ac=4)")
    print(f"G_anh/G_ball (lead) = "
          f"{float(da['lead_current'])/float(db['lead_current']):.3f}")

    fig, axes = style.figure(ncols=2, width=4.6, height=3.4)
    ax = axes[0]
    ax.step(en, N, where="mid", color="0.6", lw=1.4, label=r"channels $N(\omega)$")
    ax.plot(en, Tb, "-", color="C0", lw=1.4, label=r"$T_{\rm ball}$ ($\eta=0$)")
    ax.plot(en, Ta, "-", color="C3", lw=1.3, label=r"$T_{\rm anh}$ ($\eta=0$)")
    ax.axhline(4, color="C2", ls=":", lw=0.8)
    ax.annotate(r"$N_{\rm ac}=4$", (1, 4.15), fontsize=7, color="C2")
    ax.set_xlabel("frequency (THz)"); ax.set_ylabel("transmission (channels)")
    ax.legend(fontsize=7, loc="upper left"); ax.set_xlim(0, btop + 2)

    ax = axes[1]
    ax.plot(en[1:], j[1:] / (KB * dT / (2 * np.pi)), "-", color="C0", lw=1.4)
    ax.axhline(4, color="C2", ls=":", lw=0.8)
    ax.annotate(r"$N_{\rm ac}=4$", (5, 4.1), fontsize=7, color="C2")
    ax.set_xlabel("frequency (THz)")
    ax.set_ylabel(r"$\hbar\omega\,I(\omega)\,/\,(k_B\Delta T/2\pi)$")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.set_title("quantised heat-current plateau", fontsize=8)
    style.save(fig, "eta0_cnt33_transmission_physicality", directory=FIGDIR)
    print("figure ->", FIGDIR / "eta0_cnt33_transmission_physicality.pdf")


if __name__ == "__main__":
    main()
