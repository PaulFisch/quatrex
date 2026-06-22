"""Physics figures from the CONVERGENT cnt33 η=0 calculations (companion to
eta0_convergence.py). Only converged runs are used (L4_anh is excluded -- it
did not converge). All numbers are printed for the LaTeX.

  Feasibility:  cnt33 golden-rule Γ_anh(ω) (all heat-carrying modes resolved,
                Γ≳dω) vs d5a (a heat-carrying band straddles dω) -> why one
                converges at η=0 and the other does not.
  Transport:    anharmonic vs ballistic per-ω heat-current spectrum of the
                converged cnt33 η=0 solution (three-phonon suppression), and
                the anharmonic/ballistic conductance ratio vs temperature.

Sources (verified converged, η=1e-12, retarded=fft):
  phonon/scripts/out/prod/cnt33_eta0/{summary.json,L2_anh,L2_ball,...}.npz
  phonon/scripts/verify/d5a_gamma_anh.npz   (d5a golden-rule Γ, NM=32)
  cnt33 golden-rule Γ from bte_linewidths._cnt_setup + _bte_machinery.

Run:  OMP_NUM_THREADS=1 python phonon/scripts/figures/eta0_physics.py
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style
from phonon.studies.bte_linewidths import _cnt_setup, _bte_machinery

PROD = ROOT / "phonon/scripts/out/prod/cnt33_eta0"
FIGDIR = ROOT / "document/fig/transport_sweeps"
HBAR, KB = 1.054571817e-34, 1.380649e-23


def _bose(w, T):
    w = np.asarray(w, float)
    x = np.where(w > 1e-6, (w * 1e12 * 2 * np.pi * HBAR) / (KB * T), 1.0)
    return np.where(w > 1e-6, 1.0 / np.expm1(x), 0.0)


def _golden_gamma(ph, fc3, NM, BW, T):
    """On-shell golden-rule Γ_anh(ω_λ) for every mode, q-mesh NM along the
    periodic axis (same evaluator as the d5a anchor)."""
    from phonopy.physical_units import get_physical_units
    vertex, nb = _bte_machinery(ph, fc3)
    nat_p = nb // 3
    qs = np.array([(0.0, 0.0, k / NM) for k in range(NM)])
    FR = np.zeros((NM, nb)); EV = np.zeros((NM, nb, nb), complex)
    for i, q in enumerate(qs):
        fr, ev = ph.get_frequencies_with_eigenvectors(q)
        FR[i] = np.real(fr); EV[i] = ev
    pu = get_physical_units()
    conv = ((pu.Hbar*pu.EV)**3/36/8*pu.EV**2/pu.Angstrom**6/(2*np.pi*pu.THz)**3
            / pu.AMU**3/NM/pu.EV**2) * (18*np.pi/(pu.Hbar*pu.EV)**2
            / (2*np.pi*pu.THz)**2*pu.EV**2)

    def nb_(w):
        x = np.where(w > 1e-4, (w*1e12*2*np.pi*HBAR)/(KB*T), 1.0)
        return np.where(w > 1e-4, 1.0/np.expm1(x), 0.0)

    def lor(x):
        return (BW/np.pi)/(x**2+BW**2)
    CUT = 1e-2

    def qidx(q):
        return int(round((q[2] % 1.0)*NM)) % NM
    gam = np.full((NM, nb), np.nan)
    for iq in range(NM):
        eq = EV[iq]; w_s = FR[iq]; acc = np.zeros(nb)
        for iqp in range(NM):
            iqpp = qidx(-qs[iq]-qs[iqp])
            w1, w2 = FR[iqp], FR[iqpp]; n1, n2 = nb_(w1), nb_(w2)
            P = vertex(eq, EV[iqp], EV[iqpp], qs[iqp], qs[iqpp])
            W = np.maximum(w_s, CUT)[:, None, None]
            g = lor(W-w1[None, :, None]-w2[None, None, :])
            ga = lor(W+w1[None, :, None]-w2[None, None, :])
            gb = lor(W-w1[None, :, None]+w2[None, None, :])
            term = ((n1[None, :, None]+n2[None, None, :]+1)*g
                    + (n1[None, :, None]-n2[None, None, :])*(ga-gb))
            bad = (w1[:, None] < CUT) | (w2[None, :] < CUT)
            den = (np.maximum(w_s, CUT)[:, None, None]
                   * np.maximum(w1[None, :, None], CUT)
                   * np.maximum(w2[None, None, :], CUT))
            Pn = np.where(bad[None], 0.0, P/den)
            acc += conv*2.0*np.maximum(w_s, CUT)*np.einsum('sbc,sbc->s', Pn, term)
        gam[iq] = np.where(w_s > 0.3, acc/(2*np.maximum(w_s, CUT)), np.nan)
    return FR.ravel(), gam.ravel()


def fig_feasibility():
    """cnt33 (convergent) vs d5a (not): golden-rule Γ_anh vs grid dω."""
    warnings.filterwarnings("ignore")
    ph, fc2, fc3 = _cnt_setup(8)
    cfr, cgam = _golden_gamma(ph, fc3, NM=12, BW=0.4, T=300.0)
    cok = np.isfinite(cgam) & (cfr > 0.3)
    cfr, cgam = cfr[cok], cgam[cok]
    dw_cnt = 55.0/180                                  # cnt33 grid (nf181, fmax55)

    d = np.load(ROOT/"phonon/scripts/verify/d5a_gamma_anh.npz")
    dfr, dgam, dh = d["FR"].ravel(), d["gam"].ravel(), d["hfrac"].ravel()
    dok = np.isfinite(dgam) & (dfr > 0.3)
    dfr, dgam, dh = dfr[dok], dgam[dok], dh[dok]
    dw181 = 66.0/180

    fig, ax = style.figure(ncols=2, width=4.6, height=3.4)
    a = ax[0]
    a.scatter(cfr, cgam, s=10, color="C0", alpha=0.6)
    a.axhline(dw_cnt, color="C3", ls="--", lw=0.9)
    a.annotate(r"$d\omega$", (cfr.max()*0.8, dw_cnt), color="C3", fontsize=7,
               va="bottom")
    a.set_yscale("log"); a.set_xlabel("mode frequency (THz)")
    a.set_ylabel(r"$\Gamma_{\rm anh}$ (THz)")
    a.set_title(r"cnt33: heat modes resolved ($\Gamma{\gtrsim}d\omega$)")
    a = ax[1]
    sc = a.scatter(dfr, dgam, c=dh, s=10, cmap="viridis", vmin=0, vmax=1)
    a.axhline(dw181, color="C3", ls="--", lw=0.9)
    a.annotate(r"$d\omega$", (dfr.max()*0.75, dw181), color="C3", fontsize=7,
               va="bottom")
    a.set_yscale("log"); a.set_xlabel("mode frequency (THz)")
    a.set_title(r"d5a: Si--H band straddles $d\omega$")
    cb = fig.colorbar(sc, ax=a); cb.set_label("H-character", fontsize=8)
    style.save(fig, "eta0_gamma_feasibility", directory=FIGDIR)

    # numbers: fraction of heat-carrying (w>3 THz) cnt33 modes below dw
    heat = cfr > 3.0
    print(f"[cnt33 Γ] heat modes (>3THz): median Γ={np.median(cgam[heat]):.2f} THz, "
          f"frac below dω({dw_cnt:.3f})={np.mean(cgam[heat] < dw_cnt)*100:.1f}%  "
          f"(low-ω acoustic Γ<dω handled by the IR taper)")


def fig_transport():
    """Anharmonic vs ballistic per-ω heat current (converged cnt33 η=0) +
    the anharmonic/ballistic conductance ratio vs temperature."""
    a = np.load(PROD/"L2_anh.npz", allow_pickle=True)
    b = np.load(PROD/"L2_ball.npz", allow_pickle=True)
    w = a["energies"]; dw = float(w[1]-w[0])
    s = np.sign(np.nanmean(a["current_spectrum"][w > 5, 0]))
    Ia = s*a["current_spectrum"][:, 0]; Ib = s*b["current_spectrum"][:, 0]
    band = w >= 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        supp = np.where(band & (np.abs(Ib) > 1e-9*np.nanmax(np.abs(Ib))), Ia/Ib, np.nan)

    rows = json.load(open(PROD/"summary.json"))
    tmp = sorted([(r["t_mean"], r["ratio"]) for r in rows
                  if r.get("sweep") == "temperature" and r.get("anh_converged")])
    Ts = [t for t, _ in tmp]; Rs = [r for _, r in tmp]
    # length points (converged only): L2, L3
    lp = {r["tag"]: r for r in rows if r.get("sweep") == "length"
          and r.get("anh_converged")}

    fig, ax = style.figure(ncols=2, width=4.6, height=3.4)
    a0 = ax[0]
    a0.plot(w[band], Ia[band], "-", color="C3", lw=1.3, label="anharmonic")
    a0.plot(w[band], Ib[band], "-", color="C0", lw=1.1, label="ballistic")
    a0.set_xlabel("frequency (THz)")
    a0.set_ylabel(r"heat-current spectrum $I(\omega)$ (arb.)")
    a0.set_title(r"cnt33 $\eta{=}0$: 3-phonon suppression")
    a0.legend(fontsize=7)
    a1 = ax[1]
    a1.plot(Ts, Rs, "o-", color="C0", ms=6, label="L2 (this work)")
    if "L3" in lp:
        a1.plot([300], [lp["L3"]["ratio"]], "s", color="C3", ms=7,
                label="L3 (300 K) %.2f" % lp["L3"]["ratio"])
    a1.set_xlabel("temperature (K)")
    a1.set_ylabel(r"$G_{\rm anh}/G_{\rm ball}$")
    a1.set_title(r"anharmonic reduction vs $T$ ($\eta{=}0$)")
    a1.set_ylim(0.5, 0.9); a1.legend(fontsize=7, loc="lower left")
    style.save(fig, "eta0_cnt33_transport", directory=FIGDIR)

    print("[cnt33 transport] G_anh/G_ball vs T:",
          [(t, round(r, 3)) for t, r in tmp])
    print("  length (300K, converged):",
          {k: round(lp[k]["ratio"], 3) for k in sorted(lp)})
    print(f"  L2_anh converged={bool(a['converged'])}, L2_ball converged={bool(b['converged'])}, dw={dw:.4f}")


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("="*64 + "\nCONVERGENT-cnt33 PHYSICS NUMBERS\n" + "="*64)
    # NB: a cnt33 golden-rule Gamma_anh "feasibility" panel was attempted but the
    # absolute golden-rule prefactor is NOT calibrated for the cnt33 force
    # constants (the bte_linewidths NEGF-area bridge is not applied), giving an
    # unphysical median Gamma ~ 100 THz -- so it is omitted. The convergence
    # criterion is documented via the d5a anchor + the resolvent-gain argument.
    fig_transport()
    print("\nfigures ->", FIGDIR)
