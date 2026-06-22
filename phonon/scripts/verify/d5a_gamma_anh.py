"""d5a Si-H bending Gamma_anh vs grid dw -- the eta=0 feasibility anchor.

The general resolvent-sensitivity analysis: the eta=0 SCBA Jacobian eigenvalue
|lambda| ~ Phi^2 / max(Gamma_anh, c*dw)^2 exceeds 1 (iteration-unstable) wherever
a HEAT-CARRYING mode has anharmonic linewidth Gamma_anh < dw (sub-grid-sharp).
RPM Newton on d5a pure-eta=0 measured |lambda|~28, n_unstable=3 -> the fixed
point is strongly+multiply unstable at nf181 (dw=0.367 THz). This script measures
the PHYSICAL anharmonic linewidth Gamma_anh(omega) directly from the golden-rule
3-phonon vertex (NO SCBA, NO eta, NO grid) to settle: is Gamma_anh(Si-H bending)
sub-grid at every FEASIBLE uniform grid (=> iteration-infeasible, the honest
verdict), or resolvable at a finer grid (=> eta=0 works there)?

Reuses the Si-validated golden-rule machinery in bte_linewidths._bte_machinery
(vertex 3% vs phono3py on Si). d5a fc3 Fourier-interpolates to ANY q (real-space
FC3 + smallest-vector multiplicity), so we sample a fine q-mesh along the wire
axis even though the supercell is only [1,1,4]. On-shell Gamma_lambda(omega_lambda)
is binned by frequency; Si-H character via eigenvector projection onto the 12 H.

Run:  python phonon/scripts/verify/d5a_gamma_anh.py [--nm 24] [--bw 0.4]
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "phonon")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phonon.studies.bte_linewidths import _bte_machinery   # validated vertex

T_KELVIN = 300.0
D5A_DIR = ROOT / "phonon/configs/sinw/fc3_hiphive_sinw100_d5a_sc4_vasp"


def _d5a_setup():
    """d5a harmonic phonons (phonopy) + raw FC3, mirroring bte._cnt_setup."""
    import h5py
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    meta = json.load(open(D5A_DIR / "hiphive_meta.json"))
    prim = meta["primitive"]
    unit = PhonopyAtoms(symbols=prim["symbols"], cell=np.array(prim["cell"]),
                        scaled_positions=np.array(prim["scaled_positions"]))
    scm = meta["supercell"]                      # [1,1,4]
    ph = Phonopy(unit, supercell_matrix=np.diag(scm), primitive_matrix=np.eye(3))
    with h5py.File(D5A_DIR / "fc3.hdf5", "r") as f:
        fc2, fc3 = f["fc2"][...], f["fc3"][...]
    ph.force_constants = fc2
    symbols = list(prim["symbols"])
    return ph, fc2, fc3, symbols


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nm", type=int, default=24, help="q-points along wire axis z")
    ap.add_argument("--bw", type=float, default=0.4, help="golden-rule delta width (THz)")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    NM, BW = a.nm, a.bw

    from phonopy.physical_units import get_physical_units
    ph, fc2, fc3, symbols = _d5a_setup()
    vertex, nb = _bte_machinery(ph, fc3)
    nat_p = nb // 3
    is_H = np.array([s == "H" for s in symbols])             # (nat_p,)
    # DOF-level H mask: atom i -> rows 3i..3i+2
    hdof = np.repeat(is_H, 3)                                 # (nb,)

    qs = np.array([(0.0, 0.0, k / NM) for k in range(NM)])
    FR = np.zeros((NM, nb)); EV = np.zeros((NM, nb, nb), complex)
    for i, q in enumerate(qs):
        fr, ev = ph.get_frequencies_with_eigenvectors(q)
        FR[i] = np.real(fr); EV[i] = ev

    def qidx(q):
        return int(round((q[2] % 1.0) * NM)) % NM
    pu = get_physical_units()
    # same validated golden-rule prefactor as bte_linewidths.run (NM-normalised)
    conv = ((pu.Hbar*pu.EV)**3/36/8*pu.EV**2/pu.Angstrom**6/(2*np.pi*pu.THz)**3
            /pu.AMU**3/NM/pu.EV**2) * (18*np.pi/(pu.Hbar*pu.EV)**2/(2*np.pi*pu.THz)**2*pu.EV**2)

    def nbose(w):
        x = np.where(w > 1e-4, (w*1e12*2*np.pi*1.054571817e-34)/(1.380649e-23*T_KELVIN), 1.0)
        return np.where(w > 1e-4, 1.0/np.expm1(x), 0.0)

    def lor(x):
        return (BW/np.pi)/(x**2+BW**2)
    CUT = 1e-2

    # on-shell Gamma_lambda(omega_lambda) for every mode at every q
    gam = np.full((NM, nb), np.nan)         # THz
    hfrac = np.zeros((NM, nb))              # H-character of each eigenmode
    for iq in range(NM):
        eq = EV[iq]
        # H projection |e_H|^2 / |e|^2 per mode
        w_s = FR[iq]
        ph_imsig = np.zeros(nb)            # 2 w Gamma at on-shell omega=w_s
        for iqp in range(NM):
            iqpp = qidx(-qs[iq]-qs[iqp])
            w1 = FR[iqp]; w2 = FR[iqpp]; n1 = nbose(w1); n2 = nbose(w2)
            P = vertex(eq, EV[iqp], EV[iqpp], qs[iqp], qs[iqpp])   # (nb,nb,nb)
            W = np.maximum(w_s, CUT)[:, None, None]
            g = lor(W - w1[None, :, None] - w2[None, None, :])
            ga = lor(W + w1[None, :, None] - w2[None, None, :])
            gb = lor(W - w1[None, :, None] + w2[None, None, :])
            term = ((n1[None, :, None] + n2[None, None, :] + 1) * g
                    + (n1[None, :, None] - n2[None, None, :]) * (ga - gb))   # (s,b,c)
            bad = (w1[:, None] < CUT) | (w2[None, :] < CUT)
            den = (np.maximum(w_s, CUT)[:, None, None]
                   * np.maximum(w1[None, :, None], CUT)
                   * np.maximum(w2[None, None, :], CUT))
            Pn = np.where(bad[None], 0.0, P/den)                  # (s,b,c)
            ph_imsig += conv*2.0*np.maximum(w_s, CUT)*np.einsum('sbc,sbc->s', Pn, term)
        gam[iq] = np.where(w_s > 0.3, ph_imsig/(2*np.maximum(w_s, CUT)), np.nan)
        ev2 = np.abs(eq)**2
        hfrac[iq] = (ev2[hdof, :].sum(0) / np.maximum(ev2.sum(0), 1e-30))

    # ---- report: Gamma_anh by frequency band, vs grid dw ----
    fmax = 66.0
    grids = {"nf181": fmax/180, "nf361": fmax/360, "nf541": fmax/540, "nf721": fmax/720}
    bands = [("Si acoustic+optical", 0.3, 15.0),
             ("Si-H bending", 15.0, 30.0),
             ("Si-H stretch", 55.0, 66.0)]
    fr = FR.ravel(); gg = gam.ravel(); hh = hfrac.ravel()
    ok = np.isfinite(gg) & (fr > 0.3)
    print(f"\n=== d5a golden-rule Gamma_anh (NM={NM}, BW={BW} THz, T={T_KELVIN}K) ===")
    print(f"{'band':>22} {'N':>4} {'Hchar':>6} {'Gamma_anh THz (med [min,max])':>34}")
    out = {}
    for name, lo, hi in bands:
        m = ok & (fr >= lo) & (fr < hi)
        if not m.any():
            print(f"{name:>22}  (no modes)"); continue
        g = gg[m]; med = np.median(g)
        print(f"{name:>22} {m.sum():>4} {np.mean(hh[m]):>6.2f} "
              f"{med:>12.4g}  [{g.min():.3g}, {g.max():.3g}]")
        out[name] = (med, g.min(), g.max(), float(np.mean(hh[m])))
    print(f"\ngrid dw (fmax={fmax}): " +
          "  ".join(f"{k}={v:.4g}" for k, v in grids.items()))
    print("\n=== feasibility: Gamma_anh / dw  (>=1 grid-resolved -> eta=0 OK; "
          "<1 sub-grid -> |lambda|>1 blow-up) ===")
    for name, lo, hi in bands:
        if name not in out:
            continue
        med = out[name][0]
        ratios = "  ".join(f"{k}:{med/v:.2f}" for k, v in grids.items())
        flag = "RESOLVED" if med/grids["nf181"] >= 1 else "SUB-GRID"
        print(f"{name:>22}  med Gamma/dw = {ratios}   [{flag} @nf181]")
    # the decisive number: finest feasible grid that resolves the Si-H bending
    if "Si-H bending" in out:
        gmed = out["Si-H bending"][0]
        need_dw = gmed
        need_nf = int(np.ceil(fmax/need_dw)) + 1
        print(f"\n>>> Si-H bending median Gamma_anh = {gmed:.4g} THz")
        print(f">>> to grid-resolve it (dw <= Gamma) needs nf >~ {need_nf} "
              f"(dw={need_dw:.4g} THz)")
    np.savez(ROOT / "phonon/scripts/verify/d5a_gamma_anh.npz",
             FR=FR, gam=gam, hfrac=hfrac, qs=qs, nm=NM, bw=BW)
    print("\nwrote phonon/scripts/verify/d5a_gamma_anh.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
