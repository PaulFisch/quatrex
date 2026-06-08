"""Compare the Si-film ballistic conductance for the two FC2 sources
(2x2x2 QE-FD 'si_primitive_work' vs 5x5x5 VASP-hiphive 'si_big_hiphive')
at MATCHED settings, plus their bulk dispersions, to see whether the 5x5x5
ballistic being ~15% lower than 2x2x2/Guo is physical (softer FC2) or a bug.
Ballistic is purely harmonic (FC2), so this isolates the FC2 difference."""
import sys
import numpy as np
_W = "/usr/scratch/mont-fort11/pfischill/quatrex/phonon"
for p in ("/usr/scratch/mont-fort11/pfischill/quatrex", _W):
    if p not in sys.path:
        sys.path.insert(0, p)
from scripts.verify.si_film_kappa import load_bulk_si
from scripts.verify.si_film_ballistic import g_ballistic

FCS = [("2x2x2 QE-FD", "reaps/si_primitive_work"),
       ("5x5x5 VASP-hiphive", "reaps/si_big_hiphive")]

print("=== bulk dispersion (Gamma-optical + acoustic group velocity) ===")
disp = {}
for name, sub in FCS:
    ph, _ = load_bulk_si(fc3_subdir=sub)
    # Gamma optical + max frequency
    ph.run_mesh([13, 13, 13], with_eigenvectors=False)
    m = ph.get_mesh_dict()
    fmax = float(np.nanmax(m["frequencies"]))
    # Gamma + small-q frequencies via run_qpoints
    dq = 0.03
    ph.run_qpoints([[0, 0, 0], [dq, 0, 0], [0, dq, 0]], with_eigenvectors=False)
    qd = ph.get_qpoints_dict()
    g0 = np.array(qd["frequencies"][0])         # Gamma
    gx = np.array(qd["frequencies"][1])         # small q along x
    vg = np.sort(np.abs(gx - g0))[:3] / dq      # 3 lowest acoustic slopes (THz/frac-q)
    print(f"  {name:20s}: f_max={fmax:6.2f} THz  Gamma-optical={np.sort(g0)[-1]:6.2f} THz  "
          f"acoustic slopes(3 lowest)={np.round(vg,1)}")
    disp[name] = dict(fmax=fmax, gopt=float(np.sort(g0)[-1]))

print("\n=== film ballistic G_ball (MW/m2K), nk=8, eta_factor=0.1, nfreq=121 ===")
print(f"{'n_slabs':>7} {'L(nm)':>6} {'2x2x2':>9} {'5x5x5':>9} {'ratio 5/2':>9}")
for ns in [3, 5, 8]:
    g = {}
    for name, sub in FCS:
        ph, _ = load_bulk_si(fc3_subdir=sub)
        L = ns * float(np.linalg.norm(ph.primitive.cell[0])) * 1e-10
        G, mx = g_ballistic(ph, ns, 8, 121, 0.1, "x")
        g[name] = G / 1e6
    print(f"{ns:7d} {L*1e9:6.2f} {g['2x2x2 QE-FD']:9.1f} {g['5x5x5 VASP-hiphive']:9.1f} "
          f"{g['5x5x5 VASP-hiphive']/g['2x2x2 QE-FD']:9.3f}")
