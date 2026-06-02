"""Converged BALLISTIC cross-plane kappa(thickness) for the Si thin film.

The ballistic (Landauer) conductance per area G_ball is set by the leads and is
length-INDEPENDENT; kappa_ball(L) = G_ball * L therefore rises ~linearly with L (the
ballistic regime, before anharmonic scattering makes kappa intensive). This needs (a) a
converged transverse q_perp mesh -- the q-averaged transmission must capture all transverse
channels -- and (b) a small broadening eta so the coherent transmission is not attenuated
over the device (the F17 finite-eta artifact). No SCBA here (nk^2 cost), so we can push nk.

Usage:
  python si_film_ballistic.py --mode qconv --n-slabs 8 --nk 4 8 12 16 --eta 0.1
  python si_film_ballistic.py --mode lconv --nk 16 --eta 0.1 --n-slabs 2 4 8 16 28 40
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_W = Path("/usr/scratch/mont-fort11/pfischill/quatrex/phonon")
for p in (_W.parent, _W):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phonon_inputs.convention import get_btd_blocks
from solver.grids import build_frequency_grid, bose_full_axis
from solver.leads import (build_device_hamiltonian, compute_obc_batch,
                          ballistic_transmission_z2)
from phonon_inputs.constants import CONVERSION_THZ2, HBAR_SI, THZ_TO_RAD

sys.argv0 = None
from scripts.verify.si_film_kappa import load_bulk_si, layer_spacing_m  # reuse loader


def g_ballistic(phonon, n_slabs, nk, nfreq, eta_factor, tdir, T=300.0, dT=10.0,
                fmax=15.0):
    n_atoms = len(phonon.primitive.masses); n_dof = 3 * n_atoms
    N_D = n_slabs * n_dof
    freqs, dw, eta_w, z2_arr, pos_mask, mid = build_frequency_grid(
        (0.0, fmax, nfreq), eta_factor=eta_factor)
    q_pts = [(i / nk, j / nk) for i in range(nk) for j in range(nk)]
    nkpts = len(q_pts)
    T_L, T_R = T + dT / 2, T - dT / 2
    trans = np.zeros(len(freqs))
    for (qx, qy) in q_pts:
        H_00, H_01 = get_btd_blocks(phonon, (qx, qy), transport_direction=tdir,
                                    conversion_factor=CONVERSION_THZ2)
        H_D = build_device_hamiltonian(H_00, H_01, n_slabs)
        compute_obc_batch(z2_arr, H_00, H_01, freqs, T_L, T_R, n_slabs=n_slabs)
        H_LD = np.zeros((n_dof, N_D), dtype=complex); H_LD[:, :n_dof] = H_01
        H_DR = np.zeros((N_D, n_dof), dtype=complex); H_DR[-n_dof:, :] = H_01
        for iw, z2 in enumerate(z2_arr):
            trans[iw] += ballistic_transmission_z2(z2, H_D, H_00, H_01, H_LD, H_DR)
    trans /= nkpts
    lattice = phonon.primitive.cell; tidx = "xyz".index(tdir)
    perp = [i for i in range(3) if i != tidx]
    A_c = np.linalg.norm(np.cross(lattice[perp[0]], lattice[perp[1]])) * 1e-20
    omega = freqs * THZ_TO_RAD
    nb = bose_full_axis(freqs, T_L) - bose_full_axis(freqs, T_R)
    J = np.sum((HBAR_SI * omega * nb * trans)[pos_mask]) * dw * 1e12
    G = J / (A_c * dT)
    return G, float(trans.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["qconv", "lconv"], default="qconv")
    ap.add_argument("--nk", type=int, nargs="+", default=[8])
    ap.add_argument("--n-slabs", type=int, nargs="+", default=[8])
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--nfreq", type=int, default=121)
    ap.add_argument("--transport-dir", default="x")
    ap.add_argument("--out", default=str(_W / "scripts/out/si_film/si_film_ballistic.json"))
    args = ap.parse_args()
    phonon, _ = load_bulk_si()
    d = layer_spacing_m(phonon, args.transport_dir)
    print(f"layer spacing = {d*1e10:.4f} Ang", flush=True)
    rows = []
    if args.mode == "qconv":
        ns = args.n_slabs[0]
        for nk in args.nk:
            G, mx = g_ballistic(phonon, ns, nk, args.nfreq, args.eta, args.transport_dir)
            L = ns * d
            rows.append(dict(n_slabs=ns, nk=nk, eta=args.eta, G_ball=G, maxT=mx,
                             kappa_ball=G * L, L_nm=L * 1e9))
            print(f"  nk={nk:2d} (slabs={ns}): maxT_avg={mx:.3f}  G_ball={G:.3e}  "
                  f"kappa_ball={G*L:.2f} W/mK", flush=True)
    else:
        nk = args.nk[0]
        for ns in args.n_slabs:
            G, mx = g_ballistic(phonon, ns, nk, args.nfreq, args.eta, args.transport_dir)
            L = ns * d
            rows.append(dict(n_slabs=ns, nk=nk, eta=args.eta, G_ball=G, maxT=mx,
                             kappa_ball=G * L, L_nm=L * 1e9))
            print(f"  slabs={ns:2d} ({L*1e9:.2f} nm, nk={nk}): maxT_avg={mx:.3f}  "
                  f"G_ball={G:.3e}  kappa_ball={G*L:.2f} W/mK", flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(rows=rows, d_layer_ang=d*1e10, args=vars(args)), open(out, "w"), indent=2)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
