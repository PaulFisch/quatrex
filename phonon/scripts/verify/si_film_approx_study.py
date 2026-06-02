"""Full (unapproximated) self-energy vs Guo's approximations, in the dense q-resolved code.

The q-resolved transport driver (transmission_q) computes only the slab-DIAGONAL self-energy
(Guo approximation III). This study uses the full off-diagonal multi-slab kernel
(se_q.compute_phph_self_energy_q_dense_multi_slab) to quantify what each approximation costs
vs the unapproximated self-energy, on a transversely-periodic structure.

Method (a single first-Born step from the ballistic device G, NOT a full SCBA -- the relative
ranking of the approximations is what matters): build the ballistic device Green's functions
G_{K,K'}(q_perp,w), form Sigma^{<,>,R} under each approximation, do one Dyson solve, and read the
cross-plane heat-current conductance G = J/(A_perp dT) [same formula as scba_loop]. The corrected
(default) prefactor is used throughout.

  full         : all off-diagonal slab blocks, full FD FC3, full G range
  diag-only    : sigma_cutoff=0  (Guo approx III: only N_x x N_x diagonal blocks)
  1st-NN FC3   : vertex_cutoff=1 (Guo approx II)
  g-range=1    : g_cutoff=1      (drop G beyond nearest slab)
"""
import argparse
import sys
import warnings
from pathlib import Path

_W = Path("/usr/scratch/mont-fort11/pfischill/quatrex/phonon")
for p in (_W.parent, _W):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

import numpy as np
import h5py
from phonon.phonon_inputs.convention import get_btd_blocks
from phonon.phonon_inputs.separable import (
    build_supercell_mapping, build_realspace_fc3_matrices, build_q_diff_map)
from phonon.phonon_inputs.constants import CONVERSION_THZ2, HBAR_SI, THZ_TO_RAD
from phonon.solver.grids import build_frequency_grid, bose_full_axis
from phonon.solver.leads import (build_device_hamiltonian, compute_obc_batch,
                                  ballistic_transmission_z2, solve_green_batch)
from phonon.solver.retarded import build_retarded
from phonon.solver.se_q import compute_phph_self_energy_q_dense_multi_slab
from phonon.scripts.verify.si_film_kappa import load_bulk_si, layer_spacing_m


def heat_conductance(phonon, M_stacked, prim_indices, cell_frac, slab_indices,
                     n_atoms, n_slabs, nk, nfreq, eta_factor, tdir, approx,
                     T=300.0, dT=10.0, fmax=15.0):
    nd = 3 * n_atoms
    N_D = n_slabs * nd
    freqs, dw, eta_w, z2_arr, pos_mask, mid = build_frequency_grid(
        (0.0, fmax, nfreq), eta_factor=eta_factor)
    nfreq = len(freqs)                      # build_frequency_grid returns a symmetric grid
    q_pts = [(i / nk, j / nk) for i in range(nk) for j in range(nk)]
    nq = len(q_pts)
    q_diff = build_q_diff_map(nk, nk)
    T_L, T_R = T + dT / 2, T - dT / 2
    omega_rad = freqs * THZ_TO_RAD

    btd = [get_btd_blocks(phonon, q, transport_direction=tdir,
                          conversion_factor=CONVERSION_THZ2) for q in q_pts]
    H_D = [build_device_hamiltonian(h00, h01, n_slabs) for (h00, h01) in btd]
    obc = [compute_obc_batch(z2_arr, h00, h01, freqs, T_L, T_R, n_slabs=n_slabs)
           for (h00, h01) in btd]

    # 1) ballistic device G blocks G_{K,K'}(q,w) (Sigma_scatt = 0)
    zeroSig = np.zeros((nfreq, N_D, N_D), dtype=complex)
    gl_blk = {}; gg_blk = {}
    for iq in range(nq):
        _gr, gl, gg = solve_green_batch(z2_arr, H_D[iq], obc[iq], zeroSig, zeroSig, zeroSig)
        for K in range(n_slabs):
            for Kp in range(n_slabs):
                sK = slice(K * nd, (K + 1) * nd); sKp = slice(Kp * nd, (Kp + 1) * nd)
                gl_blk.setdefault((K, Kp), np.zeros((nq, nfreq, nd, nd), dtype=complex))
                gg_blk.setdefault((K, Kp), np.zeros((nq, nfreq, nd, nd), dtype=complex))
                gl_blk[(K, Kp)][iq] = gl[:, sK, sKp]
                gg_blk[(K, Kp)][iq] = gg[:, sK, sKp]

    # 2) self-energy under the chosen approximation
    kw = dict(sigma_cutoff=None, g_cutoff=None, vertex_cutoff=None)
    if approx == "diag-only":
        kw["sigma_cutoff"] = 0
    elif approx == "1st-NN-FC3":
        kw["vertex_cutoff"] = 1
    elif approx == "g-range1":
        kw["g_cutoff"] = 1
    sl_d, sg_d = compute_phph_self_energy_q_dense_multi_slab(
        gl_blk, gg_blk, M_stacked, prim_indices, cell_frac, slab_indices,
        n_atoms, n_slabs, nq, q_pts, q_diff, freqs, dw, tdir,
        dc_handling="interpolate", **kw)

    # 3) one Dyson solve with Sigma^{<,>,R}; 4) cross-plane heat current
    lattice = phonon.primitive.cell; tidx = "xyz".index(tdir)
    perp = [i for i in range(3) if i != tidx]
    A_c = np.linalg.norm(np.cross(lattice[perp[0]], lattice[perp[1]])) * 1e-20
    sl0 = slice(0, nd); slast = slice((n_slabs - 1) * nd, n_slabs * nd)
    JL = np.zeros(nfreq); JR = np.zeros(nfreq)
    for iq in range(nq):
        SigR = np.zeros((nfreq, N_D, N_D), dtype=complex)
        SigL = np.zeros_like(SigR); SigG = np.zeros_like(SigR)
        for (I, J), blk in sl_d.items():
            sI = slice(I * nd, (I + 1) * nd); sJ = slice(J * nd, (J + 1) * nd)
            SigL[:, sI, sJ] = blk[iq]
            SigG[:, sI, sJ] = sg_d[(I, J)][iq]
        sigr = build_retarded(SigL[None], SigG[None], freqs, method="pv")[0]
        SigR[:] = sigr
        _gr, gl, gg = solve_green_batch(z2_arr, H_D[iq], obc[iq], SigR, SigL, SigG)
        JL += HBAR_SI * omega_rad * np.real(np.trace(
            obc[iq]["Sigma_L_greater"][:, sl0, sl0] @ gl[:, sl0, sl0]
            - obc[iq]["Sigma_L_lesser"][:, sl0, sl0] @ gg[:, sl0, sl0], axis1=-2, axis2=-1))
        JR += HBAR_SI * omega_rad * np.real(np.trace(
            obc[iq]["Sigma_R_lesser"][:, slast, slast] @ gg[:, slast, slast]
            - obc[iq]["Sigma_R_greater"][:, slast, slast] @ gl[:, slast, slast], axis1=-2, axis2=-1))
    JL /= nq; JR /= nq
    J = 0.5 * (np.sum(JL[pos_mask]) + np.sum(JR[pos_mask])) * dw * 1e12
    G = J / (A_c * dT)
    # off-diagonal weight of Sigma^<
    off = sum(np.linalg.norm(sl_d[k]) for k in sl_d if k[0] != k[1])
    dia = sum(np.linalg.norm(sl_d[k]) for k in sl_d if k[0] == k[1])
    return G, off / (dia + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nk", type=int, default=4)
    ap.add_argument("--nfreq", type=int, default=61)
    ap.add_argument("--n-slabs", type=int, default=3)
    ap.add_argument("--eta-factor", type=float, default=0.1)
    ap.add_argument("--transport-dir", default="x")
    ap.add_argument("--config", default=None,
                    help="optional load_system YAML (e.g. a CNT) instead of bulk Si; "
                         "transport along its configured axis")
    args = ap.parse_args()
    if args.config:
        from phonon.finite_analysis.loader import load_system
        from phonon.solver.dense import load_fc3_raw
        axis = {"x": 0, "y": 1, "z": 2}[args.transport_dir]
        b = load_system(Path(args.config), validate=False, transport_axis=axis)
        ph = b.phonon
        fc3 = load_fc3_raw(str(Path(b.meta["fc3_path"]).expanduser().resolve()))
    else:
        ph, fc3_path = load_bulk_si()
        with h5py.File(fc3_path, "r") as f:
            fc3 = f["fc3"][:]
    nat = len(ph.primitive.masses)
    prim_indices, cell_frac, slab_indices, ref_sc = build_supercell_mapping(ph, args.transport_dir)
    M_stacked = build_realspace_fc3_matrices(fc3, nat, ph.supercell.masses, ref_sc)
    d = layer_spacing_m(ph, args.transport_dir)
    print(f"Si film, {args.n_slabs} slabs ({args.n_slabs*d*1e9:.2f} nm), nk={args.nk}, "
          f"nfreq={args.nfreq} -- full self-energy vs Guo approximations (corrected prefactor)",
          flush=True)
    G_full = None
    for approx in ["full", "diag-only", "1st-NN-FC3", "g-range1"]:
        G, offfrac = heat_conductance(
            ph, M_stacked, prim_indices, cell_frac, slab_indices, nat,
            args.n_slabs, args.nk, args.nfreq, args.eta_factor, args.transport_dir, approx)
        if approx == "full":
            G_full = G
        dev = 100.0 * (G - G_full) / G_full if G_full else 0.0
        print(f"  {approx:12s}: G_anh = {G/1e6:8.2f} MW/m2K   "
              f"({dev:+.1f}% vs full)   off-diag Sigma weight = {offfrac:.3f}", flush=True)


if __name__ == "__main__":
    main()
