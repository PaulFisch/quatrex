"""cnt33 linewidths: phono3py (golden rule) vs NEGF single-shot vs NEGF SCBA,
with the conserving vertex (raw hiphive FC3, no ASR projection).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

# The solver/phonon_inputs packages use flat intra-repo imports
# (``from phonon_inputs...``, ``from solver...``), so the phonon/ directory
# must be importable in addition to the repo root.
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "phonon")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phonon.studies import style

#: fixed physics parameters (same as the verified script)
T_KELVIN = 300.0
NE = 161
FMAX = 55.0
MIN_FREQ_THZ = 0.3      # skip (near-)zero modes in the gamma extraction
GAMMA3_FLOOR = 1e-3     # plot only modes with a resolvable phono3py gamma

#: cnt33 inputs (hiphive_meta.json + fc3.hdf5)
INPUT_DIR = ROOT / "phonon/configs/cnt/fc3_hiphive_cnt33_vasp"

OUT = Path(__file__).resolve().parent / "out"
DEFAULT_NPZ = OUT / "cnt33_linewidths.npz"
FIG_NAME = "cnt33_scba_linewidths"


def run(argv) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phonon.studies linewidths run",
        description="cnt33 equilibrium SCBA linewidths vs phono3py.")
    parser.add_argument("--nm", type=int, default=8,
                        help="q-mesh points along the tube (default 8)")
    parser.add_argument("--eta", type=float, default=0.2,
                        help="broadening eta in THz (default 0.2)")
    parser.add_argument("--niter", type=int, default=25,
                        help="max SCBA iterations (default 25)")
    parser.add_argument("--mix", type=float, default=0.1,
                        help="linear mixing factor (default 0.1)")
    parser.add_argument("--workers", type=int, default=10,
                        help="bubble kernel workers (default 10)")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ,
                        help=f"results snapshot path (default {DEFAULT_NPZ})")
    args = parser.parse_args(argv)
    NM, ETA, NITER, MIX = args.nm, args.eta, args.niter, args.mix

    warnings.filterwarnings("ignore")

    import h5py
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    from phonon.phonon_inputs.separable import (
        build_supercell_mapping, build_realspace_fc3_matrices)
    from phonon.solver.grids import bose_full_axis
    from phonon.studies.reference_kernels import (
        compute_phph_self_energy_q_dense)

    meta = json.load(open(INPUT_DIR / "hiphive_meta.json"))
    prim = meta["primitive"]
    unit = PhonopyAtoms(symbols=prim["symbols"], cell=np.array(prim["cell"]),
                        scaled_positions=np.array(prim["scaled_positions"]))
    ph = Phonopy(unit, supercell_matrix=np.diag([1, 1, 3]),
                 primitive_matrix=np.eye(3))
    with h5py.File(INPUT_DIR / "fc3.hdf5", "r") as f:
        fc2, fc3 = f["fc2"][...], f["fc3"][...]
    ph.force_constants = fc2
    nat = len(ph.primitive.masses)
    nd = 3 * nat

    pi, cf, si, ref = build_supercell_mapping(ph, "z")
    M_raw = build_realspace_fc3_matrices(fc3, nat, ph.supercell.masses, ref)

    qs = [(0.0, 0.0, k / NM) for k in range(NM)]
    nq = len(qs)
    q_idx = {round(q[2] % 1, 6): n for n, q in enumerate(qs)}
    qd = np.zeros((nq, nq), int)
    for a in range(nq):
        for b in range(nq):
            qd[a, b] = q_idx[round((qs[a][2] - qs[b][2]) % 1, 6)]

    def gmat(qfrac):
        ns = len(pi)
        Tm = np.zeros((nd, ns * 3), complex)
        phases = np.exp(-2j * np.pi * cf @ np.asarray(qfrac))
        for s in range(ns):
            for b in range(3):
                Tm[pi[s] * 3 + b, s * 3 + b] = phases[s]
        return Tm

    T_all = [gmat(q) for q in qs]
    freqs = np.linspace(-FMAX, FMAX, NE)
    freqs -= freqs[NE // 2]
    dw = float(freqs[1] - freqs[0])
    nB = bose_full_axis(freqs, T_KELVIN)
    z2 = ((freqs + 1j * ETA) ** 2)[:, None, None] * np.eye(nd)[None]

    om = np.zeros((nq, nd))
    ev = np.zeros((nq, nd, nd), complex)
    Dq = np.zeros((nq, nd, nd), complex)
    for iq, q in enumerate(qs):
        fr, e = ph.get_frequencies_with_eigenvectors(np.array(q))
        fr = np.real(fr)
        om[iq], ev[iq] = fr, e
        Dq[iq] = e @ np.diag(fr.astype(complex) ** 2) @ e.conj().T

    def greens(sigR):
        Gl = np.zeros((nq, NE, nd, nd), complex)
        Gg = np.zeros_like(Gl)
        for iq in range(nq):
            GR = np.linalg.inv(z2 - Dq[iq][None] - sigR[iq])
            A = 1j * (GR - GR.conj().transpose(0, 2, 1))
            Gl[iq] = -1j * nB[:, None, None] * A
            Gg[iq] = -1j * (nB[:, None, None] + 1.0) * A
        return Gl, Gg

    def gamma_bands(sl, sg, iq):
        g = np.full(nd, np.nan)
        for b in range(nd):
            w_b = om[iq, b]
            if w_b < MIN_FREQ_THZ:
                continue
            i_w = int(round((w_b - freqs[0]) / dw))
            m = 0.5j * (sg[iq, i_w] - sl[iq, i_w])  # -Im Sigma^R matrix
            g[b] = np.real(ev[iq][:, b].conj() @ m @ ev[iq][:, b]) / (2.0 * w_b)
        return g

    # ---- FD-closure periodic SCBA loop ----------------------------------
    sigR = np.zeros((nq, NE, nd, nd), complex)
    gam_ss = None
    for it in range(NITER):
        Gl, Gg = greens(sigR)
        sl, sg = compute_phph_self_energy_q_dense(
            Gl, Gg, M_raw, T_all, qd, nat, nq, freqs, dw,
            n_workers=args.workers, symmetry_factor=1.0)
        sig_new = 0.5 * (sg - sl)  # "half" retarded rule
        if it == 0:
            gam_ss = np.array([gamma_bands(sl, sg, iq) for iq in range(nq)])
            sigR = sig_new
            res = 1.0
        else:
            res = float(np.abs(sig_new - sigR).max()
                        / (np.abs(sig_new).max() + 1e-300))
            sigR = (1 - MIX) * sigR + MIX * sig_new
        print(f"iter {it:2d} rel dSigma={res:.3e}", flush=True)
        if it > 3 and res < 2e-3:
            break

    Glf, Ggf = greens(sigR)
    slf, sgf = compute_phph_self_energy_q_dense(
        Glf, Ggf, M_raw, T_all, qd, nat, nq, freqs, dw,
        n_workers=args.workers, symmetry_factor=1.0)
    gam_scba = np.array([gamma_bands(slf, sgf, iq) for iq in range(nq)])

    # ---- phono3py on the same mesh ---------------------------------------
    from phono3py import Phono3py

    ph3 = Phono3py(unit, supercell_matrix=np.diag([1, 1, 3]),
                   primitive_matrix=np.eye(3))
    ph3.mesh_numbers = [1, 1, NM]
    ph3.fc2 = fc2
    ph3.fc3 = fc3
    ph3.init_phph_interaction()
    ph3.run_phonon_solver()
    addresses = ph3.grid.addresses
    gp_of_iq = {}
    for iq in range(NM):
        for n in range(len(addresses)):
            if tuple(addresses[n]) == (0, 0, iq if iq <= NM // 2 else iq - NM):
                gp_of_iq[iq] = n
                break
    probe_iqs = sorted(set([NM // 4, NM // 3, NM // 2]))
    # On-shell Gamma_b extracted CORRECTLY. phono3py returns gammas of shape
    # (sigma, temp, grid, band, freqpt) -- even with NO frequency_points it builds
    # a ~201-pt auto grid -- so the previous reshape(nprobe,-1,nd)[:, -1, :]
    # SCRAMBLED the band/freqpt axes (it took Gamma at ~max frequency, mixed
    # across bands; its "median R~1.1" was a coincidence, NOT a verification, see
    # memory phonon-sse-phono3py-verification). Evaluate Gamma_b(omega) on an
    # explicit grid and interpolate at the band frequency omega_b.
    fp = np.linspace(0.3, FMAX, 400)
    gps = [gp_of_iq[i] for i in probe_iqs]
    out = ph3.run_imag_self_energy(grid_points=gps, temperatures=[T_KELVIN],
                                   frequency_points=fp)
    gfr = np.asarray(out.gammas)[0, 0]          # (probe, band, freqpt)
    g3 = np.array([[np.interp(om[iq, b], fp, gfr[c, b]) for b in range(nd)]
                   for c, iq in enumerate(probe_iqs)])
    if np.nanmedian(g3[g3 > 1e-3]) > 10.0:
        print("WARNING: phono3py on-shell gamma median > 10 THz is UNPHYSICAL -- "
              "phono3py mis-computes this quasi-1D/vacuum-cell CNT FC3 (both the "
              "imag-self-energy and RTA tc.gamma paths). This CNT linewidth "
              "comparison is NOT a valid SSE prefactor check; use bulk Si (where "
              "phono3py is standard) or the analytic golden rule. The SSE itself "
              "is verified by theory.tex eq(fgr) + the supercell ground truth + "
              "the convention-free bulk-Si kappa ~110 W/mK match.", flush=True)

    # ---- snapshot + figure -------------------------------------------------
    args.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.npz,
        om=om, gam_ss=gam_ss, gam_scba=gam_scba, g3=g3,
        probe_iqs=np.array(probe_iqs), nm=NM, eta=ETA, T=T_KELVIN,
        niter=NITER, mix=MIX,
    )
    print(f"wrote {args.npz}", flush=True)
    _figure(args.npz)
    return 0


def plot(argv) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phonon.studies linewidths plot",
        description="Re-draw the linewidth comparison figure from the npz.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ,
                        help=f"results snapshot (default {DEFAULT_NPZ})")
    args = parser.parse_args(argv)
    _figure(args.npz)
    return 0


def _figure(npz_path: Path) -> None:
    """The 4-panel comparison: per-q linewidths + the parity panel."""
    data = np.load(npz_path)
    om, gam_ss, gam_scba, g3 = (data["om"], data["gam_ss"],
                                data["gam_scba"], data["g3"])
    probe_iqs = [int(i) for i in data["probe_iqs"]]
    NM, ETA, T = int(data["nm"]), float(data["eta"]), float(data["T"])

    fig, axes = style.figure(ncols=len(probe_iqs) + 1, width=4.2, height=3.6)
    allp, alls, allg = [], [], []
    for col, iq in enumerate(probe_iqs):
        ax = axes[col]
        gp3 = g3[col]
        ok = np.isfinite(gam_scba[iq]) & (gp3 > GAMMA3_FLOOR)
        w = om[iq][ok]
        ax.plot(w, gp3[ok], "o", ms=4, color="k",
                label="phono3py (golden rule)")
        ax.plot(w, gam_ss[iq][ok], "s", ms=4, color="C0",
                label="NEGF single-shot")
        ax.plot(w, gam_scba[iq][ok], "^", ms=4, color="C3", label="NEGF SCBA")
        ax.set_yscale("log")
        ax.set_xlabel("mode frequency (THz)")
        ax.set_title(f"q=(0,0,{iq}/{NM})", fontsize=10)
        if col == 0:
            ax.set_ylabel(r"linewidth $\gamma$ (THz)")
            ax.legend(fontsize=7)
        allp += list(gp3[ok])
        alls += list(gam_ss[iq][ok])
        allg += list(gam_scba[iq][ok])
    ax = axes[-1]
    allp, alls, allg = map(np.asarray, (allp, alls, allg))
    ax.plot(allp, alls, "s", ms=4, color="C0", alpha=0.7, label="single-shot")
    ax.plot(allp, allg, "^", ms=4, color="C3", alpha=0.7, label="SCBA")
    lim = [min(allp.min(), alls.min(), allg.min()) * 0.5,
           max(allp.max(), alls.max(), allg.max()) * 2]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\gamma$ phono3py (THz)")
    ax.set_ylabel(r"$\gamma$ NEGF (THz)")
    ax.legend(fontsize=7)
    ax.set_title("parity", fontsize=10)
    print(f"median R single-shot = {np.median(alls / allp):.3f}, "
          f"SCBA = {np.median(allg / allp):.3f}")
    fig.suptitle(f"CNT(3,3) 3-phonon linewidths, conserving vertex "
                 f"(mesh {NM}, eta {ETA}, T={T:.0f}K)", fontsize=11)
    style.save(fig, FIG_NAME)
