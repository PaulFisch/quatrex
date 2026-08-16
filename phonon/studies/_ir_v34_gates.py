"""V3 + V4 gates for the CM-channel subtraction
(phonon/docs/ir_residue_derivation.md Sec. 6).

(phonon/docs/ir_residue_derivation.md Sec. 6).
Run:  QTX_ARRAY_MODULE=numpy OMP_NUM_THREADS=1         python phonon/studies/_ir_v34_gates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.studies._ir_conserve_gate import (  # noqa: E402
    N, ND, NSLAB, chain_blocks, cm_channel, contact_sigmas, ring, s3_vertex,
    s_pair)
from quatrex.phonon.ir_subtraction import bose  # noqa: E402

W_MAX = 6.0
T_L, T_R = 305.0, 295.0
G2 = 4e-3         # vertex strength (scales the unit-norm S3 vertex)
MIX = 0.2
MAX_IT = 400
TOL = 1e-9


def lead_data(dw, npos, d00, d01, d10):
    ws = np.arange(1, npos + 1) * dw
    sig_l, sig_r = contact_sigmas(ws, d00, d01, d10)
    return ws, sig_l, sig_r


def keldysh(ws, sig_l, sig_r, dev, sl_anh, sg_anh, sr_anh, t_l, t_r):
    """Dense G^{<,>} on the positive axis with lead + anharmonic Sigma."""
    npos = len(ws)
    gl = np.zeros((npos, N, N), complex)
    gg = np.zeros((npos, N, N), complex)
    for i, w in enumerate(ws):
        sys_m = (w * w) * np.eye(N) - dev - sr_anh[i]
        sys_m[:ND, :ND] -= sig_l[i]
        sys_m[-ND:, -ND:] -= sig_r[i]
        gr = np.linalg.solve(sys_m, np.eye(N))
        ga = gr.conj().T
        gam_l = np.zeros((N, N), complex)
        gam_r = np.zeros((N, N), complex)
        gam_l[:ND, :ND] = 1j * (sig_l[i] - sig_l[i].conj().T)
        gam_r[-ND:, -ND:] = 1j * (sig_r[i] - sig_r[i].conj().T)
        n_l, n_r = bose(w, t_l), bose(w, t_r)
        sl_tot = 1j * (n_l * gam_l + n_r * gam_r) + sl_anh[i]
        sg_tot = 1j * ((n_l + 1) * gam_l + (n_r + 1) * gam_r) + sg_anh[i]
        gl[i] = gr @ sl_tot @ ga
        gg[i] = gr @ sg_tot @ ga
    return gl, gg


def full_axis(pos_l, pos_g, npos):
    """Legs on the full axis via the bosonic fold; w=0 bin zero."""
    nax = 2 * npos + 1
    legs_l = np.zeros((nax, N, N), complex)
    legs_g = np.zeros((nax, N, N), complex)
    for i in range(npos):
        legs_l[npos + 1 + i] = pos_l[i]
        legs_g[npos + 1 + i] = pos_g[i]
        legs_l[npos - 1 - i] = pos_g[i].T
        legs_g[npos - 1 - i] = pos_l[i].T
    return legs_l, legs_g


def bubble_full(phi, legs_l, legs_g, npos, dw):
    """Exact full-axis convolution via the tau-domain (FFT) trick: the
    ring is bilinear, so FFT the legs along omega, ring pointwise in
    tau, inverse FFT (zero-padded -> linear convolution). Identical to
    the direct double sum (production _compute_fft_first structure)."""
    nax = 2 * npos + 1
    n_fft = 2 * nax - 1
    out = []
    for legs in (legs_l, legs_g):
        ft = np.fft.fft(legs, n=n_fft, axis=0)
        t1 = np.einsum("ace,tcb->taeb", phi, ft)
        t2 = np.einsum("taeb,ted->tabd", t1, ft)
        s_tau = np.einsum("tabd,jdb->taj", t2, phi)
        conv = np.fft.ifft(s_tau, axis=0)
        # linear-conv index m' = i + j in [0, 2*nax-2]; physical
        # m = m' - 2*npos; keep m in [-npos, npos]
        out.append(conv[npos:npos + nax] * dw)
    return out[0], out[1]


def scba(dw, sub, t_l, t_r, verbose=False):
    """eta=0 SCBA on the chain bed; returns (converged, n_it, res_traj,
    fixed-point positive-axis sigma^{<,>}, legs)."""
    d00, d01, d10 = chain_blocks()
    npos = int(round(W_MAX / dw))
    ws, sig_l, sig_r = lead_data(dw, npos, d00, d01, d10)
    dev = np.zeros((N, N), complex)
    for i in range(NSLAB):
        dev[i * ND:(i + 1) * ND, i * ND:(i + 1) * ND] = d00
        if i + 1 < NSLAB:
            dev[i * ND:(i + 1) * ND, (i + 1) * ND:(i + 2) * ND] = d01
            dev[(i + 1) * ND:(i + 2) * ND, i * ND:(i + 1) * ND] = d10
    v_l, v_r, P = cm_channel(d00, d01, d10)
    s_full_l = np.zeros((2 * npos + 1, N, N), complex)
    s_full_g = np.zeros((2 * npos + 1, N, N), complex)
    for k in range(-npos, npos + 1):
        if k:
            s_full_l[k + npos], s_full_g[k + npos] = s_pair(
                k * dw, v_l, v_r, P, t_l, t_r)
    phi = s3_vertex() * G2

    sl = np.zeros((npos, N, N), complex)   # anharmonic sigma^< (pos axis)
    sg = np.zeros((npos, N, N), complex)
    sr = np.zeros((npos, N, N), complex)
    res_traj = []
    for it in range(MAX_IT):
        gl, gg = keldysh(ws, sig_l, sig_r, dev, sl, sg, sr, t_l, t_r)
        legs_l, legs_g = full_axis(gl, gg, npos)
        if sub:
            legs_l = legs_l - s_full_l
            legs_g = legs_g - s_full_g
        bl, bg = bubble_full(phi, legs_l, legs_g, npos, dw)
        new_l, new_g = bl[npos + 1:], bg[npos + 1:]
        nrm = max(np.linalg.norm(new_l), 1e-300)
        res = np.linalg.norm(new_l - sl) / nrm
        res_traj.append(res)
        if not np.isfinite(res) or res > 1e6:
            return False, it + 1, res_traj, (sl, sg), (gl, gg)
        sl = (1 - MIX) * sl + MIX * new_l
        sg = (1 - MIX) * sg + MIX * new_g
        # retarded: production "half" convention -- skew part only,
        # sign fixed by requiring damping (Gamma_Sigma >= 0)
        gam_s = 1j * (sg - sl)
        gam_s = 0.5 * (gam_s + np.conj(np.swapaxes(gam_s, -2, -1)))
        sr = -0.5j * gam_s
        if res < TOL:
            return True, it + 1, res_traj, (sl, sg), (gl, gg)
    return False, MAX_IT, res_traj, (sl, sg), (gl, gg)


def main():
    report = {}
    print("V4: eta=0 SCBA on the chain bed, bare vs CM-subtracted")
    print(f"    (vertex G2={G2}, mix={MIX}, tol={TOL:g})")
    print(f"{'dw':>7} {'variant':>8} {'conv':>6} {'it':>5} "
          f"{'final res':>11} {'lambda_est':>10}")
    for dw in (0.1, 0.05, 0.025):
        for sub in (False, True):
            conv, nit, traj, _, _ = scba(dw, sub, T_L, T_R)
            lam = np.nan
            if len(traj) > 12 and np.isfinite(traj[-1]):
                r = np.mean(np.array(traj[-6:]) /
                            np.array(traj[-7:-1]))
                lam = 1.0 - (1.0 - r) / MIX
            tag = "sub" if sub else "bare"
            print(f"{dw:7.3f} {tag:>8} {str(conv):>6} {nit:5d} "
                  f"{traj[-1]:11.3e} {lam:10.4f}")
            report[f"{tag}_{dw}"] = dict(
                converged=bool(conv), n_it=int(nit),
                final_res=float(traj[-1]), lam=float(lam))

    # V3: equilibrium (T=300/300) at the subtracted fixed point
    print("\nV3: equilibrium identities (T=300 both leads)")
    from quatrex.phonon.ir_subtraction import bose as _bose
    uu = {}
    for dw in (0.1, 0.05, 0.025):
        conv, nit, _, (sl, sg), (gl, gg) = scba(dw, True, 300.0, 300.0)
        npos = int(round(W_MAX / dw))
        ws = np.arange(1, npos + 1) * dw
        # detailed balance of the anharmonic sigma at the fixed point
        db = []
        for i, w in enumerate(ws):
            n = _bose(w, 300.0)
            det = np.linalg.norm(sg[i] * n - sl[i] * (n + 1.0))
            db.append(det / (np.linalg.norm(sg[i] * n) + 1e-300))
        # equal-time <ww>: full-axis sum of iG^< (bare) vs of i(G-S)^<
        d00, d01, d10 = chain_blocks()
        v_l, v_r, P = cm_channel(d00, d01, d10)
        tr_bare, tr_sub = 0.0, 0.0
        for i, w in enumerate(ws):
            s_l, _ = s_pair(w, v_l, v_r, P, 300.0, 300.0)
            # both signs of w: fold contributes the transpose -- traces equal
            tr_bare += 2 * np.trace(1j * gl[i]).real
            tr_sub += 2 * np.trace(1j * (gl[i] - s_l)).real
        uu[dw] = (tr_bare * dw / (2 * np.pi), tr_sub * dw / (2 * np.pi))
        print(f"  dw={dw:6.3f}: converged={conv} ({nit} it), "
              f"detailed balance max {max(db):.2e}, "
              f"Tr<ww>-sum bare {uu[dw][0]:9.4f}  sub {uu[dw][1]:9.4f}")
        report[f"v3_{dw}"] = dict(db_max=float(max(db)),
                                  uu_bare=uu[dw][0], uu_sub=uu[dw][1])

    outdir = ROOT / "phonon/studies/out/ir_residue"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "v34_gates.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {outdir / 'v34_gates.json'}")


if __name__ == "__main__":
    main()
