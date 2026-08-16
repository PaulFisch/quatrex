"""Kill-test for the exact-IR (residue) program: does the film vertex
annihilate the divergent near-DC channel of the bubble legs?

    cluster/mos2f3nu/dynamical_matrix.mat, dense-solve the ballistic
    against the recorded engine data (run_ballistic.npz);
    vertex (cluster/mos2f3scp/fc3_blocks.hdf5 -- NOT the diagonal-only
eta: uses exactly the eta recorded in run_ballistic.npz (the branch
Run:  OMP_NUM_THREADS=1 python phonon/studies/_ir_killtest.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
NU = ROOT / "cluster/mos2f3nu"
VERTEX = ROOT / "cluster/mos2f3scp/fc3_blocks.hdf5"
OUT = ROOT / "phonon/studies/out/ir_residue"

MASSES = {"Mo": 95.95, "S": 32.06}
NSLAB = 3
ND = 18


# ---------------------------------------------------------------------------
# device blocks at Gamma
# ---------------------------------------------------------------------------
def load_gamma_blocks():
    raw = loadmat(NU / "dynamical_matrix.mat")
    blocks = {}
    for key, val in raw.items():
        if key.startswith("__"):
            continue
        n1, n2, n3 = (int(x) for x in re.findall(r"-?\d+", key))
        blocks.setdefault(n3, np.zeros((ND, ND), complex))
        blocks[n3] += val  # Gamma: all transverse phases = 1
    d00, d01, d10 = blocks[0], blocks[1], blocks[-1]
    assert np.allclose(d00, d00.conj().T, atol=1e-10), "D00 not hermitian"
    assert np.allclose(d10, d01.conj().T, atol=1e-10), "D10 != D01^dagger"
    return d00, d01, d10


def translations():
    """Mass-weighted uniform translations (null vectors of the crystal ASR)."""
    species = []
    for line in (NU / "structure.xyz").read_text().splitlines()[2:]:
        if line.strip():
            species.append(line.split()[0])
    sqm = np.sqrt([MASSES[s] for s in species])
    t = np.zeros((3, ND))
    for beta in range(3):
        t[beta, beta::3] = sqm
    return t / np.linalg.norm(t, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# surface GF via the PRODUCTION spectral (NEVP) OBC solver -- the run being
# reproduced had eta = eta_obc = 0 exactly; branch selection is by group
# velocity / decay in the mode-matching solver, not by broadening.
# ---------------------------------------------------------------------------
_SPECTRAL = None


def _spectral():
    global _SPECTRAL
    if _SPECTRAL is None:
        from qttools.boundary_conditions.obc import Spectral
        from qttools.nevp import Full
        _SPECTRAL = Spectral(nevp=Full(), block_sections=1)
    return _SPECTRAL


def bose(w_thz, T):
    from quatrex.phonon.ir_subtraction import bose as _b
    return _b(w_thz, T)


def contact_sigmas(ws, eta, d00, d01, d10):
    """Batched left/right retarded contact self-energies at frequencies ws.

    Mirrors src/quatrex/phonon/solver.py:_compute_obc (block_sections=1,
    s_00 = 0): left g_00 from (m_00, m_01, m_10) with contact='left';
    right via the flip trick with contact='right'."""
    obc = _spectral()
    z2 = (ws * ws + 2j * eta * np.abs(ws)).astype(complex)
    eye = np.eye(ND)
    m_00 = z2[:, None, None] * eye - d00[None]
    m_01 = np.broadcast_to(-d01, m_00.shape).copy()
    m_10 = np.broadcast_to(-d10, m_00.shape).copy()
    g_00 = obc(m_00, m_01, m_10, "left")
    sig_l = m_10 @ g_00 @ m_01

    flip = lambda a: np.flip(a, axis=(-2, -1))
    g_nn = obc(flip(m_00), flip(m_10), flip(m_01), "right")
    g_nn = flip(g_nn)
    sig_r = m_01 @ g_nn @ m_10
    return sig_l, sig_r


def device_g_batch(ws, eta, d00, d01, d10, t_left, t_right):
    """Ballistic G^R, G^<, G^> of the NSLAB-block device (batched over ws)."""
    sig_l, sig_r = contact_sigmas(ws, eta, d00, d01, d10)
    n = NSLAB * ND
    dev = np.zeros((n, n), complex)
    for i in range(NSLAB):
        dev[i * ND:(i + 1) * ND, i * ND:(i + 1) * ND] = d00
        if i + 1 < NSLAB:
            dev[i * ND:(i + 1) * ND, (i + 1) * ND:(i + 2) * ND] = d01
            dev[(i + 1) * ND:(i + 2) * ND, i * ND:(i + 1) * ND] = d10
    z2 = (ws * ws + 2j * eta * np.abs(ws)).astype(complex)
    out_r, out_l, out_g = [], [], []
    for i, w in enumerate(ws):
        sys = z2[i] * np.eye(n) - dev
        sys[:ND, :ND] -= sig_l[i]
        sys[-ND:, -ND:] -= sig_r[i]
        gr = np.linalg.solve(sys, np.eye(n))
        ga = gr.conj().T
        gam_l = np.zeros((n, n), complex)
        gam_r = np.zeros((n, n), complex)
        gam_l[:ND, :ND] = 1j * (sig_l[i] - sig_l[i].conj().T)
        gam_r[-ND:, -ND:] = 1j * (sig_r[i] - sig_r[i].conj().T)
        n_l, n_r = bose(w, t_left), bose(w, t_right)
        # production occupation-positive convention: Sigma^< = +i n Gamma
        gl = gr @ (1j * (n_l * gam_l + n_r * gam_r)) @ ga
        gg = gr @ (1j * ((n_l + 1) * gam_l + (n_r + 1) * gam_r)) @ ga
        out_r.append(gr); out_l.append(gl); out_g.append(gg)
    return np.array(out_r), np.array(out_l), np.array(out_g)


def device_g(w, eta, d00, d01, d10, t_left, t_right):
    gr, gl, gg = device_g_batch(np.array([w]), eta, d00, d01, d10,
                                t_left, t_right)
    return gr[0], gl[0], gg[0]


# ---------------------------------------------------------------------------
# vertex
# ---------------------------------------------------------------------------
def load_vertex():
    import h5py
    n = NSLAB * ND
    phi = np.zeros((n, n, n))
    with h5py.File(VERTEX, "r") as f:
        for key in f["fc3_blocks"]:
            I, K, Kp = (int(x) for x in key.split("_"))
            blk = f["fc3_blocks"][key][()]
            assert np.abs(blk.imag).max() < 1e-12 * max(np.abs(blk.real).max(), 1e-300)
            phi[I * ND:(I + 1) * ND, K * ND:(K + 1) * ND,
                Kp * ND:(Kp + 1) * ND] = blk.real
    return phi


def ring(phi, fa, fb):
    """Production pairing (bubble.py): S[a,J] = Phi[a,c,e] Fa[c,b] Fb[e,d] Phi[J,d,b]."""
    t1 = np.einsum("ace,cb->aeb", phi, fa)
    t2 = np.einsum("aeb,ed->abd", t1, fb)
    return np.einsum("abd,jdb->aj", t2, phi)


def slope(x, y):
    return np.diff(np.log(y)) / np.diff(np.log(x))


# ---------------------------------------------------------------------------
def main():
    import sys
    for p in (str(ROOT), str(ROOT / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)

    d00, d01, d10 = load_gamma_blocks()
    t18 = translations()

    # crystal ASR at Gamma: (D00 + D01 + D10) t = 0
    asr_fc2 = max(np.linalg.norm((d00 + d01 + d10) @ t18[b]) for b in range(3))
    scale_fc2 = np.linalg.norm(d00)
    print(f"fc2 ASR at Gamma: max |(D00+D01+D10) t| / |D00| = "
          f"{asr_fc2 / scale_fc2:.3e}")

    rec = np.load(NU / "run_ballistic.npz")
    eta = float(rec["eta"])
    t_l, t_r = float(rec["t_left"]), float(rec["t_right"])
    en = rec["energies"]
    print(f"recorded run: eta={eta:g}, T_L/T_R={t_l}/{t_r}")

    # ---- parity check against the recorded engine G at q=Gamma ----
    print("\nparity vs run_ballistic.npz (max_i |Im G^<_ii|, q=Gamma):")
    ok = True
    for ib in (1, 2, 3, 5):
        w = float(en[ib])
        _, gl, _ = device_g(w, eta, d00, d01, d10, t_l, t_r)
        mine = np.abs(np.imag(np.diag(gl))).max()
        theirs = np.abs(rec["gl_diag_imag"][ib, 0, 0, :]).max()
        r = mine / theirs
        ok &= abs(r - 1) < 0.05
        print(f"  w={w:8.4f}  dense {mine:12.4e}  engine {theirs:12.4e}  ratio {r:.4f}")
    print(f"  parity: {'OK' if ok else 'FAILED'}")

    # ---- divergent-channel structure: eigenvectors of w^2 G^< ----
    w0 = 1e-3
    _, gl0, _ = device_g(w0, eta, d00, d01, d10, t_l, t_r)
    c2 = (w0 * w0) * (-1j * gl0)
    c2 = 0.5 * (c2 + c2.conj().T)
    evals, evecs = np.linalg.eigh(c2.real.astype(float))
    idx = np.argsort(np.abs(evals))[::-1]
    t54 = np.zeros((3, NSLAB * ND))
    for b in range(3):
        t54[b] = np.tile(t18[b], NSLAB)
    t54 /= np.linalg.norm(t54, axis=1, keepdims=True)
    proj = t54.T @ np.linalg.solve(t54 @ t54.T, t54)
    print(f"\nw^2 G^< channel at w={w0:g}: top |eigenvalues| "
          f"{np.round(np.abs(evals[idx[:5]]), 4).tolist()}")
    for r in range(3):
        v = evecs[:, idx[r]]
        print(f"  eigvec {r}: translation-subspace overlap "
              f"{np.linalg.norm(proj @ v) ** 2:.6f}")

    # ---- vertex: per-leg translation annihilation ----
    phi = load_vertex()
    nrm = np.linalg.norm(phi)
    print(f"\nvertex: |Phi| = {nrm:.4e} (15-block scp build)")
    legs = {0: "ace->a", 1: "leg c", 2: "leg e"}
    ann = {}
    for leg in range(3):
        rmax = 0.0
        for b in range(3):
            contr = np.tensordot(phi, t54[b], axes=([leg], [0]))
            rmax = max(rmax, np.linalg.norm(contr) / nrm)
        ann[leg] = rmax
        print(f"  leg {leg}: max_beta |Phi . t| / |Phi| = {rmax:.3e}")

    # ---- decisive contraction scaling + rank-3 channel decomposition ----
    # With ANY finite translation leak the asymptotic slope of the full
    # contraction stays -2; the discriminating measurement is the split
    #   G = (G - QGQ) + QGQ,  Q = 1 - P_t  (P_t = translation projector):
    # if X1 is carried by the channel part while X1_perp = ring(QGQ, .)
    # is regular, the divergence is entirely the (device-truncated,
    # uncancellable) translation channel -- rank 3, analytically known.
    ws = np.geomspace(1e-3, 1.0, 13)
    w_ref = 3.0
    _, gl_ref, _ = device_g(w_ref, eta, d00, d01, d10, t_l, t_r)
    Q = np.eye(NSLAB * ND) - proj
    leg_norm, x1, x2 = [], [], []
    x1_perp, x1_chan, x2_perp = [], [], []
    for w in ws:
        _, gl, gg = device_g(w, eta, d00, d01, d10, t_l, t_r)
        glp = Q @ gl @ Q
        ggp = Q @ gg @ Q
        leg_norm.append(np.linalg.norm(gl))
        x1.append(np.linalg.norm(ring(phi, gl, gl_ref)))
        x2.append(np.linalg.norm(ring(phi, gl, gg.T)))
        x1_perp.append(np.linalg.norm(ring(phi, glp, gl_ref)))
        x1_chan.append(np.linalg.norm(ring(phi, gl - glp, gl_ref)))
        x2_perp.append(np.linalg.norm(ring(phi, glp, ggp.T)))
    leg_norm, x1, x2, x1_perp, x1_chan, x2_perp = map(
        np.array, (leg_norm, x1, x2, x1_perp, x1_chan, x2_perp))
    s_leg, s_x1, s_x2 = slope(ws, leg_norm), slope(ws, x1), slope(ws, x2)
    s_x1p, s_x2p = slope(ws, x1_perp), slope(ws, x2_perp)
    print(f"\nscaling over w' in [{ws[0]:g}, {ws[-1]:g}] THz:")
    print(f"  |G^<(w')|                       slopes {np.round(s_leg[:6], 3).tolist()}")
    print(f"  X1  = |ring(G^<, G^<({w_ref}))|      slopes {np.round(s_x1[:6], 3).tolist()}")
    print(f"  X1p = |ring(QG^<Q, G^<({w_ref}))|    slopes {np.round(s_x1p[:6], 3).tolist()}")
    print(f"  X2  = |ring(G^<, G^>^T)|        slopes {np.round(s_x2[:6], 3).tolist()}")
    print(f"  X2p = |ring(QG^<Q, QG^>Q^T)|    slopes {np.round(s_x2p[:6], 3).tolist()}")
    print("\n  w'         X1            X1_chan       X1_perp      chan/total")
    for i in (0, 4, 8, 12):
        print(f"  {ws[i]:8.4f}  {x1[i]:.4e}  {x1_chan[i]:.4e}  "
              f"{x1_perp[i]:.4e}  {x1_chan[i] / x1[i]:.4f}")
    # crossover: where does the channel part drop below the regular part?
    cross = ws[np.argmin(np.abs(np.log(x1_chan / x1_perp)))]
    print(f"\n  channel/regular crossover near w' ~ {cross:.3f} THz "
          f"(mask cut used: 1.5 THz)")

    full_div = s_x1[:4].mean() < -0.5
    perp_reg = s_x1p[:4].mean() > -0.5 and s_x2p[:4].mean() > -1.0
    if not full_div:
        verdict = ("GO-SIMPLE: vertex cancels the channel; plain quadrature "
                   "program proceeds")
    elif perp_reg:
        verdict = ("GO-RESCOPED: divergence is ENTIRELY the rank-3 "
                   "translation channel (device-truncation artifact, "
                   "uncancellable by the device vertex); the regular "
                   "remainder is quadrature-friendly. Program = exact "
                   "rank-3 channel subtraction. Report to Paul before P1.")
    else:
        verdict = ("NO-GO: even the translation-projected contraction "
                   "diverges; the exact integral does not exist as posed. "
                   "STOP and report.")
    print(f"\nVERDICT: {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "killtest_film.json").write_text(json.dumps({
        "eta": eta, "t_left": t_l, "t_right": t_r,
        "fc2_asr_rel": asr_fc2 / scale_fc2,
        "parity_ok": bool(ok),
        "vertex_translation_annihilation_per_leg": ann,
        "ws": ws.tolist(), "leg_norm": leg_norm.tolist(),
        "x1": x1.tolist(), "x2": x2.tolist(),
        "x1_perp": x1_perp.tolist(), "x1_chan": x1_chan.tolist(),
        "x2_perp": x2_perp.tolist(),
        "slopes": {"leg": s_leg.tolist(), "x1": s_x1.tolist(),
                   "x2": s_x2.tolist(), "x1_perp": s_x1p.tolist(),
                   "x2_perp": s_x2p.tolist()},
        "crossover_thz": float(cross),
        "verdict": verdict}, indent=1))
    print(f"wrote {OUT / 'killtest_film.json'}")


if __name__ == "__main__":
    main()
