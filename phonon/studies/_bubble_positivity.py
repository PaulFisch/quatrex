"""Does the three-phonon bubble preserve positivity, and if not, where?

Companion to `phonon/docs/bubble_positivity.md`. The theory note proves

    -i G^{<,>} >= 0  and  vertex leg-exchange symmetry   ==>  -i Sigma >= 0

because the ring S[a,J] = Phi_L[a,c,e] A[c,b] B[e,d] Phi_R[J,d,b] is the
congruence M (A (x) B) M^dagger with M[a,(c,e)] = Phi_L[a,c,e], PROVIDED

    Phi_R[(J,Kb,Ka)][a,d,b] = conj(Phi_L[(J,Ka,Kb)][a,b,d]).       (1)

Gamma-only (nq=1) both factors are the same real-space dict, so (1) is
"real AND symmetric under exchanging the two contracted legs". At
coupled-q the left factor is conjugated in the code
(sse_phonon_phonon.py:1846-1848), so reality is NOT needed and (1)
becomes the q-carrying exchange

    Phi(q2,q1)[(J,Kb,Ka)][a,d,b] = Phi(q1,q2)[(J,Ka,Kb)][a,b,d].    (1')

Nothing in the tree ever checked either one on a SHIPPED vertex: the
audit referenced at phonon/solver/se_finite.py:372
(phonon/scripts/verify/audit_qfold_trs.py) does not exist. Sub-command
`vertex` is that audit.

Run:  QTX_ARRAY_MODULE=numpy python phonon/studies/_bubble_positivity.py vertex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phonon/studies/out/positivity"

# label, fc3_blocks.hdf5, qfold_vertices.npz (None = Gamma-only device)
SYSTEMS = [
    ("MoS2 film L3 (u121, FAILS)", "cluster/mos2f3/fc3_blocks.hdf5",
     "cluster/mos2f3/qfold_vertices.npz"),
    ("MoS2 film L3 (nu grid, FAILS)", "cluster/mos2f3nu/fc3_blocks.hdf5",
     "cluster/mos2f3nu/qfold_vertices.npz"),
    ("MoS2 film L3 (o4 vertex)", "cluster/mos2f3o4/fc3_blocks.hdf5", None),
    ("MoS2 film L3 (scp vertex)", "cluster/mos2f3scp/fc3_blocks.hdf5", None),
    ("Si film nk9r (control)", "cluster/sifilm_nk9r/fc3_blocks.hdf5",
     "cluster/sifilm_nk9r/qfold_vertices.npz"),
    ("CNT33 L4 (control)", "cluster/l4gpu/fc3_blocks.hdf5", None),
    ("CNT33 cal (control)", "cluster/cnt_cal/fc3_blocks.hdf5", None),
]


def _load_gamma(path: Path):
    """Read the block-sparse fc3 dict straight from HDF5.

    Bypasses `load_device_fc3` so we do not have to supply block_sizes
    (they are in the file) and so no nn_only truncation is applied --
    the audit must see exactly what was shipped.
    """
    import h5py

    with h5py.File(str(path), "r") as f:
        if "fc3_blocks" not in f:
            return None, None
        sizes = np.asarray(f["meta/block_sizes"], dtype=int)
        grp = f["fc3_blocks"]
        blocks = {}
        for name in grp.keys():
            ds = grp[name]
            key = (int(ds.attrs["I"]), int(ds.attrs["J"]), int(ds.attrs["K"]))
            blocks[key] = np.asarray(ds, dtype=np.complex128)
    return blocks, sizes


def _exchange_defect(blocks: dict) -> dict:
    """Relative violation of the leg-exchange symmetry (1) on a dict.

    For every block (I, Ka, Kb) compare Phi[(I,Kb,Ka)][a,d,b] against
    Phi[(I,Ka,Kb)][a,b,d], i.e. `.transpose(0, 2, 1)`. A block present
    on one side and absent on the other is itself a violation (the
    missing partner is an implicit zero), so it is counted, not skipped.
    """
    scale = max((float(np.abs(v).max()) for v in blocks.values()), default=0.0)
    if scale == 0.0:
        return {"scale": 0.0, "rel": 0.0, "n_blocks": len(blocks)}
    worst = 0.0
    worst_key = None
    n_missing = 0
    for (I, Ka, Kb), phi in blocks.items():
        partner = blocks.get((I, Kb, Ka))
        if partner is None:
            partner = np.zeros_like(phi)
            n_missing += 1
        d = float(np.abs(partner - phi.transpose(0, 2, 1)).max())
        if d > worst:
            worst, worst_key = d, (I, Ka, Kb)
    return {"scale": scale, "rel": worst / scale, "worst_block": worst_key,
            "n_blocks": len(blocks), "n_missing_partner": n_missing}


def _reality_defect(blocks: dict) -> float:
    scale = max((float(np.abs(v).max()) for v in blocks.values()), default=0.0)
    if scale == 0.0:
        return 0.0
    return max(float(np.abs(v.imag).max()) for v in blocks.values()) / scale


def _s3_defect(blocks: dict) -> dict:
    """Full S3: also the (output leg <-> first contracted leg) exchange,
    Phi[(Ka,I,Kb)][b,a,d] vs Phi[(I,Ka,Kb)][a,b,d]. Not required by the
    congruence -- reported because it is the physical property the fit
    is supposed to have, so a clean S3 with a broken (1) would point at
    the device truncation rather than the fit."""
    scale = max((float(np.abs(v).max()) for v in blocks.values()), default=0.0)
    if scale == 0.0:
        return {"rel": 0.0}
    worst, worst_key = 0.0, None
    for (I, Ka, Kb), phi in blocks.items():
        partner = blocks.get((Ka, I, Kb))
        if partner is None:
            partner = np.zeros_like(phi)
        d = float(np.abs(partner - phi.transpose(1, 0, 2)).max())
        if d > worst:
            worst, worst_key = d, (I, Ka, Kb)
    return {"rel": worst / scale, "worst_block": worst_key}


def _qfold_defect(path: Path) -> dict:
    """Relative violation of (1') across every (q1, q2) pair."""
    sys.path.insert(0, str(ROOT / "src"))
    from quatrex.phonon.qfold import load_qfold

    vertices, q_diff_map, nk_shape = load_qfold(path)
    scale = 0.0
    for blocks in vertices.values():
        for v in blocks.values():
            scale = max(scale, float(np.abs(v).max()))
    if scale == 0.0:
        return {"scale": 0.0, "rel": 0.0}

    worst, worst_key = 0.0, None
    n_pairs = n_missing = 0
    for (iq1, iq2), blocks in vertices.items():
        partner_pair = vertices.get((iq2, iq1))
        if partner_pair is None:
            n_missing += 1
            continue
        n_pairs += 1
        for (I, Ka, Kb), phi in blocks.items():
            partner = partner_pair.get((I, Kb, Ka))
            if partner is None:
                partner = np.zeros_like(phi)
            d = float(np.abs(partner - phi.transpose(0, 2, 1)).max())
            if d > worst:
                worst, worst_key = d, (iq1, iq2, I, Ka, Kb)
    return {"scale": scale, "rel": worst / scale, "worst": worst_key,
            "n_q_pairs": n_pairs, "n_missing_q_partner": n_missing,
            "nk_shape": list(nk_shape), "n_q": len(vertices)}


def cmd_vertex() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    print("Leg-exchange symmetry of the SHIPPED vertices.")
    print("The ring is a congruence -- hence -i Sigma >= 0 -- only if these")
    print("are zero to roundoff. See phonon/docs/bubble_positivity.md eq (1).\n")
    print(f"{'system':32s} {'blocks':>7} {'reality':>10} {'exch (1)':>10} "
          f"{'S3':>10} {'qfold (1p)':>11}")
    for label, fc3_rel, qf_rel in SYSTEMS:
        fc3 = ROOT / fc3_rel
        if not fc3.exists():
            print(f"{label:32s}  MISSING {fc3_rel}")
            continue
        blocks, sizes = _load_gamma(fc3)
        if blocks is None:
            print(f"{label:32s}  no /fc3_blocks (legacy dense)")
            continue
        ex = _exchange_defect(blocks)
        re_ = _reality_defect(blocks)
        s3 = _s3_defect(blocks)
        entry = {"fc3": fc3_rel, "block_sizes": sizes.tolist(),
                 "reality_rel": re_, "exchange": ex, "s3": s3}
        qtxt = "     -"
        if qf_rel is not None and (ROOT / qf_rel).exists():
            qd = _qfold_defect(ROOT / qf_rel)
            entry["qfold"] = qd
            qtxt = f"{qd['rel']:11.2e}"
        rep[label] = entry
        print(f"{label:32s} {ex['n_blocks']:7d} {re_:10.2e} "
              f"{ex['rel']:10.2e} {s3['rel']:10.2e} {qtxt}")

    print("\nreality  : max|Im Phi| / max|Phi|            (needed only at nq=1)")
    print("exch (1) : Phi[(I,Kb,Ka)] vs Phi[(I,Ka,Kb)]^T  (needed always)")
    print("S3       : output-leg exchange (context, not required)")
    print("qfold(1p): Phi(q2,q1)[(J,Kb,Ka)] vs Phi(q1,q2)[(J,Ka,Kb)]^T")
    (OUT / "vertex_symmetry.json").write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT / 'vertex_symmetry.json'}")


# ---------------------------------------------------------------------------
# N2/N3: does the block-band truncation of Sigma cost positivity, and how much
# does putting several transport cells in one block buy back?
#
# Theory (bubble_positivity.md Thm 3 + Prop 4): Sigma is Hadamard-masked onto
# the block-tridiagonal, that mask is indefinite (min eig 1-sqrt(2) on three
# blocks), and the damage is bounded by the discarded weight. Blocking c cells
# per slab keeps |I-J| <= 1 in units of c cells, so it moves the short links
# into the untouched diagonal block. Offline the blocking is ONLY a choice of
# mask on the same dense Sigma, so the whole ladder costs nothing.
#
# Everything here is the Gamma slice (all transverse phases = 1). That is exact
# for the Gamma-only devices (CNT) and a decoupled-q slice for the films; the
# mask acts on BLOCK indices, so the structural question is q-independent.
# ---------------------------------------------------------------------------

# label, dir, transport axis, cells-per-block ladder, vertex override
DEVICES = [
    ("MoS2 film L3 (nu vertex, FAILS)", "cluster/mos2f3", 2, (1, 3), None),
    ("MoS2 film L3 (scp 15-blk vertex)", "cluster/mos2f3", 2, (1, 3),
     "cluster/mos2f3scp/fc3_blocks.hdf5"),
    # The 6-cell MoS2 pair (phonon/studies/engine/reblock_device.py): same
    # 108-dof device, two blockings. mos2f6x1's c=2 rung must reproduce
    # mos2f6x2's c=1 rung exactly -- that cross-check validates the build.
    ("MoS2 film 6 cells, 6x1 blocking", "cluster/mos2f6x1", 2,
     (1, 2, 3, 6), None),
    ("MoS2 film 6 cells, 3x2 blocking", "cluster/mos2f6x2", 2, (1, 3), None),
    ("Si film L8 (control)", "cluster/prod/geom/sifilm_L8_nk9", 0,
     (1, 2, 4), None),
    ("CNT33 L4 (control)", "cluster/l4gpu", 2, (1, 2), None),
]


def _gamma_blocks(dirpath: Path, tdir: int, nd: int):
    """Gamma-point transport blocks D(0), D(+1), D(-1) from the .mat.

    Mirrors phonon/studies/_ir_killtest.py:load_gamma_blocks, generalised
    to an arbitrary transport axis: at Gamma every transverse phase is 1,
    so the transverse keys simply sum.
    """
    import re

    from scipy.io import loadmat

    raw = loadmat(dirpath / "dynamical_matrix.mat")
    acc: dict[int, np.ndarray] = {}
    for key, val in raw.items():
        if key.startswith("__"):
            continue
        idx = [int(x) for x in re.findall(r"-?\d+", key)]
        n = idx[tdir]
        acc.setdefault(n, np.zeros((nd, nd), complex))
        acc[n] += val
    d00, d01 = acc[0], acc[1]
    d10 = acc[-1]
    assert np.allclose(d00, d00.conj().T, atol=1e-9), "D00 not hermitian"
    assert np.allclose(d10, d01.conj().T, atol=1e-9), "D10 != D01^dagger"
    return d00, d01, d10


def _ballistic_g(ws, d00, d01, d10, nslab, nd, t_left, t_right):
    """Dense ballistic G^{<,>} via the PRODUCTION spectral (NEVP) OBC at eta=0.

    Returns (gl, gg, gamma_worst) with gl/gg shaped (ne, N, N) in the
    occupation-positive convention, and gamma_worst the worst relative
    negative eigenvalue of the contact Gamma (hypothesis H3 -- nothing in
    production ever checks it).
    """
    from qttools.boundary_conditions.obc import Spectral
    from qttools.nevp import Full

    from quatrex.phonon.ir_subtraction import bose

    obc = Spectral(nevp=Full(), block_sections=1)
    ws = np.asarray(ws, float)
    z2 = (ws * ws).astype(complex)          # eta = 0 EXACTLY
    eye = np.eye(nd)
    m_00 = z2[:, None, None] * eye - d00[None]
    m_01 = np.broadcast_to(-d01, m_00.shape).copy()
    m_10 = np.broadcast_to(-d10, m_00.shape).copy()
    sig_l = m_10 @ obc(m_00, m_01, m_10, "left") @ m_01

    def flip(a):
        return np.flip(a, axis=(-2, -1))

    sig_r = m_01 @ flip(obc(flip(m_00), flip(m_10), flip(m_01), "right")) @ m_10

    n = nslab * nd
    dev = np.zeros((n, n), complex)
    for i in range(nslab):
        dev[i * nd:(i + 1) * nd, i * nd:(i + 1) * nd] = d00
        if i + 1 < nslab:
            dev[i * nd:(i + 1) * nd, (i + 1) * nd:(i + 2) * nd] = d01
            dev[(i + 1) * nd:(i + 2) * nd, i * nd:(i + 1) * nd] = d10

    gl = np.zeros((len(ws), n, n), complex)
    gg = np.zeros_like(gl)
    gam_worst = 0.0
    gam_scale = 0.0
    for i, w in enumerate(ws):
        gam_l = 1j * (sig_l[i] - sig_l[i].conj().T)
        gam_r = 1j * (sig_r[i] - sig_r[i].conj().T)
        for gam in (gam_l, gam_r):
            ev = np.linalg.eigvalsh(0.5 * (gam + gam.conj().T))
            gam_scale = max(gam_scale, float(np.abs(ev).max()))
            gam_worst = max(gam_worst, -float(ev.min()))
        if w <= 0.0:
            continue
        sysm = z2[i] * np.eye(n) - dev
        sysm[:nd, :nd] -= sig_l[i]
        sysm[-nd:, -nd:] -= sig_r[i]
        gr = np.linalg.solve(sysm, np.eye(n))
        ga = gr.conj().T
        gl_s = np.zeros((n, n), complex)
        gg_s = np.zeros((n, n), complex)
        n_l, n_r = bose(w, t_left), bose(w, t_right)
        gl_s[:nd, :nd] = 1j * n_l * gam_l
        gl_s[-nd:, -nd:] += 1j * n_r * gam_r
        gg_s[:nd, :nd] = 1j * (n_l + 1) * gam_l
        gg_s[-nd:, -nd:] += 1j * (n_r + 1) * gam_r
        gl[i] = gr @ gl_s @ ga
        gg[i] = gr @ gg_s @ ga
    return gl, gg, gam_worst / (gam_scale + 1e-300)


def _dense_vertex(path: Path, nslab: int, nd: int) -> np.ndarray:
    import h5py

    n = nslab * nd
    phi = np.zeros((n, n, n))
    with h5py.File(str(path), "r") as f:
        for key in f["fc3_blocks"]:
            ds = f["fc3_blocks"][key]
            I, K, Kp = int(ds.attrs["I"]), int(ds.attrs["J"]), int(ds.attrs["K"])
            blk = np.asarray(ds)
            phi[I * nd:(I + 1) * nd, K * nd:(K + 1) * nd,
                Kp * nd:(Kp + 1) * nd] = blk.real
    return phi


def _sigma_dense(phi, gl, gg, dw, max_bytes=1_500_000_000):
    """Dense Sigma^< by the production 3-term bosonic fold.

    sse_phonon_phonon.py:1351-1359 + :1407-1408 --
        Sigma^< = ring(g^<, g^<) + ring(g^<, rev g^>) + ring(rev g^>, g^<)
    with rev(X) the tau-axis index reversal AND the ji-transpose
    (:1110-1111, :2301). Each term is a ring of two PSD legs, so EACH must
    be PSD on its own -- returned separately, which is what validates this
    reimplementation (a wrong fold breaks terms 2/3 but not term 1).
    """
    from phonon.solver.bubble import _bubble_contract_batched_matmul

    from quatrex.phonon.units import bubble_prefactor_thz

    ne = gl.shape[0]
    n_fft = 2 * ne - 1

    def fwd(x):
        pad = np.zeros((n_fft,) + x.shape[1:], complex)
        pad[:ne] = x
        pad[0] = 0.0                       # production zeroes the omega=0 bin
        return np.fft.fft(pad, axis=0)

    def rev(xf):
        out = np.empty_like(xf)
        out[0] = xf[0]
        out[1:] = xf[:0:-1]
        return out.swapaxes(-2, -1)        # the ji-transpose of the fold

    glf, ggf = fwd(gl), fwd(gg)
    ggr = rev(ggf)
    pre = bubble_prefactor_thz(dw)

    def ring(a, b):
        s = _bubble_contract_batched_matmul(phi, phi, a, b, max_bytes=max_bytes)
        return pre * np.fft.ifft(s, axis=0)[:ne]

    t1, t2, t3 = ring(glf, glf), ring(glf, ggr), ring(ggr, glf)
    # production sign flip (:1681): stored Sigma has -i Sigma >= 0. The
    # omega=0 OUTPUT bin is masked too (:1661-1663) -- without it that
    # single bin carries 67-94% of the Frobenius norm (the delocalised
    # uniform-translation channel) and swamps every weight ratio.
    out = [-t1, -t2, -t3]
    for x in out:
        x[0] = 0.0
    return out


def _block_profile(sig, nslab, nd):
    """||Sigma_d|| / ||Sigma_0|| by block distance d = |I-J|.

    The quantity Proposition 4 bounds the mask damage with: what the
    |I-J| <= 1 truncation throws away relative to what it keeps.
    """
    out = {}
    for dd in range(nslab):
        tot = 0.0
        for i in range(nslab):
            for j in range(nslab):
                if abs(i - j) != dd:
                    continue
                tot += float(np.linalg.norm(
                    sig[..., i * nd:(i + 1) * nd, j * nd:(j + 1) * nd]) ** 2)
        out[dd] = np.sqrt(tot)
    n0 = out[0] + 1e-300
    return {d: v / n0 for d, v in out.items()}


def _psd_metric(mats, ws, rel_floor=1e-6):
    """Worst relative negative eigenvalue of the hermitian part of -i M.

    Normalised by the GLOBAL max |eigenvalue| over all omega. Per-frequency
    normalisation is the trap that made even a ballistic control "fail"
    (see mos2_conservation_audit.md); bins below rel_floor of the global
    scale are pure noise and are skipped outright.
    """
    lam_min = np.full(len(ws), np.nan)
    lam_absmax = np.zeros(len(ws))
    for i, w in enumerate(ws):
        if w <= 0.0:
            continue
        m = -1j * mats[i]
        m = 0.5 * (m + m.conj().T)
        ev = np.linalg.eigvalsh(m)
        lam_min[i] = ev.min()
        lam_absmax[i] = np.abs(ev).max()
    scale = float(lam_absmax.max())
    if scale <= 0.0:
        return {"worst_rel": 0.0, "scale": 0.0}
    live = lam_absmax > rel_floor * scale
    worst = float(np.nanmin(np.where(live, lam_min, np.nan)))
    idx = int(np.nanargmin(np.where(live, lam_min, np.nan)))
    return {"worst_rel": -worst / scale, "scale": scale,
            "omega_at_worst": float(ws[idx]), "n_live": int(live.sum())}


def _mask(nslab, nd, cells, taper=None):
    """DOF-level 0/1 (or tapered) mask for |I-J| <= 1 in units of `cells`
    transport cells per block. cells >= nslab means no mask at all."""
    blk = np.arange(nslab) // cells
    d = np.abs(blk[:, None] - blk[None, :])
    w = np.where(d <= 1, 1.0, 0.0)
    if taper is not None:                  # Bartlett band-1: w = [1, 1/2]
        w = np.where(d == 0, taper[0], np.where(d == 1, taper[1], 0.0))
    return np.kron(w, np.ones((nd, nd)))


def cmd_blocking() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    print("Block-band truncation of Sigma: does it cost positivity, and does")
    print("putting several transport cells in one block buy it back?")
    print("Gamma slice, eta = 0, production spectral OBC, 3-term bosonic fold.\n")
    for label, rel, tdir, ladder, vtx in DEVICES:
        d = ROOT / rel
        fc3 = Path(vtx) if vtx else (d / "fc3_blocks.hdf5")
        fc3 = ROOT / fc3 if not fc3.is_absolute() else fc3
        if not (d / "dynamical_matrix.mat").exists() or not fc3.exists():
            print(f"{label}: MISSING inputs")
            continue
        import h5py
        with h5py.File(str(fc3), "r") as f:
            sizes = np.asarray(f["meta/block_sizes"], int)
        nslab, nd = len(sizes), int(sizes[0])
        ws = np.load(d / "phonon_energies.npy")
        dw = float(ws[1] - ws[0])

        d00, d01, d10 = _gamma_blocks(d, tdir, nd)
        gl, gg, gam_worst = _ballistic_g(ws, d00, d01, d10, nslab, nd,
                                         305.0, 295.0)
        leg_l = _psd_metric(gl, ws)
        leg_g = _psd_metric(gg, ws)
        phi = _dense_vertex(fc3, nslab, nd)
        terms = _sigma_dense(phi, gl, gg, dw)
        sl = terms[0] + terms[1] + terms[2]

        print(f"--- {label}   ({nslab} slabs x {nd} dof = {nslab*nd})")
        print(f"    contact Gamma worst neg (H3)      {gam_worst:.2e}")
        print(f"    legs  -iG^< / -iG^> worst neg     {leg_l['worst_rel']:.2e}"
              f" / {leg_g['worst_rel']:.2e}")
        tm = [_psd_metric(t, ws)["worst_rel"] for t in terms]
        print(f"    fold terms 1/2/3 worst neg        "
              f"{tm[0]:.2e} / {tm[1]:.2e} / {tm[2]:.2e}   (each must be ~0)")
        full = _psd_metric(sl, ws)
        print(f"    Sigma^< UNMASKED worst neg        {full['worst_rel']:.2e}"
              f"   <- Theorem 1 on real inputs")

        iw = int(np.argmin(np.abs(ws - 0.5 * ws.max())))
        prof_all = _block_profile(sl, nslab, nd)
        prof_mid = _block_profile(sl[iw], nslab, nd)
        print("    ||Sigma_d||/||Sigma_0|| by block distance d:")
        print("      all omega  " + "  ".join(
            f"d{d}={v:.3f}" for d, v in prof_all.items()))
        print(f"      w={ws[iw]:.2f} THz " + "  ".join(
            f"d{d}={v:.3f}" for d, v in prof_mid.items()))

        rows = {}
        fro = np.linalg.norm(sl)
        for c in ladder:
            m = _mask(nslab, nd, c)
            met = _psd_metric(sl * m, ws)
            disc = float(np.linalg.norm(sl * (1.0 - m)) / (fro + 1e-300))
            tag = ("no mask" if c >= nslab
                   else f"{c} cell/block ({nslab // c} blocks)")
            rows[tag] = {"worst_rel": met["worst_rel"], "discarded": disc,
                         "omega_at_worst": met.get("omega_at_worst")}
            print(f"    band |I-J|<=1, {tag:24s} worst neg {met['worst_rel']:.2e}"
                  f"   discarded {disc:.2%}")
        mb = _mask(nslab, nd, 1, taper=(1.0, 0.5))
        met = _psd_metric(sl * mb, ws)
        rows["bartlett band-1, 1 cell/block"] = {
            "worst_rel": met["worst_rel"],
            "discarded": float(np.linalg.norm(sl * (1.0 - mb)) / (fro + 1e-300))}
        print(f"    bartlett band-1 (1 cell/block)        "
              f"worst neg {met['worst_rel']:.2e}")

        rep[label] = {"dir": rel, "nslab": nslab, "nd": nd,
                      "gamma_worst_neg": gam_worst,
                      "leg_lesser_worst_neg": leg_l["worst_rel"],
                      "leg_greater_worst_neg": leg_g["worst_rel"],
                      "fold_terms_worst_neg": tm,
                      "sigma_unmasked_worst_neg": full["worst_rel"],
                      "block_profile_all_omega": prof_all,
                      "block_profile_mid_omega": prof_mid,
                      "mid_omega_thz": float(ws[iw]),
                      "masks": rows}
        print()
    (OUT / "blocking.json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {OUT / 'blocking.json'}")


COMMANDS = {"vertex": cmd_vertex, "blocking": cmd_blocking}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "vertex"
    if cmd not in COMMANDS:
        raise SystemExit(f"unknown command {cmd!r}; have {list(COMMANDS)}")
    COMMANDS[cmd]()
