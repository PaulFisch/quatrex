"""ASR audit + minimal S3-symmetric projection for film 3-phonon vertices.

The MoS2 film FC3 fit violates the acoustic sum rule (translation
invariance) at the ~4e-3 level (relative to rms entry), while the
converging systems (CNT, Si film) are machine-exact. This tool

  * ``audit``    measures per-class ASR residuals, S3 symmetry deviation
                 and reality of ``fc3_blocks.hdf5`` (and, if given, of the
                 q-folded ``qfold_vertices.npz`` via an exact inverse fold);
  * ``project``  removes the violation by alternating orthogonal
                 projections onto the affine sets {per-axis ASR on
                 enforceable rows} & {S3 symmetry} & {reality} -- all
                 linear subspaces, so von Neumann alternation converges to
                 the MINIMAL (Frobenius) correction;
  * ``selftest`` builds synthetic S3-symmetric fixtures with a controlled
                 ASR violation, folds them through the REAL production
                 fold path (phonon.solver.se_q._build_folded_vertices) and
                 verifies the inverse fold + projection end to end.

Conventions (verified against the build code):

  * q-mesh: the fold uses the plain Gamma-centered fractions
    q_1d = arange(nk)/nk, iq = ix*nk + iy
    (phonon/studies/engine/build_inputs.py:330-331). ``kshift.npy`` =
    0.5 - 0.5/nk (build_inputs.py:436) NEVER enters the fold: it is the
    config kpoint_shift that makes quatrex.grid.kpoints.monkhorst_pack
    (kpts = (m + 0.5)/n - 0.5 + shift, kpoints.py:30-31) land exactly on
    m/n in index order (validated in
    src/quatrex/phonon/sse_phonon_phonon.py:240-271).
  * fold phases: exp(-2j*pi * q . c) on the two CONTRACTED legs, no phase
    and no normalisation on the external leg
    (phonon/solver/se_q.py:41-46); c is the MINIMUM-IMAGED transverse
    cell of each supercell atom (phonon_inputs/separable.py:137-143),
    i.e. c in {-(w-1)//2 .. w//2} for supercell width w.  For the MoS2
    [4,4,3] reap w=4 < nk=5: the 25-point q-grid over-determines the 16
    cell pairs, the empty torus residues are exactly zero, and the
    inverse (torus IDFT then support restriction) is exact.
  * npz keys: "v|iq1|iq2|I|K|Kp" + q_diff_map + nk_shape
    (src/quatrex/phonon/qfold.py:30-46); q_diff_map[iq,iqp] =
    ((ix-ixp)%nk)*nk + (iy-iyp)%nk (phonon_inputs/separable.py:257-274).
  * cell arithmetic for the S3 action / ASR sums is mod w (the supercell
    translation group), NOT mod nk: on an even-width supercell the +w/2
    and -w/2 offsets are identified (min-image keeps +w/2), so mod-nk
    negation would map data off-support.

ASR (mass-weighted vertex): sum over atoms j of sqrt(m_j) *
Phi[..., 3j+beta, ...] over the summed leg's full crystal support
(slabs x transverse cells x atoms). A slab row is ENFORCEABLE iff the
crystal slab support of the summed leg lies inside the device; edge rows
((0,0)/(L-1,L-1) fixed pairs) are left untouched (the same edge
violation exists in the converging systems and is benign).

The former (2026-06-12, commit 268b3174) per-leg projection was removed
because it was leg-ASYMMETRIC and over-strong; this one is exactly
S3-symmetric, minimal, and aborts if ||dPhi||/||Phi|| > 0.05.

Usage:
  python _asr_project_film.py audit   --fc3 F.hdf5 [--qfold Q.npz]
  python _asr_project_film.py project --fc3 F.hdf5 [--qfold Q.npz] --out DIR
  python _asr_project_film.py selftest [--workdir DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "phonon"), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Mo2 S4 (6 atoms x 3 cart = 18 dof/slab), the MoS2 film production order.
DEFAULT_MASSES = "95.95,95.95,32.06,32.06,32.06,32.06"

PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def _perm_key(key, p):
    return (key[p[0]], key[p[1]], key[p[2]])


# ---------------------------------------------------------------------------
# fc3_blocks.hdf5 I/O (schema: phonon_inputs/quatrex_writer.py:154-200)
# ---------------------------------------------------------------------------


def load_fc3_hdf5(path):
    import h5py

    blocks = {}
    max_imag = 0.0
    with h5py.File(str(path), "r") as f:
        block_sizes = np.asarray(f["meta/block_sizes"][:], dtype=np.int64)
        units = str(f["meta"].attrs.get("units", "THz^2"))
        for name, ds in f["fc3_blocks"].items():
            key = tuple(int(x) for x in name.split("_"))
            arr = np.asarray(ds[:])
            if np.iscomplexobj(arr):
                max_imag = max(max_imag, float(np.abs(arr.imag).max()))
                arr = arr.real
            blocks[key] = np.ascontiguousarray(arr.astype(np.float64))
    return blocks, block_sizes, units, max_imag


def write_fc3_hdf5(blocks, block_sizes, path, units="THz^2"):
    """Exact write_fc3_blocks schema (quatrex_writer.py:154-200)."""
    import h5py

    block_sizes = np.asarray(block_sizes, dtype=np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(blocks.keys())
    with h5py.File(str(path), "w") as f:
        meta = f.create_group("meta")
        meta.create_dataset("block_sizes", data=block_sizes)
        meta.attrs["units"] = units
        meta.create_dataset("keys", data=np.asarray(keys, dtype=np.int64))
        grp = f.create_group("fc3_blocks")
        for (I, J, K) in keys:
            block = np.asarray(blocks[(I, J, K)])
            ds = grp.create_dataset(
                f"{I}_{J}_{K}", data=block.astype(np.complex128)
            )
            ds.attrs["I"] = I
            ds.attrs["J"] = J
            ds.attrs["K"] = K
            ds.attrs["b_I"] = block.shape[0]
            ds.attrs["b_J"] = block.shape[1]
            ds.attrs["b_K"] = block.shape[2]


# ---------------------------------------------------------------------------
# Slab constraint classes
# ---------------------------------------------------------------------------


def _axis_triple(axis, P, Q, K):
    """Device triplet with the summed slab K put on `axis`, fixed (P, Q)
    filling the remaining axes in order."""
    if axis == 0:
        return (K, P, Q)
    if axis == 1:
        return (P, K, Q)
    return (P, Q, K)


def slab_classes(n_slabs, reach, keys):
    """All (axis, fixed-pair) ASR constraint classes.

    Returns a list of dicts with axis, pair, supp (device slab list),
    triples (present device triplets), enforceable (crystal support of
    the summed leg inside the device), vacuous (no blocks at all).
    """
    keys = set(keys)
    out = []
    for axis in (0, 1, 2):
        for P in range(n_slabs):
            for Q in range(n_slabs):
                crystal = [
                    K
                    for K in range(min(P, Q) - reach, max(P, Q) + reach + 1)
                    if abs(K - P) <= reach and abs(K - Q) <= reach
                ]
                supp = [K for K in crystal if 0 <= K < n_slabs]
                triples = [
                    _axis_triple(axis, P, Q, K)
                    for K in supp
                    if _axis_triple(axis, P, Q, K) in keys
                ]
                vacuous = len(triples) == 0
                enforceable = (
                    not vacuous
                    and len(crystal) == len(supp)
                    and len(triples) == len(supp)
                )
                out.append(
                    dict(
                        axis=axis,
                        pair=(P, Q),
                        supp=supp,
                        triples=triples,
                        enforceable=enforceable,
                        vacuous=vacuous,
                    )
                )
    return out


# ---------------------------------------------------------------------------
# Gamma-rep (fc3_blocks.hdf5): ASR + S3
# ---------------------------------------------------------------------------


def _asr_values_gamma(blocks, w, cls):
    """Constraint values v (summed leg on cls['axis']) for one class."""
    nat = len(w)
    v = None
    for T in cls["triples"]:
        B = blocks[T]
        nd = B.shape[0]
        if cls["axis"] == 0:
            t = B.reshape(nat, 3, nd, nd)
            c = np.einsum("j,jabc->abc", w, t)
        elif cls["axis"] == 1:
            t = B.reshape(nd, nat, 3, nd)
            c = np.einsum("j,ajbc->abc", w, t)
        else:
            t = B.reshape(nd, nd, nat, 3)
            c = np.einsum("j,abjc->abc", w, t)
        v = c if v is None else v + c
    return v


def _asr_project_class_gamma(blocks, w, mtot, cls):
    v = _asr_values_gamma(blocks, w, cls)
    nat = len(w)
    denom = len(cls["triples"]) * mtot
    for T in cls["triples"]:
        B = blocks[T]
        nd = B.shape[0]
        if cls["axis"] == 0:
            t = B.reshape(nat, 3, nd, nd)
            t -= w[:, None, None, None] * v[None] / denom
        elif cls["axis"] == 1:
            t = B.reshape(nd, nat, 3, nd)
            t -= w[None, :, None, None] * v[:, None, :, :] / denom
        else:
            t = B.reshape(nd, nd, nat, 3)
            t -= w[None, None, :, None] * v[:, :, None, :] / denom


def _class_residual_gamma(blocks, w, mtot, cls):
    v = _asr_values_gamma(blocks, w, cls)
    rms_v = float(np.sqrt(np.mean(v**2)))
    ent = np.concatenate([blocks[T].ravel() for T in cls["triples"]])
    rms_e = float(np.sqrt(np.mean(ent**2)))
    scale = np.sqrt(len(cls["triples"]) * mtot) * rms_e
    return rms_v / max(scale, 1e-300), rms_v, rms_e


def s3_symmetrize_gamma(blocks):
    acc = {k: np.zeros_like(v) for k, v in blocks.items()}
    for k, B in blocks.items():
        for p in PERMS:
            acc[_perm_key(k, p)] += np.transpose(B, p)
    return {k: v / 6.0 for k, v in acc.items()}


def s3_deviation_gamma(blocks):
    dev = 0.0
    for k, B in blocks.items():
        for p in PERMS:
            dev = max(
                dev, float(np.abs(np.transpose(B, p) - blocks[_perm_key(k, p)]).max())
            )
    ent = np.concatenate([b.ravel() for b in blocks.values()])
    rms_e = float(np.sqrt(np.mean(ent**2)))
    return dev, dev / max(rms_e, 1e-300)


def audit_gamma(blocks, masses, n_slabs, reach, max_imag, label="fc3_blocks"):
    w = np.sqrt(masses)
    mtot = float(masses.sum())
    classes = slab_classes(n_slabs, reach, blocks.keys())
    print(f"\n== ASR audit [{label}] (Gamma rep, {len(blocks)} blocks, "
          f"n_slabs={n_slabs}, reach={reach}) ==")
    print(f"reality: max|imag| of file = {max_imag:.3e}")
    dev, rdev = s3_deviation_gamma(blocks)
    print(f"S3 symmetry: max|Phi - perm(Phi)| = {dev:.3e} "
          f"(rel to rms entry: {rdev:.3e})")
    pooled = {}
    for cls in classes:
        if cls["vacuous"]:
            continue
        rel, rms_v, rms_e = _class_residual_gamma(blocks, w, mtot, cls)
        tag = "interior" if cls["enforceable"] else "edge    "
        print(f"  axis {cls['axis']} pair {cls['pair']} supp {cls['supp']} "
              f"[{tag}]: rel={rel:.3e} (rms_v={rms_v:.3e})")
        pooled.setdefault(tag.strip(), []).append(rel)
    vac = [c for c in classes if c["vacuous"]]
    print(f"  vacuous fixed pairs (no blocks, |P-Q|>reach or empty): "
          f"{sorted({(c['axis'], c['pair']) for c in vac})}")
    summary = {}
    for tag, vals in pooled.items():
        summary[tag] = (float(np.max(vals)), float(np.sqrt(np.mean(np.square(vals)))))
        print(f"  {tag}: max rel = {summary[tag][0]:.3e}, "
              f"rms of class rels = {summary[tag][1]:.3e}")
    summary["s3_rel"] = rdev
    summary["max_imag"] = max_imag
    return summary


def project_gamma(blocks, masses, n_slabs, reach, tol=1e-13, max_sweeps=500,
                  verbose=True):
    w = np.sqrt(masses)
    mtot = float(masses.sum())
    classes = [
        c for c in slab_classes(n_slabs, reach, blocks.keys()) if c["enforceable"]
    ]
    orig = {k: v.copy() for k, v in blocks.items()}
    blocks = {k: v.copy() for k, v in blocks.items()}
    res = np.inf
    for sweep in range(1, max_sweeps + 1):
        for cls in classes:
            _asr_project_class_gamma(blocks, w, mtot, cls)
        blocks = s3_symmetrize_gamma(blocks)
        res = max(
            _class_residual_gamma(blocks, w, mtot, cls)[0] for cls in classes
        )
        if verbose and (sweep <= 5 or sweep % 20 == 0):
            print(f"  sweep {sweep:3d}: enforceable rel residual = {res:.3e}")
        if res < tol:
            break
    num = np.sqrt(sum(float(np.sum((blocks[k] - orig[k]) ** 2)) for k in orig))
    den = np.sqrt(sum(float(np.sum(orig[k] ** 2)) for k in orig))
    corr = num / max(den, 1e-300)
    print(f"  converged after {sweep} sweeps: residual={res:.3e}, "
          f"||dPhi||/||Phi|| = {corr:.6e}")
    return blocks, corr, res


# ---------------------------------------------------------------------------
# qfold_vertices.npz: load / inverse fold / refold
# ---------------------------------------------------------------------------


def load_qfold_npz(path):
    """Parse the save_qfold format (src/quatrex/phonon/qfold.py:30-46)."""
    npz = np.load(str(path))
    q_diff_map = np.asarray(npz["q_diff_map"], dtype=int)
    nk_shape = tuple(int(k) for k in npz["nk_shape"])
    vkeys = {}
    for key in npz.files:
        if not key.startswith("v|"):
            continue
        _, iq1, iq2, I, K, Kp = key.split("|")
        vkeys.setdefault((int(iq1), int(iq2)), {})[
            (int(I), int(K), int(Kp))
        ] = key
    return npz, vkeys, q_diff_map, nk_shape


def _dft_apply(A, F, axes):
    """Apply matrix F (out, in) along each axis in `axes` of A."""
    for ax in axes:
        A = np.moveaxis(np.tensordot(F[ax], A, axes=(1, ax)), 0, ax)
    return A


def invert_fold(npz, vkeys, nk_shape, support_tol=1e-10):
    """Exact inverse of the per-leg transverse Bloch fold.

    Returns (C, cells, meta): C = {(I,K,Kp): real (W, W, nd, nd, nd)}
    cell-resolved blocks indexed by flat leg cells f = rx*wy + ry
    (rx, ry = residues mod (wx, wy), true cell = r if r <= w//2 else
    r - w), cells = (cells_x, cells_y) true-cell lists in residue order.

    Fails plainly (ValueError) if the (q1, q2) pair set is incomplete or
    the discovered cell support is not a contiguous minimum-image set --
    the inverse would be ill-posed then.
    """
    nkx, nky = nk_shape
    nq = nkx * nky
    pairs = set(vkeys.keys())
    want = {(i, j) for i in range(nq) for j in range(nq)}
    if pairs != want:
        raise ValueError(
            f"inverse fold ill-posed: npz holds {len(pairs)}/{nq * nq} "
            f"(q1,q2) pairs; missing e.g. {sorted(want - pairs)[:5]}"
        )
    triples = sorted(vkeys[(0, 0)].keys())
    for pr in pairs:
        if sorted(vkeys[pr].keys()) != triples:
            raise ValueError(f"triplet set differs at pair {pr}")

    # torus IDFT per leg axis: G[t, m] = exp(+2j pi m t / n) / n inverts
    # the fold phase exp(-2j pi (m/n) c) on residues t = c mod n.
    Ginv = {}
    for ax, n in ((0, nkx), (1, nky), (2, nkx), (3, nky)):
        t = np.arange(n)
        Ginv[ax] = np.exp(2j * np.pi * np.outer(t, t) / n) / n

    C_torus = {}
    max_imag_rel = 0.0
    rms_all = 0.0
    n_ent = 0
    for T in triples:
        nd = None
        Q = None
        for (iq1, iq2), keys in vkeys.items():
            arr = np.asarray(npz[keys[T]])
            if Q is None:
                nd = arr.shape[0]
                Q = np.empty((nq, nq, nd, nd, nd), dtype=np.complex128)
            Q[iq1, iq2] = arr
        Q = Q.reshape(nkx, nky, nkx, nky, nd, nd, nd)
        C = _dft_apply(Q, Ginv, (0, 1, 2, 3))
        scale = float(np.sqrt(np.mean(np.abs(C) ** 2)))
        max_imag_rel = max(
            max_imag_rel, float(np.abs(C.imag).max()) / max(scale, 1e-300)
        )
        C_torus[T] = np.ascontiguousarray(C.real)
        rms_all += float(np.sum(C.real**2))
        n_ent += C.size
    rms_all = np.sqrt(rms_all / n_ent)

    # Infer the true (min-imaged) cell support per transverse axis.
    weight = {0: None, 1: None}
    for T in triples:
        A = np.abs(C_torus[T])
        wx = A.sum(axis=(1, 2, 3, 4, 5, 6)) + A.sum(axis=(0, 1, 3, 4, 5, 6))
        wy = A.sum(axis=(0, 2, 3, 4, 5, 6)) + A.sum(axis=(0, 1, 2, 4, 5, 6))
        weight[0] = wx if weight[0] is None else weight[0] + wx
        weight[1] = wy if weight[1] is None else weight[1] + wy
    cells = []
    for ax, n in ((0, nkx), (1, nky)):
        nz = np.nonzero(weight[ax] > support_tol * weight[ax].max())[0]
        w_ax = len(nz)
        # expected: min-image reps of Z_w embedded at residue c % n
        expect = sorted(
            (r if r <= w_ax // 2 else r - w_ax) % n for r in range(w_ax)
        )
        if sorted(nz.tolist()) != expect:
            raise ValueError(
                f"inverse fold ill-posed: transverse axis {ax} support "
                f"residues {sorted(nz.tolist())} are not a minimum-image "
                f"set (expected {expect} for width {w_ax})"
            )
        # residue-ordered true cells: index r=0..w-1 <-> true cell
        cells.append([r if r <= w_ax // 2 else r - w_ax for r in range(w_ax)])

    # leakage on empty torus residues (must be numerically zero)
    gx = [c % nkx for c in cells[0]]
    gy = [c % nky for c in cells[1]]
    leak2 = 0.0
    tot2 = 0.0
    mask = np.zeros((nkx, nky, nkx, nky), dtype=bool)
    mask[np.ix_(gx, gy, gx, gy)] = True
    for T in triples:
        A2 = np.sum(C_torus[T] ** 2, axis=(4, 5, 6))
        tot2 += float(A2.sum())
        leak2 += float(A2[~mask].sum())
    leak_rel = np.sqrt(leak2 / max(tot2, 1e-300))

    wx, wy = len(cells[0]), len(cells[1])
    C = {}
    for T in triples:
        c = C_torus[T][np.ix_(gx, gy, gx, gy)]
        nd = c.shape[4]
        C[T] = np.ascontiguousarray(
            c.reshape(wx * wy, wx * wy, nd, nd, nd)
        )
    meta = dict(
        max_imag_rel=max_imag_rel,
        leak_rel=leak_rel,
        widths=(wx, wy),
        rms=rms_all,
    )
    return C, cells, meta


def refold(C, cells, nk_shape):
    """Forward fold: phases exp(-2j pi (m/n) c_true) per leg axis
    (se_q.py:41-46 with q = m/n)."""
    nkx, nky = nk_shape
    wx, wy = len(cells[0]), len(cells[1])
    F = {}
    for ax, (n, cs) in ((0, (nkx, cells[0])), (1, (nky, cells[1])),
                        (2, (nkx, cells[0])), (3, (nky, cells[1]))):
        m = np.arange(n)
        F[ax] = np.exp(-2j * np.pi * np.outer(m, np.asarray(cs)) / n)
    out = {}
    for T, c in C.items():
        nd = c.shape[2]
        A = c.reshape(wx, wy, wx, wy, nd, nd, nd).astype(np.complex128)
        Q = _dft_apply(A, F, (0, 1, 2, 3))
        out[T] = Q.reshape(nkx * nky, nkx * nky, nd, nd, nd)
    return out


def roundtrip_error(npz, vkeys, C, cells, nk_shape):
    Q = refold(C, cells, nk_shape)
    num2 = 0.0
    den2 = 0.0
    for (iq1, iq2), keys in vkeys.items():
        for T, key in keys.items():
            ref = np.asarray(npz[key])
            num2 += float(np.sum(np.abs(Q[T][iq1, iq2] - ref) ** 2))
            den2 += float(np.sum(np.abs(ref) ** 2))
    return np.sqrt(num2 / max(den2, 1e-300))


def write_qfold_npz(path, C, cells, nk_shape, q_diff_map):
    """Exact save_qfold format (src/quatrex/phonon/qfold.py:30-46)."""
    Q = refold(C, cells, nk_shape)
    nq = nk_shape[0] * nk_shape[1]
    out = {
        "q_diff_map": np.asarray(q_diff_map, dtype=np.int64),
        "nk_shape": np.asarray(nk_shape, dtype=np.int64),
    }
    for (I, K, Kp), q in Q.items():
        for iq1 in range(nq):
            for iq2 in range(nq):
                out[f"v|{iq1}|{iq2}|{I}|{K}|{Kp}"] = np.ascontiguousarray(
                    q[iq1, iq2].astype(np.complex128)
                )
    np.savez(str(path), **out)


# ---------------------------------------------------------------------------
# Cell-rep: group tables, S3 action, ASR
# ---------------------------------------------------------------------------


def _sub_table(cells):
    """SUB[f1, f2] = flat index of (cell_f1 - cell_f2), arithmetic mod
    the SUPERCELL widths (residue-ordered flat index f = rx*wy + ry)."""
    wx, wy = len(cells[0]), len(cells[1])
    W = wx * wy
    SUB = np.empty((W, W), dtype=np.int64)
    for f1 in range(W):
        r1x, r1y = divmod(f1, wy)
        for f2 in range(W):
            r2x, r2y = divmod(f2, wy)
            SUB[f1, f2] = ((r1x - r2x) % wx) * wy + (r1y - r2y) % wy
    return SUB


def _s3_cell_maps(SUB):
    """Per permutation: flat (t2', t3') target-index permutation pi and
    its inverse, over the W^2 x W^2 joint cell grid.

    Action derivation: legs carry (slab, cell, dof) with the external at
    cell 0; permuting legs then translating so the (new) external is at
    cell 0 gives  t2' = c[p1] - c[p0],  t3' = c[p2] - c[p0]  (mod w)."""
    W = SUB.shape[0]
    T2, T3 = np.meshgrid(np.arange(W), np.arange(W), indexing="ij")
    Z = np.zeros((W, W), dtype=np.int64)
    maps = {}
    for p in PERMS:
        slots = [Z, T2, T3]
        u = slots[p[0]]
        t2p = SUB[slots[p[1]], u]
        t3p = SUB[slots[p[2]], u]
        pi = (t2p * W + t3p).ravel()
        inv = np.empty_like(pi)
        inv[pi] = np.arange(W * W)
        maps[p] = inv
    return maps


def _s3_transform_cell(A, p, inv_map):
    """One S3 group element applied to a cell-resolved block (source
    array for triple T contributes to target triple perm_key(T, p))."""
    W = A.shape[0]
    nd = A.shape[2]
    arr = np.transpose(A, (0, 1, 2 + p[0], 2 + p[1], 2 + p[2]))
    arr = arr.reshape(W * W, nd, nd, nd)[inv_map]
    return np.ascontiguousarray(arr.reshape(W, W, nd, nd, nd))


def s3_symmetrize_cell(C, maps):
    acc = {k: np.zeros_like(v) for k, v in C.items()}
    for T, A in C.items():
        for p in PERMS:
            acc[_perm_key(T, p)] += _s3_transform_cell(A, p, maps[p])
    return {k: v / 6.0 for k, v in acc.items()}


def s3_deviation_cell(C, maps):
    dev = 0.0
    for T, A in C.items():
        for p in PERMS:
            dev = max(
                dev,
                float(
                    np.abs(
                        _s3_transform_cell(A, p, maps[p]) - C[_perm_key(T, p)]
                    ).max()
                ),
            )
    ent2 = sum(float(np.sum(v**2)) for v in C.values())
    n = sum(v.size for v in C.values())
    rms_e = np.sqrt(ent2 / n)
    return dev, dev / max(rms_e, 1e-300)


def _asr_values_cell(C, w, cls):
    """Axis-1/2 constraint values in the cell rep (cells summed too)."""
    nat = len(w)
    v = None
    for T in cls["triples"]:
        A = C[T]
        W = A.shape[0]
        nd = A.shape[2]
        if cls["axis"] == 1:
            t = A.reshape(W, W, nd, nat, 3, nd)
            c = np.einsum("j,qtajbc->tabc", w, t)
        elif cls["axis"] == 2:
            t = A.reshape(W, W, nd, nd, nat, 3)
            c = np.einsum("j,qtabjc->qabc", w, t)
        else:
            raise ValueError("use _asr0_values_cell for axis 0")
        v = c if v is None else v + c
    return v


def _asr_project_class_cell(C, w, mtot, cls):
    v = _asr_values_cell(C, w, cls)
    nat = len(w)
    T0 = cls["triples"][0]
    W = C[T0].shape[0]
    nd = C[T0].shape[2]
    denom = len(cls["triples"]) * W * mtot
    for T in cls["triples"]:
        A = C[T]
        if cls["axis"] == 1:
            t = A.reshape(W, W, nd, nat, 3, nd)
            t -= (
                w[None, None, None, :, None, None]
                * v[None, :, :, None, :, :]
                / denom
            )
        else:
            t = A.reshape(W, W, nd, nd, nat, 3)
            t -= (
                w[None, None, None, None, :, None]
                * v[:, None, :, :, None, :]
                / denom
            )


def _asr0_values_cell(C, w, cls, SUB):
    """Axis-0 (external leg) constraint values: uniform displacement of
    the external field = sum over external slabs I, cells u, atoms i with
    the stored kernel translated back to external cell 0:
    Phi(ext u) = C[sub(t2,u), sub(t3,u)]."""
    nat = len(w)
    T0 = cls["triples"][0]
    W = C[T0].shape[0]
    nd = C[T0].shape[2]
    acc = np.zeros((W, W, nd, nd, nd))
    for T in cls["triples"]:
        A = C[T]
        for u in range(W):
            g = SUB[:, u]
            acc += A[np.ix_(g, g)]
    t = acc.reshape(W, W, nat, 3, nd, nd)
    return np.einsum("i,qtiabc->qtabc", w, t)


def _class_residual_cell(C, w, mtot, cls, SUB=None):
    if cls["axis"] == 0:
        v = _asr0_values_cell(C, w, cls, SUB)
        W = C[cls["triples"][0]].shape[0]
        nslots = len(cls["triples"]) * W
    else:
        v = _asr_values_cell(C, w, cls)
        W = C[cls["triples"][0]].shape[0]
        nslots = len(cls["triples"]) * W
    rms_v = float(np.sqrt(np.mean(v**2)))
    ent2 = sum(float(np.sum(C[T] ** 2)) for T in cls["triples"])
    n = sum(C[T].size for T in cls["triples"])
    rms_e = np.sqrt(ent2 / n)
    scale = np.sqrt(nslots * mtot) * rms_e
    return rms_v / max(scale, 1e-300), rms_v, rms_e


def audit_qfold(C, cells, meta, masses, n_slabs, reach, rt_err,
                gamma_blocks=None, label="qfold"):
    w = np.sqrt(masses)
    mtot = float(masses.sum())
    SUB = _sub_table(cells)
    maps = _s3_cell_maps(SUB)
    classes = slab_classes(n_slabs, reach, C.keys())
    print(f"\n== ASR audit [{label}] (cell rep, {len(C)} triplets, "
          f"widths={meta['widths']}, cells_x={cells[0]}, cells_y={cells[1]}) ==")
    print(f"round-trip (inverse fold -> refold) rel error = {rt_err:.3e}")
    print(f"reality: max|imag|/rms of inverse-folded cells = "
          f"{meta['max_imag_rel']:.3e}")
    print(f"empty-torus-residue leakage (rel) = {meta['leak_rel']:.3e}")
    dev, rdev = s3_deviation_cell(C, maps)
    print(f"S3 symmetry (mod-w cell action): max dev = {dev:.3e} "
          f"(rel to rms entry: {rdev:.3e})")
    pooled = {}
    for cls in classes:
        if cls["vacuous"]:
            continue
        rel, rms_v, _ = _class_residual_cell(C, w, mtot, cls, SUB)
        tag = "interior" if cls["enforceable"] else "edge    "
        print(f"  axis {cls['axis']} pair {cls['pair']} supp {cls['supp']} "
              f"[{tag}]: rel={rel:.3e} (rms_v={rms_v:.3e})")
        pooled.setdefault(tag.strip(), []).append(rel)
    summary = {}
    for tag, vals in pooled.items():
        summary[tag] = (float(np.max(vals)), float(np.sqrt(np.mean(np.square(vals)))))
        print(f"  {tag}: max rel = {summary[tag][0]:.3e}")
    if gamma_blocks is not None:
        num2 = den2 = 0.0
        for T, A in C.items():
            if T not in gamma_blocks:
                continue
            g = A.sum(axis=(0, 1))
            num2 += float(np.sum((g - gamma_blocks[T]) ** 2))
            den2 += float(np.sum(gamma_blocks[T] ** 2))
        print(f"  cross-check sum_cells(C) vs fc3_blocks.hdf5: rel diff = "
              f"{np.sqrt(num2 / max(den2, 1e-300)):.3e} (informational; "
              "equal only if both files come from the same build)")
    summary["s3_rel"] = rdev
    summary["rt_err"] = rt_err
    return summary


def project_qfold(C, cells, masses, n_slabs, reach, tol=1e-13, max_sweeps=500,
                  verbose=True):
    w = np.sqrt(masses)
    mtot = float(masses.sum())
    SUB = _sub_table(cells)
    maps = _s3_cell_maps(SUB)
    all_classes = slab_classes(n_slabs, reach, C.keys())
    proj_classes = [
        c for c in all_classes if c["enforceable"] and c["axis"] in (1, 2)
    ]
    chk_classes = [c for c in all_classes if c["enforceable"]]
    orig = {k: v.copy() for k, v in C.items()}
    C = {k: v.copy() for k, v in C.items()}
    res = np.inf
    for sweep in range(1, max_sweeps + 1):
        for cls in proj_classes:
            _asr_project_class_cell(C, w, mtot, cls)
        C = s3_symmetrize_cell(C, maps)
        # post-symmetrisation the axis-0 rows equal S3 images of axis-1
        # rows, but measure all three explicitly for the stop criterion
        res = max(
            _class_residual_cell(C, w, mtot, cls, SUB)[0] for cls in chk_classes
        )
        if verbose and (sweep <= 5 or sweep % 20 == 0):
            print(f"  sweep {sweep:3d}: enforceable rel residual = {res:.3e}")
        if res < tol:
            break
    num = np.sqrt(sum(float(np.sum((C[k] - orig[k]) ** 2)) for k in orig))
    den = np.sqrt(sum(float(np.sum(orig[k] ** 2)) for k in orig))
    corr = num / max(den, 1e-300)
    print(f"  converged after {sweep} sweeps: residual={res:.3e}, "
          f"||dPhi||/||Phi|| = {corr:.6e}")
    return C, corr, res


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _parse_masses(s, nd):
    masses = np.array([float(x) for x in s.split(",") if x])
    if 3 * len(masses) != nd:
        raise SystemExit(
            f"--masses has {len(masses)} atoms but blocks have nd={nd} "
            f"(need {nd // 3} atoms)"
        )
    return masses


def _infer_reach(keys):
    return max(
        max(abs(k[0] - k[1]), abs(k[0] - k[2]), abs(k[1] - k[2])) for k in keys
    )


def cmd_audit(args):
    blocks, block_sizes, units, max_imag = load_fc3_hdf5(args.fc3)
    n_slabs = len(block_sizes)
    nd = int(block_sizes[0])
    masses = _parse_masses(args.masses, nd)
    reach = _infer_reach(blocks.keys())
    audit_gamma(blocks, masses, n_slabs, reach, max_imag,
                label=str(args.fc3))
    if args.qfold:
        npz, vkeys, qdm, nk_shape = load_qfold_npz(args.qfold)
        C, cells, meta = invert_fold(npz, vkeys, nk_shape)
        rt = roundtrip_error(npz, vkeys, C, cells, nk_shape)
        n_slabs_q = max(max(T) for T in C) + 1
        reach_q = _infer_reach(C.keys())
        audit_qfold(C, cells, meta, masses, n_slabs_q, reach_q, rt,
                    gamma_blocks=blocks, label=str(args.qfold))
    else:
        print("\n(no --qfold given: q-folded vertex audit skipped)")


def cmd_project(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    blocks, block_sizes, units, max_imag = load_fc3_hdf5(args.fc3)
    n_slabs = len(block_sizes)
    nd = int(block_sizes[0])
    masses = _parse_masses(args.masses, nd)
    reach = _infer_reach(blocks.keys())

    print("### fc3_blocks.hdf5: pre-projection audit")
    audit_gamma(blocks, masses, n_slabs, reach, max_imag, label="pre")
    print("\n### fc3_blocks.hdf5: alternating projection")
    proj, corr, res = project_gamma(blocks, masses, n_slabs, reach,
                                    tol=args.tol, max_sweeps=args.max_sweeps)
    if corr > args.max_corr:
        raise SystemExit(
            f"ABORT: fc3 correction ||dPhi||/||Phi|| = {corr:.3e} > "
            f"{args.max_corr} -- refusing to write (this is no longer a "
            "small ASR cleanup)"
        )
    write_fc3_hdf5(proj, block_sizes, out / "fc3_blocks.hdf5", units=units)
    print(f"wrote {out / 'fc3_blocks.hdf5'}")
    print("\n### fc3_blocks.hdf5: post-projection audit (re-read from disk)")
    blocks2, bs2, un2, mi2 = load_fc3_hdf5(out / "fc3_blocks.hdf5")
    audit_gamma(blocks2, masses, len(bs2), reach, mi2, label="post")
    print(f"\nfc3_blocks.hdf5 applied correction: ||dPhi||/||Phi|| = {corr:.6e}")

    if args.qfold:
        print("\n### qfold_vertices.npz: inverse fold")
        npz, vkeys, qdm, nk_shape = load_qfold_npz(args.qfold)
        C, cells, meta = invert_fold(npz, vkeys, nk_shape)
        rt = roundtrip_error(npz, vkeys, C, cells, nk_shape)
        print(f"round-trip rel error = {rt:.3e}")
        if rt > 1e-12:
            raise SystemExit(
                f"ABORT: inverse-fold round-trip error {rt:.3e} > 1e-12 -- "
                "fold convention mismatch, refusing to project"
            )
        n_slabs_q = max(max(T) for T in C) + 1
        reach_q = _infer_reach(C.keys())
        print("\n### qfold_vertices.npz: pre-projection audit")
        audit_qfold(C, cells, meta, masses, n_slabs_q, reach_q, rt,
                    gamma_blocks=blocks, label="pre")
        print("\n### qfold_vertices.npz: alternating projection (cell rep)")
        Cp, corr_q, res_q = project_qfold(C, cells, masses, n_slabs_q,
                                          reach_q, tol=args.tol,
                                          max_sweeps=args.max_sweeps)
        if corr_q > args.max_corr:
            raise SystemExit(
                f"ABORT: qfold correction {corr_q:.3e} > {args.max_corr}"
            )
        write_qfold_npz(out / "qfold_vertices.npz", Cp, cells, nk_shape, qdm)
        print(f"wrote {out / 'qfold_vertices.npz'}")
        print("\n### qfold_vertices.npz: post-projection audit "
              "(re-read from disk)")
        npz2, vkeys2, qdm2, nk2 = load_qfold_npz(out / "qfold_vertices.npz")
        C2, cells2, meta2 = invert_fold(npz2, vkeys2, nk2)
        rt2 = roundtrip_error(npz2, vkeys2, C2, cells2, nk2)
        audit_qfold(C2, cells2, meta2, masses, n_slabs_q, reach_q, rt2,
                    gamma_blocks=blocks2, label="post")
        print(f"\nqfold_vertices.npz applied correction: "
              f"||dPhi||/||Phi|| = {corr_q:.6e}")
    else:
        print("\n(no --qfold given: only fc3_blocks.hdf5 was projected; "
              "run with --qfold on the cluster for the production npz)")


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------


def _fixture_supercell(widths, nz, nat):
    """Hand-built supercell mapping (transport z, transverse x/y),
    matching build_supercell_mapping's min-imaged transverse cell_frac
    (separable.py:137-143) and raw transport slab_indices."""
    wx, wy = widths
    reps_x = [r if r <= wx // 2 else r - wx for r in range(wx)]
    reps_y = [r if r <= wy // 2 else r - wy for r in range(wy)]
    atoms = []  # (prim, cx, cy, cz)
    for cx in reps_x:
        for cy in reps_y:
            for cz in range(nz):
                for p in range(nat):
                    atoms.append((p, cx, cy, cz))
    n_super = len(atoms)
    prim_indices = np.array([a[0] for a in atoms], dtype=int)
    cell_frac = np.array([[a[1], a[2], a[3]] for a in atoms], dtype=float)
    slab_indices = np.array([a[3] for a in atoms], dtype=int)
    index = {a: i for i, a in enumerate(atoms)}
    return atoms, index, prim_indices, cell_frac, slab_indices


def _translation_perms(atoms, index, widths, nz):
    """Supercell-translation atom permutations (group Z_wx x Z_wy x Z_nz)."""
    wx, wy = widths

    def wrap(c, w):
        r = c % w
        return r if r <= w // 2 else r - w

    perms = []
    for ux in range(wx):
        for uy in range(wy):
            for uz in range(nz):
                perm = np.empty(len(atoms), dtype=int)
                for i, (p, cx, cy, cz) in enumerate(atoms):
                    tgt = (p, wrap(cx + ux, wx), wrap(cy + uy, wy),
                           (cz + uz) % nz)
                    perm[i] = index[tgt]
                perms.append(perm)
    return perms


def _sym_translate(G, dof_perms):
    """S3-symmetrize + translation-average a supercell (D,D,D) tensor."""
    S = np.zeros_like(G)
    for p in PERMS:
        S += np.transpose(G, p)
    S /= 6.0
    T = np.zeros_like(S)
    for dp in dof_perms:
        T += S[np.ix_(dp, dp, dp)]
    T /= len(dof_perms)
    return T


def _s3_symmetrize_super(G):
    S = np.zeros_like(G)
    for p in PERMS:
        S += np.transpose(G, p)
    return S / 6.0


def _transport_mask(atoms, nz):
    """Dof-level mask selecting entries whose transport offsets unroll
    consistently to a finite-reach Z-crystal kernel (pairwise min-image
    |d| <= nz//2, matching the device truncation in
    fc3_device.build_device_fc3_blocks). A ring-periodic random kernel
    carries O(1) weight on the wrap couplings (e.g. offsets (-1,+1) on
    nz=3) that the device build DROPS; physical kernels decay, so their
    truncation is lossless. Permutation- and translation-invariant for
    odd nz."""
    hw = nz // 2
    z = np.array([a[3] for a in atoms])
    d = (z[None, :] - z[:, None]) % nz
    d = np.where(d > hw, d - nz, d)  # (ns, ns) min-imaged offsets
    keep = (
        (np.abs(d)[:, :, None] <= hw)
        & (np.abs(d)[:, None, :] <= hw)
        & (np.abs(d[:, :, None] - d[:, None, :]) <= hw)
    )
    for ax in range(3):
        keep = np.repeat(keep, 3, axis=ax)
    return keep


def _supercell_asr_residual(G, w_super, mtot_super):
    D = G.shape[0]
    ns = D // 3
    rms_e = float(np.sqrt(np.mean(G**2)))
    worst = 0.0
    for axis in range(3):
        if axis == 0:
            v = np.einsum("j,jabc->abc", w_super, G.reshape(ns, 3, D, D))
        elif axis == 1:
            v = np.einsum("j,ajbc->abc", w_super, G.reshape(D, ns, 3, D))
        else:
            v = np.einsum("j,abjc->abc", w_super, G.reshape(D, D, ns, 3))
        rel = float(np.sqrt(np.mean(v**2))) / (
            np.sqrt(mtot_super) * max(rms_e, 1e-300)
        )
        worst = max(worst, rel)
    return worst


def _supercell_asr_project(G, w_super, mtot_super, mask, tol=1e-14,
                           max_sweeps=500):
    """Alternating projection onto {ring-ASR per axis} & {S3 symmetry} &
    {finite-reach transport support}. Translation invariance of the
    start is preserved by every step (all three are translation-
    covariant), so it is not re-imposed in the loop."""
    D = G.shape[0]
    ns = D // 3
    for _ in range(max_sweeps):
        t = G.reshape(ns, 3, D, D)
        v = np.einsum("j,jabc->abc", w_super, t)
        t -= w_super[:, None, None, None] * v[None] / mtot_super
        t = G.reshape(D, ns, 3, D)
        v = np.einsum("j,ajbc->abc", w_super, t)
        t -= w_super[None, :, None, None] * v[:, None, :, :] / mtot_super
        t = G.reshape(D, D, ns, 3)
        v = np.einsum("j,abjc->abc", w_super, t)
        t -= w_super[None, None, :, None] * v[:, :, None, :] / mtot_super
        G = _s3_symmetrize_super(G)
        G *= mask
        if _supercell_asr_residual(G, w_super, mtot_super) < tol:
            break
    return G


def _m_stacked_from_full(M_full, atoms, index, nat):
    """M_stacked[a*D:(a+1)*D, :] = M_full[dof(ref atom a), :, :] --
    the build_realspace_fc3_matrices layout (separable.py:160-213)."""
    D = M_full.shape[0]
    nd = 3 * nat
    M_stacked = np.zeros((nd * D, D))
    for p in range(nat):
        s_ref = index[(p, 0, 0, 0)]
        for al in range(3):
            a = 3 * p + al
            M_stacked[a * D:(a + 1) * D, :] = M_full[3 * s_ref + al]
    return M_stacked


def _cell_reference_blocks(M_full, atoms, index, widths, nz, nat, n_slabs,
                           cells):
    """Directly grouped cell-resolved device blocks (the ground truth the
    inverse fold must reproduce): C[(I,K,Kp)][f2,f3][a, 3p2+al, 3p3+be]."""
    wx, wy = widths
    flat = {}
    for fx, cx in enumerate(cells[0]):
        for fy, cy in enumerate(cells[1]):
            flat[(cx, cy)] = fx * wy + fy
    W = wx * wy
    nd = 3 * nat

    def mimg(d):
        d = d % nz
        return d - nz if d > nz // 2 else d

    offs = {}
    for i2, (p2, cx2, cy2, cz2) in enumerate(atoms):
        d2 = mimg(cz2)
        f2 = flat[(cx2, cy2)]
        for i3, (p3, cx3, cy3, cz3) in enumerate(atoms):
            d3 = mimg(cz3)
            f3 = flat[(cx3, cy3)]
            key = (d2, d3)
            if key not in offs:
                offs[key] = np.zeros((W, W, nd, nd, nd))
            blk = offs[key]
            for p1 in range(nat):
                s1 = index[(p1, 0, 0, 0)]
                blk[f2, f3, 3 * p1:3 * p1 + 3, 3 * p2:3 * p2 + 3,
                    3 * p3:3 * p3 + 3] += M_full[
                        3 * s1:3 * s1 + 3, 3 * i2:3 * i2 + 3,
                        3 * i3:3 * i3 + 3]
    half = nz // 2
    C = {}
    for I in range(n_slabs):
        for K in range(n_slabs):
            for Kp in range(n_slabs):
                dk, dkp = K - I, Kp - I
                if max(abs(dk), abs(dkp), abs(K - Kp)) > half:
                    continue
                if (dk, dkp) in offs:
                    C[(I, K, Kp)] = offs[(dk, dkp)]
    return C


def _selftest_one(widths, nz, nk, nat, masses, eps, workdir, rng):
    from phonon.solver.fc3_device import build_device_fc3_blocks
    from phonon.solver.se_q import _build_folded_vertices
    from phonon_inputs.separable import build_q_diff_map
    from phonon_inputs.quatrex_writer import write_fc3_blocks
    from quatrex.phonon.qfold import save_qfold

    n_slabs = nz
    nd = 3 * nat
    tag = f"w{widths[0]}x{widths[1]}_nz{nz}_nk{nk}"
    print(f"\n---- selftest fixture {tag} "
          f"(nat={nat}, masses={masses.tolist()}, eps={eps:g}) ----")
    atoms, index, prim_indices, cell_frac, slab_indices = _fixture_supercell(
        widths, nz, nat)
    D = 3 * len(atoms)
    dof_perms = [
        np.repeat(3 * ap, 3) + np.tile(np.arange(3), len(ap))
        for ap in _translation_perms(atoms, index, widths, nz)
    ]
    m_super = np.array([masses[a[0]] for a in atoms])
    w_super = np.sqrt(m_super)
    mtot_super = float(m_super.sum())

    mask = _transport_mask(atoms, nz)
    base = mask * _sym_translate(rng.standard_normal((D, D, D)), dof_perms)
    base = _supercell_asr_project(base, w_super, mtot_super, mask)
    res_base = _supercell_asr_residual(base, w_super, mtot_super)
    viol = mask * _sym_translate(rng.standard_normal((D, D, D)), dof_perms)
    viol *= np.linalg.norm(base) / np.linalg.norm(viol)
    M_full = base + eps * viol
    print(f"supercell base ASR residual = {res_base:.3e}; injected "
          f"S3-symmetric finite-reach violation at eps = {eps:g}")

    M_stacked = _m_stacked_from_full(M_full, atoms, index, nat)

    # --- Gamma device blocks through the production builder ---
    gamma = build_device_fc3_blocks(
        M_stacked, prim_indices, slab_indices, nat, n_slabs)
    gamma = {k: v.copy() for k, v in gamma.items()}
    fc3_path = workdir / f"fc3_blocks_{tag}.hdf5"
    write_fc3_blocks(gamma, np.array([nd] * n_slabs), fc3_path)

    # --- q-folded vertices through the PRODUCTION fold path ---
    q_1d = np.arange(nk) / nk
    q_points = [(qa, qb) for qa in q_1d for qb in q_1d]
    q_diff_map = build_q_diff_map(nk, nk)
    vertices = _build_folded_vertices(
        M_stacked, prim_indices, cell_frac, slab_indices, nat, n_slabs,
        nk * nk, q_points, q_diff_map, "z")
    qfold_path = workdir / f"qfold_{tag}.npz"
    save_qfold(qfold_path, vertices, q_diff_map, (nk, nk))

    ok = True

    # kshift convention: the config shift 1/2 - 1/(2 nk) makes the
    # engine's monkhorst_pack land exactly on the m/nk fractions the
    # fold used -- kshift never enters the fold itself.
    try:
        from quatrex.grid.kpoints import monkhorst_pack

        shift = 0.5 - 0.5 / nk
        mesh = np.asarray(monkhorst_pack((nk, nk), (shift, shift)))
        want = np.stack(
            np.meshgrid(q_1d, q_1d, indexing="ij"), axis=-1
        ).reshape(-1, 2)
        kdev = float(np.abs(mesh - want).max())
        print(f"kshift convention: monkhorst_pack(({nk},{nk}), "
              f"shift={shift:g}) vs m/n mesh: max dev = {kdev:.3e}")
        ok &= kdev < 1e-12
    except Exception as e:  # pragma: no cover
        print(f"kshift convention check SKIPPED (import failed: {e})")

    # (a) inverse fold recovers the directly-grouped input blocks
    blocks_g, bs, units, mi = load_fc3_hdf5(fc3_path)
    masses_chk = _parse_masses(",".join(str(m) for m in masses), nd)
    npz, vkeys, qdm, nk_shape = load_qfold_npz(qfold_path)
    C, cells, meta = invert_fold(npz, vkeys, nk_shape)
    C_ref = _cell_reference_blocks(M_full, atoms, index, widths, nz, nat,
                                   n_slabs, cells)
    num2 = den2 = 0.0
    assert sorted(C.keys()) == sorted(C_ref.keys()), (
        sorted(C.keys()), sorted(C_ref.keys()))
    for T in C:
        num2 += float(np.sum((C[T] - C_ref[T]) ** 2))
        den2 += float(np.sum(C_ref[T] ** 2))
    inv_err = np.sqrt(num2 / max(den2, 1e-300))
    rt = roundtrip_error(npz, vkeys, C, cells, nk_shape)
    print(f"(a) inverse fold vs direct real-space grouping: rel err = "
          f"{inv_err:.3e}; round-trip rel err = {rt:.3e}")
    ok &= inv_err < 1e-12 and rt < 1e-12

    # (b) audits show the injected violation, projection removes it
    reach = _infer_reach(blocks_g.keys())
    sg = audit_gamma(blocks_g, masses_chk, n_slabs, reach, mi,
                     label=f"gamma {tag} pre")
    sq = audit_qfold(C, cells, meta, masses_chk, n_slabs, reach, rt,
                     gamma_blocks=blocks_g, label=f"qfold {tag} pre")
    pre_int = sg.get("interior", (0.0, 0.0))[0]
    ok &= pre_int > eps / 100  # the violation must be visible
    ok &= sg["s3_rel"] < 1e-12 and sq["s3_rel"] < 1e-12

    print(f"\n-- project gamma [{tag}] --")
    pg, corr_g, res_g = project_gamma(blocks_g, masses_chk, n_slabs, reach)
    print(f"-- project qfold [{tag}] --")
    pq, corr_q, res_q = project_qfold(C, cells, masses_chk, n_slabs, reach)

    sg2 = audit_gamma(pg, masses_chk, n_slabs, reach, 0.0,
                      label=f"gamma {tag} post")
    sq2 = audit_qfold(pq, cells, meta, masses_chk, n_slabs, reach, rt,
                      label=f"qfold {tag} post")
    post_int_g = sg2.get("interior", (0.0, 0.0))[0]
    post_int_q = sq2.get("interior", (0.0, 0.0))[0]
    ok &= post_int_g < 1e-12 and post_int_q < 1e-12
    ok &= sg2["s3_rel"] < 1e-12 and sq2["s3_rel"] < 1e-12
    corr_ok = (eps / 50 < corr_g < 5 * eps) and (eps / 50 < corr_q < 5 * eps)
    ok &= corr_ok
    print(f"\n[{tag}] correction norms: gamma {corr_g:.3e}, qfold {corr_q:.3e} "
          f"(injected eps = {eps:g}) -> {'OK' if corr_ok else 'MISMATCH'}")

    # write/re-read round trip of the projected npz through my writer
    out_npz = workdir / f"qfold_{tag}_proj.npz"
    write_qfold_npz(out_npz, pq, cells, nk_shape, qdm)
    npz2, vkeys2, qdm2, nk2 = load_qfold_npz(out_npz)
    C2, cells2, meta2 = invert_fold(npz2, vkeys2, nk2)
    d2 = max(
        float(np.abs(C2[T] - pq[T]).max()) for T in pq
    ) / max(float(np.sqrt(np.mean(pq[(0, 0, 0)] ** 2))), 1e-300)
    print(f"[{tag}] projected npz write/re-read/invert consistency: "
          f"{d2:.3e}")
    ok &= d2 < 1e-10 and np.array_equal(qdm2, qdm) and tuple(nk2) == (nk, nk)

    print(f"---- fixture {tag}: {'PASS' if ok else 'FAIL'} ----")
    return ok


def cmd_selftest(args):
    workdir = Path(args.workdir) if args.workdir else None
    if workdir is None:
        import tempfile

        workdir = Path(tempfile.mkdtemp(prefix="asr_selftest_"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"selftest workdir: {workdir}")
    rng = np.random.default_rng(7)
    masses = np.array([95.95, 32.06])
    t0 = time.time()
    ok = True
    # odd width == nk (full torus) and even width < nk (support subset +
    # even-width half-shell identification, the MoS2 [4,4] < nk=5 case)
    ok &= _selftest_one((3, 3), 3, 3, 2, masses, 3e-3, workdir, rng)
    ok &= _selftest_one((2, 2), 3, 3, 2, masses, 3e-3, workdir, rng)
    print(f"\nselftest total time: {time.time() - t0:.1f} s")
    if not ok:
        raise SystemExit("SELFTEST FAIL")
    print("SELFTEST PASS")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("audit", help="measure ASR/S3/reality residuals")
    pa.add_argument("--fc3", required=True)
    pa.add_argument("--qfold", default=None)
    pa.add_argument("--masses", default=DEFAULT_MASSES)
    pa.set_defaults(func=cmd_audit)

    pp = sub.add_parser("project", help="minimal S3-symmetric ASR projection")
    pp.add_argument("--fc3", required=True)
    pp.add_argument("--qfold", default=None)
    pp.add_argument("--out", required=True)
    pp.add_argument("--masses", default=DEFAULT_MASSES)
    pp.add_argument("--tol", type=float, default=1e-13)
    pp.add_argument("--max-sweeps", type=int, default=500)
    pp.add_argument("--max-corr", type=float, default=0.05)
    pp.set_defaults(func=cmd_project)

    ps = sub.add_parser("selftest", help="synthetic end-to-end validation")
    ps.add_argument("--workdir", default=None)
    ps.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
