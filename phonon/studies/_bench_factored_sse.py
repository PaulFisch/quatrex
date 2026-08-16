"""Micro-benchmark: dense coupled-q contraction vs the legacy and the current
factored (tensor-decomposed) kernels, on the real sifilm ns3_nk9 shapes.

Usage:
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from quatrex.phonon.bubble import phi_perms, ring_contract_pre
from quatrex.phonon.bubble_factored import (
    _VARIANTS,
    GramTables,
    contract_tau_q_factored,
)
from quatrex.phonon.vertex_factors import VertexFactors


def build_fixture(rng, nd, nq, R):
    offsets = np.array([-1, 0, 1], dtype=np.int64)
    UB = (rng.standard_normal((len(offsets), nq, nd, R))
          + 1j * rng.standard_normal((len(offsets), nq, nd, R)))
    return VertexFactors(
        D=rng.standard_normal((nd, R)),
        lambdas=np.sort(np.abs(rng.standard_normal(R)))[::-1],
        offsets=offsets, UB=UB, UC=UB,
        q_diff_map=None, nk_shape=None, ansatz="INDSCAL", meta={},
    )


def build_pair_index(nslabs, offsets=(-1, 0, 1), g_band=1):
    """(I, J) -> [(K1, K2, K1p, K2p)], the PRODUCTION enumeration.

    Mirrors ``SigmaPhononPhonon._phi_pair_index``: the FC3 support of the
    factored vertex is the full offset product, and the only other constraint is
    the G band, applied to each line separately. (An earlier version of this
    bench also imposed |K1-K2| <= 1, which production does not -- that couples
    the two lines and would destroy the product structure the kernel exploits.)
    """
    phi = {(I, I + d1, I + d2)
           for I in range(nslabs) for d1 in offsets for d2 in offsets
           if 0 <= I + d1 < nslabs and 0 <= I + d2 < nslabs}
    quads = {}
    for (I, K1, K2) in sorted(phi):
        for J in range(max(0, I - 1), min(nslabs, I + 2)):
            for K1p in range(max(0, K1 - g_band), min(nslabs, K1 + g_band + 1)):
                for K2p in range(max(0, K2 - g_band), min(nslabs, K2 + g_band + 1)):
                    if (J, K2p, K1p) in phi:
                        quads.setdefault((I, J), []).append((K1, K2, K1p, K2p))
    return quads


def contract_legacy(quads_by_pair, block_sizes, q_diff_map, q_own, nq, g_dicts,
                    Dt, UB, UC, off_pos, n_tau, dtype):
    """The superseded kernel: per-quad Hadamard + explicit O(N_q^2) q'-sum."""
    grams = GramTables(g_dicts, UB, UC, off_pos, 0, n_tau, np, True)
    R = Dt.shape[1]
    DtT = Dt.T.copy()
    res = {}
    for (I, J), quads in quads_by_pair.items():
        bs_i, bs_j = int(block_sizes[I]), int(block_sizes[J])
        out_l = np.zeros((n_tau, nq, bs_i, bs_j), dtype=dtype)
        out_g = np.zeros((n_tau, nq, bs_i, bs_j), dtype=dtype)
        h_l = np.zeros((q_own, n_tau, R, R), dtype=dtype)
        h_g = np.zeros((q_own, n_tau, R, R), dtype=dtype)
        for (K1, K2, K1p, K2p) in quads:
            pa = {v: grams.get(v, (K1, K1p), K1 - I, K1p - J, role="a")
                  for v in _VARIANTS}
            pb = {v: grams.get(v, (K2, K2p), K2 - I, K2p - J, role="b")
                  for v in _VARIANTS}
            pb_l_gr = pb["l"] + pb["gr"]
            pb_g_lr = pb["g"] + pb["lr"]
            for iq_ext in range(q_own):
                idx = q_diff_map[iq_ext]
                h_l[iq_ext] += np.einsum("qwrs,qwrs->wrs", pa["l"], pb_l_gr[idx])
                h_l[iq_ext] += np.einsum("qwrs,qwrs->wrs", pa["gr"], pb["l"][idx])
                h_g[iq_ext] += np.einsum("qwrs,qwrs->wrs", pa["g"], pb_g_lr[idx])
                h_g[iq_ext] += np.einsum("qwrs,qwrs->wrs", pa["lr"], pb["g"][idx])
        for iq_ext in range(q_own):
            out_l[:, iq_ext] = (Dt @ h_l[iq_ext]) @ DtT
            out_g[:, iq_ext] = (Dt @ h_g[iq_ext]) @ DtT
        res[(I, J)] = (out_l, out_g)
    return res


def contract_dense(quads_by_pair, vf, q_diff_map, q_own, nq, nd, n_tau, g_dicts):
    """The dense per-task ring loop (vertex reconstruction is offline).

    The pre-permuted vertex pair is cached on the momenta and the transport
    offsets, which is what makes it repeat across the block index I. Keying it on
    id() -- as production did -- never hits and materialises one permuted copy
    per task (~8 GB here), which would handicap the baseline.
    """
    recon, perms = {}, {}

    def phi(iq1, iq2, d_k, d_kp):
        key = (iq1, iq2, d_k, d_kp)
        if key not in recon:
            recon[key] = vf.reconstruct_block(iq1, iq2, d_k, d_kp)
        return recon[key]

    def pre_perm(iqp, iq2, d1, d2, e1, e2):
        key = (iqp, iq2, d1, d2, e1, e2)
        if key not in perms:
            pl = phi(iqp, iq2, d1, d2)
            pr = phi(iq2, iqp, e1, e2)
            perms[key] = phi_perms(np.conj(pl), pr, np)
        return perms[key]

    # Warm the vertex + permutation caches OUTSIDE the timer: in production the
    # vertex is reconstructed once, offline, not per SCBA iteration.
    for (I, J), quads in quads_by_pair.items():
        for iq_ext in range(q_own):
            for iqp in range(nq):
                iq2 = int(q_diff_map[iq_ext, iqp])
                for (K1, K2, K1p, K2p) in quads:
                    pre_perm(iqp, iq2, K1 - I, K2 - I, K2p - J, K1p - J)

    tic = time.perf_counter()
    out = {}
    for (I, J), quads in quads_by_pair.items():
        out_l = np.zeros((n_tau, nq, nd, nd), complex)
        out_g = np.zeros((n_tau, nq, nd, nd), complex)
        for iq_ext in range(q_own):
            for iqp in range(nq):
                iq2 = int(q_diff_map[iq_ext, iqp])
                for (K1, K2, K1p, K2p) in quads:
                    PL, PR, nI, bK2, nJ = pre_perm(
                        iqp, iq2, K1 - I, K2 - I, K2p - J, K1p - J)
                    la, lb = (K1, K1p), (K2, K2p)

                    def ring(ga, gb):
                        return ring_contract_pre(
                            PL, PR, nI, bK2, nJ, g_dicts[ga][la][:, iqp],
                            g_dicts[gb][lb][:, iq2], np)

                    out_l[:, iq_ext] += (
                        ring("l", "l") + ring("l", "gr") + ring("gr", "l"))
                    out_g[:, iq_ext] += (
                        ring("g", "g") + ring("g", "lr") + ring("lr", "g"))
        out[(I, J)] = (out_l, out_g)

    return out, time.perf_counter() - tic


def _max_rel_err(ref, got, q_own):
    worst = 0.0
    for key in ref:
        for i in (0, 1):
            a, b = ref[key][i][:, :q_own], got[key][i][:, :q_own]
            worst = max(worst, np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranks", default="8,16,32,64,128")
    p.add_argument("--ntau", type=int, default=60)
    p.add_argument("--nk", type=int, default=9)
    p.add_argument("--nslabs", type=int, default=3)
    p.add_argument("--nd", type=int, default=6, help="DOFs per slab (2 Si x 3)")
    p.add_argument("--qown", type=int, default=0,
                   help="external q evaluated; 0 = all N_q (total work)")
    p.add_argument("--skip-legacy", action="store_true",
                   help="the legacy kernel is O(N_q^2) and very slow at high R")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    nq = args.nk * args.nk
    q_own = args.qown or nq
    nd, nslabs, n_tau = args.nd, args.nslabs, args.ntau
    rng = np.random.default_rng(0)
    q_diff_map = np.array([[
        ((ae // args.nk - be // args.nk) % args.nk) * args.nk
        + ((ae % args.nk) - (be % args.nk)) % args.nk
        for be in range(nq)] for ae in range(nq)])

    quads_by_pair = build_pair_index(nslabs)
    n_quads = sum(len(v) for v in quads_by_pair.values())
    block_sizes = np.array([nd] * nslabs)

    links = {(K, Kp) for K in range(nslabs)
             for Kp in range(max(0, K - 1), min(nslabs, K + 2))}
    g_dicts = {v: {lk: (rng.standard_normal((n_tau, nq, nd, nd))
                        + 1j * rng.standard_normal((n_tau, nq, nd, nd)))
                   for lk in links}
               for v in ("l", "g", "lr", "gr")}

    print(f"shapes: nslabs={nslabs} nd={nd} nq={nq} n_tau={n_tau} q_own={q_own} "
          f"pairs={len(quads_by_pair)} quads={n_quads}", flush=True)
    print(f"{'R':>4} {'dense':>9} {'legacy':>9} {'new':>9} "
          f"{'new/dense':>10} {'new/legacy':>11}  parity", flush=True)

    for rank in (int(r) for r in args.ranks.split(",") if r):
        vf = build_fixture(rng, nd, nq, rank)
        pos = vf.offset_index()
        dt = vf.D * vf.lambdas[None, :]

        dense_out, t_dense = contract_dense(
            quads_by_pair, vf, q_diff_map, q_own, nq, nd, n_tau, g_dicts)

        t_legacy = float("nan")
        if not args.skip_legacy:
            tic = time.perf_counter()
            contract_legacy(quads_by_pair, block_sizes, q_diff_map, q_own, nq,
                            g_dicts, dt, vf.UB, vf.UC, pos, n_tau, np.complex128)
            t_legacy = time.perf_counter() - tic

        tic = time.perf_counter()
        new_out = contract_tau_q_factored(
            quads_by_pair, block_sizes, (args.nk, args.nk), 0, q_own, nq,
            g_dicts, dt, vf.UB, vf.UC, pos, 0, n_tau, np, True, np.complex128)
        t_new = time.perf_counter() - tic

        parity = ""
        if args.verify:
            err = _max_rel_err(dense_out, new_out, q_own)
            parity = f"  {err:.1e}"
            assert err < 1e-9, f"R={rank}: new vs dense rel err {err:.2e}"

        print(f"{rank:>4} {t_dense:8.2f}s {t_legacy:8.2f}s {t_new:8.2f}s "
              f"{t_dense / t_new:9.1f}x {t_legacy / t_new:10.1f}x{parity}",
              flush=True)


if __name__ == "__main__":
    main()
