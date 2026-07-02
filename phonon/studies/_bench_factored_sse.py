"""Micro-benchmark: dense coupled-q vertex-pair contraction vs the factored
(tensor-decomposed) kernel, on the real sifilm ns3_nk9 per-rank shapes.

The dense path is the exact per-task 3+3 ``ring_contract_pre`` loop of
``SigmaPhononPhonon._contract_dense_q``; the factored path is
``quatrex.phonon.bubble_factored.contract_tau_q_factored``. Both consume the
SAME synthetic factors (the dense vertices are reconstructed from them), so
outputs must agree to rtol ~1e-9 -- the benchmark doubles as a large-shape
parity check.

Campaign gate (plan TD-E): factored >= 10x dense at R = 64 on ns3 shapes.

Usage:
    OMP_NUM_THREADS=1 python phonon/studies/_bench_factored_sse.py \
        [--ranks 8,16,32,64,128] [--ntau 60] [--nk 9] [--nslabs 3] [--qown 3]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from quatrex.phonon.bubble import phi_perms, ring_contract_pre
from quatrex.phonon.bubble_factored import contract_tau_q_factored
from quatrex.phonon.vertex_factors import VertexFactors


def build_fixture(rng, nslabs, nd, nq, R):
    offsets = np.array(sorted({min(d, nslabs - 1 - abs(d) * 0)  # [-1, 0, 1]
                               for d in (-1, 0, 1)}), dtype=np.int64)
    UB = (rng.standard_normal((len(offsets), nq, nd, R))
          + 1j * rng.standard_normal((len(offsets), nq, nd, R)))
    vf = VertexFactors(
        D=rng.standard_normal((nd, R)),
        lambdas=np.sort(np.abs(rng.standard_normal(R)))[::-1],
        offsets=offsets, UB=UB, UC=UB,
        q_diff_map=None, nk_shape=None, ansatz="INDSCAL", meta={},
    )
    return vf


def build_pair_index(nslabs):
    """(I, J) -> [(K1, K2, K1p, K2p)] with the production NN windows."""
    quads = {}
    for I in range(nslabs):
        for K1 in range(max(0, I - 1), min(nslabs, I + 2)):
            for K2 in range(max(0, I - 1), min(nslabs, I + 2)):
                if abs(K1 - K2) > 1:
                    continue
                for J in range(max(0, I - 1), min(nslabs, I + 2)):
                    for K1p in range(max(0, K1 - 1), min(nslabs, K1 + 2)):
                        for K2p in range(max(0, K2 - 1), min(nslabs, K2 + 2)):
                            if abs(K2p - J) > 1 or abs(K1p - J) > 1:
                                continue
                            if abs(K1p - K2p) > 1:
                                continue
                            quads.setdefault((I, J), []).append(
                                (K1, K2, K1p, K2p))
    return quads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranks", default="8,16,32,64,128")
    p.add_argument("--ntau", type=int, default=60,
                   help="per-rank tau slice (n_fft=241 / stack=4)")
    p.add_argument("--nk", type=int, default=9)
    p.add_argument("--nslabs", type=int, default=3)
    p.add_argument("--nd", type=int, default=6, help="DOFs per slab (2 Si x 3)")
    p.add_argument("--qown", type=int, default=3,
                   help="external q per rank (nq / q_comm_size = 81/27)")
    p.add_argument("--verify", action="store_true",
                   help="also check dense == factored on the outputs")
    a = p.parse_args()

    nq = a.nk * a.nk
    nd, nslabs, n_tau = a.nd, a.nslabs, a.ntau
    rng = np.random.default_rng(0)
    q_diff_map = np.array([[  # 2D Gamma-centered mesh difference map
        ((ae // a.nk - be // a.nk) % a.nk) * a.nk
        + ((ae % a.nk) - (be % a.nk)) % a.nk
        for be in range(nq)] for ae in range(nq)])

    quads_by_pair = build_pair_index(nslabs)
    n_quads = sum(len(v) for v in quads_by_pair.values())
    block_sizes = np.array([nd] * nslabs)

    # Shared random G bands, q-flattened tau-domain layout (tau, nq, b, b).
    links = {(K, Kp) for K in range(nslabs)
             for Kp in range(max(0, K - 1), min(nslabs, K + 2))}
    g_dicts = {v: {lk: (rng.standard_normal((n_tau, nq, nd, nd))
                        + 1j * rng.standard_normal((n_tau, nq, nd, nd)))
                   for lk in links}
               for v in ("l", "g", "lr", "gr")}

    print(f"shapes: nslabs={nslabs} nd={nd} nq={nq} n_tau={n_tau} "
          f"q_own={a.qown} quads={n_quads}", flush=True)

    for R in (int(r) for r in a.ranks.split(",") if r):
        vf = build_fixture(rng, nslabs, nd, nq, R)
        pos = vf.offset_index()

        # ---- dense path: reconstruct blocks + per-task fold loop ---------
        # (vertex reconstruction is offline in production -- not timed)
        recon = {}

        def phi(iq1, iq2, dK, dKp):
            key = (iq1, iq2, dK, dKp)
            if key not in recon:
                recon[key] = vf.reconstruct_block(iq1, iq2, dK, dKp)
            return recon[key]

        qtasks = {}
        for (I, J), quads in quads_by_pair.items():
            for iq_ext in range(a.qown):
                for iqp in range(nq):
                    iq2 = int(q_diff_map[iq_ext, iqp])
                    for (K1, K2, K1p, K2p) in quads:
                        pl = phi(iqp, iq2, K1 - I, K2 - I)
                        pr = phi(iq2, iqp, K2p - J, K1p - J)
                        qtasks.setdefault((I, J), []).append(
                            (iq_ext, iqp, iq2, K1, K1p, K2, K2p)
                            + phi_perms(np.conj(pl), pr, np))

        t0 = time.perf_counter()
        dense_out = {}
        for (I, J), tasks in qtasks.items():
            out_l = np.zeros((n_tau, nq, nd, nd), complex)
            out_g = np.zeros((n_tau, nq, nd, nd), complex)
            for (iq_ext, iqp, iq2, K1, K1p, K2, K2p, *pre) in tasks:
                PL, PR, nI, bK2, nJ = pre
                la, lb = (K1, K1p), (K2, K2p)
                out_l[:, iq_ext] += (
                    ring_contract_pre(PL, PR, nI, bK2, nJ,
                                      g_dicts["l"][la][:, iqp],
                                      g_dicts["l"][lb][:, iq2], np)
                    + ring_contract_pre(PL, PR, nI, bK2, nJ,
                                        g_dicts["l"][la][:, iqp],
                                        g_dicts["gr"][lb][:, iq2], np)
                    + ring_contract_pre(PL, PR, nI, bK2, nJ,
                                        g_dicts["gr"][la][:, iqp],
                                        g_dicts["l"][lb][:, iq2], np))
                out_g[:, iq_ext] += (
                    ring_contract_pre(PL, PR, nI, bK2, nJ,
                                      g_dicts["g"][la][:, iqp],
                                      g_dicts["g"][lb][:, iq2], np)
                    + ring_contract_pre(PL, PR, nI, bK2, nJ,
                                        g_dicts["g"][la][:, iqp],
                                        g_dicts["lr"][lb][:, iq2], np)
                    + ring_contract_pre(PL, PR, nI, bK2, nJ,
                                        g_dicts["lr"][la][:, iqp],
                                        g_dicts["g"][lb][:, iq2], np))
            dense_out[(I, J)] = (out_l, out_g)
        t_dense = time.perf_counter() - t0

        # ---- factored path ------------------------------------------------
        Dt = vf.D * vf.lambdas[None, :]
        t0 = time.perf_counter()
        fact_out = contract_tau_q_factored(
            quads_by_pair, block_sizes, q_diff_map, 0, a.qown, nq,
            g_dicts, Dt, vf.UB, vf.UC, pos, 0, n_tau, np,
            shared_legs=True, dtype=np.complex128)
        t_fact = time.perf_counter() - t0

        line = (f"R={R:4d}: dense {t_dense:7.2f}s  factored {t_fact:7.2f}s  "
                f"speedup {t_dense / t_fact:6.1f}x")
        if a.verify:
            worst = 0.0
            for key, (dl, dg) in dense_out.items():
                fl, fg = fact_out[key]
                den = max(np.abs(dl).max(), 1e-300)
                worst = max(worst, np.abs(fl - dl).max() / den,
                            np.abs(fg - dg).max() / max(np.abs(dg).max(),
                                                        1e-300))
            line += f"  parity max rel err {worst:.2e}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
