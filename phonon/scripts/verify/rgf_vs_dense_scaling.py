"""Solver-cost scaling vs number of transport cells (= n_slabs = num blocks).

Selected inversion of a block-tridiagonal phonon-Dyson-like matrix with a fixed
block size and a growing number of blocks, comparing:
  - block-tridiagonal RGF (textbook forward/backward recursion) -> O(num_blocks)
  - dense numpy inversion (full N x N)                           -> O(num_blocks^3)
Verifies RGF == dense on the diagonal blocks. This is the "how does cost scale
with n_slabs" answer (Part 1c): block-tridiagonal RGF is linear in the number of
cells while the dense reference is cubic. (qttools RGF==dense correctness to
machine precision is separately established in CLAUDE.md F6.)

Block size is chosen large (BLAS-bound) so the asymptotic scaling is visible and
not Python/overhead dominated. Single process; the multi-rank energy-parallel
scaling is dist_scaling.py. Writes phonon/scripts/out/rgf_vs_dense_scaling.csv.
"""
import csv
import time
from pathlib import Path

import numpy as np

BS = 192                          # block size (~ a slab's DOF); large -> BLAS-bound
NBLKS = [2, 4, 8, 16, 24, 32]
OUT = Path("/usr/scratch/mont-fort11/pfischill/quatrex/phonon/scripts/out")
OUT.mkdir(parents=True, exist_ok=True)


def build_blocks(nblk, bs, seed=0):
    """Diagonally-dominant complex block-tridiagonal: diag D[i], off-diag E[i]."""
    rng = np.random.default_rng(seed)
    D = [rng.standard_normal((bs, bs)) + 1j * rng.standard_normal((bs, bs))
         + 20.0 * np.eye(bs) for _ in range(nblk)]
    E = [0.1 * (rng.standard_normal((bs, bs)) + 1j * rng.standard_normal((bs, bs)))
         for _ in range(nblk - 1)]   # E[i] couples block i <-> i+1
    return [d + d.conj().T for d in D], E


def assemble_dense(D, E, bs):
    nblk = len(D)
    N = nblk * bs
    A = np.zeros((N, N), dtype=complex)
    for i in range(nblk):
        A[i*bs:(i+1)*bs, i*bs:(i+1)*bs] = D[i]
    for i in range(nblk - 1):
        A[i*bs:(i+1)*bs, (i+1)*bs:(i+2)*bs] = E[i]
        A[(i+1)*bs:(i+2)*bs, i*bs:(i+1)*bs] = E[i].conj().T
    return A


def rgf_diag(D, E):
    """Textbook block-tridiagonal RGF: returns the diagonal blocks of A^{-1}.
    Left-connected gL recursion + backward dressing -> O(num_blocks) block ops."""
    n = len(D)
    gL = [None] * n
    gL[0] = np.linalg.inv(D[0])
    for i in range(1, n):
        sig = E[i-1].conj().T @ gL[i-1] @ E[i-1]
        gL[i] = np.linalg.inv(D[i] - sig)
    G = [None] * n
    G[n-1] = gL[n-1]
    for i in range(n-2, -1, -1):
        G[i] = gL[i] + gL[i] @ E[i] @ G[i+1] @ E[i].conj().T @ gL[i]
    return G


rows = []
for nblk in NBLKS:
    D, E = build_blocks(nblk, BS)
    # RGF
    t0 = time.time(); G_rgf = rgf_diag(D, E); t_rgf = time.time() - t0
    # dense
    A = assemble_dense(D, E, BS)
    t0 = time.time(); A_inv = np.linalg.inv(A); t_dense = time.time() - t0
    # correctness on diagonal blocks
    err = max(float(np.max(np.abs(G_rgf[b] - A_inv[b*BS:(b+1)*BS, b*BS:(b+1)*BS])))
              for b in range(nblk))
    rows.append(dict(num_blocks=nblk, N=nblk*BS, t_rgf=t_rgf, t_dense=t_dense,
                     speedup=t_dense / t_rgf, rgf_vs_dense_maxerr=err))
    print(f"nblk={nblk:>3} N={nblk*BS:>5}  RGF={t_rgf:.4f}s  dense={t_dense:.4f}s  "
          f"dense/RGF={t_dense/t_rgf:.1f}x  err={err:.2e}", flush=True)

with open(OUT / "rgf_vs_dense_scaling.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"[done] wrote {OUT / 'rgf_vs_dense_scaling.csv'}", flush=True)
