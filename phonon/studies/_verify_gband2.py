"""Unit verification of RGF second_offdiagonals against a dense reference.

Random block-tridiagonal A (stabilised), random skew-hermitian
block-tridiagonal Sigma^{<,>}, OBC blocks on both ends. Checks:

  1. legacy path vs flag=True: the block-tridiagonal outputs are
     bit-identical;
  2. the (i, i+2) / (i+2, i) blocks of X^{<,>} = A^-1 Sigma A^-dagger
     match the dense reference to machine precision.

Memory-light (small blocks, small stack).
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")
from qttools import sparse  # noqa: E402
from qttools.comm import comm  # noqa: E402

_mpi = {"all_gather": "device_mpi", "all_to_all": "device_mpi", "all_reduce": "device_mpi",
        "bcast": "device_mpi", "send_recv": "device_mpi"}
comm.configure(block_comm_size=1, block_comm_config=dict(_mpi),
               stack_comm_config=dict(_mpi), override=True)

from qttools.datastructures.dsdbcoo import DSDBCOO  # noqa: E402
from qttools.greens_function_solver.rgf import RGF  # noqa: E402
from qttools.greens_function_solver.solver import OBCBlocks  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(7)
    nb, b, ns = 6, 5, 3          # blocks, block size, stack
    N = nb * b
    block_sizes = np.full(nb, b)

    def rand_c(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    # Block-tridiagonal A, diagonally dominated for a safe inverse.
    A = np.zeros((ns, N, N), complex)
    for i in range(nb):
        s = slice(i * b, (i + 1) * b)
        A[:, s, s] = rand_c(ns, b, b) + 8.0 * np.eye(b)
        if i + 1 < nb:
            sj = slice((i + 1) * b, (i + 2) * b)
            A[:, s, sj] = rand_c(ns, b, b)
            A[:, sj, s] = rand_c(ns, b, b)

    # Skew-hermitian block-tridiagonal Sigma^{<,>}.
    def rand_skewherm():
        S = np.zeros((ns, N, N), complex)
        for i in range(nb):
            s = slice(i * b, (i + 1) * b)
            d = rand_c(ns, b, b)
            S[:, s, s] = 0.5 * (d - d.conj().swapaxes(-2, -1))
            if i + 1 < nb:
                sj = slice((i + 1) * b, (i + 2) * b)
                o = rand_c(ns, b, b)
                S[:, s, sj] = o
                S[:, sj, s] = -o.conj().swapaxes(-2, -1)
        return S

    SL, SG = rand_skewherm(), rand_skewherm()

    # OBC blocks (retarded on both ends, lesser/greater sources).
    obc = OBCBlocks(num_blocks=nb)
    obc_r0, obc_rN = 0.05 * rand_c(ns, b, b), 0.05 * rand_c(ns, b, b)
    obc.retarded[0], obc.retarded[nb - 1] = obc_r0, obc_rN
    ol0, olN = rand_skewherm(), rand_skewherm()
    obc.lesser[0] = ol0[:, :b, :b]
    obc.lesser[nb - 1] = olN[:, -b:, -b:]
    obc.greater[0] = ol0[:, b:2*b, b:2*b] - ol0[:, b:2*b, b:2*b].conj().swapaxes(-2, -1)
    obc.greater[nb - 1] = olN[:, :b, :b]

    # Dense reference with OBC folded in.
    A_eff = A.copy()
    A_eff[:, :b, :b] -= obc_r0
    A_eff[:, -b:, -b:] -= obc_rN
    SL_eff = SL.copy()
    SL_eff[:, :b, :b] += obc.lesser[0]
    SL_eff[:, -b:, -b:] += obc.lesser[nb - 1]
    SG_eff = SG.copy()
    SG_eff[:, :b, :b] += obc.greater[0]
    SG_eff[:, -b:, -b:] += obc.greater[nb - 1]
    X = np.linalg.inv(A_eff)
    Xd = X.conj().swapaxes(-2, -1)
    XL_ref = X @ SL_eff @ Xd
    XG_ref = X @ SG_eff @ Xd

    # DSDBCOO objects: A/Sigma on the tridiagonal pattern; outputs on the
    # pentadiagonal pattern (so the d2 writes land).
    def block_band_pattern(band):
        rows, cols = [], []
        for i in range(nb):
            for j in range(nb):
                if abs(i - j) <= band:
                    r = np.arange(i * b, (i + 1) * b)
                    c = np.arange(j * b, (j + 1) * b)
                    rr, cc = np.meshgrid(r, c, indexing="ij")
                    rows.append(rr.ravel())
                    cols.append(cc.ravel())
        return sparse.coo_matrix(
            (np.ones(sum(len(r) for r in rows)),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(N, N),
        )

    def to_dsdb(dense, band):
        m = DSDBCOO.from_sparray(
            block_band_pattern(band).astype(np.complex128), block_sizes,
            global_stack_shape=(ns,))
        for i in range(nb):
            for j in range(nb):
                if abs(i - j) <= band:
                    m.blocks[i, j] = dense[:, i*b:(i+1)*b, j*b:(j+1)*b]
        return m

    a_d = to_dsdb(A, 1)
    sl_d = to_dsdb(SL, 1)
    sg_d = to_dsdb(SG, 1)

    solver = RGF(max_batch_size=2)

    def run(flag, band):
        xl = to_dsdb(np.zeros_like(A), band)
        xg = to_dsdb(np.zeros_like(A), band)
        xr = to_dsdb(np.zeros_like(A), band)
        xl.data[:] = 0; xg.data[:] = 0; xr.data[:] = 0
        solver.selected_solve(
            a=a_d, sigma_lesser=sl_d, sigma_greater=sg_d,
            out=(xl, xg, xr), obc_blocks=obc, return_retarded=True,
            second_offdiagonals=flag)
        return xl, xg

    xl0, xg0 = run(False, 1)
    xl2, xg2 = run(True, 2)

    # 1) tridiagonal outputs bit-identical.
    ok_bit = True
    for i in range(nb):
        for j in range(nb):
            if abs(i - j) <= 1:
                for m0, m2 in ((xl0, xl2), (xg0, xg2)):
                    if not np.array_equal(np.asarray(m0.blocks[i, j]),
                                          np.asarray(m2.blocks[i, j])):
                        ok_bit = False
                        print(f"BIT-DIFF at block ({i},{j})")
    print(f"tridiagonal bit-identity: {'PASS' if ok_bit else 'FAIL'}")

    # 2) d2 blocks vs dense reference.
    worst = 0.0
    for i in range(nb - 2):
        j = i + 2
        for tag, m, ref in (("xl", xl2, XL_ref), ("xg", xg2, XG_ref)):
            for (r, c) in ((i, j), (j, i)):
                got = np.asarray(m.blocks[r, c])
                want = ref[:, r*b:(r+1)*b, c*b:(c+1)*b]
                rel = (np.linalg.norm(got - want)
                       / max(np.linalg.norm(want), 1e-300))
                worst = max(worst, rel)
                if rel > 1e-10:
                    print(f"  {tag} ({r},{c}): rel={rel:.2e}  FAIL")
    print(f"second off-diagonals vs dense: worst rel = {worst:.2e} "
          f"{'PASS' if worst < 1e-10 else 'FAIL'}")

    # 3) sanity: tridiagonal blocks also match dense.
    worst_t = 0.0
    for i in range(nb):
        for j in range(nb):
            if abs(i - j) <= 1:
                got = np.asarray(xl2.blocks[i, j])
                want = XL_ref[:, i*b:(i+1)*b, j*b:(j+1)*b]
                worst_t = max(worst_t, np.linalg.norm(got - want)
                              / max(np.linalg.norm(want), 1e-300))
    print(f"tridiagonal vs dense: worst rel = {worst_t:.2e} "
          f"{'PASS' if worst_t < 1e-10 else 'FAIL'}")
    return 0 if ok_bit and worst < 1e-10 and worst_t < 1e-10 else 1


if __name__ == "__main__":
    sys.exit(main())
