#\!/usr/bin/env python
"""Tridiagonal (quatrex RGF) vs dense on a genuine phonon Dyson matrix.

A(z2) = z2 I - H_D - Sigma_leads, block-tridiagonal, built from a mass-weighted
3D diatomic-chain dynamical matrix with Sancho-Rubio open-boundary leads on the
end blocks. Compares the selected (block-tridiagonal) retarded Green's function
from: (1) numpy dense inv * mask, (2) qttools Inv, (3) qttools RGF.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon", _REPO / "src"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")
import numpy as np
from qttools import sparse, xp
from qttools.datastructures import DSDBCOO
from qttools.greens_function_solver import RGF, Inv
from phonon.solver.leads import sancho_rubio
from qttools.comm import comm
_cfg = {"all_to_all":"device_mpi","all_gather":"device_mpi","all_reduce":"device_mpi","bcast":"device_mpi"}
comm.configure(block_comm_size=1, block_comm_config=_cfg, stack_comm_config=_cfg, override=True)

rng = np.random.default_rng(0)
nb, n_slabs = 3, 6
K = rng.standard_normal((nb, nb)); K = K @ K.T + nb*np.eye(nb)
C = 0.3 * rng.standard_normal((nb, nb))
H_00 = (2*K).astype(complex)
H_01 = (-C).astype(complex)
block_sizes = np.full(n_slabs, nb)
N = nb*n_slabs
off = np.concatenate([[0], np.cumsum(block_sizes)])

def build_A(z2):
    A = np.zeros((N, N), dtype=complex)
    for l in range(n_slabs):
        s = slice(l*nb, (l+1)*nb)
        A[s, s] = z2*np.eye(nb) - H_00
        if l < n_slabs-1:
            s2 = slice((l+1)*nb, (l+2)*nb)
            A[s, s2] = -H_01
            A[s2, s] = -H_01.conj().T
    A[0:nb, 0:nb]   -= H_01.conj().T @ sancho_rubio(z2, H_00, H_01) @ H_01
    A[-nb:, -nb:]   -= H_01 @ sancho_rubio(z2, H_00, H_01.conj().T) @ H_01.conj().T
    return A

mask = np.zeros((N, N), bool)
for i in range(n_slabs):
    for j in range(n_slabs):
        if abs(i-j) <= 1:
            mask[off[i]:off[i+1], off[j]:off[j+1]] = True

def solve_qttools(SolverCls, A):
    ds = DSDBCOO.from_sparray(sparse.coo_matrix(A), block_sizes, (1,))
    g = DSDBCOO.zeros_like(ds)
    SolverCls(max_batch_size=1).selected_inv(ds, out=g)
    return np.asarray(g.to_dense())[0]

print(f"{'freq[THz]':>9} {'|Inv-dense|':>13} {'|RGF-dense|':>13} {'|RGF-Inv|':>12} {'|A|':>10}")
maxerr = 0.0
for f in (1.0, 3.0, 7.5, 12.0, 16.0):
    z2 = (f + 1j*0.05)**2
    A = build_A(z2)
    Gd = np.linalg.inv(A) * mask
    Gi = solve_qttools(Inv, A) * mask
    Gr = solve_qttools(RGF, A) * mask
    e_id = np.max(np.abs(Gi-Gd)); e_rd = np.max(np.abs(Gr-Gd)); e_ri = np.max(np.abs(Gr-Gi))
    maxerr = max(maxerr, e_rd, e_id)
    print(f"{f:>9.2f} {e_id:>13.2e} {e_rd:>13.2e} {e_ri:>12.2e} {np.abs(Gd).max():>10.3e}")
print(f"\nMAX selected-GR discrepancy across all freqs: {maxerr:.2e}")
print("PASS" if maxerr < 1e-9 else "FAIL")
