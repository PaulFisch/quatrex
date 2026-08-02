"""Unit tests for the quartic (SCP) loop static self-energy."""
import numpy as np
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from quatrex.phonon.static_self_energy import (  # noqa: E402
    CONVERSION_THZ2, sigma_loop_blocks)


def test_sigma_loop_matches_dense_einsum():
    rng = np.random.default_rng(5)
    n_blocks, n_dof = 3, 4
    N = n_blocks * n_dof
    # symmetric quartic device tensor with +-1-slab support
    fc4 = {}
    dense = np.zeros((N, N, N, N))
    for I in range(n_blocks):
        for J in (I,):
            for K in range(max(0, I - 1), min(n_blocks, I + 2)):
                for Kp in range(max(0, K - 1), min(n_blocks, K + 2)):
                    blk = rng.standard_normal((n_dof,) * 4)
                    fc4[(I, J, K, Kp)] = blk
                    dense[I*n_dof:(I+1)*n_dof, J*n_dof:(J+1)*n_dof,
                          K*n_dof:(K+1)*n_dof, Kp*n_dof:(Kp+1)*n_dof] += blk
    uu = rng.standard_normal((N, N))
    uu = 0.5 * (uu + uu.T)
    got = sigma_loop_blocks(fc4, uu, n_blocks, n_dof)
    ref = CONVERSION_THZ2 * 0.5 * np.einsum("abcd,cd->ab", dense, uu)
    ref = 0.5 * (ref + ref.T)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-10)
    # Hermitian real by construction
    np.testing.assert_allclose(got, got.T, atol=0)


def test_sigma_loop_stiffens_for_positive_quartic():
    # A positive-diagonal quartic with positive <uu> must give a
    # positive-semidefinite (stiffening) Sigma_L on the diagonal.
    n_blocks, n_dof = 1, 3
    fc4 = {(0, 0, 0, 0): np.einsum(
        "ab,cd->abcd", np.eye(n_dof), np.eye(n_dof))}
    uu = np.diag([0.1, 0.2, 0.3])
    sig = sigma_loop_blocks(fc4, uu, n_blocks, n_dof)
    evals = np.linalg.eigvalsh(sig)
    assert evals.min() > 0
