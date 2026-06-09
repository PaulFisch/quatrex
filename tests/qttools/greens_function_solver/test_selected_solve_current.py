# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Tests the boundary (lead) current of ``selected_solve(return_current=True)``.

This covers what ``test_selected_solve`` does not: the Meir-Wingreen contact
current with NONZERO OBC lesser/greater blocks. It guards two fixes:

* The RGF lead current must be **batching-invariant**. It is computed per
  energy batch (sliced to ``stack_slice``); computing it once after the batch
  loop used only the last batch's diagonal blocks against the full-energy OBC
  -> a shape error / wrong value whenever ``nfreq > max_batch_size``.
* RGF and Inv must **agree**. Both ADD the lesser/greater contact self-energy
  (the physical ``Sigma^<_tot = Sigma^<_scatter + Sigma^<_lead`` in-scattering
  convention); the Inv previously subtracted it.
"""

import numpy as np
import pytest

from qttools import sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBCOO, DSDBCSR
from qttools.greens_function_solver import RGF, Inv
from qttools.greens_function_solver.solver import OBCBlocks


@pytest.fixture(autouse=True, scope="module")
def _configure_comm():
    cfg = {k: "device_mpi" for k in ("all_to_all", "all_gather", "all_reduce", "bcast")}
    comm.configure(
        block_comm_size=1, block_comm_config=cfg, stack_comm_config=cfg, override=True
    )


def _build(dsdb, nfreq, bs, rng):
    """A diagonally-dominant block-tridiagonal system, distinct per stack point,
    with physical (skew-Hermitian) lesser/greater OBC at the two leads."""
    nb = len(bs)
    n = int(bs.sum())
    off = np.hstack(([0], np.cumsum(bs)))

    def cb(m, k):
        return rng.standard_normal((m, k)) + 1j * rng.standard_normal((m, k))

    A = np.zeros((n, n), complex)
    for i in range(nb):
        bi = slice(off[i], off[i + 1])
        A[bi, bi] = cb(bs[i], bs[i]) + 4 * nb * np.eye(bs[i])
        if i + 1 < nb:
            bj = slice(off[i + 1], off[i + 2])
            A[bi, bj] = cb(bs[i], bs[i + 1])
            A[bj, bi] = cb(bs[i + 1], bs[i])
    Bl = A - A.conj().T
    Bg = 2.0 * A - (2.0 * A).conj().T

    gss = (nfreq,)
    Am = dsdb.from_sparray(sparse.coo_matrix(A).astype(complex), bs, gss)
    Blm = dsdb.from_sparray(sparse.coo_matrix(Bl).astype(complex), bs, gss)
    Bgm = dsdb.from_sparray(sparse.coo_matrix(Bg).astype(complex), bs, gss)
    fac = (1.0 + 0.17 * np.arange(nfreq))[:, None]  # make every stack point distinct
    for M in (Am, Blm, Bgm):
        M.data[:] = M.data * fac

    obc = OBCBlocks(num_blocks=nb)
    for idx in (0, nb - 1):
        m = int(bs[idx])
        r = rng.standard_normal((nfreq, m, m)) + 1j * rng.standard_normal((nfreq, m, m))
        l = rng.standard_normal((nfreq, m, m)) + 1j * rng.standard_normal((nfreq, m, m))
        g = rng.standard_normal((nfreq, m, m)) + 1j * rng.standard_normal((nfreq, m, m))
        obc.retarded[idx] = xp.asarray(r)
        obc.lesser[idx] = xp.asarray(l - l.conj().swapaxes(-2, -1))
        obc.greater[idx] = xp.asarray(g - g.conj().swapaxes(-2, -1))
    return Am, Blm, Bgm, obc


def _current(solver_cls, dsdb, max_batch_size, nfreq, bs):
    rng = np.random.default_rng(0)  # identical system regardless of batching
    Am, Blm, Bgm, obc = _build(dsdb, nfreq, bs, rng)
    Xl = dsdb.zeros_like(Am)
    Xg = dsdb.zeros_like(Am)
    cur = solver_cls(max_batch_size=max_batch_size).selected_solve(
        Am, Blm, Bgm, obc_blocks=obc, out=[Xl, Xg], return_current=True
    )
    return np.asarray(cur)


@pytest.mark.parametrize("dsdb", [DSDBCOO, DSDBCSR])
def test_lead_current_batching_invariant_and_solver_agreement(dsdb):
    nfreq, bs = 7, np.array([3, 3, 3, 3])
    # nfreq=7 with max_batch_size=2 -> 4 batches (>1); =nfreq -> a single batch.
    rgf_nb = _current(RGF, dsdb, nfreq, nfreq, bs)
    rgf_b = _current(RGF, dsdb, 2, nfreq, bs)
    inv_nb = _current(Inv, dsdb, nfreq, nfreq, bs)
    inv_b = _current(Inv, dsdb, 2, nfreq, bs)

    # (1) batching-invariance (the RGF fix; Inv as a control). equal_nan: the
    # Inv leaves internal interfaces NaN -- those NaN positions must match, the
    # finite lead columns must agree to tolerance.
    assert xp.allclose(rgf_nb, rgf_b, atol=1e-12, rtol=0, equal_nan=True), "RGF current depends on batching"
    assert xp.allclose(inv_nb, inv_b, atol=1e-12, rtol=0, equal_nan=True), "Inv current depends on batching"

    # (2) RGF == Inv on the lead columns (the OBC-sign fix). Inv leaves the
    # internal interfaces NaN, so only the two leads are the shared contract.
    leads = [0, -1]
    assert xp.allclose(
        rgf_b[..., leads], inv_b[..., leads], atol=1e-10, rtol=0
    ), "RGF and Inv lead currents disagree (OBC lesser/greater sign?)"
