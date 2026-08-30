# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Small algebraic tests for the host dense half of the phonon JVP."""

from types import SimpleNamespace

import numpy as np

from quatrex.core.phonon_jvp import PhononJVP


def _buffer(data):
    return SimpleNamespace(data=np.asarray(data, dtype=np.complex128))


def test_prepare_reconstructs_each_frequency_q_batch_element():
    """Frequency and q are independent batch axes in the dense Dyson half.

    This catches both tempting but wrong broadcasts: using the first q-point
    dynamical matrix at every q, or interleaving q before frequency when the
    sparse buffers use C-order ``(frequency, q, nnz)`` storage.
    """
    nf, nq, n = 2, 3, 2
    freqs = np.array([1.5, 2.5])
    rows = np.repeat(np.arange(n), n)
    cols = np.tile(np.arange(n), n)
    nnz = rows.size

    # A genuinely q-dependent Hermitian dynamical matrix, with the leading
    # singleton used by matrix inputs in production.
    d_q = np.empty((1, nq, n, n), dtype=np.complex128)
    for q in range(nq):
        d_q[0, q] = np.array(
            [[0.10 + 0.03 * q, 0.01j * (q + 1)],
             [-0.01j * (q + 1), 0.25 + 0.02 * q]])

    z2 = np.broadcast_to(freqs[:, None] ** 2, (nf, nq))
    obc_r = np.empty((nf, nq, n, n), dtype=np.complex128)
    obc_l = np.empty_like(obc_r)
    obc_g = np.empty_like(obc_r)
    for iw in range(nf):
        for q in range(nq):
            obc_r[iw, q] = np.diag(
                [-0.02j * (iw + 1), -0.03j * (q + 1)])
            obc_l[iw, q] = 1j * np.diag(
                [0.04 + 0.01 * iw, 0.05 + 0.01 * q])
            obc_g[iw, q] = 1j * np.diag(
                [0.08 + 0.01 * q, 0.09 + 0.01 * iw])
    a = z2[..., None, None] * np.eye(n) - d_q - obc_r
    gr = np.linalg.inv(a)
    ga = gr.conj().swapaxes(-2, -1)

    src_l = np.empty_like(gr)
    src_g = np.empty_like(gr)
    for iw in range(nf):
        for q in range(nq):
            src_l[iw, q] = 1j * np.diag(
                [0.2 + 0.01 * iw, 0.3 + 0.02 * q])
            src_g[iw, q] = 1j * np.diag(
                [0.5 + 0.03 * q, 0.7 + 0.01 * iw])
    gl = gr @ (src_l + obc_l) @ ga
    gg = gr @ (src_g + obc_g) @ ga

    def flat(m):
        return m[..., rows, cols]

    zeros = np.zeros((nf, nq, nnz), dtype=np.complex128)
    data = SimpleNamespace(
        sigma_lesser_prev=_buffer(flat(src_l)),
        sigma_greater_prev=_buffer(flat(src_g)),
        sigma_retarded_hermitian_prev=_buffer(zeros),
        sigma_lesser=_buffer(zeros),
        sigma_greater=_buffer(zeros),
        sigma_retarded_hermitian=_buffer(zeros),
        g_lesser=_buffer(flat(gl)),
        g_greater=_buffer(flat(gg)),
    )
    solver = SimpleNamespace(
        local_frequencies=freqs,
        eta=0.0,
        _ir_floor_diag=None,
        dynamical_matrix=SimpleNamespace(blocks={(0, 0): d_q}),
        obc_blocks=SimpleNamespace(
            # In a one-block device each sole entry is already the sum of the
            # left and right reservoirs; [0] and [-1] are the same object.
            retarded=[obc_r], lesser=[obc_l], greater=[obc_g]),
    )

    # Construct only the state used by prepare(); __init__ needs a complete
    # SCBA object and is exercised by the production validation instead.
    jvp = PhononJVP.__new__(PhononJVP)
    jvp._data = data
    jvp._solver = solver
    jvp._stack_shape = (nf, nq)
    jvp._n_batch = nf * nq
    jvp._n_local = nf * nq
    jvp._nnz = nnz
    jvp._rows = rows
    jvp._cols = cols
    jvp._block_offsets = np.array([0, n])
    jvp._block_sizes = np.array([n])
    jvp._nb = 1
    jvp._N = n
    jvp._bt_mask = np.ones(nnz, dtype=bool)
    jvp._g_mask = np.ones(nnz, dtype=bool)
    jvp.recon_check_tol = 1e-12

    recon = jvp.prepare()

    assert recon < 1e-13
    np.testing.assert_allclose(
        jvp._GR.reshape(nf, nq, n, n), gr, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        jvp._GL.reshape(nf, nq, n, n), gl, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        jvp._GG.reshape(nf, nq, n, n), gg, rtol=1e-13, atol=1e-13)
