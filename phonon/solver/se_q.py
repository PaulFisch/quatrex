"""q-dependent dense phonon-phonon self-energy driver.

Computes the FC3 Fourier transform Φ(q, q', q'') on a Γ-centered q-mesh
and contracts with the q-resolved Green's functions. Parallelisable over
external q-points via ``ThreadPoolExecutor`` or a ``forkserver`` pool.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context

import numpy as np

from phonon_inputs.constants import HBAR_SI


def _se_worker_iq(args):
    """Process one chunk of iq_ext values using FFT-domain Phi contraction."""
    (iq_ext_list, GL_fft, GG_fft, Phi_all, q_diff_map,
     n_freq, n_dof, n_kpts, n_fft, freq_sl_start, freq_sl_stop,
     prefactor, qp_batch) = args

    freq_sl = slice(freq_sl_start, freq_sl_stop)
    n_out = len(iq_ext_list)
    nd = n_dof
    nd2 = nd * nd
    sig_l = np.zeros((n_out, n_freq, nd, nd), dtype=complex)
    sig_g = np.zeros((n_out, n_freq, nd, nd), dtype=complex)

    iq_prime_all = np.arange(n_kpts)

    for idx, iq_ext in enumerate(iq_ext_list):
        iq_diff_arr = q_diff_map[iq_ext]
        PL_all = Phi_all[iq_prime_all, iq_diff_arr]
        PR_all = Phi_all[iq_diff_arr, iq_prime_all]

        for G_fft, sig_out in [(GL_fft, sig_l), (GG_fft, sig_g)]:
            S_hat = np.zeros((n_fft, nd, nd), dtype=complex)

            for p0 in range(0, n_kpts, qp_batch):
                p1 = min(p0 + qp_batch, n_kpts)
                B = p1 - p0
                ps = slice(p0, p1)
                iq_d = iq_diff_arr[ps]

                PL = PL_all[ps]
                PR = PR_all[ps]
                G1 = G_fft[p0:p1]
                G2 = G_fft[iq_d]

                PL_flat = PL.reshape(B, nd2, nd)
                A = np.matmul(PL_flat[:, None, :, :], G2)
                A = A.reshape(B, n_fft, nd, nd, nd)
                A = A.transpose(0, 1, 2, 4, 3)
                B5 = np.matmul(A, G1[:, :, None, :, :])
                B5_flat = B5.reshape(B, n_fft * nd, nd2)
                PR_flat = PR.reshape(B, nd, nd2)
                S_chunk = np.matmul(B5_flat, PR_flat.transpose(0, 2, 1))
                S_hat += S_chunk.sum(axis=0).reshape(n_fft, nd, nd)

            sig_out[idx] = prefactor * np.fft.ifft(S_hat, axis=0)[freq_sl]

    return iq_ext_list, sig_l, sig_g


def compute_phph_self_energy_q_dense(
    G_lesser_q, G_greater_q, M_stacked, T_all_q, q_diff_map,
    nat_prim, n_kpts, omega_grid_thz, dw_thz, n_workers=None,
):
    """Compute q-dependent phonon-phonon self-energy.

    Returns ``(Σ^<, Σ^>)`` only; Σ^R is rebuilt from the mixed pair in
    the SCBA loop via :func:`phonon.solver.retarded.build_retarded`.

    Returns
    -------
    Sigma_lesser, Sigma_greater : (n_kpts, n_freq, n_dof, n_dof)
    """
    n_freq = len(omega_grid_thz)
    n_dof = nat_prim * 3
    dim_t = M_stacked.shape[1]

    n_fft = 2 * n_freq - 1
    mid = n_freq // 2
    freq_sl = slice(mid, mid + n_freq)

    prefactor = 0.5j * HBAR_SI * dw_thz / (2 * np.pi) / n_kpts

    G_l_clean = G_lesser_q.copy()
    G_l_clean[:, mid] = 0.0
    G_g_clean = G_greater_q.copy()
    G_g_clean[:, mid] = 0.0

    G_padded = np.zeros((n_kpts, n_fft, n_dof, n_dof), dtype=complex)
    G_padded[:, :n_freq] = G_l_clean
    GL_fft = np.fft.fft(G_padded, axis=1)
    G_padded[:, :] = 0
    G_padded[:, :n_freq] = G_g_clean
    GG_fft = np.fft.fft(G_padded, axis=1)
    del G_padded, G_l_clean, G_g_clean

    M_blocks = M_stacked.reshape(n_dof, dim_t, dim_t)
    T_arr = np.array(T_all_q)
    TM = np.einsum('qci,aij->qacj', T_arr, M_blocks)
    T_arr_H = T_arr.conj().transpose(0, 2, 1).copy()
    Phi_all = np.einsum('qacj,rjd->qracd', TM, T_arr_H)
    del TM, T_arr_H, T_arr, M_blocks

    target_bytes = 16 * 1024 * 1024
    bytes_per_qp = n_fft * n_dof**3 * 16
    qp_batch = max(1, min(n_kpts, target_bytes // max(bytes_per_qp, 1)))

    if n_workers is None:
        n_workers = min(n_kpts, os.cpu_count() or 1)
    n_workers = min(n_workers, n_kpts)
    if n_kpts > 1:
        n_workers = min(n_workers, max(1, n_kpts // 2))

    common = (GL_fft, GG_fft, Phi_all, q_diff_map,
              n_freq, n_dof, n_kpts, n_fft,
              freq_sl.start, freq_sl.stop, prefactor, qp_batch)

    if n_workers <= 1:
        _, sig_l, sig_g = _se_worker_iq(
            (list(range(n_kpts)), *common))
    else:
        chunks = [[] for _ in range(n_workers)]
        for iq in range(n_kpts):
            chunks[iq % n_workers].append(iq)
        chunks = [c for c in chunks if c]
        work_args = [(chunk, *common) for chunk in chunks]

        sig_l = np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)
        sig_g = np.zeros_like(sig_l)

        use_threads = os.environ.get("QUATREX_SE_THREADS", "1") == "1"
        if use_threads:
            with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                futures = [executor.submit(_se_worker_iq, wa) for wa in work_args]
                for fut in futures:
                    iq_list, sl, sg = fut.result()
                    for i, iq in enumerate(iq_list):
                        sig_l[iq] = sl[i]
                        sig_g[iq] = sg[i]
        else:
            ctx = get_context("forkserver")
            with ctx.Pool(processes=len(chunks)) as pool:
                for iq_list, sl, sg in pool.map(_se_worker_iq, work_args):
                    for i, iq in enumerate(iq_list):
                        sig_l[iq] = sl[i]
                        sig_g[iq] = sg[i]

    return sig_l, sig_g
