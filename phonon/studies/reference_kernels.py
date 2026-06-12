"""Independent reference 3-phonon self-energy kernels (validation only).

These are NOT the production path -- the solver computes the self-energy with
the unified :func:`phonon.solver.se_finite.compute_phph_self_energy`. The two
kernels here are deliberately separate, simpler implementations kept solely so
the validation studies can cross-check the production kernel against an
independent algorithm:

* :func:`compute_phph_self_energy_finite` -- the Gamma-only bubble as two plain
  :func:`bubble_dense` calls on a single vertex block.
* :func:`compute_phph_self_energy_q_dense` -- the q-resolved slab-diagonal bubble
  built from the gathering-matrix Fourier vertex (T(q) M T(q')), contracted with
  a streaming batched einsum over a multiprocessing pool.

Used by :mod:`phonon.studies.linewidths` (and historically by the legacy
validation modules in ``phonon.studies``).
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

# The solver/phonon_inputs packages use flat intra-repo imports
# (``from phonon_inputs...``, ``from solver...``), so the phonon/ directory
# must be importable in addition to the repo root.
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "phonon")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phonon.phonon_inputs.constants import HBAR_SI, PHPH_SYMMETRY_FACTOR
from phonon.solver.bubble import bubble_dense


# ---------------------------------------------------------------------------
# Gamma-only single-block reference (two bubble_dense calls)
# ---------------------------------------------------------------------------


def compute_phph_self_energy_finite(
    G_lesser, G_greater, Phi, omega_grid_thz, dw_thz, symmetry_factor=None,
):
    """Phonon-phonon self-energy for one isolated Gamma vertex block.

    Returns ``(Sigma^<, Sigma^>)`` of shape ``(n_freq, n_dof, n_dof)``.
    """
    if symmetry_factor is None:
        symmetry_factor = PHPH_SYMMETRY_FACTOR
    n_freq = len(omega_grid_thz)
    n_fft = 2 * n_freq - 1
    mid = n_freq // 2
    freq_sl = slice(mid, mid + n_freq)
    prefactor = symmetry_factor * 0.5j * HBAR_SI * dw_thz / (2 * np.pi)

    sig_l = bubble_dense(
        phi_left=Phi, phi_right=Phi, G_a=G_lesser, G_b=G_lesser,
        n_fft=n_fft, prefactor=prefactor,
        out_slice=freq_sl, zero_freq_idx=mid, dc_handling="zero",
    )
    sig_g = bubble_dense(
        phi_left=Phi, phi_right=Phi, G_a=G_greater, G_b=G_greater,
        n_fft=n_fft, prefactor=prefactor,
        out_slice=freq_sl, zero_freq_idx=mid, dc_handling="zero",
    )
    return sig_l, sig_g


# ---------------------------------------------------------------------------
# q-resolved slab-diagonal reference (gathering-matrix Fourier vertex)
# ---------------------------------------------------------------------------


def _se_worker_iq(args):
    """Process one chunk of iq_ext values using FFT-domain Phi contraction.

    ``Phi_all`` is either the dense vertex tensor (n_kpts, n_kpts, nd, nd, nd) or,
    for the low-memory streaming path, the tuple ``("stream", TM, T_arr_H)`` with
    ``TM`` (n_kpts, nd, nd, dim_t) and ``T_arr_H`` (n_kpts, dim_t, nd); the
    per-pair vertex Phi(q1,q2)=TM[q1] @ T_arr_H[q2] is then built on the fly per
    batch, dropping the peak memory from O(n_kpts^2 nd^3) to O(qp_batch nd^3)."""
    (iq_ext_list, GL_fft, GG_fft, Phi_all, q_diff_map,
     n_freq, n_dof, n_kpts, n_fft, freq_sl_start, freq_sl_stop,
     prefactor, qp_batch) = args

    stream = isinstance(Phi_all, tuple) and Phi_all[0] == "stream"
    if stream:
        _, TM, T_arr_H = Phi_all

    freq_sl = slice(freq_sl_start, freq_sl_stop)
    n_out = len(iq_ext_list)
    nd = n_dof
    nd2 = nd * nd
    sig_l = np.zeros((n_out, n_freq, nd, nd), dtype=complex)
    sig_g = np.zeros((n_out, n_freq, nd, nd), dtype=complex)

    iq_prime_all = np.arange(n_kpts)

    for idx, iq_ext in enumerate(iq_ext_list):
        iq_diff_arr = q_diff_map[iq_ext]
        if not stream:
            PL_all = Phi_all[iq_prime_all, iq_diff_arr]
            PR_all = Phi_all[iq_diff_arr, iq_prime_all]

        for G_fft, sig_out in [(GL_fft, sig_l), (GG_fft, sig_g)]:
            S_hat = np.zeros((n_fft, nd, nd), dtype=complex)

            for p0 in range(0, n_kpts, qp_batch):
                p1 = min(p0 + qp_batch, n_kpts)
                B = p1 - p0
                ps = slice(p0, p1)
                iq_d = iq_diff_arr[ps]

                if stream:
                    # Phi(q1,q2)[a,c,d] = TM[q1][a,c,j] @ T_arr_H[q2][j,d]
                    PL = np.einsum('pacj,pjd->pacd', TM[p0:p1], T_arr_H[iq_d])
                    PR = np.einsum('pacj,pjd->pacd', TM[iq_d], T_arr_H[p0:p1])
                else:
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
    symmetry_factor=None, stream_phi=False,
):
    """Compute the q-dependent slab-diagonal phonon-phonon self-energy.

    Returns ``(Sigma^<, Sigma^>)`` of shape ``(n_kpts, n_freq, n_dof, n_dof)``.
    ``symmetry_factor`` defaults to ``constants.PHPH_SYMMETRY_FACTOR``; pass
    ``1.0`` for the legacy (Luisier) convention.
    """
    if symmetry_factor is None:
        symmetry_factor = PHPH_SYMMETRY_FACTOR
    n_freq = len(omega_grid_thz)
    n_dof = nat_prim * 3
    dim_t = M_stacked.shape[1]

    n_fft = 2 * n_freq - 1
    mid = n_freq // 2
    freq_sl = slice(mid, mid + n_freq)

    prefactor = symmetry_factor * 0.5j * HBAR_SI * dw_thz / (2 * np.pi) / n_kpts

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
    if stream_phi:
        # Stream the vertex: keep only TM and T_arr_H, building Phi(q1,q2) per
        # batch in the worker.
        Phi_all = ("stream", TM, T_arr_H)
        del T_arr, M_blocks
    else:
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
