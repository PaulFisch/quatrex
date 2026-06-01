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

from phonon_inputs.constants import HBAR_SI, PHPH_SYMMETRY_FACTOR


def _se_worker_iq(args):
    """Process one chunk of iq_ext values using FFT-domain Phi contraction.

    ``Phi_all`` is either the dense vertex tensor (n_kpts, n_kpts, nd, nd, nd) or,
    for the low-memory streaming path, the tuple ``("stream", TM, T_arr_H)`` with
    ``TM`` (n_kpts, nd, nd, dim_t) and ``T_arr_H`` (n_kpts, dim_t, nd); the per-pair
    vertex Phi(q1,q2)=TM[q1] @ T_arr_H[q2] is then built on the fly per batch,
    dropping the peak memory from O(n_kpts^2 nd^3) to O(qp_batch nd^3)."""
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
    """Compute q-dependent phonon-phonon self-energy.

    Returns ``(Σ^<, Σ^>)`` only; Σ^R is rebuilt from the mixed pair in
    the SCBA loop via :func:`phonon.solver.retarded.build_retarded`.

    Parameters
    ----------
    symmetry_factor : float or None
        Bubble symmetry factor (see ``constants.PHPH_SYMMETRY_FACTOR``).
        ``None`` uses the physically-correct default ``1/4``; pass ``1.0``
        for the legacy (Luisier, 4x-too-large) convention.

    Returns
    -------
    Sigma_lesser, Sigma_greater : (n_kpts, n_freq, n_dof, n_dof)
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
        # Stream the vertex: keep only TM (n_kpts, nd, nd, dim_t) and T_arr_H,
        # building Phi(q1,q2) per batch in the worker. Peak memory drops from
        # O(n_kpts^2 nd^3) to O(qp_batch nd^3) -- the GPU-memory path.
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


def _qfold_device_blocks(M_stacked, prim_indices, cell_frac, slab_indices,
                         n_atoms, n_slabs, q1_frac, q2_frac, transport_direction,
                         vertex_cutoff=None):
    """q-folded device FC3 blocks {(I,K,K'): Phi(q1,q2)[a,b,c]}.

    Reuses :func:`fc3_device.build_device_fc3_blocks` on a phase-modified M_stacked:
    the two CONTRACTED legs (b<-q1, c<-q2) carry transverse Bloch phases
    exp(-2*pi*i q . R_perp), exactly as :func:`separable.build_gathering_matrix`,
    so summing supercell images reproduces T(q1) M T(q2) per device block.
    The external leg (a) carries no phase. q1_frac/q2_frac are 2-component
    transverse fractions.
    """
    from .fc3_device import build_device_fc3_blocks
    n_dof = 3 * n_atoms
    n_super = len(prim_indices)
    dim_sc = n_super * 3
    tidx = "xyz".index(transport_direction)
    perp = [i for i in range(3) if i != tidx]
    q1 = np.zeros(3); q1[perp[0]], q1[perp[1]] = q1_frac
    q2 = np.zeros(3); q2[perp[0]], q2[perp[1]] = q2_frac
    ph1 = np.exp(-2j * np.pi * cell_frac @ q1)        # (n_super,)
    ph2 = np.exp(-2j * np.pi * cell_frac @ q2)
    Mb = M_stacked.reshape(n_dof, dim_sc, dim_sc).astype(complex)
    # apply per-supercell-atom phases on the two contracted legs (3 dofs/atom)
    p1 = np.repeat(ph1, 3); p2 = np.repeat(ph2, 3)
    Mq = (Mb * p1[None, :, None]) * p2[None, None, :]
    Mq_stacked = Mq.reshape(n_dof * dim_sc, dim_sc)
    return build_device_fc3_blocks(
        Mq_stacked, prim_indices, slab_indices, n_atoms, n_slabs,
        vertex_cutoff=vertex_cutoff)


def compute_phph_self_energy_q_dense_multi_slab(
    g_lesser_blocks_q, g_greater_blocks_q,
    M_stacked, prim_indices, cell_frac, slab_indices,
    n_atoms, n_slabs, n_kpts, q_points, q_diff_map,
    omega_grid_thz, dw_thz, transport_direction="x", *,
    sigma_cutoff=None, g_cutoff=None, vertex_cutoff=None,
    dc_handling="interpolate", symmetry_factor=None,
):
    """FULL (off-diagonal-in-slab) q-resolved 3-phonon self-energy.

    Generalises :func:`compute_phph_self_energy_q_dense` (single on-site block)
    to a multi-slab device with inter-slab self-energy blocks Sigma_{IJ}(q_perp,w),
    coupling the transverse momenta by conservation q_ext=q'+(q_ext-q'). The slab
    structure mirrors the Gamma-only template
    :func:`se_finite.compute_phph_self_energy_finite_multi_slab`; the q-fold and the
    internal-momentum sum are added on top. Reduces EXACTLY to that Gamma-only
    multi-slab self-energy at a 1x1 mesh.

    G blocks: dicts ``{(K,K'): (n_kpts, n_freq, n_dof, n_dof)}``.
    Returns ``({(I,J): Sigma^<(n_kpts,n_freq,n_dof,n_dof)}, {(I,J): Sigma^>...})``.

    ``sigma_cutoff`` bounds output |I-J| (None=full), ``g_cutoff`` bounds input G
    range, ``vertex_cutoff`` bounds the FC3 slab reach; setting sigma_cutoff=0 is
    Guo's diagonal-block approximation (III). symmetry_factor: see PHPH_SYMMETRY_FACTOR.
    """
    from .se_finite import _build_pair_index, _filter_g_blocks
    from .bubble import precompute_g_fft, bubble_dense_from_fft

    if symmetry_factor is None:
        symmetry_factor = PHPH_SYMMETRY_FACTOR
    n_freq = len(omega_grid_thz)
    n_dof = 3 * n_atoms
    n_fft = 2 * n_freq - 1
    mid = n_freq // 2
    freq_sl = slice(mid, mid + n_freq)
    prefactor = symmetry_factor * 0.5j * HBAR_SI * dw_thz / (2 * np.pi) / n_kpts

    gl = _filter_g_blocks(g_lesser_blocks_q, g_cutoff)
    gg = _filter_g_blocks(g_greater_blocks_q, g_cutoff)
    g_keys = set(gl.keys())

    # Per-(q, block) pre-FFT of the G blocks (shared across all bubbles at that q).
    def _fft_all(gblk):
        out = {}
        for (K, Kp), arr in gblk.items():
            out[(K, Kp)] = [precompute_g_fft(arr[iq], n_fft=n_fft, zero_freq_idx=mid,
                                             dc_handling=dc_handling)
                            for iq in range(n_kpts)]
        return out
    gl_fft = _fft_all(gl)
    gg_fft = _fft_all(gg)

    # Pair-index (I,J,K1,K2,K1',K2') from the real (Gamma) device blocks; the q-fold
    # changes only the numbers in the blocks, not which (I,K,K') keys exist.
    phi_real = _qfold_device_blocks(
        M_stacked, prim_indices, cell_frac, slab_indices, n_atoms, n_slabs,
        (0.0, 0.0), (0.0, 0.0), transport_direction, vertex_cutoff=vertex_cutoff)
    pair_index = _build_pair_index(phi_real, g_keys, n_slabs, sigma_cutoff=sigma_cutoff)

    # cache q-folded device dicts by (iq1, iq2)
    _cache = {}
    def folded(iq1, iq2):
        key = (iq1, iq2)
        d = _cache.get(key)
        if d is None:
            d = _qfold_device_blocks(
                M_stacked, prim_indices, cell_frac, slab_indices, n_atoms, n_slabs,
                q_points[iq1], q_points[iq2], transport_direction,
                vertex_cutoff=vertex_cutoff)
            _cache[key] = d
        return d

    sl_out = {(I, J): np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)
              for (I, J) in pair_index}
    sg_out = {(I, J): np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)
              for (I, J) in pair_index}

    for iq_ext in range(n_kpts):
        for iqp in range(n_kpts):              # internal q'
            iq2 = int(q_diff_map[iq_ext, iqp])  # q_ext - q'
            phiL = folded(iqp, iq2)            # left vertex legs (q', q_ext-q')
            phiR = folded(iq2, iqp)            # right vertex legs (q_ext-q', q')
            for (I, J), quads in pair_index.items():
                for (K1, K2, K1p, K2p, _pl, _pr) in quads:
                    pl = phiL.get((I, K1, K2))
                    pr = phiR.get((J, K2p, K1p))
                    if pl is None or pr is None:
                        continue
                    for gfft, out in ((gl_fft, sl_out), (gg_fft, sg_out)):
                        ga = gfft.get((K1, K1p))
                        gb = gfft.get((K2, K2p))
                        if ga is None or gb is None:
                            continue
                        out[(I, J)][iq_ext] += bubble_dense_from_fft(
                            phi_left=pl, phi_right=pr,
                            G_a_fft=ga[iqp], G_b_fft=gb[iq2],
                            ne=n_freq, prefactor=prefactor, out_slice=freq_sl)
    return sl_out, sg_out
