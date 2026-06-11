"""Unified multi-slab 3-phonon self-energy kernel.

The single production kernel is :func:`compute_phph_self_energy`. It computes
Sigma^{<,>}_{IJ}(q, omega) for every (I, J) slab pair allowed by the cutoffs,
summing over every inner (K1, K2, K1', K2') block quadruple the FC3 support and
``g_cutoff`` permit, and -- when ``n_kpts > 1`` -- over the internal transverse
momentum (the coupled-q convolution). It takes a pre-built vertex map, so the
Gamma-only (``n_kpts == 1``) and the transversely-periodic paths share one body.

:func:`compute_phph_self_energy_finite_multi_slab` is a thin Gamma-only wrapper
(single q-point) kept for direct callers. The q-resolved wrapper that builds the
q-folded vertices lives in :mod:`phonon.solver.se_q`.

The heavy bubble loop (memory budgeting + thread pool + block accumulation)
lives in :func:`_run_bubble_tasks`. The kernel returns Sigma^{<,>} only; Sigma^R
is rebuilt from the mixed pair in the SCBA loop via
:func:`phonon.solver.retarded.build_retarded`. Independent reference
implementations used only for validation live in
``phonon/scripts/verify/reference_kernels.py``.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from phonon_inputs.constants import HBAR_SI, PHPH_SYMMETRY_FACTOR
from .bubble import (
    bubble_chunk_peak_bytes_per_w,
    bubble_dense_from_fft,
    precompute_g_fft,
)


def bubble_prefactor(dw_thz, n_kpts=1, symmetry_factor=None):
    """Common bubble prefactor i/2 * hbar * dw / (2 pi) / n_kpts.

    ``symmetry_factor`` (see ``constants.PHPH_SYMMETRY_FACTOR``) defaults to
    the physically-correct value; pass ``1.0`` for the legacy (Luisier,
    4x-too-large) convention.
    """
    if symmetry_factor is None:
        symmetry_factor = PHPH_SYMMETRY_FACTOR
    return symmetry_factor * 0.5j * HBAR_SI * dw_thz / (2 * np.pi) / n_kpts


# ---------------------------------------------------------------------------
# Block bookkeeping
# ---------------------------------------------------------------------------


def _filter_g_blocks(g_blocks, g_cutoff):
    """Drop G blocks with ``|K - K'| > g_cutoff`` (no-op when None)."""
    if g_cutoff is None:
        return g_blocks
    return {
        key: blk for key, blk in g_blocks.items()
        if abs(key[0] - key[1]) <= g_cutoff
    }


def _build_pair_index(phi_blocks, g_keys, n_slabs, *, sigma_cutoff):
    """Enumerate (I, J) -> [(K1, K2, K1', K2', phi_left, phi_right), ...].

    For each ``phi_left = phi_blocks[(I, K1, K2)]`` and
    ``phi_right = phi_blocks[(J, K2', K1')]``, the bubble integrand
    contributes only when both ``(K1, K1')`` and ``(K2, K2')`` are present
    in the G dict. ``sigma_cutoff`` bounds the produced ``|I - J|``.
    """
    d_sigma = (n_slabs - 1) if sigma_cutoff is None else int(sigma_cutoff)

    pair_index: dict[tuple[int, int], list] = {}
    for (I, K1, K2), phi_left in phi_blocks.items():
        j_lo = max(0, I - d_sigma)
        j_hi = min(n_slabs, I + d_sigma + 1)
        for J in range(j_lo, j_hi):
            for K1_prime in range(n_slabs):
                if (K1, K1_prime) not in g_keys:
                    continue
                for K2_prime in range(n_slabs):
                    if (K2, K2_prime) not in g_keys:
                        continue
                    phi_right = phi_blocks.get((J, K2_prime, K1_prime))
                    if phi_right is None:
                        continue
                    pair_index.setdefault((I, J), []).append(
                        (K1, K2, K1_prime, K2_prime, phi_left, phi_right)
                    )
    return pair_index


# ---------------------------------------------------------------------------
# Memory budgeting helpers
# ---------------------------------------------------------------------------


def _default_n_threads():
    """Worker-thread count for the bubble loop.

    Honours ``QUATREX_PHPH_THREADS`` first, else uses every visible core.
    The bubble kernel releases the GIL during BLAS matmul / FFT, so workers
    run in parallel; cap BLAS to 1 thread per worker (``OMP_NUM_THREADS=1``
    / ``OPENBLAS_NUM_THREADS=1``) to avoid oversubscription.
    """
    env = os.environ.get("QUATREX_PHPH_THREADS")
    if env is not None:
        return max(1, int(env))
    return max(1, os.cpu_count() or 1)


def _available_memory_bytes():
    """Best-effort per-process available RAM.

    Resolution order: ``QUATREX_PHPH_MEMORY_GB`` env var, then :mod:`psutil`,
    then ``/proc/meminfo``'s ``MemAvailable``, then a 8 GiB fallback.
    """
    env = os.environ.get("QUATREX_PHPH_MEMORY_GB")
    if env is not None:
        return int(float(env) * (1 << 30))
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kib = int(line.split()[1])
                    return kib * 1024
    except (FileNotFoundError, ValueError):
        pass
    return 8 * (1 << 30)


def _bubble_peak_bytes_per_worker(
    *, n_fft, nI, nJ, bK1, bK1p, bK2, bK2p, itemsize=16,
):
    """Upper bound on the transient memory one un-chunked bubble holds.

    The matmul kernel allocates a handful of ``(n_fft, n_dof^3)``-sized
    intermediates; the two largest dominate. Count four of them as live
    simultaneously plus the small output, with 25% slack.
    """
    big = n_fft * nI * max(bK1, bK2p) * max(bK1p, bK2p) * itemsize
    small = n_fft * nI * nJ * itemsize
    return int(1.25 * (4 * big + 2 * small))


class _NoThreadLimit:
    """Context manager used when threadpoolctl is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _blas_thread_limit():
    """Limit BLAS to one thread inside the parallel region, if possible."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1, user_api="blas")
    except ImportError:
        return _NoThreadLimit()


# ---------------------------------------------------------------------------
# Shared bubble task runner
# ---------------------------------------------------------------------------


def _run_bubble_tasks(
    tasks, compute_one, *,
    n_fft, nI, nJ, bK1, bK1p, bK2, bK2p,
    fixed_bytes, n_threads=None, label="phph",
):
    """Run bubble ``tasks`` on a memory-budgeted thread pool.

    ``compute_one(task, max_bytes) -> (key, block)`` computes one bubble;
    ``max_bytes`` bounds its transient memory by chunking the frequency
    axis (``None`` disables chunking). Blocks returned with the same
    ``key`` are summed. Returns ``{key: block}``.

    The budget logic: take 70% of available RAM, subtract the resident
    ``fixed_bytes`` (pre-FFT'd G blocks, vertices, output buffers), then pick
    the largest worker count for which each worker still gets at least one
    one-frequency chunk. If even that does not fit, raise ``MemoryError``.
    Shared by the Gamma-only and q-resolved multi-slab kernels.
    """
    out: dict = {}

    def _accumulate(key_block):
        key, blk = key_block
        existing = out.get(key)
        out[key] = blk if existing is None else existing + blk

    if not tasks:
        return out

    full_peak = _bubble_peak_bytes_per_worker(
        n_fft=n_fft, nI=nI, nJ=nJ, bK1=bK1, bK1p=bK1p, bK2=bK2, bK2p=bK2p,
    )
    # Transient of a single one-frequency chunk, with the same 25% slack.
    per_w_peak = int(1.25 * bubble_chunk_peak_bytes_per_w(
        nI=nI, nJ=nJ, bK1=bK1, bK1p=bK1p, bK2=bK2, bK2p=bK2p,
    ))

    total_budget = int(0.7 * _available_memory_bytes())
    worker_budget = total_budget - fixed_bytes
    if worker_budget < per_w_peak:
        raise MemoryError(
            f"[{label}] fixed G/phi storage is "
            f"{fixed_bytes / (1 << 30):.2f} GiB and a single frequency chunk "
            f"needs {per_w_peak / (1 << 30):.3f} GiB, but only "
            f"{total_budget / (1 << 30):.2f} GiB is available. Reduce "
            "n_slabs / vertex_cutoff / g_cutoff, or raise "
            "QUATREX_PHPH_MEMORY_GB."
        )

    n_workers_requested = max(
        1, n_threads if n_threads is not None else _default_n_threads())
    n_workers_by_memory = max(1, worker_budget // per_w_peak)
    n_workers = min(n_workers_requested, len(tasks), n_workers_by_memory)

    per_worker_share = worker_budget // n_workers
    if per_worker_share >= full_peak:
        per_worker_max_bytes = None  # plenty of headroom, no chunking
    else:
        per_worker_max_bytes = int(per_worker_share)
        n_chunks_est = -(-full_peak // max(per_worker_max_bytes, 1))
        print(
            f"  [{label}] memory cap: budget="
            f"{total_budget / (1 << 30):.1f} GiB, fixed="
            f"{fixed_bytes / (1 << 30):.2f} GiB, workers={n_workers} "
            f"(requested={n_workers_requested}), per-worker="
            f"{per_worker_share / (1 << 30):.2f} GiB; omega-chunking each "
            f"bubble into ~{n_chunks_est} chunk(s)",
            flush=True,
        )

    if n_workers == 1 or len(tasks) <= 1:
        for task in tasks:
            _accumulate(compute_one(task, per_worker_max_bytes))
    else:
        with _blas_thread_limit(), ThreadPoolExecutor(
                max_workers=n_workers) as pool:
            for key_block in pool.map(
                    lambda t: compute_one(t, per_worker_max_bytes), tasks):
                _accumulate(key_block)
    return out


# ---------------------------------------------------------------------------
# Multi-slab kernel
# ---------------------------------------------------------------------------


def compute_phph_self_energy(
    g_lesser_blocks_q,
    g_greater_blocks_q,
    vertices,
    n_slabs,
    n_kpts,
    q_diff_map,
    omega_grid_thz,
    dw_thz,
    *,
    sigma_cutoff=None,
    g_cutoff=None,
    dc_handling="interpolate",
    n_threads=None,
    symmetry_factor=None,
):
    """Unified multi-slab 3-phonon self-energy (Gamma or q-resolved).

    Computes ``Sigma^{<,>}_{IJ}(q_ext, omega)`` for every (I, J) slab pair the
    cutoffs allow, summing the inner (K1, K2, K1', K2') block quadruple and, for
    ``n_kpts > 1``, the internal transverse momentum q' with
    ``q2 = q_ext - q'`` (the coupled-q convolution).

    Parameters
    ----------
    g_lesser_blocks_q, g_greater_blocks_q
        Device G blocks ``{(K, K'): G^x[n_kpts, n_freq, n_dof, n_dof]}``. The
        leading q axis has length 1 for a Gamma-only device.
    vertices
        Pre-built device vertex per transverse-momentum pair,
        ``{(iq1, iq2): {(I, K, K'): Phi[n_dof, n_dof, n_dof]}}``. The q=Gamma
        entry must be keyed ``(0, 0)`` (used for the (I, J) pair index). For a
        finite device pass ``{(0, 0): device_vertex}`` and ``q_diff_map=[[0]]``.
        Build these once (they do not depend on G) via
        :func:`phonon.solver.fc3_device.build_device_fc3_blocks` (Gamma) or
        :func:`phonon.solver.se_q._build_folded_vertices` (q-resolved).
    n_slabs, n_kpts
        Number of transport cells and transverse q-points.
    q_diff_map
        ``q_diff_map[iq_ext, iq'] = index of (q_ext - q')`` on the mesh.
    omega_grid_thz, dw_thz
        Symmetric ``(-fmax, ..., fmax)`` frequency axis and its spacing (THz).
    sigma_cutoff
        Maximum ``|I - J|`` for produced Sigma blocks. ``None`` = full; ``0``
        keeps only the slab-diagonal blocks (Guo approximation III).
    g_cutoff
        Maximum ``|K - K'|`` for G blocks used in the inner sum.
    dc_handling
        Treatment of the omega = 0 sample of G before the bubble FFT.

    Returns
    -------
    sigma_lesser, sigma_greater
        ``{(I, J): Sigma^x[n_kpts, n_freq, n_I, n_J]}`` dicts (allocated for
        every reachable (I, J); pairs with no contribution stay zero).
    """
    n_freq = len(omega_grid_thz)
    n_fft = 2 * n_freq - 1
    mid = n_freq // 2
    freq_sl = slice(mid, mid + n_freq)
    prefactor = bubble_prefactor(dw_thz, n_kpts=n_kpts,
                                 symmetry_factor=symmetry_factor)

    gl = _filter_g_blocks(g_lesser_blocks_q, g_cutoff)
    gg = _filter_g_blocks(g_greater_blocks_q, g_cutoff)
    g_keys = set(gl.keys())
    if set(gg.keys()) != g_keys:
        raise ValueError(
            "g_lesser and g_greater blocks must share the same (K, K') keys; "
            f"lesser-only={g_keys - set(gg.keys())}, "
            f"greater-only={set(gg.keys()) - g_keys}"
        )

    pair_index = _build_pair_index(
        vertices[(0, 0)], g_keys, n_slabs, sigma_cutoff=sigma_cutoff)
    if not g_keys or not pair_index:
        return {}, {}

    n_dof = next(iter(gl.values())).shape[-1]

    # Pre-FFT each (q, block) once; every bubble at that q reuses it.
    def _fft_all(gblk):
        return {
            (K, Kp): [precompute_g_fft(arr[iq], n_fft=n_fft, zero_freq_idx=mid,
                                       dc_handling=dc_handling)
                      for iq in range(n_kpts)]
            for (K, Kp), arr in gblk.items()
        }
    gl_fft = _fft_all(gl)
    gg_fft = _fft_all(gg)

    # One task per (external q, internal q', I, J, inner quadruple, kind). The
    # internal-q' sum and the (K1, K2, K1', K2') sum both fold into the
    # accumulation by key (I, J, iq_ext, kind).
    tasks = []
    for iq_ext in range(n_kpts):
        for iqp in range(n_kpts):
            iq2 = int(q_diff_map[iq_ext, iqp])
            phiL = vertices[(iqp, iq2)]        # legs (q', q_ext - q')
            phiR = vertices[(iq2, iqp)]        # legs (q_ext - q', q')
            for (I, J), quads in pair_index.items():
                for (K1, K2, K1p, K2p, _pl, _pr) in quads:
                    pl = phiL.get((I, K1, K2))
                    pr = phiR.get((J, K2p, K1p))
                    if pl is None or pr is None:
                        continue
                    # LEFT vertex conjugated: Phi(q', q-q')^* pairs with
                    # Phi(q-q', q'); the unconjugated pairing breaks
                    # Sigma(-q) = Sigma(q)^T and disagrees with the
                    # real-space supercell ground truth (see
                    # phonon/scripts/verify/audit_qfold_trs.py). Real
                    # vertices at Gamma -> nq==1 results unchanged.
                    tasks.append(
                        (I, J, iq_ext, iqp, iq2, K1, K2, K1p, K2p,
                         np.conj(pl), pr))

    sl_out = {(I, J): np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)
              for (I, J) in pair_index}
    sg_out = {(I, J): np.zeros((n_kpts, n_freq, n_dof, n_dof), dtype=complex)
              for (I, J) in pair_index}
    if not tasks:
        return sl_out, sg_out

    def compute_one(kind_task, max_bytes):
        kind, task = kind_task
        I, J, iq_ext, iqp, iq2, K1, K2, K1p, K2p, pl, pr = task
        gx_fft = gl_fft if kind == "lesser" else gg_fft
        blk = bubble_dense_from_fft(
            phi_left=pl, phi_right=pr,
            G_a_fft=gx_fft[(K1, K1p)][iqp], G_b_fft=gx_fft[(K2, K2p)][iq2],
            ne=n_freq, prefactor=prefactor, out_slice=freq_sl,
            max_bytes=max_bytes,
        )
        return (I, J, iq_ext, kind), blk

    itemsize = 16  # complex128
    unique_phi = {id(p): p for d in vertices.values() for p in d.values()}
    fixed_bytes = (
        sum(g.nbytes for blk in gl_fft.values() for g in blk)
        + sum(g.nbytes for blk in gg_fft.values() for g in blk)
        + sum(p.nbytes for p in unique_phi.values())
        + 2 * len(pair_index) * n_kpts * n_freq * n_dof * n_dof * itemsize
    )

    accumulated = _run_bubble_tasks(
        [("lesser", t) for t in tasks] + [("greater", t) for t in tasks],
        compute_one,
        n_fft=n_fft, nI=n_dof, nJ=n_dof,
        bK1=n_dof, bK1p=n_dof, bK2=n_dof, bK2p=n_dof,
        fixed_bytes=fixed_bytes, n_threads=n_threads, label="phph",
    )

    for (I, J, iq_ext, kind), blk in accumulated.items():
        out = sl_out if kind == "lesser" else sg_out
        out[(I, J)][iq_ext] = blk
    return sl_out, sg_out


def compute_phph_self_energy_finite_multi_slab(
    g_lesser_blocks,
    g_greater_blocks,
    phi_dev_blocks,
    n_slabs,
    omega_grid_thz,
    dw_thz,
    *,
    sigma_cutoff=None,
    g_cutoff=None,
    dc_handling="interpolate",
    n_threads=None,
    symmetry_factor=None,
):
    """Gamma-only multi-slab self-energy: thin wrapper over
    :func:`compute_phph_self_energy` with a single (Gamma) q-point.

    ``g_*_blocks`` are ``{(K, K'): G^x[n_freq, n_dof, n_dof]}`` (no q axis);
    ``phi_dev_blocks`` is ``{(I, K, K'): Phi}`` from
    :func:`phonon.solver.fc3_device.build_device_fc3_blocks`. Returns
    ``{(I, J): Sigma^x[n_freq, n_I, n_J]}`` (no q axis).
    """
    gl_q = {k: v[np.newaxis] for k, v in g_lesser_blocks.items()}
    gg_q = {k: v[np.newaxis] for k, v in g_greater_blocks.items()}
    sl, sg = compute_phph_self_energy(
        gl_q, gg_q, {(0, 0): phi_dev_blocks}, n_slabs, 1,
        np.array([[0]]), omega_grid_thz, dw_thz,
        sigma_cutoff=sigma_cutoff, g_cutoff=g_cutoff,
        dc_handling=dc_handling, n_threads=n_threads,
        symmetry_factor=symmetry_factor,
    )
    return ({k: v[0] for k, v in sl.items()},
            {k: v[0] for k, v in sg.items()})
