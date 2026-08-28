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
    bubble_dense_from_fft_task_batched,
    bubble_task_batch_bytes,
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


def shell_binner(shell_bins):
    """``|K - K'| -> bin index`` for the leg-distance decomposition.

    ``shell_bins`` is a list of inclusive ``(lo, hi)`` ranges that must
    partition the distances a device can carry. Returns ``None`` when
    ``shell_bins`` is ``None``, which is the default (undecomposed) path.
    """
    if shell_bins is None:
        return None
    edges = [(int(lo), int(hi)) for lo, hi in shell_bins]
    if any(hi < lo for lo, hi in edges):
        raise ValueError(f"shell_binner: empty range in {shell_bins}")
    for (a_lo, a_hi), (b_lo, b_hi) in zip(edges, edges[1:]):
        if b_lo != a_hi + 1:
            raise ValueError(
                f"shell_binner: bins must partition contiguously, got "
                f"{shell_bins}")
    if edges[0][0] != 0:
        raise ValueError("shell_binner: bins must start at distance 0")

    def which(d):
        for i, (lo, hi) in enumerate(edges):
            if lo <= d <= hi:
                return i
        raise ValueError(
            f"shell_binner: leg distance {d} outside {shell_bins}")

    return which


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


def _resident_g_stack(g_fft, xp):
    """Pack ``{(K, K'): [array per q]}`` into one resident array plus an index.

    Returns ``(stacked, key_index)`` with ``stacked[b, iq]`` the pre-FFT'd
    block for ``key_index[(K, K')] == b``. The point is that the gather for a
    task batch becomes fancy indexing INTO this array rather than a Python
    loop building a new one: on a GPU the dict form would mean a host round
    trip per task, which is the cost the batching exists to remove.
    """
    keys = sorted(g_fft)
    stacked = xp.asarray(np.stack([np.stack(g_fft[k]) for k in keys]))
    return stacked, {k: i for i, k in enumerate(keys)}


def _run_bubble_tasks_task_batched(
    kind_tasks, *, gl_fft, gg_fft, key_of, n_freq, n_fft, prefactor,
    freq_sl, n_dof, fixed_bytes, xp=None, label="phph", verbose=False,
):
    """Every bubble in one batched contraction per chunk, instead of one call
    each.

    The incumbent ``_run_bubble_tasks`` issues one ``bubble_dense_from_fft``
    per task through a thread pool -- 251384 calls for six SCBA iterations on a
    16-cell 4-DOF chain, and 98 % of the iteration. Two measurements motivate
    this path and neither is about Python overhead on the CPU:

    * the thread pool HURTS. Scanned on a 256-core tortin node, the per-task
      loop runs fastest at ONE worker for d = 1, 2, 4 and at two for d = 6; a
      16-worker pool is 0.15-0.26x of serial at d = 2-4. The tasks are small
      enough that pool and GIL traffic outweigh the parallelism.
    * batching is a wash on the CPU (2.5x at d = 1, 0.68-0.90x at d = 4-6,
      where numpy is already at its bound) but it is what makes a GPU usable:
      on a GH200 the per-task path runs at 0.09-1.3x of the CPU while the
      batched one runs at 13x, 5.5x, 72x, 111x for d = 1, 2, 4, 6.

    Results are summed by ``key_of(kind, task)``, exactly as the per-task
    runner accumulates them, so the two paths agree to the bit on numpy.
    """
    if xp is None:
        xp = np
    out: dict = {}
    if not kind_tasks:
        return out

    gl_s, gl_i = _resident_g_stack(gl_fft, xp)
    gg_s, gg_i = _resident_g_stack(gg_fft, xp)

    per_task = bubble_task_batch_bytes(
        n_fft=n_fft, nI=n_dof, nJ=n_dof, bK1=n_dof, bK1p=n_dof,
        bK2=n_dof, bK2p=n_dof)
    # The chunk has to be sized against the memory it will actually live in.
    # On a GPU that is the DEVICE, and the host figure is both wrong and much
    # larger: at d = 6 the full task set is ~278 GiB of transients, so a host
    # budget picks a chunk that OOMs the card immediately.
    if xp is not np:
        free, _total = xp.cuda.Device().mem_info
        budget = int(0.5 * free) - int(gl_s.nbytes + gg_s.nbytes)
        where = "device"
    else:
        budget = int(0.7 * _available_memory_bytes()) - fixed_bytes
        where = "host"
    if budget < per_task:
        raise MemoryError(
            f"[{label}] task-batched bubble needs "
            f"{per_task / (1 << 20):.1f} MiB for a single task but only "
            f"{budget / (1 << 20):.1f} MiB is free in {where} memory after "
            "the resident G and vertex storage. Reduce n_slabs / g_cutoff, "
            "or raise QUATREX_PHPH_MEMORY_GB.")
    chunk = max(1, min(len(kind_tasks), int(budget // per_task)))
    cap = os.environ.get("QX_PHPH_CHUNK")
    if cap:
        chunk = max(1, min(chunk, int(cap)))
    if verbose:
        print(f"  [{label}] task-batched: {len(kind_tasks)} tasks in "
              f"{-(-len(kind_tasks) // chunk)} chunk(s) of <= {chunk} "
              f"({per_task * chunk / (1 << 30):.2f} GiB transient, "
              f"{where} budget {budget / (1 << 30):.1f} GiB)", flush=True)

    for c0 in range(0, len(kind_tasks), chunk):
        sub = kind_tasks[c0:c0 + chunk]
        pl = xp.asarray(np.stack([t[9] for _k, t in sub]))
        pr = xp.asarray(np.stack([t[10] for _k, t in sub]))
        ia = np.empty(len(sub), dtype=np.int64)
        ib = np.empty(len(sub), dtype=np.int64)
        qa = np.empty(len(sub), dtype=np.int64)
        qb = np.empty(len(sub), dtype=np.int64)
        lesser = np.empty(len(sub), dtype=bool)
        for n, (kind, t) in enumerate(sub):
            idx = gl_i if kind == "lesser" else gg_i
            ia[n], ib[n] = idx[(t[5], t[7])], idx[(t[6], t[8])]
            qa[n], qb[n] = t[3], t[4]
            lesser[n] = kind == "lesser"

        blocks = xp.empty((len(sub), n_freq, n_dof, n_dof), dtype=complex)
        for src, stack, mask in ((gl_s, gl_s, lesser), (gg_s, gg_s, ~lesser)):
            if not mask.any():
                continue
            sel = np.flatnonzero(mask)
            xsel = xp.asarray(sel)
            ga = stack[xp.asarray(ia[sel]), xp.asarray(qa[sel])]
            gb = stack[xp.asarray(ib[sel]), xp.asarray(qb[sel])]
            blocks[xsel] = bubble_dense_from_fft_task_batched(
                phi_left=pl[xsel], phi_right=pr[xsel],
                G_a_fft=ga, G_b_fft=gb, ne=n_freq, prefactor=prefactor,
                out_slice=freq_sl, max_bytes=None, xp=xp)

        host = blocks if xp is np else xp.asnumpy(blocks)
        for n, (kind, t) in enumerate(sub):
            key = key_of(kind, t)
            prev = out.get(key)
            out[key] = host[n] if prev is None else prev + host[n]
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
    shell_bins=None,
    shells_out=None,
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
    shell_bins, shells_out
        Optional EXACT decomposition of the result by the leg-distance shell
        of each internal Green-function link. ``Sigma`` is bilinear in ``G``,
        so writing ``G = sum_m G^(m)`` by shell ``m = |K - K'|`` gives

            Sigma_IJ = sum_{m, m'} Sigma_IJ^{(m, m')}

        exactly -- of which a ``g_cutoff`` sweep is only the partial sums, and
        cumulative ones at that (raising the band changes blocks that already
        existed, through interference). ``shell_bins`` is a contiguous list of
        inclusive ``(lo, hi)`` distance ranges starting at 0; ``shells_out``,
        if given, is filled with ``{(bin_a, bin_b): (sigma_lesser,
        sigma_greater)}`` in the same block-dict layout as the return value.
        The returned totals are unchanged either way -- they are the sum over
        shells, formed from the same accumulation -- so the default path is
        bit-identical to not passing these at all.

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

    which_shell = shell_binner(shell_bins)

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
        if which_shell is None:
            return (I, J, iq_ext, kind), blk
        return ((I, J, iq_ext, kind,
                 which_shell(abs(K1 - K1p)), which_shell(abs(K2 - K2p))), blk)

    itemsize = 16  # complex128
    unique_phi = {id(p): p for d in vertices.values() for p in d.values()}
    fixed_bytes = (
        sum(g.nbytes for blk in gl_fft.values() for g in blk)
        + sum(g.nbytes for blk in gg_fft.values() for g in blk)
        + sum(p.nbytes for p in unique_phi.values())
        + 2 * len(pair_index) * n_kpts * n_freq * n_dof * n_dof * itemsize
    )

    kind_tasks = ([("lesser", t) for t in tasks]
                  + [("greater", t) for t in tasks])

    # Task-batching and the GPU are OPT-IN and off by default: the per-task
    # path above is the one every stored result was produced with, and the two
    # agree to the bit on numpy (tests/quatrex/phonon/test_bubble_kernel.py).
    # QX_PHPH_XP=cupy implies batching -- the per-task form on a GPU measures
    # 0.09-1.3x of the CPU, so offering it would only be a trap.
    xp_name = os.environ.get("QX_PHPH_XP", "numpy").lower()
    want_batched = (os.environ.get("QX_PHPH_BATCHED", "0") == "1"
                    or xp_name != "numpy")
    if want_batched:
        if xp_name in ("cupy", "gpu"):
            import cupy as _xp
        elif xp_name == "numpy":
            _xp = np
        else:
            raise ValueError(f"QX_PHPH_XP={xp_name!r}: expected numpy or cupy")

        def key_of(kind, t):
            if which_shell is None:
                return (t[0], t[1], t[2], kind)
            return (t[0], t[1], t[2], kind,
                    which_shell(abs(t[5] - t[7])),
                    which_shell(abs(t[6] - t[8])))

        accumulated = _run_bubble_tasks_task_batched(
            kind_tasks, gl_fft=gl_fft, gg_fft=gg_fft, key_of=key_of,
            n_freq=n_freq, n_fft=n_fft, prefactor=prefactor, freq_sl=freq_sl,
            n_dof=n_dof, fixed_bytes=fixed_bytes, xp=_xp, label="phph",
            verbose=os.environ.get("QX_PHPH_VERBOSE") == "1",
        )
    else:
        accumulated = _run_bubble_tasks(
            kind_tasks,
            compute_one,
            n_fft=n_fft, nI=n_dof, nJ=n_dof,
            bK1=n_dof, bK1p=n_dof, bK2=n_dof, bK2p=n_dof,
            fixed_bytes=fixed_bytes, n_threads=n_threads, label="phph",
        )

    if which_shell is None:
        for (I, J, iq_ext, kind), blk in accumulated.items():
            out = sl_out if kind == "lesser" else sg_out
            out[(I, J)][iq_ext] = blk
        return sl_out, sg_out

    n_bins = len(shell_bins)
    if shells_out is not None:
        for m in range(n_bins):
            for mp in range(n_bins):
                shells_out[(m, mp)] = (
                    {k: np.zeros_like(v) for k, v in sl_out.items()},
                    {k: np.zeros_like(v) for k, v in sg_out.items()})
    for (I, J, iq_ext, kind, m, mp), blk in accumulated.items():
        out = sl_out if kind == "lesser" else sg_out
        out[(I, J)][iq_ext] += blk
        if shells_out is not None:
            pair = shells_out[(m, mp)][0 if kind == "lesser" else 1]
            pair[(I, J)][iq_ext] += blk
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
    shell_bins=None,
    shells_out=None,
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
        shell_bins=shell_bins, shells_out=shells_out,
    )
    if shells_out is not None:
        for key, (sl_s, sg_s) in list(shells_out.items()):
            shells_out[key] = ({k: v[0] for k, v in sl_s.items()},
                               {k: v[0] for k, v in sg_s.items()})
    return ({k: v[0] for k, v in sl.items()},
            {k: v[0] for k, v in sg.items()})
