"""Phonon-phonon self-energy drivers for a finite Gamma-only device.

Two entry points live here:

* :func:`compute_phph_self_energy_finite` — legacy single-slab driver
  used by :func:`phonon.solver.dense.transmission_finite` when the
  caller leaves all cutoffs at their legacy values
  (``sigma_cutoff=0, vertex_cutoff=0, g_cutoff=0``).
* :func:`compute_phph_self_energy_finite_multi_slab` — full multi-slab
  driver. Computes ``Sigma^{<,>}_{IJ}`` for every supported ``(I, J)``
  pair within the supplied cutoffs, accumulating contributions from
  every ``(K1, K2, K1', K2')`` inner block quadruple compatible with
  the FC3 support and ``g_cutoff``. ``bubble_dense`` supports
  off-diagonal G already so no kernel rewrite is needed.

Both routines feed Sigma^{<,>} only; Sigma^R is rebuilt from the mixed
pair in the SCBA loop via :func:`phonon.solver.retarded.build_retarded`.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from phonon_inputs.constants import HBAR_SI, PHPH_SYMMETRY_FACTOR
from .bubble import (
    bubble_chunk_peak_bytes_per_w,
    bubble_dense,
    bubble_dense_from_fft,
    precompute_g_fft,
)


# ---------------------------------------------------------------------------
# Legacy single-slab driver
# ---------------------------------------------------------------------------


def compute_phph_self_energy_finite(
    G_lesser, G_greater, Phi, omega_grid_thz, dw_thz, symmetry_factor=None,
):
    """Phonon-phonon self-energy for a single primitive-cell slab.

    Uses the Gamma-projected vertex ``Phi`` on the slab block; treats
    the slab as fully isolated (no inter-slab coupling). Reproduced
    exactly by
    :func:`compute_phph_self_energy_finite_multi_slab` when invoked
    on a one-slab device with ``dc_handling="zero"``.

    ``symmetry_factor`` (see ``constants.PHPH_SYMMETRY_FACTOR``) defaults
    to the physically-correct ``1/4``; pass ``1.0`` for the legacy
    (Luisier) convention.

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
        phi_left=Phi, phi_right=Phi,
        G_a=G_lesser, G_b=G_lesser,
        n_fft=n_fft, prefactor=prefactor,
        out_slice=freq_sl, zero_freq_idx=mid,
        dc_handling="zero",
    )
    sig_g = bubble_dense(
        phi_left=Phi, phi_right=Phi,
        G_a=G_greater, G_b=G_greater,
        n_fft=n_fft, prefactor=prefactor,
        out_slice=freq_sl, zero_freq_idx=mid,
        dc_handling="zero",
    )
    return sig_l, sig_g


# ---------------------------------------------------------------------------
# Multi-slab driver
# ---------------------------------------------------------------------------


def _filter_g_blocks(
    g_blocks: dict[tuple[int, int], np.ndarray],
    g_cutoff: int | None,
) -> dict[tuple[int, int], np.ndarray]:
    """Drop G blocks with ``|K - K'| > g_cutoff``."""
    if g_cutoff is None:
        return g_blocks
    return {
        key: blk for key, blk in g_blocks.items()
        if abs(key[0] - key[1]) <= g_cutoff
    }


def _build_pair_index(
    phi_blocks: dict[tuple[int, int, int], np.ndarray],
    g_keys: set[tuple[int, int]],
    n_slabs: int,
    *,
    sigma_cutoff: int | None,
) -> dict[tuple[int, int], list]:
    """Enumerate (I, J) → [(K1, K2, K1', K2', phi_left, phi_right), ...].

    For each ``phi_left = phi_blocks[(I, K1, K2)]`` and
    ``phi_right = phi_blocks[(J, K2', K1')]``, the bubble integrand
    contributes only when both ``(K1, K1')`` and ``(K2, K2')`` are
    present in the G dict.
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


def _default_n_threads() -> int:
    """How many worker threads to use for the (I, J) loop.

    Honours ``QUATREX_PHPH_THREADS`` first. Otherwise auto-detects
    via ``os.cpu_count()`` and uses every visible core. The bubble
    kernel releases the GIL during BLAS matmul/FFT, so the workers
    can run in parallel; cap BLAS threads to 1 per worker (via
    ``OMP_NUM_THREADS=1`` / ``OPENBLAS_NUM_THREADS=1``) to avoid
    oversubscription on many-core hosts.
    """
    env = os.environ.get("QUATREX_PHPH_THREADS")
    if env is not None:
        return max(1, int(env))
    return max(1, os.cpu_count() or 1)


def _available_memory_bytes() -> int:
    """Best-effort estimate of the per-process available RAM.

    Resolution order:

    1. ``QUATREX_PHPH_MEMORY_GB`` env var (in GiB).
    2. :mod:`psutil`'s ``virtual_memory().available`` if installed.
    3. ``/proc/meminfo``'s ``MemAvailable`` field on Linux.
    4. Conservative 8 GiB fallback.
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
    *, n_fft: int, nI: int, nJ: int,
    bK1: int, bK1p: int, bK2: int, bK2p: int,
    itemsize: int = 16,
) -> int:
    """Conservative upper bound on the transient memory a single
    bubble call holds at its peak.

    The matmul kernel allocates ``T1``, ``T1_t`` (reshape-forced copy
    after transpose), ``T2``, ``T2_r`` (reshape after transpose), the
    matmul output ``S_hat``, and the inverse-FFT result. ``T1`` and
    ``T2`` dominate; we count the four ``(n_fft, n_dof^3)``-sized
    intermediates as alive simultaneously plus a small slack for
    ``S_hat`` and the IFFT output.
    """
    big = n_fft * nI * max(bK1, bK2p) * max(bK1p, bK2p) * itemsize
    small = n_fft * nI * nJ * itemsize  # S_hat + ifft result
    # Four live big tensors + 2 small outputs + ~25 % slack for
    # transient temporaries inside BLAS / numpy.
    return int(4 * big + 2 * small + 0.25 * (4 * big + 2 * small))


def compute_phph_self_energy_finite_multi_slab(
    g_lesser_blocks: dict[tuple[int, int], np.ndarray],
    g_greater_blocks: dict[tuple[int, int], np.ndarray],
    phi_dev_blocks: dict[tuple[int, int, int], np.ndarray],
    n_slabs: int,
    omega_grid_thz: np.ndarray,
    dw_thz: float,
    *,
    sigma_cutoff: int | None = None,
    g_cutoff: int | None = None,
    dc_handling: str = "interpolate",
    n_threads: int | None = None,
    symmetry_factor: float | None = None,
) -> tuple[
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
]:
    """Full multi-slab 3-phonon self-energy on a Gamma-only finite device.

    Computes ``Sigma^{<,>}_{IJ}(omega)`` over the symmetric omega grid
    used by the dense reference solver. The cubic-vertex support is
    set by ``phi_dev_blocks``; the inner block-quadruple is set by
    that support intersected with ``g_lesser_blocks``/``g_greater_blocks``;
    output ``(I, J)`` pairs are restricted to ``|I - J| <= sigma_cutoff``.

    Parameters
    ----------
    g_lesser_blocks, g_greater_blocks
        Device-resolved G blocks ``{(K, K'): G^x[(n_freq, nK, nKp)]}``.
        Must be in the same z-sorted DOF ordering as ``phi_dev_blocks``.
    phi_dev_blocks
        Device-resolved FC3 vertex ``{(I, K, K'): Phi[n_dof, n_dof, n_dof]}``
        — produced by :func:`phonon.solver.fc3_device.build_device_fc3_blocks`.
    n_slabs
        Number of transport cells in the device.
    omega_grid_thz
        Symmetric ``(-fmax, ..., fmax)`` frequency axis (THz).
    dw_thz
        Grid spacing (THz).
    sigma_cutoff
        Maximum ``|I - J|`` for produced Sigma blocks. ``None`` = no
        truncation (every ``(I, J)`` reachable through the FC3 + G
        support is computed).
    g_cutoff
        Maximum ``|K - K'|`` for G blocks used in the inner sum.
        ``None`` = use every block present in the dict.
    dc_handling
        Forwarded to :func:`bubble_dense` to control the omega = 0
        sample of G. ``"interpolate"`` (default) replaces it with the
        midpoint of its neighbours; ``"zero"`` reproduces the legacy
        behaviour; ``"keep"`` leaves it untouched.

    Returns
    -------
    sigma_lesser, sigma_greater
        ``{(I, J): Sigma^x[n_freq, n_I, n_J]}`` dicts. Only entries
        with nonzero contribution are present; the SCBA driver
        scatters them into the full device-sized Sigma buffers.
    """
    n_freq = len(omega_grid_thz)
    n_fft = 2 * n_freq - 1
    mid = n_freq // 2
    freq_sl = slice(mid, mid + n_freq)
    if symmetry_factor is None:
        symmetry_factor = PHPH_SYMMETRY_FACTOR
    prefactor = symmetry_factor * 0.5j * HBAR_SI * dw_thz / (2 * np.pi)

    gl = _filter_g_blocks(g_lesser_blocks, g_cutoff)
    gg = _filter_g_blocks(g_greater_blocks, g_cutoff)
    g_keys = set(gl.keys())
    if set(gg.keys()) != g_keys:
        raise ValueError(
            "g_lesser_blocks and g_greater_blocks must share the same "
            f"(K, K') keys; lesser-only={g_keys - set(gg.keys())}, "
            f"greater-only={set(gg.keys()) - g_keys}"
        )

    # Pre-FFT every distinct G block once: every (I, J) inner loop
    # touches the same handful of G blocks many times, so this saves
    # O(n_pairs) redundant FFTs per SCBA iter.
    gl_fft = {
        key: precompute_g_fft(
            blk, n_fft=n_fft, zero_freq_idx=mid, dc_handling=dc_handling,
        )
        for key, blk in gl.items()
    }
    gg_fft = {
        key: precompute_g_fft(
            blk, n_fft=n_fft, zero_freq_idx=mid, dc_handling=dc_handling,
        )
        for key, blk in gg.items()
    }

    pair_index = _build_pair_index(
        phi_dev_blocks, g_keys, n_slabs, sigma_cutoff=sigma_cutoff,
    )

    tasks: list[tuple] = []
    for (I, J), pairs in pair_index.items():
        for K1, K2, K1p, K2p, phi_left, phi_right in pairs:
            tasks.append(
                (I, J, K1, K2, K1p, K2p, phi_left, phi_right, "lesser"),
            )
            tasks.append(
                (I, J, K1, K2, K1p, K2p, phi_left, phi_right, "greater"),
            )

    # --- memory budgeting -------------------------------------------
    n_workers_requested = max(
        1, n_threads if n_threads is not None else _default_n_threads()
    )
    n_workers_by_tasks = max(1, len(tasks))
    per_worker_max_bytes: int | None = None

    if tasks:
        # Uniform-DOF transport cells: every phi/G block has the same
        # shape, so one representative sample fixes the sizing.
        sample_phi_left = next(iter(phi_dev_blocks.values()))
        sample_G = next(iter(gl_fft.values()))
        nI_s, bK1_s, bK2_s = sample_phi_left.shape
        nJ_s = sample_phi_left.shape[0]
        bK1p_s = sample_G.shape[2]
        bK2p_s = bK2_s

        itemsize = 16  # complex128
        full_peak = _bubble_peak_bytes_per_worker(
            n_fft=n_fft, nI=nI_s, nJ=nJ_s,
            bK1=bK1_s, bK1p=bK1p_s, bK2=bK2_s, bK2p=bK2p_s,
        )
        per_w = bubble_chunk_peak_bytes_per_w(
            nI=nI_s, nJ=nJ_s,
            bK1=bK1_s, bK1p=bK1p_s, bK2=bK2_s, bK2p=bK2p_s,
        )
        # Account for the four big transients held per chunk, with
        # the same 25 % slack as the full-peak estimate.
        per_w_peak = int(1.25 * per_w)

        fixed_bytes = (
            sum(g.nbytes for g in gl_fft.values())
            + sum(g.nbytes for g in gg_fft.values())
            + sum(p.nbytes for p in {id(p): p
                                     for p in phi_dev_blocks.values()}.values())
            + 2 * len(pair_index) * n_freq * nI_s * nJ_s * itemsize
        )

        total_budget = int(0.7 * _available_memory_bytes())
        worker_budget = total_budget - fixed_bytes

        if worker_budget < per_w_peak:
            # Even a single one-frequency chunk does not fit alongside
            # the fixed G/phi storage. This is unrecoverable without
            # also chunking the shared dicts — surface a clear error.
            raise MemoryError(
                f"Multi-slab bubble: fixed G/phi storage is "
                f"{fixed_bytes / (1 << 30):.2f} GiB and a single "
                f"frequency chunk needs {per_w_peak / (1 << 30):.3f} GiB, "
                f"but only {total_budget / (1 << 30):.2f} GiB is "
                "available. Reduce n_slabs / vertex_cutoff / g_cutoff, "
                "or set QUATREX_PHPH_MEMORY_GB higher."
            )

        # Largest worker count for which each worker still gets at
        # least a one-frequency chunk.
        n_workers_by_memory = max(1, worker_budget // per_w_peak)
        n_workers = min(
            n_workers_requested, n_workers_by_tasks, n_workers_by_memory,
        )

        per_worker_share = worker_budget // n_workers
        if per_worker_share >= full_peak:
            per_worker_max_bytes = None  # plenty of headroom, no chunking
        else:
            per_worker_max_bytes = int(per_worker_share)
            n_chunks_est = -(-full_peak // max(per_worker_max_bytes, 1))
            print(
                f"  [phph] memory cap: budget="
                f"{total_budget / (1 << 30):.1f} GiB, fixed="
                f"{fixed_bytes / (1 << 30):.2f} GiB, "
                f"workers={n_workers} (requested={n_workers_requested}), "
                f"per-worker={per_worker_share / (1 << 30):.2f} GiB; "
                f"omega-chunking each bubble into ~{n_chunks_est} "
                f"chunk(s)",
                flush=True,
            )
    else:
        n_workers = 1

    def _compute_one_bubble(task):
        I, J, K1, K2, K1p, K2p, phi_left, phi_right, kind = task
        gx_fft = gl_fft if kind == "lesser" else gg_fft
        blk = bubble_dense_from_fft(
            phi_left=phi_left, phi_right=phi_right,
            G_a_fft=gx_fft[(K1, K1p)],
            G_b_fft=gx_fft[(K2, K2p)],
            ne=n_freq, prefactor=prefactor,
            out_slice=freq_sl,
            max_bytes=per_worker_max_bytes,
        )
        return (I, J, kind), blk

    sl_out: dict[tuple[int, int], np.ndarray] = {}
    sg_out: dict[tuple[int, int], np.ndarray] = {}

    def _accumulate(key_block):
        (I, J, kind), blk = key_block
        target = sl_out if kind == "lesser" else sg_out
        existing = target.get((I, J))
        target[(I, J)] = blk if existing is None else existing + blk

    if n_workers == 1 or len(tasks) <= 1:
        for t in tasks:
            _accumulate(_compute_one_bubble(t))
    else:
        # threadpoolctl caps BLAS threads inside the parallel region
        # so we don't over-subscribe the host (each worker still
        # benefits from BLAS-level parallelism for the matmuls, but
        # only with as many threads as we have unused cores).
        try:
            from threadpoolctl import threadpool_limits
            cm = threadpool_limits(limits=1, user_api="blas")
        except ImportError:
            class _Noop:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            cm = _Noop()
        with cm, ThreadPoolExecutor(max_workers=n_workers) as pool:
            for kb in pool.map(_compute_one_bubble, tasks):
                _accumulate(kb)

    return sl_out, sg_out
