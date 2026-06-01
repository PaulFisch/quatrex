# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Anharmonic 3-phonon scattering self-energy

Implements
    Sigma^{<,>}_{ab}(omega) = (i hbar / 2) * sum_{c,d,e,f}
        Phi_{a c d}
        * [ G^{<,>}_{cf} * G^{<,>}_{de} ](omega)
        * Phi_{b e f}

The cubic vertex Phi is supplied as a block-sparse dict
(see quatrex.phonon.fc3_loader). Internal units are THz / THz^2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, xp
from qttools.comm import comm as ranks
from qttools.datastructures import DSDBSparse
from qttools.profiling import Profiler

from quatrex.core.config import QuatrexConfig
from quatrex.core.fft_utils import hilbert_transform
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.phonon.bubble import bubble_dense
from quatrex.phonon.fc3_loader import (
    PhiBlocks,
    load_device_fc3,
)
from quatrex.phonon.units import bubble_prefactor_thz

profiler = Profiler()


class SigmaPhononPhonon(ScatteringSelfEnergy):
    """3-phonon SCBA scattering self-energy.

    Parameters
    ----------
    config : QuatrexConfig
        Quatrex configuration object. Reads ``config.phonon.fc3_path``
        and ``config.phonon.retarded_method``.
    phonon_frequencies : NDArray
        The phonon frequency grid in **THz** (uniform). Local slice for
        the calling rank — the full axis is recovered via
        ``comm.stack.all_gather_v`` inside :meth:`compute`.
    block_sizes : NDArray
        Transport-cell DOF sizes ``(N_blocks,)``.
    phi_blocks : PhiBlocks, optional
        Pre-built block-sparse Phi dict in THz^2. If not provided, the
        loader is invoked on ``config.phonon.fc3_path``.
    """

    def __init__(
        self,
        config: QuatrexConfig,
        phonon_frequencies: NDArray,
        block_sizes: NDArray,
        phi_blocks: PhiBlocks | None = None,
    ) -> None:
        # Local energy slice; the full axis is gathered in compute().
        self.local_frequencies = np.asarray(phonon_frequencies)

        self.block_sizes = np.asarray(block_sizes, dtype=int)
        self.n_blocks = int(self.block_sizes.shape[0])
        self.block_offsets = np.concatenate(([0], np.cumsum(self.block_sizes)))

        retarded_method = getattr(config.phonon, "retarded_method", "fft")
        if retarded_method not in ("half", "fft"):
            raise ValueError(
                f"Unknown retarded_method={retarded_method!r}; "
                "use 'half' or 'fft'."
            )
        self.retarded_method = retarded_method

        if phi_blocks is None:
            fc3_path = getattr(config.phonon, "fc3_path", None)
            if fc3_path is None:
                raise ValueError(
                    "config.phonon.fc3_path must be set when phi_blocks "
                    "is not provided to SigmaPhononPhonon."
                )
            phi_blocks = load_device_fc3(
                Path(fc3_path),
                block_sizes=self.block_sizes,
            )
        self.phi_blocks = phi_blocks

        # Precompute pair index: for each Sigma block (I, J)
        # collect the list of inner block-pairs (K1, K2) that
        # contribute to it. Computed once.
        self._phi_pair_index: dict[
            tuple[int, int], list[tuple[int, int, NDArray, NDArray]]
        ] = {}
        for (I, K1, K2), phi_left in self.phi_blocks.items():
            for J in range(
                max(0, I - 1), min(self.n_blocks, I + 2)
            ):
                phi_right = self.phi_blocks.get((J, K2, K1))
                if phi_right is None:
                    continue
                self._phi_pair_index.setdefault((I, J), []).append(
                    (K1, K2, phi_left, phi_right)
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @profiler.profile(label="SigmaPhononPhonon", level="default", comm=comm)
    def compute(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
    ) -> None:
        """Compute the 3-phonon self-energy contribution.

        The bubble runs in ``stack`` distribution (each rank holds an
        omega slice of every BT-band block). Only the diagonal blocks
        $g_{KK}$ enter the contraction, so only those are gathered to
        full ω; off-diagonal $G$ blocks are never communicated. The
        ``(I,J)`` output loop is split across ``comm.block``: each
        rank computes only the ``(I,J)`` outputs whose row index
        ``I`` lies in its local block range.

        The input distribution state of every buffer is restored on
        exit, so the routine is composable with both the legacy
        ``stack``-state SCBA loop and the production
        ``stack`` → ``nnz`` → ``stack`` SCBA pattern.

        Parameters
        ----------
        g_lesser, g_greater
            Phonon Green's functions.
        out
            ``(sigma_lesser, sigma_greater, sigma_retarded)`` —
            additive: this routine adds its contribution into the
            existing buffers.
        """
        sigma_lesser, sigma_greater, sigma_retarded = out

        all_bufs = (
            g_lesser, g_greater, sigma_lesser, sigma_greater, sigma_retarded,
        )

        # Remember each buffer's incoming distribution state so we
        # can restore it on exit. ``dtranspose`` toggles state
        # unconditionally, so we must keep track ourselves.
        incoming_states = {id(m): m.distribution_state for m in all_bufs}

        # Move to stack distribution: the FFT-along-ω needs the full
        # axis on a single rank, which is what ``stack`` provides
        # after the diagonal-block all-gather below.
        for m in all_bufs:
            if m.distribution_state != "stack":
                m.dtranspose()

        try:
            self._compute_in_stack(
                g_lesser, g_greater,
                sigma_lesser, sigma_greater, sigma_retarded,
            )
        finally:
            # Restore incoming distribution states. ``dtranspose``
            # toggles unconditionally, so if the buffer was nnz on
            # entry, calling ``dtranspose`` once now takes it back
            # from stack to nnz.
            for m in all_bufs:
                if m.distribution_state != incoming_states[id(m)]:
                    m.dtranspose()

    # ------------------------------------------------------------------
    # Internals — stack-state compute path
    # ------------------------------------------------------------------

    def _compute_in_stack(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        sigma_lesser: DSDBSparse,
        sigma_greater: DSDBSparse,
        sigma_retarded: DSDBSparse,
    ) -> None:
        """Bubble + write-back, assuming all buffers are in stack."""
        # Local block range (global indices) for this comm.block rank.
        block_start, block_end = self._block_range(g_lesser)
        num_local_blocks = block_end - block_start

        # Gather diagonal G blocks at full ω across comm.stack, then
        # (when comm.block.size > 1) collect them across comm.block so
        # every rank has every diagonal block at full ω.
        gl_diag, gg_diag = self._gather_diagonal_blocks(
            g_lesser, g_greater, block_start, num_local_blocks,
        )
        if not gl_diag:
            return  # nothing local to do

        # Full-ω axis recovered from the gather, plus the frequency
        # axis itself (needed for the Hilbert transform that closes
        # the retarded SSE).
        ne_full = next(iter(gl_diag.values())).shape[0]
        full_freqs = self._gather_frequencies(ne_full)
        n_fft = 2 * ne_full - 1
        prefactor = bubble_prefactor_thz(
            float(full_freqs[1] - full_freqs[0])
        )

        # Local ω slice: we accumulate at full ω on this rank, then
        # write only the slice owned by this comm.stack rank.
        e_lo = int(np.sum(g_lesser.stack_section_sizes[: ranks.stack.rank]))
        ne_local = int(self.local_frequencies.shape[0])
        e_hi = e_lo + ne_local

        sl_view = sigma_lesser.stack[...]
        sg_view = sigma_greater.stack[...]
        sr_view = sigma_retarded.stack[...]

        for (I_glob, J_glob), pairs in self._phi_pair_index.items():
            # Distribute (I, J) outputs across comm.block: each rank
            # only handles outputs whose row I is in its block range.
            if not (block_start <= I_glob < block_end):
                continue

            # Output indices are local-to-rank. The block indexer
            # accepts col indices in [0, num_blocks - block_start);
            # filter to keep only outputs that resolve to a valid
            # local addressable block.
            J_loc = J_glob - block_start
            if J_loc < 0 or J_loc >= len(g_lesser.local_block_sizes):
                continue
            I_loc = I_glob - block_start

            sl_full = None
            sg_full = None
            for K1, K2, phi_left, phi_right in pairs:
                if K1 not in gl_diag or K2 not in gl_diag:
                    continue
                sl = self._bubble_block(
                    phi_left=phi_left, phi_right=phi_right,
                    G_inner_a=gl_diag[K1], G_inner_b=gl_diag[K2],
                    n_fft=n_fft, prefactor=prefactor,
                )
                sg = self._bubble_block(
                    phi_left=phi_left, phi_right=phi_right,
                    G_inner_a=gg_diag[K1], G_inner_b=gg_diag[K2],
                    n_fft=n_fft, prefactor=prefactor,
                )
                sl_full = sl if sl_full is None else sl_full + sl
                sg_full = sg if sg_full is None else sg_full + sg

            if sl_full is None:
                continue

            delta = sg_full - sl_full
            sr_full = 0.5 * delta
            if self.retarded_method == "fft":
                sr_full = sr_full + 0.5j * hilbert_transform(delta, full_freqs)

            sl_view.blocks[I_loc, J_loc] = (
                sl_view.blocks[I_loc, J_loc] + sl_full[e_lo:e_hi]
            )
            sg_view.blocks[I_loc, J_loc] = (
                sg_view.blocks[I_loc, J_loc] + sg_full[e_lo:e_hi]
            )
            sr_view.blocks[I_loc, J_loc] = (
                sr_view.blocks[I_loc, J_loc] + sr_full[e_lo:e_hi]
            )

    # ------------------------------------------------------------------
    # Internals — block-bubble FFT contraction
    # ------------------------------------------------------------------

    def _bubble_block(
        self,
        *,
        phi_left: NDArray,
        phi_right: NDArray,
        G_inner_a: NDArray,
        G_inner_b: NDArray,
        n_fft: int,
        prefactor: complex,
    ) -> NDArray:
        """FFT 3-phonon bubble for one block triplet pair (THz²)."""
        ne = G_inner_a.shape[0]
        return bubble_dense(
            phi_left=phi_left,
            phi_right=phi_right,
            G_a=G_inner_a,
            G_b=G_inner_b,
            n_fft=n_fft,
            prefactor=prefactor,
            out_slice=slice(0, ne),
            zero_freq_idx=None,
            xp=xp,
        )

    # ------------------------------------------------------------------
    # Internals — distribution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _block_range(m: DSDBSparse) -> tuple[int, int]:
        """Return the [start, end) global-block range owned by this
        ``comm.block`` rank.
        """
        start = int(m.block_section_offsets[ranks.block.rank])
        end = int(m.block_section_offsets[ranks.block.rank + 1])
        return start, end

    def _gather_diagonal_blocks(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        block_start: int,
        num_local_blocks: int,
    ) -> tuple[
        dict[int, NDArray],
        dict[int, NDArray],
    ]:
        """Gather diagonal $G_{KK}$ blocks at full ω on every rank.

        Returns dicts mapping the **global** transport-cell index ``K``
        to a ``(ne_full, n_K, n_K)`` array. Only diagonal blocks are
        gathered: the 3-phonon bubble (\\cref{eq:phph_pair_index})
        consumes ``G[(K1, K1)]`` and ``G[(K2, K2)]`` exclusively, so
        BT-off-diagonal $G$ blocks never enter the contraction.

        For ``comm.block.size > 1`` we additionally
        ``comm.block.all_gather`` the locally-gathered diagonals so
        every block rank can address every global ``K`` (the FC3
        block-pair index may reference $K$ outside this rank's range).
        Currently requires uniform block sizes; mixed sizes raise a
        clear ``NotImplementedError``.
        """
        gl_view = g_lesser.stack[...]
        gg_view = g_greater.stack[...]

        # Local diagonals at full ω.
        gl_local: dict[int, NDArray] = {}
        gg_local: dict[int, NDArray] = {}
        for K_loc in range(num_local_blocks):
            K_glob = block_start + K_loc
            local_l = xp.asarray(gl_view.blocks[K_loc, K_loc])
            local_g = xp.asarray(gg_view.blocks[K_loc, K_loc])
            if ranks.stack.size == 1:
                gl_local[K_glob] = local_l
                gg_local[K_glob] = local_g
            else:
                gl_local[K_glob] = ranks.stack.all_gather_v(
                    local_l, axis=0, mask=g_lesser._stack_padding_mask,
                )
                gg_local[K_glob] = ranks.stack.all_gather_v(
                    local_g, axis=0, mask=g_greater._stack_padding_mask,
                )

        if ranks.block.size == 1:
            return gl_local, gg_local

        # Cross-rank gather across comm.block. Currently uniform block
        # sizes only (consistent with how SCBAData allocates them).
        block_sizes_arr = np.asarray(self.block_sizes)
        if int(block_sizes_arr.min()) != int(block_sizes_arr.max()):
            raise NotImplementedError(
                "Non-uniform block sizes are not yet supported in the "
                "multi-block-rank phonon-phonon SSE distribution path. "
                "Run with a single comm.block rank or use uniform "
                "block sizes (SCBAData enforces uniformity by default)."
            )
        nbs = int(block_sizes_arr[0])
        ne_full = next(iter(gl_local.values())).shape[0]

        # Pack local diagonals into a contiguous tensor, gather across
        # comm.block, then re-key by global block index.
        local_l_stack = xp.stack(
            [gl_local[block_start + K_loc] for K_loc in range(num_local_blocks)],
            axis=0,
        )
        local_g_stack = xp.stack(
            [gg_local[block_start + K_loc] for K_loc in range(num_local_blocks)],
            axis=0,
        )
        global_l_stack = ranks.block.all_gather_v(local_l_stack, axis=0)
        global_g_stack = ranks.block.all_gather_v(local_g_stack, axis=0)
        # Expected shape after gather: (n_blocks, ne_full, nbs, nbs).
        assert global_l_stack.shape[0] == self.n_blocks, (
            f"comm.block gather produced {global_l_stack.shape[0]} blocks; "
            f"expected {self.n_blocks}"
        )
        del nbs  # only used by the assert above; silence linter

        gl_diag = {
            K: global_l_stack[K] for K in range(self.n_blocks)
        }
        gg_diag = {
            K: global_g_stack[K] for K in range(self.n_blocks)
        }
        return gl_diag, gg_diag

    def _gather_frequencies(self, ne_full: int) -> NDArray:
        """All-gather the local frequency slice into the full axis."""
        if ranks.stack.size == 1:
            assert self.local_frequencies.shape[0] == ne_full
            return self.local_frequencies
        return ranks.stack.all_gather_v(
            xp.asarray(self.local_frequencies, dtype=float),
            axis=0,
        )
