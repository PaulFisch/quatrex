# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Anharmonic 3-phonon scattering self-energy

Implements
    Sigma^{<,>}_{ab}(omega) = (i hbar / 2) * sum_{c,d,e,f}
        Phi_{a c d}
        * [ G^{<,>}_{cf} * G^{<,>}_{de} ](omega)
        * Phi_{b e f}

The cubic vertex Phi is supplied as a block-sparse dict
(see quatrex.phonon.fc3_loader); the convolution is performed via
FFT along the energy axis. Internal units are THz / THz^2
(see quatrex.phonon.units), matching the standalone reference at
phonon/phonon_inputs/anharmonic.py.
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
        Pre-built block-sparse Phi dict in THz². If not provided, the
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

        Parameters
        ----------
        g_lesser, g_greater
            Phonon Green's functions; will be operated on in stack
            distribution.
        out
            ``(sigma_lesser, sigma_greater, sigma_retarded)`` —
            additive: this routine adds its contribution.
        """
        sigma_lesser, sigma_greater, sigma_retarded = out

        # Ensure we are in stack distribution (matches Dyson solver).
        for m in (
            g_lesser, g_greater, sigma_lesser, sigma_greater, sigma_retarded,
        ):
            if m.distribution_state != "stack":
                m.dtranspose()

        # Gather full ω-axis for every BTD G block
        gl_full, gg_full = self._gather_full_btd(g_lesser, g_greater)

        # Local axis for output assignment + Hilbert transform.
        ne_local = int(self.local_frequencies.shape[0])
        # Full ω-axis recovered from the gather.
        ne_full = next(iter(gl_full.values())).shape[0]
        full_freqs = self._gather_frequencies(ne_full)
        n_fft = 2 * ne_full - 1
        prefactor = bubble_prefactor_thz(
            float(full_freqs[1] - full_freqs[0])
        )

        # Compute Σ^{<,>} blocks at the full w resolution on every
        # rank, then slice back to the local energy chunk.
        sl_full: dict[tuple[int, int], NDArray] = {}
        sg_full: dict[tuple[int, int], NDArray] = {}
        for (I, J), pairs in self._phi_pair_index.items():
            for K1, K2, phi_left, phi_right in pairs:
                key_a = (K1, K1)
                key_b = (K2, K2)
                if key_a not in gl_full or key_b not in gl_full:
                    continue  # skip if G block is missing (shouldn't happen)
                self._accumulate_pair(
                    I=I, J=J,
                    phi_left=phi_left, phi_right=phi_right,
                    gl_a=gl_full[key_a], gl_b=gl_full[key_b],
                    gg_a=gg_full[key_a], gg_b=gg_full[key_b],
                    sl_out=sl_full, sg_out=sg_full,
                    n_fft=n_fft, prefactor=prefactor,
                )

        # Sigma^R from the bosonic Hilbert transform on the FULL axis,
        # so the integral is consistent across ranks.
        sr_full: dict[tuple[int, int], NDArray] = {}
        for key in set(sl_full) | set(sg_full):
            sl = sl_full.get(key)
            sg = sg_full.get(key)
            if sl is None:
                sl = xp.zeros_like(sg)
            if sg is None:
                sg = xp.zeros_like(sl)
            delta = sg - sl
            sr = 0.5 * delta
            if self.retarded_method == "fft":
                sr = sr + 0.5j * hilbert_transform(delta, full_freqs)
            sr_full[key] = sr

        # Slice back to local energy chunk and write into the
        # DSDBSparse outputs (additive).
        e_lo = self._local_energy_offset(ne_local, ne_full)
        e_hi = e_lo + ne_local
        sigma_lesser_view = sigma_lesser.stack[...]
        sigma_greater_view = sigma_greater.stack[...]
        sigma_retarded_view = sigma_retarded.stack[...]
        for (I, J), block in sl_full.items():
            sigma_lesser_view.blocks[I, J] = (
                sigma_lesser_view.blocks[I, J] + block[e_lo:e_hi]
            )
        for (I, J), block in sg_full.items():
            sigma_greater_view.blocks[I, J] = (
                sigma_greater_view.blocks[I, J] + block[e_lo:e_hi]
            )
        for (I, J), block in sr_full.items():
            sigma_retarded_view.blocks[I, J] = (
                sigma_retarded_view.blocks[I, J] + block[e_lo:e_hi]
            )

    # ------------------------------------------------------------------
    # Internals — block-bubble FFT contraction
    # ------------------------------------------------------------------

    def _accumulate_pair(
        self,
        *,
        I: int,
        J: int,
        phi_left: NDArray,
        phi_right: NDArray,
        gl_a: NDArray,
        gl_b: NDArray,
        gg_a: NDArray,
        gg_b: NDArray,
        sl_out: dict[tuple[int, int], NDArray],
        sg_out: dict[tuple[int, int], NDArray],
        n_fft: int,
        prefactor: complex,
    ) -> None:
        for gx_a, gx_b, sx_out in (
            (gl_a, gl_b, sl_out),
            (gg_a, gg_b, sg_out),
        ):
            sigma_block = self._bubble_block(
                phi_left=phi_left,
                phi_right=phi_right,
                G_inner_a=gx_a,
                G_inner_b=gx_b,
                n_fft=n_fft,
                prefactor=prefactor,
            )
            key = (I, J)
            if key not in sx_out:
                sx_out[key] = xp.zeros_like(sigma_block)
            sx_out[key] = sx_out[key] + sigma_block

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

    def _gather_full_btd(
        self, g_lesser: DSDBSparse, g_greater: DSDBSparse
    ) -> tuple[
        dict[tuple[int, int], NDArray],
        dict[tuple[int, int], NDArray],
    ]:
        """All-gather BTD G blocks across ``comm.stack``.

        Returns dicts mapping ``(I, J)`` (with ``|I-J| <= 1``) to a
        ``(ne_full, b_I, b_J)`` array, replicated on every rank.
        """
        gl_view = g_lesser.stack[...]
        gg_view = g_greater.stack[...]

        if ranks.stack.size == 1:
            out_l = {}
            out_g = {}
            for I in range(self.n_blocks):
                for dJ in (-1, 0, 1):
                    J = I + dJ
                    if 0 <= J < self.n_blocks:
                        out_l[(I, J)] = xp.asarray(gl_view.blocks[I, J])
                        out_g[(I, J)] = xp.asarray(gg_view.blocks[I, J])
            return out_l, out_g

        out_l: dict[tuple[int, int], NDArray] = {}
        out_g: dict[tuple[int, int], NDArray] = {}
        for I in range(self.n_blocks):
            for dJ in (-1, 0, 1):
                J = I + dJ
                if not (0 <= J < self.n_blocks):
                    continue
                local_l = xp.asarray(gl_view.blocks[I, J])
                local_g = xp.asarray(gg_view.blocks[I, J])
                full_l = ranks.stack.all_gather_v(
                    local_l,
                    axis=0,
                    mask=g_lesser._stack_padding_mask,
                )
                full_g = ranks.stack.all_gather_v(
                    local_g,
                    axis=0,
                    mask=g_greater._stack_padding_mask,
                )
                out_l[(I, J)] = full_l
                out_g[(I, J)] = full_g
        return out_l, out_g

    def _gather_frequencies(self, ne_full: int) -> NDArray:
        """All-gather the local frequency slice into the full axis."""
        if ranks.stack.size == 1:
            assert self.local_frequencies.shape[0] == ne_full
            return self.local_frequencies
        return ranks.stack.all_gather_v(
            xp.asarray(self.local_frequencies, dtype=float),
            axis=0,
        )

    def _local_energy_offset(self, ne_local: int, ne_full: int) -> int:
        """Offset of this rank's local energy slice in the full axis.

        Falls back to a contiguous-equal-split assumption if no
        explicit offset can be queried; this matches how
        ``get_electron_energies`` partitions linspace grids in
        production.
        """
        if ranks.stack.size == 1:
            return 0
        # Contiguous equal split (matches grid.energies handling).
        rank = ranks.stack.rank
        return rank * (ne_full // ranks.stack.size)
