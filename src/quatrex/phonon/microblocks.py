# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Primitive-cell views of grouped phonon transport blocks.

The Dyson equation is cheapest when several primitive transport cells are
grouped into one block.  The cubic force-constant tensor need not use the same
partition.  This module supplies the exact index map used by the production
phonon-phonon self-energy to contract primitive FC3 blocks against slices of
the grouped Green function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qttools import NDArray


@dataclass(frozen=True)
class MicroblockLayout:
    """Map uniform primitive microblocks onto possibly unequal Dyson blocks."""

    block_sizes: tuple[int, ...]
    micro_dof: int
    micro_offsets: tuple[int, ...]
    micro_to_block: tuple[int, ...]
    micro_local_index: tuple[int, ...]

    @classmethod
    def from_block_sizes(
        cls, block_sizes, micro_dof: int
    ) -> "MicroblockLayout":
        sizes = tuple(int(v) for v in np.asarray(block_sizes, dtype=int))
        d = int(micro_dof)
        if d <= 0:
            raise ValueError("sse_microblock_dof must be positive when enabled")
        bad = [v for v in sizes if v <= 0 or v % d]
        if bad:
            raise ValueError(
                "Every Dyson block must contain an integer number of "
                f"{d}-DOF SSE microblocks; incompatible sizes are {bad}."
            )

        counts = [v // d for v in sizes]
        offsets = tuple(int(v) for v in np.concatenate(([0], np.cumsum(counts))))
        to_block: list[int] = []
        local: list[int] = []
        for block, count in enumerate(counts):
            to_block.extend([block] * count)
            local.extend(range(count))
        return cls(sizes, d, offsets, tuple(to_block), tuple(local))

    @property
    def n_blocks(self) -> int:
        return len(self.block_sizes)

    @property
    def n_microblocks(self) -> int:
        return len(self.micro_to_block)

    @property
    def cells_per_block(self) -> tuple[int, ...]:
        return tuple(v // self.micro_dof for v in self.block_sizes)

    def locate(self, micro: int) -> tuple[int, slice]:
        m = int(micro)
        if not 0 <= m < self.n_microblocks:
            raise IndexError(
                f"microblock {m} outside [0, {self.n_microblocks})"
            )
        block = self.micro_to_block[m]
        lo = self.micro_local_index[m] * self.micro_dof
        return block, slice(lo, lo + self.micro_dof)

    def grouped_pair(self, left: int, right: int) -> tuple[int, int]:
        return self.micro_to_block[int(left)], self.micro_to_block[int(right)]

    def slice_link(self, block: NDArray, left: int, right: int) -> NDArray:
        """Return the primitive matrix view from one grouped matrix block."""
        _bi, si = self.locate(left)
        _bj, sj = self.locate(right)
        return block[..., si, sj]


@dataclass(frozen=True)
class MicroQuad:
    """One primitive FC3 bubble term before frequency convolution."""

    out_left: int
    out_right: int
    k1: int
    k2: int
    k1p: int
    k2p: int
    phi_left: NDArray
    phi_right: NDArray


def build_micro_pair_index(
    phi_blocks: dict[tuple[int, int, int], NDArray],
    layout: MicroblockLayout,
    g_band: int,
) -> tuple[
    dict[tuple[int, int], dict[tuple[int, int], list[MicroQuad]]], int, int
]:
    """Index every generated primitive self-energy pair.

    The returned first key is a grouped Dyson output pair.  The second key is
    the primitive output pair inside it.  ``vertex_span`` is the largest
    primitive offset present in either contracted FC3 leg and
    ``sigma_span`` is the largest output separation actually generated.

    A grouped block-tridiagonal Dyson operator can represent same-block and
    adjacent-block output pairs.  If the requested primitive Green-function
    band generates a pair beyond that range, the grouping is incomplete and
    this routine raises instead of discarding the term.
    """
    d = layout.micro_dof
    n = layout.n_microblocks
    b = int(g_band)
    if b < 0:
        raise ValueError("sse_microblock_g_band must be non-negative")

    right_by_links: dict[tuple[int, int], list[tuple[int, NDArray]]] = {}
    vertex_span = 0
    for (i, k1, k2), phi in phi_blocks.items():
        if not (0 <= i < n and 0 <= k1 < n and 0 <= k2 < n):
            raise ValueError(
                f"primitive FC3 key {(i, k1, k2)} is incompatible with "
                f"the {n}-microblock device"
            )
        if tuple(np.shape(phi)) != (d, d, d):
            raise ValueError(
                f"primitive FC3 block {(i, k1, k2)} has shape "
                f"{np.shape(phi)}, expected {(d, d, d)}"
            )
        vertex_span = max(vertex_span, abs(k1 - i), abs(k2 - i))
        # A right vertex is consumed as Phi[(J, K2', K1')].
        right_by_links.setdefault((k2, k1), []).append((i, phi))

    grouped: dict[
        tuple[int, int], dict[tuple[int, int], list[MicroQuad]]
    ] = {}
    sigma_span = 0
    for (i, k1, k2), phi_left in phi_blocks.items():
        for k1p in range(max(0, k1 - b), min(n, k1 + b + 1)):
            for k2p in range(max(0, k2 - b), min(n, k2 + b + 1)):
                for j, phi_right in right_by_links.get((k1p, k2p), ()):
                    bi, bj = layout.grouped_pair(i, j)
                    if abs(bi - bj) > 1:
                        raise ValueError(
                            "The requested primitive bubble generates "
                            f"Sigma[{i},{j}] (distance {abs(i - j)}) across "
                            f"non-adjacent Dyson blocks {bi},{bj}. Increase "
                            "the number of primitive cells per Dyson block."
                        )
                    quad = MicroQuad(
                        i, j, k1, k2, k1p, k2p, phi_left, phi_right
                    )
                    grouped.setdefault((bi, bj), {}).setdefault(
                        (i, j), []
                    ).append(quad)
                    sigma_span = max(sigma_span, abs(i - j))

    return grouped, vertex_span, sigma_span


def micro_link_views(
    grouped: dict[tuple[int, int], NDArray],
    layout: MicroblockLayout,
    links: set[tuple[int, int]],
) -> dict[tuple[int, int], NDArray]:
    """Create zero-copy primitive views for the requested Green links."""
    out = {}
    for k, kp in links:
        bi, _si = layout.locate(k)
        bj, _sj = layout.locate(kp)
        key = (bi, bj)
        if key not in grouped:
            raise KeyError(
                f"grouped Green block {key} required by primitive link "
                f"{(k, kp)} is absent"
            )
        out[(k, kp)] = layout.slice_link(grouped[key], k, kp)
    return out


def grouped_band_indices(
    block_sizes, band: int = 1, start_block: int = 0,
    end_block: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar row/column indices for a complete grouped block band.

    Every scalar entry of a selected block is present.  This is the required
    production sparsity for the microblock self-energy: an atom-distance mask
    can cut legitimate primitive output shells and is also ambiguous for a
    non-orthogonal transport lattice.  ``min(I,J)`` assigns each block pair to
    one arrow-distributed block rank.
    """
    sizes = np.asarray(block_sizes, dtype=int)
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    n = sizes.size
    lo = int(start_block)
    hi = n if end_block is None else int(end_block)
    b = int(band)
    if not (0 <= lo <= hi <= n):
        raise ValueError(f"invalid block section [{lo}, {hi}) for {n} blocks")
    if b < 0:
        raise ValueError("band must be non-negative")
    rows, cols = [], []
    for i in range(n):
        for j in range(max(0, i - b), min(n, i + b + 1)):
            if not lo <= min(i, j) < hi:
                continue
            rr, cc = np.meshgrid(
                np.arange(offsets[i], offsets[i + 1]),
                np.arange(offsets[j], offsets[j + 1]), indexing="ij")
            rows.append(rr.ravel())
            cols.append(cc.ravel())
    return (
        np.concatenate(rows) if rows else np.empty(0, dtype=int),
        np.concatenate(cols) if cols else np.empty(0, dtype=int),
    )
