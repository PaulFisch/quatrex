# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Serialization of the TENSOR-DECOMPOSED 3-phonon device vertex for the
transversely-periodic (k>1) anharmonic self-energy.

Instead of the dense q-folded dict {(iq1, iq2): {(I, K, K'): Phi[b,b,b]}}
(O(N_q^2) pairs, GB-scale, replicated per rank), the factored representation
stores the exact per-leg factorisation of the same folded blocks:

    Phi~(q1, q2)[(I, K, K')][a, b, c]
        = sum_r lam_r * D[a, r] * UB[K-I][iq1][b, r] * UC[K'-I][iq2][c, r]

with D real (unphased external leg) and UB/UC per-transport-offset,
per-transverse-momentum device factor arrays (complex; UB is UC for the
S2-symmetric INDSCAL ansatz). Total size O(n_off * N_q * n_dof * R) --
tens of MB where the dense dict is GBs.

Produced offline by phonon/phonon_inputs/fc3_factor_device.py via the
input builder; consumed by SigmaPhononPhonon's factored coupled-q branch.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qttools import NDArray

FORMAT_VERSION = 1


@dataclass
class VertexFactors:
    """The factored q-folded device vertex."""

    D: NDArray            # (n_dof, R) real -- external leg
    lambdas: NDArray      # (R,) -- CP weights, sorted by descending weight
    offsets: NDArray      # (n_off,) int -- transport offsets d (min. image)
    UB: NDArray           # (n_off, N_q, n_dof, R) complex -- leg b at q1
    UC: NDArray           # (n_off, N_q, n_dof, R) complex -- leg c at q2
    q_diff_map: NDArray   # (N_q, N_q) int
    nk_shape: tuple[int, ...]
    ansatz: str
    meta: dict

    @property
    def rank(self) -> int:
        return int(self.lambdas.shape[0])

    @property
    def n_kpts(self) -> int:
        return int(self.UB.shape[1])

    def offset_index(self) -> dict[int, int]:
        return {int(d): i for i, d in enumerate(self.offsets)}

    def truncate(self, rank: int) -> "VertexFactors":
        """Keep the leading ``rank`` components (columns are weight-sorted).

        The factors are COPIED, not sliced: a view would keep the full-rank
        arrays alive, so `sse_vertex_rank` would free no memory.

        CP/INDSCAL is NOT nested: the best rank-R fit is not the leading R
        columns of a higher-rank fit, and weight-sorting orders the components
        without making a prefix of them optimal. Truncating the R=128 film
        factors is 4.4x (R=8) to 83x (R=64) further from the reference than a
        dedicated rank-R fit. Prefer one fit per rank (they are cached on the
        bulk-FC3 hash, so this is free); truncation is a memory knob, not a
        substitute for refitting.
        """
        if rank <= 0 or rank >= self.rank:
            return self
        warnings.warn(
            f"truncating a rank-{self.rank} {self.ansatz} fit to rank {rank}: "
            "CP is not nested, so this is a strictly worse vertex than a "
            f"dedicated rank-{rank} fit. Use decomposed_vertices_r{rank}.npz "
            "if it exists.",
            stacklevel=2,
        )
        return VertexFactors(
            D=self.D[:, :rank].copy(), lambdas=self.lambdas[:rank].copy(),
            offsets=self.offsets, UB=self.UB[..., :rank].copy(),
            UC=self.UC[..., :rank].copy(), q_diff_map=self.q_diff_map,
            nk_shape=self.nk_shape, ansatz=self.ansatz,
            meta={**self.meta, "truncated_to": rank},
        )

    def reconstruct_block(self, iq1: int, iq2: int, dK: int, dKp: int) -> NDArray:
        """Dense Phi~(q1,q2) block for offsets (K-I, K'-I) -- tests/self-checks."""
        pos = self.offset_index()
        ub = self.UB[pos[int(dK)], iq1]      # (n_dof, R)
        uc = self.UC[pos[int(dKp)], iq2]     # (n_dof, R)
        return np.einsum("r,ar,br,cr->abc", self.lambdas, self.D, ub, uc)


def save_decomposed(path: str | Path, vf: VertexFactors) -> None:
    """Write the factored vertex to ``path`` (.npz)."""
    np.savez_compressed(
        str(path),
        format_version=np.int64(FORMAT_VERSION),
        D=np.asarray(vf.D, dtype=np.float64),
        lambdas=np.asarray(vf.lambdas, dtype=np.float64),
        offsets=np.asarray(vf.offsets, dtype=np.int64),
        UB=np.asarray(vf.UB, dtype=np.complex128),
        UC=np.asarray(vf.UC, dtype=np.complex128),
        q_diff_map=np.asarray(vf.q_diff_map, dtype=np.int64),
        nk_shape=np.asarray(vf.nk_shape, dtype=np.int64),
        ansatz=str(vf.ansatz),
        meta=np.array(vf.meta, dtype=object),
    )


def load_decomposed(path: str | Path, rank: int = 0) -> VertexFactors:
    """Load the factored vertex; ``rank > 0`` truncates to the leading terms."""
    npz = np.load(str(path), allow_pickle=True)
    version = int(npz["format_version"])
    if version != FORMAT_VERSION:
        raise ValueError(
            f"decomposed-vertex format {version} != supported {FORMAT_VERSION}"
        )
    vf = VertexFactors(
        D=np.asarray(npz["D"], dtype=np.float64),
        lambdas=np.asarray(npz["lambdas"], dtype=np.float64),
        offsets=np.asarray(npz["offsets"], dtype=np.int64),
        UB=np.asarray(npz["UB"], dtype=np.complex128),
        UC=np.asarray(npz["UC"], dtype=np.complex128),
        q_diff_map=np.asarray(npz["q_diff_map"], dtype=int),
        nk_shape=tuple(int(k) for k in npz["nk_shape"]),
        ansatz=str(npz["ansatz"]),
        meta=dict(npz["meta"].item()) if "meta" in npz.files else {},
    )
    return vf.truncate(rank) if rank else vf


def reblock_decomposed(vf: VertexFactors, cells_per_block: int) -> VertexFactors:
    r"""Lift primitive-cell factors into an exact supercell factorisation.

    If a supercell contains ``c`` primitive cells, primitive offset
    ``delta`` becomes ``c*Delta + v - u`` for external subcell ``u`` and
    internal subcell ``v``.  Replicating every CP component once per external
    subcell therefore represents every reblocked dense vertex block exactly
    relative to ``vf``.  The rank grows from ``R`` to ``c*R``; there is no
    refit and no new FC3 approximation.

    ``support_pairs`` cannot in general be lifted by independent per-leg
    factors because its admissibility may couple the two internal offsets.
    Current production Si factors have no such mask.  Refuse that uncommon
    case rather than manufacture unsupported blocks.
    """
    c = int(cells_per_block)
    if c < 1:
        raise ValueError("cells_per_block must be positive")
    if c == 1:
        return vf
    if vf.meta.get("support_pairs") is not None:
        raise NotImplementedError(
            "reblocking factors with a coupled support_pairs mask requires "
            "component-wise pair support")

    primitive_offsets = [int(x) for x in vf.offsets]
    offset_pos = vf.offset_index()
    # All supercell offsets for which at least one (u,v) maps to a stored
    # primitive offset.
    super_offsets = sorted({
        delta
        for u in range(c)
        for v in range(c)
        for delta in range(
            int(np.floor((min(primitive_offsets) - (v - u)) / c)),
            int(np.ceil((max(primitive_offsets) - (v - u)) / c)) + 1)
        if c * delta + v - u in offset_pos
    })
    d, rank = int(vf.D.shape[0]), int(vf.rank)
    nq = int(vf.n_kpts)
    lifted_rank = c * rank
    D = np.zeros((c * d, lifted_rank), dtype=np.asarray(vf.D).dtype)
    lambdas = np.tile(np.asarray(vf.lambdas), c)
    UB = np.zeros((len(super_offsets), nq, c * d, lifted_rank),
                  dtype=np.asarray(vf.UB).dtype)
    UC = np.zeros_like(UB, dtype=np.asarray(vf.UC).dtype)

    for u in range(c):
        rr = slice(u * rank, (u + 1) * rank)
        D[u * d:(u + 1) * d, rr] = np.asarray(vf.D)
        for ids, delta in enumerate(super_offsets):
            for v in range(c):
                primitive_delta = c * delta + v - u
                ip = offset_pos.get(primitive_delta)
                if ip is None:
                    continue
                rows = slice(v * d, (v + 1) * d)
                UB[ids, :, rows, rr] = np.asarray(vf.UB)[ip]
                UC[ids, :, rows, rr] = np.asarray(vf.UC)[ip]

    return VertexFactors(
        D=D, lambdas=lambdas,
        offsets=np.asarray(super_offsets, dtype=np.int64),
        UB=UB, UC=UC, q_diff_map=np.asarray(vf.q_diff_map),
        nk_shape=tuple(vf.nk_shape), ansatz=f"{vf.ansatz}-reblock{c}",
        meta={
            **vf.meta,
            "primitive_rank": rank,
            "primitive_n_dof": d,
            "cells_per_block": c,
            "rank": lifted_rank,
            "n_dof": c * d,
            "reblock_exact_relative_to_primitive_factors": True,
        },
    )
