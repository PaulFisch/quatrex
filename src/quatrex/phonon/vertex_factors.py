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
        """Keep the leading ``rank`` components (columns are weight-sorted)."""
        if rank <= 0 or rank >= self.rank:
            return self
        return VertexFactors(
            D=self.D[:, :rank], lambdas=self.lambdas[:rank],
            offsets=self.offsets, UB=self.UB[..., :rank],
            UC=self.UC[..., :rank], q_diff_map=self.q_diff_map,
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
