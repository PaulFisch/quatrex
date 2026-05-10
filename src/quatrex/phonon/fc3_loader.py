# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Block-sparse FC3 container and HDF5 loader.

The 3-phonon vertex Phi_{a,b,c} is local in real space (set by the FC3
cutoff). For a block-tridiagonal device with transport cells indexed by
``I``, all non-zero entries land in block-triplets ``(I, J, K)`` with
``|I-J|, |I-K|, |J-K| <= 1`` (assuming the user keeps the FC3 support
inside one nearest-neighbour shell, see §2.4 of
``docs/anharmonic_phph.tex``). Storing those blocks in a dict avoids
the dense ``O(N_dof^3)`` cost.

Two entry points are exposed:

* :func:`fc3_to_phi_blocks` — dense → block-sparse projection (used by
  tests; emits a Frobenius truncation warning).
* :func:`load_device_fc3` — HDF5 streaming reader. The expected schema
  is the one produced by
  :func:`phonon_inputs.quatrex_writer.write_fc3_blocks`. A legacy
  fallback path still accepts a dense ``(N_dof, N_dof, N_dof)``
  device-sized FC3 stored under ``/fc3``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from qttools import NDArray

PhiBlocks = dict[tuple[int, int, int], NDArray]


def _block_offsets(block_sizes: NDArray) -> NDArray:
    return np.concatenate(([0], np.cumsum(block_sizes)))


def fc3_to_phi_blocks(
    phi_dense: NDArray,
    block_sizes: NDArray,
    *,
    nn_only: bool = True,
    truncation_warn: float = 0.01,
) -> PhiBlocks:
    """Project a dense device Phi onto a block-sparse triplet dict."""
    block_sizes = np.asarray(block_sizes, dtype=int)
    if phi_dense.shape != (block_sizes.sum(),) * 3:
        raise ValueError(
            f"phi_dense.shape={phi_dense.shape} does not match "
            f"({block_sizes.sum()},) * 3"
        )

    offsets = _block_offsets(block_sizes)
    n_blocks = len(block_sizes)

    blocks: PhiBlocks = {}
    kept_norm_sq = 0.0
    total_norm_sq = float(np.vdot(phi_dense.ravel(), phi_dense.ravel()).real)

    for I in range(n_blocks):
        sI = slice(offsets[I], offsets[I + 1])
        for J in range(n_blocks):
            sJ = slice(offsets[J], offsets[J + 1])
            for K in range(n_blocks):
                if nn_only and (
                    abs(I - J) > 1 or abs(I - K) > 1 or abs(J - K) > 1
                ):
                    continue
                sK = slice(offsets[K], offsets[K + 1])
                block = phi_dense[sI, sJ, sK]
                if not np.any(block):
                    continue
                blocks[(I, J, K)] = np.ascontiguousarray(block)
                kept_norm_sq += float(
                    np.vdot(block.ravel(), block.ravel()).real
                )

    if nn_only and total_norm_sq > 0.0:
        dropped_rel = max(0.0, 1.0 - kept_norm_sq / total_norm_sq)
        if dropped_rel > truncation_warn:
            warnings.warn(
                f"FC3 nearest-neighbour truncation dropped "
                f"{dropped_rel:.2%} of the Frobenius norm "
                f"(threshold {truncation_warn:.2%}). Consider enlarging "
                f"the primitive-cells-per-transport-cell factor.",
                stacklevel=2,
            )

    return blocks


def load_device_fc3(
    fc3_path: str | Path,
    *,
    block_sizes: NDArray,
    nn_only: bool = True,
    truncation_warn: float = 0.01,
) -> PhiBlocks:
    """Load a block-sparse, mass-weighted Phi dict from disk.

    Two HDF5 schemas are accepted, in order of preference:

    1. The block-sparse layout written by
       :func:`phonon_inputs.quatrex_writer.write_fc3_blocks`
       (``/fc3_blocks/I_J_K``). This is the canonical production path.
    2. Legacy: a dense ``(N_dof, N_dof, N_dof)`` array under ``/fc3``;
       projected onto the block-tridiagonal pattern via
       :func:`fc3_to_phi_blocks`.
    """
    import h5py

    block_sizes = np.asarray(block_sizes, dtype=int)
    n_dof = int(block_sizes.sum())

    fc3_path = Path(fc3_path)
    with h5py.File(str(fc3_path), "r") as f:
        if "fc3_blocks" in f:
            stored_sizes = np.asarray(f["meta/block_sizes"], dtype=np.int64)
            if list(stored_sizes) != list(block_sizes):
                raise ValueError(
                    f"FC3 file block_sizes={stored_sizes.tolist()} do not "
                    f"match solver block_sizes={block_sizes.tolist()}."
                )
            grp = f["fc3_blocks"]
            phi_blocks: PhiBlocks = {}
            for name in grp.keys():
                ds = grp[name]
                I = int(ds.attrs["I"])
                J = int(ds.attrs["J"])
                K = int(ds.attrs["K"])
                if nn_only and (
                    abs(I - J) > 1 or abs(I - K) > 1 or abs(J - K) > 1
                ):
                    continue
                phi_blocks[(I, J, K)] = np.asarray(ds, dtype=np.complex128)
            return phi_blocks

        if "fc3" in f:
            raw = np.asarray(f["fc3"])
            if raw.shape != (n_dof, n_dof, n_dof):
                raise ValueError(
                    f"Legacy /fc3 shape {raw.shape} does not match "
                    f"({n_dof},) * 3."
                )
            return fc3_to_phi_blocks(
                raw,
                block_sizes,
                nn_only=nn_only,
                truncation_warn=truncation_warn,
            )

    raise ValueError(
        f"FC3 file {fc3_path} contains neither '/fc3_blocks' nor '/fc3'."
    )
