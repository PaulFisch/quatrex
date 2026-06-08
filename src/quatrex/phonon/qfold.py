# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Serialization of q-folded 3-phonon device vertices for the
transversely-periodic (k>1) anharmonic self-energy.

A transversely-periodic film couples the transverse momenta in the
3-phonon bubble (crystal-momentum conservation):

    Sigma(q_ext, w) = (1/N_q) sum_{q'} Phi(q', q2) G(q', w') G(q2, w-w')
                      Phi(q2, q')          with q2 = q_ext - q'.

The q-folded device vertex ``Phi(q1, q2)`` (transverse Bloch phases on
the two contracted legs) does not depend on G, so it is built once,
offline, by the input builder (reusing ``phonon.solver.se_q``) and
stored here. The production self-energy then only reads arrays

"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qttools import NDArray

# {(iq1, iq2): {(I, K, Kp): Phi[n_dof, n_dof, n_dof]}}
QVertices = dict[tuple[int, int], dict[tuple[int, int, int], NDArray]]


def save_qfold(
    path: str | Path,
    vertices: QVertices,
    q_diff_map: NDArray,
    nk_shape: tuple[int, ...],
) -> None:
    """Write the folded vertices + q-difference map to ``path`` (.npz)."""
    out: dict[str, NDArray] = {
        "q_diff_map": np.asarray(q_diff_map, dtype=np.int64),
        "nk_shape": np.asarray(nk_shape, dtype=np.int64),
    }
    for (iq1, iq2), blocks in vertices.items():
        for (I, K, Kp), phi in blocks.items():
            out[f"v|{iq1}|{iq2}|{I}|{K}|{Kp}"] = np.ascontiguousarray(
                np.asarray(phi, dtype=np.complex128)
            )
    np.savez(str(path), **out)


def load_qfold(path: str | Path) -> tuple[QVertices, NDArray, tuple[int, ...]]:
    """Load folded vertices, ``q_diff_map`` and the transverse mesh shape.

    Returns ``(vertices, q_diff_map, nk_shape)``; ``n_kpts`` is
    ``int(np.prod(nk_shape))`` and ``q_diff_map`` is ``(n_kpts, n_kpts)``.
    """
    npz = np.load(str(path))
    q_diff_map = np.asarray(npz["q_diff_map"], dtype=int)
    nk_shape = tuple(int(k) for k in npz["nk_shape"])
    vertices: QVertices = {}
    for key in npz.files:
        if not key.startswith("v|"):
            continue
        _, iq1, iq2, I, K, Kp = key.split("|")
        vertices.setdefault((int(iq1), int(iq2)), {})[
            (int(I), int(K), int(Kp))
        ] = np.asarray(npz[key], dtype=np.complex128)
    return vertices, q_diff_map, nk_shape
