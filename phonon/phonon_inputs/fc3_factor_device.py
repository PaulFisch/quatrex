"""Offline pipeline: bulk FC3 -> CP-family factors -> per-q device factor arrays.

The factored coupled-q SSE consumes, instead of the dense q-folded vertex dict
{(iq1, iq2): {(I, K, K'): Phi[b,b,b]}} (O(N_q^2 · blocks · b^3), ~GBs), the
exact per-leg factorisation of the same objects:

    Phi~(q1, q2)[(I, K, K')][a, b, c]
        = sum_r lam_r * D[a, r] * U[K-I][iq1][b, r] * U[K'-I][iq2][c, r]

where U[d][iq] is the transport-offset-d, transverse-momentum-iq device
gather of the (shared, INDSCAL) contracted-leg factor V:

    U[d][iq][3p + beta, r] = sum_{s : offset(s)=d, prim(s)=p}
                             exp(-2*pi*i * cell_frac[s] . q_iq) * V[3s+beta, r]

This mirrors phonon/solver/se_q.py:_qfold_device_blocks (phases on the two
CONTRACTED legs, external leg unphased) composed with
phonon/solver/fc3_device.py:build_device_fc3_blocks (per-(prim, slab-offset)
accumulation) EXACTLY -- both are linear per leg, so the factorisation of the
folded blocks is exact given the factorisation of M_stacked.

The FIT targets the same M_stacked (solver THz^2 units, build_supercell_mapping
gauge) that the dense qfold chain consumes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .fc3_compression import (
    export_production_factors,
    fit_production,
    target_from_dense,
)


def _fc3_hash(M_stacked: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(M_stacked).tobytes())
    return h.hexdigest()[:16]


def fit_film_fc3_factors(
    M_stacked: np.ndarray,
    n_atoms: int,
    n_super: int,
    rank: int,
    ansatz: str = "INDSCAL",
    cache_dir: str | Path | None = None,
    cache_label: str | None = None,
    masses_super: np.ndarray | None = None,
    **fit_kwargs,
) -> dict:
    """Fit the mass-weighted bulk FC3 and return the production factor export.

    The fit is a property of the bulk tensor only (independent of the device
    slab count and the transverse mesh), so it is cached per
    (ansatz, rank, tensor-hash) and shared by every (ns, nk) build.
    """
    n_dof = 3 * n_atoms
    dim_sc = 3 * n_super
    T = np.asarray(M_stacked, dtype=np.float64).reshape(n_dof, dim_sc, dim_sc)

    # The physical ASR on the mass-weighted target is the sqrt-mass-
    # weighted sum; uniform masses reduce to the legacy plain projector
    # (and keep the legacy cache tag valid).
    asr_w = None
    if masses_super is not None:
        m = np.asarray(masses_super, dtype=float)
        if not np.allclose(m, m[0]):
            asr_w = np.sqrt(m)
    tag = f"fc3_factors_{ansatz.lower()}_r{rank}_{_fc3_hash(T)}"
    if asr_w is not None:
        import hashlib
        tag += "_mw" + hashlib.sha256(asr_w.tobytes()).hexdigest()[:8]
    if cache_label:
        if not cache_label.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                "factor cache_label may contain only letters, digits, '-' "
                "and '_'"
            )
        tag += f"_{cache_label}"
    cache = None if cache_dir is None else Path(cache_dir) / f"{tag}.npz"
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=True)
        exp = {k: z[k] for k in z.files if k != "meta"}
        exp["meta"] = z["meta"].item()
        print(f"[fc3-factors] cache hit {cache.name} "
              f"(rel_err={exp['meta']['rel_err']:.4f})", flush=True)
        return exp

    target = target_from_dense(T, n_super, asr_weights=asr_w)
    res = fit_production(target, rank=rank, ansatz=ansatz, **fit_kwargs)
    exp = export_production_factors(res, target)
    if cache_label:
        exp["meta"] = {**exp["meta"], "fit_cache_label": cache_label}
    asr = res.info["asr"]
    print(f"[fc3-factors] {ansatz} R={rank}: rel_err={res.rel_err:.4f} "
          f"asr_j/norm={asr['leg_j'] / (asr['norm'] or 1.0):.2e}", flush=True)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **{k: v for k, v in exp.items()
                                      if k != "meta"},
                            meta=np.array(exp["meta"], dtype=object))
    return exp


def build_device_factor_arrays(
    export: dict,
    prim_indices: np.ndarray,
    cell_frac: np.ndarray,
    slab_indices: np.ndarray,
    n_atoms: int,
    q_points,
    transport_direction: str,
) -> dict:
    """Map real-space factors to per-(transport-offset, q) device factor arrays.

    Returns
    -------
    dict with:
      D        : (n_dof, R) float64 -- unphased external leg
      lambdas  : (R,) float64
      offsets  : (n_off,) int64     -- transport offsets d (minimum image)
      UB, UC   : (n_off, N_q, n_dof, R) complex128 -- contracted-leg device
                 factors. For INDSCAL UB is UC's alias (same V).
    """
    from phonon.solver.fc3_device import _minimum_image_offset

    n_dof = 3 * n_atoms
    n_super = len(prim_indices)
    n_super_z = int(slab_indices.max()) + 1
    R = export["lambdas"].shape[0]

    tidx = "xyz".index(transport_direction)
    perp = [i for i in range(3) if i != tidx]
    qmat = np.zeros((len(q_points), 3))
    for iq, (qa, qb) in enumerate(q_points):
        qmat[iq, perp[0]], qmat[iq, perp[1]] = qa, qb
    # phases[iq, s] = exp(-2*pi*i cell_frac[s] . q_iq)  (se_q.py convention)
    phases = np.exp(-2j * np.pi * (qmat @ np.asarray(cell_frac, float).T))

    offs = np.array(sorted({_minimum_image_offset(int(slab_indices[s]),
                                                  n_super_z)
                            for s in range(n_super)}), dtype=np.int64)
    off_pos = {int(d): i for i, d in enumerate(offs)}

    def gather(V):
        """V (dim_sc, R) -> U (n_off, N_q, n_dof, R)."""
        U = np.zeros((len(offs), len(q_points), n_dof, R), dtype=np.complex128)
        for s in range(n_super):
            d = off_pos[_minimum_image_offset(int(slab_indices[s]), n_super_z)]
            p = int(prim_indices[s])
            block = V[3 * s:3 * s + 3, :]                     # (3, R)
            # U[d, :, 3p:3p+3, :] += phases[:, s, None, None] * block
            U[d, :, 3 * p:3 * p + 3, :] += (
                phases[:, s][:, None, None] * block[None, :, :])
        return U

    if "V" in export:            # INDSCAL: shared contracted leg
        UB = gather(np.asarray(export["V"], float))
        UC = UB
        D = np.asarray(export["D"], np.float64)
    else:                        # CP: independent legs
        UB = gather(np.asarray(export["B"], float))
        UC = gather(np.asarray(export["C"], float))
        D = np.asarray(export["A"], np.float64)

    return {"D": D, "lambdas": np.asarray(export["lambdas"], np.float64),
            "offsets": offs, "UB": UB, "UC": UC,
            "meta": dict(export.get("meta", {}))}
