"""Observable-level error of the decomposed SSE, against the FULL-vertex SSE.

The FC3 fit residual ``eps_R = ||Phi - Phi_R||_F / ||Phi||_F`` is an error on the
TENSOR. The self-energy is a contraction of that tensor against two Green's
functions, so it need not inherit it: the components CP discards may be the ones
the bubble weights least. This script measures what actually reaches the physics.

It is the ONE-SHOT error: both self-energies are evaluated on the SAME Green's
function (the device's harmonic/ballistic G at the real temperature), so what is
measured is purely the vertex error propagated through the bubble -- no SCBA
feedback, no error cancellation from re-converging. That isolates the vertex->Sigma
map, which is the quantity the rank-truncation bound in the theory chapter claims.

Reports, per rank, the relative error in
  * Sigma^<, Sigma^>, Sigma^R           (the self-energy itself),
  * Gamma = i(Sigma^> - Sigma^<)        (the scattering rate -- the physical one),
  * the omega-resolved error, in-band and at the peak of the spectrum,
against the dense q-folded vertex.

Usage:
    QX_CONFIG=<dense film config> python phonon/studies/_rank_error_sse.py \
        --factors DIR --ranks 8,16,32,64,128
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from qttools import xp
from quatrex.core.config import parse_config, setup_context
from quatrex.phonon.qfold import load_qfold
from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon
from quatrex.phonon.vertex_factors import load_decomposed


def _rel(a, b):
    """Relative error ||a - b|| / ||b|| in the max norm."""
    denom = float(xp.max(xp.abs(b)))
    if denom == 0.0:
        return float("nan")
    return float(xp.max(xp.abs(a - b))) / denom


def _rel_l2(a, b):
    denom = float(xp.sqrt(xp.sum(xp.abs(b) ** 2)))
    if denom == 0.0:
        return float("nan")
    return float(xp.sqrt(xp.sum(xp.abs(a - b) ** 2))) / denom


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--factors", required=True,
                   help="directory holding decomposed_vertices_r{R}.npz")
    p.add_argument("--ranks", default="8,16,32,64,128")
    p.add_argument("--cache", default=None,
                   help="npz to cache the dense-vertex Sigma in (it is the "
                        "expensive part, and it does not change)")
    args = p.parse_args()

    config = parse_config(Path(os.environ["QX_CONFIG"]))
    setup_context(config)

    from quatrex.core.scba import SCBA

    scba = SCBA(config)
    solver = scba.phonon_solver
    data = scba.data

    # The harmonic (ballistic) Green's function of the real device: Sigma = 0.
    # Both self-energies are evaluated on THIS G, so the only difference between
    # them is the vertex.
    for m in (data.sigma_lesser, data.sigma_greater, data.sigma_retarded_hermitian):
        m.data[:] = 0.0
    data.g_retarded.allocate_data()
    solver.solve(
        data.sigma_lesser, data.sigma_greater, data.sigma_retarded_hermitian,
        out=(data.g_lesser, data.g_greater, data.g_retarded),
    )
    for m in (data.g_lesser, data.g_greater):
        m.dtranspose(discard=False)

    freqs = np.asarray(solver.local_frequencies)
    block_sizes = data.g_lesser.block_sizes

    def sigma_of(**vertex):
        """Sigma^{<,>,R} on the fixed ballistic G, for one vertex source."""
        sse = SigmaPhononPhonon(
            config, phonon_frequencies=freqs, block_sizes=block_sizes,
            dynamical_matrix=solver.dynamical_matrix, **vertex,
        )
        outs = []
        for name in ("sigma_lesser", "sigma_greater", "sigma_retarded_hermitian"):
            buf = getattr(data, name)
            if buf.distribution_state != "nnz":
                buf.dtranspose()
            buf.data[:] = 0.0
            outs.append(buf)
        sse.compute(data.g_lesser, data.g_greater, out=tuple(outs))
        return tuple(xp.array(o.data, copy=True) for o in outs)

    # --- the FULL SSE: the dense q-folded vertex ------------------------------
    # Cached: it is by far the most expensive thing here (the dense vertex is
    # what the decomposition exists to avoid).
    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        blob = np.load(cache)
        sl_ref, sg_ref, sr_ref = (xp.asarray(blob[k]) for k in ("sl", "sg", "sr"))
    else:
        vertices, q_diff_map, nk_shape = load_qfold(Path(config.phonon.qfold_path))
        sl_ref, sg_ref, sr_ref = sigma_of(
            qfold=(vertices, q_diff_map, int(np.prod(nk_shape)))
        )
        del vertices
        if cache is not None:
            np.savez_compressed(cache, sl=sl_ref, sg=sg_ref, sr=sr_ref)

    # The factored vertices are mutually exclusive with the dense one, so the
    # dense path must be forgotten before the rank-R constructions.
    config.phonon.qfold_path = None
    gamma_ref = 1j * (sg_ref - sl_ref)

    # in-band mask: where the spectrum actually carries weight
    band = xp.abs(xp.asarray(freqs)) > 1e-9
    peak = float(xp.max(xp.abs(gamma_ref)))

    print(f"{'R':>4} {'eps_R(FC3)':>11} | {'Sigma^<':>9} {'Sigma^>':>9} "
          f"{'Sigma^R':>9} {'Gamma':>9} | {'Gamma(L2)':>10}")
    print("-" * 72)

    for rank in (int(r) for r in args.ranks.split(",") if r):
        path = Path(args.factors) / f"decomposed_vertices_r{rank}.npz"
        if not path.exists():
            path = Path(args.factors) / "decomposed_vertices.npz"
        vf = load_decomposed(path, rank=rank)
        sl, sg, sr = sigma_of(vfactors=vf)
        gamma = 1j * (sg - sl)

        eps = vf.meta.get("rel_err", float("nan")) if isinstance(vf.meta, dict) else float("nan")
        print(f"{rank:>4} {eps:>11.4f} | {_rel(sl, sl_ref):>9.2e} "
              f"{_rel(sg, sg_ref):>9.2e} {_rel(sr, sr_ref):>9.2e} "
              f"{_rel(gamma, gamma_ref):>9.2e} | {_rel_l2(gamma, gamma_ref):>10.2e}")

    print()
    print(f"peak |Gamma| = {peak:.4g} THz^2;  {int(band.sum())} in-band frequencies")
    print("Errors are relative to the FULL (dense q-folded) vertex evaluated on the")
    print("SAME ballistic G -- so they are the vertex error propagated through the")
    print("bubble, with no SCBA feedback.")


if __name__ == "__main__":
    main()
