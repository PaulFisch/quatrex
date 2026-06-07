"""Production full-ring phonon-phonon SSE vs the dense reference on REAL,
multi-cell silicon force constants.

Builds a real N_B-slab device vertex from the bulk-Si 5x5x5 hiPhive FC3
(``reaps/si_big_hiphive``) via ``build_device_fc3_blocks`` — which yields
genuine off-diagonal ``(I,K,K')`` blocks (``K != K'``), i.e. the
"beyond next-nearest-neighbour" regime — and checks that the production
``SigmaPhononPhonon.compute`` (the distributed FFT-first pipeline) matches
an independent dense oracle to ~1e-15. Run::

    python phonon/scripts/verify/phph_real_fc3_parity.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
sys.path.insert(0, str(_REPO / "phonon"))

from qttools import xp  # noqa: E402
from qttools.comm import comm as ranks  # noqa: E402
from qttools.datastructures import DSDBCOO  # noqa: E402

if not ranks._is_configured:
    _cfg = {k: "device_mpi" for k in ("all_to_all", "all_gather", "all_reduce", "bcast")}
    ranks.configure(block_comm_size=1, block_comm_config=_cfg,
                    stack_comm_config=_cfg, override=True)

from phonon_inputs.separable import (  # noqa: E402
    build_realspace_fc3_matrices,
    build_supercell_mapping,
)
from phonon.solver.fc3_device import build_device_fc3_blocks  # noqa: E402
from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon  # noqa: E402

# Reuse the independent dense oracle + minimal config from the unit tests.
_spec = importlib.util.spec_from_file_location(
    "_t", str(_REPO / "tests/quatrex/phonon/test_sse_phonon_phonon.py"))
_t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t)


def main(n_slabs: int = 3, ne: int = 21) -> int:
    data = _REPO / "phonon/reaps/si_big_hiphive"
    from phonopy import load as ph_load
    phonon = ph_load(str(data / "phono3py.yaml"))
    n_prim = len(phonon.primitive.masses)
    n_dof = 3 * n_prim

    prim_indices, _cell_frac, slab_indices, ref_sc = build_supercell_mapping(phonon, "z")
    with h5py.File(data / "fc3.hdf5", "r") as f:
        fc3_raw = f["fc3"][...]
    M_stacked = build_realspace_fc3_matrices(
        fc3_raw, n_prim, phonon.supercell.masses, ref_sc)

    phi_dev = build_device_fc3_blocks(
        M_stacked, prim_indices, slab_indices, n_prim, n_slabs)
    offdiag = [k for k in phi_dev if k[1] != k[2]]
    if not offdiag:
        raise SystemExit("no off-diagonal device Phi — parity test would be vacuous")
    phi_blocks = {k: np.ascontiguousarray(v.astype(np.complex128))
                  for k, v in phi_dev.items()}
    print(f"device Phi: {len(phi_dev)} blocks, {len(offdiag)} off-diagonal (K!=K')")

    nbs = n_dof
    block_sizes = np.array([nbs] * n_slabs)
    N = int(block_sizes.sum())
    rng = np.random.default_rng(7)
    gl_band, gg_band = {}, {}
    for K in range(n_slabs):
        for Kp in range(max(0, K - 1), min(n_slabs, K + 2)):
            gl_band[(K, Kp)] = (rng.standard_normal((ne, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nbs, nbs)))
            gg_band[(K, Kp)] = (rng.standard_normal((ne, nbs, nbs))
                                + 1j * rng.standard_normal((ne, nbs, nbs)))

    rows, cols = [], []
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    for I in range(n_slabs):
        for J in range(max(0, I - 1), min(n_slabs, I + 2)):
            for i in range(nbs):
                for j in range(nbs):
                    rows.append(offs[I] + i)
                    cols.append(offs[J] + j)
    pattern = csr_matrix((np.ones(len(rows), dtype=np.complex128),
                          (np.array(rows), np.array(cols))), shape=(N, N))
    bufs = [DSDBCOO.from_sparray(pattern, block_sizes, global_stack_shape=(ne,))
            for _ in range(5)]
    g_l, g_g, s_l, s_g, s_r = bufs
    for m in bufs:
        m.data[:] = 0.0
    glv, ggv = g_l.stack[...], g_g.stack[...]
    for (K, Kp) in gl_band:
        glv.blocks[K, Kp] = gl_band[(K, Kp)]
        ggv.blocks[K, Kp] = gg_band[(K, Kp)]

    freqs = np.linspace(0.0, 20.0, ne)
    dw = float(freqs[1] - freqs[0])
    ssp = SigmaPhononPhonon(_t._make_cfg("fft"), phonon_frequencies=freqs,
                            block_sizes=block_sizes, phi_blocks=phi_blocks)
    ssp.compute(g_l, g_g, out=(s_l, s_g, s_r))

    sl_ref, sg_ref, sr_ref = _t._ref_compute_multiblock(
        phi_blocks, gl_band, gg_band, block_sizes, dw)
    slv, sgv, srv = s_l.stack[...], s_g.stack[...], s_r.stack[...]
    maxerr = 0.0
    for I in range(n_slabs):
        for J in range(max(0, I - 1), min(n_slabs, I + 2)):
            for v, ref in ((slv, sl_ref), (sgv, sg_ref), (srv, sr_ref)):
                got = np.asarray(v.blocks[I, J])
                exp = ref.get((I, J), 0)
                maxerr = max(maxerr, float(
                    np.max(np.abs(got - exp)) / (np.max(np.abs(exp)) + 1e-300)))
    print(f"REAL Si FC3 (N_B={n_slabs}, off-diagonal ring): max rel err = {maxerr:.2e}")
    print("PASS" if maxerr < 1e-10 else "FAIL")
    return 0 if maxerr < 1e-10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
