"""Build PRODUCTION quatrex phonon-transport inputs for a benchmark system.

Committed, parameterized port of the (ephemeral) /tmp builders. Reuses the
verified input machinery: ``phonon_inputs.convention.get_btd_blocks`` +
``separable`` + ``phonon.solver.fc3_device`` / ``se_q`` +
``phonon_inputs.quatrex_writer`` + ``quatrex.phonon.qfold``. THz convention.

Two device families:

  * CNT (``cnt33`` / ``cnt80``) -- transversely FINITE (Gamma-only, k==1), a
    1-D BTD device. No qfold. Writes ``dynamical_matrix.mat`` ([ix,0,0] keys),
    ``fc3_blocks.hdf5`` (Gamma device FC3), ``structure.xyz``,
    ``phonon_energies.npy``.
  * Si film (``sifilm``) -- transversely PERIODIC (k>1). The real-space cells
    are the exact transverse-IDFT of the dense H(q) over the production
    monkhorst mesh q=k/nk; the q-folded 3-phonon vertices go to
    ``qfold_vertices.npz``. Writes the above plus ``qfold_vertices.npz`` and a
    ``kshift.npy`` (the kpoint_shift the config must use).

Usage:
    python build_inputs.py --system cnt33  -L 4         --out DIR
    python build_inputs.py --system cnt80  -L 2         --out DIR
    python build_inputs.py --system sifilm --nslabs 5 --nk 8 --out DIR \
        [--fc3-subdir reaps/si_big_hiphive] [--tdir x]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

# Repo layout: <root>/phonon and <root> both importable (phonon_inputs.*,
# phonon.*, quatrex.*).
_ROOT = Path(__file__).resolve().parents[3]
_PHON = _ROOT / "phonon"
for _p in (_ROOT, _PHON):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import h5py
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phonon_inputs.convention import get_btd_blocks
from phonon_inputs.constants import CONVERSION_THZ2
from phonon_inputs.separable import (
    build_supercell_mapping,
    build_realspace_fc3_matrices,
    build_q_diff_map,
)
from phonon.solver.fc3_device import build_device_fc3_blocks
from phonon_inputs.quatrex_writer import write_fc3_blocks, write_structure_xyz

# Gamma-only (k==1) finite-device FC3 reaps. build_cnt() is species-agnostic, so
# the multi-species SiNW (Si+H) wires flow through it unchanged.
CNT_DIR = {
    "cnt33": _PHON / "configs/cnt/fc3_hiphive_cnt33_vasp",
    "cnt80": _PHON / "configs/cnt/fc3_hiphive_cnt80_vasp",
    "sinw_d5a": _PHON / "configs/sinw/fc3_hiphive_sinw100_d5a_sc4_vasp",
    "sinw_d11a": _PHON / "configs/sinw/fc3_hiphive_sinw100_d11a_vasp",
    # SrTiO3 (5-atom cubic perovskite, [3,3,3]) -- Gamma-only finite slab,
    # best-effort strong-anharmonicity transport (the bulk is transversely
    # periodic; this is a finite-cross-section device).
    "srtio3": _PHON / "configs/perovskite/fc3_hiphive_srtio3_333_vasp",
}


def _key_str(k):
    """scipy.savemat real-space key, matching the production .mat loader."""
    return f"[{k[0]}, {k[1]}, {k[2]}]"


def build_cnt(system, ncells, tdir, nfreq, fmax, emin, out):
    """Gamma-only (k==1) CNT device. Port of /tmp/build_cnt_inputs.py."""
    D = CNT_DIR[system]
    meta = json.load(open(D / "hiphive_meta.json"))
    prim = meta["primitive"]
    scm = meta["supercell"]
    ti = "xyz".index(tdir)
    n_qz = int(scm[ti] if not isinstance(scm, dict) else scm["xyz"[ti]])
    unit = PhonopyAtoms(
        symbols=prim["symbols"],
        cell=np.array(prim["cell"]),
        scaled_positions=np.array(prim["scaled_positions"]),
    )
    phonon = Phonopy(unit, supercell_matrix=np.diag(scm), primitive_matrix=np.eye(3))
    with h5py.File(D / "fc3.hdf5", "r") as f:
        fc2 = f["fc2"][...]
        fc3_raw = f["fc3"][...]
    phonon.force_constants = fc2
    nat = len(phonon.primitive.masses)
    nd = 3 * nat
    print(
        f"{system}: prim {nat} atoms, ndof/cell={nd}, supercell {scm} "
        f"(n_qz={n_qz}), transport={tdir}, ncells={ncells}, "
        f"grid {emin}-{fmax}/{nfreq}",
        flush=True,
    )

    h00, h01 = get_btd_blocks(
        phonon, (0.0, 0.0), transport_direction=tdir, n_qz=n_qz,
        conversion_factor=CONVERSION_THZ2,
    )
    print(
        f"||H00||={np.linalg.norm(h00):.1f} ||H01||={np.linalg.norm(h01):.1f} "
        f"THz^2; H00 herm err {np.abs(h00 - h00.conj().T).max():.1e}",
        flush=True,
    )
    # Dispersion sanity (no imaginary modes).
    fr = []
    for kz in np.linspace(0, 0.5, 21):
        hk = h00 + h01 * np.exp(2j * np.pi * kz) + h01.conj().T * np.exp(-2j * np.pi * kz)
        fr.append(np.sqrt(np.clip(np.linalg.eigvalsh(hk), 0, None)))
    fr = np.array(fr)
    print(f"dispersion: min {fr.min():.3f}, max {fr.max():.3f} THz; "
          f"acoustic lowest-4 @kz->0: {np.round(fr[0, :4], 4)}", flush=True)

    def key(n):
        k = [0, 0, 0]
        k[ti] = n
        return _key_str(k)

    savemat(str(out / "dynamical_matrix.mat"), {
        key(0): h00.astype(complex),
        key(1): h01.astype(complex),
        key(-1): h01.conj().T.astype(complex),
    })
    np.save(out / "phonon_energies.npy", np.linspace(emin, fmax, nfreq).astype(float))

    prim_idx, _cf, slab_idx, ref = build_supercell_mapping(phonon, tdir)
    M = build_realspace_fc3_matrices(
        fc3_raw, nat, phonon.supercell.masses, ref,
    )
    phi = build_device_fc3_blocks(M, prim_idx, slab_idx, nat, ncells)
    phi = {k: np.ascontiguousarray(v.astype(complex)) for k, v in phi.items()}
    write_fc3_blocks(phi, np.array([nd] * ncells), out / "fc3_blocks.hdf5", units="THz^2")
    write_structure_xyz(phonon.primitive, out / "structure.xyz")
    print(f"fc3 device: {len(phi)} blocks, off-diag "
          f"{sum(1 for k in phi if k[1] != k[2])}, "
          f"max|Phi|={max(np.abs(v).max() for v in phi.values()):.3e}", flush=True)


def _load_bulk_si(fc3_subdir):
    """Bulk-Si phonopy (FD FC2) + the FC3 path. Port of build_sifilm."""
    from phono3py import load as phono3py_load
    d = _PHON / fc3_subdir
    with h5py.File(d / "fc2.hdf5", "r") as f:
        fc2 = f["force_constants"][:]
    ph3 = phono3py_load(phono3py_yaml=str(d / "phono3py.yaml"), produce_fc=False, log_level=0)
    cell = PhonopyAtoms(symbols=ph3.unitcell.symbols, cell=ph3.unitcell.cell,
                        scaled_positions=ph3.unitcell.scaled_positions)
    phonon = Phonopy(cell, supercell_matrix=ph3.supercell_matrix, primitive_matrix=np.eye(3))
    phonon.force_constants = fc2
    return phonon, str(d / "fc3.hdf5")


def build_sifilm(nslabs, nk, tdir, nfreq, fmax, emin, fc3_subdir, out, nproc=1):
    """Transversely-periodic (k>1) Si film. Port of /tmp/build_sifilm_inputs.py."""
    from phonon.solver.se_q import _build_folded_vertices
    from quatrex.phonon.qfold import save_qfold

    assert nk % 2 == 1, "use ODD nk (clean Gamma-centered IDFT)"
    phonon, fc3_path = _load_bulk_si(fc3_subdir)
    nat = len(phonon.primitive.masses)
    nd = 3 * nat
    tidx = "xyz".index(tdir)
    perp = [i for i in range(3) if i != tidx]
    print(f"Si film: prim {nat} atoms, nd/cell={nd}, transport={tdir}, perp={perp}, "
          f"nk={nk}x{nk}, n_slabs={nslabs}, fc3={fc3_subdir}", flush=True)

    q_1d = np.arange(nk) / nk
    q_points = [(qa, qb) for qa in q_1d for qb in q_1d]
    q_diff_map = build_q_diff_map(nk, nk)
    n_kpts = nk * nk

    H00 = np.zeros((n_kpts, nd, nd), complex)
    H01 = np.zeros((n_kpts, nd, nd), complex)
    for iq, (qa, qb) in enumerate(q_points):
        h00, h01 = get_btd_blocks(phonon, (qa, qb), transport_direction=tdir,
                                  conversion_factor=CONVERSION_THZ2)
        H00[iq], H01[iq] = h00, h01
    print(f"dense H(q): ||H00(G)||={np.linalg.norm(H00[0]):.1f} "
          f"||H01(G)||={np.linalg.norm(H01[0]):.1f} THz^2; "
          f"H00 herm err {np.abs(H00[0] - H00[0].conj().T).max():.1e}", flush=True)

    rng = range(-(nk // 2), nk // 2 + 1)
    cells = [(cy, cz) for cy in rng for cz in rng]
    qarr = np.array(q_points)
    mats = {}
    for (cy, cz) in cells:
        ph = np.exp(-2j * np.pi * (cy * qarr[:, 0] + cz * qarr[:, 1])) / n_kpts
        D00 = np.tensordot(ph, H00, axes=(0, 0))
        Dp1 = np.tensordot(ph, H01, axes=(0, 0))
        Dm1 = np.tensordot(ph, np.conj(np.transpose(H01, (0, 2, 1))), axes=(0, 0))

        def key(ix):
            k = [0, 0, 0]
            k[tidx] = ix
            k[perp[0]] = cy
            k[perp[1]] = cz
            return (k[0], k[1], k[2])

        mats[key(0)] = D00.astype(complex)
        mats[key(1)] = Dp1.astype(complex)
        mats[key(-1)] = Dm1.astype(complex)

    def fold(ix, qa, qb):
        acc = np.zeros((nd, nd), complex)
        for (cy, cz) in cells:
            kk = [0, 0, 0]
            kk[tidx] = ix
            kk[perp[0]] = cy
            kk[perp[1]] = cz
            acc += mats[(kk[0], kk[1], kk[2])] * np.exp(2j * np.pi * (cy * qa + cz * qb))
        return acc

    worst = 0.0
    for iq in (0, 1, n_kpts // 3, n_kpts - 1):
        qa, qb = q_points[iq]
        e00 = np.abs(fold(0, qa, qb) - H00[iq]).max() / (np.abs(H00[iq]).max() + 1e-30)
        e01 = np.abs(fold(1, qa, qb) - H01[iq]).max() / (np.abs(H01[iq]).max() + 1e-30)
        worst = max(worst, e00, e01)
    print(f"IDFT round-trip self-check: worst rel err = {worst:.2e}", flush=True)
    assert worst < 1e-10, "round-trip FAILED"

    savemat(str(out / "dynamical_matrix.mat"),
            {_key_str(k): v for k, v in mats.items()})
    print(f"dynamical_matrix.mat: {len(mats)} blocks "
          f"({len(cells)} transverse cells x 3 transport)", flush=True)

    prim_idx, cell_frac, slab_idx, ref_sc = build_supercell_mapping(phonon, tdir)
    with h5py.File(fc3_path, "r") as f:
        fc3 = f["fc3"][:]
    M_stacked = build_realspace_fc3_matrices(fc3, nat, phonon.supercell.masses, ref_sc)
    print(f"building {n_kpts}^2-pair folded vertices (n_slabs={nslabs}, nproc={nproc})...", flush=True)
    vertices = _build_folded_vertices(M_stacked, prim_idx, cell_frac, slab_idx, nat,
                                      nslabs, n_kpts, q_points, q_diff_map, tdir,
                                      nproc=nproc)
    print(f"  folded vertex pairs: {len(vertices)}", flush=True)
    save_qfold(out / "qfold_vertices.npz", vertices, q_diff_map, (nk, nk))

    gamma = {k: np.ascontiguousarray(v.astype(complex)) for k, v in vertices[(0, 0)].items()}
    write_fc3_blocks(gamma, np.array([nd] * nslabs), out / "fc3_blocks.hdf5", units="THz^2")
    print(f"fc3_blocks.hdf5: {len(gamma)} Gamma device blocks", flush=True)
    write_structure_xyz(phonon.primitive, out / "structure.xyz")
    np.save(out / "phonon_energies.npy", np.linspace(emin, fmax, nfreq).astype(float))
    print(f"phonon_energies.npy: {nfreq} pts {emin}..{fmax} THz", flush=True)

    shift = round(0.5 - 0.5 / nk, 10)
    np.save(out / "kshift.npy", np.array(shift))
    print(f"REQUIRED config: kpoint_grid with {nk} on the two perp axes {perp}, "
          f"kpoint_shift={shift} on those axes, transport={tdir}, "
          f"num_transport_cells={nslabs}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", required=True,
                   choices=["cnt33", "cnt80", "sinw_d5a", "sinw_d11a", "srtio3", "sifilm"])
    p.add_argument("-L", "--ncells", type=int, default=2, help="CNT transport cells")
    p.add_argument("--nslabs", type=int, default=5, help="film layers along transport")
    p.add_argument("--nk", type=int, default=8, help="film transverse mesh (odd)")
    p.add_argument("--tdir", default=None, help="transport axis (CNT: z, film: x)")
    p.add_argument("--nfreq", type=int, default=None)
    p.add_argument("--fmax", type=float, default=None)
    p.add_argument("--emin", type=float, default=0.0,
                   help="grid start (0.0 = w0=0, best heat-flow conservation)")
    p.add_argument("--fc3-subdir", default="reaps/si_big_hiphive",
                   help="film bulk-Si FC reap (default the 5^3 hiphive)")
    # The FC3 vertex is used raw (plain truncation). The former ASR
    # projection was removed 2026-06-12: it was leg-asymmetric (broke the
    # vertex S3 symmetry -> energy conservation) and over-strong (suppressed
    # linewidths ~4-5x vs phono3py); hiphive fits are ASR-exact raw.
    p.add_argument("--out", required=True)
    p.add_argument("--nproc", type=int, default=1,
                   help="parallel workers for the O(nk^2) folded-vertex build")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.system in ("cnt33", "cnt80", "sinw_d5a", "sinw_d11a", "srtio3"):
        tdir = a.tdir or "z"
        if a.system in ("cnt33", "cnt80"):
            nfreq = a.nfreq or 161
            fmax = a.fmax or 55.0
        elif a.system == "srtio3":  # perovskite optical max ~24 THz
            nfreq = a.nfreq or 121
            fmax = a.fmax or 26.0
        else:  # SiNW: Si optical max ~15.5 THz -> tighter window than the CNT's 55
            nfreq = a.nfreq or 101
            fmax = a.fmax or 18.0
        emin = a.emin if a.emin is not None else 0.0
        build_cnt(a.system, a.ncells, tdir, nfreq, fmax, emin, out)
    else:
        tdir = a.tdir or "x"
        nfreq = a.nfreq or 121
        fmax = a.fmax or 15.0
        build_sifilm(a.nslabs, a.nk, tdir, nfreq, fmax, a.emin, a.fc3_subdir, out, a.nproc)
    print(f"inputs -> {out}", flush=True)


if __name__ == "__main__":
    main()
