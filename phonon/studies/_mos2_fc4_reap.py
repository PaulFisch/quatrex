"""Device FC4 blocks for the quartic (SCP) loop self-energy.

Run: python phonon/studies/_mos2_fc4_reap.py         --fcp cluster/mos2_scp300v2/fcp_o4.fcp         --meta cluster/mos2_film_reap_scp/hiphive_meta.json         --nslabs 3 --out cluster/mos2film_L3_nk5_scp/fc4_blocks.hdf5
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "phonon")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SUPERCELL = (4, 4, 3)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fcp", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--nslabs", type=int, required=True)
    p.add_argument("--tdir", default="z")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from ase import Atoms
    from hiphive import ForceConstantPotential
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    meta = json.load(open(a.meta))
    prim = meta["primitive"]
    unit = PhonopyAtoms(symbols=prim["symbols"],
                        cell=np.array(prim["cell"]),
                        scaled_positions=np.array(prim["scaled_positions"]))
    ph = Phonopy(unit, supercell_matrix=np.diag(SUPERCELL),
                 primitive_matrix=np.eye(3))
    sc = Atoms(symbols=ph.supercell.symbols, cell=ph.supercell.cell,
               positions=ph.supercell.positions, pbc=True)
    masses = np.asarray(sc.get_masses(), dtype=float)
    n_super = len(sc)
    nat = len(unit)
    n_dof = 3 * nat

    fcp = ForceConstantPotential.read(a.fcp)
    fcs = fcp.get_force_constants(sc)
    fc_dict = fcs.get_fc_dict()
    quads = {c: t for c, t in fc_dict.items() if len(c) == 4}
    print(f"order-4 clusters: {len(quads)}", flush=True)

    # supercell atom -> (primitive atom, transport cell index)
    ti = "xyz".index(a.tdir)
    n_sz = SUPERCELL[ti]
    frac = sc.get_scaled_positions()
    zcell = np.floor(frac[:, ti] * n_sz + 1e-9).astype(int)
    # primitive index: phonopy supercell ordering repeats the unit cell
    # per lattice point; recover via matching scaled positions mod cell
    from phonon_inputs.separable import build_supercell_mapping
    prim_idx, _cf, slab_idx, _ref = build_supercell_mapping(ph, a.tdir)
    assert np.array_equal(np.sort(np.unique(prim_idx)), np.arange(nat))

    def mio(d):        # minimum-image transport offset
        return (d + n_sz // 2) % n_sz - n_sz // 2

    # Gather: anchor leg 0 in transport cell 0 (translation invariance);
    # distribute the sorted-cluster tensor over ALL index permutations.
    bulk: dict[tuple[int, int, int], np.ndarray] = {}
    for cluster, tensor in quads.items():
        atoms4 = list(cluster)
        t = np.asarray(tensor, dtype=float)
        w = np.prod([np.sqrt(masses[s]) for s in atoms4])
        t = t / w
        seen = set()
        for perm in itertools.permutations(range(4)):
            ap = tuple(atoms4[i] for i in perm)
            if ap in seen:
                continue
            seen.add(ap)
            tp = np.transpose(t, perm)
            i0 = ap[0]
            if int(slab_idx[i0]) % n_sz != 0:
                continue          # anchor images once; translations restore
            offs = [mio(int(slab_idx[s]) - int(slab_idx[i0])) for s in ap]
            key = tuple(offs[1:])
            blk = bulk.setdefault(key, np.zeros((n_dof,) * 4))
            ps = [int(prim_idx[s]) for s in ap]
            blk[3*ps[0]:3*ps[0]+3, 3*ps[1]:3*ps[1]+3,
                3*ps[2]:3*ps[2]+3, 3*ps[3]:3*ps[3]+3] += tp
    print(f"bulk offset blocks: {sorted(bulk.keys())}", flush=True)

    # Device blocks with boundary clipping
    L = a.nslabs
    dev: dict[tuple[int, int, int, int], np.ndarray] = {}
    for (dj, dk, dl), blk in bulk.items():
        if np.abs(blk).max() == 0.0:
            continue
        for I in range(L):
            J, K, Kp = I + dj, I + dk, I + dl
            if all(0 <= X < L for X in (J, K, Kp)):
                dkey = (I, J, K, Kp)
                dev[dkey] = dev.get(dkey, 0) + blk
    print(f"device blocks: {len(dev)} "
          f"(max |Phi4| {max(np.abs(b).max() for b in dev.values()):.3f} "
          "eV/(A^4 amu^2))", flush=True)

    with h5py.File(a.out, "w") as f:
        g = f.create_group("fc4_blocks")
        for (I, J, K, Kp), blk in dev.items():
            d = g.create_dataset(f"{I}_{J}_{K}_{Kp}", data=blk,
                                 compression="gzip")
            d.attrs["IJKKp"] = (I, J, K, Kp)
        f.attrs["n_dof"] = n_dof
        f.attrs["n_slabs"] = L
        f.attrs["units"] = "eV/(A^4 amu^2), mass-weighted"
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
