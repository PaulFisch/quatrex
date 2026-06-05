#!/usr/bin/env python
"""Sparse FC4 export for a LARGE supercell (no dense n^4 materialisation), then
build a 5x5x5 (250-atom) bulk-Si FC4 from the EXISTING si_big displacements.

The dense get_fc_array(order=4) for 250 atoms is 2.36 TiB; instead we query the
ForceConstants per atom-quad (fcs[s1,s2,s3,s4], which hiphive unfolds by
symmetry) only for s1 in the reference rows and (s2,s3,s4) within the FC4 cutoff
of s1 -- exactly the compact-reference the device loop consumes. Validated first
against the dense route on the small 2x2x2 cell, then applied to si_big.
"""
from __future__ import annotations
import sys, json, warnings, itertools
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import ase.io, h5py
from ase.neighborlist import neighbor_list
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.utilities import prepare_structures
from trainstation import Optimizer
from phonon_inputs.hiphive_fc3 import _atoms_from_meta, _reference_supercell_atoms
from solver.fc4_device import build_compact_reference_fc4_from_dense


def build_compact_reference_fc4_sparse(fcs, ref_sc_atoms, ideal, cutoff, tol=1e-8):
    """Compact-reference FC4 {(s1,s2,s3,s4): T[3,3,3,3]} via per-quad queries,
    without materialising the dense n^4 tensor. Candidates for each reference
    atom s1 are s1 itself plus all atoms within `cutoff` (all 4 cluster atoms
    must be mutually within the FC4 cutoff, so within cutoff of s1)."""
    n = len(ideal)
    # neighbour list: atoms within cutoff of each atom
    i_idx, j_idx = neighbor_list("ij", ideal, cutoff)
    nbr = {a: {a} for a in range(n)}
    for a, b in zip(i_idx, j_idx):
        nbr[int(a)].add(int(b))
    out = {}
    for s1 in (int(a) for a in ref_sc_atoms):
        cand = sorted(nbr[s1])
        for s2, s3, s4 in itertools.product(cand, repeat=3):
            T = np.asarray(fcs[s1, s2, s3, s4])
            if np.max(np.abs(T)) > tol:
                out[(s1, s2, s3, s4)] = T
    return out


def fit(wd, extra_c4):
    m = json.loads((Path(wd) / "hiphive_meta.json").read_text())
    prim = _atoms_from_meta(m["primitive"]); ideal = _atoms_from_meta(m["supercell_atoms"])
    n = m["n_structures"]
    rat = [ase.io.read(str(Path(wd) / f"disp-{i:05d}" / "vasprun.xml")) for i in range(1, n + 1)]
    st = prepare_structures(rat, ideal)
    cuts = list(m["cutoffs"])[:2] + [extra_c4]
    cs = ClusterSpace(prim, cuts); sc = StructureContainer(cs)
    for s in st: sc.add_structure(s)
    opt = Optimizer(sc.get_fit_data(), fit_method="rfe", train_size=0.9, seed=0); opt.train()
    fcs = ForceConstantPotential(cs, opt.parameters).get_force_constants(ideal)
    ref = _reference_supercell_atoms(prim, ideal)
    return prim, ideal, fcs, ref, opt, cuts, m


# --- validate sparse == dense on the small 2x2x2 cell ----------------------
print("== validation on 2x2x2 bulk Si ==")
_, ideal2, fcs2, ref2, opt2, cuts2, _ = fit("phonon/configs/si_primitive/fc3_hiphive_si_fc4_vasp", 3.0)
dense = build_compact_reference_fc4_from_dense(fcs2.get_fc_array(order=4), ref2)
sparse = build_compact_reference_fc4_sparse(fcs2, ref2, ideal2, cuts2[2])
kd, ks = set(dense), set(sparse)
ok = kd == ks and all(np.max(np.abs(dense[k] - sparse[k])) < 1e-9 for k in kd)
print(f"  dense {len(dense)} quads, sparse {len(sparse)} quads, keys equal {kd==ks}, "
      f"values match {ok}")
if not ok:
    print("  SPARSE EXPORT INVALID -- aborting"); sys.exit(1)
print("  -> sparse exporter VALIDATED")

# --- si_big (5x5x5, 250 atoms): fit + sparse export ------------------------
print("\n== si_big 5x5x5 FC4 (sparse) ==")
C4 = 3.0
prim, ideal, fcs, ref, opt, cuts, m = fit("phonon/configs/si_primitive/fc3_hiphive_si_big", C4)
print(f"  {len(ideal)} atoms, cutoffs {cuts}, RMSE tr {opt.rmse_train:.4f} te {opt.rmse_test:.4f}")
compact = build_compact_reference_fc4_sparse(fcs, ref, ideal, C4)
maxf4 = max((np.max(np.abs(v)) for v in compact.values()), default=0.0)
print(f"  sparse FC4: {len(compact)} compact quads, max|FC4|={maxf4:.3e} eV/A^4")
OUT = Path("/tmp/claude/si_big_fc4"); OUT.mkdir(parents=True, exist_ok=True)
out = OUT / "fc3.hdf5"
fc2 = fcs.get_fc_array(order=2); fc3 = fcs.get_fc_array(order=3)
with h5py.File(out, "w") as f:
    f.create_dataset("fc2", data=fc2, compression="gzip")
    f.create_dataset("fc3", data=fc3, compression="gzip")
    f.create_dataset("fc4_atoms", data=np.array(list(compact.keys()), dtype=np.int64))
    f.create_dataset("fc4_values", data=np.array(list(compact.values()), dtype=np.float64),
                     compression="gzip")
    f.attrs["fc4_n_super"] = len(ideal)
print(f"  wrote {out} (fc2 {fc2.shape}, fc3 {fc3.shape}, {len(compact)} fc4 quads)")
