#!/usr/bin/env python
"""Build a CNT(3,3) FC4 (no new DFT) and export a device-compatible fc3.hdf5 with
fc2 + fc3 + compact-reference fc4, so the quartic loop can be tested on the CNT.

Uses a SHORT FC4 cutoff (2.0 A) because the longer 2.5 A quartic overfits the 30
structures (test RMSE 0.253 -> 0.321); the 2.0 A / 198-cluster fit IMPROVES it
(0.253 -> 0.244). rfe (ardr would prune the small quartic signal).
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import ase.io, h5py
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.utilities import prepare_structures
from trainstation import Optimizer
from phonon_inputs.hiphive_fc3 import _atoms_from_meta, _reference_supercell_atoms
from solver.fc4_device import build_compact_reference_fc4_from_dense

WD = _REPO / "phonon/configs/cnt/fc3_hiphive_cnt33_vasp"
OUT = Path("/tmp/claude/cnt_fc4"); OUT.mkdir(parents=True, exist_ok=True)
m = json.loads((WD / "hiphive_meta.json").read_text())
prim = _atoms_from_meta(m["primitive"]); ideal = _atoms_from_meta(m["supercell_atoms"])
n_super = len(ideal)
c2, c3 = m["cutoffs"][0], m["cutoffs"][1]; c4 = 2.0
rat = [ase.io.read(str(WD / f"disp-{i:05d}" / "vasprun.xml")) for i in range(1, m["n_structures"] + 1)]
st = prepare_structures(rat, ideal)
print(f"CNT FC4 export: {len(st)} structures, {n_super} atoms, cutoffs {c2}/{c3}/{c4}")

cs = ClusterSpace(prim, [c2, c3, c4]); sc = StructureContainer(cs)
for s in st:
    sc.add_structure(s)
opt = Optimizer(sc.get_fit_data(), fit_method="rfe", train_size=0.9, seed=0)
opt.train()
print(f"  rfe: RMSE tr {opt.rmse_train:.4f} te {opt.rmse_test:.4f}")
fcs = ForceConstantPotential(cs, opt.parameters).get_force_constants(ideal)
fc2 = fcs.get_fc_array(order=2); fc3 = fcs.get_fc_array(order=3)
out = OUT / "fc3.hdf5"
with h5py.File(out, "w") as f:
    f.create_dataset("fc3", data=fc3, compression="gzip")
    f.create_dataset("fc2", data=fc2, compression="gzip")
print(f"  wrote fc2/fc3; building dense FC4 ({n_super}^4 -> "
      f"{n_super**4 * 81 * 8 / (1 << 30):.0f} GiB)...")
fc4 = fcs.get_fc_array(order=4)
ref_sc = _reference_supercell_atoms(prim, ideal)
compact = build_compact_reference_fc4_from_dense(fc4, ref_sc)
del fc4
atoms_arr = np.array(list(compact.keys()), dtype=np.int64)
vals_arr = np.array(list(compact.values()), dtype=np.float64)
with h5py.File(out, "a") as f:
    f.create_dataset("fc4_atoms", data=atoms_arr)
    f.create_dataset("fc4_values", data=vals_arr, compression="gzip")
    f.attrs["fc4_n_super"] = n_super
print(f"  FC4: {len(compact)} compact quadruples, max|FC4|={np.max(np.abs(vals_arr)):.3e} "
      f"-> {out}")
