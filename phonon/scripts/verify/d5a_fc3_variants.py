#!/usr/bin/env python
"""Refit the d5a FC3 in several quality variants (NO new DFT, reuse the 44
rattled structures) and export each as a device-compatible fc3.hdf5, so we can
test whether FC3-side changes move the NEGF heat-flow conservation:

  rfe_base : rfe, cutoffs [7.88, 4.0]            (current production baseline)
  rfe_rot  : rfe + rotational sum rules (Huang + Born-Huang, post_fit)
  rfe_long : rfe, cutoffs [7.88, 5.0]            (longer-range FC3)
  ardr_base: ardr, cutoffs [7.88, 4.0]           (the noisier old fit)

Prints RMSE + FC2/FC3 ASR residuals + rotational residual for each, and writes
/tmp/claude/fc3_variants/<name>/fc3.hdf5 (fc2 + fc3 datasets).
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
from phonon_inputs.hiphive_convergence import (
    _apply_rotational_sum_rules, _rotational_residual_pair)
# Use the EXACT primitive + supercell ordering the reap used (meta), not a
# fresh make_supercell -- otherwise the exported fc3 atom indexing does not
# match what the device loader expects (gives a spurious conservation offset).
from phonon_inputs.hiphive_fc3 import _atoms_from_meta

WD = _REPO / "phonon/configs/sinw/fc3_hiphive_sinw100_d5a_sc4_vasp"
OUT = Path("/tmp/claude/fc3_variants"); OUT.mkdir(parents=True, exist_ok=True)
meta = json.loads((WD / "hiphive_meta.json").read_text())
prim = _atoms_from_meta(meta["primitive"])
ideal = _atoms_from_meta(meta["supercell_atoms"])
fc2c, fc3c = meta["cutoffs"][:2]
rattled = [ase.io.read(str(WD / f"disp-{i:05d}" / "vasprun.xml"))
           for i in range(1, meta["n_structures"] + 1)]
structures = prepare_structures(rattled, ideal)
print(f"d5a refit: {len(structures)} structures, base cutoffs {fc2c}/{fc3c} A\n")

VARIANTS = [
    ("rfe_base",  [fc2c, fc3c], "rfe", "off"),
    ("rfe_rot",   [fc2c, fc3c], "rfe", "post_fit"),
    ("rfe_long",  [fc2c, 5.0],  "rfe", "off"),
    ("ardr_base", [fc2c, fc3c], "ardr", "off"),
]

for name, cutoffs, method, rot in VARIANTS:
    cs = ClusterSpace(prim, cutoffs)
    sc = StructureContainer(cs)
    for s in structures:
        sc.add_structure(s)
    opt = Optimizer(sc.get_fit_data(), fit_method=method, train_size=0.9, seed=0)
    opt.train()
    params = opt.parameters
    rot_before = rot_after = float("nan")
    if rot == "post_fit":
        rot_before, _ = _rotational_residual_pair(cs, params, mode="off")
        params = _apply_rotational_sum_rules(cs, params)
        rot_after, _ = _rotational_residual_pair(cs, params, mode="off")
    fcs = ForceConstantPotential(cs, params).get_force_constants(ideal)
    asr = {}
    for order in (2, 3):
        try:
            fcs.assert_acoustic_sum_rules(order=order, tol=1e-3); asr[order] = 0.0
        except AssertionError as e:
            import re
            m = re.search(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", str(e))
            asr[order] = float(m.group(1)) if m else float("nan")
    fc2 = fcs.get_fc_array(order=2); fc3 = fcs.get_fc_array(order=3)
    d = OUT / name; d.mkdir(exist_ok=True)
    with h5py.File(d / "fc3.hdf5", "w") as f:
        f.create_dataset("fc3", data=fc3, compression="gzip")
        f.create_dataset("fc2", data=fc2, compression="gzip")
    print(f"{name:10s}: RMSE tr {opt.rmse_train:.4f} te {opt.rmse_test:.4f} | "
          f"ASR2={asr[2]:.1e} ASR3={asr[3]:.1e} | "
          f"rot {rot_before:.2e}->{rot_after:.2e} | "
          f"max|FC3|={np.max(np.abs(fc3)):.2e} -> {d/'fc3.hdf5'}")
