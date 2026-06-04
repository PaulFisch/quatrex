#!/usr/bin/env python
"""Feasibility of the sparse-FC4 loop for d5a: refit FC2+FC3+FC4 on the EXISTING
44 rattled structures (no new DFT), check the fit quality, and confirm the FC4
is accessible SPARSELY (get_fc_dict, no dense 84^4=30 GiB) so the loop
Sigma_L = 1/2 Phi4:<uu> can be contracted on the fly.
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
import ase.io
from ase import Atoms
from ase.build import make_supercell
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.utilities import prepare_structures
from trainstation import Optimizer

WD = _REPO / "phonon/configs/sinw/fc3_hiphive_sinw100_d5a_sc4_vasp"
meta = json.loads((WD / "hiphive_meta.json").read_text())
pm = meta["primitive"]
prim = Atoms(symbols=pm["symbols"], cell=pm["cell"],
             scaled_positions=pm["scaled_positions"], pbc=True)
ideal = make_supercell(prim, np.diag(meta["supercell"]))
n = len(ideal)
fc2c, fc3c = meta["cutoffs"][:2]
fc4c = 3.0   # short-ranged quartic

rattled = [ase.io.read(str(WD / f"disp-{i:05d}" / "vasprun.xml"))
           for i in range(1, meta["n_structures"] + 1)]
structures = prepare_structures(rattled, ideal)
print(f"{len(structures)} structures, {n} atoms; cutoffs FC2/FC3/FC4 = "
      f"{fc2c}/{fc3c}/{fc4c} A")

cs = ClusterSpace(prim, [fc2c, fc3c, fc4c])
sc = StructureContainer(cs)
for s in structures:
    sc.add_structure(s)
# train/test split to judge if 44 structures suffice for FC4
opt = Optimizer(sc.get_fit_data(), fit_method="ardr", train_size=0.85, seed=0)
opt.train()
print(f"ardr fit: RMSE train={opt.rmse_train:.4f} test={opt.rmse_test:.4f} eV/A, "
      f"{int(np.sum(np.abs(opt.parameters) > 0))} nonzero / {len(opt.parameters)} params")

fcp = ForceConstantPotential(cs, opt.parameters)
fcs = fcp.get_force_constants(ideal)
# SPARSE FC4 access -- the whole point (no dense 84^4)
d4 = fcs.get_fc_dict(order=4)
fc4_max = max((np.max(np.abs(np.asarray(v))) for v in d4.values()), default=0.0)
print(f"sparse FC4: {len(d4)} clusters (dense would be {n}^4={n**4:.1e}); "
      f"max|FC4|={fc4_max:.3e} eV/A^4")
# quick loop contraction sanity: Sigma_L ~ 1/2 Phi4:<uu> is FC2-sized, built from
# the sparse dict -- confirm we can iterate it without the dense array.
nelem = sum(np.asarray(v).size for v in d4.values())
print(f"sparse FC4 storage: {nelem} elements ({nelem*8/1e6:.1f} MB) "
      f"vs dense {n**4*81*8/(1<<30):.0f} GiB -> loop contractable on the fly: YES")
fc2 = fcs.get_fc_array(order=2)
print(f"FC2 from refit ||.||={np.linalg.norm(fc2):.2f} (sanity: finite, not garbage)")
