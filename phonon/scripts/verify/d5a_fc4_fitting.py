#!/usr/bin/env python
"""Can we extract a usable d5a FC4 from the EXISTING 44 rattled structures?
Compare regressions (ardr/rfe/ridge) on the FC2+FC3+FC4 ClusterSpace: a usable
FC4 must be NON-ZERO and have a test RMSE no worse than the FC2+FC3-only fit
(otherwise it's overfit noise -> the data doesn't constrain the quartic).
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
fc2c, fc3c = meta["cutoffs"][:2]
rattled = [ase.io.read(str(WD / f"disp-{i:05d}" / "vasprun.xml"))
           for i in range(1, meta["n_structures"] + 1)]
structures = prepare_structures(rattled, ideal)


def fit(cutoffs, method, **kw):
    cs = ClusterSpace(prim, cutoffs)
    sc = StructureContainer(cs)
    for s in structures:
        sc.add_structure(s)
    opt = Optimizer(sc.get_fit_data(), fit_method=method, train_size=0.85, seed=0, **kw)
    opt.train()
    fcs = ForceConstantPotential(cs, opt.parameters).get_force_constants(ideal)
    fc4 = None
    try:
        d4 = fcs.get_fc_dict(order=4)
        fc4 = max((np.max(np.abs(np.asarray(v))) for v in d4.values()), default=0.0)
    except Exception:
        pass
    return opt.rmse_train, opt.rmse_test, fc4


print(f"d5a: {len(structures)} structures, FC2/FC3 cutoffs {fc2c}/{fc3c} A\n")
# baseline: FC2+FC3 only
r2, t2, _ = fit([fc2c, fc3c], "ardr")
print(f"FC2+FC3 only  (ardr): RMSE train {r2:.4f} test {t2:.4f}")
print("FC2+FC3+FC4 (c4=3.0):")
for m in ("ardr", "rfe", "ridge"):
    try:
        rtr, tte, f4 = fit([fc2c, fc3c, 3.0], m)
        verdict = ("FC4 zeroed" if (f4 or 0) < 1e-9 else
                   ("usable" if tte <= t2 * 1.15 else "OVERFIT noise"))
        print(f"  {m:6s}: RMSE train {rtr:.4f} test {tte:.4f}  max|FC4|={f4:.3e}  -> {verdict}")
    except Exception as e:
        print(f"  {m:6s}: FAILED {e}")
