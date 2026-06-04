#!/usr/bin/env python
"""CSLD/TDEP-style 'direct loop' estimate for d5a: fit an EFFECTIVE (renormalised)
FC2 from the existing thermally-rattled structures (no new DFT) and compare the
soft-twist frequency to the harmonic FC2. The effective FC2 absorbs the static
anharmonic renormalisation (loop + tadpole) at the rattle's effective amplitude
-- the same idea CSLD/TDEP/SSCHA use to get the loop without an explicit FC4.
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
fc2_cut = meta["cutoffs"][0]

# load the rattled structures + forces (ASE reads vasprun.xml)
rattled = []
for i in range(1, meta["n_structures"] + 1):
    d = WD / f"disp-{i:05d}"
    a = ase.io.read(str(d / "vasprun.xml"))
    rattled.append(a)
print(f"loaded {len(rattled)} rattled structures ({n} atoms), FC2 cutoff {fc2_cut} A")
structures = prepare_structures(rattled, ideal)

# effective (renormalised) FC2: best harmonic fit to the thermal data
cs = ClusterSpace(prim, [fc2_cut])
sc = StructureContainer(cs)
for s in structures:
    sc.add_structure(s)
opt = Optimizer(sc.get_fit_data(), fit_method="ardr", train_size=1.0)
opt.train()
print(f"effective-FC2 fit: RMSE={opt.rmse_train:.4f} eV/A, {len(opt.parameters)} params")
fcp = ForceConstantPotential(cs, opt.parameters)
fc2_eff = fcp.get_force_constants(ideal).get_fc_array(order=2)

# harmonic FC2 (from the original reap)
import h5py
with h5py.File(WD / "fc3.hdf5", "r") as f:
    fc2_harm = f["fc2"][:]


def gamma_freqs(fc2):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    pa = PhonopyAtoms(symbols=pm["symbols"], cell=pm["cell"],
                      scaled_positions=pm["scaled_positions"])
    ph = Phonopy(pa, supercell_matrix=np.diag(meta["supercell"]),
                 primitive_matrix=np.eye(3))
    ph.force_constants = fc2
    ph.run_qpoints([[0, 0, 0]])
    return np.sort(ph.get_qpoints_dict()["frequencies"][0])  # THz


fh = gamma_freqs(fc2_harm)
fe = gamma_freqs(fc2_eff)
print("\n  lowest 6 Gamma freqs (THz):")
print(f"    harmonic : {np.array2string(fh[:6], precision=4)}")
print(f"    effective: {np.array2string(fe[:6], precision=4)}")
# soft twist = lowest non-translational mode (skip 3 acoustic ~0)
soft_h = fh[fh > 0.005][0] if np.any(fh > 0.005) else fh[3]
soft_e = fe[fe > 0.005][0] if np.any(fe > 0.005) else fe[3]
print(f"\n  soft-twist mode: harmonic {soft_h:.4f} THz -> effective {soft_e:.4f} THz "
      f"({'stiffened' if soft_e > soft_h else 'softened'} "
      f"{abs(soft_e - soft_h) / max(soft_h, 1e-6) * 100:.0f}%)")
print(f"  effective-FC2 renorm ||dFC2||/||FC2|| = "
      f"{np.linalg.norm(fc2_eff - fc2_harm) / np.linalg.norm(fc2_harm):.4f}")
