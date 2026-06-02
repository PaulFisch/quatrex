"""Bulk Ge thermal conductivity (phono3py RTA) from the freshly-computed Ge FC2/FC3.

Literature Ge kappa(300 K) ~ 60 W/mK (Glassbrenner-Slack experiment). As for Si (F7/F13),
a 2x2x2 FC3 supercell underestimates by ~20-30%, so ~40-55 W/mK is the expected DFT-RTA
result; the 11^3 -> 19^3 mesh value should rise. This is the new-material validation.

Usage:  python ge_bulk_kappa.py [fc3_dir]   (default reaps/ge_primitive_work)
"""
import sys
import warnings
from pathlib import Path

_W = Path("/usr/scratch/mont-fort11/pfischill/quatrex/phonon")
for p in (_W.parent, _W):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

import numpy as np
import phono3py

d = Path(sys.argv[1]) if len(sys.argv) > 1 else _W / "reaps/ge_primitive_work"
yaml = d / "phono3py.yaml"
fc3 = d / "fc3.hdf5"
fc2 = d / "fc2.hdf5"
print(f"Ge bulk kappa from {d}", flush=True)

ph3 = phono3py.load(phono3py_yaml=str(yaml), log_level=1, produce_fc=False)
import h5py
with h5py.File(fc3, "r") as f:
    ph3.fc3 = f["fc3"][:]
with h5py.File(fc2, "r") as f:
    ph3.fc2 = f["force_constants"][:]
print(f"  primitive: {ph3.primitive.symbols}, a = {np.linalg.norm(ph3.primitive.cell[0]):.4f} A",
      flush=True)

for mesh in (11, 19):
    ph3.mesh_numbers = [mesh, mesh, mesh]
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(temperatures=[200, 300, 400, 500, 600],
                                 write_kappa=True)
    tc = ph3.thermal_conductivity
    kappa = tc.kappa[0]  # (n_temp, 6) -> xx component is [:,0]
    T = tc.temperatures
    kxx = kappa[:, 0]
    print(f"  mesh {mesh}^3: kappa_xx(200/300/400/500/600 K) = "
          f"{np.round(kxx, 1).tolist()} W/mK", flush=True)
    if mesh == 19:
        i300 = int(np.argmin(np.abs(np.array(T) - 300)))
        print(f"  >>> Ge kappa(300 K) = {kxx[i300]:.1f} W/mK "
              f"(literature ~60; expect ~40-55 from 2x2x2 FC3) <<<", flush=True)
print("[done]")
