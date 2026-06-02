"""kappa(T) from the DFPT FC3 (D3Q, norm-conserving pseudo) vs the FD FC3 (symfc),
for bulk Si. Both force constants are on the same 2x2x2 supercell / same lattice,
so the DFPT FC is loaded into the FD phono3py structure and kappa is recomputed.
"""
from pathlib import Path

import numpy as np
import h5py
import phono3py

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
fd_dir = _REPO / "phonon/reaps/si_primitive_work"
dfpt_fc3 = _REPO / "phonon_old/input_calc/dfpt/fc3.hdf5"

with h5py.File(dfpt_fc3, "r") as f:
    d_fc2 = np.array(f["fc2"])
    d_fc3 = np.array(f["fc3"])     # compact (n_patom, n_super, n_super, 3,3,3)
print(f"DFPT fc2 {d_fc2.shape}, fc3 {d_fc3.shape}")

MESH = 13
res = {}
for tag in ("FD", "DFPT"):
    ph3 = phono3py.load(str(fd_dir / "phono3py.yaml"),
                        fc3_filename=str(fd_dir / "fc3.hdf5"),
                        fc2_filename=str(fd_dir / "fc2.hdf5"), log_level=0)
    if tag == "DFPT":
        ph3.fc2 = d_fc2
        ph3.fc3 = d_fc3
    ph3.mesh_numbers = [MESH, MESH, MESH]
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(temperatures=[200, 300, 400, 500, 600],
                                 write_kappa=False)
    k = np.array(ph3.thermal_conductivity.kappa)[0, :, 0]  # xx vs T
    res[tag] = k
    print(f"[{tag}] kappa(200..600) = {np.round(k, 1)} W/mK")

print("\n=== DFPT vs FD kappa(T), bulk Si ===")
for i, T in enumerate([200, 300, 400, 500, 600]):
    print(f"T={T}: FD {res['FD'][i]:.1f}  DFPT {res['DFPT'][i]:.1f}  "
          f"diff {100*abs(res['DFPT'][i]-res['FD'][i])/res['FD'][i]:.0f}%")
