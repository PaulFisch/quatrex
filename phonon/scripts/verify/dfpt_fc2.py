"""Run DFPT FC2 for bulk Si (scf -> ph.x -> q2r.x), skipping the FC3 (d3q.x
plugin not installed). Produces fc2.dat (QE force-constant file) + parses the
FC2 array, for the DFPT-vs-FD-vs-hiPhive dispersion comparison.
"""
import sys
import warnings
from pathlib import Path

_REPO = Path("/usr/scratch/mont-fort11/pfischill/quatrex")
for p in (_REPO / "phonon", _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from phonon_inputs.config import load_config  # noqa: E402
from phonon_inputs.structure import load_structure  # noqa: E402
from phonon_inputs import dfpt  # noqa: E402

cfg_path = _REPO / "phonon/configs/si_primitive/dfpt.yaml"
cfg = load_config(cfg_path)
cell = load_structure(cfg.structure)
work_dir = cfg_path.parent / cfg.dfpt.work_dir
work_dir.mkdir(parents=True, exist_ok=True)

print("=== DFPT FC2: sow ===", flush=True)
dfpt.sow(cell, work_dir, cfg.qe, cfg.dfpt)
print("=== SCF ===", flush=True)
dfpt.run_scf(work_dir, cfg.qe, timeout=3600)
print("=== ph.x (DFPT) ===", flush=True)
dfpt.run_ph(work_dir, cfg.dfpt)
print("=== q2r.x (FC2) ===", flush=True)
dfpt.run_q2r(work_dir, cfg.dfpt)

nat = len(cell)
fc2, info = dfpt._parse_q2r_fc2(work_dir / "fc2.dat", nat, cfg.dfpt.q_mesh)
np.savez(work_dir / "dfpt_fc2.npz", fc2=fc2)
print(f"=== DONE: FC2 shape {fc2.shape}, max {np.max(np.abs(fc2)):.4e} eV/A^2; "
      f"fc2.dat at {work_dir/'fc2.dat'} ===", flush=True)
