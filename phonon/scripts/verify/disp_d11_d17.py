import subprocess, sys
from pathlib import Path
_REPO=Path("/usr/scratch/mont-fort11/pfischill/quatrex")
for cfg in ["sinw100_d11a_vasp_sc4.yaml","sinw100_d17a_vasp_sc4.yaml"]:
    print(f"\n######## {cfg} ########",flush=True)
    subprocess.run([sys.executable,"-u",str(_REPO/"phonon/scripts/verify/dispersion_check.py"),
                    str(_REPO/"phonon/configs/sinw"/cfg),"2"])
