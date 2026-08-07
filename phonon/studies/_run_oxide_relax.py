"""VASP relaxation of the oxide-embedded d5a wire (Phase 2, step 1).

Fixed-cell (ISIF=2) ionic relaxation of the 168-atom wire+SiO2-shell
structure with PBE+D3 (IVDW=11 via vasp.incar_extra). Mirrors
phonon_inputs.pipeline Step 1; idempotent (a completed CONTCAR skips).

After convergence, prints the wire-geometry drift (the shell must
perturb, not distort, the wire) and the shell-wire contact statistics.

Run on the cluster (needs the VASP binary + POTCARs):
    python phonon/scripts/tortin.py launch --name oxrelax -- \
        python phonon/studies/_run_oxide_relax.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "phonon"))

CFG = REPO / "phonon/configs/sinw/sinw100_d5a_oxide_vasp_sc3.yaml"
N_WIRE = 21  # Si9H12 -- the first 21 atoms of the inline structure


def main() -> int:
    from phonon_inputs.config import load_config
    from phonon_inputs.structure import load_structure
    from phonon_inputs.qe_interface import run_vasp_relax

    config = load_config(CFG)
    base_dir = CFG.parent
    cell = load_structure(config.structure)
    ref = cell.positions.copy()

    rc = config.relax
    relaxed = run_vasp_relax(
        cell, base_dir / rc.work_dir, config.vasp,
        calculation=rc.calculation,
        forc_conv_thr=rc.forc_conv_thr,
        timeout=172800,  # 48 h -- 168-atom PBE+D3 relax, not a wire cell
        # legs continue from the previous leg's CONTCAR (oxrelax1-4 each
        # restarted from the original structure and re-paid the descent)
        restart_from_contcar=True,
    )

    # Wire drift diagnostics (first N_WIRE atoms are the wire).
    drift = np.linalg.norm(relaxed.positions[:N_WIRE] - ref[:N_WIRE], axis=1)
    print(f"[relax] wire drift: mean {drift.mean():.3f} A, "
          f"max {drift.max():.3f} A", flush=True)
    if drift.max() > 0.5:
        print("[warn ] wire distorted by the shell (max drift > 0.5 A) -- "
              "inspect before sampling.", flush=True)
    # Contact statistics: nearest shell atom per wire-H.
    wp = relaxed.positions[:N_WIRE]
    sp = relaxed.positions[N_WIRE:]
    hw = [i for i, s in enumerate(relaxed.symbols[:N_WIRE]) if s == "H"]
    dmin = [float(np.min(np.linalg.norm(sp - wp[i], axis=1))) for i in hw]
    print(f"[relax] wire-H to shell contact: min {min(dmin):.2f} A, "
          f"median {np.median(dmin):.2f} A", flush=True)
    print("[done ] oxide relax complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
