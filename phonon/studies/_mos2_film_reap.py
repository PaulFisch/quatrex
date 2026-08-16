"""Re-materialize the MoS2 FCs on a [4,4,3] supercell for the film build.

Run: python phonon/studies/_mos2_film_reap.py         --reap cluster/mos2_reap --out cluster/mos2_film_reap
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from ase import Atoms

SUPERCELL = (4, 4, 3)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reap", required=True,
                   help="dir with fcp.fcp + hiphive_meta.json")
    p.add_argument("--out", required=True)
    p.add_argument("--fc2-fcp", default=None,
                   help="optional separate FCP for the HARMONIC part (e.g. "
                        "the SCP-renormalised fcp_scp300.fcp); fc3 still "
                        "comes from --reap's fcp.fcp")
    a = p.parse_args()
    reap, out = Path(a.reap), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from hiphive import ForceConstantPotential
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    meta = json.load(open(reap / "hiphive_meta.json"))
    prim = meta["primitive"]
    # PHONOPY supercell ordering, not ase.repeat: the downstream loader
    # and slab mapping index the FC arrays in phonopy convention, and
    # the fit pipeline's atoms_ideal used it too (verified: this
    # round-trips the saved [4,4,1] fc2 to 0.0 exactly, while
    # ase.repeat ordering scrambles it).
    unit = PhonopyAtoms(symbols=prim["symbols"],
                        cell=np.array(prim["cell"]),
                        scaled_positions=np.array(prim["scaled_positions"]))
    ph = Phonopy(unit, supercell_matrix=np.diag(SUPERCELL),
                 primitive_matrix=np.eye(3))
    fcp = ForceConstantPotential.read(str(reap / "fcp.fcp"))
    sc = Atoms(symbols=ph.supercell.symbols, cell=ph.supercell.cell,
               positions=ph.supercell.positions, pbc=True)
    print(f"primitive {len(unit)} atoms -> phonopy supercell {SUPERCELL} "
          f"({len(sc)} atoms)", flush=True)

    fcs = fcp.get_force_constants(sc)
    fc2 = fcs.get_fc_array(order=2)          # (n, n, 3, 3)
    if a.fc2_fcp is not None:
        fcp2 = ForceConstantPotential.read(str(a.fc2_fcp))
        fc2 = fcp2.get_force_constants(sc).get_fc_array(order=2)
        print(f"fc2 REPLACED from {a.fc2_fcp}", flush=True)
    print(f"fc2 {fc2.shape}, max {np.abs(fc2).max():.3f} eV/A^2",
          flush=True)
    fc3 = fcs.get_fc_array(order=3)          # (n, n, n, 3, 3, 3)
    print(f"fc3 {fc3.shape}, max {np.abs(fc3).max():.3f} eV/A^3, "
          f"{fc3.nbytes / 1e9:.1f} GB dense", flush=True)

    # acoustic-sum sanity on the re-materialized fc2
    asr = np.abs(fc2.sum(axis=1)).max()
    print(f"fc2 ASR residual (sum_j): {asr:.2e} eV/A^2", flush=True)

    with h5py.File(out / "fc3.hdf5", "w") as f:
        f.create_dataset("fc2", data=fc2, compression="gzip")
        f.create_dataset("fc3", data=fc3, compression="gzip")
    meta_out = dict(meta)
    meta_out["supercell"] = list(SUPERCELL)
    meta_out["rematerialized_from"] = str(reap / "fcp.fcp")
    json.dump(meta_out, open(out / "hiphive_meta.json", "w"), indent=1)
    print(f"wrote {out}/fc3.hdf5 + hiphive_meta.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
