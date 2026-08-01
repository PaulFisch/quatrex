"""MoS2 SCP route: 4th-order refit + self-consistent phonons at T.

The eta=0 film SCBA diverges because the cubic-only bubble softens the
already-soft vdW interlayer modes without the quartic stiffening that
stabilises real layered crystals at 300 K. This script produces the
SCP-renormalised effective fc2(T):

  1. refit the merged displacement batches with orders 2+3+4 -- the
     production fit ([6.0, 4.0]) has NO quartic; the 4th-order cutoff
     must span the vdW gap (cross-gap S-S 3.53 A), else the interlayer
     stiffening is absent by construction. Gate: CV force RMSE must not
     degrade vs the 3rd-order fit.
  2. hiphive self_consistent_harmonic_model at T with the 4th-order
     FCP as the force calculator (QM_statistics: quantum amplitudes).
  3. gates vs EXPERIMENT: interlayer shear/breathing (bulk 2H-MoS2
     Raman ~1.0 / ~1.7 THz at 300 K), no imaginary modes on the mesh,
     c-axis acoustic bandwidth. Writes fcp_scp<T>.fcp + fc2 + a
     0K-fit-vs-SCP comparison table.

Run on a cluster node:
  python phonon/studies/_mos2_scp.py \
      --data <workdir> [<workdir_b> <workdir_c>] --out <outdir> \
      [--c4 4.0 --temperature 300 --alpha 0.2 --n-iter 40 --n-struct 60]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "phonon")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def gamma_gates(fc2, prim_atoms, supercell_matrix, mesh=(8, 8, 4)):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    unit = PhonopyAtoms(symbols=prim_atoms.get_chemical_symbols(),
                        cell=np.array(prim_atoms.cell),
                        scaled_positions=prim_atoms.get_scaled_positions())
    ph = Phonopy(unit, supercell_matrix=np.diag(supercell_matrix),
                 primitive_matrix=np.eye(3))
    ph.force_constants = fc2
    ph.run_mesh(list(mesh), with_eigenvectors=False)
    freqs = ph.get_mesh_dict()["frequencies"]
    g = ph.get_frequencies([0, 0, 0])
    a = ph.get_frequencies([0, 0, 0.5])
    return {
        "gamma_low6_THz": [float(x) for x in np.sort(g)[:6]],
        "A_low4_THz": [float(x) for x in np.sort(a)[:4]],
        "mesh_fmin_THz": float(freqs.min()),
        "mesh_fmax_THz": float(freqs.max()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--c4", type=float, default=4.0,
                   help="4th-order cutoff (A); must exceed the cross-gap "
                        "S-S distance 3.53 A for interlayer stiffening")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--n-iter", type=int, default=40)
    p.add_argument("--n-struct", type=int, default=60)
    p.add_argument("--classical", action="store_true",
                   help="classical amplitudes (default: QM_statistics)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import ase
    from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer
    from hiphive.calculators import ForceConstantCalculator
    from hiphive.self_consistent_phonons import self_consistent_harmonic_model
    from hiphive.utilities import prepare_structures
    from trainstation import CrossValidationEstimator, Optimizer
    from phonon_inputs import thirdorder as _to
    from phonon_inputs.hiphive_convergence import _apply_rotational_sum_rules
    from phonon_inputs.hiphive_fc3 import (
        _atoms_from_meta, _fold_positions_to_min_image, _load_meta,
        _read_positions_from_vasp_poscar)

    data_dirs = [Path(d) for d in a.data]
    meta = _load_meta(data_dirs[0])
    atoms_ideal = _atoms_from_meta(meta["supercell_atoms"])
    prim = _atoms_from_meta(meta["primitive"])
    n_super = len(atoms_ideal)

    rattled = []
    for dd in data_dirs:
        m = _load_meta(dd)
        for i in range(1, m["n_structures"] + 1):
            disp = dd / f"disp-{i:05d}"
            if not (disp / "vasprun.xml").exists():
                continue
            forces = _to._parse_vasp_forces(disp, n_super)
            pos = _read_positions_from_vasp_poscar(disp / "POSCAR", n_super)
            pos = _fold_positions_to_min_image(pos, atoms_ideal.positions,
                                               atoms_ideal.cell)
            rat = ase.Atoms(symbols=list(atoms_ideal.get_chemical_symbols()),
                            cell=atoms_ideal.cell, positions=pos, pbc=True)
            rat.arrays["forces"] = forces
            rattled.append(rat)
    print(f"structures: {len(rattled)}", flush=True)

    cutoffs3 = list(meta["cutoffs"])            # [6.0, 4.0]
    cutoffs4 = cutoffs3 + [a.c4]                # + quartic spanning the gap
    results = {"cutoffs4": cutoffs4, "temperature": a.temperature}

    for tag, co in (("o3", cutoffs3), ("o4", cutoffs4)):
        with contextlib.redirect_stdout(io.StringIO()):
            cs = ClusterSpace(prim, co)
            sc = StructureContainer(cs)
            for s in prepare_structures(rattled, atoms_ideal):
                sc.add_structure(s)
        fd = sc.get_fit_data()
        with contextlib.redirect_stdout(io.StringIO()):
            cve = CrossValidationEstimator(fd, fit_method="least-squares",
                                           validation_method="k-fold",
                                           n_splits=5)
            cve.validate()
            opt = Optimizer(fd, fit_method="least-squares", train_size=1.0)
            opt.train()
        print(f"{tag}: cutoffs {co}  n_dofs {cs.n_dofs}  "
              f"CV RMSE {cve.rmse_validation:.5e}", flush=True)
        results[f"rmse_cv_{tag}"] = float(cve.rmse_validation)
        results[f"n_dofs_{tag}"] = int(cs.n_dofs)
        if tag == "o4":
            params4 = _apply_rotational_sum_rules(cs, opt.parameters)
            cs4, fcp4 = cs, ForceConstantPotential(cs, params4)

    fcp4.write(str(out / "fcp_o4.fcp"))
    fc2_bare = fcp4.get_force_constants(atoms_ideal).get_fc_array(order=2)
    g0 = gamma_gates(fc2_bare, prim, [4, 4, 1])
    print("bare (0 K fit) gates:", json.dumps(g0), flush=True)
    results["gates_bare"] = g0

    # ---- SCP at T ----
    fcs4 = fcp4.get_force_constants(atoms_ideal)
    calc = ForceConstantCalculator(fcs4)
    with contextlib.redirect_stdout(io.StringIO()):
        cs2 = ClusterSpace(prim, [cutoffs3[0]])   # harmonic-only model space
    print(f"SCP: T={a.temperature} K, alpha={a.alpha}, "
          f"{a.n_iter} iters x {a.n_struct} structures, "
          f"{'classical' if a.classical else 'QM'} amplitudes", flush=True)
    param_traj = self_consistent_harmonic_model(
        atoms_ideal, calc, cs2, a.temperature, a.alpha,
        a.n_iter, a.n_struct, QM_statistics=not a.classical)
    drift = [float(np.linalg.norm(param_traj[i] - param_traj[i - 1]))
             for i in range(1, len(param_traj))]
    print("SCP param drift (every 5th):",
          [f"{d:.3e}" for d in drift[::5]], flush=True)
    results["scp_drift_last"] = drift[-1]

    fcp_scp = ForceConstantPotential(cs2, param_traj[-1])
    fcp_scp.write(str(out / f"fcp_scp{int(a.temperature)}.fcp"))
    fc2_scp = fcp_scp.get_force_constants(atoms_ideal).get_fc_array(order=2)
    gs = gamma_gates(fc2_scp, prim, [4, 4, 1])
    print(f"SCP({a.temperature:.0f} K) gates:", json.dumps(gs), flush=True)
    results["gates_scp"] = gs

    import h5py
    with h5py.File(out / f"fc2_scp{int(a.temperature)}.hdf5", "w") as f:
        f.create_dataset("fc2", data=fc2_scp, compression="gzip")
    json.dump(results, open(out / "scp_results.json", "w"), indent=1)
    print(f"wrote {out}/scp_results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
