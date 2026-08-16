"""MoS2 fc3 refit: alternative fit methods on the existing displacement data.

Run: python phonon/studies/_mos2_fc3_refit.py         --data cluster/mos2_refit/data --out cluster/mos2_refit
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

METHODS = ["least-squares", "ridge", "bayesian-ridge", "ardr"]


def load_container(data_dirs: list[Path]):
    """Rebuild the hiphive StructureContainer from one or more reaped
    VASP dirs (extra displacement batches merge into one fit)."""
    import ase
    from hiphive import ClusterSpace, StructureContainer
    from hiphive.utilities import prepare_structures
    from phonon_inputs import thirdorder as _to
    from phonon_inputs.hiphive_fc3 import (
        _atoms_from_meta, _fold_positions_to_min_image, _load_meta,
        _read_positions_from_vasp_poscar)

    meta = _load_meta(data_dirs[0])
    atoms_ideal = _atoms_from_meta(meta["supercell_atoms"])
    primitive = _atoms_from_meta(meta["primitive"])
    n_super = len(atoms_ideal)

    rattled = []
    for data_dir in data_dirs:
        m = _load_meta(data_dir)
        if not np.allclose(_atoms_from_meta(m["supercell_atoms"]).positions,
                           atoms_ideal.positions, atol=1e-8):
            raise ValueError(f"{data_dir}: supercell differs from {data_dirs[0]}")
        n_batch = 0
        for i in range(1, m["n_structures"] + 1):
            disp_dir = data_dir / f"disp-{i:05d}"
            if not (disp_dir / "vasprun.xml").exists():
                continue  # batch still running -- use what's there
            forces = _to._parse_vasp_forces(disp_dir, n_super)
            positions = _read_positions_from_vasp_poscar(
                disp_dir / "POSCAR", n_super)
            positions = _fold_positions_to_min_image(
                positions, atoms_ideal.positions, atoms_ideal.cell)
            rat = ase.Atoms(symbols=list(atoms_ideal.get_chemical_symbols()),
                            cell=atoms_ideal.cell, positions=positions, pbc=True)
            rat.arrays["forces"] = forces
            rattled.append(rat)
            n_batch += 1
        print(f"  {data_dir}: {n_batch} structures", flush=True)
    print(f"loaded {len(rattled)} rattled structures ({n_super} atoms)", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        cs = ClusterSpace(primitive, list(meta["cutoffs"]))
        sc = StructureContainer(cs)
        for s in prepare_structures(rattled, atoms_ideal):
            sc.add_structure(s)
    print(f"cluster space: {cs.n_dofs} DOF, cutoffs {list(meta['cutoffs'])}",
          flush=True)
    return meta, atoms_ideal, primitive, cs, sc


def cross_gap_metrics(fc3: np.ndarray, zfrac: np.ndarray) -> dict:
    """Max/frob of fc3 split by MoS2-layer membership ([4,4,1]: 2 layers)."""
    lay = (zfrac >= 0.5).astype(int)
    same = ((lay[:, None, None] == lay[None, :, None])
            & (lay[:, None, None] == lay[None, None, :]))
    a = np.abs(fc3).max(axis=(3, 4, 5))
    fr2 = (fc3 ** 2).sum(axis=(3, 4, 5))
    return {
        "cross_max": float(a[~same].max()),
        "cross_frob": float(np.sqrt(fr2[~same].sum())),
        "intra_max": float(a[same].max()),
        "intra_frob": float(np.sqrt(fr2[same].sum())),
    }


def fc2_gates(fc2: np.ndarray, primitive, supercell_matrix) -> dict:
    """Min/max frequency on a q-mesh + ASR residual for a [4,4,1] fc2."""
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    unit = PhonopyAtoms(symbols=primitive.get_chemical_symbols(),
                        cell=np.array(primitive.cell),
                        scaled_positions=primitive.get_scaled_positions())
    ph = Phonopy(unit, supercell_matrix=np.diag(supercell_matrix),
                 primitive_matrix=np.eye(3))
    ph.force_constants = fc2
    ph.run_mesh([8, 8, 4], with_eigenvectors=False)
    freqs = ph.get_mesh_dict()["frequencies"]
    return {
        "fmin_THz": float(freqs.min()),
        "fmax_THz": float(freqs.max()),
        "asr_resid": float(np.abs(fc2.sum(axis=1)).max()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, nargs="+",
                   help="one or more reaped VASP workdirs (batches merge)")
    p.add_argument("--out", required=True)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--skip-hdf5", action="store_true",
                   help="metrics only, no fc3_<method>.hdf5 output")
    a = p.parse_args()
    data_dirs, out = [Path(d) for d in a.data], Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import h5py
    from hiphive import ForceConstantPotential
    from trainstation import CrossValidationEstimator, Optimizer
    from phonon_inputs.hiphive_convergence import _apply_rotational_sum_rules

    meta, atoms_ideal, primitive, cs, sc = load_container(data_dirs)
    fit_data = sc.get_fit_data()
    A, y = fit_data
    print(f"fit matrix: {A.shape}, target {y.shape}", flush=True)
    zfrac = atoms_ideal.get_scaled_positions()[:, 2]
    sc_matrix = [r[i] for i, r in enumerate(meta["supercell_matrix"])] \
        if isinstance(meta.get("supercell_matrix"), list) else [4, 4, 1]

    results = {}
    for method in a.methods.split(","):
        method = method.strip()
        print(f"\n=== {method} ===", flush=True)
        # honest CV
        with contextlib.redirect_stdout(io.StringIO()):
            cve = CrossValidationEstimator(
                fit_data, fit_method=method,
                validation_method="k-fold", n_splits=a.n_splits)
            cve.validate()
        rmse_cv = float(cve.rmse_validation)

        # fold-stability of the cross-gap signal: refit on each CV-style
        # subset and track the cross-gap frobenius norm
        rng = np.random.default_rng(7)
        n = len(sc)
        rows_per = A.shape[0] // n
        cross_frobs = []
        for k in range(a.n_splits):
            test = rng.choice(n, size=max(1, n // a.n_splits), replace=False)
            train_rows = np.concatenate(
                [np.arange(i * rows_per, (i + 1) * rows_per)
                 for i in range(n) if i not in set(test)])
            with contextlib.redirect_stdout(io.StringIO()):
                o = Optimizer((A[train_rows], y[train_rows]),
                              fit_method=method, train_size=1.0)
                o.train()
            pk = _apply_rotational_sum_rules(cs, o.parameters)
            fcp_k = ForceConstantPotential(cs, pk)
            fc3_k = fcp_k.get_force_constants(atoms_ideal).get_fc_array(order=3)
            cross_frobs.append(cross_gap_metrics(fc3_k, zfrac)["cross_frob"])
        cross_frobs = np.array(cross_frobs)

        # production-parity final fit (all data) + rotational projection
        with contextlib.redirect_stdout(io.StringIO()):
            opt = Optimizer(fit_data, fit_method=method, train_size=1.0)
            opt.train()
        params = _apply_rotational_sum_rules(cs, opt.parameters)
        fcp = ForceConstantPotential(cs, params)
        fcs = fcp.get_force_constants(atoms_ideal)
        fc2 = fcs.get_fc_array(order=2)
        fc3 = fcs.get_fc_array(order=3)

        m = cross_gap_metrics(fc3, zfrac)
        g = fc2_gates(fc2, primitive, sc_matrix)
        r = {
            "rmse_train": float(opt.rmse_train),
            "rmse_cv": rmse_cv,
            "cross_frob_folds_mean": float(cross_frobs.mean()),
            "cross_frob_folds_std": float(cross_frobs.std()),
            **m, **g,
        }
        results[method] = r
        print(f"  RMSE train {r['rmse_train']:.4e}  CV {rmse_cv:.4e}")
        print(f"  cross-gap fc3: max {m['cross_max']:.4f}  frob {m['cross_frob']:.4f}"
              f"  (folds {cross_frobs.mean():.4f} +/- {cross_frobs.std():.4f})")
        print(f"  intra fc3: max {m['intra_max']:.4f}  frob {m['intra_frob']:.4f}")
        print(f"  fc2 gates: fmin {g['fmin_THz']:+.4f}  fmax {g['fmax_THz']:.4f}"
              f"  ASR {g['asr_resid']:.2e}", flush=True)

        fcp.write(str(out / f"fcp_{method}.fcp"))
        if not a.skip_hdf5:
            with h5py.File(out / f"fc3_{method}.hdf5", "w") as f:
                f.create_dataset("fc2", data=fc2, compression="gzip")
                f.create_dataset("fc3", data=fc3, compression="gzip")

    json.dump(results, open(out / "refit_results.json", "w"), indent=1)
    print(f"\nwrote {out}/refit_results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
