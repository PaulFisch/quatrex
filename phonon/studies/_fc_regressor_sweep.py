"""Regressor sweep on an existing displacement set: methods x training size.

Generalises ``_mos2_fc3_refit.py`` to any reaped work_dir (VASP or QE) and
adds the two axes that comparison needs and the reap does not record: the
number of NON-ZERO coefficients, and a k-fold cross-validated error. The
first matters because ``n_parameters`` is the cluster-space dimension and is
identical for least squares and ARD, so sparsity is otherwise invisible; the
second because a single hold-out split is noisy at these training-set sizes.

Fransson, Eriksson & Erhart, npj Comput. Mater. 6, 135 (2020) is the
reference protocol: sparsity against cross-validated error is their Fig. 7,
and their warning is that the CV score alone does not rank models -- a
physical observable has to. This script therefore also reports the gates
that are physical: imaginary modes, the frequency range, the acoustic sum
rule, and (optionally) the cross-gap third-order weight of a layered
crystal.

Run:
  python phonon/studies/_fc_regressor_sweep.py --data <workdir> --out <dir>
  python phonon/studies/_fc_regressor_sweep.py --data <workdir> --out <dir> \
      --methods least-squares,lasso,ardr,rfe --train-sizes 5,10,20,40
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "phonon")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# every method trainstation 1.2 exposes (fit_methods.py:719)
ALL_METHODS = [
    "least-squares", "least-squares-with-reg-matrix", "ridge", "bayesian-ridge",
    "lasso", "adaptive-lasso", "elasticnet", "omp", "split-bregman",
    "ardr", "rfe",
]
DEFAULT_METHODS = ["least-squares", "ridge", "bayesian-ridge", "lasso",
                   "adaptive-lasso", "ardr", "rfe"]

# A coefficient below this fraction of the largest one counts as pruned.
# Sparsity-promoting fits drive parameters to exact zero, so the threshold
# only guards against denormal dust; the count is insensitive to it.
NONZERO_REL_TOL = 1e-10


def load_container(data_dirs: list[Path], cutoffs=None):
    """Rebuild a hiphive StructureContainer from reaped VASP or QE dirs."""
    import ase
    from hiphive import ClusterSpace, StructureContainer
    from hiphive.utilities import prepare_structures
    from phonon_inputs import thirdorder as _to
    from phonon_inputs.hiphive_fc3 import (
        _atoms_from_meta, _fold_positions_to_min_image, _load_meta,
        _read_positions_from_qe_input, _read_positions_from_vasp_poscar)

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
            out_file = data_dir / f"disp-{i:05d}.out"
            inp_file = data_dir / f"disp-{i:05d}.in"
            if (disp_dir / "vasprun.xml").exists():
                forces = _to._parse_vasp_forces(disp_dir, n_super)
                positions = _read_positions_from_vasp_poscar(
                    disp_dir / "POSCAR", n_super)
            elif out_file.exists() and inp_file.exists():
                forces = _to._parse_qe_forces(out_file, n_super)
                positions = _read_positions_from_qe_input(inp_file, n_super)
            else:
                continue  # not finished / not present -- use what is there
            positions = _fold_positions_to_min_image(
                positions, atoms_ideal.positions, atoms_ideal.cell)
            rat = ase.Atoms(symbols=list(atoms_ideal.get_chemical_symbols()),
                            cell=atoms_ideal.cell, positions=positions, pbc=True)
            rat.arrays["forces"] = forces
            rattled.append(rat)
            n_batch += 1
        print(f"  {data_dir}: {n_batch} structures", flush=True)
    if not rattled:
        raise SystemExit(f"no usable displacement data under {data_dirs}")

    cut = list(cutoffs) if cutoffs else list(meta["cutoffs"])
    with contextlib.redirect_stdout(io.StringIO()):
        cs = ClusterSpace(primitive, cut)
        sc = StructureContainer(cs)
        for s in prepare_structures(rattled, atoms_ideal):
            sc.add_structure(s)
    print(f"loaded {len(rattled)} structures ({n_super} atoms); "
          f"cluster space {cs.n_dofs} DOF, cutoffs {cut}", flush=True)
    return meta, atoms_ideal, primitive, cs, sc


def layer_split_metrics(fc3: np.ndarray, zfrac: np.ndarray) -> dict:
    """Third-order weight split by layer membership (layered crystals only)."""
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


def fc2_gates(fc2, primitive, supercell_matrix, mesh) -> dict:
    """Frequency range, imaginary-mode count and ASR residual on a q-mesh."""
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    unit = PhonopyAtoms(symbols=primitive.get_chemical_symbols(),
                        cell=np.array(primitive.cell),
                        scaled_positions=primitive.get_scaled_positions())
    ph = Phonopy(unit, supercell_matrix=np.diag(supercell_matrix),
                 primitive_matrix=np.eye(3))
    ph.force_constants = fc2
    ph.run_mesh(list(mesh), with_eigenvectors=False)
    freqs = ph.get_mesh_dict()["frequencies"]
    return {
        "fmin_THz": float(freqs.min()),
        "fmax_THz": float(freqs.max()),
        # phonopy reports an unstable mode as a negative frequency
        "n_imaginary": int((freqs < -1e-6).sum()),
        "frac_imaginary": float((freqs < -1e-6).mean()),
        "asr_resid": float(np.abs(fc2.sum(axis=1)).max()),
    }


def _fit(fit_data, method, kwargs):
    from trainstation import Optimizer
    with contextlib.redirect_stdout(io.StringIO()):
        opt = Optimizer(fit_data, fit_method=method, train_size=1.0, **kwargs)
        opt.train()
    return opt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                   help=f"comma list, or 'all' for {len(ALL_METHODS)} methods")
    p.add_argument("--train-sizes", default="",
                   help="comma list of structure counts; default = all data")
    p.add_argument("--cutoffs", default="",
                   help="override the fit cutoffs, e.g. '7.0,5.0'")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--mesh", default="",
                   help="phonopy q-mesh for the gates, e.g. '8,8,4'")
    p.add_argument("--layer-axis", action="store_true",
                   help="also report the cross-layer fc3 weight (layered only)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--write-fcp", action="store_true")
    a = p.parse_args()

    methods = ALL_METHODS if a.methods.strip() == "all" else \
        [m.strip() for m in a.methods.split(",") if m.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cutoffs = [float(c) for c in a.cutoffs.split(",")] if a.cutoffs else None

    from hiphive import ForceConstantPotential
    from trainstation import CrossValidationEstimator
    from phonon_inputs.hiphive_convergence import _apply_rotational_sum_rules

    meta, atoms_ideal, primitive, cs, sc = load_container(
        [Path(d) for d in a.data], cutoffs)
    n_struct = len(sc)
    sizes = [int(s) for s in a.train_sizes.split(",")] if a.train_sizes \
        else [n_struct]
    sizes = [s for s in sizes if s <= n_struct]

    smat = meta.get("supercell") or meta.get("supercell_matrix") or [1, 1, 1]
    if isinstance(smat[0], (list, tuple)):
        smat = [smat[i][i] for i in range(3)]
    mesh = [int(m) for m in a.mesh.split(",")] if a.mesh else [8, 8, 8]
    zfrac = atoms_ideal.get_scaled_positions()[:, 2]

    rng = np.random.default_rng(a.seed)
    rows = np.arange(len(sc))
    results = []
    for size in sizes:
        # StructureContainer.add_structure will not take a FitStructure back,
        # so subset by asking for the fit data of selected indices directly.
        if size >= n_struct:
            fit_data = sc.get_fit_data()
        else:
            pick = sorted(int(i) for i in rng.choice(rows, size, replace=False))
            fit_data = sc.get_fit_data(pick)

        for method in methods:
            t0 = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    cve = CrossValidationEstimator(
                        fit_data, fit_method=method,
                        validation_method="k-fold", n_splits=a.n_splits)
                    cve.validate()
                rmse_cv = float(cve.rmse_validation)

                opt = _fit(fit_data, method, {})
                raw = np.asarray(opt.parameters)
                nz = int((np.abs(raw) > NONZERO_REL_TOL *
                          max(np.abs(raw).max(), 1e-300)).sum())

                params = _apply_rotational_sum_rules(cs, raw)
                fcp = ForceConstantPotential(cs, params)
                fcs = fcp.get_force_constants(atoms_ideal)
                fc2 = fcs.get_fc_array(order=2)
                rec = {
                    "method": method, "train_size": int(size),
                    "n_dofs": int(cs.n_dofs), "n_nonzero": nz,
                    "sparsity": float(1.0 - nz / cs.n_dofs),
                    "rmse_train": float(opt.rmse_train), "rmse_cv": rmse_cv,
                    "seconds": round(time.perf_counter() - t0, 2),
                    **fc2_gates(fc2, primitive, smat, mesh),
                }
                if cs.cutoffs.max_order >= 3:
                    fc3 = fcs.get_fc_array(order=3)
                    rec["fc3_frob"] = float(np.sqrt((fc3 ** 2).sum()))
                    if a.layer_axis:
                        rec.update(layer_split_metrics(fc3, zfrac))
                if a.write_fcp:
                    fcp.write(str(out / f"fcp_{method}_n{size}.fcp"))
            except Exception as exc:                      # keep the grid going
                rec = {"method": method, "train_size": int(size),
                       "n_dofs": int(cs.n_dofs),
                       "seconds": round(time.perf_counter() - t0, 2),
                       "error": f"{type(exc).__name__}: {exc}"}
            results.append(rec)
            if "error" in rec:
                print(f"  n={size:<4d} {method:<28s} FAILED {rec['error'][:70]}",
                      flush=True)
            else:
                print(f"  n={size:<4d} {method:<28s} cv={rec['rmse_cv']:.4e} "
                      f"nz={rec['n_nonzero']:>6d}/{rec['n_dofs']:<6d} "
                      f"imag={rec['n_imaginary']:<5d} {rec['seconds']:>7.1f}s",
                      flush=True)

    payload = {"data": [str(d) for d in a.data], "n_structures": n_struct,
               "cutoffs": cutoffs or list(meta["cutoffs"]),
               "mesh": mesh, "n_splits": a.n_splits, "results": results}
    (out / "regressor_sweep.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}/regressor_sweep.json ({len(results)} cells)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
