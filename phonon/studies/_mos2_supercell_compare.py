"""Compare two hiphive force sets of the same material at different supercell
thickness -- built for the 2H-MoS2 4x4x1 vs 4x4x2 question, but nothing here is
MoS2-specific beyond the layered-crystal metrics.

The three modes answer the three questions the 4x4x2 campaign was run to settle
(see phonon/docs/mos2_kappa_z_not_converged.md section 6):

  harmonic   pair-cutoff ladder -> A-point frequency, Gamma-A velocity,
             minimum frequency on a mesh.  Is the imaginary mode a sampling
             artefact or a truncation artefact?
  kappa      cubic-cutoff ladder -> phono3py RTA kappa_xx / kappa_zz.  Does the
             transport number transfer between cells?
  interlayer fitted FC2/FC3 weight split intra-layer vs cross-gap, plus the
             rigid-layer Raman pair.  Which fitted quantity actually differs?

Refits only -- every mode consumes stored DFT forces and runs no DFT.

usage:
  python _mos2_supercell_compare.py harmonic   <work_dir> <c3> <c2> [<c2> ...]
  python _mos2_supercell_compare.py kappa      <work_dir> <mesh> <c2>:<c3> ...
  python _mos2_supercell_compare.py interlayer <work_dir> <c2> <c3>
"""
from __future__ import annotations

import contextlib
import gc
import io
import sys
import time
from pathlib import Path

import numpy as np

THZ_TO_CM = 33.35641
#: z gap that separates two sandwiches; the S-Mo-S internal spacing is 1.54 A
#: and the vdW S-S gap 3.02 A, so anything in between labels layers correctly.
SANDWICH_GAP_A = 2.0


def _load(work_dir: Path):
    """Meta plus the rattled structures, prepared for a StructureContainer."""
    import ase
    from hiphive.utilities import prepare_structures

    from phonon_inputs import thirdorder as _to
    from phonon_inputs.hiphive_fc3 import (
        _atoms_from_meta, _fold_positions_to_min_image, _load_meta,
        _read_positions_from_vasp_poscar,
    )

    meta = _load_meta(work_dir)
    atoms_ideal = _atoms_from_meta(meta["supercell_atoms"])
    primitive = _atoms_from_meta(meta["primitive"])
    n_super = len(atoms_ideal)

    if meta["calculator"] != "vasp":
        raise NotImplementedError(f"calculator {meta['calculator']!r}")

    rattled = []
    for i in range(1, meta["n_structures"] + 1):
        d = work_dir / f"disp-{i:05d}"
        pos = _fold_positions_to_min_image(
            _read_positions_from_vasp_poscar(d / "POSCAR", n_super),
            atoms_ideal.positions, atoms_ideal.cell,
        )
        r = ase.Atoms(symbols=list(atoms_ideal.get_chemical_symbols()),
                      cell=atoms_ideal.cell, positions=pos, pbc=True)
        r.arrays["forces"] = _to._parse_vasp_forces(d, n_super)
        rattled.append(r)

    return meta, atoms_ideal, primitive, prepare_structures(rattled, atoms_ideal)


def _fit(primitive, atoms_ideal, structures, cutoffs, orders=(2,)):
    """Least-squares cluster-expansion fit with rotational sum rules applied.

    Mirrors reap()'s policy (rotational_sum_rule='post_fit') so the numbers are
    comparable with the production force constants, but takes train_size=1.0 --
    these are ladders over cutoffs, not held-out generalisation tests.
    """
    from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer
    from trainstation import Optimizer

    from phonon_inputs.hiphive_convergence import _apply_rotational_sum_rules

    with contextlib.redirect_stdout(io.StringIO()):
        cs = ClusterSpace(primitive, list(cutoffs))
        sc = StructureContainer(cs)
        for s in structures:
            sc.add_structure(s)
        opt = Optimizer(sc.get_fit_data(), fit_method="least-squares",
                        train_size=1.0)
        opt.train()
        params = _apply_rotational_sum_rules(cs, opt.parameters)
        fcs = ForceConstantPotential(cs, params).get_force_constants(atoms_ideal)
        out = {o: fcs.get_fc_array(order=o) for o in orders}
        del fcs, sc, cs
    gc.collect()
    return out, float(opt.rmse_train), int(opt.n_parameters)


def _phonopy(primitive, meta, fc2):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    unit = PhonopyAtoms(symbols=primitive.get_chemical_symbols(),
                        cell=np.array(primitive.cell),
                        scaled_positions=primitive.get_scaled_positions())
    ph = Phonopy(unit, supercell_matrix=np.diag(meta["supercell"]),
                 primitive_matrix=np.eye(3), log_level=0)
    ph.force_constants = fc2
    return ph


def _sandwich_labels(atoms_ideal):
    """Layer index per atom, from gaps in the sorted z coordinate.

    A fractional `z >= 0.5` split is only correct for a cell one primitive cell
    thick; for a 4x4x2 cell it merges two sandwiches and calls the intra-cell
    van der Waals gap intra-layer.
    """
    z = atoms_ideal.positions[:, 2]
    order = np.argsort(z)
    lab = np.zeros(len(z), dtype=int)
    cur = 0
    for k in range(1, len(z)):
        if z[order[k]] - z[order[k - 1]] > SANDWICH_GAP_A:
            cur += 1
        lab[order[k]] = cur
    return lab


def _header(work_dir, meta, atoms_ideal):
    box = np.round(np.linalg.norm(np.array(atoms_ideal.cell), axis=1), 3)
    print(f"{work_dir.name}  supercell {meta['supercell']} "
          f"({len(atoms_ideal)} atoms)  box {box} A  "
          f"{meta['n_structures']} structures", flush=True)


def cmd_harmonic(work_dir, argv):
    c3 = float(argv[0])
    c2_list = [float(x) for x in argv[1:]]
    meta, ideal, prim, structures = _load(work_dir)
    _header(work_dir, meta, ideal)
    lim = 0.5 * min(np.linalg.norm(np.array(ideal.cell), axis=1))
    print(f"  isotropic cutoff limit (min L / 2): {lim:.3f} A\n")

    nq = 41
    path = np.zeros((nq, 3))
    path[:, 2] = np.linspace(0, 0.5, nq)
    mesh = np.array([[i / 8, j / 8, k / 8]
                     for i in range(8) for j in range(8) for k in range(8)])
    cz = np.array(prim.cell)[2, 2]

    hdr = (f"{'c2':>5} {'DOF':>5} {'rmse':>9} {'A-acou':>8} "
           f"{'v_LA m/s':>9} {'min(mesh)':>10}")
    print(hdr)
    print("-" * len(hdr))
    for c2 in c2_list:
        fcs, rmse, ndof = _fit(prim, ideal, structures, [c2, c3])
        ph = _phonopy(prim, meta, fcs[2])

        ph.run_qpoints(path)
        fr = np.sort(np.asarray(ph.get_qpoints_dict()["frequencies"]), axis=1)
        # Longitudinal acoustic along Gamma-A is the third branch: the two
        # below it are the interlayer-shear-like transverse pair.
        kz = path[:, 2] * 2 * np.pi / cz
        slope = np.polyfit(kz[:5], fr[:5, 2] * 2 * np.pi, 1)[0]

        ph.run_qpoints(mesh)
        fmin = float(np.asarray(ph.get_qpoints_dict()["frequencies"]).min())

        print(f"{c2:5.1f} {ndof:5d} {rmse:9.5f} {fr[-1, 0]:8.3f} "
              f"{slope * 100:9.0f} {fmin:10.4f}", flush=True)


def cmd_kappa(work_dir, argv):
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms

    mesh = int(argv[0])
    pairs = [tuple(float(x) for x in a.split(":")) for a in argv[1:]]
    meta, ideal, prim, structures = _load(work_dir)
    _header(work_dir, meta, ideal)
    print(f"  RTA, 300 K, {mesh}^3 mesh, isotopes off\n")

    hdr = (f"{'c2':>5} {'c3':>5} {'DOF':>6} {'rmse':>9} "
           f"{'k_xx':>9} {'k_zz':>9} {'s':>6}")
    print(hdr)
    print("-" * len(hdr))
    for c2, c3 in pairs:
        t0 = time.time()
        fcs, rmse, ndof = _fit(prim, ideal, structures, [c2, c3], orders=(2, 3))
        unit = PhonopyAtoms(symbols=prim.get_chemical_symbols(),
                            cell=np.array(prim.cell),
                            scaled_positions=prim.get_scaled_positions())
        ph3 = Phono3py(unit, supercell_matrix=np.diag(meta["supercell"]),
                       primitive_matrix=np.eye(3), log_level=0)
        ph3.fc2, ph3.fc3 = fcs[2], fcs[3]
        ph3.mesh_numbers = [mesh] * 3
        ph3.init_phph_interaction()
        ph3.run_thermal_conductivity(temperatures=[300.0], is_LBTE=False,
                                     write_kappa=False)
        k = np.array(ph3.thermal_conductivity.kappa)[0, 0]
        print(f"{c2:5.1f} {c3:5.1f} {ndof:6d} {rmse:9.5f} "
              f"{k[0]:9.3f} {k[2]:9.4f} {time.time() - t0:6.0f}", flush=True)
        del fcs, ph3
        gc.collect()


def cmd_interlayer(work_dir, argv):
    c2, c3 = float(argv[0]), float(argv[1])
    meta, ideal, prim, structures = _load(work_dir)
    _header(work_dir, meta, ideal)
    fcs, rmse, ndof = _fit(prim, ideal, structures, [c2, c3], orders=(2, 3))
    fc2, fc3 = fcs[2], fcs[3]

    lab = _sandwich_labels(ideal)
    same2 = lab[:, None] == lab[None, :]
    same3 = ((lab[:, None, None] == lab[None, :, None])
             & (lab[:, None, None] == lab[None, None, :]))
    f2 = (fc2 ** 2).sum(axis=(2, 3))
    f3 = (fc3 ** 2).sum(axis=(3, 4, 5))
    # Frobenius over the whole supercell grows as sqrt(N_atoms); dividing by
    # sqrt(n_cells) makes two cell sizes directly comparable.
    norm = np.sqrt(np.prod(meta["supercell"]))

    ph = _phonopy(prim, meta, fc2)
    ph.run_qpoints([[0.0, 0.0, 0.0]])
    fr = np.sort(np.asarray(ph.get_qpoints_dict()["frequencies"])[0]) * THZ_TO_CM

    print(f"  cutoffs [{c2}, {c3}]   rmse {rmse:.5f} eV/A   DOF {ndof}")
    print(f"  sandwiches            : {lab.max() + 1}  "
          f"(atoms each {np.bincount(lab)})")
    print(f"  FC2 intra / interlayer: {np.sqrt(f2[same2].sum()) / norm:9.4f} "
          f"{np.sqrt(f2[~same2].sum()) / norm:9.4f}")
    print(f"  FC3 intra / cross-gap : {np.sqrt(f3[same3].sum()) / norm:9.4f} "
          f"{np.sqrt(f3[~same3].sum()) / norm:9.4f}")
    print(f"  FC3 cross-gap max     : {np.abs(fc3)[~same3].max():9.4f} eV/A^3")
    print(f"  Gamma shear  (E2g^2)  : {fr[3]:8.2f} cm^-1   [measured  32.5]")
    print(f"  Gamma breath (B2g^2)  : {fr[5]:8.2f} cm^-1   [measured  55.7]")
    print(f"  Gamma E1g             : {fr[6]:8.2f} cm^-1   [measured 286.0]",
          flush=True)


MODES = {"harmonic": cmd_harmonic, "kappa": cmd_kappa,
         "interlayer": cmd_interlayer}


def main(argv):
    if len(argv) < 3 or argv[0] not in MODES:
        sys.exit(__doc__)
    MODES[argv[0]](Path(argv[1]), argv[2:])


if __name__ == "__main__":
    main(sys.argv[1:])
