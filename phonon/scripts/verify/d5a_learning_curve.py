#!/usr/bin/env python
"""A-PRIORI test of whether the d5a rattled structures are FIT-CONVERGED.
Refit FC2+FC3(+FC4) on increasing subsets of the EXISTING structures and watch
the held-out test RMSE (force prediction on structures the fit never saw). If
test RMSE has plateaued by N=44 -> more DFT will NOT improve the FCs; if it is
still falling -> more structures would help. This needs no transport SCBA and no
new DFT, and it is the principled way to answer "are the structures converged?"
before spending on the bubble.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _REPO / "phonon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import ase.io
from ase import Atoms
from ase.build import make_supercell
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.utilities import prepare_structures
from trainstation import Optimizer

WD = _REPO / "phonon/configs/sinw/fc3_hiphive_sinw100_d5a_sc4_vasp"
meta = json.loads((WD / "hiphive_meta.json").read_text())
pm = meta["primitive"]
prim = Atoms(symbols=pm["symbols"], cell=pm["cell"],
             scaled_positions=pm["scaled_positions"], pbc=True)
ideal = make_supercell(prim, np.diag(meta["supercell"]))
fc2c, fc3c = meta["cutoffs"][:2]
N = meta["n_structures"]
rattled = [ase.io.read(str(WD / f"disp-{i:05d}" / "vasprun.xml")) for i in range(1, N + 1)]
structures = prepare_structures(rattled, ideal)
print(f"d5a: {len(structures)} structures, FC2/FC3 cutoffs {fc2c}/{fc3c} A")

# fixed held-out test set (last 9 structures, ~20%); train on growing prefixes
n_test = max(8, N // 5)
test_idx = list(range(N - n_test, N))
train_pool = list(range(0, N - n_test))
print(f"held-out test set = {n_test} structures (fixed); train pool = {len(train_pool)}\n")


def build_sc(cs, idxs):
    sc = StructureContainer(cs)
    for i in idxs:
        sc.add_structure(structures[i])
    return sc


def fit_eval(cutoffs, method, n_train):
    cs = ClusterSpace(prim, cutoffs)
    tr = build_sc(cs, train_pool[:n_train])
    te = build_sc(cs, test_idx)
    A_tr, y_tr = tr.get_fit_data()
    A_te, y_te = te.get_fit_data()
    opt = Optimizer((A_tr, y_tr), fit_method=method, train_size=1.0, seed=0)
    opt.train()
    p = opt.parameters
    rmse_test = float(np.sqrt(np.mean((A_te @ p - y_te) ** 2)))
    return opt.rmse_train, rmse_test


grid = [8, 12, 18, 24, 28, len(train_pool)]
for cutoffs, methods in (([fc2c, fc3c], ("ardr",)),
                         ([fc2c, fc3c, 3.0], ("rfe",))):
    tag = "+".join(f"{c}" for c in cutoffs)
    print(f"== ClusterSpace cutoffs {tag} A ==")
    for m in methods:
        print(f"  method={m}: N_train -> (train RMSE | test RMSE) eV/A")
        prev = None
        for n in grid:
            rtr, rte = fit_eval(cutoffs, m, n)
            d = "" if prev is None else f"  (dtest {rte - prev:+.4f})"
            print(f"    {n:3d} -> {rtr:.4f} | {rte:.4f}{d}")
            prev = rte
    print()
