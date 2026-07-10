"""Numerical check of the first-Born tadpole statement (theory
20_self_energy.tex TODO at line ~311): for a T=0-relaxed structure the
FIRST-BORN cubic tadpole -- built from the harmonic equilibrium <uu> --
is small (it measures thermal expansion, growing with T), while the
SELF-CONSISTENT (SCP) tadpole need not vanish.

For each structure and temperature this computes, on the harmonic device:
  uu0     = equilibrium <w w> (mode sum),
  <w>     = -Phi^+ s with s the tadpole source from uu0,
  Sigma_T = first-Born tadpole Phi3 : <w>   [THz^2],
and reports ||Sigma_T|| / ||D|| plus the mean-force residual. The stored
SCP snapshots (cluster/snapshots/study_d5a_T*_tadpole.npz) provide the
self-consistent counterpoint where available.

Run: python phonon/scripts/verify/first_born_tadpole.py
Output: phonon/scripts/verify/first_born_tadpole.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for p in (str(_REPO / "phonon"), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from phonon.finite_analysis.loader import load_system  # noqa: E402
from phonon_inputs.constants import CONVERSION_THZ2  # noqa: E402
from phonon_inputs.convention import get_btd_blocks  # noqa: E402
from solver.fc3_device import build_device_fc3_blocks  # noqa: E402
from solver.static_se import (  # noqa: E402
    device_fc3_mass_weighted,
    equilibrium_uu_modesum,
    mean_displacement,
    mean_force,
    sigma_tadpole,
    tadpole_source,
)
from solver.zero_modes import build_dynamical_zero_mode_projector  # noqa: E402

CFGS = {
    "d5a": ("phonon/configs/sinw/sinw100_d5a_vasp_sc4_fc4.yaml", None),
    "cnt33": ("phonon/configs/cnt/cnt33_vasp.yaml", None),
}
TEMPS = (30.0, 100.0, 300.0, 600.0)
N_SLABS = 1

out: dict = {}
for name, (cfg, _f) in CFGS.items():
    b = load_system(str(_REPO / cfg), validate=False, transport_axis=2)
    ph = b.phonon
    n_atoms = len(ph.primitive.masses)
    n_dof = 3 * n_atoms
    masses_dof = np.repeat(np.asarray(ph.primitive.masses, float), 3)

    H_00, H_01 = get_btd_blocks(
        ph, (0.0, 0.0), transport_direction="z",
        conversion_factor=CONVERSION_THZ2)
    D = np.asarray(H_00, dtype=complex).real
    D = 0.5 * (D + D.T)

    from solver.dense import _load_mass_weighted_fc3

    M_stacked, mapping, _raw = _load_mass_weighted_fc3(
        ph, b.meta["fc3_path"], None, "z", enforce_asr=True,
        vertex_scale=1.0, verbose=False)
    prim_indices, _cell_frac, slab_indices, _ref = mapping
    blocks = build_device_fc3_blocks(
        M_stacked, prim_indices, slab_indices, n_atoms, N_SLABS)
    fc3_mw = device_fc3_mass_weighted(blocks, N_SLABS, n_dof)

    projector = build_dynamical_zero_mode_projector(H_00, H_01, N_SLABS)
    res_T = {}
    for T in TEMPS:
        uu0 = equilibrium_uu_modesum(D, T)
        s = tadpole_source(fc3_mw, uu0)
        w_mean = mean_displacement(fc3_mw, uu0, D,
                                   optical_projector=projector)
        sig_T = sigma_tadpole(fc3_mw, w_mean)
        f_mean = mean_force(fc3_mw, uu0, masses_dof)
        res_T[f"T{int(T)}"] = {
            "norm_sigma_T_thz2": float(np.linalg.norm(sig_T)),
            "norm_D_thz2": float(np.linalg.norm(D)),
            "rel_sigma_T": float(np.linalg.norm(sig_T) / np.linalg.norm(D)),
            "max_mean_disp_A": float(np.max(np.abs(w_mean))),
            "max_mean_force_eV_A": float(np.max(np.abs(f_mean))),
            "norm_source": float(np.linalg.norm(s)),
        }
        print(f"{name} T={T:g}: |Sigma_T|/|D| = "
              f"{res_T[f'T{int(T)}']['rel_sigma_T']:.3e}, "
              f"max<u> = {res_T[f'T{int(T)}']['max_mean_disp_A']:.3e} A")
    out[name] = res_T

# Self-consistent counterpoint from the stored SCP study snapshots.
snaps = sorted((_REPO / "cluster/snapshots").glob("study_d5a_T*_tadpole.npz"))
scp = {}
for sp in snaps:
    d = np.load(sp, allow_pickle=True)
    if "sigma_static" in d and "device_D" in d:
        ss = np.asarray(d["sigma_static"])
        DD = np.asarray(d["device_D"])
        scp[sp.name] = float(np.linalg.norm(ss) / np.linalg.norm(DD))
out["scp_snapshots_rel_sigma_static"] = scp
print("SCP counterpoint:", scp)

dst = Path(__file__).with_suffix(".json")
dst.write_text(json.dumps(out, indent=2))
print(f"saved {dst}")
