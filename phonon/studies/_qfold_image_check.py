"""Does the vertex fold and the device Hamiltonian pick the same periodic image?

H is built from phonopy's dynamical matrix, whose Fourier sum uses the shortest
vectors ``R + tau_kappa - tau_i`` averaged over ties
(``phonon_inputs/convention.py::get_btd_blocks_folded``, then gauge-transformed
A -> B). The three-phonon vertex is folded with one wrapped cell index per atom,
no basis offset and no tie average (``solver/se_q.py::_qfold_device_blocks``,
``separable.build_gathering_matrix``). Same convention, possibly different image.

This script asks the question at the level of the FC2, where it is
eigenvector-free and exact: rebuild ``D_B(q_perp)`` with the vertex fold's cell
sum and compare it against phonopy's, over the production transverse mesh.

The answer is FC-weighted, which is the point -- counting how many atom pairs
have a degenerate image is an upper bound, not an error, because the pairs that
are ambiguous are the far ones and the FC cutoff may put no weight there.

Usage::

    python -m phonon.studies._qfold_image_check --reap DIR --nk 9 [--tdir x]

``DIR`` needs ``fc2.hdf5`` and ``phono3py.yaml``. For the production Si film
that is ``reaps/si_big_hiphive`` on the cluster (5x5x5, nk=9); the checked-in
``reaps/si_primitive_work`` (2x2x2) is the contrasting case where the box is
small enough that every neighbour lands on a tie shell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "phonon"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def load_reap(reap: Path):
    import h5py
    from phono3py import load as phono3py_load
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    with h5py.File(str(reap / "fc2.hdf5"), "r") as f:
        fc2 = f["force_constants"][:]
    ph3 = phono3py_load(phono3py_yaml=str(reap / "phono3py.yaml"),
                        produce_fc=False, log_level=0)
    cell = PhonopyAtoms(symbols=ph3.unitcell.symbols, cell=ph3.unitcell.cell,
                        scaled_positions=ph3.unitcell.scaled_positions)
    phonon = Phonopy(cell, supercell_matrix=ph3.supercell_matrix,
                     primitive_matrix=np.eye(3))
    phonon.force_constants = fc2
    return phonon, fc2


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reap", required=True, type=Path)
    p.add_argument("--nk", type=int, default=9)
    p.add_argument("--tdir", default="x")
    args = p.parse_args(argv)

    from phonon.phonon_inputs.constants import CONVERSION_THZ2
    from phonon.phonon_inputs.convention import gauge_transform_A_to_B
    from phonon.phonon_inputs.separable import build_supercell_mapping

    phonon, fc2 = load_reap(args.reap)
    prim_indices, cell_frac, _, ref_sc_atoms = build_supercell_mapping(
        phonon, args.tdir
    )
    tau = np.asarray(phonon.primitive.scaled_positions)
    masses = np.asarray(phonon.primitive.masses, dtype=float)
    nat = len(masses)
    tidx = "xyz".index(args.tdir)
    perp = [i for i in range(3) if i != tidx]
    width = int(round(np.abs(cell_frac).max())) + 1
    print(f"{args.reap}: {nat} primitive atoms, {len(prim_indices)} supercell "
          f"atoms, transport {args.tdir}, nk = {args.nk}")

    def cell_fold(q, sign):
        """D_B(q) the way the vertex fold sums: one wrapped cell index, no
        basis offset, no tie average."""
        ph = np.exp(sign * 2j * np.pi * (cell_frac @ np.asarray(q, float)))
        d = np.zeros((3 * nat, 3 * nat), dtype=complex)
        for i in range(nat):
            for j in range(nat):
                sel = prim_indices == j
                d[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = np.einsum(
                    "sab,s->ab", fc2[ref_sc_atoms[i]][sel], ph[sel]
                ) / np.sqrt(masses[i] * masses[j])
        return 0.5 * (d + d.conj().T) * CONVERSION_THZ2

    worst = {1.0: 0.0, -1.0: 0.0}
    worst_dw = 0.0
    for ka in range(args.nk):
        for kb in range(args.nk):
            q = np.zeros(3)
            q[perp[0]], q[perp[1]] = ka / args.nk, kb / args.nk
            d_ref = gauge_transform_A_to_B(
                phonon.get_dynamical_matrix_at_q(q), q, tau
            ) * CONVERSION_THZ2
            scale = np.abs(d_ref).max() + 1e-30
            for sign in worst:
                worst[sign] = max(
                    worst[sign], float(np.abs(cell_fold(q, sign) - d_ref).max() / scale)
                )
            w_c = np.sqrt(np.abs(np.linalg.eigvalsh(cell_fold(q, 1.0))))
            w_r = np.sqrt(np.abs(np.linalg.eigvalsh(d_ref)))
            worst_dw = max(worst_dw, float(np.abs(np.sort(w_c) - np.sort(w_r)).max()))

    print(f"  transverse supercell width ~ {width} cells")
    print(f"  worst rel |dD_B| over {args.nk**2} q_perp, exp(+2pi i q.R) "
          f"[convention B]: {worst[1.0]:.3e}")
    print(f"  worst rel |dD_B|,                exp(-2pi i q.R) "
          f"[the vertex legs' sign]: {worst[-1.0]:.3e}")
    print(f"  worst |dw|: {worst_dw:.3e} THz")
    if worst[1.0] < 1e-12:
        print("  -> the fold IS phonopy's shortest-vector sum on this bed: no "
              "atom pair with an ambiguous image carries FC weight.")
    else:
        print("  -> the two differ: ambiguous-image pairs carry weight here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
