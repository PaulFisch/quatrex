"""Toy model test for the SCBA self-energy implementation.

Creates a synthetic simple-cubic 1-atom crystal with:
  - Nearest-neighbor harmonic springs (FC2)
  - Nearest-neighbor cubic anharmonicity (FC3)

Feeds it through the EXISTING separable_anharmonic_transmission code
and validates against:
  1. Analytically known ballistic transmission
  2. SCBA convergence vs n_slabs, q-mesh, mixing

This tests the actual code path, not a reimplementation.
"""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent
sys.path.insert(0, str(work_dir))

from phonon_inputs.separable import (
    build_supercell_mapping,
    decompose_fc3_supercell,
    separable_anharmonic_transmission,
)
from phonon_inputs.convention import get_btd_blocks
from phonon_inputs.constants import CONVERSION_THZ2


# -----------------------------------------------------------------------
# 1. Build synthetic Phonopy + FC3
# -----------------------------------------------------------------------

def make_toy_phonon(a=3.0, mass=28.0, k_spring=5.0, supercell_n=2):
    """Create a simple-cubic 1-atom Phonopy with nearest-neighbor FC2.

    Parameters
    ----------
    a : float
        Lattice constant (Angstrom).
    mass : float
        Atomic mass (amu).
    k_spring : float
        Spring constant in eV/A^2.
    supercell_n : int
        Supercell size (n x n x n).

    Returns
    -------
    phonon : Phonopy
    """
    cell = PhonopyAtoms(
        symbols=["Si"],
        cell=np.diag([a, a, a]),
        scaled_positions=[[0.0, 0.0, 0.0]],
    )
    sc_mat = np.diag([supercell_n, supercell_n, supercell_n])
    phonon = Phonopy(cell, supercell_matrix=sc_mat, primitive_matrix=np.eye(3))

    # Build FC2 for nearest-neighbor springs
    n_sc = len(phonon.supercell.positions)
    fc2 = np.zeros((n_sc, n_sc, 3, 3))
    sc_pos = phonon.supercell.positions
    sc_cell = phonon.supercell.cell

    for i in range(n_sc):
        for j in range(n_sc):
            if i == j:
                continue
            diff = sc_pos[j] - sc_pos[i]
            # Minimum image convention
            frac = np.linalg.solve(sc_cell.T, diff)
            frac -= np.round(frac)
            diff_min = sc_cell.T @ frac
            dist = np.linalg.norm(diff_min)
            if abs(dist - a) < 0.1:  # nearest neighbor
                # Spring along bond direction
                n_hat = diff_min / dist
                fc2[i, j] = -k_spring * np.outer(n_hat, n_hat)
                fc2[i, i] -= fc2[i, j]  # ASR: diagonal = -sum of off-diag

    phonon.force_constants = fc2
    return phonon


def make_toy_fc3(phonon, phi3=0.5, a=3.0):
    """Create a nearest-neighbor FC3 for the toy model.

    Uses a simple isotropic cubic coupling: FC3[i,j,k,a,a,a] = phi3
    for nearest-neighbor triplets, with full permutation symmetry.

    Parameters
    ----------
    phonon : Phonopy
    phi3 : float
        Anharmonic coupling strength (eV/A^3).
    a : float
        Lattice constant.

    Returns
    -------
    fc3 : ndarray, shape (n_sc, n_sc, n_sc, 3, 3, 3)
    """
    n_sc = len(phonon.supercell.positions)
    sc_pos = phonon.supercell.positions
    sc_cell = phonon.supercell.cell

    def min_image_dist(i, j):
        diff = sc_pos[j] - sc_pos[i]
        frac = np.linalg.solve(sc_cell.T, diff)
        frac -= np.round(frac)
        return np.linalg.norm(sc_cell.T @ frac)

    # Find nearest neighbors
    nn = {}
    for i in range(n_sc):
        nn[i] = []
        for j in range(n_sc):
            if i == j:
                continue
            if abs(min_image_dist(i, j) - a) < 0.1:
                nn[i].append(j)

    # Build unsymmetrized FC3
    fc3_raw = np.zeros((n_sc, n_sc, n_sc, 3, 3, 3))
    for i in range(n_sc):
        for j in nn[i]:
            for k in nn[i]:
                for alpha in range(3):
                    fc3_raw[i, j, k, alpha, alpha, alpha] = phi3

    # Enforce permutation symmetry: average over S_3
    fc3 = np.zeros_like(fc3_raw)
    for i in range(n_sc):
        for j in range(n_sc):
            for k in range(n_sc):
                fc3[i, j, k] += fc3_raw[i, j, k]
                fc3[i, j, k] += fc3_raw[j, i, k].transpose(1, 0, 2)
                fc3[i, j, k] += fc3_raw[k, j, i].transpose(2, 1, 0)
                fc3[i, j, k] += fc3_raw[i, k, j].transpose(0, 2, 1)
                fc3[i, j, k] += fc3_raw[j, k, i].transpose(1, 2, 0)
                fc3[i, j, k] += fc3_raw[k, i, j].transpose(2, 0, 1)
    fc3 /= 6.0

    return fc3


def save_toy_hdf5(fc3, fc2, path):
    """Save FC3 + FC2 to HDF5."""
    with h5py.File(path, "w") as f:
        f.create_dataset("fc3", data=fc3)
        f.create_dataset("fc2", data=fc2)


# -----------------------------------------------------------------------
# 2. Tests
# -----------------------------------------------------------------------

def test_ballistic(phonon):
    """Test that ballistic transmission is physical."""
    print("=" * 60)
    print("TEST 1: Ballistic transmission")
    print("=" * 60)

    # Check dispersion at zone boundary
    q_test = np.array([0.5, 0.0, 0.0])
    D = phonon.get_dynamical_matrix_at_q(q_test) * CONVERSION_THZ2
    evals = np.linalg.eigvalsh(D)
    freqs = np.sqrt(np.maximum(evals, 0))
    print(f"  Frequencies at q=(0.5,0,0): {freqs} THz")
    print(f"  Max phonon freq ~ {freqs.max():.1f} THz")

    # Ballistic transmission via existing BTD code
    from phonon_inputs.validation import _ballistic_transmission
    from phonon_inputs.separable import _build_device_hamiltonian

    nfreq = 101
    fmin, fmax = 0.5, freqs.max() * 1.2
    freq_arr = np.linspace(fmin, fmax, nfreq)
    dw = freq_arr[1] - freq_arr[0]
    eta = dw**2 * 0.5
    n_dof = 3

    trans = np.zeros(nfreq)
    nk = 4
    for ix in range(nk):
        for iy in range(nk):
            qx, qy = ix / nk, iy / nk
            H_00, H_01 = get_btd_blocks(
                phonon, (qx, qy), transport_direction="x",
                conversion_factor=CONVERSION_THZ2,
            )
            H_D = _build_device_hamiltonian(H_00, H_01, 1)
            H_LD = np.zeros((n_dof, n_dof), dtype=complex)
            H_LD[:, :n_dof] = H_01
            H_DR = np.zeros((n_dof, n_dof), dtype=complex)
            H_DR[:, :] = H_01
            for iw in range(nfreq):
                trans[iw] += _ballistic_transmission(
                    freq_arr[iw]**2, H_D, H_00, H_01, H_00, H_01,
                    H_LD, H_DR, eta=eta,
                )
    trans /= nk * nk

    print(f"  Ballistic T: max = {trans.max():.4f} (should be <= 3)")
    print(f"  Sum(T*dw) = {np.sum(trans)*dw:.2f} (integrated transmission)")
    assert trans.max() <= 3.1, "Transmission exceeds 3 modes!"
    print("  PASS")


def test_svd_decomposition(phonon, fc3):
    """Test that SVD decomposition works."""
    print("\n" + "=" * 60)
    print("TEST 2: FC3 SVD decomposition")
    print("=" * 60)

    print(f"  FC3 shape: {fc3.shape}")
    print(f"  FC3 max: {np.max(np.abs(fc3)):.4e}")
    print(f"  FC3 nonzero: {np.count_nonzero(fc3)}/{fc3.size} "
          f"({np.count_nonzero(fc3)/fc3.size:.4f})")

    prim_indices, cell_frac, slab_indices, ref_sc_atoms = \
        build_supercell_mapping(phonon, "x")
    n_atoms = len(phonon.primitive.masses)
    masses_super = phonon.supercell.masses

    F_list, H, svals, trans_atoms = decompose_fc3_supercell(
        fc3, n_atoms, masses_super, prim_indices, slab_indices,
        ref_sc_atoms, rank=None, tol=1e-15,
    )
    R = len(F_list)
    print(f"  Full SVD rank: {R}")
    print(f"  Singular values: {svals}")
    print(f"  Same-slab atoms: {len(trans_atoms)}, dim_t = {3*len(trans_atoms)}")
    print(f"  n_dof = {3*n_atoms}")
    print("  PASS")


def test_scba_convergence(phonon, fc3):
    """Test SCBA convergence for various n_slabs and parameters."""
    print("\n" + "=" * 60)
    print("TEST 3: SCBA convergence vs n_slabs")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
        fc3_path = f.name
    save_toy_hdf5(fc3, phonon.force_constants, fc3_path)

    freq_max = 18.0  # from test_ballistic
    common = dict(
        freq_range_thz=(0.5, freq_max, 51),
        transport_direction="x",
        eta_factor=0.5,
        temperature=300.0,
        delta_T=10.0,
        scba_tol=0.005,
        rank=None,
        verbose=False,
    )

    print(f"\n  {'n_slabs':>8} {'q_mesh':>7} {'mixing':>7} {'G_ball':>10} "
          f"{'G_anh':>10} {'conserv':>9} {'iters':>6} {'OK':>4}")
    print("  " + "-" * 65)

    for ns in [1, 2, 4, 6, 8, 10]:
        converged = False
        for nk in [2, 4]:
            for mix in [0.3, 0.1, 0.05]:
                try:
                    r = separable_anharmonic_transmission(
                        phonon, fc3_path,
                        q_mesh_transverse=(nk, nk),
                        n_slabs=ns,
                        max_scba_iter=40,
                        mixing=mix,
                        **common,
                    )
                    G_ball = r["thermal_conductance_ballistic"] / 1e6
                    G_anh = r["thermal_conductance_anharmonic"] / 1e6
                    conserv = r["heat_flow_conservation"]
                    n_iter = r["n_scba_iterations"]
                    ok = conserv < 0.05

                    print(f"  {ns:>8} {nk}x{nk:>4} {mix:>7.2f} {G_ball:>10.1f} "
                          f"{G_anh:>10.1f} {conserv:>9.4f} {n_iter:>6} "
                          f"{'YES' if ok else 'NO':>4}")

                    if ok:
                        converged = True
                        break
                except Exception as e:
                    print(f"  {ns:>8} {nk}x{nk:>4} {mix:>7.2f} "
                          f"{'ERR':>10} {str(e)[:30]}")
            if converged:
                break

        if not converged:
            print(f"  ** n_slabs={ns}: FAILED to converge with any settings **")

    Path(fc3_path).unlink(missing_ok=True)


def main():
    print("Building toy model: simple cubic, 1 atom, NN springs + FC3\n")
    phonon = make_toy_phonon(a=3.0, mass=28.0, k_spring=5.0, supercell_n=2)
    print(f"Primitive: {len(phonon.primitive.masses)} atom(s), "
          f"a = {phonon.primitive.cell[0,0]:.1f} A")
    print(f"Supercell: {len(phonon.supercell.masses)} atoms "
          f"({phonon.supercell_matrix.diagonal()})")

    fc3 = make_toy_fc3(phonon, phi3=0.5, a=3.0)

    test_ballistic(phonon)
    test_svd_decomposition(phonon, fc3)
    test_scba_convergence(phonon, fc3)

    print("\n" + "=" * 60)
    print("TEST 4: Strong anharmonicity (phi3=5.0) + 4x4 q-mesh")
    print("=" * 60)

    fc3_strong = make_toy_fc3(phonon, phi3=5.0, a=3.0)

    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
        fc3_path = f.name
    save_toy_hdf5(fc3_strong, phonon.force_constants, fc3_path)

    common = dict(
        freq_range_thz=(0.5, 18.0, 51),
        transport_direction="x",
        eta_factor=0.5,
        temperature=300.0,
        delta_T=10.0,
        scba_tol=0.005,
        rank=None,
        verbose=False,
    )

    print(f"\n  {'n_slabs':>8} {'q_mesh':>7} {'mixing':>7} {'G_ball':>10} "
          f"{'G_anh':>10} {'conserv':>9} {'iters':>6} {'OK':>4}")
    print("  " + "-" * 65)

    for ns in [1, 2, 4, 6, 10, 20]:
        converged = False
        for nk in [4, 2]:
            for mix in [0.3, 0.1, 0.05]:
                try:
                    r = separable_anharmonic_transmission(
                        phonon, fc3_path,
                        q_mesh_transverse=(nk, nk),
                        n_slabs=ns,
                        max_scba_iter=40,
                        mixing=mix,
                        **common,
                    )
                    G_ball = r["thermal_conductance_ballistic"] / 1e6
                    G_anh = r["thermal_conductance_anharmonic"] / 1e6
                    conserv = r["heat_flow_conservation"]
                    n_iter = r["n_scba_iterations"]
                    ok = conserv < 0.05

                    print(f"  {ns:>8} {nk}x{nk:>4} {mix:>7.2f} {G_ball:>10.1f} "
                          f"{G_anh:>10.1f} {conserv:>9.4f} {n_iter:>6} "
                          f"{'YES' if ok else 'NO':>4}")

                    if ok:
                        converged = True
                        break
                except Exception as e:
                    print(f"  {ns:>8} {nk}x{nk:>4} {mix:>7.2f} "
                          f"{'ERR':>10} {str(e)[:30]}")
            if converged:
                break

        if not converged:
            print(f"  ** n_slabs={ns}: FAILED **")

    Path(fc3_path).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("Summary: compare weak vs strong anharmonicity to see")
    print("whether divergence depends on FC3 strength, q-mesh, or n_slabs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
