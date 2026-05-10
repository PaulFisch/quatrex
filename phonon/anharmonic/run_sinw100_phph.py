"""Drive ``SigmaPhononPhonon`` on the real reaps/hiphive_sinw100_vasp_larger
dataset (53-atom Si:H slab, 159 DOFs, transport along z).

This is a single-block (Γ-only) acceptance run: it wires the production
``SigmaPhononPhonon.compute()`` against the actual hiphive FC3 file and
checks parity with a direct call to ``_bubble_block``. It also exercises
the §8.2 round-trip writer/loader on a real, non-toy Φ.

Run::

    /home/paul/miniconda3/envs/quatrex-dev/bin/python \
        phonon/anharmonic/run_sinw100_phph.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "phonon"))

DATASET = ROOT / "phonon" / "input_calc" / "reaps/hiphive_sinw100_vasp_larger"


def _build_phonopy(meta: dict):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    prim = PhonopyAtoms(
        symbols=meta["primitive"]["symbols"],
        cell=np.asarray(meta["primitive"]["cell"]),
        scaled_positions=np.asarray(meta["primitive"]["scaled_positions"]),
    )
    sc_matrix = np.diag(meta["supercell"])
    return Phonopy(prim, supercell_matrix=sc_matrix, primitive_matrix=np.eye(3))


def _make_minimal_config():
    from quatrex.core.config import QuatrexConfig

    return QuatrexConfig(
        config_dir=".", input_dir=".", output_dir=".",
        formalism="negf", simulation_type="phonon",
        scba={"phonon": True, "max_iterations": 1},
        device={
            "transport_direction": "z",
            "num_transport_cells": 1,
            "neighbor_cell_cutoff": [1, 1, 1],
            "kpoint_grid": [1, 1, 1],
            "construct_from_unit_cell": False,
            "block_size": 159,
            "num_orbitals_per_atom": {"X": 4},
        },
        electron={
            "energy_window_min": -1.0, "energy_window_max": 1.0,
            "energy_window_num": 4,
        },
        phonon={
            "model": "negf",
            "fc3_path": "fc3_blocks_sinw100.hdf5",
            "left_temperature": 300.0, "right_temperature": 310.0,
        },
        compute={"comm": {"block_comm_size": 1}},
    )


def _make_phonon_only_cfg():
    class _Phonon:
        retarded_method = "fft"
        fc3_path = None
    class _Cfg:
        phonon = _Phonon()
    return _Cfg()


def main() -> int:
    print("=" * 70)
    print("sinw100 (53-atom Si:H slab) phonon-phonon SCBA acceptance run")
    print("=" * 70)

    # --- 1. Load metadata + phonopy -----------------------------------
    meta = json.loads((DATASET / "hiphive_meta.json").read_text())
    phonon = _build_phonopy(meta)
    n_prim = len(phonon.primitive.masses)
    n_sc = len(phonon.supercell.masses)
    n_dof = 3 * n_prim
    print(f"  Primitive: {n_prim} atoms ({n_dof} DOFs)")
    print(f"  Supercell: {n_sc} atoms (cell {meta['supercell']})")

    # --- 2. Build Phi at Gamma via standalone unrolling ---------------
    from phonon_inputs.separable import (  # type: ignore[import-not-found]
        build_supercell_mapping,
        build_realspace_fc3_matrices,
        build_gathering_matrix,
    )

    print("  Loading FC3 ...")
    t0 = time.perf_counter()
    with h5py.File(DATASET / "fc3.hdf5", "r") as f:
        fc3_raw = f["fc3"][...]
    print(f"    fc3.hdf5: shape={fc3_raw.shape}, dtype={fc3_raw.dtype}, "
          f"loaded in {time.perf_counter() - t0:.2f}s")

    prim_indices, cell_frac, _, ref_sc_atoms = build_supercell_mapping(
        phonon, transport_direction="z"
    )
    masses_super = phonon.supercell.masses

    print("  Building real-space FC3 matrices ...")
    t0 = time.perf_counter()
    M_stacked = build_realspace_fc3_matrices(
        fc3_raw, n_prim, masses_super, ref_sc_atoms
    )
    dim_sc = n_sc * 3
    M_blocks = M_stacked.reshape(n_dof, dim_sc, dim_sc)
    print(f"    M_stacked: {M_stacked.shape}, "
          f"||M||={np.linalg.norm(M_stacked):.3e} (THz²·...)"
          f", {time.perf_counter() - t0:.2f}s")

    print("  Gathering Φ at Γ ...")
    t0 = time.perf_counter()
    T0 = build_gathering_matrix(
        prim_indices, cell_frac, (0.0, 0.0), n_prim, "z"
    )
    Phi = np.einsum("ci,aij,dj->acd", T0, M_blocks, T0.conj())
    Phi = Phi.astype(np.complex128)
    print(f"    Phi: {Phi.shape}, ||Phi||={np.linalg.norm(Phi):.3e} THz², "
          f"{time.perf_counter() - t0:.2f}s")

    # --- 3. Round-trip Φ via the new HDF5 writer ----------------------
    from phonon_inputs.quatrex_writer import (  # type: ignore[import-not-found]
        write_fc3_blocks,
    )
    from quatrex.phonon.fc3_loader import (
        fc3_to_phi_blocks,
        load_device_fc3,
    )

    block_sizes = np.array([n_dof], dtype=np.int64)
    phi_blocks_in = fc3_to_phi_blocks(Phi, block_sizes, nn_only=True)
    out_h5 = DATASET / "fc3_blocks_sinw100.hdf5"
    write_fc3_blocks(phi_blocks_in, block_sizes, out_h5, units="THz^2")
    phi_blocks_loaded = load_device_fc3(out_h5, block_sizes=block_sizes)
    for key, block in phi_blocks_in.items():
        np.testing.assert_array_equal(phi_blocks_loaded[key], block)
    print(f"  FC3 round-trip OK ({out_h5.name}, "
          f"{len(phi_blocks_in)} block(s))")

    # --- 4. Build a real DSDBSparse G and run SigmaPhononPhonon -------
    from quatrex.core.config import setup_context
    setup_context(_make_minimal_config())

    from qttools import sparse, xp
    from qttools.datastructures import DSDBCSR
    from quatrex.phonon.sse_phonon_phonon import SigmaPhononPhonon
    from quatrex.phonon.units import bubble_prefactor_thz

    # Small ω-grid: keep n_fft = 2*ne-1 and the bubble's
    # intermediate (n_fft, n_dof, n_dof, n_dof) tractable in RAM.
    # For n_dof=159, ne=11 → n_fft=21 → ~1.3 GB intermediate.
    ne = 11
    freqs_thz = np.linspace(-16.0, 16.0, ne)
    dw_thz = float(freqs_thz[1] - freqs_thz[0])

    rng = np.random.default_rng(0)
    G_l = (
        rng.standard_normal((ne, n_dof, n_dof))
        + 1j * rng.standard_normal((ne, n_dof, n_dof))
    )
    G_g = (
        rng.standard_normal((ne, n_dof, n_dof))
        + 1j * rng.standard_normal((ne, n_dof, n_dof))
    )

    pattern = sparse.random(
        n_dof, n_dof, density=1.0, format="coo",
        dtype=np.complex128, random_state=rng,
    )
    pattern.data[:] = 1.0

    g_lesser = DSDBCSR.from_sparray(
        pattern, block_sizes=block_sizes,
        global_stack_shape=(ne,), symmetry=False,
    )
    g_greater = DSDBCSR.zeros_like(g_lesser)
    sigma_l = DSDBCSR.zeros_like(g_lesser)
    sigma_g = DSDBCSR.zeros_like(g_lesser)
    sigma_r = DSDBCSR.zeros_like(g_lesser)

    g_lesser.stack[...].blocks[0, 0] = xp.asarray(G_l)
    g_greater.stack[...].blocks[0, 0] = xp.asarray(G_g)

    ssp = SigmaPhononPhonon(
        _make_phonon_only_cfg(),
        phonon_frequencies=freqs_thz,
        block_sizes=block_sizes,
        phi_blocks=phi_blocks_loaded,
    )

    print(f"  Running SigmaPhononPhonon.compute()  "
          f"(ne={ne}, n_dof={n_dof}, n_fft={2*ne-1}) ...")
    t0 = time.perf_counter()
    ssp.compute(g_lesser, g_greater, out=(sigma_l, sigma_g, sigma_r))
    t_compute = time.perf_counter() - t0
    print(f"    compute(): {t_compute:.2f}s")

    # --- 5. Reference: direct bubble call -----------------------------
    print("  Reference _bubble_block ...")
    t0 = time.perf_counter()
    sl_ref = ssp._bubble_block(
        phi_left=Phi, phi_right=Phi,
        G_inner_a=G_l, G_inner_b=G_l,
        n_fft=2 * ne - 1, prefactor=bubble_prefactor_thz(dw_thz),
    )
    sg_ref = ssp._bubble_block(
        phi_left=Phi, phi_right=Phi,
        G_inner_a=G_g, G_inner_b=G_g,
        n_fft=2 * ne - 1, prefactor=bubble_prefactor_thz(dw_thz),
    )
    print(f"    reference: {time.perf_counter() - t0:.2f}s")

    sl_new = np.asarray(sigma_l.stack[...].blocks[0, 0])
    sg_new = np.asarray(sigma_g.stack[...].blocks[0, 0])
    sr_new = np.asarray(sigma_r.stack[...].blocks[0, 0])

    abs_l = np.max(np.abs(sl_new - sl_ref))
    abs_g = np.max(np.abs(sg_new - sg_ref))
    rel_l = abs_l / max(np.max(np.abs(sl_ref)), 1e-300)
    rel_g = abs_g / max(np.max(np.abs(sg_ref)), 1e-300)

    print()
    print("  Σ^< : max |Σ_ref|        = {:.4e}".format(np.max(np.abs(sl_ref))))
    print("  Σ^< : max |Σ_new - Σ_ref|= {:.4e}  (rel {:.4e})"
          .format(abs_l, rel_l))
    print("  Σ^> : max |Σ_new - Σ_ref|= {:.4e}  (rel {:.4e})"
          .format(abs_g, rel_g))
    print("  Σ^R : max |Σ^R|          = {:.4e}".format(np.max(np.abs(sr_new))))

    tol = 1e-10
    if max(rel_l, rel_g) > tol:
        print(f"\nFAIL: rel error {max(rel_l, rel_g):.4e} > {tol:.0e}")
        return 1
    print(f"\nPASS: SigmaPhononPhonon ↔ _bubble_block agree to {tol:.0e}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
