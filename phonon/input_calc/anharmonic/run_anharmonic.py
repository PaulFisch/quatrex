"""Anharmonic phonon transport via SCBA.

Computes ballistic and anharmonic thermal transport for bulk Si.
Supports two workflows:

1. Primitive cell (2-atom FCC): FC2 + FC3 from phono3py, no remapping.
   Requires fc3_prim/fc3.hdf5 and fc3_prim/phono3py_disp.yaml.

2. Conventional cell (8-atom cubic): FC2 from phonopy, FC3 from
   thirdorder.py with remapping. Legacy mode for backward compatibility.

Usage:
    python run_anharmonic.py [--nk 4] [--nfreq 101] [--max-iter 20]
                             [--cell primitive|conventional]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
work_dir = script_dir.parent  # input_calc/
sys.path.insert(0, str(work_dir))

from phonon_inputs.structure import load_phonopy_calculation
from phonon_inputs.force_constants import load_fc3_thirdorder, load_fc3_phono3py
from phonon_inputs.anharmonic import anharmonic_transmission

# ---------------------------------------------------------------------------
# Primitive cell workflow (preferred)
# ---------------------------------------------------------------------------

def load_primitive_cell_dfpt(work_dir):
    """Load FC2 + FC3 from the DFPT pipeline on the same primitive/supercell basis.

    Uses the DFPT-produced fc3.hdf5 together with the phono3py YAML from the
    finite-displacement setup to reconstruct the same primitive/supercell mapping.
    """
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    from phonon_inputs.force_constants import load_fc3_dfpt_hdf5

    dfpt_dir = work_dir / "dfpt"
    dfpt_h5 = dfpt_dir / "fc3.hdf5"

    # Reuse the same phono3py YAML that defines the primitive/supercell mapping.
    ref_dir = work_dir / "fc3_prim"
    yaml_path = ref_dir / "phono3py_disp.yaml"

    if not dfpt_h5.exists():
        raise FileNotFoundError(
            f"DFPT FC3 not found at {dfpt_h5}.\n"
            "Run the DFPT pipeline first:\n"
            "  python -m phonon_inputs dfpt-sow  --config config_dfpt_dir.yaml\n"
            "  python -m phonon_inputs dfpt-run  --config config_dfpt_dir.yaml\n"
            "  python -m phonon_inputs dfpt-reap --config config_dfpt_dir.yaml"
        )

    print("Loading FC3 from DFPT HDF5...")
    payload = load_fc3_dfpt_hdf5(
        dfpt_hdf5=dfpt_h5,
        phono3py_yaml=yaml_path,
    )
    fc2 = payload["fc2"]
    fc3_data = payload["fc3_data"]
    ph3 = payload["ph3"]

    print(f"  FC3 blocks: {fc3_data['n_blocks']}")
    print(f"  FC2 shape: {fc2.shape}")

    cell = PhonopyAtoms(
        symbols=ph3.unitcell.symbols,
        cell=ph3.unitcell.cell,
        scaled_positions=ph3.unitcell.scaled_positions,
    )
    phonon = Phonopy(
        cell,
        supercell_matrix=ph3.supercell_matrix,
        primitive_matrix=np.eye(3),
    )
    phonon.force_constants = fc2

    n_atoms = len(phonon.primitive.masses)
    print(f"  Primitive cell: {n_atoms} atoms, a1 = {phonon.primitive.cell[0]}")

    return phonon, fc3_data


def load_primitive_cell(work_dir):
    """Load FC2 + FC3 from phono3py for the 2-atom FCC primitive cell.

    The phono3py calculation provides both FC2 and FC3 computed on the
    same basis, so no remapping is needed.
    """
    import h5py
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    fc3_dir = work_dir / "fc3_prim"
    fc3_h5 = fc3_dir / "fc3.hdf5"
    yaml_path = fc3_dir / "phono3py_disp.yaml"

    if not fc3_h5.exists():
        raise FileNotFoundError(
            f"FC3 not found at {fc3_h5}.\n"
            "Run the phono3py pipeline first:\n"
            "  python -m phonon_inputs fc3-sow  --config config_prim.yaml\n"
            "  python -m phonon_inputs fc3-run  --config config_prim.yaml\n"
            "  python -m phonon_inputs fc3-reap --config config_prim.yaml"
        )

    # Load FC3 via phono3py
    print("Loading FC3 from phono3py (primitive cell)...")
    fc3_data = load_fc3_phono3py(
        phono3py_yaml=yaml_path,
        fc3_hdf5=fc3_h5,
    )
    print(f"  FC3 blocks: {fc3_data['n_blocks']}")

    # Load FC2 from the same HDF5 (produced by symfc alongside FC3)
    with h5py.File(fc3_h5, "r") as f:
        fc2 = f["fc2"][:]
    print(f"  FC2 shape: {fc2.shape}")

    # Reconstruct phonopy object with FC2
    from phono3py import load as phono3py_load
    ph3 = phono3py_load(
        phono3py_yaml=str(yaml_path),
        produce_fc=False,
        log_level=0,
    )

    cell = PhonopyAtoms(
        symbols=ph3.unitcell.symbols,
        cell=ph3.unitcell.cell,
        scaled_positions=ph3.unitcell.scaled_positions,
    )
    phonon = Phonopy(cell, supercell_matrix=ph3.supercell_matrix,
                     primitive_matrix=np.eye(3))
    phonon.force_constants = fc2

    n_atoms = len(phonon.primitive.masses)
    print(f"  Primitive cell: {n_atoms} atoms, "
          f"a1 = {phonon.primitive.cell[0]}")

    return phonon, fc3_data


# ---------------------------------------------------------------------------
# Conventional cell workflow (legacy, with remapping)
# ---------------------------------------------------------------------------

def _remap_fc3_to_conventional(fc3_data, a_conv, conv_cell, conv_frac):
    """Remap FC3 from 2-atom FCC to 8-atom conventional cell."""
    fcc_cell = np.array([
        [0.0, a_conv / 2, a_conv / 2],
        [a_conv / 2, 0.0, a_conv / 2],
        [a_conv / 2, a_conv / 2, 0.0],
    ])
    fcc_frac = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    fcc_cart = fcc_frac @ fcc_cell
    conv_cart = conv_frac @ conv_cell
    conv_inv = np.linalg.inv(conv_cell.T)
    fcc_inv = np.linalg.inv(fcc_cell.T)
    n_conv = len(conv_frac)

    shift = np.array([a_conv / 8] * 3)
    # Try both signs
    for sign in [1, -1]:
        s = sign * shift
        ok = all(
            any(
                np.allclose(fcc_inv @ (conv_cart[i] + s - fcc_cart[j]),
                            np.round(fcc_inv @ (conv_cart[i] + s - fcc_cart[j])),
                            atol=0.05)
                for j in range(2)
            )
            for i in range(n_conv)
        )
        if ok:
            shift = s
            break
    else:
        raise ValueError("Could not determine origin shift for diamond cell.")

    def _fcc_to_conv(fcc_atom, fcc_R_cart):
        pos = fcc_cart[fcc_atom] + fcc_R_cart - shift
        for i in range(n_conv):
            diff = pos - conv_cart[i]
            diff_frac = conv_inv @ diff
            cell_idx = np.round(diff_frac).astype(int)
            err = np.linalg.norm(diff - cell_idx.astype(float) @ conv_cell)
            if err < 0.5:
                return i, cell_idx
        raise ValueError(f"Failed: FCC atom {fcc_atom}, R={fcc_R_cart}")

    a_fcc_orig = 2 * np.linalg.norm(fcc_cell[0]) / np.sqrt(2)
    scale = a_conv / a_fcc_orig if abs(a_conv / a_fcc_orig - 1.0) > 1e-6 else 1.0

    new_blocks = []
    for block in fc3_data["blocks"]:
        R_j = block["cell_j"] * scale
        R_k = block["cell_k"] * scale
        ai, Ri = _fcc_to_conv(block["atom_i"], np.zeros(3))
        aj, Rj = _fcc_to_conv(block["atom_j"], R_j)
        ak, Rk = _fcc_to_conv(block["atom_k"], R_k)
        new_blocks.append({
            "cell_j": (Rj - Ri).astype(float) @ conv_cell,
            "cell_k": (Rk - Ri).astype(float) @ conv_cell,
            "atom_i": ai, "atom_j": aj, "atom_k": ak,
            "tensor": block["tensor"],
        })
    return {"n_blocks": len(new_blocks), "blocks": new_blocks}


def load_conventional_cell(work_dir):
    """Load harmonic FC from phonopy + FC3 from thirdorder (with remapping)."""
    phonon_si = load_phonopy_calculation(
        phonopy_yaml=work_dir / "old" / "scf_disp" / "phonopy_disp.yaml",
        force_sets_filename=work_dir / "old" / "scf_disp" / "FORCE_SETS",
        calculator="qe",
    )
    a_conv = phonon_si.unitcell.cell[0, 0]

    fc3_path = work_dir / "old" / "fc3_si" / "FORCE_CONSTANTS_3RD"
    if not fc3_path.exists():
        raise FileNotFoundError(f"FC3 not found: {fc3_path}")

    print("Loading FC3 from thirdorder.py (2-atom FCC, remapping)...")
    fc3_raw = load_fc3_thirdorder(fc3_path)
    fc3_data = _remap_fc3_to_conventional(
        fc3_raw, a_conv,
        phonon_si.unitcell.cell, phonon_si.unitcell.scaled_positions,
    )
    print(f"  Remapped: {fc3_data['n_blocks']} blocks")

    return phonon_si, fc3_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scba(phonon, fc3_data, nk, nfreq, max_iter, n_slabs, temperature,
             delta_T, mixing, transport_direction):
    """Run SCBA."""
    return anharmonic_transmission(
        phonon, fc3_data,
        q_mesh_transverse=(nk, nk),
        freq_range_thz=(0.5, 15.0, nfreq),
        transport_direction=transport_direction,
        eta_factor=0.5,
        temperature=temperature,
        delta_T=delta_T,
        max_scba_iter=max_iter,
        scba_tol=0.005,
        mixing=mixing,
        fc3_mode="full",
        n_slabs=n_slabs,
        verbose=True,
    )


def save_checkpoint(results, path):
    data = {}
    for key, result in results.items():
        prefix = f"slabs{key}_"
        for rkey, val in result.items():
            if isinstance(val, np.ndarray):
                data[prefix + rkey] = val
            elif isinstance(val, (int, float)):
                data[prefix + rkey] = np.array(val)
            elif isinstance(val, list):
                data[prefix + rkey] = np.array(val)
    np.savez(path, **data)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path):
    path = Path(path)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    results = {}
    for key in data.files:
        parts = key.split("_", 1)
        n_slabs = int(parts[0].replace("slabs", ""))
        rkey = parts[1]
        if n_slabs not in results:
            results[n_slabs] = {}
        val = data[key]
        results[n_slabs][rkey] = val.item() if val.ndim == 0 else val
    return results


def plot_results(results, output_path, temperature):
    n_panels = len(results)
    fig, axes = plt.subplots(1, max(n_panels, 2), figsize=(7 * max(n_panels, 2), 5))
    if n_panels == 1:
        axes = [axes[0], axes[1]]

    for idx, (n_slabs, result) in enumerate(sorted(results.items())):
        ax = axes[idx]
        freqs = result["freqs_thz"]
        ax.plot(freqs, result["spectral_heat_current_ballistic"], "b-", lw=1.5,
                label="Ballistic")
        ax.plot(freqs, result["spectral_heat_current"], "r--", lw=1.5,
                label="Anharmonic (SCBA)")

        G_ball = result["thermal_conductance_ballistic"]
        G_anh = result["thermal_conductance_anharmonic"]
        reduction = (1 - G_anh / G_ball) * 100 if G_ball > 0 else 0

        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("Spectral heat current (W)")
        ax.set_title(f"Si {n_slabs} slab(s), T={temperature} K\n"
                     f"G_ball={G_ball / 1e6:.0f}, G_anh={G_anh / 1e6:.0f} MW/m²K "
                     f"({reduction:.0f}% reduction)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, None)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close("all")
    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Anharmonic phonon transport (SCBA)")
    parser.add_argument("--nk", type=int, default=4, help="Transverse k-mesh (nk x nk)")
    parser.add_argument("--nfreq", type=int, default=101, help="Frequency points")
    parser.add_argument("--max-iter", type=int, default=20, help="Max SCBA iterations")
    parser.add_argument("--n-slabs", type=int, nargs="+", default=[1],
                        help="Device slab counts")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature (K)")
    parser.add_argument("--delta-T", type=float, default=10.0, help="Delta T (K)")
    parser.add_argument("--mixing", type=float, default=0.3, help="SCBA mixing")
    parser.add_argument("--checkpoint", type=str, default="anharmonic_checkpoint.npz")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--cell", choices=["primitive", "conventional"],
                        default="primitive",
                        help="Cell type: primitive (2-atom, phono3py) or "
                             "conventional (8-atom, legacy thirdorder)")
    args = parser.parse_args()

    if args.cell == "primitive":
        transport_dir = "x"  # along a1 = [011] for FCC
    else:
        transport_dir = "z"  # along [001] for conventional

    print("=" * 60)
    print(f"Anharmonic phonon transport: Si bulk ({args.cell} cell)")
    print(f"  k-mesh: {args.nk}x{args.nk}, freq points: {args.nfreq}")
    print(f"  temperature: {args.temperature} K, delta_T: {args.delta_T} K")
    print(f"  transport direction: {transport_dir}")
    print(f"  slabs: {args.n_slabs}, max SCBA iter: {args.max_iter}")
    print("=" * 60)

    if args.cell == "primitive":
        phonon, fc3_data = load_primitive_cell(work_dir)
    else:
        phonon, fc3_data = load_conventional_cell(work_dir)

    n_atoms = len(phonon.primitive.masses)
    print(f"  {n_atoms} atoms per cell")

    results = {}
    if args.resume:
        loaded = load_checkpoint(script_dir / args.checkpoint)
        if loaded:
            results = loaded
            print(f"Resumed: {list(results.keys())} slabs done")

    for n_slabs in args.n_slabs:
        if n_slabs in results:
            print(f"\nSkipping {n_slabs} slab(s) (in checkpoint)")
            continue

        print(f"\n{'=' * 60}")
        print(f"Running SCBA: {n_slabs} slab(s)")
        print(f"{'=' * 60}")

        t0 = time.time()
        result = run_scba(
            phonon, fc3_data,
            nk=args.nk, nfreq=args.nfreq, max_iter=args.max_iter,
            n_slabs=n_slabs, temperature=args.temperature,
            delta_T=args.delta_T, mixing=args.mixing,
            transport_direction=transport_dir,
        )
        elapsed = time.time() - t0

        G_ball = result["thermal_conductance_ballistic"]
        G_anh = result["thermal_conductance_anharmonic"]
        print(f"\n  Completed in {elapsed:.1f} s")
        print(f"  G_ballistic:  {G_ball / 1e6:.1f} MW/(m^2 K)")
        print(f"  G_anharmonic: {G_anh / 1e6:.1f} MW/(m^2 K)")
        if G_ball > 0:
            print(f"  Reduction:    {(1 - G_anh / G_ball) * 100:.1f}%")

        results[n_slabs] = result
        save_checkpoint(results, script_dir / args.checkpoint)

    plot_results(results, script_dir / "anharmonic_results.png", args.temperature)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for n_slabs in sorted(results.keys()):
        r = results[n_slabs]
        G_ball = r["thermal_conductance_ballistic"]
        G_anh = r["thermal_conductance_anharmonic"]
        red = (1 - G_anh / G_ball) * 100 if G_ball > 0 else 0
        print(f"  {n_slabs} slab(s): G_ball={G_ball / 1e6:.1f}, "
              f"G_anh={G_anh / 1e6:.1f} MW/m²K ({red:.1f}% reduction)")


if __name__ == "__main__":
    main()
