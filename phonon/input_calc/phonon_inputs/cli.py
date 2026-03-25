"""Command-line interface for phonon input generation.

Usage:
    python -m phonon_inputs generate --config config.yaml
    python -m phonon_inputs extract-blocks --config config.yaml
    python -m phonon_inputs validate --config config.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def cmd_generate(config_path: str, skip_dft: bool = False) -> None:
    """Full pipeline: structure -> FC -> blocks -> quatrex files."""
    from .config import load_config
    from .convention import extract_blocks
    from .force_constants import produce_force_constants
    from .qe_interface import load_existing_forces, run_qe_displacements
    from .quatrex_writer import write_all
    from .structure import create_phonopy_from_config, load_structure

    config = load_config(config_path)
    phonon = create_phonopy_from_config(config)

    n_super = len(phonon.supercell.positions)
    n_disp = len(phonon.displacements)
    print(f"Supercell: {n_super} atoms, displacements: {n_disp}")

    # Get forces
    work_dir = Path(config_path).parent / "scf_disp"
    if skip_dft:
        forces = load_existing_forces(phonon, work_dir)
    else:
        forces = run_qe_displacements(phonon, work_dir, config.qe)

    # Force constants
    produce_force_constants(phonon, forces=forces)
    print(f"Force constants shape: {phonon.force_constants.shape}")

    # Extract blocks
    q_mesh = tuple(config.block_extraction.q_mesh)
    blocks = extract_blocks(
        phonon,
        q_mesh=q_mesh,
        amplitude_cutoff=config.block_extraction.amplitude_cutoff,
    )
    print(f"Extracted {len(blocks)} real-space blocks")

    # Write quatrex inputs
    cell = phonon.primitive
    out = write_all(
        cell, blocks, config.quatrex_output,
        transport_direction=config.block_extraction.transport_direction,
    )
    print(f"Wrote quatrex inputs to {out}")


def cmd_extract_blocks(config_path: str) -> None:
    """Extract Convention B blocks from existing phonopy FC."""
    from .config import load_config
    from .convention import extract_blocks
    from .force_constants import produce_force_constants
    from .qe_interface import load_existing_forces
    from .quatrex_writer import write_all
    from .structure import create_phonopy_from_config

    config = load_config(config_path)
    phonon = create_phonopy_from_config(config)

    work_dir = Path(config_path).parent / "scf_disp"
    forces = load_existing_forces(phonon, work_dir)
    produce_force_constants(phonon, forces=forces)

    q_mesh = tuple(config.block_extraction.q_mesh)
    blocks = extract_blocks(
        phonon,
        q_mesh=q_mesh,
        amplitude_cutoff=config.block_extraction.amplitude_cutoff,
    )
    print(f"Extracted {len(blocks)} blocks")
    for key in sorted(blocks):
        print(f"  {key}: max |H| = {np.max(np.abs(blocks[key])):.3e}")

    cell = phonon.primitive
    out = write_all(
        cell, blocks, config.quatrex_output,
        transport_direction=config.block_extraction.transport_direction,
    )
    print(f"Wrote quatrex inputs to {out}")


def cmd_validate(config_path: str) -> None:
    """Run validation checks on extracted blocks."""
    from .config import load_config
    from .convention import extract_blocks
    from .force_constants import produce_force_constants
    from .qe_interface import load_existing_forces
    from .structure import create_phonopy_from_config
    from .validation import (
        check_block_symmetry,
        check_gamma_point,
        reference_transmission,
        thermal_conductance,
    )

    config = load_config(config_path)
    phonon = create_phonopy_from_config(config)

    work_dir = Path(config_path).parent / "scf_disp"
    forces = load_existing_forces(phonon, work_dir)
    produce_force_constants(phonon, forces=forces)

    q_mesh = tuple(config.block_extraction.q_mesh)
    blocks = extract_blocks(phonon, q_mesh=q_mesh)

    # Gamma check
    gamma = check_gamma_point(blocks)
    print("Gamma-point check:")
    print(f"  Acoustic freqs (THz): {gamma['acoustic_freqs_thz']}")
    print(f"  Symmetry error: {gamma['symmetry_error']:.2e}")

    # Block symmetry
    sym = check_block_symmetry(blocks)
    print(f"  H(R)^T = H(-R) max error: {sym['max_error']:.2e}")

    # Reference transmission
    td = config.block_extraction.transport_direction
    kg = config.quatrex_output.kpoint_grid
    tidx = "xyz".index(td)
    perp_k = [kg[i] for i in range(3) if i != tidx]

    print(f"\nReference transmission ({perp_k[0]}x{perp_k[1]} q-mesh)...")
    freqs, trans = reference_transmission(
        phonon, tuple(perp_k), transport_direction=td,
    )
    print(f"  Max transmission: {trans.max():.4f}")

    # Thermal conductance
    T = 300.0
    G = thermal_conductance(
        freqs, trans, T, phonon.primitive.cell, td
    )
    print(f"  Thermal conductance @ {T} K: {G / 1e6:.1f} MW/(m^2 K)")


def main():
    parser = argparse.ArgumentParser(
        prog="phonon_inputs",
        description="Generate quatrex NEGF phonon transport inputs.",
    )
    sub = parser.add_subparsers(dest="command")

    p_gen = sub.add_parser("generate", help="Full pipeline")
    p_gen.add_argument("--config", required=True, help="YAML config file")
    p_gen.add_argument("--skip-dft", action="store_true",
                       help="Load existing QE outputs instead of running")

    p_ext = sub.add_parser("extract-blocks", help="Extract blocks only")
    p_ext.add_argument("--config", required=True, help="YAML config file")

    p_val = sub.add_parser("validate", help="Run validation checks")
    p_val.add_argument("--config", required=True, help="YAML config file")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args.config, skip_dft=args.skip_dft)
    elif args.command == "extract-blocks":
        cmd_extract_blocks(args.config)
    elif args.command == "validate":
        cmd_validate(args.config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
