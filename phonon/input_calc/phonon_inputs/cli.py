"""Command-line interface for phonon input generation.

Usage:
    python -m phonon_inputs generate --config config.yaml
    python -m phonon_inputs extract-blocks --config config.yaml
    python -m phonon_inputs validate --config config.yaml
    python -m phonon_inputs fc3-sow --config config.yaml
    python -m phonon_inputs fc3-run --config config.yaml
    python -m phonon_inputs fc3-reap --config config.yaml
    python -m phonon_inputs fc3-all --config config.yaml
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


def cmd_fc3_sow(config_path: str) -> None:
    """Generate phono3py displaced supercells and write QE inputs."""
    from .config import load_config
    from .structure import load_structure
    from .thirdorder import sow

    config = load_config(config_path)
    cell = load_structure(config.structure)

    tc = config.thirdorder
    work_dir = Path(config_path).parent / tc.work_dir
    supercell = tuple(tc.supercell)

    n_disp = sow(
        cell, work_dir, config.qe, supercell,
        cutoff_pair_distance=tc.cutoff_pair_distance,
        distance=tc.displacement_distance,
    )
    print(f"\n{n_disp} QE input files in {work_dir}")
    print("Run 'fc3-run' to execute QE, then 'fc3-reap' to produce FC3.")


def cmd_fc3_run(config_path: str) -> None:
    """Run QE for all FC3 displacements."""
    from .config import load_config
    from .thirdorder import run_displacements

    config = load_config(config_path)
    tc = config.thirdorder
    work_dir = Path(config_path).parent / tc.work_dir

    run_displacements(work_dir, config.qe.pw_command, tc.pw_timeout)


def cmd_fc3_reap(config_path: str) -> None:
    """Read QE forces and produce FC3 via phono3py + symfc."""
    from .config import load_config
    from .thirdorder import reap

    config = load_config(config_path)
    tc = config.thirdorder
    work_dir = Path(config_path).parent / tc.work_dir

    fc3_path = reap(work_dir, fc_calculator=tc.fc_calculator)
    print(f"\nFC3 file: {fc3_path}")


def cmd_fc3_all(config_path: str) -> None:
    """Full FC3 pipeline: sow + run + reap."""
    from .config import load_config
    from .structure import load_structure
    from .thirdorder import generate_fc3

    config = load_config(config_path)
    cell = load_structure(config.structure)

    tc = config.thirdorder
    work_dir = Path(config_path).parent / tc.work_dir
    supercell = tuple(tc.supercell)

    fc3_path = generate_fc3(
        cell, work_dir, config.qe, supercell,
        cutoff_pair_distance=tc.cutoff_pair_distance,
        distance=tc.displacement_distance,
        fc_calculator=tc.fc_calculator,
    )
    print(f"\nFC3 file: {fc3_path}")


def cmd_pipeline(config_path: str, skip_relax: bool = False) -> None:
    """Full pipeline: relax -> FC2 + FC3 (via phono3py + symfc)."""
    from .pipeline import run_pipeline

    run_pipeline(config_path, skip_relax=skip_relax)


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

    p_fc3_sow = sub.add_parser("fc3-sow", help="Generate phono3py displacements + QE inputs")
    p_fc3_sow.add_argument("--config", required=True, help="YAML config file")

    p_fc3_run = sub.add_parser("fc3-run", help="Run QE for FC3 displacements")
    p_fc3_run.add_argument("--config", required=True, help="YAML config file")

    p_fc3_reap = sub.add_parser("fc3-reap", help="Read forces, produce FC3 via symfc")
    p_fc3_reap.add_argument("--config", required=True, help="YAML config file")

    p_fc3_all = sub.add_parser("fc3-all", help="Full FC3 pipeline: sow + run + reap")
    p_fc3_all.add_argument("--config", required=True, help="YAML config file")

    p_pipe = sub.add_parser("pipeline", help="Full pipeline: relax -> FC2 + FC3")
    p_pipe.add_argument("--config", required=True, help="YAML config file")
    p_pipe.add_argument("--skip-relax", action="store_true",
                        help="Skip structural relaxation")

    args = parser.parse_args()

    commands = {
        "generate": lambda: cmd_generate(args.config, skip_dft=args.skip_dft),
        "extract-blocks": lambda: cmd_extract_blocks(args.config),
        "validate": lambda: cmd_validate(args.config),
        "fc3-sow": lambda: cmd_fc3_sow(args.config),
        "fc3-run": lambda: cmd_fc3_run(args.config),
        "fc3-reap": lambda: cmd_fc3_reap(args.config),
        "fc3-all": lambda: cmd_fc3_all(args.config),
        "pipeline": lambda: cmd_pipeline(
            args.config, skip_relax=args.skip_relax,
        ),
    }

    if args.command in commands:
        commands[args.command]()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
