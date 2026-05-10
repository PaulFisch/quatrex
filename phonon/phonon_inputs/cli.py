"""Command-line interface for phonon input generation.

Usage:
    python -m phonon_inputs generate --config config.yaml
    python -m phonon_inputs extract-blocks --config config.yaml
    python -m phonon_inputs validate --config config.yaml
    python -m phonon_inputs fc3-sow --config config.yaml
    python -m phonon_inputs fc3-run --config config.yaml
    python -m phonon_inputs fc3-reap --config config.yaml
    python -m phonon_inputs fc3-all --config config.yaml
    python -m phonon_inputs fc3-hiphive-sow  --config config.yaml
    python -m phonon_inputs fc3-hiphive-run  --config config.yaml
    python -m phonon_inputs fc3-hiphive-reap --config config.yaml
    python -m phonon_inputs fc3-hiphive-all  --config config.yaml
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


def _get_dft_config(config):
    """Return the DFT config and command for the active calculator."""
    calc = config.thirdorder.calculator
    if calc == "vasp":
        return config.vasp, config.vasp.vasp_command
    return config.qe, config.qe.pw_command


def _get_hiphive_dft_config(config):
    """Return the DFT config and command for the hiphive calculator."""
    calc = config.hiphive.calculator
    if calc == "vasp":
        return config.vasp, config.vasp.vasp_command
    return config.qe, config.qe.pw_command


def cmd_fc3_sow(config_path: str) -> None:
    """Generate phono3py displaced supercells and write DFT inputs."""
    from .config import load_config
    from .structure import load_structure
    from .thirdorder import sow

    config = load_config(config_path)
    cell = load_structure(config.structure)

    tc = config.thirdorder
    dft_config, _ = _get_dft_config(config)
    work_dir = Path(config_path).parent / tc.work_dir
    supercell = tuple(tc.supercell)

    n_disp = sow(
        cell, work_dir, dft_config, supercell,
        cutoff_pair_distance=tc.cutoff_pair_distance,
        distance=tc.displacement_distance,
        calculator=tc.calculator,
    )
    print(f"\n{n_disp} displacement inputs in {work_dir}")
    print("Run 'fc3-run' to execute DFT, then 'fc3-reap' to produce FC3.")


def cmd_fc3_run(config_path: str) -> None:
    """Run DFT for all FC3 displacements."""
    from .config import load_config
    from .thirdorder import run_displacements

    config = load_config(config_path)
    tc = config.thirdorder
    _, dft_command = _get_dft_config(config)
    work_dir = Path(config_path).parent / tc.work_dir

    run_displacements(work_dir, dft_command, tc.pw_timeout,
                      calculator=tc.calculator)


def cmd_fc3_reap(config_path: str) -> None:
    """Read DFT forces and produce FC3 via phono3py + symfc."""
    from .config import load_config
    from .thirdorder import reap

    config = load_config(config_path)
    tc = config.thirdorder
    work_dir = Path(config_path).parent / tc.work_dir

    fc3_path = reap(work_dir, fc_calculator=tc.fc_calculator,
                    calculator=tc.calculator)
    print(f"\nFC3 file: {fc3_path}")


def cmd_fc3_all(config_path: str) -> None:
    """Full FC3 pipeline: sow + run + reap."""
    from .config import load_config
    from .structure import load_structure
    from .thirdorder import generate_fc3

    config = load_config(config_path)
    cell = load_structure(config.structure)

    tc = config.thirdorder
    dft_config, _ = _get_dft_config(config)
    work_dir = Path(config_path).parent / tc.work_dir
    supercell = tuple(tc.supercell)

    fc3_path = generate_fc3(
        cell, work_dir, dft_config, supercell,
        cutoff_pair_distance=tc.cutoff_pair_distance,
        distance=tc.displacement_distance,
        fc_calculator=tc.fc_calculator,
        calculator=tc.calculator,
    )
    print(f"\nFC3 file: {fc3_path}")


def cmd_hiphive_sow(config_path: str) -> None:
    """Generate hiphive rattled supercells and write DFT inputs."""
    from .config import load_config
    from .hiphive_fc3 import sow
    from .structure import load_structure

    config = load_config(config_path)
    cell = load_structure(config.structure)

    hh = config.hiphive
    dft_config, _ = _get_hiphive_dft_config(config)
    work_dir = Path(config_path).parent / hh.work_dir

    n_disp = sow(cell, work_dir, dft_config, hh)
    print(f"\n{n_disp} rattled inputs in {work_dir}")
    print("Run 'fc3-hiphive-run' to execute DFT, then 'fc3-hiphive-reap' to fit FC3.")


def cmd_hiphive_run(config_path: str) -> None:
    """Run DFT for all hiphive rattled structures."""
    from .config import load_config
    from .hiphive_fc3 import run_displacements

    config = load_config(config_path)
    hh = config.hiphive
    dft_config, dft_command = _get_hiphive_dft_config(config)
    work_dir = Path(config_path).parent / hh.work_dir

    run_displacements(
        work_dir, dft_command,
        timeout=hh.pw_timeout,
        calculator=hh.calculator,
        dft_config=dft_config,
    )


def cmd_hiphive_reap(config_path: str) -> None:
    """Fit hiphive cluster expansion and write fc3.hdf5."""
    from .config import load_config
    from .hiphive_fc3 import reap

    config = load_config(config_path)
    hh = config.hiphive
    work_dir = Path(config_path).parent / hh.work_dir

    fc3_path = reap(work_dir, hh_config=hh)
    print(f"\nFC3 file: {fc3_path}")


def cmd_hiphive_all(config_path: str) -> None:
    """Full hiphive FC3 pipeline: sow + run + reap."""
    from .config import load_config
    from .hiphive_fc3 import generate_fc3
    from .structure import load_structure

    config = load_config(config_path)
    cell = load_structure(config.structure)

    hh = config.hiphive
    dft_config, _ = _get_hiphive_dft_config(config)
    work_dir = Path(config_path).parent / hh.work_dir

    fc3_path = generate_fc3(cell, work_dir, dft_config, hh)
    print(f"\nFC3 file: {fc3_path}")


def cmd_dfpt_sow(config_path: str) -> None:
    """Generate DFPT input files (SCF + ph.x + d3q.x)."""
    from .config import load_config
    from .dfpt import sow
    from .structure import load_structure

    config = load_config(config_path)
    cell = load_structure(config.structure)

    dc = config.dfpt
    work_dir = Path(config_path).parent / dc.work_dir

    n = sow(cell, work_dir, config.qe, dc)
    print(f"\n{n} d3q.x triplet inputs + SCF/ph/q2r/qq2rr in {work_dir}")
    print("Run 'dfpt-run' to execute, then 'dfpt-reap' to produce FC2+FC3.")


def cmd_dfpt_run(config_path: str) -> None:
    """Run all DFPT calculations (SCF -> ph -> q2r -> d3q -> qq2rr)."""
    from .config import load_config
    from .dfpt import run_all

    config = load_config(config_path)
    dc = config.dfpt
    work_dir = Path(config_path).parent / dc.work_dir

    run_all(work_dir, config.qe, dc)


def cmd_dfpt_reap(config_path: str) -> None:
    """Parse DFPT outputs and produce fc3.hdf5."""
    from .config import load_config
    from .dfpt import reap
    from .structure import load_structure

    config = load_config(config_path)
    cell = load_structure(config.structure)
    dc = config.dfpt
    work_dir = Path(config_path).parent / dc.work_dir

    fc3_path = reap(work_dir, cell, dc.q_mesh)
    print(f"\nFC2+FC3 file: {fc3_path}")


def cmd_dfpt_all(config_path: str) -> None:
    """Full DFPT pipeline: sow + run + reap."""
    from .config import load_config
    from .dfpt import generate_fc_dfpt
    from .structure import load_structure

    config = load_config(config_path)
    cell = load_structure(config.structure)

    dc = config.dfpt
    work_dir = Path(config_path).parent / dc.work_dir

    fc3_path = generate_fc_dfpt(cell, work_dir, config.qe, dc)
    print(f"\nFC2+FC3 file: {fc3_path}")


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

    p_fc3_sow = sub.add_parser("fc3-sow", help="Generate phono3py displacements + DFT inputs (QE/VASP)")
    p_fc3_sow.add_argument("--config", required=True, help="YAML config file")

    p_fc3_run = sub.add_parser("fc3-run", help="Run DFT for FC3 displacements (QE/VASP)")
    p_fc3_run.add_argument("--config", required=True, help="YAML config file")

    p_fc3_reap = sub.add_parser("fc3-reap", help="Read forces, produce FC3 via symfc")
    p_fc3_reap.add_argument("--config", required=True, help="YAML config file")

    p_fc3_all = sub.add_parser("fc3-all", help="Full FC3 pipeline: sow + run + reap")
    p_fc3_all.add_argument("--config", required=True, help="YAML config file")

    p_hh_sow = sub.add_parser(
        "fc3-hiphive-sow", help="Generate hiphive rattled supercells + DFT inputs"
    )
    p_hh_sow.add_argument("--config", required=True, help="YAML config file")

    p_hh_run = sub.add_parser(
        "fc3-hiphive-run", help="Run DFT for hiphive rattled structures"
    )
    p_hh_run.add_argument("--config", required=True, help="YAML config file")

    p_hh_reap = sub.add_parser(
        "fc3-hiphive-reap", help="Fit hiphive cluster expansion -> fc3.hdf5"
    )
    p_hh_reap.add_argument("--config", required=True, help="YAML config file")

    p_hh_all = sub.add_parser(
        "fc3-hiphive-all", help="Full hiphive FC3 pipeline: sow + run + reap"
    )
    p_hh_all.add_argument("--config", required=True, help="YAML config file")

    p_dfpt_sow = sub.add_parser("dfpt-sow", help="Generate DFPT input files")
    p_dfpt_sow.add_argument("--config", required=True, help="YAML config file")

    p_dfpt_run = sub.add_parser("dfpt-run", help="Run DFPT calculations")
    p_dfpt_run.add_argument("--config", required=True, help="YAML config file")

    p_dfpt_reap = sub.add_parser("dfpt-reap", help="Parse DFPT outputs -> fc3.hdf5")
    p_dfpt_reap.add_argument("--config", required=True, help="YAML config file")

    p_dfpt_all = sub.add_parser("dfpt-all", help="Full DFPT: sow + run + reap")
    p_dfpt_all.add_argument("--config", required=True, help="YAML config file")

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
        "fc3-hiphive-sow": lambda: cmd_hiphive_sow(args.config),
        "fc3-hiphive-run": lambda: cmd_hiphive_run(args.config),
        "fc3-hiphive-reap": lambda: cmd_hiphive_reap(args.config),
        "fc3-hiphive-all": lambda: cmd_hiphive_all(args.config),
        "dfpt-sow": lambda: cmd_dfpt_sow(args.config),
        "dfpt-run": lambda: cmd_dfpt_run(args.config),
        "dfpt-reap": lambda: cmd_dfpt_reap(args.config),
        "dfpt-all": lambda: cmd_dfpt_all(args.config),
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
