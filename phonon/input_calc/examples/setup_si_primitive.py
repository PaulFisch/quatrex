"""Set up Si FCC primitive cell from scratch: relax -> FC2 -> FC3.

This script demonstrates the full pipeline for computing phonon force
constants starting from an approximate lattice constant.

The pipeline:
1. vc-relax: Optimizes cell + ionic positions to find the PBE
   equilibrium lattice constant (expected: a ≈ 5.466 Å for Si).
2. FC2: Phonopy displacements → QE SCF → harmonic force constants.
3. FC3: Phono3py displacements → QE SCF → anharmonic force constants
   via symfc.

After running this script, use run_anharmonic.py to compute thermal
transport with SCBA.

Usage:
    python examples/setup_si_primitive.py

    # Skip relaxation (use known lattice constant):
    python examples/setup_si_primitive.py --skip-relax

    # Only generate inputs, run QE externally, then reap:
    python examples/setup_si_primitive.py --sow-only
"""

import argparse
from pathlib import Path

import numpy as np


def make_si_primitive_config(a: float = 5.43) -> dict:
    """Create a config dict for Si FCC primitive cell.

    Parameters
    ----------
    a : float
        Conventional cubic lattice constant in Angstrom.
        Default 5.43 is an approximate value; vc-relax will find
        the PBE equilibrium (≈ 5.466 Å).
    """
    half = a / 2
    return {
        "structure": {
            "source": "inline",
            "symbols": ["Si", "Si"],
            "lattice": [
                [0.0, half, half],
                [half, 0.0, half],
                [half, half, 0.0],
            ],
            "scaled_positions": [
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
            ],
        },
        "supercell": {
            "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "displacement_distance": 0.01,
        },
        "qe": {
            "pseudo_dir": "./pseudo",
            "pseudopotentials": {"Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF"},
            "ecutwfc": 60,
            "ecutrho_factor": 8,
            "kpoints_scf": [2, 2, 2],
            "kpoints_relax": [8, 8, 8],
            "conv_thr": 1e-10,
            "smearing": "gaussian",
            "degauss": 0.01,
            "pw_command": "mpirun -np 4 pw.x",
        },
        "relax": {
            "calculation": "vc-relax",
            "forc_conv_thr": 1e-4,
            "press_conv_thr": 0.5,
            "work_dir": "./relax",
        },
        "thirdorder": {
            "supercell": [2, 2, 2],
            "cutoff_pair_distance": 5.5,
            "displacement_distance": 0.03,
            "fc_calculator": "symfc",
            "work_dir": "./fc3_prim",
            "pw_timeout": 3600,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Set up Si primitive cell: relax -> FC2 -> FC3"
    )
    parser.add_argument(
        "--skip-relax", action="store_true",
        help="Skip vc-relax, use PBE equilibrium a=5.4662 A directly",
    )
    parser.add_argument(
        "--skip-fc3", action="store_true",
        help="Only compute FC2 (skip anharmonic FC3)",
    )
    parser.add_argument(
        "--sow-only", action="store_true",
        help="Only generate QE input files (don't run DFT)",
    )
    parser.add_argument(
        "--a0", type=float, default=5.43,
        help="Initial lattice constant in Angstrom (default: 5.43)",
    )
    args = parser.parse_args()

    # If skipping relax, use the known PBE equilibrium value
    a = 5.4662 if args.skip_relax else args.a0
    config_dict = make_si_primitive_config(a)

    # Write config to disk so the pipeline can reference paths
    work_dir = Path(__file__).resolve().parent.parent
    config_path = work_dir / "config_prim_auto.yaml"

    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote config: {config_path}")

    if args.sow_only:
        # Just generate displacement inputs
        from phonon_inputs.config import config_from_dict
        from phonon_inputs.structure import load_structure, create_phonopy
        from phonon_inputs.qe_interface import write_qe_scf_input, run_qe_relax
        from phonon_inputs.thirdorder import sow

        config = config_from_dict(config_dict)
        cell = load_structure(config.structure)

        if not args.skip_relax:
            rc = config.relax
            relax_dir = work_dir / rc.work_dir
            print("\nGenerating vc-relax input...")
            write_qe_scf_input(
                relax_dir / "relax.in", cell, config.qe,
                calculation=rc.calculation,
                forc_conv_thr=rc.forc_conv_thr,
                press_conv_thr=rc.press_conv_thr,
            )
            print(f"  -> {relax_dir / 'relax.in'}")
            print("  Run QE manually, then re-run with --skip-relax")
            return

        # FC2 displacements
        sc_matrix = np.array(config.supercell.matrix)
        phonon = create_phonopy(
            cell, sc_matrix,
            primitive_matrix=np.eye(3),
            displacement_distance=config.supercell.displacement_distance,
        )
        fc2_dir = work_dir / "scf_disp"
        fc2_dir.mkdir(parents=True, exist_ok=True)
        (fc2_dir / "results").mkdir(exist_ok=True)
        from phonopy.structure.atoms import PhonopyAtoms as PA
        for i, scell in enumerate(phonon.supercells_with_displacements):
            displaced = PA(
                symbols=phonon.supercell.symbols,
                cell=scell.cell,
                scaled_positions=scell.scaled_positions,
            )
            write_qe_scf_input(fc2_dir / f"disp-{i+1:03d}.in", displaced, config.qe)
        print(f"\nFC2: {len(phonon.displacements)} QE inputs in {fc2_dir}")

        # FC3 displacements
        if not args.skip_fc3:
            tc = config.thirdorder
            fc3_dir = work_dir / tc.work_dir
            n = sow(
                cell, fc3_dir, config.qe,
                supercell=tuple(tc.supercell),
                cutoff_pair_distance=tc.cutoff_pair_distance,
                distance=tc.displacement_distance,
            )
            print(f"FC3: {n} QE inputs in {fc3_dir}")

        print("\nRun all QE jobs, then use:")
        print("  python -m phonon_inputs pipeline --config config_prim_auto.yaml "
              "--skip-relax")
        return

    # Full pipeline
    from phonon_inputs.pipeline import run_pipeline

    summary = run_pipeline(
        config_path,
        skip_relax=args.skip_relax,
        skip_fc3=args.skip_fc3,
    )

    cell = summary["relaxed_cell"]
    a_vec = np.linalg.norm(cell.cell, axis=1)
    a_conv = a_vec[0] * np.sqrt(2)  # FCC -> conventional
    print(f"\nFinal lattice constant: a = {a_conv:.4f} A")
    print(f"FC2 shape: {summary['fc2_shape']}")
    if "fc3_path" in summary:
        print(f"FC3 file: {summary['fc3_path']}")
    print("\nReady for anharmonic transport:")
    print("  python run_anharmonic.py --cell primitive")


if __name__ == "__main__":
    main()
