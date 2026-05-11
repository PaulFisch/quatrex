"""End-to-end phonon force constant pipeline.

Orchestrates: DFT relaxation -> FC2 + FC3 (phono3py + symfc).

The phono3py reap step produces both FC2 and FC3 from the same set
of displaced supercell calculations, so there is no need for a
separate phonopy FC2 step.

Usage via CLI:
    python -m phonon_inputs pipeline --config config.yaml
    python -m phonon_inputs pipeline --config config.yaml --skip-relax
"""

from pathlib import Path

import numpy as np

from .config import load_config
from .qe_interface import run_qe_relax, run_vasp_relax
from .structure import load_structure


def run_pipeline(
    config_path: str | Path,
    skip_relax: bool = False,
) -> dict:
    """Run the full phonon force constant pipeline.

    Steps:
    1. (Optional) DFT structural relaxation via vc-relax
    2. FC2 + FC3 via phono3py: displacements -> QE SCF -> symfc

    The phono3py reap step produces both FC2 and FC3 in fc3.hdf5.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML config file.
    skip_relax : bool
        Skip structural relaxation; use structure from config as-is.

    Returns
    -------
    summary : dict
        Keys: "cell", "fc3_path".
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    base_dir = config_path.parent
    summary = {}

    # ------------------------------------------------------------------
    # Step 1: Structural relaxation
    # ------------------------------------------------------------------
    cell = load_structure(config.structure)

    if not skip_relax:
        rc = config.relax
        relax_dir = base_dir / rc.work_dir
        print("=" * 60)
        print("Step 1: Structural relaxation")
        print(f"  {rc.calculation}, calculator: {rc.calculator}, "
              f"work_dir: {relax_dir}")
        print("=" * 60)

        if rc.calculator == "vasp":
            cell = run_vasp_relax(
                cell, relax_dir, config.vasp,
                calculation=rc.calculation,
                forc_conv_thr=rc.forc_conv_thr,
                press_conv_thr=rc.press_conv_thr,
            )
        else:
            cell = run_qe_relax(
                cell, relax_dir, config.qe,
                calculation=rc.calculation,
                forc_conv_thr=rc.forc_conv_thr,
                press_conv_thr=rc.press_conv_thr,
            )

        print(f"  Relaxed: {len(cell.symbols)} atoms")
        for i, v in enumerate(cell.cell):
            print(f"    a{i+1} = [{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}] "
                  f"(|a{i+1}| = {np.linalg.norm(v):.4f} A)")
    else:
        print("Step 1: Relaxation skipped (using structure from config)")

    summary["cell"] = cell

    # ------------------------------------------------------------------
    # Step 2: FC2 + FC3
    # ------------------------------------------------------------------
    method = config.fc_method
    valid_methods = {"finite_displacement", "thirdorder", "dfpt", "hiphive"}
    if method not in valid_methods:
        raise ValueError(
            f"fc_method={method!r} is not recognised. "
            f"Expected one of {sorted(valid_methods)}."
        )

    if method == "dfpt":
        from .dfpt import generate_fc_dfpt

        dc = config.dfpt
        fc_dir = base_dir / dc.work_dir

        print("\n" + "=" * 60)
        print("Step 2: Force constants (FC2 + FC3) via DFPT (ph.x + D3Q)")
        print(f"  q-mesh: {dc.q_mesh}, work_dir: {fc_dir}")
        print(f"  k-mesh (SCF): {dc.kpoints}")
        print("=" * 60)

        fc3_path = generate_fc_dfpt(cell, fc_dir, config.qe, dc)

    elif method == "hiphive":
        from .hiphive_fc3 import generate_fc3 as hiphive_generate_fc3

        hh = config.hiphive
        fc_dir = base_dir / hh.work_dir
        dft_config = config.vasp if hh.calculator == "vasp" else config.qe

        print("\n" + "=" * 60)
        print("Step 2: Force constants (FC2 + FC3) via hiphive (rattled SCs)")
        print(f"  Supercell: {tuple(hh.supercell)}, work_dir: {fc_dir}")
        print(f"  n_structures: {hh.n_structures}, rattle_std: {hh.rattle_std} A "
              f"(method: {hh.rattle_method})")
        print(f"  Cutoffs: FC2={hh.cutoffs[0]} A, FC3={hh.cutoffs[1]} A; "
              f"fit: {hh.fit_method}")
        print("=" * 60)

        fc3_path = hiphive_generate_fc3(cell, fc_dir, dft_config, hh)

    else:
        from .thirdorder import generate_fc3

        tc = config.thirdorder
        fc_dir = base_dir / tc.work_dir
        supercell = tuple(tc.supercell)

        print("\n" + "=" * 60)
        print("Step 2: Force constants (FC2 + FC3) via phono3py + symfc")
        print(f"  Supercell: {supercell}, work_dir: {fc_dir}")
        if tc.cutoff_pair_distance:
            print(f"  Pair cutoff: {tc.cutoff_pair_distance} A")
        print(f"  FC calculator: {tc.fc_calculator}")
        print("=" * 60)

        dft_config = config.vasp if tc.calculator == "vasp" else config.qe
        fc3_path = generate_fc3(
            cell, fc_dir, dft_config, supercell,
            cutoff_pair_distance=tc.cutoff_pair_distance,
            distance=tc.displacement_distance,
            fc_calculator=tc.fc_calculator,
            calculator=tc.calculator,
        )

    print(f"\n  FC2 + FC3 saved to: {fc3_path}")
    summary["fc3_path"] = fc3_path

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Pipeline complete")
    print(f"  Output: {fc3_path}")
    print("=" * 60)

    return summary
