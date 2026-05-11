"""Shared scaffolding for ``examples/setup_*.py`` scripts.

Pure scaffolding — no chemistry. Each setup script imports from here for
argparse boilerplate, YAML writing, and the boring DFT/relax/hiphive/
thirdorder default-block dicts.

The chemistry-specific generation (atom positions, lattice, potcar map)
stays in the individual setup script. The helpers compose into a final
config dict that the existing ``phonon_inputs.config_from_dict`` parses.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# argparse + YAML I/O                                                         #
# --------------------------------------------------------------------------- #


def add_common_args(
    parser: argparse.ArgumentParser, *, default_out: Path,
) -> None:
    """Standard ``--out`` argument with a sensible default."""
    parser.add_argument(
        "--out", type=Path, default=default_out,
        help=f"Output YAML path (default: {default_out})",
    )


def write_config(cfg: dict, out_path: Path) -> Path:
    """Write a config dict as YAML and report a one-line summary.

    Uses a custom dumper so that the structure is compact and matches the
    hand-written canonical configs:
      * ``structure.symbols``       — inline, e.g. ``[Si, Si, H, H]``
      * ``structure.lattice``       — outer block, inner inline rows
      * ``structure.scaled_positions`` — outer block, inner inline triples
      * Everything else             — block style (one key per line)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    class _CompactDumper(yaml.SafeDumper):
        pass

    def _list_representer(dumper, data):
        # Inner-most sequences (no nested lists) → inline; outer → block.
        flow = not any(isinstance(x, (list, tuple)) for x in data)
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=flow,
        )

    _CompactDumper.add_representer(list, _list_representer)
    _CompactDumper.add_representer(tuple, _list_representer)

    out_path.write_text(
        yaml.dump(cfg, Dumper=_CompactDumper, sort_keys=False)
    )
    syms = cfg["structure"]["symbols"]
    counts = summarise_atoms(syms)
    summary = " + ".join(f"{n} {sp}" for sp, n in counts.items())
    print(f"Wrote: {out_path}\n  {summary}  ({len(syms)} atoms)")
    return out_path


def summarise_atoms(symbols: list[str]) -> dict[str, int]:
    """Count atoms per species, preserving first-appearance order."""
    out: dict[str, int] = {}
    for s in symbols:
        out[s] = out.get(s, 0) + 1
    return out


# --------------------------------------------------------------------------- #
# DFT / pipeline default blocks                                               #
# --------------------------------------------------------------------------- #


def default_vasp_block(
    potcar_map: dict[str, str],
    *,
    kpoints_scf: tuple[int, int, int] = (1, 1, 4),
    encut: int = 400,
    ediff: float = 1.0e-7,
    sigma: float = 0.01,
    ncore: int = 8,
    kpar: int = 4,
    vasp_command: str = "ulimit -s unlimited; mpirun -np 128 vasp_std",
    potcar_dir: str = "/home/jiacao/vasp/potpaw_PBE",
) -> dict[str, Any]:
    return {
        "vasp": {
            "potcar_dir": potcar_dir, "potcar_map": dict(potcar_map),
            "encut": encut, "ediff": ediff,
            "ismear": 0, "sigma": sigma,
            "prec": "Accurate", "lreal": "Auto",
            "lwave": False, "lcharg": False,
            "ncore": ncore, "kpar": kpar,
            "kpoints_scf": list(kpoints_scf),
            "vasp_command": vasp_command,
        }
    }


def default_qe_block(
    pseudopotentials: dict[str, str],
    *,
    pseudo_dir: str = "./pseudo",
    ecutwfc: int = 60,
    ecutrho_factor: int = 8,
    kpoints_scf: tuple[int, int, int] = (2, 2, 2),
    kpoints_relax: tuple[int, int, int] = (8, 8, 8),
    pw_command: str = "mpirun -np 4 pw.x",
) -> dict[str, Any]:
    return {
        "qe": {
            "pseudo_dir": pseudo_dir,
            "pseudopotentials": dict(pseudopotentials),
            "ecutwfc": ecutwfc, "ecutrho_factor": ecutrho_factor,
            "kpoints_scf": list(kpoints_scf),
            "kpoints_relax": list(kpoints_relax),
            "conv_thr": 1e-10, "smearing": "gaussian", "degauss": 0.01,
            "pw_command": pw_command,
        }
    }


def default_relax_block(
    work_dir: str, *,
    fc_method: str = "hiphive",
    calculation: str = "relax",
    forc_conv_thr: float = 0.005,
    calculator: str = "vasp",
) -> dict[str, Any]:
    return {
        "fc_method": fc_method,
        "relax": {
            "calculation": calculation, "forc_conv_thr": forc_conv_thr,
            "calculator": calculator, "work_dir": work_dir,
        },
    }


def default_hiphive_block(
    work_dir: str,
    supercell: list[int],
    *,
    cutoffs: tuple[float, float] = (5.0, 4.0),
    n_structures: int = 6,
    rattle_std: float = 0.03,
    rattle_d_min: float = 1.3,
    rattle_n_iter: int = 10,
    rattle_seed: int = 42,
    calculator: str = "vasp",
    pw_timeout: int = 7200,
) -> dict[str, Any]:
    return {
        "hiphive": {
            "supercell": list(supercell),
            "n_structures": n_structures, "rattle_method": "mc",
            "rattle_std": rattle_std, "rattle_d_min": rattle_d_min,
            "rattle_n_iter": rattle_n_iter, "rattle_seed": rattle_seed,
            "cutoffs": list(cutoffs),
            "fit_method": "least-squares", "fit_kwargs": {},
            "calculator": calculator, "work_dir": work_dir,
            "pw_timeout": pw_timeout,
        }
    }


def default_thirdorder_block(
    work_dir: str,
    supercell: list[int],
    *,
    cutoff_pair_distance: float | None = None,
    displacement_distance: float = 0.03,
    fc_calculator: str = "symfc",
    calculator: str = "vasp",
    pw_timeout: int = 7200,
) -> dict[str, Any]:
    return {
        "thirdorder": {
            "supercell": list(supercell),
            "cutoff_pair_distance": cutoff_pair_distance,
            "displacement_distance": displacement_distance,
            "fc_calculator": fc_calculator,
            "calculator": calculator, "work_dir": work_dir,
            "pw_timeout": pw_timeout,
        }
    }
