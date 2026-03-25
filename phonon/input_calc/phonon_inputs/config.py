"""Configuration dataclasses and YAML loader."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass
class StructureConfig:
    """Crystal structure specification.

    Provide either `path` to a structure file (with `source` indicating
    the format), or inline `symbols`, `lattice`, `scaled_positions`.
    """

    source: str = "inline"  # "phonopy_yaml", "cif", "poscar", "qe_input", "inline"
    path: str | None = None
    symbols: list[str] | None = None
    lattice: list[list[float]] | None = None  # 3x3 in Angstrom
    scaled_positions: list[list[float]] | None = None


@dataclass
class SupercellConfig:
    matrix: list[list[int]] = field(
        default_factory=lambda: [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    )
    displacement_distance: float = 0.01  # Angstrom


@dataclass
class QEConfig:
    """Quantum ESPRESSO parameters."""

    pseudo_dir: str = "./pseudo"
    pseudopotentials: dict[str, str] = field(default_factory=dict)
    ecutwfc: float = 60.0
    ecutrho_factor: float = 8.0
    kpoints_scf: list[int] = field(default_factory=lambda: [4, 4, 4])
    kpoints_relax: list[int] = field(default_factory=lambda: [8, 8, 8])
    conv_thr: float = 1e-10
    smearing: str = "gaussian"
    degauss: float = 0.01
    pw_command: str = "pw.x"


@dataclass
class BlockExtractionConfig:
    """IDFT / convention transform settings."""

    q_mesh: list[int] = field(default_factory=lambda: [3, 3, 3])
    transport_direction: str = "z"
    amplitude_cutoff: float = 1e10  # (rad/s)^2; blocks below this are dropped


@dataclass
class QuatrexOutputConfig:
    """quatrex input generation settings."""

    output_dir: str = "./quatrex_inputs"
    num_transport_cells: int = 4
    kpoint_grid: list[int] = field(default_factory=lambda: [4, 4, 1])
    kpoint_shift: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    neighbor_cell_cutoff: list[int] = field(default_factory=lambda: [1, 1, 1])
    eta: float = 1e-8
    left_temperature: float = 301.0
    right_temperature: float = 299.0


@dataclass
class PhononInputConfig:
    """Top-level configuration."""

    structure: StructureConfig = field(default_factory=StructureConfig)
    supercell: SupercellConfig = field(default_factory=SupercellConfig)
    qe: QEConfig = field(default_factory=QEConfig)
    block_extraction: BlockExtractionConfig = field(
        default_factory=BlockExtractionConfig
    )
    quatrex_output: QuatrexOutputConfig = field(default_factory=QuatrexOutputConfig)


def _dict_to_dataclass(cls, d):
    """Recursively convert a dict to a nested dataclass instance."""
    if not isinstance(d, dict):
        return d
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in d.items():
        if k in fieldtypes:
            ft = cls.__dataclass_fields__[k].type
            # Check if the field type is itself a dataclass
            origin = getattr(ft, "__origin__", None)
            if hasattr(ft, "__dataclass_fields__"):
                kwargs[k] = _dict_to_dataclass(ft, v)
            else:
                kwargs[k] = v
    return cls(**kwargs)


def load_config(path: Path | str) -> PhononInputConfig:
    """Load configuration from a YAML file."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _dict_to_dataclass(PhononInputConfig, raw or {})


def config_from_dict(d: dict) -> PhononInputConfig:
    """Create configuration from a dictionary."""
    return _dict_to_dataclass(PhononInputConfig, d)
