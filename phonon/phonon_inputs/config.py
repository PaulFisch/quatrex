"""Configuration dataclasses and YAML loader."""

import warnings
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
class VASPConfig:
    """VASP parameters."""

    potcar_dir: str = "./potcar"
    potcar_map: dict[str, str] = field(default_factory=dict)
    encut: float = 500.0
    ediff: float = 1e-8
    ismear: int = 0
    sigma: float = 0.05
    prec: str = "Accurate"
    lreal: str = "Auto"
    lwave: bool = False
    lcharg: bool = False
    ncore: int | None = None       # cores per orbital (good: sqrt(ntasks/kpar))
    kpar: int | None = None        # k-point parallelization groups
    kpoints_scf: list[int] = field(default_factory=lambda: [4, 4, 4])
    vasp_command: str = "vasp_std"
    nsw: int = 300                 # max ionic steps for relaxation


@dataclass
class BlockExtractionConfig:
    """IDFT / convention transform settings."""

    q_mesh: list[int] = field(default_factory=lambda: [3, 3, 3])
    transport_direction: str = "z"
    amplitude_cutoff: float = 1e10  # (rad/s)^2; blocks below this are dropped


@dataclass
class QuatrexOutputConfig:
    """quatrex input generation settings."""

    output_dir: str = "./outputs/quatrex_inputs"
    num_transport_cells: int = 4
    kpoint_grid: list[int] = field(default_factory=lambda: [4, 4, 1])
    kpoint_shift: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    neighbor_cell_cutoff: list[int] = field(default_factory=lambda: [1, 1, 1])
    eta: float = 1e-8
    left_temperature: float = 301.0
    right_temperature: float = 299.0
    # Heterogeneous L|D|R device (all four must be set together or all None)
    num_left_cells: int | None = None
    num_right_cells: int | None = None
    left_matrix: str | None = None
    right_matrix: str | None = None


@dataclass
class ThirdOrderConfig:
    """Third-order force constant (FC3) settings via phono3py + symfc."""

    supercell: list[int] = field(default_factory=lambda: [2, 2, 2])
    cutoff_pair_distance: float | None = None  # Angstrom; None = all pairs
    displacement_distance: float = 0.03  # Angstrom
    fc_calculator: str = "symfc"  # "symfc" or None
    calculator: str = "qe"  # "qe" or "vasp"
    work_dir: str = "./fc3"
    pw_timeout: int = 3600  # seconds per DFT job


@dataclass
class ConvergenceConfig:
    """Convergence sweep for the hiphive FC3 fit.

    Subsamples ``sizes`` structures from a master pool of ``pool_size``
    DFT-evaluated rattled supercells and refits each subset with every
    method in :attr:`fit_methods`. The output of
    :func:`hiphive_convergence.run_convergence_check` is a JSON summary
    plus a side-by-side plot of RMSE / dispersion stability vs.
    ``n_structures``.

    Attributes
    ----------
    sizes
        Subsample sizes to sweep. Must be sorted ascending.
    pool_size
        Master DFT pool size. ``sow`` produces this many displacements;
        ``run_convergence_check`` subsamples down to each ``sizes`` entry.
    test_fraction
        ``train_size`` argument forwarded to ``trainstation.Optimizer``
        (the test fraction is ``1 - train_size``).
    seed
        RNG seed for the subsampling permutation. Reproducibility across
        sweeps requires this to be fixed.
    fit_methods
        Tuple of ``trainstation`` fit methods to compare. See
        ``phonon_inputs/HIPHIVE_FITTING_NOTES.md`` for the recommended
        set under different regimes.
    dispersion_q_mesh
        Q-mesh on which the post-fit dispersion is evaluated for the
        ``dispersion_max_thz`` and ``n_imaginary`` metrics.
    """

    sizes: list[int] = field(default_factory=lambda: [6, 12, 18, 24])
    pool_size: int = 32
    test_fraction: float = 0.2
    seed: int = 0
    fit_methods: tuple[str, ...] = ("rfe-cv", "ardr")
    dispersion_q_mesh: list[int] = field(default_factory=lambda: [8, 8, 8])


@dataclass
class HiphiveConfig:
    """Hiphive FC3 settings via randomized rattled supercells.

    Builds a force-constant potential up to 3rd order from a small set
    of rattled supercell DFT calculations. Compared to phono3py finite
    displacements, hiphive needs many fewer DFT runs but each run
    contains forces from many simultaneously displaced atoms.

    Parameters
    ----------
    supercell : 3 ints
        Diagonal supercell multipliers used for the rattled structures.
    n_structures : int
        Number of rattled supercells to generate.
    rattle_method : str
        "mc" for Monte-Carlo rattling (recommended; respects d_min),
        or "normal" for plain normal-distribution rattling.
    rattle_std : float
        Rattle standard deviation in Angstrom. For mc rattling the
        final RMS displacement also depends on n_iter.
    rattle_d_min : float
        Minimum interatomic distance allowed during MC rattling, in
        Angstrom. Ignored for "normal" rattling.
    rattle_n_iter : int
        Number of MC iterations per atom (mc rattling only).
    rattle_seed : int
        Seed for the rattle RNG.
    cutoffs : list[float]
        Cutoff radii (Angstrom) per cluster order. The list length sets
        the maximum order; e.g. [6.0, 4.0] gives 2nd order to 6 A and
        3rd order to 4 A.
    fit_method : str
        scikit-learn / trainstation fit method, e.g. "least-squares",
        "lasso", "rfe", "ardr".
    fit_kwargs : dict
        Extra kwargs forwarded to the fit method (e.g. {"alpha": 1e-4}).
    fc_calculator : str
        Name written into the produced phono3py params file. Pure
        bookkeeping for downstream consumers.
    calculator : str
        DFT calculator: "qe" or "vasp".
    work_dir : str
        Directory (relative to the config file) where rattled inputs
        and force outputs are stored.
    pw_timeout : int
        Per-DFT-job timeout in seconds.
    """

    supercell: list[int] = field(default_factory=lambda: [2, 2, 2])
    n_structures: int = 5
    rattle_method: str = "mc"
    rattle_std: float = 0.03
    rattle_d_min: float = 2.0
    rattle_n_iter: int = 10
    rattle_seed: int = 42
    cutoffs: list[float] = field(default_factory=lambda: [6.0, 4.0])
    fit_method: str = "least-squares"
    fit_kwargs: dict = field(default_factory=dict)
    fc_calculator: str = "hiphive"
    calculator: str = "qe"
    work_dir: str = "./fc3_hiphive"
    pw_timeout: int = 3600
    # Phase-5 additions: rotational sum-rule enforcement and the
    # n_structures convergence sweep. See HIPHIVE_FITTING_NOTES.md for
    # the rationale and the suggested default per system type.
    rotational_sum_rule: str = "off"  # "off" | "post_fit" | "constrained"
    convergence: ConvergenceConfig | None = None
    # Phase-6 additions: phonon-mode rattle ("phonon-rattled" pool from
    # hiphive.structure_generation.generate_phonon_rattled_structures).
    # Requires a seed FC2; the bootstrap stage runs a small mc-rattle
    # pool to fit one, then the main pool uses the seed.
    phonon_rattle_temperature_k: float = 300.0
    phonon_rattle_bootstrap_n: int = 4
    phonon_rattle_bootstrap_seed: int = 0
    phonon_rattle_imag_freq_factor: float = 1.0
    phonon_rattle_qm: bool = True


@dataclass
class DFPTConfig:
    """DFPT force constant settings via QE ph.x + D3Q.

    Uses Density Functional Perturbation Theory for FC2 (ph.x + q2r.x)
    and third-order DFPT for FC3 (d3q.x + d3_qq2rr.x).
    """

    q_mesh: list[int] = field(default_factory=lambda: [2, 2, 2])
    kpoints: list[int] = field(default_factory=lambda: [8, 8, 8])
    tr2_ph: float = 1e-14  # ph.x SCF convergence threshold
    ph_command: str = "ph.x"
    q2r_command: str = "q2r.x"
    d3q_command: str = "d3q.x"
    d3_qq2rr_command: str = "d3_qq2rr.x"
    d3_asr_command: str = "d3_asr.x"
    d3_sparse_command: str = "d3_sparse.x"
    asr: str = "simple"  # acoustic sum rule: "simple", "crystal", or "no"
    sparse_thr: float | None = 1e-5  # FC3 sparsification threshold (Ry/bohr^3); None = skip
    work_dir: str = "./dfpt"
    ph_timeout: int = 7200  # seconds per ph.x run
    d3q_timeout: int = 14400  # seconds per d3q.x run


@dataclass
class RelaxConfig:
    """Structural relaxation parameters."""

    calculation: str = "vc-relax"  # "relax" or "vc-relax"
    forc_conv_thr: float = 1e-4  # Ry/bohr (QE) or eV/A (VASP, mapped to EDIFFG)
    press_conv_thr: float = 0.5  # kbar (vc-relax only)
    calculator: str = "qe"        # "qe" or "vasp"
    work_dir: str = "./relax"


@dataclass
class PhononInputConfig:
    """Top-level configuration."""

    structure: StructureConfig = field(default_factory=StructureConfig)
    supercell: SupercellConfig = field(default_factory=SupercellConfig)
    qe: QEConfig = field(default_factory=QEConfig)
    vasp: VASPConfig = field(default_factory=VASPConfig)
    block_extraction: BlockExtractionConfig = field(
        default_factory=BlockExtractionConfig
    )
    quatrex_output: QuatrexOutputConfig = field(default_factory=QuatrexOutputConfig)
    thirdorder: ThirdOrderConfig = field(default_factory=ThirdOrderConfig)
    hiphive: HiphiveConfig = field(default_factory=HiphiveConfig)
    dfpt: DFPTConfig = field(default_factory=DFPTConfig)
    fc_method: str = "finite_displacement"  # "finite_displacement", "dfpt", or "hiphive"
    relax: RelaxConfig = field(default_factory=RelaxConfig)


def _unwrap_optional_dataclass(ft):
    """If ``ft`` is ``Dataclass | None`` (or ``Optional[Dataclass]``),
    return the dataclass type; otherwise return None.
    """
    import typing
    if hasattr(ft, "__dataclass_fields__"):
        return ft
    args = typing.get_args(ft)
    if not args:
        return None
    dc = [a for a in args if hasattr(a, "__dataclass_fields__")]
    return dc[0] if len(dc) == 1 else None


def _dict_to_dataclass(cls, d, *, path: str = ""):
    """Recursively convert a dict to a nested dataclass instance.

    Unknown keys are surfaced via :func:`warnings.warn` rather than silently
    dropped — they almost always indicate a misplaced field (e.g. putting
    top-level ``fc_method`` under ``relax:``) and the silent-drop behaviour
    has cost real DFT time in the past.
    """
    if not isinstance(d, dict):
        return d
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in d.items():
        if k in fieldtypes:
            ft = cls.__dataclass_fields__[k].type
            nested = _unwrap_optional_dataclass(ft)
            if nested is not None and v is not None:
                child_path = f"{path}.{k}" if path else k
                kwargs[k] = _dict_to_dataclass(nested, v, path=child_path)
            else:
                kwargs[k] = v
        else:
            here = f"{path}.{k}" if path else k
            warnings.warn(
                f"Unknown config key {here!r} (not a field of "
                f"{cls.__name__}); ignored. Check for misplaced keys.",
                stacklevel=3,
            )
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
