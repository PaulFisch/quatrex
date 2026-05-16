"""Load a finite-structure phonon system into a uniform :class:`SystemBundle`.

This module is the single source of truth for how the rest of the
``finite_analysis`` package reads a system. A bundle carries:

  * the phonopy object (with primitive + supercell + masses),
  * FC2 in eV/Å² shape ``(n_super, n_super, 3, 3)`` (phono3py convention),
  * FC3 in eV/Å³ either compact ``(nat_prim, n_super, n_super, 3, 3, 3)`` or
    full ``(n_super, n_super, n_super, 3, 3, 3)`` (phono3py convention),
  * the corresponding mass-weighted, dense ``(n_dof, dim_sc, dim_sc)`` FC3
    target tensor used by the compression and SSE machinery (THz²·√(amu·Å²)),
  * the slab decomposition along the transport direction (``block_sizes``)
    derived by clustering supercell atoms in z,
  * a small metadata blob (paths, supercell matrix, source).

Loading is non-destructive: nothing is recomputed if an FC reap directory is
already on disk. The bundle holds **arrays only** — no MPI, no quatrex
dependency. A second loader, :func:`load_quatrex_blocks`, takes a bundle and
projects FC3 onto the block-tridiagonal dictionary the quatrex SSE expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phonon_inputs.config import config_from_dict
from phonon_inputs.fc3_compression import FC3Target, build_fc3_target
from phonon_inputs.structure import load_structure


@dataclass
class SystemBundle:
    """Everything downstream analyses need to know about one finite system."""

    name: str
    phonon: Phonopy
    fc2: np.ndarray
    """Phonopy convention, ``(n_super, n_super, 3, 3)``, units eV/Å²."""

    fc3_raw: np.ndarray
    """Phono3py convention, compact or full, units eV/Å³."""

    fc3_target: FC3Target
    """Mass-weighted target produced by :func:`build_fc3_target`. THz²·√(amu·Å²).

    The ``T`` field is ``(n_dof, dim_sc, dim_sc)``; ``T_lifted`` is the
    S3-symmetric ``(dim_sc, dim_sc, dim_sc)`` lift used by all the
    compression methods and by the SSE bubble.
    """

    masses: np.ndarray
    """Per-atom masses in amu, length ``n_super``."""

    sc_positions: np.ndarray
    """Cartesian positions of the supercell atoms in Å, ``(n_super, 3)``."""

    sc_cell: np.ndarray
    """Supercell lattice vectors in Å, ``(3, 3)`` row-vector convention."""

    block_sizes: np.ndarray
    """Per-slab DOF counts along transport direction, ``sum == 3 * n_super``."""

    atom_perm: np.ndarray
    """Permutation that sorts supercell atoms by transport-axis fractional
    coordinate. Apply to ``masses``, ``sc_positions``, and (with index
    duplication for the 3-DOF expansion) to FC2/FC3 if you need
    ``block_sizes`` to address contiguous slices in the FC arrays."""

    transport_axis: int
    """0, 1, or 2 — which Cartesian axis carries transport."""

    meta: dict[str, Any] = field(default_factory=dict)
    """Free-form: paths, supercell matrix, source ('phono3py'/'hiphive'/'dfpt')."""

    @property
    def n_super(self) -> int:
        return int(self.phonon.supercell.masses.shape[0])

    @property
    def n_dof(self) -> int:
        return 3 * self.n_super

    @property
    def n_slabs(self) -> int:
        return int(self.block_sizes.shape[0])


# --------------------------------------------------------------------------- #
# YAML / FC reap loading                                                      #
# --------------------------------------------------------------------------- #


def _build_phonopy(
    cfg,
    supercell_override: list | None = None,
    primitive_override: PhonopyAtoms | None = None,
) -> Phonopy:
    """Build a :class:`Phonopy` object.

    Resolution order for the primitive cell:
      1. ``primitive_override`` if supplied (typically the primitive from
         ``hiphive_meta.json``, which is the structure hiphive actually fit
         on — the relaxed CONTCAR positions, not the YAML's pre-relax draft).
      2. ``load_structure(cfg.structure)`` (the YAML structure block) as
         the fallback for non-hiphive reaps (phono3py, DFPT).

    Resolution order for the supercell matrix:
      1. ``supercell_override`` (typically ``hiphive_meta.json``'s ``supercell``).
      2. ``cfg.thirdorder.supercell`` (the YAML's thirdorder block).
    """
    cell = primitive_override if primitive_override is not None else load_structure(cfg.structure)
    sc = supercell_override if supercell_override is not None else cfg.thirdorder.supercell
    sc_matrix = np.array(sc)
    if sc_matrix.ndim == 1:
        sc_matrix = np.diag(sc_matrix)
    return Phonopy(
        cell,
        supercell_matrix=sc_matrix,
        primitive_matrix=np.eye(3),
    )


def _hiphive_meta_at(fc3_path: Path) -> dict | None:
    """Read ``hiphive_meta.json`` next to ``fc3.hdf5`` if present.

    Hiphive reaps drop this file alongside ``fc3.hdf5`` at sow time, holding
    the primitive (symbols, cell, scaled_positions) and supercell layout
    the FC arrays were fit on. Using it as the source of truth for the
    analysis-time primitive avoids a class of bugs where the YAML primitive
    has drifted from the relaxed positions hiphive actually saw (see
    `scratch/imag_audit/INVESTIGATION_NOTES.md` for the d9a case).
    """
    import json
    meta_path = fc3_path.with_name("hiphive_meta.json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _supercell_from_reap_metadata(fc3_path: Path) -> list | None:
    """Look for ``hiphive_meta.json`` next to ``fc3.hdf5`` and return its
    ``supercell`` field. Returns None if the file is absent.

    Kept for backwards compatibility; new code should call
    :func:`_hiphive_meta_at` for the full meta blob.
    """
    meta = _hiphive_meta_at(fc3_path)
    return None if meta is None else meta.get("supercell")


def _primitive_from_reap_metadata(fc3_path: Path) -> PhonopyAtoms | None:
    """Reconstruct the primitive cell from ``hiphive_meta.json`` if present.

    Returns a :class:`PhonopyAtoms` built from the meta's ``primitive``
    block (symbols, cell, scaled_positions). Used by :func:`load_system` so
    that the analysis-time phonopy is identical to the cell hiphive fit on.
    """
    meta = _hiphive_meta_at(fc3_path)
    if meta is None or "primitive" not in meta:
        return None
    p = meta["primitive"]
    try:
        return PhonopyAtoms(
            symbols=list(p["symbols"]),
            cell=np.asarray(p["cell"]),
            scaled_positions=np.asarray(p["scaled_positions"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _read_fc_hdf5(fc3_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read FC2/FC3 from a phono3py-style ``fc3.hdf5``."""
    with h5py.File(fc3_path, "r") as f:
        fc2 = np.asarray(f["fc2"][:], dtype=np.float64)
        fc3 = np.asarray(f["fc3"][:], dtype=np.float64)
    return fc2, fc3


def _resolve_fc3_path(cfg, config_dir: Path) -> Path:
    """Pick an existing fc3.hdf5 produced by either the FD or hiphive flow.

    Preference order:
      1. ``thirdorder.work_dir/fc3.hdf5`` (FD via phono3py + symfc),
      2. ``hiphive.work_dir/fc3.hdf5``    (hiphive),
      3. ``dfpt.work_dir/fc3.hdf5``       (D3Q DFPT).
    """
    candidates: list[tuple[str, Path]] = []
    tdir = getattr(cfg.thirdorder, "work_dir", None)
    if tdir is not None:
        candidates.append(("phono3py", Path(tdir).expanduser()))
    hh = getattr(cfg, "hiphive", None)
    if hh is not None and getattr(hh, "work_dir", None) is not None:
        candidates.append(("hiphive", Path(hh.work_dir).expanduser()))
    dfpt = getattr(cfg, "dfpt", None)
    if dfpt is not None and getattr(dfpt, "work_dir", None) is not None:
        candidates.append(("dfpt", Path(dfpt.work_dir).expanduser()))

    for source, raw in candidates:
        wd = raw if raw.is_absolute() else (config_dir / raw).resolve()
        for cand in (wd / "fc3.hdf5", wd / "results" / "fc3.hdf5"):
            if cand.exists():
                return cand
    raise FileNotFoundError(
        f"No fc3.hdf5 found under any work_dir from {[c[1] for c in candidates]}."
    )


# --------------------------------------------------------------------------- #
# Slab decomposition                                                          #
# --------------------------------------------------------------------------- #


def cluster_into_slabs(
    sc_positions: np.ndarray,
    sc_cell: np.ndarray,
    transport_axis: int,
    n_slabs_hint: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Group supercell atoms into contiguous slabs along ``transport_axis``.

    Atoms are sorted by their fractional coordinate along the transport axis
    and partitioned into ``n_slabs_hint`` equal-population bins. If
    ``n_slabs_hint`` is None, the largest histogram bin count for which
    every bin is non-empty is chosen.

    Returns
    -------
    block_sizes : ndarray ``(n_slabs,)`` — DOFs per slab (= 3 × atom count).
    atom_perm : ndarray ``(n_super,)`` — atom permutation that maps the
        original supercell ordering into z-sorted order. Callers that need
        ``block_sizes`` to address contiguous slices of the FC arrays should
        also apply this permutation to all atom-indexed objects.
    """
    inv = np.linalg.inv(sc_cell)
    frac = sc_positions @ inv
    z = frac[:, transport_axis] % 1.0
    n_atoms = sc_positions.shape[0]

    if n_slabs_hint is None:
        for cand in range(min(n_atoms, 16), 0, -1):
            counts, _ = np.histogram(z, bins=cand, range=(0.0, 1.0))
            if (counts > 0).all():
                n_slabs_hint = cand
                break
        else:  # pragma: no cover
            n_slabs_hint = 1

    atom_perm = np.argsort(z, kind="stable")

    sizes_atoms = np.full(n_slabs_hint, n_atoms // n_slabs_hint, dtype=np.int64)
    sizes_atoms[: n_atoms % n_slabs_hint] += 1
    block_sizes = (3 * sizes_atoms).astype(np.int64)

    if (block_sizes <= 0).any():
        raise RuntimeError(
            f"Slab decomposition produced an empty slab: {block_sizes}"
        )
    return block_sizes, atom_perm


# --------------------------------------------------------------------------- #
# Main entry point                                                            #
# --------------------------------------------------------------------------- #


def load_system(
    config_path: str | Path,
    *,
    name: str | None = None,
    transport_axis: int = 2,
    n_slabs_hint: int | None = None,
    fc3_path_override: str | Path | None = None,
    validate: bool = True,
) -> SystemBundle:
    """Load one finite-structure system into a :class:`SystemBundle`.

    Parameters
    ----------
    config_path : path-like
        ``phonon_inputs`` YAML config (the same file that drives the pipeline).
    name : str, optional
        Display / output-directory name. Defaults to the YAML filename stem.
    transport_axis : {0, 1, 2}
        Cartesian axis along which the device is partitioned into slabs.
    n_slabs_hint : int, optional
        Target slab count. If omitted, an auto-clusterer picks the largest
        partition with non-empty slabs.
    fc3_path_override : path-like, optional
        Explicit ``fc3.hdf5``. Useful when several reaps live next to each other.
    validate : bool
        Run :func:`parameter_validation.validate_config` and emit a
        ``UserWarning`` for severity ≥ "warn". Set False in test fixtures
        that intentionally exercise edge configurations.
    """
    config_path = Path(config_path).expanduser().resolve()
    name = name or config_path.stem

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    cfg = config_from_dict(raw)

    if validate:
        import warnings as _warnings

        from .parameter_validation import validate_config as _vc

        for r in _vc(cfg):
            if r.severity != "info":
                _warnings.warn(
                    f"[{r.severity}] {r.check.key}: {r.check.message} "
                    f"(actual={r.actual!r}). {r.check.recommendation}",
                    stacklevel=2,
                )

    fc3_path = (
        Path(fc3_path_override).expanduser().resolve()
        if fc3_path_override is not None
        else _resolve_fc3_path(cfg, config_path.parent)
    )
    sc_override = _supercell_from_reap_metadata(fc3_path)
    # Prefer the primitive cell hiphive itself fit on (saved in
    # hiphive_meta.json). The YAML's primitive can drift away if the YAML
    # was regenerated after the relax/sow ran on the cluster — that drift
    # causes wrong q-phase factors at non-Γ q and silently scrambles the
    # dispersion. Fall back to the YAML primitive only when there's no
    # meta sitting next to fc3.hdf5 (phono3py / DFPT reaps).
    primitive_override = _primitive_from_reap_metadata(fc3_path)
    phonon = _build_phonopy(
        cfg,
        supercell_override=sc_override,
        primitive_override=primitive_override,
    )
    fc2, fc3 = _read_fc_hdf5(fc3_path)

    n_super_expected = phonon.supercell.masses.shape[0]
    if fc3.shape[0] != n_super_expected and fc3.shape[1] != n_super_expected:
        raise ValueError(
            f"FC3 axis sizes {fc3.shape[:3]} do not match the supercell "
            f"({n_super_expected} atoms) implied by the YAML / reap metadata. "
            "Check that the supercell matrix matches the reap."
        )
    phonon.force_constants = fc2

    fc3_target = build_fc3_target(fc3, phonon)
    masses = np.asarray(phonon.supercell.masses, dtype=np.float64)
    sc_positions = np.asarray(phonon.supercell.positions, dtype=np.float64)
    sc_cell = np.asarray(phonon.supercell.cell, dtype=np.float64)

    block_sizes, atom_perm = cluster_into_slabs(
        sc_positions, sc_cell, transport_axis, n_slabs_hint
    )

    meta = {
        "config_path": str(config_path),
        "fc3_path": str(fc3_path),
        "supercell_matrix": np.array(phonon.supercell_matrix).tolist(),
        "transport_axis": transport_axis,
        "n_super": int(masses.shape[0]),
        "nat_prim": int(phonon.primitive.masses.shape[0]),
        "z_already_sorted": bool(np.array_equal(atom_perm, np.arange(len(atom_perm)))),
    }

    return SystemBundle(
        name=name,
        phonon=phonon,
        fc2=fc2,
        fc3_raw=fc3,
        fc3_target=fc3_target,
        masses=masses,
        sc_positions=sc_positions,
        sc_cell=sc_cell,
        block_sizes=block_sizes,
        atom_perm=atom_perm,
        transport_axis=transport_axis,
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# Quatrex projection                                                          #
# --------------------------------------------------------------------------- #


def load_quatrex_blocks(
    bundle: SystemBundle,
    *,
    truncation_warn: float = 0.01,
    asr_project: bool = False,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Project the bundle's mass-weighted FC3 onto NN-tridiagonal blocks.

    Permutes the dense FC3 into z-sorted DOF order so ``bundle.block_sizes``
    addresses contiguous slices, then calls
    :func:`quatrex.phonon.fc3_loader.fc3_to_phi_blocks`. The returned dict
    is keyed by ``(I, J, K)`` with ``|I-J|, |I-K|, |J-K| <= 1``. A warning
    is emitted if the dropped Frobenius weight exceeds ``truncation_warn``.

    Parameters
    ----------
    asr_project : bool, default False
        If True, project the lifted FC3 onto the ASR null-space along
        legs 2 and 3 *before* the block-tridiagonal cut. Use when the
        upstream FC3 fit (e.g. hiphive least-squares) does not enforce
        ASR — without this, Σ(ω→0) carries a spurious Drude-like weight.
        See :data:`constants.ASR_REL_RESIDUAL_WARN`.
    """
    from quatrex.phonon.fc3_loader import fc3_to_phi_blocks

    from ._utils import expand_atom_perm_to_dofs

    dof_perm = expand_atom_perm_to_dofs(bundle.atom_perm)
    phi_dense = bundle.fc3_target.T_lifted[np.ix_(dof_perm, dof_perm, dof_perm)]
    if asr_project:
        from phonon_inputs.fc3_compression import asr_project_factor
        n_super = bundle.n_super
        # Project along the second leg, then the third. (Leg 1 is the
        # primitive-DOF index — not subject to the supercell ASR.)
        phi_dense = asr_project_factor(phi_dense, n_super, axis=1)
        phi_dense = asr_project_factor(phi_dense, n_super, axis=2)
    return fc3_to_phi_blocks(
        phi_dense, bundle.block_sizes, nn_only=True,
        truncation_warn=truncation_warn,
    )
