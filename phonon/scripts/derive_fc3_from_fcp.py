"""Derive ``fc3.hdf5`` from a committed hiphive ``fcp.fcp``.

The hiphive reap normally writes both, but only the ``fcp.fcp`` is committed
for the SrTiO3 fits, so any consumer that resolves an ``fc3.hdf5`` under a
``work_dir`` (``finite_analysis.loader._resolve_fc3_path``) fails. This
rebuilds the missing file from the potential, which is the same operation the
reap performs (``ForceConstantPotential.get_force_constants`` on the ideal
supercell, then the fc2/fc3 arrays to HDF5).

Two compatibility notes, both load-bearing:

* The committed ``.fcp`` files were pickled by a numpy whose
  ``_frombuffer`` takes a fifth argument -- the axis permutation used for
  arrays stored in ``'K'`` (keep-layout) order. numpy 2.1's version takes
  four and raises TypeError. :func:`_patch_frombuffer` supplies the
  five-argument form; its inverse is checked against a synthetic
  non-contiguous array before any file is read.
* The supercell is built with the pipeline's own ``_build_supercell`` +
  ``structure_to_ase``, which delegate to phonopy, so the atom ordering of
  the emitted arrays matches both the original fit and the finite-difference
  pipeline atom for atom.

Usage:  python phonon/scripts/derive_fc3_from_fcp.py [--force] [CONFIG.yaml ...]
        (default: both perovskite SrTiO3 configs)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_CONFIGS = [
    "phonon/configs/perovskite/srtio3_small_vasp.yaml",
    "phonon/configs/perovskite/srtio3_vasp.yaml",
]


def _patch_frombuffer() -> None:
    """Teach numpy 2.1's unpickler the five-argument ``_frombuffer``."""
    from numpy._core import numeric

    orig = numeric._frombuffer

    def _frombuffer(buf, dtype, shape, order, axis_order=None):
        if axis_order is None:
            return orig(buf, dtype, shape, order)
        phys = tuple(shape[i] for i in axis_order)
        arr = np.frombuffer(buf, dtype=dtype).reshape(phys)
        return arr.transpose(np.argsort(axis_order))

    # verify the inverse before trusting it on real data
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 4, 5))
    perm = (0, 2, 1)
    back = _frombuffer(bytearray(a.transpose(perm).copy().tobytes()),
                       np.dtype("float64"), a.shape, "K", perm)
    if not np.array_equal(back, a):
        raise RuntimeError("the 'K'-order _frombuffer inverse is wrong")
    numeric._frombuffer = _frombuffer


def derive(config_path: Path, force: bool = False) -> Path | None:
    import h5py
    from hiphive import ForceConstantPotential

    from phonon_inputs.config import load_config
    from phonon_inputs.hiphive_fc3 import _build_supercell
    from phonon_inputs.structure import load_structure, structure_to_ase

    cfg = load_config(str(config_path))
    hh = cfg.hiphive
    work_dir = Path(hh.work_dir)
    if not work_dir.is_absolute():
        work_dir = (config_path.parent / work_dir).resolve()

    out = work_dir / "fc3.hdf5"
    if out.exists() and not force:
        print(f"{out} exists; use --force to rebuild")
        return out
    fcp_path = work_dir / "fcp.fcp"
    if not fcp_path.exists():
        print(f"no fcp.fcp under {work_dir}; skipping")
        return None

    # cubic Pm-3m: every atom sits on a special position, so the YAML
    # geometry IS the relaxed one and the (absent) relax output is not needed.
    cell = load_structure(cfg.structure)
    atoms_ideal = structure_to_ase(_build_supercell(cell, tuple(hh.supercell)))

    fcp = ForceConstantPotential.read(str(fcp_path))
    fcs = fcp.get_force_constants(atoms_ideal)
    fc2 = fcs.get_fc_array(order=2)
    fc3 = fcs.get_fc_array(order=3)

    n = len(atoms_ideal)
    if fc2.shape != (n, n, 3, 3) or fc3.shape != (n, n, n, 3, 3, 3):
        raise RuntimeError(f"unexpected shapes {fc2.shape}, {fc3.shape} for {n} atoms")

    # cheap physical gates -- a silently wrong unpickle would fail these
    asr2 = np.abs(fc2.sum(axis=1)).max()
    sym2 = np.abs(fc2 - fc2.transpose(1, 0, 3, 2)).max()
    asr3 = np.abs(fc3.sum(axis=2)).max()
    print(f"  {config_path.name}: n_super={n}  "
          f"|fc2|max={np.abs(fc2).max():.4e} eV/A^2  "
          f"|fc3|max={np.abs(fc3).max():.4e} eV/A^3")
    print(f"    FC2 ASR {asr2:.3e}   FC2 pair symmetry {sym2:.3e}   FC3 ASR {asr3:.3e}")

    with h5py.File(out, "w") as f:
        f.create_dataset("fc2", data=fc2, compression="gzip")
        f.create_dataset("fc3", data=fc3, compression="gzip")
    print(f"    wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    _patch_frombuffer()
    for c in args.configs:
        p = Path(c)
        if not p.is_absolute():
            p = ROOT / p
        derive(p, force=args.force)


if __name__ == "__main__":
    main()
