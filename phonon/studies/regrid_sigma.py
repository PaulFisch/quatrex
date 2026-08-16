"""Interpolate a saved self-energy snapshot onto a new frequency grid.

Usage:
  python phonon/studies/regrid_sigma.py --sigma old_sigma.npz       --old-grid old/phonon_energies.npy --new-grid new/phonon_energies.npy       --out new_sigma.npz [--scale 1.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_grid(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        return np.asarray(np.load(path)["energies"], dtype=float)
    return np.asarray(np.load(path), dtype=float).ravel()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sigma", type=Path, required=True,
                   help="QX_SAVE_SIGMA snapshot (.npz)")
    p.add_argument("--old-grid", type=Path, required=True,
                   help="grid the snapshot lives on (.npy, or run.npz "
                        "with an 'energies' key)")
    p.add_argument("--new-grid", type=Path, required=True,
                   help="target grid (.npy or run.npz)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="extra scale on the loaded Sigma (cf. "
                        "QX_SIGMA_SCALE) [1.0]")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    old = _load_grid(a.old_grid)
    new = _load_grid(a.new_grid)
    snap = np.load(a.sigma)

    out = {}
    for key in ("sigma_lesser", "sigma_greater", "sigma_retarded"):
        sig = np.asarray(snap[key])
        if sig.shape[0] != old.size:
            raise SystemExit(
                f"{key}: frequency axis {sig.shape[0]} does not match the "
                f"old grid ({old.size} pts); wrong --old-grid?")
        flat = sig.reshape(sig.shape[0], -1)
        res = np.empty((new.size, flat.shape[1]), dtype=flat.dtype)
        for j in range(flat.shape[1]):
            res[:, j] = (np.interp(new, old, flat[:, j].real, left=0.0,
                                   right=0.0)
                         + 1j * np.interp(new, old, flat[:, j].imag,
                                          left=0.0, right=0.0))
        out[key] = a.scale * res.reshape((new.size,) + sig.shape[1:])

    np.savez(a.out, **out)
    print(f"wrote {a.out}: {old.size} -> {new.size} frequency points "
          f"(scale {a.scale}); use QX_SIGMA_INIT={a.out} on the new grid.")


if __name__ == "__main__":
    main()
