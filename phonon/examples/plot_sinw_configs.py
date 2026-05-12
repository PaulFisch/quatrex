"""Render each SiNW config as a multi-panel PDF (end-on + side view).

Reads every ``sinw*.yaml`` under ``phonon/configs/sinw/`` and writes one
``<config-stem>.pdf`` per file to ``document/src/fig/sinw/`` (created if
missing). Each PDF has two panels:

  - left: end-on view (looking down the wire z-axis); shows the
    cross-section: Si core + H passivation shell.
  - right: side view (looking along x); shows ``n_z`` periodic copies
    so the wire-period is visible.

Usage:
    python phonon/examples/plot_sinw_configs.py
    python phonon/examples/plot_sinw_configs.py --n-z 3   # tile 3 periods
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from ase import Atoms
from ase.visualize.plot import plot_atoms


def _atoms_from_yaml(cfg_path: Path) -> Atoms:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    s = cfg["structure"]
    cell = np.array(s["lattice"])
    pos = np.array(s["scaled_positions"]) @ cell
    return Atoms(
        symbols=list(s["symbols"]),
        positions=pos, cell=cell, pbc=(False, False, True),
    )


def _figure_title(cfg_path: Path) -> str:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    s = cfg["structure"]
    pos = np.array(s["scaled_positions"]) @ np.array(s["lattice"])
    syms = s["symbols"]
    L = np.array(s["lattice"])
    cx, cy = 0.5 * L[0, 0], 0.5 * L[1, 1]
    si_idx = [i for i, t in enumerate(syms) if t == "Si"]
    si_r = np.hypot(pos[si_idx, 0] - cx, pos[si_idx, 1] - cy)
    n_si = len(si_idx)
    n_h = len(syms) - n_si
    return (
        f"{cfg_path.stem}\n"
        f"{len(syms)} atoms ({n_si} Si + {n_h} H), "
        f"Si-core diameter = {2 * si_r.max():.2f} A, "
        f"period c = {L[2, 2]:.2f} A"
    )


def plot_config(cfg_path: Path, out_pdf: Path, *, n_z: int = 2) -> None:
    atoms = _atoms_from_yaml(cfg_path)
    side = atoms.repeat((1, 1, n_z))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    plot_atoms(atoms, axes[0], radii=0.45, rotation="0x,0y,0z")
    axes[0].set_title(f"End-on  (axis = z)")
    axes[0].set_xlabel("x [A]")
    axes[0].set_ylabel("y [A]")
    axes[0].set_aspect("equal")

    plot_atoms(side, axes[1], radii=0.45, rotation="-90x,0y,0z")
    axes[1].set_title(f"Side ({n_z} periods)")
    axes[1].set_xlabel("x [A]")
    axes[1].set_ylabel("z [A]")
    axes[1].set_aspect("equal")

    fig.suptitle(_figure_title(cfg_path), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", dpi=150)
    plt.close(fig)
    print(f"  {cfg_path.name}  ->  {out_pdf}")


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_in = repo / "phonon" / "configs" / "sinw"
    default_out = repo / "document" / "src" / "fig" / "sinw"

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--in-dir", type=Path, default=default_in)
    p.add_argument("--out-dir", type=Path, default=default_out)
    p.add_argument("--n-z", type=int, default=2,
                   help="Number of unit-cell periods to tile in side view")
    args = p.parse_args()

    configs = sorted(args.in_dir.glob("sinw*.yaml"))
    if not configs:
        raise SystemExit(f"No sinw*.yaml found in {args.in_dir}")
    print(f"Rendering {len(configs)} configs to {args.out_dir}")
    for cfg in configs:
        plot_config(cfg, args.out_dir / f"{cfg.stem}.pdf", n_z=args.n_z)


if __name__ == "__main__":
    main()
