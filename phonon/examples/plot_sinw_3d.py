"""Render each SiNW config as a 3D PDF.

For every ``sinw*.yaml`` under ``phonon/configs/sinw/`` writes one
``<config-stem>_3d.pdf`` to ``document/src/fig/sinw/``.

The figure has two 3D panels rendered with ``mpl_toolkits.mplot3d``:

  - left: isometric view (elev=20, azim=-60)
  - right: top-down + tilted (elev=70, azim=-60)

Atoms are drawn as size-scaled scatter spheres (Si grey, H pale). Bonds
are drawn as line segments using ``ase.neighborlist.NeighborList`` with
``natural_cutoffs`` scaled by 1.2 — periodic-image aware along z.

Usage:
    python phonon/examples/plot_sinw_3d.py
    python phonon/examples/plot_sinw_3d.py --n-z 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from ase import Atoms
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase.neighborlist import NeighborList, natural_cutoffs
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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


def _build_bond_segments(atoms: Atoms, scale: float = 1.2) -> np.ndarray:
    """Return shape ``(n_bonds, 2, 3)`` of bond endpoints in Cartesian."""
    cutoffs = [scale * c for c in natural_cutoffs(atoms)]
    nl = NeighborList(cutoffs, self_interaction=False, bothways=False, skin=0.0)
    nl.update(atoms)
    cell = atoms.get_cell().array
    pos = atoms.get_positions()
    segs: list[np.ndarray] = []
    for i in range(len(atoms)):
        neighbours, offsets = nl.get_neighbors(i)
        for j, off in zip(neighbours, offsets):
            p2 = pos[j] + off @ cell
            segs.append(np.array([pos[i], p2]))
    return np.array(segs) if segs else np.zeros((0, 2, 3))


def _equal_aspect_3d(ax, pos: np.ndarray) -> None:
    """Match the three axis ranges so spheres look spherical."""
    extents = np.ptp(pos, axis=0)
    span = max(extents.max(), 1.0)
    mid = pos.mean(axis=0)
    for set_lim, m in zip(
        (ax.set_xlim, ax.set_ylim, ax.set_zlim), mid,
    ):
        set_lim(m - span / 2, m + span / 2)


def _draw_panel(ax, atoms: Atoms, *, elev: float, azim: float) -> None:
    pos = atoms.get_positions()
    Zs = atoms.get_atomic_numbers()
    colors = jmol_colors[Zs]
    radii = np.array([covalent_radii[z] for z in Zs])
    sizes = (radii * 60.0) ** 2

    segs = _build_bond_segments(atoms, scale=1.2)
    if len(segs):
        lc = Line3DCollection(
            segs, colors=(0.3, 0.3, 0.3, 0.7), linewidths=1.2,
        )
        ax.add_collection3d(lc)

    ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        s=sizes, c=colors, edgecolors="k", linewidths=0.4, depthshade=True,
    )
    ax.set_xlabel("x [A]"); ax.set_ylabel("y [A]"); ax.set_zlabel("z [A]")
    ax.view_init(elev=elev, azim=azim)
    _equal_aspect_3d(ax, pos)
    ax.set_box_aspect((1, 1, 1))


def _figure_title(cfg_path: Path) -> str:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    s = cfg["structure"]
    L = np.array(s["lattice"])
    pos = np.array(s["scaled_positions"]) @ L
    syms = s["symbols"]
    cx, cy = 0.5 * L[0, 0], 0.5 * L[1, 1]
    si = [i for i, t in enumerate(syms) if t == "Si"]
    si_r = np.hypot(pos[si, 0] - cx, pos[si, 1] - cy)
    n_si = len(si); n_h = len(syms) - n_si
    return (
        f"{cfg_path.stem}   |   {len(syms)} atoms ({n_si} Si + {n_h} H)"
        f"   |   Si-core = {2 * si_r.max():.2f} A   |   c = {L[2, 2]:.2f} A"
    )


def plot_config_3d(cfg_path: Path, out_pdf: Path, *, n_z: int = 2) -> None:
    atoms = _atoms_from_yaml(cfg_path).repeat((1, 1, n_z))

    fig = plt.figure(figsize=(11, 5.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    _draw_panel(ax1, atoms, elev=20, azim=-60)
    ax1.set_title("Isometric")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _draw_panel(ax2, atoms, elev=70, azim=-60)
    ax2.set_title("Top-down (tilted)")

    fig.suptitle(_figure_title(cfg_path), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
                   help="Periodic copies tiled along z before rendering")
    args = p.parse_args()

    configs = sorted(args.in_dir.glob("sinw*.yaml"))
    if not configs:
        raise SystemExit(f"No sinw*.yaml found in {args.in_dir}")
    print(f"Rendering {len(configs)} configs to {args.out_dir}")
    for cfg in configs:
        plot_config_3d(cfg, args.out_dir / f"{cfg.stem}_3d.pdf", n_z=args.n_z)


if __name__ == "__main__":
    main()
