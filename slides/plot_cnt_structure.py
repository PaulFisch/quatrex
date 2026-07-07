"""Render the (3,3) armchair CNT geometry for the status slides.

Uses the exact unit cell from phonon/configs/cnt/cnt33_vasp.yaml (12 atoms,
T = sqrt(3)*1.42 A along z), replicates it along the tube axis, draws C-C
bonds, and writes fig/cnt_structure.pdf with a side view + an axial view.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- (3,3) unit cell from configs/cnt/cnt33_vasp.yaml ---------------------
LATTICE = np.array([
    [14.0000, 0.0, 0.0],
    [0.0, 14.0000, 0.0],
    [0.0, 0.0, 2.4595],
])
SCALED = np.array([
    [0.645286, 0.500000, 0.000000],
    [0.611295, 0.593388, 0.000000],
    [0.572643, 0.625821, 0.500000],
    [0.474771, 0.643079, 0.500000],
    [0.427357, 0.625821, 0.000000],
    [0.363476, 0.549691, 0.000000],
    [0.354714, 0.500000, 0.500000],
    [0.388705, 0.406612, 0.500000],
    [0.427357, 0.374179, 0.000000],
    [0.525229, 0.356921, 0.000000],
    [0.572643, 0.374179, 0.500000],
    [0.636524, 0.450309, 0.500000],
])

N_CELLS = 7          # cells stacked along z for the picture
BOND_MAX = 1.6       # A, C-C bond cutoff


def build() -> np.ndarray:
    cz = LATTICE[2, 2]
    cell = SCALED @ LATTICE
    pos = np.vstack([cell + np.array([0.0, 0.0, k * cz]) for k in range(N_CELLS)])
    # centre the tube on the x-y origin
    pos[:, 0] -= 7.0
    pos[:, 1] -= 7.0
    return pos


def bonds(pos: np.ndarray) -> list[tuple[int, int]]:
    out = []
    for i in range(len(pos)):
        d = np.linalg.norm(pos[i + 1:] - pos[i], axis=1)
        for off, dist in enumerate(d):
            if dist < BOND_MAX:
                out.append((i, i + 1 + off))
    return out


def main() -> None:
    pos = build()
    bs = bonds(pos)
    z = pos[:, 2]

    # Compact single-row inset: short side view + small axial ring.
    fig = plt.figure(figsize=(3.7, 1.25))

    # --- side view (tube axis horizontal) --------------------------------
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for i, j in bs:
        ax.plot(pos[[i, j], 2], pos[[i, j], 0], pos[[i, j], 1],
                color="0.4", lw=0.8, zorder=1)
    ax.scatter(pos[:, 2], pos[:, 0], pos[:, 1], s=14, c="#1f6feb",
               edgecolors="k", linewidths=0.2, depthshade=True, zorder=2)
    ax.view_init(elev=16, azim=-72)
    ax.set_box_aspect((N_CELLS * 2.4595, 8, 8))
    ax.set_axis_off()

    # --- axial view (look down z): the (3,3) ring ------------------------
    ax2 = fig.add_subplot(1, 2, 2)
    one = pos[z < (1.05 * 2.4595)]
    ax2.scatter(one[:, 0], one[:, 1], s=26, c="#1f6feb",
                edgecolors="k", linewidths=0.3, zorder=3)
    for i in range(len(one)):
        for j in range(i + 1, len(one)):
            if np.linalg.norm(one[i, :2] - one[j, :2]) < 1.55:
                ax2.plot(one[[i, j], 0], one[[i, j], 1], color="0.4", lw=0.9, zorder=2)
    th = np.linspace(0, 2 * np.pi, 200)
    r = np.linalg.norm(one[:, :2], axis=1).mean()
    ax2.plot(r * np.cos(th), r * np.sin(th), ls="--", color="0.75", lw=0.6)
    ax2.set_aspect("equal"); ax2.set_axis_off()

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0)
    out = Path(__file__).resolve().parent / "fig" / "cnt_structure.pdf"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
