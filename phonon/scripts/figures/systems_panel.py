"""The six systems the results chapter uses, drawn from their own configs
(fig:res_systems).

Every panel is built from the committed structure section of the
force-constant config that produced that system's force constants, so the
figure cannot drift from the inputs:

    bulk Si    ase.build.bulk (a from phonon/configs/si_primitive/hiphive_big.yaml)
    SiNW d5a   phonon/configs/sinw/sinw100_d5a_vasp_sc4.yaml
    SiNW d11a  phonon/configs/sinw/sinw100_d11a_vasp_sc4.yaml
    CNT (3,3)  phonon/configs/cnt/cnt33_vasp.yaml
    2H-MoS2    phonon/configs/mos2/mos2_bulk_vasp.gen.b.yaml   (relaxed cell)
    SrTiO3     phonon/configs/perovskite/srtio3_small_vasp.yaml

Run:  python phonon/scripts/figures/systems_panel.py
Figure -> document/fig/transport_sweeps/systems_panel.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.build import bulk
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase.utils import rotate
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "phonon")):
    if p not in sys.path:
        sys.path.insert(0, p)
from phonon.studies import style  # noqa: E402

FIGDIR = ROOT / "document/fig/transport_sweeps"
CFG = ROOT / "phonon/configs"

#: species whose contacts are ionic rather than covalent: drawing them as
#: bonds turns the perovskite panel into a mesh.
NO_BONDS = {"Sr"}


def atoms_from_cfg(relpath: str) -> Atoms:
    """The `structure:` section of a phonon_inputs config as an Atoms."""
    d = yaml.safe_load((CFG / relpath).read_text())["structure"]
    return Atoms(symbols=d["symbols"],
                 cell=np.array(d["lattice"], dtype=float),
                 scaled_positions=np.array(d["scaled_positions"], dtype=float),
                 pbc=True)


def close_cell(atoms: Atoms, tol: float = 1e-3) -> Atoms:
    """Add the periodic images of atoms sitting on a cell face.

    A cell drawn with only its stored basis has one corner atom and half its
    faces; repeating the boundary atoms gives the textbook picture of the
    cell (eight corners, six face centres) without changing the basis.
    """
    out = atoms.copy()
    frac = atoms.get_scaled_positions()
    for i, f in enumerate(frac):
        edges = [k for k in range(3) if f[k] < tol]
        for mask in range(1, 1 << len(edges)):
            shift = np.zeros(3)
            for bit, k in enumerate(edges):
                if mask >> bit & 1:
                    shift[k] = 1.0
            out += Atoms(numbers=[atoms.numbers[i]],
                         positions=[(f + shift) @ atoms.cell],
                         cell=atoms.cell, pbc=True)
    return out


def draw(ax, atoms, rotation="", repeat=(1, 1, 1), radius_scale=0.42,
         bond_scale=1.25, span=None, cell=False, close=False):
    """Ball-and-stick projection of ``atoms * repeat`` onto ``ax``.

    Bonds are drawn between pairs closer than ``bond_scale`` times the sum
    of their covalent radii, within the replicated cluster only, so no bond
    is drawn across the periodic boundary of the drawn region.
    """
    drawn = close_cell(atoms) if close else atoms
    drawn = drawn * repeat
    rot = rotate(rotation)
    pos = drawn.get_positions() @ rot
    num = drawn.get_atomic_numbers()
    rad = covalent_radii[num]
    sym = np.array(drawn.get_chemical_symbols())

    bondable = ~np.isin(sym, list(NO_BONDS))
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    cut = bond_scale * (rad[:, None] + rad[None, :])
    cut[~bondable, :] = 0.0
    cut[:, ~bondable] = 0.0
    ia, ib = np.where(np.triu(dist < cut, k=1))
    ax.add_collection(LineCollection(
        [[(pos[a, 0], pos[a, 1]), (pos[b, 0], pos[b, 1])]
         for a, b in zip(ia, ib)],
        colors="0.45", linewidths=1.0, zorder=1))

    if cell:
        origin = np.zeros(3)
        vecs = np.asarray(atoms.cell) @ rot
        edges = []
        for i in range(3):
            j, k = [m for m in range(3) if m != i]
            for shift in (origin, vecs[j], vecs[k], vecs[j] + vecs[k]):
                edges.append([tuple(shift[:2]), tuple((shift + vecs[i])[:2])])
        ax.add_collection(LineCollection(edges, colors="0.6", linewidths=0.6,
                                         linestyles=(0, (3, 2)), zorder=0))

    for k in np.argsort(pos[:, 2]):
        ax.add_patch(Circle((pos[k, 0], pos[k, 1]), radius_scale * rad[k],
                            facecolor=jmol_colors[num[k]], edgecolor="0.15",
                            linewidth=0.6, zorder=2 + 1e-4 * pos[k, 2]))

    ax.set_aspect("equal")
    ax.set_axis_off()
    cx = 0.5 * (pos[:, 0].max() + pos[:, 0].min())
    cy = 0.5 * (pos[:, 1].max() + pos[:, 1].min())
    if span is None:
        span = max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1])) + 2.0
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)


def wire_envelope(atoms: Atoms) -> float:
    """Radius of the hydrogen envelope of a [100] wire, in AA."""
    pos = atoms.get_positions()
    axis = pos[:, :2] - pos[:, :2].mean(axis=0)
    h = np.array(atoms.get_chemical_symbols()) == "H"
    return float(np.linalg.norm(axis[h], axis=1).max())


def main():
    si = bulk("Si", "diamond", a=5.468, cubic=True)
    d5a = atoms_from_cfg("sinw/sinw100_d5a_vasp_sc4.yaml")
    d11a = atoms_from_cfg("sinw/sinw100_d11a_vasp_sc4.yaml")
    cnt = atoms_from_cfg("cnt/cnt33_vasp.yaml")
    mos2 = atoms_from_cfg("mos2/mos2_bulk_vasp.gen.b.yaml")
    sto = atoms_from_cfg("perovskite/srtio3_small_vasp.yaml")

    fig, axes = style.doc_figure(ncols=3, nrows=2, frac=1.0, aspect=0.60)

    draw(axes[0, 0], si, rotation="-72x,12y", span=9.6, cell=True, close=True)
    axes[0, 0].set_title("bulk Si", pad=2)

    # the two wires share a span, so the panels compare their cross-sections
    draw(axes[0, 1], d5a, repeat=(1, 1, 2), span=16.0)
    axes[0, 1].set_title("SiNW d5a", pad=2)
    draw(axes[0, 2], d11a, repeat=(1, 1, 2), span=16.0)
    axes[0, 2].set_title("SiNW d11a", pad=2)

    draw(axes[1, 0], cnt, repeat=(1, 1, 4), rotation="-78x,10y", span=12.5)
    axes[1, 0].set_title("CNT (3,3)", pad=2)

    draw(axes[1, 1], mos2, repeat=(3, 1, 1), rotation="-90x", span=14.5)
    axes[1, 1].set_title("2H-MoS$_2$", pad=2)

    draw(axes[1, 2], sto, rotation="-72x,12y", span=7.6, radius_scale=0.34,
         cell=True, close=True)
    axes[1, 2].set_title("SrTiO$_3$", pad=2)

    # mark the van der Waals gap the cross-plane transport runs through
    gap_ax = axes[1, 1]
    zs = np.sort(mos2.get_positions()[
        np.array(mos2.get_chemical_symbols()) == "S", 2])
    gap = float(zs[2] - zs[1])
    gap_ax.text(0.5, 0.5, "van der Waals gap", transform=gap_ax.transAxes,
                ha="center", va="center", fontsize=7.5, color="0.25")

    style.panel_labels(axes)
    style.save(fig, "systems_panel", directory=FIGDIR)

    print("=" * 64)
    print("systems drawn (atoms per transport cell, degrees of freedom)")
    print("=" * 64)
    for name, at in [("bulk Si (primitive)", bulk("Si", "diamond", a=5.468)),
                     ("SiNW d5a", d5a), ("SiNW d11a", d11a),
                     ("CNT (3,3)", cnt), ("2H-MoS2", mos2),
                     ("SrTiO3", sto)]:
        a, b, c = at.cell.lengths()
        print(f"{name:20s} {at.get_chemical_formula():10s} "
              f"n={len(at):3d}  3n={3*len(at):3d}  "
              f"cell = {a:.3f} x {b:.3f} x {c:.3f} AA")
    print(f"d5a  H-envelope radius {wire_envelope(d5a):.2f} AA "
          f"(diameter {2*wire_envelope(d5a):.2f})")
    print(f"d11a H-envelope radius {wire_envelope(d11a):.2f} AA "
          f"(diameter {2*wire_envelope(d11a):.2f})")
    print(f"MoS2 interlayer S-S separation along c: {gap:.3f} AA")


if __name__ == "__main__":
    main()
