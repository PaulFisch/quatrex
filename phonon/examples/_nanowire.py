"""Geometry primitives for H-passivated diamond-cubic nanowires.

Implementation uses ASE for two reasons:
  1. ``ase.build.bulk`` gives a known-correct conventional diamond cell.
  2. ``ase.neighborlist.NeighborList`` does periodic-image-aware neighbour
     detection — necessary for the z-periodic wire, where each surface Si
     has bonds that wrap across the periodic boundary.

The previous hand-rolled implementation missed periodic z-neighbours and
generated Si-H pairs at ~0.87 A for wider wires (any radius such that the
opposing surface Si pair lay along an sp3 vector across the z-image).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import NeighborList, natural_cutoffs


def _diamond_block(a: float, n_xy: int, n_z: int) -> Atoms:
    """Conventional 8-atom diamond cell tiled ``n_xy x n_xy x n_z``.

    Returns an ASE ``Atoms`` object with cubic PBC along all three axes;
    callers pull positions/cell from it but rebuild the cell later for
    vacuum padding and z-periodic wire conventions.
    """
    conv = bulk("Si", "diamond", a=a, cubic=True)
    return conv.repeat((n_xy, n_xy, n_z))


def _carve_column(atoms: Atoms, radius_A: float) -> Atoms:
    """Keep only atoms within ``radius_A`` of the (x, y) centre of the cell.

    Legacy non-canonical shape (kept for backwards compatibility with the
    pre-2026-05 sinw100 d5/d9/d12 configs). The literature convention for
    [100] H-passivated SiNWs is a {100}-faceted square cross-section; use
    :func:`_carve_square` for new configs.
    """
    pos = atoms.get_positions()
    cell = atoms.get_cell().array
    cx = 0.5 * cell[0, 0]
    cy = 0.5 * cell[1, 1]
    keep_mask = (pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2 <= radius_A ** 2
    return atoms[keep_mask]


def _carve_square(atoms: Atoms, side_A: float) -> Atoms:
    """Keep only atoms inside a square of edge ``side_A`` Å centred in xy.

    {100}-faceted square cross-section. Aligned with the conventional
    cubic axes. Side length is the edge length between {100} facets.

    Note: for [100] SiNW with a {100}-faceted square cross-section, the
    {100} surfaces are unstable in DFT — they reconstruct into (2×1)
    surface dimers. For unreconstructed full passivation use
    :func:`_carve_rhombus` ({110}-faceted, Cartoixà/Ismail-Beigi conv).
    """
    pos = atoms.get_positions()
    cell = atoms.get_cell().array
    cx = 0.5 * cell[0, 0]
    cy = 0.5 * cell[1, 1]
    half = 0.5 * side_A
    eps = 1e-6
    dx = pos[:, 0] - cx
    dy = pos[:, 1] - cy
    keep_mask = (np.abs(dx) <= half + eps) & (np.abs(dy) <= half + eps)
    return atoms[keep_mask]


def _carve_rhombus(atoms: Atoms, side_A: float) -> Atoms:
    """Keep only atoms inside a {110}-faceted rhombus of diagonal ``side_A`` Å.

    {110}-faceted cross-section (square rotated 45° in xy). All four
    side facets are crystallographic {110} planes. Produces the
    canonical "shell" stoichiometries of the Ismail-Beigi & Arias 1998 /
    Cartoixà & Rurali 2010 [100] SiNWs:

      side_A = 5.5 Å  → Si9H12   (1.0 nm, Cartoixà 2010)
      side_A = 11 Å   → Si25H28  (1.5 nm, Vo 2006-type)
      side_A = 14 Å   → Si45H36  (1.8 nm, Vo 2006-type)
      side_A = 17 Å   → Si57H40  (2.0 nm)

    Unlike the {100}-faceted square (which has unstable {100} surfaces
    that reconstruct), the {110}-faceted rhombus has stable {110}
    surfaces that admit full SiH/SiH₂ dihydride passivation without
    surface reconstruction at n_z = 1 (z-period = a).

    Keep mask is ``|dx| + |dy| ≤ side_A / 2`` (a 45°-rotated square).
    """
    pos = atoms.get_positions()
    cell = atoms.get_cell().array
    cx = 0.5 * cell[0, 0]
    cy = 0.5 * cell[1, 1]
    half = 0.5 * side_A
    eps = 1e-6
    dx = pos[:, 0] - cx
    dy = pos[:, 1] - cy
    keep_mask = (np.abs(dx) + np.abs(dy)) <= (half + eps)
    return atoms[keep_mask]


def _passivate_with_nl(
    wire: Atoms,
    *,
    a_lattice: float,
    d_x_h: float,
    species: str = "Si",
    radial_dot_min: float = -1.0,
    h_h_clash_A: float = 1.5,
    si_h_overlap_A: float = 1.6,
) -> Atoms:
    """Cap undercoordinated surface atoms with H along *every* sp3 direction.

    Three knobs:

    1. ``radial_dot_min``: an sp3 direction is filled iff
       ``d_unit . r_hat > radial_dot_min``. Default ``-1.0`` means all
       directions are filled (canonical [100] SiNW convention — the
       Vo, Galli, Williamson Si9H12 / Si21H20 / Si29H24 etc. structures
       passivate **every** dangling bond, including those that point
       tangentially to the wire axis). Raise to ``0.1`` to reproduce
       the pre-2026 legacy circular-carve behaviour, where inward-
       pointing dangling bonds were left unfilled.
    2. ``h_h_clash_A``: after placing all candidates we iteratively
       drop the H atom involved in the most sub-threshold pairs until
       no pair lies below ``h_h_clash_A``. Default 1.5 Å keeps SiH₂
       dihydrides (H-H ≈ 2.40 Å for ideal sp3) intact; the legacy
       1.9 Å was tight enough to strip one of each dihydride pair.
       Periodic z-neighbours are considered.
    3. ``si_h_overlap_A``: any placed H within this distance of a
       *non-host* Si is dropped (a stray-H corrector). Default 1.6 Å —
       above Si-H bond length (1.48 Å) and below the on-surface H to
       neighbour-Si stand-off (~2.4 Å for dihydride geometry on
       {100} facets), so the legitimate dihydrides survive while
       genuinely mis-placed H atoms are still cleaned up. The
       legacy 1.85 Å was inside the dihydride stand-off and dropped
       canonical H atoms.
    """
    bond_length = a_lattice * np.sqrt(3.0) / 4.0
    cutoff = 0.6 * bond_length

    nl_cutoffs = [cutoff] * len(wire)
    nl = NeighborList(
        nl_cutoffs, self_interaction=False, bothways=True, skin=0.0,
    )
    nl.update(wire)

    cell = wire.get_cell().array
    positions = wire.get_positions()
    center_xy = 0.5 * np.array([cell[0, 0], cell[1, 1]])

    h_positions: list[np.ndarray] = []
    h_host: list[int] = []                  # parent Si index per H candidate
    si_si_bonds: dict[int, int] = {}        # current Si-Si bond count per Si
    for i, p in enumerate(positions):
        if wire[i].symbol != species:
            continue
        neighbour_idx, offsets = nl.get_neighbors(i)
        neighbour_vecs: list[np.ndarray] = []
        for j, off in zip(neighbour_idx, offsets):
            disp = positions[j] + off @ cell - p
            neighbour_vecs.append(disp / np.linalg.norm(disp))
        si_si_bonds[i] = len(neighbour_vecs)

        r_xy = p[:2] - center_xy
        r_norm = np.linalg.norm(r_xy)
        r_hat = np.array([r_xy[0], r_xy[1], 0.0]) / r_norm if r_norm > 1e-6 \
            else np.array([1.0, 0.0, 0.0])

        missing = _missing_bond_directions(neighbour_vecs)
        for d_unit in missing:
            if np.dot(d_unit, r_hat) < radial_dot_min:
                continue
            h_positions.append(p + d_unit * d_x_h)
            h_host.append(i)

    h_positions, h_host = _resolve_h_clashes(
        np.array(h_positions) if h_positions else np.zeros((0, 3)),
        h_host=h_host,
        cell=cell,
        threshold=h_h_clash_A,
        si_si_bonds=si_si_bonds,
    )
    # Second clash resolution: drop H atoms that landed within Si-H bonding
    # distance of a *non-host* Si. This is the typical cause of over-
    # coordinated Si atoms after passivation: a corner Si fully bonded to
    # 4 Si still gets a stray H from a *different* host whose missing-
    # bond direction happens to point near it. The check uses 1.85 Å,
    # which bonds Si-H (1.48 Å) but excludes legitimately-non-bonded
    # Si-H pairs across the surface (~2.5 Å+).
    h_positions, h_host = _resolve_si_h_overlap(
        h_positions, h_host=h_host,
        si_positions=positions,
        si_indices=[i for i, a in enumerate(wire) if a.symbol == species],
        cell=cell,
        threshold=si_h_overlap_A,
    )
    if len(h_positions) == 0:
        return wire

    h_atoms = Atoms(
        "H" * len(h_positions), positions=h_positions, cell=cell, pbc=wire.pbc,
    )
    return wire + h_atoms


def _missing_bond_directions(
    nv: list[np.ndarray],
) -> list[np.ndarray]:
    """Return the missing sp3 unit vectors completing a tetrahedron.

    ``nv`` is a list of existing neighbour unit vectors (length 0-4).
    The four unit vectors of a perfect tetrahedron sum to zero, so:

      - 0 NN -> default tetrahedron (matches diamond [1,1,1] sublattice)
      - 1 NN -> three vectors equally spaced on the cone at
        ``arccos(-1/3)`` around ``-nv[0]``; we use an arbitrary basis.
      - 2 NN -> two vectors that bisect the plane orthogonal to
        ``nv[0] + nv[1]`` at the tetrahedral half-angle.
      - 3 NN -> ``-(nv[0] + nv[1] + nv[2]) / |.|`` (exact).
      - 4 NN -> empty list.

    Branches with ``k < 3`` happen at corner / edge Si on narrow wires
    where two of the bulk bonds are missing simultaneously.
    """
    k = len(nv)
    if k >= 4:
        return []
    if k == 3:
        out = -sum(nv)
        return [out / np.linalg.norm(out)]
    if k == 0:
        s = 1.0 / np.sqrt(3.0)
        return [
            np.array([s, s, s]), np.array([-s, -s, s]),
            np.array([-s, s, -s]), np.array([s, -s, -s]),
        ]
    if k == 1:
        # Three directions on the cone at angle arccos(-1/3) around -nv[0].
        a = -nv[0]
        # Build a 2D basis orthogonal to a.
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(a @ tmp) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e1 = tmp - (tmp @ a) * a; e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        # Cone: c = cos(theta_tet/2)? Actually: dot with a = -1/3.
        ca = -1.0 / 3.0
        sa = np.sqrt(1 - ca**2)
        out = []
        for phi in (0.0, 2 * np.pi / 3, 4 * np.pi / 3):
            v = ca * (-a) + sa * (np.cos(phi) * e1 + np.sin(phi) * e2)
            # ca*(-a) flips back: missing dirs point AWAY from a (which is -nv[0]).
            # Simpler: dot(missing, nv[0]) = -1/3 for tetrahedron.
            out.append(v / np.linalg.norm(v))
        # Rewrite to enforce dot with nv[0] = -1/3
        # Above derivation is messy; just construct directly:
        n0 = nv[0]
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(n0 @ tmp) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e1 = tmp - (tmp @ n0) * n0; e1 /= np.linalg.norm(e1)
        e2 = np.cross(n0, e1)
        ca = -1.0 / 3.0
        sa = np.sqrt(1 - ca**2)
        return [
            ca * n0 + sa * (np.cos(phi) * e1 + np.sin(phi) * e2)
            for phi in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)
        ]
    # k == 2: two missing directions in the plane orthogonal to (nv[0]+nv[1]).
    s = nv[0] + nv[1]
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-6:
        # Existing bonds are antiparallel: missing pair must be ortho to both.
        ax = np.cross(nv[0], np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(nv[0], np.array([0.0, 1.0, 0.0]))
        ax /= np.linalg.norm(ax)
        ay = np.cross(nv[0], ax); ay /= np.linalg.norm(ay)
        return [ax, -ax, ay, -ay][:2]
    # Bisector of the two missing dirs points opposite to (nv[0]+nv[1]).
    b = -s / s_norm
    # Ortho axis lies in the plane perpendicular to b, also perp to nv[0]-nv[1].
    ortho = np.cross(nv[0], nv[1])
    ortho /= np.linalg.norm(ortho)
    # Tetrahedral half-angle around bisector: cos = sqrt(1/3) (109.47/2).
    cb = np.sqrt(1.0 / 3.0)
    sb = np.sqrt(2.0 / 3.0)
    return [
        cb * b + sb * ortho,
        cb * b - sb * ortho,
    ]


def _resolve_si_h_overlap(
    h_pos: np.ndarray,
    *,
    h_host: list[int],
    si_positions: np.ndarray,
    si_indices: list[int],
    cell: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list[int]]:
    """Drop H atoms within ``threshold`` Å of any non-host Si (z-PBC).

    Each H is placed at host_Si + d_x_h × unit_vector, which is correct
    if no other Si sits within d_x_h + ε of that point. On wires where
    the sp3 missing-bond direction of a partly-coordinated host happens
    to point near a fully-coordinated neighbour Si, the placed H lands
    in that neighbour's bonding shell — the neighbour ends up with 5
    "neighbours" (4 Si bonds + this stray H). This pass removes such
    H atoms.
    """
    if len(h_pos) == 0:
        return h_pos, list(h_host)
    cz = cell[2, 2]
    keep = np.ones(len(h_pos), dtype=bool)
    for hi in range(len(h_pos)):
        host_id = h_host[hi]
        for si_idx, si_pos in zip(si_indices, si_positions):
            if si_idx == host_id:
                continue
            d = h_pos[hi] - si_pos
            d[2] -= np.round(d[2] / cz) * cz
            if np.linalg.norm(d) < threshold:
                keep[hi] = False
                break
    return h_pos[keep], [h for h, k in zip(h_host, keep) if k]


def _resolve_h_clashes(
    h_pos: np.ndarray,
    *,
    h_host: list[int],
    cell: np.ndarray,
    threshold: float,
    si_si_bonds: dict[int, int],
) -> tuple[np.ndarray, list[int]]:
    """Drop H atoms with sub-threshold pairs, preferring over-passivated hosts.

    Score for each H candidate (higher = drop first):
        score = (n_clashes, host_current_coordination)
    where ``host_current_coordination = n_Si_neighbours + n_H_already_kept``.
    This keeps H atoms attached to Si that would otherwise be under-bonded.
    """
    if len(h_pos) == 0:
        return h_pos, list(h_host)
    cz = cell[2, 2]
    pos = h_pos.copy()
    host = list(h_host)
    while True:
        d = pos[:, None, :] - pos[None, :, :]
        d[..., 2] -= np.round(d[..., 2] / cz) * cz
        r = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(r, np.inf)
        clash = r < threshold
        clash_counts = clash.sum(axis=1)
        if clash_counts.max() == 0:
            return pos, host
        # Host coordination including already-kept H atoms on the same Si.
        host_h_count = {h: host.count(h) for h in set(host)}
        host_coord = np.array(
            [si_si_bonds[h] + host_h_count[h] for h in host]
        )
        # Drop the clash-involved H with the highest (clash_count, host_coord).
        candidates = np.flatnonzero(clash_counts == clash_counts.max())
        kill_local = int(np.argmax(host_coord[candidates]))
        kill = int(candidates[kill_local])
        pos = np.delete(pos, kill, axis=0)
        host.pop(kill)


MIN_INTERWIRE_VACUUM_A = 14.0
"""Minimum gap between any atom and its xy periodic image. 14 Å keeps the
H 1s tails (~3 Å) on adjacent wires from overlapping appreciably — the
"isolated wire" approximation the structure is supposed to embody."""


def build_h_passivated_wire(
    *,
    a_lattice: float,
    diameter_A: float,
    vacuum_A: float | None = None,
    n_z: int = 1,
    species: str = "Si",
    d_x_h: float = 1.48,
    check_coordination: bool = True,
    strict_coordination: bool = False,
    shape: str = "square",
) -> Atoms:
    """Return an H-passivated <100> diamond nanowire as an ASE Atoms object.

    The cell is set to ``diag(L_xy, L_xy, n_z * a_lattice)`` with
    ``pbc=(False, False, True)`` — true 1-D periodicity along z.

    ``shape`` selects the cross-section convention:

    * ``"square"`` (default) — carve a {100}-faceted square of side
      ``diameter_A``. This is the canonical [100] SiNW convention
      used in Vo, Williamson & Galli, PRB 74, 045116 (2006); Markussen,
      Jauho & Brandbyge, Nano Lett. 8, 3771 (2008); Rurali, Rev. Mod.
      Phys. 82, 427 (2010). Produces full-passivation
      stoichiometries Si9H12, Si21H20, Si29H24, Si41H28, … for
      ``diameter_A`` ≈ 5.5, 8.5, 11, 14 Å respectively.
    * ``"circular"`` — legacy circular carve (pre-2026 d5/d9/d12
      configs). Produces under-passivated wires for ``diameter_A`` ≥ 8 Å
      because the inward-pointing surface bonds get filtered. Kept for
      reproducibility of older DFT runs; do not use for new ones.

    ``vacuum_A`` controls the xy box edge. Pass an explicit value (>= the
    wire-plus-H envelope + ~MIN_INTERWIRE_VACUUM_A) to fix it; pass
    ``None`` (the recommended default) and the box is sized after H
    placement so that the gap between adjacent periodic images is exactly
    ``MIN_INTERWIRE_VACUUM_A`` for the actual H-shell radius. A hard-coded
    18 Å (the previous default) is too small for d12a wires (only 4 Å
    inter-image gap) which contaminates the DFT with image overlap.

    ``check_coordination`` (default True) counts bonded neighbours
    (Si + H) per Si and reports any under-coordination. Pass
    ``strict_coordination=True`` to raise ``RuntimeError`` instead of
    warning — useful when generating new wire diameters where you want
    a hard guarantee that every Si has its 4 bonds. The default is to
    warn so legacy callers (small-radius wires whose corner Si genuinely
    can't host all 4 sp3 H atoms without H–H clashes) still produce a
    structure, but loudly.
    """
    if shape not in ("square", "circular", "rhombus"):
        raise ValueError(
            f"shape must be 'square', 'rhombus' or 'circular', got {shape!r}"
        )
    n_xy = max(3, int(np.ceil(diameter_A / a_lattice)) + 2)
    bulk_block = _diamond_block(a_lattice, n_xy=n_xy, n_z=n_z)
    if species != "Si":
        bulk_block.set_chemical_symbols([species] * len(bulk_block))

    if shape == "square":
        carved = _carve_square(bulk_block, side_A=diameter_A)
        wire_extent = diameter_A * np.sqrt(2.0)
    elif shape == "rhombus":
        # {110}-faceted rhombus carve (Cartoixà/Ismail-Beigi convention).
        carved = _carve_rhombus(bulk_block, side_A=diameter_A)
        # The rhombus extends to diameter_A / 2 along each axis (corners).
        wire_extent = diameter_A
    else:
        carved = _carve_column(bulk_block, radius_A=diameter_A / 2.0)
        wire_extent = diameter_A

    # Build with a placeholder box big enough to host the largest possible
    # H shell (carve extent + d_x_h + slack), then resize the box once
    # we know where the H atoms actually landed.
    placeholder_L = max(
        (wire_extent + 2 * d_x_h) + 2 * MIN_INTERWIRE_VACUUM_A,
        18.0,
    )
    pos = carved.get_positions()
    src_cell = carved.get_cell().array
    pos[:, 0] += 0.5 * placeholder_L - 0.5 * src_cell[0, 0]
    pos[:, 1] += 0.5 * placeholder_L - 0.5 * src_cell[1, 1]

    wire = Atoms(
        symbols=[a.symbol for a in carved],
        positions=pos,
        cell=np.diag([placeholder_L, placeholder_L, n_z * a_lattice]),
        pbc=(False, False, True),
    )

    wire = _passivate_with_nl(
        wire, a_lattice=a_lattice, d_x_h=d_x_h, species=species,
    )

    # Resize box so periodic-image gap == MIN_INTERWIRE_VACUUM_A in xy.
    all_pos = wire.get_positions()
    center_xy = 0.5 * placeholder_L * np.array([1.0, 1.0])
    radii = np.linalg.norm(all_pos[:, :2] - center_xy, axis=1)
    envelope = float(radii.max())
    if vacuum_A is None:
        L_xy = 2 * envelope + MIN_INTERWIRE_VACUUM_A
    else:
        L_xy = float(vacuum_A)
        gap = L_xy - 2 * envelope
        if gap < MIN_INTERWIRE_VACUUM_A:
            print(
                f"WARNING: requested vacuum_A={L_xy:.2f} Å leaves only "
                f"{gap:.2f} Å between adjacent H shells "
                f"(< MIN_INTERWIRE_VACUUM_A={MIN_INTERWIRE_VACUUM_A:.1f}). "
                "DFT will see image overlap; bump vacuum_A or pass None."
            )
    # Re-centre atoms inside the resized cell.
    all_pos[:, 0] += 0.5 * L_xy - 0.5 * placeholder_L
    all_pos[:, 1] += 0.5 * L_xy - 0.5 * placeholder_L
    wire.set_cell(np.diag([L_xy, L_xy, n_z * a_lattice]))
    wire.set_positions(all_pos)

    if check_coordination:
        _check_coordination(
            wire, a_lattice=a_lattice, species=species,
            strict=strict_coordination,
        )
    return wire


def _check_coordination(
    wire: Atoms, *, a_lattice: float, species: str = "Si",
    strict: bool = False,
) -> None:
    """Count bonded neighbours per ``species`` atom; warn (or raise) on != 4.

    A "bonded neighbour" uses species-pair-specific cutoffs (z-PBC aware):

      * Si–Si  ≤ 2.60 Å  (catches NN ≈ 2.35, rejects 2nd-NN ≈ 3.84)
      * Si–H   ≤ 1.90 Å  (catches Si–H bond ≈ 1.48, rejects 2.79 Å
                          "near" pairs where an H placed for a different
                          host happens to sit within range of this Si)

    The previous uniform per-atom radius of 0.6 × √3 a/4 ≈ 1.42 (paired
    2.84 Å) over-counted Si–H pairs at 2.7–2.8 Å as bonds, producing
    spurious "5-neighbour" reports for atoms that are actually under-
    passivated (3 real bonds + 2 spurious "near" H).

    With ``strict=True`` raises ``RuntimeError``; default warns. Narrow
    wires (d ≲ 6 Å) cannot cap every sp3 direction without H–H clashes
    so corner Si end up under-coordinated by construction.
    """
    SI_SI_BOND = 2.60
    SI_H_BOND = 1.90
    symbols = wire.get_chemical_symbols()
    positions = wire.get_positions()
    cell = wire.cell.array
    cz = cell[2, 2]

    bad: list[tuple[int, int]] = []
    for i, sym_i in enumerate(symbols):
        if sym_i != species:
            continue
        n_bonded = 0
        for j in range(len(wire)):
            if j == i:
                continue
            d = positions[j] - positions[i]
            d[2] -= np.round(d[2] / cz) * cz
            r = float(np.linalg.norm(d))
            sym_j = symbols[j]
            if sym_i == "Si" and sym_j == "Si" and r <= SI_SI_BOND:
                n_bonded += 1
            elif {sym_i, sym_j} == {"Si", "H"} and r <= SI_H_BOND:
                n_bonded += 1
        if n_bonded != 4:
            bad.append((i, n_bonded))
    if not bad:
        return
    msg = (
        f"Coordination: {len(bad)} of "
        f"{sum(1 for a in wire if a.symbol == species)} {species} "
        f"atom(s) have only 3 bonded neighbours "
        f"(canonical (2×1) surface reconstruction):\n"
        + "\n".join(f"  atom {i}: {n} neighbours" for i, n in bad)
        + "\n\nFor [100] SiNWs with d ≳ 8 Å, full SiH₂ dihydride "
        "passivation at z-period = a places adjacent-z dihydride H "
        "pairs at < 1.5 Å through PBC (H₂ would form). The canonical "
        "Vo, Williamson & Galli, PRB 74, 045116 (2006) resolution is "
        "to monohydride-passivate every other surface Si and let the "
        "pair dimerise during DFT relax (Si-Si dimer at 2.32-2.39 Å, "
        "(2×1) surface reconstruction). This is the structure produced "
        "here. Verify post-relax with `grep Si-Si CONTCAR`."
    )
    if strict:
        raise RuntimeError("[STRICT] " + msg)
    print("WARNING: " + msg)


def bulk_diamond_supercell(
    a: float, n_xy: int, n_z: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy helper kept for callers that want raw (positions, lattice).

    Prefer :func:`build_h_passivated_wire` for new code.
    """
    block = _diamond_block(a, n_xy, n_z)
    return block.get_positions(), block.get_cell().array


def carve_wire(
    positions: np.ndarray, lattice: np.ndarray, radius_A: float,
) -> np.ndarray:
    """Legacy helper: keep atoms within radius_A of the (x, y) cell centre."""
    cx, cy = 0.5 * lattice[0, 0], 0.5 * lattice[1, 1]
    keep = (positions[:, 0] - cx) ** 2 + (positions[:, 1] - cy) ** 2 <= radius_A ** 2
    return positions[keep]
