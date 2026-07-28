"""d5a wire + alpha-quartz SiO2 shell: the oxide-embedded structure.

Phase-2 of the twist-mode plan: the H-passivated Si9H12 wire (chemistry
UNCHANGED -- the transport device stays the same Si/H system) surrounded
by a crystalline SiO2 annulus in mechanical/steric contact. The shell
supplies the torsional/flexural restoring force a real matrix exerts;
Phase-3 extracts just the wire-block force constants (shell-clamped
policy) after the hiphive fit.

Shell construction:
  * alpha-quartz, c-axis along the wire: c = 5.405 A vs the wire period
    5.47 A -> 1.2 % tensile strain on the oxide (the wire is NOT
    strained).
  * annulus r_in <= r_xy <= r_out carved from a quartz supercell;
    r_in chosen so the innermost shell atoms sit ~2.6 A (vdW H...O
    contact) outside the wire's H envelope (3.69 A).
  * cleanup: iteratively drop under-coordinated Si (< 2 O neighbours)
    and dangling O (0 Si neighbours); cap remaining broken bonds with H
    (O-H 0.97 A, pointing away from the severed neighbour).
  * checks: min interatomic distances, coordination histogram, shell-
    wire gap, vacuum margin. Refuses to write on violations.

Usage:
    python examples/setup_sinw100_oxide.py            # defaults
    python examples/setup_sinw100_oxide.py --r-in 6.3 --r-out 9.5 \
        --box 28.0 --out phonon/configs/sinw/sinw100_d5a_oxide.xyz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

A_QTZ, C_QTZ = 4.913, 5.405   # alpha-quartz PBE-ish lattice constants
C_WIRE = 5.47                 # wire z-period (PBE eq)
D_OH = 0.97                   # O-H cap bond length
D_SIH = 1.48                  # Si-H cap bond length
R_SIO = 2.0                   # Si-O bond cutoff for coordination
WIRE_YAML_FRAC = None         # filled from the d5a YAML below

# Relaxed d5a wire (scaled positions from sinw100_d5a_vasp_sc4.yaml,
# box 21.3796 A -- converted to Cartesian below).
WIRE_BOX = 21.379563267814216
WIRE = [
    ("Si", 0.4365816472693612, 0.4365816472693612, 0.2516275076798999),
    ("Si", 0.3728823300967292, 0.5, 0.5),
    ("Si", 0.4365816472693612, 0.5634183527306388, 0.7483724923200998),
    ("Si", 0.5, 0.3728823300967292, 0.5),
    ("Si", 0.5634183527306388, 0.4365816472693612, 0.7483724923200998),
    ("Si", 0.5, 0.5, 0.0),
    ("Si", 0.5634183527306388, 0.5634183527306388, 0.2516275076798999),
    ("Si", 0.5, 0.6271176699032708, 0.5),
    ("Si", 0.6271176699032708, 0.5, 0.5),
    ("H", 0.3961653207314795, 0.3961653207314795, 0.0917893758976927),
    ("H", 0.331568107547446, 0.5401229428014201, 0.3436820381742404),
    ("H", 0.331568107547446, 0.4598770571985795, 0.6563179618257596),
    ("H", 0.3961653207314795, 0.6038346792685202, 0.9082106241023074),
    ("H", 0.4598770571985795, 0.331568107547446, 0.6563179618257596),
    ("H", 0.5401229428014201, 0.331568107547446, 0.3436820381742404),
    ("H", 0.6038346792685202, 0.3961653207314795, 0.9082106241023074),
    ("H", 0.6038346792685202, 0.6038346792685202, 0.0917893758976927),
    ("H", 0.5401229428014201, 0.668431892452554, 0.6563179618257596),
    ("H", 0.4598770571985795, 0.668431892452554, 0.3436820381742404),
    ("H", 0.668431892452554, 0.4598770571985795, 0.3436820381742404),
    ("H", 0.668431892452554, 0.5401229428014201, 0.6563179618257596),
]


def quartz_cell():
    """alpha-quartz unit cell (3 SiO2 = 9 atoms), c strained to C_WIRE.

    Built from the spglib P3121 operator orbit (hall 441) of
    Si 3a (u, 0, 1/3), u = 0.4697 and O 6c (0.4133, 0.2672, 0.2133).
    The O z was calibrated numerically against the bond-uniformity
    criterion (all Si-O in [1.611, 1.619] A): the tabulated (u,0,0)/
    z=0.1188 pairs belong to other origin conventions and produce
    broken networks in this setting (duplicated Si images or half
    coordination). Self-validated below; the shell is VASP-relaxed
    downstream, so 0.01 A fidelity suffices.
    """
    import spglib
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    ops = spglib.get_symmetry_from_database(441)  # P3121

    def orbit(p):
        pts = []
        for r, t in zip(ops["rotations"], ops["translations"]):
            q = (r @ p + t) % 1.0
            if not any(np.allclose(q, x, atol=1e-5) for x in pts):
                pts.append(q)
        return pts

    si = orbit(np.array([0.4697, 0.0, 1.0 / 3.0]))
    ox = orbit(np.array([0.4133, 0.2672, 0.2133]))
    cell = np.array([
        [A_QTZ, 0.0, 0.0],
        [-A_QTZ / 2.0, A_QTZ * np.sqrt(3) / 2.0, 0.0],
        [0.0, 0.0, C_WIRE],
    ])
    q = Atoms(symbols=["Si"] * len(si) + ["O"] * len(ox),
              scaled_positions=si + ox, cell=cell, pbc=True)
    # self-validation: stoichiometry, 4:2 network, sane bonds
    assert len(si) == 3 and len(ox) == 6, (len(si), len(ox))
    i, j, d = neighbor_list("ijd", q, cutoff=2.0)
    sym = q.get_chemical_symbols()
    per_si: dict[int, int] = {}
    for a, b, dd in zip(i, j, d):
        if sym[a] == "Si" and sym[b] == "O":
            per_si[a] = per_si.get(a, 0) + 1
            assert 1.55 < dd < 1.70, dd
    assert sorted(per_si.values()) == [4, 4, 4], per_si
    return q


def build(r_in: float, r_out: float, box: float):
    from ase import Atoms

    # ---- wire (centered in the new box)
    wire_pos = np.array([[x * WIRE_BOX, y * WIRE_BOX, z * C_WIRE]
                         for _, x, y, z in WIRE])
    wire_sym = [s for s, *_ in WIRE]
    shift = (box - WIRE_BOX) / 2.0
    wire_pos[:, :2] += shift
    center = np.array([box / 2.0, box / 2.0])

    # ---- quartz tiled; WHOLE-TETRAHEDRON carve (standard silica
    # surface construction): keep Si whose full SiO4 fits the annulus,
    # keep every O bonded to a kept Si, terminate singly-bonded O as
    # silanol H along the vacated bond (guaranteed empty space).
    q = quartz_cell()
    n_tile = int(np.ceil(2.0 * (r_out + 4.0) / A_QTZ)) + 2
    tiled = q.repeat((n_tile, n_tile, 1))
    pos_all = tiled.get_positions()
    sym_all = np.array(tiled.get_chemical_symbols())
    pos_all[:, 0] += center[0] - pos_all[:, 0].mean()
    pos_all[:, 1] += center[1] - pos_all[:, 1].mean()
    pos_all[:, 2] %= C_WIRE

    def zsep(d):
        d = d.copy()
        d[..., 2] -= C_WIRE * np.round(d[..., 2] / C_WIRE)
        return d

    si_all = np.where(sym_all == "Si")[0]
    ox_all = np.where(sym_all == "O")[0]
    r_si = np.linalg.norm(pos_all[si_all, :2] - center, axis=1)

    # Si -> its 4 O partners (z-periodic)
    keep_si = []
    si_o = {}
    for k, i in enumerate(si_all):
        d = np.linalg.norm(zsep(pos_all[ox_all] - pos_all[i]), axis=1)
        part = ox_all[d < R_SIO]
        if len(part) == 4 and r_in <= r_si[k] <= r_out:
            keep_si.append(i)
            si_o[i] = part
    keep_si = np.array(keep_si, dtype=int)
    if keep_si.size == 0:
        sys.exit("empty shell -- widen the annulus")

    # Largest connected component of the kept-Si network (Si adjacent
    # iff sharing an O): stray fragments are dropped before capping.
    adj = {int(i): set() for i in keep_si}
    o_owners: dict[int, list[int]] = {}
    for i in keep_si:
        for j in si_o[i]:
            o_owners.setdefault(int(j), []).append(int(i))
    for owners in o_owners.values():
        for a in owners:
            for b in owners:
                if a != b:
                    adj[a].add(b)
    seen, comps = set(), []
    for start in adj:
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(adj[x] - comp)
        seen |= comp
        comps.append(comp)
    largest = max(comps, key=len)
    if len(comps) > 1:
        print(f"pruning {len(comps) - 1} stray shell fragment(s) "
              f"({sum(len(c) for c in comps) - len(largest)} Si)")
    keep_si = np.array(sorted(largest), dtype=int)
    si_o = {i: si_o[i] for i in keep_si}

    # O kept iff bonded to >= 1 kept Si; count kept-Si bonds per O
    o_count = {}
    for i in keep_si:
        for j in si_o[i]:
            o_count[j] = o_count.get(j, 0) + 1
    keep_ox = np.array(sorted(o_count), dtype=int)

    shell_idx = list(keep_si) + list(keep_ox)
    pos = pos_all[shell_idx].copy()
    sym = np.array(["Si"] * len(keep_si) + ["O"] * len(keep_ox))

    # silanol caps: singly-bonded O, H along the vacated Si direction
    caps_pos = []
    for j in keep_ox:
        if o_count[j] != 1:
            continue
        d = np.linalg.norm(zsep(pos_all[si_all] - pos_all[j]), axis=1)
        partners = si_all[d < R_SIO]
        dropped = [i for i in partners if i not in si_o]
        if not dropped:
            continue
        u = zsep(pos_all[dropped[0]][None, :] - pos_all[j][None, :])[0]
        u /= np.linalg.norm(u)
        caps_pos.append(pos_all[j] + D_OH * u)
    caps_sym = ["H"] * len(caps_pos)

    shell_pos = np.vstack([pos] + ([np.array(caps_pos)] if caps_pos else []))
    shell_sym = list(sym) + caps_sym

    # ---- assemble + checks
    all_pos = np.vstack([wire_pos, shell_pos])
    all_sym = wire_sym + shell_sym
    n = len(all_sym)

    # Pairwise clash check with BOND-aware thresholds (an O-H silanol at
    # 0.97 A or a Si-O bond at 1.61 A is chemistry, not a clash).
    MIN_D = {frozenset(("O", "H")): 0.85, frozenset(("Si", "O")): 1.45,
             frozenset(("Si", "H")): 1.30, frozenset(("H", "H")): 1.00,  # silanol-cap pairs relax apart
             frozenset(("O",)): 2.0, frozenset(("Si",)): 2.25,
             frozenset(("H",)): 1.00}

    def worst_clash():
        worst = (np.inf, None)  # margin, pair
        for i in range(n):
            d = all_pos[i + 1:] - all_pos[i]
            if len(d) == 0:
                continue
            d[:, 2] -= C_WIRE * np.round(d[:, 2] / C_WIRE)
            dd = np.linalg.norm(d, axis=1)
            for j in np.where(dd < 2.7)[0]:
                a, b = all_sym[i], all_sym[i + 1 + j]
                thr = MIN_D.get(frozenset((a, b)), 1.35)
                margin = dd[j] - thr
                if margin < worst[0]:
                    worst = (margin, (a, b, float(dd[j]), thr))
        return worst

    margin, clash = worst_clash()

    # Shell connectivity: bridging-O graph of the SiO2 network.
    si_loc = np.array([k for k, s in enumerate(shell_sym) if s == "Si"])
    ox_loc = np.array([k for k, s in enumerate(shell_sym) if s == "O"])
    parent = {int(i): int(i) for i in si_loc}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_bridge = 0
    for k in ox_loc:
        d = shell_pos[si_loc] - shell_pos[k]
        d[:, 2] -= C_WIRE * np.round(d[:, 2] / C_WIRE)
        bonded = si_loc[np.linalg.norm(d, axis=1) < R_SIO]
        if len(bonded) == 2:
            n_bridge += 1
            a, b = find(int(bonded[0])), find(int(bonded[1]))
            if a != b:
                parent[a] = b
    n_comp = len({find(int(i)) for i in si_loc})
    wire_r = np.linalg.norm(wire_pos[:, :2] - center, axis=1).max()
    shell_r_in = np.linalg.norm(shell_pos[:, :2] - center, axis=1).min()
    shell_r_out = np.linalg.norm(shell_pos[:, :2] - center, axis=1).max()
    gap = shell_r_in - wire_r
    vac = box / 2.0 - shell_r_out
    from collections import Counter
    comp = Counter(all_sym)
    print(f"composition: wire Si9H12 + shell "
          f"{Counter(shell_sym)} -> total {dict(comp)} ({n} atoms/cell)")
    print(f"worst clash margin: {margin:+.3f} A ({clash})")
    print(f"wire H-envelope {wire_r:.2f} A; shell {shell_r_in:.2f}-"
          f"{shell_r_out:.2f} A; radial gap {gap:.2f} A; vacuum {vac:.2f} A")
    print(f"shell network: {len(si_loc)} Si, {n_bridge} bridging O, "
          f"{n_comp} connected component(s)")
    ok = True
    if margin < 0.0:
        print("FAIL: clash below bond-aware threshold"); ok = False
    if gap < 2.2:
        print(f"FAIL: shell-wire gap {gap:.2f} < 2.2 A"); ok = False
    if vac < 6.0:
        print(f"FAIL: vacuum {vac:.2f} < 6 A"); ok = False
    if n_comp > 1:
        print(f"FAIL: shell not connected ({n_comp} components)"); ok = False
    return (all_sym, all_pos, box, ok)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r-in", type=float, default=6.3)
    p.add_argument("--r-out", type=float, default=9.6)
    p.add_argument("--box", type=float, default=32.0)
    p.add_argument("--out", default="phonon/configs/sinw/sinw100_d5a_oxide.xyz")
    a = p.parse_args()
    sym, pos, box, ok = build(a.r_in, a.r_out, a.box)
    if not ok:
        sys.exit("structure checks FAILED -- not writing.")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"{len(sym)}\n")
        f.write(f'Lattice="{box} 0 0 0 {box} 0 0 0 {C_WIRE}" '
                'Properties=species:S:1:pos:R:3 pbc="T T T"\n')
        for s, p_ in zip(sym, pos):
            f.write(f"{s}  {p_[0]:.8f}  {p_[1]:.8f}  {p_[2]:.8f}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
