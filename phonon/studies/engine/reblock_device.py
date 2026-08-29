"""Re-block a transversely-periodic film device: N primitive transport cells,
C of them per BTD block.

``phonon/docs/bubble_positivity.md``. Putting C transport cells in one
    python phonon/studies/engine/reblock_device.py         --src cluster/mos2f3 --cells 6 --per-block 2 --out cluster/mos2f6x2
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "phonon"))


# ---------------------------------------------------------------------------
# source device
# ---------------------------------------------------------------------------
def _read_mat(path: Path, tdir: int):
    """{(transverse key): {transport shift: D}} from a production .mat."""
    from scipy.io import loadmat

    raw = loadmat(str(path))
    out: dict[tuple, dict[int, np.ndarray]] = {}
    for key, val in raw.items():
        if key.startswith("__"):
            continue
        idx = [int(x) for x in re.findall(r"-?\d+", key)]
        perp = tuple(v for i, v in enumerate(idx) if i != tdir)
        out.setdefault(perp, {})[idx[tdir]] = np.asarray(val)
    return out


def _read_fc3(path: Path):
    import h5py

    with h5py.File(str(path), "r") as f:
        sizes = np.asarray(f["meta/block_sizes"], int)
        blocks = {}
        for name in f["fc3_blocks"]:
            ds = f["fc3_blocks"][name]
            blocks[(int(ds.attrs["I"]), int(ds.attrs["J"]),
                    int(ds.attrs["K"]))] = np.asarray(ds)
    return blocks, sizes


def _assert_slab_replicas(blocks: dict, n_src: int, tag: str) -> None:
    """Every vertex block must depend only on the slab OFFSETS, not on
    the absolute slab index -- otherwise replication along transport is
    not exact and this tool must not be used."""
    scale = max((float(np.abs(v).max()) for v in blocks.values()), default=0.0)
    if scale == 0.0:
        return
    worst = 0.0
    for (I, J, K), v in blocks.items():
        for s in range(1, n_src):
            shifted = (I + s, J + s, K + s)
            if max(shifted) >= n_src:
                continue
            other = blocks.get(shifted)
            if other is None:
                worst = max(worst, float(np.abs(v).max()))
                continue
            worst = max(worst, float(np.abs(other - v).max()))
    if worst > 1e-12 * scale:
        raise SystemExit(
            f"{tag}: slabs are NOT translationally equivalent (worst "
            f"relative difference {worst / scale:.2e}). Replication would "
            "not be exact; build the longer device with build_inputs.py "
            "instead.")


def _replicate(blocks: dict, n_src: int, n_tgt: int) -> dict:
    """Extend a slab-translationally-invariant block dict to n_tgt slabs."""
    out = {}
    for (I, J, K), v in blocks.items():
        for s in range(-n_src, n_tgt):
            t = (I + s, J + s, K + s)
            if min(t) < 0 or max(t) >= n_tgt:
                continue
            out[t] = v
    return out


def _merge(blocks: dict, n_prim: int, c: int, nd: int) -> dict:
    """Merge c consecutive primitive slabs into one block.

    Exact re-partition: the (u, v, w) sub-position of the merged block
    (I, J, K) is the primitive triple (cI+u, cJ+v, cK+w), zero where the
    primitive vertex has no entry.
    """
    nb, ndn = n_prim // c, c * nd
    out = {}
    for I in range(nb):
        for J in range(nb):
            for K in range(nb):
                blk = None
                for u in range(c):
                    for v in range(c):
                        for w in range(c):
                            src = blocks.get((c * I + u, c * J + v, c * K + w))
                            if src is None:
                                continue
                            if blk is None:
                                blk = np.zeros((ndn, ndn, ndn),
                                               dtype=np.asarray(src).dtype)
                            blk[u * nd:(u + 1) * nd, v * nd:(v + 1) * nd,
                                w * nd:(w + 1) * nd] = src
                if blk is not None:
                    out[(I, J, K)] = blk
    return out


def _write_fc3_blocks(phi_blocks: dict, block_sizes, path: Path,
                      units: str = "THz^2") -> None:
    """Write fc3_blocks.hdf5 in the production schema.

    Inlined rather than imported from phonon_inputs.quatrex_writer: that
    module pulls in phonopy at import time, which is not installed in
    the daint venv, and this tool must run wherever the device inputs
    live. Schema mirrors quatrex_writer.write_fc3_blocks exactly (the
    parity of the two writers is pinned by
    tests/quatrex/phonon/test_reblock_device.py).
    """
    import h5py

    block_sizes = np.asarray(block_sizes, dtype=np.int64)
    keys = sorted(phi_blocks)
    with h5py.File(str(path), "w") as f:
        meta = f.create_group("meta")
        meta.create_dataset("block_sizes", data=block_sizes)
        meta.attrs["units"] = units
        meta.create_dataset("keys", data=np.asarray(keys, dtype=np.int64))
        grp = f.create_group("fc3_blocks")
        for (I, J, K) in keys:
            blk = np.asarray(phi_blocks[(I, J, K)])
            ds = grp.create_dataset(f"{I}_{J}_{K}",
                                    data=blk.astype(np.complex128))
            ds.attrs["I"], ds.attrs["J"], ds.attrs["K"] = I, J, K
            ds.attrs["b_I"], ds.attrs["b_J"], ds.attrs["b_K"] = blk.shape


def _dense_vertex(blocks: dict, nb: int, ndn: int) -> np.ndarray:
    n = nb * ndn
    phi = np.zeros((n, n, n), complex)
    for (I, J, K), v in blocks.items():
        phi[I * ndn:(I + 1) * ndn, J * ndn:(J + 1) * ndn,
            K * ndn:(K + 1) * ndn] = v
    return phi


def _superblock(dmap: dict[int, np.ndarray], c: int, nd: int, shift: int):
    """The c-cell unit-cell block at transport shift `shift`.

    Mirrors src/quatrex/device/inputs.py:_get_transport_block --
    sub-block (i, j) is the primitive coupling at cell offset
    j - i + shift*c, zero beyond the primitive range.
    """
    out = np.zeros((c * nd, c * nd), complex)
    for i in range(c):
        for j in range(c):
            d = dmap.get(j - i + shift * c)
            if d is not None:
                out[i * nd:(i + 1) * nd, j * nd:(j + 1) * nd] = d
    return out


def _dense_fc2(mats: dict, perp, n_cells: int, nd: int) -> np.ndarray:
    """Dense n_cells-block FC2 at one transverse key, primitive blocking."""
    n = n_cells * nd
    out = np.zeros((n, n), complex)
    for i in range(n_cells):
        for j in range(n_cells):
            d = mats[perp].get(j - i)
            if d is not None:
                out[i * nd:(i + 1) * nd, j * nd:(j + 1) * nd] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="source device dir")
    ap.add_argument("--cells", type=int, required=True,
                    help="total primitive transport cells in the target")
    ap.add_argument("--per-block", type=int, required=True,
                    help="primitive cells per BTD block")
    ap.add_argument("--out", required=True)
    ap.add_argument("--decomposed-path", default=None,
                    help="optional primitive VertexFactors archive to lift "
                    "exactly into the new blocking")
    ap.add_argument("--preserve-primitive-vertices", action="store_true",
                    help="keep Gamma/q-folded FC3 and optional factors in "
                    "primitive-cell blocking; emit an SSE microblock config "
                    "instead of merging/lifting the vertices")
    ap.add_argument("--micro-g-band", type=int, default=3,
                    help="primitive Green band written with "
                    "--preserve-primitive-vertices (default: 3)")
    ap.add_argument("--factor-support", choices=["dense", "stored"],
                    default="dense",
                    help="with preserved primitive factors, restrict their "
                    "transport-offset pairs to the support present in the "
                    "dense FC3 (default), or retain the archive metadata")
    ap.add_argument("--skip-qfold", action="store_true",
                    help="do not materialise the dense q-folded vertex")
    ap.add_argument("--tdir", default=None, help="transport axis (default: "
                    "read from the source config)")
    a = ap.parse_args()

    src = ROOT / a.src if not Path(a.src).is_absolute() else Path(a.src)
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    cfg_txt = (src / "quatrex_config.toml").read_text()
    tdir = a.tdir or re.search(r'transport_direction\s*=\s*"(\w)"',
                               cfg_txt).group(1)
    tidx = "xyz".index(tdir)
    c, n_cells = a.per_block, a.cells
    if n_cells % c:
        raise SystemExit(f"--cells {n_cells} not divisible by --per-block {c}")
    nb = n_cells // c

    # ---- source ---------------------------------------------------------
    mats = _read_mat(src / "dynamical_matrix.mat", tidx)
    fc3, sizes = _read_fc3(src / "fc3_blocks.hdf5")
    n_src, nd = len(sizes), int(sizes[0])
    shifts = sorted({s for m in mats.values() for s in m})
    print(f"source {src.name}: {n_src} slabs x {nd} dof, transport {tdir}, "
          f"{len(mats)} transverse keys, transport shifts {shifts}")
    if max(abs(s) for s in shifts) != 1:
        raise SystemExit(
            f"source .mat has transport shifts {shifts}; this tool assumes a "
            "nearest-neighbour (folded) FC2 export.")
    _assert_slab_replicas(fc3, n_src, "fc3_blocks.hdf5")
    print(f"  slab translational equivalence: OK (fc3)")

    out.mkdir(parents=True, exist_ok=True)
    ndn = c * nd

    # ---- FC2: write the c-cell block as the unit cell, +-1 keys ---------
    new_mats = {}
    for perp, dmap in mats.items():
        for s in (-1, 0, 1):
            key = list(perp[:tidx]) + [s] + list(perp[tidx:])
            new_mats[f"[{key[0]}, {key[1]}, {key[2]}]"] = _superblock(
                dmap, c, nd, s)
    # exactness gate: the dense device operator must not change
    for perp in list(mats)[:8]:
        ref = _dense_fc2(mats, perp, n_cells, nd)
        got = np.zeros_like(ref)
        dmap_new = {s: _superblock(mats[perp], c, nd, s) for s in (-1, 0, 1)}
        for i in range(nb):
            for j in range(nb):
                d = dmap_new.get(j - i)
                if d is not None:
                    got[i * ndn:(i + 1) * ndn, j * ndn:(j + 1) * ndn] = d
        err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-300)
        if err > 1e-13:
            raise SystemExit(f"FC2 re-blocking changed the operator at "
                             f"transverse key {perp}: rel err {err:.2e}")
    print(f"  FC2 re-block exact on {min(8, len(mats))} transverse keys "
          f"(dense {n_cells}-cell operator unchanged)")
    from scipy.io import savemat
    savemat(str(out / "dynamical_matrix.mat"), new_mats)
    print(f"dynamical_matrix.mat: {len(new_mats)} keys of {ndn}x{ndn}")

    # ---- Gamma fc3 ------------------------------------------------------
    prim = _replicate(fc3, n_src, n_cells)
    if a.preserve_primitive_vertices:
        _write_fc3_blocks(
            {k: np.ascontiguousarray(v.astype(complex))
             for k, v in prim.items()},
            np.array([nd] * n_cells), out / "fc3_blocks.hdf5")
        print(f"  fc3 kept primitive: {len(prim)} blocks of {nd}^3 "
              f"for {nb} grouped Dyson blocks")
    else:
        merged = _merge(prim, n_cells, c, nd)
        ref = _dense_vertex(prim, n_cells, nd)
        got = _dense_vertex(merged, nb, ndn)
        err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-300)
        if err > 1e-14:
            raise SystemExit(f"fc3 merge changed the dense vertex: {err:.2e}")
        print(f"  fc3 merge exact (dense {n_cells}-cell vertex unchanged, "
              f"{len(prim)} primitive -> {len(merged)} merged blocks)")
        _write_fc3_blocks({k: np.ascontiguousarray(v.astype(complex))
                           for k, v in merged.items()},
                          np.array([ndn] * nb), out / "fc3_blocks.hdf5")

    # ---- q-folded vertices ---------------------------------------------
    qf = src / "qfold_vertices.npz"
    if qf.exists() and not a.skip_qfold:
        from quatrex.phonon.qfold import load_qfold, save_qfold  # noqa: E402

        V, q_diff_map, nk_shape = load_qfold(qf)
        first = True
        newV = {}
        for (q1, q2), blocks in V.items():
            if first:
                _assert_slab_replicas(blocks, n_src, f"qfold {(q1, q2)}")
                first = False
            replicated = _replicate(blocks, n_src, n_cells)
            newV[(q1, q2)] = (
                replicated if a.preserve_primitive_vertices
                else _merge(replicated, n_cells, c, nd))
        save_qfold(out / "qfold_vertices.npz", newV, q_diff_map, nk_shape)
        print(f"qfold_vertices.npz: {len(newV)} q-pairs x "
              f"{len(next(iter(newV.values())))} blocks of "
              f"{nd if a.preserve_primitive_vertices else ndn}^3")

    factor_requires_reconstruct = False
    if a.decomposed_path:
        from quatrex.phonon.vertex_factors import (
            load_decomposed, reblock_decomposed, save_decomposed,
        )

        source_factors = Path(a.decomposed_path)
        if not source_factors.is_absolute():
            source_factors = ROOT / source_factors
        primitive_factors = load_decomposed(source_factors)
        if a.preserve_primitive_vertices:
            if a.factor_support == "dense":
                support = sorted({
                    (int(k1 - i), int(k2 - i))
                    for i, k1, k2 in prim
                })
                available = set(map(int, primitive_factors.offsets))
                missing = sorted({x for pair in support for x in pair}
                                 - available)
                if missing:
                    raise SystemExit(
                        "dense FC3 support contains offsets absent from the "
                        f"factor archive: {missing}")
                primitive_factors.meta = {
                    **primitive_factors.meta,
                    "support_pairs": support,
                    "support_source": "dense FC3 transport offsets",
                }
                axis_a = {x for x, _ in support}
                axis_b = {y for _, y in support}
                factor_requires_reconstruct = set(support) != {
                    (x, y) for x in axis_a for y in axis_b
                }
            save_decomposed(out / "decomposed_vertices.npz",
                            primitive_factors)
            print(f"decomposed_vertices.npz: primitive factors preserved "
                  f"at rank {primitive_factors.rank}, "
                  f"dof {primitive_factors.D.shape[0]}, support "
                  f"{primitive_factors.meta.get('support_pairs', 'stored')}")
        else:
            lifted = reblock_decomposed(primitive_factors, c)
            save_decomposed(out / "decomposed_vertices.npz", lifted)
            print(f"decomposed_vertices.npz: exact factor lift rank "
                  f"{lifted.meta['primitive_rank']} -> {lifted.rank}, "
                  f"dof {lifted.meta['primitive_n_dof']} -> "
                  f"{lifted.D.shape[0]}")

    # ---- structure, grids, config ---------------------------------------
    lines = (src / "structure.xyz").read_text().splitlines()
    nat = int(lines[0])
    lat = [float(x) for x in re.search(r'Lattice="([^"]+)"',
                                       lines[1]).group(1).split()]
    avec = np.array(lat).reshape(3, 3)
    tvec = avec[tidx].copy()
    body = []
    for k in range(c):
        for ln in lines[2:2 + nat]:
            p = ln.split()
            xyz = np.array([float(p[1]), float(p[2]), float(p[3])]) + k * tvec
            body.append(f"{p[0]:2s} {xyz[0]:14.8f} {xyz[1]:14.8f} "
                        f"{xyz[2]:14.8f}")
    avec[tidx] = tvec * c
    latstr = " ".join(f"{v:.10g}" for v in avec.reshape(-1))
    (out / "structure.xyz").write_text(
        f"{nat * c}\n"
        f'Lattice="{latstr}" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        + "\n".join(body) + "\n")
    print(f"structure.xyz: {nat * c} atoms, transport lattice x{c}")

    for f in ("phonon_energies.npy", "kshift.npy"):
        if (src / f).exists():
            shutil.copy2(src / f, out / f)

    cfg = cfg_txt
    cfg = cfg.replace(f"/cluster/{src.name}", f"/cluster/{out.name}")
    cfg = re.sub(r"num_transport_cells\s*=\s*\d+",
                 f"num_transport_cells = {nb}", cfg)
    if a.decomposed_path:
        factor_line = (f'decomposed_vertices_path = "{out.resolve()}/'
                       'decomposed_vertices.npz"')
        if re.search(r"^decomposed_vertices_path\s*=.*$", cfg,
                     flags=re.MULTILINE):
            cfg = re.sub(r"^decomposed_vertices_path\s*=.*$", factor_line,
                         cfg, flags=re.MULTILINE)
        else:
            cfg = cfg.replace("[phonon]\n", "[phonon]\n" + factor_line + "\n")
        cfg = re.sub(r"^qfold_path\s*=.*\n?", "", cfg,
                     flags=re.MULTILINE)
        if factor_requires_reconstruct:
            line = 'decomposed_kernel = "reconstruct"'
            if re.search(r"^decomposed_kernel\s*=.*$", cfg,
                         flags=re.MULTILINE):
                cfg = re.sub(r"^decomposed_kernel\s*=.*$", line, cfg,
                             flags=re.MULTILINE)
            else:
                cfg = cfg.replace("[phonon]\n", "[phonon]\n" + line + "\n")
    elif qf.exists() and not a.skip_qfold:
        qfold_line = f'qfold_path = "{out.resolve()}/qfold_vertices.npz"'
        cfg = re.sub(r"^decomposed_vertices_path\s*=.*\n?", "", cfg,
                     flags=re.MULTILINE)
        if re.search(r"^qfold_path\s*=.*$", cfg, flags=re.MULTILINE):
            cfg = re.sub(r"^qfold_path\s*=.*$", qfold_line, cfg,
                         flags=re.MULTILINE)
        else:
            cfg = cfg.replace("[phonon]\n", "[phonon]\n" + qfold_line + "\n")
    else:
        # Gamma-only output: do not inherit a source representation whose
        # archive was deliberately not copied.
        cfg = re.sub(
            r"^(?:qfold_path|decomposed_vertices_path)\s*=.*\n?", "", cfg,
            flags=re.MULTILINE)
    if a.preserve_primitive_vertices:
        micro_lines = (
            f"sse_microblock_dof = {nd}\n"
            f"sse_microblock_g_band = {a.micro_g_band}\n")
        for field in ("sse_microblock_dof", "sse_microblock_g_band"):
            cfg = re.sub(rf"^{field}\s*=.*\n?", "", cfg,
                         flags=re.MULTILINE)
        cfg = cfg.replace("[phonon]\n", "[phonon]\n" + micro_lines)
    (out / "quatrex_config.toml").write_text(cfg)
    print(f"quatrex_config.toml: num_transport_cells = {nb} "
          f"({nb} blocks x {ndn} dof = {nb * ndn}, {n_cells} primitive cells)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
