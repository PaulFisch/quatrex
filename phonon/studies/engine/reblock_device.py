"""Re-block a transversely-periodic film device: N primitive transport
cells, C of them per BTD block.

Why. The anharmonic self-energy is written only on the block-tridiagonal
(``sse_phonon_phonon.py:410``, and the RGF reads nothing else), so the
retained interaction range is +-1 BLOCK whatever ``sse_g_band`` says.
That Hadamard mask is indefinite -- min eigenvalue ``1 - 2 cos(pi/(n+1))``
in the number of blocks -- and it is the live PSD defect measured in
``phonon/docs/bubble_positivity.md``. Putting C transport cells in one
block widens the retained physical range to +-C cells AND reduces the
block count, so the mask gets both smaller support to damage and better
conditioning. Unlike a taper it costs no vertex weight: it is an exact
re-partition of the same operator.

What this does. Everything here is a re-partition plus (for N > the
source device) a replication along transport; no DFT, no hiphive refit,
and no q-fold recomputation. The output is written as a device whose
UNIT CELL is the C-cell block, with ``+-1`` transport keys, so the
production loader's ``supercell_size = extent // 2``
(``src/quatrex/device/inputs.py:993-998``) reads it back as C*nd-sized
blocks with no change anywhere in ``src/quatrex``.

Replication is exact only for a device whose slabs are translationally
equivalent, which is checked, not assumed: every per-slab vertex block
must be bit-identical across slabs (it is for the MoS2 film -- the fit
prunes the vdW-gap fc3 to exact zero, so the vertex is intra-slab).

Run (from the repo root):

    python phonon/studies/engine/reblock_device.py \
        --src cluster/mos2f3 --cells 6 --per-block 2 --out cluster/mos2f6x2
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
    from phonon_inputs.quatrex_writer import (  # noqa: E402
        write_fc3_blocks, write_structure_xyz,
    )

    prim = _replicate(fc3, n_src, n_cells)
    merged = _merge(prim, n_cells, c, nd)
    ref = _dense_vertex(prim, n_cells, nd)
    got = _dense_vertex(merged, nb, ndn)
    err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-300)
    if err > 1e-14:
        raise SystemExit(f"fc3 merge changed the dense vertex: {err:.2e}")
    print(f"  fc3 merge exact (dense {n_cells}-cell vertex unchanged, "
          f"{len(prim)} primitive -> {len(merged)} merged blocks)")
    write_fc3_blocks({k: np.ascontiguousarray(v.astype(complex))
                      for k, v in merged.items()},
                     np.array([ndn] * nb), out / "fc3_blocks.hdf5",
                     units="THz^2")

    # ---- q-folded vertices ---------------------------------------------
    qf = src / "qfold_vertices.npz"
    if qf.exists():
        from quatrex.phonon.qfold import load_qfold, save_qfold  # noqa: E402

        V, q_diff_map, nk_shape = load_qfold(qf)
        first = True
        newV = {}
        for (q1, q2), blocks in V.items():
            if first:
                _assert_slab_replicas(blocks, n_src, f"qfold {(q1, q2)}")
                first = False
            newV[(q1, q2)] = _merge(_replicate(blocks, n_src, n_cells),
                                    n_cells, c, nd)
        save_qfold(out / "qfold_vertices.npz", newV, q_diff_map, nk_shape)
        print(f"qfold_vertices.npz: {len(newV)} q-pairs x "
              f"{len(next(iter(newV.values())))} blocks of {ndn}^3")

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
    (out / "quatrex_config.toml").write_text(cfg)
    print(f"quatrex_config.toml: num_transport_cells = {nb} "
          f"({nb} blocks x {ndn} dof = {nb * ndn}, {n_cells} primitive cells)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
