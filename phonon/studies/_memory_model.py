"""Predict peak GPU memory per rank for a phonon-phonon SCBA run.

Every OOM in the 2026-08 MoS2 campaign was discovered by running into it:
four jobs died at 95-100 GB/GPU with no prior estimate. This module turns
the allocation sites into a closed formula so a launch can be sized
beforehand, and pins it with the three measured OOMs as fixtures.

The model is assembled from the actual allocation sites, not from a fit:

  B_G   = 16 * ceil(ne/P_s)    * nq * nnz_r     one G or Sigma buffer
  B_tau = 16 * ceil(n_fft/P_s) * nq * nnz_r     one tau buffer

  peak ~= 10 * B_G                              scba.py:183-207 (+system_matrix)
        +  6 * B_tau                            sse:2596  (10 with compute_linearized)
        +  4 * L  * blk                         sse:1228-1255  the *_blk dicts
        +  4 * L  * blk                         sse:2030       _stack() duplicate
        +  k * Np * blk                         sse:1363, 2046 per-pair outputs
        +  2 * ring_b3                          bubble.py:241-242  T and U
        + 32 * b^3 * (nq^2/P_q) * Q             sse:1871  perm_cache (dense qfold)
        + (3*nB + 18) * rgf_tmp                 rgf.py:161-163, 350-401
        +  4 * obc                              nevp/full.py:165-184
        + 12 * B_G          (retarded="fft")    fft_utils.py:91,114
  with blk = 16 * ceil(n_fft/P_s) * nq * b^2.

Three facts the formula makes visible, all load-bearing:

* the transverse-q axis is REPLICATED on every rank (nothing distributes
  `global_stack_shape[1:]`, dsdbsparse.py:829-837), so nq multiplies
  every term and adding nodes cannot touch it;
* `q_comm_size > 1` makes per-rank memory WORSE, because
  P_s = world / (P_b * P_q) (comm.py:1095);
* `perm_cache` scales as b^3 * nq^2 and is bounded by nothing.

Usage
-----
    QTX_ARRAY_MODULE=numpy python phonon/studies/_memory_model.py \
        cluster/mos2f4dense --ranks 8
    ... --ne 4001 --aux-dw 0.01          # what-if on the grid
    ... --check                          # run the OOM fixtures
"""

from __future__ import annotations

import argparse
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GB = 1024.0**3


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
@dataclass
class Params:
    """Everything the formula needs. Sizes are counts, not bytes."""

    ne: int                  # primary frequency points
    nnz: int                 # stored elements of the shared pattern
    nq: int                  # transverse k-points (REPLICATED per rank)
    block_sizes: list[int]
    world: int = 8           # total ranks
    p_block: int = 1         # block_comm_size
    p_q: int = 1             # q_comm_size
    aux_dw: float = 0.0      # 0 => aux grid off, ne_conv = ne
    aux_fmax: float = 0.0
    prim_top: float = 16.0   # top of the primary grid (THz)
    g_band: int = 3
    max_batch: int = 100000
    n_links: int = 0         # L, distinct (K,K') band links
    n_pairs: int = 0         # Np, output (I,J) pairs
    n_quads: int = 0         # Q, ring quads summed over pairs
    qfold: bool = True       # dense q-folded vertex (perm_cache lives)
    dense_q_batched: bool = True
    with_dg: bool = False    # compute_linearized (newton/jfnk) => 10 tau bufs
    retarded: str = "half"   # "half" | "fft"
    out_per_pair: int = 4    # out_l, out_g, out_x, out_t56
    tau_chunk_bytes: float = 256e6   # config.py:1497
    label: str = ""

    # derived
    b: int = field(init=False)
    n_blocks: int = field(init=False)

    def __post_init__(self) -> None:
        self.b = int(max(self.block_sizes))
        self.n_blocks = len(self.block_sizes)

    @property
    def p_stack(self) -> int:
        return max(1, self.world // (self.p_block * self.p_q))

    @property
    def ne_conv(self) -> int:
        if self.aux_dw <= 0.0:
            return self.ne
        top = max(self.prim_top, self.aux_fmax)
        return int(math.ceil(top / self.aux_dw - 1e-9)) + 1

    @property
    def n_fft(self) -> int:
        return 2 * self.ne_conv - 1

    @property
    def nnz_r(self) -> int:
        return int(math.ceil(self.nnz / self.p_block))

    @property
    def ne_loc(self) -> int:
        return int(math.ceil(self.ne / self.p_stack))

    @property
    def n_tau(self) -> int:
        return int(math.ceil(self.n_fft / self.p_stack))


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
def breakdown(p: Params) -> tuple[dict[str, float], dict[str, float]]:
    """(solve-phase, SSE-phase) bytes per rank, per allocation site.

    The two phases do not overlap in principle: `system_matrix` and the
    RGF/OBC temporaries are allocated and freed inside `PhononSolver.solve`
    (solver.py:379, :579), which is the only place on the phonon path that
    calls `free_mempool()` (dsdbsparse.py:779, :798). Everything else is
    reported under the SSE phase.

    The caveat that makes this a range rather than a number: the CuPy pool
    only returns blocks to the driver on `free_all_blocks`, and the SSE
    path never calls it. So the observed high-water mark sits somewhere
    between max(solve, sse) -- perfect block reuse -- and their sum -- no
    reuse at all. Closing that gap is what a `free_mempool()` between the
    SSE stages would buy.
    """
    z = 16.0  # complex128
    b_g = z * p.ne_loc * p.nq * p.nnz_r
    b_tau = z * p.n_tau * p.nq * p.nnz_r
    blk = z * p.n_tau * p.nq * p.b**2
    n_tau_buf = 10 if p.with_dg else 6

    # ---- persistent across both phases ------------------------------
    # scba.py:183-207 -- 9 buffers (Sigma and Sigma_prev are a full double
    # copy) plus system_matrix, allocated per solve.
    persistent = {"scba buffers (10 B_G)": 10 * b_g}

    # ---- solve phase -------------------------------------------------
    solve = dict(persistent)
    # rgf.py:161-163 + ~18 named temporaries in the backward sweep.
    batch = min(p.max_batch, p.ne_loc)
    solve["RGF temporaries"] = (3 * p.n_blocks + 18) * z * batch * p.nq * p.b**2
    # nevp/full.py:165-184 -- (2b)^2 per (frequency, q), unbatched.
    solve["OBC Full NEVP"] = 4 * z * p.ne_loc * p.nq * (2 * p.b) ** 2

    # ---- SSE phase ----------------------------------------------------
    sse = dict(persistent)
    sse[f"tau buffers ({n_tau_buf} B_tau)"] = n_tau_buf * b_tau
    # sse:1228-1255 -- .blocks[K,Kp] densifies, it is not a view.
    sse["band-link dicts (4L)"] = 4 * p.n_links * blk
    if p.nq > 1 and p.dense_q_batched:
        # sse:2030 -- a second full copy of the same data, built before the
        # tau-chunk loop that would have bounded it.
        sse["_stack() duplicate (4L)"] = 4 * p.n_links * blk
    # sse:1363-1374 / :2046-2051 -- all pairs live at once.
    sse["per-pair outputs"] = p.out_per_pair * p.n_pairs * blk
    # bubble.py:241-242 -- T and U. The coupled-q path chunks them against
    # sse_tau_chunk_bytes (:2071-2098); the Gamma-only GPU path does not
    # (:1496 falls through to the unbatched _contract_tau).
    if p.nq > 1 and p.dense_q_batched:
        sse["ring T,U (chunked)"] = 2.0 * p.tau_chunk_bytes
    else:
        sse["ring T,U (b^3, unbatched)"] = 2 * z * p.n_tau * p.b**3
    if p.qfold:
        # sse:1871 -- persistent, unbounded, keyed per block index.
        sse["perm_cache"] = 32.0 * p.b**3 * (p.nq**2 / max(1, p.p_q)) * p.n_quads
    if p.retarded == "fft":
        sse["Hilbert (fft mode)"] = 12 * b_g
    # dsdbsparse.py:662-713 -- dtranspose is not in-place.
    sse["dtranspose transient"] = max(b_g, b_tau)
    return solve, sse


def report(p: Params, note: str = "") -> tuple[float, float]:
    solve, sse = breakdown(p)
    head = f"{p.label or 'run'}  {note}".strip()
    print(f"\n=== {head} ===")
    print(f"  ne={p.ne} ne_conv={p.ne_conv} n_fft={p.n_fft} nnz={p.nnz} "
          f"nq={p.nq} b={p.b} nB={p.n_blocks}")
    print(f"  world={p.world} P_stack={p.p_stack} P_block={p.p_block} "
          f"P_q={p.p_q} -> ne_loc={p.ne_loc} n_tau={p.n_tau}")
    print(f"  L={p.n_links} Np={p.n_pairs} Q={p.n_quads} "
          f"max_batch={min(p.max_batch, p.ne_loc)}")
    for name, d in (("solve phase", solve), ("SSE phase", sse)):
        tot = sum(d.values())
        print(f"  -- {name}: {tot / GB:.2f} GB")
        for k, v in sorted(d.items(), key=lambda kv: -kv[1]):
            if v > 0:
                print(f"       {k:<30s} {v / GB:9.2f} GB ({100 * v / tot:5.1f} %)")
    lo = max(sum(solve.values()), sum(sse.values()))
    hi = sum(solve.values()) + sum(sse.values())
    print(f"  => peak between {lo / GB:.1f} and {hi / GB:.1f} GB/rank "
          f"(reuse / no reuse)")
    return lo, hi


# --------------------------------------------------------------------------
# derive parameters from a run directory
# --------------------------------------------------------------------------
def _zcoords(xyz: Path, n_cells: int, axis: int = 2) -> np.ndarray:
    """Orbital coordinates along the transport axis, tiled over cells.

    Mirrors create_coordinate_grid (device/inputs.py:70-106) for the
    axis-aligned case, then repeats 3x for the cartesian dof.
    """
    import re

    lines = xyz.read_text().splitlines()
    lat = [float(x) for x in
           re.search(r'Lattice="([^"]+)"', lines[1]).group(1).split()]
    cell = np.asarray(lat, dtype=float).reshape(3, 3)[axis, axis]
    n_at = int(lines[0])
    pos = np.array([[float(v) for v in ln.split()[1:4]]
                    for ln in lines[2:2 + n_at]])
    z = np.concatenate([pos[:, axis] + k * cell for k in range(n_cells)])
    return np.repeat(z, 3)


def _pattern_nnz(z: np.ndarray, cutoff: float, block_sizes: list[int],
                 g_band: int) -> int:
    """nnz of the shared pattern: box cutoff, unioned with the g_band blocks.

    Replicates core/utils.py:45-74 (strict `<`, transport axis only) and
    core/scba.py:126-168 (the extra |I-J| <= g_band blocks).
    """
    keep = np.abs(z[:, None] - z[None, :]) < cutoff
    if g_band > 1:
        off = np.hstack(([0], np.cumsum(block_sizes)))
        blk = np.zeros(z.size, dtype=int)
        for i in range(len(block_sizes)):
            blk[off[i]:off[i + 1]] = i
        keep |= np.abs(blk[:, None] - blk[None, :]) <= g_band
    return int(keep.sum())


def _fc3_index(fc3: Path, n_blocks: int, g_band: int) -> tuple[int, int, int]:
    """(L, Np, Q) from the shipped vertex, exactly as sse:434-449 builds them."""
    import h5py

    with h5py.File(str(fc3), "r") as f:
        if "fc3_blocks" not in f:
            return 0, 0, 0
        phi = {(int(d.attrs["I"]), int(d.attrs["J"]), int(d.attrs["K"]))
               for d in (f["fc3_blocks"][k] for k in f["fc3_blocks"])}
    links: set[tuple[int, int]] = set()
    pairs: set[tuple[int, int]] = set()
    quads = 0
    for (i, k1, k2) in phi:
        for j in range(max(0, i - 1), min(n_blocks, i + 2)):
            for k1p in range(max(0, k1 - g_band), min(n_blocks, k1 + g_band + 1)):
                for k2p in range(max(0, k2 - g_band),
                                 min(n_blocks, k2 + g_band + 1)):
                    if (j, k2p, k1p) not in phi:
                        continue
                    quads += 1
                    pairs.add((i, j))
                    links.add((k1, k1p))
                    links.add((k2, k2p))
    return len(links), len(pairs), quads


def from_run_dir(path: Path, world: int, **over) -> Params:
    d = Path(path)
    cfg = tomllib.loads((d / "quatrex_config.toml").read_text())
    ph = cfg.get("phonon", {})
    dev = cfg.get("device", {})
    ew = cfg.get("electron", cfg)

    n_cells = int(dev.get("num_transport_cells", ph.get("num_transport_cells", 1)))
    if "num_transport_cells" in cfg:
        n_cells = int(cfg["num_transport_cells"])
    kgrid = dev.get("kpoint_grid", cfg.get("kpoint_grid", [1, 1, 1]))
    nq = int(np.prod([k for k in kgrid if k > 1])) if any(
        k > 1 for k in kgrid) else 1
    axis = {"x": 0, "y": 1, "z": 2}[dev.get("transport_direction",
                                            cfg.get("transport_direction", "z"))]

    ne = int(cfg.get("energy_window_num", ew.get("energy_window_num", 121)))
    top = float(cfg.get("energy_window_max", ew.get("energy_window_max", 16.0)))
    cutoff = float(ph.get("interaction_cutoff", cfg.get("interaction_cutoff", 10.0)))
    g_band = int(ph.get("sse_g_band", cfg.get("sse_g_band", 3)))
    aux_dw = float(ph.get("sse_aux_grid_dw_thz", cfg.get("sse_aux_grid_dw_thz", 0.0)))
    aux_fm = float(ph.get("sse_aux_grid_fmax_thz",
                          cfg.get("sse_aux_grid_fmax_thz", 0.0)))
    mb = int(cfg.get("max_batch_size",
                     ph.get("solver", {}).get("max_batch_size", 100000)))

    z = _zcoords(d / "structure.xyz", n_cells, axis)
    n_dof = z.size
    # supercell_size = transport extent // 2 (device/inputs.py:993-998); every
    # shipped device has one cell per block unless it was re-blocked.
    n_blocks = n_cells
    bs = [n_dof // n_blocks] * n_blocks
    g_band_eff = min(g_band, n_blocks - 1) if n_blocks > 1 else 1
    nnz = _pattern_nnz(z, cutoff, bs, g_band_eff)
    L, Np, Q = _fc3_index(d / "fc3_blocks.hdf5", n_blocks, g_band_eff)

    p = dict(ne=ne, nnz=nnz, nq=nq, block_sizes=bs, world=world, prim_top=top,
             aux_dw=aux_dw, aux_fmax=aux_fm, g_band=g_band_eff, max_batch=mb,
             n_links=L, n_pairs=Np, n_quads=Q,
             qfold=(d / "qfold_vertices.npz").exists() and nq > 1,
             label=d.name)
    p.update({k: v for k, v in over.items() if v is not None})
    return Params(**p)


# --------------------------------------------------------------------------
# fixtures: the three OOMs we actually hit
# --------------------------------------------------------------------------
FIXTURES = [
    # (label, Params kwargs, observed pool bytes at death)
    ("mos2f2dense", dict(ne=15001, nnz=1296, nq=25, block_sizes=[18, 18],
                         world=4, aux_dw=0.0, prim_top=16.0, g_band=1,
                         max_batch=100000, n_links=4, n_pairs=4, n_quads=4,
                         label="mos2f2dense"), 97.9),
    ("mos2f4dense", dict(ne=15001, nnz=5184, nq=25, block_sizes=[36, 36],
                         world=8, aux_dw=0.01, aux_fmax=32.0, prim_top=16.0,
                         g_band=1, max_batch=100000, n_links=4, n_pairs=4,
                         n_quads=4, label="mos2f4dense"), 100.6),
    # Q = 4 counted from mos2f6x3/fc3_blocks.hdf5, not guessed. The
    # traceback's ~39000 allocations of 2.52 MB are phi_perms CALLS, not
    # live cache entries: 4 quads x 25^2 q-pairs x 2 stored perms = 2500
    # entries = 12.6 GB, which is the straw that broke an SSE phase already
    # sitting near 90 GB -- not a 97 GB term on its own.
    ("mos2L6n4scba", dict(ne=4001, nnz=11664, nq=25, block_sizes=[54, 54],
                          world=8, aux_dw=0.01, aux_fmax=32.0, prim_top=16.0,
                          g_band=1, max_batch=100000, n_links=4, n_pairs=4,
                          n_quads=4, label="mos2L6n4scba"), 97.4),
]


def check() -> int:
    """Gate: each fixture must be predicted to exceed the 96 GB GH200 HBM.

    The pool size at death is a LOWER bound on the true requirement (the
    allocation that failed is not in it), so the test is that the model
    puts the requirement above the ceiling -- not that it reproduces the
    death-time pool.
    """
    print("Fixtures: three measured OOMs. GH200 = 96 GB/GPU.")
    bad = 0
    for _, kw, obs in FIXTURES:
        lo, hi = report(Params(**kw), note=f"died with pool at {obs:.1f} GB")
        ok = hi / GB > 96.0 and hi / GB >= obs
        print(f"    predicts OOM: {'PASS' if ok else 'FAIL'} "
              f"(need >96 and >={obs:.1f}, got {hi / GB:.1f})")
        bad += not ok
    print(f"\n{len(FIXTURES) - bad}/{len(FIXTURES)} fixtures pass.")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rundir", nargs="?", type=Path)
    ap.add_argument("--ranks", type=int, default=8, help="world size")
    ap.add_argument("--nodes", type=int, default=None,
                    help="convenience: ranks = 4*nodes")
    ap.add_argument("--ne", type=int, default=None)
    ap.add_argument("--aux-dw", dest="aux_dw", type=float, default=None)
    ap.add_argument("--aux-fmax", dest="aux_fmax", type=float, default=None)
    ap.add_argument("--max-batch", dest="max_batch", type=int, default=None)
    ap.add_argument("--p-block", dest="p_block", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if a.check or a.rundir is None:
        raise SystemExit(check())
    world = 4 * a.nodes if a.nodes else a.ranks
    p = from_run_dir(a.rundir, world, ne=a.ne, aux_dw=a.aux_dw,
                     aux_fmax=a.aux_fmax, max_batch=a.max_batch,
                     p_block=a.p_block)
    lo, hi = report(p)
    verdict = ("FITS" if hi / GB < 96 else
               "MARGINAL" if lo / GB < 96 else "OOM")
    print(f"\n  GH200 96 GB/GPU -> {verdict}")


if __name__ == "__main__":
    main()
